//! Core aggregation implementations for GroupBy operations
//!
//! # Semantics
//!
//! * Groups are emitted in ascending key order (pandas `sort=True` default) so
//!   that results are reproducible; iterating the group `HashMap` directly
//!   produces a different row order on every run.
//! * Missing values are skipped by every reduction (pandas `skipna=True`). For
//!   float columns `NaN` counts as missing, as it does in pandas' `float64`.
//! * A reduction that is undefined for the group (mean/min/max/median/first/
//!   last of an all-missing group, std/var of fewer than two observations)
//!   yields a real NULL in the output column instead of a fabricated `0.0`.
//!   `sum` of an all-missing group is `0.0`, matching pandas.

use rayon::prelude::*;

use super::super::core::OptimizedDataFrame;
use super::grouping::NA_GROUP_KEY_MARKER;
use super::types::{AggregateOp, CustomAggregation, GroupBy, GroupKey};
use crate::column::{Column, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::StringMultiIndex;

/// Number of groups from which parallel evaluation pays for itself.
const PARALLEL_THRESHOLD: usize = 10;
/// Row count from which parallel evaluation pays for itself.
const DATA_SIZE_THRESHOLD: usize = 10_000;

/// Iterate the non-missing `i64` values of `col` for the given rows.
fn int_values<'i>(col: &'i Int64Column, rows: &'i [usize]) -> impl Iterator<Item = i64> + 'i {
    rows.iter()
        .filter_map(move |&idx| col.get(idx).ok().flatten())
}

/// Iterate the non-missing `f64` values of `col` for the given rows.
///
/// `NaN` is treated as missing, matching pandas where `NaN` is the `float64`
/// missing marker and every reduction defaults to `skipna=True`.
fn float_values<'i>(col: &'i Float64Column, rows: &'i [usize]) -> impl Iterator<Item = f64> + 'i {
    rows.iter()
        .filter_map(move |&idx| col.get(idx).ok().flatten())
        .filter(|value| !value.is_nan())
}

/// Number of non-missing entries of `col` across `rows`.
fn count_valid(col: &Column, rows: &[usize]) -> usize {
    match col {
        Column::Int64(c) => rows
            .iter()
            .filter(|&&idx| matches!(c.get(idx), Ok(Some(_))))
            .count(),
        Column::Float64(c) => float_values(c, rows).count(),
        Column::String(c) => rows
            .iter()
            .filter(|&&idx| matches!(c.get(idx), Ok(Some(_))))
            .count(),
        Column::Boolean(c) => rows
            .iter()
            .filter(|&&idx| matches!(c.get(idx), Ok(Some(_))))
            .count(),
    }
}

/// Sample variance (Bessel's correction). `None` for fewer than two values.
fn sample_variance(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sum_squared_diff = values
        .iter()
        .map(|&value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>();

    Some(sum_squared_diff / (n - 1.0))
}

/// Median of the supplied values. `None` when there is nothing to reduce.
fn median_of(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    // `total_cmp` is a genuine total order; `partial_cmp` + fallback is not
    // transitive and can leave the slice unsorted.
    values.sort_by(|a, b| a.total_cmp(b));

    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

impl<'a> GroupBy<'a> {
    /// Groups in ascending key order, so output rows are reproducible.
    pub(super) fn ordered_groups(&self) -> Vec<(&GroupKey<'a>, &Vec<usize>)> {
        let mut ordered: Vec<(&GroupKey<'a>, &Vec<usize>)> = self.groups.iter().collect();
        ordered.sort_by(|left, right| left.0.cmp(right.0));
        ordered
    }

    /// Resolve aggregation source columns once instead of per group.
    fn resolve_columns(&self, names: &[String]) -> Result<Vec<&'a Column>> {
        let mut columns = Vec::with_capacity(names.len());
        for name in names {
            let idx = *self
                .df
                .column_indices
                .get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.clone()))?;
            let column = self
                .df
                .columns
                .get(idx)
                .ok_or_else(|| Error::ColumnNotFound(name.clone()))?;
            columns.push(column);
        }
        Ok(columns)
    }

    /// Assemble the result frame from per-group key/value data.
    ///
    /// Keys and values are indexed by the *same* group ordering, so a row can
    /// never pair one group's key with another group's aggregate.
    fn build_result(
        &self,
        ordered: &[(&GroupKey<'a>, &Vec<usize>)],
        result_names: &[String],
        per_group_values: &[Vec<Option<f64>>],
    ) -> Result<OptimizedDataFrame> {
        let mut result = OptimizedDataFrame::new();
        let group_count = ordered.len();

        if self.create_multi_index && self.group_by_columns.len() > 1 {
            // MultiIndex levels are string-typed, so a missing key component
            // has to be rendered; only reachable with `dropna = false`.
            let tuples: Vec<Vec<String>> = ordered
                .iter()
                .map(|(key, _)| {
                    key.iter()
                        .map(|part| {
                            part.to_value_string()
                                .unwrap_or_else(|| NA_GROUP_KEY_MARKER.to_string())
                        })
                        .collect()
                })
                .collect();

            let names = Some(
                self.group_by_columns
                    .iter()
                    .map(|name| Some(name.clone()))
                    .collect(),
            );

            let multi_index = StringMultiIndex::from_tuples(tuples, names)?;
            result.set_index_from_multi_index(multi_index)?;
        } else {
            // Iterate `group_by_columns` (not a HashMap) so the result column
            // order is deterministic.
            for (level, col_name) in self.group_by_columns.iter().enumerate() {
                let mut values: Vec<String> = Vec::with_capacity(group_count);
                let mut nulls: Vec<bool> = Vec::with_capacity(group_count);

                for (key, _) in ordered {
                    match key.get(level).and_then(|part| part.to_value_string()) {
                        Some(text) => {
                            values.push(text);
                            nulls.push(false);
                        }
                        None => {
                            // Missing key stays missing: no literal "NULL" cell.
                            values.push(String::new());
                            nulls.push(true);
                        }
                    }
                }

                let column = if nulls.iter().any(|&is_null| is_null) {
                    StringColumn::with_nulls(values, nulls)
                } else {
                    StringColumn::new(values)
                };
                result.add_column(col_name.clone(), Column::String(column))?;
            }
        }

        // Aggregation result columns, in the order the caller requested them.
        for (agg_idx, alias) in result_names.iter().enumerate() {
            let mut values: Vec<f64> = Vec::with_capacity(group_count);
            let mut nulls: Vec<bool> = Vec::with_capacity(group_count);

            for group_values in per_group_values {
                match group_values.get(agg_idx).copied().flatten() {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(0.0);
                        nulls.push(true);
                    }
                }
            }

            let column = if nulls.iter().any(|&is_null| is_null) {
                Float64Column::with_nulls(values, nulls)
            } else {
                Float64Column::new(values)
            };
            result.add_column(alias.clone(), Column::Float64(column))?;
        }

        result.row_count = group_count;

        Ok(result)
    }

    /// Whether parallel group evaluation is worthwhile for this frame.
    fn should_parallelize(&self) -> bool {
        self.groups.len() >= PARALLEL_THRESHOLD
            || (self.df.row_count >= DATA_SIZE_THRESHOLD && self.groups.len() > 3)
    }

    /// Shared implementation of [`GroupBy::aggregate`] / [`GroupBy::par_aggregate`].
    fn aggregate_impl(
        &self,
        aggregations: &[(String, AggregateOp, String)],
        parallel: bool,
    ) -> Result<OptimizedDataFrame> {
        let source_names: Vec<String> = aggregations
            .iter()
            .map(|(col_name, _, _)| col_name.clone())
            .collect();
        let columns = self.resolve_columns(&source_names)?;
        let result_names: Vec<String> = aggregations
            .iter()
            .map(|(_, _, alias)| alias.clone())
            .collect();

        let ordered = self.ordered_groups();

        let evaluate = |rows: &[usize]| -> Result<Vec<Option<f64>>> {
            columns
                .iter()
                .zip(aggregations.iter())
                .map(|(column, (_, op, _))| self.calculate_aggregation(column, *op, rows))
                .collect()
        };

        let per_group_values: Vec<Vec<Option<f64>>> = if parallel {
            ordered
                .par_iter()
                .map(|&(_, rows)| evaluate(rows))
                .collect::<Result<Vec<_>>>()?
        } else {
            ordered
                .iter()
                .map(|&(_, rows)| evaluate(rows))
                .collect::<Result<Vec<_>>>()?
        };

        self.build_result(&ordered, &result_names, &per_group_values)
    }

    /// Shared implementation of the custom-aggregation entry points.
    fn aggregate_custom_impl(
        &self,
        aggregations: &[CustomAggregation],
        parallel: bool,
    ) -> Result<OptimizedDataFrame> {
        // Verify that custom functions are provided for AggregateOp::Custom
        for agg in aggregations {
            if agg.op == AggregateOp::Custom && agg.custom_fn.is_none() {
                return Err(Error::OperationFailed(
                    "Custom aggregation function is required for AggregateOp::Custom".to_string(),
                ));
            }
        }

        let source_names: Vec<String> = aggregations.iter().map(|agg| agg.column.clone()).collect();
        let columns = self.resolve_columns(&source_names)?;
        let result_names: Vec<String> = aggregations
            .iter()
            .map(|agg| agg.result_name.clone())
            .collect();

        let ordered = self.ordered_groups();

        let evaluate = |rows: &[usize]| -> Result<Vec<Option<f64>>> {
            columns
                .iter()
                .zip(aggregations.iter())
                .map(|(&column, agg)| match (column, &agg.custom_fn) {
                    (Column::Int64(int_col), Some(custom_fn)) if agg.op == AggregateOp::Custom => {
                        let values: Vec<f64> = int_values(int_col, rows)
                            .map(|value| value as f64)
                            .collect();
                        Ok(Some(custom_fn(&values)))
                    }
                    (Column::Float64(float_col), Some(custom_fn))
                        if agg.op == AggregateOp::Custom =>
                    {
                        let values: Vec<f64> = float_values(float_col, rows).collect();
                        Ok(Some(custom_fn(&values)))
                    }
                    _ => self.calculate_aggregation(column, agg.op, rows),
                })
                .collect()
        };

        let per_group_values: Vec<Vec<Option<f64>>> = if parallel {
            ordered
                .par_iter()
                .map(|&(_, rows)| evaluate(rows))
                .collect::<Result<Vec<_>>>()?
        } else {
            ordered
                .iter()
                .map(|&(_, rows)| evaluate(rows))
                .collect::<Result<Vec<_>>>()?
        };

        self.build_result(&ordered, &result_names, &per_group_values)
    }

    /// Execute aggregation operations for each group in parallel
    ///
    /// # Arguments
    /// * `aggregations` - List of aggregation operations (column name, operation, result column name)
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn par_aggregate<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        let aggregations: Vec<(String, AggregateOp, String)> = aggregations.into_iter().collect();
        let parallel = self.should_parallelize();
        self.aggregate_impl(&aggregations, parallel)
    }

    /// Execute custom aggregations in parallel
    ///
    /// # Arguments
    /// * `aggregations` - List of custom aggregation operations
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn par_aggregate_custom<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = CustomAggregation>,
    {
        let aggregations: Vec<CustomAggregation> = aggregations.into_iter().collect();
        let parallel = self.should_parallelize();
        self.aggregate_custom_impl(&aggregations, parallel)
    }

    /// Execute aggregation operations for each group with support for custom functions
    ///
    /// # Arguments
    /// * `aggregations` - List of custom aggregation operations
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn aggregate_custom<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = CustomAggregation>,
    {
        let aggregations: Vec<CustomAggregation> = aggregations.into_iter().collect();
        self.aggregate_custom_impl(&aggregations, false)
    }

    /// Execute aggregation operations for each group
    ///
    /// # Arguments
    /// * `aggregations` - List of aggregation operations (column name, operation, result column name)
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn aggregate<I>(&self, aggregations: I) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, AggregateOp, String)>,
    {
        let aggregations: Vec<(String, AggregateOp, String)> = aggregations.into_iter().collect();
        self.aggregate_impl(&aggregations, false)
    }

    /// Reduce one column over one group.
    ///
    /// Returns `Ok(None)` when the reduction is undefined for the group (an
    /// all-missing group, or fewer than two observations for std/var) so the
    /// caller can emit a NULL rather than substituting `0.0`.
    pub(crate) fn calculate_aggregation(
        &self,
        col: &Column,
        op: AggregateOp,
        row_indices: &[usize],
    ) -> Result<Option<f64>> {
        // Count and Custom are handled uniformly across column types.
        match op {
            AggregateOp::Count => return Ok(Some(count_valid(col, row_indices) as f64)),
            AggregateOp::Custom => {
                return Err(Error::OperationFailed(
                    "Custom aggregation requires a custom function, use aggregate_custom instead"
                        .to_string(),
                ))
            }
            _ => {}
        }

        match col {
            Column::Int64(int_col) => match op {
                AggregateOp::Sum => {
                    // Accumulate in i128: exact and overflow-free.
                    let sum: i128 = int_values(int_col, row_indices)
                        .map(i128::from)
                        .sum::<i128>();
                    Ok(Some(sum as f64))
                }
                AggregateOp::Mean => {
                    let mut sum: i128 = 0;
                    let mut count: usize = 0;
                    for value in int_values(int_col, row_indices) {
                        sum += i128::from(value);
                        count += 1;
                    }
                    if count == 0 {
                        Ok(None)
                    } else {
                        Ok(Some(sum as f64 / count as f64))
                    }
                }
                AggregateOp::Min => Ok(int_values(int_col, row_indices)
                    .min()
                    .map(|value| value as f64)),
                AggregateOp::Max => Ok(int_values(int_col, row_indices)
                    .max()
                    .map(|value| value as f64)),
                AggregateOp::Std => {
                    let values: Vec<f64> = int_values(int_col, row_indices)
                        .map(|value| value as f64)
                        .collect();
                    Ok(sample_variance(&values).map(f64::sqrt))
                }
                AggregateOp::Var => {
                    let values: Vec<f64> = int_values(int_col, row_indices)
                        .map(|value| value as f64)
                        .collect();
                    Ok(sample_variance(&values))
                }
                AggregateOp::Median => {
                    let mut values: Vec<f64> = int_values(int_col, row_indices)
                        .map(|value| value as f64)
                        .collect();
                    Ok(median_of(&mut values))
                }
                // pandas `first`/`last` return the first/last *observed* value.
                AggregateOp::First => Ok(int_values(int_col, row_indices)
                    .next()
                    .map(|value| value as f64)),
                AggregateOp::Last => Ok(int_values(int_col, row_indices)
                    .last()
                    .map(|value| value as f64)),
                AggregateOp::Count | AggregateOp::Custom => Ok(None),
            },
            Column::Float64(float_col) => match op {
                AggregateOp::Sum => {
                    // Kahan compensated summation for numerical stability.
                    let mut sum = 0.0;
                    let mut compensation = 0.0;
                    for value in float_values(float_col, row_indices) {
                        let y = value - compensation;
                        let t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                    }
                    Ok(Some(sum))
                }
                AggregateOp::Mean => {
                    let mut sum = 0.0;
                    let mut compensation = 0.0;
                    let mut count: usize = 0;
                    for value in float_values(float_col, row_indices) {
                        let y = value - compensation;
                        let t = sum + y;
                        compensation = (t - sum) - y;
                        sum = t;
                        count += 1;
                    }
                    if count == 0 {
                        Ok(None)
                    } else {
                        Ok(Some(sum / count as f64))
                    }
                }
                AggregateOp::Min => Ok(float_values(float_col, row_indices)
                    .fold(None, |acc: Option<f64>, value| {
                        Some(acc.map_or(value, |current| current.min(value)))
                    })),
                AggregateOp::Max => Ok(float_values(float_col, row_indices)
                    .fold(None, |acc: Option<f64>, value| {
                        Some(acc.map_or(value, |current| current.max(value)))
                    })),
                AggregateOp::Std => {
                    let values: Vec<f64> = float_values(float_col, row_indices).collect();
                    Ok(sample_variance(&values).map(f64::sqrt))
                }
                AggregateOp::Var => {
                    let values: Vec<f64> = float_values(float_col, row_indices).collect();
                    Ok(sample_variance(&values))
                }
                AggregateOp::Median => {
                    let mut values: Vec<f64> = float_values(float_col, row_indices).collect();
                    Ok(median_of(&mut values))
                }
                AggregateOp::First => Ok(float_values(float_col, row_indices).next()),
                AggregateOp::Last => Ok(float_values(float_col, row_indices).last()),
                AggregateOp::Count | AggregateOp::Custom => Ok(None),
            },
            other => Err(Error::OperationFailed(format!(
                "Aggregation operation {:?} is not supported for column type {:?}",
                op,
                other.column_type()
            ))),
        }
    }
}

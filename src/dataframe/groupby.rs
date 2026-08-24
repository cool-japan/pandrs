//! Enhanced GroupBy functionality for DataFrames with pandas-like named aggregations
//!
//! This module provides comprehensive groupby operations with support for:
//! - Named aggregations (similar to pandas .agg({'col': {'alias': 'func'}}) syntax)
//! - Multiple aggregation functions per column
//! - Custom aggregation functions
//! - Multi-level column names for results

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Enumeration representing aggregation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    /// Sum aggregation
    Sum,
    /// Mean (average) aggregation
    Mean,
    /// Minimum value aggregation
    Min,
    /// Maximum value aggregation
    Max,
    /// Count of non-null values
    Count,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Median value
    Median,
    /// First value
    First,
    /// Last value
    Last,
    /// Count of unique values
    Nunique,
    /// Custom aggregation function
    Custom,
}

impl AggFunc {
    /// Get the string representation of the aggregation function
    pub fn as_str(&self) -> &'static str {
        match self {
            AggFunc::Sum => "sum",
            AggFunc::Mean => "mean",
            AggFunc::Min => "min",
            AggFunc::Max => "max",
            AggFunc::Count => "count",
            AggFunc::Std => "std",
            AggFunc::Var => "var",
            AggFunc::Median => "median",
            AggFunc::First => "first",
            AggFunc::Last => "last",
            AggFunc::Nunique => "nunique",
            AggFunc::Custom => "custom",
        }
    }
}

/// Type for custom aggregation functions
pub type CustomAggFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Specification for a named aggregation
#[derive(Clone)]
pub struct NamedAgg {
    /// Column to aggregate
    pub column: String,
    /// Aggregation function to apply
    pub func: AggFunc,
    /// Alias for the result column
    pub alias: String,
    /// Optional custom function (required when func is Custom)
    pub custom_fn: Option<CustomAggFn>,
}

impl std::fmt::Debug for NamedAgg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedAgg")
            .field("column", &self.column)
            .field("func", &self.func)
            .field("alias", &self.alias)
            .field(
                "custom_fn",
                &self.custom_fn.as_ref().map(|_| "<custom_function>"),
            )
            .finish()
    }
}

impl NamedAgg {
    /// Create a new named aggregation
    pub fn new(column: String, func: AggFunc, alias: String) -> Self {
        Self {
            column,
            func,
            alias,
            custom_fn: None,
        }
    }

    /// Create a named aggregation with a custom function
    pub fn custom<F>(column: String, alias: String, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        Self {
            column,
            func: AggFunc::Custom,
            alias,
            custom_fn: Some(Arc::new(func)),
        }
    }
}

/// Builder for creating multiple named aggregations for a single column
pub struct ColumnAggBuilder {
    column: String,
    aggregations: Vec<(AggFunc, String, Option<CustomAggFn>)>,
}

impl std::fmt::Debug for ColumnAggBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnAggBuilder")
            .field("column", &self.column)
            .field(
                "aggregations",
                &self
                    .aggregations
                    .iter()
                    .map(|(func, alias, custom_fn)| {
                        (func, alias, custom_fn.as_ref().map(|_| "<custom_function>"))
                    })
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ColumnAggBuilder {
    /// Create a new column aggregation builder
    pub fn new(column: String) -> Self {
        Self {
            column,
            aggregations: Vec::new(),
        }
    }

    /// Add a standard aggregation function with an alias
    pub fn agg(mut self, func: AggFunc, alias: String) -> Self {
        self.aggregations.push((func, alias, None));
        self
    }

    /// Add a custom aggregation function with an alias
    pub fn custom<F>(mut self, alias: String, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.aggregations
            .push((AggFunc::Custom, alias, Some(Arc::new(func))));
        self
    }

    /// Build the named aggregations
    pub fn build(self) -> Vec<NamedAgg> {
        self.aggregations
            .into_iter()
            .map(|(func, alias, custom_fn)| NamedAgg {
                column: self.column.clone(),
                func,
                alias,
                custom_fn,
            })
            .collect()
    }
}

/// DataFrame GroupBy with enhanced functionality
#[derive(Debug)]
pub struct DataFrameGroupBy {
    /// Original DataFrame
    df: DataFrame,
    /// Grouping column(s)
    group_by_columns: Vec<String>,
    /// Grouped indices for each group key
    groups: HashMap<Vec<String>, Vec<usize>>,
}

impl DataFrameGroupBy {
    /// Create a new DataFrame GroupBy
    pub fn new(df: DataFrame, group_by_columns: Vec<String>) -> Result<Self> {
        // Verify that all grouping columns exist
        for col in &group_by_columns {
            if !df.contains_column(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        // Materialize every grouping column ONCE up front. Previously each
        // column was re-fetched (a full `Vec<String>` clone of the column)
        // on every row of the outer loop below -- O(rows) work repeated
        // `rows` times, i.e. O(rows^2 * group_columns) total. Measured:
        // 500 rows 5.153ms / 1,000 21.512ms / 2,000 122.275ms / 4,000
        // 582.751ms / 16,000 ~14,300ms (~O(n^2.06-2.51)). Hoisting to
        // O(rows * group_columns) mirrors the same fix already applied to
        // `ApplyExt::apply` (Axis::Row) in `src/dataframe/apply.rs`.
        let group_column_values: Vec<Vec<String>> = group_by_columns
            .iter()
            .map(|col_name| df.get_column_string_values(col_name))
            .collect::<Result<Vec<_>>>()?;

        // Create groups based on the grouping columns
        let row_count = df.row_count();
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..row_count {
            let mut key = Vec::with_capacity(group_by_columns.len());

            for col_values in &group_column_values {
                if row_idx < col_values.len() {
                    key.push(col_values[row_idx].clone());
                } else {
                    key.push("NULL".to_string());
                }
            }

            groups.entry(key).or_default().push(row_idx);
        }

        Ok(Self {
            df,
            group_by_columns,
            groups,
        })
    }

    /// Get the number of groups
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }

    /// Get the size of each group
    pub fn size(&self) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        let mut group_keys = Vec::new();
        let mut sizes = Vec::new();

        for (key, indices) in &self.groups {
            let key_str = key.join("_");
            group_keys.push(key_str);
            sizes.push(indices.len().to_string());
        }

        let group_series = Series::new(group_keys, Some("group".to_string()))?;
        let size_series = Series::new(sizes, Some("size".to_string()))?;

        result.add_column("group".to_string(), group_series)?;
        result.add_column("size".to_string(), size_series)?;

        Ok(result)
    }

    /// Apply named aggregations (pandas-like .agg() functionality)
    pub fn agg(&self, named_aggs: Vec<NamedAgg>) -> Result<DataFrame> {
        if named_aggs.is_empty() {
            return Err(Error::InvalidValue(
                "At least one aggregation must be specified".to_string(),
            ));
        }

        // Verify all columns exist
        for agg in &named_aggs {
            if !self.df.contains_column(&agg.column) {
                return Err(Error::ColumnNotFound(agg.column.clone()));
            }
        }

        let mut result = DataFrame::new();

        // Create columns for group keys
        for (i, group_col) in self.group_by_columns.iter().enumerate() {
            let mut group_values = Vec::new();
            for key in self.groups.keys() {
                group_values.push(key[i].clone());
            }
            let group_series = Series::new(group_values, Some(group_col.clone()))?;
            result.add_column(group_col.clone(), group_series)?;
        }

        // Materialize each distinct target column ONCE, not once per group.
        // Previously `calculate_aggregation` called
        // `self.df.get_column_string_values(column)` itself, from inside
        // the `for indices in self.groups.values()` loop below -- so a
        // column referenced by one named aggregation was re-fetched (a full
        // column clone) once per group, i.e. O(groups * rows) instead of
        // O(rows) for that column.
        let mut column_cache: HashMap<&str, Vec<String>> = HashMap::new();
        for agg in &named_aggs {
            if !column_cache.contains_key(agg.column.as_str()) {
                let values = self.df.get_column_string_values(&agg.column)?;
                column_cache.insert(agg.column.as_str(), values);
            }
        }

        // Apply each named aggregation
        for agg in &named_aggs {
            let mut agg_values = Vec::new();
            let column_values = column_cache.get(agg.column.as_str()).ok_or_else(|| {
                Error::InvalidValue(format!(
                    "internal error: column '{}' was not pre-materialized in column_cache",
                    agg.column
                ))
            })?;

            for indices in self.groups.values() {
                let agg_result = self.calculate_aggregation(
                    &agg.column,
                    agg.func,
                    indices,
                    &agg.custom_fn,
                    column_values,
                )?;
                agg_values.push(agg_result.to_string());
            }

            let agg_series = Series::new(agg_values, Some(agg.alias.clone()))?;
            result.add_column(agg.alias.clone(), agg_series)?;
        }

        Ok(result)
    }

    /// Apply multiple aggregations using a builder pattern
    pub fn agg_multi(&self, builders: Vec<ColumnAggBuilder>) -> Result<DataFrame> {
        let mut named_aggs = Vec::new();

        for builder in builders {
            named_aggs.extend(builder.build());
        }

        self.agg(named_aggs)
    }

    /// Apply aggregations using a HashMap specification (similar to pandas)
    /// Example: {"price": [("mean", "avg_price"), ("std", "price_std")]}
    pub fn agg_dict(&self, agg_spec: HashMap<String, Vec<(AggFunc, String)>>) -> Result<DataFrame> {
        let mut named_aggs = Vec::new();

        for (column, specs) in agg_spec {
            for (func, alias) in specs {
                named_aggs.push(NamedAgg::new(column.clone(), func, alias));
            }
        }

        self.agg(named_aggs)
    }

    /// Convenience method for simple aggregations
    pub fn sum(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Sum, format!("{}_sum", column));
        self.agg(vec![agg])
    }

    /// Convenience method for mean aggregation
    pub fn mean(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Mean,
            format!("{}_mean", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for count aggregation
    pub fn count(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Count,
            format!("{}_count", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for min aggregation
    pub fn min(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Min, format!("{}_min", column));
        self.agg(vec![agg])
    }

    /// Convenience method for max aggregation
    pub fn max(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Max, format!("{}_max", column));
        self.agg(vec![agg])
    }

    /// Convenience method for std aggregation
    pub fn std(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Std, format!("{}_std", column));
        self.agg(vec![agg])
    }

    /// Convenience method for var aggregation
    pub fn var(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(column.to_string(), AggFunc::Var, format!("{}_var", column));
        self.agg(vec![agg])
    }

    /// Convenience method for median aggregation
    pub fn median(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Median,
            format!("{}_median", column),
        );
        self.agg(vec![agg])
    }

    /// Convenience method for nunique aggregation
    pub fn nunique(&self, column: &str) -> Result<DataFrame> {
        let agg = NamedAgg::new(
            column.to_string(),
            AggFunc::Nunique,
            format!("{}_nunique", column),
        );
        self.agg(vec![agg])
    }

    /// Apply a custom aggregation function
    pub fn apply<F>(&self, column: &str, alias: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        let agg = NamedAgg::custom(column.to_string(), alias.to_string(), func);
        self.agg(vec![agg])
    }

    /// Filter groups based on a condition
    pub fn filter<F>(&self, condition: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> bool,
    {
        let mut filtered_indices = Vec::new();

        for indices in self.groups.values() {
            // Create a subset DataFrame for this group
            let group_df = self.create_group_dataframe(indices)?;

            // Apply the condition
            if condition(&group_df) {
                filtered_indices.extend(indices);
            }
        }

        // Create result DataFrame with filtered rows
        self.create_subset_dataframe(&filtered_indices)
    }

    /// Transform groups using a function.
    ///
    /// Like pandas' `GroupBy.transform`, `func` must return a DataFrame
    /// with exactly one output row per input row of the group it was given
    /// (a reduction like `.mean()` must be broadcast back to the group's
    /// shape by `func` itself). The result is realigned to `self.df`'s
    /// original row order.
    pub fn transform<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> Result<DataFrame>,
    {
        let row_count = self.df.row_count();
        let mut transformed_parts = Vec::new();
        // Original row index that each row of the (group-order)
        // concatenation of `transformed_parts` corresponds to.
        let mut original_index_of: Vec<usize> = Vec::with_capacity(row_count);

        for indices in self.groups.values() {
            let group_df = self.create_group_dataframe(indices)?;
            let transformed = func(&group_df)?;

            if transformed.row_count() != indices.len() {
                return Err(Error::InvalidValue(format!(
                    "transform function must return exactly one row per input row: \
                     group had {} row(s) but the transform produced {}",
                    indices.len(),
                    transformed.row_count()
                )));
            }

            original_index_of.extend_from_slice(indices);
            transformed_parts.push(transformed);
        }

        // `self.groups` (a HashMap) iterates in an arbitrary, not
        // input-order, sequence, so the straightforward concatenation of
        // `transformed_parts` is ordered by that arbitrary group order --
        // not by `self.df`'s original row order. pandas' `transform`
        // always returns a result aligned to the input's row order, so
        // re-project every concatenated row back to its original position
        // rather than handing back this group-shuffled order.
        let concatenated = self.concatenate_dataframes(transformed_parts)?;

        if row_count == 0 {
            return Ok(concatenated);
        }

        // `self.groups` partitions every row index in `0..row_count`
        // exactly once, so `original_index_of` is a permutation of
        // `0..row_count` and this inversion is total.
        let mut position_of_original = vec![0usize; row_count];
        for (position, &original_idx) in original_index_of.iter().enumerate() {
            position_of_original[original_idx] = position;
        }

        let mut result = DataFrame::new();
        for col_name in concatenated.column_names() {
            let col_values = concatenated.get_column_string_values(col_name)?;
            let mut realigned = Vec::with_capacity(row_count);
            for orig_idx in 0..row_count {
                let position = position_of_original[orig_idx];
                let value = col_values.get(position).ok_or_else(|| {
                    Error::InvalidValue(format!(
                        "transform realignment out of bounds: position {} for column '{}' ({} rows)",
                        position,
                        col_name,
                        col_values.len()
                    ))
                })?;
                realigned.push(value.clone());
            }
            let series = Series::new(realigned, Some(col_name.to_string()))?;
            result.add_column(col_name.to_string(), series)?;
        }

        Ok(result)
    }

    /// Calculate aggregation for a column and group.
    ///
    /// `column_values` is the full column, already materialized as strings
    /// and indexed by original row index -- callers hoist this once per
    /// column (see `agg`) rather than re-fetching it for every group, which
    /// used to make aggregation O(groups * rows) instead of O(rows) per
    /// column.
    fn calculate_aggregation(
        &self,
        column: &str,
        func: AggFunc,
        indices: &[usize],
        custom_fn: &Option<CustomAggFn>,
        column_values: &[String],
    ) -> Result<f64> {
        // `Count` and `Nunique` describe the rows themselves (how many rows,
        // how many distinct rendered values) rather than a numeric summary, so
        // -- working on any column, numeric or not, as they do in pandas --
        // they never require a successful f64 parse. Unlike the numeric
        // reductions below, NEITHER applies the `skipna` NA exclusion:
        //
        //   * base `Count` reports the group's ROW count (its `size`) and
        //     DELIBERATELY differs from the typed/split `group_by` path, whose
        //     `count_valid` returns the number of non-missing (non-`NaN`/
        //     non-NULL) observations. The base column is materialized as plain
        //     strings with no accompanying null mask, so there is no reliable
        //     per-type NA test to reproduce that here -- e.g. for a `String`
        //     column no string value is unambiguously "missing". Callers that
        //     need the non-missing count on a numeric column should use the
        //     typed path (or `count` a column after dropping its NAs).
        //   * base `Nunique` counts distinct RENDERED cell strings, so a
        //     missing numeric cell's "NaN" text is counted as one distinct
        //     value where pandas' `nunique()` would drop it. There is no
        //     `AggregateOp::Nunique` on the typed path, so there is no parity
        //     target to align this against; it is left as a known divergence.
        match func {
            AggFunc::Count => return Ok(indices.len() as f64),
            AggFunc::Nunique => {
                let mut unique: Vec<&String> = indices
                    .iter()
                    .filter_map(|&idx| column_values.get(idx))
                    .collect();
                unique.sort();
                unique.dedup();
                return Ok(unique.len() as f64);
            }
            _ => {}
        }

        // Every remaining aggregation (Sum/Mean/Min/Max/Std/Var/Median/
        // First/Last/Custom) is inherently numeric. A cell that fails to
        // parse as f64 is a genuine data problem for a numeric aggregation
        // -- e.g. a truly non-numeric string in what's supposed to be a
        // numeric column -- so fail loudly instead of silently dropping it
        // from the group via `.parse().ok()` (which used to shrink the
        // group without telling anyone and skew every remaining
        // aggregate), mirroring `DataFrame::get_column_numeric_values`'s
        // existing convention.
        //
        // A legitimate *missing* numeric value renders as the literal text
        // "NaN" (see `get_column_string_values`), which `str::parse::<f64>`
        // accepts and round-trips to `f64::NAN`. Such missing cells are
        // EXCLUDED from the accumulation below (pandas `skipna=True`),
        // exactly as the typed/split `group_by` path excludes them: its
        // `float_values` helper filters `!value.is_nan()` and
        // `Float64Column::mean` divides by the count of non-`NaN`
        // observations. Excluding here, at construction, makes every numeric
        // reduction that follows (mean, sum, min, max, std, var, median,
        // first, last, custom) skip missing values uniformly -- so e.g.
        // mean over [10, NaN, 20] is 15.0 (the NaN is dropped from BOTH the
        // running sum AND the divisor), not the propagated `NaN` this path
        // used to return, and an all-`NaN` `min`/`max` no longer leaks the
        // `+-INF` fold seed. Only `NaN` is treated as missing: legitimate
        // `+-inf` observations are kept, matching the typed path (whose
        // filter is `!is_nan()`, not `is_finite()`).
        let mut group_values = Vec::with_capacity(indices.len());
        for &idx in indices {
            let raw = column_values.get(idx).ok_or_else(|| {
                Error::InvalidValue(format!(
                    "Row index {} out of bounds for column '{}' ({} rows)",
                    idx,
                    column,
                    column_values.len()
                ))
            })?;
            let value = raw.trim().parse::<f64>().map_err(|_| {
                Error::InvalidValue(format!(
                    "Value '{}' in column '{}' cannot be converted to numeric for aggregation '{}'",
                    raw,
                    column,
                    func.as_str()
                ))
            })?;
            // `NaN` is the base DataFrame's missing-value marker: skip it so
            // it is excluded from every numeric aggregation (skipna=True),
            // consistent with the typed path's null semantics.
            if value.is_nan() {
                continue;
            }
            group_values.push(value);
        }

        if group_values.is_empty() {
            // Reached when the group is entirely missing: every cell parsed
            // to the `NaN` marker and was skipped above (an all-NA group).
            // (`DataFrameGroupBy::new` never creates a zero-row group -- every
            // key in `self.groups` had at least one row index pushed onto it
            // -- so an empty `indices` is not the cause.) Honor pandas' own
            // per-function convention for an all-missing group, matching the
            // typed/split path's `Ok(Some(0.0))` for `sum` and `Ok(None)` ->
            // NULL cell for the undefined reductions:
            //   * `Sum` of nothing is the additive identity, 0.0.
            //   * `Custom` still runs, on the empty slice -- the typed path's
            //     `aggregate_custom_impl` calls `custom_fn(&[])` for an
            //     all-missing group rather than short-circuiting, so a caller
            //     counting observations sees 0, not a fabricated `NaN`.
            //   * every other reduction is undefined for an all-missing group
            //     -> `f64::NAN`, the base DataFrame's missing marker (the
            //     typed path emits a real NULL; both mean "no value", never a
            //     misleading 0.0). This is also what turns an all-`NaN`
            //     `min`/`max` into `NaN` instead of the `+-INF` the numeric
            //     `fold` seed would otherwise leak.
            return match func {
                AggFunc::Sum => Ok(0.0),
                AggFunc::Custom => {
                    if let Some(custom_fn) = custom_fn {
                        Ok(custom_fn(&group_values))
                    } else {
                        Err(Error::InvalidValue(
                            "Custom function not provided".to_string(),
                        ))
                    }
                }
                _ => Ok(f64::NAN),
            };
        }

        match func {
            AggFunc::Sum => Ok(group_values.iter().sum()),
            AggFunc::Mean => Ok(group_values.iter().sum::<f64>() / group_values.len() as f64),
            AggFunc::Min => Ok(group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            AggFunc::Max => Ok(group_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            AggFunc::Std => {
                if group_values.len() <= 1 {
                    // pandas: std/var with the default ddof=1 is undefined
                    // (NaN) for a single observation, not 0.0.
                    Ok(f64::NAN)
                } else {
                    let mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
                    let variance = group_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (group_values.len() - 1) as f64;
                    Ok(variance.sqrt())
                }
            }
            AggFunc::Var => {
                if group_values.len() <= 1 {
                    Ok(f64::NAN)
                } else {
                    let mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
                    Ok(group_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (group_values.len() - 1) as f64)
                }
            }
            AggFunc::Median => {
                let mut sorted = group_values;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len() % 2 == 0 {
                    Ok((sorted[mid - 1] + sorted[mid]) / 2.0)
                } else {
                    Ok(sorted[mid])
                }
            }
            AggFunc::First => group_values.first().copied().ok_or_else(|| {
                Error::InvalidValue("Cannot compute First aggregation on empty group".to_string())
            }),
            AggFunc::Last => group_values.last().copied().ok_or_else(|| {
                Error::InvalidValue("Cannot compute Last aggregation on empty group".to_string())
            }),
            AggFunc::Custom => {
                if let Some(custom_fn) = custom_fn {
                    Ok(custom_fn(&group_values))
                } else {
                    Err(Error::InvalidValue(
                        "Custom function not provided".to_string(),
                    ))
                }
            }
            AggFunc::Count | AggFunc::Nunique => {
                // Unreachable via the public API: both are returned early
                // above, before any numeric parsing. Kept as a real `Err`
                // rather than `unreachable!()` so a future edit that
                // removes the early return fails safely instead of
                // panicking.
                Err(Error::InvalidValue(format!(
                    "internal error: AggFunc::{:?} should have been handled before numeric parsing",
                    func
                )))
            }
        }
    }

    /// Create a DataFrame for a specific group
    fn create_group_dataframe(&self, indices: &[usize]) -> Result<DataFrame> {
        let mut group_df = DataFrame::new();

        for col_name in self.df.column_names() {
            let column_values = self.df.get_column_string_values(&col_name)?;
            let group_values: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let group_series = Series::new(group_values, Some(col_name.clone()))?;
            group_df.add_column(col_name.clone(), group_series)?;
        }

        Ok(group_df)
    }

    /// Create a subset DataFrame with specific row indices
    fn create_subset_dataframe(&self, indices: &[usize]) -> Result<DataFrame> {
        let mut subset_df = DataFrame::new();

        for col_name in self.df.column_names() {
            let column_values = self.df.get_column_string_values(&col_name)?;
            let subset_values: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let subset_series = Series::new(subset_values, Some(col_name.clone()))?;
            subset_df.add_column(col_name.clone(), subset_series)?;
        }

        Ok(subset_df)
    }

    /// Concatenate multiple DataFrames
    fn concatenate_dataframes(&self, dataframes: Vec<DataFrame>) -> Result<DataFrame> {
        if dataframes.is_empty() {
            return Ok(DataFrame::new());
        }

        let mut result = DataFrame::new();
        let first_df = &dataframes[0];

        // Every part must expose the same column set. Previously only
        // `first_df.column_names()` was consulted, so a `func` (from
        // `transform`/`filter`) that returned a different set of columns
        // for some group would have those columns silently dropped from
        // the output -- with no indication anything was lost.
        let expected_columns: std::collections::HashSet<&String> =
            first_df.column_names().iter().collect();
        for (i, df) in dataframes.iter().enumerate().skip(1) {
            let these_columns: std::collections::HashSet<&String> =
                df.column_names().iter().collect();
            if these_columns != expected_columns {
                return Err(Error::InvalidValue(format!(
                    "cannot concatenate group results with mismatched columns: \
                     part 0 has {:?}, part {} has {:?}",
                    first_df.column_names(),
                    i,
                    df.column_names()
                )));
            }
        }

        for col_name in first_df.column_names() {
            let mut all_values = Vec::new();

            for df in &dataframes {
                let column_values = df.get_column_string_values(col_name)?;
                all_values.extend(column_values);
            }

            let concat_series = Series::new(all_values, Some(col_name.to_string()))?;
            result.add_column(col_name.to_string(), concat_series)?;
        }

        Ok(result)
    }
}

/// Extension trait to add groupby functionality to DataFrame
pub trait GroupByExt {
    /// Group DataFrame by one or more columns
    fn groupby<S: AsRef<str>>(&self, columns: &[S]) -> Result<DataFrameGroupBy>;

    /// Group DataFrame by a single column (convenience method)
    fn groupby_single(&self, column: &str) -> Result<DataFrameGroupBy>;
}

impl GroupByExt for DataFrame {
    fn groupby<S: AsRef<str>>(&self, columns: &[S]) -> Result<DataFrameGroupBy> {
        let group_columns: Vec<String> = columns.iter().map(|s| s.as_ref().to_string()).collect();
        DataFrameGroupBy::new(self.clone(), group_columns)
    }

    fn groupby_single(&self, column: &str) -> Result<DataFrameGroupBy> {
        DataFrameGroupBy::new(self.clone(), vec![column.to_string()])
    }
}

// Helper macros for creating aggregation specifications

/// Create a named aggregation
#[macro_export]
macro_rules! named_agg {
    ($column:expr, $func:expr, $alias:expr) => {
        NamedAgg::new($column.to_string(), $func, $alias.to_string())
    };
}

/// Create multiple named aggregations for a column
#[macro_export]
macro_rules! column_aggs {
    ($column:expr, $(($func:expr, $alias:expr)),+) => {
        {
            let mut builder = ColumnAggBuilder::new($column.to_string());
            $(
                builder = builder.agg($func, $alias.to_string());
            )+
            builder
        }
    };
}

/// Create aggregation specification (similar to pandas)
#[macro_export]
macro_rules! agg_spec {
    ($($column:expr => [$(($func:expr, $alias:expr)),+]),+) => {
        {
            let mut spec = std::collections::HashMap::new();
            $(
                spec.insert($column.to_string(), vec![$(($func, $alias.to_string())),+]);
            )+
            spec
        }
    };
}

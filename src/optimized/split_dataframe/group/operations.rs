//! Transform, filter, and convenience methods for GroupBy operations

use std::sync::Arc;

use rayon::prelude::*;

use super::super::core::OptimizedDataFrame;
use super::types::{AggregateOp, CustomAggregation, GroupBy};
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

/// Vertically concatenate transformed group frames.
///
/// The first frame defines the schema; every other frame must agree on column
/// count and column types. NULL values are carried through the concatenation
/// (the data vector keeps a placeholder slot for each missing entry so that the
/// null mask stays aligned with the values).
fn concat_group_frames(frames: &[OptimizedDataFrame]) -> Result<OptimizedDataFrame> {
    let Some(template) = frames.first() else {
        return Ok(OptimizedDataFrame::new());
    };

    let mut result = OptimizedDataFrame::new();

    for (col_idx, template_col) in template.columns.iter().enumerate() {
        let col_name = template
            .column_names
            .get(col_idx)
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "transformed group has {} columns but only {} column names",
                    template.columns.len(),
                    template.column_names.len()
                ))
            })?
            .clone();

        // Collect every frame's column, requiring a matching type.
        let mut columns = Vec::with_capacity(frames.len());
        for frame in frames {
            let column = frame.columns.get(col_idx).ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "transformed group is missing column '{}'",
                    col_name
                ))
            })?;
            if column.column_type() != template_col.column_type() {
                return Err(Error::InvalidOperation(format!(
                    "transformed groups disagree on the type of column '{}': {:?} vs {:?}",
                    col_name,
                    template_col.column_type(),
                    column.column_type()
                )));
            }
            columns.push(column);
        }

        let column = match template_col {
            Column::Int64(_) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for column in &columns {
                    if let Column::Int64(col) = column {
                        for i in 0..col.len() {
                            let value = col.get(i)?;
                            values.push(value.unwrap_or_default());
                            nulls.push(value.is_none());
                        }
                    }
                }
                if nulls.iter().any(|&is_null| is_null) {
                    Column::Int64(Int64Column::with_nulls(values, nulls))
                } else {
                    Column::Int64(Int64Column::new(values))
                }
            }
            Column::Float64(_) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for column in &columns {
                    if let Column::Float64(col) = column {
                        for i in 0..col.len() {
                            let value = col.get(i)?;
                            values.push(value.unwrap_or_default());
                            nulls.push(value.is_none());
                        }
                    }
                }
                if nulls.iter().any(|&is_null| is_null) {
                    Column::Float64(Float64Column::with_nulls(values, nulls))
                } else {
                    Column::Float64(Float64Column::new(values))
                }
            }
            Column::String(_) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for column in &columns {
                    if let Column::String(col) = column {
                        for i in 0..col.len() {
                            let value = col.get(i)?;
                            values.push(value.unwrap_or_default().to_string());
                            nulls.push(value.is_none());
                        }
                    }
                }
                if nulls.iter().any(|&is_null| is_null) {
                    Column::String(StringColumn::with_nulls(values, nulls))
                } else {
                    Column::String(StringColumn::new(values))
                }
            }
            Column::Boolean(_) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for column in &columns {
                    if let Column::Boolean(col) = column {
                        for i in 0..col.len() {
                            let value = col.get(i)?;
                            values.push(value.unwrap_or_default());
                            nulls.push(value.is_none());
                        }
                    }
                }
                if nulls.iter().any(|&is_null| is_null) {
                    Column::Boolean(BooleanColumn::with_nulls(values, nulls))
                } else {
                    Column::Boolean(BooleanColumn::new(values))
                }
            }
        };

        result.add_column(col_name, column)?;
    }

    Ok(result)
}

impl<'a> GroupBy<'a> {
    /// Apply a custom aggregation function to a column in parallel
    ///
    /// # Arguments
    /// * `column` - Column to aggregate
    /// * `result_name` - Name for the result column
    /// * `func` - Custom aggregation function
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn par_custom<F>(
        &self,
        column: &str,
        result_name: &str,
        func: F,
    ) -> Result<OptimizedDataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        let custom_fn = Arc::new(func);

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.par_aggregate_custom(vec![custom_agg])
    }

    /// Filter groups based on a predicate function
    ///
    /// Surviving rows are returned in their original row order, as pandas'
    /// `DataFrameGroupBy.filter` does.
    ///
    /// # Arguments
    /// * `filter_fn` - Function that determines if a group should be included
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing only rows from groups that satisfy the predicate
    pub fn filter<F>(&self, filter_fn: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&OptimizedDataFrame) -> bool + Send + Sync + 'static,
    {
        let filter_fn = Arc::new(filter_fn);

        // Collect indices of groups that pass the filter
        let mut filtered_indices = Vec::new();

        for (_, row_indices) in self.ordered_groups() {
            // Create a DataFrame for this group
            let group_df = self.df.filter_by_indices(row_indices)?;

            // Apply the filter function to determine if this group passes
            if filter_fn(&group_df) {
                filtered_indices.extend(row_indices.iter().copied());
            }
        }

        // Restore the original row order (group order must not leak into the result)
        filtered_indices.sort_unstable();

        // Create a new DataFrame with the filtered rows
        self.df.filter_by_indices(&filtered_indices)
    }

    /// Filter groups based on a predicate function in parallel
    ///
    /// # Arguments
    /// * `filter_fn` - Function that determines if a group should be included
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing only rows from groups that satisfy the predicate
    pub fn par_filter<F>(&self, filter_fn: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&OptimizedDataFrame) -> bool + Send + Sync + 'static,
    {
        let filter_fn = Arc::new(filter_fn);

        // Optimization threshold - only use parallel for enough groups
        const PARALLEL_THRESHOLD: usize = 8;

        if self.groups.len() < PARALLEL_THRESHOLD {
            return self.filter(move |df| filter_fn(df));
        }

        let ordered = self.ordered_groups();

        // Process groups in parallel; failures propagate instead of silently
        // dropping the group's rows from the result.
        let kept: Vec<Option<&Vec<usize>>> = ordered
            .par_iter()
            .map(|&(_, row_indices)| {
                let group_df = self.df.filter_by_indices(row_indices)?;
                Ok(if filter_fn(&group_df) {
                    Some(row_indices)
                } else {
                    None
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut filtered_indices: Vec<usize> = Vec::new();
        for row_indices in kept.into_iter().flatten() {
            filtered_indices.extend(row_indices.iter().copied());
        }
        filtered_indices.sort_unstable();

        // Create a new DataFrame with the filtered rows
        self.df.filter_by_indices(&filtered_indices)
    }

    /// Transform each group with a given function
    ///
    /// Groups are processed in ascending key order, so the concatenated result
    /// is reproducible instead of depending on hash-map iteration order.
    ///
    /// # Arguments
    /// * `transform_fn` - Function that transforms each group
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing transformed data
    pub fn transform<F>(&self, transform_fn: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync + 'static,
    {
        let transform_fn = Arc::new(transform_fn);

        // Apply transformation to each group, in deterministic key order
        let mut transformed_dfs = Vec::with_capacity(self.groups.len());
        for (_, row_indices) in self.ordered_groups() {
            let group_df = self.df.filter_by_indices(row_indices)?;
            transformed_dfs.push(transform_fn(&group_df)?);
        }

        concat_group_frames(&transformed_dfs)
    }

    /// Transform each group with a given function in parallel
    ///
    /// # Arguments
    /// * `transform_fn` - Function that transforms each group
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing transformed data
    pub fn par_transform<F>(&self, transform_fn: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync + 'static,
    {
        let transform_fn = Arc::new(transform_fn);

        // Optimization threshold - only use parallel for enough groups
        const PARALLEL_THRESHOLD: usize = 8;

        if self.groups.len() < PARALLEL_THRESHOLD {
            return self.transform(move |df| transform_fn(df));
        }

        let ordered = self.ordered_groups();

        // Each group is transformed exactly once and the results stay in group
        // order, so the concatenation matches the serial implementation.
        let transformed_dfs: Vec<OptimizedDataFrame> = ordered
            .par_iter()
            .map(|&(_, row_indices)| {
                let group_df = self.df.filter_by_indices(row_indices)?;
                transform_fn(&group_df)
            })
            .collect::<Result<Vec<_>>>()?;

        concat_group_frames(&transformed_dfs)
    }

    /// Aggregation shortcut method: Sum
    pub fn sum(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_sum", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Sum, agg_name)])
    }

    /// Aggregation shortcut method: Mean
    pub fn mean(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_mean", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Mean, agg_name)])
    }

    /// Aggregation shortcut method: Minimum value
    pub fn min(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_min", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Min, agg_name)])
    }

    /// Aggregation shortcut method: Maximum value
    pub fn max(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_max", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Max, agg_name)])
    }

    /// Aggregation shortcut method: Count
    pub fn count(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_count", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Count, agg_name)])
    }

    /// Aggregation shortcut method: Standard deviation
    pub fn std(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_std", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Std, agg_name)])
    }

    /// Aggregation shortcut method: Variance
    pub fn var(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_var", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Var, agg_name)])
    }

    /// Aggregation shortcut method: Median
    pub fn median(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_median", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Median, agg_name)])
    }

    /// Aggregation shortcut method: First value
    pub fn first(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_first", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::First, agg_name)])
    }

    /// Aggregation shortcut method: Last value
    pub fn last(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_last", column);
        self.aggregate(vec![(column.to_string(), AggregateOp::Last, agg_name)])
    }

    /// Apply multiple aggregation operations at once
    pub fn agg(&self, aggs: &[(&str, AggregateOp)]) -> Result<OptimizedDataFrame> {
        let aggregations = aggs
            .iter()
            .map(|(col, op)| {
                let op_name = match op {
                    AggregateOp::Sum => "sum",
                    AggregateOp::Mean => "mean",
                    AggregateOp::Min => "min",
                    AggregateOp::Max => "max",
                    AggregateOp::Count => "count",
                    AggregateOp::Std => "std",
                    AggregateOp::Var => "var",
                    AggregateOp::Median => "median",
                    AggregateOp::First => "first",
                    AggregateOp::Last => "last",
                    AggregateOp::Custom => "custom",
                };
                let agg_name = format!("{}_{}", col, op_name);
                (col.to_string(), *op, agg_name)
            })
            .collect::<Vec<_>>();

        self.aggregate(aggregations)
    }

    /// Apply multiple aggregation operations at once in parallel
    pub fn par_agg(&self, aggs: &[(&str, AggregateOp)]) -> Result<OptimizedDataFrame> {
        let aggregations = aggs
            .iter()
            .map(|(col, op)| {
                let op_name = match op {
                    AggregateOp::Sum => "sum",
                    AggregateOp::Mean => "mean",
                    AggregateOp::Min => "min",
                    AggregateOp::Max => "max",
                    AggregateOp::Count => "count",
                    AggregateOp::Std => "std",
                    AggregateOp::Var => "var",
                    AggregateOp::Median => "median",
                    AggregateOp::First => "first",
                    AggregateOp::Last => "last",
                    AggregateOp::Custom => "custom",
                };
                let agg_name = format!("{}_{}", col, op_name);
                (col.to_string(), *op, agg_name)
            })
            .collect::<Vec<_>>();

        self.par_aggregate(aggregations)
    }

    /// Parallel version of sum aggregation
    pub fn par_sum(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_sum", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Sum, agg_name)])
    }

    /// Parallel version of mean aggregation
    pub fn par_mean(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_mean", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Mean, agg_name)])
    }

    /// Parallel version of min aggregation
    pub fn par_min(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_min", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Min, agg_name)])
    }

    /// Parallel version of max aggregation
    pub fn par_max(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_max", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Max, agg_name)])
    }

    /// Parallel version of count aggregation
    pub fn par_count(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_count", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Count, agg_name)])
    }

    /// Parallel version of std aggregation
    pub fn par_std(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_std", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Std, agg_name)])
    }

    /// Parallel version of var aggregation
    pub fn par_var(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_var", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Var, agg_name)])
    }

    /// Parallel version of median aggregation
    pub fn par_median(&self, column: &str) -> Result<OptimizedDataFrame> {
        let agg_name = format!("{}_median", column);
        self.par_aggregate(vec![(column.to_string(), AggregateOp::Median, agg_name)])
    }

    /// Apply a custom aggregation function to a column
    ///
    /// # Arguments
    /// * `column` - Column to aggregate
    /// * `result_name` - Name for the result column
    /// * `func` - Custom aggregation function
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn custom<F>(&self, column: &str, result_name: &str, func: F) -> Result<OptimizedDataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        let custom_fn = Arc::new(func);

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }
}

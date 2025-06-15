//! Transform, filter, and convenience methods for GroupBy operations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use super::super::core::OptimizedDataFrame;
use super::types::{AggregateFn, AggregateOp, CustomAggregation, GroupBy};
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::Result;

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

        for (_, row_indices) in &self.groups {
            // Create a DataFrame for this group
            let group_df = self.df.filter_by_indices(row_indices)?;

            // Apply the filter function to determine if this group passes
            if filter_fn(&group_df) {
                filtered_indices.extend(row_indices.iter().copied());
            }
        }

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

        // Create group keys and indices lists
        let mut group_keys = Vec::with_capacity(self.groups.len());
        let mut row_indices_list = Vec::with_capacity(self.groups.len());

        for (key, indices) in &self.groups {
            group_keys.push(key.clone());
            row_indices_list.push(indices.clone());
        }

        // Process groups in parallel to identify those that pass the filter
        let filtered_indices = Mutex::new(Vec::new());

        group_keys
            .into_par_iter()
            .zip(row_indices_list.into_par_iter())
            .for_each(|(_, row_indices)| {
                // Create a DataFrame for this group
                if let Ok(group_df) = self.df.filter_by_indices(&row_indices) {
                    // Apply the filter function to determine if this group passes
                    if filter_fn(&group_df) {
                        let mut indices = filtered_indices.lock().unwrap();
                        indices.extend(row_indices.iter().copied());
                    }
                }
            });

        // Create a new DataFrame with the filtered rows
        let indices = filtered_indices.into_inner().unwrap();
        self.df.filter_by_indices(&indices)
    }

    /// Transform each group with a given function
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

        // Collect all transformed DataFrames first
        let mut transformed_dfs = Vec::new();

        // Apply transformation to each group
        for (_, row_indices) in &self.groups {
            // Create a DataFrame for this group
            let group_df = self.df.filter_by_indices(row_indices)?;

            // Apply the transformation function
            let transformed = transform_fn(&group_df)?;
            transformed_dfs.push(transformed);
        }

        // If no groups, return empty DataFrame
        if transformed_dfs.is_empty() {
            return Ok(OptimizedDataFrame::new());
        }

        // Use the first transformed DataFrame as template for column structure
        let template = &transformed_dfs[0];
        let mut result = OptimizedDataFrame::new();

        // Collect data for each column
        for (col_idx, template_col) in template.columns.iter().enumerate() {
            let col_name = &template.column_names[col_idx];

            match template_col {
                Column::Int64(_) => {
                    let mut all_data = Vec::new();
                    for df in &transformed_dfs {
                        if let Some(Column::Int64(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<i64> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Int64(Int64Column::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Int64(Int64Column::new(values)),
                        )?;
                    }
                }
                Column::Float64(_) => {
                    let mut all_data = Vec::new();
                    for df in &transformed_dfs {
                        if let Some(Column::Float64(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<f64> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Float64(Float64Column::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Float64(Float64Column::new(values)),
                        )?;
                    }
                }
                Column::String(_) => {
                    let mut all_data = Vec::new();
                    for df in &transformed_dfs {
                        if let Some(Column::String(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<String> = all_data
                        .iter()
                        .filter_map(|x| x.as_ref())
                        .map(|s| s.to_string())
                        .collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::String(StringColumn::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::String(StringColumn::new(values)),
                        )?;
                    }
                }
                Column::Boolean(_) => {
                    let mut all_data = Vec::new();
                    for df in &transformed_dfs {
                        if let Some(Column::Boolean(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<bool> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Boolean(BooleanColumn::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Boolean(BooleanColumn::new(values)),
                        )?;
                    }
                }
            }
        }

        Ok(result)
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

        // Create group indices lists
        let mut row_indices_list = Vec::with_capacity(self.groups.len());
        for (_, indices) in &self.groups {
            row_indices_list.push(indices.clone());
        }

        // First, transform the first group to get the structure of the result
        if row_indices_list.is_empty() {
            return Ok(OptimizedDataFrame::new());
        }

        let first_group_df = self.df.filter_by_indices(&row_indices_list[0])?;
        let first_transformed = transform_fn(&first_group_df)?;

        // Process all groups in parallel to get transformed results
        let results: Result<Vec<OptimizedDataFrame>> = row_indices_list
            .into_par_iter()
            .map(|row_indices| {
                let group_df = self.df.filter_by_indices(&row_indices)?;
                transform_fn(&group_df)
            })
            .collect();

        let mut all_transformed = results?;
        all_transformed.insert(0, first_transformed);

        // If no transformed DataFrames, return empty
        if all_transformed.is_empty() {
            return Ok(OptimizedDataFrame::new());
        }

        // Use the first DataFrame as template
        let template = &all_transformed[0];
        let mut result = OptimizedDataFrame::new();

        // Collect data for each column
        for (col_idx, template_col) in template.columns.iter().enumerate() {
            let col_name = &template.column_names[col_idx];

            match template_col {
                Column::Int64(_) => {
                    let mut all_data = Vec::new();
                    for df in &all_transformed {
                        if let Some(Column::Int64(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<i64> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Int64(Int64Column::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Int64(Int64Column::new(values)),
                        )?;
                    }
                }
                Column::Float64(_) => {
                    let mut all_data = Vec::new();
                    for df in &all_transformed {
                        if let Some(Column::Float64(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<f64> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Float64(Float64Column::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Float64(Float64Column::new(values)),
                        )?;
                    }
                }
                Column::String(_) => {
                    let mut all_data = Vec::new();
                    for df in &all_transformed {
                        if let Some(Column::String(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<String> = all_data
                        .iter()
                        .filter_map(|x| x.as_ref())
                        .map(|s| s.to_string())
                        .collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::String(StringColumn::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::String(StringColumn::new(values)),
                        )?;
                    }
                }
                Column::Boolean(_) => {
                    let mut all_data = Vec::new();
                    for df in &all_transformed {
                        if let Some(Column::Boolean(col)) = df.columns.get(col_idx) {
                            for i in 0..col.len() {
                                all_data.push(col.get(i).unwrap_or(None));
                            }
                        }
                    }
                    let values: Vec<bool> = all_data.iter().filter_map(|&x| x).collect();
                    let nulls: Vec<bool> = all_data.iter().map(|x| x.is_none()).collect();

                    if nulls.iter().any(|&is_null| is_null) {
                        result.add_column(
                            col_name.clone(),
                            Column::Boolean(BooleanColumn::with_nulls(values, nulls)),
                        )?;
                    } else {
                        result.add_column(
                            col_name.clone(),
                            Column::Boolean(BooleanColumn::new(values)),
                        )?;
                    }
                }
            }
        }

        Ok(result)
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

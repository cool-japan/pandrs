//! Grouping and aggregation functionality for OptimizedDataFrame

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use super::core::OptimizedDataFrame;
use crate::column::string_pool::StringPool;
use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, MultiIndex, StringMultiIndex};

/// Enumeration representing aggregation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count
    Count,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Median
    Median,
    /// First
    First,
    /// Last
    Last,
    /// Custom (requires custom function)
    Custom,
}

/// Type for a filter function that determines if a group should be included in the result
pub type FilterFn = Arc<dyn Fn(&OptimizedDataFrame) -> bool + Send + Sync>;

/// Type for a transform function that transforms each group's data
pub type TransformFn = Arc<dyn Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync>;

/// Type for a custom aggregation function
pub type AggregateFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Structure representing grouping results
pub struct GroupBy<'a> {
    /// Original DataFrame
    df: &'a OptimizedDataFrame,
    /// Grouping key columns
    group_by_columns: Vec<String>,
    /// Row indices for each group
    groups: HashMap<Vec<String>, Vec<usize>>,
    /// Whether to create multi-index for result
    create_multi_index: bool,
}

impl OptimizedDataFrame {
    /// Group DataFrame
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by<I, S>(&self, columns: I) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.group_by_with_options(columns, true)
    }

    /// Group DataFrame with options
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    /// * `as_multi_index` - Whether to create a multi-index for the result (when multiple columns)
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by_with_options<I, S>(
        &self,
        columns: I,
        as_multi_index: bool,
    ) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let group_by_columns: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        // Verify existence of each column
        for column in &group_by_columns {
            if !self.column_indices.contains_key(column) {
                return Err(Error::ColumnNotFound(column.clone()));
            }
        }

        // Create grouping keys
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..self.row_count {
            let mut key = Vec::with_capacity(group_by_columns.len());

            for col_name in &group_by_columns {
                let col_idx = self.column_indices[col_name];
                let col = &self.columns[col_idx];

                let key_part = match col {
                    Column::Int64(int_col) => {
                        if let Ok(Some(val)) = int_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Float64(float_col) => {
                        if let Ok(Some(val)) = float_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::String(str_col) => {
                        if let Ok(Some(val)) = str_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Boolean(bool_col) => {
                        if let Ok(Some(val)) = bool_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                };

                key.push(key_part);
            }

            groups.entry(key).or_default().push(row_idx);
        }

        // Only use multi-index when we have multiple grouping columns and option is enabled
        let create_multi_index = as_multi_index && group_by_columns.len() > 1;

        Ok(GroupBy {
            df: self,
            group_by_columns,
            groups,
            create_multi_index,
        })
    }

    /// Group DataFrame using parallel processing
    ///
    /// # Arguments
    /// * `group_by_columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<HashMap<String, Self>>` - Grouping results (map of keys and DataFrames)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};

        // Optimization threshold based on data size
        const PARALLEL_THRESHOLD: usize = 50_000;

        // Get column indices for grouping keys
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self
                .column_indices
                .get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }

        // Generate group keys and group each row's index
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // Serial processing is more efficient for small data
            let mut groups = HashMap::new();

            for row_idx in 0..self.row_count {
                // Generate group key for this row
                let mut key_parts = Vec::with_capacity(group_col_indices.len());

                for &col_idx in &group_col_indices {
                    let column = &self.columns[col_idx];
                    let part = match column {
                        Column::Int64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::Float64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::String(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::Boolean(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                    };
                    key_parts.push(part);
                }

                let group_key = key_parts.join("_");

                match groups.entry(group_key) {
                    Entry::Vacant(e) => {
                        e.insert(vec![row_idx]);
                    }
                    Entry::Occupied(mut e) => {
                        e.get_mut().push(row_idx);
                    }
                }
            }

            groups
        } else {
            // For large data, use parallel processing + lock-free approach
            // 1. Create local group maps in parallel
            // 2. Merge them
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);

            // Step 1: Create local intermediate group maps in parallel
            let local_maps: Vec<HashMap<String, Vec<usize>>> = (0..self.row_count)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_groups = HashMap::new();

                    for &row_idx in chunk {
                        // Generate group key for this row
                        let mut key_parts = Vec::with_capacity(group_col_indices.len());

                        for &col_idx in &group_col_indices {
                            let column = &self.columns[col_idx];
                            let part = match column {
                                Column::Int64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::Float64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::String(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::Boolean(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                            };
                            key_parts.push(part);
                        }

                        let group_key = key_parts.join("_");

                        match local_groups.entry(group_key) {
                            Entry::Vacant(e) => {
                                e.insert(vec![row_idx]);
                            }
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(row_idx);
                            }
                        }
                    }

                    local_groups
                })
                .collect();

            // Step 2: Merge intermediate maps
            let mut merged_groups = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    match merged_groups.entry(key) {
                        Entry::Vacant(e) => {
                            e.insert(indices);
                        }
                        Entry::Occupied(mut e) => {
                            e.get_mut().extend(indices);
                        }
                    }
                }
            }

            merged_groups
        };

        // Efficiently create DataFrames for each group
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // Use serial processing for small data or when group count is small
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                let group_df = self.filter_by_indices(&indices)?;
                result.insert(key, group_df);
            }
            result
        } else {
            // Parallelize group processing for large data
            // Process each group in parallel and safely aggregate results
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));

            // Adjust chunk size to minimize overhead
            let chunk_size = (groups.len() / rayon::current_num_threads()).max(10);

            // Create list of groups and split into chunks for parallel processing
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();

            group_items.par_chunks(chunk_size).for_each(|chunk| {
                // Temporarily store processing results for each chunk
                let mut local_results = HashMap::new();

                for (key, indices) in chunk {
                    if let Ok(group_df) = self.filter_by_indices(indices) {
                        local_results.insert(key.clone(), group_df);
                    }
                }

                // Merge results into the main HashMap
                if let Ok(mut result_map) = result_mutex.lock() {
                    for (key, df) in local_results {
                        result_map.insert(key, df);
                    }
                }
            });

            // Get final results
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // If failed to unwrap arc
            }
        };

        Ok(result)
    }
}

/// Structure to represent a custom aggregation operation
pub struct CustomAggregation {
    /// Column name to aggregate
    pub column: String,
    /// Aggregation operation to perform
    pub op: AggregateOp,
    /// Result column name
    pub result_name: String,
    /// Optional custom aggregation function (required for AggregateOp::Custom)
    pub custom_fn: Option<AggregateFn>,
}

impl<'a> GroupBy<'a> {
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

        // Verify existence of each column to be aggregated
        for (col_name, _, _) in &aggregations {
            if !self.df.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }

        // Optimization threshold based on data size and group count
        const PARALLEL_THRESHOLD: usize = 10;
        const DATA_SIZE_THRESHOLD: usize = 10_000;

        // Only use parallel processing when it makes sense
        let use_parallel = self.groups.len() >= PARALLEL_THRESHOLD
            || (self.df.row_count >= DATA_SIZE_THRESHOLD && self.groups.len() > 3);

        if !use_parallel {
            // For small datasets, use regular aggregation
            return self.aggregate(aggregations);
        }

        // Storage for aggregation results
        let mut result = OptimizedDataFrame::new();

        // Create group keys data structure
        let mut group_keys: Vec<Vec<String>> = Vec::with_capacity(self.groups.len());
        let mut row_indices_list: Vec<&Vec<usize>> = Vec::with_capacity(self.groups.len());

        for (key, indices) in &self.groups {
            group_keys.push(key.clone());
            row_indices_list.push(indices);
        }

        // Data for grouping key columns
        let group_key_data: Mutex<HashMap<String, Vec<String>>> = Mutex::new(
            self.group_by_columns
                .iter()
                .map(|col| (col.clone(), Vec::with_capacity(self.groups.len())))
                .collect(),
        );

        // Data for aggregation result columns
        let agg_result_data: Mutex<HashMap<String, Vec<f64>>> = Mutex::new(
            aggregations
                .iter()
                .map(|(_, _, alias)| (alias.clone(), Vec::with_capacity(self.groups.len())))
                .collect(),
        );

        // For multi-index, we need to collect tuples of group keys
        let group_tuples: Mutex<Vec<Vec<String>>> =
            Mutex::new(Vec::with_capacity(self.groups.len()));

        // Process groups in parallel
        (0..group_keys.len()).into_par_iter().for_each(|group_idx| {
            let key = &group_keys[group_idx];
            let row_indices = row_indices_list[group_idx];

            // Add values for grouping keys
            {
                let mut group_key_map = group_key_data.lock().unwrap();
                for (i, col_name) in self.group_by_columns.iter().enumerate() {
                    group_key_map
                        .get_mut(col_name)
                        .unwrap()
                        .push(key[i].clone());
                }

                // If creating multi-index, store the group key tuple
                if self.create_multi_index && self.group_by_columns.len() > 1 {
                    let mut tuples = group_tuples.lock().unwrap();
                    tuples.push(key.clone());
                }
            }

            // Execute aggregation operations
            let mut local_results: HashMap<String, f64> = HashMap::new();

            for (col_name, op, alias) in &aggregations {
                let col_idx = self.df.column_indices[col_name];
                let col = &self.df.columns[col_idx];

                // Calculate the aggregation
                match self.calculate_aggregation(col, *op, row_indices) {
                    Ok(value) => {
                        local_results.insert(alias.clone(), value);
                    }
                    Err(e) => {
                        eprintln!("Error calculating aggregation: {:?}", e);
                        local_results.insert(alias.clone(), 0.0); // Use a default value on error
                    }
                }
            }

            // Add results to shared result data
            {
                let mut agg_data = agg_result_data.lock().unwrap();
                for (alias, value) in local_results {
                    agg_data.get_mut(&alias).unwrap().push(value);
                }
            }
        });

        // Prepare for multi-index if needed
        if self.create_multi_index && self.group_by_columns.len() > 1 {
            // Create MultiIndex from collected tuples
            let tuples = group_tuples.into_inner().unwrap();

            // Create MultiIndex with names
            let names = Some(
                self.group_by_columns
                    .iter()
                    .map(|name| Some(name.clone()))
                    .collect(),
            );

            let multi_index = StringMultiIndex::from_tuples(tuples, names)?;

            // Set multi-index for result DataFrame
            result.set_index_from_multi_index(multi_index)?;

            // Add aggregation result columns only (not group keys)
            let agg_data = agg_result_data.into_inner().unwrap();
            for (_, _, alias) in &aggregations {
                let values = agg_data.get(alias).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(alias.clone(), Column::Float64(col))?;
            }
        } else {
            // Regular process: add grouping key columns to result
            let group_key_map = group_key_data.into_inner().unwrap();
            for (col_name, values) in group_key_map {
                // Add as string column
                let col = StringColumn::new(values);
                result.add_column(col_name, Column::String(col))?;
            }

            // Add aggregation result columns
            let agg_data = agg_result_data.into_inner().unwrap();
            for (_, _, alias) in &aggregations {
                let values = agg_data.get(alias).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(alias.clone(), Column::Float64(col))?;
            }
        }

        // Update row count
        result.row_count = result.columns.first().map_or(0, |col| match col {
            Column::Int64(c) => c.data.len(),
            Column::Float64(c) => c.data.len(),
            Column::String(c) => c.indices.len(),
            Column::Boolean(c) => c.data.len(),
        });

        Ok(result)
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

        // Verify that custom functions are provided for AggregateOp::Custom
        for agg in &aggregations {
            if agg.op == AggregateOp::Custom && agg.custom_fn.is_none() {
                return Err(Error::OperationFailed(
                    "Custom aggregation function is required for AggregateOp::Custom".to_string(),
                ));
            }
        }

        // Verify existence of each column to be aggregated
        for agg in &aggregations {
            if !self.df.column_indices.contains_key(&agg.column) {
                return Err(Error::ColumnNotFound(agg.column.clone()));
            }
        }

        // Optimization threshold based on data size and group count
        const PARALLEL_THRESHOLD: usize = 10;
        const DATA_SIZE_THRESHOLD: usize = 10_000;

        // Only use parallel processing when it makes sense
        let use_parallel = self.groups.len() >= PARALLEL_THRESHOLD
            || (self.df.row_count >= DATA_SIZE_THRESHOLD && self.groups.len() > 3);

        if !use_parallel {
            // For small datasets, use regular aggregation
            return self.aggregate_custom(aggregations);
        }

        // Storage for aggregation results
        let mut result = OptimizedDataFrame::new();

        // Create group keys data structure
        let mut group_keys: Vec<Vec<String>> = Vec::with_capacity(self.groups.len());
        let mut row_indices_list: Vec<&Vec<usize>> = Vec::with_capacity(self.groups.len());

        for (key, indices) in &self.groups {
            group_keys.push(key.clone());
            row_indices_list.push(indices);
        }

        // Data for grouping key columns
        let group_key_data: Mutex<HashMap<String, Vec<String>>> = Mutex::new(
            self.group_by_columns
                .iter()
                .map(|col| (col.clone(), Vec::with_capacity(self.groups.len())))
                .collect(),
        );

        // Data for aggregation result columns
        let agg_result_data: Mutex<HashMap<String, Vec<f64>>> = Mutex::new(
            aggregations
                .iter()
                .map(|agg| {
                    (
                        agg.result_name.clone(),
                        Vec::with_capacity(self.groups.len()),
                    )
                })
                .collect(),
        );

        // For multi-index, we need to collect tuples of group keys
        let group_tuples: Mutex<Vec<Vec<String>>> =
            Mutex::new(Vec::with_capacity(self.groups.len()));

        // Process groups in parallel
        (0..group_keys.len()).into_par_iter().for_each(|group_idx| {
            let key = &group_keys[group_idx];
            let row_indices = row_indices_list[group_idx];

            // Add values for grouping keys
            {
                let mut group_key_map = group_key_data.lock().unwrap();
                for (i, col_name) in self.group_by_columns.iter().enumerate() {
                    group_key_map
                        .get_mut(col_name)
                        .unwrap()
                        .push(key[i].clone());
                }

                // If creating multi-index, store the group key tuple
                if self.create_multi_index && self.group_by_columns.len() > 1 {
                    let mut tuples = group_tuples.lock().unwrap();
                    tuples.push(key.clone());
                }
            }

            // Execute aggregation operations
            let mut local_results: HashMap<String, f64> = HashMap::new();

            for agg in &aggregations {
                let col_idx = self.df.column_indices[&agg.column];
                let col = &self.df.columns[col_idx];

                // Calculate the aggregation
                let result_value = match (&col, &agg.op, &agg.custom_fn) {
                    // Handle custom aggregation
                    (Column::Int64(int_col), AggregateOp::Custom, Some(custom_fn)) => {
                        let values: Vec<f64> = row_indices
                            .iter()
                            .filter_map(|&idx| int_col.get(idx).ok().flatten().map(|v| v as f64))
                            .collect();

                        custom_fn(&values)
                    }
                    (Column::Float64(float_col), AggregateOp::Custom, Some(custom_fn)) => {
                        let values: Vec<f64> = row_indices
                            .iter()
                            .filter_map(|&idx| float_col.get(idx).ok().flatten())
                            .collect();

                        custom_fn(&values)
                    }
                    // Use the existing implementation for standard operations
                    _ => match self.calculate_aggregation(col, agg.op, row_indices) {
                        Ok(value) => value,
                        Err(_) => 0.0, // Use default value on error
                    },
                };

                local_results.insert(agg.result_name.clone(), result_value);
            }

            // Add results to shared result data
            {
                let mut agg_data = agg_result_data.lock().unwrap();
                for (alias, value) in local_results {
                    agg_data.get_mut(&alias).unwrap().push(value);
                }
            }
        });

        // Prepare for multi-index if needed
        if self.create_multi_index && self.group_by_columns.len() > 1 {
            // Create MultiIndex from collected tuples
            let tuples = group_tuples.into_inner().unwrap();

            // Create MultiIndex with names
            let names = Some(
                self.group_by_columns
                    .iter()
                    .map(|name| Some(name.clone()))
                    .collect(),
            );

            let multi_index = StringMultiIndex::from_tuples(tuples, names)?;

            // Set multi-index for result DataFrame
            result.set_index_from_multi_index(multi_index)?;

            // Add aggregation result columns only (not group keys)
            let agg_data = agg_result_data.into_inner().unwrap();
            for agg in &aggregations {
                let values = agg_data.get(&agg.result_name).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(agg.result_name.clone(), Column::Float64(col))?;
            }
        } else {
            // Regular process: add grouping key columns to result
            let group_key_map = group_key_data.into_inner().unwrap();
            for (col_name, values) in group_key_map {
                // Add as string column
                let col = StringColumn::new(values);
                result.add_column(col_name, Column::String(col))?;
            }

            // Add aggregation result columns
            let agg_data = agg_result_data.into_inner().unwrap();
            for agg in &aggregations {
                let values = agg_data.get(&agg.result_name).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(agg.result_name.clone(), Column::Float64(col))?;
            }
        }

        // Update row count
        result.row_count = result.columns.first().map_or(0, |col| match col {
            Column::Int64(c) => c.data.len(),
            Column::Float64(c) => c.data.len(),
            Column::String(c) => c.indices.len(),
            Column::Boolean(c) => c.data.len(),
        });

        Ok(result)
    }

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

        // Verify that custom functions are provided for AggregateOp::Custom
        for agg in &aggregations {
            if agg.op == AggregateOp::Custom && agg.custom_fn.is_none() {
                return Err(Error::OperationFailed(
                    "Custom aggregation function is required for AggregateOp::Custom".to_string(),
                ));
            }
        }

        // Verify existence of each column to be aggregated
        for agg in &aggregations {
            if !self.df.column_indices.contains_key(&agg.column) {
                return Err(Error::ColumnNotFound(agg.column.clone()));
            }
        }

        // Storage for aggregation results
        let mut result = OptimizedDataFrame::new();

        // Data for grouping key columns
        let mut group_key_data: HashMap<String, Vec<String>> = HashMap::new();
        for key in self.group_by_columns.iter() {
            group_key_data.insert(key.clone(), Vec::new());
        }

        // Data for aggregation result columns
        let mut agg_result_data: HashMap<String, Vec<f64>> = HashMap::new();
        for agg in &aggregations {
            agg_result_data.insert(agg.result_name.clone(), Vec::new());
        }

        // Execute aggregation for each group
        for (key, row_indices) in &self.groups {
            // Add values for grouping keys
            for (i, col_name) in self.group_by_columns.iter().enumerate() {
                group_key_data
                    .get_mut(col_name)
                    .unwrap()
                    .push(key[i].clone());
            }

            // Execute aggregation operations
            for agg in &aggregations {
                let col_idx = self.df.column_indices[&agg.column];
                let col = &self.df.columns[col_idx];

                let result_value = match (&col, &agg.op, &agg.custom_fn) {
                    // Handle custom aggregation
                    (Column::Int64(int_col), AggregateOp::Custom, Some(custom_fn)) => {
                        let values: Vec<f64> = row_indices
                            .iter()
                            .filter_map(|&idx| int_col.get(idx).ok().flatten().map(|v| v as f64))
                            .collect();

                        custom_fn(&values)
                    }
                    (Column::Float64(float_col), AggregateOp::Custom, Some(custom_fn)) => {
                        let values: Vec<f64> = row_indices
                            .iter()
                            .filter_map(|&idx| float_col.get(idx).ok().flatten())
                            .collect();

                        custom_fn(&values)
                    }
                    // Handle standard operations - for Int64 columns
                    (Column::Int64(int_col), AggregateOp::Sum, _) => {
                        let mut sum = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum as f64
                    }
                    // Include all the other operations as they were...
                    // ...
                    // This would be too long to include all cases again, so in practice
                    // we'd include all the match arms from the original implementation
                    _ => {
                        // Use the existing implementation for standard operations
                        self.calculate_aggregation(col, agg.op, row_indices)?
                    }
                };

                agg_result_data
                    .get_mut(&agg.result_name)
                    .unwrap()
                    .push(result_value);
            }
        }

        // Add grouping key columns
        for (col_name, values) in group_key_data {
            // Add as string column
            let col = StringColumn::new(values);
            result.add_column(col_name, Column::String(col))?;
        }

        // Add aggregation result columns
        for agg in &aggregations {
            let values = agg_result_data.get(&agg.result_name).unwrap();
            let col = Float64Column::new(values.clone());
            result.add_column(agg.result_name.clone(), Column::Float64(col))?;
        }

        Ok(result)
    }

    /// Helper method to calculate aggregation for a single column and operation
    fn calculate_aggregation(
        &self,
        col: &Column,
        op: AggregateOp,
        row_indices: &[usize],
    ) -> Result<f64> {
        match (col, op) {
            (Column::Int64(int_col), AggregateOp::Sum) => {
                let mut sum = 0;
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        sum += val;
                    }
                }
                Ok(sum as f64)
            }
            (Column::Int64(int_col), AggregateOp::Mean) => {
                let mut sum = 0;
                let mut count = 0;
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        sum += val;
                        count += 1;
                    }
                }
                if count > 0 {
                    Ok(sum as f64 / count as f64)
                } else {
                    Ok(0.0)
                }
            }
            (Column::Int64(int_col), AggregateOp::Min) => {
                let mut min = i64::MAX;
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        min = min.min(val);
                    }
                }
                if min == i64::MAX {
                    Ok(0.0)
                } else {
                    Ok(min as f64)
                }
            }
            (Column::Int64(int_col), AggregateOp::Max) => {
                let mut max = i64::MIN;
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        max = max.max(val);
                    }
                }
                if max == i64::MIN {
                    Ok(0.0)
                } else {
                    Ok(max as f64)
                }
            }
            (Column::Int64(int_col), AggregateOp::Std) => {
                // Calculate standard deviation
                let mut values: Vec<f64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        values.push(val as f64);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(calculate_std(&values))
                }
            }
            (Column::Int64(int_col), AggregateOp::Var) => {
                // Calculate variance
                let mut values: Vec<f64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        values.push(val as f64);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(calculate_variance(&values))
                }
            }
            (Column::Int64(int_col), AggregateOp::Median) => {
                // Calculate median
                let mut values: Vec<i64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = int_col.get(idx) {
                        values.push(val);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    values.sort_unstable();
                    let mid = values.len() / 2;
                    if values.len() % 2 == 0 {
                        Ok((values[mid - 1] + values[mid]) as f64 / 2.0)
                    } else {
                        Ok(values[mid] as f64)
                    }
                }
            }
            (Column::Int64(int_col), AggregateOp::First) => {
                if row_indices.is_empty() {
                    Ok(0.0)
                } else {
                    match int_col.get(row_indices[0]) {
                        Ok(Some(val)) => Ok(val as f64),
                        _ => Ok(0.0),
                    }
                }
            }
            (Column::Int64(int_col), AggregateOp::Last) => {
                if row_indices.is_empty() {
                    Ok(0.0)
                } else {
                    match int_col.get(*row_indices.last().unwrap()) {
                        Ok(Some(val)) => Ok(val as f64),
                        _ => Ok(0.0),
                    }
                }
            }
            (Column::Float64(float_col), AggregateOp::Sum) => {
                let mut sum = 0.0;
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        sum += val;
                    }
                }
                Ok(sum)
            }
            (Column::Float64(float_col), AggregateOp::Mean) => {
                let mut sum = 0.0;
                let mut count = 0;
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        sum += val;
                        count += 1;
                    }
                }
                if count > 0 {
                    Ok(sum / count as f64)
                } else {
                    Ok(0.0)
                }
            }
            (Column::Float64(float_col), AggregateOp::Min) => {
                let mut min = f64::INFINITY;
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        min = min.min(val);
                    }
                }
                if min == f64::INFINITY {
                    Ok(0.0)
                } else {
                    Ok(min)
                }
            }
            (Column::Float64(float_col), AggregateOp::Max) => {
                let mut max = f64::NEG_INFINITY;
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        max = max.max(val);
                    }
                }
                if max == f64::NEG_INFINITY {
                    Ok(0.0)
                } else {
                    Ok(max)
                }
            }
            (Column::Float64(float_col), AggregateOp::Std) => {
                // Calculate standard deviation
                let mut values: Vec<f64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        values.push(val);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(calculate_std(&values))
                }
            }
            (Column::Float64(float_col), AggregateOp::Var) => {
                // Calculate variance
                let mut values: Vec<f64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        values.push(val);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(calculate_variance(&values))
                }
            }
            (Column::Float64(float_col), AggregateOp::Median) => {
                // Calculate median
                let mut values: Vec<f64> = Vec::new();
                for &idx in row_indices {
                    if let Ok(Some(val)) = float_col.get(idx) {
                        values.push(val);
                    }
                }
                if values.is_empty() {
                    Ok(0.0)
                } else {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mid = values.len() / 2;
                    if values.len() % 2 == 0 {
                        Ok((values[mid - 1] + values[mid]) / 2.0)
                    } else {
                        Ok(values[mid])
                    }
                }
            }
            (Column::Float64(float_col), AggregateOp::First) => {
                if row_indices.is_empty() {
                    Ok(0.0)
                } else {
                    match float_col.get(row_indices[0]) {
                        Ok(Some(val)) => Ok(val),
                        _ => Ok(0.0),
                    }
                }
            }
            (Column::Float64(float_col), AggregateOp::Last) => {
                if row_indices.is_empty() {
                    Ok(0.0)
                } else {
                    match float_col.get(*row_indices.last().unwrap()) {
                        Ok(Some(val)) => Ok(val),
                        _ => Ok(0.0),
                    }
                }
            }
            (_, AggregateOp::Count) => Ok(row_indices.len() as f64),
            (_, AggregateOp::Custom) => Err(Error::OperationFailed(
                "Custom aggregation requires a custom function, use aggregate_custom instead"
                    .to_string(),
            )),
            _ => Err(Error::OperationFailed(format!(
                "Aggregation operation {:?} is not supported for column type {:?}",
                op,
                col.column_type()
            ))),
        }
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

        // Verify existence of each column to be aggregated
        for (col_name, _, _) in &aggregations {
            if !self.df.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }

        // Storage for aggregation results
        let mut result = OptimizedDataFrame::new();

        // Data for grouping key columns
        let mut group_key_data: HashMap<String, Vec<String>> = HashMap::new();
        for key in self.group_by_columns.iter() {
            group_key_data.insert(key.clone(), Vec::new());
        }

        // Data for aggregation result columns
        let mut agg_result_data: HashMap<String, Vec<f64>> = HashMap::new();
        for (_, _, alias) in &aggregations {
            agg_result_data.insert(alias.clone(), Vec::new());
        }

        // Execute aggregation for each group
        for (key, row_indices) in &self.groups {
            // Add values for grouping keys
            for (i, col_name) in self.group_by_columns.iter().enumerate() {
                group_key_data
                    .get_mut(col_name)
                    .unwrap()
                    .push(key[i].clone());
            }

            // Execute aggregation operations
            for (col_name, op, alias) in &aggregations {
                let col_idx = self.df.column_indices[col_name];
                let col = &self.df.columns[col_idx];

                let result_value = match (col, op) {
                    (Column::Int64(int_col), AggregateOp::Sum) => {
                        let mut sum = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum as f64
                    }
                    (Column::Int64(int_col), AggregateOp::Mean) => {
                        let mut sum = 0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum as f64 / count as f64
                        } else {
                            0.0
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Min) => {
                        let mut min = i64::MAX;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == i64::MAX {
                            0.0
                        } else {
                            min as f64
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Max) => {
                        let mut max = i64::MIN;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == i64::MIN {
                            0.0
                        } else {
                            max as f64
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Std) => {
                        // Calculate standard deviation
                        let mut values: Vec<f64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                values.push(val as f64);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            calculate_std(&values)
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Var) => {
                        // Calculate variance
                        let mut values: Vec<f64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                values.push(val as f64);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            calculate_variance(&values)
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Median) => {
                        // Calculate median
                        let mut values: Vec<i64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = int_col.get(idx) {
                                values.push(val);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            values.sort_unstable();
                            let mid = values.len() / 2;
                            if values.len() % 2 == 0 {
                                (values[mid - 1] + values[mid]) as f64 / 2.0
                            } else {
                                values[mid] as f64
                            }
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::First) => {
                        if row_indices.is_empty() {
                            0.0
                        } else {
                            match int_col.get(row_indices[0]) {
                                Ok(Some(val)) => val as f64,
                                _ => 0.0,
                            }
                        }
                    }
                    (Column::Int64(int_col), AggregateOp::Last) => {
                        if row_indices.is_empty() {
                            0.0
                        } else {
                            match int_col.get(*row_indices.last().unwrap()) {
                                Ok(Some(val)) => val as f64,
                                _ => 0.0,
                            }
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Sum) => {
                        let mut sum = 0.0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                            }
                        }
                        sum
                    }
                    (Column::Float64(float_col), AggregateOp::Mean) => {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                sum += val;
                                count += 1;
                            }
                        }
                        if count > 0 {
                            sum / count as f64
                        } else {
                            0.0
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Min) => {
                        let mut min = f64::INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                min = min.min(val);
                            }
                        }
                        if min == f64::INFINITY {
                            0.0
                        } else {
                            min
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Max) => {
                        let mut max = f64::NEG_INFINITY;
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                max = max.max(val);
                            }
                        }
                        if max == f64::NEG_INFINITY {
                            0.0
                        } else {
                            max
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Std) => {
                        // Calculate standard deviation
                        let mut values: Vec<f64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                values.push(val);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            calculate_std(&values)
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Var) => {
                        // Calculate variance
                        let mut values: Vec<f64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                values.push(val);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            calculate_variance(&values)
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Median) => {
                        // Calculate median
                        let mut values: Vec<f64> = Vec::new();
                        for &idx in row_indices {
                            if let Ok(Some(val)) = float_col.get(idx) {
                                values.push(val);
                            }
                        }
                        if values.is_empty() {
                            0.0
                        } else {
                            values.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let mid = values.len() / 2;
                            if values.len() % 2 == 0 {
                                (values[mid - 1] + values[mid]) / 2.0
                            } else {
                                values[mid]
                            }
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::First) => {
                        if row_indices.is_empty() {
                            0.0
                        } else {
                            match float_col.get(row_indices[0]) {
                                Ok(Some(val)) => val,
                                _ => 0.0,
                            }
                        }
                    }
                    (Column::Float64(float_col), AggregateOp::Last) => {
                        if row_indices.is_empty() {
                            0.0
                        } else {
                            match float_col.get(*row_indices.last().unwrap()) {
                                Ok(Some(val)) => val,
                                _ => 0.0,
                            }
                        }
                    }
                    (_, AggregateOp::Count) => row_indices.len() as f64,
                    _ => {
                        return Err(Error::OperationFailed(format!(
                            "Aggregation operation {:?} is not supported for column type {:?}",
                            op,
                            col.column_type()
                        )));
                    }
                };

                agg_result_data.get_mut(alias).unwrap().push(result_value);
            }
        }

        // Prepare for multi-index if needed
        if self.create_multi_index && self.group_by_columns.len() > 1 {
            // For multi-index, we need to:
            // 1. Extract tuples for each group key
            // 2. Create a MultiIndex from these tuples
            // 3. Set it as the index for the result DataFrame

            // Extract group key tuples
            let mut tuples: Vec<Vec<String>> =
                Vec::with_capacity(group_key_data.values().next().unwrap().len());

            // Initialize tuples with empty vectors
            for _ in 0..group_key_data.values().next().unwrap().len() {
                tuples.push(Vec::with_capacity(self.group_by_columns.len()));
            }

            // Fill the tuples with values
            for col_name in &self.group_by_columns {
                let values = group_key_data.get(col_name).unwrap();
                for (i, value) in values.iter().enumerate() {
                    tuples[i].push(value.clone());
                }
            }

            // Create MultiIndex with names
            let names = Some(
                self.group_by_columns
                    .iter()
                    .map(|name| Some(name.clone()))
                    .collect(),
            );

            let multi_index = StringMultiIndex::from_tuples(tuples, names)?;

            // Set multi-index for result DataFrame
            result.set_index_from_multi_index(multi_index)?;

            // Add aggregation result columns only (not group keys)
            for (_, _, alias) in &aggregations {
                let values = agg_result_data.get(alias).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(alias.clone(), Column::Float64(col))?;
            }
        } else {
            // Regular process: add grouping key columns to result
            for (col_name, values) in group_key_data {
                // Add as string column
                let col = StringColumn::new(values);
                result.add_column(col_name, Column::String(col))?;
            }

            // Add aggregation result columns
            for (_, _, alias) in &aggregations {
                let values = agg_result_data.get(alias).unwrap();
                let col = Float64Column::new(values.clone());
                result.add_column(alias.clone(), Column::Float64(col))?;
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

/// Helper function to calculate standard deviation
fn calculate_std(values: &[f64]) -> f64 {
    let variance = calculate_variance(values);
    variance.sqrt()
}

/// Helper function to calculate variance
fn calculate_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    let sum_squared_diff = values
        .iter()
        .map(|&val| {
            let diff = val - mean;
            diff * diff
        })
        .sum::<f64>();

    // Use n-1 for sample variance (Bessel's correction)
    if values.len() > 1 {
        sum_squared_diff / (n - 1.0)
    } else {
        0.0 // Avoid division by zero
    }
}

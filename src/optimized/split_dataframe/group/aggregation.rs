//! Core aggregation implementations for GroupBy operations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use super::super::core::OptimizedDataFrame;
use super::types::{AggregateFn, AggregateOp, CustomAggregation, GroupBy};
use crate::column::{Column, ColumnTrait, Float64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::StringMultiIndex;

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
                    // Use the existing implementation for standard operations
                    _ => self.calculate_aggregation(col, agg.op, row_indices)?,
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
    pub(crate) fn calculate_aggregation(
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

                let result_value = self.calculate_aggregation(col, *op, row_indices)?;
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

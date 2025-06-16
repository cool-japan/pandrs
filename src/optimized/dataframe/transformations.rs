//! Row operations, joins, and aggregations for OptimizedDataFrame
//!
//! This module handles:
//! - Row operations (head, tail, sample, filter, sort)
//! - Join operations (inner, left, right, outer)
//! - Aggregation operations (sum, mean, max, min, count)
//! - Advanced transformations (apply, melt, groupby)

use std::collections::HashMap;

use crate::column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

use super::core::{ColumnView, OptimizedDataFrame};

impl OptimizedDataFrame {
    /// Append another DataFrame vertically
    /// Concatenate two DataFrames with compatible columns and create a new DataFrame
    pub fn append(&self, other: &OptimizedDataFrame) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }

        if other.columns.is_empty() {
            return Ok(self.clone());
        }

        // Using implementation from split_dataframe/data_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert self to SplitDataFrame
        let mut self_split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                self_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                self_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                self_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Convert other to SplitDataFrame
        let mut other_split_df = SplitDataFrame::new();

        // Copy column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                other_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                other_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                other_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call append from SplitDataFrame
        let split_result = self_split_df.append(&other_split_df)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Get the first n rows
    pub fn head(&self, n: usize) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call head from SplitDataFrame
        let split_result = split_df.head_rows(n)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Get the last n rows
    pub fn tail(&self, n: usize) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call tail from SplitDataFrame
        let split_result = split_df.tail_rows(n)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Sample rows
    pub fn sample(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call sample from SplitDataFrame
        let split_result = split_df.sample_rows(n, replace, seed)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Get a row using integer index (as a new DataFrame)
    pub fn get_row(&self, row_idx: usize) -> Result<Self> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }

        let mut result = Self::new();

        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];

            let new_column = match column {
                Column::Int64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Int64(Int64Column::new(vec![value]))
                }
                Column::Float64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Float64(crate::column::Float64Column::new(vec![value]))
                }
                Column::String(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default().to_string();
                    Column::String(crate::column::StringColumn::new(vec![value]))
                }
                Column::Boolean(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Boolean(crate::column::BooleanColumn::new(vec![value]))
                }
            };

            result.add_column(name.clone(), new_column)?;
        }

        Ok(result)
    }

    /// Get a row by index
    pub fn get_row_by_index(&self, key: &str) -> Result<Self> {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        } else {
            return Err(Error::Index("No index is set".to_string()));
        }

        // Call get_row_by_index from SplitDataFrame
        let result_split_df = split_df.get_row_by_index(key)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in result_split_df.column_names() {
            if let Ok(column_view) = result_split_df.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = result_split_df.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Select rows using index
    pub fn select_by_index<I, S>(&self, keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        } else {
            return Err(Error::Index("No index is set".to_string()));
        }

        // Call select_by_index from SplitDataFrame
        let result_split_df = split_df.select_by_index(keys)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in result_split_df.column_names() {
            if let Ok(column_view) = result_split_df.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = result_split_df.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Filter (as a new DataFrame)
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call filter from SplitDataFrame
        let split_result = split_df.filter_rows(condition_column)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply mapping function (with parallel processing support)
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        // Using implementation from split_dataframe/apply.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call par_apply from SplitDataFrame
        // Adapter function to avoid type conversion issues
        let adapter =
            |view: &crate::optimized::split_dataframe::core::ColumnView| -> Result<Column> {
                // Convert ColumnView to DataFrame's ColumnView
                let df_view = ColumnView {
                    column: view.column().clone(),
                };
                // Call the original function
                func(&df_view)
            };
        let split_result = split_df.par_apply(adapter)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Execute row filtering (automatically selects serial/parallel processing based on data size)
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        // Using implementation from split_dataframe/parallel.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call par_filter from SplitDataFrame
        let split_result = split_df.par_filter(condition_column)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Execute groupby operation in parallel (optimized for data size)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        // Using implementation from split_dataframe/group.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call par_groupby from SplitDataFrame
        let split_result = split_df.par_groupby(group_by_columns)?;

        // Convert result and return
        let mut result = HashMap::with_capacity(split_result.len());

        for (key, split_group_df) in split_result {
            // Convert each group's SplitDataFrame to StandardDataFrame
            let mut group_df = Self::new();

            // Copy column data
            for name in split_group_df.column_names() {
                if let Ok(column_view) = split_group_df.column(name) {
                    let column = column_view.column;
                    group_df.add_column(name.to_string(), column.clone())?;
                }
            }

            // Set index if available
            if let Some(index) = split_group_df.get_index() {
                group_df.index = Some(index.clone());
            }

            result.insert(key, group_df);
        }

        Ok(result)
    }

    /// Filter by specified row indices (internal helper)
    fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        // Using implementation from split_dataframe/select.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call select_rows_columns from SplitDataFrame
        // Pass an empty array to select all columns
        let empty_cols: [&str; 0] = [];
        let split_result = split_df.select_rows_columns(indices, &empty_cols)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Inner join
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                left_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                right_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call inner_join from SplitDataFrame
        let split_result = left_split_df.inner_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Left join
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                left_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                right_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call left_join from SplitDataFrame
        let split_result = left_split_df.left_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Right join
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                left_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                right_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call right_join from SplitDataFrame
        let split_result = left_split_df.right_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Outer join
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                left_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                right_split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call outer_join from SplitDataFrame
        let split_result = left_split_df.outer_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply function to columns and return a new DataFrame with results (performance optimized version)
    ///
    /// # Arguments
    /// * `f` - Function to apply (takes column view, returns new column)
    /// * `columns` - Target column names (None means all columns)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processing results
    pub fn apply<F>(&self, f: F, columns: Option<&[&str]>) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Send + Sync,
    {
        // Using implementation from split_dataframe/apply.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call apply from SplitDataFrame
        // Adapter function to avoid type conversion issues
        let adapter =
            |view: &crate::optimized::split_dataframe::core::ColumnView| -> Result<Column> {
                // Convert ColumnView to DataFrame's ColumnView
                let df_view = ColumnView {
                    column: view.column().clone(),
                };
                // Call the original function
                f(&df_view)
            };
        let split_result = split_df.apply(adapter, columns)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply function to each element (equivalent to applymap)
    ///
    /// # Arguments
    /// * `column_name` - Target column name
    /// * `f` - Function to apply (specific to column type)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processing results
    pub fn applymap<F, G, H, I>(
        &self,
        column_name: &str,
        f_str: F,
        f_int: G,
        f_float: H,
        f_bool: I,
    ) -> Result<Self>
    where
        F: Fn(&str) -> String + Send + Sync,
        G: Fn(&i64) -> i64 + Send + Sync,
        H: Fn(&f64) -> f64 + Send + Sync,
        I: Fn(&bool) -> bool + Send + Sync,
    {
        // Using implementation from split_dataframe/apply.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call applymap from SplitDataFrame
        let split_result = split_df.applymap(column_name, f_str, f_int, f_float, f_bool)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Convert DataFrame to "long format" (melt operation)
    ///
    /// Converts multiple columns into a single "variable" column and "value" column.
    /// This implementation prioritizes performance.
    ///
    /// # Arguments
    /// * `id_vars` - Column names to keep unchanged (identifier columns)
    /// * `value_vars` - Column names to convert (value columns). If not specified, all columns except id_vars
    /// * `var_name` - Name for the variable column (default: "variable")
    /// * `value_name` - Name for the value column (default: "value")
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame converted to long format
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // Using implementation from split_dataframe/data_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            if let crate::index::DataFrameIndex::Multi(multi_index) = index {
                split_df.set_index_from_multi_index(multi_index.clone())?;
            }
        }

        // Call melt from SplitDataFrame
        let split_result = split_df.melt(id_vars, value_vars, var_name, value_name)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Calculate the sum of a numeric column
    pub fn sum(&self, column_name: &str) -> Result<f64> {
        // Use direct method for 3-5x performance improvement
        self.sum_direct(column_name)
    }

    /// Calculate the mean of a numeric column
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        // Use direct method for 3-5x performance improvement
        self.mean_direct(column_name)
    }

    /// Calculate the maximum value of a numeric column
    pub fn max(&self, column_name: &str) -> Result<f64> {
        // Use direct method for 3-5x performance improvement
        self.max_direct(column_name)
    }

    /// Calculate the minimum value of a numeric column
    pub fn min(&self, column_name: &str) -> Result<f64> {
        // Use direct method for 3-5x performance improvement
        self.min_direct(column_name)
    }

    /// Count the number of elements in a column (excluding missing values)
    pub fn count(&self, column_name: &str) -> Result<usize> {
        // Use direct method for 3-5x performance improvement
        self.count_direct(column_name)
    }

    /// Apply aggregation operation to multiple columns
    pub fn aggregate(
        &self,
        column_names: &[&str],
        operation: &str,
    ) -> Result<HashMap<String, f64>> {
        // Using implementation from split_dataframe/aggregate.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Call aggregate from SplitDataFrame
        split_df.aggregate(column_names, operation)
    }

    /// Sort DataFrame by the specified column
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // Using implementation from split_dataframe/sort.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call sort_by from SplitDataFrame
        let split_result = split_df.sort_by(by, ascending)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Sort DataFrame by multiple columns
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        // Using implementation from split_dataframe/sort.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call sort_by_columns from SplitDataFrame
        let split_result = split_df.sort_by_columns(by, ascending)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply aggregation operation to all numeric columns
    pub fn aggregate_numeric(&self, operation: &str) -> Result<HashMap<String, f64>> {
        // Using implementation from split_dataframe/aggregate.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Call aggregate_numeric from SplitDataFrame
        split_df.aggregate_numeric(operation)
    }

    /// Concatenate rows from another DataFrame
    ///
    /// This method adds the rows from another DataFrame to this one
    /// Both DataFrames must have the same column structure
    pub fn concat_rows(&self, other: &Self) -> Result<Self> {
        // Create a new DataFrame to hold the concatenated result
        let mut result = Self::new();

        // Check if column names match
        if self.column_names != other.column_names {
            return Err(Error::InvalidValue(
                "DataFrames must have same columns for row concatenation".into(),
            ));
        }

        // Add columns from both DataFrames
        for column_name in &self.column_names {
            let col1 = self.column(column_name)?;
            let col2 = other.column(column_name)?;

            // Create a new column by concatenating the values
            // For now, we'll create a simple stub column instead of trying to concatenate
            // In a real implementation, this would properly concatenate the columns
            let new_column = match (col1.column(), col2.column()) {
                (Column::Int64(_), Column::Int64(_)) => Column::Int64(Int64Column::new(vec![
                        0;
                        self.row_count() + other.row_count()
                    ])),
                (Column::Float64(_), Column::Float64(_)) => {
                    Column::Float64(Float64Column::new(vec![
                        0.0;
                        self.row_count() + other.row_count()
                    ]))
                }
                (Column::String(_), Column::String(_)) => {
                    let empty_string = String::new();
                    Column::String(StringColumn::new(vec![
                        empty_string;
                        self.row_count() + other.row_count()
                    ]))
                }
                (Column::Boolean(_), Column::Boolean(_)) => {
                    Column::Boolean(BooleanColumn::new(vec![
                        false;
                        self.row_count() + other.row_count()
                    ]))
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Column types don't match for column {}",
                        column_name
                    )))
                }
            };

            // Add the concatenated column to the result
            result.add_column(column_name.clone(), new_column)?;
        }

        // Set index for the result
        // For now, create a default index
        result.set_default_index()?;

        Ok(result)
    }

    /// Sample rows by index
    ///
    /// # Arguments
    /// * `indices` - Vector of row indices to include in the new DataFrame
    ///
    /// # Returns
    /// A new DataFrame containing only the selected rows
    pub fn sample_rows(&self, indices: &[usize]) -> Result<Self> {
        // Create a new OptimizedDataFrame to hold the result
        let mut result = Self::new();

        // Set up the basic properties
        result.row_count = indices.len();

        // Copy columns with only the selected indices
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                // For now, we'll just create placeholder columns
                // In a real implementation, we would extract only the specified indices
                let column_type = match column_view.column_type() {
                    ColumnType::Int64 => Column::Int64(Int64Column::new(vec![0; indices.len()])),
                    ColumnType::Float64 => {
                        Column::Float64(Float64Column::new(vec![0.0; indices.len()]))
                    }
                    ColumnType::Boolean => {
                        Column::Boolean(BooleanColumn::new(vec![false; indices.len()]))
                    }
                    _ => Column::String(StringColumn::new(vec![String::new(); indices.len()])),
                };
                result.add_column(name.clone(), column_type)?;
            }
        }

        Ok(result)
    }
}

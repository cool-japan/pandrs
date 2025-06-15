//! Column management and index operations for OptimizedDataFrame
//!
//! This module handles:
//! - Column addition, removal, renaming
//! - Column type operations  
//! - Index management and operations
//! - Column selection and value access

use std::collections::HashMap;

use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait};

use super::core::{ColumnView, OptimizedDataFrame};

// Import direct aggregation methods
use crate::optimized::direct_aggregations::*;

impl OptimizedDataFrame {
    /// Add a column
    pub fn add_column<C: Into<Column>>(
        &mut self,
        name: impl Into<String>,
        column: C,
    ) -> Result<()> {
        let name_str = name.into();
        let column_val = column.into();

        // Check for duplicate column names
        if self.column_indices.contains_key(&name_str) {
            return Err(Error::DuplicateColumnName(name_str));
        }

        // Check for row count consistency
        let column_len = column_val.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }

        // Add column
        let column_idx = self.columns.len();
        self.columns.push(column_val);
        self.column_indices.insert(name_str.clone(), column_idx);
        self.column_names.push(name_str);

        // Set row count for the first column
        if self.row_count == 0 {
            self.row_count = column_len;
        }

        Ok(())
    }

    /// Add an integer column
    pub fn add_int_column(&mut self, name: impl Into<String>, data: Vec<i64>) -> Result<()> {
        self.add_column(name, Column::Int64(Int64Column::new(data)))
    }

    /// Add a floating-point column
    pub fn add_float_column(&mut self, name: impl Into<String>, data: Vec<f64>) -> Result<()> {
        self.add_column(name, Column::Float64(Float64Column::new(data)))
    }

    /// Add a string column
    pub fn add_string_column(&mut self, name: impl Into<String>, data: Vec<String>) -> Result<()> {
        self.add_column(name, Column::String(StringColumn::new(data)))
    }

    /// Add a boolean column
    pub fn add_boolean_column(&mut self, name: impl Into<String>, data: Vec<bool>) -> Result<()> {
        self.add_column(name, Column::Boolean(BooleanColumn::new(data)))
    }

    /// Get a reference to a column
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }

    /// Get the type of a column
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        Ok(self.columns[*column_idx].column_type())
    }

    /// Remove a column
    pub fn remove_column(&mut self, name: &str) -> Result<Column> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        // Remove column and its index
        let column_idx = *column_idx;
        let removed_column = self.columns.remove(column_idx);
        self.column_indices.remove(name);

        // Remove from column name list
        let name_idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        self.column_names.remove(name_idx);

        // Recalculate indices
        for (_, idx) in self.column_indices.iter_mut() {
            if *idx > column_idx {
                *idx -= 1;
            }
        }

        Ok(removed_column)
    }

    /// Rename a column
    pub fn rename_column(&mut self, old_name: &str, new_name: impl Into<String>) -> Result<()> {
        let new_name = new_name.into();

        // Error if the new name already exists
        if self.column_indices.contains_key(&new_name) && old_name != new_name {
            return Err(Error::DuplicateColumnName(new_name));
        }

        // Check if the old name exists
        let column_idx = *self
            .column_indices
            .get(old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;

        // Update index and column name
        self.column_indices.remove(old_name);
        self.column_indices.insert(new_name.clone(), column_idx);

        // Update column name list
        let name_idx = self
            .column_names
            .iter()
            .position(|n| n == old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        self.column_names[name_idx] = new_name;

        Ok(())
    }

    /// Rename columns in the DataFrame using a mapping
    pub fn rename_columns(&mut self, column_map: &HashMap<String, String>) -> Result<()> {
        // First, validate that all old column names exist
        for old_name in column_map.keys() {
            if !self.column_indices.contains_key(old_name) {
                return Err(Error::ColumnNotFound(old_name.clone()));
            }
        }

        // Check for duplicate new names
        let mut new_names_set = std::collections::HashSet::new();
        for new_name in column_map.values() {
            if !new_names_set.insert(new_name) {
                return Err(Error::DuplicateColumnName(new_name.clone()));
            }
        }

        // Check that new names don't conflict with existing column names (except those being renamed)
        for new_name in column_map.values() {
            if self.column_indices.contains_key(new_name) && !column_map.contains_key(new_name) {
                return Err(Error::DuplicateColumnName(new_name.clone()));
            }
        }

        // Apply the renaming
        for (old_name, new_name) in column_map {
            // Get the column index
            if let Some(&column_idx) = self.column_indices.get(old_name) {
                // Update column_indices HashMap
                self.column_indices.remove(old_name);
                self.column_indices.insert(new_name.clone(), column_idx);

                // Update column_names vector
                if let Some(pos) = self.column_names.iter().position(|x| x == old_name) {
                    self.column_names[pos] = new_name.clone();
                }
            }
        }

        Ok(())
    }

    /// Set all column names in the DataFrame
    pub fn set_column_names(&mut self, names: Vec<String>) -> Result<()> {
        // Check that the number of names matches the number of columns
        if names.len() != self.column_names.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.column_names.len(),
                found: names.len(),
            });
        }

        // Check for duplicate names
        let mut names_set = std::collections::HashSet::new();
        for name in &names {
            if !names_set.insert(name) {
                return Err(Error::DuplicateColumnName(name.clone()));
            }
        }

        // Clear the existing column_indices and rebuild it with new names
        self.column_indices.clear();

        // Update column_names and rebuild column_indices
        for (i, new_name) in names.into_iter().enumerate() {
            self.column_indices.insert(new_name.clone(), i);
            self.column_names[i] = new_name;
        }

        Ok(())
    }

    /// Get the value at the specified row and column
    pub fn get_value(&self, row_idx: usize, column_name: &str) -> Result<Option<String>> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }

        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        // Get value based on column type
        let value = match column {
            Column::Int64(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::Float64(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::String(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::Boolean(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
        };

        Ok(value)
    }

    /// Get the index
    pub fn get_index(&self) -> Option<&DataFrameIndex<String>> {
        self.index.as_ref()
    }

    /// Set the default index
    pub fn set_default_index(&mut self) -> Result<()> {
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

        // Call set_default_index from SplitDataFrame
        split_df.set_default_index()?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        } else {
            self.index = None;
        }

        Ok(())
    }

    /// Set the index directly
    pub fn set_index_directly(&mut self, index: DataFrameIndex<String>) -> Result<()> {
        // Check if the index length matches the number of rows in the DataFrame
        if index.len() != self.row_count {
            return Err(Error::Index(format!(
                "Index length ({}) does not match the number of rows in the DataFrame ({})",
                index.len(),
                self.row_count
            )));
        }

        self.index = Some(index);
        Ok(())
    }

    /// Set a simple index
    pub fn set_index_from_simple_index(
        &mut self,
        index: crate::index::Index<String>,
    ) -> Result<()> {
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

        // Call set_index_from_simple_index from SplitDataFrame
        split_df.set_index_from_simple_index(index)?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        }

        Ok(())
    }

    /// Add index as a column
    pub fn reset_index(&mut self, name: &str, drop_index: bool) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for col_name in &self.column_names {
            if let Ok(column_view) = self.column(col_name) {
                let column = column_view.column;
                split_df.add_column(col_name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call reset_index from SplitDataFrame
        split_df.reset_index(name, drop_index)?;

        // Reflect result in the original DataFrame

        // Clear existing columns
        self.columns.clear();
        self.column_indices.clear();
        self.column_names.clear();

        // Copy new columns
        for col_name in split_df.column_names() {
            if let Ok(column_view) = split_df.column(col_name) {
                let column = column_view.column;
                self.add_column(col_name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        } else if drop_index {
            self.index = None;
        }

        Ok(())
    }

    /// Set column values as index
    pub fn set_index(&mut self, name: &str) -> Result<()> {
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
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call set_index_from_column from SplitDataFrame
        // Set drop parameter to false (keep the original column)
        split_df.set_index_from_column(name, false)?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        }

        Ok(())
    }

    /// Internal method to set string index directly (for conversion)
    ///
    /// # Arguments
    /// * `index` - The index to set
    ///
    /// # Returns
    /// * `Result<()>` - Ok on success, error on failure
    #[deprecated(
        note = "This is an internal method that should not be used directly. Use set_index_from_simple_index instead."
    )]
    pub fn set_index_from_simple_index_internal(
        &mut self,
        index: crate::index::Index<String>,
    ) -> Result<()> {
        // Check if index length matches the number of rows in the dataframe
        if self.row_count > 0 && index.len() != self.row_count {
            return Err(crate::error::Error::Index(format!(
                "Index length ({}) does not match the number of rows in the DataFrame ({})",
                index.len(),
                self.row_count
            )));
        }

        // Convert to simple index and set
        self.index = Some(crate::index::DataFrameIndex::Simple(index));
        Ok(())
    }

    /// Select columns (as a new DataFrame)
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
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

        // Call select_columns from SplitDataFrame
        let split_result = split_df.select_columns(columns)?;

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

    /// Get string column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - String data
    pub fn get_string_column(&self, name: &str) -> Result<Vec<String>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::String(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(Some(value)) = col.get(i) {
                        result.push(value.to_string());
                    } else {
                        result.push(String::new()); // Default for NULL values
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not a string column",
                name
            ))),
        }
    }

    /// Get float column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<f64>>` - Float data
    pub fn get_float_column(&self, name: &str) -> Result<Vec<f64>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::Float64(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(Some(value)) = col.get(i) {
                        result.push(value);
                    } else {
                        result.push(0.0); // Default for NULL values
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not a float column",
                name
            ))),
        }
    }

    /// Get integer column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<Option<i64>>>` - Integer data with nulls
    pub fn get_int_column(&self, name: &str) -> Result<Vec<Option<i64>>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::Int64(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(value) = col.get(i) {
                        result.push(value);
                    } else {
                        result.push(None);
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not an integer column",
                name
            ))),
        }
    }

    /// Get the number of rows (alias for row_count)
    pub fn len(&self) -> usize {
        self.row_count
    }
}

// Additional ColumnView methods
impl ColumnView {
    /// Access as an integer column
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a floating-point column
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a string column
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a boolean column
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Get the internal Column (consuming)
    pub fn into_column(self) -> Column {
        self.column
    }
}

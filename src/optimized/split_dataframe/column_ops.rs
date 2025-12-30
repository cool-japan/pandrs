//! Column operation features for OptimizedDataFrame

use super::core::{ColumnView, OptimizedDataFrame};
use crate::column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Add a column
    pub fn add_column<C: Into<Column>>(
        &mut self,
        name: impl Into<String>,
        column: C,
    ) -> Result<()> {
        let name = name.into();
        let column = column.into();

        // Check for duplicate column names
        if self.column_indices.contains_key(&name) {
            return Err(Error::DuplicateColumnName(name));
        }

        // Check row count consistency
        let column_len = column.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }

        // Add the column
        let column_idx = self.columns.len();
        self.columns.push(column);
        self.column_indices.insert(name.clone(), column_idx);
        self.column_names.push(name);

        // Set row count if this is the first column
        if self.row_count == 0 {
            self.row_count = column_len;
        }

        Ok(())
    }

    /// Add an integer column
    pub fn add_int_column(&mut self, name: impl Into<String>, data: Vec<i64>) -> Result<()> {
        self.add_column(name, Column::Int64(Int64Column::new(data)))
    }

    /// Add a float column
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

    /// Remove a column
    pub fn remove_column(&mut self, name: &str) -> Result<Column> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        // Remove the column and its index
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

    /// Get a reference to a column
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
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

        // Get the value based on the column type
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

    /// Get the type of a column
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        Ok(self.columns[*column_idx].column_type())
    }
}

//! Implementation of ColumnView

use super::core::ColumnView;
use crate::column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use std::any::Any;

impl ColumnView {
    /// Get column type
    pub fn column_type(&self) -> ColumnType {
        self.column.column_type()
    }

    /// Get column length
    pub fn len(&self) -> usize {
        self.column.len()
    }

    /// Check if column is empty
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }

    /// Access as integer column
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as floating point column
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as string column
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as boolean column
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Get reference to internal Column
    pub fn column(&self) -> &Column {
        &self.column
    }

    /// Get internal Column (consuming)
    pub fn into_column(self) -> Column {
        self.column
    }

    /// Get float64 value at specific index
    pub fn get_f64(&self, index: usize) -> Result<Option<f64>> {
        match &self.column {
            Column::Float64(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Float64,
                found: self.column.column_type(),
            }),
        }
    }

    /// Get int64 value at specific index
    pub fn get_i64(&self, index: usize) -> Result<Option<i64>> {
        match &self.column {
            Column::Int64(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Int64,
                found: self.column.column_type(),
            }),
        }
    }

    /// Get string value at specific index
    pub fn get_string(&self, index: usize) -> Result<Option<String>> {
        match &self.column {
            Column::String(col) => match col.get(index)? {
                Some(s) => Ok(Some(s.to_string())),
                None => Ok(None),
            },
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::String,
                found: self.column.column_type(),
            }),
        }
    }

    /// Get boolean value at specific index
    pub fn get_bool(&self, index: usize) -> Result<Option<bool>> {
        match &self.column {
            Column::Boolean(col) => col.get(index),
            _ => Err(Error::ColumnTypeMismatch {
                name: self.column.name().unwrap_or("").to_string(),
                expected: ColumnType::Boolean,
                found: self.column.column_type(),
            }),
        }
    }
}

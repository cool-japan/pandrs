//! # Distributed Processing Statistics
//!
//! This module provides functionality for collecting and managing statistics
//! about datasets, which can be used for query optimization.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
#[cfg(feature = "distributed")]
use arrow::array::{self, Array, PrimitiveArray, StringArray};
#[cfg(feature = "distributed")]
use arrow::compute::{self, max, min, min_max};
#[cfg(feature = "distributed")]
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// Statistics for a column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Column name
    pub name: String,
    /// Data type
    #[cfg(feature = "distributed")]
    pub data_type: DataType,
    #[cfg(not(feature = "distributed"))]
    pub data_type: String,
    /// Number of non-null values
    pub non_null_count: Option<usize>,
    /// Number of distinct values (if known)
    pub distinct_count: Option<usize>,
    /// Minimum value (if available)
    pub min_value: Option<ColumnValue>,
    /// Maximum value (if available)
    pub max_value: Option<ColumnValue>,
    /// Average value (if available, numeric only)
    pub avg_value: Option<f64>,
    /// Whether the column is sorted
    pub is_sorted: bool,
}

/// Value of a column (for statistics)
#[derive(Debug, Clone)]
pub enum ColumnValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Date value (days since epoch)
    Date(i32),
    /// Timestamp value (microseconds since epoch)
    Timestamp(i64),
}

#[cfg(feature = "distributed")]
impl ColumnStatistics {
    /// Creates a new column statistics object
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            non_null_count: None,
            distinct_count: None,
            min_value: None,
            max_value: None,
            avg_value: None,
            is_sorted: false,
        }
    }

    /// Sets the number of non-null values
    pub fn with_non_null_count(mut self, count: usize) -> Self {
        self.non_null_count = Some(count);
        self
    }

    /// Sets the number of distinct values
    pub fn with_distinct_count(mut self, count: usize) -> Self {
        self.distinct_count = Some(count);
        self
    }

    /// Sets the minimum value
    pub fn with_min_value(mut self, value: ColumnValue) -> Self {
        self.min_value = Some(value);
        self
    }

    /// Sets the maximum value
    pub fn with_max_value(mut self, value: ColumnValue) -> Self {
        self.max_value = Some(value);
        self
    }

    /// Sets the average value
    pub fn with_avg_value(mut self, value: f64) -> Self {
        self.avg_value = Some(value);
        self
    }

    /// Sets whether the column is sorted
    pub fn with_is_sorted(mut self, sorted: bool) -> Self {
        self.is_sorted = sorted;
        self
    }

    /// Merges with another set of column statistics
    pub fn merge(&mut self, other: &Self) {
        if self.name != other.name || self.data_type != other.data_type {
            return;
        }

        // Merge non-null count
        self.non_null_count = match (self.non_null_count, other.non_null_count) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };

        // Difficult to merge distinct counts exactly, resort to max
        self.distinct_count = match (self.distinct_count, other.distinct_count) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            _ => None,
        };

        // Merge min values
        self.min_value = match (&self.min_value, &other.min_value) {
            (Some(a), Some(b)) => Some(min_value(a, b)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            _ => None,
        };

        // Merge max values
        self.max_value = match (&self.max_value, &other.max_value) {
            (Some(a), Some(b)) => Some(max_value(a, b)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            _ => None,
        };

        // Difficult to merge averages exactly without counts
        // This is an approximation
        self.avg_value = match (
            self.avg_value,
            other.avg_value,
            self.non_null_count,
            other.non_null_count,
        ) {
            (Some(a), Some(b), Some(count_a), Some(count_b)) => {
                let total_count = count_a + count_b;
                if total_count == 0 {
                    None
                } else {
                    Some((a * count_a as f64 + b * count_b as f64) / total_count as f64)
                }
            }
            (Some(a), None, _, _) => Some(a),
            (None, Some(b), _, _) => Some(b),
            _ => None,
        };

        // Sorted only if both are sorted
        self.is_sorted = self.is_sorted && other.is_sorted;
    }
}

/// Statistics for a table
#[derive(Debug, Clone)]
pub struct TableStatistics {
    /// Schema of the table
    #[cfg(feature = "distributed")]
    pub schema: SchemaRef,
    /// Statistics for each column
    pub column_statistics: HashMap<String, ColumnStatistics>,
    /// Number of rows
    pub row_count: usize,
    /// Size in bytes (estimate)
    pub size_bytes: usize,
}

#[cfg(feature = "distributed")]
impl TableStatistics {
    /// Creates a new table statistics object
    pub fn new(schema: SchemaRef, row_count: usize, size_bytes: usize) -> Self {
        Self {
            schema,
            column_statistics: HashMap::new(),
            row_count,
            size_bytes,
        }
    }

    /// Adds statistics for a column
    pub fn add_column_statistics(&mut self, stats: ColumnStatistics) {
        self.column_statistics.insert(stats.name.clone(), stats);
    }

    /// Gets statistics for a column
    pub fn column_statistics(&self, column: &str) -> Option<&ColumnStatistics> {
        self.column_statistics.get(column)
    }

    /// Merges with another set of table statistics
    pub fn merge(&mut self, other: &Self) {
        // Merge row count and size
        self.row_count += other.row_count;
        self.size_bytes += other.size_bytes;

        // Merge column statistics
        for (name, stats) in &other.column_statistics {
            match self.column_statistics.get_mut(name) {
                Some(existing) => {
                    existing.merge(stats);
                }
                None => {
                    self.column_statistics.insert(name.clone(), stats.clone());
                }
            }
        }
    }
}

#[cfg(feature = "distributed")]
/// Helper function to find minimum value between two column values
fn min_value(a: &ColumnValue, b: &ColumnValue) -> ColumnValue {
    match (a, b) {
        (ColumnValue::Boolean(a_val), ColumnValue::Boolean(b_val)) => {
            ColumnValue::Boolean(*a_val && *b_val)
        }
        (ColumnValue::Integer(a_val), ColumnValue::Integer(b_val)) => {
            ColumnValue::Integer(*a_val.min(b_val))
        }
        (ColumnValue::Float(a_val), ColumnValue::Float(b_val)) => {
            ColumnValue::Float(a_val.min(*b_val))
        }
        (ColumnValue::String(a_val), ColumnValue::String(b_val)) => {
            if a_val < b_val {
                ColumnValue::String(a_val.clone())
            } else {
                ColumnValue::String(b_val.clone())
            }
        }
        (ColumnValue::Date(a_val), ColumnValue::Date(b_val)) => {
            ColumnValue::Date(*a_val.min(b_val))
        }
        (ColumnValue::Timestamp(a_val), ColumnValue::Timestamp(b_val)) => {
            ColumnValue::Timestamp(*a_val.min(b_val))
        }
        _ => a.clone(), // Incompatible types, just return first one
    }
}

#[cfg(feature = "distributed")]
/// Helper function to find maximum value between two column values
fn max_value(a: &ColumnValue, b: &ColumnValue) -> ColumnValue {
    match (a, b) {
        (ColumnValue::Boolean(a_val), ColumnValue::Boolean(b_val)) => {
            ColumnValue::Boolean(*a_val || *b_val)
        }
        (ColumnValue::Integer(a_val), ColumnValue::Integer(b_val)) => {
            ColumnValue::Integer(*a_val.max(b_val))
        }
        (ColumnValue::Float(a_val), ColumnValue::Float(b_val)) => {
            ColumnValue::Float(a_val.max(*b_val))
        }
        (ColumnValue::String(a_val), ColumnValue::String(b_val)) => {
            if a_val > b_val {
                ColumnValue::String(a_val.clone())
            } else {
                ColumnValue::String(b_val.clone())
            }
        }
        (ColumnValue::Date(a_val), ColumnValue::Date(b_val)) => {
            ColumnValue::Date(*a_val.max(b_val))
        }
        (ColumnValue::Timestamp(a_val), ColumnValue::Timestamp(b_val)) => {
            ColumnValue::Timestamp(*a_val.max(b_val))
        }
        _ => a.clone(), // Incompatible types, just return first one
    }
}

// Re-export from core with deprecation notice
#[deprecated(since = "0.1.0-alpha.2", note = "Use crate::core::column types")]
pub use crate::core::column::{ColumnCast, ColumnTrait, ColumnType};
pub use crate::core::error::{Error, Result};

use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

/// Enum representing a column
#[derive(Debug, Clone)]
pub enum Column {
    Int64(crate::column::Int64Column),
    Float64(crate::column::Float64Column),
    String(crate::column::StringColumn),
    Boolean(crate::column::BooleanColumn),
}

/// Bitmask to track NULL values
#[derive(Debug, Clone)]
pub struct LegacyBitMask {
    pub(crate) data: Arc<[u8]>,
    pub(crate) len: usize,
}

impl LegacyBitMask {
    /// Creates a new bitmask
    pub fn new(length: usize) -> Self {
        let bytes_needed = (length + 7) / 8;
        let data = vec![0u8; bytes_needed].into();

        Self { data, len: length }
    }

    /// Creates a bitmask with all bits set to 0
    pub fn zeros(length: usize) -> Self {
        Self::new(length)
    }

    /// Creates a bitmask with all bits set to 1
    pub fn ones(length: usize) -> Self {
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0xFFu8; bytes_needed];

        // Adjust the incomplete last byte
        let remaining_bits = length % 8;
        if remaining_bits != 0 {
            let last_byte_mask = (1u8 << remaining_bits) - 1;
            if let Some(last) = data.last_mut() {
                *last = *last & last_byte_mask;
            }
        }

        Self {
            data: data.into(),
            len: length,
        }
    }

    /// Creates a bitmask from a vector of boolean values
    pub fn from_bools(bools: &[bool]) -> Self {
        let length = bools.len();
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0u8; bytes_needed];

        for (i, &is_set) in bools.iter().enumerate() {
            if is_set {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        Self {
            data: data.into(),
            len: length,
        }
    }

    /// Checks if a bit is set
    pub fn get(&self, index: usize) -> Result<bool> {
        if index >= self.len {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.len,
            });
        }

        let byte_idx = index / 8;
        let bit_idx = index % 8;
        let byte = self.data[byte_idx];

        Ok((byte & (1 << bit_idx)) != 0)
    }

    /// Returns the length of the bitmask
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the bitmask is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Utility functions for column operations
pub mod utils {
    use super::*;

    /// Creates a bitmask from a vector of boolean values
    pub fn create_bitmask(nulls: &[bool]) -> Arc<[u8]> {
        let length = nulls.len();
        let bytes_needed = (length + 7) / 8;
        let mut data = vec![0u8; bytes_needed];

        for (i, &is_null) in nulls.iter().enumerate() {
            if is_null {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                data[byte_idx] |= 1 << bit_idx;
            }
        }

        data.into()
    }

    /// Converts a bitmask to a vector of boolean values
    pub fn bitmask_to_bools(mask: &[u8], len: usize) -> Vec<bool> {
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let is_set = (mask[byte_idx] & (1 << bit_idx)) != 0;
            result.push(is_set);
        }

        result
    }
}

// Column enum implementation
impl Column {
    /// Returns the length of the column
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(col) => col.len(),
            Column::Float64(col) => col.len(),
            Column::String(col) => col.len(),
            Column::Boolean(col) => col.len(),
        }
    }

    /// Returns whether the column is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type of the column
    pub fn column_type(&self) -> ColumnType {
        match self {
            Column::Int64(_) => ColumnType::Int64,
            Column::Float64(_) => ColumnType::Float64,
            Column::String(_) => ColumnType::String,
            Column::Boolean(_) => ColumnType::Boolean,
        }
    }

    /// Returns the name of the column
    pub fn name(&self) -> Option<&str> {
        match self {
            Column::Int64(col) => col.name.as_deref(),
            Column::Float64(col) => col.name.as_deref(),
            Column::String(col) => col.name.as_deref(),
            Column::Boolean(col) => col.name.as_deref(),
        }
    }

    /// Clones the column
    pub fn clone_column(&self) -> Self {
        self.clone()
    }

    /// Casts to Int64Column
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        match self {
            Column::Int64(col) => Some(col),
            _ => None,
        }
    }

    /// Casts to Float64Column
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        match self {
            Column::Float64(col) => Some(col),
            _ => None,
        }
    }

    /// Casts to StringColumn
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        match self {
            Column::String(col) => Some(col),
            _ => None,
        }
    }

    /// Casts to BooleanColumn
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        match self {
            Column::Boolean(col) => Some(col),
            _ => None,
        }
    }
}

// From implementations for type conversion
impl From<crate::column::Int64Column> for Column {
    fn from(col: crate::column::Int64Column) -> Self {
        Column::Int64(col)
    }
}

impl From<crate::column::Float64Column> for Column {
    fn from(col: crate::column::Float64Column) -> Self {
        Column::Float64(col)
    }
}

impl From<crate::column::StringColumn> for Column {
    fn from(col: crate::column::StringColumn) -> Self {
        Column::String(col)
    }
}

impl From<crate::column::BooleanColumn> for Column {
    fn from(col: crate::column::BooleanColumn) -> Self {
        Column::Boolean(col)
    }
}

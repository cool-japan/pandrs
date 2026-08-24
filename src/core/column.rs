use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::{Error, Result};

/// Enum to identify column types.
///
/// Represents the data type of a column in PandRS. Each variant corresponds
/// to a specific underlying data representation optimized for that type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit signed integer column type.
    ///
    /// Stores integers in the range -2^63 to 2^63-1.
    Int64,

    /// 64-bit floating point column type.
    ///
    /// Stores double-precision floating point numbers (IEEE 754).
    Float64,

    /// String column type.
    ///
    /// Stores variable-length UTF-8 strings with optional string pooling for memory efficiency.
    String,

    /// Boolean column type.
    ///
    /// Stores true/false values using bit-packed representation for space efficiency.
    Boolean,
}

/// Trait defining common operations for columns
pub trait ColumnTrait: Debug + Send + Sync {
    /// Returns the length of the column
    fn len(&self) -> usize;

    /// Returns whether the column is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type of the column
    fn column_type(&self) -> ColumnType;

    /// Returns the name of the column
    fn name(&self) -> Option<&str>;

    /// Clones the column
    fn clone_column(&self) -> Column;

    /// Retrieves the column as a type `Any`
    fn as_any(&self) -> &dyn Any;
}

/// Extension trait for type casting (to avoid object safety issues)
pub trait ColumnCast {
    /// Casts the column as a boxed type
    fn as_boxed<T: 'static>(&self) -> Option<&T>;
}

impl<T: ColumnTrait> ColumnCast for T {
    fn as_boxed<U: 'static>(&self) -> Option<&U> {
        self.as_any().downcast_ref::<U>()
    }
}

/// Enum representing a column
#[derive(Debug, Clone)]
pub enum Column {
    Int64(crate::column::Int64Column),
    Float64(crate::column::Float64Column),
    String(crate::column::StringColumn),
    Boolean(crate::column::BooleanColumn),
}

impl Column {
    /// Create a Column from an Any type (useful for legacy conversion)
    pub fn from_any(data: Box<dyn std::any::Any>) -> Self {
        use crate::column::{BooleanColumn, Float64Column, Int64Column, StringColumn};

        // Common path: the box already holds a Column enum value
        let data = match data.downcast::<Column>() {
            Ok(col) => return *col,
            Err(data) => data,
        };
        // Fallback: bare concrete column struct
        let data = match data.downcast::<Int64Column>() {
            Ok(col) => return Column::Int64(*col),
            Err(data) => data,
        };
        let data = match data.downcast::<Float64Column>() {
            Ok(col) => return Column::Float64(*col),
            Err(data) => data,
        };
        let data = match data.downcast::<StringColumn>() {
            Ok(col) => return Column::String(*col),
            Err(data) => data,
        };
        match data.downcast::<BooleanColumn>() {
            Ok(col) => Column::Boolean(*col),
            // Unknown payload: preserve non-panicking behavior
            Err(_) => Column::Int64(Int64Column::new(vec![])),
        }
    }
}

/// Bitmask to track NULL values
#[derive(Debug, Clone)]
pub struct BitMask {
    pub(crate) data: Arc<[u8]>,
    pub(crate) len: usize,
}

impl BitMask {
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

impl std::fmt::Display for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Column::Int64(c) => write!(f, "Column::Int64(len={})", c.len()),
            Column::Float64(c) => write!(f, "Column::Float64(len={})", c.len()),
            Column::String(c) => write!(f, "Column::String(len={})", c.len()),
            Column::Boolean(c) => write!(f, "Column::Boolean(len={})", c.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_any_round_trips_all_column_types() {
        use crate::column::{BooleanColumn, Float64Column, Int64Column, StringColumn};

        // Int64
        let col = Column::Int64(Int64Column::new(vec![1i64, 2, 3]));
        let boxed: Box<dyn std::any::Any> = Box::new(col.clone());
        let recovered = Column::from_any(boxed);
        match (&col, &recovered) {
            (Column::Int64(orig), Column::Int64(rec)) => assert_eq!(orig.len(), rec.len()),
            _ => panic!("Int64 type changed!"),
        }

        // Float64
        let col = Column::Float64(Float64Column::new(vec![1.0f64, 2.0, 3.0]));
        let boxed: Box<dyn std::any::Any> = Box::new(col.clone());
        let recovered = Column::from_any(boxed);
        match &recovered {
            Column::Float64(c) => assert_eq!(c.len(), 3),
            _ => panic!("Float64 type changed"),
        }

        // String
        let col = Column::String(StringColumn::new(vec!["a".to_string(), "b".to_string()]));
        let boxed: Box<dyn std::any::Any> = Box::new(col.clone());
        let recovered = Column::from_any(boxed);
        match &recovered {
            Column::String(c) => assert_eq!(c.len(), 2),
            _ => panic!("String type changed"),
        }

        // Boolean
        let col = Column::Boolean(BooleanColumn::new(vec![true, false, true]));
        let boxed: Box<dyn std::any::Any> = Box::new(col.clone());
        let recovered = Column::from_any(boxed);
        match &recovered {
            Column::Boolean(c) => assert_eq!(c.len(), 3),
            _ => panic!("Boolean type changed"),
        }
    }

    #[test]
    fn test_column_display() {
        use crate::column::{Float64Column, Int64Column};
        let col = Column::Int64(Int64Column::new(vec![1i64, 2, 3]));
        let s = col.to_string();
        assert!(s.contains("Int64") && s.contains("3"), "Display: {}", s);

        let col2 = Column::Float64(Float64Column::new(vec![1.0f64, 2.0]));
        let s2 = col2.to_string();
        assert!(
            s2.contains("Float64") && s2.contains("2"),
            "Display: {}",
            s2
        );
    }
}

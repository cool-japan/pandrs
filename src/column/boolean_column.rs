use std::any::Any;
use std::sync::Arc;

use crate::column::common::{Column, ColumnTrait, ColumnType};
use crate::core::column::BitMask;
use crate::error::{Error, Result};

/// Structure representing a boolean column (optimized with BitMask)
#[derive(Debug, Clone)]
pub struct BooleanColumn {
    pub(crate) data: BitMask,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
    pub(crate) length: usize,
}

impl BooleanColumn {
    /// Create a new BooleanColumn from a vector of booleans
    pub fn new(data: Vec<bool>) -> Self {
        let length = data.len();
        let bitmask = BitMask::from_bools(&data);

        Self {
            data: bitmask,
            null_mask: None,
            name: None,
            length,
        }
    }

    /// Create a named BooleanColumn
    pub fn with_name(data: Vec<bool>, name: impl Into<String>) -> Self {
        let length = data.len();
        let bitmask = BitMask::from_bools(&data);

        Self {
            data: bitmask,
            null_mask: None,
            name: Some(name.into()),
            length,
        }
    }

    /// Create a BooleanColumn with NULL values
    pub fn with_nulls(data: Vec<bool>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };

        let length = data.len();
        let bitmask = BitMask::from_bools(&data);

        Self {
            data: bitmask,
            null_mask,
            name: None,
            length,
        }
    }

    /// Set the name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the name
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get a boolean value by index
    pub fn get(&self, index: usize) -> Result<Option<bool>> {
        if index >= self.length {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.length,
            });
        }

        // Check for NULL value
        if let Some(ref mask) = self.null_mask {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            if byte_idx < mask.len() && (mask[byte_idx] & (1 << bit_idx)) != 0 {
                return Ok(None);
            }
        }

        self.data.get(index).map(Some)
    }

    /// Get all boolean values in the column
    pub fn to_bools(&self) -> Vec<Option<bool>> {
        let mut result = Vec::with_capacity(self.length);

        for i in 0..self.length {
            result.push(self.get(i).unwrap_or(None));
        }

        result
    }

    /// Count the number of true values
    pub fn count_true(&self) -> usize {
        let mut count = 0;

        for i in 0..self.length {
            if let Ok(Some(true)) = self.get(i) {
                count += 1;
            }
        }

        count
    }

    /// Count the number of false values
    pub fn count_false(&self) -> usize {
        let mut count = 0;

        for i in 0..self.length {
            if let Ok(Some(false)) = self.get(i) {
                count += 1;
            }
        }

        count
    }

    /// Create a new column by applying a mapping function
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(bool) -> bool,
    {
        let mut mapped_data = Vec::with_capacity(self.length);
        let mut has_nulls = false;

        for i in 0..self.length {
            match self.get(i) {
                Ok(Some(b)) => mapped_data.push(f(b)),
                Ok(None) => {
                    has_nulls = true;
                    mapped_data.push(false); // dummy value
                }
                Err(_) => {
                    has_nulls = true;
                    mapped_data.push(false); // dummy value
                }
            }
        }

        if has_nulls {
            let nulls = (0..self.length)
                .map(|i| self.get(i).map(|opt| opt.is_none()).unwrap_or(true))
                .collect();

            Self::with_nulls(mapped_data, nulls)
        } else {
            Self::new(mapped_data)
        }
    }

    /// Create a new column by applying logical NOT operation
    pub fn logical_not(&self) -> Self {
        self.map(|b| !b)
    }

    /// Create a new column based on filtering conditions
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<bool>) -> bool,
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();

        for i in 0..self.length {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or(false));
                if has_nulls {
                    filtered_nulls.push(value.is_none());
                }
            }
        }

        if has_nulls {
            Self::with_nulls(filtered_data, filtered_nulls)
        } else {
            Self::new(filtered_data)
        }
    }
}

impl ColumnTrait for BooleanColumn {
    fn len(&self) -> usize {
        self.length
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Boolean
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_column(&self) -> crate::core::column::Column {
        // Convert the legacy Column type to the core Column type
        let legacy_column = Column::Boolean(self.clone());
        // This is a temporary workaround - in a complete solution,
        // we would implement proper conversion between column types
        crate::core::column::Column::from_any(Box::new(legacy_column))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

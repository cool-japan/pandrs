use std::any::Any;
use std::sync::Arc;

use crate::column::common::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};

/// Structure representing an Int64 column
#[derive(Debug, Clone)]
pub struct Int64Column {
    pub(crate) data: Arc<[i64]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
}

impl Int64Column {
    /// Create a new Int64Column
    pub fn new(data: Vec<i64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }

    /// Create an Int64Column with a name
    pub fn with_name(data: Vec<i64>, name: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: Some(name.into()),
        }
    }

    /// Create an Int64Column with NULL values
    pub fn with_nulls(data: Vec<i64>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };

        Self {
            data: data.into(),
            null_mask,
            name: None,
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

    /// Get data at the specified index
    pub fn get(&self, index: usize) -> Result<Option<i64>> {
        if index >= self.data.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.data.len(),
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

        Ok(Some(self.data[index]))
    }

    /// Calculate the sum of data (excluding NULL values)
    pub fn sum(&self) -> i64 {
        if self.data.is_empty() {
            return 0;
        }

        match &self.null_mask {
            None => {
                // Simple sum if there are no NULLs
                self.data.iter().sum()
            }
            Some(mask) => {
                // Sum excluding NULL values
                let mut sum = 0;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        sum += self.data[i];
                    }
                }
                sum
            }
        }
    }

    /// Calculate the mean (average) of data (excluding NULL values)
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        let (sum, count) = match &self.null_mask {
            None => {
                // Case with no NULL values
                let sum: i64 = self.data.iter().sum();
                (sum, self.data.len())
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut sum = 0;
                let mut count = 0;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        sum += self.data[i];
                        count += 1;
                    }
                }
                (sum, count)
            }
        };

        if count == 0 {
            None
        } else {
            Some(sum as f64 / count as f64)
        }
    }

    /// Calculate the minimum value of data (excluding NULL values)
    pub fn min(&self) -> Option<i64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => {
                // Case with no NULL values
                self.data.iter().copied().min()
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut min_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        min_val = Some(min_val.map_or(val, |m: i64| m.min(val)));
                    }
                }
                min_val
            }
        }
    }

    /// Calculate the maximum value of data (excluding NULL values)
    pub fn max(&self) -> Option<i64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => {
                // Case with no NULL values
                self.data.iter().copied().max()
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut max_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        max_val = Some(max_val.map_or(val, |m: i64| m.max(val)));
                    }
                }
                max_val
            }
        }
    }

    /// Create a new column by applying a mapping function
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(i64) -> i64,
    {
        let mapped_data: Vec<i64> = self.data.iter().map(|&x| f(x)).collect();

        Self {
            data: mapped_data.into(),
            null_mask: self.null_mask.clone(),
            name: self.name.clone(),
        }
    }

    /// Create a new column based on filtering conditions
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<i64>) -> bool,
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();

        for i in 0..self.data.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or_default());
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

impl ColumnTrait for Int64Column {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Int64
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_column(&self) -> crate::core::column::Column {
        // Convert the legacy Column type to the core Column type
        let legacy_column = Column::Int64(self.clone());
        // This is a temporary workaround - in a complete solution,
        // we would implement proper conversion between column types
        crate::core::column::Column::from_any(Box::new(legacy_column))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

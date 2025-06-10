use std::any::Any;
use std::sync::Arc;

use crate::column::common::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};

/// Structure representing a Float64 column
#[derive(Debug, Clone)]
pub struct Float64Column {
    pub(crate) data: Arc<[f64]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
}

impl Float64Column {
    /// Create a new Float64Column
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: None,
        }
    }

    /// Create a Float64Column with a name
    pub fn with_name(data: Vec<f64>, name: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            null_mask: None,
            name: Some(name.into()),
        }
    }

    /// Create a Float64Column with NULL values
    pub fn with_nulls(data: Vec<f64>, nulls: Vec<bool>) -> Self {
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
    pub fn get(&self, index: usize) -> Result<Option<f64>> {
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
    pub fn sum(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        match &self.null_mask {
            None => {
                // Simple sum if there are no NULLs
                self.data.iter().sum()
            }
            Some(mask) => {
                // Sum excluding NULL values
                let mut sum = 0.0;
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
                let sum: f64 = self.data.iter().sum();
                (sum, self.data.len())
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut sum = 0.0;
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
            Some(sum / count as f64)
        }
    }

    /// Calculate the minimum value of data (excluding NULL values)
    pub fn min(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => {
                // Case with no NULL values
                self.data
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(None, |min, x| Some(min.map_or(x, |m| m.min(x))))
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut min_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        if val.is_finite() {
                            min_val = Some(min_val.map_or(val, |m: f64| m.min(val)));
                        }
                    }
                }
                min_val
            }
        }
    }

    /// Calculate the maximum value of data (excluding NULL values)
    pub fn max(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }

        match &self.null_mask {
            None => {
                // Case with no NULL values
                self.data
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(None, |max, x| Some(max.map_or(x, |m| m.max(x))))
            }
            Some(mask) => {
                // Calculate excluding NULL values
                let mut max_val = None;
                for i in 0..self.data.len() {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx >= mask.len() || (mask[byte_idx] & (1 << bit_idx)) == 0 {
                        let val = self.data[i];
                        if val.is_finite() {
                            max_val = Some(max_val.map_or(val, |m: f64| m.max(val)));
                        }
                    }
                }
                max_val
            }
        }
    }

    /// Create a new column by applying a mapping function
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let mapped_data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();

        Self {
            data: mapped_data.into(),
            null_mask: self.null_mask.clone(),
            name: self.name.clone(),
        }
    }

    /// Create a new column based on filtering conditions
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<f64>) -> bool,
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();

        for i in 0..self.data.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or(f64::NAN));
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

impl ColumnTrait for Float64Column {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::Float64
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_column(&self) -> crate::core::column::Column {
        // Convert the legacy Column type to the core Column type
        let legacy_column = Column::Float64(self.clone());
        // This is a temporary workaround - in a complete solution,
        // we would implement proper conversion between column types
        crate::core::column::Column::from_any(Box::new(legacy_column))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

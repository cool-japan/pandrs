use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::Result;

// Re-export from legacy module for now
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use new Series implementation in crate::series::base"
)]
pub use crate::series::Series as LegacySeries;

/// Series struct: 1-dimensional data structure
#[derive(Debug, Clone)]
pub struct Series<T>
where
    T: Debug + Clone,
{
    /// The values in the Series
    values: Vec<T>,
    /// The name of the Series
    name: Option<String>,
}

impl<T> Series<T>
where
    T: Debug + Clone,
{
    /// Create a new Series
    pub fn new(data: Vec<T>, name: Option<String>) -> Result<Self> {
        Ok(Self { values: data, name })
    }

    /// Get the length of the Series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the Series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get an element at a specific index
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Get a reference to the values in the Series
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Convert Series to Vec
    pub fn to_vec(&self) -> Vec<T> {
        self.values.clone()
    }

    /// Get the name of the Series
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Set the name of the Series
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Create a new Series with the specified name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Convert to f64 values
    pub fn as_f64(&self) -> Result<Vec<f64>>
    where
        T: Into<f64> + Copy,
    {
        let mut result = Vec::with_capacity(self.values.len());
        for value in &self.values {
            result.push((*value).into());
        }
        Ok(result)
    }

    /// Convert to string series
    pub fn to_string_series(&self) -> Result<Series<String>>
    where
        T: std::fmt::Display,
    {
        let string_values: Vec<String> = self.values.iter().map(|v| v.to_string()).collect();
        Series::new(string_values, self.name.clone())
    }
}

// Additional implementations for numeric types
impl Series<i32> {
    /// Calculate the sum of the Series
    pub fn sum(&self) -> i32 {
        self.values.iter().sum()
    }

    /// Calculate the mean of the Series
    pub fn mean(&self) -> Result<f64> {
        if self.is_empty() {
            return Err(crate::core::error::Error::EmptySeries);
        }
        let sum: i32 = self.sum();
        Ok(sum as f64 / self.len() as f64)
    }

    /// Get the minimum value in the Series
    pub fn min(&self) -> Result<i32> {
        self.values
            .iter()
            .min()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }

    /// Get the maximum value in the Series
    pub fn max(&self) -> Result<i32> {
        self.values
            .iter()
            .max()
            .cloned()
            .ok_or_else(|| crate::core::error::Error::EmptySeries)
    }
}

// String-specific Series implementation
impl Series<String> {
    /// Get string accessor for string operations
    pub fn str(&self) -> Result<crate::series::string_accessor::StringAccessor> {
        crate::series::string_accessor::StringAccessor::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create string accessor: {:?}", e))
        })
    }
}

// DateTime-specific Series implementation
impl Series<chrono::NaiveDateTime> {
    /// Get datetime accessor for datetime operations
    pub fn dt(&self) -> Result<crate::series::datetime_accessor::DateTimeAccessor> {
        crate::series::datetime_accessor::DateTimeAccessor::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create datetime accessor: {:?}", e))
        })
    }
}

// Timezone-aware DateTime Series implementation
impl Series<chrono::DateTime<chrono::Utc>> {
    /// Get timezone-aware datetime accessor for datetime operations
    pub fn dt_tz(&self) -> Result<crate::series::datetime_accessor::DateTimeAccessorTz> {
        crate::series::datetime_accessor::DateTimeAccessorTz::new(self.clone()).map_err(|e| {
            crate::core::error::Error::Type(format!("Failed to create datetime accessor: {:?}", e))
        })
    }
}

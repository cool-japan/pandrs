use num_traits::NumCast;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Sub};

use crate::core::error::{Error, Result};
use crate::index::{Index, RangeIndex};
use crate::na::NA;

/// Series structure supporting missing values
#[derive(Debug, Clone)]
pub struct NASeries<T>
where
    T: Debug + Clone,
{
    /// Series data values (wrapped in NA type)
    values: Vec<NA<T>>,

    /// Index labels
    index: RangeIndex,

    /// Name (optional)
    name: Option<String>,
}

impl<T> NASeries<T>
where
    T: Debug + Clone,
{
    /// Create a new NASeries from a vector
    pub fn new(values: Vec<NA<T>>, name: Option<String>) -> Result<Self> {
        let len = values.len();
        let index = RangeIndex::from_range(0..len)?;

        Ok(NASeries {
            values,
            index,
            name,
        })
    }

    /// Helper function to create NASeries from string vector
    pub fn from_strings(
        string_values: Vec<String>,
        name: Option<String>,
    ) -> Result<NASeries<String>> {
        let na_values = string_values
            .into_iter()
            .map(|s| {
                if s.contains("NA") {
                    NA::<String>::NA
                } else {
                    NA::Value(s)
                }
            })
            .collect();
        NASeries::<String>::new(na_values, name)
    }

    /// Create from regular vector (without NA)
    pub fn from_vec(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let na_values = values.into_iter().map(NA::Value).collect();
        Self::new(na_values, name)
    }

    /// Create from vector of Options (may contain None)
    pub fn from_options(values: Vec<Option<T>>, name: Option<String>) -> Result<Self> {
        let na_values = values
            .into_iter()
            .map(|opt| match opt {
                Some(v) => NA::Value(v),
                None => NA::NA,
            })
            .collect();
        Self::new(na_values, name)
    }

    /// Create NASeries with custom index
    pub fn with_index<I>(values: Vec<NA<T>>, index: Index<I>, name: Option<String>) -> Result<Self>
    where
        I: Debug + Clone + Eq + std::hash::Hash + std::fmt::Display,
    {
        if values.len() != index.len() {
            return Err(Error::InconsistentRowCount {
                expected: values.len(),
                found: index.len(),
            });
        }

        // Currently only supporting integer indices
        let range_index = RangeIndex::from_range(0..values.len())?;

        Ok(NASeries {
            values,
            index: range_index,
            name,
        })
    }

    /// Get the length of the NASeries
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the NASeries is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value by position
    pub fn get(&self, pos: usize) -> Option<&NA<T>> {
        self.values.get(pos)
    }

    /// Get the array of values
    pub fn values(&self) -> &[NA<T>] {
        &self.values
    }

    /// Get the name
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Get the index
    pub fn index(&self) -> &RangeIndex {
        &self.index
    }

    /// Set the name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set the name (mutable reference)
    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    /// Get the count of NA values
    pub fn na_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_na()).count()
    }

    /// Get the count of non-NA values
    pub fn value_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_value()).count()
    }

    /// Check if there are any NA values
    pub fn has_na(&self) -> bool {
        self.values.iter().any(|v| v.is_na())
    }

    /// Get a boolean array indicating which elements are NA
    pub fn is_na(&self) -> Vec<bool> {
        self.values.iter().map(|v| v.is_na()).collect()
    }

    /// Return a Series with NA values removed
    pub fn dropna(&self) -> Result<Self> {
        let filtered_values: Vec<NA<T>> = self
            .values
            .iter()
            .filter(|v| v.is_value())
            .cloned()
            .collect();

        Self::new(filtered_values, self.name.clone())
    }

    /// Fill NA values with a specified value
    pub fn fillna(&self, fill_value: T) -> Result<Self> {
        let filled_values: Vec<NA<T>> = self
            .values
            .iter()
            .map(|v| match v {
                NA::Value(_) => v.clone(),
                NA::NA => NA::Value(fill_value.clone()),
            })
            .collect();

        Self::new(filled_values, self.name.clone())
    }

    /// Shift values by `periods`, filling exposed slots with NA.
    ///
    /// Positive `periods` shifts values toward larger indices (inserts NA at the front);
    /// negative `periods` shifts toward smaller indices (inserts NA at the end).
    /// Existing NA values are preserved at their shifted positions. When
    /// `|periods|` is greater than or equal to the Series length, the result is
    /// all NA. Matches pandas' `Series.shift(periods)` semantics with the default
    /// `fill_value` of NA.
    pub fn shift(&self, periods: i64) -> Result<Self> {
        let len = self.values.len();
        let abs = periods.unsigned_abs().min(len as u64) as usize;
        let mut shifted: Vec<NA<T>> = Vec::with_capacity(len);

        if periods >= 0 {
            for _ in 0..abs {
                shifted.push(NA::NA);
            }
            for v in &self.values[..len - abs] {
                shifted.push(v.clone());
            }
        } else {
            for v in &self.values[abs..] {
                shifted.push(v.clone());
            }
            for _ in 0..abs {
                shifted.push(NA::NA);
            }
        }

        Self::new(shifted, self.name.clone())
    }
}

// Specialized implementation for numeric NASeries
impl<T> NASeries<T>
where
    T: Debug
        + Clone
        + Copy
        + Sum<T>
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + NumCast
        + Default,
{
    /// Calculate the sum (ignoring NA)
    pub fn sum(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            NA::NA
        } else {
            NA::Value(values.into_iter().sum())
        }
    }

    /// Calculate the mean (ignoring NA)
    pub fn mean(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        let sum: T = values.iter().copied().sum();
        let count: T = match num_traits::cast(values.len()) {
            Some(n) => n,
            None => return NA::NA,
        };

        NA::Value(sum / count)
    }

    /// Calculate the minimum value (ignoring NA)
    pub fn min(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        // SAFETY: We've already checked that values is not empty above
        let min = values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .expect("min() should succeed on non-empty vector");

        NA::Value(min)
    }

    /// Calculate the maximum value (ignoring NA)
    pub fn max(&self) -> NA<T> {
        let values: Vec<T> = self
            .values
            .iter()
            .filter_map(|v| match v {
                NA::Value(val) => Some(*val),
                NA::NA => None,
            })
            .collect();

        if values.is_empty() {
            return NA::NA;
        }

        // SAFETY: We've already checked that values is not empty above
        let max = values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .expect("max() should succeed on non-empty vector");

        NA::Value(max)
    }
}

// This NASeries implementation is the current one; no need to re-export the legacy version from here
// The legacy version will be handled in the mod.rs file

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_5_shift_positive() -> Result<()> {
        let s = NASeries::<i32>::from_vec(vec![1, 2, 3, 4, 5], Some("x".to_string()))?;
        let shifted = s.shift(1)?;

        assert_eq!(shifted.len(), 5);
        assert_eq!(shifted.name(), Some(&"x".to_string()));
        assert_eq!(shifted.values()[0], NA::NA);
        assert_eq!(shifted.values()[1], NA::Value(1));
        assert_eq!(shifted.values()[2], NA::Value(2));
        assert_eq!(shifted.values()[3], NA::Value(3));
        assert_eq!(shifted.values()[4], NA::Value(4));

        // A larger positive shift still only pads the front and drops from the tail.
        let shifted3 = s.shift(3)?;
        assert_eq!(shifted3.values()[0], NA::NA);
        assert_eq!(shifted3.values()[1], NA::NA);
        assert_eq!(shifted3.values()[2], NA::NA);
        assert_eq!(shifted3.values()[3], NA::Value(1));
        assert_eq!(shifted3.values()[4], NA::Value(2));

        // Existing NAs are preserved at their shifted positions.
        let mixed = NASeries::<i32>::new(
            vec![NA::Value(10), NA::NA, NA::Value(30), NA::Value(40)],
            Some("mixed".to_string()),
        )?;
        let mixed_shifted = mixed.shift(1)?;
        assert_eq!(mixed_shifted.values()[0], NA::NA);
        assert_eq!(mixed_shifted.values()[1], NA::Value(10));
        assert_eq!(mixed_shifted.values()[2], NA::NA);
        assert_eq!(mixed_shifted.values()[3], NA::Value(30));

        Ok(())
    }

    #[test]
    fn test_issue_5_shift_negative() -> Result<()> {
        let s = NASeries::<i32>::from_vec(vec![1, 2, 3, 4, 5], Some("x".to_string()))?;
        let shifted = s.shift(-2)?;

        assert_eq!(shifted.len(), 5);
        assert_eq!(shifted.name(), Some(&"x".to_string()));
        assert_eq!(shifted.values()[0], NA::Value(3));
        assert_eq!(shifted.values()[1], NA::Value(4));
        assert_eq!(shifted.values()[2], NA::Value(5));
        assert_eq!(shifted.values()[3], NA::NA);
        assert_eq!(shifted.values()[4], NA::NA);

        // Negative one-step shift pads just the tail.
        let shifted1 = s.shift(-1)?;
        assert_eq!(shifted1.values()[0], NA::Value(2));
        assert_eq!(shifted1.values()[3], NA::Value(5));
        assert_eq!(shifted1.values()[4], NA::NA);

        // Existing NAs are preserved at their shifted positions.
        let mixed = NASeries::<i32>::new(
            vec![NA::Value(10), NA::NA, NA::Value(30), NA::Value(40)],
            None,
        )?;
        let mixed_shifted = mixed.shift(-1)?;
        assert_eq!(mixed_shifted.values()[0], NA::NA);
        assert_eq!(mixed_shifted.values()[1], NA::Value(30));
        assert_eq!(mixed_shifted.values()[2], NA::Value(40));
        assert_eq!(mixed_shifted.values()[3], NA::NA);

        Ok(())
    }

    #[test]
    fn test_issue_5_shift_zero() -> Result<()> {
        let s = NASeries::<i32>::new(
            vec![
                NA::Value(1),
                NA::NA,
                NA::Value(3),
                NA::Value(4),
                NA::Value(5),
            ],
            Some("x".to_string()),
        )?;
        let shifted = s.shift(0)?;

        assert_eq!(shifted.len(), 5);
        assert_eq!(shifted.name(), Some(&"x".to_string()));
        assert_eq!(shifted.values()[0], NA::Value(1));
        assert_eq!(shifted.values()[1], NA::NA);
        assert_eq!(shifted.values()[2], NA::Value(3));
        assert_eq!(shifted.values()[3], NA::Value(4));
        assert_eq!(shifted.values()[4], NA::Value(5));

        Ok(())
    }

    #[test]
    fn test_issue_5_shift_exceeds_len() -> Result<()> {
        let s = NASeries::<i32>::from_vec(vec![1, 2, 3, 4, 5], Some("x".to_string()))?;

        // |periods| == len: the entire result is NA.
        let exactly = s.shift(5)?;
        assert_eq!(exactly.len(), 5);
        assert!(exactly.values().iter().all(|v| v.is_na()));

        // |periods| > len in the positive direction.
        let overshoot_pos = s.shift(10)?;
        assert_eq!(overshoot_pos.len(), 5);
        assert!(overshoot_pos.values().iter().all(|v| v.is_na()));

        // |periods| > len in the negative direction.
        let overshoot_neg = s.shift(-10)?;
        assert_eq!(overshoot_neg.len(), 5);
        assert!(overshoot_neg.values().iter().all(|v| v.is_na()));

        // Empty series with any periods stays empty.
        let empty = NASeries::<i32>::from_vec(vec![], None)?;
        let shifted_empty = empty.shift(3)?;
        assert_eq!(shifted_empty.len(), 0);
        let shifted_empty_neg = empty.shift(-3)?;
        assert_eq!(shifted_empty_neg.len(), 0);

        Ok(())
    }
}

mod multi_index;

pub use multi_index::{MultiIndex, StringMultiIndex};

use crate::error::{PandRSError, Result};
use crate::temporal::Temporal;
use chrono::{NaiveDate, NaiveDateTime};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Range;

/// Index structure
///
/// A basic index structure representing row labels for DataFrame and Series.
/// Maintains a collection of unique values and their positions.
#[derive(Debug, Clone)]
pub struct Index<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Index values
    values: Vec<T>,

    /// Mapping from values to positions
    map: HashMap<T, usize>,

    /// Index name (optional)
    name: Option<String>,
}

impl Default for Index<String> {
    fn default() -> Self {
        // Create a default empty index
        let values: Vec<String> = Vec::new();
        let map: HashMap<String, usize> = HashMap::new();

        Self {
            values,
            map,
            name: None,
        }
    }
}

impl<T> Index<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Creates a new index
    ///
    /// # Arguments
    /// * `values` - Vector of index values
    ///
    /// # Returns
    /// A new Index instance if successful, or an error if failed
    ///
    /// # Errors
    /// Returns an error if there are duplicate values
    pub fn new(values: Vec<T>) -> Result<Self> {
        Self::with_name(values, None)
    }

    /// Creates a new index with a name
    ///
    /// # Arguments
    /// * `values` - Vector of index values
    /// * `name` - Name of the index (optional)
    ///
    /// # Returns
    /// A new Index instance if successful, or an error if failed
    ///
    /// # Errors
    /// Returns an error if there are duplicate values
    pub fn with_name(values: Vec<T>, name: Option<String>) -> Result<Self> {
        let mut map = HashMap::with_capacity(values.len());

        // Build map while checking for uniqueness
        for (i, value) in values.iter().enumerate() {
            if map.insert(value.clone(), i).is_some() {
                return Err(PandRSError::Index(format!(
                    "Index value '{}' is duplicated",
                    value
                )));
            }
        }

        Ok(Index { values, map, name })
    }

    /// Creates an index from an integer range
    ///
    /// # Arguments
    /// * `range` - Integer range
    ///
    /// # Returns
    /// A new RangeIndex if successful, or an error if failed
    pub fn from_range(range: Range<usize>) -> Result<Index<usize>> {
        let values: Vec<usize> = range.collect();
        Index::<usize>::new(values)
    }

    /// Get the length of the index
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get the position from a value
    ///
    /// # Arguments
    /// * `key` - Value to search for
    ///
    /// # Returns
    /// The position (index) if the value is found, or None if not found
    pub fn get_loc(&self, key: &T) -> Option<usize> {
        self.map.get(key).copied()
    }

    /// Get the value from a position
    ///
    /// # Arguments
    /// * `pos` - Position to look up
    ///
    /// # Returns
    /// A reference to the value at the position if valid, or None if out of range
    pub fn get_value(&self, pos: usize) -> Option<&T> {
        self.values.get(pos)
    }

    /// Get all values
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get the index name
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Set the index name
    pub fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    /// Copy the index with a new name
    pub fn rename(&self, name: Option<String>) -> Self {
        let mut new_index = self.clone();
        new_index.name = name;
        new_index
    }
}

/// Common trait for index types
///
/// Provides common functionality for different types of indices.
pub trait IndexTrait {
    /// Get the length of the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> IndexTrait for Index<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> IndexTrait for MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    fn len(&self) -> usize {
        self.len()
    }
}

/// Index type used by DataFrame
///
/// An enum for uniformly handling both single-level indices and multi-level indices.
#[derive(Debug, Clone)]
pub enum DataFrameIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Single-level index
    Simple(Index<T>),
    /// Multi-level index
    Multi(MultiIndex<T>),
}

impl<T> IndexTrait for DataFrameIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    fn len(&self) -> usize {
        match self {
            DataFrameIndex::Simple(idx) => idx.len(),
            DataFrameIndex::Multi(idx) => idx.len(),
        }
    }
}

impl<T> DataFrameIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Create from a simple index
    pub fn from_simple(index: Index<T>) -> Self {
        DataFrameIndex::Simple(index)
    }

    /// Create from a multi-index
    pub fn from_multi(index: MultiIndex<T>) -> Self {
        DataFrameIndex::Multi(index)
    }

    /// Create a default index
    ///
    /// # Arguments
    /// * `len` - Length of the index
    ///
    /// # Returns
    /// A default index with sequential numbers from 0 to len-1
    pub fn default_with_len(len: usize) -> Result<DataFrameIndex<String>> {
        let range_idx = Index::<usize>::from_range(0..len)?;
        let string_values: Vec<String> = range_idx.values().iter().map(|i| i.to_string()).collect();
        let string_idx = Index::<String>::new(string_values)?;
        Ok(DataFrameIndex::Simple(string_idx))
    }

    /// Check if the index is a multi-index
    pub fn is_multi(&self) -> bool {
        matches!(self, DataFrameIndex::Multi(_))
    }
}

/// Alias for integer index type
pub type RangeIndex = Index<usize>;

/// Alias for string index type
pub type StringIndex = Index<String>;

/// Extension methods for date/time conversion
impl StringIndex {
    /// Convert index values to an array of date/time objects
    ///
    /// If the index contains date strings, converts them to NaiveDate.
    /// If conversion fails, uses the current date.
    pub fn to_datetime_vec(&self) -> Result<Vec<NaiveDate>> {
        let mut result = Vec::with_capacity(self.len());

        for value in &self.values {
            // Parse date format string
            match NaiveDate::parse_from_str(value, "%Y-%m-%d") {
                Ok(date) => result.push(date),
                Err(_) => {
                    // If date parsing fails, use the current date
                    result.push(
                        NaiveDate::parse_from_str("2023-01-01", "%Y-%m-%d")
                            .unwrap_or_else(|_| NaiveDate::now()),
                    );
                }
            }
        }

        Ok(result)
    }
}

/// Extension methods for DataFrameIndex
impl DataFrameIndex<String> {
    /// Convert index values to an array of date/time objects
    pub fn to_datetime_vec(&self) -> Result<Vec<NaiveDate>> {
        match self {
            DataFrameIndex::Simple(idx) => idx.to_datetime_vec(),
            DataFrameIndex::Multi(_) => {
                // For multi-index, use the first level
                Err(PandRSError::NotImplemented(
                    "Date/time conversion for multi-index is not currently supported".to_string(),
                ))
            }
        }
    }

    /// Get string values of the index
    pub fn string_values(&self) -> Option<Vec<String>> {
        match self {
            DataFrameIndex::Simple(idx) => Some(idx.values().iter().map(|v| v.clone()).collect()),
            DataFrameIndex::Multi(multi_idx) => {
                // Return simplified string representation of multi-index
                Some(
                    multi_idx
                        .tuples()
                        .iter()
                        .map(|tuple| tuple.join(", "))
                        .collect(),
                )
            }
        }
    }
}

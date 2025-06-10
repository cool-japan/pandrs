use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::{Error, Result};

/// Trait defining common operations for indexes
pub trait IndexTrait: Debug + Clone + Send + Sync {
    /// Returns the length of the index
    fn len(&self) -> usize;

    /// Returns whether the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the position of a value in the index
    fn get_position(&self, value: &str) -> Option<usize>;

    /// Gets the value at a specified position
    fn get_value(&self, position: usize) -> Result<String>;

    /// Gets all values in the index as a vector of strings
    fn get_values(&self) -> Vec<String>;

    /// Clones the index
    fn clone_index(&self) -> Index;
}

/// Enum representing an index
#[derive(Debug, Clone)]
pub enum Index {
    Range(RangeIndex),
    String(StringIndex),
}

/// Range-based index (0, 1, 2, ...)
#[derive(Debug, Clone)]
pub struct RangeIndex {
    start: usize,
    len: usize,
    name: Option<String>,
}

impl RangeIndex {
    /// Creates a new range index
    pub fn new(start: usize, len: usize, name: Option<String>) -> Self {
        Self { start, len, name }
    }

    /// Creates a range index from 0 to len-1
    pub fn from_len(len: usize) -> Self {
        Self::new(0, len, None)
    }

    /// Returns the name of the index
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl IndexTrait for RangeIndex {
    fn len(&self) -> usize {
        self.len
    }

    fn get_position(&self, value: &str) -> Option<usize> {
        match value.parse::<usize>() {
            Ok(pos) if pos >= self.start && pos < self.start + self.len => Some(pos - self.start),
            _ => None,
        }
    }

    fn get_value(&self, position: usize) -> Result<String> {
        if position < self.len {
            Ok((self.start + position).to_string())
        } else {
            Err(Error::IndexOutOfBounds {
                index: position,
                size: self.len,
            })
        }
    }

    fn get_values(&self) -> Vec<String> {
        (self.start..self.start + self.len)
            .map(|i| i.to_string())
            .collect()
    }

    fn clone_index(&self) -> Index {
        Index::Range(self.clone())
    }
}

/// String-based index
#[derive(Debug, Clone)]
pub struct StringIndex {
    values: Arc<[String]>,
    map: std::collections::HashMap<String, usize>,
    name: Option<String>,
}

impl StringIndex {
    /// Creates a new string index
    pub fn new(values: Vec<String>, name: Option<String>) -> Self {
        let mut map = std::collections::HashMap::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            map.insert(v.clone(), i);
        }

        Self {
            values: values.into(),
            map,
            name,
        }
    }

    /// Returns the name of the index
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl IndexTrait for StringIndex {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn get_position(&self, value: &str) -> Option<usize> {
        self.map.get(value).copied()
    }

    fn get_value(&self, position: usize) -> Result<String> {
        if position < self.values.len() {
            Ok(self.values[position].clone())
        } else {
            Err(Error::IndexOutOfBounds {
                index: position,
                size: self.values.len(),
            })
        }
    }

    fn get_values(&self) -> Vec<String> {
        self.values.to_vec()
    }

    fn clone_index(&self) -> Index {
        Index::String(self.clone())
    }
}

// Index enum implementation
impl Index {
    /// Creates a new range index
    pub fn range(start: usize, len: usize, name: Option<String>) -> Self {
        Index::Range(RangeIndex::new(start, len, name))
    }

    /// Creates a range index from 0 to len-1
    pub fn from_len(len: usize) -> Self {
        Index::Range(RangeIndex::from_len(len))
    }

    /// Creates a new string index
    pub fn string(values: Vec<String>, name: Option<String>) -> Self {
        Index::String(StringIndex::new(values, name))
    }

    /// Returns the length of the index
    pub fn len(&self) -> usize {
        match self {
            Index::Range(idx) => idx.len(),
            Index::String(idx) => idx.len(),
        }
    }

    /// Returns whether the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the position of a value in the index
    pub fn get_position(&self, value: &str) -> Option<usize> {
        match self {
            Index::Range(idx) => idx.get_position(value),
            Index::String(idx) => idx.get_position(value),
        }
    }

    /// Gets the value at a specified position
    pub fn get_value(&self, position: usize) -> Result<String> {
        match self {
            Index::Range(idx) => idx.get_value(position),
            Index::String(idx) => idx.get_value(position),
        }
    }

    /// Gets all values in the index as a vector of strings
    pub fn get_values(&self) -> Vec<String> {
        match self {
            Index::Range(idx) => idx.get_values(),
            Index::String(idx) => idx.get_values(),
        }
    }

    /// Returns the name of the index
    pub fn name(&self) -> Option<&str> {
        match self {
            Index::Range(idx) => idx.name(),
            Index::String(idx) => idx.name(),
        }
    }
}

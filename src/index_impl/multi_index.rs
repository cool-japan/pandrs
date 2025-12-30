use crate::error::{PandRSError, Result};
use crate::index::Index;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// MultiIndex structure
///
/// Represents a multi-level hierarchical index.
/// Provides similar functionality to `MultiIndex` in Python's pandas.
#[derive(Debug, Clone)]
pub struct MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Labels for each level
    levels: Vec<Vec<T>>,
    
    /// Codes indicating the index positions for each level
    codes: Vec<Vec<i32>>,
    
    /// Names for each level
    names: Vec<Option<String>>,
    
    /// Mapping from MultiIndex values to positions
    map: HashMap<Vec<T>, usize>,
}

impl<T> MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Create a new MultiIndex
    ///
    /// # Arguments
    /// * `levels` - List of unique values for each level
    /// * `codes` - Codes indicating index positions for each level (-1 for missing values)
    /// * `names` - Names for each level (optional)
    pub fn new(
        levels: Vec<Vec<T>>,
        codes: Vec<Vec<i32>>,
        names: Option<Vec<Option<String>>>,
    ) -> Result<Self> {
        // Basic validation
        if levels.is_empty() {
            return Err(PandRSError::Index("At least one level is required".into()));
        }
        
        if levels.len() != codes.len() {
            return Err(PandRSError::Index(
                "The lengths of levels and codes must match".into()
            ));
        }
        
        // Validate codes
        for (level_idx, level_codes) in codes.iter().enumerate() {
            let max_code = levels[level_idx].len() as i32 - 1;
            for &code in level_codes.iter() {
                if code > max_code && code != -1 {
                    return Err(PandRSError::Index(format!(
                        "Code {} in level {} is out of valid range",
                        code, level_idx
                    )));
                }
            }
        }
        
        // Check row consistency
        let n_rows = if !codes.is_empty() { codes[0].len() } else { 0 };
        for level_codes in &codes {
            if level_codes.len() != n_rows {
                return Err(PandRSError::Index(
                    "All levels must have the same number of rows".into()
                ));
            }
        }
        
        // Validate names length
        let names = match names {
            Some(n) => {
                if n.len() != levels.len() {
                    return Err(PandRSError::Index(
                        "The number of names must match the number of levels".into()
                    ));
                }
                n
            }
            None => vec![None; levels.len()],
        };
        
        // Build the map
        let mut map = HashMap::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row_values = Vec::with_capacity(levels.len());
            
            for level_idx in 0..levels.len() {
                let code = codes[level_idx][i];
                if code == -1 {
                    // Missing values are not supported
                    return Err(PandRSError::NotImplemented(
                        "Missing values are not supported in the current MultiIndex implementation".into()
                    ));
                } else {
                    row_values.push(levels[level_idx][code as usize].clone());
                }
            }
            
            if map.insert(row_values.clone(), i).is_some() {
                return Err(PandRSError::Index(
                    "Duplicate values are not allowed in MultiIndex".into()
                ));
            }
        }
        
        Ok(MultiIndex {
            levels,
            codes,
            names,
            map,
        })
    }
    
    /// Create a MultiIndex from a list of tuples, similar to pandas.MultiIndex.from_tuples
    pub fn from_tuples(tuples: Vec<Vec<T>>, names: Option<Vec<Option<String>>>) -> Result<Self> {
        if tuples.is_empty() {
            return Err(PandRSError::Index("An empty list of tuples was provided".into()));
        }
        
        let n_levels = tuples[0].len();
        
        // Ensure all tuples have the same length
        for (i, tuple) in tuples.iter().enumerate() {
            if tuple.len() != n_levels {
                return Err(PandRSError::Index(format!(
                    "All tuples must have the same length. Tuple {} has length {}, but the first tuple has length {}.",
                    i, tuple.len(), n_levels
                )));
            }
        }
        
        // Collect unique values for each level
        let mut unique_values: Vec<Vec<T>> = vec![Vec::new(); n_levels];
        let mut level_maps: Vec<HashMap<T, i32>> = vec![HashMap::new(); n_levels];
        
        // Build codes for each level
        let mut codes: Vec<Vec<i32>> = vec![vec![-1; tuples.len()]; n_levels];
        
        for (row_idx, tuple) in tuples.iter().enumerate() {
            for (level_idx, value) in tuple.iter().enumerate() {
                let level_map = &mut level_maps[level_idx];
                
                let code = match level_map.get(value) {
                    Some(&code) => code,
                    None => {
                        let new_code = unique_values[level_idx].len() as i32;
                        unique_values[level_idx].push(value.clone());
                        level_map.insert(value.clone(), new_code);
                        new_code
                    }
                };
                
                codes[level_idx][row_idx] = code;
            }
        }
        
        MultiIndex::new(unique_values, codes, names)
    }
    
    /// Get the tuple at a specific position
    pub fn get_tuple(&self, pos: usize) -> Option<Vec<T>> {
        if pos >= self.len() {
            return None;
        }
        
        let mut result = Vec::with_capacity(self.levels.len());
        for level_idx in 0..self.levels.len() {
            let code = self.codes[level_idx][pos];
            if code == -1 {
                // Missing values are not supported
                return None;
            }
            result.push(self.levels[level_idx][code as usize].clone());
        }
        
        Some(result)
    }
    
    /// Get the position of a tuple
    pub fn get_loc(&self, key: &[T]) -> Option<usize> {
        if key.len() != self.levels.len() {
            return None;
        }
        
        // Convert to Vec<T> and search in the map
        let key_vec = key.to_vec();
        self.map.get(&key_vec).copied()
    }
    
    /// Get the number of rows in the index
    pub fn len(&self) -> usize {
        if self.codes.is_empty() {
            0
        } else {
            self.codes[0].len()
        }
    }
    
    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get the number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
    
    /// Get the values of each level
    pub fn levels(&self) -> &[Vec<T>] {
        &self.levels
    }
    
    /// Get the codes of each level
    pub fn codes(&self) -> &[Vec<i32>] {
        &self.codes
    }
    
    /// Get the names of each level
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }
    
    /// Slice based on a specific level
    pub fn get_level_values(&self, level: usize) -> Result<Index<T>> {
        if level >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "Level {} is out of range. Valid levels are 0 to {}",
                level,
                self.levels.len() - 1
            )));
        }
        
        let mut values = Vec::with_capacity(self.len());
        for &code in &self.codes[level] {
            if code == -1 {
                return Err(PandRSError::NotImplemented(
                    "Extracting level values with missing values is not supported".into()
                ));
            }
            values.push(self.levels[level][code as usize].clone());
        }
        
        Index::new(values)
    }
    
    /// Swap levels and create a new MultiIndex
    pub fn swaplevel(&self, i: usize, j: usize) -> Result<Self> {
        if i >= self.levels.len() || j >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "Level indices are out of range. Valid levels are 0 to {}",
                self.levels.len() - 1
            )));
        }
        
        let mut new_levels = self.levels.clone();
        let mut new_codes = self.codes.clone();
        let mut new_names = self.names.clone();
        
        // Swap levels
        new_levels.swap(i, j);
        new_codes.swap(i, j);
        new_names.swap(i, j);
        
        MultiIndex::new(new_levels, new_codes, Some(new_names))
    }
    
    /// Set the names of the levels
    pub fn set_names(&mut self, names: Vec<Option<String>>) -> Result<()> {
        if names.len() != self.levels.len() {
            return Err(PandRSError::Index(format!(
                "The number of names ({}) must match the number of levels ({})",
                names.len(),
                self.levels.len()
            )));
        }
        
        self.names = names;
        Ok(())
    }
}

// Alias for MultiIndex with String type
pub type StringMultiIndex = MultiIndex<String>;
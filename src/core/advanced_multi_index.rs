//! # Advanced MultiIndex with Cross-Section Selection
//!
//! This module provides a comprehensive MultiIndex implementation with advanced
//! cross-section selection capabilities, including partial indexing, level slicing,
//! boolean indexing, and hierarchical navigation.
//!
//! Features:
//! - Cross-section (xs) selection for partial key matching
//! - Level-wise operations and transformations
//! - Boolean and fancy indexing for multi-level data
//! - Hierarchical aggregation and grouping operations
//! - Performance-optimized lookup and navigation

use crate::core::error::{Error, Result};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Value type for MultiIndex entries
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexValue {
    String(String),
    Integer(i64),
    Float(OrderedFloat),
    Boolean(bool),
    Null,
}

impl fmt::Display for IndexValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexValue::String(s) => write!(f, "{}", s),
            IndexValue::Integer(i) => write!(f, "{}", i),
            IndexValue::Float(OrderedFloat(fl)) => write!(f, "{}", fl),
            IndexValue::Boolean(b) => write!(f, "{}", b),
            IndexValue::Null => write!(f, "null"),
        }
    }
}

/// Wrapper for f64 to implement Ord and Eq
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Advanced MultiIndex with cross-section selection capabilities
#[derive(Debug, Clone)]
pub struct AdvancedMultiIndex {
    /// Unique values for each level (sorted for efficient lookup)
    levels: Vec<Vec<IndexValue>>,
    /// Level names
    level_names: Vec<Option<String>>,
    /// Index tuples for each row
    tuples: Vec<Vec<IndexValue>>,
    /// Fast lookup map: tuple -> row index
    tuple_to_index: HashMap<Vec<IndexValue>, usize>,
    /// Level-wise lookup maps for cross-section operations
    level_maps: Vec<BTreeMap<IndexValue, Vec<usize>>>,
    /// Cached cross-section results for performance
    xs_cache: HashMap<CrossSectionKey, Vec<usize>>,
}

/// Key for cross-section selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrossSectionKey {
    pub level: usize,
    pub key: IndexValue,
    pub drop_level: bool,
}

/// Result of cross-section selection
#[derive(Debug, Clone)]
pub struct CrossSectionResult {
    /// Matching row indices
    pub indices: Vec<usize>,
    /// Resulting MultiIndex (if levels were not dropped)
    pub index: Option<AdvancedMultiIndex>,
    /// Whether the selection was successful
    pub found: bool,
}

/// Selection criteria for advanced indexing
#[derive(Debug, Clone)]
pub enum SelectionCriteria {
    /// Exact match on specific levels
    Exact(Vec<(usize, IndexValue)>),
    /// Partial match (cross-section)
    Partial(Vec<(usize, IndexValue)>),
    /// Range selection on a level
    Range(usize, IndexValue, IndexValue),
    /// Boolean mask
    Boolean(Vec<bool>),
    /// Index positions
    Positions(Vec<usize>),
    /// Level-wise selection
    Level(usize, Vec<IndexValue>),
}

impl AdvancedMultiIndex {
    /// Create a new AdvancedMultiIndex
    pub fn new(
        tuples: Vec<Vec<IndexValue>>,
        level_names: Option<Vec<Option<String>>>,
    ) -> Result<Self> {
        if tuples.is_empty() {
            return Err(Error::InvalidInput("Empty tuples provided".to_string()));
        }

        let n_levels = tuples[0].len();
        if n_levels == 0 {
            return Err(Error::InvalidInput("Empty tuple structure".to_string()));
        }

        // Validate all tuples have same length
        for (i, tuple) in tuples.iter().enumerate() {
            if tuple.len() != n_levels {
                return Err(Error::InvalidInput(format!(
                    "Tuple {} has length {}, expected {}",
                    i,
                    tuple.len(),
                    n_levels
                )));
            }
        }

        // Extract unique values for each level
        let mut levels = vec![Vec::new(); n_levels];
        let mut level_sets = vec![HashSet::new(); n_levels];

        for tuple in &tuples {
            for (level_idx, value) in tuple.iter().enumerate() {
                if level_sets[level_idx].insert(value.clone()) {
                    levels[level_idx].push(value.clone());
                }
            }
        }

        // Sort levels for efficient lookup
        for level in &mut levels {
            level.sort();
        }

        // Build tuple-to-index mapping
        let mut tuple_to_index = HashMap::with_capacity(tuples.len());
        for (i, tuple) in tuples.iter().enumerate() {
            if tuple_to_index.insert(tuple.clone(), i).is_some() {
                return Err(Error::InvalidInput(format!(
                    "Duplicate tuple found: {:?}",
                    tuple
                )));
            }
        }

        // Build level-wise lookup maps
        let mut level_maps = vec![BTreeMap::new(); n_levels];
        for (row_idx, tuple) in tuples.iter().enumerate() {
            for (level_idx, value) in tuple.iter().enumerate() {
                level_maps[level_idx]
                    .entry(value.clone())
                    .or_insert_with(Vec::new)
                    .push(row_idx);
            }
        }

        // Set up level names
        let level_names = level_names.unwrap_or_else(|| vec![None; n_levels]);
        if level_names.len() != n_levels {
            return Err(Error::InvalidInput(format!(
                "Level names length {} doesn't match levels count {}",
                level_names.len(),
                n_levels
            )));
        }

        Ok(Self {
            levels,
            level_names,
            tuples,
            tuple_to_index,
            level_maps,
            xs_cache: HashMap::new(),
        })
    }

    /// Create from arrays of values for each level
    pub fn from_arrays(
        arrays: Vec<Vec<IndexValue>>,
        names: Option<Vec<Option<String>>>,
    ) -> Result<Self> {
        if arrays.is_empty() {
            return Err(Error::InvalidInput("Empty arrays provided".to_string()));
        }

        let n_rows = arrays[0].len();

        // Validate all arrays have same length
        for (i, array) in arrays.iter().enumerate() {
            if array.len() != n_rows {
                return Err(Error::InvalidInput(format!(
                    "Array {} has length {}, expected {}",
                    i,
                    array.len(),
                    n_rows
                )));
            }
        }

        // Transpose to get tuples
        let mut tuples = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut tuple = Vec::with_capacity(arrays.len());
            for array in &arrays {
                tuple.push(array[row_idx].clone());
            }
            tuples.push(tuple);
        }

        Self::new(tuples, names)
    }

    /// Get the number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Get level names
    pub fn level_names(&self) -> &[Option<String>] {
        &self.level_names
    }

    /// Get unique values for a level
    pub fn get_level_values(&self, level: usize) -> Result<&[IndexValue]> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }
        Ok(&self.levels[level])
    }

    /// Get tuple at specific row
    pub fn get_tuple(&self, index: usize) -> Result<&[IndexValue]> {
        if index >= self.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.len(),
            });
        }
        Ok(&self.tuples[index])
    }

    /// Cross-section selection: select rows with specific value at given level
    pub fn xs(
        &mut self,
        key: IndexValue,
        level: usize,
        drop_level: bool,
    ) -> Result<CrossSectionResult> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        let cache_key = CrossSectionKey {
            level,
            key: key.clone(),
            drop_level,
        };

        // Check cache first
        if let Some(cached_indices) = self.xs_cache.get(&cache_key) {
            let result_index = if drop_level && self.n_levels() > 1 {
                Some(self.drop_level_from_selection(cached_indices, level)?)
            } else {
                None
            };

            return Ok(CrossSectionResult {
                indices: cached_indices.clone(),
                index: result_index,
                found: !cached_indices.is_empty(),
            });
        }

        // Find matching indices
        let indices = self.level_maps[level]
            .get(&key)
            .map(|v| v.clone())
            .unwrap_or_else(Vec::new);

        // Cache result
        self.xs_cache.insert(cache_key, indices.clone());

        let result_index = if drop_level && self.n_levels() > 1 && !indices.is_empty() {
            Some(self.drop_level_from_selection(&indices, level)?)
        } else {
            None
        };

        Ok(CrossSectionResult {
            indices: indices.clone(),
            index: result_index,
            found: !indices.is_empty(),
        })
    }

    /// Advanced selection with multiple criteria
    pub fn select(&self, criteria: SelectionCriteria) -> Result<Vec<usize>> {
        match criteria {
            SelectionCriteria::Exact(level_values) => self.select_exact(level_values),
            SelectionCriteria::Partial(level_values) => self.select_partial(level_values),
            SelectionCriteria::Range(level, start, end) => self.select_range(level, start, end),
            SelectionCriteria::Boolean(mask) => self.select_boolean(mask),
            SelectionCriteria::Positions(positions) => self.select_positions(positions),
            SelectionCriteria::Level(level, values) => self.select_level(level, values),
        }
    }

    /// Select with exact match on multiple levels
    fn select_exact(&self, level_values: Vec<(usize, IndexValue)>) -> Result<Vec<usize>> {
        if level_values.is_empty() {
            return Ok((0..self.len()).collect());
        }

        // Start with all indices from first constraint
        let (first_level, first_value) = &level_values[0];
        let mut result_indices = self.level_maps[*first_level]
            .get(first_value)
            .map(|v| v.clone())
            .unwrap_or_else(Vec::new);

        // Apply additional constraints by intersection
        for (level, value) in level_values.iter().skip(1) {
            if let Some(level_indices) = self.level_maps[*level].get(value) {
                result_indices = result_indices
                    .into_iter()
                    .filter(|idx| level_indices.contains(idx))
                    .collect();
            } else {
                return Ok(Vec::new()); // No matches
            }
        }

        Ok(result_indices)
    }

    /// Select with partial match (any of the constraints)
    fn select_partial(&self, level_values: Vec<(usize, IndexValue)>) -> Result<Vec<usize>> {
        let mut result_indices = HashSet::new();

        for (level, value) in level_values {
            if let Some(level_indices) = self.level_maps[level].get(&value) {
                for &idx in level_indices {
                    result_indices.insert(idx);
                }
            }
        }

        let mut result: Vec<usize> = result_indices.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Select with range on a specific level
    fn select_range(&self, level: usize, start: IndexValue, end: IndexValue) -> Result<Vec<usize>> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        let mut result_indices = Vec::new();
        for (&ref value, indices) in self.level_maps[level].range(start..=end) {
            result_indices.extend(indices);
        }

        result_indices.sort();
        Ok(result_indices)
    }

    /// Select with boolean mask
    fn select_boolean(&self, mask: Vec<bool>) -> Result<Vec<usize>> {
        if mask.len() != self.len() {
            return Err(Error::InvalidInput(format!(
                "Boolean mask length {} doesn't match index length {}",
                mask.len(),
                self.len()
            )));
        }

        let result_indices = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &include)| if include { Some(i) } else { None })
            .collect();

        Ok(result_indices)
    }

    /// Select specific positions
    fn select_positions(&self, positions: Vec<usize>) -> Result<Vec<usize>> {
        for &pos in &positions {
            if pos >= self.len() {
                return Err(Error::IndexOutOfBounds {
                    index: pos,
                    size: self.len(),
                });
            }
        }

        Ok(positions)
    }

    /// Select all rows with any of the specified values at a level
    fn select_level(&self, level: usize, values: Vec<IndexValue>) -> Result<Vec<usize>> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        let mut result_indices = HashSet::new();

        for value in values {
            if let Some(level_indices) = self.level_maps[level].get(&value) {
                for &idx in level_indices {
                    result_indices.insert(idx);
                }
            }
        }

        let mut result: Vec<usize> = result_indices.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Create a new index by dropping a level from selected rows
    fn drop_level_from_selection(
        &self,
        indices: &[usize],
        drop_level: usize,
    ) -> Result<AdvancedMultiIndex> {
        if self.n_levels() <= 1 {
            return Err(Error::InvalidInput(
                "Cannot drop level from single-level index".to_string(),
            ));
        }

        let mut new_tuples = Vec::with_capacity(indices.len());
        for &idx in indices {
            let original_tuple = &self.tuples[idx];
            let mut new_tuple = Vec::with_capacity(original_tuple.len() - 1);

            for (level_idx, value) in original_tuple.iter().enumerate() {
                if level_idx != drop_level {
                    new_tuple.push(value.clone());
                }
            }
            new_tuples.push(new_tuple);
        }

        // Create new level names
        let mut new_level_names = Vec::with_capacity(self.level_names.len() - 1);
        for (level_idx, name) in self.level_names.iter().enumerate() {
            if level_idx != drop_level {
                new_level_names.push(name.clone());
            }
        }

        AdvancedMultiIndex::new(new_tuples, Some(new_level_names))
    }

    /// Get all unique tuples for a partial key at specified levels
    pub fn get_group_keys(&self, levels: &[usize]) -> Result<Vec<Vec<IndexValue>>> {
        for &level in levels {
            if level >= self.n_levels() {
                return Err(Error::IndexOutOfBounds {
                    index: level,
                    size: self.n_levels(),
                });
            }
        }

        let mut unique_keys = HashSet::new();
        for tuple in &self.tuples {
            let key: Vec<IndexValue> = levels.iter().map(|&i| tuple[i].clone()).collect();
            unique_keys.insert(key);
        }

        let mut result: Vec<Vec<IndexValue>> = unique_keys.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get indices for rows matching partial key
    pub fn get_group_indices(&self, levels: &[usize], key: &[IndexValue]) -> Result<Vec<usize>> {
        if levels.len() != key.len() {
            return Err(Error::InvalidInput(format!(
                "Levels length {} doesn't match key length {}",
                levels.len(),
                key.len()
            )));
        }

        for &level in levels {
            if level >= self.n_levels() {
                return Err(Error::IndexOutOfBounds {
                    index: level,
                    size: self.n_levels(),
                });
            }
        }

        let mut result_indices = Vec::new();
        for (row_idx, tuple) in self.tuples.iter().enumerate() {
            let matches = levels
                .iter()
                .zip(key.iter())
                .all(|(&level_idx, key_value)| &tuple[level_idx] == key_value);

            if matches {
                result_indices.push(row_idx);
            }
        }

        Ok(result_indices)
    }

    /// Create a slice of the index
    pub fn slice(&self, start: usize, end: usize) -> Result<AdvancedMultiIndex> {
        if start > end || end > self.len() {
            return Err(Error::InvalidInput(format!(
                "Invalid slice range: {}..{} for length {}",
                start,
                end,
                self.len()
            )));
        }

        let sliced_tuples = self.tuples[start..end].to_vec();
        AdvancedMultiIndex::new(sliced_tuples, Some(self.level_names.clone()))
    }

    /// Reorder levels
    pub fn reorder_levels(&self, new_order: &[usize]) -> Result<AdvancedMultiIndex> {
        if new_order.len() != self.n_levels() {
            return Err(Error::InvalidInput(format!(
                "New order length {} doesn't match levels count {}",
                new_order.len(),
                self.n_levels()
            )));
        }

        for &level in new_order {
            if level >= self.n_levels() {
                return Err(Error::IndexOutOfBounds {
                    index: level,
                    size: self.n_levels(),
                });
            }
        }

        let mut reordered_tuples = Vec::with_capacity(self.len());
        for tuple in &self.tuples {
            let reordered_tuple: Vec<IndexValue> =
                new_order.iter().map(|&i| tuple[i].clone()).collect();
            reordered_tuples.push(reordered_tuple);
        }

        let reordered_names: Vec<Option<String>> = new_order
            .iter()
            .map(|&i| self.level_names[i].clone())
            .collect();

        AdvancedMultiIndex::new(reordered_tuples, Some(reordered_names))
    }

    /// Clear the cross-section cache
    pub fn clear_cache(&mut self) {
        self.xs_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            size: self.xs_cache.len(),
            max_recommended_size: 1000, // Could be configurable
        }
    }
}

/// Cache statistics for performance monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_recommended_size: usize,
}

impl CacheStats {
    pub fn should_clear(&self) -> bool {
        self.size > self.max_recommended_size
    }
}

impl fmt::Display for AdvancedMultiIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "AdvancedMultiIndex with {} levels and {} rows",
            self.n_levels(),
            self.len()
        )?;

        // Show level names if available
        if self.level_names.iter().any(|name| name.is_some()) {
            write!(f, "Level names: [")?;
            for (i, name) in self.level_names.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match name {
                    Some(n) => write!(f, "'{}'", n)?,
                    None => write!(f, "None")?,
                }
            }
            writeln!(f, "]")?;
        }

        // Show sample of tuples
        let max_show = 10.min(self.len());
        for i in 0..max_show {
            write!(f, "  (")?;
            for (j, value) in self.tuples[i].iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", value)?;
            }
            writeln!(f, ")")?;
        }

        if self.len() > max_show {
            writeln!(f, "  ... ({} more rows)", self.len() - max_show)?;
        }

        Ok(())
    }
}

// Convenience constructors for IndexValue
impl From<String> for IndexValue {
    fn from(s: String) -> Self {
        IndexValue::String(s)
    }
}

impl From<&str> for IndexValue {
    fn from(s: &str) -> Self {
        IndexValue::String(s.to_string())
    }
}

impl From<i64> for IndexValue {
    fn from(i: i64) -> Self {
        IndexValue::Integer(i)
    }
}

impl From<f64> for IndexValue {
    fn from(f: f64) -> Self {
        IndexValue::Float(OrderedFloat(f))
    }
}

impl From<bool> for IndexValue {
    fn from(b: bool) -> Self {
        IndexValue::Boolean(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_multi_index_creation() {
        let tuples = vec![
            vec![IndexValue::from("A"), IndexValue::from(1)],
            vec![IndexValue::from("A"), IndexValue::from(2)],
            vec![IndexValue::from("B"), IndexValue::from(1)],
            vec![IndexValue::from("B"), IndexValue::from(2)],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();
        assert_eq!(index.n_levels(), 2);
        assert_eq!(index.len(), 4);
    }

    #[test]
    fn test_cross_section_selection() {
        let tuples = vec![
            vec![IndexValue::from("A"), IndexValue::from(1)],
            vec![IndexValue::from("A"), IndexValue::from(2)],
            vec![IndexValue::from("B"), IndexValue::from(1)],
            vec![IndexValue::from("B"), IndexValue::from(2)],
        ];

        let mut index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // Select all rows with 'A' at level 0
        let result = index.xs(IndexValue::from("A"), 0, false).unwrap();
        assert_eq!(result.indices, vec![0, 1]);
        assert!(result.found);

        // Select all rows with value 1 at level 1
        let result = index.xs(IndexValue::from(1), 1, false).unwrap();
        assert_eq!(result.indices, vec![0, 2]);
        assert!(result.found);
    }

    #[test]
    fn test_exact_selection() {
        let tuples = vec![
            vec![
                IndexValue::from("A"),
                IndexValue::from(1),
                IndexValue::from("X"),
            ],
            vec![
                IndexValue::from("A"),
                IndexValue::from(2),
                IndexValue::from("Y"),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from(1),
                IndexValue::from("X"),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from(2),
                IndexValue::from("Z"),
            ],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // Exact match on multiple levels
        let criteria =
            SelectionCriteria::Exact(vec![(0, IndexValue::from("A")), (1, IndexValue::from(1))]);

        let result = index.select(criteria).unwrap();
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_range_selection() {
        let tuples = vec![
            vec![IndexValue::from("A"), IndexValue::from(1)],
            vec![IndexValue::from("A"), IndexValue::from(5)],
            vec![IndexValue::from("B"), IndexValue::from(3)],
            vec![IndexValue::from("B"), IndexValue::from(7)],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // Range selection on level 1 (values 2-6)
        let criteria = SelectionCriteria::Range(1, IndexValue::from(2), IndexValue::from(6));
        let result = index.select(criteria).unwrap();
        assert_eq!(result, vec![1, 2]); // Values 5 and 3
    }

    #[test]
    fn test_boolean_selection() {
        let tuples = vec![
            vec![IndexValue::from("A"), IndexValue::from(1)],
            vec![IndexValue::from("A"), IndexValue::from(2)],
            vec![IndexValue::from("B"), IndexValue::from(1)],
            vec![IndexValue::from("B"), IndexValue::from(2)],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        let criteria = SelectionCriteria::Boolean(vec![true, false, true, false]);
        let result = index.select(criteria).unwrap();
        assert_eq!(result, vec![0, 2]);
    }

    #[test]
    fn test_group_operations() {
        let tuples = vec![
            vec![
                IndexValue::from("A"),
                IndexValue::from("X"),
                IndexValue::from(1),
            ],
            vec![
                IndexValue::from("A"),
                IndexValue::from("Y"),
                IndexValue::from(2),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from("X"),
                IndexValue::from(3),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from("Y"),
                IndexValue::from(4),
            ],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // Get unique keys for first two levels
        let keys = index.get_group_keys(&[0, 1]).unwrap();
        assert_eq!(keys.len(), 4); // All combinations are unique

        // Get indices for specific group
        let indices = index
            .get_group_indices(&[0], &[IndexValue::from("A")])
            .unwrap();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_level_reordering() {
        let tuples = vec![
            vec![
                IndexValue::from("A"),
                IndexValue::from(1),
                IndexValue::from("X"),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from(2),
                IndexValue::from("Y"),
            ],
        ];

        let names = Some(vec![
            Some("first".to_string()),
            Some("second".to_string()),
            Some("third".to_string()),
        ]);

        let index = AdvancedMultiIndex::new(tuples, names).unwrap();

        // Reorder levels: [2, 0, 1]
        let reordered = index.reorder_levels(&[2, 0, 1]).unwrap();

        assert_eq!(reordered.level_names()[0], Some("third".to_string()));
        assert_eq!(reordered.level_names()[1], Some("first".to_string()));
        assert_eq!(reordered.level_names()[2], Some("second".to_string()));

        // Check that data was reordered correctly
        let first_tuple = reordered.get_tuple(0).unwrap();
        assert_eq!(first_tuple[0], IndexValue::from("X")); // Originally at level 2
        assert_eq!(first_tuple[1], IndexValue::from("A")); // Originally at level 0
        assert_eq!(first_tuple[2], IndexValue::from(1)); // Originally at level 1
    }
}

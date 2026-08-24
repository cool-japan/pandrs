use crate::error::{PandRSError, Result};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// MultiIndex structure
///
/// Represents a hierarchical index with multiple levels.
/// Provides functionality similar to pandas `MultiIndex` in Python.
#[derive(Debug, Clone)]
pub struct MultiIndex<T>
where
    T: Debug + Clone + Eq + Hash + Display,
{
    /// Labels for each level
    levels: Vec<Vec<T>>,

    /// Codes indicating the index of values for each level
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
    /// Get all tuples
    pub fn tuples(&self) -> Vec<Vec<T>> {
        let mut result = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            if let Some(tuple) = self.get_tuple(i) {
                result.push(tuple);
            }
        }
        result
    }
    /// Creates a new MultiIndex
    ///
    /// # Arguments
    /// * `levels` - List of unique values for each level
    /// * `codes` - Codes indicating index positions at each level (-1 represents missing values)
    /// * `names` - Names for each level (optional)
    ///
    /// # Returns
    /// A new MultiIndex instance if successful, otherwise an error
    ///
    /// # Errors
    /// - If levels are empty
    /// - If lengths of levels and codes don't match
    /// - If codes are out of range
    /// - If row counts across levels don't match
    /// - If names length doesn't match levels length when specified
    /// - If missing values are included (not currently supported)
    /// - If duplicate values are included
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
                "Lengths of levels and codes must match".into(),
            ));
        }

        // Validate code values
        for (level_idx, level_codes) in codes.iter().enumerate() {
            let max_code = levels[level_idx].len() as i32 - 1;
            for &code in level_codes.iter() {
                if code > max_code && code != -1 {
                    return Err(PandRSError::Index(format!(
                        "Code {} at level {} is out of valid range",
                        level_idx, code
                    )));
                }
            }
        }

        // Check row count consistency
        let n_rows = if !codes.is_empty() { codes[0].len() } else { 0 };
        for level_codes in &codes {
            if level_codes.len() != n_rows {
                return Err(PandRSError::Index(
                    "All levels must have the same number of rows".into(),
                ));
            }
        }

        // Verify name count
        let names = match names {
            Some(n) => {
                if n.len() != levels.len() {
                    return Err(PandRSError::Index(
                        "Names must have the same length as levels".into(),
                    ));
                }
                n
            }
            None => vec![None; levels.len()],
        };

        // Build mapping
        // Rows where ANY level code is -1 (missing) are excluded from the reverse lookup map,
        // because a missing key is not addressable by get_loc (pandas semantics).
        let mut map = HashMap::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row_values = Vec::with_capacity(levels.len());
            let mut has_missing = false;

            for level_idx in 0..levels.len() {
                let code = codes[level_idx][i];
                if code == -1 {
                    has_missing = true;
                    break;
                } else {
                    row_values.push(levels[level_idx][code as usize].clone());
                }
            }

            // Skip insertion into the reverse-lookup map for rows with any missing code.
            // Such rows are stored in self.codes but are not addressable via get_loc.
            if !has_missing {
                if map.insert(row_values.clone(), i).is_some() {
                    return Err(PandRSError::Index(
                        "Duplicate values are not allowed in MultiIndex".into(),
                    ));
                }
            }
        }

        Ok(MultiIndex {
            levels,
            codes,
            names,
            map,
        })
    }

    /// Creates a MultiIndex from a list of tuples
    ///
    /// Provides functionality similar to pandas.MultiIndex.from_tuples in Python.
    ///
    /// # Arguments
    /// * `tuples` - Vector of tuples (each tuple must have the same length)
    /// * `names` - Names for each level (optional)
    ///
    /// # Returns
    /// A new MultiIndex instance if successful, otherwise an error
    ///
    /// # Errors
    /// - If an empty tuple list is passed
    /// - If tuple lengths don't match
    pub fn from_tuples(tuples: Vec<Vec<T>>, names: Option<Vec<Option<String>>>) -> Result<Self> {
        if tuples.is_empty() {
            return Err(PandRSError::Index("Empty tuple list was passed".into()));
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

    /// Gets a tuple at a specific position
    ///
    /// # Arguments
    /// * `pos` - Position to retrieve
    ///
    /// # Returns
    /// The tuple if position is valid, None if out of range
    pub fn get_tuple(&self, pos: usize) -> Option<Vec<T>> {
        if pos >= self.len() {
            return None;
        }

        let mut result = Vec::with_capacity(self.levels.len());
        for level_idx in 0..self.levels.len() {
            let code = self.codes[level_idx][pos];
            if code == -1 {
                // Missing values are not currently supported
                return None;
            }
            result.push(self.levels[level_idx][code as usize].clone());
        }

        Some(result)
    }

    /// Gets the position from a tuple
    ///
    /// # Arguments
    /// * `key` - Tuple to search for
    ///
    /// # Returns
    /// The position if tuple is found, None if not found
    pub fn get_loc(&self, key: &[T]) -> Option<usize> {
        if key.len() != self.levels.len() {
            return None;
        }

        // Convert to Vec<T> and search in map
        let key_vec = key.to_vec();
        self.map.get(&key_vec).copied()
    }

    /// Gets the length (number of rows) of the index
    pub fn len(&self) -> usize {
        if self.codes.is_empty() {
            0
        } else {
            self.codes[0].len()
        }
    }

    /// Determines if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets the number of levels
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Gets the values for each level
    pub fn levels(&self) -> &[Vec<T>] {
        &self.levels
    }

    /// Gets the codes for each level
    pub fn codes(&self) -> &[Vec<i32>] {
        &self.codes
    }

    /// Gets the names for each level
    pub fn names(&self) -> &[Option<String>] {
        &self.names
    }

    /// Gets the values for a specific level
    ///
    /// Returns a `Vec<Option<T>>` where `None` indicates a missing value (code == -1),
    /// following pandas semantics for MultiIndex missing-value handling.
    ///
    /// # Arguments
    /// * `level` - Index of the level to retrieve
    ///
    /// # Returns
    /// Vector of optional level values if successful, error if level is out of range
    ///
    /// # Errors
    /// - If level is out of range
    pub fn get_level_values(&self, level: usize) -> Result<Vec<Option<T>>> {
        if level >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "Level {} is out of range. Valid levels are from 0 to {}",
                level,
                self.levels.len() - 1
            )));
        }

        let mut values = Vec::with_capacity(self.len());
        for &code in &self.codes[level] {
            if code == -1 {
                values.push(None);
            } else {
                values.push(Some(self.levels[level][code as usize].clone()));
            }
        }

        Ok(values)
    }

    /// Gets the tuple for a row, with `None` elements for any level that has a -1 (missing) code.
    ///
    /// Unlike `get_tuple` (which returns `None` when any level code is missing, for backwards
    /// compatibility), this method always returns `Some(Vec<Option<T>>)` for valid row indices,
    /// with individual `None` elements for missing level values.
    ///
    /// # Arguments
    /// * `row` - Row index to retrieve
    ///
    /// # Returns
    /// `Some(Vec<Option<T>>)` with per-level optionals, or `None` if `row >= self.len()`
    pub fn get_tuple_opt(&self, row: usize) -> Option<Vec<Option<T>>> {
        if row >= self.len() {
            return None;
        }

        let mut result = Vec::with_capacity(self.levels.len());
        for level_idx in 0..self.levels.len() {
            let code = self.codes[level_idx][row];
            if code == -1 {
                result.push(None);
            } else {
                result.push(Some(self.levels[level_idx][code as usize].clone()));
            }
        }

        Some(result)
    }

    /// Creates a new MultiIndex by swapping levels
    ///
    /// # Arguments
    /// * `i` - Index of the first level to swap
    /// * `j` - Index of the second level to swap
    ///
    /// # Returns
    /// A new MultiIndex with swapped levels if successful, error if failed
    ///
    /// # Errors
    /// - If level is out of range
    pub fn swaplevel(&self, i: usize, j: usize) -> Result<Self> {
        if i >= self.levels.len() || j >= self.levels.len() {
            return Err(PandRSError::Index(format!(
                "Level specification is out of range. Valid levels are from 0 to {}",
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

    /// Sets the names for the specified levels
    ///
    /// # Arguments
    /// * `names` - Vector of new level names
    ///
    /// # Returns
    /// () if successful, error if failed
    ///
    /// # Errors
    /// - If the length of names doesn't match the number of levels
    pub fn set_names(&mut self, names: Vec<Option<String>>) -> Result<()> {
        if names.len() != self.levels.len() {
            return Err(PandRSError::Index(format!(
                "Number of names {} must match the number of levels {}",
                names.len(),
                self.levels.len()
            )));
        }

        self.names = names;
        Ok(())
    }
}

/// Alias for string MultiIndex
pub type StringMultiIndex = MultiIndex<String>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that constructing a MultiIndex with a -1 code (missing-value sentinel) succeeds,
    /// that get_loc returns None for the partially-missing row (since it is excluded from the
    /// reverse-lookup map), and that get_level_values returns None at the missing position.
    #[test]
    fn test_multiindex_missing_value() {
        // Level 0 has values ["A", "B"], level 1 has values ["1", "2"].
        // Row 1 has code -1 at level 0, which means it has a missing value at that level.
        let levels = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["1".to_string(), "2".to_string()],
        ];

        // Row 0: (A, 1), Row 1: (MISSING, 2), Row 2: (B, 1)
        let codes = vec![vec![0_i32, -1, 1], vec![0_i32, 1, 0]];

        let mi =
            MultiIndex::new(levels, codes, None).expect("MultiIndex with -1 code must succeed");

        assert_eq!(mi.len(), 3);

        // Row 0 is fully specified — must be locatable.
        let loc_row0 = mi.get_loc(&["A".to_string(), "1".to_string()]);
        assert!(
            loc_row0.is_some(),
            "Row 0 (A,1) must be addressable via get_loc"
        );
        assert_eq!(loc_row0.unwrap(), 0);

        // Row 1 has a missing code, so it must NOT appear in the reverse-lookup map.
        // There is no fully-resolved key for row 1 — searching for ("MISSING", "2") is
        // semantically undefined; instead, we verify that no spurious entry was inserted.
        let loc_not_present = mi.get_loc(&["MISSING".to_string(), "2".to_string()]);
        assert!(
            loc_not_present.is_none(),
            "A key containing the literal string MISSING must not be found"
        );

        // get_level_values must return Vec<Option<T>> with None at the missing position.
        let level0_vals = mi
            .get_level_values(0)
            .expect("get_level_values(0) must succeed");
        assert_eq!(level0_vals.len(), 3);
        assert_eq!(level0_vals[0], Some("A".to_string()));
        assert_eq!(level0_vals[1], None, "Row 1 level 0 must be None (missing)");
        assert_eq!(level0_vals[2], Some("B".to_string()));

        let level1_vals = mi
            .get_level_values(1)
            .expect("get_level_values(1) must succeed");
        assert_eq!(level1_vals[0], Some("1".to_string()));
        assert_eq!(level1_vals[1], Some("2".to_string()));
        assert_eq!(level1_vals[2], Some("1".to_string()));
    }

    /// Verifies that get_tuple_opt returns Some with None elements for -1 codes,
    /// and that get_tuple (backwards-compat) returns None for rows with any missing code.
    #[test]
    fn test_multiindex_get_tuple_opt() {
        let levels = vec![
            vec!["X".to_string(), "Y".to_string()],
            vec!["p".to_string(), "q".to_string()],
        ];

        // Row 0: (X, p), Row 1: (MISSING, q), Row 2: (Y, MISSING)
        let codes = vec![vec![0_i32, -1, 1], vec![0_i32, 1, -1]];

        let mi =
            MultiIndex::new(levels, codes, None).expect("MultiIndex with -1 codes must succeed");

        // get_tuple_opt for row 0 must return all-Some.
        let opt0 = mi.get_tuple_opt(0).expect("row 0 must exist");
        assert_eq!(opt0, vec![Some("X".to_string()), Some("p".to_string())]);

        // get_tuple_opt for row 1: level 0 is missing.
        let opt1 = mi.get_tuple_opt(1).expect("row 1 must exist");
        assert_eq!(opt1, vec![None, Some("q".to_string())]);

        // get_tuple_opt for row 2: level 1 is missing.
        let opt2 = mi.get_tuple_opt(2).expect("row 2 must exist");
        assert_eq!(opt2, vec![Some("Y".to_string()), None]);

        // Out-of-bounds must return None.
        assert!(mi.get_tuple_opt(3).is_none());

        // get_tuple (backwards compat) must return None for rows with missing codes.
        assert!(
            mi.get_tuple(1).is_none(),
            "get_tuple must return None for rows with any -1 code"
        );
        assert!(
            mi.get_tuple(2).is_none(),
            "get_tuple must return None for rows with any -1 code"
        );
        // get_tuple for a fully-specified row must still work.
        assert_eq!(
            mi.get_tuple(0),
            Some(vec!["X".to_string(), "p".to_string()])
        );
    }
}

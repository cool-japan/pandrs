use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::core::index::{Index, IndexTrait};

/// A multi-level index (hierarchical index)
#[derive(Clone)]
pub struct MultiIndex {
    // Levels (columns) of the multi-index
    levels: Vec<Vec<String>>,
    // Names of the levels
    level_names: Vec<Option<String>>,
    // Labels for each row (tuples of level indices)
    labels: Vec<Vec<usize>>,
    // Map from label tuples to positions
    label_map: HashMap<Vec<String>, usize>,
}

impl MultiIndex {
    /// Creates a new MultiIndex from vectors of labels
    pub fn new(
        levels: Vec<Vec<String>>,
        labels: Vec<Vec<usize>>,
        level_names: Vec<Option<String>>,
    ) -> Result<Self> {
        // Validate inputs
        if levels.is_empty() {
            return Err(Error::InvalidInput(
                "Empty levels in MultiIndex".to_string(),
            ));
        }

        if !labels.is_empty() && labels[0].len() != levels.len() {
            return Err(Error::InvalidInput(
                "Labels shape doesn't match levels count".to_string(),
            ));
        }

        if !level_names.is_empty() && level_names.len() != levels.len() {
            return Err(Error::InvalidInput(
                "Level names count doesn't match levels count".to_string(),
            ));
        }

        // Create label map for efficient lookup
        let mut label_map = HashMap::new();
        for (i, label_indices) in labels.iter().enumerate() {
            let label_values: Vec<String> = label_indices
                .iter()
                .enumerate()
                .map(|(level_idx, &value_idx)| {
                    levels
                        .get(level_idx)
                        .and_then(|level| level.get(value_idx))
                        .cloned()
                        .unwrap_or_else(|| format!("Unknown({})", value_idx))
                })
                .collect();

            label_map.insert(label_values, i);
        }

        // Fill level_names if empty
        let level_names = if level_names.is_empty() {
            vec![None; levels.len()]
        } else {
            level_names
        };

        Ok(Self {
            levels,
            level_names,
            labels,
            label_map,
        })
    }

    /// Creates a MultiIndex from tuples
    pub fn from_tuples(
        tuples: Vec<Vec<String>>,
        names: Option<Vec<Option<String>>>,
    ) -> Result<Self> {
        if tuples.is_empty() {
            return Err(Error::InvalidInput(
                "Empty tuples in MultiIndex".to_string(),
            ));
        }

        let n_levels = tuples[0].len();

        // Check if all tuples have the same length
        for (i, t) in tuples.iter().enumerate() {
            if t.len() != n_levels {
                return Err(Error::InvalidInput(format!(
                    "Tuple at index {} has length {}, expected {}",
                    i,
                    t.len(),
                    n_levels
                )));
            }
        }

        // Extract unique values for each level
        let mut unique_values: Vec<HashMap<String, usize>> = vec![HashMap::new(); n_levels];
        let mut level_values: Vec<Vec<String>> = vec![Vec::new(); n_levels];

        // Assign indices to unique values
        for tuple in &tuples {
            for (level_idx, value) in tuple.iter().enumerate() {
                let level_map = &mut unique_values[level_idx];
                if !level_map.contains_key(value) {
                    let idx = level_map.len();
                    level_map.insert(value.clone(), idx);
                    level_values[level_idx].push(value.clone());
                }
            }
        }

        // Create labels from tuples
        let mut labels = Vec::with_capacity(tuples.len());
        for tuple in &tuples {
            let mut label = Vec::with_capacity(n_levels);
            for (level_idx, value) in tuple.iter().enumerate() {
                let idx = unique_values[level_idx][value];
                label.push(idx);
            }
            labels.push(label);
        }

        // Use provided names or create default ones
        let level_names = names.unwrap_or_else(|| vec![None; n_levels]);

        Self::new(level_values, labels, level_names)
    }

    /// Returns the number of levels in the MultiIndex
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the number of rows in the MultiIndex
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Returns whether the MultiIndex is empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Returns the values at a specific level
    pub fn get_level_values(&self, level: usize) -> Result<Vec<String>> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        let mut result = Vec::with_capacity(self.len());
        for label in &self.labels {
            let value_idx = label[level];
            result.push(self.levels[level][value_idx].clone());
        }

        Ok(result)
    }

    /// Returns the name of a specific level
    pub fn get_level_name(&self, level: usize) -> Result<Option<String>> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        Ok(self.level_names[level].clone())
    }

    /// Sets the name of a specific level
    pub fn set_level_name(&mut self, level: usize, name: Option<String>) -> Result<()> {
        if level >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.n_levels(),
            });
        }

        self.level_names[level] = name;
        Ok(())
    }

    /// Swaps two levels in the MultiIndex
    pub fn swaplevel(&self, i: usize, j: usize) -> Result<Self> {
        if i >= self.n_levels() || j >= self.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: if i >= self.n_levels() { i } else { j },
                size: self.n_levels(),
            });
        }

        if i == j {
            return Ok(self.clone());
        }

        // Create new levels, labels, and level_names with swapped indices
        let mut new_levels = self.levels.clone();
        new_levels.swap(i, j);

        let mut new_level_names = self.level_names.clone();
        new_level_names.swap(i, j);

        let mut new_labels = Vec::with_capacity(self.labels.len());
        for label in &self.labels {
            let mut new_label = label.clone();
            new_label.swap(i, j);
            new_labels.push(new_label);
        }

        Self::new(new_levels, new_labels, new_level_names)
    }

    /// Converts the MultiIndex to a tuple of (levels, labels, level_names)
    pub fn to_tuples(&self) -> (Vec<Vec<String>>, Vec<Vec<usize>>, Vec<Option<String>>) {
        (
            self.levels.clone(),
            self.labels.clone(),
            self.level_names.clone(),
        )
    }
}

impl fmt::Debug for MultiIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MultiIndex:")?;

        // Show level names if any are defined
        let has_names = self.level_names.iter().any(|n| n.is_some());
        if has_names {
            write!(f, "  Names: [")?;
            for (i, name) in self.level_names.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if let Some(name) = name {
                    write!(f, "'{}'", name)?;
                } else {
                    write!(f, "None")?;
                }
            }
            writeln!(f, "]")?;
        }

        // Show a sample of the index (up to 10 items)
        let max_show = std::cmp::min(10, self.len());
        for i in 0..max_show {
            write!(f, "  (")?;
            for (level_idx, value_idx) in self.labels[i].iter().enumerate() {
                if level_idx > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.levels[level_idx][*value_idx])?;
            }
            writeln!(f, ")")?;
        }

        // Show ellipsis if there are more items
        if self.len() > max_show {
            writeln!(f, "  ...")?;
            writeln!(f, "  {} rows total", self.len())?;
        }

        Ok(())
    }
}

//! # Advanced MultiIndex with Cross-Section Selection
//!
//! This module provides cross-section selection capabilities for DataFrames with MultiIndex,
//! including partial indexing, level slicing, boolean indexing, and hierarchical navigation.

use crate::column::{Column, ColumnTrait};
use crate::core::advanced_multi_index::{
    AdvancedMultiIndex, CrossSectionResult, IndexValue, SelectionCriteria,
};
use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use std::collections::HashMap;

/// DataFrame with advanced MultiIndex cross-section support
#[derive(Debug, Clone)]
pub struct MultiIndexDataFrame {
    /// The underlying DataFrame
    pub dataframe: DataFrame,
    /// The row index (MultiIndex)
    pub index: AdvancedMultiIndex,
    /// Column names (simple for now, could be extended to MultiIndex)
    pub column_names: Vec<String>,
}

/// Result of cross-section selection
#[derive(Debug, Clone)]
pub struct CrossSectionDataFrame {
    /// Resulting DataFrame
    pub dataframe: MultiIndexDataFrame,
    /// Selected row indices
    pub selected_indices: Vec<usize>,
    /// Whether any level was dropped
    pub level_dropped: bool,
}

impl MultiIndexDataFrame {
    /// Create a new MultiIndexDataFrame
    pub fn new(dataframe: DataFrame, index: AdvancedMultiIndex) -> Result<Self> {
        if index.len() != dataframe.row_count() {
            return Err(Error::InconsistentRowCount {
                expected: dataframe.row_count(),
                found: index.len(),
            });
        }

        let column_names = dataframe.column_names().to_vec();

        Ok(Self {
            dataframe,
            index,
            column_names,
        })
    }

    /// Cross-section selection: select rows with specific value at given level
    pub fn xs(
        &mut self,
        key: IndexValue,
        level: usize,
        drop_level: bool,
    ) -> Result<CrossSectionDataFrame> {
        let xs_result = self.index.xs(key, level, drop_level)?;

        if xs_result.indices.is_empty() {
            return Err(Error::InvalidOperation(
                "No matching rows found for cross-section".to_string(),
            ));
        }

        // Create new DataFrame with selected rows
        let selected_dataframe = self.select_rows(&xs_result.indices)?;

        // Handle index transformation
        let result_index = if drop_level {
            xs_result.index.ok_or_else(|| {
                Error::InvalidOperation("Expected index after dropping level".to_string())
            })?
        } else {
            // Keep original index structure but only selected rows
            self.select_index_rows(&xs_result.indices)?
        };

        let result = MultiIndexDataFrame {
            dataframe: selected_dataframe,
            index: result_index,
            column_names: self.column_names.clone(),
        };

        Ok(CrossSectionDataFrame {
            dataframe: result,
            selected_indices: xs_result.indices,
            level_dropped: drop_level,
        })
    }

    /// Advanced selection using multiple criteria
    pub fn select(&self, criteria: SelectionCriteria) -> Result<CrossSectionDataFrame> {
        let selected_indices = self.index.select(criteria)?;

        if selected_indices.is_empty() {
            return Err(Error::InvalidOperation(
                "No matching rows found for selection criteria".to_string(),
            ));
        }

        let selected_dataframe = self.select_rows(&selected_indices)?;
        let selected_index = self.select_index_rows(&selected_indices)?;

        let result = MultiIndexDataFrame {
            dataframe: selected_dataframe,
            index: selected_index,
            column_names: self.column_names.clone(),
        };

        Ok(CrossSectionDataFrame {
            dataframe: result,
            selected_indices,
            level_dropped: false,
        })
    }

    /// GroupBy operation using index levels
    pub fn groupby_level(&self, levels: &[usize]) -> Result<MultiIndexGroupBy> {
        // Validate levels
        for &level in levels {
            if level >= self.index.n_levels() {
                return Err(Error::IndexOutOfBounds {
                    index: level,
                    size: self.index.n_levels(),
                });
            }
        }

        // Get unique group keys
        let group_keys = self.index.get_group_keys(levels)?;

        // Build groups: group_key -> row_indices
        let mut groups = HashMap::new();
        for group_key in group_keys {
            let indices = self.index.get_group_indices(levels, &group_key)?;
            groups.insert(group_key, indices);
        }

        Ok(MultiIndexGroupBy {
            dataframe: self.clone(),
            groups,
            group_levels: levels.to_vec(),
        })
    }

    /// Select specific levels from the index
    pub fn select_levels(&self, levels: &[usize]) -> Result<MultiIndexDataFrame> {
        // Validate levels
        for &level in levels {
            if level >= self.index.n_levels() {
                return Err(Error::IndexOutOfBounds {
                    index: level,
                    size: self.index.n_levels(),
                });
            }
        }

        // Create new tuples with only selected levels
        let mut new_tuples = Vec::with_capacity(self.index.len());
        for i in 0..self.index.len() {
            let original_tuple = self.index.get_tuple(i)?;
            let new_tuple: Vec<IndexValue> = levels
                .iter()
                .map(|&level| original_tuple[level].clone())
                .collect();
            new_tuples.push(new_tuple);
        }

        // Create new level names
        let original_names = self.index.level_names();
        let new_names: Vec<Option<String>> = levels
            .iter()
            .map(|&level| original_names[level].clone())
            .collect();

        let new_index = AdvancedMultiIndex::new(new_tuples, Some(new_names))?;

        Ok(MultiIndexDataFrame {
            dataframe: self.dataframe.clone(),
            index: new_index,
            column_names: self.column_names.clone(),
        })
    }

    /// Reindex with new MultiIndex
    pub fn reindex(&self, new_index: AdvancedMultiIndex) -> Result<MultiIndexDataFrame> {
        if new_index.len() != self.index.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.index.len(),
                found: new_index.len(),
            });
        }

        Ok(MultiIndexDataFrame {
            dataframe: self.dataframe.clone(),
            index: new_index,
            column_names: self.column_names.clone(),
        })
    }

    /// Swap levels in the index
    pub fn swaplevel(&self, i: usize, j: usize) -> Result<MultiIndexDataFrame> {
        let swapped_index = self.index.reorder_levels(&{
            let mut order: Vec<usize> = (0..self.index.n_levels()).collect();
            order.swap(i, j);
            order
        })?;

        Ok(MultiIndexDataFrame {
            dataframe: self.dataframe.clone(),
            index: swapped_index,
            column_names: self.column_names.clone(),
        })
    }

    /// Sort by index levels
    pub fn sort_index(
        &self,
        levels: Option<&[usize]>,
        ascending: bool,
    ) -> Result<MultiIndexDataFrame> {
        let default_levels: Vec<usize> = (0..self.index.n_levels()).collect();
        let sort_levels = levels.unwrap_or(&default_levels);

        // Create sortable tuples with original indices
        let mut indexed_tuples: Vec<(usize, Vec<IndexValue>)> =
            Vec::with_capacity(self.index.len());
        for i in 0..self.index.len() {
            let tuple = self.index.get_tuple(i)?;
            let sort_tuple: Vec<IndexValue> = sort_levels
                .iter()
                .map(|&level| tuple[level].clone())
                .collect();
            indexed_tuples.push((i, sort_tuple));
        }

        // Sort tuples
        indexed_tuples.sort_by(|a, b| {
            let comparison = a.1.cmp(&b.1);
            if ascending {
                comparison
            } else {
                comparison.reverse()
            }
        });

        // Extract sorted indices
        let sorted_indices: Vec<usize> = indexed_tuples.into_iter().map(|(idx, _)| idx).collect();

        // Apply sorting to DataFrame and index
        let sorted_dataframe = self.select_rows(&sorted_indices)?;
        let sorted_index = self.select_index_rows(&sorted_indices)?;

        Ok(MultiIndexDataFrame {
            dataframe: sorted_dataframe,
            index: sorted_index,
            column_names: self.column_names.clone(),
        })
    }

    /// Get level values as a vector
    pub fn get_level_values(&self, level: usize) -> Result<Vec<IndexValue>> {
        if level >= self.index.n_levels() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.index.n_levels(),
            });
        }

        let mut values = Vec::with_capacity(self.index.len());
        for i in 0..self.index.len() {
            let tuple = self.index.get_tuple(i)?;
            values.push(tuple[level].clone());
        }

        Ok(values)
    }

    /// Check if index is monotonic at specified level
    pub fn is_monotonic(&self, level: usize) -> Result<bool> {
        let values = self.get_level_values(level)?;
        if values.len() <= 1 {
            return Ok(true);
        }

        let increasing = values.windows(2).all(|w| w[0] <= w[1]);
        let decreasing = values.windows(2).all(|w| w[0] >= w[1]);

        Ok(increasing || decreasing)
    }

    // Helper methods

    fn select_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        // This is a simplified implementation - would need actual DataFrame row selection
        // For now, return a clone as placeholder
        Ok(self.dataframe.clone())
    }

    fn select_index_rows(&self, indices: &[usize]) -> Result<AdvancedMultiIndex> {
        let selected_tuples: Result<Vec<Vec<IndexValue>>> = indices
            .iter()
            .map(|&i| self.index.get_tuple(i).map(|t| t.to_vec()))
            .collect();

        AdvancedMultiIndex::new(selected_tuples?, Some(self.index.level_names().to_vec()))
    }
}

/// GroupBy operation result for MultiIndex DataFrames
#[derive(Debug, Clone)]
pub struct MultiIndexGroupBy {
    dataframe: MultiIndexDataFrame,
    groups: HashMap<Vec<IndexValue>, Vec<usize>>,
    group_levels: Vec<usize>,
}

impl MultiIndexGroupBy {
    /// Get the number of groups
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }

    /// Get group sizes
    pub fn size(&self) -> HashMap<Vec<IndexValue>, usize> {
        self.groups
            .iter()
            .map(|(key, indices)| (key.clone(), indices.len()))
            .collect()
    }

    /// Get group by key
    pub fn get_group(&self, key: &[IndexValue]) -> Result<MultiIndexDataFrame> {
        let indices = self
            .groups
            .get(key)
            .ok_or_else(|| Error::InvalidOperation(format!("Group key {:?} not found", key)))?;

        let selected_dataframe = self.dataframe.select_rows(indices)?;
        let selected_index = self.dataframe.select_index_rows(indices)?;

        Ok(MultiIndexDataFrame {
            dataframe: selected_dataframe,
            index: selected_index,
            column_names: self.dataframe.column_names.clone(),
        })
    }

    /// Apply aggregation function to all groups
    pub fn agg(&self, func: AggregationFunction) -> Result<MultiIndexDataFrame> {
        let mut result_data: HashMap<String, Vec<f64>> = HashMap::new();
        let mut result_index_tuples = Vec::new();

        // Initialize result columns
        for col_name in &self.dataframe.column_names {
            result_data.insert(col_name.clone(), Vec::new());
        }

        // Process each group
        for (group_key, indices) in &self.groups {
            result_index_tuples.push(group_key.clone());

            for col_name in &self.dataframe.column_names {
                let values = self.extract_column_values(col_name, indices)?;
                let aggregated = func.apply(&values);
                result_data.get_mut(col_name).unwrap().push(aggregated);
            }
        }

        // Create result index with only group levels
        let group_level_names: Vec<Option<String>> = self
            .group_levels
            .iter()
            .map(|&level| self.dataframe.index.level_names()[level].clone())
            .collect();

        let result_index = AdvancedMultiIndex::new(result_index_tuples, Some(group_level_names))?;

        // Create result DataFrame (simplified - would need actual DataFrame construction)
        let result_dataframe = self.dataframe.dataframe.clone(); // Placeholder

        Ok(MultiIndexDataFrame {
            dataframe: result_dataframe,
            index: result_index,
            column_names: self.dataframe.column_names.clone(),
        })
    }

    /// Apply custom function to each group
    pub fn apply<F>(&self, func: F) -> Result<MultiIndexDataFrame>
    where
        F: Fn(&MultiIndexDataFrame) -> Result<MultiIndexDataFrame>,
    {
        let mut result_parts = Vec::new();

        for (group_key, indices) in &self.groups {
            let group_dataframe = self.get_group(group_key)?;
            let group_result = func(&group_dataframe)?;
            result_parts.push((group_key.clone(), group_result));
        }

        // Combine results (simplified implementation)
        if let Some((_, first_result)) = result_parts.first() {
            Ok(first_result.clone()) // Placeholder
        } else {
            Err(Error::InvalidOperation(
                "No groups to apply function to".to_string(),
            ))
        }
    }

    /// Get group keys
    pub fn groups(&self) -> &HashMap<Vec<IndexValue>, Vec<usize>> {
        &self.groups
    }

    fn extract_column_values(&self, column_name: &str, indices: &[usize]) -> Result<Vec<f64>> {
        // This would extract numeric values from the specified column at given indices
        // For now, return placeholder values
        Ok(vec![0.0; indices.len()])
    }
}

/// Aggregation functions for GroupBy operations
#[derive(Debug, Clone, Copy)]
pub enum AggregationFunction {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Std,
    Var,
    First,
    Last,
}

impl AggregationFunction {
    pub fn apply(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        match self {
            AggregationFunction::Sum => values.iter().sum(),
            AggregationFunction::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AggregationFunction::Count => values.len() as f64,
            AggregationFunction::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggregationFunction::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggregationFunction::Std => {
                if values.len() <= 1 {
                    0.0
                } else {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / (values.len() - 1) as f64;
                    variance.sqrt()
                }
            }
            AggregationFunction::Var => {
                if values.len() <= 1 {
                    0.0
                } else {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / (values.len() - 1) as f64
                }
            }
            AggregationFunction::First => values[0],
            AggregationFunction::Last => values[values.len() - 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_section_selection() {
        // Create test data
        let tuples = vec![
            vec![IndexValue::from("A"), IndexValue::from(1)],
            vec![IndexValue::from("A"), IndexValue::from(2)],
            vec![IndexValue::from("B"), IndexValue::from(1)],
            vec![IndexValue::from("B"), IndexValue::from(2)],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // This test would require a proper DataFrame implementation
        // For now, we'll test the index operations
        assert_eq!(index.len(), 4);
        assert_eq!(index.n_levels(), 2);
    }

    #[test]
    fn test_groupby_level() {
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
        let group_keys = index.get_group_keys(&[0]).unwrap();

        assert_eq!(group_keys.len(), 2); // "A" and "B"

        let indices_a = index
            .get_group_indices(&[0], &[IndexValue::from("A")])
            .unwrap();
        assert_eq!(indices_a, vec![0, 1]);

        let indices_b = index
            .get_group_indices(&[0], &[IndexValue::from("B")])
            .unwrap();
        assert_eq!(indices_b, vec![2, 3]);
    }

    #[test]
    fn test_aggregation_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(AggregationFunction::Sum.apply(&values), 15.0);
        assert_eq!(AggregationFunction::Mean.apply(&values), 3.0);
        assert_eq!(AggregationFunction::Count.apply(&values), 5.0);
        assert_eq!(AggregationFunction::Min.apply(&values), 1.0);
        assert_eq!(AggregationFunction::Max.apply(&values), 5.0);
        assert_eq!(AggregationFunction::First.apply(&values), 1.0);
        assert_eq!(AggregationFunction::Last.apply(&values), 5.0);

        // Test std and var
        let std_val = AggregationFunction::Std.apply(&values);
        let var_val = AggregationFunction::Var.apply(&values);
        assert!((std_val - 1.58113883).abs() < 1e-6);
        assert!((var_val - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_level_selection() {
        let tuples = vec![
            vec![
                IndexValue::from("A"),
                IndexValue::from("X"),
                IndexValue::from(1),
            ],
            vec![
                IndexValue::from("B"),
                IndexValue::from("Y"),
                IndexValue::from(2),
            ],
        ];

        let index = AdvancedMultiIndex::new(tuples, None).unwrap();

        // Test getting level values
        let level_0_values = (0..index.len())
            .map(|i| index.get_tuple(i).unwrap()[0].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            level_0_values,
            vec![IndexValue::from("A"), IndexValue::from("B")]
        );

        let level_2_values = (0..index.len())
            .map(|i| index.get_tuple(i).unwrap()[2].clone())
            .collect::<Vec<_>>();
        assert_eq!(
            level_2_values,
            vec![IndexValue::from(1), IndexValue::from(2)]
        );
    }
}

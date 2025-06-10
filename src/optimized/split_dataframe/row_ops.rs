//! Row operations functionality for OptimizedDataFrame

use rayon::prelude::*;
use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use super::data_ops; // Reference to data operations module
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, IndexTrait};

impl OptimizedDataFrame {
    /// Filter rows (as a new DataFrame)
    ///
    /// Extracts only rows where the value in the condition column (boolean type) is true.
    ///
    /// # Arguments
    /// * `condition_column` - Name of the boolean column to use as filter condition
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `filter_rows`.
    pub fn filter_rows(&self, condition_column: &str) -> Result<Self> {
        // Get condition column
        let column_idx = self
            .column_indices
            .get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;

        let condition = &self.columns[*column_idx];

        // Verify that the condition column is boolean type
        if let Column::Boolean(bool_col) = condition {
            // Collect indices of rows where the value is true
            let mut indices = Vec::new();
            for i in 0..bool_col.len() {
                if let Ok(Some(true)) = bool_col.get(i) {
                    indices.push(i);
                }
            }

            // Create a new DataFrame
            let mut result = Self::new();

            // Filter each column
            for (i, name) in self.column_names.iter().enumerate() {
                let column = &self.columns[i];

                let filtered_column = match column {
                    Column::Int64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0); // Default value
                            }
                        }
                        Column::Int64(Int64Column::new(filtered_data))
                    }
                    Column::Float64(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(0.0); // Default value
                            }
                        }
                        Column::Float64(Float64Column::new(filtered_data))
                    }
                    Column::String(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val.to_string());
                            } else {
                                filtered_data.push(String::new()); // Default value
                            }
                        }
                        Column::String(StringColumn::new(filtered_data))
                    }
                    Column::Boolean(col) => {
                        let mut filtered_data = Vec::with_capacity(indices.len());
                        for &idx in &indices {
                            if let Ok(Some(val)) = col.get(idx) {
                                filtered_data.push(val);
                            } else {
                                filtered_data.push(false); // Default value
                            }
                        }
                        Column::Boolean(BooleanColumn::new(filtered_data))
                    }
                };

                result.add_column(name.clone(), filtered_column)?;
            }

            // Process the index
            if let Some(ref idx) = self.index {
                if let DataFrameIndex::Simple(ref simple_idx) = idx {
                    let mut new_index_values = Vec::with_capacity(indices.len());

                    for &old_idx in &indices {
                        if old_idx < simple_idx.len() {
                            let value = simple_idx
                                .get_value(old_idx)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| old_idx.to_string());
                            new_index_values.push(value);
                        } else {
                            new_index_values.push(old_idx.to_string());
                        }
                    }

                    let new_index = crate::index::Index::new(new_index_values)?;
                    result.set_index_from_simple_index(new_index)?;
                }
            }

            Ok(result)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: crate::column::ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }

    /// Filter by specified row indices
    ///
    /// # Arguments
    /// * `indices` - Array of row indices to extract
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `filter_rows_by_indices`.
    pub fn filter_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        // Use parallel processing
        use rayon::prelude::*;

        let mut result = Self::new();

        // Filter each column in parallel
        let column_results: Result<Vec<(String, Column)>> = self
            .column_names
            .par_iter()
            .map(|name| {
                let i = self.column_indices[name];
                let column = &self.columns[i];

                let filtered_column = match column {
                    Column::Int64(col) => {
                        let filtered_data: Vec<i64> = indices
                            .iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(0) // Default value
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Int64(Int64Column::new(filtered_data))
                    }
                    Column::Float64(col) => {
                        let filtered_data: Vec<f64> = indices
                            .iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(0.0) // Default value
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Float64(Float64Column::new(filtered_data))
                    }
                    Column::String(col) => {
                        let filtered_data: Vec<String> = indices
                            .iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val.to_string())
                                    } else {
                                        Some(String::new()) // Default value
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::String(StringColumn::new(filtered_data))
                    }
                    Column::Boolean(col) => {
                        let filtered_data: Vec<bool> = indices
                            .iter()
                            .filter_map(|&idx| {
                                if idx < col.len() {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        Some(val)
                                    } else {
                                        Some(false) // Default value
                                    }
                                } else {
                                    None
                                }
                            })
                            .collect();
                        Column::Boolean(BooleanColumn::new(filtered_data))
                    }
                };

                Ok((name.clone(), filtered_column))
            })
            .collect();

        // Process results
        let columns = column_results?;
        for (name, column) in columns {
            result.add_column(name, column)?;
        }

        // Process the index
        if let Some(ref idx) = self.index {
            if let DataFrameIndex::Simple(ref simple_idx) = idx {
                let valid_indices: Vec<usize> = indices
                    .iter()
                    .filter(|&&i| i < self.row_count)
                    .cloned()
                    .collect();

                if !valid_indices.is_empty() {
                    let mut new_index_values = Vec::with_capacity(valid_indices.len());

                    for &old_idx in &valid_indices {
                        if old_idx < simple_idx.len() {
                            let value = simple_idx
                                .get_value(old_idx)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| old_idx.to_string());
                            new_index_values.push(value);
                        } else {
                            new_index_values.push(old_idx.to_string());
                        }
                    }

                    let new_index = crate::index::Index::new(new_index_values)?;
                    result.set_index_from_simple_index(new_index)?;
                }
            }
        }

        Ok(result)
    }

    /// Get the first n rows
    ///
    /// # Arguments
    /// * `n` - Number of rows to retrieve
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with the first n rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `head_rows`.
    pub fn head_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let indices: Vec<usize> = (0..n).collect();
        self.filter_rows_by_indices(&indices)
    }

    /// Get the last n rows
    ///
    /// # Arguments
    /// * `n` - Number of rows to retrieve
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with the last n rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `tail_rows`.
    pub fn tail_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let start = self.row_count.saturating_sub(n);
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.filter_rows_by_indices(&indices)
    }

    /// Sample rows from the DataFrame
    ///
    /// # Arguments
    /// * `n` - Number of rows to sample
    /// * `replace` - Whether to sample with replacement
    /// * `seed` - Random seed value (for reproducibility)
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with sampled rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `sample_rows`.
    pub fn sample_rows(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        use rand::rngs::StdRng;
        use rand::{seq::SliceRandom, Rng, RngCore, SeedableRng};

        if self.row_count == 0 {
            return Ok(Self::new());
        }

        let row_indices: Vec<usize> = (0..self.row_count).collect();

        // Initialize random number generator
        let mut rng = if let Some(seed_val) = seed {
            StdRng::seed_from_u64(seed_val)
        } else {
            // API changed due to dependency updates, so using a method to generate seed
            let mut seed_bytes = [0u8; 32];
            rand::rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        };

        // Sample row indices
        let sampled_indices = if replace {
            // Sampling with replacement
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.random_range(0..self.row_count);
                samples.push(idx);
            }
            samples
        } else {
            // Sampling without replacement
            let sample_size = std::cmp::min(n, self.row_count);
            let mut indices_copy = row_indices.clone();
            indices_copy.shuffle(&mut rng);
            indices_copy[0..sample_size].to_vec()
        };

        self.filter_rows_by_indices(&sampled_indices)
    }
}

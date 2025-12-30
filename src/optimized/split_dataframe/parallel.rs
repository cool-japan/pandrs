//! Parallel processing functionality for OptimizedDataFrame

use rayon::prelude::*;
use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Parallel row filtering
    ///
    /// Extracts only rows where the value in the condition column (Boolean type) is true,
    /// applying parallel processing for large datasets.
    ///
    /// # Arguments
    /// * `condition_column` - Name of Boolean column to use as filter condition
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        // Threshold for optimal parallelization (smaller datasets benefit from serial processing)
        const PARALLEL_THRESHOLD: usize = 100_000;

        // Get condition column
        let column_idx = self
            .column_indices
            .get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;

        let condition = &self.columns[*column_idx];

        // Verify the condition column is boolean type
        if let Column::Boolean(bool_col) = condition {
            let row_count = bool_col.len();

            // Choose serial/parallel processing based on data size
            let indices: Vec<usize> = if row_count < PARALLEL_THRESHOLD {
                // Serial processing (small data)
                (0..row_count)
                    .filter_map(|i| {
                        if let Ok(Some(true)) = bool_col.get(i) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // Parallel processing (large data)
                // Optimize chunk size to reduce parallelization overhead
                let chunk_size = (row_count / rayon::current_num_threads()).max(1000);

                // First convert range to array, then process chunks
                (0..row_count)
                    .collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .filter_map(|&i| {
                                if let Ok(Some(true)) = bool_col.get(i) {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };

            if indices.is_empty() {
                // Return empty DataFrame
                let mut result = Self::new();
                for name in &self.column_names {
                    let col_idx = self.column_indices[name];
                    let empty_col = match &self.columns[col_idx] {
                        Column::Int64(_) => Column::Int64(Int64Column::new(Vec::new())),
                        Column::Float64(_) => Column::Float64(Float64Column::new(Vec::new())),
                        Column::String(_) => Column::String(StringColumn::new(Vec::new())),
                        Column::Boolean(_) => Column::Boolean(BooleanColumn::new(Vec::new())),
                    };
                    result.add_column(name.clone(), empty_col)?;
                }
                return Ok(result);
            }

            // Create new DataFrame
            let mut result = Self::new();

            // Pre-allocate vector for result columns
            let mut result_columns = Vec::with_capacity(self.column_names.len());

            // Choose column processing method based on data size
            if indices.len() < PARALLEL_THRESHOLD || self.column_names.len() < 4 {
                // Serial processing (small data or few columns)
                for name in &self.column_names {
                    let i = self.column_indices[name];
                    let column = &self.columns[i];

                    let filtered_column = match column {
                        Column::Int64(col) => {
                            let filtered_data: Vec<i64> = indices
                                .iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0 // Default value
                                    }
                                })
                                .collect();
                            Column::Int64(Int64Column::new(filtered_data))
                        }
                        Column::Float64(col) => {
                            let filtered_data: Vec<f64> = indices
                                .iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        0.0 // Default value
                                    }
                                })
                                .collect();
                            Column::Float64(Float64Column::new(filtered_data))
                        }
                        Column::String(col) => {
                            let filtered_data: Vec<String> = indices
                                .iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val.to_string()
                                    } else {
                                        String::new() // Default value
                                    }
                                })
                                .collect();
                            Column::String(StringColumn::new(filtered_data))
                        }
                        Column::Boolean(col) => {
                            let filtered_data: Vec<bool> = indices
                                .iter()
                                .map(|&idx| {
                                    if let Ok(Some(val)) = col.get(idx) {
                                        val
                                    } else {
                                        false // Default value
                                    }
                                })
                                .collect();
                            Column::Boolean(BooleanColumn::new(filtered_data))
                        }
                    };

                    result_columns.push((name.clone(), filtered_column));
                }
            } else {
                // Parallel processing for large data
                // Process each column in parallel (coarse-grained parallelism at column level)
                result_columns = self
                    .column_names
                    .par_iter()
                    .map(|name| {
                        let i = self.column_indices[name];
                        let column = &self.columns[i];

                        let indices_len = indices.len();
                        let filtered_column = match column {
                            Column::Int64(col) => {
                                // Split large index list for processing
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);

                                // Use chunks to ensure all elements are processed
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<i64> = chunk
                                        .iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0 // Default value
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }

                                Column::Int64(Int64Column::new(filtered_data))
                            }
                            Column::Float64(col) => {
                                // Split large index list for processing
                                let chunk_size = (indices_len / 8).max(1000);
                                let mut filtered_data = Vec::with_capacity(indices_len);

                                // Use chunks to ensure all elements are processed
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<f64> = chunk
                                        .iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val
                                            } else {
                                                0.0 // Default value
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }

                                Column::Float64(Float64Column::new(filtered_data))
                            }
                            Column::String(col) => {
                                // String processing is especially heavy, use finer chunks
                                let chunk_size = (indices_len / 16).max(500);
                                let mut filtered_data = Vec::with_capacity(indices_len);

                                // Use chunks to ensure all elements are processed
                                for chunk in indices.chunks(chunk_size) {
                                    let chunk_data: Vec<String> = chunk
                                        .iter()
                                        .map(|&idx| {
                                            if let Ok(Some(val)) = col.get(idx) {
                                                val.to_string()
                                            } else {
                                                String::new() // Default value
                                            }
                                        })
                                        .collect();
                                    filtered_data.extend(chunk_data);
                                }

                                Column::String(StringColumn::new(filtered_data))
                            }
                            Column::Boolean(col) => {
                                let filtered_data: Vec<bool> = indices
                                    .iter()
                                    .map(|&idx| {
                                        if let Ok(Some(val)) = col.get(idx) {
                                            val
                                        } else {
                                            false // Default value
                                        }
                                    })
                                    .collect();
                                Column::Boolean(BooleanColumn::new(filtered_data))
                            }
                        };

                        (name.clone(), filtered_column)
                    })
                    .collect();
            }

            // Add results to DataFrame
            for (name, column) in result_columns {
                result.add_column(name, column)?;
            }

            // Copy index
            if let Some(ref idx) = self.index {
                result.index = Some(idx.clone());
            }

            Ok(result)
        } else {
            Err(Error::OperationFailed(format!(
                "Column '{}' is not of boolean type",
                condition_column
            )))
        }
    }
}

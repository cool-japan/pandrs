use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// Sort DataFrame by column
    ///
    /// # Arguments
    /// * `by` - Name of the column to sort by
    /// * `ascending` - Whether to sort in ascending order
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame that is sorted
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // Get column index
        let column_idx = self
            .column_indices
            .get(by)
            .ok_or_else(|| Error::ColumnNotFound(by.to_string()))?;

        let column = &self.columns[*column_idx];

        // Create an array of row indices (from 0 to row_count-1)
        let mut indices: Vec<usize> = (0..self.row_count()).collect();

        // Sort based on column type
        match column.column_type() {
            ColumnType::Int64 => {
                let col = column.as_int64().unwrap();
                // Create pairs of row index and value
                let mut pairs: Vec<(usize, Option<i64>)> = indices
                    .iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();

                // Sort: NULL values are placed at the end
                pairs.sort_by(|a, b| match (&a.1, &b.1) {
                    (None, None) => Ordering::Equal,
                    (None, _) => Ordering::Greater,
                    (_, None) => Ordering::Less,
                    (Some(val_a), Some(val_b)) => {
                        if ascending {
                            val_a.cmp(val_b)
                        } else {
                            val_b.cmp(val_a)
                        }
                    }
                });

                // Extract sorted row indices
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            }
            ColumnType::Float64 => {
                let col = column.as_float64().unwrap();
                // Create pairs of row index and value
                let mut pairs: Vec<(usize, Option<f64>)> = indices
                    .iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();

                // Sort: NULL values are placed at the end
                pairs.sort_by(|a, b| match (&a.1, &b.1) {
                    (None, None) => Ordering::Equal,
                    (None, _) => Ordering::Greater,
                    (_, None) => Ordering::Less,
                    (Some(val_a), Some(val_b)) => {
                        if ascending {
                            val_a.partial_cmp(val_b).unwrap_or(Ordering::Equal)
                        } else {
                            val_b.partial_cmp(val_a).unwrap_or(Ordering::Equal)
                        }
                    }
                });

                // Extract sorted row indices
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            }
            ColumnType::String => {
                let col = column.as_string().unwrap();
                // Create pairs of row index and value
                let mut pairs: Vec<(usize, Option<String>)> = indices
                    .iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten().map(|s| s.to_string())))
                    .collect();

                // Sort: NULL values are placed at the end
                pairs.sort_by(|a, b| match (&a.1, &b.1) {
                    (None, None) => Ordering::Equal,
                    (None, _) => Ordering::Greater,
                    (_, None) => Ordering::Less,
                    (Some(val_a), Some(val_b)) => {
                        if ascending {
                            val_a.cmp(val_b)
                        } else {
                            val_b.cmp(val_a)
                        }
                    }
                });

                // Extract sorted row indices
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            }
            ColumnType::Boolean => {
                let col = column.as_boolean().unwrap();
                // Create pairs of row index and value
                let mut pairs: Vec<(usize, Option<bool>)> = indices
                    .iter()
                    .map(|&idx| (idx, col.get(idx).ok().flatten()))
                    .collect();

                // Sort: NULL values are placed at the end
                pairs.sort_by(|a, b| match (&a.1, &b.1) {
                    (None, None) => Ordering::Equal,
                    (None, _) => Ordering::Greater,
                    (_, None) => Ordering::Less,
                    (Some(val_a), Some(val_b)) => {
                        if ascending {
                            val_a.cmp(val_b)
                        } else {
                            val_b.cmp(val_a)
                        }
                    }
                });

                // Extract sorted row indices
                indices = pairs.into_iter().map(|(idx, _)| idx).collect();
            }
        }

        // Create a new DataFrame from sorted row indices
        self.select_rows_by_indices_internal(&indices)
    }

    /// Sort DataFrame by multiple columns
    ///
    /// # Arguments
    /// * `by` - Array of column names to sort by
    /// * `ascending` - Array of boolean flags indicating ascending/descending order for each column (if None, all are ascending)
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame that is sorted
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        if by.is_empty() {
            return Err(Error::EmptyColumnList);
        }

        // Verify column names exist
        for &col_name in by {
            if !self.column_indices.contains_key(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }

        // Set ascending array
        let is_ascending: Vec<bool> = match ascending {
            Some(asc) => {
                if asc.len() != by.len() {
                    return Err(Error::InconsistentArrayLengths {
                        expected: by.len(),
                        found: asc.len(),
                    });
                }
                asc.to_vec()
            }
            None => vec![true; by.len()], // Default is ascending
        };

        // Create an array of row indices (from 0 to row_count-1)
        let mut indices: Vec<usize> = (0..self.row_count()).collect();

        // Sort by multiple keys
        indices.sort_by(|&a, &b| {
            // Compare each column
            for (col_idx, (&col_name, &asc)) in by.iter().zip(is_ascending.iter()).enumerate() {
                let column_idx = self.column_indices[col_name];
                let column = &self.columns[column_idx];

                let cmp = match column.column_type() {
                    ColumnType::Int64 => {
                        let col = column.as_int64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();

                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc {
                                    v_a.cmp(&v_b)
                                } else {
                                    v_b.cmp(&v_a)
                                }
                            }
                        }
                    }
                    ColumnType::Float64 => {
                        let col = column.as_float64().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();

                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc {
                                    v_a.partial_cmp(&v_b).unwrap_or(Ordering::Equal)
                                } else {
                                    v_b.partial_cmp(&v_a).unwrap_or(Ordering::Equal)
                                }
                            }
                        }
                    }
                    ColumnType::String => {
                        let col = column.as_string().unwrap();
                        let val_a = col.get(a).ok().flatten().map(|s| s.to_string());
                        let val_b = col.get(b).ok().flatten().map(|s| s.to_string());

                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc {
                                    v_a.cmp(&v_b)
                                } else {
                                    v_b.cmp(&v_a)
                                }
                            }
                        }
                    }
                    ColumnType::Boolean => {
                        let col = column.as_boolean().unwrap();
                        let val_a = col.get(a).ok().flatten();
                        let val_b = col.get(b).ok().flatten();

                        match (val_a, val_b) {
                            (None, None) => Ordering::Equal,
                            (None, _) => Ordering::Greater,
                            (_, None) => Ordering::Less,
                            (Some(v_a), Some(v_b)) => {
                                if asc {
                                    v_a.cmp(&v_b)
                                } else {
                                    v_b.cmp(&v_a)
                                }
                            }
                        }
                    }
                };

                // Return result if not equal
                if cmp != Ordering::Equal {
                    return cmp;
                }

                // Proceed to the next column
            }

            // If all columns are equal
            Ordering::Equal
        });

        // Create a new DataFrame from sorted row indices
        self.select_rows_by_indices_internal(&indices)
    }

    /// Select rows based on row indices (using implementation from select module)
    fn select_rows_by_indices_internal(&self, indices: &[usize]) -> Result<Self> {
        // Use function implemented in select.rs
        use crate::optimized::split_dataframe::select;
        select::select_rows_by_indices_impl(self, indices)
    }
}

//! Function application capabilities for OptimizedDataFrame

use rayon::prelude::*;
use std::collections::HashMap;

use super::core::{ColumnView, OptimizedDataFrame};
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Apply a function to columns and return a new DataFrame (performance optimized version)
    ///
    /// # Arguments
    /// * `f` - Function to apply (takes a column view and returns a new column)
    /// * `columns` - Target column names (None for all columns)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processed results
    pub fn apply<F>(&self, f: F, columns: Option<&[&str]>) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Send + Sync,
    {
        let mut result = Self::new();

        // Determine target columns
        let target_columns = if let Some(cols) = columns {
            // Target only specified columns
            cols.iter()
                .map(|&name| {
                    self.column_indices
                        .get(name)
                        .ok_or_else(|| Error::ColumnNotFound(name.to_string()))
                        .map(|&idx| (name, idx))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Target all columns
            self.column_names
                .iter()
                .map(|name| {
                    let idx = self.column_indices[name];
                    (name.as_str(), idx)
                })
                .collect()
        };

        // Apply function to each column (using parallel processing for performance optimization)
        let processed_columns: Result<Vec<(String, Column)>> = target_columns
            .into_par_iter()  // Parallel iteration
            .map(|(name, idx)| {
                // Create column view
                let view = ColumnView {
                    column: self.columns[idx].clone(),
                };

                // Apply function to generate new column
                let new_column = f(&view)?;

                // Ensure new column has same row count
                if new_column.len() != self.row_count {
                    return Err(Error::LengthMismatch {
                        expected: self.row_count,
                        actual: new_column.len(),
                    });
                }

                Ok((name.to_string(), new_column))
            })
            .collect();

        // Add processed columns to DataFrame
        for (name, column) in processed_columns? {
            result.add_column(name, column)?;
        }

        // Copy untargeted columns
        if columns.is_some() {
            for (name, idx) in self
                .column_names
                .iter()
                .map(|name| (name, self.column_indices[name]))
            {
                if !result.column_indices.contains_key(name) {
                    result.add_column(name.clone(), self.columns[idx].clone_column())?;
                }
            }
        }

        // Copy index to new DataFrame
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }

        Ok(result)
    }

    /// Apply function element-wise (equivalent to applymap)
    ///
    /// # Arguments
    /// * `column_name` - Target column name
    /// * `f` - Function to apply (type-specific functions)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processed results
    pub fn applymap<F, G, H, I>(
        &self,
        column_name: &str,
        f_str: F,
        f_int: G,
        f_float: H,
        f_bool: I,
    ) -> Result<Self>
    where
        F: Fn(&str) -> String + Send + Sync,
        G: Fn(&i64) -> i64 + Send + Sync,
        H: Fn(&f64) -> f64 + Send + Sync,
        I: Fn(&bool) -> bool + Send + Sync,
    {
        // Check column existence
        let col_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*col_idx];

        // Process based on type
        let new_column = match column {
            Column::Int64(int_col) => {
                let mut new_data = Vec::with_capacity(int_col.len());

                for i in 0..int_col.len() {
                    if let Ok(Some(val)) = int_col.get(i) {
                        new_data.push(f_int(&val));
                    } else {
                        // Keep NULL values as is
                        new_data.push(0); // Default value
                    }
                }

                Column::Int64(Int64Column::new(new_data))
            }
            Column::Float64(float_col) => {
                let mut new_data = Vec::with_capacity(float_col.len());

                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        new_data.push(f_float(&val));
                    } else {
                        // Keep NULL values as is
                        new_data.push(0.0); // Default value
                    }
                }

                Column::Float64(Float64Column::new(new_data))
            }
            Column::String(str_col) => {
                let mut new_data = Vec::with_capacity(str_col.len());

                for i in 0..str_col.len() {
                    if let Ok(Some(val)) = str_col.get(i) {
                        new_data.push(f_str(val));
                    } else {
                        // Keep NULL values as is
                        new_data.push(String::new()); // Default value
                    }
                }

                Column::String(StringColumn::new(new_data))
            }
            Column::Boolean(bool_col) => {
                let mut new_data = Vec::with_capacity(bool_col.len());

                for i in 0..bool_col.len() {
                    if let Ok(Some(val)) = bool_col.get(i) {
                        new_data.push(f_bool(&val));
                    } else {
                        // Keep NULL values as is
                        new_data.push(false); // Default value
                    }
                }

                Column::Boolean(BooleanColumn::new(new_data))
            }
        };

        // Create result DataFrame
        let mut result = self.clone();

        // Replace existing column
        result.columns[*col_idx] = new_column;

        Ok(result)
    }

    /// Apply a function to columns using parallel processing
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        // Same as apply but always uses parallel processing internally
        self.apply(func, None)
    }
}

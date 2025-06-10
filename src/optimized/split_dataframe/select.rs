//! Selection functionality for OptimizedDataFrame

use std::collections::HashSet;

use crate::column::Column;
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// Select columns to create a new DataFrame
    ///
    /// # Arguments
    /// * `columns` - Array of column names to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected columns
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        let mut df = Self::new();

        // Create a set of column names (for existence check)
        let column_set: HashSet<&str> = self.column_names.iter().map(|s| s.as_str()).collect();

        // Add specified columns to the new DataFrame
        for &col_name in columns {
            if !column_set.contains(col_name) {
                // Return error if column doesn't exist
                return Err(crate::error::Error::ColumnNotFound(col_name.to_string()));
            }

            let col_idx = self.column_indices.get(col_name).unwrap();
            let column = &self.columns[*col_idx];

            df.add_column(col_name.to_string(), column.clone())?;
        }

        // Copy index
        if let Some(ref index) = self.index {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Select rows by index
    ///
    /// # Arguments
    /// * `indices` - Array of row indices to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected rows
    ///
    /// Note: A method with the same name exists in sort.rs but that one is private
    pub fn select_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut df = Self::new();

        // Process each column
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let column = &self.columns[col_idx];

            // Extract data from selected rows
            let new_column = match column {
                Column::Int64(col) => {
                    let values: Vec<i64> = indices
                        .iter()
                        .filter_map(|&idx| {
                            if idx < self.row_count {
                                col.get(idx).ok().flatten()
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Int64(crate::column::Int64Column::new(values))
                }
                Column::Float64(col) => {
                    let values: Vec<f64> = indices
                        .iter()
                        .filter_map(|&idx| {
                            if idx < self.row_count {
                                col.get(idx).ok().flatten()
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Float64(crate::column::Float64Column::new(values))
                }
                Column::String(col) => {
                    let values: Vec<String> = indices
                        .iter()
                        .filter_map(|&idx| {
                            if idx < self.row_count {
                                col.get(idx).ok().flatten().map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::String(crate::column::StringColumn::new(values))
                }
                Column::Boolean(col) => {
                    let values: Vec<bool> = indices
                        .iter()
                        .filter_map(|&idx| {
                            if idx < self.row_count {
                                col.get(idx).ok().flatten()
                            } else {
                                None
                            }
                        })
                        .collect();
                    Column::Boolean(crate::column::BooleanColumn::new(values))
                }
            };

            df.add_column(col_name.clone(), new_column)?;
        }

        // Create new index
        // NOTE: We could extract corresponding values from the existing index,
        // but for simplicity, we create a new sequential index here
        df.set_default_index()?;

        Ok(df)
    }

    /// Select both rows and columns
    ///
    /// # Arguments
    /// * `row_indices` - Array of row indices to select
    /// * `columns` - Array of column names to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected rows and columns
    pub fn select_rows_columns(&self, row_indices: &[usize], columns: &[&str]) -> Result<Self> {
        // First select columns
        let cols_selected = self.select_columns(columns)?;

        // Then select rows
        cols_selected.select_rows_by_indices(row_indices)
    }

    /// Select rows using a mask
    ///
    /// # Arguments
    /// * `mask` - Boolean vector representing selection condition (True rows are selected)
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with rows matching the condition
    pub fn select_by_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.row_count {
            return Err(crate::error::Error::Format(format!(
                "Mask length ({}) does not match DataFrame row count ({})",
                mask.len(),
                self.row_count
            )));
        }

        // Create a list of indices from the mask
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        // Execute selection by indices
        self.select_rows_by_indices(&indices)
    }
}

/// Implementation for selecting rows based on row indices (used by other modules)
pub(crate) fn select_rows_by_indices_impl(
    df: &OptimizedDataFrame,
    indices: &[usize],
) -> Result<OptimizedDataFrame> {
    // Return an empty DataFrame if there are no rows
    if indices.is_empty() {
        return Ok(OptimizedDataFrame::new());
    }

    let mut result = OptimizedDataFrame::new();

    // Process each column
    for (name, &column_idx) in &df.column_indices {
        let column = &df.columns[column_idx];

        // Get data from row indices based on column type
        let selected_col = match column {
            Column::Int64(col) => {
                let selected_data: Vec<i64> = indices
                    .iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Int64(crate::column::Int64Column::new(selected_data))
            }
            Column::Float64(col) => {
                let selected_data: Vec<f64> = indices
                    .iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Float64(crate::column::Float64Column::new(selected_data))
            }
            Column::String(col) => {
                let selected_data: Vec<String> = indices
                    .iter()
                    .map(|&idx| {
                        col.get(idx)
                            .ok()
                            .flatten()
                            .map(|s| s.to_string())
                            .unwrap_or_default()
                    })
                    .collect();
                Column::String(crate::column::StringColumn::new(selected_data))
            }
            Column::Boolean(col) => {
                let selected_data: Vec<bool> = indices
                    .iter()
                    .map(|&idx| col.get(idx).ok().flatten().unwrap_or_default())
                    .collect();
                Column::Boolean(crate::column::BooleanColumn::new(selected_data))
            }
        };

        result.add_column(name.clone(), selected_col)?;
    }

    // Get index
    if let Some(ref idx) = df.get_index() {
        // TODO: Process index selection
        match idx {
            crate::index::DataFrameIndex::Simple(simple_idx) => {
                result.set_index_from_simple_index(simple_idx.clone())?;
            }
            _ => {
                // Temporary: Set default index
                result.set_default_index()?;
            }
        }
    }

    Ok(result)
}

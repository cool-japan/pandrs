//! Data operations functionality for OptimizedDataFrame

use rayon::prelude::*;
use std::collections::HashMap;

use super::core::{ColumnView, OptimizedDataFrame};
use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait};

impl OptimizedDataFrame {
    /// Select columns (as a new DataFrame)
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();

        for &name in columns {
            let column_idx = self
                .column_indices
                .get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

            let column = self.columns[*column_idx].clone();
            result.add_column(name.to_string(), column)?;
        }

        // Copy index
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }

        Ok(result)
    }

    /// Filter rows (as a new DataFrame)
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        // Get condition column
        let column_idx = self
            .column_indices
            .get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;

        let condition = &self.columns[*column_idx];

        // Verify that the condition column is Boolean type
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

            Ok(result)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }

    /// Filter by specified row indices (internal helper)
    pub(crate) fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut result = Self::new();

        // Filter each column
        for (i, name) in self.column_names.iter().enumerate() {
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

            result.add_column(name.clone(), filtered_column)?;
        }

        // Copy index
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }

        Ok(result)
    }

    /// Get the first n rows
    pub fn head(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let indices: Vec<usize> = (0..n).collect();
        self.filter_by_indices(&indices)
    }

    /// Get the last n rows
    pub fn tail(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let start = self.row_count.saturating_sub(n);
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.filter_by_indices(&indices)
    }

    /// Convert DataFrame to "long format" (melt operation)
    ///
    /// Converts multiple columns into a single "variable" column and "value" column.
    /// This implementation prioritizes performance.
    ///
    /// # Arguments
    /// * `id_vars` - Column names to keep unchanged (identifier columns)
    /// * `value_vars` - Column names to convert (value columns). If None, all columns except id_vars
    /// * `var_name` - Name for the variable column (default: "variable")
    /// * `value_name` - Name for the value column (default: "value")
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame converted to long format
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // Set default values for arguments
        let var_name = var_name.unwrap_or("variable");
        let value_name = value_name.unwrap_or("value");

        // If value_vars is not specified, use all columns except id_vars
        let value_vars = if let Some(vars) = value_vars {
            vars.to_vec()
        } else {
            self.column_names
                .iter()
                .filter(|name| !id_vars.contains(&name.as_str()))
                .map(|s| s.as_str())
                .collect()
        };

        // Check for non-existent column names
        for col in id_vars.iter().chain(value_vars.iter()) {
            if !self.column_indices.contains_key(*col) {
                return Err(Error::ColumnNotFound((*col).to_string()));
            }
        }

        // Precompute the size of the result (performance optimization)
        let result_rows = self.row_count * value_vars.len();

        // Extract data for ID columns
        let mut id_columns = Vec::with_capacity(id_vars.len());
        for &id_col in id_vars {
            let idx = self.column_indices[id_col];
            id_columns.push((id_col, &self.columns[idx]));
        }

        // Extract data for value columns
        let mut value_columns = Vec::with_capacity(value_vars.len());
        for &val_col in &value_vars {
            let idx = self.column_indices[val_col];
            value_columns.push((val_col, &self.columns[idx]));
        }

        // Create the result DataFrame
        let mut result = Self::new();

        // Create the variable name column
        let mut var_col_data = Vec::with_capacity(result_rows);
        for &value_col_name in &value_vars {
            for _ in 0..self.row_count {
                var_col_data.push(value_col_name.to_string());
            }
        }
        result.add_column(
            var_name.to_string(),
            Column::String(StringColumn::new(var_col_data)),
        )?;

        // Replicate and add ID columns
        for &(id_col_name, col) in &id_columns {
            match col {
                Column::Int64(int_col) => {
                    // Integer column
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = int_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // Use default value for NULLs
                                repeated_data.push(0);
                            }
                        }
                    }
                    result.add_column(
                        id_col_name.to_string(),
                        Column::Int64(Int64Column::new(repeated_data)),
                    )?;
                }
                Column::Float64(float_col) => {
                    // Float column
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = float_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // Use default value for NULLs
                                repeated_data.push(0.0);
                            }
                        }
                    }
                    result.add_column(
                        id_col_name.to_string(),
                        Column::Float64(Float64Column::new(repeated_data)),
                    )?;
                }
                Column::String(str_col) => {
                    // String column
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = str_col.get(i) {
                                repeated_data.push(val.to_string());
                            } else {
                                // Use default value for NULLs
                                repeated_data.push(String::new());
                            }
                        }
                    }
                    result.add_column(
                        id_col_name.to_string(),
                        Column::String(StringColumn::new(repeated_data)),
                    )?;
                }
                Column::Boolean(bool_col) => {
                    // Boolean column
                    let mut repeated_data = Vec::with_capacity(result_rows);
                    for _ in 0..value_vars.len() {
                        for i in 0..self.row_count {
                            if let Ok(Some(val)) = bool_col.get(i) {
                                repeated_data.push(val);
                            } else {
                                // Use default value for NULLs
                                repeated_data.push(false);
                            }
                        }
                    }
                    result.add_column(
                        id_col_name.to_string(),
                        Column::Boolean(BooleanColumn::new(repeated_data)),
                    )?;
                }
            }
        }

        // Create value columns (optimized for each type)
        // To optimize, collect all values as strings first
        let mut all_values = Vec::with_capacity(result_rows);

        for (_, col) in value_columns {
            match col {
                Column::Int64(int_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = int_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                }
                Column::Float64(float_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = float_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                }
                Column::String(str_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = str_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                }
                Column::Boolean(bool_col) => {
                    for i in 0..self.row_count {
                        if let Ok(Some(val)) = bool_col.get(i) {
                            all_values.push(val.to_string());
                        } else {
                            all_values.push(String::new());
                        }
                    }
                }
            }
        }

        // Determine the appropriate type and add data
        let is_all_int = all_values
            .iter()
            .all(|s| s.parse::<i64>().is_ok() || s.is_empty());

        let is_all_float = !is_all_int
            && all_values
                .iter()
                .all(|s| s.parse::<f64>().is_ok() || s.is_empty());

        let is_all_bool = !is_all_int
            && !is_all_float
            && all_values.iter().all(|s| {
                let lower = s.to_lowercase();
                lower.is_empty()
                    || lower == "true"
                    || lower == "false"
                    || lower == "1"
                    || lower == "0"
                    || lower == "yes"
                    || lower == "no"
            });

        // Add columns based on the determined type
        if is_all_int {
            let int_values: Vec<i64> = all_values
                .iter()
                .map(|s| s.parse::<i64>().unwrap_or(0))
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Int64(Int64Column::new(int_values)),
            )?;
        } else if is_all_float {
            let float_values: Vec<f64> = all_values
                .iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Float64(Float64Column::new(float_values)),
            )?;
        } else if is_all_bool {
            let bool_values: Vec<bool> = all_values
                .iter()
                .map(|s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "1" || lower == "yes"
                })
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Boolean(BooleanColumn::new(bool_values)),
            )?;
        } else {
            // Default to string type
            result.add_column(
                value_name.to_string(),
                Column::String(StringColumn::new(all_values)),
            )?;
        }

        Ok(result)
    }

    /// Concatenate DataFrames vertically
    ///
    /// # Arguments
    /// * `other` - DataFrame to concatenate
    ///
    /// # Returns
    /// * `Result<Self>` - Concatenated DataFrame
    pub fn append(&self, other: &Self) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }

        if other.columns.is_empty() {
            return Ok(self.clone());
        }

        // Create the result DataFrame
        let mut result = Self::new();

        // Create a set of all column names
        let mut all_columns = std::collections::HashSet::new();

        for name in &self.column_names {
            all_columns.insert(name.clone());
        }

        for name in &other.column_names {
            all_columns.insert(name.clone());
        }

        // Prepare new column data
        for col_name in all_columns {
            let self_has_column = self.column_indices.contains_key(&col_name);
            let other_has_column = other.column_indices.contains_key(&col_name);

            // If both DataFrames have the column
            if self_has_column && other_has_column {
                let self_col_idx = self.column_indices[&col_name];
                let other_col_idx = other.column_indices[&col_name];

                let self_col = &self.columns[self_col_idx];
                let other_col = &other.columns[other_col_idx];

                // Concatenate columns of the same type
                if self_col.column_type() == other_col.column_type() {
                    match (self_col, other_col) {
                        (Column::Int64(self_int), Column::Int64(other_int)) => {
                            let mut combined_data =
                                Vec::with_capacity(self_int.len() + other_int.len());

                            // Add self data
                            for i in 0..self_int.len() {
                                if let Ok(Some(val)) = self_int.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // Default value
                                }
                            }

                            // Add other data
                            for i in 0..other_int.len() {
                                if let Ok(Some(val)) = other_int.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Int64(Int64Column::new(combined_data)),
                            )?;
                        }
                        (Column::Float64(self_float), Column::Float64(other_float)) => {
                            let mut combined_data =
                                Vec::with_capacity(self_float.len() + other_float.len());

                            // Add self data
                            for i in 0..self_float.len() {
                                if let Ok(Some(val)) = self_float.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // Default value
                                }
                            }

                            // Add other data
                            for i in 0..other_float.len() {
                                if let Ok(Some(val)) = other_float.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Float64(Float64Column::new(combined_data)),
                            )?;
                        }
                        (Column::String(self_str), Column::String(other_str)) => {
                            let mut combined_data =
                                Vec::with_capacity(self_str.len() + other_str.len());

                            // Add self data
                            for i in 0..self_str.len() {
                                if let Ok(Some(val)) = self_str.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // Default value
                                }
                            }

                            // Add other data
                            for i in 0..other_str.len() {
                                if let Ok(Some(val)) = other_str.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::String(StringColumn::new(combined_data)),
                            )?;
                        }
                        (Column::Boolean(self_bool), Column::Boolean(other_bool)) => {
                            let mut combined_data =
                                Vec::with_capacity(self_bool.len() + other_bool.len());

                            // Add self data
                            for i in 0..self_bool.len() {
                                if let Ok(Some(val)) = self_bool.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // Default value
                                }
                            }

                            // Add other data
                            for i in 0..other_bool.len() {
                                if let Ok(Some(val)) = other_bool.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Boolean(BooleanColumn::new(combined_data)),
                            )?;
                        }
                        _ => {
                            // Concatenate as strings if types do not match
                            let mut combined_data =
                                Vec::with_capacity(self.row_count + other.row_count);

                            // Add self data
                            for i in 0..self.row_count {
                                let value = match self_col {
                                    Column::Int64(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::Float64(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::String(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::Boolean(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                };
                                combined_data.push(value);
                            }

                            // Add other data
                            for i in 0..other.row_count {
                                let value = match other_col {
                                    Column::Int64(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::Float64(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::String(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                    Column::Boolean(col) => col
                                        .get(i)
                                        .ok()
                                        .flatten()
                                        .map(|v| v.to_string())
                                        .unwrap_or_default(),
                                };
                                combined_data.push(value);
                            }

                            result.add_column(
                                col_name,
                                Column::String(StringColumn::new(combined_data)),
                            )?;
                        }
                    }
                }
                // Concatenate as strings if types do not match
                else {
                    let mut combined_data = Vec::with_capacity(self.row_count + other.row_count);

                    // Add self data
                    for i in 0..self.row_count {
                        let value = match self_col {
                            Column::Int64(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::Float64(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::String(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::Boolean(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                        };
                        combined_data.push(value);
                    }

                    // Add other data
                    for i in 0..other.row_count {
                        let value = match other_col {
                            Column::Int64(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::Float64(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::String(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                            Column::Boolean(col) => col
                                .get(i)
                                .ok()
                                .flatten()
                                .map(|v| v.to_string())
                                .unwrap_or_default(),
                        };
                        combined_data.push(value);
                    }

                    result
                        .add_column(col_name, Column::String(StringColumn::new(combined_data)))?;
                }
            }
            // If the column exists in only one DataFrame
            else {
                let total_rows = self.row_count + other.row_count;

                if self_has_column {
                    let col_idx = self.column_indices[&col_name];
                    let column = &self.columns[col_idx];

                    match column {
                        Column::Int64(int_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Add self data
                            for i in 0..int_col.len() {
                                if let Ok(Some(val)) = int_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // Default value
                                }
                            }

                            // Fill missing data with default values
                            combined_data.resize(total_rows, 0);

                            result.add_column(
                                col_name,
                                Column::Int64(Int64Column::new(combined_data)),
                            )?;
                        }
                        Column::Float64(float_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Add self data
                            for i in 0..float_col.len() {
                                if let Ok(Some(val)) = float_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // Default value
                                }
                            }

                            // Fill missing data with default values
                            combined_data.resize(total_rows, 0.0);

                            result.add_column(
                                col_name,
                                Column::Float64(Float64Column::new(combined_data)),
                            )?;
                        }
                        Column::String(str_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Add self data
                            for i in 0..str_col.len() {
                                if let Ok(Some(val)) = str_col.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // Default value
                                }
                            }

                            // Fill missing data with default values
                            combined_data.resize(total_rows, String::new());

                            result.add_column(
                                col_name,
                                Column::String(StringColumn::new(combined_data)),
                            )?;
                        }
                        Column::Boolean(bool_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Add self data
                            for i in 0..bool_col.len() {
                                if let Ok(Some(val)) = bool_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // Default value
                                }
                            }

                            // Fill missing data with default values
                            combined_data.resize(total_rows, false);

                            result.add_column(
                                col_name,
                                Column::Boolean(BooleanColumn::new(combined_data)),
                            )?;
                        }
                    }
                } else if other_has_column {
                    let col_idx = other.column_indices[&col_name];
                    let column = &other.columns[col_idx];

                    match column {
                        Column::Int64(int_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Fill missing data with default values
                            combined_data.resize(self.row_count, 0);

                            // Add other data
                            for i in 0..int_col.len() {
                                if let Ok(Some(val)) = int_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Int64(Int64Column::new(combined_data)),
                            )?;
                        }
                        Column::Float64(float_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Fill missing data with default values
                            combined_data.resize(self.row_count, 0.0);

                            // Add other data
                            for i in 0..float_col.len() {
                                if let Ok(Some(val)) = float_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(0.0); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Float64(Float64Column::new(combined_data)),
                            )?;
                        }
                        Column::String(str_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Fill missing data with default values
                            combined_data.resize(self.row_count, String::new());

                            // Add other data
                            for i in 0..str_col.len() {
                                if let Ok(Some(val)) = str_col.get(i) {
                                    combined_data.push(val.to_string());
                                } else {
                                    combined_data.push(String::new()); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::String(StringColumn::new(combined_data)),
                            )?;
                        }
                        Column::Boolean(bool_col) => {
                            let mut combined_data = Vec::with_capacity(total_rows);

                            // Fill missing data with default values
                            combined_data.resize(self.row_count, false);

                            // Add other data
                            for i in 0..bool_col.len() {
                                if let Ok(Some(val)) = bool_col.get(i) {
                                    combined_data.push(val);
                                } else {
                                    combined_data.push(false); // Default value
                                }
                            }

                            result.add_column(
                                col_name,
                                Column::Boolean(BooleanColumn::new(combined_data)),
                            )?;
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

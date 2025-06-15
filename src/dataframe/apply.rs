use std::collections::HashMap;
use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;
// Removed temporal and window imports to break circular dependencies

/// Axis for function application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Apply function to each column
    Column = 0,
    /// Apply function to each row
    Row = 1,
}

/// Apply functionality for DataFrames (simplified to avoid compilation issues)
pub trait ApplyExt {
    /// Apply a function to each column or row
    fn apply<F>(&self, f: F, axis: Axis, result_name: Option<String>) -> Result<Series<String>>
    where
        F: Fn(&Series<String>) -> String;

    /// Apply a function to each element
    fn applymap<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str) -> String;

    /// Replace values based on a condition
    fn mask<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool;

    /// Replace values based on a condition (inverse of mask)
    fn where_func<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool;

    /// Replace values with corresponding values
    fn replace(&self, replace_map: &HashMap<String, String>) -> Result<DataFrame>;

    /// Detect duplicate rows
    fn duplicated(&self, subset: Option<&[String]>, keep: Option<&str>) -> Result<Series<bool>>;

    /// Drop duplicate rows
    fn drop_duplicates(&self, subset: Option<&[String]>, keep: Option<&str>) -> Result<DataFrame>;
}

/// Implementation of ApplyExt for DataFrame
impl ApplyExt for DataFrame {
    fn apply<F>(&self, f: F, axis: Axis, result_name: Option<String>) -> Result<Series<String>>
    where
        F: Fn(&Series<String>) -> String,
    {
        match axis {
            Axis::Column => {
                let mut results = Vec::new();

                for column_name in &self.column_names() {
                    // Get column as string series
                    let string_values = self.get_column_string_values(column_name)?;
                    let series = Series::new(string_values, Some(column_name.to_string()))?;

                    // Apply function to the series
                    let result = f(&series);
                    results.push(result);
                }

                Series::new(results, result_name)
            }
            Axis::Row => {
                let mut results = Vec::new();

                for row_idx in 0..self.row_count() {
                    let mut row_values = Vec::new();

                    // Collect all values for this row
                    for column_name in &self.column_names() {
                        let string_values = self.get_column_string_values(column_name)?;
                        if row_idx < string_values.len() {
                            row_values.push(string_values[row_idx].clone());
                        }
                    }

                    // Create series for this row and apply function
                    let row_series = Series::new(row_values, Some(format!("row_{}", row_idx)))?;
                    let result = f(&row_series);
                    results.push(result);
                }

                Series::new(results, result_name)
            }
        }
    }

    fn applymap<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str) -> String,
    {
        let mut result = DataFrame::new();

        for column_name in &self.column_names() {
            // Get column values as strings
            let string_values = self.get_column_string_values(column_name)?;

            // Apply function to each element
            let transformed_values: Vec<String> = string_values.iter().map(|val| f(val)).collect();

            // Create new series with transformed values
            let new_series = Series::new(transformed_values, Some(column_name.to_string()))?;
            result.add_column(column_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    fn mask<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result = DataFrame::new();

        for column_name in &self.column_names() {
            // Get column values as strings
            let string_values = self.get_column_string_values(column_name)?;

            // Apply mask: replace values that satisfy condition with 'other'
            let masked_values: Vec<String> = string_values
                .iter()
                .map(|val| {
                    if condition(val) {
                        other.to_string()
                    } else {
                        val.clone()
                    }
                })
                .collect();

            // Create new series with masked values
            let new_series = Series::new(masked_values, Some(column_name.to_string()))?;
            result.add_column(column_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    fn where_func<F>(&self, condition: F, other: &str) -> Result<DataFrame>
    where
        F: Fn(&str) -> bool,
    {
        let mut result = DataFrame::new();

        for column_name in &self.column_names() {
            // Get column values as strings
            let string_values = self.get_column_string_values(column_name)?;

            // Apply where: keep values that satisfy condition, replace others with 'other'
            let where_values: Vec<String> = string_values
                .iter()
                .map(|val| {
                    if condition(val) {
                        val.clone()
                    } else {
                        other.to_string()
                    }
                })
                .collect();

            // Create new series with where values
            let new_series = Series::new(where_values, Some(column_name.to_string()))?;
            result.add_column(column_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    fn replace(&self, replace_map: &HashMap<String, String>) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for column_name in &self.column_names() {
            // Get column values as strings
            let string_values = self.get_column_string_values(column_name)?;

            // Apply replacements
            let replaced_values: Vec<String> = string_values
                .iter()
                .map(|val| replace_map.get(val).cloned().unwrap_or_else(|| val.clone()))
                .collect();

            // Create new series with replaced values
            let new_series = Series::new(replaced_values, Some(column_name.to_string()))?;
            result.add_column(column_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    fn duplicated(&self, subset: Option<&[String]>, keep: Option<&str>) -> Result<Series<bool>> {
        let keep_option = keep.unwrap_or("first");
        let mut result = vec![false; self.row_count()];

        // Determine which columns to check for duplicates
        let columns_to_check = if let Some(subset_cols) = subset {
            subset_cols.to_vec()
        } else {
            self.column_names()
        };

        // Validate that all subset columns exist
        for col_name in &columns_to_check {
            if !self.contains_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }

        // Build row representations for comparison
        let mut row_data = Vec::new();
        for row_idx in 0..self.row_count() {
            let mut row_values = Vec::new();
            for col_name in &columns_to_check {
                let column_values = self.get_column_string_values(col_name)?;
                if row_idx < column_values.len() {
                    row_values.push(column_values[row_idx].clone());
                }
            }
            row_data.push(row_values);
        }

        // Find duplicates based on keep strategy
        match keep_option {
            "first" => {
                let mut seen = std::collections::HashSet::new();
                for (idx, row) in row_data.iter().enumerate() {
                    if seen.contains(row) {
                        result[idx] = true;
                    } else {
                        seen.insert(row.clone());
                    }
                }
            }
            "last" => {
                let mut seen = std::collections::HashSet::new();
                // Process in reverse to mark earlier occurrences as duplicates
                for (idx, row) in row_data.iter().enumerate().rev() {
                    if seen.contains(row) {
                        result[idx] = true;
                    } else {
                        seen.insert(row.clone());
                    }
                }
            }
            "false" => {
                // Mark all duplicates (both first and subsequent occurrences)
                let mut counts = std::collections::HashMap::new();
                for row in &row_data {
                    *counts.entry(row.clone()).or_insert(0) += 1;
                }
                for (idx, row) in row_data.iter().enumerate() {
                    if counts[row] > 1 {
                        result[idx] = true;
                    }
                }
            }
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Invalid keep option: {}. Must be 'first', 'last', or 'false'",
                    keep_option
                )));
            }
        }

        Series::new(result, Some("duplicated".to_string()))
    }

    fn drop_duplicates(&self, subset: Option<&[String]>, keep: Option<&str>) -> Result<DataFrame> {
        let keep_option = keep.unwrap_or("first");

        // Determine which columns to check for duplicates
        let columns_to_check = if let Some(subset_cols) = subset {
            subset_cols.to_vec()
        } else {
            self.column_names()
        };

        // Validate that all subset columns exist
        for col_name in &columns_to_check {
            if !self.contains_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }
        }

        // Build row representations for comparison
        let mut row_data = Vec::new();
        for row_idx in 0..self.row_count() {
            let mut row_values = Vec::new();
            for col_name in &columns_to_check {
                let column_values = self.get_column_string_values(col_name)?;
                if row_idx < column_values.len() {
                    row_values.push(column_values[row_idx].clone());
                }
            }
            row_data.push(row_values);
        }

        // Determine which rows to keep
        let mut rows_to_keep = Vec::new();
        match keep_option {
            "first" => {
                let mut seen = std::collections::HashSet::new();
                for (idx, row) in row_data.iter().enumerate() {
                    if !seen.contains(row) {
                        seen.insert(row.clone());
                        rows_to_keep.push(idx);
                    }
                }
            }
            "last" => {
                let mut seen = std::collections::HashSet::new();
                // Process in reverse to keep last occurrences
                for (idx, row) in row_data.iter().enumerate().rev() {
                    if !seen.contains(row) {
                        seen.insert(row.clone());
                        rows_to_keep.push(idx);
                    }
                }
                rows_to_keep.reverse(); // Restore original order
            }
            "false" => {
                // Keep no duplicates (remove all duplicate rows)
                let mut counts = std::collections::HashMap::new();
                for row in &row_data {
                    *counts.entry(row.clone()).or_insert(0) += 1;
                }
                for (idx, row) in row_data.iter().enumerate() {
                    if counts[row] == 1 {
                        rows_to_keep.push(idx);
                    }
                }
            }
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Invalid keep option: {}. Must be 'first', 'last', or 'false'",
                    keep_option
                )));
            }
        }

        // Create result DataFrame with selected rows
        let mut result = DataFrame::new();

        for column_name in &self.column_names() {
            let column_values = self.get_column_string_values(column_name)?;
            let filtered_values: Vec<String> = rows_to_keep
                .iter()
                .filter_map(|&row_idx| column_values.get(row_idx).cloned())
                .collect();

            let new_series = Series::new(filtered_values, Some(column_name.to_string()))?;
            result.add_column(column_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    // Window operations removed to break circular dependencies and fix compilation timeouts
}

/// Re-export Axis for backward compatibility
pub use crate::dataframe::apply::Axis as LegacyAxis;

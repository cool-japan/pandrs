// Test utilities for PandRS tests

use pandrs::error::{PandRSError, Result};
use pandrs::series::{CategoricalOrder, StringCategorical};
use pandrs::DataFrame;
use std::collections::HashMap;

/// Categorical functionality for DataFrame tests
pub trait CategoricalExt {
    /// Convert a column to a categorical data type
    fn astype_categorical(
        &self,
        column_name: &str,
        categories: Option<Vec<String>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self>
    where
        Self: Sized;

    /// Add a categorical column to the DataFrame
    fn add_categorical_column(
        &mut self,
        column_name: String,
        categorical: StringCategorical,
    ) -> Result<()>;

    /// Set the order type of a categorical column
    fn set_categorical_ordered(&mut self, column_name: &str, order: CategoricalOrder)
        -> Result<()>;

    /// Get aggregates for categorical columns
    fn get_categorical_aggregates(
        &self,
        column_names: &[&str],
        value_column: &str,
        aggregator: impl Fn(Vec<String>) -> Result<usize>,
    ) -> Result<HashMap<Vec<String>, usize>>;

    /// Reorder categories in a categorical column
    fn reorder_categories(&mut self, column_name: &str, new_categories: Vec<String>) -> Result<()>;

    /// Add categories to a categorical column
    fn add_categories(&mut self, column_name: &str, categories: Vec<String>) -> Result<()>;

    /// Remove categories from a categorical column
    fn remove_categories(&mut self, column_name: &str, categories: &[String]) -> Result<()>;
}

// Implementation wrapper for DataFrame
impl CategoricalExt for DataFrame {
    fn astype_categorical(
        &self,
        column_name: &str,
        _categories: Option<Vec<String>>,
        ordered: Option<CategoricalOrder>,
    ) -> Result<Self> {
        // Forward to the DataFrame implementation
        let mut result = self.clone();

        // Setting as categorical in the test context
        let order_bool = match ordered {
            Some(CategoricalOrder::Ordered) => true,
            _ => false,
        };

        // Get column data and convert
        let values = result.get_column_string_values(column_name)?;
        let cat = StringCategorical::new(values, None, order_bool)?;

        // Create a new DataFrame with all columns except the one we're replacing,
        // then add the categorical column back
        let mut new_df = DataFrame::new();

        // Copy all columns except the one we're converting
        for col in result.column_names() {
            if col != column_name {
                let series = result.get_column_string_values(&col)?;
                new_df.add_column(
                    col.to_string(),
                    pandrs::series::Series::new(series, Some(col.to_string()))?,
                )?;
            }
        }

        // Add the new categorical column
        let series = cat.to_series(Some(column_name.to_string()))?;
        new_df.add_column(column_name.to_string(), series)?;

        // Replace result with new DataFrame
        result = new_df;

        Ok(result)
    }

    fn add_categorical_column(
        &mut self,
        column_name: String,
        categorical: StringCategorical,
    ) -> Result<()> {
        // Convert to series and add
        let series = categorical.to_series(Some(column_name.clone()))?;
        self.add_column(column_name, series)
    }

    fn set_categorical_ordered(
        &mut self,
        column_name: &str,
        order: CategoricalOrder,
    ) -> Result<()> {
        // Get the column data
        if !self.contains_column(column_name) {
            return Err(pandrs::error::PandRSError::ColumnNotFound(
                column_name.to_string(),
            ));
        }

        // Get current data and convert to categorical with new order
        let values = self.get_column_string_values(column_name)?;
        let order_bool = match order {
            CategoricalOrder::Ordered => true,
            CategoricalOrder::Unordered => false,
        };

        let cat = StringCategorical::new(values, None, order_bool)?;

        // Create a new DataFrame with all columns except the one we're replacing,
        // then add the categorical column back
        let mut new_df = DataFrame::new();

        // Copy all columns except the one we're converting
        for col in self.column_names() {
            if col != column_name {
                let series = self.get_column_string_values(&col)?;
                new_df.add_column(
                    col.to_string(),
                    pandrs::series::Series::new(series, Some(col.to_string()))?,
                )?;
            }
        }

        // Add the new categorical column
        let series = cat.to_series(Some(column_name.to_string()))?;
        new_df.add_column(column_name.to_string(), series)?;

        // Replace columns with new DataFrame's columns
        *self = new_df;

        Ok(())
    }

    fn get_categorical_aggregates(
        &self,
        column_names: &[&str],
        value_column: &str,
        aggregator: impl Fn(Vec<String>) -> Result<usize>,
    ) -> Result<HashMap<Vec<String>, usize>> {
        // Check if all columns exist
        for &col_name in column_names {
            if !self.contains_column(col_name) {
                return Err(pandrs::error::PandRSError::ColumnNotFound(
                    col_name.to_string(),
                ));
            }
        }

        if !self.contains_column(value_column) {
            return Err(pandrs::error::PandRSError::ColumnNotFound(
                value_column.to_string(),
            ));
        }

        // Group data by categorical columns
        let row_count = self.row_count();
        let mut grouped_data: HashMap<Vec<String>, Vec<String>> = HashMap::new();

        // For each row, create a key from categorical columns
        for row_idx in 0..row_count {
            // Create key from categorical column values
            let mut key = Vec::new();
            for &col_name in column_names {
                let values = self.get_column_string_values(col_name)?;
                if row_idx < values.len() {
                    key.push(values[row_idx].clone());
                }
            }

            // Get value column data
            let values = self.get_column_string_values(value_column)?;
            if row_idx < values.len() {
                grouped_data
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(values[row_idx].clone());
            }
        }

        // Apply aggregation function to each group
        let mut result = HashMap::new();
        for (key, values) in grouped_data {
            let agg_value = aggregator(values)?;
            result.insert(key, agg_value);
        }

        Ok(result)
    }

    fn reorder_categories(&mut self, column_name: &str, new_categories: Vec<String>) -> Result<()> {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(PandRSError::ColumnNotFound(column_name.to_string()));
        }

        // Get column data
        let values = self.get_column_string_values(column_name)?;

        // Validate new categories
        let current_values: std::collections::HashSet<String> = values.iter().cloned().collect();
        let new_cats_set: std::collections::HashSet<String> =
            new_categories.iter().cloned().collect();

        // Check if all existing values are in the new categories
        for value in &current_values {
            if !new_cats_set.contains(value) {
                return Err(PandRSError::InvalidValue(format!(
                    "Value '{}' exists in column but not in new categories",
                    value
                )));
            }
        }

        // Create a new categorical with the updated order
        let cat = StringCategorical::new(values, Some(new_categories), true)?;

        // Replace the column
        let series = cat.to_series(Some(column_name.to_string()))?;

        // Create a new DataFrame with all columns except the one we're replacing
        let mut new_df = DataFrame::new();

        // Copy all columns except the one we're replacing
        for col in self.column_names() {
            if col != column_name {
                let col_values = self.get_column_string_values(&col)?;
                new_df.add_column(
                    col.to_string(),
                    pandrs::series::Series::new(col_values, Some(col.to_string()))?,
                )?;
            }
        }

        // Add the updated column
        new_df.add_column(column_name.to_string(), series)?;

        // Replace the DataFrame
        *self = new_df;

        Ok(())
    }

    fn add_categories(&mut self, column_name: &str, categories: Vec<String>) -> Result<()> {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(PandRSError::ColumnNotFound(column_name.to_string()));
        }

        // Get column data
        let values = self.get_column_string_values(column_name)?;

        // Get current categories (unique values)
        let mut current_categories: Vec<String> = values.iter().cloned().collect();
        current_categories.sort();
        current_categories.dedup();

        // Merge current categories with new ones
        let mut all_categories = current_categories;
        for cat in categories {
            if !all_categories.contains(&cat) {
                all_categories.push(cat);
            }
        }

        // Create a new categorical with the updated categories
        let cat = StringCategorical::new(values, Some(all_categories), true)?;

        // Replace the column
        let series = cat.to_series(Some(column_name.to_string()))?;

        // Create a new DataFrame with all columns except the one we're replacing
        let mut new_df = DataFrame::new();

        // Copy all columns except the one we're replacing
        for col in self.column_names() {
            if col != column_name {
                let col_values = self.get_column_string_values(&col)?;
                new_df.add_column(
                    col.to_string(),
                    pandrs::series::Series::new(col_values, Some(col.to_string()))?,
                )?;
            }
        }

        // Add the updated column
        new_df.add_column(column_name.to_string(), series)?;

        // Replace the DataFrame
        *self = new_df;

        Ok(())
    }

    fn remove_categories(&mut self, column_name: &str, categories: &[String]) -> Result<()> {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(PandRSError::ColumnNotFound(column_name.to_string()));
        }

        // Get column data
        let values = self.get_column_string_values(column_name)?;

        // Get current categories (unique values)
        let mut current_categories: Vec<String> = values.iter().cloned().collect();
        current_categories.sort();
        current_categories.dedup();

        // Remove specified categories
        let remove_set: std::collections::HashSet<&String> = categories.iter().collect();
        let filtered_categories: Vec<String> = current_categories
            .into_iter()
            .filter(|cat| !remove_set.contains(cat))
            .collect();

        // Check if any of the removed categories have data
        for value in &values {
            if remove_set.contains(&value) {
                return Err(PandRSError::InvalidValue(format!(
                    "Cannot remove category '{}' as it has data",
                    value
                )));
            }
        }

        // Create a new categorical with the updated categories
        let cat = StringCategorical::new(values, Some(filtered_categories), true)?;

        // Replace the column
        let series = cat.to_series(Some(column_name.to_string()))?;

        // Create a new DataFrame with all columns except the one we're replacing
        let mut new_df = DataFrame::new();

        // Copy all columns except the one we're replacing
        for col in self.column_names() {
            if col != column_name {
                let col_values = self.get_column_string_values(&col)?;
                new_df.add_column(
                    col.to_string(),
                    pandrs::series::Series::new(col_values, Some(col.to_string()))?,
                )?;
            }
        }

        // Add the updated column
        new_df.add_column(column_name.to_string(), series)?;

        // Replace the DataFrame
        *self = new_df;

        Ok(())
    }
}

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::dataframe::DataFrame;
use crate::error::{PandRSError, Result};
use crate::series::Series;

/// Structure representing grouped results
#[derive(Debug)]
pub struct GroupBy<'a, K, T>
where
    K: Debug + Eq + Hash + Clone,
    T: Debug + Clone,
{
    // Use underscore to suppress warnings for unused fields
    #[allow(dead_code)]
    /// Group keys
    keys: Vec<K>,

    /// Grouped values
    groups: HashMap<K, Vec<usize>>,

    /// Original series
    source: &'a Series<T>,

    // Use underscore to suppress warnings for unused fields
    #[allow(dead_code)]
    /// Group name
    name: Option<String>,
}

impl<'a, K, T> GroupBy<'a, K, T>
where
    K: Debug + Eq + Hash + Clone,
    T: Debug + Clone,
{
    /// Create a new group
    pub fn new(keys: Vec<K>, source: &'a Series<T>, name: Option<String>) -> Result<Self> {
        // Check if the length of keys matches the source
        if keys.len() != source.len() {
            return Err(PandRSError::Consistency(format!(
                "Length of keys ({}) and source ({}) do not match",
                keys.len(),
                source.len()
            )));
        }

        // Create groups
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone()).or_insert_with(Vec::new).push(i);
        }

        Ok(GroupBy {
            keys,
            groups,
            source,
            name,
        })
    }

    /// Get the number of groups
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Return the size of each group
    pub fn size(&self) -> HashMap<K, usize> {
        self.groups
            .iter()
            .map(|(k, indices)| (k.clone(), indices.len()))
            .collect()
    }

    /// Calculate the sum for each group
    pub fn sum(&self) -> Result<HashMap<K, T>>
    where
        T: Copy + std::iter::Sum,
    {
        let mut results = HashMap::new();

        for (key, indices) in &self.groups {
            let values: Vec<T> = indices
                .iter()
                .filter_map(|&i| self.source.get(i).cloned())
                .collect();

            if !values.is_empty() {
                results.insert(key.clone(), values.into_iter().sum());
            }
        }

        Ok(results)
    }

    /// Calculate the mean for each group
    pub fn mean(&self) -> Result<HashMap<K, f64>>
    where
        T: Copy + Into<f64>,
    {
        let mut results = HashMap::new();

        for (key, indices) in &self.groups {
            let values: Vec<f64> = indices
                .iter()
                .filter_map(|&i| self.source.get(i).map(|&v| v.into()))
                .collect();

            if !values.is_empty() {
                let sum: f64 = values.iter().sum();
                let mean = sum / values.len() as f64;
                results.insert(key.clone(), mean);
            }
        }

        Ok(results)
    }
}

/// DataFrame grouping functionality
pub struct DataFrameGroupBy<'a, K>
where
    K: Debug + Eq + Hash + Clone,
{
    // Use underscore to suppress warnings for unused fields
    #[allow(dead_code)]
    /// Group keys
    keys: Vec<K>,

    /// Grouped row indices
    groups: HashMap<K, Vec<usize>>,

    // Use underscore to suppress warnings for unused fields
    #[allow(dead_code)]
    /// Original DataFrame
    source: &'a DataFrame,

    // Use underscore to suppress warnings for unused fields
    #[allow(dead_code)]
    /// Column name used for grouping
    by: String,
}

impl<'a, K> DataFrameGroupBy<'a, K>
where
    K: Debug + Eq + Hash + Clone,
{
    /// Create a new DataFrame group
    pub fn new(keys: Vec<K>, source: &'a DataFrame, by: String) -> Result<Self> {
        // Check if the length of keys matches the row count
        if keys.len() != source.row_count() {
            return Err(PandRSError::Consistency(format!(
                "Length of keys ({}) and DataFrame row count ({}) do not match",
                keys.len(),
                source.row_count()
            )));
        }

        // Create groups
        let mut groups = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            groups.entry(key.clone()).or_insert_with(Vec::new).push(i);
        }

        Ok(DataFrameGroupBy {
            keys,
            groups,
            source,
            by,
        })
    }

    /// Get the number of groups
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Return the size of each group
    pub fn size(&self) -> HashMap<K, usize> {
        self.groups
            .iter()
            .map(|(k, indices)| (k.clone(), indices.len()))
            .collect()
    }

    /// Get the size of each group as a DataFrame
    pub fn size_as_df(&self) -> Result<DataFrame> {
        // Create a DataFrame for results
        let mut result = DataFrame::new();

        // Create columns for group keys and values
        let mut keys = Vec::new();
        let mut sizes = Vec::new();

        for (key, indices) in &self.groups {
            keys.push(format!("{:?}", key)); // Convert key to string
            sizes.push(indices.len().to_string()); // Convert size to string
        }

        // Add group key column
        let key_column = Series::new(keys, Some("group_key".to_string()))?;
        result.add_column("group_key".to_string(), key_column)?;

        // Add size column
        let size_column = Series::new(sizes, Some("size".to_string()))?;
        result.add_column("size".to_string(), size_column)?;

        Ok(result)
    }

    /// Simple aggregation function
    pub fn aggregate(&self, column_name: &str, func_name: &str) -> Result<DataFrame> {
        // Check if column exists
        if !self.source.contains_column(column_name) {
            return Err(PandRSError::Column(format!(
                "Column '{}' not found",
                column_name
            )));
        }

        // Create DataFrame for results
        let mut result = DataFrame::new();

        // Create group key and value columns
        let mut keys = Vec::new();
        let mut aggregated_values = Vec::new();

        // Get column data - using a stub for now
        // In a real implementation we would get numeric values from the column
        let column_data: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            // Add group key
            keys.push(format!("{:?}", key));

            // Extract data for this group
            let group_data: Vec<f64> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_data.len() {
                        Some(column_data[idx])
                    } else {
                        None
                    }
                })
                .collect();

            // Apply aggregation function
            let result_value = if group_data.is_empty() {
                "0.0".to_string()
            } else {
                match func_name {
                    "sum" => group_data.iter().sum::<f64>().to_string(),
                    "mean" => {
                        (group_data.iter().sum::<f64>() / group_data.len() as f64).to_string()
                    }
                    "min" => group_data
                        .iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b))
                        .to_string(),
                    "max" => group_data
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                        .to_string(),
                    "count" => group_data.len().to_string(),
                    _ => "0.0".to_string(),
                }
            };

            aggregated_values.push(result_value);
        }

        // Add group key column
        let key_column = Series::new(keys, Some("group_key".to_string()))?;
        result.add_column("group_key".to_string(), key_column)?;

        // Add aggregation result column
        let result_column_name = format!("{}_{}", column_name, func_name);
        let value_column = Series::new(aggregated_values, Some(result_column_name.clone()))?;
        result.add_column(result_column_name, value_column)?;

        Ok(result)
    }
}

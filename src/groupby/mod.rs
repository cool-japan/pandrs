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

    /// Get the size of each group as a DataFrame.
    ///
    /// Rows are ordered deterministically (sorted by the group key's string
    /// form) rather than following raw `HashMap` iteration order, which
    /// varied from run to run.
    pub fn size_as_df(&self) -> Result<DataFrame> {
        // Create a DataFrame for results
        let mut result = DataFrame::new();

        let mut entries: Vec<(String, usize)> = self
            .groups
            .iter()
            .map(|(k, indices)| (format!("{:?}", k), indices.len()))
            .collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut keys = Vec::with_capacity(entries.len());
        let mut sizes = Vec::with_capacity(entries.len());
        for (key, size) in entries {
            keys.push(key);
            sizes.push(size.to_string());
        }

        // Add group key column
        let key_column = Series::new(keys, Some("group_key".to_string()))?;
        result.add_column("group_key".to_string(), key_column)?;

        // Add size column
        let size_column = Series::new(sizes, Some("size".to_string()))?;
        result.add_column("size".to_string(), size_column)?;

        Ok(result)
    }

    /// Simple aggregation function.
    ///
    /// Fetches the real numeric values of `column_name` from the source
    /// DataFrame (previously a hardcoded empty `Vec`, which made every
    /// group's data look empty and every aggregate come back `"0.0"`
    /// regardless of the real data) and reduces each group's values with
    /// `func_name` (one of `"sum"`, `"mean"`, `"min"`, `"max"`, `"count"`).
    /// Groups are emitted in a deterministic order (sorted by the group
    /// key's string form) rather than raw `HashMap` iteration order.
    pub fn aggregate(&self, column_name: &str, func_name: &str) -> Result<DataFrame> {
        // Check if column exists
        if !self.source.contains_column(column_name) {
            return Err(PandRSError::Column(format!(
                "Column '{}' not found",
                column_name
            )));
        }

        if !matches!(func_name, "sum" | "mean" | "min" | "max" | "count") {
            return Err(PandRSError::Consistency(format!(
                "Unsupported aggregation function '{}' (expected one of: sum, mean, min, max, count)",
                func_name
            )));
        }

        // Fetch the column's real numeric values once (not a stub empty Vec).
        let column_data: Vec<f64> = self.source.get_column_numeric_values(column_name)?;

        // Deterministic group order: sort by the same string form used for
        // the output key column, instead of raw (unordered) HashMap
        // iteration.
        let mut sorted_groups: Vec<(String, &Vec<usize>)> = self
            .groups
            .iter()
            .map(|(k, indices)| (format!("{:?}", k), indices))
            .collect();
        sorted_groups.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut keys = Vec::with_capacity(sorted_groups.len());
        let mut aggregated_values = Vec::with_capacity(sorted_groups.len());

        for (key_str, indices) in sorted_groups {
            let group_data: Vec<f64> = indices
                .iter()
                .filter_map(|&idx| column_data.get(idx).copied())
                .collect();

            // Every key in `self.groups` owns at least one source row (see
            // `GroupBy::new`), so an empty `group_data` here only happens if
            // the column read returned fewer values than the DataFrame's
            // row count -- an inconsistency worth surfacing rather than
            // papering over with a fabricated "0.0".
            if group_data.is_empty() {
                return Err(PandRSError::Consistency(format!(
                    "No numeric values found for group {} in column '{}'",
                    key_str, column_name
                )));
            }

            let result_value = match func_name {
                "sum" => group_data.iter().sum::<f64>().to_string(),
                "mean" => (group_data.iter().sum::<f64>() / group_data.len() as f64).to_string(),
                "min" => group_data
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b))
                    .to_string(),
                "max" => group_data
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    .to_string(),
                "count" => group_data.len().to_string(),
                _ => {
                    return Err(PandRSError::Consistency(format!(
                        "Unsupported aggregation function '{}'",
                        func_name
                    )))
                }
            };

            keys.push(key_str);
            aggregated_values.push(result_value);
        }

        // Create DataFrame for results
        let mut result = DataFrame::new();

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

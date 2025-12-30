//! GroupBy operations for DataFrames
//!
//! Provides pandas-compatible GroupBy functionality for aggregating data.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::collections::HashMap;

/// GroupBy object that holds the grouped data
pub struct DataFrameGroupBy<'a> {
    df: &'a DataFrame,
    group_columns: Vec<String>,
    /// Maps group key (as string representation) to row indices
    groups: HashMap<String, Vec<usize>>,
    /// Maps group key to the actual group values
    group_keys: HashMap<String, Vec<String>>,
}

impl<'a> DataFrameGroupBy<'a> {
    /// Create a new GroupBy object
    pub fn new(df: &'a DataFrame, by: &[&str]) -> Result<Self> {
        if by.is_empty() {
            return Err(Error::InvalidValue(
                "GroupBy requires at least one column".to_string(),
            ));
        }

        // Validate columns exist
        for col in by {
            if !df.contains_column(col) {
                return Err(Error::InvalidValue(format!(
                    "Column '{}' not found in DataFrame",
                    col
                )));
            }
        }

        let group_columns: Vec<String> = by.iter().map(|s| s.to_string()).collect();
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut group_keys: HashMap<String, Vec<String>> = HashMap::new();

        let row_count = df.row_count();

        // Build group indices
        for row_idx in 0..row_count {
            let mut key_parts: Vec<String> = Vec::new();

            for col in &group_columns {
                let value = if let Ok(values) = df.get_column_string_values(col) {
                    values.get(row_idx).cloned().unwrap_or_default()
                } else if let Ok(values) = df.get_column_numeric_values(col) {
                    let v = values.get(row_idx).copied().unwrap_or(f64::NAN);
                    if v.is_nan() {
                        "NaN".to_string()
                    } else {
                        v.to_string()
                    }
                } else {
                    "".to_string()
                };
                key_parts.push(value);
            }

            let key = key_parts.join("|||");
            groups
                .entry(key.clone())
                .or_insert_with(Vec::new)
                .push(row_idx);
            group_keys.entry(key).or_insert(key_parts);
        }

        Ok(Self {
            df,
            group_columns,
            groups,
            group_keys,
        })
    }

    /// Get the number of groups
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }

    /// Get group sizes
    pub fn size(&self) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        let mut group_col_values: Vec<Vec<String>> = vec![Vec::new(); self.group_columns.len()];
        let mut sizes: Vec<f64> = Vec::new();

        for (key, indices) in &self.groups {
            if let Some(key_values) = self.group_keys.get(key) {
                for (i, val) in key_values.iter().enumerate() {
                    group_col_values[i].push(val.clone());
                }
            }
            sizes.push(indices.len() as f64);
        }

        // Add group columns
        for (i, col_name) in self.group_columns.iter().enumerate() {
            result.add_column(
                col_name.clone(),
                Series::new(group_col_values[i].clone(), Some(col_name.clone()))?,
            )?;
        }

        // Add size column
        result.add_column(
            "size".to_string(),
            Series::new(sizes, Some("size".to_string()))?,
        )?;

        Ok(result)
    }

    /// Count rows per group
    pub fn count(&self) -> Result<DataFrame> {
        self.size()
    }

    /// Sum numeric columns per group
    pub fn sum(&self) -> Result<DataFrame> {
        self.aggregate(|values| values.iter().filter(|v| !v.is_nan()).sum())
    }

    /// Mean of numeric columns per group
    pub fn mean(&self) -> Result<DataFrame> {
        self.aggregate(|values| {
            let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if valid.is_empty() {
                f64::NAN
            } else {
                valid.iter().sum::<f64>() / valid.len() as f64
            }
        })
    }

    /// Minimum of numeric columns per group
    pub fn min(&self) -> Result<DataFrame> {
        self.aggregate(|values| {
            values
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .fold(f64::INFINITY, f64::min)
        })
    }

    /// Maximum of numeric columns per group
    pub fn max(&self) -> Result<DataFrame> {
        self.aggregate(|values| {
            values
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
    }

    /// Standard deviation of numeric columns per group (sample std, using n-1)
    pub fn std(&self) -> Result<DataFrame> {
        self.aggregate(|values| {
            let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if valid.len() <= 1 {
                f64::NAN
            } else {
                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                let variance: f64 = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (valid.len() - 1) as f64;
                variance.sqrt()
            }
        })
    }

    /// Variance of numeric columns per group (sample variance, using n-1)
    pub fn var(&self) -> Result<DataFrame> {
        self.aggregate(|values| {
            let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if valid.len() <= 1 {
                f64::NAN
            } else {
                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (valid.len() - 1) as f64
            }
        })
    }

    /// First value of each group
    pub fn first(&self) -> Result<DataFrame> {
        self.aggregate_first_last(true)
    }

    /// Last value of each group
    pub fn last(&self) -> Result<DataFrame> {
        self.aggregate_first_last(false)
    }

    /// Internal method to apply an aggregation function
    fn aggregate<F>(&self, agg_fn: F) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut result = DataFrame::new();

        // Get numeric columns (excluding group columns)
        let numeric_cols: Vec<String> = self
            .df
            .column_names()
            .into_iter()
            .filter(|col| {
                !self.group_columns.contains(col) && self.df.get_column_numeric_values(col).is_ok()
            })
            .collect();

        // Prepare group column values
        let mut group_col_values: Vec<Vec<String>> = vec![Vec::new(); self.group_columns.len()];

        // Prepare aggregated values for each numeric column
        let mut agg_values: HashMap<String, Vec<f64>> = HashMap::new();
        for col in &numeric_cols {
            agg_values.insert(col.clone(), Vec::new());
        }

        // Process each group
        for (key, indices) in &self.groups {
            // Add group key values
            if let Some(key_values) = self.group_keys.get(key) {
                for (i, val) in key_values.iter().enumerate() {
                    group_col_values[i].push(val.clone());
                }
            }

            // Aggregate each numeric column
            for col in &numeric_cols {
                if let Ok(all_values) = self.df.get_column_numeric_values(col) {
                    let group_values: Vec<f64> = indices
                        .iter()
                        .filter_map(|&i| all_values.get(i).copied())
                        .collect();
                    let aggregated = agg_fn(&group_values);
                    agg_values.get_mut(col).unwrap().push(aggregated);
                }
            }
        }

        // Build result DataFrame
        // Add group columns
        for (i, col_name) in self.group_columns.iter().enumerate() {
            result.add_column(
                col_name.clone(),
                Series::new(group_col_values[i].clone(), Some(col_name.clone()))?,
            )?;
        }

        // Add aggregated columns
        for col in &numeric_cols {
            if let Some(values) = agg_values.get(col) {
                result.add_column(col.clone(), Series::new(values.clone(), Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    /// Internal method for first/last aggregations
    fn aggregate_first_last(&self, first: bool) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        // Get all non-group columns
        let other_cols: Vec<String> = self
            .df
            .column_names()
            .into_iter()
            .filter(|col| !self.group_columns.contains(col))
            .collect();

        // Prepare group column values
        let mut group_col_values: Vec<Vec<String>> = vec![Vec::new(); self.group_columns.len()];

        // Prepare values for each column
        let mut numeric_values: HashMap<String, Vec<f64>> = HashMap::new();
        let mut string_values: HashMap<String, Vec<String>> = HashMap::new();

        for col in &other_cols {
            if self.df.get_column_numeric_values(col).is_ok() {
                numeric_values.insert(col.clone(), Vec::new());
            } else if self.df.get_column_string_values(col).is_ok() {
                string_values.insert(col.clone(), Vec::new());
            }
        }

        // Process each group
        for (key, indices) in &self.groups {
            // Add group key values
            if let Some(key_values) = self.group_keys.get(key) {
                for (i, val) in key_values.iter().enumerate() {
                    group_col_values[i].push(val.clone());
                }
            }

            let target_idx = if first {
                *indices.first().unwrap()
            } else {
                *indices.last().unwrap()
            };

            // Get first/last value for each column
            for col in &other_cols {
                if let Ok(all_values) = self.df.get_column_numeric_values(col) {
                    let value = all_values.get(target_idx).copied().unwrap_or(f64::NAN);
                    numeric_values.get_mut(col).unwrap().push(value);
                } else if let Ok(all_values) = self.df.get_column_string_values(col) {
                    let value = all_values.get(target_idx).cloned().unwrap_or_default();
                    string_values.get_mut(col).unwrap().push(value);
                }
            }
        }

        // Build result DataFrame
        // Add group columns
        for (i, col_name) in self.group_columns.iter().enumerate() {
            result.add_column(
                col_name.clone(),
                Series::new(group_col_values[i].clone(), Some(col_name.clone()))?,
            )?;
        }

        // Add other columns (preserve order from original DataFrame)
        for col in &other_cols {
            if let Some(values) = numeric_values.get(col) {
                result.add_column(col.clone(), Series::new(values.clone(), Some(col.clone()))?)?;
            } else if let Some(values) = string_values.get(col) {
                result.add_column(col.clone(), Series::new(values.clone(), Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    /// Apply multiple aggregations at once
    pub fn agg(&self, aggs: &[(&str, &str)]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        // Prepare group column values
        let mut group_col_values: Vec<Vec<String>> = vec![Vec::new(); self.group_columns.len()];

        // First pass: collect group keys
        for (key, _) in &self.groups {
            if let Some(key_values) = self.group_keys.get(key) {
                for (i, val) in key_values.iter().enumerate() {
                    group_col_values[i].push(val.clone());
                }
            }
        }

        // Add group columns
        for (i, col_name) in self.group_columns.iter().enumerate() {
            result.add_column(
                col_name.clone(),
                Series::new(group_col_values[i].clone(), Some(col_name.clone()))?,
            )?;
        }

        // Process each aggregation
        for (col, agg_name) in aggs {
            if !self.df.contains_column(col) {
                continue;
            }

            if let Ok(all_values) = self.df.get_column_numeric_values(col) {
                let mut agg_values: Vec<f64> = Vec::new();

                for (_, indices) in &self.groups {
                    let group_values: Vec<f64> = indices
                        .iter()
                        .filter_map(|&i| all_values.get(i).copied())
                        .collect();

                    let aggregated = match *agg_name {
                        "sum" => group_values.iter().filter(|v| !v.is_nan()).sum(),
                        "mean" => {
                            let valid: Vec<f64> = group_values
                                .iter()
                                .filter(|v| !v.is_nan())
                                .copied()
                                .collect();
                            if valid.is_empty() {
                                f64::NAN
                            } else {
                                valid.iter().sum::<f64>() / valid.len() as f64
                            }
                        }
                        "min" => group_values
                            .iter()
                            .filter(|v| !v.is_nan())
                            .copied()
                            .fold(f64::INFINITY, f64::min),
                        "max" => group_values
                            .iter()
                            .filter(|v| !v.is_nan())
                            .copied()
                            .fold(f64::NEG_INFINITY, f64::max),
                        "count" => group_values.iter().filter(|v| !v.is_nan()).count() as f64,
                        "std" => {
                            let valid: Vec<f64> = group_values
                                .iter()
                                .filter(|v| !v.is_nan())
                                .copied()
                                .collect();
                            if valid.len() <= 1 {
                                f64::NAN
                            } else {
                                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                                let variance: f64 =
                                    valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                        / (valid.len() - 1) as f64;
                                variance.sqrt()
                            }
                        }
                        "var" => {
                            let valid: Vec<f64> = group_values
                                .iter()
                                .filter(|v| !v.is_nan())
                                .copied()
                                .collect();
                            if valid.len() <= 1 {
                                f64::NAN
                            } else {
                                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                                valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                    / (valid.len() - 1) as f64
                            }
                        }
                        "first" => group_values.first().copied().unwrap_or(f64::NAN),
                        "last" => group_values.last().copied().unwrap_or(f64::NAN),
                        _ => f64::NAN,
                    };

                    agg_values.push(aggregated);
                }

                let result_col_name = format!("{}_{}", col, agg_name);
                result.add_column(
                    result_col_name.clone(),
                    Series::new(agg_values, Some(result_col_name))?,
                )?;
            }
        }

        Ok(result)
    }
}

/// Extension trait to add groupby method to DataFrame (multi-column support)
pub trait PandasGroupByExt {
    /// Group DataFrame by one or more columns
    ///
    /// # Example
    /// ```ignore
    /// use pandrs::dataframe::pandas_compat::PandasGroupByExt;
    ///
    /// let result = df.groupby_multi(&["category"]).unwrap().sum().unwrap();
    /// ```
    fn groupby_multi(&self, by: &[&str]) -> Result<DataFrameGroupBy>;
}

impl PandasGroupByExt for DataFrame {
    fn groupby_multi(&self, by: &[&str]) -> Result<DataFrameGroupBy> {
        DataFrameGroupBy::new(self, by)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(
                vec![10.0, 20.0, 30.0, 40.0, 50.0],
                Some("value".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "score".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("score".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_groupby_sum() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().sum().unwrap();

        assert_eq!(result.row_count(), 2);

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        // Find indices for A and B
        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A: 10 + 30 + 50 = 90
        assert_eq!(values[a_idx], 90.0);
        // B: 20 + 40 = 60
        assert_eq!(values[b_idx], 60.0);
    }

    #[test]
    fn test_groupby_mean() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().mean().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A: (10 + 30 + 50) / 3 = 30
        assert_eq!(values[a_idx], 30.0);
        // B: (20 + 40) / 2 = 30
        assert_eq!(values[b_idx], 30.0);
    }

    #[test]
    fn test_groupby_min() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().min().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A: min(10, 30, 50) = 10
        assert_eq!(values[a_idx], 10.0);
        // B: min(20, 40) = 20
        assert_eq!(values[b_idx], 20.0);
    }

    #[test]
    fn test_groupby_max() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().max().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A: max(10, 30, 50) = 50
        assert_eq!(values[a_idx], 50.0);
        // B: max(20, 40) = 40
        assert_eq!(values[b_idx], 40.0);
    }

    #[test]
    fn test_groupby_count() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().count().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let sizes = result.get_column_numeric_values("size").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A: 3 rows
        assert_eq!(sizes[a_idx], 3.0);
        // B: 2 rows
        assert_eq!(sizes[b_idx], 2.0);
    }

    #[test]
    fn test_groupby_std() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().std().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();

        // A: std of [10, 30, 50] with sample std (n-1)
        // mean = 30, variance = ((10-30)^2 + (30-30)^2 + (50-30)^2) / 2 = 400
        // std = 20
        assert!((values[a_idx] - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_groupby_first() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().first().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A first: 10
        assert_eq!(values[a_idx], 10.0);
        // B first: 20
        assert_eq!(values[b_idx], 20.0);
    }

    #[test]
    fn test_groupby_last() {
        let df = create_test_df();
        let result = df.groupby_multi(&["category"]).unwrap().last().unwrap();

        let cats = result.get_column_string_values("category").unwrap();
        let values = result.get_column_numeric_values("value").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        let b_idx = cats.iter().position(|c| c == "B").unwrap();

        // A last: 50
        assert_eq!(values[a_idx], 50.0);
        // B last: 40
        assert_eq!(values[b_idx], 40.0);
    }

    #[test]
    fn test_groupby_multiple_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "cat1".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
                Some("cat1".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "cat2".to_string(),
            Series::new(
                vec![
                    "X".to_string(),
                    "Y".to_string(),
                    "X".to_string(),
                    "Y".to_string(),
                ],
                Some("cat2".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.groupby_multi(&["cat1", "cat2"]).unwrap().sum().unwrap();

        // Should have 4 groups: (A,X), (A,Y), (B,X), (B,Y)
        assert_eq!(result.row_count(), 4);
    }

    #[test]
    fn test_groupby_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec!["A".to_string(), "A".to_string(), "A".to_string()],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(vec![10.0, f64::NAN, 30.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.groupby_multi(&["category"]).unwrap().sum().unwrap();

        let values = result.get_column_numeric_values("value").unwrap();
        // Sum should ignore NaN: 10 + 30 = 40
        assert_eq!(values[0], 40.0);
    }

    #[test]
    fn test_groupby_agg() {
        let df = create_test_df();
        let result = df
            .groupby_multi(&["category"])
            .unwrap()
            .agg(&[("value", "sum"), ("value", "mean"), ("score", "max")])
            .unwrap();

        assert!(result.contains_column("value_sum"));
        assert!(result.contains_column("value_mean"));
        assert!(result.contains_column("score_max"));

        let value_sums = result.get_column_numeric_values("value_sum").unwrap();
        let cats = result.get_column_string_values("category").unwrap();

        let a_idx = cats.iter().position(|c| c == "A").unwrap();
        assert_eq!(value_sums[a_idx], 90.0);
    }

    #[test]
    fn test_ngroups() {
        let df = create_test_df();
        let gb = df.groupby_multi(&["category"]).unwrap();
        assert_eq!(gb.ngroups(), 2);
    }
}

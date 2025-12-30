//! Module providing parallel processing functionality

use crate::error::Result;
use crate::na::NA;
use crate::series::NASeries;
use crate::DataFrame;
use crate::Series;
use rayon::prelude::*;

/// Parallel processing extension: Series parallel processing
impl<T> Series<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// Apply a function to all elements in parallel
    pub fn par_map<F, R>(&self, f: F) -> Series<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let new_values: Vec<R> = self.values().par_iter().map(|v| f(v)).collect();

        Series::new(new_values, self.name().cloned()).unwrap()
    }

    /// Filter elements in parallel based on a condition function
    pub fn par_filter<F>(&self, f: F) -> Series<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let filtered_values: Vec<T> = self.values().par_iter().filter(|v| f(v)).cloned().collect();

        Series::new(filtered_values, self.name().cloned()).unwrap()
    }
}

/// Parallel processing extension: Series with NA values
impl<T> NASeries<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// Apply a function to all elements in parallel (ignoring NA)
    pub fn par_map<F, R>(&self, f: F) -> NASeries<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let new_values: Vec<NA<R>> = self
            .values()
            .par_iter()
            .map(|v| match v {
                NA::Value(val) => NA::Value(f(val)),
                NA::NA => NA::NA,
            })
            .collect();

        NASeries::new(new_values, self.name().cloned()).unwrap()
    }

    /// Filter elements in parallel based on a condition function (excluding NA)
    pub fn par_filter<F>(&self, f: F) -> NASeries<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let filtered_values: Vec<NA<T>> = self
            .values()
            .par_iter()
            .filter(|v| match v {
                NA::Value(val) => f(val),
                NA::NA => false,
            })
            .cloned()
            .collect();

        NASeries::new(filtered_values, self.name().cloned()).unwrap()
    }
}

/// Parallel processing extension: DataFrame parallel processing
impl DataFrame {
    /// Apply a function to all columns in parallel
    pub fn par_apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str, usize, &str) -> String + Send + Sync,
    {
        let mut result = DataFrame::new();

        // Process each column in parallel
        let column_names = self.column_names().to_vec();

        // Prepare row count and column names
        let n_rows = self.row_count();

        // Process each column
        for col_name in &column_names {
            // Get string values
            let values = self.get_column_string_values(col_name)?;

            // Create new values in parallel
            let new_values: Vec<String> = (0..n_rows)
                .into_par_iter()
                .map(|i| {
                    let val = if i < values.len() { &values[i] } else { "" };
                    f(col_name, i, val)
                })
                .collect();

            // Add new column
            let new_series = Series::new(new_values, Some(col_name.clone()))?;
            result.add_column(col_name.clone(), new_series)?;
        }

        Ok(result)
    }

    /// Filter rows in parallel
    pub fn par_filter_rows<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let mut result = DataFrame::new();

        // Get column names
        let column_names = self.column_names().to_vec();

        // Filter row indices in parallel
        let row_indices: Vec<usize> = (0..self.row_count())
            .into_par_iter()
            .filter(|&i| f(i))
            .collect();

        // Filter each column
        for col_name in &column_names {
            let values = self.get_column_string_values(col_name)?;

            // Get filtered values
            let filtered_values: Vec<String> = row_indices
                .par_iter()
                .filter_map(|&i| {
                    if i < values.len() {
                        Some(values[i].clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Add new column
            let new_series = Series::new(filtered_values, Some(col_name.clone()))?;
            result.add_column(col_name.clone(), new_series)?;
        }

        Ok(result)
    }

    /// Execute groupby operation in parallel
    pub fn par_groupby<K>(&self, key_func: K) -> Result<HashMap<String, DataFrame>>
    where
        K: Fn(usize) -> String + Send + Sync,
    {
        // Group map
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        // Calculate keys for each row and group row indices
        (0..self.row_count())
            .into_par_iter()
            .map(|i| (key_func(i), i))
            .collect::<Vec<_>>()
            .into_iter() // Process results sequentially
            .for_each(|(key, idx)| {
                groups.entry(key).or_insert_with(Vec::new).push(idx);
            });

        // Create DataFrame for each group
        let mut result = HashMap::new();

        for (key, indices) in groups {
            let group_df = self.par_filter_rows(|i| indices.contains(&i))?;
            result.insert(key, group_df);
        }

        Ok(result)
    }
}

/// Utilities for parallel data operations
pub struct ParallelUtils;

impl ParallelUtils {
    /// Sort a vector in parallel
    pub fn par_sort<T>(mut values: Vec<T>) -> Vec<T>
    where
        T: Ord + Send,
    {
        values.par_sort();
        values
    }

    /// Aggregate vector elements in parallel
    pub fn par_sum<T>(values: &[T]) -> T
    where
        T: Send + Sync + std::iter::Sum + Copy,
    {
        values.par_iter().copied().sum()
    }

    /// Calculate mean in parallel
    pub fn par_mean<T>(values: &[T]) -> Option<f64>
    where
        T: Send + Sync + Copy + Into<f64>,
    {
        if values.is_empty() {
            return None;
        }

        let sum: f64 = values.par_iter().map(|&v| v.into()).sum();

        Some(sum / values.len() as f64)
    }

    /// Find minimum value in parallel
    pub fn par_min<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        values.par_iter().min().copied()
    }

    /// Find maximum value in parallel
    pub fn par_max<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        values.par_iter().max().copied()
    }
}

use std::collections::HashMap;

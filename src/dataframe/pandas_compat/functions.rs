//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

// Import types from the types module
use super::types::{Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue};

// Import helper modules
use super::helpers::{aggregations, comparison_ops, math_ops, string_ops, window_ops};
/// Helper function to select specific rows by indices
fn select_rows_by_indices(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let selected: Vec<f64> = indices
                .iter()
                .filter_map(|&i| values.get(i).copied())
                .collect();
            new_df.add_column(
                col_name.clone(),
                Series::new(selected, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            let selected: Vec<String> = indices
                .iter()
                .filter_map(|&i| values.get(i).cloned())
                .collect();
            new_df.add_column(
                col_name.clone(),
                Series::new(selected, Some(col_name.clone()))?,
            )?;
        }
    }
    Ok(new_df)
}
/// Pandas compatibility extension trait for DataFrame
// Import trait definition
use super::trait_def::PandasCompatExt;

impl PandasCompatExt for DataFrame {
    fn assign<F, T>(&self, name: &str, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrame) -> Vec<T>,
        T: Into<SeriesValue>,
    {
        let values = func(self);
        let mut df = self.clone();
        if values.is_empty() {
            return Ok(df);
        }
        let first = values.first().map(|v| {
            let sv: SeriesValue = unsafe { std::ptr::read(v as *const T) }.into();
            sv
        });
        match first {
            Some(SeriesValue::Int(_)) => {
                let int_values: Vec<i64> = values
                    .into_iter()
                    .map(|v| match v.into() {
                        SeriesValue::Int(i) => i,
                        _ => 0,
                    })
                    .collect();
                df.add_column(
                    name.to_string(),
                    Series::new(int_values, Some(name.to_string()))?,
                )?;
            }
            Some(SeriesValue::Float(_)) => {
                let float_values: Vec<f64> = values
                    .into_iter()
                    .map(|v| match v.into() {
                        SeriesValue::Float(f) => f,
                        SeriesValue::Int(i) => i as f64,
                        _ => 0.0,
                    })
                    .collect();
                df.add_column(
                    name.to_string(),
                    Series::new(float_values, Some(name.to_string()))?,
                )?;
            }
            Some(SeriesValue::String(_)) => {
                let string_values: Vec<String> = values
                    .into_iter()
                    .map(|v| match v.into() {
                        SeriesValue::String(s) => s,
                        _ => String::new(),
                    })
                    .collect();
                df.add_column(
                    name.to_string(),
                    Series::new(string_values, Some(name.to_string()))?,
                )?;
            }
            Some(SeriesValue::Bool(_)) => {
                let bool_values: Vec<bool> = values
                    .into_iter()
                    .map(|v| match v.into() {
                        SeriesValue::Bool(b) => b,
                        _ => false,
                    })
                    .collect();
                df.add_column(
                    name.to_string(),
                    Series::new(bool_values, Some(name.to_string()))?,
                )?;
            }
            None => {}
        }
        Ok(df)
    }
    fn assign_many(&self, assignments: Vec<(&str, Vec<f64>)>) -> Result<DataFrame> {
        let mut df = self.clone();
        for (name, values) in assignments {
            df.add_column(
                name.to_string(),
                Series::new(values, Some(name.to_string()))?,
            )?;
        }
        Ok(df)
    }
    fn pipe<F, R>(&self, func: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        func(self)
    }
    fn pipe_result<F>(&self, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&Self) -> Result<DataFrame>,
    {
        func(self)
    }
    fn isin(&self, column: &str, values: &[&str]) -> Result<Vec<bool>> {
        let col_values = self.get_column_string_values(column)?;
        let value_set: HashSet<&str> = values.iter().copied().collect();
        let result: Vec<bool> = col_values
            .iter()
            .map(|s| value_set.contains(s.as_str()))
            .collect();
        Ok(result)
    }
    fn isin_numeric(&self, column: &str, values: &[f64]) -> Result<Vec<bool>> {
        let col_values = self.get_column_numeric_values(column)?;
        let value_set: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
        let result: Vec<bool> = col_values
            .iter()
            .map(|v| value_set.contains(&v.to_bits()))
            .collect();
        Ok(result)
    }
    fn nlargest(&self, n: usize, column: &str) -> Result<DataFrame> {
        let col_values = self.get_column_numeric_values(column)?;
        let mut indexed_values: Vec<(usize, f64)> = col_values.into_iter().enumerate().collect();
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed_values.truncate(n);
        let indices: Vec<usize> = indexed_values.into_iter().map(|(i, _)| i).collect();
        select_rows_by_indices(self, &indices)
    }
    fn nsmallest(&self, n: usize, column: &str) -> Result<DataFrame> {
        let col_values = self.get_column_numeric_values(column)?;
        let mut indexed_values: Vec<(usize, f64)> = col_values.into_iter().enumerate().collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        indexed_values.truncate(n);
        let indices: Vec<usize> = indexed_values.into_iter().map(|(i, _)| i).collect();
        select_rows_by_indices(self, &indices)
    }
    fn idxmax(&self, column: &str) -> Result<Option<usize>> {
        let values = self.get_column_numeric_values(column)?;
        let max_idx = values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i);
        Ok(max_idx)
    }
    fn idxmin(&self, column: &str) -> Result<Option<usize>> {
        let values = self.get_column_numeric_values(column)?;
        let min_idx = values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i);
        Ok(min_idx)
    }
    fn rank(&self, column: &str, method: RankMethod) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let n = values.len();
        let mut indexed_values: Vec<(usize, f64)> = values.into_iter().enumerate().collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut ranks = vec![f64::NAN; n];
        let mut i = 0;
        while i < indexed_values.len() {
            let mut j = i;
            while j < indexed_values.len() && indexed_values[j].1 == indexed_values[i].1 {
                j += 1;
            }
            let rank = match method {
                RankMethod::Average => (i + j + 1) as f64 / 2.0,
                RankMethod::Min => (i + 1) as f64,
                RankMethod::Max => j as f64,
                RankMethod::First => 0.0,
                RankMethod::Dense => 0.0,
            };
            for k in i..j {
                let idx = indexed_values[k].0;
                ranks[idx] = if method == RankMethod::First {
                    (k + 1) as f64
                } else {
                    rank
                };
            }
            i = j;
        }
        if method == RankMethod::Dense {
            let mut dense_rank = 0.0;
            let mut i = 0;
            while i < indexed_values.len() {
                dense_rank += 1.0;
                let mut j = i;
                while j < indexed_values.len() && indexed_values[j].1 == indexed_values[i].1 {
                    ranks[indexed_values[j].0] = dense_rank;
                    j += 1;
                }
                i = j;
            }
        }
        Ok(ranks)
    }
    fn clip(&self, column: &str, lower: Option<f64>, upper: Option<f64>) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let clipped: Vec<f64> = values
            .iter()
            .map(|&v| {
                let v = lower.map_or(v, |l| v.max(l));
                upper.map_or(v, |u| v.min(u))
            })
            .collect();
        let mut df = self.clone();
        df.add_column(
            column.to_string(),
            Series::new(clipped, Some(column.to_string()))?,
        )?;
        Ok(df)
    }
    fn between(&self, column: &str, lower: f64, upper: f64) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        let result: Vec<bool> = values.iter().map(|&v| v >= lower && v <= upper).collect();
        Ok(result)
    }
    fn transpose(&self) -> Result<DataFrame> {
        let col_names = self.column_names();
        let n_rows = self.row_count();
        let n_cols = col_names.len();
        if n_rows == 0 || n_cols == 0 {
            return Ok(DataFrame::new());
        }
        let mut new_df = DataFrame::new();
        let all_values: Vec<Vec<String>> = col_names
            .iter()
            .map(|col| self.get_column_string_values(col).unwrap_or_default())
            .collect();
        for i in 0..n_rows {
            let col_name = format!("row_{}", i);
            let values: Vec<String> = all_values
                .iter()
                .map(|col_vals| col_vals.get(i).cloned().unwrap_or_default())
                .collect();
            new_df.add_column(col_name.clone(), Series::new(values, Some(col_name))?)?;
        }
        Ok(new_df)
    }
    fn cumsum(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let mut cumsum = 0.0;
        let result: Vec<f64> = values
            .iter()
            .map(|&v| {
                cumsum += v;
                cumsum
            })
            .collect();
        Ok(result)
    }
    fn cumprod(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let mut cumprod = 1.0;
        let result: Vec<f64> = values
            .iter()
            .map(|&v| {
                cumprod *= v;
                cumprod
            })
            .collect();
        Ok(result)
    }
    fn cummax(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let mut cummax = f64::NEG_INFINITY;
        let result: Vec<f64> = values
            .iter()
            .map(|&v| {
                cummax = cummax.max(v);
                cummax
            })
            .collect();
        Ok(result)
    }
    fn cummin(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let mut cummin = f64::INFINITY;
        let result: Vec<f64> = values
            .iter()
            .map(|&v| {
                cummin = cummin.min(v);
                cummin
            })
            .collect();
        Ok(result)
    }
    fn shift(&self, column: &str, periods: i32) -> Result<Vec<Option<f64>>> {
        let values = self.get_column_numeric_values(column)?;
        let n = values.len();
        let result: Vec<Option<f64>> = (0..n)
            .map(|i| {
                let src_idx = i as i32 - periods;
                if src_idx >= 0 && src_idx < n as i32 {
                    Some(values[src_idx as usize])
                } else {
                    None
                }
            })
            .collect();
        Ok(result)
    }
    fn nunique(&self) -> Result<Vec<(String, usize)>> {
        let col_names = self.column_names();
        let mut results = Vec::new();
        for col_name in col_names {
            let values = self.get_column_string_values(&col_name)?;
            let unique_values: HashSet<String> = values.into_iter().collect();
            results.push((col_name, unique_values.len()));
        }
        Ok(results)
    }
    fn memory_usage(&self) -> usize {
        let col_names = self.column_names();
        let n_rows = self.row_count();
        let base_size = col_names.len() * n_rows * 8;
        base_size + col_names.len() * 64 + 256
    }
    fn value_counts(&self, column: &str) -> Result<Vec<(String, usize)>> {
        let values = self.get_column_string_values(column)?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in values {
            *counts.entry(value).or_insert(0) += 1;
        }
        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        Ok(result)
    }
    fn value_counts_numeric(&self, column: &str) -> Result<Vec<(f64, usize)>> {
        let values = self.get_column_numeric_values(column)?;
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for value in values {
            *counts.entry(value.to_bits()).or_insert(0) += 1;
        }
        let mut result: Vec<(f64, usize)> = counts
            .into_iter()
            .map(|(bits, count)| (f64::from_bits(bits), count))
            .collect();
        result.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
        });
        Ok(result)
    }
    fn describe(&self, column: &str) -> Result<DescribeStats> {
        let values = self.get_column_numeric_values(column)?;
        if values.is_empty() {
            return Err(Error::Empty("Cannot describe empty column".to_string()));
        }
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std = variance.sqrt();
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let min = sorted[0];
        let max = sorted[count - 1];
        let percentile = |p: f64| -> f64 {
            let idx = p / 100.0 * (count - 1) as f64;
            let lower = idx.floor() as usize;
            let upper = idx.ceil() as usize;
            if lower == upper {
                sorted[lower]
            } else {
                let weight = idx - lower as f64;
                sorted[lower] * (1.0 - weight) + sorted[upper] * weight
            }
        };
        Ok(DescribeStats {
            count,
            mean,
            std,
            min,
            q25: percentile(25.0),
            q50: percentile(50.0),
            q75: percentile(75.0),
            max,
        })
    }
    fn apply<F, T>(&self, func: F, axis: Axis) -> Result<Vec<T>>
    where
        F: Fn(&[f64]) -> T,
    {
        match axis {
            Axis::Rows => {
                let col_names = self.column_names();
                let mut result = Vec::with_capacity(self.row_count());
                for i in 0..self.row_count() {
                    let row_values: Vec<f64> = col_names
                        .iter()
                        .filter_map(|col| {
                            self.get_column_numeric_values(col)
                                .ok()
                                .and_then(|vals| vals.get(i).copied())
                        })
                        .collect();
                    result.push(func(&row_values));
                }
                Ok(result)
            }
            Axis::Columns => {
                let col_names = self.column_names();
                let mut result = Vec::with_capacity(col_names.len());
                for col in col_names {
                    if let Ok(values) = self.get_column_numeric_values(&col) {
                        result.push(func(&values));
                    }
                }
                Ok(result)
            }
        }
    }
    fn corr(&self) -> Result<CorrelationMatrix> {
        let col_names = self.column_names();
        let n_cols = col_names.len();
        if n_cols == 0 {
            return Err(Error::Empty(
                "Cannot compute correlation on empty DataFrame".to_string(),
            ));
        }
        let mut columns_data: Vec<Vec<f64>> = Vec::new();
        let mut valid_columns: Vec<String> = Vec::new();
        for col_name in col_names {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                columns_data.push(values);
                valid_columns.push(col_name);
            }
        }
        if columns_data.is_empty() {
            return Err(Error::Empty("No numeric columns found".to_string()));
        }
        let n = columns_data.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    matrix[i][j] = pearson_correlation(&columns_data[i], &columns_data[j]);
                }
            }
        }
        Ok(CorrelationMatrix {
            columns: valid_columns,
            values: matrix,
        })
    }
    fn cov(&self) -> Result<CorrelationMatrix> {
        let col_names = self.column_names();
        let mut columns_data: Vec<Vec<f64>> = Vec::new();
        let mut valid_columns: Vec<String> = Vec::new();
        for col_name in col_names {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                columns_data.push(values);
                valid_columns.push(col_name);
            }
        }
        if columns_data.is_empty() {
            return Err(Error::Empty("No numeric columns found".to_string()));
        }
        let n = columns_data.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = covariance(&columns_data[i], &columns_data[j]);
            }
        }
        Ok(CorrelationMatrix {
            columns: valid_columns,
            values: matrix,
        })
    }
    fn pct_change(&self, column: &str, periods: usize) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let mut result = vec![f64::NAN; values.len()];
        for i in periods..values.len() {
            let prev = values[i - periods];
            let curr = values[i];
            result[i] = if prev == 0.0 {
                f64::NAN
            } else {
                (curr - prev) / prev
            };
        }
        Ok(result)
    }
    fn diff(&self, column: &str, periods: usize) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let mut result = vec![f64::NAN; values.len()];
        for i in periods..values.len() {
            result[i] = values[i] - values[i - periods];
        }
        Ok(result)
    }
    fn replace(&self, column: &str, to_replace: &[&str], values: &[&str]) -> Result<DataFrame> {
        if to_replace.len() != values.len() {
            return Err(Error::InvalidValue(
                "to_replace and values must have same length".to_string(),
            ));
        }
        let replacement_map: HashMap<&str, &str> = to_replace
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (*k, *v))
            .collect();
        let col_values = self.get_column_string_values(column)?;
        let replaced: Vec<String> = col_values
            .iter()
            .map(|v| {
                replacement_map
                    .get(v.as_str())
                    .map(|&new| new.to_string())
                    .unwrap_or_else(|| v.clone())
            })
            .collect();
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(replaced.clone(), Some(col_name.clone()))?,
                )?;
            } else {
                if let Ok(values) = self.get_column_string_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                }
            }
        }
        Ok(df)
    }
    fn replace_numeric(
        &self,
        column: &str,
        to_replace: &[f64],
        values: &[f64],
    ) -> Result<DataFrame> {
        if to_replace.len() != values.len() {
            return Err(Error::InvalidValue(
                "to_replace and values must have same length".to_string(),
            ));
        }
        let replacement_map: HashMap<u64, f64> = to_replace
            .iter()
            .zip(values.iter())
            .map(|(k, v)| (k.to_bits(), *v))
            .collect();
        let col_values = self.get_column_numeric_values(column)?;
        let replaced: Vec<f64> = col_values
            .iter()
            .map(|&v| replacement_map.get(&v.to_bits()).copied().unwrap_or(v))
            .collect();
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(replaced.clone(), Some(col_name.clone()))?,
                )?;
            } else {
                if let Ok(values) = self.get_column_string_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                }
            }
        }
        Ok(df)
    }
    fn sample(&self, n: usize, replace: bool) -> Result<DataFrame> {
        use rand::seq::SliceRandom;
        use rand::Rng;
        let n_rows = self.row_count();
        if n_rows == 0 {
            return Ok(DataFrame::new());
        }
        if !replace && n > n_rows {
            return Err(Error::InvalidValue(format!(
                "Cannot sample {} rows without replacement from {} rows",
                n, n_rows
            )));
        }
        let mut rng = rand::rng();
        let indices: Vec<usize> = if replace {
            (0..n).map(|_| rng.random_range(0..n_rows)).collect()
        } else {
            let mut all_indices: Vec<usize> = (0..n_rows).collect();
            all_indices.shuffle(&mut rng);
            all_indices.into_iter().take(n).collect()
        };
        select_rows_by_indices(self, &indices)
    }
    fn drop_columns(&self, labels: &[&str]) -> Result<DataFrame> {
        let mut df = DataFrame::new();
        let drop_set: HashSet<&str> = labels.iter().copied().collect();
        for col_name in self.column_names() {
            if !drop_set.contains(col_name.as_str()) {
                if let Ok(values) = self.get_column_numeric_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(values) = self.get_column_string_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                }
            }
        }
        Ok(df)
    }
    fn rename_columns(&self, mapper: &HashMap<String, String>) -> Result<DataFrame> {
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            let new_name = mapper.get(&col_name).unwrap_or(&col_name);
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                df.add_column(
                    new_name.clone(),
                    Series::new(values, Some(new_name.clone()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                df.add_column(
                    new_name.clone(),
                    Series::new(values, Some(new_name.clone()))?,
                )?;
            }
        }
        Ok(df)
    }
    fn abs(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let abs_values: Vec<f64> = values.iter().map(|&v| v.abs()).collect();
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(abs_values.clone(), Some(col_name.clone()))?,
                )?;
            } else {
                if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                    df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                }
            }
        }
        Ok(df)
    }
    fn round(&self, column: &str, decimals: i32) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let multiplier = 10f64.powi(decimals);
        let rounded: Vec<f64> = values
            .iter()
            .map(|&v| (v * multiplier).round() / multiplier)
            .collect();
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(rounded.clone(), Some(col_name.clone()))?,
                )?;
            } else {
                if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                    df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                }
            }
        }
        Ok(df)
    }
    fn quantile(&self, column: &str, q: f64) -> Result<f64> {
        if q < 0.0 || q > 1.0 {
            return Err(Error::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }
        let values = self.get_column_numeric_values(column)?;
        if values.is_empty() {
            return Err(Error::Empty(
                "Cannot compute quantile of empty column".to_string(),
            ));
        }
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = sorted.len();
        let idx = q * (n - 1) as f64;
        let lower = idx.floor() as usize;
        let upper = idx.ceil() as usize;
        if lower == upper {
            Ok(sorted[lower])
        } else {
            let weight = idx - lower as f64;
            Ok(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }
    fn head(&self, n: usize) -> Result<DataFrame> {
        let n_rows = self.row_count();
        let take = n.min(n_rows);
        let indices: Vec<usize> = (0..take).collect();
        select_rows_by_indices(self, &indices)
    }
    fn tail(&self, n: usize) -> Result<DataFrame> {
        let n_rows = self.row_count();
        if n_rows == 0 {
            return Ok(DataFrame::new());
        }
        let start = if n >= n_rows { 0 } else { n_rows - n };
        let indices: Vec<usize> = (start..n_rows).collect();
        select_rows_by_indices(self, &indices)
    }
    fn unique(&self, column: &str) -> Result<Vec<String>> {
        let values = self.get_column_string_values(column)?;
        let unique_set: HashSet<String> = values.into_iter().collect();
        let mut result: Vec<String> = unique_set.into_iter().collect();
        result.sort();
        Ok(result)
    }
    fn unique_numeric(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let unique_set: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
        let mut result: Vec<f64> = unique_set.into_iter().map(f64::from_bits).collect();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(result)
    }
    fn fillna(&self, column: &str, value: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let filled: Vec<f64> = values
            .iter()
            .map(|&v| if v.is_nan() { value } else { v })
            .collect();
        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(filled.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(df)
    }

    fn fillna_method(&self, column: &str, method: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;

        let filled: Vec<f64> = match method {
            "ffill" | "forward" => {
                // Forward fill - propagate last valid observation forward
                let mut last_valid = f64::NAN;
                values
                    .iter()
                    .map(|&v| {
                        if !v.is_nan() {
                            last_valid = v;
                            v
                        } else if !last_valid.is_nan() {
                            last_valid
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            }
            "bfill" | "backward" => {
                // Backward fill - use next valid observation to fill gap
                let mut result = values.clone();
                let mut next_valid = f64::NAN;

                for i in (0..result.len()).rev() {
                    if !result[i].is_nan() {
                        next_valid = result[i];
                    } else if !next_valid.is_nan() {
                        result[i] = next_valid;
                    }
                }
                result
            }
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Invalid fill method: '{}'. Use 'ffill' or 'bfill'.",
                    method
                )));
            }
        };

        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(filled.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(df)
    }

    fn interpolate(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;

        // Linear interpolation
        let mut interpolated = values.clone();

        // Find first and last valid indices
        let first_valid = values.iter().position(|v| !v.is_nan());
        let last_valid = values.iter().rposition(|v| !v.is_nan());

        if let (Some(first), Some(last)) = (first_valid, last_valid) {
            let mut prev_valid_idx = first;
            let mut prev_valid_val = values[first];

            for i in (first + 1)..=last {
                if !values[i].is_nan() {
                    // Found next valid value, interpolate between prev and current
                    if i > prev_valid_idx + 1 {
                        // There are NaN values to interpolate
                        let gap_size = (i - prev_valid_idx) as f64;
                        let value_diff = values[i] - prev_valid_val;

                        for j in (prev_valid_idx + 1)..i {
                            let position = (j - prev_valid_idx) as f64;
                            interpolated[j] = prev_valid_val + (value_diff * position / gap_size);
                        }
                    }

                    prev_valid_idx = i;
                    prev_valid_val = values[i];
                }
            }
        }

        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(interpolated.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(df)
    }

    fn dropna(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let valid_indices: Vec<usize> = values
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v.is_nan())
            .map(|(i, _)| i)
            .collect();
        select_rows_by_indices(self, &valid_indices)
    }
    fn isna(&self, column: &str) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().map(|v| v.is_nan()).collect())
    }
    fn sum_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let sum: f64 = values.iter().filter(|v| !v.is_nan()).sum();
                results.push((col_name, sum));
            }
        }
        Ok(results)
    }
    fn mean_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid_values: Vec<f64> =
                    values.iter().filter(|v| !v.is_nan()).copied().collect();
                if !valid_values.is_empty() {
                    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                    results.push((col_name, mean));
                }
            }
        }
        Ok(results)
    }
    fn std_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid_values: Vec<f64> =
                    values.iter().filter(|v| !v.is_nan()).copied().collect();
                if valid_values.len() > 1 {
                    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                    let variance: f64 =
                        valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (valid_values.len() - 1) as f64;
                    results.push((col_name, variance.sqrt()));
                }
            }
        }
        Ok(results)
    }
    fn var_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid_values: Vec<f64> =
                    values.iter().filter(|v| !v.is_nan()).copied().collect();
                if valid_values.len() > 1 {
                    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                    let variance: f64 =
                        valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                            / (valid_values.len() - 1) as f64;
                    results.push((col_name, variance));
                }
            }
        }
        Ok(results)
    }
    fn min_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid_values: Vec<f64> =
                    values.iter().filter(|v| !v.is_nan()).copied().collect();
                if !valid_values.is_empty() {
                    let min = valid_values.iter().cloned().fold(f64::INFINITY, f64::min);
                    results.push((col_name, min));
                }
            }
        }
        Ok(results)
    }
    fn max_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid_values: Vec<f64> =
                    values.iter().filter(|v| !v.is_nan()).copied().collect();
                if !valid_values.is_empty() {
                    let max = valid_values
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    results.push((col_name, max));
                }
            }
        }
        Ok(results)
    }
    fn sort_values(&self, column: &str, ascending: bool) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mut indexed_values: Vec<(usize, f64)> =
            values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_values.sort_by(|a, b| {
            let cmp = a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
            if ascending {
                cmp
            } else {
                cmp.reverse()
            }
        });
        let sorted_indices: Vec<usize> = indexed_values.iter().map(|(i, _)| *i).collect();
        select_rows_by_indices(self, &sorted_indices)
    }
    fn sort_by_columns(&self, columns: &[&str], ascending: &[bool]) -> Result<DataFrame> {
        if columns.len() != ascending.len() {
            return Err(Error::InvalidValue(
                "Number of columns must match number of ascending flags".to_string(),
            ));
        }
        if columns.is_empty() {
            return Ok(self.clone());
        }
        let mut column_values: Vec<Vec<f64>> = Vec::new();
        for &col in columns {
            column_values.push(self.get_column_numeric_values(col)?);
        }
        let n_rows = self.row_count();
        let mut indices: Vec<usize> = (0..n_rows).collect();
        indices.sort_by(|&i, &j| {
            for (col_idx, (&col_name, &asc)) in columns.iter().zip(ascending.iter()).enumerate() {
                let vals = &column_values[col_idx];
                let vi = vals.get(i).copied().unwrap_or(f64::NAN);
                let vj = vals.get(j).copied().unwrap_or(f64::NAN);
                let cmp = vi.partial_cmp(&vj).unwrap_or(Ordering::Equal);
                let ord_cmp = if asc { cmp } else { cmp.reverse() };
                if ord_cmp != Ordering::Equal {
                    return ord_cmp;
                }
            }
            Ordering::Equal
        });
        select_rows_by_indices(self, &indices)
    }

    fn merge(
        &self,
        other: &DataFrame,
        on: &str,
        how: super::merge::JoinType,
        suffixes: (&str, &str),
    ) -> Result<DataFrame> {
        super::merge::merge(self, other, on, how, suffixes)
    }

    fn where_cond(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;

        if condition.len() != values.len() {
            return Err(Error::InvalidValue(
                "Condition length must match column length".to_string(),
            ));
        }

        let replaced: Vec<f64> = values
            .iter()
            .zip(condition.iter())
            .map(|(&v, &cond)| if cond { v } else { other })
            .collect();

        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(replaced.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(df)
    }

    fn mask(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;

        if condition.len() != values.len() {
            return Err(Error::InvalidValue(
                "Condition length must match column length".to_string(),
            ));
        }

        let replaced: Vec<f64> = values
            .iter()
            .zip(condition.iter())
            .map(|(&v, &cond)| if cond { other } else { v })
            .collect();

        let mut df = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                df.add_column(
                    col_name.clone(),
                    Series::new(replaced.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(df)
    }

    fn drop_duplicates(&self, subset: Option<&[&str]>, keep: &str) -> Result<DataFrame> {
        let columns_to_check: Vec<String> = match subset {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => self.column_names(),
        };

        // Validate columns exist
        for col in &columns_to_check {
            if !self.contains_column(col) {
                return Err(Error::InvalidValue(format!(
                    "Column '{}' not found in DataFrame",
                    col
                )));
            }
        }

        // Build row keys for comparison
        let n_rows = self.row_count();
        let mut row_keys: Vec<String> = Vec::with_capacity(n_rows);

        for row_idx in 0..n_rows {
            let mut key_parts: Vec<String> = Vec::new();
            for col in &columns_to_check {
                if let Ok(values) = self.get_column_string_values(col) {
                    key_parts.push(values.get(row_idx).cloned().unwrap_or_default());
                } else if let Ok(values) = self.get_column_numeric_values(col) {
                    let v = values.get(row_idx).copied().unwrap_or(f64::NAN);
                    key_parts.push(v.to_bits().to_string());
                }
            }
            row_keys.push(key_parts.join("|||"));
        }

        // Find indices to keep based on 'keep' parameter
        let mut seen: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, key) in row_keys.iter().enumerate() {
            seen.entry(key.clone()).or_insert_with(Vec::new).push(idx);
        }

        let mut indices_to_keep: Vec<usize> = Vec::new();
        match keep {
            "first" => {
                for (_, indices) in &seen {
                    if let Some(&first) = indices.first() {
                        indices_to_keep.push(first);
                    }
                }
            }
            "last" => {
                for (_, indices) in &seen {
                    if let Some(&last) = indices.last() {
                        indices_to_keep.push(last);
                    }
                }
            }
            "none" | "false" => {
                for (_, indices) in &seen {
                    if indices.len() == 1 {
                        indices_to_keep.push(indices[0]);
                    }
                }
            }
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                    keep
                )));
            }
        }

        // Sort indices to preserve original order
        indices_to_keep.sort_unstable();

        select_rows_by_indices(self, &indices_to_keep)
    }

    fn select_dtypes(&self, include: &[&str]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col_name in self.column_names() {
            let is_numeric = self.get_column_numeric_values(&col_name).is_ok();
            let is_string = !is_numeric && self.get_column_string_values(&col_name).is_ok();

            let should_include = include.iter().any(|&dtype| {
                (dtype == "numeric" || dtype == "number" || dtype == "float64" || dtype == "int64")
                    && is_numeric
                    || (dtype == "string" || dtype == "object" || dtype == "str") && is_string
            });

            if should_include {
                if let Ok(values) = self.get_column_numeric_values(&col_name) {
                    result.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(values) = self.get_column_string_values(&col_name) {
                    result.add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
                }
            }
        }

        Ok(result)
    }

    fn any_numeric(&self) -> Result<Vec<(String, bool)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let has_any = values.iter().any(|&v| !v.is_nan() && v != 0.0);
                results.push((col_name, has_any));
            }
        }
        Ok(results)
    }

    fn all_numeric(&self) -> Result<Vec<(String, bool)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let all_true = values.iter().all(|&v| !v.is_nan() && v != 0.0);
                results.push((col_name, all_true));
            }
        }
        Ok(results)
    }

    fn count_valid(&self) -> Result<Vec<(String, usize)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let count = values.iter().filter(|v| !v.is_nan()).count();
                results.push((col_name, count));
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                // For strings, count non-empty values
                let count = values.iter().filter(|v| !v.is_empty()).count();
                results.push((col_name, count));
            }
        }
        Ok(results)
    }

    fn reverse_columns(&self) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let columns = self.column_names();

        for col_name in columns.into_iter().rev() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }

        Ok(result)
    }

    fn reverse_rows(&self) -> Result<DataFrame> {
        let n_rows = self.row_count();
        let indices: Vec<usize> = (0..n_rows).rev().collect();
        select_rows_by_indices(self, &indices)
    }

    fn notna(&self, column: &str) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().map(|v| !v.is_nan()).collect())
    }

    fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: &str,
        value_name: &str,
    ) -> Result<DataFrame> {
        // Validate id_vars exist
        for id_var in id_vars {
            if !self.contains_column(id_var) {
                return Err(Error::InvalidValue(format!(
                    "Column '{}' not found in DataFrame",
                    id_var
                )));
            }
        }

        // Determine value columns
        let value_columns: Vec<String> = match value_vars {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => {
                let id_set: std::collections::HashSet<&str> = id_vars.iter().copied().collect();
                self.column_names()
                    .into_iter()
                    .filter(|c| !id_set.contains(c.as_str()))
                    .collect()
            }
        };

        if value_columns.is_empty() {
            return Err(Error::InvalidValue("No value columns to melt".to_string()));
        }

        let n_rows = self.row_count();
        let n_value_cols = value_columns.len();
        let total_rows = n_rows * n_value_cols;

        let mut result = DataFrame::new();

        // Add id columns (repeated for each value column)
        for id_var in id_vars {
            if let Ok(values) = self.get_column_numeric_values(id_var) {
                let mut repeated: Vec<f64> = Vec::with_capacity(total_rows);
                for _ in 0..n_value_cols {
                    repeated.extend(values.iter().copied());
                }
                result.add_column(
                    id_var.to_string(),
                    Series::new(repeated, Some(id_var.to_string()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(id_var) {
                let mut repeated: Vec<String> = Vec::with_capacity(total_rows);
                for _ in 0..n_value_cols {
                    repeated.extend(values.iter().cloned());
                }
                result.add_column(
                    id_var.to_string(),
                    Series::new(repeated, Some(id_var.to_string()))?,
                )?;
            }
        }

        // Add variable column
        let mut var_values: Vec<String> = Vec::with_capacity(total_rows);
        for col in &value_columns {
            for _ in 0..n_rows {
                var_values.push(col.clone());
            }
        }
        result.add_column(
            var_name.to_string(),
            Series::new(var_values, Some(var_name.to_string()))?,
        )?;

        // Add value column
        let mut all_values: Vec<f64> = Vec::with_capacity(total_rows);
        for col in &value_columns {
            if let Ok(values) = self.get_column_numeric_values(col) {
                all_values.extend(values);
            } else {
                // Non-numeric columns get NaN
                for _ in 0..n_rows {
                    all_values.push(f64::NAN);
                }
            }
        }
        result.add_column(
            value_name.to_string(),
            Series::new(all_values, Some(value_name.to_string()))?,
        )?;

        Ok(result)
    }

    fn explode(&self, column: &str, separator: &str) -> Result<DataFrame> {
        let string_values = self.get_column_string_values(column)?;
        let n_rows = self.row_count();

        // Split each value and calculate new row count
        let split_values: Vec<Vec<&str>> = string_values
            .iter()
            .map(|v| v.split(separator).map(|s| s.trim()).collect())
            .collect();

        let total_new_rows: usize = split_values.iter().map(|v| v.len().max(1)).sum();

        // Build mapping of old row index to new rows
        let mut row_mapping: Vec<(usize, &str)> = Vec::with_capacity(total_new_rows);
        for (row_idx, parts) in split_values.iter().enumerate() {
            if parts.is_empty() {
                row_mapping.push((row_idx, ""));
            } else {
                for part in parts {
                    row_mapping.push((row_idx, part));
                }
            }
        }

        let mut result = DataFrame::new();

        // Copy all columns, expanding as needed
        for col_name in self.column_names() {
            if &col_name == column {
                // This is the exploded column
                let new_values: Vec<String> =
                    row_mapping.iter().map(|(_, val)| val.to_string()).collect();
                result.add_column(
                    col_name.clone(),
                    Series::new(new_values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let new_values: Vec<f64> =
                    row_mapping.iter().map(|(idx, _)| values[*idx]).collect();
                result.add_column(
                    col_name.clone(),
                    Series::new(new_values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                let new_values: Vec<String> = row_mapping
                    .iter()
                    .map(|(idx, _)| values[*idx].clone())
                    .collect();
                result.add_column(
                    col_name.clone(),
                    Series::new(new_values, Some(col_name.clone()))?,
                )?;
            }
        }

        Ok(result)
    }

    fn duplicated(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
        let columns_to_check: Vec<String> = match subset {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => self.column_names(),
        };

        // Validate columns exist
        for col in &columns_to_check {
            if !self.contains_column(col) {
                return Err(Error::InvalidValue(format!(
                    "Column '{}' not found in DataFrame",
                    col
                )));
            }
        }

        let n_rows = self.row_count();
        let mut row_keys: Vec<String> = Vec::with_capacity(n_rows);

        for row_idx in 0..n_rows {
            let mut key_parts: Vec<String> = Vec::new();
            for col in &columns_to_check {
                if let Ok(values) = self.get_column_string_values(col) {
                    key_parts.push(values.get(row_idx).cloned().unwrap_or_default());
                } else if let Ok(values) = self.get_column_numeric_values(col) {
                    let v = values.get(row_idx).copied().unwrap_or(f64::NAN);
                    key_parts.push(v.to_bits().to_string());
                }
            }
            row_keys.push(key_parts.join("|||"));
        }

        // Track occurrences
        let mut first_occurrence: HashMap<String, usize> = HashMap::new();
        let mut last_occurrence: HashMap<String, usize> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for (idx, key) in row_keys.iter().enumerate() {
            first_occurrence.entry(key.clone()).or_insert(idx);
            last_occurrence.insert(key.clone(), idx);
            *counts.entry(key.clone()).or_insert(0) += 1;
        }

        // Build duplicate mask
        let mut is_duplicate = vec![false; n_rows];
        match keep {
            "first" => {
                for (idx, key) in row_keys.iter().enumerate() {
                    if first_occurrence.get(key) != Some(&idx) {
                        is_duplicate[idx] = true;
                    }
                }
            }
            "last" => {
                for (idx, key) in row_keys.iter().enumerate() {
                    if last_occurrence.get(key) != Some(&idx) {
                        is_duplicate[idx] = true;
                    }
                }
            }
            "none" | "false" => {
                for (idx, key) in row_keys.iter().enumerate() {
                    if counts.get(key).copied().unwrap_or(0) > 1 {
                        is_duplicate[idx] = true;
                    }
                }
            }
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                    keep
                )));
            }
        }

        Ok(is_duplicate)
    }

    fn copy(&self) -> DataFrame {
        self.clone()
    }

    fn to_dict(&self) -> Result<HashMap<String, Vec<String>>> {
        let mut result = HashMap::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_string_values(&col_name) {
                result.insert(col_name, values);
            } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result.insert(col_name, values.iter().map(|v| v.to_string()).collect());
            }
        }
        Ok(result)
    }

    fn first_valid_index(&self, column: &str) -> Result<Option<usize>> {
        let values = self.get_column_numeric_values(column)?;
        for (idx, v) in values.iter().enumerate() {
            if !v.is_nan() {
                return Ok(Some(idx));
            }
        }
        Ok(None)
    }

    fn last_valid_index(&self, column: &str) -> Result<Option<usize>> {
        let values = self.get_column_numeric_values(column)?;
        for (idx, v) in values.iter().enumerate().rev() {
            if !v.is_nan() {
                return Ok(Some(idx));
            }
        }
        Ok(None)
    }

    fn product_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
                if !valid.is_empty() {
                    let product = valid.iter().fold(1.0, |acc, &x| acc * x);
                    results.push((col_name, product));
                }
            }
        }
        Ok(results)
    }

    fn median_all(&self) -> Result<Vec<(String, f64)>> {
        let mut results = Vec::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let mut valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
                if !valid.is_empty() {
                    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                    let mid = valid.len() / 2;
                    let median = if valid.len() % 2 == 0 {
                        (valid[mid - 1] + valid[mid]) / 2.0
                    } else {
                        valid[mid]
                    };
                    results.push((col_name, median));
                }
            }
        }
        Ok(results)
    }

    fn skew(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.len() < 3 {
            return Ok(f64::NAN);
        }

        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(f64::NAN);
        }

        let m3 = valid.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
        let skewness = m3 / std_dev.powi(3);

        // Adjust for sample bias (Fisher's formula)
        let adjustment = ((n * (n - 1.0)).sqrt()) / (n - 2.0);
        Ok(skewness * adjustment)
    }

    fn kurtosis(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.len() < 4 {
            return Ok(f64::NAN);
        }

        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(f64::NAN);
        }

        let m4 = valid.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
        let kurtosis = m4 / std_dev.powi(4) - 3.0; // Excess kurtosis

        // Adjust for sample bias
        let adjustment = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurtosis + 6.0);
        Ok(adjustment)
    }

    fn add_prefix(&self, prefix: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            let new_name = format!("{}{}", prefix, col_name);
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            }
        }
        Ok(result)
    }

    fn add_suffix(&self, suffix: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            let new_name = format!("{}{}", col_name, suffix);
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            }
        }
        Ok(result)
    }

    fn filter_by_mask(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.row_count() {
            return Err(Error::InvalidValue(
                "Mask length must match number of rows".to_string(),
            ));
        }

        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();

        select_rows_by_indices(self, &indices)
    }

    fn mode_numeric(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let mut counts: HashMap<u64, usize> = HashMap::new();

        for v in values.iter().filter(|v| !v.is_nan()) {
            *counts.entry(v.to_bits()).or_insert(0) += 1;
        }

        if counts.is_empty() {
            return Ok(vec![]);
        }

        let max_count = *counts.values().max().unwrap();
        let mut modes: Vec<f64> = counts
            .iter()
            .filter(|(_, &c)| c == max_count)
            .map(|(&bits, _)| f64::from_bits(bits))
            .collect();

        modes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(modes)
    }

    fn mode_string(&self, column: &str) -> Result<Vec<String>> {
        let values = self.get_column_string_values(column)?;
        let mut counts: HashMap<String, usize> = HashMap::new();

        for v in values.iter().filter(|v| !v.is_empty()) {
            *counts.entry(v.clone()).or_insert(0) += 1;
        }

        if counts.is_empty() {
            return Ok(vec![]);
        }

        let max_count = *counts.values().max().unwrap();
        let mut modes: Vec<String> = counts
            .iter()
            .filter(|(_, &c)| c == max_count)
            .map(|(k, _)| k.clone())
            .collect();

        modes.sort();
        Ok(modes)
    }

    fn percentile(&self, column: &str, n: f64) -> Result<f64> {
        // Convert percentile (0-100) to quantile (0-1)
        self.quantile(column, n / 100.0)
    }

    fn ewma(&self, column: &str, span: usize) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;

        if span < 1 {
            return Err(Error::InvalidValue("Span must be at least 1".to_string()));
        }

        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Vec::with_capacity(values.len());

        let mut ewma_value: Option<f64> = None;

        for &v in &values {
            if v.is_nan() {
                result.push(f64::NAN);
            } else {
                ewma_value = Some(match ewma_value {
                    Some(prev) => alpha * v + (1.0 - alpha) * prev,
                    None => v,
                });
                result.push(ewma_value.unwrap());
            }
        }

        Ok(result)
    }

    fn iloc(&self, index: usize) -> Result<HashMap<String, String>> {
        if index >= self.row_count() {
            return Err(Error::InvalidValue(format!(
                "Index {} out of bounds for DataFrame with {} rows",
                index,
                self.row_count()
            )));
        }

        let mut result = HashMap::new();
        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_string_values(&col_name) {
                result.insert(col_name, values[index].clone());
            } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result.insert(col_name, values[index].to_string());
            }
        }
        Ok(result)
    }

    fn iloc_range(&self, start: usize, end: usize) -> Result<DataFrame> {
        if start > end {
            return Err(Error::InvalidValue(
                "Start index must be less than or equal to end index".to_string(),
            ));
        }

        let n_rows = self.row_count();
        let end = end.min(n_rows);
        let start = start.min(n_rows);

        let indices: Vec<usize> = (start..end).collect();
        select_rows_by_indices(self, &indices)
    }

    fn info(&self) -> String {
        let mut info = String::new();
        let n_rows = self.row_count();
        let columns = self.column_names();
        let n_cols = columns.len();

        info.push_str(&format!("<DataFrame>\n"));
        info.push_str(&format!(
            "RangeIndex: {} entries, 0 to {}\n",
            n_rows,
            n_rows.saturating_sub(1)
        ));
        info.push_str(&format!("Data columns (total {} columns):\n", n_cols));
        info.push_str(&format!(" #   Column  Non-Null Count  Dtype\n"));
        info.push_str(&format!("---  ------  --------------  -----\n"));

        for (idx, col) in columns.iter().enumerate() {
            let (non_null, dtype) = if let Ok(values) = self.get_column_numeric_values(col) {
                let non_null = values.iter().filter(|v| !v.is_nan()).count();
                (non_null, "float64")
            } else if let Ok(values) = self.get_column_string_values(col) {
                let non_null = values.iter().filter(|v| !v.is_empty()).count();
                (non_null, "object")
            } else {
                (0, "unknown")
            };
            info.push_str(&format!(
                " {}   {}  {} non-null  {}\n",
                idx, col, non_null, dtype
            ));
        }

        info.push_str(&format!(
            "dtypes: float64({}), object({})\n",
            columns
                .iter()
                .filter(|c| self.get_column_numeric_values(c).is_ok())
                .count(),
            columns
                .iter()
                .filter(|c| self.get_column_string_values(c).is_ok()
                    && self.get_column_numeric_values(c).is_err())
                .count()
        ));
        info.push_str(&format!("memory usage: {} bytes\n", self.memory_usage()));

        info
    }

    fn equals(&self, other: &DataFrame) -> bool {
        if self.row_count() != other.row_count() {
            return false;
        }

        let cols1 = self.column_names();
        let cols2 = other.column_names();
        if cols1 != cols2 {
            return false;
        }

        for col in &cols1 {
            if let (Ok(v1), Ok(v2)) = (
                self.get_column_numeric_values(col),
                other.get_column_numeric_values(col),
            ) {
                for (a, b) in v1.iter().zip(v2.iter()) {
                    if a.is_nan() && b.is_nan() {
                        continue;
                    }
                    if (a - b).abs() > f64::EPSILON {
                        return false;
                    }
                }
            } else if let (Ok(v1), Ok(v2)) = (
                self.get_column_string_values(col),
                other.get_column_string_values(col),
            ) {
                if v1 != v2 {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }

    fn compare(&self, other: &DataFrame) -> Result<DataFrame> {
        if self.row_count() != other.row_count() {
            return Err(Error::InvalidValue(
                "DataFrames must have the same number of rows".to_string(),
            ));
        }

        let mut result = DataFrame::new();
        let n_rows = self.row_count();

        // Find common columns
        let cols1: std::collections::HashSet<_> = self.column_names().into_iter().collect();
        let cols2: std::collections::HashSet<_> = other.column_names().into_iter().collect();
        let common_cols: Vec<_> = cols1.intersection(&cols2).cloned().collect();

        for col in &common_cols {
            if let (Ok(v1), Ok(v2)) = (
                self.get_column_numeric_values(col),
                other.get_column_numeric_values(col),
            ) {
                let diff: Vec<f64> = v1
                    .iter()
                    .zip(v2.iter())
                    .map(|(a, b)| {
                        if a.is_nan() && b.is_nan() {
                            0.0
                        } else if a.is_nan() || b.is_nan() {
                            f64::NAN
                        } else {
                            a - b
                        }
                    })
                    .collect();
                result.add_column(
                    format!("{}_diff", col),
                    Series::new(diff, Some(format!("{}_diff", col)))?,
                )?;
            }
        }

        Ok(result)
    }

    fn keys(&self) -> Vec<String> {
        self.column_names()
    }

    fn pop_column(&self, column: &str) -> Result<(DataFrame, Vec<f64>)> {
        let values = self.get_column_numeric_values(column)?;
        let new_df = self.drop_columns(&[column])?;
        Ok((new_df, values))
    }

    fn insert_column(&self, loc: usize, name: &str, values: Vec<f64>) -> Result<DataFrame> {
        if values.len() != self.row_count() {
            return Err(Error::InvalidValue(
                "Column length must match DataFrame row count".to_string(),
            ));
        }

        let columns = self.column_names();
        let loc = loc.min(columns.len());

        let mut result = DataFrame::new();

        // Add columns before insertion point
        for col in columns.iter().take(loc) {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        // Add new column
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;

        // Add columns after insertion point
        for col in columns.iter().skip(loc) {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn rolling_sum(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_sum(self, column, window, min_periods)
    }

    fn rolling_mean(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_mean(self, column, window, min_periods)
    }

    fn rolling_std(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_std(self, column, window, min_periods)
    }

    fn rolling_min(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_min(self, column, window, min_periods)
    }

    fn rolling_max(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_max(self, column, window, min_periods)
    }

    fn rolling_var(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_var(self, column, window, min_periods)
    }

    fn rolling_median(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>> {
        window_ops::rolling_median(self, column, window, min_periods)
    }

    fn rolling_count(&self, column: &str, window: usize) -> Result<Vec<usize>> {
        window_ops::rolling_count(self, column, window)
    }

    fn rolling_apply<F>(
        &self,
        column: &str,
        window: usize,
        func: F,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        window_ops::rolling_apply(self, column, window, func, min_periods)
    }

    fn cumcount(&self, column: &str) -> Result<Vec<usize>> {
        let values = self.get_column_numeric_values(column)?;
        let mut count = 0usize;
        let mut result = Vec::with_capacity(values.len());

        for v in &values {
            if !v.is_nan() {
                count += 1;
            }
            result.push(count);
        }

        Ok(result)
    }

    fn nth(&self, n: i32) -> Result<HashMap<String, String>> {
        let n_rows = self.row_count() as i32;
        let actual_index = if n >= 0 {
            n as usize
        } else {
            (n_rows + n) as usize
        };

        if actual_index >= self.row_count() {
            return Err(Error::InvalidValue(format!(
                "Index {} out of bounds for DataFrame with {} rows",
                n,
                self.row_count()
            )));
        }

        self.iloc(actual_index)
    }

    fn transform<F>(&self, column: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64,
    {
        let values = self.get_column_numeric_values(column)?;
        let transformed: Vec<f64> = values.iter().map(|&v| func(v)).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(transformed.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(result)
    }

    fn crosstab(&self, col1: &str, col2: &str) -> Result<DataFrame> {
        let values1 = self.get_column_string_values(col1)?;
        let values2 = self.get_column_string_values(col2)?;

        // Get unique values for both columns
        let mut unique1: Vec<String> = values1
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let mut unique2: Vec<String> = values2
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique1.sort();
        unique2.sort();

        // Count occurrences
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for (v1, v2) in values1.iter().zip(values2.iter()) {
            *counts.entry((v1.clone(), v2.clone())).or_insert(0) += 1;
        }

        // Build result DataFrame
        let mut result = DataFrame::new();

        // Add index column (col1 values)
        result.add_column(
            col1.to_string(),
            Series::new(unique1.clone(), Some(col1.to_string()))?,
        )?;

        // Add count columns for each unique value of col2
        for u2 in &unique2 {
            let column_counts: Vec<f64> = unique1
                .iter()
                .map(|u1| counts.get(&(u1.clone(), u2.clone())).copied().unwrap_or(0) as f64)
                .collect();
            result.add_column(u2.clone(), Series::new(column_counts, Some(u2.clone()))?)?;
        }

        Ok(result)
    }

    fn expanding_sum(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_sum(self, column, min_periods)
    }

    fn expanding_mean(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_mean(self, column, min_periods)
    }

    fn expanding_std(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_std(self, column, min_periods)
    }

    fn expanding_min(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_min(self, column, min_periods)
    }

    fn expanding_max(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_max(self, column, min_periods)
    }

    fn expanding_var(&self, column: &str, min_periods: usize) -> Result<Vec<f64>> {
        window_ops::expanding_var(self, column, min_periods)
    }

    fn expanding_apply<F>(&self, column: &str, func: F, min_periods: usize) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        window_ops::expanding_apply(self, column, func, min_periods)
    }

    fn align(&self, other: &DataFrame) -> Result<(DataFrame, DataFrame)> {
        let cols1: std::collections::HashSet<_> = self.column_names().into_iter().collect();
        let cols2: std::collections::HashSet<_> = other.column_names().into_iter().collect();
        let all_cols: Vec<_> = cols1.union(&cols2).cloned().collect();

        let mut result1 = DataFrame::new();
        let mut result2 = DataFrame::new();

        for col in &all_cols {
            // For self
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else {
                let nan_vals: Vec<f64> = vec![f64::NAN; self.row_count()];
                result1.add_column(col.clone(), Series::new(nan_vals, Some(col.clone()))?)?;
            }

            // For other
            if let Ok(vals) = other.get_column_numeric_values(col) {
                result2.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = other.get_column_string_values(col) {
                result2.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else {
                let nan_vals: Vec<f64> = vec![f64::NAN; other.row_count()];
                result2.add_column(col.clone(), Series::new(nan_vals, Some(col.clone()))?)?;
            }
        }

        Ok((result1, result2))
    }

    fn reindex_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for &col in columns {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
            } else {
                // Column doesn't exist, fill with NaN
                let nan_vals: Vec<f64> = vec![f64::NAN; self.row_count()];
                result.add_column(
                    col.to_string(),
                    Series::new(nan_vals, Some(col.to_string()))?,
                )?;
            }
        }

        Ok(result)
    }

    fn value_range(&self, column: &str) -> Result<(f64, f64)> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.is_empty() {
            return Err(Error::InvalidValue("No valid values in column".to_string()));
        }

        let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok((min, max))
    }

    fn zscore(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.len() < 2 {
            return Err(Error::InvalidValue(
                "Need at least 2 values for z-score".to_string(),
            ));
        }

        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let std_dev = (valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        if std_dev == 0.0 {
            return Err(Error::InvalidValue(
                "Standard deviation is zero".to_string(),
            ));
        }

        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    f64::NAN
                } else {
                    (v - mean) / std_dev
                }
            })
            .collect())
    }

    fn normalize(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let (min, max) = self.value_range(column)?;

        let range = max - min;
        if range == 0.0 {
            return Err(Error::InvalidValue(
                "Range is zero, cannot normalize".to_string(),
            ));
        }

        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    f64::NAN
                } else {
                    (v - min) / range
                }
            })
            .collect())
    }

    fn cut(&self, column: &str, bins: usize) -> Result<Vec<String>> {
        if bins == 0 {
            return Err(Error::InvalidValue(
                "Number of bins must be > 0".to_string(),
            ));
        }

        let values = self.get_column_numeric_values(column)?;
        let (min, max) = self.value_range(column)?;

        let bin_width = (max - min) / bins as f64;
        let mut edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();
        edges[bins] = max + 0.001; // Include max value

        let mut result = Vec::with_capacity(values.len());
        for v in &values {
            if v.is_nan() {
                result.push("NaN".to_string());
            } else {
                for i in 0..bins {
                    if *v >= edges[i] && *v < edges[i + 1] {
                        result.push(format!("({:.2}, {:.2}]", edges[i], edges[i + 1]));
                        break;
                    }
                }
            }
        }

        Ok(result)
    }

    fn qcut(&self, column: &str, q: usize) -> Result<Vec<String>> {
        if q == 0 {
            return Err(Error::InvalidValue(
                "Number of quantiles must be > 0".to_string(),
            ));
        }

        let values = self.get_column_numeric_values(column)?;
        let mut valid: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, &v)| (i, v))
            .collect();

        if valid.is_empty() {
            return Err(Error::InvalidValue("No valid values".to_string()));
        }

        valid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Calculate quantile edges
        let mut edges: Vec<f64> = Vec::with_capacity(q + 1);
        for i in 0..=q {
            let idx = (valid.len() as f64 * i as f64 / q as f64) as usize;
            let idx = idx.min(valid.len() - 1);
            edges.push(valid[idx].1);
        }

        let mut result = vec!["".to_string(); values.len()];
        for (orig_idx, v) in values.iter().enumerate() {
            if v.is_nan() {
                result[orig_idx] = "NaN".to_string();
            } else {
                for i in 0..q {
                    let lower = edges[i];
                    let upper = if i == q - 1 {
                        edges[i + 1] + 0.001
                    } else {
                        edges[i + 1]
                    };
                    if *v >= lower && *v < upper {
                        result[orig_idx] = format!("Q{}", i + 1);
                        break;
                    }
                }
            }
        }

        Ok(result)
    }

    fn stack(&self, columns: Option<&[&str]>) -> Result<DataFrame> {
        let cols_to_stack: Vec<String> = if let Some(cols) = columns {
            cols.iter().map(|s| s.to_string()).collect()
        } else {
            // Stack all numeric columns
            self.column_names()
                .into_iter()
                .filter(|c| self.get_column_numeric_values(c).is_ok())
                .collect()
        };

        if cols_to_stack.is_empty() {
            return Err(Error::InvalidValue("No columns to stack".to_string()));
        }

        let n_rows = self.row_count();
        let mut row_indices: Vec<f64> = Vec::with_capacity(n_rows * cols_to_stack.len());
        let mut variables: Vec<String> = Vec::with_capacity(n_rows * cols_to_stack.len());
        let mut values: Vec<f64> = Vec::with_capacity(n_rows * cols_to_stack.len());

        for row_idx in 0..n_rows {
            for col_name in &cols_to_stack {
                row_indices.push(row_idx as f64);
                variables.push(col_name.clone());
                if let Ok(col_values) = self.get_column_numeric_values(col_name) {
                    values.push(col_values[row_idx]);
                } else {
                    values.push(f64::NAN);
                }
            }
        }

        let mut result = DataFrame::new();
        result.add_column(
            "row_index".to_string(),
            Series::new(row_indices, Some("row_index".to_string()))?,
        )?;
        result.add_column(
            "variable".to_string(),
            Series::new(variables, Some("variable".to_string()))?,
        )?;
        result.add_column(
            "value".to_string(),
            Series::new(values, Some("value".to_string()))?,
        )?;

        Ok(result)
    }

    fn unstack(&self, index_col: &str, columns_col: &str, values_col: &str) -> Result<DataFrame> {
        let index_values = self.get_column_string_values(index_col)?;
        let column_values = self.get_column_string_values(columns_col)?;
        let data_values = self.get_column_numeric_values(values_col)?;

        // Get unique indices and columns
        let mut unique_indices: Vec<String> = index_values
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let mut unique_cols: Vec<String> = column_values
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_indices.sort();
        unique_cols.sort();

        // Create lookup map
        let mut data_map: HashMap<(String, String), f64> = HashMap::new();
        for i in 0..index_values.len() {
            data_map.insert(
                (index_values[i].clone(), column_values[i].clone()),
                data_values[i],
            );
        }

        // Build result DataFrame
        let mut result = DataFrame::new();
        result.add_column(
            index_col.to_string(),
            Series::new(unique_indices.clone(), Some(index_col.to_string()))?,
        )?;

        for col in &unique_cols {
            let col_data: Vec<f64> = unique_indices
                .iter()
                .map(|idx| {
                    data_map
                        .get(&(idx.clone(), col.clone()))
                        .copied()
                        .unwrap_or(f64::NAN)
                })
                .collect();
            result.add_column(col.clone(), Series::new(col_data, Some(col.clone()))?)?;
        }

        Ok(result)
    }

    fn pivot(&self, index: &str, columns: &str, values: &str) -> Result<DataFrame> {
        // Pivot is essentially the same as unstack
        self.unstack(index, columns, values)
    }

    fn astype(&self, column: &str, dtype: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col_name in self.column_names() {
            if &col_name == column {
                match dtype.to_lowercase().as_str() {
                    "float64" | "float" | "f64" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            result.add_column(
                                col_name.clone(),
                                Series::new(values, Some(col_name.clone()))?,
                            )?;
                        } else if let Ok(values) = self.get_column_string_values(&col_name) {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
                                .collect();
                            result.add_column(
                                col_name.clone(),
                                Series::new(converted, Some(col_name.clone()))?,
                            )?;
                        }
                    }
                    "int64" | "int" | "i64" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            let converted: Vec<f64> = values.iter().map(|v| v.floor()).collect();
                            result.add_column(
                                col_name.clone(),
                                Series::new(converted, Some(col_name.clone()))?,
                            )?;
                        } else if let Ok(values) = self.get_column_string_values(&col_name) {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|s| s.parse::<i64>().map(|i| i as f64).unwrap_or(f64::NAN))
                                .collect();
                            result.add_column(
                                col_name.clone(),
                                Series::new(converted, Some(col_name.clone()))?,
                            )?;
                        }
                    }
                    "string" | "str" | "object" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            let converted: Vec<String> = values
                                .iter()
                                .map(|v| {
                                    if v.is_nan() {
                                        "NaN".to_string()
                                    } else {
                                        v.to_string()
                                    }
                                })
                                .collect();
                            result.add_column(
                                col_name.clone(),
                                Series::new(converted, Some(col_name.clone()))?,
                            )?;
                        } else if let Ok(values) = self.get_column_string_values(&col_name) {
                            result.add_column(
                                col_name.clone(),
                                Series::new(values, Some(col_name.clone()))?,
                            )?;
                        }
                    }
                    "bool" | "boolean" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|v| if *v != 0.0 && !v.is_nan() { 1.0 } else { 0.0 })
                                .collect();
                            result.add_column(
                                col_name.clone(),
                                Series::new(converted, Some(col_name.clone()))?,
                            )?;
                        }
                    }
                    _ => return Err(Error::InvalidValue(format!("Unknown dtype: {}", dtype))),
                }
            } else {
                // Copy other columns unchanged
                if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                    result
                        .add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    result
                        .add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
                }
            }
        }

        Ok(result)
    }

    fn applymap<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64,
    {
        let mut result = DataFrame::new();

        for col_name in self.column_names() {
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let transformed: Vec<f64> = values.iter().map(|&v| func(v)).collect();
                result.add_column(
                    col_name.clone(),
                    Series::new(transformed, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                // String columns pass through unchanged
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }

        Ok(result)
    }

    fn agg(&self, column: &str, funcs: &[&str]) -> Result<HashMap<String, f64>> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        let mut results = HashMap::new();

        for func in funcs {
            let value = match func.to_lowercase().as_str() {
                "sum" => valid.iter().sum(),
                "mean" => {
                    if valid.is_empty() {
                        f64::NAN
                    } else {
                        valid.iter().sum::<f64>() / valid.len() as f64
                    }
                }
                "min" => {
                    if valid.is_empty() {
                        f64::NAN
                    } else {
                        valid.iter().cloned().fold(f64::INFINITY, f64::min)
                    }
                }
                "max" => {
                    if valid.is_empty() {
                        f64::NAN
                    } else {
                        valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    }
                }
                "std" => {
                    if valid.len() < 2 {
                        f64::NAN
                    } else {
                        let n = valid.len() as f64;
                        let mean = valid.iter().sum::<f64>() / n;
                        let variance =
                            valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
                        variance.sqrt()
                    }
                }
                "var" => {
                    if valid.len() < 2 {
                        f64::NAN
                    } else {
                        let n = valid.len() as f64;
                        let mean = valid.iter().sum::<f64>() / n;
                        valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
                    }
                }
                "count" => valid.len() as f64,
                "first" => *valid.first().unwrap_or(&f64::NAN),
                "last" => *valid.last().unwrap_or(&f64::NAN),
                "median" => {
                    if valid.is_empty() {
                        f64::NAN
                    } else {
                        let mut sorted = valid.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unknown aggregation function: {}",
                        func
                    )))
                }
            };
            results.insert(func.to_string(), value);
        }

        Ok(results)
    }

    fn dtypes(&self) -> Vec<(String, String)> {
        self.column_names()
            .into_iter()
            .map(|col| {
                let dtype = if self.get_column_numeric_values(&col).is_ok() {
                    "float64".to_string()
                } else if self.get_column_string_values(&col).is_ok() {
                    "object".to_string()
                } else {
                    "unknown".to_string()
                };
                (col, dtype)
            })
            .collect()
    }

    fn set_values(&self, column: &str, indices: &[usize], values: &[f64]) -> Result<DataFrame> {
        if indices.len() != values.len() {
            return Err(Error::InvalidValue(
                "Indices and values must have the same length".to_string(),
            ));
        }

        let mut col_values = self.get_column_numeric_values(column)?;

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= col_values.len() {
                return Err(Error::InvalidValue(format!("Index {} out of bounds", idx)));
            }
            col_values[idx] = values[i];
        }

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(col_values.clone(), Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn query_eq(&self, column: &str, value: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mask: Vec<bool> = values
            .iter()
            .map(|&v| (v - value).abs() < f64::EPSILON)
            .collect();
        self.filter_by_mask(&mask)
    }

    fn query_gt(&self, column: &str, value: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mask: Vec<bool> = values.iter().map(|&v| v > value).collect();
        self.filter_by_mask(&mask)
    }

    fn query_lt(&self, column: &str, value: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mask: Vec<bool> = values.iter().map(|&v| v < value).collect();
        self.filter_by_mask(&mask)
    }

    fn query_contains(&self, column: &str, pattern: &str) -> Result<DataFrame> {
        let values = self.get_column_string_values(column)?;
        let mask: Vec<bool> = values.iter().map(|v| v.contains(pattern)).collect();
        self.filter_by_mask(&mask)
    }

    fn select_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for &col in columns {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
            } else {
                return Err(Error::InvalidValue(format!("Column '{}' not found", col)));
            }
        }

        Ok(result)
    }

    fn add_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        self.transform(column, |x| x + value)
    }

    fn mul_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        self.transform(column, |x| x * value)
    }

    fn sub_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        self.transform(column, |x| x - value)
    }

    fn div_scalar(&self, column: &str, value: f64) -> Result<DataFrame> {
        self.transform(column, |x| x / value)
    }

    fn pow(&self, column: &str, exponent: f64) -> Result<DataFrame> {
        self.transform(column, |x| x.powf(exponent))
    }

    fn sqrt(&self, column: &str) -> Result<DataFrame> {
        self.transform(column, |x| x.sqrt())
    }

    fn log(&self, column: &str) -> Result<DataFrame> {
        self.transform(column, |x| x.ln())
    }

    fn exp(&self, column: &str) -> Result<DataFrame> {
        self.transform(column, |x| x.exp())
    }

    fn col_add(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;

        if v1.len() != v2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a + b).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;

        Ok(result)
    }

    fn col_mul(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;

        if v1.len() != v2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;

        Ok(result)
    }

    fn col_sub(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;

        if v1.len() != v2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a - b).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;

        Ok(result)
    }

    fn col_div(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;

        if v1.len() != v2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a / b).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;

        Ok(result)
    }

    // === Additional pandas-compatible methods implementations ===

    fn iterrows(&self) -> Vec<(usize, HashMap<String, SeriesValue>)> {
        let mut result = Vec::new();
        let columns = self.column_names();

        for row_idx in 0..self.row_count() {
            let mut row_data = HashMap::new();
            for col in &columns {
                if let Ok(vals) = self.get_column_numeric_values(col) {
                    if row_idx < vals.len() {
                        row_data.insert(col.clone(), SeriesValue::Float(vals[row_idx]));
                    }
                } else if let Ok(vals) = self.get_column_string_values(col) {
                    if row_idx < vals.len() {
                        row_data.insert(col.clone(), SeriesValue::String(vals[row_idx].clone()));
                    }
                }
            }
            result.push((row_idx, row_data));
        }
        result
    }

    fn at(&self, row: usize, column: &str) -> Result<SeriesValue> {
        if row >= self.row_count() {
            return Err(Error::InvalidValue(format!(
                "Row index {} out of bounds",
                row
            )));
        }

        if let Ok(vals) = self.get_column_numeric_values(column) {
            if row < vals.len() {
                return Ok(SeriesValue::Float(vals[row]));
            }
        } else if let Ok(vals) = self.get_column_string_values(column) {
            if row < vals.len() {
                return Ok(SeriesValue::String(vals[row].clone()));
            }
        }

        Err(Error::ColumnNotFound(column.to_string()))
    }

    fn iat(&self, row: usize, col_idx: usize) -> Result<SeriesValue> {
        let columns = self.column_names();
        if col_idx >= columns.len() {
            return Err(Error::InvalidValue(format!(
                "Column index {} out of bounds",
                col_idx
            )));
        }
        self.at(row, &columns[col_idx])
    }

    fn drop_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        let indices_set: std::collections::HashSet<usize> = indices.iter().cloned().collect();
        let mut result = DataFrame::new();

        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                let filtered: Vec<f64> = vals
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !indices_set.contains(i))
                    .map(|(_, v)| *v)
                    .collect();
                result.add_column(col.clone(), Series::new(filtered, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                let filtered: Vec<String> = vals
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !indices_set.contains(i))
                    .map(|(_, v)| v.clone())
                    .collect();
                result.add_column(col.clone(), Series::new(filtered, Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn set_index(&self, column: &str, drop: bool) -> Result<(DataFrame, Vec<String>)> {
        let index_values = self.get_column_string_values(column)?;

        let mut result = DataFrame::new();
        for col in self.column_names() {
            if drop && col == column {
                continue;
            }
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        Ok((result, index_values))
    }

    fn reset_index(&self, index_values: Option<&[String]>, name: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        // Add index as column if provided
        if let Some(idx_vals) = index_values {
            result.add_column(
                name.to_string(),
                Series::new(idx_vals.to_vec(), Some(name.to_string()))?,
            )?;
        }

        // Copy all existing columns
        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn to_records(&self) -> Vec<HashMap<String, SeriesValue>> {
        self.iterrows().into_iter().map(|(_, row)| row).collect()
    }

    fn items(&self) -> Vec<(String, Vec<SeriesValue>)> {
        let mut result = Vec::new();

        for col in self.column_names() {
            let mut values = Vec::new();
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                values = vals.iter().map(|v| SeriesValue::Float(*v)).collect();
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                values = vals
                    .iter()
                    .map(|v| SeriesValue::String(v.clone()))
                    .collect();
            }
            result.push((col, values));
        }

        result
    }

    fn update(&self, other: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col in self.column_names() {
            if other.contains_column(&col) {
                // Use values from other if available
                if let Ok(other_vals) = other.get_column_numeric_values(&col) {
                    if let Ok(self_vals) = self.get_column_numeric_values(&col) {
                        let updated: Vec<f64> = self_vals
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| {
                                if i < other_vals.len() && !other_vals[i].is_nan() {
                                    other_vals[i]
                                } else {
                                    v
                                }
                            })
                            .collect();
                        result.add_column(col.clone(), Series::new(updated, Some(col.clone()))?)?;
                    }
                } else if let Ok(other_vals) = other.get_column_string_values(&col) {
                    if let Ok(self_vals) = self.get_column_string_values(&col) {
                        let updated: Vec<String> = self_vals
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                if i < other_vals.len() && !other_vals[i].is_empty() {
                                    other_vals[i].clone()
                                } else {
                                    v.clone()
                                }
                            })
                            .collect();
                        result.add_column(col.clone(), Series::new(updated, Some(col.clone()))?)?;
                    }
                }
            } else {
                // Keep original values
                if let Ok(vals) = self.get_column_numeric_values(&col) {
                    result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
                } else if let Ok(vals) = self.get_column_string_values(&col) {
                    result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
                }
            }
        }

        Ok(result)
    }

    fn combine<F>(&self, other: &DataFrame, func: F) -> Result<DataFrame>
    where
        F: Fn(Option<f64>, Option<f64>) -> f64,
    {
        let mut result = DataFrame::new();

        // Get all unique columns from both DataFrames
        let mut all_cols: Vec<String> = self.column_names();
        for col in other.column_names() {
            if !all_cols.contains(&col) {
                all_cols.push(col);
            }
        }

        let max_rows = std::cmp::max(self.row_count(), other.row_count());

        for col in all_cols {
            let self_vals = self.get_column_numeric_values(&col).ok();
            let other_vals = other.get_column_numeric_values(&col).ok();

            let combined: Vec<f64> = (0..max_rows)
                .map(|i| {
                    let v1 = self_vals.as_ref().and_then(|v| v.get(i).copied());
                    let v2 = other_vals.as_ref().and_then(|v| v.get(i).copied());
                    func(v1, v2)
                })
                .collect();

            result.add_column(col.clone(), Series::new(combined, Some(col.clone()))?)?;
        }

        Ok(result)
    }

    fn shape(&self) -> (usize, usize) {
        (self.row_count(), self.column_names().len())
    }

    fn size(&self) -> usize {
        self.row_count() * self.column_names().len()
    }

    fn empty(&self) -> bool {
        self.row_count() == 0 || self.column_names().is_empty()
    }

    fn first_row(&self) -> Result<HashMap<String, SeriesValue>> {
        if self.row_count() == 0 {
            return Err(Error::InvalidValue("DataFrame is empty".to_string()));
        }
        let rows = self.iterrows();
        Ok(rows.into_iter().next().unwrap().1)
    }

    fn last_row(&self) -> Result<HashMap<String, SeriesValue>> {
        if self.row_count() == 0 {
            return Err(Error::InvalidValue("DataFrame is empty".to_string()));
        }
        let rows = self.iterrows();
        Ok(rows.into_iter().last().unwrap().1)
    }

    fn get_value(&self, row: usize, column: &str, default: SeriesValue) -> SeriesValue {
        self.at(row, column).unwrap_or(default)
    }

    fn lookup(
        &self,
        lookup_col: &str,
        other: &DataFrame,
        other_col: &str,
        result_col: &str,
    ) -> Result<DataFrame> {
        let lookup_vals = self.get_column_string_values(lookup_col)?;
        let other_keys = other.get_column_string_values(other_col)?;
        let other_result = other.get_column_string_values(result_col).or_else(|_| {
            other
                .get_column_numeric_values(result_col)
                .map(|v| v.iter().map(|x| x.to_string()).collect())
        })?;

        // Build lookup map
        let mut lookup_map: HashMap<String, String> = HashMap::new();
        for (i, key) in other_keys.iter().enumerate() {
            if i < other_result.len() {
                lookup_map.insert(key.clone(), other_result[i].clone());
            }
        }

        // Perform lookup
        let result_values: Vec<String> = lookup_vals
            .iter()
            .map(|k| lookup_map.get(k).cloned().unwrap_or_default())
            .collect();

        // Build result DataFrame
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }
        result.add_column(
            format!("{}_result", result_col),
            Series::new(result_values, Some(format!("{}_result", result_col)))?,
        )?;

        Ok(result)
    }

    fn get_column_by_index(&self, idx: usize) -> Result<(String, Vec<SeriesValue>)> {
        let columns = self.column_names();
        if idx >= columns.len() {
            return Err(Error::InvalidValue(format!(
                "Column index {} out of bounds",
                idx
            )));
        }

        let col_name = &columns[idx];
        let mut values = Vec::new();

        if let Ok(vals) = self.get_column_numeric_values(col_name) {
            values = vals.iter().map(|v| SeriesValue::Float(*v)).collect();
        } else if let Ok(vals) = self.get_column_string_values(col_name) {
            values = vals
                .iter()
                .map(|v| SeriesValue::String(v.clone()))
                .collect();
        }

        Ok((col_name.clone(), values))
    }

    fn swap_columns(&self, col1: &str, col2: &str) -> Result<DataFrame> {
        if !self.contains_column(col1) {
            return Err(Error::ColumnNotFound(col1.to_string()));
        }
        if !self.contains_column(col2) {
            return Err(Error::ColumnNotFound(col2.to_string()));
        }

        let mut result = DataFrame::new();
        for col in self.column_names() {
            let target_col = if col == col1 {
                col2
            } else if col == col2 {
                col1
            } else {
                &col
            };

            if let Ok(vals) = self.get_column_numeric_values(target_col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(target_col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn sort_columns(&self, ascending: bool) -> Result<DataFrame> {
        let mut columns = self.column_names();
        if ascending {
            columns.sort();
        } else {
            columns.sort_by(|a, b| b.cmp(a));
        }

        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.reindex_columns(&col_refs)
    }

    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<DataFrame> {
        let mut mapper = HashMap::new();
        mapper.insert(old_name.to_string(), new_name.to_string());
        self.rename_columns(&mapper)
    }

    fn to_categorical(&self, column: &str) -> Result<(DataFrame, HashMap<String, i64>)> {
        let values = self.get_column_string_values(column)?;

        // Build category map
        let mut category_map: HashMap<String, i64> = HashMap::new();
        let mut next_code: i64 = 0;

        let codes: Vec<f64> = values
            .iter()
            .map(|v| {
                if let Some(&code) = category_map.get(v) {
                    code as f64
                } else {
                    let code = next_code;
                    category_map.insert(v.clone(), code);
                    next_code += 1;
                    code as f64
                }
            })
            .collect();

        // Build result DataFrame
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if col == column {
                result.add_column(col.clone(), Series::new(codes.clone(), Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }

        Ok((result, category_map))
    }

    fn row_hash(&self) -> Vec<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let rows = self.iterrows();
        rows.iter()
            .map(|(_, row)| {
                let mut hasher = DefaultHasher::new();
                for (col, val) in row {
                    col.hash(&mut hasher);
                    match val {
                        SeriesValue::Float(f) => f.to_bits().hash(&mut hasher),
                        SeriesValue::Int(i) => i.hash(&mut hasher),
                        SeriesValue::String(s) => s.hash(&mut hasher),
                        SeriesValue::Bool(b) => b.hash(&mut hasher),
                    }
                }
                hasher.finish()
            })
            .collect()
    }

    fn sample_frac(&self, frac: f64, replace: bool) -> Result<DataFrame> {
        if frac < 0.0 || frac > 1.0 {
            return Err(Error::InvalidValue(
                "Fraction must be between 0 and 1".to_string(),
            ));
        }
        let n = (self.row_count() as f64 * frac).round() as usize;
        // Use the PandasCompatExt sample implementation
        PandasCompatExt::sample(self, n, replace)
    }

    fn take(&self, indices: &[usize]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                let taken: Vec<f64> = indices
                    .iter()
                    .filter_map(|&i| vals.get(i).copied())
                    .collect();
                result.add_column(col.clone(), Series::new(taken, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                let taken: Vec<String> = indices
                    .iter()
                    .filter_map(|&i| vals.get(i).cloned())
                    .collect();
                result.add_column(col.clone(), Series::new(taken, Some(col.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn duplicated_rows(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
        let cols_to_check: Vec<String> = subset
            .map(|s| s.iter().map(|&c| c.to_string()).collect())
            .unwrap_or_else(|| self.column_names());

        let hashes = self.row_hash();
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut result = vec![false; self.row_count()];

        match keep {
            "first" => {
                for (i, hash) in hashes.iter().enumerate() {
                    if seen.contains_key(hash) {
                        result[i] = true;
                    } else {
                        seen.insert(*hash, i);
                    }
                }
            }
            "last" => {
                // First pass: find last occurrence
                for (i, hash) in hashes.iter().enumerate() {
                    seen.insert(*hash, i);
                }
                // Second pass: mark all but last
                let mut seen_final: HashMap<u64, bool> = HashMap::new();
                for (i, hash) in hashes.iter().enumerate() {
                    if let Some(&last_idx) = seen.get(hash) {
                        if i != last_idx {
                            result[i] = true;
                        }
                    }
                    seen_final.insert(*hash, true);
                }
            }
            "none" | _ => {
                // Count occurrences
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for hash in &hashes {
                    *counts.entry(*hash).or_insert(0) += 1;
                }
                // Mark all rows with count > 1
                for (i, hash) in hashes.iter().enumerate() {
                    if counts.get(hash).copied().unwrap_or(0) > 1 {
                        result[i] = true;
                    }
                }
            }
        }

        // Use cols_to_check for subset-based comparison (simplified - using hash for now)
        let _ = cols_to_check;

        Ok(result)
    }

    fn get_column_as_f64(&self, column: &str) -> Result<Vec<f64>> {
        self.get_column_numeric_values(column)
    }

    fn get_column_as_string(&self, column: &str) -> Result<Vec<String>> {
        if let Ok(vals) = self.get_column_string_values(column) {
            Ok(vals)
        } else if let Ok(vals) = self.get_column_numeric_values(column) {
            Ok(vals.iter().map(|v| v.to_string()).collect())
        } else {
            Err(Error::ColumnNotFound(column.to_string()))
        }
    }

    fn groupby_apply<F>(&self, by: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> Result<HashMap<String, f64>>,
    {
        let group_vals = self.get_column_string_values(by).or_else(|_| {
            self.get_column_numeric_values(by)
                .map(|v| v.iter().map(|x| x.to_string()).collect())
        })?;

        // Get unique groups
        let mut unique_groups: Vec<String> = Vec::new();
        for val in &group_vals {
            if !unique_groups.contains(val) {
                unique_groups.push(val.clone());
            }
        }

        // Process each group
        let mut result_data: HashMap<String, Vec<f64>> = HashMap::new();
        let mut group_names: Vec<String> = Vec::new();

        for group in unique_groups {
            // Find indices for this group
            let indices: Vec<usize> = group_vals
                .iter()
                .enumerate()
                .filter(|(_, v)| *v == &group)
                .map(|(i, _)| i)
                .collect();

            // Create subset DataFrame
            let subset = self.take(&indices)?;

            // Apply function
            let agg_result = func(&subset)?;

            group_names.push(group);
            for (key, value) in agg_result {
                result_data.entry(key).or_default().push(value);
            }
        }

        // Build result DataFrame
        let mut result = DataFrame::new();
        result.add_column(
            by.to_string(),
            Series::new(group_names, Some(by.to_string()))?,
        )?;

        for (col, vals) in result_data {
            result.add_column(col.clone(), Series::new(vals, Some(col))?)?;
        }

        Ok(result)
    }

    fn corr_columns(&self, col1: &str, col2: &str) -> Result<f64> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;
        Ok(pearson_correlation(&v1, &v2))
    }

    fn cov_columns(&self, col1: &str, col2: &str) -> Result<f64> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;
        Ok(covariance(&v1, &v2))
    }

    fn var_column(&self, column: &str, ddof: usize) -> Result<f64> {
        let vals = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.len() <= ddof {
            return Ok(f64::NAN);
        }

        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - ddof as f64);

        Ok(variance)
    }

    fn std_column(&self, column: &str, ddof: usize) -> Result<f64> {
        Ok(self.var_column(column, ddof)?.sqrt())
    }

    fn str_lower(&self, column: &str) -> Result<DataFrame> {
        string_ops::str_lower(self, column)
    }

    fn str_upper(&self, column: &str) -> Result<DataFrame> {
        string_ops::str_upper(self, column)
    }

    fn str_strip(&self, column: &str) -> Result<DataFrame> {
        string_ops::str_strip(self, column)
    }

    fn str_contains(&self, column: &str, pattern: &str) -> Result<Vec<bool>> {
        string_ops::str_contains(self, column, pattern)
    }

    fn str_replace(&self, column: &str, pattern: &str, replacement: &str) -> Result<DataFrame> {
        string_ops::str_replace(self, column, pattern, replacement)
    }

    fn str_split(&self, column: &str, delimiter: &str) -> Result<Vec<Vec<String>>> {
        string_ops::str_split(self, column, delimiter)
    }

    fn str_len(&self, column: &str) -> Result<Vec<usize>> {
        string_ops::str_len(self, column)
    }

    fn sem(&self, column: &str, ddof: usize) -> Result<f64> {
        aggregations::sem(self, column, ddof)
    }

    fn mad(&self, column: &str) -> Result<f64> {
        aggregations::mad(self, column)
    }

    fn ffill(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mut filled = Vec::with_capacity(values.len());
        let mut last_valid = f64::NAN;

        for v in &values {
            if !v.is_nan() {
                last_valid = *v;
            }
            filled.push(last_valid);
        }

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(filled.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn bfill(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mut filled = vec![f64::NAN; values.len()];
        let mut last_valid = f64::NAN;

        // Iterate backwards
        for i in (0..values.len()).rev() {
            if !values[i].is_nan() {
                last_valid = values[i];
            }
            filled[i] = last_valid;
        }

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(filled.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }

        Ok(result)
    }

    fn pct_rank(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let n = values.len();
        let mut result = vec![f64::NAN; n];

        // Get indices sorted by value
        let mut indexed: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i, *v))
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let valid_count = indexed.len();
        if valid_count == 0 {
            return Ok(result);
        }

        // Assign ranks (0 to 1)
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            result[*idx] = rank as f64 / (valid_count - 1).max(1) as f64;
        }

        Ok(result)
    }

    fn abs_column(&self, column: &str) -> Result<DataFrame> {
        math_ops::abs_column(self, column)
    }

    fn round_column(&self, column: &str, decimals: i32) -> Result<DataFrame> {
        math_ops::round_column(self, column, decimals)
    }

    fn argmax(&self, column: &str) -> Result<usize> {
        let values = self.get_column_numeric_values(column)?;
        let mut max_idx = 0;
        let mut max_val = f64::NEG_INFINITY;

        for (i, v) in values.iter().enumerate() {
            if !v.is_nan() && *v > max_val {
                max_val = *v;
                max_idx = i;
            }
        }

        if max_val == f64::NEG_INFINITY {
            return Err(Error::InvalidValue(
                "No valid values found in column".to_string(),
            ));
        }

        Ok(max_idx)
    }

    fn argmin(&self, column: &str) -> Result<usize> {
        let values = self.get_column_numeric_values(column)?;
        let mut min_idx = 0;
        let mut min_val = f64::INFINITY;

        for (i, v) in values.iter().enumerate() {
            if !v.is_nan() && *v < min_val {
                min_val = *v;
                min_idx = i;
            }
        }

        if min_val == f64::INFINITY {
            return Err(Error::InvalidValue(
                "No valid values found in column".to_string(),
            ));
        }

        Ok(min_idx)
    }

    fn gt(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::gt(self, column, value)
    }

    fn ge(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::ge(self, column, value)
    }

    fn lt(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::lt(self, column, value)
    }

    fn le(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::le(self, column, value)
    }

    fn eq_value(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::eq_value(self, column, value)
    }

    fn ne_value(&self, column: &str, value: f64) -> Result<Vec<bool>> {
        comparison_ops::ne_value(self, column, value)
    }

    fn clip_lower(&self, column: &str, min: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let clipped: Vec<f64> = values.iter().map(|v| v.max(min)).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(clipped.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(result)
    }

    fn clip_upper(&self, column: &str, max: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let clipped: Vec<f64> = values.iter().map(|v| v.min(max)).collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(clipped.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(result)
    }

    fn any_column(&self, column: &str) -> Result<bool> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().any(|v| !v.is_nan() && *v != 0.0))
    }

    fn all_column(&self, column: &str) -> Result<bool> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().all(|v| !v.is_nan() && *v != 0.0))
    }

    fn count_na(&self, column: &str) -> Result<usize> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().filter(|v| v.is_nan()).count())
    }

    fn prod(&self, column: &str) -> Result<f64> {
        aggregations::prod(self, column)
    }

    fn coalesce(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;

        if values1.len() != values2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let coalesced: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| if v1.is_nan() { *v2 } else { *v1 })
            .collect();

        let mut result = self.copy();
        result.add_column(
            result_name.to_string(),
            Series::new(coalesced, Some(result_name.to_string()))?,
        )?;
        Ok(result)
    }

    fn first_valid(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        for v in &values {
            if !v.is_nan() {
                return Ok(*v);
            }
        }
        Ok(f64::NAN)
    }

    fn last_valid(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        for v in values.iter().rev() {
            if !v.is_nan() {
                return Ok(*v);
            }
        }
        Ok(f64::NAN)
    }

    fn add_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;

        if values1.len() != values2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 + v2)
            .collect();

        let mut result = self.copy();
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;
        Ok(result)
    }

    fn sub_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;

        if values1.len() != values2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 - v2)
            .collect();

        let mut result = self.copy();
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;
        Ok(result)
    }

    fn mul_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;

        if values1.len() != values2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 * v2)
            .collect();

        let mut result = self.copy();
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;
        Ok(result)
    }

    fn div_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;

        if values1.len() != values2.len() {
            return Err(Error::InvalidValue(
                "Columns must have the same length".to_string(),
            ));
        }

        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 / v2)
            .collect();

        let mut result = self.copy();
        result.add_column(
            result_name.to_string(),
            Series::new(result_values, Some(result_name.to_string()))?,
        )?;
        Ok(result)
    }

    fn mod_column(&self, column: &str, divisor: f64) -> Result<DataFrame> {
        math_ops::mod_column(self, column, divisor)
    }

    fn floordiv(&self, column: &str, divisor: f64) -> Result<DataFrame> {
        math_ops::floordiv(self, column, divisor)
    }

    fn neg(&self, column: &str) -> Result<DataFrame> {
        math_ops::neg(self, column)
    }

    fn sign(&self, column: &str) -> Result<Vec<i32>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values
            .iter()
            .map(|v| {
                if v.is_nan() {
                    0
                } else if *v > 0.0 {
                    1
                } else if *v < 0.0 {
                    -1
                } else {
                    0
                }
            })
            .collect())
    }

    fn is_finite(&self, column: &str) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().map(|v| v.is_finite()).collect())
    }

    fn is_infinite(&self, column: &str) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().map(|v| v.is_infinite()).collect())
    }

    fn replace_inf(&self, column: &str, replacement: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let result_values: Vec<f64> = values
            .iter()
            .map(|v| if v.is_infinite() { replacement } else { *v })
            .collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(result_values.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(result)
    }

    fn str_startswith(&self, column: &str, prefix: &str) -> Result<Vec<bool>> {
        string_ops::str_startswith(self, column, prefix)
    }

    fn str_endswith(&self, column: &str, suffix: &str) -> Result<Vec<bool>> {
        string_ops::str_endswith(self, column, suffix)
    }

    fn str_pad_left(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        string_ops::str_pad_left(self, column, width, fillchar)
    }

    fn str_pad_right(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        string_ops::str_pad_right(self, column, width, fillchar)
    }

    fn str_slice(&self, column: &str, start: usize, end: Option<usize>) -> Result<DataFrame> {
        string_ops::str_slice(self, column, start, end)
    }

    fn floor(&self, column: &str) -> Result<DataFrame> {
        math_ops::floor(self, column)
    }

    fn ceil(&self, column: &str) -> Result<DataFrame> {
        math_ops::ceil(self, column)
    }

    fn trunc(&self, column: &str) -> Result<DataFrame> {
        math_ops::trunc(self, column)
    }

    fn fract(&self, column: &str) -> Result<DataFrame> {
        math_ops::fract(self, column)
    }

    fn reciprocal(&self, column: &str) -> Result<DataFrame> {
        math_ops::reciprocal(self, column)
    }

    fn count_value(&self, column: &str, value: f64) -> Result<usize> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values
            .iter()
            .filter(|v| !v.is_nan() && (*v - value).abs() < f64::EPSILON)
            .count())
    }

    fn fillna_zero(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let result_values: Vec<f64> = values
            .iter()
            .map(|v| if v.is_nan() { 0.0 } else { *v })
            .collect();

        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result.add_column(
                    col_name.clone(),
                    Series::new(result_values.clone(), Some(col_name))?,
                )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
        Ok(result)
    }

    fn nunique_all(&self) -> Result<HashMap<String, usize>> {
        let mut result = HashMap::new();
        for col_name in self.column_names() {
            let count = if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                let unique: std::collections::HashSet<_> = vals
                    .iter()
                    .filter(|v| !v.is_nan())
                    .map(|v| v.to_bits())
                    .collect();
                unique.len()
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                let unique: std::collections::HashSet<_> = vals.iter().collect();
                unique.len()
            } else {
                0
            };
            result.insert(col_name, count);
        }
        Ok(result)
    }

    fn is_between(
        &self,
        column: &str,
        lower: f64,
        upper: f64,
        inclusive: bool,
    ) -> Result<Vec<bool>> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values
            .iter()
            .map(|v| {
                if v.is_nan() {
                    false
                } else if inclusive {
                    *v >= lower && *v <= upper
                } else {
                    *v > lower && *v < upper
                }
            })
            .collect())
    }

    fn str_count(&self, column: &str, pattern: &str) -> Result<Vec<usize>> {
        string_ops::str_count(self, column, pattern)
    }

    fn str_repeat(&self, column: &str, n: usize) -> Result<DataFrame> {
        string_ops::str_repeat(self, column, n)
    }

    fn str_center(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
        string_ops::str_center(self, column, width, fillchar)
    }

    fn str_zfill(&self, column: &str, width: usize) -> Result<DataFrame> {
        string_ops::str_zfill(self, column, width)
    }

    fn is_numeric_column(&self, column: &str) -> bool {
        DataFrame::get_column_numeric_values(self, column).is_ok()
    }

    fn is_string_column(&self, column: &str) -> bool {
        DataFrame::get_column_string_values(self, column).is_ok()
    }

    fn has_nulls(&self, column: &str) -> Result<bool> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().any(|v| v.is_nan()))
    }

    fn describe_column(&self, column: &str) -> Result<HashMap<String, f64>> {
        aggregations::describe_column(self, column)
    }

    fn memory_usage_column(&self, column: &str) -> Result<usize> {
        if let Ok(vals) = self.get_column_numeric_values(column) {
            Ok(vals.len() * std::mem::size_of::<f64>())
        } else if let Ok(vals) = self.get_column_string_values(column) {
            Ok(vals.iter().map(|s| s.len()).sum::<usize>()
                + vals.len() * std::mem::size_of::<String>())
        } else {
            Err(Error::ColumnNotFound(column.to_string()))
        }
    }

    fn range(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

        if valid.is_empty() {
            return Ok(f64::NAN);
        }

        let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Ok(max - min)
    }

    fn abs_sum(&self, column: &str) -> Result<f64> {
        let values = self.get_column_numeric_values(column)?;
        Ok(values.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).sum())
    }

    fn is_unique(&self, column: &str) -> Result<bool> {
        if let Ok(vals) = self.get_column_numeric_values(column) {
            let unique: std::collections::HashSet<_> = vals
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.to_bits())
                .collect();
            let valid_count = vals.iter().filter(|v| !v.is_nan()).count();
            Ok(unique.len() == valid_count)
        } else if let Ok(vals) = self.get_column_string_values(column) {
            let unique: std::collections::HashSet<_> = vals.iter().collect();
            Ok(unique.len() == vals.len())
        } else {
            Err(Error::ColumnNotFound(column.to_string()))
        }
    }

    fn mode_with_count(&self, column: &str) -> Result<(f64, usize)> {
        let values = self.get_column_numeric_values(column)?;
        let mut counts: HashMap<u64, usize> = HashMap::new();

        for v in &values {
            if !v.is_nan() {
                *counts.entry(v.to_bits()).or_insert(0) += 1;
            }
        }

        if counts.is_empty() {
            return Ok((f64::NAN, 0));
        }

        let (mode_bits, count) = counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
        Ok((f64::from_bits(mode_bits), count))
    }

    fn geometric_mean(&self, column: &str) -> Result<f64> {
        aggregations::geometric_mean(self, column)
    }

    fn harmonic_mean(&self, column: &str) -> Result<f64> {
        aggregations::harmonic_mean(self, column)
    }

    fn iqr(&self, column: &str) -> Result<f64> {
        aggregations::iqr(self, column)
    }

    fn cv(&self, column: &str) -> Result<f64> {
        aggregations::cv(self, column)
    }

    fn percentile_value(&self, column: &str, q: f64) -> Result<f64> {
        aggregations::percentile_value(self, column, q)
    }

    fn trimmed_mean(&self, column: &str, trim_fraction: f64) -> Result<f64> {
        aggregations::trimmed_mean(self, column, trim_fraction)
    }
}
/// Helper function to compute Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}
/// Helper function to compute covariance
fn covariance(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    sum / n
}
#[cfg(test)]
mod tests {
    use super::*;
    fn create_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                    "David".to_string(),
                    "Eve".to_string(),
                ],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df
    }
    #[test]
    fn test_pipe() {
        let df = create_test_df();
        let result = df.pipe(|d| d.row_count());
        assert_eq!(result, 5);
    }
    #[test]
    fn test_isin() {
        let df = create_test_df();
        let mask = df.isin("name", &["Alice", "Bob", "Unknown"]).unwrap();
        assert_eq!(mask, vec![true, true, false, false, false]);
    }
    #[test]
    fn test_nlargest() {
        let df = create_test_df();
        let result = df.nlargest(3, "a").unwrap();
        assert_eq!(result.row_count(), 3);
    }
    #[test]
    fn test_nsmallest() {
        let df = create_test_df();
        let result = df.nsmallest(2, "a").unwrap();
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_idxmax() {
        let df = create_test_df();
        let idx = df.idxmax("a").unwrap();
        assert_eq!(idx, Some(4));
    }
    #[test]
    fn test_idxmin() {
        let df = create_test_df();
        let idx = df.idxmin("a").unwrap();
        assert_eq!(idx, Some(0));
    }
    #[test]
    fn test_rank_average() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        let ranks = df.rank("x", RankMethod::Average).unwrap();
        assert_eq!(ranks[1], 1.5);
        assert_eq!(ranks[3], 1.5);
        assert_eq!(ranks[0], 3.0);
    }
    #[test]
    fn test_between() {
        let df = create_test_df();
        let mask = df.between("a", 2.0, 4.0).unwrap();
        assert_eq!(mask, vec![false, true, true, true, false]);
    }
    #[test]
    fn test_cumsum() {
        let df = create_test_df();
        let result = df.cumsum("a").unwrap();
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }
    #[test]
    fn test_cumprod() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.cumprod("x").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 6.0, 24.0]);
    }
    #[test]
    fn test_cummax() {
        let df = create_test_df();
        let result = df.cummax("b").unwrap();
        assert_eq!(result, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    }
    #[test]
    fn test_cummin() {
        let df = create_test_df();
        let result = df.cummin("b").unwrap();
        assert_eq!(result, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_shift() {
        let df = create_test_df();
        let result = df.shift("a", 1).unwrap();
        assert_eq!(result[0], None);
        assert_eq!(result[1], Some(1.0));
        assert_eq!(result[2], Some(2.0));
    }
    #[test]
    fn test_nunique() {
        let df = create_test_df();
        let result = df.nunique().unwrap();
        for (_, count) in &result {
            assert_eq!(*count, 5);
        }
    }
    #[test]
    fn test_memory_usage() {
        let df = create_test_df();
        let mem = df.memory_usage();
        assert!(mem > 0);
    }
    #[test]
    fn test_assign_many() {
        let df = create_test_df();
        let result = df
            .assign_many(vec![("c", vec![10.0, 20.0, 30.0, 40.0, 50.0])])
            .unwrap();
        assert!(result.contains_column("c"));
    }
    #[test]
    fn test_value_counts() {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "C".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        let counts = PandasCompatExt::value_counts(&df, "category").unwrap();
        assert_eq!(&counts[0].0, "A");
        assert_eq!(counts[0].1, 3);
        assert_eq!(counts[1].1, 1);
        assert_eq!(counts[2].1, 1);
    }
    #[test]
    fn test_value_counts_numeric() {
        let df = create_test_df();
        let result = df.value_counts_numeric("a").unwrap();
        assert_eq!(result.len(), 5);
        for (_, count) in &result {
            assert_eq!(*count, 1);
        }
    }
    #[test]
    fn test_describe() {
        let df = create_test_df();
        let stats = df.describe("a").unwrap();
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.0001);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.q50, 3.0);
    }
    #[test]
    fn test_apply_rows() {
        let df = create_test_df();
        let result: Vec<f64> = df.apply(|row| row.iter().sum::<f64>(), Axis::Rows).unwrap();
        assert!((result[0] - 6.0).abs() < 0.0001);
        assert!((result[4] - 6.0).abs() < 0.0001);
    }
    #[test]
    fn test_apply_columns() {
        let df = create_test_df();
        let result: Vec<f64> = df
            .apply(|col| col.iter().sum::<f64>(), Axis::Columns)
            .unwrap();
        assert!((result[0] - 15.0).abs() < 0.0001);
        assert!((result[1] - 15.0).abs() < 0.0001);
    }
    #[test]
    fn test_corr() {
        let df = create_test_df();
        let corr_matrix = df.corr().unwrap();
        assert!((corr_matrix.values[0][0] - 1.0).abs() < 0.0001);
        assert!((corr_matrix.values[1][1] - 1.0).abs() < 0.0001);
        assert!(corr_matrix.values[0][1] < -0.99);
        assert!(corr_matrix.values[1][0] < -0.99);
    }
    #[test]
    fn test_cov() {
        let df = create_test_df();
        let cov_matrix = df.cov().unwrap();
        assert!((cov_matrix.values[0][1] - cov_matrix.values[1][0]).abs() < 0.0001);
        let (rows, cols) = cov_matrix.shape();
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
    }
    #[test]
    fn test_pct_change() {
        let df = create_test_df();
        let result = df.pct_change("a", 1).unwrap();
        assert!(result[0].is_nan());
        assert!((result[1] - 1.0).abs() < 0.0001);
        assert!((result[2] - 0.5).abs() < 0.0001);
    }
    #[test]
    fn test_diff() {
        let df = create_test_df();
        let result = df.diff("a", 1).unwrap();
        assert!(result[0].is_nan());
        for i in 1..result.len() {
            assert!((result[i] - 1.0).abs() < 0.0001);
        }
    }
    #[test]
    fn test_diff_periods() {
        let df = create_test_df();
        let result = df.diff("a", 2).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        for i in 2..result.len() {
            assert!((result[i] - 2.0).abs() < 0.0001);
        }
    }
    #[test]
    fn test_replace() {
        let mut df = DataFrame::new();
        df.add_column(
            "status".to_string(),
            Series::new(
                vec!["ok".to_string(), "fail".to_string(), "ok".to_string()],
                Some("status".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        let result = df
            .replace("status", &["ok", "fail"], &["success", "error"])
            .unwrap();
        let values = result.get_column_string_values("status").unwrap();
        assert_eq!(values[0], "success");
        assert_eq!(values[1], "error");
        assert_eq!(values[2], "success");
    }
    #[test]
    fn test_replace_numeric() {
        let df = create_test_df();
        let result = df.replace_numeric("a", &[1.0, 2.0], &[10.0, 20.0]).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values[0], 10.0);
        assert_eq!(values[1], 20.0);
        assert_eq!(values[2], 3.0);
    }
    #[test]
    fn test_correlation_matrix_get() {
        let df = create_test_df();
        let corr = df.corr().unwrap();
        let val = corr.get("a", "b").unwrap();
        assert!(val < -0.99);
        let self_corr = corr.get("a", "a").unwrap();
        assert!((self_corr - 1.0).abs() < 0.0001);
    }
    #[test]
    fn test_sample() {
        let df = create_test_df();
        let result = PandasCompatExt::sample(&df, 3, false).unwrap();
        assert_eq!(result.row_count(), 3);
        let result2 = PandasCompatExt::sample(&df, 10, true).unwrap();
        assert_eq!(result2.row_count(), 10);
    }
    #[test]
    fn test_sample_too_many() {
        let df = create_test_df();
        let result = PandasCompatExt::sample(&df, 10, false);
        assert!(result.is_err());
    }
    #[test]
    fn test_drop_columns() {
        let df = create_test_df();
        let result = df.drop_columns(&["a"]).unwrap();
        assert!(!result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(result.contains_column("name"));
    }
    #[test]
    fn test_rename_columns() {
        let df = create_test_df();
        let mut mapper = HashMap::new();
        mapper.insert("a".to_string(), "alpha".to_string());
        mapper.insert("b".to_string(), "beta".to_string());
        let result = df.rename_columns(&mapper).unwrap();
        assert!(result.contains_column("alpha"));
        assert!(result.contains_column("beta"));
        assert!(!result.contains_column("a"));
        assert!(!result.contains_column("b"));
    }
    #[test]
    fn test_abs() {
        let mut df = DataFrame::new();
        df.add_column(
            "values".to_string(),
            Series::new(vec![-1.0, -2.0, 3.0, -4.0, 5.0], Some("values".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.abs("values").unwrap();
        let values = result.get_column_numeric_values("values").unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 5.0);
    }
    #[test]
    fn test_round() {
        let mut df = DataFrame::new();
        df.add_column(
            "values".to_string(),
            Series::new(vec![1.123, 2.567, 3.999], Some("values".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.round("values", 2).unwrap();
        let values = result.get_column_numeric_values("values").unwrap();
        assert!((values[0] - 1.12).abs() < 0.001);
        assert!((values[1] - 2.57).abs() < 0.001);
        assert!((values[2] - 4.00).abs() < 0.001);
    }
    #[test]
    fn test_quantile() {
        let df = create_test_df();
        let median = df.quantile("a", 0.5).unwrap();
        assert_eq!(median, 3.0);
        let q25 = df.quantile("a", 0.25).unwrap();
        assert_eq!(q25, 2.0);
        let q75 = df.quantile("a", 0.75).unwrap();
        assert_eq!(q75, 4.0);
    }
    #[test]
    fn test_quantile_invalid() {
        let df = create_test_df();
        assert!(df.quantile("a", 1.5).is_err());
        assert!(df.quantile("a", -0.5).is_err());
    }
    #[test]
    fn test_head() {
        let df = create_test_df();
        let result = PandasCompatExt::head(&df, 3).unwrap();
        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_head_more_than_available() {
        let df = create_test_df();
        let result = PandasCompatExt::head(&df, 100).unwrap();
        assert_eq!(result.row_count(), 5);
    }
    #[test]
    fn test_tail() {
        let df = create_test_df();
        let result = PandasCompatExt::tail(&df, 3).unwrap();
        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_unique() {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "C".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        let result = df.unique("category").unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&"A".to_string()));
        assert!(result.contains(&"B".to_string()));
        assert!(result.contains(&"C".to_string()));
    }
    #[test]
    fn test_unique_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "nums".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], Some("nums".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.unique_numeric("nums").unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_fillna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        let result = df.fillna("a", 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }
    #[test]
    fn test_fillna_with_negative() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.fillna("a", -999.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, -999.0, 3.0]);
    }

    #[test]
    fn test_fillna_ffill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.fillna_method("a", "ffill").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        // First value stays NaN (no prior value to fill from)
        assert!(values[0] == 1.0);
        assert_eq!(values[1], 1.0); // Filled with previous value
        assert_eq!(values[2], 1.0); // Filled with previous value
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 4.0); // Filled with previous value
        assert_eq!(values[5], 6.0);
    }

    #[test]
    fn test_fillna_bfill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.fillna_method("a", "bfill").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 4.0); // Filled with next value
        assert_eq!(values[2], 4.0); // Filled with next value
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 6.0); // Filled with next value
        assert_eq!(values[5], 6.0);
    }

    #[test]
    fn test_fillna_ffill_leading_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.fillna_method("a", "ffill").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        // Leading NaNs stay as NaN
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 3.0); // Filled with previous value
        assert_eq!(values[4], 5.0);
    }

    #[test]
    fn test_fillna_bfill_trailing_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.fillna_method("a", "bfill").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 3.0); // Filled with next value
        assert_eq!(values[2], 3.0);
        // Trailing NaNs stay as NaN
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }

    #[test]
    fn test_fillna_invalid_method() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.fillna_method("a", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_linear() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.interpolate("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0); // Linear interpolation: 1 + (4-1)*1/3
        assert_eq!(values[2], 3.0); // Linear interpolation: 1 + (4-1)*2/3
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 5.0); // Linear interpolation: 4 + (6-4)*1/2
        assert_eq!(values[5], 6.0);
    }

    #[test]
    fn test_interpolate_single_gap() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, f64::NAN, 8.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.interpolate("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 2.0);
        assert_eq!(values[1], 5.0); // Linear interpolation: 2 + (8-2)*1/2
        assert_eq!(values[2], 8.0);
    }

    #[test]
    fn test_interpolate_leading_trailing_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, 2.0, f64::NAN, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.interpolate("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        // Leading NaN stays as NaN
        assert!(values[0].is_nan());
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0); // Interpolated
        assert_eq!(values[3], 4.0);
        // Trailing NaN stays as NaN
        assert!(values[4].is_nan());
    }

    #[test]
    fn test_interpolate_no_gaps() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.interpolate("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        // No changes expected
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dropna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.dropna("a").unwrap();
        assert_eq!(result.row_count(), 3);
        let values_a = result.get_column_numeric_values("a").unwrap();
        let values_b = result.get_column_numeric_values("b").unwrap();
        assert_eq!(values_a, vec![1.0, 3.0, 5.0]);
        assert_eq!(values_b, vec![10.0, 30.0, 50.0]);
    }
    #[test]
    fn test_isna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        let result = df.isna("a").unwrap();
        assert_eq!(result, vec![false, true, false, true, false]);
    }
    #[test]
    fn test_sum_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sum_all().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 6.0));
        assert_eq!(result[1], ("b".to_string(), 60.0));
    }
    #[test]
    fn test_sum_all_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sum_all().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 4.0));
    }
    #[test]
    fn test_mean_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.mean_all().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 2.0));
        assert_eq!(result[1], ("b".to_string(), 20.0));
    }
    #[test]
    fn test_std_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.std_all().unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0].1 - 1.5811).abs() < 0.001);
    }
    #[test]
    fn test_var_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.var_all().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 2.5));
    }
    #[test]
    fn test_min_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 2.0, 8.0, 1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 5.0, 15.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.min_all().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 1.0));
        assert_eq!(result[1], ("b".to_string(), 5.0));
    }
    #[test]
    fn test_max_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 2.0, 8.0, 1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 5.0, 15.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.max_all().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 8.0));
        assert_eq!(result[1], ("b".to_string(), 20.0));
    }
    #[test]
    fn test_sort_values_ascending() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 2.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![30.0, 10.0, 40.0, 20.0, 50.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sort_values("a", true).unwrap();
        let values_a = result.get_column_numeric_values("a").unwrap();
        let values_b = result.get_column_numeric_values("b").unwrap();
        assert_eq!(values_a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(values_b, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }
    #[test]
    fn test_sort_values_descending() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 2.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sort_values("a", false).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_sort_by_columns_single() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sort_by_columns(&["a"], &[true]).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_sort_by_columns_multiple() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![20.0, 10.0, 10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sort_by_columns(&["a", "b"], &[true, true]).unwrap();
        let values_a = result.get_column_numeric_values("a").unwrap();
        let values_b = result.get_column_numeric_values("b").unwrap();
        assert_eq!(values_a, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(values_b, vec![10.0, 20.0, 10.0, 20.0]);
    }
    #[test]
    fn test_sort_by_columns_mixed_order() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![20.0, 10.0, 10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        let result = df.sort_by_columns(&["a", "b"], &[true, false]).unwrap();
        let values_a = result.get_column_numeric_values("a").unwrap();
        let values_b = result.get_column_numeric_values("b").unwrap();
        assert_eq!(values_a, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(values_b, vec![20.0, 10.0, 20.0, 10.0]);
    }
    #[test]
    fn test_sort_by_columns_error_mismatch() {
        let df = create_test_df();
        let result = df.sort_by_columns(&["a", "b"], &[true]);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_cond() {
        let df = create_test_df();
        // Keep value where condition is True, replace with -1 where False
        let condition = vec![true, false, true, false, true];
        let result = df.where_cond("a", &condition, -1.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, -1.0, 3.0, -1.0, 5.0]);
    }

    #[test]
    fn test_where_cond_all_true() {
        let df = create_test_df();
        let condition = vec![true, true, true, true, true];
        let result = df.where_cond("a", &condition, 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_where_cond_all_false() {
        let df = create_test_df();
        let condition = vec![false, false, false, false, false];
        let result = df.where_cond("a", &condition, 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_where_cond_length_mismatch() {
        let df = create_test_df();
        let condition = vec![true, false]; // Wrong length
        let result = df.where_cond("a", &condition, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_mask() {
        let df = create_test_df();
        // Replace value where condition is True, keep where False
        let condition = vec![true, false, true, false, true];
        let result = df.mask("a", &condition, -1.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![-1.0, 2.0, -1.0, 4.0, -1.0]);
    }

    #[test]
    fn test_mask_all_true() {
        let df = create_test_df();
        let condition = vec![true, true, true, true, true];
        let result = df.mask("a", &condition, 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mask_all_false() {
        let df = create_test_df();
        let condition = vec![false, false, false, false, false];
        let result = df.mask("a", &condition, 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_mask_length_mismatch() {
        let df = create_test_df();
        let condition = vec![true, false]; // Wrong length
        let result = df.mask("a", &condition, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_duplicates_keep_first() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 10.0, 30.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.drop_duplicates(None, "first").unwrap();
        assert_eq!(result.row_count(), 3);
        let values_a = result.get_column_numeric_values("a").unwrap();
        assert!(values_a.contains(&1.0));
        assert!(values_a.contains(&2.0));
        assert!(values_a.contains(&3.0));
    }

    #[test]
    fn test_drop_duplicates_keep_last() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.drop_duplicates(Some(&["a"]), "last").unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_drop_duplicates_keep_none() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        // "none" removes all duplicates, keeping only unique rows
        let result = df.drop_duplicates(Some(&["a"]), "none").unwrap();
        assert_eq!(result.row_count(), 3); // Only 2, 3, 4 are unique
        let values = result.get_column_numeric_values("a").unwrap();
        assert!(values.contains(&2.0));
        assert!(values.contains(&3.0));
        assert!(values.contains(&4.0));
        assert!(!values.contains(&1.0)); // Duplicates removed
    }

    #[test]
    fn test_drop_duplicates_subset() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 1.0, 2.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        // Drop duplicates based only on column 'a'
        let result = df.drop_duplicates(Some(&["a"]), "first").unwrap();
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_drop_duplicates_invalid_keep() {
        let df = create_test_df();
        let result = df.drop_duplicates(None, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_duplicates_no_duplicates() {
        let df = create_test_df();
        let result = df.drop_duplicates(None, "first").unwrap();
        assert_eq!(result.row_count(), 5);
    }

    #[test]
    fn test_select_dtypes_numeric() {
        let df = create_test_df();
        let result = df.select_dtypes(&["numeric"]).unwrap();
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(!result.contains_column("name"));
    }

    #[test]
    fn test_select_dtypes_string() {
        let df = create_test_df();
        let result = df.select_dtypes(&["string"]).unwrap();
        assert!(!result.contains_column("a"));
        assert!(!result.contains_column("b"));
        assert!(result.contains_column("name"));
    }

    #[test]
    fn test_select_dtypes_both() {
        let df = create_test_df();
        let result = df.select_dtypes(&["numeric", "string"]).unwrap();
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(result.contains_column("name"));
    }

    #[test]
    fn test_select_dtypes_aliases() {
        let df = create_test_df();

        // Test "number" alias
        let result = df.select_dtypes(&["number"]).unwrap();
        assert!(result.contains_column("a"));
        assert!(!result.contains_column("name"));

        // Test "object" alias for string
        let result2 = df.select_dtypes(&["object"]).unwrap();
        assert!(!result2.contains_column("a"));
        assert!(result2.contains_column("name"));
    }

    #[test]
    fn test_any_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "all_zero".to_string(),
            Series::new(vec![0.0, 0.0, 0.0], Some("all_zero".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "has_nonzero".to_string(),
            Series::new(vec![0.0, 1.0, 0.0], Some("has_nonzero".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "has_nan".to_string(),
            Series::new(vec![f64::NAN, f64::NAN, 1.0], Some("has_nan".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.any_numeric().unwrap();
        assert_eq!(result.len(), 3);

        for (name, has_any) in &result {
            match name.as_str() {
                "all_zero" => assert!(!has_any),
                "has_nonzero" => assert!(*has_any),
                "has_nan" => assert!(*has_any),
                _ => panic!("Unexpected column"),
            }
        }
    }

    #[test]
    fn test_all_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "all_nonzero".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("all_nonzero".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "has_zero".to_string(),
            Series::new(vec![1.0, 0.0, 3.0], Some("has_zero".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "has_nan".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("has_nan".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.all_numeric().unwrap();
        assert_eq!(result.len(), 3);

        for (name, all_true) in &result {
            match name.as_str() {
                "all_nonzero" => assert!(*all_true),
                "has_zero" => assert!(!all_true),
                "has_nan" => assert!(!all_true),
                _ => panic!("Unexpected column"),
            }
        }
    }

    #[test]
    fn test_count_valid_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.count_valid().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 3));
    }

    #[test]
    fn test_count_valid_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "".to_string(),
                    "Charlie".to_string(),
                    "".to_string(),
                ],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.count_valid().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("name".to_string(), 2)); // Non-empty strings
    }

    #[test]
    fn test_count_valid_mixed() {
        let df = create_test_df();
        let result = df.count_valid().unwrap();
        assert_eq!(result.len(), 3); // 3 columns

        // All values are valid in test df
        for (_, count) in &result {
            assert_eq!(*count, 5);
        }
    }

    #[test]
    fn test_reverse_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![3.0, 4.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "c".to_string(),
            Series::new(vec![5.0, 6.0], Some("c".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.reverse_columns().unwrap();
        let cols = result.column_names();
        assert_eq!(
            cols,
            vec!["c".to_string(), "b".to_string(), "a".to_string()]
        );
    }

    #[test]
    fn test_reverse_columns_single() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.reverse_columns().unwrap();
        let cols = result.column_names();
        assert_eq!(cols, vec!["a".to_string()]);
    }

    #[test]
    fn test_reverse_rows() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.reverse_rows().unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reverse_rows_with_strings() {
        let mut df = DataFrame::new();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string(), "C".to_string()],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.reverse_rows().unwrap();
        let values = result.get_column_string_values("name").unwrap();
        assert_eq!(
            values,
            vec!["C".to_string(), "B".to_string(), "A".to_string()]
        );
    }

    #[test]
    fn test_reverse_rows_preserves_columns() {
        let df = create_test_df();
        let result = df.reverse_rows().unwrap();

        assert_eq!(result.row_count(), 5);

        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);

        let b_values = result.get_column_numeric_values("b").unwrap();
        assert_eq!(b_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let names = result.get_column_string_values("name").unwrap();
        assert_eq!(names[0], "Eve");
        assert_eq!(names[4], "Alice");
    }

    // Tests for new pandas-compatible methods

    #[test]
    fn test_notna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.notna("a").unwrap();
        assert_eq!(result, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_melt_basic() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("id".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "y".to_string(),
            Series::new(vec![3.0, 4.0], Some("y".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.melt(&["id"], None, "variable", "value").unwrap();

        assert_eq!(result.row_count(), 4); // 2 rows * 2 value columns
        assert!(result.contains_column("id"));
        assert!(result.contains_column("variable"));
        assert!(result.contains_column("value"));
    }

    #[test]
    fn test_melt_with_value_vars() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("id".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "y".to_string(),
            Series::new(vec![3.0, 4.0], Some("y".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "z".to_string(),
            Series::new(vec![5.0, 6.0], Some("z".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.melt(&["id"], Some(&["x", "y"]), "var", "val").unwrap();

        assert_eq!(result.row_count(), 4); // 2 rows * 2 specified value columns
        assert!(result.contains_column("var"));
        assert!(result.contains_column("val"));
    }

    #[test]
    fn test_explode() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "tags".to_string(),
            Series::new(
                vec!["a,b,c".to_string(), "x,y".to_string()],
                Some("tags".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.explode("tags", ",").unwrap();

        assert_eq!(result.row_count(), 5); // 3 + 2 = 5 total rows
        let tags = result.get_column_string_values("tags").unwrap();
        assert_eq!(tags, vec!["a", "b", "c", "x", "y"]);
    }

    #[test]
    fn test_explode_preserves_other_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "items".to_string(),
            Series::new(
                vec!["a,b".to_string(), "c".to_string()],
                Some("items".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.explode("items", ",").unwrap();

        let ids = result.get_column_numeric_values("id").unwrap();
        assert_eq!(ids, vec![1.0, 1.0, 2.0]); // id=1 repeated for 2 items
    }

    #[test]
    fn test_duplicated_keep_first() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.duplicated(Some(&["a"]), "first").unwrap();
        assert_eq!(result, vec![false, false, true, false, true]);
    }

    #[test]
    fn test_duplicated_keep_last() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.duplicated(Some(&["a"]), "last").unwrap();
        assert_eq!(result, vec![true, true, false, false, false]);
    }

    #[test]
    fn test_duplicated_keep_none() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.duplicated(Some(&["a"]), "none").unwrap();
        assert_eq!(result, vec![true, false, true, false]); // Both 1s are duplicates
    }

    #[test]
    fn test_copy() {
        let df = create_test_df();
        let copied = df.copy();

        assert_eq!(df.row_count(), copied.row_count());
        assert_eq!(df.column_names(), copied.column_names());
    }

    #[test]
    fn test_to_dict() {
        let df = create_test_df();
        let dict = df.to_dict().unwrap();

        assert!(dict.contains_key("a"));
        assert!(dict.contains_key("b"));
        assert!(dict.contains_key("name"));
        assert_eq!(dict.get("name").unwrap().len(), 5);
    }

    #[test]
    fn test_first_valid_index() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, 4.0, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.first_valid_index("a").unwrap();
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_first_valid_index_no_nan() {
        let df = create_test_df();
        let result = df.first_valid_index("a").unwrap();
        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_first_valid_index_all_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![f64::NAN, f64::NAN, f64::NAN], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.first_valid_index("a").unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_last_valid_index() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, 2.0, 3.0, f64::NAN, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.last_valid_index("a").unwrap();
        assert_eq!(result, Some(2));
    }

    #[test]
    fn test_product_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.product_all().unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 24.0); // 1 * 2 * 3 * 4
    }

    #[test]
    fn test_product_all_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.product_all().unwrap();
        assert_eq!(result[0].1, 6.0); // 2 * 3 (NaN skipped)
    }

    #[test]
    fn test_median_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.median_all().unwrap();
        assert_eq!(result[0].1, 3.0);
    }

    #[test]
    fn test_median_all_even() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.median_all().unwrap();
        assert_eq!(result[0].1, 2.5); // (2 + 3) / 2
    }

    #[test]
    fn test_skew() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.skew("a").unwrap();
        assert!((result - 0.0).abs() < 0.1); // Symmetric distribution has ~0 skewness
    }

    #[test]
    fn test_skew_insufficient_data() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.skew("a").unwrap();
        assert!(result.is_nan()); // Need at least 3 values
    }

    #[test]
    fn test_kurtosis() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.kurtosis("a").unwrap();
        assert!(!result.is_nan());
    }

    #[test]
    fn test_kurtosis_insufficient_data() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.kurtosis("a").unwrap();
        assert!(result.is_nan()); // Need at least 4 values
    }

    #[test]
    fn test_add_prefix() {
        let df = create_test_df();
        let result = df.add_prefix("col_").unwrap();

        assert!(result.contains_column("col_a"));
        assert!(result.contains_column("col_b"));
        assert!(result.contains_column("col_name"));
        assert!(!result.contains_column("a"));
    }

    #[test]
    fn test_add_suffix() {
        let df = create_test_df();
        let result = df.add_suffix("_new").unwrap();

        assert!(result.contains_column("a_new"));
        assert!(result.contains_column("b_new"));
        assert!(result.contains_column("name_new"));
        assert!(!result.contains_column("a"));
    }

    #[test]
    fn test_filter_by_mask() {
        let df = create_test_df();
        let mask = vec![true, false, true, false, true];
        let result = df.filter_by_mask(&mask).unwrap();

        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_filter_by_mask_length_mismatch() {
        let df = create_test_df();
        let mask = vec![true, false]; // Wrong length
        let result = df.filter_by_mask(&mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_mode_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.mode_numeric("a").unwrap();
        assert_eq!(result, vec![3.0]); // 3 appears most frequently
    }

    #[test]
    fn test_mode_numeric_multiple() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 1.0, 2.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.mode_numeric("a").unwrap();
        assert_eq!(result, vec![1.0, 2.0]); // Both 1 and 2 appear twice
    }

    #[test]
    fn test_mode_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "cat".to_string(),
            Series::new(
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "a".to_string(),
                    "c".to_string(),
                ],
                Some("cat".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.mode_string("cat").unwrap();
        assert_eq!(result, vec!["a".to_string()]);
    }

    #[test]
    fn test_percentile() {
        let df = create_test_df();
        let p50 = df.percentile("a", 50.0).unwrap();
        assert_eq!(p50, 3.0); // Same as quantile(0.5)
    }

    #[test]
    fn test_ewma() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.ewma("a", 3).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0); // First value is unchanged
        assert!(result[4] > result[3]); // EWMA should increase for increasing values
    }

    #[test]
    fn test_ewma_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.ewma("a", 2).unwrap();
        assert_eq!(result[0], 1.0);
        assert!(result[1].is_nan()); // NaN propagates
    }

    #[test]
    fn test_iloc() {
        let df = create_test_df();
        let row = df.iloc(2).unwrap();

        assert_eq!(row.get("a").unwrap(), "3");
        assert_eq!(row.get("name").unwrap(), "Charlie");
    }

    #[test]
    fn test_iloc_out_of_bounds() {
        let df = create_test_df();
        let result = df.iloc(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_iloc_range() {
        let df = create_test_df();
        let result = df.iloc_range(1, 4).unwrap();

        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_iloc_range_clamped() {
        let df = create_test_df();
        let result = df.iloc_range(3, 100).unwrap();

        assert_eq!(result.row_count(), 2); // Clamped to [3, 5)
    }

    #[test]
    fn test_iloc_range_invalid() {
        let df = create_test_df();
        let result = df.iloc_range(4, 2);
        assert!(result.is_err()); // start > end
    }

    // ============== Tests for new methods ==============

    #[test]
    fn test_info() {
        let df = create_test_df();
        let info = df.info();

        assert!(info.contains("<DataFrame>"));
        assert!(info.contains("RangeIndex: 5 entries"));
        assert!(info.contains("Data columns (total 3 columns)"));
        assert!(info.contains("float64"));
        assert!(info.contains("object"));
        assert!(info.contains("memory usage:"));
    }

    #[test]
    fn test_info_empty() {
        let df = DataFrame::new();
        let info = df.info();

        assert!(info.contains("RangeIndex: 0 entries"));
        assert!(info.contains("Data columns (total 0 columns)"));
    }

    #[test]
    fn test_equals_same() {
        let df = create_test_df();
        assert!(df.equals(&df));
    }

    #[test]
    fn test_equals_identical() {
        let df1 = create_test_df();
        let df2 = create_test_df();
        assert!(df1.equals(&df2));
    }

    #[test]
    fn test_equals_different_values() {
        let df1 = create_test_df();
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 99.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                    "David".to_string(),
                    "Eve".to_string(),
                ],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(!df1.equals(&df2));
    }

    #[test]
    fn test_equals_different_columns() {
        let df1 = create_test_df();
        let mut df2 = DataFrame::new();
        df2.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        assert!(!df1.equals(&df2));
    }

    #[test]
    fn test_equals_nan_handling() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        assert!(df1.equals(&df2)); // NaN equals NaN in equals()
    }

    #[test]
    fn test_compare() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 5.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df1.compare(&df2).unwrap();
        let diff = result.get_column_numeric_values("a_diff").unwrap();

        assert_eq!(diff[0], 0.0);
        assert_eq!(diff[1], -3.0); // 2 - 5 = -3
        assert_eq!(diff[2], 0.0);
    }

    #[test]
    fn test_compare_different_rows() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df1.compare(&df2);
        assert!(result.is_err());
    }

    #[test]
    fn test_keys() {
        let df = create_test_df();
        let keys = df.keys();

        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"name".to_string()));
    }

    #[test]
    fn test_pop_column() {
        let df = create_test_df();
        let (new_df, values) = df.pop_column("a").unwrap();

        assert!(!new_df.contains_column("a"));
        assert!(new_df.contains_column("b"));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_pop_column_nonexistent() {
        let df = create_test_df();
        let result = df.pop_column("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_column() {
        let df = create_test_df();
        let new_df = df
            .insert_column(1, "new_col", vec![10.0, 20.0, 30.0, 40.0, 50.0])
            .unwrap();

        let cols = new_df.column_names();
        assert_eq!(cols.len(), 4);
        assert!(cols.contains(&"new_col".to_string()));

        let values = new_df.get_column_numeric_values("new_col").unwrap();
        assert_eq!(values, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_insert_column_wrong_length() {
        let df = create_test_df();
        let result = df.insert_column(1, "bad_col", vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_sum("a", 3, None).unwrap();

        assert!(result[0].is_nan()); // Window not full
        assert!(result[1].is_nan()); // Window not full
        assert_eq!(result[2], 6.0); // 1 + 2 + 3
        assert_eq!(result[3], 9.0); // 2 + 3 + 4
        assert_eq!(result[4], 12.0); // 3 + 4 + 5
    }

    #[test]
    fn test_rolling_sum_min_periods() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_sum("a", 3, Some(1)).unwrap();

        assert_eq!(result[0], 1.0); // Only 1 value, min_periods=1
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[2], 6.0); // 1 + 2 + 3
        assert_eq!(result[3], 9.0); // 2 + 3 + 4
        assert_eq!(result[4], 12.0); // 3 + 4 + 5
    }

    #[test]
    fn test_rolling_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_mean("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // (1 + 2 + 3) / 3
        assert_eq!(result[3], 3.0); // (2 + 3 + 4) / 3
        assert_eq!(result[4], 4.0); // (3 + 4 + 5) / 3
    }

    #[test]
    fn test_rolling_mean_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_mean("a", 3, Some(2)).unwrap();

        // Window [1] has only 1 valid value, min_periods=2 means NaN
        assert!(result[0].is_nan());
        // Window [1, NaN] has only 1 valid value, min_periods=2 means NaN
        assert!(result[1].is_nan());
        // Window [1, NaN, 3] has 2 valid values: (1 + 3) / 2 = 2
        assert_eq!(result[2], 2.0);
        // Window [NaN, 3, 4] has 2 valid values: (3 + 4) / 2 = 3.5
        assert!((result[3] - 3.5).abs() < 0.001);
        // Window [3, 4, 5] has 3 valid values: (3 + 4 + 5) / 3 = 4
        assert_eq!(result[4], 4.0);
    }

    #[test]
    fn test_rolling_std() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_std("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 0.001); // std of [1, 2, 3]
        assert!((result[3] - 1.0).abs() < 0.001); // std of [2, 3, 4]
        assert!((result[4] - 1.0).abs() < 0.001); // std of [3, 4, 5]
    }

    #[test]
    fn test_rolling_min() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_min("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0); // min of [3, 1, 4]
        assert_eq!(result[3], 1.0); // min of [1, 4, 1]
        assert_eq!(result[4], 1.0); // min of [4, 1, 5]
    }

    #[test]
    fn test_rolling_max() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_max("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 4.0); // max of [3, 1, 4]
        assert_eq!(result[3], 4.0); // max of [1, 4, 1]
        assert_eq!(result[4], 5.0); // max of [4, 1, 5]
    }

    #[test]
    fn test_rolling_var() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_var("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 0.0001); // var of [1, 2, 3] = 1.0
        assert!((result[3] - 1.0).abs() < 0.0001); // var of [2, 3, 4] = 1.0
        assert!((result[4] - 1.0).abs() < 0.0001); // var of [3, 4, 5] = 1.0
    }

    #[test]
    fn test_rolling_median() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.rolling_median("a", 3, None).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 3.0); // median of [3, 1, 4] sorted = [1, 3, 4]
        assert_eq!(result[3], 1.0); // median of [1, 4, 1] sorted = [1, 1, 4]
        assert_eq!(result[4], 4.0); // median of [4, 1, 5] sorted = [1, 4, 5]
    }

    #[test]
    fn test_rolling_count() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.rolling_count("a", 3).unwrap();

        assert_eq!(result[0], 1); // window [1.0]
        assert_eq!(result[1], 1); // window [1.0, NaN]
        assert_eq!(result[2], 2); // window [1.0, NaN, 3.0]
        assert_eq!(result[3], 1); // window [NaN, 3.0, NaN]
        assert_eq!(result[4], 2); // window [3.0, NaN, 5.0]
    }

    #[test]
    fn test_rolling_apply() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        // Custom function: sum of squares
        let result = df
            .rolling_apply("a", 3, |vals| vals.iter().map(|v| v * v).sum(), None)
            .unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 14.0); // 1^2 + 2^2 + 3^2 = 14
        assert_eq!(result[3], 29.0); // 2^2 + 3^2 + 4^2 = 29
        assert_eq!(result[4], 50.0); // 3^2 + 4^2 + 5^2 = 50
    }

    #[test]
    fn test_expanding_var() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_var("a", 2).unwrap();

        assert!(result[0].is_nan());
        assert!((result[1] - 0.5).abs() < 0.0001); // var of [1, 2] = 0.5
        assert!((result[2] - 1.0).abs() < 0.0001); // var of [1, 2, 3] = 1.0
        assert!((result[3] - 1.6667).abs() < 0.001); // var of [1, 2, 3, 4] ≈ 1.667
        assert!((result[4] - 2.5).abs() < 0.0001); // var of [1, 2, 3, 4, 5] = 2.5
    }

    #[test]
    fn test_expanding_apply() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        // Custom function: product of all values
        let result = df
            .expanding_apply("a", |vals| vals.iter().product(), 1)
            .unwrap();

        assert_eq!(result[0], 1.0); // 1
        assert_eq!(result[1], 2.0); // 1 * 2
        assert_eq!(result[2], 6.0); // 1 * 2 * 3
        assert_eq!(result[3], 24.0); // 1 * 2 * 3 * 4
        assert_eq!(result[4], 120.0); // 1 * 2 * 3 * 4 * 5
    }

    #[test]
    fn test_cumcount() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.cumcount("a").unwrap();

        assert_eq!(result, vec![1, 1, 2, 2, 3]); // Count increments only for non-NaN
    }

    #[test]
    fn test_nth_positive() {
        let df = create_test_df();
        let row = df.nth(2).unwrap();

        assert_eq!(row.get("a").unwrap(), "3");
        assert_eq!(row.get("name").unwrap(), "Charlie");
    }

    #[test]
    fn test_nth_negative() {
        let df = create_test_df();
        let row = df.nth(-1).unwrap(); // Last row

        assert_eq!(row.get("a").unwrap(), "5");
        assert_eq!(row.get("name").unwrap(), "Eve");
    }

    #[test]
    fn test_nth_negative_second() {
        let df = create_test_df();
        let row = df.nth(-2).unwrap(); // Second to last row

        assert_eq!(row.get("a").unwrap(), "4");
        assert_eq!(row.get("name").unwrap(), "David");
    }

    #[test]
    fn test_nth_out_of_bounds() {
        let df = create_test_df();
        let result = df.nth(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform() {
        let df = create_test_df();
        let result = df.transform("a", |x| x * 2.0).unwrap();

        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_transform_preserves_other_columns() {
        let df = create_test_df();
        let result = df.transform("a", |x| x * 2.0).unwrap();

        // Other columns should be unchanged
        let b_values = result.get_column_numeric_values("b").unwrap();
        assert_eq!(b_values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_crosstab() {
        let mut df = DataFrame::new();
        df.add_column(
            "gender".to_string(),
            Series::new(
                vec![
                    "M".to_string(),
                    "F".to_string(),
                    "M".to_string(),
                    "F".to_string(),
                    "M".to_string(),
                ],
                Some("gender".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.crosstab("gender", "category").unwrap();

        assert!(result.contains_column("gender"));
        assert!(result.contains_column("A"));
        assert!(result.contains_column("B"));
    }

    #[test]
    fn test_expanding_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_sum("a", 1).unwrap();

        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0); // 1 + 2
        assert_eq!(result[2], 6.0); // 1 + 2 + 3
        assert_eq!(result[3], 10.0); // 1 + 2 + 3 + 4
        assert_eq!(result[4], 15.0); // 1 + 2 + 3 + 4 + 5
    }

    #[test]
    fn test_expanding_sum_min_periods() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_sum("a", 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 6.0); // 1 + 2 + 3
        assert_eq!(result[3], 10.0); // 1 + 2 + 3 + 4
        assert_eq!(result[4], 15.0);
    }

    #[test]
    fn test_expanding_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, 4.0, 6.0, 8.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_mean("a", 1).unwrap();

        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0); // (2 + 4) / 2
        assert_eq!(result[2], 4.0); // (2 + 4 + 6) / 3
        assert_eq!(result[3], 5.0); // (2 + 4 + 6 + 8) / 4
    }

    #[test]
    fn test_expanding_std() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_std("a", 2).unwrap();

        assert!(result[0].is_nan()); // Need at least 2 values
        assert!((result[1] - 0.7071).abs() < 0.001); // std of [1, 2]
    }

    #[test]
    fn test_expanding_min() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_min("a", 1).unwrap();

        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 1.0); // min(3, 1)
        assert_eq!(result[2], 1.0); // min(3, 1, 4)
        assert_eq!(result[3], 1.0); // min(3, 1, 4, 1)
        assert_eq!(result[4], 1.0); // min(3, 1, 4, 1, 5)
    }

    #[test]
    fn test_expanding_max() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.expanding_max("a", 1).unwrap();

        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 3.0); // max(3, 1)
        assert_eq!(result[2], 4.0); // max(3, 1, 4)
        assert_eq!(result[3], 4.0); // max(3, 1, 4, 1)
        assert_eq!(result[4], 5.0); // max(3, 1, 4, 1, 5)
    }

    #[test]
    fn test_align() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let (aligned1, aligned2) = df1.align(&df2).unwrap();

        // Both should have columns 'a' and 'b'
        assert!(aligned1.contains_column("a") || aligned1.contains_column("b"));
        assert!(aligned2.contains_column("a") || aligned2.contains_column("b"));
    }

    #[test]
    fn test_reindex_columns() {
        let df = create_test_df();
        let result = df.reindex_columns(&["b", "a"]).unwrap();

        let cols = result.column_names();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], "b");
        assert_eq!(cols[1], "a");
    }

    #[test]
    fn test_reindex_columns_with_missing() {
        let df = create_test_df();
        let result = df.reindex_columns(&["a", "nonexistent", "b"]).unwrap();

        let cols = result.column_names();
        assert_eq!(cols.len(), 3);
        assert!(result.contains_column("nonexistent")); // Should be filled with NaN
    }

    #[test]
    fn test_value_range() {
        let df = create_test_df();
        let (min, max) = df.value_range("a").unwrap();

        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_value_range_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, 2.0, 5.0, f64::NAN, 1.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let (min, max) = df.value_range("a").unwrap();

        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_zscore() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.zscore("a").unwrap();

        // Mean = 3, std = 1.5811
        assert!(result[0] < 0.0); // Below mean
        assert!((result[2]).abs() < 0.001); // At mean, z-score should be ~0
        assert!(result[4] > 0.0); // Above mean
    }

    #[test]
    fn test_zscore_insufficient_values() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.zscore("a");
        assert!(result.is_err()); // Need at least 2 values
    }

    #[test]
    fn test_normalize() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 50.0, 100.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.normalize("a").unwrap();

        assert_eq!(result[0], 0.0); // (0 - 0) / (100 - 0) = 0
        assert_eq!(result[1], 0.5); // (50 - 0) / (100 - 0) = 0.5
        assert_eq!(result[2], 1.0); // (100 - 0) / (100 - 0) = 1
    }

    #[test]
    fn test_normalize_with_negative() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-10.0, 0.0, 10.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.normalize("a").unwrap();

        assert_eq!(result[0], 0.0); // min
        assert_eq!(result[1], 0.5); // middle
        assert_eq!(result[2], 1.0); // max
    }

    #[test]
    fn test_normalize_constant_values() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 5.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.normalize("a");
        assert!(result.is_err()); // Range is zero
    }

    #[test]
    fn test_cut() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.5, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.cut("a", 2).unwrap();

        assert_eq!(result.len(), 4);
        // Values should be binned into 2 intervals
        assert!(result[0].contains("1.00") || result[0].contains("("));
    }

    #[test]
    fn test_cut_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.cut("a", 2).unwrap();

        assert_eq!(result[1], "NaN");
    }

    #[test]
    fn test_cut_zero_bins() {
        let df = create_test_df();
        let result = df.cut("a", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_qcut() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.qcut("a", 4).unwrap(); // Quartiles

        assert_eq!(result.len(), 8);
        // Should have Q1, Q2, Q3, Q4 labels
        assert!(result.iter().any(|s| s.starts_with("Q")));
    }

    #[test]
    fn test_qcut_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.qcut("a", 2).unwrap();

        assert_eq!(result[1], "NaN");
    }

    #[test]
    fn test_qcut_zero_quantiles() {
        let df = create_test_df();
        let result = df.qcut("a", 0);
        assert!(result.is_err());
    }

    // ============== Tests for additional methods ==============

    #[test]
    fn test_stack() {
        let df = create_test_df();
        let result = df.stack(Some(&["a", "b"])).unwrap();

        assert!(result.contains_column("row_index"));
        assert!(result.contains_column("variable"));
        assert!(result.contains_column("value"));
        // 5 rows * 2 columns = 10 stacked rows
        assert_eq!(result.row_count(), 10);
    }

    #[test]
    fn test_stack_all_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "y".to_string(),
            Series::new(vec![3.0, 4.0], Some("y".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.stack(None).unwrap();
        assert_eq!(result.row_count(), 4); // 2 rows * 2 columns
    }

    #[test]
    fn test_unstack() {
        let mut df = DataFrame::new();
        df.add_column(
            "idx".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
                Some("idx".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "col".to_string(),
            Series::new(
                vec![
                    "X".to_string(),
                    "Y".to_string(),
                    "X".to_string(),
                    "Y".to_string(),
                ],
                Some("col".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "val".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("val".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.unstack("idx", "col", "val").unwrap();

        assert!(result.contains_column("idx"));
        assert!(result.contains_column("X"));
        assert!(result.contains_column("Y"));
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_pivot() {
        let mut df = DataFrame::new();
        df.add_column(
            "date".to_string(),
            Series::new(
                vec![
                    "Mon".to_string(),
                    "Mon".to_string(),
                    "Tue".to_string(),
                    "Tue".to_string(),
                ],
                Some("date".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "type".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                ],
                Some("type".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.pivot("date", "type", "value").unwrap();

        assert!(result.contains_column("date"));
        assert!(result.contains_column("A"));
        assert!(result.contains_column("B"));
    }

    #[test]
    fn test_astype_to_int() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.7, 2.3, 3.9], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.astype("a", "int64").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_astype_to_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.astype("a", "string").unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values[0], "1");
        assert_eq!(values[1], "2");
        assert_eq!(values[2], "3");
    }

    #[test]
    fn test_astype_to_bool() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 5.0, 0.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.astype("a", "bool").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_applymap() {
        let df = create_test_df();
        let result = df.applymap(|x| x * 2.0).unwrap();

        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let b_values = result.get_column_numeric_values("b").unwrap();
        assert_eq!(b_values, vec![10.0, 8.0, 6.0, 4.0, 2.0]);
    }

    #[test]
    fn test_agg_multiple() {
        let df = create_test_df();
        let result = df
            .agg("a", &["sum", "mean", "min", "max", "count"])
            .unwrap();

        assert_eq!(result.get("sum").unwrap(), &15.0);
        assert_eq!(result.get("mean").unwrap(), &3.0);
        assert_eq!(result.get("min").unwrap(), &1.0);
        assert_eq!(result.get("max").unwrap(), &5.0);
        assert_eq!(result.get("count").unwrap(), &5.0);
    }

    #[test]
    fn test_agg_statistics() {
        let df = create_test_df();
        let result = df.agg("a", &["std", "var", "median"]).unwrap();

        assert!((result.get("std").unwrap() - 1.5811).abs() < 0.001);
        assert_eq!(result.get("var").unwrap(), &2.5);
        assert_eq!(result.get("median").unwrap(), &3.0);
    }

    #[test]
    fn test_dtypes() {
        let df = create_test_df();
        let dtypes = df.dtypes();

        assert!(dtypes
            .iter()
            .any(|(name, dtype)| name == "a" && dtype == "float64"));
        assert!(dtypes
            .iter()
            .any(|(name, dtype)| name == "name" && dtype == "object"));
    }

    #[test]
    fn test_set_values() {
        let df = create_test_df();
        let result = df
            .set_values("a", &[0, 2, 4], &[100.0, 300.0, 500.0])
            .unwrap();

        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![100.0, 2.0, 300.0, 4.0, 500.0]);
    }

    #[test]
    fn test_set_values_mismatch() {
        let df = create_test_df();
        let result = df.set_values("a", &[0, 1], &[100.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_query_eq() {
        let df = create_test_df();
        let result = df.query_eq("a", 3.0).unwrap();
        assert_eq!(result.row_count(), 1);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values[0], 3.0);
    }

    #[test]
    fn test_query_gt() {
        let df = create_test_df();
        let result = df.query_gt("a", 3.0).unwrap();
        assert_eq!(result.row_count(), 2);
        let values = result.get_column_numeric_values("a").unwrap();
        assert!(values.iter().all(|&v| v > 3.0));
    }

    #[test]
    fn test_query_lt() {
        let df = create_test_df();
        let result = df.query_lt("a", 3.0).unwrap();
        assert_eq!(result.row_count(), 2);
        let values = result.get_column_numeric_values("a").unwrap();
        assert!(values.iter().all(|&v| v < 3.0));
    }

    #[test]
    fn test_query_contains() {
        let df = create_test_df();
        let result = df.query_contains("name", "li").unwrap();
        assert_eq!(result.row_count(), 2); // Alice and Charlie
    }

    #[test]
    fn test_select_columns() {
        let df = create_test_df();
        let result = df.select_columns(&["a", "name"]).unwrap();

        // Verify data can be retrieved from selected columns
        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let name_values = result.get_column_string_values("name").unwrap();
        assert_eq!(name_values[0], "Alice");

        // Check that non-selected column is missing
        assert!(result.get_column_numeric_values("b").is_err());
    }

    #[test]
    fn test_select_columns_nonexistent() {
        let df = create_test_df();
        let result = df.select_columns(&["a", "nonexistent"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_scalar() {
        let df = create_test_df();
        let result = df.add_scalar("a", 10.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let df = create_test_df();
        let result = df.mul_scalar("a", 2.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_sub_scalar() {
        let df = create_test_df();
        let result = df.sub_scalar("a", 1.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_div_scalar() {
        let df = create_test_df();
        let result = df.div_scalar("a", 2.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_pow() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.pow("a", 2.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_sqrt() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![4.0, 9.0, 16.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.sqrt("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_log() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![
                    1.0,
                    std::f64::consts::E,
                    std::f64::consts::E * std::f64::consts::E,
                ],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.log("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert!((values[0] - 0.0).abs() < 0.001);
        assert!((values[1] - 1.0).abs() < 0.001);
        assert!((values[2] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_exp() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.exp("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert!((values[0] - 1.0).abs() < 0.001);
        assert!((values[1] - std::f64::consts::E).abs() < 0.001);
    }

    #[test]
    fn test_col_add() {
        let df = create_test_df();
        let result = df.col_add("a", "b", "sum").unwrap();

        assert!(result.contains_column("sum"));
        let values = result.get_column_numeric_values("sum").unwrap();
        assert_eq!(values, vec![6.0, 6.0, 6.0, 6.0, 6.0]); // a + b = constant 6
    }

    #[test]
    fn test_col_mul() {
        let df = create_test_df();
        let result = df.col_mul("a", "b", "product").unwrap();

        assert!(result.contains_column("product"));
        let values = result.get_column_numeric_values("product").unwrap();
        assert_eq!(values, vec![5.0, 8.0, 9.0, 8.0, 5.0]); // 1*5, 2*4, 3*3, 4*2, 5*1
    }

    #[test]
    fn test_col_sub() {
        let df = create_test_df();
        let result = df.col_sub("a", "b", "diff").unwrap();

        assert!(result.contains_column("diff"));
        let values = result.get_column_numeric_values("diff").unwrap();
        assert_eq!(values, vec![-4.0, -2.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_col_div() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![2.0, 4.0, 5.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.col_div("a", "b", "ratio").unwrap();

        let values = result.get_column_numeric_values("ratio").unwrap();
        assert_eq!(values, vec![5.0, 5.0, 6.0]);
    }

    // === Tests for new pandas-compatible methods ===

    #[test]
    fn test_iterrows() {
        let df = create_test_df();
        let rows = df.iterrows();

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].0, 0); // First row index is 0

        // Check first row data
        let first_row = &rows[0].1;
        assert!(first_row.contains_key("a"));
        if let SeriesValue::Float(val) = first_row.get("a").unwrap() {
            assert_eq!(*val, 1.0);
        }
    }

    #[test]
    fn test_at() {
        let df = create_test_df();

        // Test numeric column
        let val = df.at(2, "a").unwrap();
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 3.0);
        }

        // Test string column
        let val = df.at(0, "name").unwrap();
        if let SeriesValue::String(s) = val {
            assert_eq!(s, "Alice");
        }

        // Test out of bounds
        assert!(df.at(100, "a").is_err());
    }

    #[test]
    fn test_iat() {
        let df = create_test_df();

        // Access first column, third row
        let val = df.iat(2, 0).unwrap();
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 3.0);
        }

        // Test out of bounds
        assert!(df.iat(0, 100).is_err());
    }

    #[test]
    fn test_drop_rows() {
        let df = create_test_df();
        let result = df.drop_rows(&[1, 3]).unwrap();

        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_set_index() {
        let df = create_test_df();
        let (result, index) = df.set_index("name", true).unwrap();

        assert_eq!(index.len(), 5);
        assert_eq!(index[0], "Alice");
        assert!(!result.contains_column("name")); // Dropped
    }

    #[test]
    fn test_reset_index() {
        let df = create_test_df();
        let index_vals: Vec<String> = vec![
            "idx0".to_string(),
            "idx1".to_string(),
            "idx2".to_string(),
            "idx3".to_string(),
            "idx4".to_string(),
        ];
        let result = df.reset_index(Some(&index_vals), "index").unwrap();

        assert!(result.contains_column("index"));
        let vals = result.get_column_string_values("index").unwrap();
        assert_eq!(vals[0], "idx0");
    }

    #[test]
    fn test_to_records() {
        let df = create_test_df();
        let records = df.to_records();

        assert_eq!(records.len(), 5);
        assert!(records[0].contains_key("a"));
    }

    #[test]
    fn test_items() {
        let df = create_test_df();
        let items = df.items();

        assert!(!items.is_empty());
        // Find "a" column
        let a_item = items.iter().find(|(name, _)| name == "a").unwrap();
        assert_eq!(a_item.1.len(), 5);
    }

    #[test]
    fn test_update() {
        let df = create_test_df();

        let mut other = DataFrame::new();
        other
            .add_column(
                "a".to_string(),
                Series::new(vec![100.0, 200.0], Some("a".to_string())).unwrap(),
            )
            .unwrap();

        let result = df.update(&other).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values[0], 100.0);
        assert_eq!(values[1], 200.0);
        assert_eq!(values[2], 3.0); // Unchanged
    }

    #[test]
    fn test_shape() {
        let df = create_test_df();
        let (rows, cols) = df.shape();

        assert_eq!(rows, 5);
        assert_eq!(cols, 3);
    }

    #[test]
    fn test_size() {
        let df = create_test_df();
        assert_eq!(df.size(), 15); // 5 rows * 3 columns
    }

    #[test]
    fn test_empty() {
        let df = create_test_df();
        assert!(!df.empty());

        let empty_df = DataFrame::new();
        assert!(empty_df.empty());
    }

    #[test]
    fn test_first_last_row() {
        let df = create_test_df();

        let first = df.first_row().unwrap();
        if let SeriesValue::Float(v) = first.get("a").unwrap() {
            assert_eq!(*v, 1.0);
        }

        let last = df.last_row().unwrap();
        if let SeriesValue::Float(v) = last.get("a").unwrap() {
            assert_eq!(*v, 5.0);
        }
    }

    #[test]
    fn test_get_value() {
        let df = create_test_df();

        let val = df.get_value(0, "a", SeriesValue::Float(0.0));
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 1.0);
        }

        // Test default
        let val = df.get_value(100, "a", SeriesValue::Float(999.0));
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 999.0);
        }
    }

    #[test]
    fn test_get_column_by_index() {
        let df = create_test_df();
        let (name, values) = df.get_column_by_index(0).unwrap();

        assert!(!name.is_empty());
        assert_eq!(values.len(), 5);

        // Test out of bounds
        assert!(df.get_column_by_index(100).is_err());
    }

    #[test]
    fn test_swap_columns() {
        let df = create_test_df();
        let original_a = df.get_column_numeric_values("a").unwrap();
        let original_b = df.get_column_numeric_values("b").unwrap();

        let result = df.swap_columns("a", "b").unwrap();

        // After swap, column "a" should have "b"'s original values
        let new_a = result.get_column_numeric_values("a").unwrap();
        let new_b = result.get_column_numeric_values("b").unwrap();

        assert_eq!(new_a, original_b);
        assert_eq!(new_b, original_a);
    }

    #[test]
    fn test_sort_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "c".to_string(),
            Series::new(vec![1.0, 2.0], Some("c".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 6.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.sort_columns(true).unwrap();
        let cols = result.column_names();

        assert_eq!(cols[0], "a");
        assert_eq!(cols[1], "b");
        assert_eq!(cols[2], "c");
    }

    #[test]
    fn test_rename_column() {
        let df = create_test_df();
        let result = df.rename_column("a", "new_a").unwrap();

        assert!(result.contains_column("new_a"));
        assert!(!result.contains_column("a"));
    }

    #[test]
    fn test_to_categorical() {
        let df = create_test_df();
        let (result, mapping) = df.to_categorical("name").unwrap();

        // Check that codes are numeric
        let codes = result.get_column_numeric_values("name").unwrap();
        assert_eq!(codes.len(), 5);

        // Check mapping exists
        assert!(mapping.contains_key("Alice"));
    }

    #[test]
    fn test_row_hash() {
        let df = create_test_df();
        let hashes = df.row_hash();

        assert_eq!(hashes.len(), 5);
        // All rows should have different hashes
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_sample_frac() {
        let df = create_test_df();
        let result = df.sample_frac(0.4, false).unwrap();

        // 0.4 of 5 rows = 2 rows
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_take() {
        let df = create_test_df();
        let result = df.take(&[0, 2, 4]).unwrap();

        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_duplicated_rows() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let dupes = df.duplicated_rows(None, "first").unwrap();
        // First occurrence is not duplicate, subsequent ones are
        assert_eq!(dupes.len(), 5);
    }

    #[test]
    fn test_get_column_as_f64() {
        let df = create_test_df();
        let values: Vec<f64> = PandasCompatExt::get_column_as_f64(&df, "a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_get_column_as_string() {
        let df = create_test_df();

        // String column
        let values = df.get_column_as_string("name").unwrap();
        assert_eq!(values[0], "Alice");

        // Numeric column converted to string
        let values = df.get_column_as_string("a").unwrap();
        assert_eq!(values[0], "1");
    }

    #[test]
    fn test_corr_columns() {
        let df = create_test_df();
        let corr = df.corr_columns("a", "b").unwrap();

        // a and b have perfect negative correlation
        assert!((corr - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_cov_columns() {
        let df = create_test_df();
        let cov = df.cov_columns("a", "b").unwrap();

        // Should be negative (inverse relationship)
        assert!(cov < 0.0);
    }

    #[test]
    fn test_var_column() {
        let df = create_test_df();
        let var = df.var_column("a", 0).unwrap();

        // Variance of [1,2,3,4,5] with ddof=0 is 2.0
        assert_eq!(var, 2.0);
    }

    #[test]
    fn test_std_column() {
        let df = create_test_df();
        let std = df.std_column("a", 0).unwrap();

        // Std of [1,2,3,4,5] with ddof=0 is sqrt(2)
        assert!((std - std::f64::consts::SQRT_2).abs() < 0.0001);
    }

    #[test]
    fn test_str_lower() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["HELLO".to_string(), "WORLD".to_string()],
                Some("text".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_lower("text").unwrap();
        let values = result.get_column_string_values("text").unwrap();

        assert_eq!(values[0], "hello");
        assert_eq!(values[1], "world");
    }

    #[test]
    fn test_str_upper() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["hello".to_string(), "world".to_string()],
                Some("text".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_upper("text").unwrap();
        let values = result.get_column_string_values("text").unwrap();

        assert_eq!(values[0], "HELLO");
        assert_eq!(values[1], "WORLD");
    }

    #[test]
    fn test_str_strip() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["  hello  ".to_string(), "  world  ".to_string()],
                Some("text".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_strip("text").unwrap();
        let values = result.get_column_string_values("text").unwrap();

        assert_eq!(values[0], "hello");
        assert_eq!(values[1], "world");
    }

    #[test]
    fn test_str_contains() {
        let df = create_test_df();
        let contains = df.str_contains("name", "li").unwrap();

        // Alice and Charlie contain "li"
        assert_eq!(contains.iter().filter(|&&b| b).count(), 2);
    }

    #[test]
    fn test_str_replace() {
        let df = create_test_df();
        let result = df.str_replace("name", "Alice", "Alicia").unwrap();
        let values = result.get_column_string_values("name").unwrap();

        assert_eq!(values[0], "Alicia");
    }

    #[test]
    fn test_str_split() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["a,b,c".to_string(), "d,e,f".to_string()],
                Some("text".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let splits = df.str_split("text", ",").unwrap();

        assert_eq!(splits[0], vec!["a", "b", "c"]);
        assert_eq!(splits[1], vec!["d", "e", "f"]);
    }

    #[test]
    fn test_str_len() {
        let df = create_test_df();
        let lengths = df.str_len("name").unwrap();

        assert_eq!(lengths[0], 5); // "Alice" = 5
        assert_eq!(lengths[1], 3); // "Bob" = 3
    }

    #[test]
    fn test_combine() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df1
            .combine(&df2, |v1, v2| v1.unwrap_or(0.0) + v2.unwrap_or(0.0))
            .unwrap();

        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_lookup() {
        let df = create_test_df();

        let mut lookup_df = DataFrame::new();
        lookup_df
            .add_column(
                "key".to_string(),
                Series::new(
                    vec!["Alice".to_string(), "Bob".to_string()],
                    Some("key".to_string()),
                )
                .unwrap(),
            )
            .unwrap();
        lookup_df
            .add_column(
                "value".to_string(),
                Series::new(
                    vec!["A".to_string(), "B".to_string()],
                    Some("value".to_string()),
                )
                .unwrap(),
            )
            .unwrap();

        let result = df.lookup("name", &lookup_df, "key", "value").unwrap();
        assert!(result.contains_column("value_result"));
    }

    #[test]
    fn test_sem() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.sem("a", 1).unwrap();
        // SEM = std / sqrt(n) = sqrt(2.5) / sqrt(5) ≈ 0.707
        assert!((result - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_mad() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.mad("a").unwrap();
        // MAD = mean(|x - mean(x)|) = mean(|1-3|, |2-3|, |3-3|, |4-3|, |5-3|) = (2+1+0+1+2)/5 = 1.2
        assert!((result - 1.2).abs() < 0.0001);
    }

    #[test]
    fn test_ffill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.ffill("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 1.0); // filled forward
        assert_eq!(values[2], 1.0); // filled forward
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 4.0); // filled forward
    }

    #[test]
    fn test_bfill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.bfill("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values[0], 3.0); // filled backward
        assert_eq!(values[1], 3.0); // filled backward
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 5.0); // filled backward
        assert_eq!(values[4], 5.0);
    }

    #[test]
    fn test_pct_rank() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.pct_rank("a").unwrap();

        // Sorted values: 1.0, 1.0, 3.0, 4.0, 5.0
        // Ranks should be 0.0, 0.0, 0.5, 0.75, 1.0
        assert!((result[1] - 0.0).abs() < 0.01); // smallest
        assert!((result[4] - 1.0).abs() < 0.01); // largest
    }

    #[test]
    fn test_abs_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-1.0, -2.0, 3.0, -4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.abs_column("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_round_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.234, 2.567, 3.891], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.round_column("a", 2).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();

        assert_eq!(values, vec![1.23, 2.57, 3.89]);
    }

    #[test]
    fn test_argmax_argmin() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        assert_eq!(df.argmax("a").unwrap(), 4); // 5.0 at index 4
        assert_eq!(df.argmin("a").unwrap(), 1); // 1.0 at index 1
    }

    #[test]
    fn test_gt_lt_ge_le() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let gt = df.gt("a", 3.0).unwrap();
        assert_eq!(gt, vec![false, false, false, true, true]);

        let ge = df.ge("a", 3.0).unwrap();
        assert_eq!(ge, vec![false, false, true, true, true]);

        let lt = df.lt("a", 3.0).unwrap();
        assert_eq!(lt, vec![true, true, false, false, false]);

        let le = df.le("a", 3.0).unwrap();
        assert_eq!(le, vec![true, true, true, false, false]);
    }

    #[test]
    fn test_eq_ne_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 2.0, 1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let eq = df.eq_value("a", 2.0).unwrap();
        assert_eq!(eq, vec![false, true, false, true, false]);

        let ne = df.ne_value("a", 2.0).unwrap();
        assert_eq!(ne, vec![true, false, true, false, true]);
    }

    #[test]
    fn test_clip_lower_upper() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let clipped = df.clip_lower("a", 2.5).unwrap();
        let values = clipped.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.5, 2.5, 3.0, 4.0, 5.0]);

        let clipped = df.clip_upper("a", 3.5).unwrap();
        let values = clipped.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 3.5, 3.5]);
    }

    #[test]
    fn test_any_all_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "c".to_string(),
            Series::new(vec![0.0, 0.0, 0.0], Some("c".to_string())).unwrap(),
        )
        .unwrap();

        assert!(df.any_column("a").unwrap()); // has non-zero
        assert!(!df.all_column("a").unwrap()); // has zero
        assert!(df.all_column("b").unwrap()); // all non-zero
        assert!(!df.any_column("c").unwrap()); // all zero
    }

    #[test]
    fn test_count_na() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(df.count_na("a").unwrap(), 3);
    }

    #[test]
    fn test_prod() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        assert_eq!(df.prod("a").unwrap(), 24.0); // 1 * 2 * 3 * 4 = 24
    }

    #[test]
    fn test_coalesce() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, f64::NAN], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.coalesce("a", "b", "c").unwrap();
        let values = result.get_column_numeric_values("c").unwrap();

        assert_eq!(values[0], 1.0); // from a (not NaN)
        assert_eq!(values[1], 20.0); // from b (a is NaN)
        assert_eq!(values[2], 3.0); // from a
        assert_eq!(values[3], 40.0); // from b (a is NaN)
    }

    #[test]
    fn test_first_last_valid() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(df.first_valid("a").unwrap(), 3.0);
        assert_eq!(df.last_valid("a").unwrap(), 4.0);
    }

    #[test]
    fn test_column_arithmetic() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.add_columns("a", "b", "sum").unwrap();
        assert_eq!(
            result.get_column_numeric_values("sum").unwrap(),
            vec![11.0, 22.0, 33.0]
        );

        let result = df.sub_columns("b", "a", "diff").unwrap();
        assert_eq!(
            result.get_column_numeric_values("diff").unwrap(),
            vec![9.0, 18.0, 27.0]
        );

        let result = df.mul_columns("a", "b", "prod").unwrap();
        assert_eq!(
            result.get_column_numeric_values("prod").unwrap(),
            vec![10.0, 40.0, 90.0]
        );

        let result = df.div_columns("b", "a", "quot").unwrap();
        assert_eq!(
            result.get_column_numeric_values("quot").unwrap(),
            vec![10.0, 10.0, 10.0]
        );
    }

    #[test]
    fn test_mod_floordiv() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![7.0, 10.0, 15.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.mod_column("a", 3.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 1.0, 0.0]);

        let result = df.floordiv("a", 3.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_neg_sign() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-2.0, 0.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.neg("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, 0.0, -3.0]);

        let signs = df.sign("a").unwrap();
        assert_eq!(signs, vec![-1, 0, 1]);
    }

    #[test]
    fn test_is_finite_infinite() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let finite = df.is_finite("a").unwrap();
        assert_eq!(finite, vec![true, false, false, false]);

        let infinite = df.is_infinite("a").unwrap();
        assert_eq!(infinite, vec![false, true, true, false]);
    }

    #[test]
    fn test_replace_inf() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.replace_inf("a", 0.0).unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_str_startswith_endswith() {
        let df = create_test_df();

        let starts = df.str_startswith("name", "A").unwrap();
        assert_eq!(starts[0], true); // Alice
        assert_eq!(starts[1], false); // Bob

        let ends = df.str_endswith("name", "e").unwrap();
        assert_eq!(ends[0], true); // Alice
        assert_eq!(ends[1], false); // Bob
    }

    #[test]
    fn test_str_pad() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["a".to_string(), "bb".to_string(), "ccc".to_string()],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_pad_left("a", 5, '0').unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["0000a", "000bb", "00ccc"]);

        let result = df.str_pad_right("a", 5, '-').unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["a----", "bb---", "ccc--"]);
    }

    #[test]
    fn test_str_slice() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["hello".to_string(), "world".to_string()],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_slice("a", 0, Some(3)).unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["hel", "wor"]);

        let result = df.str_slice("a", 2, None).unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["llo", "rld"]);
    }

    #[test]
    fn test_floor_ceil_trunc() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.7, -1.7, 2.3, -2.3], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.floor("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, -2.0, 2.0, -3.0]);

        let result = df.ceil("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![2.0, -1.0, 3.0, -2.0]);

        let result = df.trunc("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, -1.0, 2.0, -2.0]);
    }

    #[test]
    fn test_fract_reciprocal() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.5, 2.7, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.fract("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert!((values[0] - 0.5).abs() < 0.0001);
        assert!((values[1] - 0.7).abs() < 0.0001);
        assert!((values[2] - 0.0).abs() < 0.0001);

        let result = df.reciprocal("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert!((values[0] - 0.6667).abs() < 0.001);
        assert!((values[1] - 0.3704).abs() < 0.001);
        assert!((values[2] - 0.25).abs() < 0.0001);
    }

    #[test]
    fn test_count_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        assert_eq!(df.count_value("a", 1.0).unwrap(), 3);
        assert_eq!(df.count_value("a", 2.0).unwrap(), 1);
        assert_eq!(df.count_value("a", 5.0).unwrap(), 0);
    }

    #[test]
    fn test_fillna_zero() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, f64::NAN], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.fillna_zero("a").unwrap();
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_nunique_all() {
        let df = create_test_df();
        let result = df.nunique_all().unwrap();

        assert_eq!(result.get("a").unwrap(), &5); // 1, 2, 3, 4, 5
        assert_eq!(result.get("name").unwrap(), &5); // Alice, Bob, Charlie, David, Eve
    }

    #[test]
    fn test_is_between() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.is_between("a", 2.0, 4.0, true).unwrap();
        assert_eq!(result, vec![false, true, true, true, false]);

        let result = df.is_between("a", 2.0, 4.0, false).unwrap();
        assert_eq!(result, vec![false, false, true, false, false]);
    }

    #[test]
    fn test_str_count() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["aaa".to_string(), "aba".to_string(), "bbb".to_string()],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let counts = df.str_count("a", "a").unwrap();
        assert_eq!(counts, vec![3, 2, 0]);
    }

    #[test]
    fn test_str_repeat() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["ab".to_string(), "cd".to_string()],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_repeat("a", 3).unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["ababab", "cdcdcd"]);
    }

    #[test]
    fn test_str_center_zfill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["ab".to_string(), "c".to_string()],
                Some("a".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.str_center("a", 5, '-').unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["-ab--", "--c--"]);

        let result = df.str_zfill("a", 5).unwrap();
        let values = result.get_column_string_values("a").unwrap();
        assert_eq!(values, vec!["000ab", "0000c"]);
    }

    #[test]
    fn test_column_type_detection() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                ],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // Test that we can get numeric values from numeric column
        let numeric_result = df.get_column_numeric_values("a");
        assert!(numeric_result.is_ok(), "Should get numeric values from 'a'");

        // Test that we can get string values from string column
        let string_result = df.get_column_string_values("name");
        assert!(
            string_result.is_ok(),
            "Should get string values from 'name'"
        );
    }

    #[test]
    fn test_has_nulls() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        assert!(df.has_nulls("a").unwrap());
        assert!(!df.has_nulls("b").unwrap());
    }

    #[test]
    fn test_describe_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let stats = df.describe_column("a").unwrap();

        assert_eq!(stats.get("count").unwrap(), &5.0);
        assert!((stats.get("mean").unwrap() - 3.0).abs() < 0.0001);
        assert!((stats.get("min").unwrap() - 1.0).abs() < 0.0001);
        assert!((stats.get("max").unwrap() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_range_abs_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-2.0, 1.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        assert_eq!(df.range("a").unwrap(), 7.0); // 5 - (-2) = 7
        assert_eq!(df.abs_sum("a").unwrap(), 8.0); // |-2| + |1| + |5| = 8
    }

    #[test]
    fn test_is_unique() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 1.0, 2.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        assert!(df.is_unique("a").unwrap());
        assert!(!df.is_unique("b").unwrap());
    }

    #[test]
    fn test_mode_with_count() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 2.0, 3.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let (mode, count) = df.mode_with_count("a").unwrap();
        assert_eq!(mode, 2.0);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_geometric_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 4.0, 8.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.geometric_mean("a").unwrap();
        // Geometric mean of 1,2,4,8 = (1*2*4*8)^(1/4) = 64^0.25 = 2.828...
        assert!((result - 2.828).abs() < 0.01);
    }

    #[test]
    fn test_harmonic_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.harmonic_mean("a").unwrap();
        // Harmonic mean of 1,2,4 = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 = 1.714...
        assert!((result - 1.714).abs() < 0.01);
    }

    #[test]
    fn test_iqr() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.iqr("a").unwrap();
        // IQR should be approximately Q3 - Q1
        assert!(result > 1.0 && result < 3.0);
    }

    #[test]
    fn test_cv() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = df.cv("a").unwrap();
        // CV = std / mean, should be positive
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn test_percentile_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let p50 = df.percentile_value("a", 0.5).unwrap();
        assert!((p50 - 3.0).abs() < 0.5);

        let p0 = df.percentile_value("a", 0.0).unwrap();
        assert_eq!(p0, 1.0);

        let p100 = df.percentile_value("a", 1.0).unwrap();
        assert_eq!(p100, 5.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 100.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        // Trim 20% from each end (removing 1.0 and 100.0)
        let result = df.trimmed_mean("a", 0.2).unwrap();
        // Should be mean of [2.0, 3.0, 4.0] = 3.0
        assert!((result - 3.0).abs() < 0.1);
    }
}

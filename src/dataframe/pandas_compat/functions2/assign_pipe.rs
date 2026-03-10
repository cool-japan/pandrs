//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Pandas compatibility extension trait for DataFrame - assign, pipe, and data manipulation methods
use super::super::trait_def::PandasCompatExt;
use super::super::types::{
    Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue,
};
use super::helpers::select_rows_by_indices;
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

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
        let first = values
            .first()
            .map(|v| {
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
        let mut indexed_values: Vec<(usize, f64)> = col_values
            .into_iter()
            .enumerate()
            .collect();
        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed_values.truncate(n);
        let indices: Vec<usize> = indexed_values.into_iter().map(|(i, _)| i).collect();
        select_rows_by_indices(self, &indices)
    }
    fn nsmallest(&self, n: usize, column: &str) -> Result<DataFrame> {
        let col_values = self.get_column_numeric_values(column)?;
        let mut indexed_values: Vec<(usize, f64)> = col_values
            .into_iter()
            .enumerate()
            .collect();
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
        let mut indexed_values: Vec<(usize, f64)> = values
            .into_iter()
            .enumerate()
            .collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let mut ranks = vec![f64::NAN; n];
        let mut i = 0;
        while i < indexed_values.len() {
            let mut j = i;
            while j < indexed_values.len() && indexed_values[j].1 == indexed_values[i].1
            {
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
                while j < indexed_values.len()
                    && indexed_values[j].1 == indexed_values[i].1
                {
                    ranks[indexed_values[j].0] = dense_rank;
                    j += 1;
                }
                i = j;
            }
        }
        Ok(ranks)
    }
    fn clip(
        &self,
        column: &str,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<DataFrame> {
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
        let result: Vec<bool> = values
            .iter()
            .map(|&v| v >= lower && v <= upper)
            .collect();
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
        result
            .sort_by(|a, b| {
                b.1.cmp(&a.1).then(a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
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
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / count as f64;
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
            return Err(
                Error::Empty("Cannot compute correlation on empty DataFrame".to_string()),
            );
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
                    matrix[i][j] = pearson_correlation(
                        &columns_data[i],
                        &columns_data[j],
                    );
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
            result[i] = if prev == 0.0 { f64::NAN } else { (curr - prev) / prev };
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
    fn replace(
        &self,
        column: &str,
        to_replace: &[&str],
        values: &[&str],
    ) -> Result<DataFrame> {
        if to_replace.len() != values.len() {
            return Err(
                Error::InvalidValue(
                    "to_replace and values must have same length".to_string(),
                ),
            );
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
            return Err(
                Error::InvalidValue(
                    "to_replace and values must have same length".to_string(),
                ),
            );
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
            return Err(
                Error::InvalidValue(
                    format!(
                        "Cannot sample {} rows without replacement from {} rows", n,
                        n_rows
                    ),
                ),
            );
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
                    df.add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
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
                    df.add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    df.add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
                }
            }
        }
        Ok(df)
    }
    fn quantile(&self, column: &str, q: f64) -> Result<f64> {
        if q < 0.0 || q > 1.0 {
            return Err(
                Error::InvalidValue("Quantile must be between 0 and 1".to_string()),
            );
        }
        let values = self.get_column_numeric_values(column)?;
        if values.is_empty() {
            return Err(
                Error::Empty("Cannot compute quantile of empty column".to_string()),
            );
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
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            }
        }
        Ok(df)
    }
    fn fillna_method(&self, column: &str, method: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let filled: Vec<f64> = match method {
            "ffill" | "forward" => {
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
                return Err(
                    Error::InvalidValue(
                        format!(
                            "Invalid fill method: '{}'. Use 'ffill' or 'bfill'.", method
                        ),
                    ),
                );
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
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            }
        }
        Ok(df)
    }
    fn interpolate(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mut interpolated = values.clone();
        let first_valid = values.iter().position(|v| !v.is_nan());
        let last_valid = values.iter().rposition(|v| !v.is_nan());
        if let (Some(first), Some(last)) = (first_valid, last_valid) {
            let mut prev_valid_idx = first;
            let mut prev_valid_val = values[first];
            for i in (first + 1)..=last {
                if !values[i].is_nan() {
                    if i > prev_valid_idx + 1 {
                        let gap_size = (i - prev_valid_idx) as f64;
                        let value_diff = values[i] - prev_valid_val;
                        for j in (prev_valid_idx + 1)..i {
                            let position = (j - prev_valid_idx) as f64;
                            interpolated[j] = prev_valid_val
                                + (value_diff * position / gap_size);
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
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
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
                let valid_values: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
                if !valid_values.is_empty() {
                    let mean = valid_values.iter().sum::<f64>()
                        / valid_values.len() as f64;
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
                let valid_values: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
                if valid_values.len() > 1 {
                    let mean = valid_values.iter().sum::<f64>()
                        / valid_values.len() as f64;
                    let variance: f64 = valid_values
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>() / (valid_values.len() - 1) as f64;
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
                let valid_values: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
                if valid_values.len() > 1 {
                    let mean = valid_values.iter().sum::<f64>()
                        / valid_values.len() as f64;
                    let variance: f64 = valid_values
                        .iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>() / (valid_values.len() - 1) as f64;
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
                let valid_values: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
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
                let valid_values: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
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
        let mut indexed_values: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed_values
            .sort_by(|a, b| {
                let cmp = a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
                if ascending { cmp } else { cmp.reverse() }
            });
        let sorted_indices: Vec<usize> = indexed_values
            .iter()
            .map(|(i, _)| *i)
            .collect();
        select_rows_by_indices(self, &sorted_indices)
    }
    fn sort_by_columns(
        &self,
        columns: &[&str],
        ascending: &[bool],
    ) -> Result<DataFrame> {
        if columns.len() != ascending.len() {
            return Err(
                Error::InvalidValue(
                    "Number of columns must match number of ascending flags".to_string(),
                ),
            );
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
        indices
            .sort_by(|&i, &j| {
                for (col_idx, (&col_name, &asc)) in columns
                    .iter()
                    .zip(ascending.iter())
                    .enumerate()
                {
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
        how: super::super::merge::JoinType,
        suffixes: (&str, &str),
    ) -> Result<DataFrame> {
        super::super::merge::merge(self, other, on, how, suffixes)
    }
    fn where_cond(
        &self,
        column: &str,
        condition: &[bool],
        other: f64,
    ) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        if condition.len() != values.len() {
            return Err(
                Error::InvalidValue(
                    "Condition length must match column length".to_string(),
                ),
            );
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
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            }
        }
        Ok(df)
    }
    fn mask(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        if condition.len() != values.len() {
            return Err(
                Error::InvalidValue(
                    "Condition length must match column length".to_string(),
                ),
            );
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
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                df.add_column(
                    col_name.clone(),
                    Series::new(vals, Some(col_name.clone()))?,
                )?;
            }
        }
        Ok(df)
    }
    fn drop_duplicates(&self, subset: Option<&[&str]>, keep: &str) -> Result<DataFrame> {
        let columns_to_check: Vec<String> = match subset {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => self.column_names(),
        };
        for col in &columns_to_check {
            if !self.contains_column(col) {
                return Err(
                    Error::InvalidValue(
                        format!("Column '{}' not found in DataFrame", col),
                    ),
                );
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
                return Err(
                    Error::InvalidValue(
                        format!(
                            "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                            keep
                        ),
                    ),
                );
            }
        }
        indices_to_keep.sort_unstable();
        select_rows_by_indices(self, &indices_to_keep)
    }
    fn select_dtypes(&self, include: &[&str]) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            let is_numeric = self.get_column_numeric_values(&col_name).is_ok();
            let is_string = !is_numeric
                && self.get_column_string_values(&col_name).is_ok();
            let should_include = include
                .iter()
                .any(|&dtype| {
                    (dtype == "numeric" || dtype == "number" || dtype == "float64"
                        || dtype == "int64") && is_numeric
                        || (dtype == "string" || dtype == "object" || dtype == "str")
                            && is_string
                });
            if should_include {
                if let Ok(values) = self.get_column_numeric_values(&col_name) {
                    result
                        .add_column(
                            col_name.clone(),
                            Series::new(values, Some(col_name.clone()))?,
                        )?;
                } else if let Ok(values) = self.get_column_string_values(&col_name) {
                    result
                        .add_column(
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(values, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
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
        for id_var in id_vars {
            if !self.contains_column(id_var) {
                return Err(
                    Error::InvalidValue(
                        format!("Column '{}' not found in DataFrame", id_var),
                    ),
                );
            }
        }
        let value_columns: Vec<String> = match value_vars {
            Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
            None => {
                let id_set: std::collections::HashSet<&str> = id_vars
                    .iter()
                    .copied()
                    .collect();
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
        for id_var in id_vars {
            if let Ok(values) = self.get_column_numeric_values(id_var) {
                let mut repeated: Vec<f64> = Vec::with_capacity(total_rows);
                for _ in 0..n_value_cols {
                    repeated.extend(values.iter().copied());
                }
                result
                    .add_column(
                        id_var.to_string(),
                        Series::new(repeated, Some(id_var.to_string()))?,
                    )?;
            } else if let Ok(values) = self.get_column_string_values(id_var) {
                let mut repeated: Vec<String> = Vec::with_capacity(total_rows);
                for _ in 0..n_value_cols {
                    repeated.extend(values.iter().cloned());
                }
                result
                    .add_column(
                        id_var.to_string(),
                        Series::new(repeated, Some(id_var.to_string()))?,
                    )?;
            }
        }
        let mut var_values: Vec<String> = Vec::with_capacity(total_rows);
        for col in &value_columns {
            for _ in 0..n_rows {
                var_values.push(col.clone());
            }
        }
        result
            .add_column(
                var_name.to_string(),
                Series::new(var_values, Some(var_name.to_string()))?,
            )?;
        let mut all_values: Vec<f64> = Vec::with_capacity(total_rows);
        for col in &value_columns {
            if let Ok(values) = self.get_column_numeric_values(col) {
                all_values.extend(values);
            } else {
                for _ in 0..n_rows {
                    all_values.push(f64::NAN);
                }
            }
        }
        result
            .add_column(
                value_name.to_string(),
                Series::new(all_values, Some(value_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn explode(&self, column: &str, separator: &str) -> Result<DataFrame> {
        let string_values = self.get_column_string_values(column)?;
        let n_rows = self.row_count();
        let split_values: Vec<Vec<&str>> = string_values
            .iter()
            .map(|v| v.split(separator).map(|s| s.trim()).collect())
            .collect();
        let total_new_rows: usize = split_values.iter().map(|v| v.len().max(1)).sum();
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
        for col_name in self.column_names() {
            if &col_name == column {
                let new_values: Vec<String> = row_mapping
                    .iter()
                    .map(|(_, val)| val.to_string())
                    .collect();
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(new_values, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(values) = self.get_column_numeric_values(&col_name) {
                let new_values: Vec<f64> = row_mapping
                    .iter()
                    .map(|(idx, _)| values[*idx])
                    .collect();
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(new_values, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                let new_values: Vec<String> = row_mapping
                    .iter()
                    .map(|(idx, _)| values[*idx].clone())
                    .collect();
                result
                    .add_column(
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
        for col in &columns_to_check {
            if !self.contains_column(col) {
                return Err(
                    Error::InvalidValue(
                        format!("Column '{}' not found in DataFrame", col),
                    ),
                );
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
        let mut first_occurrence: HashMap<String, usize> = HashMap::new();
        let mut last_occurrence: HashMap<String, usize> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for (idx, key) in row_keys.iter().enumerate() {
            first_occurrence.entry(key.clone()).or_insert(idx);
            last_occurrence.insert(key.clone(), idx);
            *counts.entry(key.clone()).or_insert(0) += 1;
        }
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
                return Err(
                    Error::InvalidValue(
                        format!(
                            "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                            keep
                        ),
                    ),
                );
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
                let valid: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
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
                let mut valid: Vec<f64> = values
                    .iter()
                    .filter(|v| !v.is_nan())
                    .copied()
                    .collect();
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
        let kurtosis = m4 / std_dev.powi(4) - 3.0;
        let adjustment = ((n - 1.0) / ((n - 2.0) * (n - 3.0)))
            * ((n + 1.0) * kurtosis + 6.0);
        Ok(adjustment)
    }
    fn add_prefix(&self, prefix: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            let new_name = format!("{}{}", prefix, col_name);
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result
                    .add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            }
        }
        Ok(result)
    }
    fn add_suffix(&self, suffix: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            let new_name = format!("{}{}", col_name, suffix);
            if let Ok(values) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            } else if let Ok(values) = self.get_column_string_values(&col_name) {
                result
                    .add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
            }
        }
        Ok(result)
    }
    fn filter_by_mask(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.row_count() {
            return Err(
                Error::InvalidValue("Mask length must match number of rows".to_string()),
            );
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
        let max_count = *counts
            .values()
            .max()
            .ok_or_else(|| {
                Error::InsufficientData(
                    "No valid values for mode calculation".to_string(),
                )
            })?;
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
        let max_count = *counts
            .values()
            .max()
            .ok_or_else(|| {
                Error::InsufficientData(
                    "No valid values for mode calculation".to_string(),
                )
            })?;
        let mut modes: Vec<String> = counts
            .iter()
            .filter(|(_, &c)| c == max_count)
            .map(|(k, _)| k.clone())
            .collect();
        modes.sort();
        Ok(modes)
    }
    fn percentile(&self, column: &str, n: f64) -> Result<f64> {
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
                ewma_value = Some(
                    match ewma_value {
                        Some(prev) => alpha * v + (1.0 - alpha) * prev,
                        None => v,
                    },
                );
                result.push(ewma_value.expect("EWMA value just set"));
            }
        }
        Ok(result)
    }
    fn iloc(&self, index: usize) -> Result<HashMap<String, String>> {
        if index >= self.row_count() {
            return Err(
                Error::InvalidValue(
                    format!(
                        "Index {} out of bounds for DataFrame with {} rows", index, self
                        .row_count()
                    ),
                ),
            );
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
            return Err(
                Error::InvalidValue(
                    "Start index must be less than or equal to end index".to_string(),
                ),
            );
        }
        let n_rows = self.row_count();
        let end = end.min(n_rows);
        let start = start.min(n_rows);
        let indices: Vec<usize> = (start..end).collect();
        select_rows_by_indices(self, &indices)
    }
}

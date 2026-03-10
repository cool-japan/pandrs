//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Pandas compatibility extension trait for DataFrame - statistical and aggregation methods
use super::super::trait_def::PandasCompatExt;
use super::super::types::{
    Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue,
};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

impl PandasCompatExt for DataFrame {
    fn info(&self) -> String {
        let mut info = String::new();
        let n_rows = self.row_count();
        let columns = self.column_names();
        let n_cols = columns.len();
        info.push_str(&format!("<DataFrame>\n"));
        info.push_str(
            &format!(
                "RangeIndex: {} entries, 0 to {}\n", n_rows, n_rows.saturating_sub(1)
            ),
        );
        info.push_str(&format!("Data columns (total {} columns):\n", n_cols));
        info.push_str(&format!(" #   Column  Non-Null Count  Dtype\n"));
        info.push_str(&format!("---  ------  --------------  -----\n"));
        for (idx, col) in columns.iter().enumerate() {
            let (non_null, dtype) = if let Ok(values) = self
                .get_column_numeric_values(col)
            {
                let non_null = values.iter().filter(|v| !v.is_nan()).count();
                (non_null, "float64")
            } else if let Ok(values) = self.get_column_string_values(col) {
                let non_null = values.iter().filter(|v| !v.is_empty()).count();
                (non_null, "object")
            } else {
                (0, "unknown")
            };
            info.push_str(
                &format!(" {}   {}  {} non-null  {}\n", idx, col, non_null, dtype),
            );
        }
        info.push_str(
            &format!(
                "dtypes: float64({}), object({})\n", columns.iter().filter(| c | self
                .get_column_numeric_values(c).is_ok()).count(), columns.iter().filter(| c
                | self.get_column_string_values(c).is_ok() && self
                .get_column_numeric_values(c).is_err()).count()
            ),
        );
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
            return Err(
                Error::InvalidValue(
                    "DataFrames must have the same number of rows".to_string(),
                ),
            );
        }
        let mut result = DataFrame::new();
        let n_rows = self.row_count();
        let cols1: std::collections::HashSet<_> = self
            .column_names()
            .into_iter()
            .collect();
        let cols2: std::collections::HashSet<_> = other
            .column_names()
            .into_iter()
            .collect();
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
                result
                    .add_column(
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
    fn insert_column(
        &self,
        loc: usize,
        name: &str,
        values: Vec<f64>,
    ) -> Result<DataFrame> {
        if values.len() != self.row_count() {
            return Err(
                Error::InvalidValue(
                    "Column length must match DataFrame row count".to_string(),
                ),
            );
        }
        let columns = self.column_names();
        let loc = loc.min(columns.len());
        let mut result = DataFrame::new();
        for col in columns.iter().take(loc) {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }
        result
            .add_column(name.to_string(), Series::new(values, Some(name.to_string()))?)?;
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
        let actual_index = if n >= 0 { n as usize } else { (n_rows + n) as usize };
        if actual_index >= self.row_count() {
            return Err(
                Error::InvalidValue(
                    format!(
                        "Index {} out of bounds for DataFrame with {} rows", n, self
                        .row_count()
                    ),
                ),
            );
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(transformed.clone(), Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            }
        }
        Ok(result)
    }
    fn crosstab(&self, col1: &str, col2: &str) -> Result<DataFrame> {
        let values1 = self.get_column_string_values(col1)?;
        let values2 = self.get_column_string_values(col2)?;
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
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        for (v1, v2) in values1.iter().zip(values2.iter()) {
            *counts.entry((v1.clone(), v2.clone())).or_insert(0) += 1;
        }
        let mut result = DataFrame::new();
        result
            .add_column(
                col1.to_string(),
                Series::new(unique1.clone(), Some(col1.to_string()))?,
            )?;
        for u2 in &unique2 {
            let column_counts: Vec<f64> = unique1
                .iter()
                .map(|u1| {
                    counts.get(&(u1.clone(), u2.clone())).copied().unwrap_or(0) as f64
                })
                .collect();
            result
                .add_column(u2.clone(), Series::new(column_counts, Some(u2.clone()))?)?;
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
    fn expanding_apply<F>(
        &self,
        column: &str,
        func: F,
        min_periods: usize,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        window_ops::expanding_apply(self, column, func, min_periods)
    }
    fn align(&self, other: &DataFrame) -> Result<(DataFrame, DataFrame)> {
        let cols1: std::collections::HashSet<_> = self
            .column_names()
            .into_iter()
            .collect();
        let cols2: std::collections::HashSet<_> = other
            .column_names()
            .into_iter()
            .collect();
        let all_cols: Vec<_> = cols1.union(&cols2).cloned().collect();
        let mut result1 = DataFrame::new();
        let mut result2 = DataFrame::new();
        for col in &all_cols {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else {
                let nan_vals: Vec<f64> = vec![f64::NAN; self.row_count()];
                result1
                    .add_column(col.clone(), Series::new(nan_vals, Some(col.clone()))?)?;
            }
            if let Ok(vals) = other.get_column_numeric_values(col) {
                result2.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = other.get_column_string_values(col) {
                result2.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else {
                let nan_vals: Vec<f64> = vec![f64::NAN; other.row_count()];
                result2
                    .add_column(col.clone(), Series::new(nan_vals, Some(col.clone()))?)?;
            }
        }
        Ok((result1, result2))
    }
    fn reindex_columns(&self, columns: &[&str]) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for &col in columns {
            if let Ok(vals) = self.get_column_numeric_values(col) {
                result
                    .add_column(
                        col.to_string(),
                        Series::new(vals, Some(col.to_string()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result
                    .add_column(
                        col.to_string(),
                        Series::new(vals, Some(col.to_string()))?,
                    )?;
            } else {
                let nan_vals: Vec<f64> = vec![f64::NAN; self.row_count()];
                result
                    .add_column(
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
            return Err(
                Error::InvalidValue("Need at least 2 values for z-score".to_string()),
            );
        }
        let n = valid.len() as f64;
        let mean = valid.iter().sum::<f64>() / n;
        let std_dev = (valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0))
            .sqrt();
        if std_dev == 0.0 {
            return Err(Error::InvalidValue("Standard deviation is zero".to_string()));
        }
        Ok(
            values
                .iter()
                .map(|&v| { if v.is_nan() { f64::NAN } else { (v - mean) / std_dev } })
                .collect(),
        )
    }
    fn normalize(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let (min, max) = self.value_range(column)?;
        let range = max - min;
        if range == 0.0 {
            return Err(
                Error::InvalidValue("Range is zero, cannot normalize".to_string()),
            );
        }
        Ok(
            values
                .iter()
                .map(|&v| { if v.is_nan() { f64::NAN } else { (v - min) / range } })
                .collect(),
        )
    }
    fn cut(&self, column: &str, bins: usize) -> Result<Vec<String>> {
        if bins == 0 {
            return Err(Error::InvalidValue("Number of bins must be > 0".to_string()));
        }
        let values = self.get_column_numeric_values(column)?;
        let (min, max) = self.value_range(column)?;
        let bin_width = (max - min) / bins as f64;
        let mut edges: Vec<f64> = (0..=bins)
            .map(|i| min + i as f64 * bin_width)
            .collect();
        edges[bins] = max + 0.001;
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
            return Err(
                Error::InvalidValue("Number of quantiles must be > 0".to_string()),
            );
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
        let mut variables: Vec<String> = Vec::with_capacity(
            n_rows * cols_to_stack.len(),
        );
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
        result
            .add_column(
                "row_index".to_string(),
                Series::new(row_indices, Some("row_index".to_string()))?,
            )?;
        result
            .add_column(
                "variable".to_string(),
                Series::new(variables, Some("variable".to_string()))?,
            )?;
        result
            .add_column(
                "value".to_string(),
                Series::new(values, Some("value".to_string()))?,
            )?;
        Ok(result)
    }
    fn unstack(
        &self,
        index_col: &str,
        columns_col: &str,
        values_col: &str,
    ) -> Result<DataFrame> {
        let index_values = self.get_column_string_values(index_col)?;
        let column_values = self.get_column_string_values(columns_col)?;
        let data_values = self.get_column_numeric_values(values_col)?;
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
        let mut data_map: HashMap<(String, String), f64> = HashMap::new();
        for i in 0..index_values.len() {
            data_map
                .insert(
                    (index_values[i].clone(), column_values[i].clone()),
                    data_values[i],
                );
        }
        let mut result = DataFrame::new();
        result
            .add_column(
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
        self.unstack(index, columns, values)
    }
    fn astype(&self, column: &str, dtype: &str) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                match dtype.to_lowercase().as_str() {
                    "float64" | "float" | "f64" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            result
                                .add_column(
                                    col_name.clone(),
                                    Series::new(values, Some(col_name.clone()))?,
                                )?;
                        } else if let Ok(values) = self
                            .get_column_string_values(&col_name)
                        {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|s| s.parse::<f64>().unwrap_or(f64::NAN))
                                .collect();
                            result
                                .add_column(
                                    col_name.clone(),
                                    Series::new(converted, Some(col_name.clone()))?,
                                )?;
                        }
                    }
                    "int64" | "int" | "i64" => {
                        if let Ok(values) = self.get_column_numeric_values(&col_name) {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|v| v.floor())
                                .collect();
                            result
                                .add_column(
                                    col_name.clone(),
                                    Series::new(converted, Some(col_name.clone()))?,
                                )?;
                        } else if let Ok(values) = self
                            .get_column_string_values(&col_name)
                        {
                            let converted: Vec<f64> = values
                                .iter()
                                .map(|s| {
                                    s.parse::<i64>().map(|i| i as f64).unwrap_or(f64::NAN)
                                })
                                .collect();
                            result
                                .add_column(
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
                                    if v.is_nan() { "NaN".to_string() } else { v.to_string() }
                                })
                                .collect();
                            result
                                .add_column(
                                    col_name.clone(),
                                    Series::new(converted, Some(col_name.clone()))?,
                                )?;
                        } else if let Ok(values) = self
                            .get_column_string_values(&col_name)
                        {
                            result
                                .add_column(
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
                            result
                                .add_column(
                                    col_name.clone(),
                                    Series::new(converted, Some(col_name.clone()))?,
                                )?;
                        }
                    }
                    _ => {
                        return Err(
                            Error::InvalidValue(format!("Unknown dtype: {}", dtype)),
                        );
                    }
                }
            } else {
                if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                    result
                        .add_column(
                            col_name.clone(),
                            Series::new(vals, Some(col_name.clone()))?,
                        )?;
                } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                    result
                        .add_column(
                            col_name.clone(),
                            Series::new(vals, Some(col_name.clone()))?,
                        )?;
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(transformed, Some(col_name.clone()))?,
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
                        let variance = valid
                            .iter()
                            .map(|x| (x - mean).powi(2))
                            .sum::<f64>() / (n - 1.0);
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
                        sorted
                            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                }
                _ => {
                    return Err(
                        Error::InvalidValue(
                            format!("Unknown aggregation function: {}", func),
                        ),
                    );
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
    fn set_values(
        &self,
        column: &str,
        indices: &[usize],
        values: &[f64],
    ) -> Result<DataFrame> {
        if indices.len() != values.len() {
            return Err(
                Error::InvalidValue(
                    "Indices and values must have the same length".to_string(),
                ),
            );
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(col_values.clone(), Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
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
                result
                    .add_column(
                        col.to_string(),
                        Series::new(vals, Some(col.to_string()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(col) {
                result
                    .add_column(
                        col.to_string(),
                        Series::new(vals, Some(col.to_string()))?,
                    )?;
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
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            }
        }
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn col_mul(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;
        if v1.len() != v2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            }
        }
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn col_sub(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;
        if v1.len() != v2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            }
        }
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn col_div(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame> {
        let v1 = self.get_column_numeric_values(col1)?;
        let v2 = self.get_column_numeric_values(col2)?;
        if v1.len() != v2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a / b)
            .collect();
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            } else if let Ok(vals) = self.get_column_string_values(&col_name) {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(vals, Some(col_name.clone()))?,
                    )?;
            }
        }
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
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
                        row_data
                            .insert(
                                col.clone(),
                                SeriesValue::String(vals[row_idx].clone()),
                            );
                    }
                }
            }
            result.push((row_idx, row_data));
        }
        result
    }
    fn at(&self, row: usize, column: &str) -> Result<SeriesValue> {
        if row >= self.row_count() {
            return Err(Error::InvalidValue(format!("Row index {} out of bounds", row)));
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
            return Err(
                Error::InvalidValue(format!("Column index {} out of bounds", col_idx)),
            );
        }
        self.at(row, &columns[col_idx])
    }
    fn drop_rows(&self, indices: &[usize]) -> Result<DataFrame> {
        let indices_set: std::collections::HashSet<usize> = indices
            .iter()
            .cloned()
            .collect();
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                let filtered: Vec<f64> = vals
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !indices_set.contains(i))
                    .map(|(_, v)| *v)
                    .collect();
                result
                    .add_column(col.clone(), Series::new(filtered, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                let filtered: Vec<String> = vals
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !indices_set.contains(i))
                    .map(|(_, v)| v.clone())
                    .collect();
                result
                    .add_column(col.clone(), Series::new(filtered, Some(col.clone()))?)?;
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
    fn reset_index(
        &self,
        index_values: Option<&[String]>,
        name: &str,
    ) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        if let Some(idx_vals) = index_values {
            result
                .add_column(
                    name.to_string(),
                    Series::new(idx_vals.to_vec(), Some(name.to_string()))?,
                )?;
        }
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
                values = vals.iter().map(|v| SeriesValue::String(v.clone())).collect();
            }
            result.push((col, values));
        }
        result
    }
    fn update(&self, other: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if other.contains_column(&col) {
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
                        result
                            .add_column(
                                col.clone(),
                                Series::new(updated, Some(col.clone()))?,
                            )?;
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
                        result
                            .add_column(
                                col.clone(),
                                Series::new(updated, Some(col.clone()))?,
                            )?;
                    }
                }
            } else {
                if let Ok(vals) = self.get_column_numeric_values(&col) {
                    result
                        .add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
                } else if let Ok(vals) = self.get_column_string_values(&col) {
                    result
                        .add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
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
        Ok(
            rows
                .into_iter()
                .next()
                .ok_or_else(|| Error::InsufficientData("No rows available".to_string()))?
                .1,
        )
    }
    fn last_row(&self) -> Result<HashMap<String, SeriesValue>> {
        if self.row_count() == 0 {
            return Err(Error::InvalidValue("DataFrame is empty".to_string()));
        }
        let rows = self.iterrows();
        Ok(
            rows
                .into_iter()
                .last()
                .ok_or_else(|| Error::InsufficientData("No rows available".to_string()))?
                .1,
        )
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
        let other_result = other
            .get_column_string_values(result_col)
            .or_else(|_| {
                other
                    .get_column_numeric_values(result_col)
                    .map(|v| v.iter().map(|x| x.to_string()).collect())
            })?;
        let mut lookup_map: HashMap<String, String> = HashMap::new();
        for (i, key) in other_keys.iter().enumerate() {
            if i < other_result.len() {
                lookup_map.insert(key.clone(), other_result[i].clone());
            }
        }
        let result_values: Vec<String> = lookup_vals
            .iter()
            .map(|k| lookup_map.get(k).cloned().unwrap_or_default())
            .collect();
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if let Ok(vals) = self.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = self.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }
        result
            .add_column(
                format!("{}_result", result_col),
                Series::new(result_values, Some(format!("{}_result", result_col)))?,
            )?;
        Ok(result)
    }
    fn get_column_by_index(&self, idx: usize) -> Result<(String, Vec<SeriesValue>)> {
        let columns = self.column_names();
        if idx >= columns.len() {
            return Err(
                Error::InvalidValue(format!("Column index {} out of bounds", idx)),
            );
        }
        let col_name = &columns[idx];
        let mut values = Vec::new();
        if let Ok(vals) = self.get_column_numeric_values(col_name) {
            values = vals.iter().map(|v| SeriesValue::Float(*v)).collect();
        } else if let Ok(vals) = self.get_column_string_values(col_name) {
            values = vals.iter().map(|v| SeriesValue::String(v.clone())).collect();
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
        let mut result = DataFrame::new();
        for col in self.column_names() {
            if col == column {
                result
                    .add_column(
                        col.clone(),
                        Series::new(codes.clone(), Some(col.clone()))?,
                    )?;
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
            return Err(
                Error::InvalidValue("Fraction must be between 0 and 1".to_string()),
            );
        }
        let n = (self.row_count() as f64 * frac).round() as usize;
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
}

//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Pandas compatibility extension trait for DataFrame - advanced operations, string ops, and utilities
use super::super::helpers::{aggregations, string_ops};
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
                for (i, hash) in hashes.iter().enumerate() {
                    seen.insert(*hash, i);
                }
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
                let mut counts: HashMap<u64, usize> = HashMap::new();
                for hash in &hashes {
                    *counts.entry(*hash).or_insert(0) += 1;
                }
                for (i, hash) in hashes.iter().enumerate() {
                    if counts.get(hash).copied().unwrap_or(0) > 1 {
                        result[i] = true;
                    }
                }
            }
        }
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
        let group_vals = self
            .get_column_string_values(by)
            .or_else(|_| {
                self.get_column_numeric_values(by)
                    .map(|v| v.iter().map(|x| x.to_string()).collect())
            })?;
        let mut unique_groups: Vec<String> = Vec::new();
        for val in &group_vals {
            if !unique_groups.contains(val) {
                unique_groups.push(val.clone());
            }
        }
        let mut result_data: HashMap<String, Vec<f64>> = HashMap::new();
        let mut group_names: Vec<String> = Vec::new();
        for group in unique_groups {
            let indices: Vec<usize> = group_vals
                .iter()
                .enumerate()
                .filter(|(_, v)| *v == &group)
                .map(|(i, _)| i)
                .collect();
            let subset = self.take(&indices)?;
            let agg_result = func(&subset)?;
            group_names.push(group);
            for (key, value) in agg_result {
                result_data.entry(key).or_default().push(value);
            }
        }
        let mut result = DataFrame::new();
        result
            .add_column(
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
        let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (n - ddof as f64);
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
    fn str_replace(
        &self,
        column: &str,
        pattern: &str,
        replacement: &str,
    ) -> Result<DataFrame> {
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(filled.clone(), Some(col_name))?,
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
    fn bfill(&self, column: &str) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let mut filled = vec![f64::NAN; values.len()];
        let mut last_valid = f64::NAN;
        for i in (0..values.len()).rev() {
            if !values[i].is_nan() {
                last_valid = values[i];
            }
            filled[i] = last_valid;
        }
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(filled.clone(), Some(col_name))?,
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
    fn pct_rank(&self, column: &str) -> Result<Vec<f64>> {
        let values = self.get_column_numeric_values(column)?;
        let n = values.len();
        let mut result = vec![f64::NAN; n];
        let mut indexed: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(i, v)| (i, *v))
            .collect();
        indexed
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let valid_count = indexed.len();
        if valid_count == 0 {
            return Ok(result);
        }
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
            return Err(
                Error::InvalidValue("No valid values found in column".to_string()),
            );
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
            return Err(
                Error::InvalidValue("No valid values found in column".to_string()),
            );
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(clipped.clone(), Some(col_name))?,
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
    fn clip_upper(&self, column: &str, max: f64) -> Result<DataFrame> {
        let values = self.get_column_numeric_values(column)?;
        let clipped: Vec<f64> = values.iter().map(|v| v.min(max)).collect();
        let mut result = DataFrame::new();
        for col_name in self.column_names() {
            if &col_name == column {
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(clipped.clone(), Some(col_name))?,
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
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let coalesced: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| if v1.is_nan() { *v2 } else { *v1 })
            .collect();
        let mut result = self.copy();
        result
            .add_column(
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
    fn add_columns(
        &self,
        col1: &str,
        col2: &str,
        result_name: &str,
    ) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;
        if values1.len() != values2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 + v2)
            .collect();
        let mut result = self.copy();
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn sub_columns(
        &self,
        col1: &str,
        col2: &str,
        result_name: &str,
    ) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;
        if values1.len() != values2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 - v2)
            .collect();
        let mut result = self.copy();
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn mul_columns(
        &self,
        col1: &str,
        col2: &str,
        result_name: &str,
    ) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;
        if values1.len() != values2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 * v2)
            .collect();
        let mut result = self.copy();
        result
            .add_column(
                result_name.to_string(),
                Series::new(result_values, Some(result_name.to_string()))?,
            )?;
        Ok(result)
    }
    fn div_columns(
        &self,
        col1: &str,
        col2: &str,
        result_name: &str,
    ) -> Result<DataFrame> {
        let values1 = self.get_column_numeric_values(col1)?;
        let values2 = self.get_column_numeric_values(col2)?;
        if values1.len() != values2.len() {
            return Err(
                Error::InvalidValue("Columns must have the same length".to_string()),
            );
        }
        let result_values: Vec<f64> = values1
            .iter()
            .zip(values2.iter())
            .map(|(v1, v2)| v1 / v2)
            .collect();
        let mut result = self.copy();
        result
            .add_column(
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
        Ok(
            values
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
                .collect(),
        )
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(result_values.clone(), Some(col_name))?,
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
    fn str_startswith(&self, column: &str, prefix: &str) -> Result<Vec<bool>> {
        string_ops::str_startswith(self, column, prefix)
    }
    fn str_endswith(&self, column: &str, suffix: &str) -> Result<Vec<bool>> {
        string_ops::str_endswith(self, column, suffix)
    }
    fn str_pad_left(
        &self,
        column: &str,
        width: usize,
        fillchar: char,
    ) -> Result<DataFrame> {
        string_ops::str_pad_left(self, column, width, fillchar)
    }
    fn str_pad_right(
        &self,
        column: &str,
        width: usize,
        fillchar: char,
    ) -> Result<DataFrame> {
        string_ops::str_pad_right(self, column, width, fillchar)
    }
    fn str_slice(
        &self,
        column: &str,
        start: usize,
        end: Option<usize>,
    ) -> Result<DataFrame> {
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
        Ok(
            values
                .iter()
                .filter(|v| !v.is_nan() && (*v - value).abs() < f64::EPSILON)
                .count(),
        )
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
                result
                    .add_column(
                        col_name.clone(),
                        Series::new(result_values.clone(), Some(col_name))?,
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
        Ok(
            values
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
                .collect(),
        )
    }
    fn str_count(&self, column: &str, pattern: &str) -> Result<Vec<usize>> {
        string_ops::str_count(self, column, pattern)
    }
    fn str_repeat(&self, column: &str, n: usize) -> Result<DataFrame> {
        string_ops::str_repeat(self, column, n)
    }
    fn str_center(
        &self,
        column: &str,
        width: usize,
        fillchar: char,
    ) -> Result<DataFrame> {
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
            Ok(
                vals.iter().map(|s| s.len()).sum::<usize>()
                    + vals.len() * std::mem::size_of::<String>(),
            )
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
        let (mode_bits, count) = counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .ok_or_else(|| {
                Error::InsufficientData(
                    "No valid values for mode calculation".to_string(),
                )
            })?;
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

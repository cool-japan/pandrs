//! Implementation functions for PandasCompatExt - reshape and analysis (melt, rolling, expanding, pivot, astype)

use super::super::helpers::{aggregations, comparison_ops, math_ops, string_ops, window_ops};
use super::super::trait_def::PandasCompatExt;
use super::super::types::{Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue};
use super::functions::select_rows_by_indices;
use super::functions_3::{covariance, pearson_correlation};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub(super) fn explode(df: &DataFrame, column: &str, separator: &str) -> Result<DataFrame> {
    let string_values = df.get_column_string_values(column)?;
    let n_rows = df.row_count();
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
    for col_name in df.column_names() {
        if &col_name == column {
            let new_values: Vec<String> =
                row_mapping.iter().map(|(_, val)| val.to_string()).collect();
            result.add_column(
                col_name.clone(),
                Series::new(new_values, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let new_values: Vec<f64> = row_mapping.iter().map(|(idx, _)| values[*idx]).collect();
            result.add_column(
                col_name.clone(),
                Series::new(new_values, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
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

pub(super) fn duplicated(df: &DataFrame, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>> {
    let columns_to_check: Vec<String> = match subset {
        Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
        None => df.column_names(),
    };
    for col in &columns_to_check {
        if !df.contains_column(col) {
            return Err(Error::InvalidValue(format!(
                "Column '{}' not found in DataFrame",
                col
            )));
        }
    }
    let n_rows = df.row_count();
    let mut row_keys: Vec<String> = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut key_parts: Vec<String> = Vec::new();
        for col in &columns_to_check {
            if let Ok(values) = df.get_column_string_values(col) {
                key_parts.push(values.get(row_idx).cloned().unwrap_or_default());
            } else if let Ok(values) = df.get_column_numeric_values(col) {
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
            return Err(Error::InvalidValue(format!(
                "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                keep
            )));
        }
    }
    Ok(is_duplicate)
}

pub(super) fn copy(df: &DataFrame) -> DataFrame {
    df.clone()
}

pub(super) fn to_dict(df: &DataFrame) -> Result<HashMap<String, Vec<String>>> {
    let mut result = HashMap::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_string_values(&col_name) {
            result.insert(col_name, values);
        } else if let Ok(values) = df.get_column_numeric_values(&col_name) {
            result.insert(col_name, values.iter().map(|v| v.to_string()).collect());
        }
    }
    Ok(result)
}

pub(super) fn first_valid_index(df: &DataFrame, column: &str) -> Result<Option<usize>> {
    let values = df.get_column_numeric_values(column)?;
    for (idx, v) in values.iter().enumerate() {
        if !v.is_nan() {
            return Ok(Some(idx));
        }
    }
    Ok(None)
}

pub(super) fn last_valid_index(df: &DataFrame, column: &str) -> Result<Option<usize>> {
    let values = df.get_column_numeric_values(column)?;
    for (idx, v) in values.iter().enumerate().rev() {
        if !v.is_nan() {
            return Ok(Some(idx));
        }
    }
    Ok(None)
}

pub(super) fn product_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if !valid.is_empty() {
                let product = valid.iter().fold(1.0, |acc, &x| acc * x);
                results.push((col_name, product));
            }
        }
    }
    Ok(results)
}

pub(super) fn median_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
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

pub(super) fn skew(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn kurtosis(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
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
    let adjustment = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * kurtosis + 6.0);
    Ok(adjustment)
}

pub(super) fn add_prefix(df: &DataFrame, prefix: &str) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        let new_name = format!("{}{}", prefix, col_name);
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
        }
    }
    Ok(result)
}

pub(super) fn add_suffix(df: &DataFrame, suffix: &str) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        let new_name = format!("{}{}", col_name, suffix);
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            result.add_column(new_name.clone(), Series::new(values, Some(new_name))?)?;
        }
    }
    Ok(result)
}

pub(super) fn filter_by_mask(df: &DataFrame, mask: &[bool]) -> Result<DataFrame> {
    if mask.len() != df.row_count() {
        return Err(Error::InvalidValue(
            "Mask length must match number of rows".to_string(),
        ));
    }
    let indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn mode_numeric(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for v in values.iter().filter(|v| !v.is_nan()) {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return Ok(vec![]);
    }
    let max_count = *counts.values().max().ok_or_else(|| {
        Error::InsufficientData("No valid values for mode calculation".to_string())
    })?;
    let mut modes: Vec<f64> = counts
        .iter()
        .filter(|(_, &c)| c == max_count)
        .map(|(&bits, _)| f64::from_bits(bits))
        .collect();
    modes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(modes)
}

pub(super) fn mode_string(df: &DataFrame, column: &str) -> Result<Vec<String>> {
    let values = df.get_column_string_values(column)?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for v in values.iter().filter(|v| !v.is_empty()) {
        *counts.entry(v.clone()).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return Ok(vec![]);
    }
    let max_count = *counts.values().max().ok_or_else(|| {
        Error::InsufficientData("No valid values for mode calculation".to_string())
    })?;
    let mut modes: Vec<String> = counts
        .iter()
        .filter(|(_, &c)| c == max_count)
        .map(|(k, _)| k.clone())
        .collect();
    modes.sort();
    Ok(modes)
}

pub(super) fn percentile(df: &DataFrame, column: &str, n: f64) -> Result<f64> {
    df.quantile(column, n / 100.0)
}

pub(super) fn ewma(df: &DataFrame, column: &str, span: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
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
            result.push(ewma_value.expect("EWMA value just set"));
        }
    }
    Ok(result)
}

pub(super) fn iloc(df: &DataFrame, index: usize) -> Result<HashMap<String, String>> {
    if index >= df.row_count() {
        return Err(Error::InvalidValue(format!(
            "Index {} out of bounds for DataFrame with {} rows",
            index,
            df.row_count()
        )));
    }
    let mut result = HashMap::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_string_values(&col_name) {
            result.insert(col_name, values[index].clone());
        } else if let Ok(values) = df.get_column_numeric_values(&col_name) {
            result.insert(col_name, values[index].to_string());
        }
    }
    Ok(result)
}

pub(super) fn iloc_range(df: &DataFrame, start: usize, end: usize) -> Result<DataFrame> {
    if start > end {
        return Err(Error::InvalidValue(
            "Start index must be less than or equal to end index".to_string(),
        ));
    }
    let n_rows = df.row_count();
    let end = end.min(n_rows);
    let start = start.min(n_rows);
    let indices: Vec<usize> = (start..end).collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn info(df: &DataFrame) -> String {
    let mut info = String::new();
    let n_rows = df.row_count();
    let columns = df.column_names();
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
        let (non_null, dtype) = if let Ok(values) = df.get_column_numeric_values(col) {
            let non_null = values.iter().filter(|v| !v.is_nan()).count();
            (non_null, "float64")
        } else if let Ok(values) = df.get_column_string_values(col) {
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
            .filter(|c| df.get_column_numeric_values(c).is_ok())
            .count(),
        columns
            .iter()
            .filter(|c| df.get_column_string_values(c).is_ok()
                && df.get_column_numeric_values(c).is_err())
            .count()
    ));
    info.push_str(&format!("memory usage: {} bytes\n", df.memory_usage()));
    info
}

pub(super) fn equals(df: &DataFrame, other: &DataFrame) -> bool {
    if df.row_count() != other.row_count() {
        return false;
    }
    let cols1 = df.column_names();
    let cols2 = other.column_names();
    if cols1 != cols2 {
        return false;
    }
    for col in &cols1 {
        if let (Ok(v1), Ok(v2)) = (
            df.get_column_numeric_values(col),
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
            df.get_column_string_values(col),
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

pub(super) fn compare(df: &DataFrame, other: &DataFrame) -> Result<DataFrame> {
    if df.row_count() != other.row_count() {
        return Err(Error::InvalidValue(
            "DataFrames must have the same number of rows".to_string(),
        ));
    }
    let mut result = DataFrame::new();
    let n_rows = df.row_count();
    let cols1: std::collections::HashSet<_> = df.column_names().into_iter().collect();
    let cols2: std::collections::HashSet<_> = other.column_names().into_iter().collect();
    let common_cols: Vec<_> = cols1.intersection(&cols2).cloned().collect();
    for col in &common_cols {
        if let (Ok(v1), Ok(v2)) = (
            df.get_column_numeric_values(col),
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

pub(super) fn keys(df: &DataFrame) -> Vec<String> {
    df.column_names()
}

pub(super) fn pop_column(df: &DataFrame, column: &str) -> Result<(DataFrame, Vec<f64>)> {
    let values = df.get_column_numeric_values(column)?;
    let new_df = df.drop_columns(&[column])?;
    Ok((new_df, values))
}

pub(super) fn insert_column(
    df: &DataFrame,
    loc: usize,
    name: &str,
    values: Vec<f64>,
) -> Result<DataFrame> {
    if values.len() != df.row_count() {
        return Err(Error::InvalidValue(
            "Column length must match DataFrame row count".to_string(),
        ));
    }
    let columns = df.column_names();
    let loc = loc.min(columns.len());
    let mut result = DataFrame::new();
    for col in columns.iter().take(loc) {
        if let Ok(vals) = df.get_column_numeric_values(col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    result.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string()))?,
    )?;
    for col in columns.iter().skip(loc) {
        if let Ok(vals) = df.get_column_numeric_values(col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn rolling_sum(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_sum(df, column, window, min_periods)
}

pub(super) fn rolling_mean(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_mean(df, column, window, min_periods)
}

pub(super) fn rolling_std(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_std(df, column, window, min_periods)
}

pub(super) fn rolling_min(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_min(df, column, window, min_periods)
}

pub(super) fn rolling_max(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_max(df, column, window, min_periods)
}

pub(super) fn rolling_var(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_var(df, column, window, min_periods)
}

pub(super) fn rolling_median(
    df: &DataFrame,
    column: &str,
    window: usize,
    min_periods: Option<usize>,
) -> Result<Vec<f64>> {
    window_ops::rolling_median(df, column, window, min_periods)
}

pub(super) fn rolling_count(df: &DataFrame, column: &str, window: usize) -> Result<Vec<usize>> {
    window_ops::rolling_count(df, column, window)
}

pub(super) fn rolling_apply<F>(
    df: &DataFrame,
    column: &str,
    window: usize,
    func: F,
    min_periods: Option<usize>,
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    window_ops::rolling_apply(df, column, window, func, min_periods)
}

pub(super) fn cumcount(df: &DataFrame, column: &str) -> Result<Vec<usize>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn nth(df: &DataFrame, n: i32) -> Result<HashMap<String, String>> {
    let n_rows = df.row_count() as i32;
    let actual_index = if n >= 0 {
        n as usize
    } else {
        (n_rows + n) as usize
    };
    if actual_index >= df.row_count() {
        return Err(Error::InvalidValue(format!(
            "Index {} out of bounds for DataFrame with {} rows",
            n,
            df.row_count()
        )));
    }
    df.iloc(actual_index)
}

pub(super) fn transform<F>(df: &DataFrame, column: &str, func: F) -> Result<DataFrame>
where
    F: Fn(f64) -> f64,
{
    let values = df.get_column_numeric_values(column)?;
    let transformed: Vec<f64> = values.iter().map(|&v| func(v)).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(transformed.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn crosstab(df: &DataFrame, col1: &str, col2: &str) -> Result<DataFrame> {
    let values1 = df.get_column_string_values(col1)?;
    let values2 = df.get_column_string_values(col2)?;
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
    result.add_column(
        col1.to_string(),
        Series::new(unique1.clone(), Some(col1.to_string()))?,
    )?;
    for u2 in &unique2 {
        let column_counts: Vec<f64> = unique1
            .iter()
            .map(|u1| counts.get(&(u1.clone(), u2.clone())).copied().unwrap_or(0) as f64)
            .collect();
        result.add_column(u2.clone(), Series::new(column_counts, Some(u2.clone()))?)?;
    }
    Ok(result)
}

pub(super) fn expanding_sum(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_sum(df, column, min_periods)
}

pub(super) fn expanding_mean(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_mean(df, column, min_periods)
}

pub(super) fn expanding_std(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_std(df, column, min_periods)
}

pub(super) fn expanding_min(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_min(df, column, min_periods)
}

pub(super) fn expanding_max(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_max(df, column, min_periods)
}

pub(super) fn expanding_var(df: &DataFrame, column: &str, min_periods: usize) -> Result<Vec<f64>> {
    window_ops::expanding_var(df, column, min_periods)
}

pub(super) fn expanding_apply<F>(
    df: &DataFrame,
    column: &str,
    func: F,
    min_periods: usize,
) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    window_ops::expanding_apply(df, column, func, min_periods)
}

pub(super) fn align(df: &DataFrame, other: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let cols1: std::collections::HashSet<_> = df.column_names().into_iter().collect();
    let cols2: std::collections::HashSet<_> = other.column_names().into_iter().collect();
    let all_cols: Vec<_> = cols1.union(&cols2).cloned().collect();
    let mut result1 = DataFrame::new();
    let mut result2 = DataFrame::new();
    for col in &all_cols {
        if let Ok(vals) = df.get_column_numeric_values(col) {
            result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(col) {
            result1.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else {
            let nan_vals: Vec<f64> = vec![f64::NAN; df.row_count()];
            result1.add_column(col.clone(), Series::new(nan_vals, Some(col.clone()))?)?;
        }
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

pub(super) fn reindex_columns(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for &col in columns {
        if let Ok(vals) = df.get_column_numeric_values(col) {
            result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(col) {
            result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
        } else {
            let nan_vals: Vec<f64> = vec![f64::NAN; df.row_count()];
            result.add_column(
                col.to_string(),
                Series::new(nan_vals, Some(col.to_string()))?,
            )?;
        }
    }
    Ok(result)
}

pub(super) fn value_range(df: &DataFrame, column: &str) -> Result<(f64, f64)> {
    let values = df.get_column_numeric_values(column)?;
    let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return Err(Error::InvalidValue("No valid values in column".to_string()));
    }
    let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((min, max))
}

pub(super) fn zscore(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn normalize(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let (min, max) = df.value_range(column)?;
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

pub(super) fn cut(df: &DataFrame, column: &str, bins: usize) -> Result<Vec<String>> {
    if bins == 0 {
        return Err(Error::InvalidValue(
            "Number of bins must be > 0".to_string(),
        ));
    }
    let values = df.get_column_numeric_values(column)?;
    let (min, max) = df.value_range(column)?;
    let bin_width = (max - min) / bins as f64;
    let mut edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();
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

pub(super) fn qcut(df: &DataFrame, column: &str, q: usize) -> Result<Vec<String>> {
    if q == 0 {
        return Err(Error::InvalidValue(
            "Number of quantiles must be > 0".to_string(),
        ));
    }
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn stack(df: &DataFrame, columns: Option<&[&str]>) -> Result<DataFrame> {
    let cols_to_stack: Vec<String> = if let Some(cols) = columns {
        cols.iter().map(|s| s.to_string()).collect()
    } else {
        df.column_names()
            .into_iter()
            .filter(|c| df.get_column_numeric_values(c).is_ok())
            .collect()
    };
    if cols_to_stack.is_empty() {
        return Err(Error::InvalidValue("No columns to stack".to_string()));
    }
    let n_rows = df.row_count();
    let mut row_indices: Vec<f64> = Vec::with_capacity(n_rows * cols_to_stack.len());
    let mut variables: Vec<String> = Vec::with_capacity(n_rows * cols_to_stack.len());
    let mut values: Vec<f64> = Vec::with_capacity(n_rows * cols_to_stack.len());
    for row_idx in 0..n_rows {
        for col_name in &cols_to_stack {
            row_indices.push(row_idx as f64);
            variables.push(col_name.clone());
            if let Ok(col_values) = df.get_column_numeric_values(col_name) {
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

pub(super) fn unstack(
    df: &DataFrame,
    index_col: &str,
    columns_col: &str,
    values_col: &str,
) -> Result<DataFrame> {
    let index_values = df.get_column_string_values(index_col)?;
    let column_values = df.get_column_string_values(columns_col)?;
    let data_values = df.get_column_numeric_values(values_col)?;
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
        data_map.insert(
            (index_values[i].clone(), column_values[i].clone()),
            data_values[i],
        );
    }
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

pub(super) fn pivot(df: &DataFrame, index: &str, columns: &str, values: &str) -> Result<DataFrame> {
    df.unstack(index, columns, values)
}

pub(super) fn astype(df: &DataFrame, column: &str, dtype: &str) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            match dtype.to_lowercase().as_str() {
                "float64" | "float" | "f64" => {
                    if let Ok(values) = df.get_column_numeric_values(&col_name) {
                        result.add_column(
                            col_name.clone(),
                            Series::new(values, Some(col_name.clone()))?,
                        )?;
                    } else if let Ok(values) = df.get_column_string_values(&col_name) {
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
                    if let Ok(values) = df.get_column_numeric_values(&col_name) {
                        let converted: Vec<f64> = values.iter().map(|v| v.floor()).collect();
                        result.add_column(
                            col_name.clone(),
                            Series::new(converted, Some(col_name.clone()))?,
                        )?;
                    } else if let Ok(values) = df.get_column_string_values(&col_name) {
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
                    if let Ok(values) = df.get_column_numeric_values(&col_name) {
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
                    } else if let Ok(values) = df.get_column_string_values(&col_name) {
                        result.add_column(
                            col_name.clone(),
                            Series::new(values, Some(col_name.clone()))?,
                        )?;
                    }
                }
                "bool" | "boolean" => {
                    if let Ok(values) = df.get_column_numeric_values(&col_name) {
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
                _ => {
                    return Err(Error::InvalidValue(format!("Unknown dtype: {}", dtype)));
                }
            }
        } else {
            if let Ok(vals) = df.get_column_numeric_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = df.get_column_string_values(&col_name) {
                result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
    }
    Ok(result)
}

pub(super) fn applymap<F>(df: &DataFrame, func: F) -> Result<DataFrame>
where
    F: Fn(f64) -> f64,
{
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let transformed: Vec<f64> = values.iter().map(|&v| func(v)).collect();
            result.add_column(
                col_name.clone(),
                Series::new(transformed, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            result.add_column(
                col_name.clone(),
                Series::new(values, Some(col_name.clone()))?,
            )?;
        }
    }
    Ok(result)
}

pub(super) fn agg(df: &DataFrame, column: &str, funcs: &[&str]) -> Result<HashMap<String, f64>> {
    let values = df.get_column_numeric_values(column)?;
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
                )));
            }
        };
        results.insert(func.to_string(), value);
    }
    Ok(results)
}

pub(super) fn dtypes(df: &DataFrame) -> Vec<(String, String)> {
    df.column_names()
        .into_iter()
        .map(|col| {
            let dtype = if df.get_column_numeric_values(&col).is_ok() {
                "float64".to_string()
            } else if df.get_column_string_values(&col).is_ok() {
                "object".to_string()
            } else {
                "unknown".to_string()
            };
            (col, dtype)
        })
        .collect()
}

pub(super) fn set_values(
    df: &DataFrame,
    column: &str,
    indices: &[usize],
    values: &[f64],
) -> Result<DataFrame> {
    if indices.len() != values.len() {
        return Err(Error::InvalidValue(
            "Indices and values must have the same length".to_string(),
        ));
    }
    let mut col_values = df.get_column_numeric_values(column)?;
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= col_values.len() {
            return Err(Error::InvalidValue(format!("Index {} out of bounds", idx)));
        }
        col_values[idx] = values[i];
    }
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(col_values.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn query_eq(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let mask: Vec<bool> = values
        .iter()
        .map(|&v| (v - value).abs() < f64::EPSILON)
        .collect();
    df.filter_by_mask(&mask)
}

pub(super) fn query_gt(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let mask: Vec<bool> = values.iter().map(|&v| v > value).collect();
    df.filter_by_mask(&mask)
}

pub(super) fn query_lt(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let mask: Vec<bool> = values.iter().map(|&v| v < value).collect();
    df.filter_by_mask(&mask)
}

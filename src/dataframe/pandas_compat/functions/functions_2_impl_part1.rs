//! Implementation functions for PandasCompatExt - core operations (assign, filter, sort, aggregate, cumulative)

use super::super::merge as merge_mod;
use super::super::types::{Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue};
use super::functions::select_rows_by_indices;
use super::functions_3::{covariance, pearson_correlation};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub(super) fn assign<F, T>(df: &DataFrame, name: &str, func: F) -> Result<DataFrame>
where
    F: FnOnce(&DataFrame) -> Vec<T>,
    T: Into<SeriesValue>,
{
    let values = func(df);
    let mut df = df.clone();
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

pub(super) fn assign_many(df: &DataFrame, assignments: Vec<(&str, Vec<f64>)>) -> Result<DataFrame> {
    let mut df = df.clone();
    for (name, values) in assignments {
        df.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
    }
    Ok(df)
}

pub(super) fn pipe<F, R>(df: &DataFrame, func: F) -> R
where
    F: FnOnce(&DataFrame) -> R,
{
    func(df)
}

pub(super) fn pipe_result<F>(df: &DataFrame, func: F) -> Result<DataFrame>
where
    F: FnOnce(&DataFrame) -> Result<DataFrame>,
{
    func(df)
}

pub(super) fn isin(df: &DataFrame, column: &str, values: &[&str]) -> Result<Vec<bool>> {
    let col_values = df.get_column_string_values(column)?;
    let value_set: HashSet<&str> = values.iter().copied().collect();
    let result: Vec<bool> = col_values
        .iter()
        .map(|s| value_set.contains(s.as_str()))
        .collect();
    Ok(result)
}

pub(super) fn isin_numeric(df: &DataFrame, column: &str, values: &[f64]) -> Result<Vec<bool>> {
    let col_values = df.get_column_numeric_values(column)?;
    let value_set: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
    let result: Vec<bool> = col_values
        .iter()
        .map(|v| value_set.contains(&v.to_bits()))
        .collect();
    Ok(result)
}

pub(super) fn nlargest(df: &DataFrame, n: usize, column: &str) -> Result<DataFrame> {
    let col_values = df.get_column_numeric_values(column)?;
    let mut indexed_values: Vec<(usize, f64)> = col_values.into_iter().enumerate().collect();
    indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed_values.truncate(n);
    let indices: Vec<usize> = indexed_values.into_iter().map(|(i, _)| i).collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn nsmallest(df: &DataFrame, n: usize, column: &str) -> Result<DataFrame> {
    let col_values = df.get_column_numeric_values(column)?;
    let mut indexed_values: Vec<(usize, f64)> = col_values.into_iter().enumerate().collect();
    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    indexed_values.truncate(n);
    let indices: Vec<usize> = indexed_values.into_iter().map(|(i, _)| i).collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn idxmax(df: &DataFrame, column: &str) -> Result<Option<usize>> {
    let values = df.get_column_numeric_values(column)?;
    // Skip NaN explicitly (matching `argmax`'s semantics and pandas'
    // `idxmax(skipna=True)` default) instead of `max_by` with
    // `partial_cmp().unwrap_or(Equal)`: that treats every NaN as "equal" to
    // whatever it's compared against, and since `max_by` breaks ties by
    // keeping the *last* candidate, a trailing NaN could silently win over
    // the true maximum (e.g. `[5.0, NaN]` returned index 1, not 0).
    let mut max_idx: Option<usize> = None;
    let mut max_val = f64::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (max_idx.is_none() || v > max_val) {
            max_val = v;
            max_idx = Some(i);
        }
    }
    Ok(max_idx)
}

pub(super) fn idxmin(df: &DataFrame, column: &str) -> Result<Option<usize>> {
    let values = df.get_column_numeric_values(column)?;
    // See `idxmax`: skip NaN explicitly rather than relying on
    // `partial_cmp().unwrap_or(Equal)`, which can hand the result to a NaN.
    let mut min_idx: Option<usize> = None;
    let mut min_val = f64::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_nan() && (min_idx.is_none() || v < min_val) {
            min_val = v;
            min_idx = Some(i);
        }
    }
    Ok(min_idx)
}

pub(super) fn rank(df: &DataFrame, column: &str, method: RankMethod) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn clip(
    df: &DataFrame,
    column: &str,
    lower: Option<f64>,
    upper: Option<f64>,
) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let clipped: Vec<f64> = values
        .iter()
        .map(|&v| {
            let v = lower.map_or(v, |l| v.max(l));
            upper.map_or(v, |u| v.min(u))
        })
        .collect();
    let mut df = df.clone();
    df.add_column(
        column.to_string(),
        Series::new(clipped, Some(column.to_string()))?,
    )?;
    Ok(df)
}

pub(super) fn between(df: &DataFrame, column: &str, lower: f64, upper: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    let result: Vec<bool> = values.iter().map(|&v| v >= lower && v <= upper).collect();
    Ok(result)
}

pub(super) fn transpose(df: &DataFrame) -> Result<DataFrame> {
    let col_names = df.column_names();
    let n_rows = df.row_count();
    let n_cols = col_names.len();
    if n_rows == 0 || n_cols == 0 {
        return Ok(DataFrame::new());
    }
    let mut new_df = DataFrame::new();
    let all_values: Vec<Vec<String>> = col_names
        .iter()
        .map(|col| df.get_column_string_values(col).unwrap_or_default())
        .collect();

    // The source row index becomes the transposed frame's column names
    // (matching pandas' `df.T`, where the row index becomes the column
    // index) when the source actually carries an explicit index; otherwise
    // fall back to the previous positional "row_N" naming.
    let row_labels = df
        .get_index()
        .string_values()
        .filter(|labels| labels.len() == n_rows);
    let new_col_names: Vec<String> = match &row_labels {
        Some(labels) => labels.clone(),
        None => (0..n_rows).map(|i| format!("row_{}", i)).collect(),
    };

    for (i, new_col_name) in new_col_names.into_iter().enumerate() {
        let values: Vec<String> = all_values
            .iter()
            .map(|col_vals| col_vals.get(i).cloned().unwrap_or_default())
            .collect();
        new_df.add_column(
            new_col_name.clone(),
            Series::new(values, Some(new_col_name))?,
        )?;
    }

    // The source column names become the transposed frame's row index, so
    // they survive the transpose instead of being discarded outright.
    if let Ok(new_index) = crate::index::Index::<String>::new(col_names.to_vec()) {
        new_df.set_index(new_index)?;
    }

    Ok(new_df)
}

pub(super) fn cumsum(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    // A NaN must only blank out its own position (pandas `cumsum`
    // skipna=True default): `cumsum += NaN` here previously poisoned the
    // running total permanently, turning every subsequent cumulative sum
    // into NaN instead of just the row that was actually missing.
    let mut cumsum = 0.0;
    let result: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                cumsum += v;
                cumsum
            }
        })
        .collect();
    Ok(result)
}

pub(super) fn cumprod(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    // See `cumsum`: keep NaN local to its own position instead of letting
    // it poison every later cumulative product.
    let mut cumprod = 1.0;
    let result: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                cumprod *= v;
                cumprod
            }
        })
        .collect();
    Ok(result)
}

pub(super) fn cummax(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    // `f64::max` ignores NaN (returns the other operand), which kept the
    // running max moving correctly but then pushed that carried-forward
    // value at the NaN's own row instead of NaN -- pandas' `cummax` leaves
    // the NaN row as NaN while still tracking the max across it.
    let mut cummax = f64::NEG_INFINITY;
    let result: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                cummax = cummax.max(v);
                cummax
            }
        })
        .collect();
    Ok(result)
}

pub(super) fn cummin(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    // See `cummax`: report NaN at the NaN's own row instead of silently
    // filling it in with the running minimum.
    let mut cummin = f64::INFINITY;
    let result: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                cummin = cummin.min(v);
                cummin
            }
        })
        .collect();
    Ok(result)
}

pub(super) fn shift(df: &DataFrame, column: &str, periods: i32) -> Result<Vec<Option<f64>>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn nunique(df: &DataFrame) -> Result<Vec<(String, usize)>> {
    let col_names = df.column_names();
    let mut results = Vec::new();
    for col_name in col_names {
        let values = df.get_column_string_values(col_name)?;
        let unique_values: HashSet<String> = values.into_iter().collect();
        results.push((col_name.clone(), unique_values.len()));
    }
    Ok(results)
}

/// Compute the DataFrame's real approximate memory footprint by summing each
/// column's actual materialized data, rather than a placeholder formula
/// unrelated to the column contents (the previous `cols*rows*8 + cols*64 +
/// 256` charged every column -- string or numeric, wide or narrow -- the
/// same flat rate and added constants with no basis in the data).
pub(super) fn memory_usage(df: &DataFrame) -> usize {
    let mut total = std::mem::size_of::<usize>(); // DataFrame's own row-count field
    for col_name in df.column_names() {
        // The column's own name/label storage.
        total += col_name.len();
        if df.is_numeric_column(col_name) {
            // Every numeric element type this DataFrame stores (i64/f64,
            // and the narrower i32/f32) is materialized as `f64` by the
            // public accessor; 8 bytes/element matches the two widest --
            // and, in this crate, most common -- numeric element types.
            total += df.row_count() * std::mem::size_of::<f64>();
        } else if let Ok(values) = df.get_column_string_values(col_name) {
            // Real per-string byte length plus the `String` struct's own
            // (pointer, length, capacity) footprint: an actual sum over
            // the materialized data, not an estimate.
            total += values
                .iter()
                .map(|s| s.len() + std::mem::size_of::<String>())
                .sum::<usize>();
        }
    }
    total
}

pub(super) fn value_counts(df: &DataFrame, column: &str) -> Result<Vec<(String, usize)>> {
    let values = df.get_column_string_values(column)?;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    let mut result: Vec<(String, usize)> = counts.into_iter().collect();
    result.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Ok(result)
}

pub(super) fn value_counts_numeric(df: &DataFrame, column: &str) -> Result<Vec<(f64, usize)>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn describe(df: &DataFrame, column: &str) -> Result<DescribeStats> {
    let raw_values = df.get_column_numeric_values(column)?;
    if raw_values.is_empty() {
        return Err(Error::Empty("Cannot describe empty column".to_string()));
    }
    // pandas' `describe()` counts and summarizes only non-null values
    // (skipna=True); the previous implementation folded any NaN straight
    // into the running sum, turning `mean`/`std`/every percentile into NaN
    // from a single missing value anywhere in the column.
    let values: Vec<f64> = raw_values.into_iter().filter(|v| !v.is_nan()).collect();
    let count = values.len();
    if count == 0 {
        // All-NaN column: pandas reports count=0 and NaN for the rest
        // rather than erroring (an *empty* column, handled above, is the
        // only case that raises).
        return Ok(DescribeStats {
            count: 0,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            q25: f64::NAN,
            q50: f64::NAN,
            q75: f64::NAN,
            max: f64::NAN,
        });
    }
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;
    // Sample standard deviation (ddof=1), pandas' default -- undefined
    // (NaN) for a single observation rather than reporting a spurious 0.
    let std = if count < 2 {
        f64::NAN
    } else {
        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
        variance.sqrt()
    };
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min = sorted[0];
    let max = sorted[count - 1];
    let percentile = |p: f64| -> f64 {
        if count == 1 {
            return sorted[0];
        }
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

pub(super) fn apply<F, T>(df: &DataFrame, func: F, axis: Axis) -> Result<Vec<T>>
where
    F: Fn(&[f64]) -> T,
{
    match axis {
        Axis::Rows => {
            // Materialize each numeric column once up front instead of
            // re-fetching (and re-downcasting) every column on every row --
            // the previous version called `get_column_numeric_values`
            // inside the row loop, making this O(rows * cols) column
            // materializations instead of O(cols).
            let numeric_cols: Vec<Vec<f64>> = df
                .column_names()
                .iter()
                .filter_map(|col| df.get_column_numeric_values(col).ok())
                .collect();
            let mut result = Vec::with_capacity(df.row_count());
            for i in 0..df.row_count() {
                let row_values: Vec<f64> = numeric_cols
                    .iter()
                    .filter_map(|vals| vals.get(i).copied())
                    .collect();
                result.push(func(&row_values));
            }
            Ok(result)
        }
        Axis::Columns => {
            let col_names = df.column_names();
            let mut result = Vec::with_capacity(col_names.len());
            for col in col_names {
                if let Ok(values) = df.get_column_numeric_values(&col) {
                    result.push(func(&values));
                }
            }
            Ok(result)
        }
    }
}

pub(super) fn corr(df: &DataFrame) -> Result<CorrelationMatrix> {
    let col_names = df.column_names();
    let n_cols = col_names.len();
    if n_cols == 0 {
        return Err(Error::Empty(
            "Cannot compute correlation on empty DataFrame".to_string(),
        ));
    }
    let mut columns_data: Vec<Vec<f64>> = Vec::new();
    let mut valid_columns: Vec<String> = Vec::new();
    for col_name in col_names {
        if let Ok(values) = df.get_column_numeric_values(col_name) {
            columns_data.push(values);
            valid_columns.push(col_name.clone());
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

pub(super) fn cov(df: &DataFrame) -> Result<CorrelationMatrix> {
    let col_names = df.column_names();
    let mut columns_data: Vec<Vec<f64>> = Vec::new();
    let mut valid_columns: Vec<String> = Vec::new();
    for col_name in col_names {
        if let Ok(values) = df.get_column_numeric_values(col_name) {
            columns_data.push(values);
            valid_columns.push(col_name.clone());
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

pub(super) fn pct_change(df: &DataFrame, column: &str, periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn diff(df: &DataFrame, column: &str, periods: usize) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut result = vec![f64::NAN; values.len()];
    for i in periods..values.len() {
        result[i] = values[i] - values[i - periods];
    }
    Ok(result)
}

pub(super) fn replace(
    df: &DataFrame,
    column: &str,
    to_replace: &[&str],
    values: &[&str],
) -> Result<DataFrame> {
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
    let col_values = df.get_column_string_values(column)?;
    let replaced: Vec<String> = col_values
        .iter()
        .map(|v| {
            replacement_map
                .get(v.as_str())
                .map(|&new| new.to_string())
                .unwrap_or_else(|| v.clone())
        })
        .collect();
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(replaced.clone(), Some(col_name.clone()))?,
            )?;
        } else {
            if let Ok(values) = df.get_column_string_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = df.get_column_numeric_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }
    }
    Ok(new_df)
}

pub(super) fn replace_numeric(
    df: &DataFrame,
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
    let col_values = df.get_column_numeric_values(column)?;
    let replaced: Vec<f64> = col_values
        .iter()
        .map(|&v| replacement_map.get(&v.to_bits()).copied().unwrap_or(v))
        .collect();
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(replaced.clone(), Some(col_name.clone()))?,
            )?;
        } else {
            if let Ok(values) = df.get_column_string_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = df.get_column_numeric_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }
    }
    Ok(new_df)
}

pub(super) fn sample(df: &DataFrame, n: usize, replace: bool) -> Result<DataFrame> {
    use scirs2_core::random::RngExt;
    use scirs2_core::random::SliceRandom;
    let n_rows = df.row_count();
    if n_rows == 0 {
        return Ok(DataFrame::new());
    }
    if !replace && n > n_rows {
        return Err(Error::InvalidValue(format!(
            "Cannot sample {} rows without replacement from {} rows",
            n, n_rows
        )));
    }
    let mut rng = scirs2_core::random::rng();
    let indices: Vec<usize> = if replace {
        (0..n).map(|_| rng.random_range(0..n_rows)).collect()
    } else {
        let mut all_indices: Vec<usize> = (0..n_rows).collect();
        all_indices.shuffle(&mut rng);
        all_indices.into_iter().take(n).collect()
    };
    select_rows_by_indices(df, &indices)
}

pub(super) fn drop_columns(df: &DataFrame, labels: &[&str]) -> Result<DataFrame> {
    let mut new_df = DataFrame::new();
    let drop_set: HashSet<&str> = labels.iter().copied().collect();
    for col_name in df.column_names() {
        if !drop_set.contains(col_name.as_str()) {
            if let Ok(values) = df.get_column_numeric_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = df.get_column_string_values(&col_name) {
                new_df.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }
    }
    Ok(new_df)
}

pub(super) fn rename_columns(
    df: &DataFrame,
    mapper: &HashMap<String, String>,
) -> Result<DataFrame> {
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        let new_name = mapper.get(col_name.as_str()).unwrap_or(col_name);
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(
                new_name.clone(),
                Series::new(values, Some(new_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            new_df.add_column(
                new_name.clone(),
                Series::new(values, Some(new_name.clone()))?,
            )?;
        }
    }
    Ok(new_df)
}

pub(super) fn abs(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let abs_values: Vec<f64> = values.iter().map(|&v| v.abs()).collect();
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(abs_values.clone(), Some(col_name.clone()))?,
            )?;
        } else {
            if let Ok(vals) = df.get_column_numeric_values(&col_name) {
                new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = df.get_column_string_values(&col_name) {
                new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
    }
    Ok(new_df)
}

pub(super) fn round(df: &DataFrame, column: &str, decimals: i32) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let multiplier = 10f64.powi(decimals);
    let rounded: Vec<f64> = values
        .iter()
        .map(|&v| (v * multiplier).round() / multiplier)
        .collect();
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(rounded.clone(), Some(col_name.clone()))?,
            )?;
        } else {
            if let Ok(vals) = df.get_column_numeric_values(&col_name) {
                new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            } else if let Ok(vals) = df.get_column_string_values(&col_name) {
                new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
            }
        }
    }
    Ok(new_df)
}

pub(super) fn quantile(df: &DataFrame, column: &str, q: f64) -> Result<f64> {
    if q < 0.0 || q > 1.0 {
        return Err(Error::InvalidValue(
            "Quantile must be between 0 and 1".to_string(),
        ));
    }
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn head(df: &DataFrame, n: usize) -> Result<DataFrame> {
    let n_rows = df.row_count();
    let take = n.min(n_rows);
    let indices: Vec<usize> = (0..take).collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn tail(df: &DataFrame, n: usize) -> Result<DataFrame> {
    let n_rows = df.row_count();
    if n_rows == 0 {
        return Ok(DataFrame::new());
    }
    let start = if n >= n_rows { 0 } else { n_rows - n };
    let indices: Vec<usize> = (start..n_rows).collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn unique(df: &DataFrame, column: &str) -> Result<Vec<String>> {
    let values = df.get_column_string_values(column)?;
    let unique_set: HashSet<String> = values.into_iter().collect();
    let mut result: Vec<String> = unique_set.into_iter().collect();
    result.sort();
    Ok(result)
}

pub(super) fn unique_numeric(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let unique_set: HashSet<u64> = values.iter().map(|v| v.to_bits()).collect();
    let mut result: Vec<f64> = unique_set.into_iter().map(f64::from_bits).collect();
    result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    Ok(result)
}

pub(super) fn fillna(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let filled: Vec<f64> = values
        .iter()
        .map(|&v| if v.is_nan() { value } else { v })
        .collect();
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(filled.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(new_df)
}

pub(super) fn fillna_method(df: &DataFrame, column: &str, method: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
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
            return Err(Error::InvalidValue(format!(
                "Invalid fill method: '{}'. Use 'ffill' or 'bfill'.",
                method
            )));
        }
    };
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(filled.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(new_df)
}

pub(super) fn interpolate(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
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
                        interpolated[j] = prev_valid_val + (value_diff * position / gap_size);
                    }
                }
                prev_valid_idx = i;
                prev_valid_val = values[i];
            }
        }
    }
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(interpolated.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(new_df)
}

pub(super) fn dropna(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let valid_indices: Vec<usize> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, _)| i)
        .collect();
    select_rows_by_indices(df, &valid_indices)
}

pub(super) fn isna(df: &DataFrame, column: &str) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| v.is_nan()).collect())
}

pub(super) fn sum_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let sum: f64 = values.iter().filter(|v| !v.is_nan()).sum();
            results.push((col_name.clone(), sum));
        }
    }
    Ok(results)
}

pub(super) fn mean_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if !valid_values.is_empty() {
                let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                results.push((col_name.clone(), mean));
            }
        }
    }
    Ok(results)
}

pub(super) fn std_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if valid_values.len() > 1 {
                let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                let variance: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (valid_values.len() - 1) as f64;
                results.push((col_name.clone(), variance.sqrt()));
            }
        }
    }
    Ok(results)
}

pub(super) fn var_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if valid_values.len() > 1 {
                let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                let variance: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (valid_values.len() - 1) as f64;
                results.push((col_name.clone(), variance));
            }
        }
    }
    Ok(results)
}

pub(super) fn min_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if !valid_values.is_empty() {
                let min = valid_values.iter().cloned().fold(f64::INFINITY, f64::min);
                results.push((col_name.clone(), min));
            }
        }
    }
    Ok(results)
}

pub(super) fn max_all(df: &DataFrame) -> Result<Vec<(String, f64)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
            if !valid_values.is_empty() {
                let max = valid_values
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                results.push((col_name.clone(), max));
            }
        }
    }
    Ok(results)
}

/// A single column's materialized values, tagged by the DataFrame's actual
/// stored dtype (via [`DataFrame::is_numeric_column`], a concrete-type
/// downcast -- not "does this parse as a number", which is what the old
/// numeric-first-then-string dispatch effectively tested). Object columns
/// whose text happens to look numeric (e.g. `"007"`) must sort lexically,
/// not as floats.
enum SortColumn {
    Numeric(Vec<f64>),
    Text(Vec<String>),
}

impl SortColumn {
    fn extract(df: &DataFrame, column: &str) -> Result<Self> {
        if df.is_numeric_column(column) {
            Ok(SortColumn::Numeric(df.get_column_numeric_values(column)?))
        } else {
            Ok(SortColumn::Text(df.get_column_string_values(column)?))
        }
    }

    /// Total-order comparison of rows `i` and `j`. Numeric NaNs always sort
    /// last regardless of `ascending`, matching pandas' `na_position="last"`
    /// default; `f64::total_cmp` (rather than `partial_cmp().unwrap_or(Equal)`)
    /// keeps the non-NaN case a real total order so the overall sort can't
    /// produce non-transitive garbage or panic.
    fn compare(&self, i: usize, j: usize, ascending: bool) -> Ordering {
        // NB: `ascending` only flips the *value* comparison, never the
        // NaN-vs-non-NaN placement -- NaN must sort last in both
        // directions. Reversing the whole `Ordering` (including the NaN
        // branches) would have put NaN *first* under `ascending: false`.
        match self {
            SortColumn::Numeric(vals) => {
                let vi = vals.get(i).copied().unwrap_or(f64::NAN);
                let vj = vals.get(j).copied().unwrap_or(f64::NAN);
                match (vi.is_nan(), vj.is_nan()) {
                    (true, true) => Ordering::Equal,
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    (false, false) => {
                        let cmp = vi.total_cmp(&vj);
                        if ascending {
                            cmp
                        } else {
                            cmp.reverse()
                        }
                    }
                }
            }
            SortColumn::Text(vals) => {
                let vi = vals.get(i).map(|s| s.as_str()).unwrap_or("");
                let vj = vals.get(j).map(|s| s.as_str()).unwrap_or("");
                let cmp = vi.cmp(vj);
                if ascending {
                    cmp
                } else {
                    cmp.reverse()
                }
            }
        }
    }
}

pub(super) fn sort_values(df: &DataFrame, column: &str, ascending: bool) -> Result<DataFrame> {
    let col = SortColumn::extract(df, column)?;
    let n_rows = df.row_count();
    let mut indices: Vec<usize> = (0..n_rows).collect();
    indices.sort_by(|&i, &j| col.compare(i, j, ascending));
    select_rows_by_indices(df, &indices)
}

pub(super) fn sort_by_columns(
    df: &DataFrame,
    columns: &[&str],
    ascending: &[bool],
) -> Result<DataFrame> {
    if columns.len() != ascending.len() {
        return Err(Error::InvalidValue(
            "Number of columns must match number of ascending flags".to_string(),
        ));
    }
    if columns.is_empty() {
        return Ok(df.clone());
    }
    let mut column_values: Vec<SortColumn> = Vec::with_capacity(columns.len());
    for &col in columns {
        column_values.push(SortColumn::extract(df, col)?);
    }
    let n_rows = df.row_count();
    let mut indices: Vec<usize> = (0..n_rows).collect();
    indices.sort_by(|&i, &j| {
        for (col, &asc) in column_values.iter().zip(ascending.iter()) {
            let ord_cmp = col.compare(i, j, asc);
            if ord_cmp != Ordering::Equal {
                return ord_cmp;
            }
        }
        Ordering::Equal
    });
    select_rows_by_indices(df, &indices)
}

pub(super) fn merge(
    df: &DataFrame,
    other: &DataFrame,
    on: &str,
    how: merge_mod::JoinType,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    merge_mod::merge(df, other, on, how, suffixes)
}

pub(super) fn where_cond(
    df: &DataFrame,
    column: &str,
    condition: &[bool],
    other: f64,
) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
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
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(replaced.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(new_df)
}

pub(super) fn mask(
    df: &DataFrame,
    column: &str,
    condition: &[bool],
    other: f64,
) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
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
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            new_df.add_column(
                col_name.clone(),
                Series::new(replaced.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            new_df.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(new_df)
}

pub(super) fn drop_duplicates(
    df: &DataFrame,
    subset: Option<&[&str]>,
    keep: &str,
) -> Result<DataFrame> {
    let columns_to_check: Vec<String> = match subset {
        Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
        None => df.column_names().to_vec(),
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
    // Materialize each key column once (dispatched by concrete dtype via
    // `is_numeric_column`, matching `duplicated`/`duplicated_rows`) instead
    // of re-fetching it once per row per column, which was
    // O(rows^2 * cols).
    enum KeyColumn {
        Numeric(Vec<f64>),
        Text(Vec<String>),
    }
    let key_columns: Vec<KeyColumn> = columns_to_check
        .iter()
        .map(|col| {
            if df.is_numeric_column(col) {
                KeyColumn::Numeric(df.get_column_numeric_values(col).unwrap_or_default())
            } else {
                KeyColumn::Text(df.get_column_string_values(col).unwrap_or_default())
            }
        })
        .collect();
    let mut row_keys: Vec<String> = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut key_parts: Vec<String> = Vec::with_capacity(key_columns.len());
        for col in &key_columns {
            match col {
                KeyColumn::Text(values) => {
                    key_parts.push(values.get(row_idx).cloned().unwrap_or_default());
                }
                KeyColumn::Numeric(values) => {
                    let v = values.get(row_idx).copied().unwrap_or(f64::NAN);
                    key_parts.push(v.to_bits().to_string());
                }
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
            return Err(Error::InvalidValue(format!(
                "Invalid keep value: '{}'. Use 'first', 'last', or 'none'.",
                keep
            )));
        }
    }
    indices_to_keep.sort_unstable();
    select_rows_by_indices(df, &indices_to_keep)
}

pub(super) fn select_dtypes(df: &DataFrame, include: &[&str]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        let is_numeric = df.get_column_numeric_values(&col_name).is_ok();
        let is_string = !is_numeric && df.get_column_string_values(&col_name).is_ok();
        let should_include = include.iter().any(|&dtype| {
            (dtype == "numeric" || dtype == "number" || dtype == "float64" || dtype == "int64")
                && is_numeric
                || (dtype == "string" || dtype == "object" || dtype == "str") && is_string
        });
        if should_include {
            if let Ok(values) = df.get_column_numeric_values(&col_name) {
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else if let Ok(values) = df.get_column_string_values(&col_name) {
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            }
        }
    }
    Ok(result)
}

pub(super) fn any_numeric(df: &DataFrame) -> Result<Vec<(String, bool)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let has_any = values.iter().any(|&v| !v.is_nan() && v != 0.0);
            results.push((col_name.clone(), has_any));
        }
    }
    Ok(results)
}

pub(super) fn all_numeric(df: &DataFrame) -> Result<Vec<(String, bool)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let all_true = values.iter().all(|&v| !v.is_nan() && v != 0.0);
            results.push((col_name.clone(), all_true));
        }
    }
    Ok(results)
}

pub(super) fn count_valid(df: &DataFrame) -> Result<Vec<(String, usize)>> {
    let mut results = Vec::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let count = values.iter().filter(|v| !v.is_nan()).count();
            results.push((col_name.clone(), count));
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            let count = values.iter().filter(|v| !v.is_empty()).count();
            results.push((col_name.clone(), count));
        }
    }
    Ok(results)
}

pub(super) fn reverse_columns(df: &DataFrame) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    let columns = df.column_names();
    for col_name in columns.into_iter().rev() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            result.add_column(
                col_name.clone(),
                Series::new(values, Some(col_name.clone()))?,
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

pub(super) fn reverse_rows(df: &DataFrame) -> Result<DataFrame> {
    let n_rows = df.row_count();
    let indices: Vec<usize> = (0..n_rows).rev().collect();
    select_rows_by_indices(df, &indices)
}

pub(super) fn notna(df: &DataFrame, column: &str) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| !v.is_nan()).collect())
}

pub(super) fn melt(
    df: &DataFrame,
    id_vars: &[&str],
    value_vars: Option<&[&str]>,
    var_name: &str,
    value_name: &str,
) -> Result<DataFrame> {
    for id_var in id_vars {
        if !df.contains_column(id_var) {
            return Err(Error::InvalidValue(format!(
                "Column '{}' not found in DataFrame",
                id_var
            )));
        }
    }
    let value_columns: Vec<String> = match value_vars {
        Some(cols) => cols.iter().map(|s| s.to_string()).collect(),
        None => {
            let id_set: std::collections::HashSet<&str> = id_vars.iter().copied().collect();
            df.column_names()
                .iter()
                .filter(|c| !id_set.contains(c.as_str()))
                .cloned()
                .collect()
        }
    };
    if value_columns.is_empty() {
        return Err(Error::InvalidValue("No value columns to melt".to_string()));
    }
    let n_rows = df.row_count();
    let n_value_cols = value_columns.len();
    let total_rows = n_rows * n_value_cols;
    let mut result = DataFrame::new();
    for id_var in id_vars {
        if let Ok(values) = df.get_column_numeric_values(id_var) {
            let mut repeated: Vec<f64> = Vec::with_capacity(total_rows);
            for _ in 0..n_value_cols {
                repeated.extend(values.iter().copied());
            }
            result.add_column(
                id_var.to_string(),
                Series::new(repeated, Some(id_var.to_string()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(id_var) {
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
    let mut all_values: Vec<f64> = Vec::with_capacity(total_rows);
    for col in &value_columns {
        if let Ok(values) = df.get_column_numeric_values(col) {
            all_values.extend(values);
        } else {
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

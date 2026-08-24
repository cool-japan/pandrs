//! Implementation functions for PandasCompatExt - element-wise ops, string ops, comparison, and utility functions

use super::super::helpers::{aggregations, comparison_ops, math_ops, string_ops};
use super::super::trait_def::PandasCompatExt;
use super::super::types::SeriesValue;
use super::functions_3::{covariance, pearson_correlation};
use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::collections::HashMap;

pub(super) fn query_contains(df: &DataFrame, column: &str, pattern: &str) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let mask: Vec<bool> = values.iter().map(|v| v.contains(pattern)).collect();
    df.filter_by_mask(&mask)
}

pub(super) fn select_columns(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for &col in columns {
        if let Ok(vals) = df.get_column_numeric_values(col) {
            result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(col) {
            result.add_column(col.to_string(), Series::new(vals, Some(col.to_string()))?)?;
        } else {
            return Err(Error::InvalidValue(format!("Column '{}' not found", col)));
        }
    }
    Ok(result)
}

pub(super) fn add_scalar(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    df.transform(column, |x| x + value)
}

pub(super) fn mul_scalar(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    df.transform(column, |x| x * value)
}

pub(super) fn sub_scalar(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    df.transform(column, |x| x - value)
}

pub(super) fn div_scalar(df: &DataFrame, column: &str, value: f64) -> Result<DataFrame> {
    df.transform(column, |x| x / value)
}

pub(super) fn pow(df: &DataFrame, column: &str, exponent: f64) -> Result<DataFrame> {
    df.transform(column, |x| x.powf(exponent))
}

pub(super) fn sqrt(df: &DataFrame, column: &str) -> Result<DataFrame> {
    df.transform(column, |x| x.sqrt())
}

pub(super) fn log(df: &DataFrame, column: &str) -> Result<DataFrame> {
    df.transform(column, |x| x.ln())
}

pub(super) fn exp(df: &DataFrame, column: &str) -> Result<DataFrame> {
    df.transform(column, |x| x.exp())
}

pub(super) fn col_add(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    if v1.len() != v2.len() {
        return Err(Error::InvalidValue(
            "Columns must have the same length".to_string(),
        ));
    }
    let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a + b).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn col_mul(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    if v1.len() != v2.len() {
        return Err(Error::InvalidValue(
            "Columns must have the same length".to_string(),
        ));
    }
    let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn col_sub(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    if v1.len() != v2.len() {
        return Err(Error::InvalidValue(
            "Columns must have the same length".to_string(),
        ));
    }
    let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a - b).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn col_div(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    if v1.len() != v2.len() {
        return Err(Error::InvalidValue(
            "Columns must have the same length".to_string(),
        ));
    }
    let result_values: Vec<f64> = v1.iter().zip(v2.iter()).map(|(&a, &b)| a / b).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn iterrows(df: &DataFrame) -> Vec<(usize, HashMap<String, SeriesValue>)> {
    // Materialize each column once (dispatched by concrete dtype via
    // `is_numeric_column`, not "does numeric parsing happen to succeed" --
    // that would turn an object column of e.g. `"007"` into `Float(7.0)`)
    // instead of re-fetching every column on every row, which made this
    // O(rows^2 * cols).
    enum ColData {
        Num(Vec<f64>),
        Str(Vec<String>),
    }
    let columns = df.column_names();
    let column_data: Vec<(String, ColData)> = columns
        .iter()
        .filter_map(|col| {
            if df.is_numeric_column(col) {
                df.get_column_numeric_values(col)
                    .ok()
                    .map(|v| (col.clone(), ColData::Num(v)))
            } else {
                df.get_column_string_values(col)
                    .ok()
                    .map(|v| (col.clone(), ColData::Str(v)))
            }
        })
        .collect();

    let mut result = Vec::with_capacity(df.row_count());
    for row_idx in 0..df.row_count() {
        let mut row_data: HashMap<String, SeriesValue> = HashMap::new();
        for (col, data) in &column_data {
            match data {
                ColData::Num(vals) => {
                    if let Some(&v) = vals.get(row_idx) {
                        row_data.insert(col.clone(), SeriesValue::Float(v));
                    }
                }
                ColData::Str(vals) => {
                    if let Some(v) = vals.get(row_idx) {
                        row_data.insert(col.clone(), SeriesValue::String(v.clone()));
                    }
                }
            }
        }
        result.push((row_idx, row_data));
    }
    result
}

pub(super) fn at(df: &DataFrame, row: usize, column: &str) -> Result<SeriesValue> {
    if row >= df.row_count() {
        return Err(Error::InvalidValue(format!(
            "Row index {} out of bounds",
            row
        )));
    }
    if let Ok(vals) = df.get_column_numeric_values(column) {
        if row < vals.len() {
            return Ok(SeriesValue::Float(vals[row]));
        }
    } else if let Ok(vals) = df.get_column_string_values(column) {
        if row < vals.len() {
            return Ok(SeriesValue::String(vals[row].clone()));
        }
    }
    Err(Error::ColumnNotFound(column.to_string()))
}

pub(super) fn iat(df: &DataFrame, row: usize, col_idx: usize) -> Result<SeriesValue> {
    let columns = df.column_names();
    if col_idx >= columns.len() {
        return Err(Error::InvalidValue(format!(
            "Column index {} out of bounds",
            col_idx
        )));
    }
    df.at(row, &columns[col_idx])
}

pub(super) fn drop_rows(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let indices_set: std::collections::HashSet<usize> = indices.iter().cloned().collect();
    let mut result = DataFrame::new();
    for col in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            let filtered: Vec<f64> = vals
                .iter()
                .enumerate()
                .filter(|(i, _)| !indices_set.contains(i))
                .map(|(_, v)| *v)
                .collect();
            result.add_column(col.clone(), Series::new(filtered, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
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

pub(super) fn set_index(
    df: &DataFrame,
    column: &str,
    drop: bool,
) -> Result<(DataFrame, Vec<String>)> {
    let index_values = df.get_column_string_values(column)?;
    let mut result = DataFrame::new();
    for col in df.column_names() {
        if drop && col == column {
            continue;
        }
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    Ok((result, index_values))
}

pub(super) fn reset_index(
    df: &DataFrame,
    index_values: Option<&[String]>,
    name: &str,
) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    if let Some(idx_vals) = index_values {
        result.add_column(
            name.to_string(),
            Series::new(idx_vals.to_vec(), Some(name.to_string()))?,
        )?;
    }
    for col in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn to_records(df: &DataFrame) -> Vec<HashMap<String, SeriesValue>> {
    df.iterrows().into_iter().map(|(_, row)| row).collect()
}

pub(super) fn items(df: &DataFrame) -> Vec<(String, Vec<SeriesValue>)> {
    let mut result = Vec::new();
    for col in df.column_names() {
        let mut values = Vec::new();
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            values = vals.iter().map(|v| SeriesValue::Float(*v)).collect();
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            values = vals
                .iter()
                .map(|v| SeriesValue::String(v.clone()))
                .collect();
        }
        result.push((col.clone(), values));
    }
    result
}

pub(super) fn update(df: &DataFrame, other: &DataFrame) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col in df.column_names() {
        if other.contains_column(&col) {
            if let Ok(other_vals) = other.get_column_numeric_values(&col) {
                if let Ok(self_vals) = df.get_column_numeric_values(&col) {
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
                if let Ok(self_vals) = df.get_column_string_values(&col) {
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
            if let Ok(vals) = df.get_column_numeric_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            } else if let Ok(vals) = df.get_column_string_values(&col) {
                result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
            }
        }
    }
    Ok(result)
}

pub(super) fn combine<F>(df: &DataFrame, other: &DataFrame, func: F) -> Result<DataFrame>
where
    F: Fn(Option<f64>, Option<f64>) -> f64,
{
    let mut result = DataFrame::new();
    let mut all_cols: Vec<String> = df.column_names().to_vec();
    for col in other.column_names() {
        if !all_cols.contains(col) {
            all_cols.push(col.to_string());
        }
    }
    let max_rows = std::cmp::max(df.row_count(), other.row_count());
    for col in all_cols {
        let self_vals = df.get_column_numeric_values(&col).ok();
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

pub(super) fn shape(df: &DataFrame) -> (usize, usize) {
    (df.row_count(), df.column_names().len())
}

pub(super) fn size(df: &DataFrame) -> usize {
    df.row_count() * df.column_names().len()
}

pub(super) fn empty(df: &DataFrame) -> bool {
    df.row_count() == 0 || df.column_names().is_empty()
}

pub(super) fn first_row(df: &DataFrame) -> Result<HashMap<String, SeriesValue>> {
    if df.row_count() == 0 {
        return Err(Error::InvalidValue("DataFrame is empty".to_string()));
    }
    let rows = df.iterrows();
    Ok(rows
        .into_iter()
        .next()
        .ok_or_else(|| Error::InsufficientData("No rows available".to_string()))?
        .1)
}

pub(super) fn last_row(df: &DataFrame) -> Result<HashMap<String, SeriesValue>> {
    if df.row_count() == 0 {
        return Err(Error::InvalidValue("DataFrame is empty".to_string()));
    }
    let rows = df.iterrows();
    Ok(rows
        .into_iter()
        .last()
        .ok_or_else(|| Error::InsufficientData("No rows available".to_string()))?
        .1)
}

pub(super) fn get_value(
    df: &DataFrame,
    row: usize,
    column: &str,
    default: SeriesValue,
) -> SeriesValue {
    df.at(row, column).unwrap_or(default)
}

pub(super) fn lookup(
    df: &DataFrame,
    lookup_col: &str,
    other: &DataFrame,
    other_col: &str,
    result_col: &str,
) -> Result<DataFrame> {
    let lookup_vals = df.get_column_string_values(lookup_col)?;
    let other_keys = other.get_column_string_values(other_col)?;
    let other_result = other.get_column_string_values(result_col).or_else(|_| {
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
    for col in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    result.add_column(
        format!("{}_result", result_col),
        Series::new(result_values, Some(format!("{}_result", result_col)))?,
    )?;
    Ok(result)
}

pub(super) fn get_column_by_index(
    df: &DataFrame,
    idx: usize,
) -> Result<(String, Vec<SeriesValue>)> {
    let columns = df.column_names();
    if idx >= columns.len() {
        return Err(Error::InvalidValue(format!(
            "Column index {} out of bounds",
            idx
        )));
    }
    let col_name = &columns[idx];
    let mut values = Vec::new();
    if let Ok(vals) = df.get_column_numeric_values(col_name) {
        values = vals.iter().map(|v| SeriesValue::Float(*v)).collect();
    } else if let Ok(vals) = df.get_column_string_values(col_name) {
        values = vals
            .iter()
            .map(|v| SeriesValue::String(v.clone()))
            .collect();
    }
    Ok((col_name.clone(), values))
}

pub(super) fn swap_columns(df: &DataFrame, col1: &str, col2: &str) -> Result<DataFrame> {
    if !df.contains_column(col1) {
        return Err(Error::ColumnNotFound(col1.to_string()));
    }
    if !df.contains_column(col2) {
        return Err(Error::ColumnNotFound(col2.to_string()));
    }
    let mut result = DataFrame::new();
    for col in df.column_names() {
        let target_col = if col == col1 {
            col2
        } else if col == col2 {
            col1
        } else {
            &col
        };
        if let Ok(vals) = df.get_column_numeric_values(target_col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(target_col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn sort_columns(df: &DataFrame, ascending: bool) -> Result<DataFrame> {
    let mut columns = df.column_names().to_vec();
    if ascending {
        columns.sort();
    } else {
        columns.sort_by(|a, b| b.cmp(a));
    }
    let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
    df.reindex_columns(&col_refs)
}

pub(super) fn rename_column(df: &DataFrame, old_name: &str, new_name: &str) -> Result<DataFrame> {
    let mut mapper = HashMap::new();
    mapper.insert(old_name.to_string(), new_name.to_string());
    df.rename_columns(&mapper)
}

pub(super) fn to_categorical(
    df: &DataFrame,
    column: &str,
) -> Result<(DataFrame, HashMap<String, i64>)> {
    let values = df.get_column_string_values(column)?;
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
    for col in df.column_names() {
        if col == column {
            result.add_column(col.clone(), Series::new(codes.clone(), Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(vals, Some(col.clone()))?)?;
        }
    }
    Ok((result, category_map))
}

pub(super) fn row_hash(df: &DataFrame) -> Vec<u64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let rows = df.iterrows();
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

pub(super) fn sample_frac(df: &DataFrame, frac: f64, replace: bool) -> Result<DataFrame> {
    if frac < 0.0 || frac > 1.0 {
        return Err(Error::InvalidValue(
            "Fraction must be between 0 and 1".to_string(),
        ));
    }
    let n = (df.row_count() as f64 * frac).round() as usize;
    PandasCompatExt::sample(df, n, replace)
}

pub(super) fn take(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col in df.column_names() {
        if let Ok(vals) = df.get_column_numeric_values(&col) {
            let taken: Vec<f64> = indices
                .iter()
                .filter_map(|&i| vals.get(i).copied())
                .collect();
            result.add_column(col.clone(), Series::new(taken, Some(col.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col) {
            let taken: Vec<String> = indices
                .iter()
                .filter_map(|&i| vals.get(i).cloned())
                .collect();
            result.add_column(col.clone(), Series::new(taken, Some(col.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn duplicated_rows(
    df: &DataFrame,
    subset: Option<&[&str]>,
    keep: &str,
) -> Result<Vec<bool>> {
    // `duplicated_rows` and `duplicated` (functions_2_impl_part2) are the
    // same operation under two trait method names; delegate to the
    // subset-aware, dtype-aware implementation instead of maintaining a
    // second copy. The previous body computed a *whole-row* hash via
    // `df.row_hash()` unconditionally, so `subset` was read into
    // `cols_to_check` and then never consulted (`let _ = cols_to_check;`) --
    // `duplicated_rows(Some(&["a"]), ..)` compared every column, not just
    // `"a"`.
    super::functions_2_impl_part2::duplicated(df, subset, keep)
}

pub(super) fn get_column_as_f64(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    df.get_column_numeric_values(column)
}

pub(super) fn get_column_as_string(df: &DataFrame, column: &str) -> Result<Vec<String>> {
    if let Ok(vals) = df.get_column_string_values(column) {
        Ok(vals)
    } else if let Ok(vals) = df.get_column_numeric_values(column) {
        Ok(vals.iter().map(|v| v.to_string()).collect())
    } else {
        Err(Error::ColumnNotFound(column.to_string()))
    }
}

pub(super) fn groupby_apply<F>(df: &DataFrame, by: &str, func: F) -> Result<DataFrame>
where
    F: Fn(&DataFrame) -> Result<HashMap<String, f64>>,
{
    let group_vals = df.get_column_string_values(by).or_else(|_| {
        df.get_column_numeric_values(by)
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
        let subset = df.take(&indices)?;
        let agg_result = func(&subset)?;
        group_names.push(group);
        for (key, value) in agg_result {
            result_data.entry(key).or_default().push(value);
        }
    }
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

pub(super) fn corr_columns(df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    Ok(pearson_correlation(&v1, &v2))
}

pub(super) fn cov_columns(df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
    let v1 = df.get_column_numeric_values(col1)?;
    let v2 = df.get_column_numeric_values(col2)?;
    Ok(covariance(&v1, &v2))
}

pub(super) fn var_column(df: &DataFrame, column: &str, ddof: usize) -> Result<f64> {
    let vals = df.get_column_numeric_values(column)?;
    let valid: Vec<f64> = vals.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.len() <= ddof {
        return Ok(f64::NAN);
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let variance = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - ddof as f64);
    Ok(variance)
}

pub(super) fn std_column(df: &DataFrame, column: &str, ddof: usize) -> Result<f64> {
    Ok(df.var_column(column, ddof)?.sqrt())
}

pub(super) fn str_lower(df: &DataFrame, column: &str) -> Result<DataFrame> {
    string_ops::str_lower(df, column)
}

pub(super) fn str_upper(df: &DataFrame, column: &str) -> Result<DataFrame> {
    string_ops::str_upper(df, column)
}

pub(super) fn str_strip(df: &DataFrame, column: &str) -> Result<DataFrame> {
    string_ops::str_strip(df, column)
}

pub(super) fn str_contains(df: &DataFrame, column: &str, pattern: &str) -> Result<Vec<bool>> {
    string_ops::str_contains(df, column, pattern)
}

pub(super) fn str_replace(
    df: &DataFrame,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> Result<DataFrame> {
    string_ops::str_replace(df, column, pattern, replacement)
}

pub(super) fn str_split(df: &DataFrame, column: &str, delimiter: &str) -> Result<Vec<Vec<String>>> {
    string_ops::str_split(df, column, delimiter)
}

pub(super) fn str_len(df: &DataFrame, column: &str) -> Result<Vec<usize>> {
    string_ops::str_len(df, column)
}

pub(super) fn sem(df: &DataFrame, column: &str, ddof: usize) -> Result<f64> {
    aggregations::sem(df, column, ddof)
}

pub(super) fn mad(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::mad(df, column)
}

pub(super) fn ffill(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let mut filled = Vec::with_capacity(values.len());
    let mut last_valid = f64::NAN;
    for v in &values {
        if !v.is_nan() {
            last_valid = *v;
        }
        filled.push(last_valid);
    }
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(filled.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn bfill(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let mut filled = vec![f64::NAN; values.len()];
    let mut last_valid = f64::NAN;
    for i in (0..values.len()).rev() {
        if !values[i].is_nan() {
            last_valid = values[i];
        }
        filled[i] = last_valid;
    }
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(filled.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn pct_rank(df: &DataFrame, column: &str) -> Result<Vec<f64>> {
    let values = df.get_column_numeric_values(column)?;
    let n = values.len();
    let mut result = vec![f64::NAN; n];
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
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        result[*idx] = rank as f64 / (valid_count - 1).max(1) as f64;
    }
    Ok(result)
}

pub(super) fn abs_column(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::abs_column(df, column)
}

pub(super) fn round_column(df: &DataFrame, column: &str, decimals: i32) -> Result<DataFrame> {
    math_ops::round_column(df, column, decimals)
}

pub(super) fn argmax(df: &DataFrame, column: &str) -> Result<usize> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn argmin(df: &DataFrame, column: &str) -> Result<usize> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn gt(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::gt(df, column, value)
}

pub(super) fn ge(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::ge(df, column, value)
}

pub(super) fn lt(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::lt(df, column, value)
}

pub(super) fn le(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::le(df, column, value)
}

pub(super) fn eq_value(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::eq_value(df, column, value)
}

pub(super) fn ne_value(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    comparison_ops::ne_value(df, column, value)
}

pub(super) fn clip_lower(df: &DataFrame, column: &str, min: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let clipped: Vec<f64> = values.iter().map(|v| v.max(min)).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(clipped.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn clip_upper(df: &DataFrame, column: &str, max: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let clipped: Vec<f64> = values.iter().map(|v| v.min(max)).collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(clipped.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn any_column(df: &DataFrame, column: &str) -> Result<bool> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().any(|v| !v.is_nan() && *v != 0.0))
}

pub(super) fn all_column(df: &DataFrame, column: &str) -> Result<bool> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().all(|v| !v.is_nan() && *v != 0.0))
}

pub(super) fn count_na(df: &DataFrame, column: &str) -> Result<usize> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().filter(|v| v.is_nan()).count())
}

pub(super) fn prod(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::prod(df, column)
}

pub(super) fn coalesce(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let values1 = df.get_column_numeric_values(col1)?;
    let values2 = df.get_column_numeric_values(col2)?;
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
    let mut result = df.copy();
    result.add_column(
        result_name.to_string(),
        Series::new(coalesced, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn first_valid(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    for v in &values {
        if !v.is_nan() {
            return Ok(*v);
        }
    }
    Ok(f64::NAN)
}

pub(super) fn last_valid(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    for v in values.iter().rev() {
        if !v.is_nan() {
            return Ok(*v);
        }
    }
    Ok(f64::NAN)
}

pub(super) fn add_columns(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let values1 = df.get_column_numeric_values(col1)?;
    let values2 = df.get_column_numeric_values(col2)?;
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
    let mut result = df.copy();
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn sub_columns(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let values1 = df.get_column_numeric_values(col1)?;
    let values2 = df.get_column_numeric_values(col2)?;
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
    let mut result = df.copy();
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn mul_columns(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let values1 = df.get_column_numeric_values(col1)?;
    let values2 = df.get_column_numeric_values(col2)?;
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
    let mut result = df.copy();
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn div_columns(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    result_name: &str,
) -> Result<DataFrame> {
    let values1 = df.get_column_numeric_values(col1)?;
    let values2 = df.get_column_numeric_values(col2)?;
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
    let mut result = df.copy();
    result.add_column(
        result_name.to_string(),
        Series::new(result_values, Some(result_name.to_string()))?,
    )?;
    Ok(result)
}

pub(super) fn mod_column(df: &DataFrame, column: &str, divisor: f64) -> Result<DataFrame> {
    math_ops::mod_column(df, column, divisor)
}

pub(super) fn floordiv(df: &DataFrame, column: &str, divisor: f64) -> Result<DataFrame> {
    math_ops::floordiv(df, column, divisor)
}

pub(super) fn neg(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::neg(df, column)
}

pub(super) fn sign(df: &DataFrame, column: &str) -> Result<Vec<i32>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn is_finite(df: &DataFrame, column: &str) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| v.is_finite()).collect())
}

pub(super) fn is_infinite(df: &DataFrame, column: &str) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| v.is_infinite()).collect())
}

pub(super) fn replace_inf(df: &DataFrame, column: &str, replacement: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values
        .iter()
        .map(|v| if v.is_infinite() { replacement } else { *v })
        .collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(result_values.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn str_startswith(df: &DataFrame, column: &str, prefix: &str) -> Result<Vec<bool>> {
    string_ops::str_startswith(df, column, prefix)
}

pub(super) fn str_endswith(df: &DataFrame, column: &str, suffix: &str) -> Result<Vec<bool>> {
    string_ops::str_endswith(df, column, suffix)
}

pub(super) fn str_pad_left(
    df: &DataFrame,
    column: &str,
    width: usize,
    fillchar: char,
) -> Result<DataFrame> {
    string_ops::str_pad_left(df, column, width, fillchar)
}

pub(super) fn str_pad_right(
    df: &DataFrame,
    column: &str,
    width: usize,
    fillchar: char,
) -> Result<DataFrame> {
    string_ops::str_pad_right(df, column, width, fillchar)
}

pub(super) fn str_slice(
    df: &DataFrame,
    column: &str,
    start: usize,
    end: Option<usize>,
) -> Result<DataFrame> {
    string_ops::str_slice(df, column, start, end)
}

pub(super) fn floor(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::floor(df, column)
}

pub(super) fn ceil(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::ceil(df, column)
}

pub(super) fn trunc(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::trunc(df, column)
}

pub(super) fn fract(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::fract(df, column)
}

pub(super) fn reciprocal(df: &DataFrame, column: &str) -> Result<DataFrame> {
    math_ops::reciprocal(df, column)
}

pub(super) fn count_value(df: &DataFrame, column: &str, value: f64) -> Result<usize> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values
        .iter()
        .filter(|v| !v.is_nan() && (*v - value).abs() < f64::EPSILON)
        .count())
}

pub(super) fn fillna_zero(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(result_values.clone(), Some(col_name.clone()))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

pub(super) fn nunique_all(df: &DataFrame) -> Result<HashMap<String, usize>> {
    let mut result = HashMap::new();
    for col_name in df.column_names() {
        let count = if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            let unique: std::collections::HashSet<_> = vals
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| v.to_bits())
                .collect();
            unique.len()
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            let unique: std::collections::HashSet<_> = vals.iter().collect();
            unique.len()
        } else {
            0
        };
        result.insert(col_name.to_string(), count);
    }
    Ok(result)
}

pub(super) fn is_between(
    df: &DataFrame,
    column: &str,
    lower: f64,
    upper: f64,
    inclusive: bool,
) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
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

pub(super) fn str_count(df: &DataFrame, column: &str, pattern: &str) -> Result<Vec<usize>> {
    string_ops::str_count(df, column, pattern)
}

pub(super) fn str_repeat(df: &DataFrame, column: &str, n: usize) -> Result<DataFrame> {
    string_ops::str_repeat(df, column, n)
}

pub(super) fn str_center(
    df: &DataFrame,
    column: &str,
    width: usize,
    fillchar: char,
) -> Result<DataFrame> {
    string_ops::str_center(df, column, width, fillchar)
}

pub(super) fn str_zfill(df: &DataFrame, column: &str, width: usize) -> Result<DataFrame> {
    string_ops::str_zfill(df, column, width)
}

pub(super) fn is_numeric_column(df: &DataFrame, column: &str) -> bool {
    DataFrame::get_column_numeric_values(df, column).is_ok()
}

pub(super) fn is_string_column(df: &DataFrame, column: &str) -> bool {
    DataFrame::get_column_string_values(df, column).is_ok()
}

pub(super) fn has_nulls(df: &DataFrame, column: &str) -> Result<bool> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().any(|v| v.is_nan()))
}

pub(super) fn describe_column(df: &DataFrame, column: &str) -> Result<HashMap<String, f64>> {
    aggregations::describe_column(df, column)
}

pub(super) fn memory_usage_column(df: &DataFrame, column: &str) -> Result<usize> {
    if let Ok(vals) = df.get_column_numeric_values(column) {
        Ok(vals.len() * std::mem::size_of::<f64>())
    } else if let Ok(vals) = df.get_column_string_values(column) {
        Ok(
            vals.iter().map(|s| s.len()).sum::<usize>()
                + vals.len() * std::mem::size_of::<String>(),
        )
    } else {
        Err(Error::ColumnNotFound(column.to_string()))
    }
}

pub(super) fn range(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();
    if valid.is_empty() {
        return Ok(f64::NAN);
    }
    let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok(max - min)
}

pub(super) fn abs_sum(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).sum())
}

pub(super) fn is_unique(df: &DataFrame, column: &str) -> Result<bool> {
    if let Ok(vals) = df.get_column_numeric_values(column) {
        let unique: std::collections::HashSet<_> = vals
            .iter()
            .filter(|v| !v.is_nan())
            .map(|v| v.to_bits())
            .collect();
        let valid_count = vals.iter().filter(|v| !v.is_nan()).count();
        Ok(unique.len() == valid_count)
    } else if let Ok(vals) = df.get_column_string_values(column) {
        let unique: std::collections::HashSet<_> = vals.iter().collect();
        Ok(unique.len() == vals.len())
    } else {
        Err(Error::ColumnNotFound(column.to_string()))
    }
}

pub(super) fn mode_with_count(df: &DataFrame, column: &str) -> Result<(f64, usize)> {
    let values = df.get_column_numeric_values(column)?;
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for v in &values {
        if !v.is_nan() {
            *counts.entry(v.to_bits()).or_insert(0) += 1;
        }
    }
    if counts.is_empty() {
        return Ok((f64::NAN, 0));
    }
    let (mode_bits, count) = counts.into_iter().max_by_key(|(_, c)| *c).ok_or_else(|| {
        Error::InsufficientData("No valid values for mode calculation".to_string())
    })?;
    Ok((f64::from_bits(mode_bits), count))
}

pub(super) fn geometric_mean(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::geometric_mean(df, column)
}

pub(super) fn harmonic_mean(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::harmonic_mean(df, column)
}

pub(super) fn iqr(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::iqr(df, column)
}

pub(super) fn cv(df: &DataFrame, column: &str) -> Result<f64> {
    aggregations::cv(df, column)
}

pub(super) fn percentile_value(df: &DataFrame, column: &str, q: f64) -> Result<f64> {
    aggregations::percentile_value(df, column, q)
}

pub(super) fn trimmed_mean(df: &DataFrame, column: &str, trim_fraction: f64) -> Result<f64> {
    aggregations::trimmed_mean(df, column, trim_fraction)
}

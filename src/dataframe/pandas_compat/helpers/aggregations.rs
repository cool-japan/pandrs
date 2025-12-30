//! Aggregation helper functions

use crate::core::error::Result;
use crate::dataframe::DataFrame;
use std::collections::HashMap;

/// Compute standard error of the mean for a column
pub fn sem(df: &DataFrame, column: &str, ddof: usize) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid_values.len() <= ddof {
        return Ok(f64::NAN);
    }

    let n = valid_values.len() as f64;
    let mean = valid_values.iter().sum::<f64>() / n;
    let variance: f64 = valid_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (valid_values.len() - ddof) as f64;
    let std_dev = variance.sqrt();

    Ok(std_dev / n.sqrt())
}

/// Compute mean absolute deviation for a column
pub fn mad(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid_values.is_empty() {
        return Ok(f64::NAN);
    }

    let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
    let mad =
        valid_values.iter().map(|v| (v - mean).abs()).sum::<f64>() / valid_values.len() as f64;

    Ok(mad)
}

/// Compute the product of values in a column
pub fn prod(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid_values: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid_values.is_empty() {
        return Ok(f64::NAN);
    }

    Ok(valid_values.iter().product())
}

/// Get statistics for a single numeric column
pub fn describe_column(df: &DataFrame, column: &str) -> Result<HashMap<String, f64>> {
    let values = df.get_column_numeric_values(column)?;
    let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    let mut result = HashMap::new();

    if valid.is_empty() {
        result.insert("count".to_string(), 0.0);
        result.insert("mean".to_string(), f64::NAN);
        result.insert("std".to_string(), f64::NAN);
        result.insert("min".to_string(), f64::NAN);
        result.insert("max".to_string(), f64::NAN);
        return Ok(result);
    }

    let n = valid.len() as f64;
    let sum: f64 = valid.iter().sum();
    let mean = sum / n;
    let variance: f64 = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = variance.sqrt();
    let min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    result.insert("count".to_string(), n);
    result.insert("mean".to_string(), mean);
    result.insert("std".to_string(), std);
    result.insert("min".to_string(), min);
    result.insert("max".to_string(), max);

    // Quartiles
    let mut sorted = valid.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q25_idx = (sorted.len() as f64 * 0.25) as usize;
    let q50_idx = sorted.len() / 2;
    let q75_idx = (sorted.len() as f64 * 0.75) as usize;

    result.insert(
        "25%".to_string(),
        sorted.get(q25_idx).copied().unwrap_or(f64::NAN),
    );
    result.insert(
        "50%".to_string(),
        sorted.get(q50_idx).copied().unwrap_or(f64::NAN),
    );
    result.insert(
        "75%".to_string(),
        sorted.get(q75_idx).copied().unwrap_or(f64::NAN),
    );

    Ok(result)
}

/// Compute geometric mean for a column
pub fn geometric_mean(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid_values: Vec<f64> = values
        .iter()
        .filter(|v| !v.is_nan() && **v > 0.0)
        .copied()
        .collect();

    if valid_values.is_empty() {
        return Ok(f64::NAN);
    }

    let log_sum: f64 = valid_values.iter().map(|v| v.ln()).sum();
    Ok((log_sum / valid_values.len() as f64).exp())
}

/// Compute harmonic mean for a column
pub fn harmonic_mean(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid_values: Vec<f64> = values
        .iter()
        .filter(|v| !v.is_nan() && **v != 0.0)
        .copied()
        .collect();

    if valid_values.is_empty() {
        return Ok(f64::NAN);
    }

    let reciprocal_sum: f64 = valid_values.iter().map(|v| 1.0 / v).sum();
    Ok(valid_values.len() as f64 / reciprocal_sum)
}

/// Compute interquartile range (IQR) for a column
pub fn iqr(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let mut valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid.len() < 2 {
        return Ok(f64::NAN);
    }

    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q25_idx = (valid.len() as f64 * 0.25) as usize;
    let q75_idx = (valid.len() as f64 * 0.75) as usize;

    let q25 = valid.get(q25_idx).copied().unwrap_or(f64::NAN);
    let q75 = valid.get(q75_idx).copied().unwrap_or(f64::NAN);

    Ok(q75 - q25)
}

/// Compute coefficient of variation (CV) for a column
pub fn cv(df: &DataFrame, column: &str) -> Result<f64> {
    let values = df.get_column_numeric_values(column)?;
    let valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid.is_empty() {
        return Ok(f64::NAN);
    }

    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;

    if mean.abs() < f64::EPSILON {
        return Ok(f64::NAN);
    }

    let variance: f64 = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = variance.sqrt();

    Ok(std / mean.abs())
}

/// Compute specific percentile for a column
pub fn percentile_value(df: &DataFrame, column: &str, q: f64) -> Result<f64> {
    if q < 0.0 || q > 1.0 {
        return Ok(f64::NAN);
    }

    let values = df.get_column_numeric_values(column)?;
    let mut valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid.is_empty() {
        return Ok(f64::NAN);
    }

    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let idx = ((valid.len() - 1) as f64 * q) as usize;
    Ok(valid.get(idx).copied().unwrap_or(f64::NAN))
}

/// Compute trimmed mean (excluding outliers at both ends)
pub fn trimmed_mean(df: &DataFrame, column: &str, trim_fraction: f64) -> Result<f64> {
    if trim_fraction < 0.0 || trim_fraction >= 0.5 {
        return Ok(f64::NAN);
    }

    let values = df.get_column_numeric_values(column)?;
    let mut valid: Vec<f64> = values.iter().filter(|v| !v.is_nan()).copied().collect();

    if valid.is_empty() {
        return Ok(f64::NAN);
    }

    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let trim_count = (valid.len() as f64 * trim_fraction) as usize;
    let trimmed = &valid[trim_count..valid.len() - trim_count];

    if trimmed.is_empty() {
        return Ok(f64::NAN);
    }

    Ok(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
}

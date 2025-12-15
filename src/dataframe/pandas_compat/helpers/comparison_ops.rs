//! Comparison operation helper functions

use crate::core::error::Result;
use crate::dataframe::DataFrame;

/// Element-wise greater than comparison
pub fn gt(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| !v.is_nan() && *v > value).collect())
}

/// Element-wise greater than or equal comparison
pub fn ge(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| !v.is_nan() && *v >= value).collect())
}

/// Element-wise less than comparison
pub fn lt(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| !v.is_nan() && *v < value).collect())
}

/// Element-wise less than or equal comparison
pub fn le(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values.iter().map(|v| !v.is_nan() && *v <= value).collect())
}

/// Element-wise equality comparison
pub fn eq_value(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values
        .iter()
        .map(|v| !v.is_nan() && (*v - value).abs() < f64::EPSILON)
        .collect())
}

/// Element-wise not equal comparison
pub fn ne_value(df: &DataFrame, column: &str, value: f64) -> Result<Vec<bool>> {
    let values = df.get_column_numeric_values(column)?;
    Ok(values
        .iter()
        .map(|v| v.is_nan() || (*v - value).abs() >= f64::EPSILON)
        .collect())
}

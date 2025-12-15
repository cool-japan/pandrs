//! Mathematical operation helper functions

use crate::core::error::Result;
use crate::dataframe::DataFrame;
use crate::series::Series;

/// Helper function to copy DataFrame and replace one column
fn copy_df_with_column(
    df: &DataFrame,
    target_column: &str,
    new_values: Vec<f64>,
) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == target_column {
            result.add_column(
                col_name.clone(),
                Series::new(new_values.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Floor values in a column
pub fn floor(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| v.floor()).collect();
    copy_df_with_column(df, column, result_values)
}

/// Ceiling values in a column
pub fn ceil(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| v.ceil()).collect();
    copy_df_with_column(df, column, result_values)
}

/// Truncate values toward zero
pub fn trunc(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| v.trunc()).collect();
    copy_df_with_column(df, column, result_values)
}

/// Get fractional part of values
pub fn fract(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| v.fract()).collect();
    copy_df_with_column(df, column, result_values)
}

/// Apply reciprocal (1/x) to values
pub fn reciprocal(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| 1.0 / v).collect();
    copy_df_with_column(df, column, result_values)
}

/// Compute the absolute value of a numeric column
pub fn abs_column(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let abs_values: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    copy_df_with_column(df, column, abs_values)
}

/// Round values in a column to specified decimal places
pub fn round_column(df: &DataFrame, column: &str, decimals: i32) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let factor = 10.0f64.powi(decimals);
    let rounded: Vec<f64> = values
        .iter()
        .map(|v| (v * factor).round() / factor)
        .collect();
    copy_df_with_column(df, column, rounded)
}

/// Negate values in a column
pub fn neg(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| -v).collect();
    copy_df_with_column(df, column, result_values)
}

/// Compute modulo operation on a column
pub fn mod_column(df: &DataFrame, column: &str, divisor: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| v % divisor).collect();
    copy_df_with_column(df, column, result_values)
}

/// Floor division of a column
pub fn floordiv(df: &DataFrame, column: &str, divisor: f64) -> Result<DataFrame> {
    let values = df.get_column_numeric_values(column)?;
    let result_values: Vec<f64> = values.iter().map(|v| (v / divisor).floor()).collect();
    copy_df_with_column(df, column, result_values)
}

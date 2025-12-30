//! Window operations for DataFrame
//!
//! This module provides pandas-like window operations for DataFrames including rolling windows,
//! expanding windows, and exponentially weighted moving operations.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::{Series, WindowExt, WindowOps};

/// Extension trait to add window operations to DataFrame
pub trait DataFrameWindowExt {
    /// Apply a rolling window operation to a column
    fn rolling(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an expanding window operation to a column
    fn expanding(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an exponentially weighted moving operation to a column
    fn ewm(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;
}

impl DataFrameWindowExt for DataFrame {
    fn rolling(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Get the column as a Series<f64>
        let column = self.get_column_as_f64_legacy(column_name)?;

        // Apply the rolling operation
        let rolling = column.rolling(window_size)?;
        let result_series = match operation.to_lowercase().as_str() {
            "mean" => rolling.mean()?,
            "sum" => rolling.sum()?,
            "std" => rolling.std(1)?,
            "var" => rolling.var(1)?,
            "min" => rolling.min()?,
            "max" => rolling.max()?,
            "median" => rolling.median()?,
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Unsupported rolling operation: {}",
                    operation
                )))
            }
        };

        // Create a new DataFrame with the result
        let mut new_df = self.clone();
        let default_name = format!("{}_{}", column_name, operation);
        let result_column_name = new_column_name.unwrap_or(&default_name);
        new_df.add_column(
            result_column_name.to_string(),
            result_series.to_string_series()?,
        )?;

        Ok(new_df)
    }

    fn expanding(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Get the column as a Series<f64>
        let column = self.get_column_as_f64_legacy(column_name)?;

        // Apply the expanding operation
        let expanding = column.expanding(min_periods)?;
        let result_series = match operation.to_lowercase().as_str() {
            "mean" => expanding.mean()?,
            "sum" => expanding.sum()?,
            "std" => expanding.std(1)?,
            "var" => expanding.var(1)?,
            "min" => expanding.min()?,
            "max" => expanding.max()?,
            "median" => expanding.median()?,
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Unsupported expanding operation: {}",
                    operation
                )))
            }
        };

        // Create a new DataFrame with the result
        let mut new_df = self.clone();
        let default_name = format!("{}_{}", column_name, operation);
        let result_column_name = new_column_name.unwrap_or(&default_name);
        new_df.add_column(
            result_column_name.to_string(),
            result_series.to_string_series()?,
        )?;

        Ok(new_df)
    }

    fn ewm(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Get the column as a Series<f64>
        let column = self.get_column_as_f64_legacy(column_name)?;

        // Create EWM window
        let mut ewm = column.ewm();
        if let Some(span_val) = span {
            ewm = ewm.span(span_val);
        } else if let Some(alpha_val) = alpha {
            ewm = ewm.alpha(alpha_val)?;
        } else {
            return Err(Error::InvalidValue(
                "Must specify either span or alpha for EWM".to_string(),
            ));
        }

        // Apply the EWM operation
        let result_series = match operation.to_lowercase().as_str() {
            "mean" => ewm.mean()?,
            "std" => ewm.std(1)?,
            "var" => ewm.var(1)?,
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Unsupported EWM operation: {}",
                    operation
                )))
            }
        };

        // Create a new DataFrame with the result
        let mut new_df = self.clone();
        let default_name = format!("{}_{}", column_name, operation);
        let result_column_name = new_column_name.unwrap_or(&default_name);
        new_df.add_column(
            result_column_name.to_string(),
            result_series.to_string_series()?,
        )?;

        Ok(new_df)
    }
}

impl DataFrame {
    /// Legacy helper method to get a column as Series<f64>
    fn get_column_as_f64_legacy(&self, column_name: &str) -> Result<Series<f64>> {
        // Try to get the column as a string series first, then parse to f64
        if let Ok(string_series) = self.get_column::<String>(column_name) {
            // Parse string values to f64
            let mut f64_values = Vec::new();
            for value in string_series.values() {
                match value.parse::<f64>() {
                    Ok(val) => f64_values.push(val),
                    Err(_) => {
                        return Err(Error::InvalidValue(format!(
                            "Cannot convert column '{}' to numeric",
                            column_name
                        )))
                    }
                }
            }
            return Series::new(f64_values, Some(column_name.to_string()));
        }

        Err(Error::ColumnNotFound(column_name.to_string()))
    }
}

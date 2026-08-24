//! Window operations for DataFrame
//!
//! This module provides pandas-like window operations for DataFrames including rolling windows,
//! expanding windows, and exponentially weighted moving operations.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::{Series, WindowExt, WindowOps};

/// Extension trait to add window operations to DataFrame
pub trait DataFrameWindowExt {
    /// Apply a rolling window operation to a column.
    ///
    /// Equivalent to [`DataFrameWindowExt::rolling_with_options`] with
    /// `min_periods = None` (defaults to `window_size`), `center = false`,
    /// and `ddof = 1`.
    fn rolling(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply a rolling window operation to a column with explicit
    /// `min_periods`, `center`, and `ddof` (used by `"std"`/`"var"`) --
    /// the string-dispatch equivalent of the options the fluent
    /// [`crate::dataframe::enhanced_window::DataFrameRolling`] builder
    /// already exposes.
    fn rolling_with_options(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        min_periods: Option<usize>,
        center: bool,
        ddof: usize,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an expanding window operation to a column.
    ///
    /// Equivalent to [`DataFrameWindowExt::expanding_with_options`] with
    /// `ddof = 1`.
    fn expanding(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an expanding window operation to a column with an explicit
    /// `ddof` (used by `"std"`/`"var"`).
    fn expanding_with_options(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        ddof: usize,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an exponentially weighted moving operation to a column.
    ///
    /// Equivalent to [`DataFrameWindowExt::ewm_with_options`] with
    /// `ddof = 1`.
    fn ewm(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame>;

    /// Apply an exponentially weighted moving operation to a column with an
    /// explicit `ddof` (used by `"std"`/`"var"`).
    fn ewm_with_options(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        ddof: usize,
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
        self.rolling_with_options(
            window_size,
            column_name,
            operation,
            None,
            false,
            1,
            new_column_name,
        )
    }

    fn rolling_with_options(
        &self,
        window_size: usize,
        column_name: &str,
        operation: &str,
        min_periods: Option<usize>,
        center: bool,
        ddof: usize,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Get the column as a Series<f64>
        let column = self.get_column_as_f64_legacy(column_name)?;

        // Apply the rolling operation
        let rolling = column
            .rolling(window_size)?
            .min_periods(min_periods.unwrap_or(window_size))
            .center(center);
        let result_series = match operation.to_lowercase().as_str() {
            "mean" => rolling.mean()?,
            "sum" => rolling.sum()?,
            "std" => rolling.std(ddof)?,
            "var" => rolling.var(ddof)?,
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
        let col_names: Vec<&str> = self.column_names().iter().map(|s| s.as_str()).collect();
        let mut new_df = self.select_columns(&col_names)?;
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
        self.expanding_with_options(min_periods, column_name, operation, 1, new_column_name)
    }

    fn expanding_with_options(
        &self,
        min_periods: usize,
        column_name: &str,
        operation: &str,
        ddof: usize,
        new_column_name: Option<&str>,
    ) -> Result<DataFrame> {
        // Get the column as a Series<f64>
        let column = self.get_column_as_f64_legacy(column_name)?;

        // Apply the expanding operation
        let expanding = column.expanding(min_periods)?;
        let result_series = match operation.to_lowercase().as_str() {
            "mean" => expanding.mean()?,
            "sum" => expanding.sum()?,
            "std" => expanding.std(ddof)?,
            "var" => expanding.var(ddof)?,
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
        let col_names: Vec<&str> = self.column_names().iter().map(|s| s.as_str()).collect();
        let mut new_df = self.select_columns(&col_names)?;
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
        self.ewm_with_options(column_name, operation, span, alpha, 1, new_column_name)
    }

    fn ewm_with_options(
        &self,
        column_name: &str,
        operation: &str,
        span: Option<usize>,
        alpha: Option<f64>,
        ddof: usize,
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
            "std" => ewm.std(ddof)?,
            "var" => ewm.var(ddof)?,
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Unsupported EWM operation: {}",
                    operation
                )))
            }
        };

        // Create a new DataFrame with the result
        let col_names: Vec<&str> = self.column_names().iter().map(|s| s.as_str()).collect();
        let mut new_df = self.select_columns(&col_names)?;
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
    /// Helper method to get a column as `Series<f64>`, regardless of its
    /// underlying storage type.
    ///
    /// This used to accept *only* `Series<String>` columns, so a column
    /// actually stored as `Series<f64>`/`Series<i64>`/etc. (the normal case
    /// for numeric data) always failed with `ColumnNotFound` even though it
    /// existed -- silently turning every rolling/expanding/EWM call in this
    /// module into a no-op `Ok` for such DataFrames. Delegates to
    /// [`DataFrame::get_column_numeric_values`], which downcasts the
    /// column's actual element type once instead of assuming a single
    /// storage type.
    fn get_column_as_f64_legacy(&self, column_name: &str) -> Result<Series<f64>> {
        let values = self.get_column_numeric_values(column_name)?;
        Series::new(values, Some(column_name.to_string()))
    }
}

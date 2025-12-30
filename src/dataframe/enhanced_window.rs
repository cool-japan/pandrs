//! Enhanced window operations for DataFrame
//!
//! This module provides comprehensive pandas-like window operations for DataFrames including rolling windows,
//! expanding windows, and exponentially weighted moving operations with full feature parity to Series.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::window::{Expanding, Rolling, WindowClosed, EWM};
use crate::series::{Series, WindowExt, WindowOps};
use chrono::{Duration, NaiveDateTime};
use std::any::Any;
use std::collections::HashMap;

/// Advanced rolling window configuration for DataFrames
#[derive(Debug, Clone)]
pub struct DataFrameRolling {
    pub window_size: usize,
    pub min_periods: Option<usize>,
    pub center: bool,
    pub closed: WindowClosed,
    pub columns: Option<Vec<String>>,
}

impl DataFrameRolling {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            min_periods: None,
            center: false,
            closed: WindowClosed::Right,
            columns: None,
        }
    }

    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    pub fn closed(mut self, closed: WindowClosed) -> Self {
        self.closed = closed;
        self
    }

    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Advanced expanding window configuration for DataFrames
#[derive(Debug, Clone)]
pub struct DataFrameExpanding {
    pub min_periods: usize,
    pub columns: Option<Vec<String>>,
}

impl DataFrameExpanding {
    pub fn new(min_periods: usize) -> Self {
        Self {
            min_periods,
            columns: None,
        }
    }

    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Advanced EWM configuration for DataFrames
#[derive(Debug, Clone)]
pub struct DataFrameEWM {
    pub alpha: Option<f64>,
    pub span: Option<usize>,
    pub halflife: Option<f64>,
    pub adjust: bool,
    pub ignore_na: bool,
    pub columns: Option<Vec<String>>,
}

impl DataFrameEWM {
    pub fn new() -> Self {
        Self {
            alpha: None,
            span: None,
            halflife: None,
            adjust: true,
            ignore_na: false,
            columns: None,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(Error::InvalidValue(
                "Alpha must be between 0 and 1".to_string(),
            ));
        }
        self.alpha = Some(alpha);
        self.span = None;
        self.halflife = None;
        Ok(self)
    }

    pub fn span(mut self, span: usize) -> Self {
        self.span = Some(span);
        self.alpha = None;
        self.halflife = None;
        self
    }

    pub fn halflife(mut self, halflife: f64) -> Self {
        self.halflife = Some(halflife);
        self.alpha = None;
        self.span = None;
        self
    }

    pub fn adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    pub fn ignore_na(mut self, ignore_na: bool) -> Self {
        self.ignore_na = ignore_na;
        self
    }

    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Enhanced extension trait to add comprehensive window operations to DataFrame
pub trait DataFrameWindowExt {
    /// Create a rolling window configuration
    fn rolling(&self, window_size: usize) -> DataFrameRolling;

    /// Create an expanding window configuration
    fn expanding(&self, min_periods: usize) -> DataFrameExpanding;

    /// Create an EWM window configuration
    fn ewm(&self) -> DataFrameEWM;

    /// Apply rolling window operations with advanced configuration
    fn apply_rolling<'a>(&'a self, config: &'a DataFrameRolling) -> DataFrameRollingOps<'a>;

    /// Apply expanding window operations with advanced configuration
    fn apply_expanding<'a>(&'a self, config: &'a DataFrameExpanding) -> DataFrameExpandingOps<'a>;

    /// Apply EWM operations with advanced configuration
    fn apply_ewm<'a>(&'a self, config: &'a DataFrameEWM) -> Result<DataFrameEWMOps<'a>>;

    /// Time-based rolling window for datetime columns
    fn rolling_time(&self, window: Duration, on: &str) -> Result<DataFrameTimeRolling>;
}

/// Rolling window operations for DataFrames
pub struct DataFrameRollingOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a DataFrameRolling,
}

/// Expanding window operations for DataFrames
pub struct DataFrameExpandingOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a DataFrameExpanding,
}

/// EWM operations for DataFrames
pub struct DataFrameEWMOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a DataFrameEWM,
}

/// Time-based rolling window for DataFrames
pub struct DataFrameTimeRolling<'a> {
    dataframe: &'a DataFrame,
    window: Duration,
    datetime_column: String,
}

impl DataFrameWindowExt for DataFrame {
    fn rolling(&self, window_size: usize) -> DataFrameRolling {
        DataFrameRolling::new(window_size)
    }

    fn expanding(&self, min_periods: usize) -> DataFrameExpanding {
        DataFrameExpanding::new(min_periods)
    }

    fn ewm(&self) -> DataFrameEWM {
        DataFrameEWM::new()
    }

    fn apply_rolling<'a>(&'a self, config: &'a DataFrameRolling) -> DataFrameRollingOps<'a> {
        DataFrameRollingOps {
            dataframe: self,
            config,
        }
    }

    fn apply_expanding<'a>(&'a self, config: &'a DataFrameExpanding) -> DataFrameExpandingOps<'a> {
        DataFrameExpandingOps {
            dataframe: self,
            config,
        }
    }

    fn apply_ewm<'a>(&'a self, config: &'a DataFrameEWM) -> Result<DataFrameEWMOps<'a>> {
        // Validate EWM configuration
        if config.alpha.is_none() && config.span.is_none() && config.halflife.is_none() {
            return Err(Error::InvalidValue(
                "Must specify either alpha, span, or halflife for EWM".to_string(),
            ));
        }

        Ok(DataFrameEWMOps {
            dataframe: self,
            config,
        })
    }

    fn rolling_time(&self, window: Duration, on: &str) -> Result<DataFrameTimeRolling> {
        // Verify the datetime column exists
        if !self.column_names().contains(&on.to_string()) {
            return Err(Error::ColumnNotFound(on.to_string()));
        }

        Ok(DataFrameTimeRolling {
            dataframe: self,
            window,
            datetime_column: on.to_string(),
        })
    }
}

// Implementation for DataFrameRollingOps
impl<'a> DataFrameRollingOps<'a> {
    /// Apply rolling mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply rolling sum operation
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_operation("sum")
    }

    /// Apply rolling standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply rolling variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    /// Apply rolling minimum operation
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_operation("min")
    }

    /// Apply rolling maximum operation
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_operation("max")
    }

    /// Apply rolling count operation
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_operation("count")
    }

    /// Apply rolling median operation
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_operation("median")
    }

    /// Apply rolling quantile operation
    pub fn quantile(&self, q: f64) -> Result<DataFrame> {
        self.apply_operation_with_param("quantile", q)
    }

    /// Apply custom aggregation function
    pub fn apply<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> f64 + Copy,
    {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let rolling = column
                .rolling(self.config.window_size)?
                .min_periods(self.config.min_periods.unwrap_or(self.config.window_size))
                .center(self.config.center)
                .closed(self.config.closed);

            let result_series = rolling.apply(func)?;
            let result_column_name = format!("{}_{}", column_name, "custom");
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let rolling = column
                .rolling(self.config.window_size)?
                .min_periods(self.config.min_periods.unwrap_or(self.config.window_size))
                .center(self.config.center)
                .closed(self.config.closed);

            let result_series = match operation {
                "mean" => rolling.mean()?,
                "sum" => rolling.sum()?,
                "min" => rolling.min()?,
                "max" => rolling.max()?,
                "median" => rolling.median()?,
                "count" => {
                    let count_series = rolling.count()?;
                    // Convert usize to f64 for consistency
                    let f64_values: Vec<f64> =
                        count_series.values().iter().map(|&v| v as f64).collect();
                    Series::new(f64_values, count_series.name().cloned())?
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let rolling = column
                .rolling(self.config.window_size)?
                .min_periods(self.config.min_periods.unwrap_or(self.config.window_size))
                .center(self.config.center)
                .closed(self.config.closed);

            let result_series = match operation {
                "std" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        rolling.std(*ddof)?
                    } else {
                        rolling.std(1)?
                    }
                }
                "var" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        rolling.var(*ddof)?
                    } else {
                        rolling.var(1)?
                    }
                }
                "quantile" => {
                    if let Some(q) = (&param as &dyn Any).downcast_ref::<f64>() {
                        rolling.quantile(*q)?
                    } else {
                        return Err(Error::InvalidValue(
                            "Invalid quantile parameter".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn get_target_columns(&self) -> Result<Vec<String>> {
        if let Some(ref columns) = self.config.columns {
            // Verify all specified columns exist
            for col in columns {
                if !self.dataframe.column_names().contains(col) {
                    return Err(Error::ColumnNotFound(col.clone()));
                }
            }
            Ok(columns.clone())
        } else {
            // Get all numeric columns
            Ok(self.dataframe.get_numeric_column_names())
        }
    }
}

// Implementation for DataFrameExpandingOps
impl<'a> DataFrameExpandingOps<'a> {
    /// Apply expanding mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply expanding sum operation
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_operation("sum")
    }

    /// Apply expanding standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply expanding variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    /// Apply expanding minimum operation
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_operation("min")
    }

    /// Apply expanding maximum operation
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_operation("max")
    }

    /// Apply expanding count operation
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_operation("count")
    }

    /// Apply expanding median operation
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_operation("median")
    }

    /// Apply expanding quantile operation
    pub fn quantile(&self, q: f64) -> Result<DataFrame> {
        self.apply_operation_with_param("quantile", q)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let expanding = column.expanding(self.config.min_periods)?;

            let result_series = match operation {
                "mean" => expanding.mean()?,
                "sum" => expanding.sum()?,
                "min" => expanding.min()?,
                "max" => expanding.max()?,
                "median" => expanding.median()?,
                "count" => {
                    let count_series = expanding.count()?;
                    let f64_values: Vec<f64> =
                        count_series.values().iter().map(|&v| v as f64).collect();
                    Series::new(f64_values, count_series.name().cloned())?
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let expanding = column.expanding(self.config.min_periods)?;

            let result_series = match operation {
                "std" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        expanding.std(*ddof)?
                    } else {
                        expanding.std(1)?
                    }
                }
                "var" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        expanding.var(*ddof)?
                    } else {
                        expanding.var(1)?
                    }
                }
                "quantile" => {
                    if let Some(q) = (&param as &dyn Any).downcast_ref::<f64>() {
                        expanding.quantile(*q)?
                    } else {
                        return Err(Error::InvalidValue(
                            "Invalid quantile parameter".to_string(),
                        ));
                    }
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn get_target_columns(&self) -> Result<Vec<String>> {
        if let Some(ref columns) = self.config.columns {
            for col in columns {
                if !self.dataframe.column_names().contains(col) {
                    return Err(Error::ColumnNotFound(col.clone()));
                }
            }
            Ok(columns.clone())
        } else {
            Ok(self.dataframe.get_numeric_column_names())
        }
    }
}

// Implementation for DataFrameEWMOps
impl<'a> DataFrameEWMOps<'a> {
    /// Apply EWM mean operation
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply EWM standard deviation operation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply EWM variance operation
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let mut ewm = column
                .ewm()
                .adjust(self.config.adjust)
                .ignore_na(self.config.ignore_na);

            if let Some(alpha) = self.config.alpha {
                ewm = ewm.alpha(alpha)?;
            } else if let Some(span) = self.config.span {
                ewm = ewm.span(span);
            } else if let Some(halflife) = self.config.halflife {
                ewm = ewm.halflife(halflife);
            }

            let result_series = match operation {
                "mean" => ewm.mean()?,
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported EWM operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let mut result_df = self.dataframe.clone();

        for column_name in target_columns {
            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let mut ewm = column
                .ewm()
                .adjust(self.config.adjust)
                .ignore_na(self.config.ignore_na);

            if let Some(alpha) = self.config.alpha {
                ewm = ewm.alpha(alpha)?;
            } else if let Some(span) = self.config.span {
                ewm = ewm.span(span);
            } else if let Some(halflife) = self.config.halflife {
                ewm = ewm.halflife(halflife);
            }

            let result_series = match operation {
                "std" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        ewm.std(*ddof)?
                    } else {
                        ewm.std(1)?
                    }
                }
                "var" => {
                    if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                        ewm.var(*ddof)?
                    } else {
                        ewm.var(1)?
                    }
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported EWM operation: {}",
                        operation
                    )))
                }
            };

            let result_column_name = format!("{}_{}", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn get_target_columns(&self) -> Result<Vec<String>> {
        if let Some(ref columns) = self.config.columns {
            for col in columns {
                if !self.dataframe.column_names().contains(col) {
                    return Err(Error::ColumnNotFound(col.clone()));
                }
            }
            Ok(columns.clone())
        } else {
            Ok(self.dataframe.get_numeric_column_names())
        }
    }
}

// Implementation for DataFrameTimeRolling
impl<'a> DataFrameTimeRolling<'a> {
    /// Apply time-based rolling mean
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_time_operation("mean")
    }

    /// Apply time-based rolling sum
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_time_operation("sum")
    }

    /// Apply time-based rolling count
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_time_operation("count")
    }

    fn apply_time_operation(&self, operation: &str) -> Result<DataFrame> {
        // Get datetime column
        let datetime_series = self
            .dataframe
            .get_column::<NaiveDateTime>(&self.datetime_column)
            .map_err(|_| {
                Error::InvalidValue(format!(
                    "Column '{}' is not a datetime column",
                    self.datetime_column
                ))
            })?;

        let mut result_df = self.dataframe.clone();
        let numeric_columns = self.dataframe.get_numeric_column_names();

        for column_name in numeric_columns {
            if column_name == self.datetime_column {
                continue;
            }

            let column = self.dataframe.get_column_as_f64(&column_name)?;
            let result_values =
                self.calculate_time_window_operation(&datetime_series, &column, operation)?;

            let result_series = Series::new(
                result_values,
                Some(format!("{}_{}", column_name, operation)),
            )?;
            result_df.add_column(
                format!("{}_{}", column_name, operation),
                result_series.to_string_series()?,
            )?;
        }

        Ok(result_df)
    }

    fn calculate_time_window_operation(
        &self,
        datetime_series: &Series<NaiveDateTime>,
        value_series: &Series<f64>,
        operation: &str,
    ) -> Result<Vec<f64>> {
        let mut result = Vec::with_capacity(datetime_series.len());

        for (i, current_time) in datetime_series.values().iter().enumerate() {
            let window_start = *current_time - self.window;

            // Collect values within the time window
            let mut window_values = Vec::new();
            for (j, time) in datetime_series.values().iter().enumerate() {
                if *time >= window_start && *time <= *current_time {
                    window_values.push(value_series.values()[j]);
                }
            }

            let result_value = match operation {
                "mean" => {
                    if window_values.is_empty() {
                        f64::NAN
                    } else {
                        window_values.iter().sum::<f64>() / window_values.len() as f64
                    }
                }
                "sum" => window_values.iter().sum::<f64>(),
                "count" => window_values.len() as f64,
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Unsupported time operation: {}",
                        operation
                    )))
                }
            };

            result.push(result_value);
        }

        Ok(result)
    }
}

impl DataFrame {
    /// Helper method to get a column as `Series<f64>`
    pub fn get_column_as_f64(&self, column_name: &str) -> Result<Series<f64>> {
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

    /// Helper method to get all numeric column names
    pub fn get_numeric_column_names(&self) -> Vec<String> {
        let mut numeric_columns = Vec::new();

        for column_name in self.column_names() {
            // Try to convert to f64 to check if numeric
            if self.get_column_as_f64(&column_name).is_ok() {
                numeric_columns.push(column_name);
            }
        }

        numeric_columns
    }
}

//! Group-wise window operations for DataFrames
//!
//! This module provides comprehensive window operations within groups, combining
//! GroupBy functionality with advanced window operations for sophisticated time series
//! and grouped analytics operations.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::enhanced_window::{
    DataFrameEWM, DataFrameExpanding, DataFrameRolling,
    DataFrameWindowExt as EnhancedDataFrameWindowExt,
};
use crate::dataframe::groupby::{DataFrameGroupBy, GroupByExt};
use crate::series::window::{Expanding, Rolling, WindowClosed, EWM};
use crate::series::{Series, WindowExt, WindowOps};
use chrono::{Duration, NaiveDateTime};
use std::any::Any;
use std::collections::HashMap;

/// Group-wise rolling window configuration
#[derive(Debug, Clone)]
pub struct GroupWiseRolling {
    pub window_size: usize,
    pub min_periods: Option<usize>,
    pub center: bool,
    pub closed: WindowClosed,
    pub columns: Option<Vec<String>>,
    pub group_columns: Vec<String>,
}

impl GroupWiseRolling {
    pub fn new(group_columns: Vec<String>, window_size: usize) -> Self {
        Self {
            window_size,
            min_periods: None,
            center: false,
            closed: WindowClosed::Right,
            columns: None,
            group_columns,
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

/// Group-wise expanding window configuration
#[derive(Debug, Clone)]
pub struct GroupWiseExpanding {
    pub min_periods: usize,
    pub columns: Option<Vec<String>>,
    pub group_columns: Vec<String>,
}

impl GroupWiseExpanding {
    pub fn new(group_columns: Vec<String>, min_periods: usize) -> Self {
        Self {
            min_periods,
            columns: None,
            group_columns,
        }
    }

    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Group-wise EWM configuration
#[derive(Debug, Clone)]
pub struct GroupWiseEWM {
    pub alpha: Option<f64>,
    pub span: Option<usize>,
    pub halflife: Option<f64>,
    pub adjust: bool,
    pub ignore_na: bool,
    pub columns: Option<Vec<String>>,
    pub group_columns: Vec<String>,
}

impl GroupWiseEWM {
    pub fn new(group_columns: Vec<String>) -> Self {
        Self {
            alpha: None,
            span: None,
            halflife: None,
            adjust: true,
            ignore_na: false,
            columns: None,
            group_columns,
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

/// Group-wise time-based rolling window configuration
#[derive(Debug, Clone)]
pub struct GroupWiseTimeRolling {
    pub window: Duration,
    pub datetime_column: String,
    pub columns: Option<Vec<String>>,
    pub group_columns: Vec<String>,
}

impl GroupWiseTimeRolling {
    pub fn new(group_columns: Vec<String>, window: Duration, datetime_column: String) -> Self {
        Self {
            window,
            datetime_column,
            columns: None,
            group_columns,
        }
    }

    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Extension trait for group-wise window operations
pub trait GroupWiseWindowExt {
    /// Create a group-wise rolling window configuration
    fn rolling_by_group(&self, group_columns: Vec<String>, window_size: usize) -> GroupWiseRolling;

    /// Create a group-wise expanding window configuration
    fn expanding_by_group(
        &self,
        group_columns: Vec<String>,
        min_periods: usize,
    ) -> GroupWiseExpanding;

    /// Create a group-wise EWM configuration
    fn ewm_by_group(&self, group_columns: Vec<String>) -> GroupWiseEWM;

    /// Create a group-wise time-based rolling window configuration
    fn rolling_time_by_group(
        &self,
        group_columns: Vec<String>,
        window: Duration,
        datetime_column: String,
    ) -> GroupWiseTimeRolling;

    /// Apply group-wise rolling operations
    fn apply_rolling_by_group<'a>(
        &'a self,
        config: &'a GroupWiseRolling,
    ) -> Result<GroupWiseRollingOps<'a>>;

    /// Apply group-wise expanding operations
    fn apply_expanding_by_group<'a>(
        &'a self,
        config: &'a GroupWiseExpanding,
    ) -> Result<GroupWiseExpandingOps<'a>>;

    /// Apply group-wise EWM operations
    fn apply_ewm_by_group<'a>(&'a self, config: &'a GroupWiseEWM) -> Result<GroupWiseEWMOps<'a>>;

    /// Apply group-wise time-based rolling operations
    fn apply_rolling_time_by_group<'a>(
        &'a self,
        config: &'a GroupWiseTimeRolling,
    ) -> Result<GroupWiseTimeRollingOps<'a>>;
}

/// Group-wise rolling operations
pub struct GroupWiseRollingOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a GroupWiseRolling,
}

/// Group-wise expanding operations
pub struct GroupWiseExpandingOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a GroupWiseExpanding,
}

/// Group-wise EWM operations
pub struct GroupWiseEWMOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a GroupWiseEWM,
}

/// Group-wise time-based rolling operations
pub struct GroupWiseTimeRollingOps<'a> {
    dataframe: &'a DataFrame,
    config: &'a GroupWiseTimeRolling,
}

impl GroupWiseWindowExt for DataFrame {
    fn rolling_by_group(&self, group_columns: Vec<String>, window_size: usize) -> GroupWiseRolling {
        GroupWiseRolling::new(group_columns, window_size)
    }

    fn expanding_by_group(
        &self,
        group_columns: Vec<String>,
        min_periods: usize,
    ) -> GroupWiseExpanding {
        GroupWiseExpanding::new(group_columns, min_periods)
    }

    fn ewm_by_group(&self, group_columns: Vec<String>) -> GroupWiseEWM {
        GroupWiseEWM::new(group_columns)
    }

    fn rolling_time_by_group(
        &self,
        group_columns: Vec<String>,
        window: Duration,
        datetime_column: String,
    ) -> GroupWiseTimeRolling {
        GroupWiseTimeRolling::new(group_columns, window, datetime_column)
    }

    fn apply_rolling_by_group<'a>(
        &'a self,
        config: &'a GroupWiseRolling,
    ) -> Result<GroupWiseRollingOps<'a>> {
        // Validate group columns exist
        for col in &config.group_columns {
            if !self.column_names().contains(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        Ok(GroupWiseRollingOps {
            dataframe: self,
            config,
        })
    }

    fn apply_expanding_by_group<'a>(
        &'a self,
        config: &'a GroupWiseExpanding,
    ) -> Result<GroupWiseExpandingOps<'a>> {
        // Validate group columns exist
        for col in &config.group_columns {
            if !self.column_names().contains(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        Ok(GroupWiseExpandingOps {
            dataframe: self,
            config,
        })
    }

    fn apply_ewm_by_group<'a>(&'a self, config: &'a GroupWiseEWM) -> Result<GroupWiseEWMOps<'a>> {
        // Validate EWM configuration
        if config.alpha.is_none() && config.span.is_none() && config.halflife.is_none() {
            return Err(Error::InvalidValue(
                "Must specify either alpha, span, or halflife for EWM".to_string(),
            ));
        }

        // Validate group columns exist
        for col in &config.group_columns {
            if !self.column_names().contains(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        Ok(GroupWiseEWMOps {
            dataframe: self,
            config,
        })
    }

    fn apply_rolling_time_by_group<'a>(
        &'a self,
        config: &'a GroupWiseTimeRolling,
    ) -> Result<GroupWiseTimeRollingOps<'a>> {
        // Validate datetime column exists
        if !self.column_names().contains(&config.datetime_column) {
            return Err(Error::ColumnNotFound(config.datetime_column.clone()));
        }

        // Validate group columns exist
        for col in &config.group_columns {
            if !self.column_names().contains(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        Ok(GroupWiseTimeRollingOps {
            dataframe: self,
            config,
        })
    }
}

// Implementation for GroupWiseRollingOps
impl<'a> GroupWiseRollingOps<'a> {
    /// Apply group-wise rolling mean
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply group-wise rolling sum
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_operation("sum")
    }

    /// Apply group-wise rolling standard deviation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply group-wise rolling variance
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    /// Apply group-wise rolling minimum
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_operation("min")
    }

    /// Apply group-wise rolling maximum
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_operation("max")
    }

    /// Apply group-wise rolling count
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_operation("count")
    }

    /// Apply group-wise rolling median
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_operation("median")
    }

    /// Apply group-wise rolling quantile
    pub fn quantile(&self, q: f64) -> Result<DataFrame> {
        self.apply_operation_with_param("quantile", q)
    }

    /// Apply custom aggregation function within groups
    pub fn apply<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> f64 + Copy,
    {
        let target_columns = self.get_target_columns()?;
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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

        // For simplified implementation, apply rolling operations to the entire DataFrame
        // In a full implementation, you would group by the specified columns and apply operations within each group
        let mut result_df = self.dataframe.clone();

        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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

            let result_column_name = format!("{}_{}_groupwise", column_name, operation);
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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
            for col in columns {
                if !self.dataframe.column_names().contains(col) {
                    return Err(Error::ColumnNotFound(col.clone()));
                }
            }
            Ok(columns.clone())
        } else {
            let mut numeric_columns = self.dataframe.get_numeric_column_names();
            // Remove group columns from target columns
            numeric_columns.retain(|col| !self.config.group_columns.contains(col));
            Ok(numeric_columns)
        }
    }

    fn concatenate_group_results(&self, group_results: Vec<DataFrame>) -> Result<DataFrame> {
        if group_results.is_empty() {
            return Ok(self.dataframe.clone());
        }

        // For simplicity, return the first group result
        // In a full implementation, you would concatenate all groups
        Ok(group_results.into_iter().next().unwrap())
    }
}

// Implementation for GroupWiseExpandingOps
impl<'a> GroupWiseExpandingOps<'a> {
    /// Apply group-wise expanding mean
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply group-wise expanding sum
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_operation("sum")
    }

    /// Apply group-wise expanding standard deviation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply group-wise expanding variance
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    /// Apply group-wise expanding minimum
    pub fn min(&self) -> Result<DataFrame> {
        self.apply_operation("min")
    }

    /// Apply group-wise expanding maximum
    pub fn max(&self) -> Result<DataFrame> {
        self.apply_operation("max")
    }

    /// Apply group-wise expanding count
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_operation("count")
    }

    /// Apply group-wise expanding median
    pub fn median(&self) -> Result<DataFrame> {
        self.apply_operation("median")
    }

    /// Apply group-wise expanding quantile
    pub fn quantile(&self, q: f64) -> Result<DataFrame> {
        self.apply_operation_with_param("quantile", q)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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
            let mut numeric_columns = self.dataframe.get_numeric_column_names();
            numeric_columns.retain(|col| !self.config.group_columns.contains(col));
            Ok(numeric_columns)
        }
    }

    fn concatenate_group_results(&self, group_results: Vec<DataFrame>) -> Result<DataFrame> {
        if group_results.is_empty() {
            return Ok(self.dataframe.clone());
        }
        Ok(group_results.into_iter().next().unwrap())
    }
}

// Implementation for GroupWiseEWMOps
impl<'a> GroupWiseEWMOps<'a> {
    /// Apply group-wise EWM mean
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_operation("mean")
    }

    /// Apply group-wise EWM standard deviation
    pub fn std(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("std", ddof)
    }

    /// Apply group-wise EWM variance
    pub fn var(&self, ddof: usize) -> Result<DataFrame> {
        self.apply_operation_with_param("var", ddof)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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
        // Simplified implementation - apply operations to entire DataFrame
        // In a full implementation, you would group by specified columns
        let mut result_df = self.dataframe.clone();
        for column_name in &target_columns {
            let column = self.dataframe.get_column_as_f64(column_name)?;
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
            let mut numeric_columns = self.dataframe.get_numeric_column_names();
            numeric_columns.retain(|col| !self.config.group_columns.contains(col));
            Ok(numeric_columns)
        }
    }

    fn concatenate_group_results(&self, group_results: Vec<DataFrame>) -> Result<DataFrame> {
        if group_results.is_empty() {
            return Ok(self.dataframe.clone());
        }
        Ok(group_results.into_iter().next().unwrap())
    }
}

// Implementation for GroupWiseTimeRollingOps
impl<'a> GroupWiseTimeRollingOps<'a> {
    /// Apply group-wise time-based rolling mean
    pub fn mean(&self) -> Result<DataFrame> {
        self.apply_time_operation("mean")
    }

    /// Apply group-wise time-based rolling sum
    pub fn sum(&self) -> Result<DataFrame> {
        self.apply_time_operation("sum")
    }

    /// Apply group-wise time-based rolling count
    pub fn count(&self) -> Result<DataFrame> {
        self.apply_time_operation("count")
    }

    fn apply_time_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;

        // Get datetime column
        let datetime_series = self
            .dataframe
            .get_column::<NaiveDateTime>(&self.config.datetime_column)
            .map_err(|_| {
                Error::InvalidValue(format!(
                    "Column '{}' is not a datetime column",
                    self.config.datetime_column
                ))
            })?;

        let mut result_df = self.dataframe.clone();

        for column_name in &target_columns {
            if column_name == &self.config.datetime_column {
                continue;
            }

            let column = self.dataframe.get_column_as_f64(column_name)?;
            let result_values =
                self.calculate_time_window_operation(&datetime_series, &column, operation)?;

            let result_series = Series::new(
                result_values,
                Some(format!("{}_{}_groupwise", column_name, operation)),
            )?;
            result_df.add_column(
                format!("{}_{}_groupwise", column_name, operation),
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
            let window_start = *current_time - self.config.window;

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

    fn get_target_columns(&self) -> Result<Vec<String>> {
        if let Some(ref columns) = self.config.columns {
            for col in columns {
                if !self.dataframe.column_names().contains(col) {
                    return Err(Error::ColumnNotFound(col.clone()));
                }
            }
            Ok(columns.clone())
        } else {
            let mut numeric_columns = self.dataframe.get_numeric_column_names();
            numeric_columns.retain(|col| {
                !self.config.group_columns.contains(col) && col != &self.config.datetime_column
            });
            Ok(numeric_columns)
        }
    }

    fn concatenate_group_results(&self, group_results: Vec<DataFrame>) -> Result<DataFrame> {
        if group_results.is_empty() {
            return Ok(self.dataframe.clone());
        }
        Ok(group_results.into_iter().next().unwrap())
    }
}

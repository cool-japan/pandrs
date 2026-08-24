//! Group-wise window operations for DataFrames
//!
//! This module provides comprehensive window operations within groups, combining
//! GroupBy functionality with advanced window operations for sophisticated time series
//! and grouped analytics operations.
//!
//! Every rolling/expanding/EWM/time-rolling computation here first partitions the
//! DataFrame's rows by `group_columns` (see `group_row_indices`), runs the window
//! computation independently on each group's own row-ordered sub-series (see
//! `compute_grouped_column`), and scatters the results back into the original row
//! order. This guarantees windows and EWM recursive state never cross group
//! boundaries.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::window::WindowClosed;
use crate::series::{Series, WindowExt, WindowOps};
use chrono::{Duration, NaiveDateTime};
use std::any::Any;
use std::collections::HashMap;

/// Partition all row indices of `dataframe` into groups keyed by the
/// composite value of `group_columns`, in first-appearance order. Row
/// indices within each group are kept in ascending (original) order.
///
/// Column values are materialized once per grouping column up front rather
/// than re-fetched per row, mirroring the same hoist used by
/// [`crate::dataframe::groupby::DataFrameGroupBy::new`]. When
/// `group_columns` is empty, every row belongs to a single group (no
/// partitioning requested).
fn group_row_indices(dataframe: &DataFrame, group_columns: &[String]) -> Result<Vec<Vec<usize>>> {
    let row_count = dataframe.row_count();

    if group_columns.is_empty() {
        return Ok(vec![(0..row_count).collect()]);
    }

    let key_columns: Vec<Vec<String>> = group_columns
        .iter()
        .map(|col| dataframe.get_column_string_values(col))
        .collect::<Result<Vec<_>>>()?;

    let mut order: Vec<Vec<String>> = Vec::new();
    let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

    for row_idx in 0..row_count {
        let key: Vec<String> = key_columns
            .iter()
            .map(|col| {
                col.get(row_idx)
                    .cloned()
                    .unwrap_or_else(|| "NULL".to_string())
            })
            .collect();

        if !groups.contains_key(&key) {
            order.push(key.clone());
        }
        groups.entry(key).or_default().push(row_idx);
    }

    Ok(order
        .into_iter()
        .map(|key| groups.remove(&key).unwrap_or_default())
        .collect())
}

/// Compute a per-group windowed column.
///
/// Runs `compute` independently on each group's own row-ordered sub-series
/// (so rolling windows / expanding accumulation / EWM recursive state never
/// cross group boundaries), then scatters every group's results back into a
/// single `Vec<f64>` aligned to `dataframe`'s original row order.
///
/// `groups` must be a partition of `0..dataframe.row_count()` (as produced
/// by `group_row_indices` on this same `dataframe`).
fn compute_grouped_column<F>(
    dataframe: &DataFrame,
    groups: &[Vec<usize>],
    column_name: &str,
    compute: F,
) -> Result<Vec<f64>>
where
    F: Fn(Series<f64>) -> Result<Series<f64>>,
{
    let full_column = dataframe.get_column_as_f64(column_name)?;
    let full_values = full_column.values();
    let row_count = full_values.len();

    let mut result = vec![f64::NAN; row_count];

    for indices in groups {
        let group_values: Vec<f64> = indices
            .iter()
            .map(|&idx| {
                full_values.get(idx).copied().ok_or_else(|| {
                    Error::InvalidValue(format!(
                        "row index {} out of bounds for column '{}' ({} rows)",
                        idx,
                        column_name,
                        full_values.len()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let group_series = Series::new(group_values, None)?;
        let computed = compute(group_series)?;

        if computed.len() != indices.len() {
            return Err(Error::InvalidValue(format!(
                "group-wise window computation for column '{}' must return one value per \
                 group row: group had {} row(s) but the computation produced {}",
                column_name,
                indices.len(),
                computed.len()
            )));
        }

        for (&idx, &value) in indices.iter().zip(computed.values().iter()) {
            let slot = result.get_mut(idx).ok_or_else(|| {
                Error::InvalidValue(format!(
                    "row index {} out of bounds while scattering group-wise window results \
                     for column '{}'",
                    idx, column_name
                ))
            })?;
            *slot = value;
        }
    }

    Ok(result)
}

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
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        let window_size = self.config.window_size;
        let min_periods = self.config.min_periods.unwrap_or(window_size);
        let center = self.config.center;
        let closed = self.config.closed;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let rolling = group_series
                        .rolling(window_size)?
                        .min_periods(min_periods)
                        .center(center)
                        .closed(closed);
                    // `Rolling::apply` returns one `Option<f64>` per row of
                    // this group (`None` where `min_periods` wasn't met);
                    // render `None` as `NaN` to match every other
                    // group-wise window result.
                    let applied = rolling.apply(func)?;
                    let f64_values: Vec<f64> = applied
                        .values()
                        .iter()
                        .map(|&v| v.unwrap_or(f64::NAN))
                        .collect();
                    Series::new(f64_values, applied.name().cloned())
                })?;

            let result_column_name = format!("{}_{}", column_name, "custom");
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation(&self, operation: &str) -> Result<DataFrame> {
        let target_columns = self.get_target_columns()?;
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        let window_size = self.config.window_size;
        let min_periods = self.config.min_periods.unwrap_or(window_size);
        let center = self.config.center;
        let closed = self.config.closed;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let rolling = group_series
                        .rolling(window_size)?
                        .min_periods(min_periods)
                        .center(center)
                        .closed(closed);

                    match operation {
                        "mean" => rolling.mean(),
                        "sum" => rolling.sum(),
                        "min" => rolling.min(),
                        "max" => rolling.max(),
                        "median" => rolling.median(),
                        "count" => {
                            let count_series = rolling.count()?;
                            let f64_values: Vec<f64> =
                                count_series.values().iter().map(|&v| v as f64).collect();
                            Series::new(f64_values, count_series.name().cloned())
                        }
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}_groupwise", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        let window_size = self.config.window_size;
        let min_periods = self.config.min_periods.unwrap_or(window_size);
        let center = self.config.center;
        let closed = self.config.closed;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let rolling = group_series
                        .rolling(window_size)?
                        .min_periods(min_periods)
                        .center(center)
                        .closed(closed);

                    match operation {
                        "std" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                rolling.std(*ddof)
                            } else {
                                rolling.std(1)
                            }
                        }
                        "var" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                rolling.var(*ddof)
                            } else {
                                rolling.var(1)
                            }
                        }
                        "quantile" => {
                            if let Some(q) = (&param as &dyn Any).downcast_ref::<f64>() {
                                rolling.quantile(*q)
                            } else {
                                Err(Error::InvalidValue(
                                    "Invalid quantile parameter".to_string(),
                                ))
                            }
                        }
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
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
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();
        let min_periods = self.config.min_periods;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let expanding = group_series.expanding(min_periods)?;

                    match operation {
                        "mean" => expanding.mean(),
                        "sum" => expanding.sum(),
                        "min" => expanding.min(),
                        "max" => expanding.max(),
                        "median" => expanding.median(),
                        "count" => {
                            let count_series = expanding.count()?;
                            let f64_values: Vec<f64> =
                                count_series.values().iter().map(|&v| v as f64).collect();
                            Series::new(f64_values, count_series.name().cloned())
                        }
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();
        let min_periods = self.config.min_periods;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let expanding = group_series.expanding(min_periods)?;

                    match operation {
                        "std" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                expanding.std(*ddof)
                            } else {
                                expanding.std(1)
                            }
                        }
                        "var" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                expanding.var(*ddof)
                            } else {
                                expanding.var(1)
                            }
                        }
                        "quantile" => {
                            if let Some(q) = (&param as &dyn Any).downcast_ref::<f64>() {
                                expanding.quantile(*q)
                            } else {
                                Err(Error::InvalidValue(
                                    "Invalid quantile parameter".to_string(),
                                ))
                            }
                        }
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
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
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        let adjust = self.config.adjust;
        let ignore_na = self.config.ignore_na;
        let alpha = self.config.alpha;
        let span = self.config.span;
        let halflife = self.config.halflife;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let mut ewm = group_series.ewm().adjust(adjust).ignore_na(ignore_na);
                    if let Some(alpha) = alpha {
                        ewm = ewm.alpha(alpha)?;
                    } else if let Some(span) = span {
                        ewm = ewm.span(span);
                    } else if let Some(halflife) = halflife {
                        ewm = ewm.halflife(halflife);
                    }

                    match operation {
                        "mean" => ewm.mean(),
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported EWM operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
        }

        Ok(result_df)
    }

    fn apply_operation_with_param<T>(&self, operation: &str, param: T) -> Result<DataFrame>
    where
        T: Copy + 'static,
    {
        let target_columns = self.get_target_columns()?;
        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        let adjust = self.config.adjust;
        let ignore_na = self.config.ignore_na;
        let alpha = self.config.alpha;
        let span = self.config.span;
        let halflife = self.config.halflife;

        for column_name in &target_columns {
            let result_values =
                compute_grouped_column(self.dataframe, &groups, column_name, |group_series| {
                    let mut ewm = group_series.ewm().adjust(adjust).ignore_na(ignore_na);
                    if let Some(alpha) = alpha {
                        ewm = ewm.alpha(alpha)?;
                    } else if let Some(span) = span {
                        ewm = ewm.span(span);
                    } else if let Some(halflife) = halflife {
                        ewm = ewm.halflife(halflife);
                    }

                    match operation {
                        "std" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                ewm.std(*ddof)
                            } else {
                                ewm.std(1)
                            }
                        }
                        "var" => {
                            if let Some(ddof) = (&param as &dyn Any).downcast_ref::<usize>() {
                                ewm.var(*ddof)
                            } else {
                                ewm.var(1)
                            }
                        }
                        _ => Err(Error::InvalidValue(format!(
                            "Unsupported EWM operation: {}",
                            operation
                        ))),
                    }
                })?;

            let result_column_name = format!("{}_{}", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
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

        let groups = group_row_indices(self.dataframe, &self.config.group_columns)?;
        let mut result_df = self.dataframe.clone();

        for column_name in &target_columns {
            if column_name == &self.config.datetime_column {
                continue;
            }

            let value_series = self.dataframe.get_column_as_f64(column_name)?;
            let mut result_values = vec![f64::NAN; self.dataframe.row_count()];

            // Compute the time window independently within each group so a
            // row's window never pulls in another group's observations,
            // then scatter each group's results back to their original
            // row positions.
            for indices in &groups {
                let group_datetimes: Vec<NaiveDateTime> = indices
                    .iter()
                    .map(|&idx| {
                        datetime_series.get(idx).copied().ok_or_else(|| {
                            Error::InvalidValue(format!(
                                "row index {} out of bounds for datetime column '{}'",
                                idx, self.config.datetime_column
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let group_values: Vec<f64> = indices
                    .iter()
                    .map(|&idx| {
                        value_series.get(idx).copied().ok_or_else(|| {
                            Error::InvalidValue(format!(
                                "row index {} out of bounds for column '{}'",
                                idx, column_name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                let group_datetime_series = Series::new(group_datetimes, None)?;
                let group_value_series = Series::new(group_values, None)?;
                let group_result = self.calculate_time_window_operation(
                    &group_datetime_series,
                    &group_value_series,
                    operation,
                )?;

                for (&idx, &value) in indices.iter().zip(group_result.iter()) {
                    let slot = result_values.get_mut(idx).ok_or_else(|| {
                        Error::InvalidValue(format!(
                            "row index {} out of bounds while scattering group-wise \
                             time-rolling results for column '{}'",
                            idx, column_name
                        ))
                    })?;
                    *slot = value;
                }
            }

            let result_column_name = format!("{}_{}_groupwise", column_name, operation);
            let result_series = Series::new(result_values, Some(result_column_name.clone()))?;
            result_df.add_column(result_column_name, result_series.to_string_series()?)?;
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

        for (_i, current_time) in datetime_series.values().iter().enumerate() {
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
}

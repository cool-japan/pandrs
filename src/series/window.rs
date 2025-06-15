//! Window operations for Series
//!
//! This module provides pandas-like window operations including rolling windows,
//! expanding windows, and exponentially weighted moving operations.

use crate::core::error::{Error, Result};
use crate::series::base::Series;
use std::fmt::Debug;

/// Rolling window configuration and operations
#[derive(Debug, Clone)]
pub struct Rolling<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    window_size: usize,
    min_periods: Option<usize>,
    center: bool,
    closed: WindowClosed,
}

/// Expanding window configuration and operations
#[derive(Debug, Clone)]
pub struct Expanding<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    min_periods: usize,
}

/// Exponentially weighted moving window configuration and operations
#[derive(Debug, Clone)]
pub struct EWM<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    alpha: Option<f64>,
    span: Option<usize>,
    halflife: Option<f64>,
    adjust: bool,
    ignore_na: bool,
}

/// How to handle window boundaries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowClosed {
    /// Window includes both endpoints
    Both,
    /// Window includes left endpoint only
    Left,
    /// Window includes right endpoint only
    Right,
    /// Window includes neither endpoint
    Neither,
}

impl Default for WindowClosed {
    fn default() -> Self {
        WindowClosed::Right
    }
}

/// Trait for window aggregation operations
pub trait WindowOps<T>
where
    T: Debug + Clone,
{
    /// Calculate the mean of the window
    fn mean(&self) -> Result<Series<f64>>;

    /// Calculate the sum of the window
    fn sum(&self) -> Result<Series<f64>>;

    /// Calculate the standard deviation of the window
    fn std(&self, ddof: usize) -> Result<Series<f64>>;

    /// Calculate the variance of the window
    fn var(&self, ddof: usize) -> Result<Series<f64>>;

    /// Calculate the minimum value in the window
    fn min(&self) -> Result<Series<f64>>;

    /// Calculate the maximum value in the window
    fn max(&self) -> Result<Series<f64>>;

    /// Count non-null values in the window
    fn count(&self) -> Result<Series<usize>>;

    /// Calculate the median of the window
    fn median(&self) -> Result<Series<f64>>;

    /// Calculate a quantile of the window
    fn quantile(&self, q: f64) -> Result<Series<f64>>;

    /// Apply a custom aggregation function
    fn apply<F, R>(&self, func: F) -> Result<Series<R>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone;
}

// Implementation for Rolling windows
impl<T> Rolling<T>
where
    T: Debug + Clone,
{
    /// Create a new rolling window
    pub fn new(series: Series<T>, window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(Error::InvalidValue(
                "Window size must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            series,
            window_size,
            min_periods: None,
            center: false,
            closed: WindowClosed::default(),
        })
    }

    /// Set minimum number of observations required to have a value
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    /// Set whether to center the window around the current observation
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set how to handle window boundaries
    pub fn closed(mut self, closed: WindowClosed) -> Self {
        self.closed = closed;
        self
    }

    /// Get the effective minimum periods
    fn effective_min_periods(&self) -> usize {
        self.min_periods.unwrap_or(self.window_size)
    }

    /// Convert values to f64 for calculations
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        let mut result = Vec::with_capacity(self.series.len());
        for value in self.series.values() {
            result.push(Some((*value).into()));
        }
        Ok(result)
    }

    /// Apply window operation with generic aggregation function
    fn apply_window_op<F, R>(&self, mut func: F) -> Result<Series<Option<R>>>
    where
        T: Into<f64> + Copy,
        F: FnMut(&[f64]) -> R,
        R: Debug + Clone,
    {
        let values = self.values_as_f64()?;
        let mut result = Vec::with_capacity(values.len());
        let min_periods = self.effective_min_periods();

        for i in 0..values.len() {
            let (start, end) = if self.center {
                // Center the window around current position
                let half_window = self.window_size / 2;
                let start = if i >= half_window { i - half_window } else { 0 };
                let end = std::cmp::min(start + self.window_size, values.len());
                (start, end)
            } else {
                // Standard rolling window (looking backwards)
                let start = if i + 1 >= self.window_size {
                    i + 1 - self.window_size
                } else {
                    0
                };
                let end = i + 1;
                (start, end)
            };

            // Extract non-null values in the window
            let window_values: Vec<f64> = values[start..end].iter().filter_map(|&v| v).collect();

            if window_values.len() >= min_periods {
                let agg_result = func(&window_values);
                result.push(Some(agg_result));
            } else {
                result.push(None);
            }
        }

        Series::new(result, self.series.name().cloned())
    }
}

impl<T> WindowOps<T> for Rolling<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    fn mean(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().sum::<f64>() / values.len() as f64)?;

        // Convert Option<f64> to f64 (with NaN for None)
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn sum(&self) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| values.iter().sum::<f64>())?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64;
                variance.sqrt()
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn min(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().fold(f64::INFINITY, |a, &b| a.min(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn max(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn count(&self) -> Result<Series<usize>> {
        let result = self.apply_window_op(|values| values.len())?;
        let values: Vec<usize> = result.values().iter().map(|&v| v.unwrap_or(0)).collect();
        Series::new(values, result.name().cloned())
    }

    fn median(&self) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn quantile(&self, q: f64) -> Result<Series<f64>> {
        if q < 0.0 || q > 1.0 {
            return Err(Error::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        let result = self.apply_window_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (q * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn apply<F, R>(&self, func: F) -> Result<Series<R>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone,
    {
        let result = self.apply_window_op(func)?;
        let values: Vec<R> = result
            .values()
            .iter()
            .filter_map(|v| v.as_ref().cloned())
            .collect();
        Series::new(values, result.name().cloned())
    }
}

// Implementation for Expanding windows
impl<T> Expanding<T>
where
    T: Debug + Clone,
{
    /// Create a new expanding window
    pub fn new(series: Series<T>, min_periods: usize) -> Result<Self> {
        Ok(Self {
            series,
            min_periods,
        })
    }

    /// Convert values to f64 for calculations
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        let mut result = Vec::with_capacity(self.series.len());
        for value in self.series.values() {
            result.push(Some((*value).into()));
        }
        Ok(result)
    }

    /// Apply expanding operation with generic aggregation function
    fn apply_expanding_op<F, R>(&self, mut func: F) -> Result<Series<Option<R>>>
    where
        T: Into<f64> + Copy,
        F: FnMut(&[f64]) -> R,
        R: Debug + Clone,
    {
        let values = self.values_as_f64()?;
        let mut result = Vec::with_capacity(values.len());

        for i in 0..values.len() {
            // Get all values from start to current position
            let window_values: Vec<f64> = values[0..=i].iter().filter_map(|&v| v).collect();

            if window_values.len() >= self.min_periods {
                let agg_result = func(&window_values);
                result.push(Some(agg_result));
            } else {
                result.push(None);
            }
        }

        Series::new(result, self.series.name().cloned())
    }
}

impl<T> WindowOps<T> for Expanding<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    fn mean(&self) -> Result<Series<f64>> {
        let result =
            self.apply_expanding_op(|values| values.iter().sum::<f64>() / values.len() as f64)?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn sum(&self) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| values.iter().sum::<f64>())?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64;
                variance.sqrt()
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn min(&self) -> Result<Series<f64>> {
        let result =
            self.apply_expanding_op(|values| values.iter().fold(f64::INFINITY, |a, &b| a.min(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn max(&self) -> Result<Series<f64>> {
        let result = self
            .apply_expanding_op(|values| values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn count(&self) -> Result<Series<usize>> {
        let result = self.apply_expanding_op(|values| values.len())?;
        let values: Vec<usize> = result.values().iter().map(|&v| v.unwrap_or(0)).collect();
        Series::new(values, result.name().cloned())
    }

    fn median(&self) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn quantile(&self, q: f64) -> Result<Series<f64>> {
        if q < 0.0 || q > 1.0 {
            return Err(Error::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        let result = self.apply_expanding_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (q * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn apply<F, R>(&self, func: F) -> Result<Series<R>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone,
    {
        let result = self.apply_expanding_op(func)?;
        let values: Vec<R> = result
            .values()
            .iter()
            .filter_map(|v| v.as_ref().cloned())
            .collect();
        Series::new(values, result.name().cloned())
    }
}

// Implementation for EWM windows
impl<T> EWM<T>
where
    T: Debug + Clone,
{
    /// Create a new exponentially weighted moving window
    pub fn new(series: Series<T>) -> Self {
        Self {
            series,
            alpha: None,
            span: None,
            halflife: None,
            adjust: true,
            ignore_na: false,
        }
    }

    /// Set the smoothing factor alpha directly
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

    /// Set the span (window size)
    pub fn span(mut self, span: usize) -> Self {
        self.span = Some(span);
        self.alpha = None;
        self.halflife = None;
        self
    }

    /// Set the halflife
    pub fn halflife(mut self, halflife: f64) -> Self {
        self.halflife = Some(halflife);
        self.alpha = None;
        self.span = None;
        self
    }

    /// Set whether to use adjustment
    pub fn adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    /// Set whether to ignore NA values
    pub fn ignore_na(mut self, ignore_na: bool) -> Self {
        self.ignore_na = ignore_na;
        self
    }

    /// Calculate the effective alpha value
    fn get_alpha(&self) -> Result<f64> {
        if let Some(alpha) = self.alpha {
            Ok(alpha)
        } else if let Some(span) = self.span {
            Ok(2.0 / (span as f64 + 1.0))
        } else if let Some(halflife) = self.halflife {
            Ok(1.0 - (-std::f64::consts::LN_2 / halflife).exp())
        } else {
            Err(Error::InvalidValue(
                "Must specify either alpha, span, or halflife".to_string(),
            ))
        }
    }

    /// Convert values to f64 for calculations
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        let mut result = Vec::with_capacity(self.series.len());
        for value in self.series.values() {
            result.push(Some((*value).into()));
        }
        Ok(result)
    }
}

impl<T> EWM<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    /// Calculate exponentially weighted moving average
    pub fn mean(&self) -> Result<Series<f64>> {
        let alpha = self.get_alpha()?;
        let values = self.values_as_f64()?;
        let mut result = Vec::with_capacity(values.len());

        if values.is_empty() {
            return Series::new(result, self.series.name().cloned());
        }

        // Find first non-null value
        let mut ewm_val = None;
        for (i, &val) in values.iter().enumerate() {
            if let Some(v) = val {
                if ewm_val.is_none() {
                    ewm_val = Some(v);
                    result.extend(std::iter::repeat(f64::NAN).take(i));
                    result.push(v);
                } else {
                    let prev = ewm_val.unwrap();
                    ewm_val = Some(alpha * v + (1.0 - alpha) * prev);
                    result.push(ewm_val.unwrap());
                }
            } else if ewm_val.is_some() {
                result.push(ewm_val.unwrap());
            } else {
                result.push(f64::NAN);
            }
        }

        Series::new(result, self.series.name().cloned())
    }

    /// Calculate exponentially weighted moving standard deviation
    pub fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let alpha = self.get_alpha()?;
        let values = self.values_as_f64()?;
        let mut result = Vec::with_capacity(values.len());

        if values.is_empty() {
            return Series::new(result, self.series.name().cloned());
        }

        let mut ewm_mean = None;
        let mut ewm_var = None;

        for (i, &val) in values.iter().enumerate() {
            if let Some(v) = val {
                if ewm_mean.is_none() {
                    ewm_mean = Some(v);
                    ewm_var = Some(0.0);
                    result.extend(std::iter::repeat(f64::NAN).take(i + 1));
                } else {
                    let prev_mean = ewm_mean.unwrap();
                    let prev_var = ewm_var.unwrap();

                    // Update mean
                    ewm_mean = Some(alpha * v + (1.0 - alpha) * prev_mean);

                    // Update variance using recursive formula
                    let diff = v - prev_mean;
                    ewm_var = Some((1.0 - alpha) * (prev_var + alpha * diff * diff));

                    result.push(ewm_var.unwrap().sqrt());
                }
            } else if ewm_var.is_some() {
                result.push(ewm_var.unwrap().sqrt());
            } else {
                result.push(f64::NAN);
            }
        }

        Series::new(result, self.series.name().cloned())
    }

    /// Calculate exponentially weighted moving variance
    pub fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let std_series = self.std(ddof)?;
        let var_values: Vec<f64> = std_series
            .values()
            .iter()
            .map(|&v| if v.is_nan() { f64::NAN } else { v * v })
            .collect();
        Series::new(var_values, std_series.name().cloned())
    }
}

/// Extension trait to add window operations to Series
pub trait WindowExt<T>
where
    T: Debug + Clone,
{
    /// Create a rolling window
    fn rolling(&self, window_size: usize) -> Result<Rolling<T>>;

    /// Create an expanding window
    fn expanding(&self, min_periods: usize) -> Result<Expanding<T>>;

    /// Create an exponentially weighted moving window
    fn ewm(&self) -> EWM<T>;
}

impl<T> WindowExt<T> for Series<T>
where
    T: Debug + Clone,
{
    fn rolling(&self, window_size: usize) -> Result<Rolling<T>> {
        Rolling::new(self.clone(), window_size)
    }

    fn expanding(&self, min_periods: usize) -> Result<Expanding<T>> {
        Expanding::new(self.clone(), min_periods)
    }

    fn ewm(&self) -> EWM<T> {
        EWM::new(self.clone())
    }
}

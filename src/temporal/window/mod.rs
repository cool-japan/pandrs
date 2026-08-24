//! Module for windowing operations on time series data
//!
//! This module provides functionality for time series windowing operations,
//! including fixed (rolling) windows, expanding windows, and exponentially
//! weighted windows.

use std::fmt;

use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::temporal::core::Temporal;
use crate::temporal::core::TimeSeries;

/// Enum that defines the type of window
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Fixed Window (Rolling Window)
    /// Operation that slides a window of fixed size
    Fixed,

    /// Expanding Window
    /// Window that includes all points from the first point to the current point
    Expanding,

    /// Exponentially Weighted Window
    /// Window that gives higher weights to more recent data
    ExponentiallyWeighted,
}

/// Enum that defines the aggregation operation for a window
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowOperation {
    /// Mean (average) calculation
    Mean,

    /// Sum calculation
    Sum,

    /// Standard deviation calculation
    Std,

    /// Minimum value calculation
    Min,

    /// Maximum value calculation
    Max,

    /// Count of non-NA values
    Count,

    /// Median value calculation
    Median,

    /// Variance calculation
    Var,
}

impl fmt::Display for WindowType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            WindowType::Fixed => write!(f, "Fixed"),
            WindowType::Expanding => write!(f, "Expanding"),
            WindowType::ExponentiallyWeighted => write!(f, "ExponentiallyWeighted"),
        }
    }
}

/// Structure for window operations
#[derive(Debug)]
pub struct Window<'a, T: Temporal> {
    /// Reference to the original time series data
    time_series: &'a TimeSeries<T>,

    /// Type of window
    window_type: WindowType,

    /// Size of the window
    window_size: usize,

    /// Decay factor for exponential weighting (alpha)
    /// 0.0 < alpha <= 1.0, larger values give higher weights to more recent data
    alpha: Option<f64>,

    /// Whether the exponentially weighted statistics use the *adjusted*
    /// weighted average over all prior observations (pandas' `adjust=True`
    /// default) or the plain recursion `yₜ = α·xₜ + (1−α)·yₜ₋₁`.
    adjust: bool,
}

impl<'a, T: Temporal> Window<'a, T> {
    /// Create a new window operation instance
    pub fn new(
        time_series: &'a TimeSeries<T>,
        window_type: WindowType,
        window_size: usize,
    ) -> Result<Self> {
        // Validate window size
        if window_size == 0 || (window_type == WindowType::Fixed && window_size > time_series.len())
        {
            return Err(PandRSError::Consistency(format!(
                "Invalid window size ({}). Must be greater than 0 and less than or equal to the data length ({}).",
                window_size, time_series.len()
            )));
        }

        Ok(Window {
            time_series,
            window_type,
            window_size,
            alpha: None,
            adjust: true,
        })
    }

    /// Choose between the adjusted weighted average (`true`, the default and
    /// pandas' default) and the plain recursion (`false`) for exponentially
    /// weighted statistics.
    pub fn with_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    /// Set the decay factor for exponentially weighted window
    /// alpha: 0.0 < alpha <= 1.0, larger values give higher weights to more recent data
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(PandRSError::Consistency(format!(
                "Decay factor alpha ({}) must be greater than 0 and less than or equal to 1.",
                alpha
            )));
        }

        self.alpha = Some(alpha);
        Ok(self)
    }

    /// Calculate mean
    pub fn mean(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_mean(),
            WindowType::Expanding => self.expanding_window_mean(),
            WindowType::ExponentiallyWeighted => self.ewm_mean(),
        }
    }

    /// Calculate sum
    pub fn sum(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_sum(),
            WindowType::Expanding => self.expanding_window_sum(),
            WindowType::ExponentiallyWeighted => Err(PandRSError::Operation(
                "Sum operation is not supported for exponentially weighted windows.".to_string(),
            )),
        }
    }

    /// Calculate standard deviation
    pub fn std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_std(ddof),
            WindowType::Expanding => self.expanding_window_std(ddof),
            WindowType::ExponentiallyWeighted => self.ewm_std(ddof),
        }
    }

    /// Calculate minimum
    pub fn min(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_min(),
            WindowType::Expanding => self.expanding_window_min(),
            WindowType::ExponentiallyWeighted => Err(PandRSError::Operation(
                "Min operation is not supported for exponentially weighted windows.".to_string(),
            )),
        }
    }

    /// Calculate maximum
    pub fn max(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_max(),
            WindowType::Expanding => self.expanding_window_max(),
            WindowType::ExponentiallyWeighted => Err(PandRSError::Operation(
                "Max operation is not supported for exponentially weighted windows.".to_string(),
            )),
        }
    }

    /// Apply a general aggregation operation
    pub fn aggregate<F>(&self, agg_func: F, min_periods: Option<usize>) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let min_periods = min_periods.unwrap_or(1);
        if min_periods == 0 {
            return Err(PandRSError::Consistency(
                "min_periods must be greater than or equal to 1.".to_string(),
            ));
        }

        match self.window_type {
            WindowType::Fixed => self.fixed_window_aggregate(agg_func, min_periods),
            WindowType::Expanding => self.expanding_window_aggregate(agg_func, min_periods),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("General aggregation operations are not supported for exponentially weighted windows.".to_string()))
            }
        }
    }

    // Implementations for each window type

    // ------- Fixed Window Implementations -------

    /// Calculate fixed window mean
    fn fixed_window_mean(&self) -> Result<TimeSeries<T>> {
        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving average
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let sum: f64 = window_values.iter().sum();
                    let mean = sum / window_values.len() as f64;
                    result_values.push(NA::Value(mean));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate fixed window sum
    fn fixed_window_sum(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the fixed window sum implementation

        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving sum
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let sum: f64 = window_values.iter().sum();
                    result_values.push(NA::Value(sum));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate fixed window standard deviation
    fn fixed_window_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the fixed window std implementation

        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving standard deviation
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.len() <= ddof {
                    result_values.push(NA::NA);
                } else {
                    // Calculate mean
                    let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;

                    // Calculate variance
                    let variance: f64 = window_values
                        .iter()
                        .map(|v| (*v - mean).powi(2))
                        .sum::<f64>()
                        / (window_values.len() - ddof) as f64;

                    // Calculate standard deviation
                    let std_dev = variance.sqrt();
                    result_values.push(NA::Value(std_dev));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate fixed window minimum
    fn fixed_window_min(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the fixed window min implementation

        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving minimum
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let min = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    result_values.push(NA::Value(min));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate fixed window maximum
    fn fixed_window_max(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the fixed window max implementation

        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving maximum
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let max = window_values
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    result_values.push(NA::Value(max));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Apply a general aggregation function to fixed window
    fn fixed_window_aggregate<F>(&self, agg_func: F, min_periods: usize) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the fixed window aggregate implementation

        let window_size = self.window_size;
        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate moving aggregation
        for i in 0..self.time_series.len() {
            if i < window_size - 1 {
                // The first window-1 elements are NA
                result_values.push(NA::NA);
            } else {
                // Get values within the window
                let start_idx = i.checked_sub(window_size - 1).unwrap_or(0);
                let window_values: Vec<f64> = self.time_series.values()[start_idx..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.len() < min_periods {
                    result_values.push(NA::NA);
                } else {
                    let result = agg_func(&window_values);
                    result_values.push(NA::Value(result));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    // ------- Expanding Window Implementations -------

    /// Calculate expanding window mean
    fn expanding_window_mean(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window mean implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding mean
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let sum: f64 = window_values.iter().sum();
                    let mean = sum / window_values.len() as f64;
                    result_values.push(NA::Value(mean));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate expanding window sum
    fn expanding_window_sum(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window sum implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding sum
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let sum: f64 = window_values.iter().sum();
                    result_values.push(NA::Value(sum));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate expanding window standard deviation
    fn expanding_window_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window std implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding standard deviation
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.len() <= ddof {
                    result_values.push(NA::NA);
                } else {
                    // Calculate mean
                    let mean: f64 = window_values.iter().sum::<f64>() / window_values.len() as f64;

                    // Calculate variance
                    let variance: f64 = window_values
                        .iter()
                        .map(|v| (*v - mean).powi(2))
                        .sum::<f64>()
                        / (window_values.len() - ddof) as f64;

                    // Calculate standard deviation
                    let std_dev = variance.sqrt();
                    result_values.push(NA::Value(std_dev));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate expanding window minimum
    fn expanding_window_min(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window min implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding minimum
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let min = window_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    result_values.push(NA::Value(min));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate expanding window maximum
    fn expanding_window_max(&self) -> Result<TimeSeries<T>> {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window max implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding maximum
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.is_empty() {
                    result_values.push(NA::NA);
                } else {
                    let max = window_values
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    result_values.push(NA::Value(max));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Apply a general aggregation function to expanding window
    fn expanding_window_aggregate<F>(
        &self,
        agg_func: F,
        min_periods: usize,
    ) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
        // Implementation omitted for brevity - see the original window.rs file
        // This would include the expanding window aggregate implementation

        let mut result_values = Vec::with_capacity(self.time_series.len());

        // Calculate expanding aggregation
        for i in 0..self.time_series.len() {
            if i < self.window_size - 1 {
                // If the minimum window size is not met, return NA
                result_values.push(NA::NA);
            } else {
                // Get values from the beginning to the current index
                let window_values: Vec<f64> = self.time_series.values()[0..=i]
                    .iter()
                    .filter_map(|v| match v {
                        NA::Value(val) => Some(*val),
                        NA::NA => None,
                    })
                    .collect();

                if window_values.len() < min_periods {
                    result_values.push(NA::NA);
                } else {
                    let result = agg_func(&window_values);
                    result_values.push(NA::Value(result));
                }
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    // ------- Exponentially Weighted Window Implementations -------

    /// Calculate the exponentially weighted moving average.
    ///
    /// Honours `adjust` (see [`Window::with_adjust`]): with `adjust = true`
    /// (the default, matching pandas) the value is the *adjusted* weighted
    /// average over every prior observation,
    /// `yₜ = Σᵢ (1−α)ⁱ xₜ₋ᵢ / Σᵢ (1−α)ⁱ`; with `adjust = false` it is the plain
    /// recursion `yₜ = α·xₜ + (1−α)·yₜ₋₁`.
    ///
    /// **NA policy (unchanged):** positions before the first observation are
    /// `NA`; a missing observation afterwards leaves the state untouched and
    /// carries the previous value forward rather than propagating `NA`.
    fn ewm_mean(&self) -> Result<TimeSeries<T>> {
        let alpha = self.require_alpha()?;
        let values = self.time_series.values();

        if values.is_empty() {
            return TimeSeries::new(Vec::new(), Vec::new(), self.time_series.name().cloned());
        }

        let state = Self::ewm_recursion(values, alpha, self.adjust);
        let result_values: Vec<NA<f64>> = state
            .mean
            .iter()
            .map(|m| match m {
                Some(value) => NA::Value(*value),
                None => NA::NA,
            })
            .collect();

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// Calculate the exponentially weighted moving standard deviation.
    ///
    /// The variance comes from the numerically sound weighted-covariance
    /// recursion in [`Window::ewm_recursion`] (the same one pandas uses in
    /// `_libs/window/aggregations.pyx`), **not** from `E[X²] − (E[X])²`.
    /// That difference-of-squares form cancels catastrophically once the mean
    /// dominates the spread: for a series around `1e8` with unit noise the two
    /// accumulators agree to well past `f64`'s 16 significant digits, the
    /// subtraction yields a negative number, and the old
    /// `if variance > 0.0 { sqrt } else { 0.0 }` guard reported the deviation
    /// as **exactly `0.0`** — a plausible-looking answer that is off by 100%.
    ///
    /// `ddof` is now honoured (it was validated and then ignored). The
    /// correction uses the weights' effective sample size
    /// `n_eff = (Σw)² / Σw²`: the reported variance is
    /// `cov · n_eff / (n_eff − ddof)`, and `NA` where there is not enough
    /// effective sample for it (`n_eff ≤ ddof` — with pandas' default
    /// `ddof = 1` that is always the first observation).
    fn ewm_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        let alpha = self.require_alpha()?;

        // Degrees of freedom adjustment
        if ddof >= self.time_series.len() {
            return Err(PandRSError::Consistency(format!(
                "Degrees of freedom adjustment ddof ({}) is greater than or equal to the sample size ({})",
                ddof, self.time_series.len()
            )));
        }

        let values = self.time_series.values();
        if values.is_empty() {
            return TimeSeries::new(Vec::new(), Vec::new(), self.time_series.name().cloned());
        }

        let state = Self::ewm_recursion(values, alpha, self.adjust);
        let ddof = ddof as f64;

        let mut result_values = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            if state.mean[i].is_none() {
                result_values.push(NA::NA);
                continue;
            }
            let n_eff = state.n_eff[i];
            let denom = n_eff - ddof;
            if denom > 0.0 && state.cov_biased[i].is_finite() {
                // The weighted sum of squared deviations cannot be negative;
                // clamp only to absorb float rounding noise near zero.
                let variance = (state.cov_biased[i] * n_eff / denom).max(0.0);
                result_values.push(NA::Value(variance.sqrt()));
            } else {
                result_values.push(NA::NA);
            }
        }

        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }

    /// The decay factor, or an error when it was never configured.
    fn require_alpha(&self) -> Result<f64> {
        self.alpha.ok_or_else(|| {
            PandRSError::Consistency(
                "Alpha parameter is required for exponentially weighted windows.".to_string(),
            )
        })
    }

    /// The recursive EWM algorithm, tracking the weighted mean, the biased
    /// (`ddof = 0`) weighted covariance and the weight moments needed to
    /// bias-correct it for an arbitrary `ddof`.
    ///
    /// This mirrors `series::window`'s `ewm_recursion`, specialized to this
    /// module's NA policy (a missing observation costs no decay weight and
    /// leaves the running value in place, so the mean carries forward).
    /// It is duplicated rather than shared because the `series` version is a
    /// private item of a module this file does not own; the two should be
    /// unified behind one crate-internal helper when that file next changes.
    fn ewm_recursion(values: &[NA<f64>], alpha: f64, adjust: bool) -> EwmState {
        let n = values.len();
        let old_wt_factor = 1.0 - alpha;
        let new_wt = if adjust { 1.0 } else { alpha };

        let mut mean_out = vec![None; n];
        let mut cov_out = vec![f64::NAN; n];
        let mut n_eff_out = vec![f64::NAN; n];

        let mut mean: Option<f64> = None;
        let mut cov = 0.0_f64;
        let mut old_wt = 1.0_f64;
        let mut sum_wt = 1.0_f64;
        let mut sum_wt2 = 1.0_f64;

        for (i, value) in values.iter().enumerate() {
            if let NA::Value(x) = *value {
                match mean {
                    None => {
                        mean = Some(x);
                        cov = 0.0;
                    }
                    Some(old_mean) => {
                        old_wt *= old_wt_factor;
                        sum_wt *= old_wt_factor;
                        sum_wt2 *= old_wt_factor * old_wt_factor;

                        let new_mean = if old_mean != x {
                            (old_wt * old_mean + new_wt * x) / (old_wt + new_wt)
                        } else {
                            old_mean
                        };
                        cov = (old_wt * (cov + (old_mean - new_mean) * (old_mean - new_mean))
                            + new_wt * (x - new_mean) * (x - new_mean))
                            / (old_wt + new_wt);
                        mean = Some(new_mean);

                        sum_wt += new_wt;
                        sum_wt2 += new_wt * new_wt;
                        old_wt += new_wt;
                        if !adjust {
                            sum_wt /= old_wt;
                            sum_wt2 /= old_wt * old_wt;
                            old_wt = 1.0;
                        }
                    }
                }
            }

            if mean.is_some() {
                mean_out[i] = mean;
                cov_out[i] = cov;
                n_eff_out[i] = if sum_wt2 > 0.0 {
                    sum_wt * sum_wt / sum_wt2
                } else {
                    f64::NAN
                };
            }
        }

        EwmState {
            mean: mean_out,
            cov_biased: cov_out,
            n_eff: n_eff_out,
        }
    }
}

/// Running EWM state at every position: the exponentially weighted mean
/// (`None` before the first observation), the biased (`ddof = 0`) weighted
/// variance, and the weights' effective sample size `(Σw)² / Σw²`.
struct EwmState {
    mean: Vec<Option<f64>>,
    cov_biased: Vec<f64>,
    n_eff: Vec<f64>,
}

// Add window creation methods to TimeSeries struct
impl<T: Temporal> crate::temporal::core::TimeSeries<T> {
    /// Create fixed window operation
    pub fn rolling(&self, window_size: usize) -> Result<Window<T>> {
        Window::new(self, WindowType::Fixed, window_size)
    }

    /// Create expanding window operation
    pub fn expanding(&self, min_periods: usize) -> Result<Window<T>> {
        Window::new(self, WindowType::Expanding, min_periods)
    }

    /// Create exponentially weighted window operation
    pub fn ewm(&self, span: Option<usize>, alpha: Option<f64>, adjust: bool) -> Result<Window<T>> {
        // Error if both span and alpha are specified
        if span.is_some() && alpha.is_some() {
            return Err(PandRSError::Consistency(
                "Cannot specify both span and alpha. Please specify only one of them.".to_string(),
            ));
        }

        // Calculate or use alpha directly
        let alpha_value = if let Some(alpha_val) = alpha {
            alpha_val
        } else if let Some(span_val) = span {
            if span_val < 1 {
                return Err(PandRSError::Consistency(
                    "span must be greater than or equal to 1.".to_string(),
                ));
            }
            // alpha = 2/(span+1)
            2.0 / (span_val as f64 + 1.0)
        } else {
            // Default is equivalent to span=5
            2.0 / (5.0 + 1.0)
        };

        // Create window (window_size is set to 1, not actually used)
        let mut window = Window::new(self, WindowType::ExponentiallyWeighted, 1)?;
        window = window.with_alpha(alpha_value)?.with_adjust(adjust);

        Ok(window)
    }
}

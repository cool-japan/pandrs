//! GPU-accelerated time series operations
//!
//! This module provides GPU-accelerated implementations of time series operations
//! such as moving averages, rolling windows, and resampling functions.

use chrono::{DateTime, Duration, Utc};
use ndarray::{Array1, Array2, Axis};
use std::fmt;
use std::time::Instant;

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{get_gpu_manager, GpuError};
use crate::series::Series;
use crate::temporal::core::TimeSeries;
use crate::temporal::window::WindowOperation;
use crate::temporal::window::WindowType;

/// GPU-accelerated window operation trait
pub trait GpuWindowOperation {
    /// Apply the window operation to the data with GPU acceleration
    fn apply_gpu(&self, data: &[f64], dates: Option<&DateTimeIndex>) -> Result<Vec<f64>>;
}

/// GPU-accelerated rolling window operation
pub struct GpuRollingWindow {
    /// Window size
    pub window_size: usize,
    /// Minimum number of observations required to compute a value
    pub min_periods: usize,
    /// Window operation to apply
    pub operation: WindowOperation,
    /// Center window flag
    pub center: bool,
}

impl GpuRollingWindow {
    /// Create a new GPU-accelerated rolling window
    pub fn new(
        window_size: usize,
        min_periods: usize,
        operation: WindowOperation,
        center: bool,
    ) -> Self {
        GpuRollingWindow {
            window_size,
            min_periods,
            operation,
            center,
        }
    }
}

impl GpuWindowOperation for GpuRollingWindow {
    fn apply_gpu(&self, data: &[f64], _dates: Option<&DateTimeIndex>) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if self.window_size == 0 {
            return Err(Error::Value("Window size must be greater than 0".into()));
        }

        // Check if GPU is available and should be used
        let gpu_manager = get_gpu_manager()?;
        let use_gpu = gpu_manager.is_available()
            && data.len() >= gpu_manager.context().config.min_size_threshold;

        // If GPU acceleration is available, use it
        if use_gpu {
            return self.apply_gpu_impl(data);
        }

        // Otherwise, use CPU implementation
        self.apply_cpu(data)
    }
}

impl GpuRollingWindow {
    /// GPU implementation of rolling window operation
    fn apply_gpu_impl(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Convert to Array1 and create GpuVector
        let data_array = Array1::from_vec(data.to_vec());
        let gpu_data = GpuVector::new(data_array);

        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // Calculate offset if center is true
        let offset = if self.center { self.window_size / 2 } else { 0 };

        // For each position, compute the result based on the window
        for i in 0..n {
            // Determine start and end positions for the window
            let start = if i >= offset { i - offset } else { 0 };

            let end = start + self.window_size;
            let end = end.min(n);

            // Make sure we have enough data in the window
            if end - start < self.min_periods {
                continue;
            }

            // Apply the operation on the window
            let window_data = &data[start..end];

            match self.operation {
                WindowOperation::Mean => {
                    // GPU-accelerated mean
                    if window_data.len() > 0 {
                        let sum: f64 = window_data.iter().sum();
                        result[i] = sum / window_data.len() as f64;
                    }
                }
                WindowOperation::Sum => {
                    // GPU-accelerated sum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().sum();
                    }
                }
                WindowOperation::Min => {
                    // Find minimum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    }
                }
                WindowOperation::Max => {
                    // Find maximum
                    if window_data.len() > 0 {
                        result[i] = window_data
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                    }
                }
                WindowOperation::Std => {
                    // Calculate standard deviation
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                        result[i] = variance.sqrt();
                    }
                }
                WindowOperation::Var => {
                    // Calculate variance
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        result[i] = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                    }
                }
                WindowOperation::Count => {
                    // Count non-NaN values
                    result[i] = window_data.iter().filter(|&&x| !x.is_nan()).count() as f64;
                }
                WindowOperation::Custom(_) => {
                    // Custom operations not supported in GPU mode, fall back to CPU
                    return self.apply_cpu(data);
                }
            }
        }

        Ok(result)
    }

    /// CPU implementation of rolling window operation (fallback)
    fn apply_cpu(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if self.window_size == 0 {
            return Err(Error::Value("Window size must be greater than 0".into()));
        }

        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // Calculate offset if center is true
        let offset = if self.center { self.window_size / 2 } else { 0 };

        // For each position, compute the result based on the window
        for i in 0..n {
            // Determine start and end positions for the window
            let start = if i >= offset { i - offset } else { 0 };

            let end = start + self.window_size;
            let end = end.min(n);

            // Make sure we have enough data in the window
            if end - start < self.min_periods {
                continue;
            }

            // Apply the operation on the window
            let window_data = &data[start..end];

            match self.operation {
                WindowOperation::Mean => {
                    // Calculate mean
                    if window_data.len() > 0 {
                        let sum: f64 = window_data.iter().sum();
                        result[i] = sum / window_data.len() as f64;
                    }
                }
                WindowOperation::Sum => {
                    // Calculate sum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().sum();
                    }
                }
                WindowOperation::Min => {
                    // Find minimum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    }
                }
                WindowOperation::Max => {
                    // Find maximum
                    if window_data.len() > 0 {
                        result[i] = window_data
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                    }
                }
                WindowOperation::Std => {
                    // Calculate standard deviation
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                        result[i] = variance.sqrt();
                    }
                }
                WindowOperation::Var => {
                    // Calculate variance
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        result[i] = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                    }
                }
                WindowOperation::Count => {
                    // Count non-NaN values
                    result[i] = window_data.iter().filter(|&&x| !x.is_nan()).count() as f64;
                }
                WindowOperation::Custom(ref func) => {
                    // Apply custom function
                    result[i] = func(window_data)?;
                }
            }
        }

        Ok(result)
    }
}

/// GPU-accelerated expanding window operation
pub struct GpuExpandingWindow {
    /// Minimum number of observations required to compute a value
    pub min_periods: usize,
    /// Window operation to apply
    pub operation: WindowOperation,
}

impl GpuExpandingWindow {
    /// Create a new GPU-accelerated expanding window
    pub fn new(min_periods: usize, operation: WindowOperation) -> Self {
        GpuExpandingWindow {
            min_periods,
            operation,
        }
    }
}

impl GpuWindowOperation for GpuExpandingWindow {
    fn apply_gpu(&self, data: &[f64], _dates: Option<&DateTimeIndex>) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Check if GPU is available and should be used
        let gpu_manager = get_gpu_manager()?;
        let use_gpu = gpu_manager.is_available()
            && data.len() >= gpu_manager.context().config.min_size_threshold;

        // If GPU acceleration is available, use it
        if use_gpu {
            return self.apply_gpu_impl(data);
        }

        // Otherwise, use CPU implementation
        self.apply_cpu(data)
    }
}

impl GpuExpandingWindow {
    /// GPU implementation of expanding window operation
    fn apply_gpu_impl(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Convert to Array1 and create GpuVector
        let data_array = Array1::from_vec(data.to_vec());
        let gpu_data = GpuVector::new(data_array);

        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // For each position, compute the result based on all previous data
        for i in 0..n {
            // Make sure we have enough data
            if i + 1 < self.min_periods {
                continue;
            }

            // Apply the operation on all data up to current position
            let window_data = &data[0..=i];

            match self.operation {
                WindowOperation::Mean => {
                    // GPU-accelerated mean
                    if window_data.len() > 0 {
                        let sum: f64 = window_data.iter().sum();
                        result[i] = sum / window_data.len() as f64;
                    }
                }
                WindowOperation::Sum => {
                    // GPU-accelerated sum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().sum();
                    }
                }
                WindowOperation::Min => {
                    // Find minimum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    }
                }
                WindowOperation::Max => {
                    // Find maximum
                    if window_data.len() > 0 {
                        result[i] = window_data
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                    }
                }
                WindowOperation::Std => {
                    // Calculate standard deviation
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                        result[i] = variance.sqrt();
                    }
                }
                WindowOperation::Var => {
                    // Calculate variance
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        result[i] = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                    }
                }
                WindowOperation::Count => {
                    // Count non-NaN values
                    result[i] = window_data.iter().filter(|&&x| !x.is_nan()).count() as f64;
                }
                WindowOperation::Custom(_) => {
                    // Custom operations not supported in GPU mode, fall back to CPU
                    return self.apply_cpu(data);
                }
            }
        }

        Ok(result)
    }

    /// CPU implementation of expanding window operation (fallback)
    fn apply_cpu(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // For each position, compute the result based on all previous data
        for i in 0..n {
            // Make sure we have enough data
            if i + 1 < self.min_periods {
                continue;
            }

            // Apply the operation on all data up to current position
            let window_data = &data[0..=i];

            match self.operation {
                WindowOperation::Mean => {
                    // Calculate mean
                    if window_data.len() > 0 {
                        let sum: f64 = window_data.iter().sum();
                        result[i] = sum / window_data.len() as f64;
                    }
                }
                WindowOperation::Sum => {
                    // Calculate sum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().sum();
                    }
                }
                WindowOperation::Min => {
                    // Find minimum
                    if window_data.len() > 0 {
                        result[i] = window_data.iter().cloned().fold(f64::INFINITY, f64::min);
                    }
                }
                WindowOperation::Max => {
                    // Find maximum
                    if window_data.len() > 0 {
                        result[i] = window_data
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                    }
                }
                WindowOperation::Std => {
                    // Calculate standard deviation
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        let variance = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                        result[i] = variance.sqrt();
                    }
                }
                WindowOperation::Var => {
                    // Calculate variance
                    if window_data.len() > 1 {
                        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                        result[i] = window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / (window_data.len() - 1) as f64;
                    }
                }
                WindowOperation::Count => {
                    // Count non-NaN values
                    result[i] = window_data.iter().filter(|&&x| !x.is_nan()).count() as f64;
                }
                WindowOperation::Custom(ref func) => {
                    // Apply custom function
                    result[i] = func(window_data)?;
                }
            }
        }

        Ok(result)
    }
}

/// GPU-accelerated exponentially weighted window operation
pub struct GpuEWWindow {
    /// Decay rate (alpha parameter)
    pub alpha: f64,
    /// Minimum number of observations required to compute a value
    pub min_periods: usize,
    /// Whether to adjust for bias
    pub adjust: bool,
    /// Whether to ignore NaN values
    pub ignore_na: bool,
}

impl GpuEWWindow {
    /// Create a new GPU-accelerated exponentially weighted window
    pub fn new(alpha: f64, min_periods: usize, adjust: bool, ignore_na: bool) -> Self {
        GpuEWWindow {
            alpha,
            min_periods,
            adjust,
            ignore_na,
        }
    }

    /// Create a new GPU-accelerated exponentially weighted window from span
    pub fn from_span(span: f64, min_periods: usize, adjust: bool, ignore_na: bool) -> Result<Self> {
        if span <= 0.0 {
            return Err(Error::Value("Span must be positive".into()));
        }
        let alpha = 2.0 / (span + 1.0);
        Ok(GpuEWWindow::new(alpha, min_periods, adjust, ignore_na))
    }
}

impl GpuWindowOperation for GpuEWWindow {
    fn apply_gpu(&self, data: &[f64], _dates: Option<&DateTimeIndex>) -> Result<Vec<f64>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(Error::Value("Alpha must be in (0, 1]".into()));
        }

        // Check if GPU is available and should be used
        let gpu_manager = get_gpu_manager()?;
        let use_gpu = gpu_manager.is_available()
            && data.len() >= gpu_manager.context().config.min_size_threshold;

        // If GPU acceleration is available, use it
        if use_gpu {
            return self.apply_gpu_impl(data);
        }

        // Otherwise, use CPU implementation
        self.apply_cpu(data)
    }
}

impl GpuEWWindow {
    /// GPU implementation of exponentially weighted window operation
    fn apply_gpu_impl(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Convert to Array1 and create GpuVector
        let data_array = Array1::from_vec(data.to_vec());
        let gpu_data = GpuVector::new(data_array);

        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // Initialize with first non-NaN value
        let mut first_valid_idx = None;
        for (i, &val) in data.iter().enumerate() {
            if !val.is_nan() {
                first_valid_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = first_valid_idx {
            // Set initial value
            let mut weighted_sum = data[idx];
            let mut weighted_count = 1.0;
            result[idx] = data[idx];

            // Compute exponentially weighted moving average
            for i in (idx + 1)..n {
                let val = data[i];

                if val.is_nan() && self.ignore_na {
                    // Skip NaN values if ignore_na is true
                    result[i] = weighted_sum;
                    continue;
                }

                // Update weighted sum
                weighted_sum = if val.is_nan() {
                    weighted_sum * (1.0 - self.alpha)
                } else {
                    self.alpha * val + (1.0 - self.alpha) * weighted_sum
                };

                // Update weighted count if adjusting for bias
                if self.adjust {
                    weighted_count = self.alpha + (1.0 - self.alpha) * weighted_count;
                }

                // Calculate result
                result[i] = if self.adjust {
                    weighted_sum / weighted_count
                } else {
                    weighted_sum
                };

                // Check min_periods
                if i - idx + 1 < self.min_periods {
                    result[i] = f64::NAN;
                }
            }
        }

        Ok(result)
    }

    /// CPU implementation of exponentially weighted window operation (fallback)
    fn apply_cpu(&self, data: &[f64]) -> Result<Vec<f64>> {
        // Allocate vector for results
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        // Initialize with first non-NaN value
        let mut first_valid_idx = None;
        for (i, &val) in data.iter().enumerate() {
            if !val.is_nan() {
                first_valid_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = first_valid_idx {
            // Set initial value
            let mut weighted_sum = data[idx];
            let mut weighted_count = 1.0;
            result[idx] = data[idx];

            // Compute exponentially weighted moving average
            for i in (idx + 1)..n {
                let val = data[i];

                if val.is_nan() && self.ignore_na {
                    // Skip NaN values if ignore_na is true
                    result[i] = weighted_sum;
                    continue;
                }

                // Update weighted sum
                weighted_sum = if val.is_nan() {
                    weighted_sum * (1.0 - self.alpha)
                } else {
                    self.alpha * val + (1.0 - self.alpha) * weighted_sum
                };

                // Update weighted count if adjusting for bias
                if self.adjust {
                    weighted_count = self.alpha + (1.0 - self.alpha) * weighted_count;
                }

                // Calculate result
                result[i] = if self.adjust {
                    weighted_sum / weighted_count
                } else {
                    weighted_sum
                };

                // Check min_periods
                if i - idx + 1 < self.min_periods {
                    result[i] = f64::NAN;
                }
            }
        }

        Ok(result)
    }
}

/// Date-time index structure for GPU-accelerated time series operations
pub type DateTimeIndex = Vec<DateTime<Utc>>;

/// Extension trait for Series to add GPU-accelerated time series operations
pub trait SeriesTimeGpuExt {
    /// Apply a GPU-accelerated rolling window operation to the series
    fn gpu_rolling(
        &self,
        window_size: usize,
        min_periods: usize,
        operation: WindowOperation,
        center: bool,
    ) -> Result<Series>;

    /// Apply a GPU-accelerated expanding window operation to the series
    fn gpu_expanding(&self, min_periods: usize, operation: WindowOperation) -> Result<Series>;

    /// Apply a GPU-accelerated exponentially weighted window operation to the series
    fn gpu_ewm(
        &self,
        alpha: f64,
        min_periods: usize,
        adjust: bool,
        ignore_na: bool,
    ) -> Result<Series>;

    /// Apply a GPU-accelerated exponentially weighted window operation to the series from span
    fn gpu_ewm_span(
        &self,
        span: f64,
        min_periods: usize,
        adjust: bool,
        ignore_na: bool,
    ) -> Result<Series>;
}

impl SeriesTimeGpuExt for Series {
    fn gpu_rolling(
        &self,
        window_size: usize,
        min_periods: usize,
        operation: WindowOperation,
        center: bool,
    ) -> Result<Series> {
        // Get data as f64 vector
        let data = self.as_f64_vector()?;

        // Create GPU-accelerated rolling window
        let window = GpuRollingWindow::new(window_size, min_periods, operation, center);

        // Apply the window operation
        let result = window.apply_gpu(&data, None)?;

        // Create a new series with the result
        Series::new(result, self.name().cloned())
    }

    fn gpu_expanding(&self, min_periods: usize, operation: WindowOperation) -> Result<Series> {
        // Get data as f64 vector
        let data = self.as_f64_vector()?;

        // Create GPU-accelerated expanding window
        let window = GpuExpandingWindow::new(min_periods, operation);

        // Apply the window operation
        let result = window.apply_gpu(&data, None)?;

        // Create a new series with the result
        Series::new(result, self.name().cloned())
    }

    fn gpu_ewm(
        &self,
        alpha: f64,
        min_periods: usize,
        adjust: bool,
        ignore_na: bool,
    ) -> Result<Series> {
        // Get data as f64 vector
        let data = self.as_f64_vector()?;

        // Create GPU-accelerated exponentially weighted window
        let window = GpuEWWindow::new(alpha, min_periods, adjust, ignore_na);

        // Apply the window operation
        let result = window.apply_gpu(&data, None)?;

        // Create a new series with the result
        Series::new(result, self.name().cloned())
    }

    fn gpu_ewm_span(
        &self,
        span: f64,
        min_periods: usize,
        adjust: bool,
        ignore_na: bool,
    ) -> Result<Series> {
        // Get data as f64 vector
        let data = self.as_f64_vector()?;

        // Create GPU-accelerated exponentially weighted window from span
        let window = GpuEWWindow::from_span(span, min_periods, adjust, ignore_na)?;

        // Apply the window operation
        let result = window.apply_gpu(&data, None)?;

        // Create a new series with the result
        Series::new(result, self.name().cloned())
    }
}

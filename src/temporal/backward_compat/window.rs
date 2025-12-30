//! Module for windowing operations on time series data

use std::fmt;

use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::temporal::TimeSeries;
use crate::temporal::Temporal;

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
}

impl<'a, T: Temporal> Window<'a, T> {
    /// Create a new window operation instance
    pub fn new(
        time_series: &'a TimeSeries<T>,
        window_type: WindowType,
        window_size: usize,
    ) -> Result<Self> {
        // Validate window size
        if window_size == 0 || (window_type == WindowType::Fixed && window_size > time_series.len()) {
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
        })
    }
    
    /// Set the decay factor for exponentially weighted window
    /// alpha: 0.0 < alpha <= 1.0, larger values give higher weights to more recent data
    pub fn with_alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(PandRSError::Consistency(format!(
                "Decay factor alpha ({}) must be greater than 0 and less than or equal to 1.", alpha
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
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("Sum operation is not supported for exponentially weighted windows.".to_string()))
            }
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
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("Min operation is not supported for exponentially weighted windows.".to_string()))
            }
        }
    }
    
    /// Calculate maximum
    pub fn max(&self) -> Result<TimeSeries<T>> {
        match self.window_type {
            WindowType::Fixed => self.fixed_window_max(),
            WindowType::Expanding => self.expanding_window_max(),
            WindowType::ExponentiallyWeighted => {
                Err(PandRSError::Operation("Max operation is not supported for exponentially weighted windows.".to_string()))
            }
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
                    let max = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
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
    fn fixed_window_aggregate<F>(
        &self,
        agg_func: F,
        min_periods: usize,
    ) -> Result<TimeSeries<T>>
    where
        F: Fn(&[f64]) -> f64,
    {
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
                    let max = window_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
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
    
    /// Calculate exponentially weighted moving average
    fn ewm_mean(&self) -> Result<TimeSeries<T>> {
        let alpha = self.alpha.ok_or_else(|| {
            PandRSError::Consistency("Alpha parameter is required for exponentially weighted windows.".to_string())
        })?;
        
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // Calculate exponentially weighted moving average
        let values = self.time_series.values();
        
        // If there are no initial values
        if values.is_empty() {
            return Ok(TimeSeries::new(
                Vec::new(),
                Vec::new(),
                self.time_series.name().cloned(),
            )?);
        }
        
        // Find the first non-NA index
        let first_valid_idx = values.iter().position(|v| !v.is_na());
        
        if let Some(idx) = first_valid_idx {
            // Add NA up to the first valid value
            for _ in 0..idx {
                result_values.push(NA::NA);
            }
            
            // Get the first valid value
            let mut weighted_avg = if let NA::Value(first_val) = values[idx] {
                first_val
            } else {
                return Err(PandRSError::Consistency("Invalid initial value".to_string()));
            };
            
            // Add the first value
            result_values.push(NA::Value(weighted_avg));
            
            // Calculate for the remaining values
            for i in (idx + 1)..values.len() {
                match values[i] {
                    NA::Value(val) => {
                        // Update exponentially weighted average: yt = α*xt + (1-α)*yt-1
                        weighted_avg = alpha * val + (1.0 - alpha) * weighted_avg;
                        result_values.push(NA::Value(weighted_avg));
                    }
                    NA::NA => {
                        // Maintain the previous value for NA (NA does not propagate)
                        result_values.push(NA::Value(weighted_avg));
                    }
                }
            }
        } else {
            // If there are no valid values, all are NA
            for _ in 0..values.len() {
                result_values.push(NA::NA);
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
    
    /// Calculate exponentially weighted moving standard deviation
    fn ewm_std(&self, ddof: usize) -> Result<TimeSeries<T>> {
        let alpha = self.alpha.ok_or_else(|| {
            PandRSError::Consistency("Alpha parameter is required for exponentially weighted windows.".to_string())
        })?;
        
        // Degrees of freedom adjustment
        if ddof >= self.time_series.len() {
            return Err(PandRSError::Consistency(format!(
                "Degrees of freedom adjustment ddof ({}) is greater than or equal to the sample size ({})",
                ddof, self.time_series.len()
            )));
        }
        
        let mut result_values = Vec::with_capacity(self.time_series.len());
        
        // Calculate exponentially weighted moving standard deviation
        let values = self.time_series.values();
        
        // If there are no initial values
        if values.is_empty() {
            return Ok(TimeSeries::new(
                Vec::new(),
                Vec::new(),
                self.time_series.name().cloned(),
            )?);
        }
        
        // Find the first non-NA index
        let first_valid_idx = values.iter().position(|v| !v.is_na());
        
        if let Some(idx) = first_valid_idx {
            // Add NA up to the first valid value
            for _ in 0..idx {
                result_values.push(NA::NA);
            }
            
            // Get the first valid value
            let first_val = if let NA::Value(val) = values[idx] {
                val
            } else {
                return Err(PandRSError::Consistency("Invalid initial value".to_string()));
            };
            
            // Set initial values
            let mut weighted_avg = first_val;
            let mut weighted_sq_avg = first_val * first_val;
            
            // Add the first value (standard deviation is 0)
            result_values.push(NA::Value(0.0));
            
            // Calculate for the remaining values
            for i in (idx + 1)..values.len() {
                match values[i] {
                    NA::Value(val) => {
                        // Update exponentially weighted average
                        weighted_avg = alpha * val + (1.0 - alpha) * weighted_avg;
                        
                        // Update exponentially weighted squared average
                        weighted_sq_avg = alpha * val * val + (1.0 - alpha) * weighted_sq_avg;
                        
                        // Variance = E[X^2] - (E[X])^2
                        let variance = weighted_sq_avg - weighted_avg * weighted_avg;
                        
                        // Prevent variance from being negative (to counter numerical errors)
                        let std_dev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
                        
                        result_values.push(NA::Value(std_dev));
                    }
                    NA::NA => {
                        // Maintain the previous value for NA
                        result_values.push(result_values.last().unwrap().clone());
                    }
                }
            }
        } else {
            // If there are no valid values, all are NA
            for _ in 0..values.len() {
                result_values.push(NA::NA);
            }
        }
        
        TimeSeries::new(
            result_values,
            self.time_series.timestamps().to_vec(),
            self.time_series.name().cloned(),
        )
    }
}

// Implementation removed here - using main module implementation
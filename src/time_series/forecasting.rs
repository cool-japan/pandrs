//! Forecasting Module
//!
//! This module provides various forecasting algorithms for time series prediction,
//! including ARIMA, exponential smoothing, moving averages, and trend-based methods.

use crate::core::error::{Error, Result};
use crate::time_series::core::{DateTimeIndex, Frequency, TimeSeries, TimeSeriesData};
use crate::time_series::decomposition::{DecompositionMethod, SeasonalDecomposition};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Forecast result containing predictions and confidence intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Predicted values
    pub forecast: TimeSeries,
    /// Lower confidence interval
    pub lower_ci: TimeSeries,
    /// Upper confidence interval
    pub upper_ci: TimeSeries,
    /// Forecast method used
    pub method: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Forecast metrics
    pub metrics: ForecastMetrics,
    /// Confidence level (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
}

/// Forecast evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: Option<f64>,
    /// Mean Squared Error
    pub mse: Option<f64>,
    /// Root Mean Squared Error
    pub rmse: Option<f64>,
    /// Mean Absolute Percentage Error
    pub mape: Option<f64>,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: Option<f64>,
    /// Akaike Information Criterion
    pub aic: Option<f64>,
    /// Bayesian Information Criterion
    pub bic: Option<f64>,
    /// Model likelihood
    pub log_likelihood: Option<f64>,
}

/// Generic forecaster trait
pub trait Forecaster {
    /// Fit the forecasting model to the time series
    fn fit(&mut self, ts: &TimeSeries) -> Result<()>;

    /// Generate forecasts for the specified number of periods ahead
    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult>;

    /// Get model name
    fn name(&self) -> &str;

    /// Get model parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Calculate in-sample fit metrics
    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics>;
}

/// Simple Moving Average Forecaster
#[derive(Debug, Clone)]
pub struct SimpleMovingAverageForecaster {
    window: usize,
    fitted_values: Option<Vec<f64>>,
    last_values: Option<Vec<f64>>,
    index: Option<DateTimeIndex>,
}

impl SimpleMovingAverageForecaster {
    /// Create a new SMA forecaster
    pub fn new(window: usize) -> Self {
        Self {
            window,
            fitted_values: None,
            last_values: None,
            index: None,
        }
    }
}

impl Forecaster for SimpleMovingAverageForecaster {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        if ts.len() < self.window {
            return Err(Error::InvalidInput(
                "Time series must be longer than window size".to_string(),
            ));
        }

        let mut fitted = Vec::new();
        for i in 0..ts.len() {
            if i < self.window - 1 {
                fitted.push(f64::NAN);
            } else {
                let window_sum: f64 = (i + 1 - self.window..=i)
                    .filter_map(|idx| ts.values.get_f64(idx))
                    .sum();
                fitted.push(window_sum / self.window as f64);
            }
        }

        // Store last window values for forecasting
        let last_values: Vec<f64> = (ts.len() - self.window..ts.len())
            .filter_map(|idx| ts.values.get_f64(idx))
            .collect();

        self.fitted_values = Some(fitted);
        self.last_values = Some(last_values);
        self.index = Some(ts.index.clone());

        Ok(())
    }

    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult> {
        let last_values = self
            .last_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        // SMA forecast is simply the mean of the last window
        let forecast_value = last_values.iter().sum::<f64>() / last_values.len() as f64;

        // Calculate prediction intervals (assuming normal distribution)
        let residual_std = self.calculate_residual_std()?;
        let z_score = self.get_z_score(confidence_level);
        let margin = z_score * residual_std;

        // Create forecast dates
        let last_date = *index.end().unwrap();
        let frequency = index.frequency.clone().unwrap_or(Frequency::Daily);
        let duration = frequency.to_duration();

        let mut forecast_dates = Vec::new();
        for i in 1..=periods {
            forecast_dates.push(last_date + duration * i as i32);
        }

        let forecast_index = DateTimeIndex::with_frequency(forecast_dates, frequency);
        let forecast_values = vec![forecast_value; periods];
        let lower_values = vec![forecast_value - margin; periods];
        let upper_values = vec![forecast_value + margin; periods];

        let forecast_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(forecast_values),
        )?;
        let lower_ci_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(lower_values),
        )?;
        let upper_ci_ts = TimeSeries::new(forecast_index, TimeSeriesData::from_vec(upper_values))?;

        let mut parameters = HashMap::new();
        parameters.insert("window".to_string(), self.window as f64);
        parameters.insert("forecast_value".to_string(), forecast_value);

        Ok(ForecastResult {
            forecast: forecast_ts,
            lower_ci: lower_ci_ts,
            upper_ci: upper_ci_ts,
            method: "Simple Moving Average".to_string(),
            parameters,
            metrics: ForecastMetrics {
                mae: None,
                mse: None,
                rmse: None,
                mape: None,
                smape: None,
                aic: None,
                bic: None,
                log_likelihood: None,
            },
            confidence_level,
        })
    }

    fn name(&self) -> &str {
        "Simple Moving Average"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("window".to_string(), self.window as f64);
        params
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let fitted = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        calculate_forecast_metrics(ts, fitted)
    }
}

impl SimpleMovingAverageForecaster {
    fn calculate_residual_std(&self) -> Result<f64> {
        // Simplified residual calculation for demonstration
        // In practice, this would use actual residuals from fitting
        Ok(1.0) // Default standard deviation
    }

    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Approximate z-scores for common confidence levels
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96, // Default to 95%
        }
    }
}

/// Linear Trend Forecaster
#[derive(Debug, Clone)]
pub struct LinearTrendForecaster {
    slope: Option<f64>,
    intercept: Option<f64>,
    fitted_values: Option<Vec<f64>>,
    index: Option<DateTimeIndex>,
    residual_std: Option<f64>,
}

impl LinearTrendForecaster {
    /// Create a new linear trend forecaster
    pub fn new() -> Self {
        Self {
            slope: None,
            intercept: None,
            fitted_values: None,
            index: None,
            residual_std: None,
        }
    }
}

impl Default for LinearTrendForecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for LinearTrendForecaster {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        if ts.len() < 2 {
            return Err(Error::InvalidInput(
                "Time series must have at least 2 points".to_string(),
            ));
        }

        // Simple linear regression: y = mx + b
        let n = ts.len() as f64;
        let x_values: Vec<f64> = (0..ts.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();

        if y_values.len() != ts.len() {
            return Err(Error::InvalidInput(
                "Missing values not supported".to_string(),
            ));
        }

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = y_values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(&y_values)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate fitted values
        let fitted: Vec<f64> = x_values.iter().map(|&x| slope * x + intercept).collect();

        // Calculate residual standard deviation
        let residuals: Vec<f64> = y_values
            .iter()
            .zip(&fitted)
            .map(|(actual, predicted)| actual - predicted)
            .collect();
        let residual_std = (residuals.iter().map(|r| r * r).sum::<f64>() / (n - 2.0)).sqrt();

        self.slope = Some(slope);
        self.intercept = Some(intercept);
        self.fitted_values = Some(fitted);
        self.index = Some(ts.index.clone());
        self.residual_std = Some(residual_std);

        Ok(())
    }

    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult> {
        let slope = self
            .slope
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let intercept = self
            .intercept
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let residual_std = self
            .residual_std
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        // Create forecast dates
        let last_date = *index.end().unwrap();
        let frequency = index.frequency.clone().unwrap_or(Frequency::Daily);
        let duration = frequency.to_duration();

        let mut forecast_dates = Vec::new();
        let mut forecast_values = Vec::new();
        let start_x = index.len() as f64;

        let z_score = self.get_z_score(confidence_level);

        for i in 1..=periods {
            forecast_dates.push(last_date + duration * i as i32);
            let x = start_x + i as f64 - 1.0;
            let forecast_value = slope * x + intercept;
            forecast_values.push(forecast_value);
        }

        // Calculate prediction intervals
        let margin = z_score * residual_std;
        let lower_values: Vec<f64> = forecast_values.iter().map(|v| v - margin).collect();
        let upper_values: Vec<f64> = forecast_values.iter().map(|v| v + margin).collect();

        let forecast_index = DateTimeIndex::with_frequency(forecast_dates, frequency);

        let forecast_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(forecast_values),
        )?;
        let lower_ci_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(lower_values),
        )?;
        let upper_ci_ts = TimeSeries::new(forecast_index, TimeSeriesData::from_vec(upper_values))?;

        let mut parameters = HashMap::new();
        parameters.insert("slope".to_string(), slope);
        parameters.insert("intercept".to_string(), intercept);
        parameters.insert("residual_std".to_string(), residual_std);

        Ok(ForecastResult {
            forecast: forecast_ts,
            lower_ci: lower_ci_ts,
            upper_ci: upper_ci_ts,
            method: "Linear Trend".to_string(),
            parameters,
            metrics: ForecastMetrics {
                mae: None,
                mse: None,
                rmse: None,
                mape: None,
                smape: None,
                aic: None,
                bic: None,
                log_likelihood: None,
            },
            confidence_level,
        })
    }

    fn name(&self) -> &str {
        "Linear Trend"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        if let Some(slope) = self.slope {
            params.insert("slope".to_string(), slope);
        }
        if let Some(intercept) = self.intercept {
            params.insert("intercept".to_string(), intercept);
        }
        params
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let fitted = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        calculate_forecast_metrics(ts, fitted)
    }
}

impl LinearTrendForecaster {
    fn get_z_score(&self, confidence_level: f64) -> f64 {
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96,
        }
    }
}

/// Exponential Smoothing Forecaster
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingForecaster {
    alpha: f64,         // Level smoothing parameter
    beta: Option<f64>,  // Trend smoothing parameter
    gamma: Option<f64>, // Seasonal smoothing parameter
    seasonal_periods: Option<usize>,
    fitted_values: Option<Vec<f64>>,
    level: Option<f64>,
    trend: Option<f64>,
    seasonal: Option<Vec<f64>>,
    index: Option<DateTimeIndex>,
    residual_std: Option<f64>,
}

impl ExponentialSmoothingForecaster {
    /// Create a simple exponential smoothing forecaster
    pub fn simple(alpha: f64) -> Self {
        Self {
            alpha,
            beta: None,
            gamma: None,
            seasonal_periods: None,
            fitted_values: None,
            level: None,
            trend: None,
            seasonal: None,
            index: None,
            residual_std: None,
        }
    }

    /// Create a double exponential smoothing forecaster (Holt's method)
    pub fn double(alpha: f64, beta: f64) -> Self {
        Self {
            alpha,
            beta: Some(beta),
            gamma: None,
            seasonal_periods: None,
            fitted_values: None,
            level: None,
            trend: None,
            seasonal: None,
            index: None,
            residual_std: None,
        }
    }

    /// Create a triple exponential smoothing forecaster (Holt-Winters method)
    pub fn triple(alpha: f64, beta: f64, gamma: f64, seasonal_periods: usize) -> Self {
        Self {
            alpha,
            beta: Some(beta),
            gamma: Some(gamma),
            seasonal_periods: Some(seasonal_periods),
            fitted_values: None,
            level: None,
            trend: None,
            seasonal: None,
            index: None,
            residual_std: None,
        }
    }
}

impl Forecaster for ExponentialSmoothingForecaster {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        if ts.is_empty() {
            return Err(Error::InvalidInput("Empty time series".to_string()));
        }

        let values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();

        if values.len() != ts.len() {
            return Err(Error::InvalidInput(
                "Missing values not supported".to_string(),
            ));
        }

        match (&self.beta, &self.gamma, &self.seasonal_periods) {
            (None, None, None) => self.fit_simple(&values)?,
            (Some(_), None, None) => self.fit_double(&values)?,
            (Some(_), Some(_), Some(_)) => self.fit_triple(&values)?,
            _ => {
                return Err(Error::InvalidInput(
                    "Invalid smoothing configuration".to_string(),
                ))
            }
        }

        self.index = Some(ts.index.clone());
        Ok(())
    }

    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult> {
        let level = self
            .level
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        let mut forecast_values = Vec::new();

        match (&self.beta, &self.gamma, &self.seasonal_periods) {
            (None, None, None) => {
                // Simple exponential smoothing
                forecast_values = vec![level; periods];
            }
            (Some(beta), None, None) => {
                // Double exponential smoothing
                let trend = self.trend.unwrap_or(0.0);
                for h in 1..=periods {
                    forecast_values.push(level + h as f64 * trend);
                }
            }
            (Some(beta), Some(gamma), Some(seasonal_periods)) => {
                // Triple exponential smoothing
                let trend = self.trend.unwrap_or(0.0);
                let seasonal = self.seasonal.as_ref().unwrap();

                for h in 1..=periods {
                    let seasonal_idx = (h - 1) % seasonal_periods;
                    let forecast = level + h as f64 * trend + seasonal[seasonal_idx];
                    forecast_values.push(forecast);
                }
            }
            _ => return Err(Error::InvalidOperation("Invalid model state".to_string())),
        }

        // Create forecast dates
        let last_date = *index.end().unwrap();
        let frequency = index.frequency.clone().unwrap_or(Frequency::Daily);
        let duration = frequency.to_duration();

        let mut forecast_dates = Vec::new();
        for i in 1..=periods {
            forecast_dates.push(last_date + duration * i as i32);
        }

        // Calculate prediction intervals
        let residual_std = self.residual_std.unwrap_or(1.0);
        let z_score = self.get_z_score(confidence_level);
        let margin = z_score * residual_std;

        let lower_values: Vec<f64> = forecast_values.iter().map(|v| v - margin).collect();
        let upper_values: Vec<f64> = forecast_values.iter().map(|v| v + margin).collect();

        let forecast_index = DateTimeIndex::with_frequency(forecast_dates, frequency);

        let forecast_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(forecast_values),
        )?;
        let lower_ci_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(lower_values),
        )?;
        let upper_ci_ts = TimeSeries::new(forecast_index, TimeSeriesData::from_vec(upper_values))?;

        Ok(ForecastResult {
            forecast: forecast_ts,
            lower_ci: lower_ci_ts,
            upper_ci: upper_ci_ts,
            method: self.method_name(),
            parameters: self.parameters(),
            metrics: ForecastMetrics {
                mae: None,
                mse: None,
                rmse: None,
                mape: None,
                smape: None,
                aic: None,
                bic: None,
                log_likelihood: None,
            },
            confidence_level,
        })
    }

    fn name(&self) -> &str {
        match (&self.beta, &self.gamma) {
            (None, None) => "Simple Exponential Smoothing",
            (Some(_), None) => "Double Exponential Smoothing",
            (Some(_), Some(_)) => "Triple Exponential Smoothing",
            _ => "Exponential Smoothing",
        }
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), self.alpha);
        if let Some(beta) = self.beta {
            params.insert("beta".to_string(), beta);
        }
        if let Some(gamma) = self.gamma {
            params.insert("gamma".to_string(), gamma);
        }
        if let Some(level) = self.level {
            params.insert("level".to_string(), level);
        }
        if let Some(trend) = self.trend {
            params.insert("trend".to_string(), trend);
        }
        params
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let fitted = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        calculate_forecast_metrics(ts, fitted)
    }
}

impl ExponentialSmoothingForecaster {
    fn fit_simple(&mut self, values: &[f64]) -> Result<()> {
        let mut level = values[0];
        let mut fitted = vec![level];

        for &value in &values[1..] {
            level = self.alpha * value + (1.0 - self.alpha) * level;
            fitted.push(level);
        }

        let residuals: Vec<f64> = values
            .iter()
            .zip(&fitted)
            .map(|(actual, predicted)| actual - predicted)
            .collect();
        let residual_std =
            (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();

        self.level = Some(level);
        self.fitted_values = Some(fitted);
        self.residual_std = Some(residual_std);

        Ok(())
    }

    fn fit_double(&mut self, values: &[f64]) -> Result<()> {
        let beta = self.beta.unwrap();

        let mut level = values[0];
        let mut trend = if values.len() > 1 {
            values[1] - values[0]
        } else {
            0.0
        };
        let mut fitted = vec![level];

        for &value in &values[1..] {
            let prev_level = level;
            level = self.alpha * value + (1.0 - self.alpha) * (level + trend);
            trend = beta * (level - prev_level) + (1.0 - beta) * trend;
            fitted.push(level + trend);
        }

        let residuals: Vec<f64> = values
            .iter()
            .zip(&fitted)
            .map(|(actual, predicted)| actual - predicted)
            .collect();
        let residual_std =
            (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();

        self.level = Some(level);
        self.trend = Some(trend);
        self.fitted_values = Some(fitted);
        self.residual_std = Some(residual_std);

        Ok(())
    }

    fn fit_triple(&mut self, values: &[f64]) -> Result<()> {
        let beta = self.beta.unwrap();
        let gamma = self.gamma.unwrap();
        let seasonal_periods = self.seasonal_periods.unwrap();

        if values.len() < seasonal_periods {
            return Err(Error::InvalidInput(
                "Time series must be longer than seasonal periods".to_string(),
            ));
        }

        // Initialize seasonal factors
        let mut seasonal = vec![0.0; seasonal_periods];
        for i in 0..seasonal_periods {
            seasonal[i] = values[i]
                / (values.iter().take(seasonal_periods).sum::<f64>() / seasonal_periods as f64);
        }

        let mut level = values.iter().take(seasonal_periods).sum::<f64>() / seasonal_periods as f64;
        let mut trend = 0.0;
        let mut fitted = Vec::new();

        for (t, &value) in values.iter().enumerate() {
            let seasonal_idx = t % seasonal_periods;

            if t < seasonal_periods {
                fitted.push(level * seasonal[seasonal_idx]);
            } else {
                let prev_level = level;
                level = self.alpha * (value / seasonal[seasonal_idx])
                    + (1.0 - self.alpha) * (level + trend);
                trend = beta * (level - prev_level) + (1.0 - beta) * trend;
                seasonal[seasonal_idx] =
                    gamma * (value / level) + (1.0 - gamma) * seasonal[seasonal_idx];
                fitted.push((level + trend) * seasonal[seasonal_idx]);
            }
        }

        let residuals: Vec<f64> = values
            .iter()
            .zip(&fitted)
            .map(|(actual, predicted)| actual - predicted)
            .collect();
        let residual_std =
            (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();

        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonal = Some(seasonal);
        self.fitted_values = Some(fitted);
        self.residual_std = Some(residual_std);

        Ok(())
    }

    fn method_name(&self) -> String {
        match (&self.beta, &self.gamma) {
            (None, None) => "Simple Exponential Smoothing".to_string(),
            (Some(_), None) => "Double Exponential Smoothing (Holt)".to_string(),
            (Some(_), Some(_)) => "Triple Exponential Smoothing (Holt-Winters)".to_string(),
            _ => "Exponential Smoothing".to_string(),
        }
    }

    fn get_z_score(&self, confidence_level: f64) -> f64 {
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96,
        }
    }
}

/// ARIMA Forecaster (simplified implementation)
#[derive(Debug, Clone)]
pub struct ArimaForecaster {
    p: usize, // AR order
    d: usize, // Differencing order
    q: usize, // MA order
    ar_params: Option<Vec<f64>>,
    ma_params: Option<Vec<f64>>,
    fitted_values: Option<Vec<f64>>,
    residuals: Option<Vec<f64>>,
    index: Option<DateTimeIndex>,
    differenced_series: Option<Vec<f64>>,
    residual_std: Option<f64>,
}

impl ArimaForecaster {
    /// Create a new ARIMA forecaster
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            ar_params: None,
            ma_params: None,
            fitted_values: None,
            residuals: None,
            index: None,
            differenced_series: None,
            residual_std: None,
        }
    }
}

impl Forecaster for ArimaForecaster {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        if ts.len() < self.p + self.d + self.q + 1 {
            return Err(Error::InvalidInput(
                "Time series too short for ARIMA model".to_string(),
            ));
        }

        // This is a simplified ARIMA implementation
        // In practice, this would involve maximum likelihood estimation

        // Step 1: Difference the series
        let mut series = ts.clone();
        for _ in 0..self.d {
            series = series.diff(1)?;
        }

        let differenced_values: Vec<f64> = (self.d..series.len())
            .filter_map(|i| series.values.get_f64(i))
            .collect();

        // Step 2: Fit AR and MA parameters (simplified)
        let ar_params = vec![0.5; self.p]; // Simplified AR parameters
        let ma_params = vec![0.3; self.q]; // Simplified MA parameters

        // Step 3: Calculate fitted values and residuals
        let mut fitted = Vec::new();
        let mut residuals = Vec::new();

        for i in 0..differenced_values.len() {
            let mut forecast = 0.0;

            // AR component
            for j in 0..self.p {
                if i >= j + 1 {
                    forecast += ar_params[j] * differenced_values[i - j - 1];
                }
            }

            // MA component (simplified)
            for j in 0..self.q {
                if i >= j + 1 && j < residuals.len() {
                    forecast += ma_params[j] * residuals[residuals.len() - j - 1];
                }
            }

            fitted.push(forecast);
            residuals.push(differenced_values[i] - forecast);
        }

        let residual_std =
            (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();

        self.ar_params = Some(ar_params);
        self.ma_params = Some(ma_params);
        self.fitted_values = Some(fitted);
        self.residuals = Some(residuals);
        self.index = Some(ts.index.clone());
        self.differenced_series = Some(differenced_values);
        self.residual_std = Some(residual_std);

        Ok(())
    }

    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult> {
        let ar_params = self
            .ar_params
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let ma_params = self
            .ma_params
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let index = self
            .index
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let differenced = self
            .differenced_series
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        // Generate forecasts (simplified)
        let mut forecasts = Vec::new();
        let mut last_values = differenced
            .iter()
            .rev()
            .take(self.p)
            .cloned()
            .collect::<Vec<_>>();
        last_values.reverse();
        let mut last_residuals = residuals
            .iter()
            .rev()
            .take(self.q)
            .cloned()
            .collect::<Vec<_>>();
        last_residuals.reverse();

        for _ in 0..periods {
            let mut forecast = 0.0;

            // AR component
            for j in 0..self.p {
                if j < last_values.len() {
                    forecast += ar_params[j] * last_values[last_values.len() - 1 - j];
                }
            }

            // MA component
            for j in 0..self.q {
                if j < last_residuals.len() {
                    forecast += ma_params[j] * last_residuals[last_residuals.len() - 1 - j];
                }
            }

            forecasts.push(forecast);

            // Update for next iteration
            last_values.push(forecast);
            if last_values.len() > self.p {
                last_values.remove(0);
            }

            last_residuals.push(0.0); // Assume zero residual for future
            if last_residuals.len() > self.q {
                last_residuals.remove(0);
            }
        }

        // Create forecast dates
        let last_date = *index.end().unwrap();
        let frequency = index.frequency.clone().unwrap_or(Frequency::Daily);
        let duration = frequency.to_duration();

        let mut forecast_dates = Vec::new();
        for i in 1..=periods {
            forecast_dates.push(last_date + duration * i as i32);
        }

        // Calculate prediction intervals
        let residual_std = self.residual_std.unwrap_or(1.0);
        let z_score = self.get_z_score(confidence_level);
        let margin = z_score * residual_std;

        let lower_values: Vec<f64> = forecasts.iter().map(|v| v - margin).collect();
        let upper_values: Vec<f64> = forecasts.iter().map(|v| v + margin).collect();

        let forecast_index = DateTimeIndex::with_frequency(forecast_dates, frequency);

        let forecast_ts =
            TimeSeries::new(forecast_index.clone(), TimeSeriesData::from_vec(forecasts))?;
        let lower_ci_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(lower_values),
        )?;
        let upper_ci_ts = TimeSeries::new(forecast_index, TimeSeriesData::from_vec(upper_values))?;

        Ok(ForecastResult {
            forecast: forecast_ts,
            lower_ci: lower_ci_ts,
            upper_ci: upper_ci_ts,
            method: format!("ARIMA({},{},{})", self.p, self.d, self.q),
            parameters: self.parameters(),
            metrics: ForecastMetrics {
                mae: None,
                mse: None,
                rmse: None,
                mape: None,
                smape: None,
                aic: None,
                bic: None,
                log_likelihood: None,
            },
            confidence_level,
        })
    }

    fn name(&self) -> &str {
        "ARIMA"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.p as f64);
        params.insert("d".to_string(), self.d as f64);
        params.insert("q".to_string(), self.q as f64);
        params
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let fitted = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        // For ARIMA, we need to account for differencing
        let mut original_fitted = vec![f64::NAN; self.d];
        original_fitted.extend(fitted.iter());

        calculate_forecast_metrics(ts, &original_fitted)
    }
}

impl ArimaForecaster {
    fn get_z_score(&self, confidence_level: f64) -> f64 {
        match (confidence_level * 100.0) as i32 {
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.96,
        }
    }
}

/// Calculate forecast evaluation metrics
fn calculate_forecast_metrics(ts: &TimeSeries, fitted: &[f64]) -> Result<ForecastMetrics> {
    let actual_values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();

    if actual_values.len() != fitted.len() {
        return Err(Error::DimensionMismatch(
            "Actual and fitted values must have same length".to_string(),
        ));
    }

    let valid_pairs: Vec<(f64, f64)> = actual_values
        .iter()
        .zip(fitted.iter())
        .filter(|(a, f)| a.is_finite() && f.is_finite())
        .map(|(&a, &f)| (a, f))
        .collect();

    if valid_pairs.is_empty() {
        return Ok(ForecastMetrics {
            mae: None,
            mse: None,
            rmse: None,
            mape: None,
            smape: None,
            aic: None,
            bic: None,
            log_likelihood: None,
        });
    }

    let n = valid_pairs.len() as f64;

    // Mean Absolute Error
    let mae = valid_pairs
        .iter()
        .map(|(actual, fitted)| (actual - fitted).abs())
        .sum::<f64>()
        / n;

    // Mean Squared Error
    let mse = valid_pairs
        .iter()
        .map(|(actual, fitted)| (actual - fitted).powi(2))
        .sum::<f64>()
        / n;

    // Root Mean Squared Error
    let rmse = mse.sqrt();

    // Mean Absolute Percentage Error
    let mape = valid_pairs
        .iter()
        .filter(|(actual, _)| *actual != 0.0)
        .map(|(actual, fitted)| ((actual - fitted) / actual).abs())
        .sum::<f64>()
        / n
        * 100.0;

    // Symmetric Mean Absolute Percentage Error
    let smape = valid_pairs
        .iter()
        .map(|(actual, fitted)| {
            let denominator = (actual.abs() + fitted.abs()) / 2.0;
            if denominator != 0.0 {
                (actual - fitted).abs() / denominator
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n
        * 100.0;

    Ok(ForecastMetrics {
        mae: Some(mae),
        mse: Some(mse),
        rmse: Some(rmse),
        mape: Some(mape),
        smape: Some(smape),
        aic: None,            // Would require likelihood calculation
        bic: None,            // Would require likelihood calculation
        log_likelihood: None, // Would require proper likelihood calculation
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_test_series_with_trend() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..50 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let value = 10.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin(); // Trend + small seasonality
            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_simple_moving_average_forecaster() {
        let ts = create_test_series_with_trend();
        let mut forecaster = SimpleMovingAverageForecaster::new(5);

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(10, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 10);
        assert_eq!(result.method, "Simple Moving Average");
        assert!(result.confidence_level == 0.95);
    }

    #[test]
    fn test_linear_trend_forecaster() {
        let ts = create_test_series_with_trend();
        let mut forecaster = LinearTrendForecaster::new();

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(10, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 10);
        assert_eq!(result.method, "Linear Trend");

        // Check that forecast shows increasing trend
        let first_forecast = result.forecast.values.get_f64(0).unwrap();
        let last_forecast = result.forecast.values.get_f64(9).unwrap();
        assert!(
            last_forecast > first_forecast,
            "Should show increasing trend"
        );
    }

    #[test]
    fn test_simple_exponential_smoothing() {
        let ts = create_test_series_with_trend();
        let mut forecaster = ExponentialSmoothingForecaster::simple(0.3);

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(5, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 5);
        assert!(result.method.contains("Simple Exponential Smoothing"));
    }

    #[test]
    fn test_double_exponential_smoothing() {
        let ts = create_test_series_with_trend();
        let mut forecaster = ExponentialSmoothingForecaster::double(0.3, 0.1);

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(5, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 5);
        assert!(result.method.contains("Double Exponential Smoothing"));
    }

    #[test]
    fn test_arima_forecaster() {
        let ts = create_test_series_with_trend();
        let mut forecaster = ArimaForecaster::new(1, 1, 1);

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(5, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 5);
        assert_eq!(result.method, "ARIMA(1,1,1)");
    }

    #[test]
    fn test_forecast_metrics() {
        let ts = create_test_series_with_trend();
        let mut forecaster = LinearTrendForecaster::new();

        forecaster.fit(&ts).unwrap();
        let metrics = forecaster.fit_metrics(&ts).unwrap();

        assert!(metrics.mae.is_some());
        assert!(metrics.mse.is_some());
        assert!(metrics.rmse.is_some());
        assert!(metrics.mae.unwrap() >= 0.0);
        assert!(metrics.mse.unwrap() >= 0.0);
        assert!(metrics.rmse.unwrap() >= 0.0);
    }

    #[test]
    fn test_confidence_intervals() {
        let ts = create_test_series_with_trend();
        let mut forecaster = LinearTrendForecaster::new();

        forecaster.fit(&ts).unwrap();
        let result = forecaster.forecast(5, 0.95).unwrap();

        // Check that confidence intervals make sense
        for i in 0..result.forecast.len() {
            let forecast = result.forecast.values.get_f64(i).unwrap();
            let lower = result.lower_ci.values.get_f64(i).unwrap();
            let upper = result.upper_ci.values.get_f64(i).unwrap();

            assert!(lower < forecast, "Lower CI should be less than forecast");
            assert!(upper > forecast, "Upper CI should be greater than forecast");
        }
    }
}

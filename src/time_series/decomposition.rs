//! Seasonal Decomposition Module
//!
//! This module provides seasonal decomposition methods for time series analysis,
//! including additive and multiplicative decomposition, trend extraction,
//! and seasonal pattern analysis.

use crate::core::error::{Error, Result};
use crate::time_series::core::{TimeSeries, TimeSeriesData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Seasonal decomposition methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionMethod {
    /// Additive decomposition: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
    Additive,
    /// Multiplicative decomposition: Y(t) = Trend(t) * Seasonal(t) * Residual(t)
    Multiplicative,
    /// STL (Seasonal and Trend decomposition using Loess)
    STL,
    /// X-13ARIMA-SEATS (simplified version)
    X13,
}

/// Result of seasonal decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Original time series
    pub original: TimeSeries,
    /// Trend component
    pub trend: TimeSeries,
    /// Seasonal component
    pub seasonal: TimeSeries,
    /// Residual/irregular component
    pub residual: TimeSeries,
    /// Decomposition method used
    pub method: DecompositionMethod,
    /// Seasonal period
    pub period: usize,
    /// Quality metrics
    pub metrics: DecompositionMetrics,
}

/// Decomposition quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionMetrics {
    /// Proportion of variance explained by trend
    pub trend_variance_ratio: f64,
    /// Proportion of variance explained by seasonal component
    pub seasonal_variance_ratio: f64,
    /// Proportion of variance explained by residual
    pub residual_variance_ratio: f64,
    /// Signal-to-noise ratio
    pub signal_to_noise_ratio: f64,
    /// Seasonality strength
    pub seasonality_strength: f64,
    /// Trend strength
    pub trend_strength: f64,
}

/// Seasonal decomposition implementation
pub struct SeasonalDecomposition {
    method: DecompositionMethod,
    period: Option<usize>,
    extrapolate_trend: usize,
}

impl SeasonalDecomposition {
    /// Create a new seasonal decomposition
    pub fn new(method: DecompositionMethod) -> Self {
        Self {
            method,
            period: None,
            extrapolate_trend: 0,
        }
    }

    /// Set seasonal period
    pub fn with_period(mut self, period: usize) -> Self {
        self.period = Some(period);
        self
    }

    /// Set trend extrapolation
    pub fn with_extrapolate_trend(mut self, extrapolate: usize) -> Self {
        self.extrapolate_trend = extrapolate;
        self
    }

    /// Perform decomposition
    pub fn decompose(&self, ts: &TimeSeries) -> Result<DecompositionResult> {
        if ts.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot decompose empty time series".to_string(),
            ));
        }

        let period = self.infer_period(ts)?;

        match self.method {
            DecompositionMethod::Additive => self.additive_decomposition(ts, period),
            DecompositionMethod::Multiplicative => self.multiplicative_decomposition(ts, period),
            DecompositionMethod::STL => self.stl_decomposition(ts, period),
            DecompositionMethod::X13 => self.x13_decomposition(ts, period),
        }
    }

    /// Infer seasonal period from time series
    fn infer_period(&self, ts: &TimeSeries) -> Result<usize> {
        if let Some(period) = self.period {
            return Ok(period);
        }

        // Auto-detect period based on frequency
        match &ts.index.frequency {
            Some(freq) => match freq {
                crate::time_series::core::Frequency::Daily => Ok(7), // Weekly seasonality
                crate::time_series::core::Frequency::Weekly => Ok(52), // Yearly seasonality
                crate::time_series::core::Frequency::Monthly => Ok(12), // Yearly seasonality
                crate::time_series::core::Frequency::Quarterly => Ok(4), // Yearly seasonality
                crate::time_series::core::Frequency::Hour => Ok(24), // Daily seasonality
                crate::time_series::core::Frequency::Minute => Ok(60), // Hourly seasonality
                _ => Ok(12),                                         // Default assumption
            },
            None => {
                // Try to detect period using autocorrelation
                self.detect_period_autocorr(ts)
            }
        }
    }

    /// Detect period using autocorrelation
    fn detect_period_autocorr(&self, ts: &TimeSeries) -> Result<usize> {
        let max_period = std::cmp::min(ts.len() / 2, 100);
        let mut best_period = 12; // Default
        let mut max_correlation = 0.0;

        for period in 2..=max_period {
            let correlation = self.calculate_autocorrelation(ts, period)?;
            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }

        Ok(best_period)
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, ts: &TimeSeries, lag: usize) -> Result<f64> {
        if lag >= ts.len() {
            return Ok(0.0);
        }

        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < lag + 1 {
            return Ok(0.0);
        }

        let n = values.len() - lag;
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let dev1 = values[i] - mean;
            let dev2 = values[i + lag] - mean;
            numerator += dev1 * dev2;
        }

        for &val in &values {
            let dev = val - mean;
            denominator += dev * dev;
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Additive decomposition
    fn additive_decomposition(
        &self,
        ts: &TimeSeries,
        period: usize,
    ) -> Result<DecompositionResult> {
        let trend = self.extract_trend(ts, period)?;
        let detrended = self.subtract_series(ts, &trend)?;
        let seasonal = self.extract_seasonal_additive(&detrended, period)?;
        let residual = self.subtract_series(&detrended, &seasonal)?;

        let metrics = self.calculate_metrics(ts, &trend, &seasonal, &residual)?;

        Ok(DecompositionResult {
            original: ts.clone(),
            trend,
            seasonal,
            residual,
            method: DecompositionMethod::Additive,
            period,
            metrics,
        })
    }

    /// Multiplicative decomposition
    fn multiplicative_decomposition(
        &self,
        ts: &TimeSeries,
        period: usize,
    ) -> Result<DecompositionResult> {
        // Check for non-positive values
        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val <= 0.0 {
                    return Err(Error::InvalidInput(
                        "Multiplicative decomposition requires positive values".to_string(),
                    ));
                }
            }
        }

        let trend = self.extract_trend(ts, period)?;
        let detrended = self.divide_series(ts, &trend)?;
        let seasonal = self.extract_seasonal_multiplicative(&detrended, period)?;
        let residual = self.divide_series(&detrended, &seasonal)?;

        let metrics = self.calculate_metrics(ts, &trend, &seasonal, &residual)?;

        Ok(DecompositionResult {
            original: ts.clone(),
            trend,
            seasonal,
            residual,
            method: DecompositionMethod::Multiplicative,
            period,
            metrics,
        })
    }

    /// STL decomposition (simplified implementation)
    fn stl_decomposition(&self, ts: &TimeSeries, period: usize) -> Result<DecompositionResult> {
        // Simplified STL implementation
        // In practice, this would involve iterative LOESS smoothing
        self.additive_decomposition(ts, period)
    }

    /// X-13ARIMA-SEATS decomposition (simplified implementation)
    fn x13_decomposition(&self, ts: &TimeSeries, period: usize) -> Result<DecompositionResult> {
        // Simplified X-13 implementation
        // In practice, this would use ARIMA modeling and sophisticated filters
        self.additive_decomposition(ts, period)
    }

    /// Extract trend component using moving average
    fn extract_trend(&self, ts: &TimeSeries, period: usize) -> Result<TimeSeries> {
        let window_size = if period % 2 == 0 { period } else { period };

        let mut trend_values = Vec::with_capacity(ts.len());

        for i in 0..ts.len() {
            let start = if i >= window_size / 2 {
                i - window_size / 2
            } else {
                0
            };
            let end = std::cmp::min(i + window_size / 2 + 1, ts.len());

            let window_values: Vec<f64> = (start..end)
                .filter_map(|idx| ts.values.get_f64(idx))
                .filter(|v| v.is_finite())
                .collect();

            if !window_values.is_empty() {
                let trend_val = window_values.iter().sum::<f64>() / window_values.len() as f64;
                trend_values.push(trend_val);
            } else {
                trend_values.push(f64::NAN);
            }
        }

        let trend_series = TimeSeriesData::from_vec(trend_values);
        TimeSeries::new(ts.index.clone(), trend_series)
    }

    /// Extract seasonal component (additive)
    fn extract_seasonal_additive(
        &self,
        detrended: &TimeSeries,
        period: usize,
    ) -> Result<TimeSeries> {
        let mut seasonal_pattern = vec![0.0; period];
        let mut counts = vec![0; period];

        // Calculate average for each seasonal position
        for i in 0..detrended.len() {
            if let Some(val) = detrended.values.get_f64(i) {
                if val.is_finite() {
                    let season_idx = i % period;
                    seasonal_pattern[season_idx] += val;
                    counts[season_idx] += 1;
                }
            }
        }

        // Average the seasonal components
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f64;
            }
        }

        // Ensure seasonal component sums to zero
        let mean_seasonal = seasonal_pattern.iter().sum::<f64>() / period as f64;
        for val in &mut seasonal_pattern {
            *val -= mean_seasonal;
        }

        // Repeat pattern for full series length
        let mut seasonal_values = Vec::with_capacity(detrended.len());
        for i in 0..detrended.len() {
            seasonal_values.push(seasonal_pattern[i % period]);
        }

        let seasonal_series = TimeSeriesData::from_vec(seasonal_values);
        TimeSeries::new(detrended.index.clone(), seasonal_series)
    }

    /// Extract seasonal component (multiplicative)
    fn extract_seasonal_multiplicative(
        &self,
        detrended: &TimeSeries,
        period: usize,
    ) -> Result<TimeSeries> {
        let mut seasonal_pattern = vec![1.0; period];
        let mut counts = vec![0; period];

        // Calculate geometric mean for each seasonal position
        for i in 0..detrended.len() {
            if let Some(val) = detrended.values.get_f64(i) {
                if val.is_finite() && val > 0.0 {
                    let season_idx = i % period;
                    seasonal_pattern[season_idx] *= val;
                    counts[season_idx] += 1;
                }
            }
        }

        // Calculate geometric mean
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_pattern[i] = seasonal_pattern[i].powf(1.0 / counts[i] as f64);
            }
        }

        // Normalize to sum to period (for multiplicative)
        let sum_seasonal: f64 = seasonal_pattern.iter().sum();
        if sum_seasonal > 0.0 {
            for val in &mut seasonal_pattern {
                *val = *val * period as f64 / sum_seasonal;
            }
        }

        // Repeat pattern for full series length
        let mut seasonal_values = Vec::with_capacity(detrended.len());
        for i in 0..detrended.len() {
            seasonal_values.push(seasonal_pattern[i % period]);
        }

        let seasonal_series = TimeSeriesData::from_vec(seasonal_values);
        TimeSeries::new(detrended.index.clone(), seasonal_series)
    }

    /// Subtract two time series
    fn subtract_series(&self, ts1: &TimeSeries, ts2: &TimeSeries) -> Result<TimeSeries> {
        if ts1.len() != ts2.len() {
            return Err(Error::DimensionMismatch(
                "Time series must have the same length".to_string(),
            ));
        }

        let mut result_values = Vec::with_capacity(ts1.len());

        for i in 0..ts1.len() {
            let val1 = ts1.values.get_f64(i).unwrap_or(f64::NAN);
            let val2 = ts2.values.get_f64(i).unwrap_or(f64::NAN);
            result_values.push(val1 - val2);
        }

        let result_series = TimeSeriesData::from_vec(result_values);
        TimeSeries::new(ts1.index.clone(), result_series)
    }

    /// Divide two time series
    fn divide_series(&self, ts1: &TimeSeries, ts2: &TimeSeries) -> Result<TimeSeries> {
        if ts1.len() != ts2.len() {
            return Err(Error::DimensionMismatch(
                "Time series must have the same length".to_string(),
            ));
        }

        let mut result_values = Vec::with_capacity(ts1.len());

        for i in 0..ts1.len() {
            let val1 = ts1.values.get_f64(i).unwrap_or(f64::NAN);
            let val2 = ts2.values.get_f64(i).unwrap_or(f64::NAN);

            if val2 != 0.0 && val2.is_finite() {
                result_values.push(val1 / val2);
            } else {
                result_values.push(f64::NAN);
            }
        }

        let result_series = TimeSeriesData::from_vec(result_values);
        TimeSeries::new(ts1.index.clone(), result_series)
    }

    /// Calculate decomposition metrics
    fn calculate_metrics(
        &self,
        original: &TimeSeries,
        trend: &TimeSeries,
        seasonal: &TimeSeries,
        residual: &TimeSeries,
    ) -> Result<DecompositionMetrics> {
        let original_var = self.calculate_variance(original)?;
        let trend_var = self.calculate_variance(trend)?;
        let seasonal_var = self.calculate_variance(seasonal)?;
        let residual_var = self.calculate_variance(residual)?;

        let total_explained_var = trend_var + seasonal_var;
        let signal_var = total_explained_var;
        let noise_var = residual_var;

        Ok(DecompositionMetrics {
            trend_variance_ratio: if original_var > 0.0 {
                trend_var / original_var
            } else {
                0.0
            },
            seasonal_variance_ratio: if original_var > 0.0 {
                seasonal_var / original_var
            } else {
                0.0
            },
            residual_variance_ratio: if original_var > 0.0 {
                residual_var / original_var
            } else {
                0.0
            },
            signal_to_noise_ratio: if noise_var > 0.0 {
                signal_var / noise_var
            } else {
                f64::INFINITY
            },
            seasonality_strength: if original_var > 0.0 {
                1.0 - (residual_var + trend_var) / original_var
            } else {
                0.0
            },
            trend_strength: if original_var > 0.0 {
                1.0 - (residual_var + seasonal_var) / original_var
            } else {
                0.0
            },
        })
    }

    /// Calculate variance of time series
    fn calculate_variance(&self, ts: &TimeSeries) -> Result<f64> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return Ok(0.0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        Ok(variance)
    }
}

impl DecompositionResult {
    /// Reconstruct the original series from components
    pub fn reconstruct(&self) -> Result<TimeSeries> {
        match self.method {
            DecompositionMethod::Additive | DecompositionMethod::STL | DecompositionMethod::X13 => {
                self.reconstruct_additive()
            }
            DecompositionMethod::Multiplicative => self.reconstruct_multiplicative(),
        }
    }

    /// Reconstruct additive decomposition
    fn reconstruct_additive(&self) -> Result<TimeSeries> {
        let mut reconstructed_values = Vec::with_capacity(self.original.len());

        for i in 0..self.original.len() {
            let trend_val = self.trend.values.get_f64(i).unwrap_or(0.0);
            let seasonal_val = self.seasonal.values.get_f64(i).unwrap_or(0.0);
            let residual_val = self.residual.values.get_f64(i).unwrap_or(0.0);

            reconstructed_values.push(trend_val + seasonal_val + residual_val);
        }

        let reconstructed_series = TimeSeriesData::from_vec(reconstructed_values);
        TimeSeries::new(self.original.index.clone(), reconstructed_series)
    }

    /// Reconstruct multiplicative decomposition
    fn reconstruct_multiplicative(&self) -> Result<TimeSeries> {
        let mut reconstructed_values = Vec::with_capacity(self.original.len());

        for i in 0..self.original.len() {
            let trend_val = self.trend.values.get_f64(i).unwrap_or(1.0);
            let seasonal_val = self.seasonal.values.get_f64(i).unwrap_or(1.0);
            let residual_val = self.residual.values.get_f64(i).unwrap_or(1.0);

            reconstructed_values.push(trend_val * seasonal_val * residual_val);
        }

        let reconstructed_series = TimeSeriesData::from_vec(reconstructed_values);
        TimeSeries::new(self.original.index.clone(), reconstructed_series)
    }

    /// Get seasonal indices for specific periods
    pub fn get_seasonal_indices(&self) -> HashMap<usize, f64> {
        let mut indices = HashMap::new();

        for i in 0..std::cmp::min(self.period, self.seasonal.len()) {
            if let Some(val) = self.seasonal.values.get_f64(i) {
                indices.insert(i, val);
            }
        }

        indices
    }

    /// Calculate decomposition quality score
    pub fn quality_score(&self) -> f64 {
        let trend_strength = self.metrics.trend_strength.max(0.0).min(1.0);
        let seasonal_strength = self.metrics.seasonality_strength.max(0.0).min(1.0);
        let explained_variance =
            self.metrics.trend_variance_ratio + self.metrics.seasonal_variance_ratio;

        // Weighted combination of various quality measures
        0.4 * explained_variance + 0.3 * trend_strength + 0.3 * seasonal_strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeries, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_test_seasonal_series() -> TimeSeries {
        // Create a time series with trend and seasonality
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap(); // Daily data
            let trend = i as f64 * 0.1; // Linear trend
            let seasonal = (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin() * 2.0; // Weekly seasonality
            let noise = 0.1 * (i as f64 % 3.0 - 1.0); // Small noise
            let value = 10.0 + trend + seasonal + noise;

            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_seasonal_decomposition() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).unwrap();

        assert_eq!(result.period, 7);
        assert_eq!(result.trend.len(), ts.len());
        assert_eq!(result.seasonal.len(), ts.len());
        assert_eq!(result.residual.len(), ts.len());

        // Check that decomposition explains most of the variance
        let total_explained =
            result.metrics.trend_variance_ratio + result.metrics.seasonal_variance_ratio;
        assert!(
            total_explained > 0.7,
            "Decomposition should explain most variance"
        );
    }

    #[test]
    fn test_multiplicative_decomposition() {
        let mut ts = create_test_seasonal_series();

        // Make all values positive for multiplicative decomposition
        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                ts.values = TimeSeriesData::from_vec(
                    (0..ts.len())
                        .map(|j| {
                            if j == i {
                                val.abs() + 1.0
                            } else {
                                ts.values.get_f64(j).unwrap_or(1.0)
                            }
                        })
                        .collect(),
                );
            }
        }

        let decomposer =
            SeasonalDecomposition::new(DecompositionMethod::Multiplicative).with_period(7);

        let result = decomposer.decompose(&ts).unwrap();
        assert_eq!(result.method, DecompositionMethod::Multiplicative);
    }

    #[test]
    fn test_decomposition_reconstruction() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).unwrap();
        let reconstructed = result.reconstruct().unwrap();

        // Check that reconstruction is close to original
        for i in 0..ts.len() {
            let original = ts.values.get_f64(i).unwrap();
            let reconstructed_val = reconstructed.values.get_f64(i).unwrap();
            let diff = (original - reconstructed_val).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction should be very close to original"
            );
        }
    }

    #[test]
    fn test_period_detection() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive);

        let result = decomposer.decompose(&ts).unwrap();

        // Should detect weekly seasonality (period = 7)
        assert_eq!(result.period, 7);
    }

    #[test]
    fn test_quality_metrics() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).unwrap();
        let quality = result.quality_score();

        assert!(
            quality > 0.5,
            "Quality score should be reasonable for synthetic data"
        );
        assert!(quality <= 1.0, "Quality score should not exceed 1.0");
    }
}

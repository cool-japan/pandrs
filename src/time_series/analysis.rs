//! Time Series Analysis Module
//!
//! This module provides various statistical analysis methods for time series data,
//! including trend analysis, seasonality detection, stationarity tests, and
//! autocorrelation analysis.

use crate::core::error::{Error, Result};
use crate::time_series::core::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend direction: "increasing", "decreasing", "no_trend"
    pub direction: String,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Slope of the trend line
    pub slope: f64,
    /// R-squared of trend fit
    pub r_squared: f64,
    /// Statistical significance of trend
    pub p_value: f64,
    /// Confidence interval for slope
    pub slope_confidence_interval: (f64, f64),
    /// Mann-Kendall tau statistic
    pub mann_kendall_tau: f64,
    /// Sen's slope estimator
    pub sens_slope: f64,
}

/// Seasonality analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysis {
    /// Whether seasonality is detected
    pub has_seasonality: bool,
    /// Dominant seasonal period
    pub dominant_period: Option<usize>,
    /// Seasonal strength (0.0 to 1.0)
    pub strength: f64,
    /// All detected periods with their strengths
    pub detected_periods: HashMap<usize, f64>,
    /// Seasonal indices for dominant period
    pub seasonal_indices: HashMap<usize, f64>,
    /// Peak frequency in spectrum
    pub peak_frequency: Option<f64>,
    /// Spectral density at peak
    pub peak_power: Option<f64>,
}

/// Stationarity test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTest {
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical values at different significance levels
    pub critical_values: HashMap<String, f64>,
    /// Whether series is stationary
    pub is_stationary: bool,
    /// Test type
    pub test_type: String,
    /// Number of lags used
    pub lags: Option<usize>,
    /// Trend component included
    pub trend: Option<String>,
}

/// Autocorrelation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationAnalysis {
    /// Autocorrelation function values
    pub acf: Vec<f64>,
    /// Partial autocorrelation function values
    pub pacf: Vec<f64>,
    /// Lags corresponding to ACF/PACF values
    pub lags: Vec<usize>,
    /// Ljung-Box test statistic
    pub ljung_box_statistic: f64,
    /// Ljung-Box test p-value
    pub ljung_box_p_value: f64,
    /// Whether residuals are white noise
    pub is_white_noise: bool,
    /// Confidence intervals for ACF
    pub acf_confidence_intervals: Vec<(f64, f64)>,
}

/// Change point detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePointDetection {
    /// Detected change points (indices)
    pub change_points: Vec<usize>,
    /// Change point scores
    pub scores: Vec<f64>,
    /// Detection method used
    pub method: String,
    /// Threshold used for detection
    pub threshold: f64,
    /// Statistical significance of change points
    pub significance_levels: Vec<f64>,
}

impl TrendAnalysis {
    /// Analyze trend in time series
    pub fn analyze(ts: &TimeSeries) -> Result<TrendAnalysis> {
        if ts.len() < 3 {
            return Err(Error::InvalidInput(
                "Time series must have at least 3 points for trend analysis".to_string(),
            ));
        }

        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 3 {
            return Err(Error::InvalidInput(
                "Insufficient valid data points".to_string(),
            ));
        }

        // Linear trend analysis
        let (slope, intercept, r_squared) = Self::linear_regression(&values)?;

        // Mann-Kendall test
        let (mann_kendall_tau, mk_p_value) = Self::mann_kendall_test(&values)?;

        // Sen's slope
        let sens_slope = Self::sens_slope(&values)?;

        // Determine trend direction and strength
        let direction = if slope > 0.0 && mk_p_value < 0.05 {
            "increasing"
        } else if slope < 0.0 && mk_p_value < 0.05 {
            "decreasing"
        } else {
            "no_trend"
        };

        let strength = r_squared.max(mann_kendall_tau.abs());

        // Calculate confidence interval for slope (simplified)
        let slope_std_error = Self::slope_standard_error(&values, slope, intercept)?;
        let t_critical = 1.96; // For 95% confidence
        let slope_ci = (
            slope - t_critical * slope_std_error,
            slope + t_critical * slope_std_error,
        );

        Ok(super::analysis::TrendAnalysis {
            direction: direction.to_string(),
            strength,
            slope,
            r_squared,
            p_value: mk_p_value,
            slope_confidence_interval: slope_ci,
            mann_kendall_tau,
            sens_slope,
        })
    }

    /// Perform linear regression
    fn linear_regression(values: &[f64]) -> Result<(f64, f64, f64)> {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(values).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();
        let sum_y2 = values.iter().map(|y| y * y).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot = values.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
        let ss_res = x_values
            .iter()
            .zip(values)
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum::<f64>();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Ok((slope, intercept, r_squared))
    }

    /// Mann-Kendall trend test
    fn mann_kendall_test(values: &[f64]) -> Result<(f64, f64)> {
        let n = values.len();
        let mut s = 0i32;

        for i in 0..n {
            for j in (i + 1)..n {
                s += if values[j] > values[i] {
                    1
                } else if values[j] < values[i] {
                    -1
                } else {
                    0
                };
            }
        }

        let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;
        let tau = s as f64 / ((n * (n - 1)) as f64 / 2.0);

        // Calculate z-score and p-value (simplified)
        let z = if s > 0 {
            (s as f64 - 1.0) / var_s.sqrt()
        } else if s < 0 {
            (s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };

        let p_value = 2.0 * (1.0 - Self::standard_normal_cdf(z.abs()));

        Ok((tau, p_value))
    }

    /// Sen's slope estimator
    fn sens_slope(values: &[f64]) -> Result<f64> {
        let n = values.len();
        let mut slopes = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if i != j {
                    slopes.push((values[j] - values[i]) / (j - i) as f64);
                }
            }
        }

        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Median slope
        let median_idx = slopes.len() / 2;
        let sen_slope = if slopes.len() % 2 == 0 {
            (slopes[median_idx - 1] + slopes[median_idx]) / 2.0
        } else {
            slopes[median_idx]
        };

        Ok(sen_slope)
    }

    /// Calculate standard error of slope
    fn slope_standard_error(values: &[f64], slope: f64, intercept: f64) -> Result<f64> {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        // Calculate residual sum of squares
        let rss = x_values
            .iter()
            .zip(values)
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum::<f64>();

        // Calculate sum of squared deviations of x
        let x_mean = x_values.iter().sum::<f64>() / n;
        let sxx = x_values.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();

        let slope_se = (rss / ((n - 2.0) * sxx)).sqrt();
        Ok(slope_se)
    }

    /// Standard normal CDF (simplified approximation)
    fn standard_normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

impl SeasonalityAnalysis {
    /// Analyze seasonality in time series
    pub fn analyze(ts: &TimeSeries, max_period: Option<usize>) -> Result<SeasonalityAnalysis> {
        if ts.len() < 6 {
            return Err(Error::InvalidInput(
                "Time series must have at least 6 points for seasonality analysis".to_string(),
            ));
        }

        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        // First detrend the data for better seasonality detection
        let detrended_values = Self::detrend_series(&values)?;

        let max_period = max_period.unwrap_or(std::cmp::min(values.len() / 3, 50));
        let mut detected_periods = HashMap::new();

        // Analyze potential periods from 2 to max_period
        for period in 2..=max_period {
            let strength = Self::calculate_seasonal_strength(&detrended_values, period)?;
            if strength > 0.05 {
                // Lower threshold for better detection
                detected_periods.insert(period, strength);
            }
        }

        // Find dominant period, avoiding harmonics
        let dominant_period = Self::find_fundamental_period(&detected_periods);

        let has_seasonality = !detected_periods.is_empty();
        let strength = detected_periods.values().cloned().fold(0.0, f64::max);

        // Calculate seasonal indices for dominant period using original values
        let seasonal_indices = if let Some(period) = dominant_period {
            Self::calculate_seasonal_indices(&values, period)?
        } else {
            HashMap::new()
        };

        // Power spectral density analysis using detrended data
        let (peak_frequency, peak_power) = Self::analyze_spectrum(&detrended_values)?;

        Ok(super::analysis::SeasonalityAnalysis {
            has_seasonality,
            dominant_period,
            strength,
            detected_periods,
            seasonal_indices,
            peak_frequency,
            peak_power,
        })
    }

    /// Calculate seasonal strength for a given period
    fn calculate_seasonal_strength(values: &[f64], period: usize) -> Result<f64> {
        if values.len() < period * 2 {
            return Ok(0.0);
        }

        // Calculate seasonal averages
        let mut seasonal_means = vec![0.0; period];
        let mut counts = vec![0; period];

        for (i, &value) in values.iter().enumerate() {
            let season_idx = i % period;
            seasonal_means[season_idx] += value;
            counts[season_idx] += 1;
        }

        // Average the seasonal components
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_means[i] /= counts[i] as f64;
            }
        }

        // Calculate residuals after removing seasonal pattern
        let mut residuals = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            let season_idx = i % period;
            let expected = seasonal_means[season_idx];
            residuals.push(value - expected);
        }

        // Calculate variance of original values
        let mean_orig = values.iter().sum::<f64>() / values.len() as f64;
        let var_orig =
            values.iter().map(|&v| (v - mean_orig).powi(2)).sum::<f64>() / values.len() as f64;

        // Calculate variance of residuals
        let mean_resid = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let var_resid = residuals
            .iter()
            .map(|&r| (r - mean_resid).powi(2))
            .sum::<f64>()
            / residuals.len() as f64;

        // Seasonal strength is explained variance ratio
        let strength = if var_orig > 0.0 {
            (var_orig - var_resid) / var_orig
        } else {
            0.0
        };

        Ok(strength.max(0.0).min(1.0))
    }

    /// Calculate seasonal indices
    fn calculate_seasonal_indices(values: &[f64], period: usize) -> Result<HashMap<usize, f64>> {
        let mut seasonal_sums = vec![0.0; period];
        let mut counts = vec![0; period];

        for (i, &value) in values.iter().enumerate() {
            let season_idx = i % period;
            seasonal_sums[season_idx] += value;
            counts[season_idx] += 1;
        }

        let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut indices = HashMap::new();

        for i in 0..period {
            if counts[i] > 0 {
                let seasonal_mean = seasonal_sums[i] / counts[i] as f64;
                let index = if overall_mean != 0.0 {
                    seasonal_mean / overall_mean
                } else {
                    1.0
                };
                indices.insert(i, index);
            }
        }

        Ok(indices)
    }

    /// Analyze power spectrum using autocorrelation-based periodogram
    fn analyze_spectrum(values: &[f64]) -> Result<(Option<f64>, Option<f64>)> {
        let n = values.len();
        if n < 4 {
            return Ok((None, None));
        }

        let max_lag = std::cmp::min(n / 3, 50);

        let mut autocorr = Vec::new();
        for lag in 0..=max_lag {
            let corr = Self::calculate_autocorrelation(values, lag)?;
            autocorr.push(corr);
        }

        // Find significant peaks in autocorrelation (excluding lag 0)
        let mut peaks = Vec::new();
        for lag in 2..autocorr.len() {
            let corr = autocorr[lag];

            // Check if this is a local maximum
            let is_peak = lag > 0
                && lag < autocorr.len() - 1
                && corr > autocorr[lag - 1]
                && corr > autocorr[lag + 1]
                && corr > 0.1; // Lower threshold for peak detection

            if is_peak {
                peaks.push((lag, corr));
            }
        }

        // Sort peaks by correlation strength
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Return the strongest peak
        if let Some((peak_lag, peak_corr)) = peaks.first() {
            let peak_frequency = Some(1.0 / *peak_lag as f64);
            let peak_power = Some(*peak_corr);
            Ok((peak_frequency, peak_power))
        } else {
            Ok((None, None))
        }
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(values: &[f64], lag: usize) -> Result<f64> {
        if lag >= values.len() {
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

        for &val in values {
            let dev = val - mean;
            denominator += dev * dev;
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Detrend time series using linear regression
    fn detrend_series(values: &[f64]) -> Result<Vec<f64>> {
        let n = values.len();
        if n < 2 {
            return Ok(values.to_vec());
        }

        // Calculate linear trend parameters
        let x_values: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_xx = x_values.iter().map(|x| x * x).sum::<f64>();

        let n_f64 = n as f64;
        let slope = (n_f64 * sum_xy - sum_x * sum_y) / (n_f64 * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n_f64;

        // Remove trend
        let detrended: Vec<f64> = x_values
            .iter()
            .zip(values.iter())
            .map(|(x, y)| y - (slope * x + intercept))
            .collect();

        Ok(detrended)
    }

    /// Find fundamental period by avoiding harmonics
    fn find_fundamental_period(detected_periods: &HashMap<usize, f64>) -> Option<usize> {
        if detected_periods.is_empty() {
            return None;
        }

        // Sort periods by strength
        let mut periods: Vec<(usize, f64)> = detected_periods
            .iter()
            .map(|(&period, &strength)| (period, strength))
            .collect();
        periods.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find the fundamental period (smallest period that explains the seasonality)
        for &(candidate_period, candidate_strength) in &periods {
            // Check if this is likely a fundamental period
            let mut is_fundamental = true;

            // Check if any smaller period is a divisor (potential fundamental)
            for &(smaller_period, smaller_strength) in &periods {
                if smaller_period < candidate_period
                    && candidate_period % smaller_period == 0
                    && smaller_strength >= candidate_strength * 0.7
                {
                    // This candidate is likely a harmonic
                    is_fundamental = false;
                    break;
                }
            }

            if is_fundamental {
                return Some(candidate_period);
            }
        }

        // If no fundamental found, return the strongest period
        periods.first().map(|(period, _)| *period)
    }
}

impl StationarityTest {
    /// Augmented Dickey-Fuller test
    pub fn augmented_dickey_fuller(
        ts: &TimeSeries,
        lags: Option<usize>,
    ) -> Result<StationarityTest> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Time series must have at least 10 points for ADF test".to_string(),
            ));
        }

        let lags = lags.unwrap_or(((values.len() as f64).cbrt() * 12.0 / 100.0) as usize);

        // Create differenced series
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut delta_y = Vec::new();

        for i in 1..values.len() {
            delta_y.push(values[i] - values[i - 1]);
        }

        for i in lags..delta_y.len() {
            y.push(delta_y[i]);
            x.push(values[i]); // lagged level

            // Add lagged differences
            for lag in 1..=lags {
                if i >= lag {
                    x.push(delta_y[i - lag]);
                }
            }
        }

        // Simplified ADF calculation (in practice would use regression)
        let test_statistic = Self::calculate_adf_statistic(&values, lags)?;

        // Critical values (MacKinnon, 1996)
        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), -3.43);
        critical_values.insert("5%".to_string(), -2.86);
        critical_values.insert("10%".to_string(), -2.57);

        let p_value = Self::calculate_adf_p_value(test_statistic)?;
        let is_stationary = test_statistic < critical_values["5%"];

        Ok(super::analysis::StationarityTest {
            test_statistic,
            p_value,
            critical_values,
            is_stationary,
            test_type: "Augmented Dickey-Fuller".to_string(),
            lags: Some(lags),
            trend: Some("constant".to_string()),
        })
    }

    /// KPSS test for stationarity
    pub fn kpss_test(ts: &TimeSeries, trend: &str) -> Result<super::analysis::StationarityTest> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Time series must have at least 10 points for KPSS test".to_string(),
            ));
        }

        // Detrend the series
        let detrended = match trend {
            "constant" => Self::detrend_constant(&values)?,
            "linear" => Self::detrend_linear(&values)?,
            _ => {
                return Err(Error::InvalidInput(
                    "Invalid trend specification".to_string(),
                ))
            }
        };

        // Calculate partial sums
        let mut partial_sums = vec![0.0; detrended.len()];
        partial_sums[0] = detrended[0];
        for i in 1..detrended.len() {
            partial_sums[i] = partial_sums[i - 1] + detrended[i];
        }

        // Calculate long-run variance
        let long_run_var = Self::calculate_long_run_variance(&detrended)?;

        // KPSS statistic
        let n = values.len() as f64;
        let sum_of_squares: f64 = partial_sums.iter().map(|x| x * x).sum();
        let test_statistic = sum_of_squares / (n * n * long_run_var);

        // Critical values for KPSS test
        let mut critical_values = HashMap::new();
        match trend {
            "constant" => {
                critical_values.insert("1%".to_string(), 0.739);
                critical_values.insert("5%".to_string(), 0.463);
                critical_values.insert("10%".to_string(), 0.347);
            }
            "linear" => {
                critical_values.insert("1%".to_string(), 0.216);
                critical_values.insert("5%".to_string(), 0.146);
                critical_values.insert("10%".to_string(), 0.119);
            }
            _ => {}
        }

        let p_value = Self::calculate_kpss_p_value(test_statistic, trend)?;
        let is_stationary = test_statistic < critical_values["5%"];

        Ok(super::analysis::StationarityTest {
            test_statistic,
            p_value,
            critical_values,
            is_stationary,
            test_type: "KPSS".to_string(),
            lags: None,
            trend: Some(trend.to_string()),
        })
    }

    /// Calculate ADF test statistic (simplified)
    fn calculate_adf_statistic(values: &[f64], lags: usize) -> Result<f64> {
        // Simplified calculation - in practice this would involve regression
        let mut diff_values = Vec::new();
        for i in 1..values.len() {
            diff_values.push(values[i] - values[i - 1]);
        }

        let mean_diff = diff_values.iter().sum::<f64>() / diff_values.len() as f64;
        let var_diff = diff_values
            .iter()
            .map(|x| (x - mean_diff).powi(2))
            .sum::<f64>()
            / diff_values.len() as f64;

        let std_diff = var_diff.sqrt();
        let t_stat = mean_diff / (std_diff / (diff_values.len() as f64).sqrt());

        Ok(t_stat)
    }

    /// Calculate ADF p-value (simplified)
    fn calculate_adf_p_value(test_statistic: f64) -> Result<f64> {
        // Simplified p-value calculation
        let p_value = if test_statistic < -3.43 {
            0.01
        } else if test_statistic < -2.86 {
            0.05
        } else if test_statistic < -2.57 {
            0.10
        } else {
            0.15
        };

        Ok(p_value)
    }

    /// Calculate KPSS p-value (simplified)
    fn calculate_kpss_p_value(test_statistic: f64, trend: &str) -> Result<f64> {
        let critical_1 = match trend {
            "constant" => 0.739,
            "linear" => 0.216,
            _ => 0.5,
        };

        let critical_5 = match trend {
            "constant" => 0.463,
            "linear" => 0.146,
            _ => 0.3,
        };

        let p_value = if test_statistic > critical_1 {
            0.01
        } else if test_statistic > critical_5 {
            0.05
        } else {
            0.10
        };

        Ok(p_value)
    }

    /// Detrend with constant
    fn detrend_constant(values: &[f64]) -> Result<Vec<f64>> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Ok(values.iter().map(|x| x - mean).collect())
    }

    /// Detrend with linear trend
    fn detrend_linear(values: &[f64]) -> Result<Vec<f64>> {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(values).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        let detrended: Vec<f64> = x_values
            .iter()
            .zip(values)
            .map(|(x, y)| y - (slope * x + intercept))
            .collect();

        Ok(detrended)
    }

    /// Calculate long-run variance for KPSS
    fn calculate_long_run_variance(residuals: &[f64]) -> Result<f64> {
        let n = residuals.len();
        let variance = residuals.iter().map(|x| x * x).sum::<f64>() / n as f64;

        // Simplified - should include autocovariances
        Ok(variance)
    }
}

impl AutocorrelationAnalysis {
    /// Compute autocorrelation and partial autocorrelation functions
    pub fn analyze(ts: &TimeSeries, max_lags: Option<usize>) -> Result<AutocorrelationAnalysis> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Time series must have at least 10 points for autocorrelation analysis".to_string(),
            ));
        }

        let max_lags = max_lags.unwrap_or(std::cmp::min(values.len() / 4, 40));

        // Calculate ACF
        let mut acf = Vec::new();
        let mut lags = Vec::new();

        for lag in 0..=max_lags {
            lags.push(lag);
            acf.push(Self::calculate_autocorrelation(&values, lag)?);
        }

        // Calculate PACF
        let pacf = Self::calculate_pacf(&values, max_lags)?;

        // Calculate confidence intervals
        let acf_confidence_intervals = Self::calculate_acf_confidence_intervals(&values, max_lags)?;

        // Ljung-Box test
        let (ljung_box_statistic, ljung_box_p_value) = Self::ljung_box_test(&values, max_lags)?;
        let is_white_noise = ljung_box_p_value > 0.05;

        Ok(super::analysis::AutocorrelationAnalysis {
            acf,
            pacf,
            lags,
            ljung_box_statistic,
            ljung_box_p_value,
            is_white_noise,
            acf_confidence_intervals,
        })
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(values: &[f64], lag: usize) -> Result<f64> {
        if lag >= values.len() {
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

        for &val in values {
            let dev = val - mean;
            denominator += dev * dev;
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate partial autocorrelation function
    fn calculate_pacf(values: &[f64], max_lags: usize) -> Result<Vec<f64>> {
        let mut pacf = vec![1.0]; // PACF at lag 0 is always 1

        if max_lags == 0 {
            return Ok(pacf);
        }

        // Calculate ACF first
        let mut acf = Vec::new();
        for lag in 0..=max_lags {
            acf.push(Self::calculate_autocorrelation(values, lag)?);
        }

        // Calculate PACF using Yule-Walker equations (simplified)
        for k in 1..=max_lags {
            if k == 1 {
                pacf.push(acf[1]);
            } else {
                // Solve Yule-Walker equations for partial autocorrelation
                let mut numerator = acf[k];
                let denominator = 1.0;

                for j in 1..k {
                    numerator -= pacf[j] * acf[k - j];
                }

                if denominator != 0.0 {
                    pacf.push(numerator / denominator);
                } else {
                    pacf.push(0.0);
                }
            }
        }

        Ok(pacf)
    }

    /// Calculate confidence intervals for ACF
    fn calculate_acf_confidence_intervals(
        values: &[f64],
        max_lags: usize,
    ) -> Result<Vec<(f64, f64)>> {
        let n = values.len() as f64;
        let mut intervals = Vec::new();

        for lag in 0..=max_lags {
            let se = if lag == 0 { 0.0 } else { (1.0 / n).sqrt() };

            let margin = 1.96 * se; // 95% confidence interval
            intervals.push((-margin, margin));
        }

        Ok(intervals)
    }

    /// Ljung-Box test for white noise
    fn ljung_box_test(values: &[f64], max_lags: usize) -> Result<(f64, f64)> {
        let n = values.len() as f64;
        let mut lb_statistic = 0.0;

        for lag in 1..=max_lags {
            let acf_lag = Self::calculate_autocorrelation(values, lag)?;
            lb_statistic += acf_lag * acf_lag / (n - lag as f64);
        }

        lb_statistic *= n * (n + 2.0);

        // P-value calculation (simplified)
        let p_value = if lb_statistic > 20.0 {
            0.01
        } else if lb_statistic > 15.0 {
            0.05
        } else {
            0.10
        };

        Ok((lb_statistic, p_value))
    }
}

impl ChangePointDetection {
    /// Detect change points using CUSUM method
    pub fn cusum_detection(
        ts: &TimeSeries,
        threshold: Option<f64>,
    ) -> Result<ChangePointDetection> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Time series must have at least 10 points for change point detection".to_string(),
            ));
        }

        let threshold = threshold.unwrap_or(2.0);
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        let mut cusum_pos = vec![0.0; values.len()];
        let mut cusum_neg = vec![0.0; values.len()];
        let mut scores = vec![0.0; values.len()];

        for i in 1..values.len() {
            cusum_pos[i] = (cusum_pos[i - 1] + (values[i] - mean)).max(0.0);
            cusum_neg[i] = (cusum_neg[i - 1] - (values[i] - mean)).max(0.0);
            scores[i] = cusum_pos[i].max(cusum_neg[i]);
        }

        // Detect change points
        let mut change_points = Vec::new();
        let mut significance_levels = Vec::new();

        for (i, &score) in scores.iter().enumerate() {
            if score > threshold {
                change_points.push(i);
                significance_levels.push(score / threshold);
            }
        }

        Ok(super::analysis::ChangePointDetection {
            change_points,
            scores,
            method: "CUSUM".to_string(),
            threshold,
            significance_levels,
        })
    }

    /// Detect change points using Bayesian change point detection (simplified)
    pub fn bayesian_detection(
        ts: &TimeSeries,
        prior_scale: Option<f64>,
    ) -> Result<super::analysis::ChangePointDetection> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Time series must have at least 10 points for change point detection".to_string(),
            ));
        }

        let prior_scale = prior_scale.unwrap_or(0.01);

        // Simplified Bayesian change point detection
        let mut scores = Vec::new();
        let mut change_points = Vec::new();
        let mut significance_levels = Vec::new();

        for i in 2..(values.len() - 2) {
            let before_mean = values[..i].iter().sum::<f64>() / i as f64;
            let after_mean = values[i..].iter().sum::<f64>() / (values.len() - i) as f64;

            let score = (before_mean - after_mean).abs();
            scores.push(score);

            if score > prior_scale * 10.0 {
                // Simplified threshold
                change_points.push(i);
                significance_levels.push(score / (prior_scale * 10.0));
            }
        }

        // Pad scores to match original length
        let mut full_scores = vec![0.0; values.len()];
        for (i, score) in scores.iter().enumerate() {
            full_scores[i + 2] = *score;
        }

        Ok(super::analysis::ChangePointDetection {
            change_points,
            scores: full_scores,
            method: "Bayesian".to_string(),
            threshold: prior_scale * 10.0,
            significance_levels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_trending_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let value = 10.0 + i as f64 * 0.2 + (i as f64 % 10.0 - 5.0) * 0.1; // Trend with noise
            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    fn create_seasonal_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let seasonal = (2.0 * PI * i as f64 / 7.0).sin() * 5.0; // Weekly seasonality
            let value = 20.0 + seasonal + (i as f64 % 3.0 - 1.0) * 0.5; // Seasonality with noise
            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_trend_analysis() {
        let ts = create_trending_series();
        let result = TrendAnalysis::analyze(&ts).unwrap();

        assert_eq!(result.direction, "increasing");
        assert!(result.slope > 0.0);
        assert!(result.strength > 0.5);
        assert!(result.r_squared > 0.8);
    }

    #[test]
    fn test_seasonality_analysis() {
        let ts = create_seasonal_series();
        let result = SeasonalityAnalysis::analyze(&ts, Some(20)).unwrap();

        assert!(result.has_seasonality);
        assert_eq!(result.dominant_period, Some(7)); // Should detect weekly pattern
        assert!(result.strength > 0.3);
        assert!(result.detected_periods.contains_key(&7));
    }

    #[test]
    fn test_stationarity_adf() {
        let ts = create_trending_series();
        let result = StationarityTest::augmented_dickey_fuller(&ts, None).unwrap();

        assert_eq!(result.test_type, "Augmented Dickey-Fuller");
        assert!(!result.is_stationary); // Trending series should not be stationary
        assert!(result.critical_values.contains_key("5%"));
    }

    #[test]
    fn test_stationarity_kpss() {
        let ts = create_seasonal_series();
        let result = StationarityTest::kpss_test(&ts, "constant").unwrap();

        assert_eq!(result.test_type, "KPSS");
        assert!(result.critical_values.contains_key("5%"));
    }

    #[test]
    fn test_autocorrelation_analysis() {
        let ts = create_seasonal_series();
        let result = AutocorrelationAnalysis::analyze(&ts, Some(20)).unwrap();

        assert_eq!(result.acf.len(), 21); // 0 to 20 lags
        assert_eq!(result.pacf.len(), 21);
        assert_eq!(result.lags.len(), 21);
        assert!(result.acf[0] == 1.0); // ACF at lag 0 should be 1
        assert!(result.pacf[0] == 1.0); // PACF at lag 0 should be 1
    }

    #[test]
    fn test_change_point_detection() {
        // Create series with a change point
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..50 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let value = if i < 25 { 10.0 } else { 20.0 }; // Clear change at position 25
            builder = builder.add_point(timestamp, value);
        }

        let ts = builder.frequency(Frequency::Daily).build().unwrap();
        let result = ChangePointDetection::cusum_detection(&ts, Some(1.0)).unwrap();

        assert_eq!(result.method, "CUSUM");
        assert!(!result.change_points.is_empty());
        // Should detect change point around position 25
        assert!(result.change_points.iter().any(|&cp| cp >= 20 && cp <= 30));
    }
}

//! Time Series Analysis Module
//!
//! This module provides various statistical analysis methods for time series data,
//! including trend analysis, seasonality detection, stationarity tests, and
//! autocorrelation analysis.

use crate::core::error::{Error, Result};
use crate::stats::special::{chi2_sf, ln_gamma, normal_sf, student_t_ppf};
use crate::time_series::core::TimeSeries;
use crate::time_series::stats::{
    kpss_critical_values, kpss_p_value_from_table, newey_west_long_run_variance,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(test)]
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

        // 95% confidence interval for the OLS slope. The critical value is the
        // Student-t quantile at `n − 2` degrees of freedom, not the normal
        // `1.96`: with `n = 5` the correct multiplier is 3.182, so the interval
        // this used to report was 38% too narrow.
        let slope_std_error = Self::slope_standard_error(&values, slope, intercept)?;
        let t_critical = student_t_ppf(0.975, (values.len() as f64) - 2.0);
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
        let _sum_y2 = values.iter().map(|y| y * y).sum::<f64>();

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

        // Two-sided normal tail from `crate::stats::special`, the crate's
        // single special-function module. Computing it as `1 − Φ(|z|)` with a
        // local 7-digit `erf` approximation (the previous behaviour) both
        // duplicated the approximation and cancelled catastrophically in the
        // tail; `normal_sf` is the survival function directly.
        let p_value = (2.0 * normal_sf(z.abs())).clamp(0.0, 1.0);

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

        slopes.sort_by(|a, b| a.total_cmp(b));

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
        peaks.sort_by(|a, b| b.1.total_cmp(&a.1));

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
        periods.sort_by(|a, b| b.1.total_cmp(&a.1));

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

        // Run the actual ADF regression and take the t-statistic on the lagged
        // level coefficient (see `calculate_adf_statistic`).
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

        // Critical values first, so an invalid trend specification is rejected
        // before any arithmetic (shared table with `time_series::stats`).
        let (c1, c5, c10) = kpss_critical_values(trend)?;

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

        // Long-run variance via the shared Bartlett-kernel Newey-West
        // estimator with the Schwert lag rule `l = ⌊4·(n/100)^{1/4}⌋`. The
        // local helper this replaced returned the plain contemporaneous
        // variance `Σê²/n` — the *short*-run variance — so the denominator
        // ignored exactly the serial correlation the KPSS statistic exists to
        // correct for, inflating the statistic for any persistent series and
        // rejecting stationarity far too often.
        let n_obs = detrended.len();
        let bandwidth = (4.0 * (n_obs as f64 / 100.0).powf(0.25)).floor() as usize;
        let long_run_var = newey_west_long_run_variance(&detrended, bandwidth);

        // KPSS statistic
        let n = values.len() as f64;
        let sum_of_squares: f64 = partial_sums.iter().map(|x| x * x).sum();
        let test_statistic = if long_run_var > 0.0 {
            sum_of_squares / (n * n * long_run_var)
        } else {
            f64::NAN
        };

        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), c1);
        critical_values.insert("5%".to_string(), c5);
        critical_values.insert("10%".to_string(), c10);

        // Interpolated table p-value (shared with `time_series::stats`),
        // replacing a three-level `0.01 / 0.05 / 0.10` step ladder that could
        // never report anything in between.
        let p_value = if test_statistic.is_finite() {
            kpss_p_value_from_table(test_statistic, c10, c5, c1)
        } else {
            f64::NAN
        };
        let is_stationary = test_statistic < c5;

        Ok(super::analysis::StationarityTest {
            test_statistic,
            p_value,
            critical_values,
            is_stationary,
            test_type: "KPSS".to_string(),
            lags: Some(bandwidth),
            trend: Some(trend.to_string()),
        })
    }

    /// Calculate the Augmented Dickey-Fuller test statistic.
    ///
    /// Runs the OLS regression
    ///   Δyₜ = α + β·yₜ₋₁ + Σⱼ γⱼ·Δyₜ₋ⱼ + εₜ   (constant, no trend)
    /// and returns the t-statistic on the lagged-level coefficient β. Under the
    /// unit-root null β = 0; a stationary series gives β < 0 and a strongly
    /// negative t-statistic. This is the genuine ADF regression — the previous
    /// implementation built the design matrix and then discarded it, returning a
    /// one-sample t-test of the differenced mean instead.
    fn calculate_adf_statistic(values: &[f64], lags: usize) -> Result<f64> {
        let n = values.len();

        // First differences Δyₜ; `dy[k] = values[k+1] - values[k]`.
        let dy: Vec<f64> = (1..n).map(|i| values[i] - values[i - 1]).collect();

        // Build the regression design matrix. For each usable time index `t`
        // (the earliest is `lags + 1`, so all lagged differences exist), the
        // row is [1, yₜ₋₁, Δyₜ₋₁, …, Δyₜ₋ₗ] and the response is Δyₜ.
        let n_regressors = 2 + lags; // constant + lagged level + `lags` diffs
        let mut x_rows: Vec<Vec<f64>> = Vec::new();
        let mut response: Vec<f64> = Vec::new();

        for t in (lags + 1)..n {
            let mut row = Vec::with_capacity(n_regressors);
            row.push(1.0); // constant
            row.push(values[t - 1]); // lagged level yₜ₋₁
            for j in 1..=lags {
                row.push(dy[t - 1 - j]); // Δyₜ₋ⱼ
            }
            x_rows.push(row);
            response.push(dy[t - 1]); // Δyₜ
        }

        if x_rows.len() <= n_regressors {
            return Err(Error::InvalidInput(
                "Insufficient observations to estimate the ADF regression".to_string(),
            ));
        }

        let (coefficients, std_errors) = ols_with_std_errors(&x_rows, &response)
            .ok_or_else(|| Error::InvalidInput("ADF regression matrix is singular".to_string()))?;

        // The lagged level is the second regressor (index 1).
        let beta = coefficients[1];
        let se = std_errors[1];
        if !se.is_finite() || se <= 0.0 {
            return Err(Error::InvalidInput(
                "ADF regression produced a degenerate standard error".to_string(),
            ));
        }

        Ok(beta / se)
    }

    /// Approximate ADF p-value.
    ///
    /// The Dickey-Fuller statistic does not follow a standard distribution and
    /// has no elementary closed form, so this returns an **approximate** p-value
    /// by monotone interpolation of the MacKinnon constant-only critical-value
    /// surface (1% = −3.43, 5% = −2.86, 10% = −2.57). It is intended for
    /// reporting significance bands, not as an exact tail probability; the
    /// stationarity verdict itself uses the tabulated critical values directly.
    fn calculate_adf_p_value(test_statistic: f64) -> Result<f64> {
        let anchors = [(-3.43_f64, 0.01_f64), (-2.86, 0.05), (-2.57, 0.10)];

        let p = if test_statistic <= anchors[0].0 {
            let slope = (anchors[1].1 - anchors[0].1) / (anchors[1].0 - anchors[0].0);
            (anchors[0].1 + slope * (test_statistic - anchors[0].0)).clamp(0.0001, 0.01)
        } else if test_statistic <= anchors[1].0 {
            let t = (test_statistic - anchors[0].0) / (anchors[1].0 - anchors[0].0);
            anchors[0].1 + t * (anchors[1].1 - anchors[0].1)
        } else if test_statistic <= anchors[2].0 {
            let t = (test_statistic - anchors[1].0) / (anchors[2].0 - anchors[1].0);
            anchors[1].1 + t * (anchors[2].1 - anchors[1].1)
        } else {
            let slope = (anchors[2].1 - anchors[1].1) / (anchors[2].0 - anchors[1].0);
            (anchors[2].1 + slope * (test_statistic - anchors[2].0)).clamp(0.10, 0.999)
        };

        Ok(p)
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

        // Calculate confidence intervals (Bartlett bands, built from the ACF
        // just computed)
        let acf_confidence_intervals = Self::calculate_acf_confidence_intervals(&acf, &values)?;

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

        // Durbin-Levinson recursion. `phi` holds the AR coefficients of the
        // order-(k-1) fit; `v` is the running prediction-error variance ratio.
        // The partial autocorrelation at lag k is the reflection coefficient
        // phi_kk, and `v` is updated as v_k = v_{k-1} * (1 - phi_kk^2). This is
        // the correct recursion — the previous code held the denominator at the
        // constant 1.0, giving wrong PACF values for every lag >= 2.
        let mut phi = vec![0.0_f64; max_lags + 1];
        let mut v = 1.0_f64;

        for k in 1..=max_lags {
            let mut numerator = acf[k];
            for j in 1..k {
                numerator -= phi[j] * acf[k - j];
            }

            let phi_kk = if v.abs() > 1e-12 { numerator / v } else { 0.0 };
            pacf.push(phi_kk);

            // Update the AR coefficients for order k from those of order k-1.
            let prev: Vec<f64> = phi[1..k].to_vec();
            phi[k] = phi_kk;
            for j in 1..k {
                phi[j] = prev[j - 1] - phi_kk * prev[k - 1 - j];
            }

            v *= 1.0 - phi_kk * phi_kk;
        }

        Ok(pacf)
    }

    /// 95% confidence bands for the ACF using **Bartlett's formula**.
    ///
    /// Under the null that the process is MA(k−1) — i.e. that all
    /// autocorrelations beyond lag `k−1` vanish — the large-sample standard
    /// error of `r_k` is
    ///
    /// ```text
    /// se(r_k) = √( (1 + 2·Σ_{j=1}^{k−1} r_j²) / n )
    /// ```
    ///
    /// so the bands widen as earlier lags show correlation. The flat
    /// `±1.96/√n` bands this replaced are Bartlett's formula specialized to
    /// *white noise* (every `r_j = 0`); applied to a series that is visibly
    /// autocorrelated at short lags they are far too narrow at long lags and
    /// flag spurious significance.
    ///
    /// `acf[0] = r_0 = 1` has no sampling error, so its band is `(0, 0)`.
    fn calculate_acf_confidence_intervals(acf: &[f64], values: &[f64]) -> Result<Vec<(f64, f64)>> {
        let n = values.len() as f64;
        let z = 1.959_963_984_540_054_f64; // Φ⁻¹(0.975)
        let mut intervals = Vec::with_capacity(acf.len());

        let mut running = 0.0; // Σ_{j=1}^{k−1} r_j²
        for (lag, &r) in acf.iter().enumerate() {
            if lag == 0 {
                intervals.push((0.0, 0.0));
            } else {
                let se = ((1.0 + 2.0 * running) / n).sqrt();
                let margin = z * se;
                intervals.push((-margin, margin));
                running += r * r;
            }
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

        // Under H0 (white noise) the Ljung-Box statistic is asymptotically
        // chi-squared with `max_lags` degrees of freedom.
        let p_value = chi2_sf(lb_statistic, max_lags as f64);

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

    /// Detect change points with **Bayesian Online Changepoint Detection**
    /// (Adams & MacKay, 2007).
    ///
    /// `prior_scale` is the constant changepoint **hazard rate** — the prior
    /// probability that any given observation begins a new regime; the default
    /// `0.01` corresponds to an expected run length of 100 observations. It
    /// must lie strictly between 0 and 1.
    ///
    /// `scores[t]` is the posterior probability `P(r_t = 0 | x_{1:t})` that the
    /// run length collapsed at `t`; `change_points` lists the indices where
    /// that probability exceeds `0.5`, and `significance_levels` carries the
    /// probabilities themselves. See
    /// `bocpd_changepoint_probabilities` for the model and for what the
    /// previous non-Bayesian implementation actually computed.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for series shorter than 10 points or a
    /// `prior_scale` outside `(0, 1)`.
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

        let hazard = prior_scale.unwrap_or(0.01);
        if !(hazard > 0.0 && hazard < 1.0) {
            return Err(Error::InvalidInput(format!(
                "prior_scale is the constant changepoint hazard rate and must lie strictly \
                 between 0 and 1, got {hazard}"
            )));
        }

        let scores = bocpd_changepoint_probabilities(&values, hazard);

        // A run-length-zero posterior above 0.5 means the model believes, on
        // balance, that this observation started a new regime.
        const THRESHOLD: f64 = 0.5;
        let mut change_points = Vec::new();
        let mut significance_levels = Vec::new();
        for (i, &score) in scores.iter().enumerate() {
            if i > 0 && score > THRESHOLD {
                change_points.push(i);
                significance_levels.push(score);
            }
        }

        Ok(super::analysis::ChangePointDetection {
            change_points,
            scores,
            method: "BOCPD (Adams & MacKay 2007, Normal-Inverse-Gamma)".to_string(),
            threshold: THRESHOLD,
            significance_levels,
        })
    }
}

/// Bayesian Online Changepoint Detection (Adams & MacKay, 2007) with a
/// Normal-Inverse-Gamma conjugate prior on the (unknown mean, unknown variance)
/// Gaussian observation model and a constant hazard rate.
///
/// Returns, for each index `t`, the posterior probability that a changepoint
/// occurred at `t`, evaluated with **one step of lookahead**:
/// `P(r_{t+1} = 1 | x_{1:t+1})` — the probability that, having also seen
/// `x_{t+1}`, the current run is exactly one observation old and therefore
/// began at `t`.
///
/// The purely filtered quantity `P(r_t = 0 | x_{1:t})` is *not* usable as a
/// detector: at time `t` every run-length hypothesis is multiplied by the same
/// predictive `π(x_t | r)`, so the changepoint and growth branches differ only
/// by the hazard ratio and `P(r_t = 0)` never rises much above `H` however
/// dramatic the shift. One observation later the hypothesis "the run started at
/// `t`" has already absorbed `x_t` into its posterior and predicts `x_{t+1}`
/// far better than every older hypothesis, so the mass concentrates. Index `0`
/// scores `0.0` — the series has to start somewhere, and that is not a detected
/// change — and the final index falls back to its filtered `P(r_t = 0)`.
///
/// The whole recursion runs in log-space with log-sum-exp normalization, so it
/// stays numerically stable for long series where the joint run-length
/// probabilities underflow.
///
/// This replaces a "Bayesian detection" that was `|mean(x[..i]) − mean(x[i..])|
/// > prior_scale · 10`: an absolute mean-shift threshold with no prior, no
/// likelihood, no posterior and no scale invariance (its default threshold of
/// `0.1` fired on every point of any series measured in units bigger than a
/// tenth). Nothing about it was Bayesian.
///
/// ### Model
/// Within a run, `xₜ ~ N(μ, σ²)` with `(μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀)`, whose
/// posterior predictive is a Student-t:
///
/// ```text
/// xₜ | run of length r  ~  t_{2α}( μ, β(κ+1) / (α κ) )
/// ```
///
/// The prior is set empirically from the series (`μ₀ = mean`, `β₀` from its
/// variance) with weak counts `κ₀ = 1`, `α₀ = 1`, which makes the detector
/// scale-invariant.
fn bocpd_changepoint_probabilities(values: &[f64], hazard: f64) -> Vec<f64> {
    let n = values.len();
    let nf = n as f64;

    // Empirical, weakly-informative NIG prior.
    let mean0 = values.iter().sum::<f64>() / nf;
    let var0 = values.iter().map(|v| (v - mean0).powi(2)).sum::<f64>() / nf;
    let kappa0 = 1.0_f64;
    let alpha0 = 1.0_f64;
    let beta0 = if var0 > 0.0 { var0 } else { 1.0 };

    // Sufficient statistics indexed by run length: entry `r` describes the
    // hypothesis "the current run has length r".
    let mut mu = vec![mean0];
    let mut kappa = vec![kappa0];
    let mut alpha = vec![alpha0];
    let mut beta = vec![beta0];

    // log P(r_t = r, x_{1:t}); starts as the point mass at r = 0.
    let mut log_joint = vec![0.0_f64];

    let log_hazard = hazard.ln();
    let log_survive = (1.0 - hazard).ln();

    // Filtered posteriors we need for the lag-1 smoothed detector:
    // `p_reset[t] = P(r_t = 0 | x_{1:t})` and `p_len1[t] = P(r_t = 1 | x_{1:t})`.
    let mut p_reset = Vec::with_capacity(n);
    let mut p_len1 = Vec::with_capacity(n);

    for &x in values {
        let run_count = log_joint.len();

        // Predictive log-likelihood of x under each run-length hypothesis.
        let mut log_predictive = Vec::with_capacity(run_count);
        for r in 0..run_count {
            let df = 2.0 * alpha[r];
            let scale_sq = beta[r] * (kappa[r] + 1.0) / (alpha[r] * kappa[r]);
            log_predictive.push(student_t_log_pdf(x, mu[r], scale_sq, df));
        }

        // Growth: the run continues (r -> r+1); Changepoint: it resets to 0.
        let mut new_log_joint = vec![f64::NEG_INFINITY; run_count + 1];
        let mut log_reset_terms = Vec::with_capacity(run_count);
        for r in 0..run_count {
            new_log_joint[r + 1] = log_joint[r] + log_predictive[r] + log_survive;
            log_reset_terms.push(log_joint[r] + log_predictive[r] + log_hazard);
        }
        new_log_joint[0] = log_sum_exp(&log_reset_terms);

        // Normalize to a posterior over run lengths.
        let log_evidence = log_sum_exp(&new_log_joint);
        for value in new_log_joint.iter_mut() {
            *value -= log_evidence;
        }

        p_reset.push(new_log_joint[0].exp());
        p_len1.push(new_log_joint.get(1).map_or(0.0, |v| v.exp()));

        // Conjugate NIG updates, shifted by one because hypothesis `r+1` at the
        // next step descends from hypothesis `r` at this one.
        let mut new_mu = vec![mean0];
        let mut new_kappa = vec![kappa0];
        let mut new_alpha = vec![alpha0];
        let mut new_beta = vec![beta0];
        for r in 0..run_count {
            let k = kappa[r];
            let m = mu[r];
            new_mu.push((k * m + x) / (k + 1.0));
            new_kappa.push(k + 1.0);
            new_alpha.push(alpha[r] + 0.5);
            new_beta.push(beta[r] + k * (x - m) * (x - m) / (2.0 * (k + 1.0)));
        }

        mu = new_mu;
        kappa = new_kappa;
        alpha = new_alpha;
        beta = new_beta;
        log_joint = new_log_joint;
    }

    (0..n)
        .map(|t| {
            if t == 0 {
                0.0
            } else if t + 1 < n {
                p_len1[t + 1]
            } else {
                p_reset[t]
            }
        })
        .collect()
}

/// Log density of a Student-t with `df` degrees of freedom, location `loc` and
/// **squared** scale `scale_sq`.
fn student_t_log_pdf(x: f64, loc: f64, scale_sq: f64, df: f64) -> f64 {
    if !(scale_sq > 0.0) || !(df > 0.0) {
        return f64::NEG_INFINITY;
    }
    let z_sq = (x - loc) * (x - loc) / scale_sq;
    ln_gamma((df + 1.0) / 2.0)
        - ln_gamma(df / 2.0)
        - 0.5 * (df * std::f64::consts::PI * scale_sq).ln()
        - (df + 1.0) / 2.0 * (1.0 + z_sq / df).ln()
}

/// Numerically stable `ln Σ exp(xᵢ)`.
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Ordinary least squares via the normal equations.
///
/// Solves `y = X·b` and returns `(coefficients, standard_errors)`, where the
/// standard errors come from `σ̂² · diag((XᵀX)⁻¹)` with `σ̂² = RSS / (n − k)`.
/// Returns `None` when the system is under-determined (`n ≤ k`) or `XᵀX` is
/// singular. Used by the Augmented Dickey-Fuller regression.
pub(crate) fn ols_with_std_errors(x: &[Vec<f64>], y: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = x.len();
    if n == 0 || y.len() != n {
        return None;
    }
    let k = x[0].len();
    if n <= k {
        return None;
    }

    // Normal equations: XᵀX (k×k) and Xᵀy (k).
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];
    for (row, &yi) in x.iter().zip(y.iter()) {
        for a in 0..k {
            xty[a] += row[a] * yi;
            for b in 0..k {
                xtx[a][b] += row[a] * row[b];
            }
        }
    }

    let inv = invert_matrix(&xtx)?;

    // b = (XᵀX)⁻¹ Xᵀy.
    let mut beta = vec![0.0_f64; k];
    for a in 0..k {
        for c in 0..k {
            beta[a] += inv[a][c] * xty[c];
        }
    }

    // Residual sum of squares.
    let mut rss = 0.0_f64;
    for (row, &yi) in x.iter().zip(y.iter()) {
        let mut pred = 0.0;
        for a in 0..k {
            pred += row[a] * beta[a];
        }
        let e = yi - pred;
        rss += e * e;
    }

    let dof = (n - k) as f64;
    let sigma2 = rss / dof;

    let mut se = vec![0.0_f64; k];
    for a in 0..k {
        let var = sigma2 * inv[a][a];
        se[a] = if var > 0.0 { var.sqrt() } else { f64::NAN };
    }

    Some((beta, se))
}

/// Invert a square matrix by Gauss-Jordan elimination with partial pivoting.
/// Returns `None` if the matrix is (numerically) singular.
fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut inv = vec![vec![0.0_f64; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: largest magnitude in this column.
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..n {
            let v = a[r][col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);

        let diag = a[col][col];
        for j in 0..n {
            a[col][j] /= diag;
            inv[col][j] /= diag;
        }

        for r in 0..n {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    for j in 0..n {
                        a[r][j] -= factor * a[col][j];
                        inv[r][j] -= factor * inv[col][j];
                    }
                }
            }
        }
    }

    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_trending_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc
                .timestamp_opt(1640995200 + i * 86400, 0)
                .single()
                .expect("operation should succeed");
            let value = 10.0 + i as f64 * 0.2 + (i as f64 % 10.0 - 5.0) * 0.1; // Trend with noise
            builder = builder.add_point(timestamp, value);
        }

        builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed")
    }

    fn create_seasonal_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc
                .timestamp_opt(1640995200 + i * 86400, 0)
                .single()
                .expect("operation should succeed");
            let seasonal = (2.0 * PI * i as f64 / 7.0).sin() * 5.0; // Weekly seasonality
            let value = 20.0 + seasonal + (i as f64 % 3.0 - 1.0) * 0.5; // Seasonality with noise
            builder = builder.add_point(timestamp, value);
        }

        builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_trend_analysis() {
        let ts = create_trending_series();
        let result = TrendAnalysis::analyze(&ts).expect("operation should succeed");

        assert_eq!(result.direction, "increasing");
        assert!(result.slope > 0.0);
        assert!(result.strength > 0.5);
        assert!(result.r_squared > 0.8);
    }

    #[test]
    fn test_seasonality_analysis() {
        let ts = create_seasonal_series();
        let result = SeasonalityAnalysis::analyze(&ts, Some(20)).expect("operation should succeed");

        assert!(result.has_seasonality);
        assert_eq!(result.dominant_period, Some(7)); // Should detect weekly pattern
        assert!(result.strength > 0.3);
        assert!(result.detected_periods.contains_key(&7));
    }

    #[test]
    fn test_stationarity_adf() {
        let ts = create_trending_series();
        let result =
            StationarityTest::augmented_dickey_fuller(&ts, None).expect("operation should succeed");

        assert_eq!(result.test_type, "Augmented Dickey-Fuller");
        assert!(!result.is_stationary); // Trending series should not be stationary
        assert!(result.critical_values.contains_key("5%"));
    }

    #[test]
    fn test_stationarity_kpss() {
        let ts = create_seasonal_series();
        let result =
            StationarityTest::kpss_test(&ts, "constant").expect("operation should succeed");

        assert_eq!(result.test_type, "KPSS");
        assert!(result.critical_values.contains_key("5%"));
    }

    #[test]
    fn test_autocorrelation_analysis() {
        let ts = create_seasonal_series();
        let result =
            AutocorrelationAnalysis::analyze(&ts, Some(20)).expect("operation should succeed");

        assert_eq!(result.acf.len(), 21); // 0 to 20 lags
        assert_eq!(result.pacf.len(), 21);
        assert_eq!(result.lags.len(), 21);
        assert!(result.acf[0] == 1.0); // ACF at lag 0 should be 1
        assert!(result.pacf[0] == 1.0); // PACF at lag 0 should be 1
    }

    #[test]
    fn test_pacf_durbin_levinson_lag2_closed_form() {
        // The Durbin-Levinson recursion must reproduce the known closed form for
        // the lag-2 partial autocorrelation,
        //   φ₂₂ = (r₂ − r₁²) / (1 − r₁²),
        // which the previous (denominator == 1.0) implementation got wrong.
        let ts = create_seasonal_series();
        let result =
            AutocorrelationAnalysis::analyze(&ts, Some(6)).expect("operation should succeed");

        let r1 = result.acf[1];
        let r2 = result.acf[2];
        let expected = (r2 - r1 * r1) / (1.0 - r1 * r1);

        assert!(
            (result.pacf[1] - r1).abs() < 1e-12,
            "PACF lag 1 must equal ACF lag 1"
        );
        assert!(
            (result.pacf[2] - expected).abs() < 1e-9,
            "PACF lag 2 ({}) should match the closed form ({})",
            result.pacf[2],
            expected
        );
    }

    #[test]
    fn test_change_point_detection() {
        // Create series with a change point
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..50 {
            let timestamp = Utc
                .timestamp_opt(1640995200 + i * 86400, 0)
                .single()
                .expect("operation should succeed");
            let value = if i < 25 { 10.0 } else { 20.0 }; // Clear change at position 25
            builder = builder.add_point(timestamp, value);
        }

        let ts = builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed");
        let result = ChangePointDetection::cusum_detection(&ts, Some(1.0))
            .expect("operation should succeed");

        assert_eq!(result.method, "CUSUM");
        assert!(!result.change_points.is_empty());
        // Should detect change point around position 25
        assert!(result.change_points.iter().any(|&cp| cp >= 20 && cp <= 30));
    }
}

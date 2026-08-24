//! Seasonal Decomposition Module
//!
//! This module provides seasonal decomposition methods for time series analysis,
//! including additive and multiplicative decomposition, trend extraction,
//! and seasonal pattern analysis.

use crate::core::error::{Error, Result};
use crate::time_series::core::{TimeSeries, TimeSeriesData};
use crate::time_series::stats::inv_normal_cdf;
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
    robust: bool,
}

impl SeasonalDecomposition {
    /// Create a new seasonal decomposition
    pub fn new(method: DecompositionMethod) -> Self {
        Self {
            method,
            period: None,
            extrapolate_trend: 0,
            robust: false,
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

    /// Enable STL's **outer robustness loop**.
    ///
    /// With `robust = true` the decomposition runs Cleveland et al.'s robust
    /// configuration (one inner pass per outer iteration, fifteen outer
    /// iterations), recomputing bisquare weights from the remainder each time so
    /// that outliers land in the remainder instead of bending the trend and
    /// seasonal components towards them. It costs proportionally more work and
    /// only affects [`DecompositionMethod::STL`]; the classical additive and
    /// multiplicative decompositions have no robustness loop to enable.
    pub fn with_robust(mut self, robust: bool) -> Self {
        self.robust = robust;
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

    /// Detect the seasonal period as the lag with the strongest positive
    /// autocorrelation, provided it clears the white-noise significance band.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when the series is too short to test any
    /// candidate lag, or when no lag is significantly autocorrelated. The
    /// previous version seeded `best_period = 12` and returned it whenever no
    /// candidate beat a correlation of `0.0`, so a series with *no* seasonality
    /// was silently decomposed at a fabricated period of 12 — and every
    /// downstream `result.period` reported 12 as if it had been measured.
    /// Callers that know the period should set it with
    /// [`SeasonalDecomposition::with_period`].
    fn detect_period_autocorr(&self, ts: &TimeSeries) -> Result<usize> {
        let max_period = std::cmp::min(ts.len() / 2, 100);
        if max_period < 2 {
            return Err(Error::InvalidInput(format!(
                "Time series of length {} is too short to detect a seasonal period; \
                 set one explicitly with `with_period`",
                ts.len()
            )));
        }

        // White-noise band at the 5% family-wise level. Every lag in
        // `2..=max_period` is a candidate, so a per-lag 5% band would flag
        // roughly one lag in twenty *by chance* — on pure noise it picks a
        // "seasonal period" almost every time. A Bonferroni correction over the
        // number of candidates keeps the false-positive rate at 5% for the
        // whole search: z = Φ⁻¹(1 − α / (2·L)), band = z / √n.
        let candidates = (max_period - 1) as f64;
        let z = inv_normal_cdf(1.0 - 0.05 / (2.0 * candidates));
        let significance = z / (ts.len() as f64).sqrt();

        let mut best_period = None;
        let mut max_correlation = significance;

        for period in 2..=max_period {
            let correlation = self.calculate_autocorrelation(ts, period)?;
            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = Some(period);
            }
        }

        best_period.ok_or_else(|| {
            Error::InvalidInput(format!(
                "No seasonal period detected: no lag in 2..={max_period} has an autocorrelation \
                 above the family-wise 5% white-noise band ({significance:.4}); set the period \
                 explicitly with `with_period`"
            ))
        })
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

    /// STL decomposition (Seasonal-Trend decomposition using Loess), following
    /// Cleveland, Cleveland, McRae & Terpenning (1990).
    ///
    /// This is the real algorithm — an inner loop that alternates
    /// cycle-subseries loess smoothing (with a low-pass filter to keep the
    /// seasonal component free of trend) and loess smoothing of the
    /// deseasonalized series — not a classical decomposition wearing the STL
    /// label. Unlike [`Self::additive_decomposition`], the seasonal component is
    /// allowed to *evolve* from cycle to cycle. Smoothing spans follow the
    /// authors' recommended defaults for the given period; see
    /// [`crate::time_series::stl::StlParams::recommended`].
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when the series is shorter than two full
    /// periods, when `period < 2`, or when any observation is non-finite (STL
    /// is a global fit; substituting a value for a missing observation would
    /// change every component).
    fn stl_decomposition(&self, ts: &TimeSeries, period: usize) -> Result<DecompositionResult> {
        let values: Vec<f64> = (0..ts.len())
            .map(|i| ts.values.get_f64(i).unwrap_or(f64::NAN))
            .collect();

        let params = if self.robust {
            crate::time_series::stl::StlParams::robust(period)
        } else {
            crate::time_series::stl::StlParams::recommended(period)
        };
        let decomposed = crate::time_series::stl::stl(&values, period, &params)?;

        let trend = TimeSeries::new(ts.index.clone(), TimeSeriesData::from_vec(decomposed.trend))?;
        let seasonal = TimeSeries::new(
            ts.index.clone(),
            TimeSeriesData::from_vec(decomposed.seasonal),
        )?;
        let residual = TimeSeries::new(
            ts.index.clone(),
            TimeSeriesData::from_vec(decomposed.residual),
        )?;

        let metrics = self.calculate_metrics(ts, &trend, &seasonal, &residual)?;

        Ok(DecompositionResult {
            original: ts.clone(),
            trend,
            seasonal,
            residual,
            method: DecompositionMethod::STL,
            period,
            metrics,
        })
    }

    /// X-13ARIMA-SEATS decomposition.
    ///
    /// Not implemented, and deliberately so. Unlike STL — whose definition is a
    /// self-contained sequence of loess smoothers, and which is implemented in
    /// [`Self::stl_decomposition`] — "X-13ARIMA-SEATS" does not name an
    /// algorithm but a *program*: the U.S. Census Bureau's X-13 suite, whose
    /// output is defined by its own RegARIMA pre-adjustment (automatic outlier,
    /// trading-day and holiday regressors with automatic ARIMA order
    /// selection), its X-11 seasonal-filter cascade with data-dependent filter
    /// selection, and the alternative SEATS ARIMA-model-based signal
    /// extraction. Any short reimplementation would produce *different numbers*
    /// under the same name, which is exactly the kind of silent algorithm swap
    /// this crate refuses to ship. Use [`DecompositionMethod::STL`] for a
    /// loess-based decomposition or [`DecompositionMethod::Additive`] /
    /// [`DecompositionMethod::Multiplicative`] for the classical one.
    fn x13_decomposition(&self, _ts: &TimeSeries, _period: usize) -> Result<DecompositionResult> {
        Err(Error::NotImplemented(
            "X-13ARIMA-SEATS decomposition is not implemented: its output is defined by the \
             U.S. Census Bureau X-13 program (RegARIMA pre-adjustment, the X-11 filter cascade \
             and SEATS signal extraction), and an approximation under that name would report \
             different numbers as if they were X-13's. Use DecompositionMethod::STL for a \
             loess-based decomposition."
                .into(),
        ))
    }

    /// Extract trend component using moving average
    /// Extract the trend with a **centered moving average** matched to the
    /// seasonal period.
    ///
    /// * odd `m`: the ordinary centered `m`-MA, weights `1/m` on offsets
    ///   `−⌊m/2⌋ ..= ⌊m/2⌋`.
    /// * even `m`: the `2×m`-MA (Hyndman & Athanasopoulos §6.2), i.e. `m + 1`
    ///   taps with **half weight on the two endpoints**:
    ///   `1/(2m), 1/m, …, 1/m, 1/(2m)`. This is what makes the window centered
    ///   on an observation rather than half-way between two of them, and is
    ///   what every classical decomposition uses for even periods.
    ///
    /// The previous code read `if period % 2 == 0 { period } else { period }` —
    /// both branches identical — and then took a plain unweighted mean, so for
    /// an even period (`m = 12` monthly, `m = 4` quarterly) the "centered"
    /// average was offset by half a period and leaked seasonality into the
    /// trend.
    ///
    /// **Edge policy:** near the ends the kernel is truncated to the available
    /// samples and its weights renormalized to sum to one, so the output has
    /// the same length as the input and reconstruction stays exact. The trend
    /// is correspondingly less smooth in the first and last `⌊m/2⌋` points.
    /// Positions whose window contains no finite value at all are `NaN`.
    fn extract_trend(&self, ts: &TimeSeries, period: usize) -> Result<TimeSeries> {
        if period == 0 {
            return Err(Error::InvalidInput(
                "Seasonal period must be at least 1".to_string(),
            ));
        }

        let half = (period / 2) as isize;
        let m = period as f64;
        let kernel: Vec<(isize, f64)> = (-half..=half)
            .map(|offset| {
                let weight = if period % 2 == 0 && offset.abs() == half {
                    0.5 / m
                } else {
                    1.0 / m
                };
                (offset, weight)
            })
            .collect();

        let len = ts.len() as isize;
        let mut trend_values = Vec::with_capacity(ts.len());

        for i in 0..len {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for &(offset, weight) in &kernel {
                let idx = i + offset;
                if idx < 0 || idx >= len {
                    continue;
                }
                if let Some(value) = ts.values.get_f64(idx as usize) {
                    if value.is_finite() {
                        weighted_sum += weight * value;
                        weight_total += weight;
                    }
                }
            }

            trend_values.push(if weight_total > 0.0 {
                weighted_sum / weight_total
            } else {
                f64::NAN
            });
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

    /// Calculate decomposition metrics.
    ///
    /// `trend_strength` and `seasonality_strength` follow Hyndman &
    /// Athanasopoulos (*Forecasting: Principles and Practice*, §6.7):
    ///
    /// ```text
    /// F_T = max(0, 1 − Var(R) / Var(T + R))
    /// F_S = max(0, 1 − Var(R) / Var(S + R))
    /// ```
    ///
    /// i.e. each component is measured against the variance of *itself plus the
    /// remainder*, which is what bounds the result to `[0, 1]`.
    ///
    /// The formulas this replaces were `1 − (Var(R) + Var(T))/Var(Y)` and
    /// `1 − (Var(R) + Var(S))/Var(Y)`. Those measure something else entirely
    /// (they are 1 minus a *different* component's share) and are routinely
    /// **negative**: for the module's own weekly-seasonal fixture the trend
    /// variance alone exceeds the total, so the reported "seasonality strength"
    /// came out below zero — a value the field is documented to hold in
    /// `[0, 1]`.
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

        let signal_var = trend_var + seasonal_var;
        let noise_var = residual_var;

        // Var(T + R) and Var(S + R) over the positions where both components
        // are finite.
        let trend_plus_residual_var = self.calculate_sum_variance(trend, residual)?;
        let seasonal_plus_residual_var = self.calculate_sum_variance(seasonal, residual)?;

        let strength = |remainder: f64, combined: f64| -> f64 {
            if combined > 0.0 {
                (1.0 - remainder / combined).clamp(0.0, 1.0)
            } else {
                // No variation in the component plus the remainder: the
                // component carries no signal.
                0.0
            }
        };

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
            seasonality_strength: strength(residual_var, seasonal_plus_residual_var),
            trend_strength: strength(residual_var, trend_plus_residual_var),
        })
    }

    /// Variance of the pointwise sum of two components, over the positions
    /// where both are finite.
    fn calculate_sum_variance(&self, a: &TimeSeries, b: &TimeSeries) -> Result<f64> {
        let values: Vec<f64> = (0..a.len().min(b.len()))
            .filter_map(|i| match (a.values.get_f64(i), b.values.get_f64(i)) {
                (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Some(x + y),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return Ok(0.0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Ok(values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64)
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
            let timestamp = Utc
                .timestamp_opt(1640995200 + i * 86400, 0)
                .single()
                .expect("operation should succeed"); // Daily data
            let trend = i as f64 * 0.1; // Linear trend
            let seasonal = (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin() * 2.0; // Weekly seasonality
            let noise = 0.1 * (i as f64 % 3.0 - 1.0); // Small noise
            let value = 10.0 + trend + seasonal + noise;

            builder = builder.add_point(timestamp, value);
        }

        builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_seasonal_decomposition() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).expect("operation should succeed");

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

        let result = decomposer.decompose(&ts).expect("operation should succeed");
        assert_eq!(result.method, DecompositionMethod::Multiplicative);
    }

    #[test]
    fn test_decomposition_reconstruction() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).expect("operation should succeed");
        let reconstructed = result.reconstruct().expect("operation should succeed");

        // Check that reconstruction is close to original
        for i in 0..ts.len() {
            let original = ts.values.get_f64(i).expect("operation should succeed");
            let reconstructed_val = reconstructed
                .values
                .get_f64(i)
                .expect("operation should succeed");
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

        let result = decomposer.decompose(&ts).expect("operation should succeed");

        // Should detect weekly seasonality (period = 7)
        assert_eq!(result.period, 7);
    }

    #[test]
    fn test_quality_metrics() {
        let ts = create_test_seasonal_series();
        let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

        let result = decomposer.decompose(&ts).expect("operation should succeed");
        let quality = result.quality_score();

        assert!(
            quality > 0.5,
            "Quality score should be reasonable for synthetic data"
        );
        assert!(quality <= 1.0, "Quality score should not exceed 1.0");
    }
}

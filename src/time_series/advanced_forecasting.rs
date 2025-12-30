//! Advanced Forecasting Module
//!
//! This module provides advanced forecasting algorithms including:
//! - SARIMA (Seasonal ARIMA)
//! - Auto ARIMA for automatic model selection
//! - Enhanced parameter estimation
//! - Model diagnostics and selection criteria

use crate::core::error::{Error, Result};
use crate::time_series::core::{DateTimeIndex, Frequency, TimeSeries, TimeSeriesData};
use crate::time_series::forecasting::{ForecastMetrics, ForecastResult, Forecaster};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SARIMA (Seasonal ARIMA) model
/// ARIMA(p,d,q)(P,D,Q)\[m\]
#[derive(Debug, Clone)]
pub struct SarimaForecaster {
    /// Non-seasonal AR order
    p: usize,
    /// Non-seasonal differencing order
    d: usize,
    /// Non-seasonal MA order
    q: usize,
    /// Seasonal AR order
    seasonal_p: usize,
    /// Seasonal differencing order
    seasonal_d: usize,
    /// Seasonal MA order
    seasonal_q: usize,
    /// Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    seasonal_period: usize,
    /// AR parameters
    ar_params: Option<Vec<f64>>,
    /// MA parameters
    ma_params: Option<Vec<f64>>,
    /// Seasonal AR parameters
    seasonal_ar_params: Option<Vec<f64>>,
    /// Seasonal MA parameters
    seasonal_ma_params: Option<Vec<f64>>,
    /// Fitted values
    fitted_values: Option<Vec<f64>>,
    /// Residuals
    residuals: Option<Vec<f64>>,
    /// Index for forecasting
    index: Option<DateTimeIndex>,
    /// Differenced series
    differenced_series: Option<Vec<f64>>,
    /// Residual standard deviation
    residual_std: Option<f64>,
    /// Log likelihood
    log_likelihood: Option<f64>,
    /// Number of parameters (for AIC/BIC calculation)
    n_params: usize,
}

impl SarimaForecaster {
    /// Create a new SARIMA model
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> Self {
        let n_params = p + q + seasonal_p + seasonal_q + 1; // +1 for variance
        SarimaForecaster {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_period,
            ar_params: None,
            ma_params: None,
            seasonal_ar_params: None,
            seasonal_ma_params: None,
            fitted_values: None,
            residuals: None,
            index: None,
            differenced_series: None,
            residual_std: None,
            log_likelihood: None,
            n_params,
        }
    }

    /// Create a non-seasonal ARIMA model
    pub fn arima(p: usize, d: usize, q: usize) -> Self {
        Self::new(p, d, q, 0, 0, 0, 1)
    }

    /// Apply differencing to the series
    fn difference(&self, values: &[f64], order: usize) -> Vec<f64> {
        let mut result = values.to_vec();
        for _ in 0..order {
            if result.len() <= 1 {
                break;
            }
            result = result.windows(2).map(|w| w[1] - w[0]).collect();
        }
        result
    }

    /// Apply seasonal differencing
    fn seasonal_difference(&self, values: &[f64], order: usize, period: usize) -> Vec<f64> {
        let mut result = values.to_vec();
        for _ in 0..order {
            if result.len() <= period {
                break;
            }
            result = result
                .iter()
                .skip(period)
                .zip(result.iter())
                .map(|(curr, prev)| curr - prev)
                .collect();
        }
        result
    }

    /// Estimate AR parameters using Yule-Walker equations (simplified)
    fn estimate_ar_params(&self, values: &[f64], order: usize) -> Vec<f64> {
        if order == 0 || values.len() < order + 1 {
            return vec![];
        }

        // Calculate autocorrelations
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();

        let var = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
        if var.abs() < 1e-10 {
            return vec![0.0; order];
        }

        let mut autocorr = Vec::with_capacity(order + 1);
        for lag in 0..=order {
            let cov: f64 = centered
                .iter()
                .take(n - lag)
                .zip(centered.iter().skip(lag))
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64;
            autocorr.push(cov / var);
        }

        // Levinson-Durbin algorithm for solving Yule-Walker equations
        let mut phi = vec![vec![0.0; order]; order];
        let mut partial_autocorr = vec![0.0; order];

        // Order 1
        phi[0][0] = autocorr[1];
        partial_autocorr[0] = autocorr[1];

        // Higher orders
        for k in 1..order {
            let mut num = autocorr[k + 1];
            let mut den = 1.0;

            for j in 0..k {
                num -= phi[k - 1][j] * autocorr[k - j];
                den -= phi[k - 1][j] * autocorr[j + 1];
            }

            if den.abs() < 1e-10 {
                partial_autocorr[k] = 0.0;
            } else {
                partial_autocorr[k] = num / den;
            }

            phi[k][k] = partial_autocorr[k];

            for j in 0..k {
                phi[k][j] = phi[k - 1][j] - partial_autocorr[k] * phi[k - 1][k - 1 - j];
            }
        }

        // Return the AR parameters from the final iteration
        if order > 0 {
            phi[order - 1].clone()
        } else {
            vec![]
        }
    }

    /// Estimate MA parameters using innovation algorithm (simplified)
    fn estimate_ma_params(&self, values: &[f64], order: usize, ar_residuals: &[f64]) -> Vec<f64> {
        if order == 0 || ar_residuals.len() < order + 1 {
            return vec![];
        }

        // Simplified: Use sample autocorrelations of residuals
        let n = ar_residuals.len();
        let mean = ar_residuals.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = ar_residuals.iter().map(|v| v - mean).collect();

        let var = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
        if var.abs() < 1e-10 {
            return vec![0.0; order];
        }

        let mut ma_params = Vec::with_capacity(order);
        for lag in 1..=order {
            let cov: f64 = centered
                .iter()
                .take(n - lag)
                .zip(centered.iter().skip(lag))
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / n as f64;

            // Simplified estimation: use autocorrelation as starting point
            let param = (cov / var).clamp(-0.99, 0.99);
            ma_params.push(param);
        }

        ma_params
    }

    /// Calculate log-likelihood (Gaussian)
    fn calculate_log_likelihood(&self, residuals: &[f64], variance: f64) -> f64 {
        let n = residuals.len() as f64;
        if variance <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let sum_sq: f64 = residuals.iter().map(|r| r * r).sum();
        -0.5 * n * (2.0 * std::f64::consts::PI).ln()
            - 0.5 * n * variance.ln()
            - sum_sq / (2.0 * variance)
    }

    /// Calculate AIC
    pub fn aic(&self) -> Option<f64> {
        self.log_likelihood
            .map(|ll| -2.0 * ll + 2.0 * self.n_params as f64)
    }

    /// Calculate BIC
    pub fn bic(&self, n_obs: usize) -> Option<f64> {
        self.log_likelihood
            .map(|ll| -2.0 * ll + self.n_params as f64 * (n_obs as f64).ln())
    }

    /// Calculate AICc (corrected AIC for small samples)
    pub fn aicc(&self, n_obs: usize) -> Option<f64> {
        self.aic().map(|aic| {
            let k = self.n_params as f64;
            let n = n_obs as f64;
            if n - k - 1.0 > 0.0 {
                aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
            } else {
                aic
            }
        })
    }
}

impl Forecaster for SarimaForecaster {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        let min_len = self.p
            + self.d
            + self.q
            + self.seasonal_period * (self.seasonal_p + self.seasonal_d + self.seasonal_q)
            + 1;
        if ts.len() < min_len {
            return Err(Error::InvalidInput(format!(
                "Time series too short for SARIMA model. Need at least {} observations, got {}",
                min_len,
                ts.len()
            )));
        }

        // Get values
        let values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();
        if values.len() != ts.len() {
            return Err(Error::InvalidInput(
                "Missing values not supported in SARIMA".to_string(),
            ));
        }

        // Apply differencing
        let mut working_series = values.clone();

        // Non-seasonal differencing
        working_series = self.difference(&working_series, self.d);

        // Seasonal differencing
        if self.seasonal_d > 0 && self.seasonal_period > 1 {
            working_series =
                self.seasonal_difference(&working_series, self.seasonal_d, self.seasonal_period);
        }

        // Estimate AR parameters
        let ar_params = self.estimate_ar_params(&working_series, self.p);

        // Calculate AR residuals
        let mut ar_residuals = Vec::with_capacity(working_series.len());
        for i in 0..working_series.len() {
            let mut prediction = 0.0;
            for (j, &param) in ar_params.iter().enumerate() {
                if i > j {
                    prediction += param * working_series[i - j - 1];
                }
            }
            ar_residuals.push(working_series[i] - prediction);
        }

        // Estimate MA parameters
        let ma_params = self.estimate_ma_params(&working_series, self.q, &ar_residuals);

        // Estimate seasonal parameters (simplified)
        let seasonal_ar_params = if self.seasonal_p > 0 {
            self.estimate_ar_params(&working_series, self.seasonal_p)
        } else {
            vec![]
        };

        let seasonal_ma_params = if self.seasonal_q > 0 {
            self.estimate_ma_params(&working_series, self.seasonal_q, &ar_residuals)
        } else {
            vec![]
        };

        // Calculate fitted values and final residuals
        let mut fitted = Vec::with_capacity(working_series.len());
        let mut residuals = Vec::with_capacity(working_series.len());

        for i in 0..working_series.len() {
            let mut prediction = 0.0;

            // AR component
            for (j, &param) in ar_params.iter().enumerate() {
                if i > j {
                    prediction += param * working_series[i - j - 1];
                }
            }

            // MA component
            for (j, &param) in ma_params.iter().enumerate() {
                if i > j && j < residuals.len() {
                    prediction += param * residuals[residuals.len() - j - 1];
                }
            }

            // Seasonal AR component
            for (j, &param) in seasonal_ar_params.iter().enumerate() {
                let lag = (j + 1) * self.seasonal_period;
                if i >= lag {
                    prediction += param * working_series[i - lag];
                }
            }

            // Seasonal MA component
            for (j, &param) in seasonal_ma_params.iter().enumerate() {
                let lag = (j + 1) * self.seasonal_period;
                if lag <= residuals.len() {
                    prediction += param * residuals[residuals.len() - lag];
                }
            }

            fitted.push(prediction);
            residuals.push(working_series[i] - prediction);
        }

        // Calculate statistics
        let variance = residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64;
        let residual_std = variance.sqrt();
        let log_likelihood = self.calculate_log_likelihood(&residuals, variance);

        // Store results
        self.ar_params = Some(ar_params);
        self.ma_params = Some(ma_params);
        self.seasonal_ar_params = Some(seasonal_ar_params);
        self.seasonal_ma_params = Some(seasonal_ma_params);
        self.fitted_values = Some(fitted);
        self.residuals = Some(residuals);
        self.index = Some(ts.index.clone());
        self.differenced_series = Some(working_series);
        self.residual_std = Some(residual_std);
        self.log_likelihood = Some(log_likelihood);

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

        let seasonal_ar = self
            .seasonal_ar_params
            .as_ref()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let seasonal_ma = self
            .seasonal_ma_params
            .as_ref()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        // Generate forecasts
        let mut forecasts = Vec::with_capacity(periods);
        let mut extended_series = differenced.clone();
        let mut extended_residuals = residuals.clone();

        for _ in 0..periods {
            let mut forecast = 0.0;
            let n = extended_series.len();

            // AR component
            for (j, &param) in ar_params.iter().enumerate() {
                if n > j {
                    forecast += param * extended_series[n - j - 1];
                }
            }

            // MA component (residuals become 0 for future periods)
            for (j, &param) in ma_params.iter().enumerate() {
                if j < extended_residuals.len() {
                    let idx = extended_residuals.len() - j - 1;
                    if idx < residuals.len() {
                        forecast += param * extended_residuals[idx];
                    }
                }
            }

            // Seasonal AR component
            for (j, &param) in seasonal_ar.iter().enumerate() {
                let lag = (j + 1) * self.seasonal_period;
                if n >= lag {
                    forecast += param * extended_series[n - lag];
                }
            }

            // Seasonal MA component
            for (j, &param) in seasonal_ma.iter().enumerate() {
                let lag = (j + 1) * self.seasonal_period;
                if lag <= extended_residuals.len() {
                    let idx = extended_residuals.len() - lag;
                    if idx < residuals.len() {
                        forecast += param * extended_residuals[idx];
                    }
                }
            }

            forecasts.push(forecast);
            extended_series.push(forecast);
            extended_residuals.push(0.0); // Expected residual for future
        }

        // Create forecast dates
        let last_date = *index.end().unwrap();
        let frequency = index.frequency.clone().unwrap_or(Frequency::Daily);
        let duration = frequency.to_duration();

        let mut forecast_dates = Vec::with_capacity(periods);
        for i in 1..=periods {
            forecast_dates.push(last_date + duration * i as i32);
        }

        // Calculate prediction intervals (with increasing uncertainty)
        let residual_std = self.residual_std.unwrap_or(1.0);
        let z_score = get_z_score(confidence_level);

        let mut lower_values = Vec::with_capacity(periods);
        let mut upper_values = Vec::with_capacity(periods);

        for (h, &forecast) in forecasts.iter().enumerate() {
            // Prediction interval widens with horizon
            let margin = z_score * residual_std * ((h + 1) as f64).sqrt();
            lower_values.push(forecast - margin);
            upper_values.push(forecast + margin);
        }

        let forecast_index = DateTimeIndex::with_frequency(forecast_dates, frequency);

        let forecast_ts =
            TimeSeries::new(forecast_index.clone(), TimeSeriesData::from_vec(forecasts))?;
        let lower_ci_ts = TimeSeries::new(
            forecast_index.clone(),
            TimeSeriesData::from_vec(lower_values),
        )?;
        let upper_ci_ts = TimeSeries::new(forecast_index, TimeSeriesData::from_vec(upper_values))?;

        let mut parameters = self.parameters();
        if let Some(aic) = self.aic() {
            parameters.insert("aic".to_string(), aic);
        }
        if let Some(bic) = self.bic(differenced.len()) {
            parameters.insert("bic".to_string(), bic);
        }

        Ok(ForecastResult {
            forecast: forecast_ts,
            lower_ci: lower_ci_ts,
            upper_ci: upper_ci_ts,
            method: self.name().to_string(),
            parameters,
            metrics: ForecastMetrics {
                mae: None,
                mse: None,
                rmse: None,
                mape: None,
                smape: None,
                aic: self.aic(),
                bic: self.bic(differenced.len()),
                log_likelihood: self.log_likelihood,
            },
            confidence_level,
        })
    }

    fn name(&self) -> &str {
        "SARIMA"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.p as f64);
        params.insert("d".to_string(), self.d as f64);
        params.insert("q".to_string(), self.q as f64);
        params.insert("P".to_string(), self.seasonal_p as f64);
        params.insert("D".to_string(), self.seasonal_d as f64);
        params.insert("Q".to_string(), self.seasonal_q as f64);
        params.insert("m".to_string(), self.seasonal_period as f64);
        params
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let fitted = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;
        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted".to_string()))?;

        let n = fitted.len() as f64;
        let mse = residuals.iter().map(|r| r * r).sum::<f64>() / n;
        let rmse = mse.sqrt();
        let mae = residuals.iter().map(|r| r.abs()).sum::<f64>() / n;

        Ok(ForecastMetrics {
            mae: Some(mae),
            mse: Some(mse),
            rmse: Some(rmse),
            mape: None, // Would need original values
            smape: None,
            aic: self.aic(),
            bic: self.bic(fitted.len()),
            log_likelihood: self.log_likelihood,
        })
    }
}

/// Auto ARIMA for automatic model selection
#[derive(Debug, Clone)]
pub struct AutoArima {
    /// Maximum AR order to consider
    max_p: usize,
    /// Maximum differencing order
    max_d: usize,
    /// Maximum MA order to consider
    max_q: usize,
    /// Maximum seasonal AR order
    max_seasonal_p: usize,
    /// Maximum seasonal differencing
    max_seasonal_d: usize,
    /// Maximum seasonal MA order
    max_seasonal_q: usize,
    /// Seasonal period (0 for non-seasonal)
    seasonal_period: usize,
    /// Information criterion to use
    criterion: ModelSelectionCriterion,
    /// Best model found
    best_model: Option<SarimaForecaster>,
    /// Model selection results
    selection_results: Vec<ModelSelectionResult>,
}

/// Criterion for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Corrected AIC (for small samples)
    AICc,
    /// Bayesian Information Criterion
    BIC,
}

impl Default for ModelSelectionCriterion {
    fn default() -> Self {
        ModelSelectionCriterion::AICc
    }
}

/// Result of model selection for a single model
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    /// Model order (p, d, q)
    pub order: (usize, usize, usize),
    /// Seasonal order (P, D, Q, m)
    pub seasonal_order: (usize, usize, usize, usize),
    /// AIC value
    pub aic: Option<f64>,
    /// AICc value
    pub aicc: Option<f64>,
    /// BIC value
    pub bic: Option<f64>,
    /// Whether the model was successfully fitted
    pub success: bool,
}

impl AutoArima {
    /// Create a new Auto ARIMA selector
    pub fn new() -> Self {
        AutoArima {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            max_seasonal_p: 2,
            max_seasonal_d: 1,
            max_seasonal_q: 2,
            seasonal_period: 0,
            criterion: ModelSelectionCriterion::AICc,
            best_model: None,
            selection_results: Vec::new(),
        }
    }

    /// Set maximum AR order
    pub fn max_p(mut self, p: usize) -> Self {
        self.max_p = p;
        self
    }

    /// Set maximum differencing order
    pub fn max_d(mut self, d: usize) -> Self {
        self.max_d = d;
        self
    }

    /// Set maximum MA order
    pub fn max_q(mut self, q: usize) -> Self {
        self.max_q = q;
        self
    }

    /// Set seasonal period
    pub fn seasonal(mut self, period: usize) -> Self {
        self.seasonal_period = period;
        self
    }

    /// Set maximum seasonal AR order
    pub fn max_seasonal_p(mut self, p: usize) -> Self {
        self.max_seasonal_p = p;
        self
    }

    /// Set maximum seasonal differencing
    pub fn max_seasonal_d(mut self, d: usize) -> Self {
        self.max_seasonal_d = d;
        self
    }

    /// Set maximum seasonal MA order
    pub fn max_seasonal_q(mut self, q: usize) -> Self {
        self.max_seasonal_q = q;
        self
    }

    /// Set model selection criterion
    pub fn criterion(mut self, criterion: ModelSelectionCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Fit and select the best model
    pub fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        let n_obs = ts.len();
        self.selection_results.clear();

        let mut best_criterion_value = f64::INFINITY;
        let mut best_model: Option<SarimaForecaster> = None;

        // Determine differencing order using KPSS-like test (simplified)
        let d = self.estimate_differencing_order(ts)?;

        // Determine seasonal differencing if applicable
        let seasonal_d = if self.seasonal_period > 1 {
            self.estimate_seasonal_differencing_order(ts)?
        } else {
            0
        };

        // Try different model orders
        for p in 0..=self.max_p {
            for q in 0..=self.max_q {
                // Non-seasonal model
                if self.seasonal_period <= 1 {
                    let result = self.try_model(ts, p, d, q, 0, 0, 0, 1, n_obs);
                    self.selection_results.push(result.clone());

                    if result.success {
                        let criterion_value = self.get_criterion_value(&result);
                        if criterion_value < best_criterion_value {
                            best_criterion_value = criterion_value;
                            let mut model = SarimaForecaster::arima(p, d, q);
                            if model.fit(ts).is_ok() {
                                best_model = Some(model);
                            }
                        }
                    }
                } else {
                    // Seasonal models
                    for seasonal_p in 0..=self.max_seasonal_p {
                        for seasonal_q in 0..=self.max_seasonal_q {
                            let result = self.try_model(
                                ts,
                                p,
                                d,
                                q,
                                seasonal_p,
                                seasonal_d,
                                seasonal_q,
                                self.seasonal_period,
                                n_obs,
                            );
                            self.selection_results.push(result.clone());

                            if result.success {
                                let criterion_value = self.get_criterion_value(&result);
                                if criterion_value < best_criterion_value {
                                    best_criterion_value = criterion_value;
                                    let mut model = SarimaForecaster::new(
                                        p,
                                        d,
                                        q,
                                        seasonal_p,
                                        seasonal_d,
                                        seasonal_q,
                                        self.seasonal_period,
                                    );
                                    if model.fit(ts).is_ok() {
                                        best_model = Some(model);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if best_model.is_some() {
            self.best_model = best_model;
            Ok(())
        } else {
            Err(Error::InvalidOperation(
                "No suitable model found".to_string(),
            ))
        }
    }

    /// Try fitting a specific model and return results
    fn try_model(
        &self,
        ts: &TimeSeries,
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
        n_obs: usize,
    ) -> ModelSelectionResult {
        let mut model =
            SarimaForecaster::new(p, d, q, seasonal_p, seasonal_d, seasonal_q, seasonal_period);

        let success = model.fit(ts).is_ok();

        ModelSelectionResult {
            order: (p, d, q),
            seasonal_order: (seasonal_p, seasonal_d, seasonal_q, seasonal_period),
            aic: model.aic(),
            aicc: model.aicc(n_obs),
            bic: model.bic(n_obs),
            success,
        }
    }

    /// Get the criterion value based on selected criterion
    fn get_criterion_value(&self, result: &ModelSelectionResult) -> f64 {
        match self.criterion {
            ModelSelectionCriterion::AIC => result.aic.unwrap_or(f64::INFINITY),
            ModelSelectionCriterion::AICc => result.aicc.unwrap_or(f64::INFINITY),
            ModelSelectionCriterion::BIC => result.bic.unwrap_or(f64::INFINITY),
        }
    }

    /// Estimate the differencing order (simplified approach)
    fn estimate_differencing_order(&self, ts: &TimeSeries) -> Result<usize> {
        let values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();

        // Check variance ratio after differencing
        let var0 = variance(&values);
        if var0 < 1e-10 {
            return Ok(0);
        }

        let diff1: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let var1 = variance(&diff1);

        // If differencing reduces variance significantly, use d=1
        if var1 < var0 * 0.9 {
            let diff2: Vec<f64> = diff1.windows(2).map(|w| w[1] - w[0]).collect();
            let var2 = variance(&diff2);

            // Check if second differencing helps
            if var2 < var1 * 0.9 {
                Ok(2.min(self.max_d))
            } else {
                Ok(1.min(self.max_d))
            }
        } else {
            Ok(0)
        }
    }

    /// Estimate seasonal differencing order
    fn estimate_seasonal_differencing_order(&self, ts: &TimeSeries) -> Result<usize> {
        if self.seasonal_period <= 1 {
            return Ok(0);
        }

        let values: Vec<f64> = (0..ts.len()).filter_map(|i| ts.values.get_f64(i)).collect();
        if values.len() <= self.seasonal_period * 2 {
            return Ok(0);
        }

        let var0 = variance(&values);

        // Seasonal difference
        let seasonal_diff: Vec<f64> = values
            .iter()
            .skip(self.seasonal_period)
            .zip(values.iter())
            .map(|(curr, prev)| curr - prev)
            .collect();

        let var1 = variance(&seasonal_diff);

        if var1 < var0 * 0.8 {
            Ok(1.min(self.max_seasonal_d))
        } else {
            Ok(0)
        }
    }

    /// Get the best model
    pub fn best_model(&self) -> Option<&SarimaForecaster> {
        self.best_model.as_ref()
    }

    /// Get all model selection results
    pub fn selection_results(&self) -> &[ModelSelectionResult] {
        &self.selection_results
    }

    /// Get summary of model selection
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("Auto ARIMA Model Selection Summary\n");
        summary.push_str("==================================\n\n");

        if let Some(model) = &self.best_model {
            summary.push_str(&format!(
                "Best Model: ARIMA({},{},{})",
                model.p, model.d, model.q
            ));
            if model.seasonal_period > 1 {
                summary.push_str(&format!(
                    "({},{},{}){}",
                    model.seasonal_p, model.seasonal_d, model.seasonal_q, model.seasonal_period
                ));
            }
            summary.push_str("\n\n");

            if let Some(aic) = model.aic() {
                summary.push_str(&format!("AIC: {:.4}\n", aic));
            }
        } else {
            summary.push_str("No model selected.\n");
        }

        summary.push_str(&format!(
            "\nModels evaluated: {}\n",
            self.selection_results.len()
        ));

        summary
    }
}

impl Default for AutoArima {
    fn default() -> Self {
        Self::new()
    }
}

impl Forecaster for AutoArima {
    fn fit(&mut self, ts: &TimeSeries) -> Result<()> {
        AutoArima::fit(self, ts)
    }

    fn forecast(&self, periods: usize, confidence_level: f64) -> Result<ForecastResult> {
        let model = self
            .best_model
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("No model fitted".to_string()))?;
        model.forecast(periods, confidence_level)
    }

    fn name(&self) -> &str {
        "Auto ARIMA"
    }

    fn parameters(&self) -> HashMap<String, f64> {
        self.best_model
            .as_ref()
            .map(|m| m.parameters())
            .unwrap_or_default()
    }

    fn fit_metrics(&self, ts: &TimeSeries) -> Result<ForecastMetrics> {
        let model = self
            .best_model
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("No model fitted".to_string()))?;
        model.fit_metrics(ts)
    }
}

/// Helper function to get z-score for confidence level
fn get_z_score(confidence_level: f64) -> f64 {
    match (confidence_level * 100.0) as i32 {
        90 => 1.645,
        95 => 1.96,
        99 => 2.576,
        _ => 1.96,
    }
}

/// Helper function to calculate variance
fn variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_test_series_with_trend() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();
        for i in 0..100 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let value = 10.0 + i as f64 * 0.5 + (i as f64 * 0.1).sin();
            builder = builder.add_point(timestamp, value);
        }
        builder.frequency(Frequency::Daily).build().unwrap()
    }

    fn create_seasonal_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();
        for i in 0..120 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            // Trend + seasonality (period 7)
            let value =
                10.0 + i as f64 * 0.1 + 5.0 * (i as f64 * 2.0 * std::f64::consts::PI / 7.0).sin();
            builder = builder.add_point(timestamp, value);
        }
        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_sarima_non_seasonal() {
        let ts = create_test_series_with_trend();
        let mut model = SarimaForecaster::arima(1, 1, 1);

        model.fit(&ts).unwrap();
        let result = model.forecast(10, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 10);
        assert!(model.aic().is_some());
        assert!(model.log_likelihood.is_some());
    }

    #[test]
    fn test_sarima_seasonal() {
        let ts = create_seasonal_series();
        let mut model = SarimaForecaster::new(1, 1, 1, 1, 0, 1, 7);

        model.fit(&ts).unwrap();
        let result = model.forecast(14, 0.95).unwrap();

        assert_eq!(result.forecast.len(), 14);
    }

    #[test]
    fn test_auto_arima() {
        let ts = create_test_series_with_trend();
        let mut auto = AutoArima::new().max_p(2).max_d(2).max_q(2);

        auto.fit(&ts).unwrap();

        assert!(auto.best_model().is_some());
        assert!(!auto.selection_results().is_empty());

        let result = auto.forecast(5, 0.95).unwrap();
        assert_eq!(result.forecast.len(), 5);
    }

    #[test]
    fn test_auto_arima_summary() {
        let ts = create_test_series_with_trend();
        let mut auto = AutoArima::new().max_p(2).max_d(1).max_q(2);

        auto.fit(&ts).unwrap();
        let summary = auto.summary();

        assert!(summary.contains("Auto ARIMA"));
        assert!(summary.contains("Best Model"));
    }

    #[test]
    fn test_model_selection_criterion() {
        let ts = create_test_series_with_trend();

        // Test with AIC
        let mut auto_aic = AutoArima::new()
            .max_p(2)
            .max_q(2)
            .criterion(ModelSelectionCriterion::AIC);
        auto_aic.fit(&ts).unwrap();

        // Test with BIC
        let mut auto_bic = AutoArima::new()
            .max_p(2)
            .max_q(2)
            .criterion(ModelSelectionCriterion::BIC);
        auto_bic.fit(&ts).unwrap();

        // Both should find models
        assert!(auto_aic.best_model().is_some());
        assert!(auto_bic.best_model().is_some());
    }

    #[test]
    fn test_information_criteria() {
        let ts = create_test_series_with_trend();
        let mut model = SarimaForecaster::arima(1, 1, 1);
        model.fit(&ts).unwrap();

        let aic = model.aic();
        let bic = model.bic(ts.len());
        let aicc = model.aicc(ts.len());

        assert!(aic.is_some());
        assert!(bic.is_some());
        assert!(aicc.is_some());

        // AICc should be >= AIC for any sample
        assert!(aicc.unwrap() >= aic.unwrap());
    }

    #[test]
    fn test_confidence_intervals_widen() {
        let ts = create_test_series_with_trend();
        let mut model = SarimaForecaster::arima(1, 1, 1);
        model.fit(&ts).unwrap();

        let result = model.forecast(10, 0.95).unwrap();

        // Check that confidence intervals widen with horizon
        let first_width =
            result.upper_ci.values.get_f64(0).unwrap() - result.lower_ci.values.get_f64(0).unwrap();
        let last_width =
            result.upper_ci.values.get_f64(9).unwrap() - result.lower_ci.values.get_f64(9).unwrap();

        assert!(
            last_width > first_width,
            "CI should widen with forecast horizon"
        );
    }
}

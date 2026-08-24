//! Smoothing filters for [`TimeSeriesPreprocessor`].
//!
//! Split out of `preprocessing.rs` to keep both files well under the 2 000-line
//! ceiling. Every method here is a genuine implementation of the algorithm it
//! names: `Lowess` is Cleveland's locally-weighted regression, `KalmanFilter` a
//! local-level state-space smoother with a maximum-likelihood signal-to-noise
//! ratio, and `HodrickPrescott` the exact minimizer of the HP objective — none
//! of them is a moving average in disguise.

use crate::core::error::{Error, Result};
use crate::time_series::core::{TimeSeries, TimeSeriesData};
use crate::time_series::preprocessing::{
    SmoothingConfig, SmoothingMethod, TimeSeriesPreprocessor, TransformationInfo,
};

impl TimeSeriesPreprocessor {
    /// Apply the configured smoothing filter, reporting back any parameter the
    /// filter *estimated* rather than was given.
    pub(super) fn apply_smoothing(
        &self,
        ts: &TimeSeries,
        config: &SmoothingConfig,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        let mut parameters = config.parameters.clone();

        let smoothed_series = match &config.method {
            SmoothingMethod::MovingAverage { window } => self.moving_average_smooth(ts, *window)?,
            SmoothingMethod::ExponentialSmoothing { alpha } => {
                self.exponential_smooth(ts, *alpha)?
            }
            SmoothingMethod::SavitzkyGolay { window, order } => {
                self.savitzky_golay_smooth(ts, *window, *order)?
            }
            SmoothingMethod::Lowess { fraction } => self.lowess_smooth(ts, *fraction)?,
            SmoothingMethod::KalmanFilter => {
                let (series, fit) = self.kalman_smooth(ts)?;
                // The local-level model has no user-supplied tuning knob: its
                // signal-to-noise ratio is estimated by maximum likelihood. Report
                // what was chosen so the amount of smoothing is inspectable rather
                // than hidden inside the filter.
                parameters.insert("kalman_signal_to_noise".to_string(), fit.signal_to_noise);
                parameters.insert(
                    "kalman_observation_variance".to_string(),
                    fit.observation_variance,
                );
                series
            }
            SmoothingMethod::HodrickPrescott { lambda } => {
                self.hodrick_prescott_smooth(ts, *lambda)?
            }
        };

        let transform_info = TransformationInfo {
            transformation_type: format!("smoothing_{:?}", config.method),
            parameters,
            affected_values: ts.len(),
            order: 4,
        };

        Ok((smoothed_series, transform_info))
    }

    pub(super) fn moving_average_smooth(
        &self,
        ts: &TimeSeries,
        window: usize,
    ) -> Result<TimeSeries> {
        ts.rolling_mean(window)
    }

    pub(super) fn exponential_smooth(&self, ts: &TimeSeries, alpha: f64) -> Result<TimeSeries> {
        let mut smoothed_values = Vec::with_capacity(ts.len());

        if let Some(first_val) = ts.values.get_f64(0) {
            smoothed_values.push(first_val);

            for i in 1..ts.len() {
                if let Some(current_val) = ts.values.get_f64(i) {
                    let prev_smooth = smoothed_values[i - 1];
                    let new_smooth = alpha * current_val + (1.0 - alpha) * prev_smooth;
                    smoothed_values.push(new_smooth);
                } else {
                    smoothed_values.push(smoothed_values[i - 1]);
                }
            }
        }

        let smoothed_series = TimeSeriesData::from_vec(smoothed_values);
        TimeSeries::new(ts.index.clone(), smoothed_series)
    }

    /// Savitzky-Golay smoothing by local least-squares polynomial fitting.
    ///
    /// For each point a polynomial of degree `order` is fitted (by ordinary
    /// least squares) over the surrounding `window` samples and evaluated at the
    /// centre. For interior points this is exactly the classic Savitzky-Golay
    /// convolution; at the boundaries the fit adapts to the available
    /// (asymmetric) window. This honours both `window` and `order` rather than
    /// collapsing to a moving average.
    pub(super) fn savitzky_golay_smooth(
        &self,
        ts: &TimeSeries,
        window: usize,
        order: usize,
    ) -> Result<TimeSeries> {
        if window < 2 || order >= window {
            return Err(Error::InvalidInput(
                "Savitzky-Golay filter requires window >= 2 and order < window".to_string(),
            ));
        }

        let n = ts.len();
        let values: Vec<f64> = (0..n)
            .map(|i| ts.values.get_f64(i).unwrap_or(f64::NAN))
            .collect();
        let half = window / 2;

        let mut smoothed = Vec::with_capacity(n);
        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);

            // Local coordinates centred on i (so the fitted value at the centre
            // is the constant term of the polynomial).
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for j in start..end {
                if values[j].is_finite() {
                    xs.push(j as f64 - i as f64);
                    ys.push(values[j]);
                }
            }

            let degree = order.min(xs.len().saturating_sub(1));
            if xs.len() < 2 || degree == 0 {
                let fallback = if ys.is_empty() {
                    values[i]
                } else {
                    ys.iter().sum::<f64>() / ys.len() as f64
                };
                smoothed.push(fallback);
                continue;
            }

            // Normal equations for the polynomial fit: (XᵀX)·c = Xᵀy.
            let k = degree + 1;
            let mut xtx = vec![vec![0.0_f64; k]; k];
            let mut xty = vec![0.0_f64; k];
            for (idx, &x) in xs.iter().enumerate() {
                let mut powers = vec![1.0_f64; k];
                for p in 1..k {
                    powers[p] = powers[p - 1] * x;
                }
                for a in 0..k {
                    xty[a] += powers[a] * ys[idx];
                    for b in 0..k {
                        xtx[a][b] += powers[a] * powers[b];
                    }
                }
            }

            match crate::time_series::preprocessing::solve_linear_system(xtx, xty) {
                // c[0] is the polynomial value at x = 0, i.e. the smoothed point.
                Some(c) => smoothed.push(c[0]),
                None => smoothed.push(values[i]),
            }
        }

        let smoothed_series = TimeSeriesData::from_vec(smoothed);
        TimeSeries::new(ts.index.clone(), smoothed_series)
    }

    /// Collect the series as a dense `Vec<f64>`, rejecting non-finite entries.
    ///
    /// The whole-series smoothers below (LOWESS, Kalman, Hodrick-Prescott) are
    /// global fits: a missing observation cannot be quietly replaced by `0.0`
    /// or by a neighbour without changing every fitted value, so an incomplete
    /// series is reported as an error and the caller is pointed at the
    /// missing-value stage that exists for exactly this purpose.
    pub(super) fn dense_values(ts: &TimeSeries, method: &str) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(ts.len());
        for i in 0..ts.len() {
            match ts.values.get_f64(i) {
                Some(v) if v.is_finite() => values.push(v),
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "{method} needs a complete series, but observation {i} is missing or \
                         non-finite; handle missing values first (see `MissingValueStrategy`)"
                    )))
                }
            }
        }
        Ok(values)
    }

    /// LOWESS smoothing (Cleveland, 1979).
    ///
    /// Each fitted point is a locally-weighted linear regression over the
    /// `fraction · n` nearest observations, using tricube distance weights, with
    /// three bisquare robustness passes so that outliers do not drag the curve.
    /// See [`crate::time_series::loess`] for the algorithm.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when `fraction` is outside `(0, 1]` or
    /// the series contains a missing or non-finite observation.
    pub(super) fn lowess_smooth(&self, ts: &TimeSeries, fraction: f64) -> Result<TimeSeries> {
        let values = Self::dense_values(ts, "LOWESS smoothing")?;
        let x: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        // Three robustness iterations is Cleveland's recommended default.
        let smoothed = crate::time_series::loess::lowess(&x, &values, fraction, 3, 1)?;
        TimeSeries::new(ts.index.clone(), TimeSeriesData::from_vec(smoothed))
    }

    /// Kalman smoothing under a local-level (random-walk-plus-noise)
    /// state-space model.
    ///
    /// The signal-to-noise ratio is estimated from the data by maximum
    /// likelihood, then the level is extracted by the Kalman filter followed by
    /// the backward (RTS) state smoother, so every fitted point conditions on
    /// the entire sample. See [`crate::time_series::filters::local_level_smooth`].
    ///
    /// Returns the smoothed series together with the fitted hyper-parameters, so
    /// the caller can report which signal-to-noise ratio the likelihood chose.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] for an empty series or one containing a
    /// missing or non-finite observation.
    pub(super) fn kalman_smooth(
        &self,
        ts: &TimeSeries,
    ) -> Result<(TimeSeries, crate::time_series::filters::LocalLevelFit)> {
        let values = Self::dense_values(ts, "Kalman smoothing")?;
        let fit = crate::time_series::filters::local_level_smooth(&values)?;
        let series = TimeSeries::new(
            ts.index.clone(),
            TimeSeriesData::from_vec(fit.level.clone()),
        )?;
        Ok((series, fit))
    }

    /// Hodrick-Prescott filter.
    ///
    /// Returns the exact minimizer of
    /// `Σ(yₜ − τₜ)² + λ Σ(Δ²τₜ)²`, obtained by an `O(n)` banded Cholesky solve
    /// of the pentadiagonal system `(I + λDᵀD)τ = y`. `lambda` genuinely
    /// controls the smoothness; it is not a decorative parameter.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when `lambda` is negative or non-finite,
    /// or the series contains a missing or non-finite observation.
    pub(super) fn hodrick_prescott_smooth(
        &self,
        ts: &TimeSeries,
        lambda: f64,
    ) -> Result<TimeSeries> {
        let values = Self::dense_values(ts, "The Hodrick-Prescott filter")?;
        let trend = crate::time_series::filters::hodrick_prescott(&values, lambda)?;
        TimeSeries::new(ts.index.clone(), TimeSeriesData::from_vec(trend))
    }
}

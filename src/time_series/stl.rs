//! STL — Seasonal-Trend decomposition using Loess (Cleveland, Cleveland,
//! McRae & Terpenning, 1990).
//!
//! STL splits a series into `trend + seasonal + remainder` with two nested
//! loops:
//!
//! * the **inner loop** alternates between smoothing each *cycle-subseries*
//!   (all Januaries, all Februaries, …) to update the seasonal component, and
//!   smoothing the deseasonalized series to update the trend. A low-pass
//!   filter removes any low-frequency drift that leaks into the seasonal
//!   component, which is what keeps the two components identified;
//! * the **outer loop** recomputes bisquare robustness weights from the
//!   remainder, so outliers stop distorting either component.
//!
//! Every smoothing step is a genuine locally-weighted regression from
//! [`crate::time_series::loess`] — this is not a classical moving-average
//! decomposition relabelled as STL.

use crate::core::error::{Error, Result};
use crate::time_series::loess::{bisquare, loess_series, loess_series_at, median_abs};

/// Tuning parameters for [`stl`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct StlParams {
    /// Span (in cycles) of the loess smoother applied to each cycle-subseries.
    /// Larger values force the seasonal shape to change more slowly from one
    /// cycle to the next.
    pub seasonal_span: usize,
    /// Span (in observations) of the loess smoother applied to the
    /// deseasonalized series to obtain the trend.
    pub trend_span: usize,
    /// Span (in observations) of the loess smoother inside the low-pass filter.
    pub lowpass_span: usize,
    /// Inner-loop passes per outer iteration.
    pub inner_iters: usize,
    /// Outer (robustness) iterations. `0` gives the non-robust fit.
    pub outer_iters: usize,
}

impl StlParams {
    /// Cleveland et al.'s recommended defaults for a series of the given
    /// seasonal period, in the non-robust configuration:
    ///
    /// * `n_s = 7` — the smallest span the authors recommend, letting the
    ///   seasonal shape evolve slowly across cycles;
    /// * `n_l` = the smallest odd integer `≥ period`;
    /// * `n_t` = the smallest odd integer `≥ 1.5·period / (1 − 1.5/n_s)`, the
    ///   value that makes the trend smoother unable to absorb the seasonal
    ///   signal;
    /// * `n_i = 2`, `n_o = 0` — the non-robust setting (two inner passes are
    ///   enough for convergence when there are no outlier weights to settle).
    pub fn recommended(period: usize) -> Self {
        let seasonal_span = 7usize;
        let lowpass_span = next_odd(period as f64);
        let trend_span = next_odd(1.5 * period as f64 / (1.0 - 1.5 / seasonal_span as f64));
        Self {
            seasonal_span,
            trend_span,
            lowpass_span,
            inner_iters: 2,
            outer_iters: 0,
        }
    }

    /// The robust configuration: one inner pass per outer iteration and fifteen
    /// outer iterations, as recommended when the series contains outliers.
    /// Reachable from the public API through
    /// [`crate::time_series::decomposition::SeasonalDecomposition::with_robust`].
    pub fn robust(period: usize) -> Self {
        Self {
            inner_iters: 1,
            outer_iters: 15,
            ..Self::recommended(period)
        }
    }
}

/// Smallest odd integer greater than or equal to `value` (and at least 3).
fn next_odd(value: f64) -> usize {
    let ceiling = value.ceil().max(3.0) as usize;
    if ceiling % 2 == 0 {
        ceiling + 1
    } else {
        ceiling
    }
}

/// The three STL components, each the same length as the input.
#[derive(Debug, Clone)]
pub(crate) struct StlDecomposition {
    /// Trend-cycle component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder (`y − trend − seasonal`).
    pub residual: Vec<f64>,
}

/// Run STL on `values` at the given seasonal `period`.
///
/// # Errors
/// Returns [`Error::InvalidInput`] when `period < 2`, when the series is
/// shorter than two full periods (a cycle-subseries would then have a single
/// observation, so no seasonal evolution is estimable), or when the series
/// contains a non-finite value.
pub(crate) fn stl(values: &[f64], period: usize, params: &StlParams) -> Result<StlDecomposition> {
    let n = values.len();
    if period < 2 {
        return Err(Error::InvalidInput(format!(
            "STL needs a seasonal period of at least 2, got {period}"
        )));
    }
    if n < 2 * period {
        return Err(Error::InvalidInput(format!(
            "STL needs at least two full periods ({} observations), got {n}",
            2 * period
        )));
    }
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "STL requires finite values; index {bad} is {}",
            values[bad]
        )));
    }

    let mut seasonal = vec![0.0_f64; n];
    let mut trend = vec![0.0_f64; n];
    let mut robustness = vec![1.0_f64; n];

    for outer in 0..=params.outer_iters {
        for _ in 0..params.inner_iters.max(1) {
            // Step 1 — detrend.
            let detrended: Vec<f64> = values.iter().zip(&trend).map(|(&y, &t)| y - t).collect();

            // Step 2 — cycle-subseries smoothing, extended one cycle at each end.
            let extended =
                smooth_cycle_subseries(&detrended, period, params.seasonal_span, &robustness)?;

            // Step 3 — low-pass filter of the extended seasonal series.
            let lowpass = low_pass(&extended, period, params.lowpass_span)?;

            // Step 4 — the seasonal component is the extended series (trimmed
            // back to the data range) minus its low-frequency content.
            for (i, slot) in seasonal.iter_mut().enumerate() {
                *slot = extended[i + period] - lowpass[i];
            }

            // Step 5 — deseasonalize, Step 6 — smooth to get the trend.
            let deseasonalized: Vec<f64> =
                values.iter().zip(&seasonal).map(|(&y, &s)| y - s).collect();
            trend = loess_series(&deseasonalized, Some(&robustness), params.trend_span, 1)?;
        }

        if outer < params.outer_iters {
            let residual: Vec<f64> = (0..n).map(|i| values[i] - trend[i] - seasonal[i]).collect();
            let scale = 6.0 * median_abs(&residual);
            if !scale.is_finite() || scale <= 0.0 {
                // An exact fit: every bisquare weight would be 1, so further
                // outer iterations cannot change anything.
                break;
            }
            robustness = residual.iter().map(|&e| bisquare(e / scale)).collect();
        }
    }

    let residual: Vec<f64> = (0..n).map(|i| values[i] - trend[i] - seasonal[i]).collect();

    Ok(StlDecomposition {
        trend,
        seasonal,
        residual,
    })
}

/// Smooth every cycle-subseries with loess and evaluate it one cycle position
/// beyond each end, producing a series of length `n + 2·period` whose element
/// `i + period` corresponds to observation `i`.
fn smooth_cycle_subseries(
    detrended: &[f64],
    period: usize,
    span: usize,
    robustness: &[f64],
) -> Result<Vec<f64>> {
    let n = detrended.len();
    let extended_len = n + 2 * period;
    let mut extended = vec![f64::NAN; extended_len];

    for phase in 0..period {
        let indices: Vec<usize> = (phase..n).step_by(period).collect();
        if indices.is_empty() {
            continue;
        }
        let sub_y: Vec<f64> = indices.iter().map(|&i| detrended[i]).collect();
        let sub_w: Vec<f64> = indices.iter().map(|&i| robustness[i]).collect();
        let cycle_positions: Vec<f64> = (0..indices.len()).map(|c| c as f64).collect();
        // One position before the first cycle and one after the last.
        let targets: Vec<f64> = (0..indices.len() + 2).map(|c| c as f64 - 1.0).collect();

        let fitted = loess_series_at(&cycle_positions, &sub_y, Some(&sub_w), span, 1, &targets)?;

        for (slot, &value) in fitted.iter().enumerate() {
            let index = phase + slot * period;
            if index < extended_len {
                extended[index] = value;
            }
        }
    }

    if let Some(bad) = extended.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidOperation(format!(
            "STL cycle-subseries smoothing produced a non-finite value at extended index {bad}"
        )));
    }

    Ok(extended)
}

/// The STL low-pass filter: `MA(period) → MA(period) → MA(3) → loess`.
///
/// Each moving average is taken in "valid" mode, so the three passes trim the
/// extended seasonal series of length `n + 2·period` back to exactly `n`
/// points, aligned with the observations.
fn low_pass(extended: &[f64], period: usize, span: usize) -> Result<Vec<f64>> {
    let first = moving_average(extended, period)?;
    let second = moving_average(&first, period)?;
    let third = moving_average(&second, 3)?;
    loess_series(&third, None, span, 1)
}

/// Trailing-window ("valid") moving average of length `window`: the output has
/// `input.len() − window + 1` points.
fn moving_average(values: &[f64], window: usize) -> Result<Vec<f64>> {
    if window == 0 || values.len() < window {
        return Err(Error::InvalidInput(format!(
            "moving average of width {window} needs at least that many points, got {}",
            values.len()
        )));
    }
    let inv = 1.0 / window as f64;
    let mut out = Vec::with_capacity(values.len() - window + 1);
    let mut sum: f64 = values[..window].iter().sum();
    out.push(sum * inv);
    for i in window..values.len() {
        sum += values[i] - values[i - window];
        out.push(sum * inv);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn seasonal_plus_trend(n: usize, period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let trend: Vec<f64> = (0..n).map(|i| 10.0 + 0.05 * i as f64).collect();
        let seasonal: Vec<f64> = (0..n)
            .map(|i| 3.0 * (TAU * (i % period) as f64 / period as f64).sin())
            .collect();
        let values: Vec<f64> = trend.iter().zip(&seasonal).map(|(&t, &s)| t + s).collect();
        (values, trend, seasonal)
    }

    #[test]
    fn stl_recovers_trend_and_seasonality() {
        let period = 12;
        let n = 144;
        let (values, trend, seasonal) = seasonal_plus_trend(n, period);

        let result = stl(&values, period, &StlParams::recommended(period)).expect("stl");
        assert_eq!(result.trend.len(), n);
        assert_eq!(result.seasonal.len(), n);
        assert_eq!(result.residual.len(), n);

        // Components must add back up to the data, exactly.
        for i in 0..n {
            let sum = result.trend[i] + result.seasonal[i] + result.residual[i];
            assert!(
                (sum - values[i]).abs() < 1e-9,
                "index {i}: {sum} != {}",
                values[i]
            );
        }

        // The interior of the trend must track the true linear trend closely
        // (loess boundaries are naturally noisier).
        for i in period..(n - period) {
            assert!(
                (result.trend[i] - trend[i]).abs() < 0.5,
                "index {i}: trend {} vs truth {}",
                result.trend[i],
                trend[i]
            );
        }

        // The seasonal component must correlate almost perfectly with the truth.
        let corr = correlation(
            &result.seasonal[period..n - period],
            &seasonal[period..n - period],
        );
        assert!(corr > 0.98, "seasonal correlation {corr}");

        // The remainder must be small relative to the seasonal amplitude.
        let max_residual = result.residual.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(max_residual < 0.6, "max |residual| {max_residual}");
    }

    #[test]
    fn stl_is_not_a_classical_decomposition() {
        // A seasonal pattern whose amplitude grows over time is exactly what STL
        // can follow and a fixed (period-averaged) seasonal index cannot. The
        // seasonal component must therefore grow too.
        let period = 12;
        let n = 240;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let amplitude = 1.0 + 4.0 * i as f64 / n as f64;
                20.0 + amplitude * (TAU * (i % period) as f64 / period as f64).sin()
            })
            .collect();

        let result = stl(&values, period, &StlParams::recommended(period)).expect("stl");

        let early_amplitude = amplitude_of(&result.seasonal[period..2 * period]);
        let late_amplitude = amplitude_of(&result.seasonal[n - 2 * period..n - period]);
        assert!(
            late_amplitude > 1.8 * early_amplitude,
            "seasonal amplitude should grow: {early_amplitude} -> {late_amplitude}"
        );
    }

    #[test]
    fn robust_stl_resists_an_outlier() {
        let period = 12;
        let n = 144;
        let (mut values, _, _) = seasonal_plus_trend(n, period);
        values[70] += 60.0; // a gross spike

        let plain = stl(&values, period, &StlParams::recommended(period)).expect("stl");
        let robust = stl(&values, period, &StlParams::robust(period)).expect("stl");

        // The robust fit should leave the spike in the remainder instead of
        // bending the trend towards it.
        assert!(
            robust.residual[70].abs() > plain.residual[70].abs(),
            "robust remainder {} should absorb more of the spike than {}",
            robust.residual[70],
            plain.residual[70]
        );
        assert!(
            (robust.trend[70] - plain.trend[70]).abs() > 0.5,
            "robust trend {} should be pulled less than {}",
            robust.trend[70],
            plain.trend[70]
        );
    }

    #[test]
    fn stl_rejects_impossible_requests() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(stl(&values, 1, &StlParams::recommended(2)).is_err());
        assert!(stl(&values, 12, &StlParams::recommended(12)).is_err());
        let bad = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(stl(&bad, 2, &StlParams::recommended(2)).is_err());
    }

    #[test]
    fn next_odd_rounds_up_to_an_odd_integer() {
        assert_eq!(next_odd(1.0), 3);
        assert_eq!(next_odd(4.0), 5);
        assert_eq!(next_odd(5.0), 5);
        assert_eq!(next_odd(5.2), 7);
        assert_eq!(next_odd(12.0), 13);
    }

    #[test]
    fn moving_average_trims_exactly() {
        let v: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ma = moving_average(&v, 3).expect("ma");
        assert_eq!(ma.len(), 8);
        assert!((ma[0] - 1.0).abs() < 1e-12);
        assert!((ma[7] - 8.0).abs() < 1e-12);
        assert!(moving_average(&v, 0).is_err());
        assert!(moving_average(&v, 11).is_err());
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let cov: f64 = a.iter().zip(b).map(|(&x, &y)| (x - ma) * (y - mb)).sum();
        let va: f64 = a.iter().map(|&x| (x - ma) * (x - ma)).sum();
        let vb: f64 = b.iter().map(|&y| (y - mb) * (y - mb)).sum();
        if va <= 0.0 || vb <= 0.0 {
            return 0.0;
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    fn amplitude_of(values: &[f64]) -> f64 {
        let max = values.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        let min = values.iter().fold(f64::INFINITY, |m, &v| m.min(v));
        max - min
    }
}

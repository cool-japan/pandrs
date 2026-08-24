//! Locally-weighted polynomial regression (LOESS / LOWESS).
//!
//! This module implements Cleveland's (1979) locally-weighted scatterplot
//! smoother and the local-fit primitive it is built from. It is the shared
//! engine behind [`crate::time_series::preprocessing`]'s `Lowess` smoothing
//! method and the seasonal / trend smoothing steps of
//! [`crate::time_series::stl`].
//!
//! # Method
//!
//! For a query point `x₀` the `span` observations whose abscissae are nearest
//! to `x₀` are selected. Each is given the **tricube** weight
//! `w(u) = (1 − u³)³` for `u = |xⱼ − x₀| / d`, where `d` is the largest of those
//! distances, and a polynomial of degree `degree` is fitted by weighted least
//! squares in the local coordinate `xⱼ − x₀`. The fitted value at `x₀` is the
//! constant term of that polynomial.
//!
//! Robustness iterations (Cleveland 1979 §2) multiply the tricube weights by
//! bisquare weights derived from the current residuals, `(1 − (e/6s)²)²` with
//! `s = median|e|`, which lets outliers stop dragging the fit.

use crate::core::error::{Error, Result};

/// Fit a local polynomial at `at` and return its value there.
///
/// * `x` — abscissae, **sorted ascending** (the neighbour search relies on it).
/// * `y` — ordinates, same length as `x`.
/// * `prior_weights` — optional non-negative per-observation weights (the
///   robustness weights during a LOWESS iteration). `None` means all ones.
/// * `span` — number of nearest neighbours used for the local fit.
/// * `degree` — polynomial degree (`0` = local weighted mean, `1` = local
///   linear, `2` = local quadratic).
///
/// Returns `None` only when `x` is empty. When the local design is degenerate
/// (too few distinct abscissae with positive weight, or a singular normal
/// matrix) the degree is reduced until the fit is well posed, ending at the
/// weighted mean — never at a fabricated constant.
pub(crate) fn loess_at(
    x: &[f64],
    y: &[f64],
    prior_weights: Option<&[f64]>,
    at: f64,
    span: usize,
    degree: usize,
) -> Option<f64> {
    let n = x.len();
    if n == 0 || y.len() != n {
        return None;
    }

    let q = span.clamp(1, n);

    // Slide a window of `q` consecutive (hence nearest, since `x` is sorted)
    // observations until it is the closest one to `at`. This also does the right
    // thing when `at` lies outside `[x[0], x[n-1]]`: the window parks on the
    // corresponding end, which is how STL extends a cycle-subseries by one
    // position beyond each end of the data.
    let mut lo = 0usize;
    let mut hi = q; // exclusive
    while hi < n && (x[hi] - at) < (at - x[lo]) {
        lo += 1;
        hi += 1;
    }

    let d_max = (at - x[lo]).abs().max((x[hi - 1] - at).abs());

    let mut u = Vec::with_capacity(hi - lo);
    let mut v = Vec::with_capacity(hi - lo);
    let mut w = Vec::with_capacity(hi - lo);
    for j in lo..hi {
        let prior = prior_weights.map_or(1.0, |pw| pw.get(j).copied().unwrap_or(0.0));
        if !prior.is_finite() || prior <= 0.0 || !y[j].is_finite() {
            continue;
        }
        // All neighbours coincide with the query point: the tricube is 1 there.
        let tricube = if d_max <= 0.0 {
            1.0
        } else {
            let r = (x[j] - at).abs() / d_max;
            if r >= 1.0 {
                0.0
            } else {
                let c = 1.0 - r * r * r;
                c * c * c
            }
        };
        if tricube <= 0.0 {
            continue;
        }
        u.push(x[j] - at);
        v.push(y[j]);
        w.push(tricube * prior);
    }

    if u.is_empty() {
        // Every neighbour was zero-weighted (all-zero robustness weights, or a
        // window of coincident abscissae at the tricube cut-off). Fall back to
        // the unweighted mean of the finite ordinates in the window rather than
        // inventing a value.
        let finite: Vec<f64> = (lo..hi).map(|j| y[j]).filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return None;
        }
        return Some(finite.iter().sum::<f64>() / finite.len() as f64);
    }

    let distinct = count_distinct(&u);
    let mut deg = degree.min(distinct.saturating_sub(1));

    loop {
        if deg == 0 {
            let sw: f64 = w.iter().sum();
            if sw <= 0.0 {
                return None;
            }
            let wm: f64 = v.iter().zip(&w).map(|(&yi, &wi)| wi * yi).sum::<f64>() / sw;
            return Some(wm);
        }

        if let Some(value) = weighted_poly_intercept(&u, &v, &w, deg) {
            return Some(value);
        }
        deg -= 1;
    }
}

/// Number of distinct values in `u` (which is short, so an O(n²) scan is
/// cheaper than sorting).
fn count_distinct(u: &[f64]) -> usize {
    let mut distinct = 0usize;
    for (i, a) in u.iter().enumerate() {
        if !u[..i].iter().any(|b| b == a) {
            distinct += 1;
        }
    }
    distinct
}

/// Weighted least-squares polynomial fit of degree `deg` in the local
/// coordinate `u`, returning the intercept (the value at `u = 0`).
///
/// Solves the normal equations `(UᵀWU)c = UᵀWy` by Gaussian elimination with
/// partial pivoting, and reports `None` when the system is singular so the
/// caller can drop to a lower degree.
fn weighted_poly_intercept(u: &[f64], y: &[f64], w: &[f64], deg: usize) -> Option<f64> {
    let k = deg + 1;
    let mut a = vec![vec![0.0_f64; k]; k];
    let mut b = vec![0.0_f64; k];

    for ((&ui, &yi), &wi) in u.iter().zip(y).zip(w) {
        let mut powers = vec![1.0_f64; k];
        for p in 1..k {
            powers[p] = powers[p - 1] * ui;
        }
        for r in 0..k {
            b[r] += wi * powers[r] * yi;
            for c in 0..k {
                a[r][c] += wi * powers[r] * powers[c];
            }
        }
    }

    // Scale-aware singularity threshold: the normal matrix of a local fit on
    // data of magnitude `M` has entries of order `M`, so an absolute pivot
    // floor would reject well-conditioned small-scale problems and accept
    // ill-conditioned large-scale ones.
    let scale = a
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale <= 0.0 {
        return None;
    }
    let tol = scale * 1e-12;

    for col in 0..k {
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..k {
            if a[r][col].abs() > max_abs {
                max_abs = a[r][col].abs();
                pivot = r;
            }
        }
        if max_abs <= tol {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let diag = a[col][col];
        for j in col..k {
            a[col][j] /= diag;
        }
        b[col] /= diag;

        for r in 0..k {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    for j in col..k {
                        a[r][j] -= factor * a[col][j];
                    }
                    b[r] -= factor * b[col];
                }
            }
        }
    }

    let intercept = b[0];
    if intercept.is_finite() {
        Some(intercept)
    } else {
        None
    }
}

/// Smooth `y` observed at `x = 0, 1, …, n−1`, evaluating the local fit at every
/// observation.
pub(crate) fn loess_series(
    y: &[f64],
    prior_weights: Option<&[f64]>,
    span: usize,
    degree: usize,
) -> Result<Vec<f64>> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    loess_series_at(&x, y, prior_weights, span, degree, &x)
}

/// Smooth `y` observed at `x`, evaluating the local fit at each point of
/// `at_points` (which may lie outside the range of `x` — STL relies on that to
/// extend a cycle-subseries by one cycle at each end).
pub(crate) fn loess_series_at(
    x: &[f64],
    y: &[f64],
    prior_weights: Option<&[f64]>,
    span: usize,
    degree: usize,
    at_points: &[f64],
) -> Result<Vec<f64>> {
    if x.len() != y.len() {
        return Err(Error::InvalidInput(format!(
            "loess needs matching abscissae and ordinates, got {} and {}",
            x.len(),
            y.len()
        )));
    }
    if x.is_empty() {
        return Err(Error::InvalidInput(
            "loess needs at least one observation".to_string(),
        ));
    }

    at_points
        .iter()
        .map(|&at| {
            loess_at(x, y, prior_weights, at, span, degree).ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "loess could not fit a local model at {at}: no usable observation in the \
                     neighbourhood"
                ))
            })
        })
        .collect()
}

/// Cleveland's LOWESS: a robust locally-weighted scatterplot smoother.
///
/// * `fraction` — the span, as a fraction of the sample size, used for each
///   local fit. Must lie in `(0, 1]`.
/// * `robustness_iters` — number of bisquare re-weighting passes. `0` gives the
///   plain (non-robust) LOESS fit; Cleveland's recommendation is `2`–`3`.
/// * `degree` — local polynomial degree (`1` is the classical choice).
///
/// # Errors
/// Returns [`Error::InvalidInput`] when the inputs have different lengths, are
/// empty, contain a non-finite value, when `x` is not sorted ascending, or when
/// `fraction` is outside `(0, 1]`.
pub(crate) fn lowess(
    x: &[f64],
    y: &[f64],
    fraction: f64,
    robustness_iters: usize,
    degree: usize,
) -> Result<Vec<f64>> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::InvalidInput(format!(
            "LOWESS needs matching abscissae and ordinates, got {} and {}",
            n,
            y.len()
        )));
    }
    if n == 0 {
        return Err(Error::InvalidInput(
            "LOWESS needs at least one observation".to_string(),
        ));
    }
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(Error::InvalidInput(format!(
            "LOWESS span fraction must lie in (0, 1], got {fraction}"
        )));
    }
    if let Some(bad) = x.iter().chain(y.iter()).position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "LOWESS requires finite inputs; element {bad} is not finite"
        )));
    }
    if x.windows(2).any(|w| w[1] < w[0]) {
        return Err(Error::InvalidInput(
            "LOWESS requires abscissae sorted in ascending order".to_string(),
        ));
    }

    // The tricube gives the farthest neighbour weight exactly zero, so a span of
    // `q` supplies `q − 1` effectively-weighted points; ask for enough of them
    // to identify a degree-`degree` polynomial.
    let span = ((fraction * n as f64).round() as usize).clamp((degree + 2).min(n), n);

    let mut robustness: Option<Vec<f64>> = None;
    let mut fitted = vec![0.0_f64; n];

    for _ in 0..=robustness_iters {
        for (i, slot) in fitted.iter_mut().enumerate() {
            *slot = loess_at(x, y, robustness.as_deref(), x[i], span, degree).ok_or_else(|| {
                Error::InvalidOperation(format!("LOWESS could not fit a local model at index {i}"))
            })?;
        }

        let residuals: Vec<f64> = y.iter().zip(&fitted).map(|(&yi, &fi)| yi - fi).collect();
        let scale = median_abs(&residuals);
        if !scale.is_finite() || scale <= 0.0 {
            // A perfect (or degenerate) fit: further re-weighting is undefined
            // — every bisquare weight would be 1 — so stop here.
            break;
        }
        robustness = Some(
            residuals
                .iter()
                .map(|&e| bisquare(e / (6.0 * scale)))
                .collect(),
        );
    }

    Ok(fitted)
}

/// Bisquare (Tukey biweight) robustness weight: `(1 − u²)²` inside `|u| < 1`.
pub(crate) fn bisquare(u: f64) -> f64 {
    if !u.is_finite() {
        return 0.0;
    }
    let a = u.abs();
    if a >= 1.0 {
        0.0
    } else {
        let c = 1.0 - a * a;
        c * c
    }
}

/// Median of the absolute values of `values`, ignoring non-finite entries.
pub(crate) fn median_abs(values: &[f64]) -> f64 {
    let mut abs: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if abs.is_empty() {
        return 0.0;
    }
    abs.sort_by(|a, b| a.total_cmp(b));
    let mid = abs.len() / 2;
    if abs.len() % 2 == 0 {
        (abs[mid - 1] + abs[mid]) / 2.0
    } else {
        abs[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loess_reproduces_a_straight_line_exactly() {
        // A local *linear* fit is exact on data that is globally linear, at any
        // span, including at the boundaries where the window is asymmetric.
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 3.0 * v - 7.0).collect();

        let fitted = lowess(&x, &y, 0.3, 0, 1).expect("lowess");
        for (i, (&f, &expected)) in fitted.iter().zip(&y).enumerate() {
            assert!(
                (f - expected).abs() < 1e-8,
                "index {i}: fitted {f} vs exact {expected}"
            );
        }
    }

    #[test]
    fn loess_recovers_a_smooth_curve_from_noise() {
        // Deterministic pseudo-noise around a slow sine: the smoother must be
        // much closer to the underlying signal than the noisy data is.
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let signal: Vec<f64> = x
            .iter()
            .map(|&v| (v * std::f64::consts::TAU / 300.0).sin())
            .collect();
        let noisy: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| s + 0.3 * ((i as f64 * 12.9898).sin() * 43758.5453).fract())
            .collect();

        let fitted = lowess(&x, &noisy, 0.2, 2, 1).expect("lowess");

        let err_fit: f64 = fitted
            .iter()
            .zip(&signal)
            .map(|(&f, &s)| (f - s) * (f - s))
            .sum();
        let err_raw: f64 = noisy
            .iter()
            .zip(&signal)
            .map(|(&f, &s)| (f - s) * (f - s))
            .sum();
        assert!(
            err_fit < err_raw / 3.0,
            "smoothed SSE {err_fit} should be far below raw SSE {err_raw}"
        );
    }

    #[test]
    fn robustness_iterations_reject_an_outlier() {
        let x: Vec<f64> = (0..41).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&v| 2.0 * v).collect();
        y[20] = 1000.0; // a gross outlier at the centre

        let plain = lowess(&x, &y, 0.4, 0, 1).expect("lowess");
        let robust = lowess(&x, &y, 0.4, 3, 1).expect("lowess");

        let truth = 40.0;
        assert!(
            (robust[20] - truth).abs() < (plain[20] - truth).abs(),
            "robust fit {} should beat the non-robust fit {} at the outlier",
            robust[20],
            plain[20]
        );
        assert!(
            (robust[20] - truth).abs() < 5.0,
            "robust fit {} should be near the underlying line value {truth}",
            robust[20]
        );
    }

    #[test]
    fn loess_extrapolates_one_step_beyond_the_data() {
        // STL evaluates each cycle-subseries one position outside its range.
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 5.0 + 2.0 * v).collect();

        let at = vec![-1.0, 10.0];
        let out = loess_series_at(&x, &y, None, 5, 1, &at).expect("loess");
        assert!((out[0] - 3.0).abs() < 1e-8, "left extension {}", out[0]);
        assert!((out[1] - 25.0).abs() < 1e-8, "right extension {}", out[1]);
    }

    #[test]
    fn lowess_rejects_bad_arguments() {
        let x = [0.0, 1.0, 2.0];
        let y = [1.0, 2.0, 3.0];
        assert!(lowess(&x, &y, 0.0, 0, 1).is_err());
        assert!(lowess(&x, &y, 1.5, 0, 1).is_err());
        assert!(lowess(&x, &[1.0, 2.0], 0.5, 0, 1).is_err());
        assert!(lowess(&[2.0, 1.0, 0.0], &y, 0.5, 0, 1).is_err());
        assert!(lowess(&x, &[1.0, f64::NAN, 3.0], 0.5, 0, 1).is_err());
    }

    #[test]
    fn constant_input_is_reproduced_exactly() {
        let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y = vec![4.25_f64; 15];
        let fitted = lowess(&x, &y, 0.5, 2, 1).expect("lowess");
        for f in fitted {
            assert!((f - 4.25).abs() < 1e-12);
        }
    }
}

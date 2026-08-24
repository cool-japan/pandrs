//! Signal-extraction filters: the Hodrick-Prescott penalized least-squares
//! filter and a local-level state-space (Kalman) smoother.
//!
//! Both estimate a slowly-varying signal from a noisy series, and both are
//! exact — neither is a moving average wearing another algorithm's name.

use crate::core::error::{Error, Result};

/// Hodrick-Prescott trend filter.
///
/// Returns the trend `τ` minimizing
///
/// ```text
///   Σₜ (yₜ − τₜ)²  +  λ Σₜ ((τₜ₊₁ − τₜ) − (τₜ − τₜ₋₁))²
/// ```
///
/// whose first-order condition is the linear system `(I + λ·DᵀD)·τ = y` with
/// `D` the `(n−2)×n` second-difference matrix. `I + λDᵀD` is symmetric,
/// positive definite and **pentadiagonal**, so it is factored here by a banded
/// Cholesky decomposition in `O(n)` time and memory — not by forming or
/// inverting a dense `n×n` matrix.
///
/// `λ` controls smoothness: the conventional values are 100 (annual data),
/// 1 600 (quarterly) and 14 400 (monthly). `λ = 0` returns the input unchanged,
/// which is the mathematically correct limit (no smoothness penalty).
///
/// # Errors
/// Returns [`Error::InvalidInput`] when `λ` is negative or non-finite or the
/// series contains a non-finite value, and [`Error::InvalidOperation`] if the
/// factorization loses positive definiteness (only reachable through extreme
/// `λ` overflow).
pub(crate) fn hodrick_prescott(values: &[f64], lambda: f64) -> Result<Vec<f64>> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::InvalidInput(format!(
            "Hodrick-Prescott lambda must be finite and non-negative, got {lambda}"
        )));
    }
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "Hodrick-Prescott filter requires finite values; index {bad} is {}",
            values[bad]
        )));
    }

    let n = values.len();
    // With fewer than three observations the second-difference penalty has no
    // terms at all, so the minimizer is the data itself. Same for lambda = 0.
    if n < 3 || lambda == 0.0 {
        return Ok(values.to_vec());
    }

    // Symmetric band storage with half-bandwidth 2: `ab[i][k] = A[i][i + k]`.
    let mut ab = vec![[0.0_f64; 3]; n];
    for row in ab.iter_mut() {
        row[0] = 1.0; // the identity term
    }
    // Accumulate lambda * DᵀD one second-difference row at a time; the row for
    // t has coefficients (1, −2, 1) on columns (t, t+1, t+2).
    const COEFF: [f64; 3] = [1.0, -2.0, 1.0];
    for t in 0..(n - 2) {
        for a in 0..3 {
            for b in a..3 {
                ab[t + a][b - a] += lambda * COEFF[a] * COEFF[b];
            }
        }
    }

    let r = band_cholesky(&ab)?;
    Ok(band_solve(&r, values))
}

/// Cholesky factorization `A = RᵀR` of a symmetric positive-definite band
/// matrix with half-bandwidth 2, in the same band storage as the input
/// (`r[i][k] = R[i][i + k]`).
fn band_cholesky(ab: &[[f64; 3]]) -> Result<Vec<[f64; 3]>> {
    let n = ab.len();
    let m = 2usize;
    let mut r = vec![[0.0_f64; 3]; n];

    for i in 0..n {
        let mut diag = ab[i][0];
        for p in 1..=m.min(i) {
            diag -= r[i - p][p] * r[i - p][p];
        }
        if !(diag.is_finite() && diag > 0.0) {
            return Err(Error::InvalidOperation(format!(
                "band Cholesky lost positive definiteness at row {i} (pivot {diag})"
            )));
        }
        r[i][0] = diag.sqrt();

        for k in 1..=m.min(n - 1 - i) {
            let mut sum = ab[i][k];
            for p in 1..=(m - k).min(i) {
                sum -= r[i - p][p] * r[i - p][p + k];
            }
            r[i][k] = sum / r[i][0];
        }
    }

    Ok(r)
}

/// Solve `RᵀR x = b` given the band Cholesky factor `r`.
fn band_solve(r: &[[f64; 3]], b: &[f64]) -> Vec<f64> {
    let n = r.len();
    let m = 2usize;

    // Forward substitution: Rᵀ z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for p in 1..=m.min(i) {
            sum -= r[i - p][p] * z[i - p];
        }
        z[i] = sum / r[i][0];
    }

    // Back substitution: R x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = z[i];
        for k in 1..=m.min(n - 1 - i) {
            sum -= r[i][k] * x[i + k];
        }
        x[i] = sum / r[i][0];
    }

    x
}

/// Result of fitting the local-level model to a series.
#[derive(Debug, Clone)]
pub(crate) struct LocalLevelFit {
    /// Smoothed level `E[αₜ | y₁…y_n]` at every observation.
    pub level: Vec<f64>,
    /// Maximum-likelihood signal-to-noise ratio `q = σ²_η / σ²_ε`.
    pub signal_to_noise: f64,
    /// Concentrated maximum-likelihood observation variance `σ²_ε`.
    pub observation_variance: f64,
}

/// Kalman smoothing of a series under the **local-level** (random-walk-plus-
/// noise) state-space model
///
/// ```text
///   yₜ = αₜ + εₜ,        εₜ ~ N(0, σ²_ε)
///   αₜ₊₁ = αₜ + ηₜ,      ηₜ ~ N(0, q·σ²_ε)
/// ```
///
/// The single free parameter — the signal-to-noise ratio `q` — is **estimated
/// from the data** by maximizing the diffuse concentrated likelihood, so the
/// amount of smoothing adapts to the series instead of being a hidden constant.
/// The state is then extracted by the Kalman filter followed by the
/// Rauch-Tung-Striebel (equivalently, Durbin & Koopman's) backward state
/// smoother, so every returned level uses the *whole* sample, not just its
/// past.
///
/// The filter is initialized by conditioning on the first observation
/// (`a₁ = y₁` with a diffuse prior variance), and the likelihood is evaluated
/// over `t = 2…n` accordingly — the standard exact-diffuse treatment for a
/// non-stationary state.
///
/// # Errors
/// Returns [`Error::InvalidInput`] when the series is empty or contains a
/// non-finite value.
pub(crate) fn local_level_smooth(values: &[f64]) -> Result<LocalLevelFit> {
    let n = values.len();
    if n == 0 {
        return Err(Error::InvalidInput(
            "Kalman smoothing needs at least one observation".to_string(),
        ));
    }
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "Kalman smoothing requires finite values; index {bad} is {}",
            values[bad]
        )));
    }

    // A constant series (and the degenerate single observation) carries no
    // information about q: the level is the series itself and both variances
    // are zero. Reporting that honestly beats returning an arbitrary q.
    if n == 1 || values.windows(2).all(|w| w[0] == w[1]) {
        return Ok(LocalLevelFit {
            level: values.to_vec(),
            signal_to_noise: 0.0,
            observation_variance: 0.0,
        });
    }

    let log_q = maximize_local_level_likelihood(values);
    let q = log_q.exp();
    let state = filter_local_level(values, q);
    let level = smooth_local_level(&state);

    Ok(LocalLevelFit {
        level,
        signal_to_noise: q,
        observation_variance: state.concentrated_variance,
    })
}

/// Filtered quantities of the local-level model, in units of `σ²_ε`.
struct LocalLevelState {
    /// One-step-ahead state predictions `aₜ`.
    a: Vec<f64>,
    /// Prediction variances `Pₜ` (scaled by `σ²_ε`).
    p: Vec<f64>,
    /// Prediction errors `vₜ = yₜ − aₜ`.
    v: Vec<f64>,
    /// Prediction error variances `Fₜ = Pₜ + 1` (scaled by `σ²_ε`).
    f: Vec<f64>,
    /// Kalman gains `Kₜ = Pₜ / Fₜ`.
    k: Vec<f64>,
    /// Concentrated ML estimate of `σ²_ε`.
    concentrated_variance: f64,
    /// `(n − 1)·ln σ̂²_ε + Σ_{t≥2} ln Fₜ`, the profile objective to minimize.
    objective: f64,
}

/// Diffuse prior variance for the initial level, in units of `σ²_ε`. Large
/// enough that the first update is effectively `a₂ = y₁, P₂ = 1 + q` (the exact
/// diffuse result) and small enough to stay far from floating-point trouble.
const DIFFUSE_PRIOR: f64 = 1e7;

fn filter_local_level(values: &[f64], q: f64) -> LocalLevelState {
    let n = values.len();
    let mut a = vec![0.0_f64; n];
    let mut p = vec![0.0_f64; n];
    let mut v = vec![0.0_f64; n];
    let mut f = vec![0.0_f64; n];
    let mut k = vec![0.0_f64; n];

    a[0] = values[0];
    p[0] = DIFFUSE_PRIOR;

    let mut sum_scaled_sq = 0.0_f64;
    let mut sum_log_f = 0.0_f64;

    for t in 0..n {
        v[t] = values[t] - a[t];
        f[t] = p[t] + 1.0;
        k[t] = p[t] / f[t];
        // t = 0 carries the diffuse prior, so it contributes to neither the
        // concentrated variance nor the log-determinant.
        if t > 0 {
            sum_scaled_sq += v[t] * v[t] / f[t];
            sum_log_f += f[t].ln();
        }
        if t + 1 < n {
            a[t + 1] = a[t] + k[t] * v[t];
            p[t + 1] = p[t] * (1.0 - k[t]) + q;
        }
    }

    let effective = (n - 1) as f64;
    let concentrated_variance = sum_scaled_sq / effective;
    let objective = if concentrated_variance > 0.0 && concentrated_variance.is_finite() {
        effective * concentrated_variance.ln() + sum_log_f
    } else {
        f64::INFINITY
    };

    LocalLevelState {
        a,
        p,
        v,
        f,
        k,
        concentrated_variance,
        objective,
    }
}

/// Durbin & Koopman's backward state smoother:
/// `r_{t−1} = vₜ/Fₜ + (1 − Kₜ)·r_t`, `α̂ₜ = aₜ + Pₜ·r_{t−1}`.
fn smooth_local_level(state: &LocalLevelState) -> Vec<f64> {
    let n = state.a.len();
    let mut level = vec![0.0_f64; n];
    let mut r = 0.0_f64;
    for t in (0..n).rev() {
        r = state.v[t] / state.f[t] + (1.0 - state.k[t]) * r;
        level[t] = state.a[t] + state.p[t] * r;
    }
    level
}

/// Maximize the diffuse concentrated likelihood over `ln q`.
///
/// A coarse grid brackets the optimum (the profile objective is unimodal in
/// `ln q` for this model, but a grid pass makes the search independent of that
/// assumption over the searched range), then golden-section refines it.
fn maximize_local_level_likelihood(values: &[f64]) -> f64 {
    const LO: f64 = -18.0;
    const HI: f64 = 18.0;
    const GRID: usize = 72;

    let objective = |log_q: f64| filter_local_level(values, log_q.exp()).objective;

    let mut best_idx = 0usize;
    let mut best = f64::INFINITY;
    let mut grid = Vec::with_capacity(GRID + 1);
    for i in 0..=GRID {
        let log_q = LO + (HI - LO) * i as f64 / GRID as f64;
        let value = objective(log_q);
        if value < best {
            best = value;
            best_idx = i;
        }
        grid.push(log_q);
    }
    if !best.is_finite() {
        // Nothing in the search range produced a usable likelihood: fall back to
        // the smallest ratio, i.e. essentially no state noise.
        return LO;
    }

    let mut lo = grid[best_idx.saturating_sub(1)];
    let mut hi = grid[(best_idx + 1).min(GRID)];

    // Golden-section search: each pass shrinks the bracket by ~38%, so 60
    // passes take a bracket of width 0.5 below 1e-12.
    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - inv_phi * (hi - lo);
    let mut d = lo + inv_phi * (hi - lo);
    let mut fc = objective(c);
    let mut fd = objective(d);
    for _ in 0..60 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - inv_phi * (hi - lo);
            fc = objective(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + inv_phi * (hi - lo);
            fd = objective(d);
        }
    }

    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense reference solve of `(I + λDᵀD)τ = y`, used to prove the banded
    /// Cholesky path agrees with the definition of the HP filter.
    fn hp_reference(values: &[f64], lambda: f64) -> Vec<f64> {
        let n = values.len();
        let mut a = vec![vec![0.0_f64; n]; n];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        for t in 0..(n - 2) {
            let idx = [t, t + 1, t + 2];
            let c = [1.0, -2.0, 1.0];
            for p in 0..3 {
                for r in 0..3 {
                    a[idx[p]][idx[r]] += lambda * c[p] * c[r];
                }
            }
        }

        // Gauss-Jordan with partial pivoting.
        let mut b = values.to_vec();
        for col in 0..n {
            let mut pivot = col;
            for r in (col + 1)..n {
                if a[r][col].abs() > a[pivot][col].abs() {
                    pivot = r;
                }
            }
            a.swap(col, pivot);
            b.swap(col, pivot);
            let diag = a[col][col];
            for j in col..n {
                a[col][j] /= diag;
            }
            b[col] /= diag;
            for r in 0..n {
                if r != col {
                    let factor = a[r][col];
                    for j in col..n {
                        a[r][j] -= factor * a[col][j];
                    }
                    b[r] -= factor * b[col];
                }
            }
        }
        b
    }

    #[test]
    fn hp_matches_the_dense_normal_equations() {
        let values: Vec<f64> = (0..40)
            .map(|i| 10.0 + 0.5 * i as f64 + (i as f64 * 0.7).sin() * 3.0)
            .collect();
        for &lambda in &[1.0_f64, 100.0, 1600.0, 14400.0] {
            let banded = hodrick_prescott(&values, lambda).expect("hp");
            let dense = hp_reference(&values, lambda);
            for (i, (&b, &d)) in banded.iter().zip(&dense).enumerate() {
                assert!(
                    (b - d).abs() < 1e-8 * d.abs().max(1.0),
                    "lambda {lambda}, index {i}: banded {b} vs dense {d}"
                );
            }
        }
    }

    #[test]
    fn hp_leaves_a_straight_line_untouched() {
        // A line has zero second differences, so it pays no penalty and is its
        // own HP trend at every lambda.
        let values: Vec<f64> = (0..30).map(|i| 4.0 + 2.5 * i as f64).collect();
        let trend = hodrick_prescott(&values, 1600.0).expect("hp");
        for (i, (&t, &v)) in trend.iter().zip(&values).enumerate() {
            assert!((t - v).abs() < 1e-6, "index {i}: {t} vs {v}");
        }
    }

    #[test]
    fn hp_lambda_controls_smoothness() {
        let values: Vec<f64> = (0..60)
            .map(|i| i as f64 * 0.1 + if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let roughness = |v: &[f64]| -> f64 {
            v.windows(3)
                .map(|w| (w[2] - 2.0 * w[1] + w[0]).powi(2))
                .sum()
        };
        let light = hodrick_prescott(&values, 10.0).expect("hp");
        let heavy = hodrick_prescott(&values, 10_000.0).expect("hp");
        assert!(
            roughness(&heavy) < roughness(&light),
            "heavier lambda must yield a smoother trend: {} vs {}",
            roughness(&heavy),
            roughness(&light)
        );
        assert!(roughness(&light) < roughness(&values));
    }

    #[test]
    fn hp_rejects_bad_arguments() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert!(hodrick_prescott(&values, -1.0).is_err());
        assert!(hodrick_prescott(&values, f64::NAN).is_err());
        assert!(hodrick_prescott(&[1.0, f64::INFINITY, 3.0], 100.0).is_err());
        // lambda = 0 is the identity filter, not an error.
        assert_eq!(hodrick_prescott(&values, 0.0).expect("hp"), values.to_vec());
    }

    #[test]
    fn kalman_shrinks_pure_observation_noise() {
        // Constant level plus alternating measurement error: the ML fit should
        // recognise there is no state movement and average the noise away.
        let truth = 7.0_f64;
        let values: Vec<f64> = (0..60)
            .map(|i| truth + if i % 2 == 0 { 0.8 } else { -0.8 })
            .collect();

        let fit = local_level_smooth(&values).expect("kalman");
        assert_eq!(fit.level.len(), values.len());

        let sse_fit: f64 = fit.level.iter().map(|&a| (a - truth) * (a - truth)).sum();
        let sse_raw: f64 = values.iter().map(|&a| (a - truth) * (a - truth)).sum();
        assert!(
            sse_fit < sse_raw / 10.0,
            "smoothed SSE {sse_fit} vs raw SSE {sse_raw}"
        );
        assert!(
            fit.signal_to_noise < 0.1,
            "a constant level should give a small signal-to-noise ratio, got {}",
            fit.signal_to_noise
        );
        assert!(fit.observation_variance > 0.0);
    }

    #[test]
    fn kalman_still_tracks_a_level_shift() {
        // The smoother must not flatten a genuine break: it uses the whole
        // sample, so the post-break level has to sit far above the pre-break one.
        let values: Vec<f64> = (0..60)
            .map(|i| {
                let level = if i < 30 { 5.0 } else { 15.0 };
                level + if i % 2 == 0 { 0.4 } else { -0.4 }
            })
            .collect();

        let fit = local_level_smooth(&values).expect("kalman");
        assert!(
            fit.level[50] - fit.level[10] > 8.0,
            "level shift lost: {} -> {}",
            fit.level[10],
            fit.level[50]
        );
    }

    #[test]
    fn kalman_on_a_pure_random_walk_tracks_the_data() {
        // With no observation noise the ML fit should push q up and follow the
        // series closely.
        let mut values = vec![0.0_f64];
        let mut state = 0.0_f64;
        for i in 1..80 {
            state += ((i as f64 * 1.7).sin() * 2.0).round();
            values.push(state);
        }
        let fit = local_level_smooth(&values).expect("kalman");
        let sse: f64 = fit
            .level
            .iter()
            .zip(&values)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let total_variation = {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
        };
        assert!(
            sse < total_variation * 0.5,
            "smoothed level should track a random walk: SSE {sse} vs total variation \
             {total_variation}"
        );
    }

    #[test]
    fn kalman_handles_degenerate_input() {
        let constant = vec![3.0_f64; 20];
        let fit = local_level_smooth(&constant).expect("kalman");
        assert_eq!(fit.level, constant);
        assert_eq!(fit.signal_to_noise, 0.0);

        assert!(local_level_smooth(&[]).is_err());
        assert!(local_level_smooth(&[1.0, f64::NAN]).is_err());
    }
}

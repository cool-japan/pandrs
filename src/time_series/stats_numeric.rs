//! Numerical support routines shared across the time-series statistical
//! tests: the Dickey-Fuller regression, long-run variance estimation, the
//! normal quantile, and the published critical-value tables.
//!
//! Split out of `stats.rs` purely to keep both files well under the 2000-line
//! guideline; `stats.rs` re-exports every crate-visible item so existing
//! `crate::time_series::stats::…` paths keep working.

use crate::core::error::{Error, Result};
use crate::time_series::analysis::ols_with_std_errors;

// All tail probabilities route through `crate::stats::special`, the crate's
// single numerically-correct special-function module. The in-file copies that
// previously lived here (a Lanczos `log_gamma`, a buggy upper-incomplete-gamma
// continued fraction, and ad-hoc `chi2_sf` / `normal_sf` approximations) have
// been removed so every test shares one consistent implementation.

/// Compute the Augmented Dickey-Fuller t-statistic on the lagged-level
/// coefficient from the OLS regression
///   Δyₜ = α + β·yₜ₋₁ + Σⱼ γⱼ·Δyₜ₋ⱼ + εₜ   (constant, no trend).
pub(super) fn adf_regression_statistic(values: &[f64], lags: usize) -> Result<f64> {
    let n = values.len();
    let dy: Vec<f64> = (1..n).map(|i| values[i] - values[i - 1]).collect();

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

    let beta = coefficients[1];
    let se = std_errors[1];
    if !se.is_finite() || se <= 0.0 {
        return Err(Error::InvalidInput(
            "ADF regression produced a degenerate standard error".to_string(),
        ));
    }

    Ok(beta / se)
}

/// Approximate ADF p-value by monotone interpolation of the MacKinnon
/// constant-only critical-value surface. Documented as an approximation: the
/// Dickey-Fuller distribution has no elementary closed form.
pub(super) fn adf_p_value(statistic: f64) -> f64 {
    let anchors = [(-3.43_f64, 0.01_f64), (-2.86, 0.05), (-2.57, 0.10)];

    if statistic <= anchors[0].0 {
        let slope = (anchors[1].1 - anchors[0].1) / (anchors[1].0 - anchors[0].0);
        (anchors[0].1 + slope * (statistic - anchors[0].0)).clamp(0.0001, 0.01)
    } else if statistic <= anchors[1].0 {
        let t = (statistic - anchors[0].0) / (anchors[1].0 - anchors[0].0);
        anchors[0].1 + t * (anchors[1].1 - anchors[0].1)
    } else if statistic <= anchors[2].0 {
        let t = (statistic - anchors[1].0) / (anchors[2].0 - anchors[1].0);
        anchors[1].1 + t * (anchors[2].1 - anchors[1].1)
    } else {
        let slope = (anchors[2].1 - anchors[1].1) / (anchors[2].0 - anchors[1].0);
        (anchors[2].1 + slope * (statistic - anchors[2].0)).clamp(0.10, 0.999)
    }
}

/// Bartlett-kernel (Newey-West) long-run variance estimator:
///   σ² = γ₀ + 2 Σ_{j=1}^{l} (1 − j/(l+1)) γⱼ,   γⱼ = (1/n) Σ_t eₜ·eₜ₋ⱼ.
///
/// This is the crate's single long-run variance implementation; the KPSS test
/// in [`crate::time_series::analysis`] and the Phillips-Perron correction below
/// both route through it.
pub(crate) fn newey_west_long_run_variance(residuals: &[f64], bandwidth: usize) -> f64 {
    let n = residuals.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut lrv = residuals.iter().map(|e| e * e).sum::<f64>() / nf; // γ₀

    for j in 1..=bandwidth {
        if j >= n {
            break;
        }
        let mut gamma_j = 0.0;
        for t in j..n {
            gamma_j += residuals[t] * residuals[t - j];
        }
        gamma_j /= nf;
        let weight = 1.0 - (j as f64) / ((bandwidth + 1) as f64);
        lrv += 2.0 * weight * gamma_j;
    }

    lrv
}

/// Horner evaluation of a polynomial `c[0] + c[1]·x + c[2]·x² + …`.
pub(crate) fn poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Automatic Newey-West/Bartlett bandwidth following Newey & West (1994):
/// `l = floor(4 · (n/100)^{2/9})`. Used by the Phillips-Perron correction; the
/// KPSS test in this file deliberately keeps the Schwert `4·(n/100)^{1/4}`
/// rule, which is what its own literature specifies.
pub(crate) fn newey_west_bandwidth(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let l = 4.0 * (n as f64 / 100.0).powf(2.0 / 9.0);
    (l.floor() as usize).min(n.saturating_sub(1))
}

/// Inverse standard-normal CDF (probit) via Acklam's rational approximation
/// (relative error < 1.2e-9).
///
/// This is the time-series module's single normal-quantile implementation:
/// the Shapiro-Wilk normal scores, every forecast prediction interval, and the
/// Durbin-Watson small-sample fallback all call it. `crate::stats::special`
/// deliberately exposes no normal quantile (only `normal_cdf`/`normal_sf` and
/// the χ²/t/F quantiles), so adding a fourth ad-hoc `1.96` ladder would have
/// been the alternative.
pub(crate) fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Two-sided normal critical value `z` for a confidence level, i.e.
/// `Φ⁻¹((1 + confidence_level) / 2)`.
///
/// Every forecaster's prediction interval routes through this. It replaces the
/// `match (confidence_level * 100.0) as i32 { 90 => 1.645, 95 => 1.96,
/// 99 => 2.576, _ => 1.96 }` ladders that used to sit in each forecaster: a
/// request for an 80% interval silently produced a 95% interval while the
/// result still reported `confidence_level: 0.80`.
///
/// # Errors
/// Returns [`Error::InvalidInput`] unless `0 < confidence_level < 1`.
pub(crate) fn normal_critical_value(confidence_level: f64) -> Result<f64> {
    if !(confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(Error::InvalidInput(format!(
            "confidence_level must lie strictly between 0 and 1, got {confidence_level}"
        )));
    }
    Ok(inv_normal_cdf(0.5 * (1.0 + confidence_level)))
}

/// KPSS p-value by monotone interpolation of the published critical-value
/// table, clamped to `[0.01, 0.10]` outside the tabulated range.
///
/// `c10`, `c5` and `c1` are the 10%, 5% and 1% critical values for the trend
/// specification under test (Kwiatkowski et al. 1992, Table 1). This is the
/// crate's single KPSS p-value routine: the three-level step ladder that used
/// to live in [`crate::time_series::analysis`] reported literally `0.10`,
/// `0.05` or `0.01` and nothing in between.
/// KPSS asymptotic critical values `(1%, 5%, 10%)` for a trend specification
/// (Kwiatkowski, Phillips, Schmidt & Shin 1992, Table 1).
///
/// # Errors
/// Returns [`Error::InvalidInput`] for any specification other than
/// `"constant"` (level stationarity) or `"linear"` (trend stationarity).
pub(crate) fn kpss_critical_values(trend: &str) -> Result<(f64, f64, f64)> {
    match trend {
        "constant" => Ok((0.739, 0.463, 0.347)),
        "linear" => Ok((0.216, 0.146, 0.119)),
        other => Err(Error::InvalidInput(format!(
            "KPSS trend specification must be \"constant\" or \"linear\", got \"{other}\""
        ))),
    }
}

pub(crate) fn kpss_p_value_from_table(statistic: f64, c10: f64, c5: f64, c1: f64) -> f64 {
    if statistic <= c10 {
        0.10
    } else if statistic <= c5 {
        let t = (statistic - c10) / (c5 - c10);
        0.10 + t * (0.05 - 0.10)
    } else if statistic <= c1 {
        let t = (statistic - c5) / (c1 - c5);
        0.05 + t * (0.01 - 0.05)
    } else {
        0.01
    }
}

/// Mid-ranks (average ranks for ties) of `row`, in the order of `row`.
pub(super) fn average_ranks(row: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));

    let mut ranks = vec![0.0; row.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && row[order[j]] == row[order[i]] {
            j += 1;
        }
        // Ranks i+1 ..= j averaged over the tie group.
        let mid = ((i + 1 + j) as f64) / 2.0;
        for &idx in &order[i..j] {
            ranks[idx] = mid;
        }
        i = j;
    }
    ranks
}

/// `Σ (tᵢ³ − tᵢ)` over the tie groups of `row`, the numerator of the standard
/// rank-test tie correction.
pub(super) fn tie_sum(row: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = row.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    let mut total = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let t = (j - i) as f64;
        total += t * t * t - t;
        i = j;
    }
    total
}

/// Durbin-Watson 5% bounds `(d_L, d_U)` for `n` observations and one regressor
/// besides the intercept (`k' = 1`).
///
/// Values are Savin & White (1977), Table 1, linearly interpolated between the
/// tabulated sample sizes. Outside the tabulated range (`n < 15`, `n > 200`)
/// the asymptotic approximation `d ≈ 2(1 − ρ̂)` with `ρ̂ ~ N(0, 1/n)` is used:
/// `d_L = d_U = 2(1 − z_{0.975}/√n)`, which collapses the inconclusive region
/// to a point — honest for a regime the table does not cover.
pub(super) fn durbin_watson_bounds(n: usize) -> (f64, f64) {
    /// (n, d_L, d_U) at α = 0.05, k' = 1.
    const TABLE: [(f64, f64, f64); 43] = [
        (15.0, 1.077, 1.361),
        (16.0, 1.106, 1.371),
        (17.0, 1.133, 1.381),
        (18.0, 1.158, 1.391),
        (19.0, 1.180, 1.401),
        (20.0, 1.201, 1.411),
        (21.0, 1.221, 1.420),
        (22.0, 1.239, 1.429),
        (23.0, 1.257, 1.437),
        (24.0, 1.273, 1.446),
        (25.0, 1.288, 1.454),
        (26.0, 1.302, 1.461),
        (27.0, 1.316, 1.469),
        (28.0, 1.328, 1.476),
        (29.0, 1.341, 1.483),
        (30.0, 1.352, 1.489),
        (31.0, 1.363, 1.496),
        (32.0, 1.373, 1.502),
        (33.0, 1.383, 1.508),
        (34.0, 1.393, 1.514),
        (35.0, 1.402, 1.519),
        (36.0, 1.411, 1.525),
        (37.0, 1.419, 1.530),
        (38.0, 1.427, 1.535),
        (39.0, 1.435, 1.540),
        (40.0, 1.442, 1.544),
        (45.0, 1.475, 1.566),
        (50.0, 1.503, 1.585),
        (55.0, 1.528, 1.601),
        (60.0, 1.549, 1.616),
        (65.0, 1.567, 1.629),
        (70.0, 1.583, 1.641),
        (75.0, 1.598, 1.652),
        (80.0, 1.611, 1.662),
        (85.0, 1.624, 1.671),
        (90.0, 1.635, 1.679),
        (95.0, 1.645, 1.687),
        (100.0, 1.654, 1.694),
        (150.0, 1.720, 1.746),
        (200.0, 1.758, 1.778),
        (500.0, 1.845, 1.851),
        (1000.0, 1.888, 1.891),
        (2000.0, 1.921, 1.922),
    ];

    let nf = n as f64;
    let first = TABLE[0];
    let last = TABLE[TABLE.len() - 1];

    if nf < first.0 || nf > last.0 {
        // Asymptotic fallback: d = 2(1 − ρ̂), ρ̂ ≈ N(0, 1/n).
        let z = inv_normal_cdf(0.975);
        let bound = (2.0 * (1.0 - z / nf.sqrt())).max(0.0);
        return (bound, bound);
    }

    for pair in TABLE.windows(2) {
        let (n_lo, dl_lo, du_lo) = pair[0];
        let (n_hi, dl_hi, du_hi) = pair[1];
        if nf >= n_lo && nf <= n_hi {
            let t = if (n_hi - n_lo).abs() < f64::EPSILON {
                0.0
            } else {
                (nf - n_lo) / (n_hi - n_lo)
            };
            return (dl_lo + t * (dl_hi - dl_lo), du_lo + t * (du_hi - du_lo));
        }
    }

    (last.1, last.2)
}

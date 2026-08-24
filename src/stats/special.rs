//! Native special functions for statistical distributions.
//!
//! Self-contained, dependency-free, Pure-Rust implementations of the gamma,
//! regularized incomplete-gamma and regularized incomplete-beta functions, and
//! the CDF / survival / quantile functions of the normal, Student's-t,
//! chi-squared and F distributions.
//!
//! This module is the **single source of truth** for the `stats` (and
//! `time_series`) tests: every hypothesis test routes its tail probabilities
//! through here so the whole crate shares one numerically-correct
//! implementation instead of the several divergent ad-hoc approximations that
//! previously lived in `distributions.rs`, `inference/mod.rs`, `hypothesis.rs`
//! and `time_series/stats.rs`.
//!
//! The incomplete-gamma (`gser`/`gcf`) and incomplete-beta (`betacf`)
//! continued fractions follow Numerical Recipes (Press et al., 3rd ed.).
//! Accuracy is ~1e-10 to ~1e-12 relative over the usual statistical range
//! (verified against `scipy` 1.17 — see the `scipy_fixture_*` tests below);
//! the gamma continued fraction here fixes a sign error (`an = -i*(i-a)`)
//! that made the previous `chi2_sf` return ≈1.0 for essentially all inputs.
//! `normal_cdf`/`normal_sf` are implemented via [`gamma_q`] (see their doc
//! comments), so they share that same accuracy rather than the ~1.2e-7 a
//! rational `erfc` approximation would give.
//!
//! Survival functions (`chi2_sf`, `f_sf`, `student_t_two_sided_p`,
//! `student_t_sf`, `normal_sf`) and the upper half of the quantile functions
//! (`p > 0.5`) are
//! always computed from a small, well-conditioned tail probability directly
//! — never as `1.0 - cdf(...)` — because that subtraction cancels to zero
//! (or to a value with essentially no correct digits) once `cdf` is within
//! a few ULPs of 1.0, which is exactly the deep-tail regime these functions
//! exist to get right.

const EPS: f64 = 1e-14;
const TINY: f64 = 1e-300;
/// Iteration cap for [`betacf`] (the incomplete-beta continued fraction).
/// `betai` is not known to need the large-shape scaling `gamma_max_iter`
/// applies to the incomplete-gamma routines below.
const MAX_ITER: usize = 300;

/// Iteration cap for the incomplete-gamma series/continued-fraction (`gser`
/// `gcf`), scaled with the shape parameter `a`. The number of terms needed
/// for either to converge grows like `O(sqrt(a))` (the spread of the
/// Poisson-like term sequence around its peak), so a fixed 300-iteration cap
/// silently truncated (and returned a badly wrong result for, e.g.,
/// `a ~ 5000`, reachable from a 100x100 chi-squared goodness-of-fit test).
fn gamma_max_iter(a: f64) -> usize {
    300usize.max((30.0 * a.sqrt()).ceil() as usize)
}

/// Natural logarithm of the gamma function via the Lanczos approximation
/// (g = 7, 9 coefficients), with the reflection formula for `x < 0.5`.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection: Γ(x)Γ(1−x) = π / sin(πx). This function returns
        // ln|Γ(x)| (the standard convention for real arguments, where Γ can
        // be negative for x < 0), so it must take ln|sin(πx)|, not
        // ln(sin(πx)) — the latter is NaN for every x ∈ (-1, 0), where
        // sin(πx) < 0 throughout (e.g. x = -0.5 gives sin(-π/2) = -1).
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut s = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            s += ci / (z + (i as f64) + 1.0);
        }
        let t = z + 7.0 + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
    }
}

/// Regularized lower incomplete gamma `P(a, x)` via the series expansion
/// (Numerical Recipes `gser`). Valid for `x < a + 1`.
///
/// Returns `NaN` if the series fails to converge within
/// [`gamma_max_iter`] terms, rather than silently returning whatever partial
/// sum a fixed 300-term cap happened to reach (which was badly wrong for
/// large `a`, e.g. `gamma_p(1e6, 1e6) = 0.1183` instead of `0.5001`).
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let max_iter = gamma_max_iter(a);
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    let mut converged = false;
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            converged = true;
            break;
        }
    }
    if !converged {
        return f64::NAN;
    }
    (sum.ln() + (-x + a * x.ln() - ln_gamma(a))).exp()
}

/// Regularized upper incomplete gamma `Q(a, x)` via the continued fraction
/// (Numerical Recipes `gcf`). Valid for `x >= a + 1`.
///
/// Returns `NaN` on non-convergence — see [`gamma_p_series`].
fn gamma_q_cf(a: f64, x: f64) -> f64 {
    let max_iter = gamma_max_iter(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    let mut converged = false;
    for i in 1..max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            converged = true;
            break;
        }
    }
    if !converged {
        return f64::NAN;
    }
    (h.ln() + (-x + a * x.ln() - ln_gamma(a))).exp()
}

/// Regularized lower incomplete gamma function `P(a, x) = γ(a, x) / Γ(a)`.
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_p_series(a, x)
    } else {
        1.0 - gamma_q_cf(a, x)
    }
}

/// Regularized upper incomplete gamma function `Q(a, x) = 1 − P(a, x)`.
pub(crate) fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
    } else {
        gamma_q_cf(a, x)
    }
}

/// Continued fraction for the incomplete beta function (Numerical Recipes
/// `betacf`), evaluated by the modified Lentz method.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        // Even step of the recurrence.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step of the recurrence.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta function `I_x(a, b)` (Numerical Recipes `betai`).
pub(crate) fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

// ─── Normal ──────────────────────────────────────────────────────────────────
//
// erf(x) = sign(x) · P(1/2, x²) and erfc(x) = Q(1/2, x²) for x ≥ 0 are
// standard identities relating the error function to the regularized
// incomplete gamma function (Abramowitz & Stegun 6.5.16). Routing through
// `gamma_q`/`gamma_p` here — rather than a fixed-precision rational
// approximation of `erfc` such as Numerical Recipes' `erfcc` (~1.2e-7
// fractional error) — gives `normal_cdf`/`normal_sf` the same ~1e-12
// relative accuracy as the rest of this module, and keeps every tail
// probability in the crate flowing through one continued-fraction
// implementation instead of several independent approximations.

/// Standard-normal survival function `1 − Φ(z)`, i.e. `P(Z > z)`.
///
/// Computed directly from the well-conditioned tail (`gamma_q`), never as
/// `1.0 - normal_cdf(z)` — for large `z` that subtraction would cancel to
/// zero long before the true survival probability actually reaches zero.
pub(crate) fn normal_sf(z: f64) -> f64 {
    let q = gamma_q(0.5, 0.5 * z * z);
    (if z >= 0.0 { 0.5 * q } else { 1.0 - 0.5 * q }).clamp(0.0, 1.0)
}

/// Standard-normal CDF `Φ(z)`, computed as `normal_sf(-z)` (`Φ(z) = P(Z ≤ z)
/// = P(Z ≥ -z)` by symmetry) so it shares `normal_sf`'s accuracy without a
/// separate, independently-cancellation-prone derivation.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    normal_sf(-z)
}

// ─── Chi-squared ─────────────────────────────────────────────────────────────

/// Chi-squared CDF `P(X ≤ x | df)`.
pub(crate) fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    gamma_p(df / 2.0, x / 2.0)
}

/// Chi-squared survival function `P(X > x | df)`.
pub(crate) fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

/// Chi-squared quantile (inverse CDF) via bisection on [`chi2_cdf`] (for
/// `p ≤ 0.5`) or [`chi2_sf`] (for `p > 0.5`).
///
/// The `p > 0.5` branch solves `chi2_sf(x, df) = 1 - p` instead of
/// `chi2_cdf(x, df) = p`: as `p → 1`, `1 - p` stays a small,
/// well-conditioned number while `chi2_cdf(x, df)` itself saturates to
/// (and becomes numerically indistinguishable from) `1.0`, which previously
/// made bisection unable to locate deep-right-tail roots at all.
pub(crate) fn chi2_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let hi0 = df.max(1.0);
    if p > 0.5 {
        let target = 1.0 - p;
        if target == 0.0 {
            // `p` is close enough to 1.0 that `1.0 - p` underflowed to
            // exactly zero; the true quantile is not finitely representable
            // as a meaningfully-more-precise answer than +inf.
            return f64::INFINITY;
        }
        bisect(&|x| chi2_sf(x, df) - target, 0.0, hi0)
    } else {
        bisect(&|x| chi2_cdf(x, df) - p, 0.0, hi0)
    }
}

// ─── Student's t ─────────────────────────────────────────────────────────────

/// Student's-t CDF `P(T ≤ t | df)` via the incomplete beta identity.
pub(crate) fn student_t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let half_tail = 0.5 * betai(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - half_tail
    } else {
        half_tail
    }
}

/// Two-tailed Student's-t p-value `P(|T| > |t| | df)`.
pub(crate) fn student_t_two_sided_p(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    betai(df / 2.0, 0.5, x).clamp(0.0, 1.0)
}

/// One-sided upper-tail probability `P(T > t | df)` for `t ≥ 0`, computed
/// directly (this is exactly `student_t_two_sided_p(t, df) / 2` for `t ≥
/// 0`) rather than as `1.0 - student_t_cdf(t, df)`, which would reintroduce
/// the cancellation `student_t_ppf` exists to avoid.
fn student_t_sf_nonneg(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    0.5 * betai(df / 2.0, 0.5, x)
}

/// One-sided upper-tail probability `P(T > t | df)` for *any* sign of `t`
/// (unlike [`student_t_sf_nonneg`], which requires `t >= 0`).
///
/// This is the direct replacement every "greater" one-sided (and, doubled,
/// two-sided) hypothesis test in this crate must call instead of
/// `1.0 - student_t_cdf(t, df)`. `student_t_cdf` itself already returns
/// `1.0 - half_tail` internally for `t >= 0` (see its doc comment); naively
/// subtracting that *already-rounded* result from `1.0` *again* at the call
/// site double-cancels and silently rounds to exactly `0.0` for any truly
/// significant (large `t`) result, long before the true tail probability
/// actually reaches zero — the same failure mode [`f_sf`] and [`chi2_sf`]
/// exist to avoid for their respective distributions. Mirrors [`normal_sf`]'s
/// sign handling: `student_t_two_sided_p(t, df) / 2` is exactly
/// `student_t_sf_nonneg(t, df)` for `t >= 0` by construction (both reduce to
/// the same `betai` call), so branching on the sign of `t` here costs
/// nothing beyond [`student_t_two_sided_p`]'s own single `betai` evaluation.
pub(crate) fn student_t_sf(t: f64, df: f64) -> f64 {
    let two_sided = student_t_two_sided_p(t, df);
    (if t >= 0.0 {
        0.5 * two_sided
    } else {
        1.0 - 0.5 * two_sided
    })
    .clamp(0.0, 1.0)
}

/// Student's-t quantile (inverse CDF).
///
/// For `p < 0.5` this solves `student_t_sf_nonneg(-t, df) = p` for the
/// (nonnegative) upper root and negates it — using `p` directly, never `1.0
/// - p`, which would round to exactly `1.0` (and silently return the wrong
/// tail) once `p` is small enough, e.g. the `p = 1e-10` case this fixes.
/// For `p > 0.5` it solves the mirrored equation with `target = 1.0 - p`,
/// which is safe there since `p` is bounded away from 0.
///
/// The previous implementation bisected `student_t_cdf(t, df) - p` over a
/// hard-coded `[-1e7, 1e7]` bracket with no sign-change check: any true
/// quantile outside that range (Cauchy-tailed `df = 1` reaches billions for
/// modest `p`) silently returned a value of the wrong sign entirely.
pub(crate) fn student_t_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p < 0.5 {
        -bisect(&|t| student_t_sf_nonneg(t, df) - p, 0.0, 1.0)
    } else {
        let target = 1.0 - p;
        if target == 0.0 {
            return f64::INFINITY;
        }
        bisect(&|t| student_t_sf_nonneg(t, df) - target, 0.0, 1.0)
    }
}

// ─── F ───────────────────────────────────────────────────────────────────────

/// F-distribution CDF `P(X ≤ x | df1, df2)` via the incomplete beta identity.
pub(crate) fn f_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let v = df1 * x / (df1 * x + df2);
    betai(df1 / 2.0, df2 / 2.0, v)
}

/// F-distribution survival function `P(X > x | df1, df2)`.
///
/// Computed directly via the complementary incomplete-beta identity
/// (swapping `a`/`b` and using `1 - v` in place of `v`), never as
/// `1.0 - f_cdf(...)`: that subtraction cancels to exactly `0.0` once
/// `f_cdf` saturates to `1.0`, even though the true survival probability
/// can still be many orders of magnitude above zero (e.g.
/// `f_sf(1e4, 10, 10)` is `~1.26e-18`, not `0.0`).
pub(crate) fn f_sf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let v = df2 / (df1 * x + df2);
    betai(df2 / 2.0, df1 / 2.0, v)
}

/// F-distribution quantile (inverse CDF) via bisection on [`f_cdf`] (for
/// `p ≤ 0.5`) or [`f_sf`] (for `p > 0.5`) — see [`chi2_ppf`] for why the
/// survival function is used directly in the upper half rather than
/// `1.0 - f_cdf(...)`.
pub(crate) fn f_ppf(p: f64, df1: f64, df2: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        let target = 1.0 - p;
        if target == 0.0 {
            return f64::INFINITY;
        }
        bisect(&|x| f_sf(x, df1, df2) - target, 0.0, 1.0)
    } else {
        bisect(&|x| f_cdf(x, df1, df2) - p, 0.0, 1.0)
    }
}

// ─── Root finding ────────────────────────────────────────────────────────────

/// Bisection root finder for a monotone `f` (increasing or decreasing),
/// seeded with an initial `[lo, hi]` guess.
///
/// Two independent bugs in the previous fixed-bracket, fixed-tolerance
/// version are fixed here:
///
/// 1. **Bracket**: `[lo, hi]` was previously a hard cap (`±1e7`) rather than
///    a starting guess — any true root outside it (e.g. `f_ppf(0.99999999,
///    1, 1) ≈ 4.05e15`, or `student_t_ppf` for small `df`, which is
///    Cauchy-tailed and reaches billions for modest `p`) silently returned
///    the clamped boundary, or — worse, when the boundary's residual
///    happened to share a sign with `f(lo)` by coincidence — a value with
///    the *wrong sign entirely*. Here, if the seed bracket doesn't already
///    contain a sign change, it is grown geometrically (doubling the
///    interval each step) until it does.
/// 2. **Termination**: the previous absolute thresholds (`|f(mid)| < 1e-12`
///    or `hi - lo < 1e-12`) declare convergence at the same fixed absolute
///    scale regardless of the root's own magnitude, which is wrong in both
///    directions — many orders of magnitude too loose for a large root, and
///    (the reported case) up to 11 orders of magnitude too loose for a
///    small one, e.g. `chi2_ppf(1e-12, 1) ≈ 1.57e-24`. Bisecting until
///    `mid` can no longer be distinguished from `lo`/`hi` in `f64` instead
///    adapts to the root's actual magnitude and reaches full double
///    precision (this needs at most ~1074 steps across the *entire* `f64`
///    range, and typically well under 100 once a bracket is found).
///
/// Returns `NaN` if no sign change can be bracketed, or if `f` itself
/// returns non-finite values (e.g. the incomplete-gamma series/continued
/// fraction in [`gamma_p_series`]/[`gamma_q_cf`] failing to converge) —
/// propagating that honest failure rather than laundering it into a
/// plausible-looking finite answer via an arbitrary NaN-comparison branch.
fn bisect(f: &dyn Fn(f64) -> f64, lo0: f64, hi0: f64) -> f64 {
    let mut lo = lo0;
    let mut hi = hi0;
    let mut flo = f(lo);
    let mut fhi = f(hi);
    if !flo.is_finite() || !fhi.is_finite() {
        return f64::NAN;
    }
    if flo == 0.0 {
        return lo;
    }
    if fhi == 0.0 {
        return hi;
    }

    // Geometric bracket expansion.
    let mut expansions = 0;
    while (fhi < 0.0) == (flo < 0.0) {
        let width = (hi - lo).abs().max(1.0);
        lo = hi;
        flo = fhi;
        hi += 2.0 * width;
        fhi = f(hi);
        expansions += 1;
        if expansions > 2000 || !hi.is_finite() || !fhi.is_finite() {
            return f64::NAN;
        }
    }

    // Bisect to full `f64` precision (see point 2 above).
    for _ in 0..2000 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            return mid;
        }
        let fmid = f(mid);
        if !fmid.is_finite() {
            return f64::NAN;
        }
        if fmid == 0.0 {
            return mid;
        }
        if (fmid < 0.0) == (flo < 0.0) {
            lo = mid;
            flo = fmid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn chi2_survival_matches_known_critical_values() {
        // χ² 0.05 critical values: df=1 → 3.841, df=4 → 9.488, df=10 → 18.307.
        assert!(close(chi2_sf(3.841, 1.0), 0.05, 1e-3));
        assert!(close(chi2_sf(9.488, 4.0), 0.05, 1e-3));
        assert!(close(chi2_sf(18.307, 10.0), 0.05, 1e-3));
        // A large statistic must be highly significant, not ≈1.0 (the old bug).
        assert!(chi2_sf(50.0, 4.0) < 1e-6);
        // CDF + SF = 1.
        assert!(close(chi2_cdf(9.488, 4.0) + chi2_sf(9.488, 4.0), 1.0, 1e-9));
    }

    #[test]
    fn student_t_two_sided_matches_known_values() {
        // t 0.975 (two-sided 0.05): df=4 → 2.776, df=10 → 2.228, df=30 → 2.042.
        assert!(close(student_t_two_sided_p(2.776, 4.0), 0.05, 1e-3));
        assert!(close(student_t_two_sided_p(2.228, 10.0), 0.05, 1e-3));
        assert!(close(student_t_two_sided_p(2.042, 30.0), 0.05, 1e-3));
        // CDF is a probability in [0,1] (the old impl could exceed 1).
        let c = student_t_cdf(2.776, 4.0);
        assert!((0.0..=1.0).contains(&c));
        assert!(close(c, 0.975, 1e-3));
    }

    #[test]
    fn f_cdf_matches_known_critical_values() {
        // F 0.95 critical values: (2,12) → 3.885, (3,20) → 3.098.
        assert!(close(f_sf(3.885, 2.0, 12.0), 0.05, 1e-3));
        assert!(close(f_sf(3.098, 3.0, 20.0), 0.05, 1e-3));
        assert!(close(f_cdf(1.0, 10.0, 10.0), 0.5, 1e-3));
    }

    #[test]
    fn quantiles_invert_their_cdfs() {
        assert!(close(chi2_cdf(chi2_ppf(0.95, 7.0), 7.0), 0.95, 1e-6));
        assert!(close(f_cdf(f_ppf(0.9, 5.0, 9.0), 5.0, 9.0), 0.9, 1e-6));
        assert!(close(
            student_t_cdf(student_t_ppf(0.975, 8.0), 8.0),
            0.975,
            1e-6
        ));
    }

    #[test]
    fn normal_tails_are_correct() {
        // normal_cdf/normal_sf are now implemented via `gamma_q` (see their
        // doc comments), not a fixed ~1.2e-7-accurate rational `erfc`
        // approximation, so they carry this module's usual ~1e-12 accuracy.
        assert!(close(normal_sf(1.959_963_98), 0.025, 1e-6));
        assert!(close(normal_cdf(0.0), 0.5, 1e-10));
        assert!(close(normal_cdf(1.0) + normal_sf(1.0), 1.0, 1e-12));
    }

    /// Relative-error comparison, for reference values whose magnitude
    /// spans many orders (a fixed absolute tolerance is meaningless for
    /// both `~1e-24` and `~1e15` results in the same test).
    fn close_rel(actual: f64, expected: f64, rel_tol: f64) -> bool {
        if expected == 0.0 {
            return actual.abs() < rel_tol;
        }
        ((actual - expected) / expected).abs() < rel_tol
    }

    /// Scipy 1.17 (`scipy.stats` / `scipy.special`) reference values, each
    /// asserted at 1e-10 relative accuracy — a materially tighter bar than
    /// the ~1e-3 spot checks above, which only pin down textbook critical
    /// values to 3 decimal places.
    #[test]
    fn scipy_fixture_matches_at_1e_minus_10() {
        assert!(close_rel(chi2_sf(7.5, 5.0), 0.18602983360286693, 1e-10));
        assert!(close_rel(chi2_cdf(7.5, 5.0), 0.813970166397133, 1e-10));
        assert!(close_rel(chi2_ppf(0.95, 5.0), 11.070497693516351, 1e-10));
        assert!(close_rel(chi2_ppf(0.05, 5.0), 1.1454762260617692, 1e-10));

        assert!(close_rel(
            student_t_cdf(2.5, 9.0),
            0.9830690861585072,
            1e-10
        ));
        assert!(close_rel(
            student_t_sf_nonneg(2.5, 9.0),
            0.016930913841492864,
            1e-10
        ));
        assert!(close_rel(
            student_t_ppf(0.9, 9.0),
            1.3830287383966324,
            1e-10
        ));
        assert!(close_rel(
            student_t_ppf(0.1, 9.0),
            -1.3830287383966322,
            1e-10
        ));

        assert!(close_rel(f_cdf(2.5, 5.0, 12.0), 0.9101758463950645, 1e-10));
        assert!(close_rel(f_sf(2.5, 5.0, 12.0), 0.08982415360493556, 1e-10));
        assert!(close_rel(f_ppf(0.9, 5.0, 12.0), 2.3940222568422334, 1e-10));

        assert!(close_rel(betai(2.0, 3.0, 0.4), 0.5247999999999999, 1e-10));

        assert!(close_rel(normal_cdf(1.5), 0.9331927987311419, 1e-10));
        assert!(close_rel(normal_sf(1.5), 0.06680720126885806, 1e-10));
        assert!(close_rel(normal_cdf(-3.0), 0.001349898031630093, 1e-10));
        assert!(close_rel(normal_sf(5.0), 2.8665157187919344e-07, 1e-10));
    }

    /// The four (plus one) numerical edge cases this pass fixed, each
    /// verified against `scipy` 1.17. Every one of these previously
    /// returned a value that was wrong by multiple orders of magnitude, the
    /// wrong sign, or exactly the wrong boundary constant.
    #[test]
    fn scipy_fixture_edge_cases() {
        // (1) f_sf cancellation: `1.0 - f_cdf(...)` used to collapse to
        // exactly 0.0 here; the true value is ~1e-18, not 0.
        assert!(close_rel(
            f_sf(1.0e4, 10.0, 10.0),
            1.2589504948268003e-18,
            1e-6
        ));

        // (2) gamma_p/gamma_q large-shape truncation: MAX_ITER=300 used to
        // truncate this series/CF at a~1e6 (chi2 with df~2e6) and return
        // 0.1183 instead of ~0.5001.
        assert!(close_rel(gamma_p(1.0e6, 1.0e6), 0.5001329807608725, 1e-6));
        assert!(close_rel(
            gamma_q(1.0e6, 1.0e6),
            1.0 - 0.5001329807608725,
            1e-6
        ));

        // (3) student_t_ppf bracket sign bug: the old fixed [-1e7, 1e7]
        // bracket returned +1e7 (wrong sign *and* wrong magnitude) here;
        // Cauchy-tailed df=1 truly needs a quantile near -3.18e9.
        assert!(close_rel(
            student_t_ppf(1e-10, 1.0),
            -3183098861.8379064,
            1e-6
        ));
        assert!(
            student_t_ppf(1e-10, 1.0) < 0.0,
            "must be negative, not +1e7"
        );

        // (4) bisect absolute-tolerance termination, small-magnitude root:
        // the old 1e-12 absolute bracket/residual threshold "converged" ~11
        // orders of magnitude away from the true ~1.57e-24 answer.
        assert!(close_rel(
            chi2_ppf(1e-12, 1.0),
            1.5707963267948931e-24,
            1e-6
        ));

        // (5) bisect absolute hi=1e7 clamp, large-magnitude root: the old
        // code could never search past 1e7, so this returned exactly 1e7
        // instead of ~4.05e15.
        assert!(close_rel(
            f_ppf(0.99999999, 1.0, 1.0),
            4052847304964346.0,
            1e-6
        ));
    }

    /// `student_t_sf` (the sign-aware one-sided survival function, added so
    /// callers of a "greater" alternative hypothesis test never have to
    /// write `1.0 - student_t_cdf(...)` themselves) must both agree with
    /// `student_t_two_sided_p` at every sign of `t`, and — the entire
    /// reason it exists — never collapse to exactly `0.0` for a large,
    /// truly significant `t`, the way `1.0 - student_t_cdf(t, df)` does
    /// once `student_t_cdf`'s own internal `1.0 - half_tail` has already
    /// rounded to exactly `1.0`.
    #[test]
    fn student_t_sf_matches_two_sided_p_and_avoids_double_cancellation() {
        // t >= 0: student_t_sf(t, df) == student_t_two_sided_p(t, df) / 2
        // == student_t_sf_nonneg(t, df) exactly (same betai evaluation).
        assert!(close_rel(
            student_t_sf(2.5, 9.0),
            0.016930913841492864, // scipy: stats.t.sf(2.5, 9)
            1e-10
        ));
        assert_eq!(student_t_sf(2.5, 9.0), student_t_sf_nonneg(2.5, 9.0));

        // t < 0: P(T > t) = 1 - P(T <= t) = 1 - student_t_cdf(-t is not
        // needed; symmetry) = 1 - student_t_sf_nonneg(-t, df). scipy:
        // stats.t.sf(-2.5, 9) == stats.t.cdf(2.5, 9).
        assert!(close_rel(
            student_t_sf(-2.5, 9.0),
            0.9830690861585072,
            1e-10
        ));

        // CDF + SF = 1 at every sign.
        assert!(close(
            student_t_cdf(2.5, 9.0) + student_t_sf(2.5, 9.0),
            1.0,
            1e-9
        ));
        assert!(close(
            student_t_cdf(-2.5, 9.0) + student_t_sf(-2.5, 9.0),
            1.0,
            1e-9
        ));

        // The double-cancellation this function exists to avoid: for a
        // large, genuinely significant t, `1.0 - student_t_cdf(t, df)`
        // rounds to exactly 0.0 (since `student_t_cdf` itself already
        // returns `1.0 - half_tail`, and re-subtracting that from 1.0 loses
        // every remaining digit once `half_tail` is small enough) even
        // though the true survival probability is still ~9.49e-30, not 0
        // (scipy: `stats.t.sf(1e6, 5)` = 9.490167245460678e-30, while
        // `1.0 - stats.t.cdf(1e6, 5)` already collapses to exactly 0.0 in
        // `scipy`'s own float64 arithmetic too). The direct `student_t_sf`
        // must not exhibit this.
        let t = 1.0e6;
        let df = 5.0;
        let naive_double_subtraction = 1.0 - student_t_cdf(t, df);
        assert_eq!(
            naive_double_subtraction, 0.0,
            "sanity check: the naive `1.0 - cdf` path really does collapse to exactly 0.0 here"
        );
        let direct = student_t_sf(t, df);
        assert!(
            direct > 0.0,
            "student_t_sf({t}, {df}) must not collapse to exactly 0.0 like the naive path does"
        );
        assert!(close_rel(direct, 9.490167245460678e-30, 1e-6));
        assert!(close_rel(direct, 0.5 * student_t_two_sided_p(t, df), 1e-10));
    }
}

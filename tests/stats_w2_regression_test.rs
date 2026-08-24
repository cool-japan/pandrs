#![allow(clippy::result_large_err)]
//! Regression tests for the Wave-2 `src/stats/**` numerical-correctness pass.
//!
//! Each test pins down one specific bug fixed in that pass, at the crate's
//! *public* API surface (the numerically-verified internals in
//! `stats::special` — chi-squared / t / F CDFs, quantiles, the incomplete
//! gamma/beta functions — are `pub(crate)` and have their own scipy-derived
//! fixture tests inside `src/stats/special.rs`; this file cannot reach them
//! directly and instead exercises the public functions that route through
//! them). Reference values are `scipy` 1.17 (`scipy.stats` /
//! `scipy.special`) unless noted otherwise.

use pandrs::dataframe::DataFrame;
use pandrs::series::Series;
use pandrs::stats::{
    self, chi_square_test_independence, friedman_test, independent_ttest, kruskal_wallis_test,
    ks_two_sample_test, mann_whitney_u_advanced, one_sample_ttest, one_way_anova, runs_test,
    shapiro_wilk_test, wilcoxon_signed_rank_test, AlternativeHypothesis, Poisson,
};
use pandrs::Error;

fn close_rel(actual: f64, expected: f64, rel_tol: f64) -> bool {
    if expected == 0.0 {
        return actual.abs() < rel_tol;
    }
    ((actual - expected) / expected).abs() < rel_tol
}

// ─── Mann-Whitney U: direction-blind one-sided p-value + tie correction ────
//
// The pre-fix implementation collapsed the test statistic to `min(u1, u2)`
// before computing z, which discards exactly the information needed to
// distinguish `Greater` from `Less` — both one-sided directions returned
// the *same* p-value regardless of which group's ranks were actually
// larger. Fixed to test the alternative-appropriate statistic (U1 for
// `Greater`, U2 for `Less`, max(U1, U2) for `TwoSided`) with a tie-corrected
// variance, matching `scipy.stats.mannwhitneyu`'s asymptotic method exactly
// (verified in Python against this exact dataset, which has ties in both
// groups: `[1,2,2,3,4]` vs `[2,3,3,5,6]`).

#[test]
fn mann_whitney_one_sided_p_values_are_direction_aware_and_tie_corrected() {
    let g1 = [1.0, 2.0, 2.0, 3.0, 4.0];
    let g2 = [2.0, 3.0, 3.0, 5.0, 6.0];

    let two_sided = mann_whitney_u_advanced(&g1, &g2, AlternativeHypothesis::TwoSided)
        .expect("mann-whitney two-sided should succeed");
    let greater = mann_whitney_u_advanced(&g1, &g2, AlternativeHypothesis::Greater)
        .expect("mann-whitney greater should succeed");
    let less = mann_whitney_u_advanced(&g1, &g2, AlternativeHypothesis::Less)
        .expect("mann-whitney less should succeed");

    // scipy.stats.mannwhitneyu(g1, g2, alternative=..., method='asymptotic')
    assert!(
        close_rel(two_sided.p_value, 0.19882894401644235, 1e-9),
        "two-sided p was {}",
        two_sided.p_value
    );
    assert!(
        close_rel(greater.p_value, 0.9330689276222845, 1e-9),
        "greater p was {}",
        greater.p_value
    );
    assert!(
        close_rel(less.p_value, 0.09941447200822118, 1e-9),
        "less p was {}",
        less.p_value
    );

    // The old bug: `Greater` and `Less` produced the *same* p-value.
    assert!((greater.p_value - less.p_value).abs() > 0.5);
    // g1's values are (stochastically) smaller than g2's, so the evidence
    // should favor `Less`, not `Greater`.
    assert!(less.p_value < greater.p_value);

    // scipy always reports U for the first sample, regardless of alternative.
    assert!(close_rel(two_sided.statistic, 6.0, 1e-12));
    assert!(close_rel(greater.statistic, 6.0, 1e-12));
    assert!(close_rel(less.statistic, 6.0, 1e-12));
}

/// `stats::mann_whitney_u` (the independent two-sided-only implementation
/// in `inference::mann_whitney_u_impl`) must now agree with
/// `mann_whitney_u_advanced`'s `TwoSided` case on the same data, since both
/// were fixed to the same tie-corrected, continuity-corrected formula.
#[test]
fn mann_whitney_two_implementations_agree_two_sided() {
    let g1 = [1.0, 2.0, 2.0, 3.0, 4.0];
    let g2 = [2.0, 3.0, 3.0, 5.0, 6.0];

    let via_inference = stats::mann_whitney_u(g1, g2, 0.05).expect("should succeed");
    let via_nonparametric =
        mann_whitney_u_advanced(&g1, &g2, AlternativeHypothesis::TwoSided).expect("should succeed");

    assert!(close_rel(
        via_inference.p_value,
        via_nonparametric.p_value,
        1e-9
    ));
    assert!(close_rel(via_inference.p_value, 0.19882894401644235, 1e-9));
}

// ─── Wilcoxon signed-rank: same direction-blindness + tie correction ───────

#[test]
fn wilcoxon_one_sided_p_values_are_direction_aware_and_tie_corrected() {
    let before = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let after = [2.0, 2.0, 5.0, 4.0, 7.0, 6.5];

    let two_sided = wilcoxon_signed_rank_test(&before, &after, AlternativeHypothesis::TwoSided)
        .expect("wilcoxon two-sided should succeed");
    let greater = wilcoxon_signed_rank_test(&before, &after, AlternativeHypothesis::Greater)
        .expect("wilcoxon greater should succeed");
    let less = wilcoxon_signed_rank_test(&before, &after, AlternativeHypothesis::Less)
        .expect("wilcoxon less should succeed");

    // scipy.stats.wilcoxon(before, after, alternative=..., method='approx',
    //                      zero_method='wilcox', correction=True)
    assert!(
        close_rel(two_sided.p_value, 0.09751253817810959, 1e-9),
        "two-sided p was {}",
        two_sided.p_value
    );
    assert!(
        close_rel(greater.p_value, 0.9785798766633644, 1e-9),
        "greater p was {}",
        greater.p_value
    );
    assert!(
        close_rel(less.p_value, 0.048756269089054796, 1e-9),
        "less p was {}",
        less.p_value
    );

    // The old bug: `Greater` and `Less` produced the same p-value.
    assert!((greater.p_value - less.p_value).abs() > 0.5);
    // `before` is mostly smaller than `after`, so the evidence should favor
    // `Less` (before < after), not `Greater`.
    assert!(less.p_value < greater.p_value);
}

// ─── Kolmogorov-Smirnov: no artificial p-value floor ───────────────────────
//
// The pre-fix implementation floored the *returned* `p_value` field at
// 0.001 while computing `reject_null` from the true, unfloored value —
// fabricating a less-extreme p-value than was actually computed. Two
// well-separated samples should now report a p-value that can legitimately
// fall below 0.001 (this dataset's true two-sided asymptotic p is on the
// order of 1e-4, well under the old floor).

#[test]
fn ks_two_sample_p_value_is_not_floored() {
    let sample1: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let sample2: Vec<f64> = (0..20).map(|i| i as f64 + 100.0).collect();

    let result = ks_two_sample_test(&sample1, &sample2, AlternativeHypothesis::TwoSided)
        .expect("ks test should succeed");

    assert!(
        result.p_value < 0.001,
        "expected an unfloored p-value below the old 0.001 floor, got {}",
        result.p_value
    );
    // `reject_null` and the returned `p_value` must agree (the pre-fix
    // struct literal derived them from two different variables).
    assert_eq!(result.reject_null, result.p_value < 0.05);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
}

// ─── sample / stratified_sample: dtype preservation + seeded reproducibility

/// The pre-fix `sample`/`stratified_sample` reconstructed every sampled row
/// via `get_column::<String>` only; on a DataFrame with no `String` column
/// (e.g. this all-`f64`/`i64` one), every column silently failed that
/// downcast and was dropped, so the "sampled" result had *zero* columns.
#[test]
fn sample_preserves_non_string_column_types() {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new((0..20).map(|i| i as f64).collect(), Some("x".to_string())).expect("series"),
    )
    .expect("add_column");
    df.add_column(
        "y".to_string(),
        Series::new((0..20).collect::<Vec<i64>>(), Some("y".to_string())).expect("series"),
    )
    .expect("add_column");

    let sampled = stats::sample(&df, 0.5, false, Some(1)).expect("sample should succeed");
    assert_eq!(sampled.column_count(), 2, "no column should be dropped");
    assert_eq!(sampled.row_count(), 10);
    assert!(sampled.get_column::<f64>("x").is_ok());
    assert!(sampled.get_column::<i64>("y").is_ok());

    let strat_df = {
        let mut d = DataFrame::new();
        d.add_column(
            "g".to_string(),
            Series::new(
                vec!["a", "a", "b", "b", "a", "b"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                Some("g".to_string()),
            )
            .expect("series"),
        )
        .expect("add_column");
        d.add_column(
            "v".to_string(),
            Series::new(vec![1i64, 2, 3, 4, 5, 6], Some("v".to_string())).expect("series"),
        )
        .expect("add_column");
        d
    };
    let strat_sampled = stats::stratified_sample(&strat_df, "g", 0.5, false, Some(1))
        .expect("stratified_sample should succeed");
    assert_eq!(strat_sampled.column_count(), 2);
    assert!(strat_sampled.get_column::<i64>("v").is_ok());
}

/// The pre-fix implementation's doc comment claimed a seeded, reproducible
/// RNG but actually called the unseeded `scirs2_core::random::rng()` every
/// time — two calls with "the same seed" produced different samples. For
/// `stratified_sample` specifically, the stratum loop also re-seeded a
/// fresh RNG *inside* the loop and iterated a `HashMap` in unspecified
/// order, so even a single hoisted seed would not have been enough.
#[test]
fn sample_with_same_seed_is_reproducible() {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new((0..100).collect::<Vec<i64>>(), Some("x".to_string())).expect("series"),
    )
    .expect("add_column");

    let a = stats::sample(&df, 0.5, false, Some(123))
        .expect("sample should succeed")
        .get_column::<i64>("x")
        .expect("column")
        .values()
        .to_vec();
    let b = stats::sample(&df, 0.5, false, Some(123))
        .expect("sample should succeed")
        .get_column::<i64>("x")
        .expect("column")
        .values()
        .to_vec();
    assert_eq!(a, b, "same seed must draw the same sample");
}

// ─── linear_regression: no silent 0.0 coercion, no usize underflow panic ──

#[test]
fn linear_regression_rejects_unparseable_cells_instead_of_coercing_to_zero() {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).expect("series"),
    )
    .expect("add_column");
    // A string `y` column with one genuinely unparseable cell ("N/A").
    // Previously coerced to `0.0`, silently injecting a fabricated
    // observation into the fit rather than reporting the bad input.
    df.add_column(
        "y".to_string(),
        Series::new(
            vec!["2.0", "4.0", "N/A", "8.0", "10.0"]
                .into_iter()
                .map(String::from)
                .collect(),
            Some("y".to_string()),
        )
        .expect("series"),
    )
    .expect("add_column");

    let result = stats::linear_regression(&df, "y", &["x"]);
    assert!(result.is_err(), "an unparseable cell must be an error");
    assert!(matches!(result.unwrap_err(), Error::InvalidValue(_)));
}

#[test]
fn linear_regression_reports_insufficient_data_instead_of_panicking() {
    // n = 2 observations, p = 2 predictors: residual degrees of freedom
    // `n - p - 1` is `2 - 2 - 1`, which underflows a `usize` subtraction
    // before any bounds check runs (this would panic in a debug build, or
    // silently wrap to a huge `usize` — and corrupt `adj_r_squared` with it
    // — in release). With only 2 rows and 3 model columns (intercept +
    // x1 + x2), X^TX is also rank-deficient, so `matrix_inverse`'s own
    // singularity check fires first here; the point of this test is simply
    // that the *process does not panic* and a clean `Err` comes back
    // either way, exercising the exact input shape that used to be able to
    // reach the unguarded subtraction.
    let mut df = DataFrame::new();
    df.add_column(
        "x1".to_string(),
        Series::new(vec![1.0, 2.0], Some("x1".to_string())).expect("series"),
    )
    .expect("add_column");
    df.add_column(
        "x2".to_string(),
        Series::new(vec![2.0, 5.0], Some("x2".to_string())).expect("series"),
    )
    .expect("add_column");
    df.add_column(
        "y".to_string(),
        Series::new(vec![3.0, 7.0], Some("y".to_string())).expect("series"),
    )
    .expect("add_column");

    let result = stats::linear_regression(&df, "y", &["x1", "x2"]);
    assert!(
        result.is_err(),
        "under-determined regression (n <= p) must error, not panic or wrap"
    );

    // A well-conditioned `n == p + 1` case (3 rows, 2 predictors: X^TX is a
    // full-rank, invertible 3x3 matrix, so `matrix_inverse` succeeds) drives
    // the residual degrees of freedom to exactly 0 and exercises the
    // `checked_sub`/`filter(|&d| d > 0)` guard directly, independent of
    // matrix singularity detection.
    let mut df2 = DataFrame::new();
    df2.add_column(
        "x1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0], Some("x1".to_string())).expect("series"),
    )
    .expect("add_column");
    df2.add_column(
        "x2".to_string(),
        Series::new(vec![2.0, 5.0, 4.0], Some("x2".to_string())).expect("series"),
    )
    .expect("add_column");
    df2.add_column(
        "y".to_string(),
        Series::new(vec![3.0, 7.0, 8.0], Some("y".to_string())).expect("series"),
    )
    .expect("add_column");

    let result2 = stats::linear_regression(&df2, "y", &["x1", "x2"]);
    assert!(
        result2.is_err(),
        "zero residual degrees of freedom (n == p + 1) must error cleanly"
    );
    assert!(matches!(result2.unwrap_err(), Error::InsufficientData(_)));
}

// ─── Poisson::pmf: exact via ln_gamma, not a 1-term Stirling approximation ─

/// `scipy.stats.poisson.pmf` reference values. `k = 170` is past the point
/// where `k!` itself overflows `f64` (~170!), which the previous direct
/// `λᵏ / k!` implementation (with `k!` replaced by 1-term Stirling for
/// `k > 20`) could not survive; the log-space `ln_gamma`-based
/// implementation never materializes the factorial at all.
#[test]
fn poisson_pmf_matches_scipy_including_beyond_170() {
    let close_rel =
        |actual: f64, expected: f64, tol: f64| ((actual - expected) / expected).abs() < tol;

    let d = Poisson::new(50.0).expect("valid lambda");
    assert!(close_rel(d.pmf(45), 0.045826241434197924, 1e-9));

    let d170 = Poisson::new(100.0).expect("valid lambda");
    let p170 = d170.pmf(170);
    assert!(p170.is_finite() && p170 > 0.0, "pmf(170) was {p170}");
    assert!(close_rel(p170, 5.1258962876176525e-11, 1e-6));
}

// ─── `1.0 - cdf(...)` cancellation at hypothesis-test call sites ───────────
//
// `stats::special`'s own `chi2_sf`/`f_sf`/`student_t_two_sided_p`/`normal_sf`
// were already fixed to compute survival probabilities directly rather than
// as `1.0 - cdf(...)` (see `special.rs`'s own scipy-fixture tests). But
// several *callers* in `hypothesis.rs`, `nonparametric.rs`, and
// `inference/mod.rs` still computed their own p-value as
// `1.0 - some_dist.cdf(statistic)` (or, for the two-sided Student's-t case,
// `2.0 * (1.0 - cdf(|t|))`) — reintroducing exactly the same cancellation
// one layer up: a distribution's `cdf` already returns `1.0 - half_tail`
// internally for a nonnegative argument, so subtracting that already-rounded
// result from `1.0` *again* at the call site silently collapses to exactly
// `0.0` for any strongly-significant result, long before the true tail
// probability actually reaches zero.
//
// Every test below drives its statistic far enough into that regime that
// the true `scipy`-computed p-value is many orders of magnitude below the
// ~1.1e-16 threshold where `1.0 - x` first rounds to exactly `1.0` in
// `f64` — the pre-fix code would have returned `p_value == 0.0` (confirmed
// against `scipy`'s own float64 arithmetic for two of these cases below,
// where even `1.0 - scipy_cdf(...)` collapses to exactly `0.0`) in every one
// of these; the fix must return the true, still-nonzero (if extremely
// small) value instead.

#[test]
fn ttest_two_sided_survives_extreme_significance_without_collapsing_to_zero() {
    // scipy: `stats.ttest_ind(s_big, s_small, equal_var=True)`.
    let s_big = [50.0, 50.001, 49.999, 50.0005, 49.9995];
    let s_small = [1.0, 1.001, 0.999, 1.0005, 0.9995];
    let result = stats::ttest(s_big, s_small, 0.05, true).expect("ttest should succeed");
    assert!(
        result.pvalue > 0.0,
        "pvalue collapsed to exactly 0.0, was {}",
        result.pvalue
    );
    assert!(close_rel(result.pvalue, 1.316465298689405e-37, 1e-6));
    assert!(result.significant);
}

#[test]
fn independent_ttest_greater_and_two_sided_survive_extreme_significance() {
    // scipy: `stats.ttest_ind(s_big, s_small, equal_var=False, alternative=...)`.
    let s_big = [50.0, 50.001, 49.999, 50.0005, 49.9995];
    let s_small = [1.0, 1.001, 0.999, 1.0005, 0.9995];

    let greater = independent_ttest(&s_big, &s_small, AlternativeHypothesis::Greater, false)
        .expect("independent_ttest greater should succeed");
    assert!(
        greater.p_value > 0.0,
        "Greater p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(greater.p_value, 6.582326493447146e-38, 1e-6));

    let two_sided = independent_ttest(&s_big, &s_small, AlternativeHypothesis::TwoSided, false)
        .expect("independent_ttest two-sided should succeed");
    assert!(
        two_sided.p_value > 0.0,
        "TwoSided p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(two_sided.p_value, 1.3164652986894292e-37, 1e-6));

    // t > 0 here, so TwoSided must be exactly double Greater's p-value
    // (both reduce to the same `betai` evaluation, scaled by a power of 2).
    assert!(close_rel(two_sided.p_value, 2.0 * greater.p_value, 1e-9));
}

#[test]
fn one_sample_ttest_greater_and_two_sided_survive_extreme_significance() {
    // scipy: `stats.ttest_1samp([50, 50.001, ...], popmean=1.0, alternative=...)`.
    let data = [50.0, 50.001, 49.999, 50.0005, 49.9995];

    let greater = one_sample_ttest(&data, 1.0, AlternativeHypothesis::Greater)
        .expect("one_sample_ttest greater should succeed");
    assert!(
        greater.p_value > 0.0,
        "Greater p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(greater.p_value, 8.131243382652792e-21, 1e-6));

    let two_sided = one_sample_ttest(&data, 1.0, AlternativeHypothesis::TwoSided)
        .expect("one_sample_ttest two-sided should succeed");
    assert!(
        two_sided.p_value > 0.0,
        "TwoSided p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(two_sided.p_value, 1.6262486765305583e-20, 1e-6));
}

#[test]
fn one_way_anova_survives_extreme_significance() {
    // scipy: `stats.f_oneway(g1, g2, g3)`. Deliberately kept to a natural
    // (integer-valued, single-order-of-magnitude-per-group) scale rather
    // than mixing tiny within-group noise with a huge cross-group magnitude
    // spread: the latter makes the *sum-of-squares formula itself* (a
    // separate, pre-existing floating-point-summation-order sensitivity
    // unrelated to the `f_sf` fix under test here) diverge slightly from
    // `scipy`'s internal computation, which this test's tight tolerance
    // cannot absorb. With this data, `scipy` and the crate's direct
    // sum-of-squares formula agree on `F` bit-for-bit.
    let g1: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let g2: Vec<f64> = (0..20).map(|x| x as f64 + 1000.0).collect();
    let g3: Vec<f64> = (0..20).map(|x| x as f64 + 2000.0).collect();
    let groups: Vec<&[f64]> = vec![&g1, &g2, &g3];

    let result = one_way_anova(&groups).expect("anova should succeed");
    assert!(
        result.p_value > 0.0,
        "ANOVA p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(result.p_value, 2.4493203495319386e-123, 1e-6));
    assert!(result.reject_null);
}

#[test]
fn chi_square_independence_survives_extreme_significance() {
    // scipy: `stats.chi2_contingency([[100, 10], [10, 100]], correction=False)`.
    let observed = vec![vec![100.0, 10.0], vec![10.0, 100.0]];
    let result = chi_square_test_independence(&observed).expect("chi-square should succeed");
    assert!(
        result.p_value > 0.0,
        "chi-square p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(result.p_value, 6.840881185196698e-34, 1e-6));
}

#[test]
fn kruskal_wallis_survives_extreme_significance() {
    // scipy: `stats.kruskal(k1, k2, k3)`.
    let k1: Vec<f64> = (1..=30).map(|x| x as f64).collect();
    let k2: Vec<f64> = (1..=30).map(|x| x as f64 + 100_000.0).collect();
    let k3: Vec<f64> = (1..=30).map(|x| x as f64 + 200_000.0).collect();
    let groups: Vec<&[f64]> = vec![&k1, &k2, &k3];

    let result = kruskal_wallis_test(&groups).expect("kruskal-wallis should succeed");
    assert!(
        result.p_value > 0.0,
        "Kruskal-Wallis p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(result.p_value, 6.593551417550708e-18, 1e-6));
}

#[test]
fn friedman_test_survives_extreme_significance() {
    // scipy: `stats.friedmanchisquare(*cols)` on the transposed data — 100
    // subjects with a perfectly consistent rank order across the 3
    // conditions (condition 3 always ranked last, by a huge margin).
    let data: Vec<Vec<f64>> = (0..100)
        .map(|i| {
            let base = i as f64 * 0.01;
            vec![1.0 + base, 2.0 + base, 100.0 + base]
        })
        .collect();

    let result = friedman_test(&data).expect("friedman test should succeed");
    assert!(
        result.p_value > 0.0,
        "Friedman p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(result.p_value, 3.7200759760208177e-44, 1e-6));
}

#[test]
fn runs_test_survives_extreme_significance() {
    // A maximally non-random sequence (one run of 40 `true`s, then one run
    // of 40 `false`s) drives `|z|` far enough that even `scipy`'s own
    // `1.0 - norm.cdf(|z|)` collapses to exactly 0.0 in float64, while the
    // true (`norm.sf`) value is ~4.56e-18.
    let sequence: Vec<bool> = std::iter::repeat(true)
        .take(40)
        .chain(std::iter::repeat(false).take(40))
        .collect();

    let result = runs_test(&sequence).expect("runs test should succeed");
    assert!(
        result.p_value > 0.0,
        "runs-test p-value collapsed to exactly 0.0"
    );
    assert!(close_rel(result.p_value, 4.560094719366857e-18, 1e-6));
}

// ─── Shapiro-Wilk: no artificial `1e-10` p-value floor ─────────────────────
//
// Both branches of Royston's p-value transform used to clamp to
// `[1e-10, 1.0]` rather than the valid probability range `[0.0, 1.0]` —
// fabricating a less-extreme p-value than was actually computed, the same
// anti-pattern already removed from the Kolmogorov-Smirnov test's old
// `.max(0.001)` floor. A strongly non-normal sample (a tight cluster with
// one massive outlier) drives Royston's `W` statistic — and the resulting
// normal-approximation `z` — far enough that the true p-value is many
// orders of magnitude below the old floor.

#[test]
fn shapiro_wilk_p_value_is_not_floored_at_1e_minus_10() {
    let mut data: Vec<f64> = (1..=29).map(|x| x as f64).collect();
    data.push(1.0e7); // one massive outlier among an otherwise tight cluster
    let result = shapiro_wilk_test(&data).expect("shapiro_wilk_test should succeed");

    assert!(
        result.p_value < 1e-10,
        "expected a p-value below the old 1e-10 floor, got {}",
        result.p_value
    );
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    // `reject_null` and the returned `p_value` must agree.
    assert_eq!(result.reject_null, result.p_value < 0.05);
    assert!(result.reject_null);
}

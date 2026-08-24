//! Regression tests for `src/optimized/direct_aggregations.rs` (wave 4).
//!
//! Pins down bugs fixed in Wave 4 where the SIMD-accelerated `*_simd`
//! methods on `OptimizedDataFrame` diverged from their `*_direct`
//! counterparts:
//!
//! * **A2** -- `mean_simd` on an `Int64` column (no null mask) used to
//!   compute an INTEGER mean (`sum / len` performed in `i64`, truncating
//!   toward zero) and cast the already-truncated result to `f64`:
//!   `[1, 2]` -> `1.0` instead of the correct `1.5` that `mean_direct`
//!   (`Int64Column::mean`) returns. Fixed to sum in `i64` (exact, matches
//!   `Int64Column::sum`) and divide by the count in `f64`.
//! * **A4** -- `min_simd`/`max_simd` on an all-`NaN`, no-null-mask
//!   `Float64` column used to return `Ok(+-INFINITY)` (the SIMD scalar
//!   fold's identity value) where `min_direct`/`max_direct`
//!   (`Float64Column::min`/`max`) correctly return `Err` (there is no
//!   valid, non-`NaN` observation). Fixed to detect the all-`NaN` case
//!   explicitly and return the same `Err`.
//! * **Adjacent to A4** (found covering the "all-NaN f64" case for every
//!   `*_simd` method, not just min/max) -- `mean_simd` had the exact same
//!   all-`NaN` divergence as A4: `Ok(NaN)` (from `simd_mean_f64`'s
//!   documented all-NaN behavior) where `mean_direct`
//!   (`Float64Column::mean`) returns `Err`. Fixed the same way as A4.
//!   Nothing in `src/`/`examples/` depends on the old `Ok(NaN)` shape
//!   (confirmed by inspection before fixing).
//!
//! Every test below asserts `*_simd(..)` and `*_direct(..)` agree --
//! either the same `Ok` value, or both `Err` with the same message -- so a
//! future regression on either path is caught here.

use pandrs::OptimizedDataFrame;

/// Build a single-column, no-null-mask `OptimizedDataFrame` over an `i64`
/// vector (via `add_int_column`, which never sets a null mask).
fn df_int64(name: &str, data: Vec<i64>) -> OptimizedDataFrame {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column(name, data)
        .expect("add_int_column should succeed");
    df
}

/// Build a single-column, no-null-mask `OptimizedDataFrame` over an `f64`
/// vector (via `add_float_column`, which never sets a null mask).
fn df_float64(name: &str, data: Vec<f64>) -> OptimizedDataFrame {
    let mut df = OptimizedDataFrame::new();
    df.add_float_column(name, data)
        .expect("add_float_column should succeed");
    df
}

// ---------------------------------------------------------------------
// A2 -- mean_simd i64 must divide in f64, not compute a truncated integer
// mean and cast it afterward.
// ---------------------------------------------------------------------

#[test]
fn mean_simd_i64_matches_mean_direct_on_two_element_vector() {
    // The exact regression case from the audit: [1, 2] must mean 1.5, not
    // the truncated integer mean of 1.0.
    let df = df_int64("v", vec![1, 2]);

    let direct = df.mean_direct("v").expect("mean_direct should succeed");
    let simd = df.mean_simd("v").expect("mean_simd should succeed");

    assert_eq!(direct, 1.5, "mean_direct regressed off the expected 1.5");
    assert_eq!(simd, 1.5, "mean_simd must be 1.5, not the truncated 1.0");
    assert_eq!(simd, direct, "mean_simd must match mean_direct exactly");
}

#[test]
fn mean_simd_i64_matches_mean_direct_on_mixed_integer_vectors() {
    // Vectors chosen so the OLD (buggy) integer-mean would visibly diverge
    // from the true floating-point mean: non-exact positive, non-exact
    // negative (Rust's `i64` division truncates toward zero, so the old bug
    // under-reported the magnitude of a negative mean too), non-exact mixed
    // sign, plus an exact-mean case and a singleton as sanity checks that
    // the fix doesn't disturb the easy cases.
    let cases: Vec<(&str, Vec<i64>, f64)> = vec![
        ("positive_non_exact", vec![1, 2, 3, 4], 2.5),
        ("negative_non_exact", vec![-1, -2], -1.5),
        ("mixed_sign_non_exact", vec![-3, 4, -1, 10], 2.5),
        ("exact_mean", vec![7, 7, 7], 7.0),
        ("single_element", vec![42], 42.0),
        (
            "larger_mixed_vector",
            vec![100, -50, 33, -7, 21, 6, -4, 15],
            114.0 / 8.0,
        ),
    ];

    for (label, data, expected) in cases {
        let df = df_int64("v", data);
        let direct = df
            .mean_direct("v")
            .unwrap_or_else(|e| panic!("[{label}] mean_direct failed: {e}"));
        let simd = df
            .mean_simd("v")
            .unwrap_or_else(|e| panic!("[{label}] mean_simd failed: {e}"));

        assert_eq!(direct, expected, "[{label}] mean_direct mismatch");
        assert_eq!(simd, expected, "[{label}] mean_simd mismatch");
        assert_eq!(simd, direct, "[{label}] mean_simd must match mean_direct");
    }
}

#[test]
fn mean_simd_f64_normal_value_computation_is_unaffected_by_either_fix() {
    // The f64 arm's VALUE computation (simd_mean_f64) was already correct
    // and untouched by the A2 fix (that fix only changed the Int64 arm);
    // the adjacent all-NaN guard added alongside A4 only short-circuits
    // the degenerate all-NaN case, so a normal, non-exact float mean must
    // still match mean_direct exactly here.
    let df = df_float64("v", vec![1.0, 2.0, 4.0]);

    let direct = df.mean_direct("v").expect("mean_direct should succeed");
    let simd = df.mean_simd("v").expect("mean_simd should succeed");

    assert_eq!(direct, 7.0 / 3.0);
    assert_eq!(simd, direct);
}

// ---------------------------------------------------------------------
// A4 (+ the adjacent mean_simd divergence) -- all-NaN f64 must Err on
// every *_simd method that has a *_direct counterpart returning Err.
// ---------------------------------------------------------------------

#[test]
fn mean_min_max_simd_f64_all_nan_returns_err_matching_direct() {
    let df = df_float64("v", vec![f64::NAN, f64::NAN, f64::NAN]);

    let mean_direct_err = df
        .mean_direct("v")
        .expect_err("mean_direct must error on all-NaN");
    let mean_simd_err = df
        .mean_simd("v")
        .expect_err("mean_simd must error on all-NaN (adjacent to A4)");
    let min_direct_err = df
        .min_direct("v")
        .expect_err("min_direct must error on all-NaN");
    let min_simd_err = df
        .min_simd("v")
        .expect_err("min_simd must error on all-NaN (A4)");
    let max_direct_err = df
        .max_direct("v")
        .expect_err("max_direct must error on all-NaN");
    let max_simd_err = df
        .max_simd("v")
        .expect_err("max_simd must error on all-NaN (A4)");

    // Same Err "shape" (variant + message), not just "both are Err".
    assert_eq!(mean_simd_err.to_string(), mean_direct_err.to_string());
    assert_eq!(min_simd_err.to_string(), min_direct_err.to_string());
    assert_eq!(max_simd_err.to_string(), max_direct_err.to_string());
    assert!(matches!(mean_simd_err, pandrs::Error::EmptyDataFrame(_)));
    assert!(matches!(min_simd_err, pandrs::Error::EmptyDataFrame(_)));
    assert!(matches!(max_simd_err, pandrs::Error::EmptyDataFrame(_)));
}

#[test]
fn mean_min_max_simd_f64_all_nan_beyond_four_elements_returns_err() {
    // 9 elements exercises the 4-way-unrolled chunked loop (chunks_exact(4)
    // + remainder) in both the column and SIMD scalar folds, not only the
    // remainder-only path a <=4-element vector takes.
    let df = df_float64("v", vec![f64::NAN; 9]);

    assert!(df.mean_direct("v").is_err());
    assert!(df.mean_simd("v").is_err());
    assert!(df.min_direct("v").is_err());
    assert!(df.min_simd("v").is_err());
    assert!(df.max_direct("v").is_err());
    assert!(df.max_simd("v").is_err());
}

#[test]
fn mean_min_max_simd_f64_real_infinity_is_not_mistaken_for_all_nan() {
    // Guards against an over-eager fix: a column that is legitimately all
    // `+INFINITY` (no NaN at all) must still succeed with `Ok(INFINITY)` /
    // a NaN mixed with a real finite value must resolve to the finite
    // value -- the fix keys off "every value is NaN", not off "the folded
    // result happens to be infinite".
    let df_pos = df_float64("v", vec![f64::INFINITY, f64::INFINITY]);
    assert_eq!(
        df_pos.min_direct("v").expect("min_direct should succeed"),
        f64::INFINITY
    );
    assert_eq!(
        df_pos.min_simd("v").expect("min_simd should succeed"),
        f64::INFINITY
    );
    // mean([inf, inf]) == inf / 2.0 == inf under IEEE 754, on both paths.
    assert_eq!(
        df_pos.mean_direct("v").expect("mean_direct should succeed"),
        f64::INFINITY
    );
    assert_eq!(
        df_pos.mean_simd("v").expect("mean_simd should succeed"),
        f64::INFINITY
    );

    let df_neg = df_float64("v", vec![f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert_eq!(
        df_neg.max_direct("v").expect("max_direct should succeed"),
        f64::NEG_INFINITY
    );
    assert_eq!(
        df_neg.max_simd("v").expect("max_simd should succeed"),
        f64::NEG_INFINITY
    );
    assert_eq!(
        df_neg.mean_direct("v").expect("mean_direct should succeed"),
        f64::NEG_INFINITY
    );
    assert_eq!(
        df_neg.mean_simd("v").expect("mean_simd should succeed"),
        f64::NEG_INFINITY
    );

    let df_mixed = df_float64("v", vec![1.0, f64::NAN, f64::INFINITY]);
    assert_eq!(
        df_mixed.min_direct("v").expect("min_direct should succeed"),
        1.0
    );
    assert_eq!(
        df_mixed.min_simd("v").expect("min_simd should succeed"),
        1.0
    );
    assert_eq!(
        df_mixed.max_direct("v").expect("max_direct should succeed"),
        f64::INFINITY
    );
    assert_eq!(
        df_mixed.max_simd("v").expect("max_simd should succeed"),
        f64::INFINITY
    );
}

#[test]
fn min_max_simd_f64_normal_values_match_direct() {
    let cases: Vec<(&str, Vec<f64>)> = vec![
        ("simple_ascending", vec![1.0, 2.0, 3.0, 4.0, 5.0]),
        ("negative_and_positive", vec![-5.5, 3.25, 0.0, -1.0, 9.9]),
        (
            "larger_than_four_elements",
            vec![3.1, -2.2, 7.7, -8.8, 0.5, 1.1, -0.1, 6.6, 2.2],
        ),
        ("single_element", vec![42.5]),
    ];

    for (label, data) in cases {
        let df = df_float64("v", data);

        let min_direct = df
            .min_direct("v")
            .unwrap_or_else(|e| panic!("[{label}] min_direct failed: {e}"));
        let min_simd = df
            .min_simd("v")
            .unwrap_or_else(|e| panic!("[{label}] min_simd failed: {e}"));
        let max_direct = df
            .max_direct("v")
            .unwrap_or_else(|e| panic!("[{label}] max_direct failed: {e}"));
        let max_simd = df
            .max_simd("v")
            .unwrap_or_else(|e| panic!("[{label}] max_simd failed: {e}"));

        assert_eq!(min_simd, min_direct, "[{label}] min mismatch");
        assert_eq!(max_simd, max_direct, "[{label}] max mismatch");
    }
}

#[test]
fn min_max_simd_i64_normal_values_still_match_direct() {
    // Int64 min/max were never part of the A4 bug (integers can't be NaN)
    // but are pinned here for completeness alongside the f64 fix.
    let df = df_int64("v", vec![10, -20, 30, -40, 50, 5, -5]);

    assert_eq!(
        df.min_direct("v").expect("min_direct should succeed"),
        df.min_simd("v").expect("min_simd should succeed")
    );
    assert_eq!(
        df.max_direct("v").expect("max_direct should succeed"),
        df.max_simd("v").expect("max_simd should succeed")
    );
}

// ---------------------------------------------------------------------
// Empty columns -- mean/min/max must ALL error identically on both paths;
// sum (whose additive identity is a legitimate 0.0) must NOT error, on
// either path, confirming the A2/A4 fixes didn't leak into sum_simd.
// ---------------------------------------------------------------------

#[test]
fn empty_f64_column_mean_min_max_simd_and_direct_both_error() {
    let df = df_float64("v", vec![]);

    let mean_direct_err = df
        .mean_direct("v")
        .expect_err("mean_direct must error on empty");
    let mean_simd_err = df
        .mean_simd("v")
        .expect_err("mean_simd must error on empty");
    let min_direct_err = df
        .min_direct("v")
        .expect_err("min_direct must error on empty");
    let min_simd_err = df.min_simd("v").expect_err("min_simd must error on empty");
    let max_direct_err = df
        .max_direct("v")
        .expect_err("max_direct must error on empty");
    let max_simd_err = df.max_simd("v").expect_err("max_simd must error on empty");

    assert_eq!(mean_simd_err.to_string(), mean_direct_err.to_string());
    assert_eq!(min_simd_err.to_string(), min_direct_err.to_string());
    assert_eq!(max_simd_err.to_string(), max_direct_err.to_string());

    // sum's additive identity (0.0) must still be returned, not an error.
    assert_eq!(df.sum_direct("v").expect("sum_direct should succeed"), 0.0);
    assert_eq!(df.sum_simd("v").expect("sum_simd should succeed"), 0.0);
}

#[test]
fn empty_i64_column_mean_min_max_simd_and_direct_both_error() {
    let df = df_int64("v", vec![]);

    let mean_direct_err = df
        .mean_direct("v")
        .expect_err("mean_direct must error on empty");
    let mean_simd_err = df
        .mean_simd("v")
        .expect_err("mean_simd must error on empty");
    let min_direct_err = df
        .min_direct("v")
        .expect_err("min_direct must error on empty");
    let min_simd_err = df.min_simd("v").expect_err("min_simd must error on empty");
    let max_direct_err = df
        .max_direct("v")
        .expect_err("max_direct must error on empty");
    let max_simd_err = df.max_simd("v").expect_err("max_simd must error on empty");

    assert_eq!(mean_simd_err.to_string(), mean_direct_err.to_string());
    assert_eq!(min_simd_err.to_string(), min_direct_err.to_string());
    assert_eq!(max_simd_err.to_string(), max_direct_err.to_string());

    // sum's additive identity (0.0) must still be returned, not an error --
    // this also exercises the i64-empty guard that mean_simd's A2 fix
    // relies on (`col.data.len() as f64` would otherwise divide by zero).
    assert_eq!(df.sum_direct("v").expect("sum_direct should succeed"), 0.0);
    assert_eq!(df.sum_simd("v").expect("sum_simd should succeed"), 0.0);
}

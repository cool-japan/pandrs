//! Wave-3b SIMD regression tests (public-API level).
//!
//! These pin the correctness guarantees established while making the JIT SIMD
//! modules honest and consistent:
//!
//!   * `simd_sum_f64` / `simd_mean_f64` / `simd_min_f64` / `simd_max_f64` are
//!     bit-identical to the `Float64Column` NaN-skip accumulators, and
//!     `simd_sum_i64` to `Int64Column::sum`.
//!   * `simd_divide_f64` is pure IEEE (`x/0 == +-inf`, `0/0 == NaN`) rather than
//!     a guarded-`NaN` substitute.
//!   * float equality is exact (never epsilon), with `NaN != NaN`.
//!   * `i64` arithmetic wraps (two's complement), never saturates.
//!   * `simd_variance_f64` / `simd_covariance_f64` match a two-pass reference to
//!     a tight relative tolerance.
//!
//! Host-coverage note: the SIMD kernels are `#[cfg(target_arch = "x86_64")]`.
//! On x86_64 with AVX2 these assertions exercise the AVX2 kernels against the
//! scalar/column reference; on other hosts (e.g. aarch64) every SIMD entry
//! point takes its scalar fallback, so the same assertions validate the scalar
//! paths. Either way scalar==SIMD holds by construction on the host that runs.

use pandrs::optimized::jit::{
    simd_add_i64, simd_compare_f64, simd_covariance_f64, simd_divide_f64, simd_max_f64,
    simd_mean_f64, simd_min_f64, simd_multiply_i64, simd_subtract_i64, simd_sum_f64, simd_sum_i64,
    simd_variance_f64, ComparisonOp,
};
use pandrs::{Float64Column, Int64Column};

/// Fixtures that place `NaN`, `+-inf` and signed zeros in every lane and
/// remainder position of the 4-wide reduction.
fn f64_fixtures() -> Vec<Vec<f64>> {
    vec![
        vec![],
        vec![42.0],
        vec![1.0, 2.0, 3.0],
        vec![1.5, -2.5, 3.25, -4.125],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        vec![f64::NAN, 1.0, 2.0, 3.0, 4.0],
        vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN, 6.0],
        vec![-0.0, 0.0, -0.0, 0.0, -0.0],
        vec![f64::INFINITY, 1.0, f64::NEG_INFINITY, 2.0, 3.0],
        vec![1e308, 1e308, -1e308, 5.0, -3.0, 2.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ]
}

#[test]
fn simd_sum_f64_is_bit_identical_to_float64column() {
    for data in f64_fixtures() {
        let via_simd = simd_sum_f64(&data);
        let via_column = Float64Column::new(data.clone()).sum();
        assert_eq!(
            via_simd.to_bits(),
            via_column.to_bits(),
            "simd_sum_f64 != Float64Column::sum for {:?} ({} vs {})",
            data,
            via_simd,
            via_column
        );
    }
}

#[test]
fn simd_mean_f64_is_bit_identical_to_float64column_when_defined() {
    for data in f64_fixtures() {
        let via_simd = simd_mean_f64(&data);
        match Float64Column::new(data.clone()).mean() {
            Some(via_column) => assert_eq!(
                via_simd.to_bits(),
                via_column.to_bits(),
                "simd_mean_f64 != Float64Column::mean for {:?}",
                data
            ),
            None => {
                // Column reports None for empty / all-NaN; the f64 entry point
                // returns 0.0 (empty) or NaN (all-NaN). Both are the honest
                // "no observations" values, never a fabricated finite mean.
                assert!(
                    via_simd == 0.0 || via_simd.is_nan(),
                    "simd_mean_f64 must be 0.0/NaN where the column mean is None, got {}",
                    via_simd
                );
            }
        }
    }
}

#[test]
fn simd_min_max_f64_match_float64column_payload() {
    for data in f64_fixtures() {
        let col = Float64Column::new(data.clone());
        if let Some(min_col) = col.min() {
            assert_eq!(
                simd_min_f64(&data).to_bits(),
                min_col.to_bits(),
                "simd_min_f64 != Float64Column::min for {:?}",
                data
            );
        }
        if let Some(max_col) = col.max() {
            assert_eq!(
                simd_max_f64(&data).to_bits(),
                max_col.to_bits(),
                "simd_max_f64 != Float64Column::max for {:?}",
                data
            );
        }
    }
    // Empty / all-NaN collapse to the fold identity.
    assert_eq!(simd_min_f64(&[]), f64::INFINITY);
    assert_eq!(simd_max_f64(&[]), f64::NEG_INFINITY);
    let all_nan = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    assert_eq!(simd_min_f64(&all_nan), f64::INFINITY);
    assert_eq!(simd_max_f64(&all_nan), f64::NEG_INFINITY);
}

#[test]
fn simd_sum_i64_matches_int64column() {
    // Non-overflowing fixtures only (Int64Column::sum uses Iterator::sum, which
    // debug-panics on overflow; the SIMD wrapping path is checked separately).
    let fixtures: Vec<Vec<i64>> = vec![
        vec![],
        vec![7],
        vec![1, 2, 3],
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        vec![-5, -4, -3, -2, -1, 0, 1, 2, 3],
        vec![1_000_000, -2_000_000, 3_000_000, 4, 5, 6, 7],
    ];
    for data in fixtures {
        assert_eq!(
            simd_sum_i64(&data),
            Int64Column::new(data.clone()).sum(),
            "simd_sum_i64 != Int64Column::sum for {:?}",
            data
        );
    }
}

#[test]
fn simd_divide_f64_is_pure_ieee() {
    let left = vec![1.0, -1.0, 0.0, 6.0, 5.0, -5.0, 7.0, 8.0, 9.0];
    let right = vec![0.0, 0.0, 0.0, 2.0, -0.0, 0.0, 4.0, 0.0, 3.0];
    let mut result = vec![0.0; left.len()];
    simd_divide_f64(&left, &right, &mut result).expect("simd_divide_f64");

    assert_eq!(result[0], f64::INFINITY, "1.0 / 0.0 must be +inf");
    assert_eq!(result[1], f64::NEG_INFINITY, "-1.0 / 0.0 must be -inf");
    assert!(result[2].is_nan(), "0.0 / 0.0 must be NaN");
    assert_eq!(result[3], 3.0);
    assert_eq!(result[4], f64::NEG_INFINITY, "5.0 / -0.0 must be -inf");
    assert_eq!(result[5], f64::NEG_INFINITY, "-5.0 / 0.0 must be -inf");
    assert_eq!(result[6], 1.75);
    assert_eq!(result[7], f64::INFINITY);
    assert_eq!(result[8], 3.0);

    // Every finite lane matches plain IEEE division bit-for-bit.
    for i in 0..left.len() {
        if right[i] != 0.0 {
            assert_eq!(result[i].to_bits(), (left[i] / right[i]).to_bits());
        }
    }
}

#[test]
fn simd_compare_f64_is_exact_with_nan_semantics() {
    let left = vec![1.0, f64::NAN, 2.0, 3.0, f64::NAN];
    let right = vec![1.0, f64::NAN, 3.0, 3.0, 4.0];

    let mut eq = vec![false; left.len()];
    simd_compare_f64(&left, &right, ComparisonOp::Equal, &mut eq).expect("eq");
    // Exact equality; NaN == anything is false.
    assert_eq!(eq, vec![true, false, false, true, false]);

    let mut ne = vec![false; left.len()];
    simd_compare_f64(&left, &right, ComparisonOp::NotEqual, &mut ne).expect("ne");
    // NaN != anything is true (unordered), and it is the exact negation of ==.
    assert_eq!(ne, vec![false, true, true, false, true]);

    // Two distinct values whose difference (1e-20) is far below f64::EPSILON
    // (~2.2e-16). Exact equality must report them NOT equal; the old
    // epsilon-tolerance bug (`|a-b| < EPSILON`) would have called them equal.
    let a: Vec<f64> = vec![1e-20];
    let b: Vec<f64> = vec![2e-20];
    assert_ne!(a[0], b[0], "fixture values must be genuinely distinct");
    assert!(
        (a[0] - b[0]).abs() < f64::EPSILON,
        "fixture difference must be below EPSILON to exercise the old bug"
    );
    let mut close = vec![false; 1];
    simd_compare_f64(&a, &b, ComparisonOp::Equal, &mut close).expect("close");
    assert_eq!(
        close,
        vec![false],
        "near-but-unequal must not compare equal"
    );
}

#[test]
fn simd_i64_arithmetic_wraps() {
    // Addition overflow wraps (two's complement), not saturates.
    let mut out = vec![0i64; 1];
    simd_add_i64(&[i64::MAX], &[1], &mut out).expect("add");
    assert_eq!(out[0], i64::MIN, "i64::MAX + 1 must wrap to i64::MIN");

    simd_subtract_i64(&[i64::MIN], &[1], &mut out).expect("sub");
    assert_eq!(out[0], i64::MAX, "i64::MIN - 1 must wrap to i64::MAX");

    simd_multiply_i64(&[i64::MAX], &[2], &mut out).expect("mul");
    assert_eq!(out[0], i64::MAX.wrapping_mul(2), "i64 multiply must wrap");

    // Non-overflowing lanes are unaffected, including in vectors long enough to
    // hit the SIMD chunk + scalar remainder split.
    let a: Vec<i64> = (0..9).collect();
    let b: Vec<i64> = (0..9).map(|x| x * 10).collect();
    let mut sum = vec![0i64; 9];
    simd_add_i64(&a, &b, &mut sum).expect("add vec");
    for i in 0..9 {
        assert_eq!(sum[i], a[i].wrapping_add(b[i]));
    }
}

/// Two-pass reference matching `simd_variance_f64`'s algorithm.
fn ref_variance(data: &[f64], ddof: usize) -> f64 {
    let n = data.len();
    if n <= ddof {
        return f64::NAN;
    }
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let ss: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    ss / (n - ddof) as f64
}

fn ref_covariance(x: &[f64], y: &[f64], ddof: usize) -> f64 {
    let n = x.len();
    if n <= ddof {
        return f64::NAN;
    }
    let mx: f64 = x.iter().sum::<f64>() / n as f64;
    let my: f64 = y.iter().sum::<f64>() / n as f64;
    let cov: f64 = x
        .iter()
        .zip(y)
        .map(|(&xi, &yi)| (xi - mx) * (yi - my))
        .sum();
    cov / (n - ddof) as f64
}

#[test]
fn simd_variance_and_covariance_match_reference_to_tolerance() {
    // Not asserted bit-identical: the vector kernels reduce in tree order while
    // the reference reduces sequentially (float addition is not associative).
    // A tight relative tolerance is the honest contract.
    const REL_TOL: f64 = 1e-9;
    let datasets: Vec<Vec<f64>> = vec![
        vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        (0..37).map(|i| (i as f64 * 0.37).sin() * 100.0).collect(),
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        (0..64).map(|i| i as f64 - 31.5).collect(),
    ];
    for data in &datasets {
        for ddof in [0usize, 1usize] {
            let got = simd_variance_f64(data, ddof);
            let want = ref_variance(data, ddof);
            assert_close(got, want, REL_TOL, "variance");
        }
    }

    let x: Vec<f64> = (0..48).map(|i| i as f64 * 1.3).collect();
    let y: Vec<f64> = (0..48).map(|i| (i as f64 * 0.7).cos() * 5.0).collect();
    for ddof in [0usize, 1usize] {
        assert_close(
            simd_covariance_f64(&x, &y, ddof),
            ref_covariance(&x, &y, ddof),
            REL_TOL,
            "covariance",
        );
    }
}

fn assert_close(got: f64, want: f64, rel_tol: f64, what: &str) {
    if want == 0.0 {
        assert!(got.abs() <= rel_tol, "{}: {} not ~0", what, got);
    } else {
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel <= rel_tol,
            "{}: {} vs {} (rel err {})",
            what,
            got,
            want,
            rel
        );
    }
}

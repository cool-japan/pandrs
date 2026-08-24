//! # SIMD JIT Aggregation Module
//!
//! Vectorized reductions (`sum`) for the direct-aggregation fast path
//! ([`crate::optimized::dataframe::OptimizedDataFrame::sum_simd`] and friends).
//!
//! ## What is actually SIMD-accelerated here
//!
//! Only **`sum`** has a hand-written SIMD kernel, and only on `x86_64` with
//! AVX2 at runtime:
//!
//! | op        | x86_64 + AVX2                     | everything else            |
//! |-----------|-----------------------------------|----------------------------|
//! | `sum_f64` | AVX2 (`simd_sum_f64_avx2`)         | scalar 4-lane NaN-skip     |
//! | `sum_i64` | AVX2 (`simd_sum_i64_avx2`)         | scalar wrapping fold       |
//! | `mean_*`  | scalar                            | scalar                     |
//! | `min_*` / `max_*` | scalar                    | scalar                     |
//!
//! `mean`, `min` and `max` are **scalar on every target**. They are not
//! vectorized on purpose: their pandas `skipna=True` contract (skip `NaN`,
//! keep `+-inf`, preserve the sign of zero) does not line up with the x86
//! `MINPD`/`MAXPD` tie-break rules (which return the *second* operand on
//! equal inputs and propagate `NaN`), so a vector kernel could not be made
//! bit-identical to the scalar reductions without more work than it would
//! ever save. The scalar helpers below are the F10 4-way-unrolled
//! accumulators — the compiler auto-vectorizes them on any target that can.
//!
//! ## Bit-identity contract (critical)
//!
//! Every function here returns a value **bit-identical** to the corresponding
//! [`crate::column::Float64Column`] / [`crate::column::Int64Column`]
//! accumulator for the no-NULL case (which is the only case the callers route
//! through SIMD — a present null mask always falls back to the column method).
//! Concretely:
//! - `simd_sum_f64(d) == Float64Column::new(d).sum()`
//! - `simd_mean_f64(d)` equals `Float64Column::new(d).mean()` where that is
//!   `Some` (and yields `NaN` for the all-`NaN` case the column reports as
//!   `None`, since this function's return type is a plain `f64`).
//! - `simd_min_f64` / `simd_max_f64` equal the `Some` payload of
//!   `Float64Column::{min,max}`; empty / all-`NaN` yield `+-INFINITY`, matching
//!   the identity element of the fold (the callers guard emptiness separately).
//! - `simd_sum_i64(d) == Int64Column::new(d).sum()` for every non-overflowing
//!   input. Overflow **wraps** here (two's complement, like the AVX2 kernel and
//!   NumPy int64) rather than debug-panicking like `Iterator::sum`.
//!
//! The AVX2 kernels reproduce the scalar reduction *tree* exactly: four lanes
//! keyed by `index % 4`, combined as `(l0 + l1) + (l2 + l3)`, with the same
//! `NaN`-skip masking. See the unit tests for the equivalence checks.
//!
//! Note on host coverage: on non-`x86_64` hosts (e.g. aarch64) the AVX2 bodies
//! are `#[cfg]`-compiled out and every call takes the scalar path.

// ---------------------------------------------------------------------------
// f64 sum — AVX2 kernel + scalar NaN-skip fallback (bit-identical)
// ---------------------------------------------------------------------------

/// SIMD-accelerated sum of `f64` values, skipping `NaN` (pandas
/// `skipna=True`). Bit-identical to [`crate::column::Float64Column::sum`] for
/// the no-NULL case. Empty input (or all-`NaN`) sums to `0.0`, the additive
/// identity.
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_sum_f64_avx2(data) };
        }
    }

    sum_skipnan_scalar(data)
}

/// SIMD-accelerated mean of `f64` values, skipping `NaN` (pandas
/// `skipna=True`): the divisor is the count of non-`NaN` values, not `len()`.
/// Scalar on every target (see module docs). Returns `0.0` for empty input and
/// `NaN` when there are no non-`NaN` observations, matching the previous
/// behavior of this entry point.
pub fn simd_mean_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let (sum, count) = sum_count_skipnan_scalar(data);
    if count == 0 {
        return f64::NAN;
    }
    sum / count as f64
}

/// SIMD-accelerated minimum of `f64` values, skipping `NaN` but keeping
/// `+-inf` (pandas semantics). Scalar on every target (see module docs).
/// Returns `+INFINITY` for empty or all-`NaN` input (the `min` identity); the
/// callers guard emptiness before using the result.
pub fn simd_min_f64(data: &[f64]) -> f64 {
    min_skipnan_scalar(data)
}

/// SIMD-accelerated maximum of `f64` values, skipping `NaN` but keeping
/// `+-inf`. Scalar on every target (see module docs). Returns `-INFINITY` for
/// empty or all-`NaN` input (the `max` identity).
pub fn simd_max_f64(data: &[f64]) -> f64 {
    max_skipnan_scalar(data)
}

// ---------------------------------------------------------------------------
// i64 sum — AVX2 kernel + scalar wrapping fallback (bit-identical, no NaN)
// ---------------------------------------------------------------------------

/// SIMD-accelerated sum of `i64` values. Bit-identical to
/// [`crate::column::Int64Column::sum`] for every non-overflowing input;
/// overflow wraps (two's complement) instead of panicking.
pub fn simd_sum_i64(data: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_sum_i64_avx2(data) };
        }
    }

    sum_i64_scalar(data)
}

/// Integer (truncating) mean of `i64` values: `sum / len`, computed in `i64`.
///
/// Scalar on every target. **Callers that need a floating-point mean must not
/// route through this function** — divide the `i64` sum by the count in `f64`
/// instead (`simd_sum_i64(d) as f64 / d.len() as f64`). Returns `0` for empty
/// input.
pub fn simd_mean_i64(data: &[i64]) -> i64 {
    if data.is_empty() {
        return 0;
    }
    simd_sum_i64(data) / data.len() as i64
}

/// Minimum of `i64` values. Scalar on every target: AVX2 has no 64-bit integer
/// `min`/`max` instruction (that arrived with AVX-512, which this crate does
/// not use), so there is no honest SIMD kernel to route to. Bit-identical to
/// [`crate::column::Int64Column::min`]'s `Some` payload; returns `i64::MAX` for
/// empty input.
pub fn simd_min_i64(data: &[i64]) -> i64 {
    data.iter().copied().min().unwrap_or(i64::MAX)
}

/// Maximum of `i64` values. Scalar on every target (see [`simd_min_i64`] for
/// why). Returns `i64::MIN` for empty input.
pub fn simd_max_i64(data: &[i64]) -> i64 {
    data.iter().copied().max().unwrap_or(i64::MIN)
}

// ---------------------------------------------------------------------------
// Scalar NaN-skip reductions (F10 4-way unrolled).
//
// These are byte-for-byte replicas of the private accumulators in
// `crate::column::float64_column` (`sum_skipnan_unrolled`,
// `sum_count_skipnan_unrolled`, `min_skipnan_unrolled`, `max_skipnan_unrolled`)
// — same lane assignment (`index % 4`), same reduction tree, same NaN-skip —
// so the SIMD entry points above are bit-identical to the column methods. The
// float64_column accumulators are private; keeping a synchronized copy here is
// what lets the no-NULL fast path stay a true drop-in. The integration
// regression tests pin `simd_* == Float64Column::*` so the copies cannot drift.
// ---------------------------------------------------------------------------

/// Sum of `data`, skipping `NaN`, 4-way unrolled. Replica of
/// `Float64Column`'s `sum_skipnan_unrolled`.
fn sum_skipnan_scalar(data: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in acc.iter_mut().zip(chunk) {
            *lane += if v.is_nan() { 0.0 } else { v };
        }
    }

    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for &v in remainder {
        if !v.is_nan() {
            total += v;
        }
    }
    total
}

/// Sum and non-`NaN` count of `data`, 4-way unrolled. Replica of
/// `Float64Column`'s `sum_count_skipnan_unrolled`.
fn sum_count_skipnan_scalar(data: &[f64]) -> (f64, usize) {
    let mut acc = [0.0f64; 4];
    let mut cnt = [0usize; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for lane in 0..4 {
            let v = chunk[lane];
            let valid = !v.is_nan();
            acc[lane] += if valid { v } else { 0.0 };
            cnt[lane] += valid as usize;
        }
    }

    let mut total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let mut count = (cnt[0] + cnt[1]) + (cnt[2] + cnt[3]);
    for &v in remainder {
        let valid = !v.is_nan();
        total += if valid { v } else { 0.0 };
        count += valid as usize;
    }
    (total, count)
}

/// Minimum of `data`, skipping `NaN`, keeping `+-inf`, 4-way unrolled. Replica
/// of `Float64Column`'s `min_skipnan_unrolled` inner fold (returns the fold's
/// identity `+INFINITY` for empty / all-`NaN`).
fn min_skipnan_scalar(data: &[f64]) -> f64 {
    let mut lanes = [f64::INFINITY; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            *lane = lane.min(v);
        }
    }

    let mut result = lanes[0].min(lanes[1]).min(lanes[2]).min(lanes[3]);
    for &v in remainder {
        result = result.min(v);
    }
    result
}

/// Maximum of `data`, skipping `NaN`, keeping `+-inf`, 4-way unrolled. Replica
/// of `Float64Column`'s `max_skipnan_unrolled` inner fold.
fn max_skipnan_scalar(data: &[f64]) -> f64 {
    let mut lanes = [f64::NEG_INFINITY; 4];
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        for (lane, &v) in lanes.iter_mut().zip(chunk) {
            *lane = lane.max(v);
        }
    }

    let mut result = lanes[0].max(lanes[1]).max(lanes[2]).max(lanes[3]);
    for &v in remainder {
        result = result.max(v);
    }
    result
}

/// Wrapping sum of `i64` values (two's complement on overflow). Scalar
/// fallback for [`simd_sum_i64`].
fn sum_i64_scalar(data: &[i64]) -> i64 {
    data.iter().copied().fold(0i64, i64::wrapping_add)
}

// ---------------------------------------------------------------------------
// AVX2 kernels (x86_64 only).
//
// `#[target_feature(enable = "avx2")]` lets the compiler inline the intrinsics
// and use AVX2 codegen for the surrounding scalar tail instead of emitting an
// out-of-line call (F11). Each is `unsafe` and MUST only be reached after a
// runtime `is_x86_feature_detected!("avx2")` check — which the public entry
// points above enforce.
// ---------------------------------------------------------------------------

/// AVX2 `f64` sum, `NaN`-skipping, reproducing `sum_skipnan_scalar`'s 4-lane
/// reduction tree exactly (so the result is bit-identical).
///
/// Each loaded vector's `NaN` lanes are zeroed via an ordered self-compare
/// mask (`v == v` is all-ones for non-`NaN`, all-zeros for `NaN`; `AND`-ing
/// with `v` yields `+0.0` on `NaN` lanes, matching the scalar `if nan {0.0}`).
/// Lanes map to `index % 4` just like the scalar accumulator, and the final
/// combine is `(l0 + l1) + (l2 + l3)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sum_f64_avx2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_pd();
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = _mm256_loadu_pd(chunk.as_ptr());
        // Ordered self-compare: lanes that are NaN become 0-bits, others 1-bits.
        let ord = _mm256_cmp_pd(v, v, _CMP_ORD_Q);
        // NaN lanes -> +0.0 (all-zero bit pattern), non-NaN lanes preserved.
        let cleaned = _mm256_and_pd(v, ord);
        sum = _mm256_add_pd(sum, cleaned);
    }

    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), sum);
    // Same reduction tree as the scalar accumulator.
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);

    // Remainder: skip (do not add +0.0 for) NaN, matching the scalar tail.
    for &v in remainder {
        if !v.is_nan() {
            total += v;
        }
    }

    total
}

/// AVX2 `i64` sum. Integer addition is exact and associative modulo `2^64`, so
/// any lane ordering yields the same wrapping result as the scalar fold.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sum_i64_avx2(data: &[i64]) -> i64 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        sum = _mm256_add_epi64(sum, v);
    }

    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, sum);
    let mut total = lanes[0]
        .wrapping_add(lanes[1])
        .wrapping_add(lanes[2])
        .wrapping_add(lanes[3]);

    for &v in remainder {
        total = total.wrapping_add(v);
    }

    total
}

// ---------------------------------------------------------------------------
// Capability reporting (informational; honest about the current host)
// ---------------------------------------------------------------------------

/// Check whether any SIMD acceleration is available on this host. `true` only
/// on `x86_64` with SSE2 (baseline); the AVX2 sum kernels additionally require
/// runtime AVX2. Always `false` off `x86_64` — this crate ships no NEON or
/// portable-SIMD path.
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check whether AVX2 (the only feature the sum kernels use) is available.
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Get SIMD capabilities as a human-readable string. Reports `"None"` off
/// `x86_64`.
pub fn simd_capabilities() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut caps: Vec<&str> = Vec::new();
        if is_x86_feature_detected!("avx2") {
            caps.push("AVX2");
        }
        if is_x86_feature_detected!("sse4.2") {
            caps.push("SSE4.2");
        }
        if is_x86_feature_detected!("sse4.1") {
            caps.push("SSE4.1");
        }
        if is_x86_feature_detected!("ssse3") {
            caps.push("SSSE3");
        }
        if is_x86_feature_detected!("sse3") {
            caps.push("SSE3");
        }
        if is_x86_feature_detected!("sse2") {
            caps.push("SSE2");
        }
        if is_x86_feature_detected!("sse") {
            caps.push("SSE");
        }

        if caps.is_empty() {
            "None".to_string()
        } else {
            caps.join(", ")
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        "None".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sum_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = 36.0;
        let result = simd_sum_f64(&data);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_mean_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 3.0;
        let result = simd_mean_f64(&data);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_min_max_f64() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];

        let min_result = simd_min_f64(&data);
        let max_result = simd_max_f64(&data);

        assert_eq!(min_result, 1.0);
        assert_eq!(max_result, 9.0);
    }

    #[test]
    fn test_simd_sum_i64() {
        let data = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let expected = 36i64;
        let result = simd_sum_i64(&data);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_capabilities() {
        let caps = simd_capabilities();
        println!("SIMD capabilities: {}", caps);
        assert!(!caps.is_empty());
    }

    /// Regression test for issue #7:
    /// `simd_capabilities()` must compile and return a sensible
    /// value on non-x86_64 platforms (aarch64, arm, etc.).
    #[test]
    fn test_issue_7_simd_capabilities_compiles_on_non_x86_64() {
        let caps = simd_capabilities();
        assert!(
            !caps.is_empty(),
            "simd_capabilities() must never return empty"
        );

        #[cfg(not(target_arch = "x86_64"))]
        {
            assert_eq!(
                caps, "None",
                "on non-x86_64 platforms simd_capabilities() must report \"None\""
            );
        }
    }

    #[test]
    fn test_empty_arrays() {
        let empty_f64: Vec<f64> = vec![];
        let empty_i64: Vec<i64> = vec![];

        assert_eq!(simd_sum_f64(&empty_f64), 0.0);
        assert_eq!(simd_mean_f64(&empty_f64), 0.0);
        assert_eq!(simd_min_f64(&empty_f64), f64::INFINITY);
        assert_eq!(simd_max_f64(&empty_f64), f64::NEG_INFINITY);

        assert_eq!(simd_sum_i64(&empty_i64), 0);
        assert_eq!(simd_mean_i64(&empty_i64), 0);
        assert_eq!(simd_min_i64(&empty_i64), i64::MAX);
        assert_eq!(simd_max_i64(&empty_i64), i64::MIN);
    }

    /// The public sum entry point must equal the scalar NaN-skip accumulator
    /// bit-for-bit. On x86_64+AVX2 this exercises the AVX2 kernel against the
    /// scalar reference; elsewhere it is scalar==scalar. Covers NaN in every
    /// lane and remainder position, and signed zeros.
    #[test]
    fn simd_sum_f64_bit_identical_to_scalar_reference() {
        let fixtures: Vec<Vec<f64>> = vec![
            vec![],
            vec![1.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![f64::NAN, 1.0, 2.0, 3.0, 4.0],
            vec![1.0, f64::NAN, 2.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN, 6.0],
            vec![-0.0, 0.0, -0.0, 0.0],
            vec![f64::INFINITY, 1.0, f64::NEG_INFINITY, 2.0],
            vec![1e308, 1e308, -1e308, 5.0],
        ];
        for data in &fixtures {
            let simd = simd_sum_f64(data);
            let scalar = sum_skipnan_scalar(data);
            assert_eq!(
                simd.to_bits(),
                scalar.to_bits(),
                "simd_sum_f64 != scalar reference for {:?}",
                data
            );
        }
    }

    /// AVX2 i64 sum (where available) must equal the scalar wrapping fold.
    #[test]
    fn simd_sum_i64_matches_scalar_fold() {
        let fixtures: Vec<Vec<i64>> = vec![
            vec![],
            vec![7],
            vec![1, 2, 3],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![-5, -4, -3, -2, -1, 0, 1, 2, 3],
            vec![i64::MAX, 1, -1, i64::MIN, 0, 0, 0, 0],
        ];
        for data in &fixtures {
            assert_eq!(
                simd_sum_i64(data),
                sum_i64_scalar(data),
                "simd_sum_i64 != scalar fold for {:?}",
                data
            );
        }
    }

    #[test]
    fn min_max_skip_nan_keep_inf() {
        let data = vec![1.0, f64::NAN, -2.0, f64::INFINITY, 3.0];
        assert_eq!(simd_min_f64(&data), -2.0);
        assert_eq!(simd_max_f64(&data), f64::INFINITY);

        // All-NaN collapses to the fold identity.
        let all_nan = vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        assert_eq!(simd_min_f64(&all_nan), f64::INFINITY);
        assert_eq!(simd_max_f64(&all_nan), f64::NEG_INFINITY);
    }
}

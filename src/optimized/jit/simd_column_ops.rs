//! # SIMD Column Operations Module
//!
//! Element-wise column operations used by the [`crate::column::simd_operations`]
//! traits (`SIMDFloat64Ops` / `SIMDInt64Ops`).
//!
//! ## What is SIMD-accelerated
//!
//! On `x86_64`, each `f64` op dispatches at runtime to an AVX2 kernel
//! (4 lanes) or an SSE2 kernel (2 lanes), falling back to scalar; off
//! `x86_64` (e.g. aarch64) every op is scalar. The following `i64` ops have
//! **no** SIMD kernel and are scalar on every target: `multiply` (no 64-bit
//! SIMD multiply), `abs`, and `compare` under SSE2 (SSE2 lacks 64-bit
//! comparisons); AVX2 `compare` vectorizes only `Equal`/`GreaterThan` and runs
//! the other predicates scalar.
//!
//! ## Result semantics (identical across scalar / SSE2 / AVX2)
//!
//! Element-wise IEEE-754 operations are deterministic per element, so the
//! vector and scalar paths are bit-identical. The contract is:
//! - **Division** is pure IEEE: `x / 0.0 == +-inf`, `0.0 / 0.0 == NaN`. No
//!   guarded-`NaN` substitution (which diverged from the vector lanes by CPU).
//! - **Equality** is exact `==` / `!=` (never an epsilon tolerance); `NaN`
//!   compares unequal to everything and `NaN != NaN` is `true`, on every path.
//! - **`i64` arithmetic wraps** (two's complement), matching `_mm256_add_epi64`
//!   and NumPy int64 — never saturating (which diverged from the vector lanes).
//!
//! Every intrinsic function is `unsafe` and reached only behind a runtime
//! `is_x86_feature_detected!` gate; the AVX2 kernels carry
//! `#[target_feature(enable = "avx2")]` so the intrinsics inline (SSE2 is the
//! `x86_64` baseline and needs no attribute).

use crate::core::error::{Error, Result};

/// SIMD-optimized element-wise addition for f64 vectors
pub fn simd_add_f64(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_add_f64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_add_f64_sse2(left, right, result) };
        }
    }

    // Fallback implementation
    for i in 0..left.len() {
        result[i] = left[i] + right[i];
    }
    Ok(())
}

/// SIMD-optimized element-wise subtraction for f64 vectors
pub fn simd_subtract_f64(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_subtract_f64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_subtract_f64_sse2(left, right, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = left[i] - right[i];
    }
    Ok(())
}

/// SIMD-optimized element-wise multiplication for f64 vectors
pub fn simd_multiply_f64(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_multiply_f64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_multiply_f64_sse2(left, right, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = left[i] * right[i];
    }
    Ok(())
}

/// SIMD-optimized element-wise division for f64 vectors
pub fn simd_divide_f64(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_divide_f64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_divide_f64_sse2(left, right, result) };
        }
    }

    // Fallback: pure IEEE-754 division, bit-identical to the `_mm256_div_pd` /
    // `_mm_div_pd` kernels below. `x / 0.0` is `+-inf`, `0.0 / 0.0` is `NaN`
    // (NumPy / pandas semantics) — never a guarded `NaN` stand-in, which would
    // diverge from the vector path by CPU and by element index.
    for i in 0..left.len() {
        result[i] = left[i] / right[i];
    }
    Ok(())
}

/// SIMD-optimized scalar addition for f64 vectors
pub fn simd_add_scalar_f64(data: &[f64], scalar: f64, result: &mut [f64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_add_scalar_f64_avx2(data, scalar, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_add_scalar_f64_sse2(data, scalar, result) };
        }
    }

    // Fallback
    for i in 0..data.len() {
        result[i] = data[i] + scalar;
    }
    Ok(())
}

/// SIMD-optimized scalar multiplication for f64 vectors
pub fn simd_multiply_scalar_f64(data: &[f64], scalar: f64, result: &mut [f64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_multiply_scalar_f64_avx2(data, scalar, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_multiply_scalar_f64_sse2(data, scalar, result) };
        }
    }

    // Fallback
    for i in 0..data.len() {
        result[i] = data[i] * scalar;
    }
    Ok(())
}

/// SIMD-optimized absolute value for f64 vectors
pub fn simd_abs_f64(data: &[f64], result: &mut [f64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_abs_f64_avx2(data, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_abs_f64_sse2(data, result) };
        }
    }

    // Fallback
    for i in 0..data.len() {
        result[i] = data[i].abs();
    }
    Ok(())
}

/// SIMD-optimized square root for f64 vectors
pub fn simd_sqrt_f64(data: &[f64], result: &mut [f64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_sqrt_f64_avx2(data, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_sqrt_f64_sse2(data, result) };
        }
    }

    // Fallback: `f64::sqrt` already returns `NaN` for negative inputs and
    // `-0.0` for `sqrt(-0.0)`, matching `_mm256_sqrt_pd` / `_mm_sqrt_pd`. No
    // guard is needed and adding one would diverge on `-0.0`.
    for i in 0..data.len() {
        result[i] = data[i].sqrt();
    }
    Ok(())
}

/// SIMD-optimized comparison operations for f64 vectors
pub fn simd_compare_f64(
    left: &[f64],
    right: &[f64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for comparison operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_compare_f64_avx2(left, right, op, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_compare_f64_sse2(left, right, op, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }
    Ok(())
}

/// Comparison operation types
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
}

// SIMD operations for i64
/// SIMD-optimized element-wise addition for i64 vectors
pub fn simd_add_i64(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_add_i64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_add_i64_sse2(left, right, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = left[i].wrapping_add(right[i]);
    }
    Ok(())
}

/// SIMD-optimized element-wise subtraction for i64 vectors
pub fn simd_subtract_i64(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_subtract_i64_avx2(left, right, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_subtract_i64_sse2(left, right, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = left[i].wrapping_sub(right[i]);
    }
    Ok(())
}

/// SIMD-optimized element-wise multiplication for i64 vectors
pub fn simd_multiply_i64(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for element-wise operations".to_string(),
        ));
    }

    // Scalar only: AVX2/SSE2 have no full 64-bit integer multiply, so there is
    // no SIMD kernel here. Wrapping (two's complement) is used for consistency
    // with the wrapping i64 add/subtract paths and NumPy int64 semantics.
    for i in 0..left.len() {
        result[i] = left[i].wrapping_mul(right[i]);
    }
    Ok(())
}

/// SIMD-optimized scalar addition for i64 vectors
pub fn simd_add_scalar_i64(data: &[i64], scalar: i64, result: &mut [i64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_add_scalar_i64_avx2(data, scalar, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_add_scalar_i64_sse2(data, scalar, result) };
        }
    }

    // Fallback
    for i in 0..data.len() {
        result[i] = data[i].wrapping_add(scalar);
    }
    Ok(())
}

/// SIMD-optimized absolute value for i64 vectors
pub fn simd_abs_i64(data: &[i64], result: &mut [i64]) -> Result<()> {
    if data.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Input and output vectors must have same length".to_string(),
        ));
    }

    // Scalar only: no direct i64 SIMD abs. `wrapping_abs` avoids the debug-mode
    // overflow panic on `i64::MIN` (whose absolute value is not representable);
    // it maps `i64::MIN` to itself, matching two's-complement semantics.
    for i in 0..data.len() {
        result[i] = data[i].wrapping_abs();
    }
    Ok(())
}

/// SIMD-optimized comparison operations for i64 vectors
pub fn simd_compare_i64(
    left: &[i64],
    right: &[i64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    if left.len() != right.len() || left.len() != result.len() {
        return Err(Error::InvalidOperation(
            "Vector lengths must match for comparison operations".to_string(),
        ));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_compare_i64_avx2(left, right, op, result) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_compare_i64_sse2(left, right, op, result) };
        }
    }

    // Fallback
    for i in 0..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }
    Ok(())
}

// AVX2 implementations for f64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_f64_avx2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm256_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm256_add_pd(left_vec, right_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = left[i] + right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_subtract_f64_avx2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm256_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm256_sub_pd(left_vec, right_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = left[i] - right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_multiply_f64_avx2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm256_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm256_mul_pd(left_vec, right_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = left[i] * right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_divide_f64_avx2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm256_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm256_div_pd(left_vec, right_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder (pure IEEE division, matching the vector lanes)
    for i in (chunks * 4)..left.len() {
        result[i] = left[i] / right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_scalar_f64_avx2(data: &[f64], scalar: f64, result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_pd(scalar);
    let chunks = data.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm256_add_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..data.len() {
        result[i] = data[i] + scalar;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_multiply_scalar_f64_avx2(
    data: &[f64],
    scalar: f64,
    result: &mut [f64],
) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_pd(scalar);
    let chunks = data.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm256_mul_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..data.len() {
        result[i] = data[i] * scalar;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_abs_f64_avx2(data: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    // Create mask to clear sign bit
    let sign_mask = _mm256_set1_pd(-0.0);
    let chunks = data.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm256_andnot_pd(sign_mask, data_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..data.len() {
        result[i] = data[i].abs();
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_sqrt_f64_avx2(data: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = data.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm256_sqrt_pd(data_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..data.len() {
        result[i] = data[i].sqrt();
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_compare_f64_avx2(
    left: &[f64],
    right: &[f64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm256_loadu_pd(right.as_ptr().add(offset));

        let cmp_result = match op {
            ComparisonOp::Equal => _mm256_cmp_pd(left_vec, right_vec, _CMP_EQ_OQ),
            // `_CMP_NEQ_UQ` (unordered) returns true when either operand is NaN,
            // matching Rust `!=` and the SSE2 `_mm_cmpneq_pd` path. `_CMP_NEQ_OQ`
            // (ordered) would return false for NaN, diverging by CPU width.
            ComparisonOp::NotEqual => _mm256_cmp_pd(left_vec, right_vec, _CMP_NEQ_UQ),
            ComparisonOp::LessThan => _mm256_cmp_pd(left_vec, right_vec, _CMP_LT_OQ),
            ComparisonOp::LessThanEqual => _mm256_cmp_pd(left_vec, right_vec, _CMP_LE_OQ),
            ComparisonOp::GreaterThan => _mm256_cmp_pd(left_vec, right_vec, _CMP_GT_OQ),
            ComparisonOp::GreaterThanEqual => _mm256_cmp_pd(left_vec, right_vec, _CMP_GE_OQ),
        };

        // Extract results to boolean array
        let mut temp_result = [0.0; 4];
        _mm256_storeu_pd(temp_result.as_mut_ptr(), cmp_result);

        for j in 0..4 {
            result[offset + j] = temp_result[j].to_bits() != 0;
        }
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }

    Ok(())
}

// SSE2 implementations (similar pattern but with 2-element operations)
#[cfg(target_arch = "x86_64")]
unsafe fn simd_add_f64_sse2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm_add_pd(left_vec, right_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = left[i] + right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_subtract_f64_sse2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm_sub_pd(left_vec, right_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = left[i] - right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_multiply_f64_sse2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm_mul_pd(left_vec, right_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = left[i] * right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_divide_f64_sse2(left: &[f64], right: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm_loadu_pd(right.as_ptr().add(offset));
        let result_vec = _mm_div_pd(left_vec, right_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder (pure IEEE division, matching the vector lanes)
    for i in (chunks * 2)..left.len() {
        result[i] = left[i] / right[i];
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_add_scalar_f64_sse2(data: &[f64], scalar: f64, result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm_set1_pd(scalar);
    let chunks = data.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let data_vec = _mm_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm_add_pd(data_vec, scalar_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..data.len() {
        result[i] = data[i] + scalar;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_multiply_scalar_f64_sse2(
    data: &[f64],
    scalar: f64,
    result: &mut [f64],
) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm_set1_pd(scalar);
    let chunks = data.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let data_vec = _mm_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm_mul_pd(data_vec, scalar_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..data.len() {
        result[i] = data[i] * scalar;
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_abs_f64_sse2(data: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let sign_mask = _mm_set1_pd(-0.0);
    let chunks = data.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let data_vec = _mm_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm_andnot_pd(sign_mask, data_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..data.len() {
        result[i] = data[i].abs();
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_sqrt_f64_sse2(data: &[f64], result: &mut [f64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = data.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let data_vec = _mm_loadu_pd(data.as_ptr().add(offset));
        let result_vec = _mm_sqrt_pd(data_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(offset), result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..data.len() {
        result[i] = data[i].sqrt();
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_compare_f64_sse2(
    left: &[f64],
    right: &[f64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_pd(left.as_ptr().add(offset));
        let right_vec = _mm_loadu_pd(right.as_ptr().add(offset));

        let cmp_result = match op {
            ComparisonOp::Equal => _mm_cmpeq_pd(left_vec, right_vec),
            ComparisonOp::NotEqual => _mm_cmpneq_pd(left_vec, right_vec),
            ComparisonOp::LessThan => _mm_cmplt_pd(left_vec, right_vec),
            ComparisonOp::LessThanEqual => _mm_cmple_pd(left_vec, right_vec),
            ComparisonOp::GreaterThan => _mm_cmpgt_pd(left_vec, right_vec),
            ComparisonOp::GreaterThanEqual => _mm_cmpge_pd(left_vec, right_vec),
        };

        // Extract results
        let mut temp_result = [0.0; 2];
        _mm_storeu_pd(temp_result.as_mut_ptr(), cmp_result);

        for j in 0..2 {
            result[offset + j] = temp_result[j].to_bits() != 0;
        }
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }

    Ok(())
}

// AVX2 implementations for i64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_i64_avx2(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_si256(left.as_ptr().add(offset) as *const __m256i);
        let right_vec = _mm256_loadu_si256(right.as_ptr().add(offset) as *const __m256i);
        let result_vec = _mm256_add_epi64(left_vec, right_vec);
        _mm256_storeu_si256(result.as_mut_ptr().add(offset) as *mut __m256i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = left[i].wrapping_add(right[i]);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_subtract_i64_avx2(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_si256(left.as_ptr().add(offset) as *const __m256i);
        let right_vec = _mm256_loadu_si256(right.as_ptr().add(offset) as *const __m256i);
        let result_vec = _mm256_sub_epi64(left_vec, right_vec);
        _mm256_storeu_si256(result.as_mut_ptr().add(offset) as *mut __m256i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = left[i].wrapping_sub(right[i]);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_scalar_i64_avx2(data: &[i64], scalar: i64, result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_epi64x(scalar);
    let chunks = data.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let data_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);
        let result_vec = _mm256_add_epi64(data_vec, scalar_vec);
        _mm256_storeu_si256(result.as_mut_ptr().add(offset) as *mut __m256i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 4)..data.len() {
        result[i] = data[i].wrapping_add(scalar);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_compare_i64_avx2(
    left: &[i64],
    right: &[i64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let left_vec = _mm256_loadu_si256(left.as_ptr().add(offset) as *const __m256i);
        let right_vec = _mm256_loadu_si256(right.as_ptr().add(offset) as *const __m256i);

        let cmp_result = match op {
            ComparisonOp::Equal => _mm256_cmpeq_epi64(left_vec, right_vec),
            ComparisonOp::GreaterThan => _mm256_cmpgt_epi64(left_vec, right_vec),
            // For other operations, we need to implement using available operations
            _ => {
                // Fallback to scalar for complex comparisons
                for j in 0..4 {
                    let l_val = *left.get_unchecked(offset + j);
                    let r_val = *right.get_unchecked(offset + j);
                    result[offset + j] = match op {
                        ComparisonOp::NotEqual => l_val != r_val,
                        ComparisonOp::LessThan => l_val < r_val,
                        ComparisonOp::LessThanEqual => l_val <= r_val,
                        ComparisonOp::GreaterThanEqual => l_val >= r_val,
                        _ => unreachable!(),
                    };
                }
                continue;
            }
        };

        // Extract results for supported operations
        if matches!(op, ComparisonOp::Equal | ComparisonOp::GreaterThan) {
            let mut temp_result = [0i64; 4];
            _mm256_storeu_si256(temp_result.as_mut_ptr() as *mut __m256i, cmp_result);

            for j in 0..4 {
                result[offset + j] = temp_result[j] != 0;
            }
        }
    }

    // Handle remainder
    for i in (chunks * 4)..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }

    Ok(())
}

// SSE2 implementations for i64
#[cfg(target_arch = "x86_64")]
unsafe fn simd_add_i64_sse2(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_si128(left.as_ptr().add(offset) as *const __m128i);
        let right_vec = _mm_loadu_si128(right.as_ptr().add(offset) as *const __m128i);
        let result_vec = _mm_add_epi64(left_vec, right_vec);
        _mm_storeu_si128(result.as_mut_ptr().add(offset) as *mut __m128i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = left[i].wrapping_add(right[i]);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_subtract_i64_sse2(left: &[i64], right: &[i64], result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let chunks = left.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let left_vec = _mm_loadu_si128(left.as_ptr().add(offset) as *const __m128i);
        let right_vec = _mm_loadu_si128(right.as_ptr().add(offset) as *const __m128i);
        let result_vec = _mm_sub_epi64(left_vec, right_vec);
        _mm_storeu_si128(result.as_mut_ptr().add(offset) as *mut __m128i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..left.len() {
        result[i] = left[i].wrapping_sub(right[i]);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_add_scalar_i64_sse2(data: &[i64], scalar: i64, result: &mut [i64]) -> Result<()> {
    use std::arch::x86_64::*;

    let scalar_vec = _mm_set1_epi64x(scalar);
    let chunks = data.len() / 2;

    for i in 0..chunks {
        let offset = i * 2;
        let data_vec = _mm_loadu_si128(data.as_ptr().add(offset) as *const __m128i);
        let result_vec = _mm_add_epi64(data_vec, scalar_vec);
        _mm_storeu_si128(result.as_mut_ptr().add(offset) as *mut __m128i, result_vec);
    }

    // Handle remainder
    for i in (chunks * 2)..data.len() {
        result[i] = data[i].wrapping_add(scalar);
    }

    Ok(())
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_compare_i64_sse2(
    left: &[i64],
    right: &[i64],
    op: ComparisonOp,
    result: &mut [bool],
) -> Result<()> {
    // SSE2 has limited i64 comparison support, so we use scalar fallback
    for i in 0..left.len() {
        result[i] = match op {
            ComparisonOp::Equal => left[i] == right[i],
            ComparisonOp::NotEqual => left[i] != right[i],
            ComparisonOp::LessThan => left[i] < right[i],
            ComparisonOp::LessThanEqual => left[i] <= right[i],
            ComparisonOp::GreaterThan => left[i] > right[i],
            ComparisonOp::GreaterThanEqual => left[i] >= right[i],
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_f64() {
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let right = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = vec![0.0; 8];

        simd_add_f64(&left, &right, &mut result).expect("operation should succeed");

        let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for i in 0..8 {
            assert!((result[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_multiply_scalar_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 5];

        simd_multiply_scalar_f64(&data, 2.0, &mut result).expect("operation should succeed");

        let expected = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        for i in 0..5 {
            assert!((result[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_abs_f64() {
        let data = vec![-1.0, 2.0, -3.0, 4.0, -5.0];
        let mut result = vec![0.0; 5];

        simd_abs_f64(&data, &mut result).expect("operation should succeed");

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for i in 0..5 {
            assert!((result[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_simd_compare_f64() {
        let left = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let right = vec![1.0, 1.0, 4.0, 4.0, 6.0];
        let mut result = vec![false; 5];

        simd_compare_f64(&left, &right, ComparisonOp::GreaterThan, &mut result)
            .expect("operation should succeed");

        let expected = vec![false, true, false, false, false];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_add_i64() {
        let left = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let right = vec![1i64, 1, 1, 1, 1, 1, 1, 1];
        let mut result = vec![0i64; 8];

        simd_add_i64(&left, &right, &mut result).expect("operation should succeed");

        let expected = vec![2i64, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_compare_i64() {
        let left = vec![1i64, 2, 3, 4, 5];
        let right = vec![1i64, 1, 4, 4, 6];
        let mut result = vec![false; 5];

        simd_compare_i64(&left, &right, ComparisonOp::Equal, &mut result)
            .expect("operation should succeed");

        let expected = vec![true, false, false, true, false];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_length_mismatch() {
        let left = vec![1.0, 2.0];
        let right = vec![1.0];
        let mut result = vec![0.0; 2];

        let err = simd_add_f64(&left, &right, &mut result);
        assert!(err.is_err());
    }
}

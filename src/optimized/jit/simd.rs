//! # SIMD JIT Operations Module
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimized operations
//! for high-performance vectorized computations.

use super::config::SIMDConfig;

/// SIMD-optimized sum for f64 values
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_sum_f64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_sum_f64_sse2(data) };
        }
    }

    // Fallback to standard implementation
    data.iter().sum()
}

/// SIMD-optimized mean for f64 values
pub fn simd_mean_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    simd_sum_f64(data) / data.len() as f64
}

/// SIMD-optimized minimum for f64 values
pub fn simd_min_f64(data: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_min_f64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_min_f64_sse2(data) };
        }
    }

    // Fallback
    data.iter().copied().fold(f64::INFINITY, f64::min)
}

/// SIMD-optimized maximum for f64 values
pub fn simd_max_f64(data: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_max_f64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_max_f64_sse2(data) };
        }
    }

    // Fallback
    data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// SIMD-optimized sum for i64 values
pub fn simd_sum_i64(data: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_sum_i64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_sum_i64_sse2(data) };
        }
    }

    // Fallback
    data.iter().sum()
}

/// SIMD-optimized mean for i64 values
pub fn simd_mean_i64(data: &[i64]) -> i64 {
    if data.is_empty() {
        return 0;
    }
    simd_sum_i64(data) / data.len() as i64
}

/// SIMD-optimized minimum for i64 values
pub fn simd_min_i64(data: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_min_i64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_min_i64_sse2(data) };
        }
    }

    // Fallback
    data.iter().copied().min().unwrap_or(i64::MAX)
}

/// SIMD-optimized maximum for i64 values
pub fn simd_max_i64(data: &[i64]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_max_i64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { simd_max_i64_sse2(data) };
        }
    }

    // Fallback
    data.iter().copied().max().unwrap_or(i64::MIN)
}

// AVX2 implementations for f64
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sum_f64_avx2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_pd();
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm256_loadu_pd(chunk.as_ptr());
        sum = _mm256_add_pd(sum, vec);
    }

    // Extract and sum the components
    let mut result = [0.0; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), sum);
    let mut total = result[0] + result[1] + result[2] + result[3];

    // Handle remainder
    for &value in remainder {
        total += value;
    }

    total
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_min_f64_avx2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if data.is_empty() {
        return f64::INFINITY;
    }

    let mut min_vec = _mm256_set1_pd(f64::INFINITY);
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm256_loadu_pd(chunk.as_ptr());
        min_vec = _mm256_min_pd(min_vec, vec);
    }

    // Extract and find minimum
    let mut result = [0.0; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), min_vec);
    let mut min_val = result[0].min(result[1]).min(result[2]).min(result[3]);

    // Handle remainder
    for &value in remainder {
        min_val = min_val.min(value);
    }

    min_val
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_max_f64_avx2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if data.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm256_loadu_pd(chunk.as_ptr());
        max_vec = _mm256_max_pd(max_vec, vec);
    }

    // Extract and find maximum
    let mut result = [0.0; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0].max(result[1]).max(result[2]).max(result[3]);

    // Handle remainder
    for &value in remainder {
        max_val = max_val.max(value);
    }

    max_val
}

// SSE2 implementations for f64
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sum_f64_sse2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = _mm_setzero_pd();
    let chunks = data.chunks_exact(2);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm_loadu_pd(chunk.as_ptr());
        sum = _mm_add_pd(sum, vec);
    }

    // Extract and sum the components
    let mut result = [0.0; 2];
    _mm_storeu_pd(result.as_mut_ptr(), sum);
    let mut total = result[0] + result[1];

    // Handle remainder
    for &value in remainder {
        total += value;
    }

    total
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_min_f64_sse2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if data.is_empty() {
        return f64::INFINITY;
    }

    let mut min_vec = _mm_set1_pd(f64::INFINITY);
    let chunks = data.chunks_exact(2);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm_loadu_pd(chunk.as_ptr());
        min_vec = _mm_min_pd(min_vec, vec);
    }

    // Extract and find minimum
    let mut result = [0.0; 2];
    _mm_storeu_pd(result.as_mut_ptr(), min_vec);
    let mut min_val = result[0].min(result[1]);

    // Handle remainder
    for &value in remainder {
        min_val = min_val.min(value);
    }

    min_val
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_max_f64_sse2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    if data.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max_vec = _mm_set1_pd(f64::NEG_INFINITY);
    let chunks = data.chunks_exact(2);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm_loadu_pd(chunk.as_ptr());
        max_vec = _mm_max_pd(max_vec, vec);
    }

    // Extract and find maximum
    let mut result = [0.0; 2];
    _mm_storeu_pd(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0].max(result[1]);

    // Handle remainder
    for &value in remainder {
        max_val = max_val.max(value);
    }

    max_val
}

// AVX2 implementations for i64
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sum_i64_avx2(data: &[i64]) -> i64 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        sum = _mm256_add_epi64(sum, vec);
    }

    // Extract and sum the components
    let mut result = [0i64; 4];
    _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, sum);
    let mut total = result[0] + result[1] + result[2] + result[3];

    // Handle remainder
    for &value in remainder {
        total += value;
    }

    total
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_min_i64_avx2(data: &[i64]) -> i64 {
    if data.is_empty() {
        return i64::MAX;
    }

    // AVX2 doesn't have min_epi64, so we'll use a simple fallback for now
    data.iter().copied().min().unwrap_or(i64::MAX)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_max_i64_avx2(data: &[i64]) -> i64 {
    if data.is_empty() {
        return i64::MIN;
    }

    // AVX2 doesn't have max_epi64, so we'll use a simple fallback for now
    data.iter().copied().max().unwrap_or(i64::MIN)
}

// SSE2 implementations for i64
#[cfg(target_arch = "x86_64")]
unsafe fn simd_sum_i64_sse2(data: &[i64]) -> i64 {
    use std::arch::x86_64::*;

    let mut sum = _mm_setzero_si128();
    let chunks = data.chunks_exact(2);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let vec = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);
        sum = _mm_add_epi64(sum, vec);
    }

    // Extract and sum the components
    let mut result = [0i64; 2];
    _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, sum);
    let mut total = result[0] + result[1];

    // Handle remainder
    for &value in remainder {
        total += value;
    }

    total
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_min_i64_sse2(data: &[i64]) -> i64 {
    if data.is_empty() {
        return i64::MAX;
    }

    // SSE2 doesn't have min_epi64, so we'll use a simple fallback
    data.iter().copied().min().unwrap_or(i64::MAX)
}

#[cfg(target_arch = "x86_64")]
unsafe fn simd_max_i64_sse2(data: &[i64]) -> i64 {
    if data.is_empty() {
        return i64::MIN;
    }

    // SSE2 doesn't have max_epi64, so we'll use a simple fallback
    data.iter().copied().max().unwrap_or(i64::MIN)
}

/// Check if SIMD operations are available on this platform
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

/// Check if AVX2 is available
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

/// Get SIMD capabilities as a string
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
}

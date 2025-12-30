//! SIMD-accelerated string operations
//!
//! Provides high-performance string operations using SIMD instructions (AVX2/SSE2)
//! for common string operations like case conversion, character classification,
//! and pattern matching.

use std::sync::atomic::{AtomicBool, Ordering};

// Feature detection cache
static AVX2_SUPPORTED: AtomicBool = AtomicBool::new(false);
static SSE2_SUPPORTED: AtomicBool = AtomicBool::new(false);
static FEATURES_DETECTED: AtomicBool = AtomicBool::new(false);

/// Initialize SIMD feature detection
fn detect_features() {
    if FEATURES_DETECTED.load(Ordering::Relaxed) {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        AVX2_SUPPORTED.store(is_x86_feature_detected!("avx2"), Ordering::Relaxed);
        SSE2_SUPPORTED.store(is_x86_feature_detected!("sse2"), Ordering::Relaxed);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is always available on AArch64
        SSE2_SUPPORTED.store(true, Ordering::Relaxed);
    }

    FEATURES_DETECTED.store(true, Ordering::Relaxed);
}

/// Check if AVX2 is available
#[inline]
pub fn has_avx2() -> bool {
    detect_features();
    AVX2_SUPPORTED.load(Ordering::Relaxed)
}

/// Check if SSE2 is available
#[inline]
pub fn has_sse2() -> bool {
    detect_features();
    SSE2_SUPPORTED.load(Ordering::Relaxed)
}

// ============================================================================
// ASCII Detection
// ============================================================================

/// Check if a string is ASCII-only using SIMD
///
/// Returns true if all bytes are in the ASCII range (0-127)
pub fn is_ascii_simd(s: &str) -> bool {
    let bytes = s.as_bytes();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && bytes.len() >= 32 {
            return unsafe { is_ascii_avx2(bytes) };
        }
        if has_sse2() && bytes.len() >= 16 {
            return unsafe { is_ascii_sse2(bytes) };
        }
    }

    // Scalar fallback
    bytes.iter().all(|&b| b < 128)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_ascii_avx2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        let high_bits = _mm256_movemask_epi8(chunk);
        if high_bits != 0 {
            return false;
        }
        i += 32;
    }

    // Process remaining bytes
    while i < len {
        if bytes[i] >= 128 {
            return false;
        }
        i += 1;
    }

    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn is_ascii_sse2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
        let high_bits = _mm_movemask_epi8(chunk);
        if high_bits != 0 {
            return false;
        }
        i += 16;
    }

    // Process remaining bytes
    while i < len {
        if bytes[i] >= 128 {
            return false;
        }
        i += 1;
    }

    true
}

// ============================================================================
// Case Conversion (ASCII fast path)
// ============================================================================

/// Convert ASCII string to uppercase using SIMD
///
/// For non-ASCII strings, falls back to standard conversion
pub fn to_uppercase_simd(s: &str) -> String {
    if is_ascii_simd(s) {
        let bytes = s.as_bytes();

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() && bytes.len() >= 32 {
                return unsafe { to_uppercase_avx2(bytes) };
            }
            if has_sse2() && bytes.len() >= 16 {
                return unsafe { to_uppercase_sse2(bytes) };
            }
        }

        // Scalar ASCII fast path
        return to_uppercase_ascii_scalar(bytes);
    }

    // Non-ASCII fallback
    s.to_uppercase()
}

/// Convert ASCII string to lowercase using SIMD
pub fn to_lowercase_simd(s: &str) -> String {
    if is_ascii_simd(s) {
        let bytes = s.as_bytes();

        #[cfg(target_arch = "x86_64")]
        {
            if has_avx2() && bytes.len() >= 32 {
                return unsafe { to_lowercase_avx2(bytes) };
            }
            if has_sse2() && bytes.len() >= 16 {
                return unsafe { to_lowercase_sse2(bytes) };
            }
        }

        // Scalar ASCII fast path
        return to_lowercase_ascii_scalar(bytes);
    }

    // Non-ASCII fallback
    s.to_lowercase()
}

fn to_uppercase_ascii_scalar(bytes: &[u8]) -> String {
    let mut result = Vec::with_capacity(bytes.len());
    for &b in bytes {
        if b >= b'a' && b <= b'z' {
            result.push(b - 32);
        } else {
            result.push(b);
        }
    }
    unsafe { String::from_utf8_unchecked(result) }
}

fn to_lowercase_ascii_scalar(bytes: &[u8]) -> String {
    let mut result = Vec::with_capacity(bytes.len());
    for &b in bytes {
        if b >= b'A' && b <= b'Z' {
            result.push(b + 32);
        } else {
            result.push(b);
        }
    }
    unsafe { String::from_utf8_unchecked(result) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn to_uppercase_avx2(bytes: &[u8]) -> String {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut result: Vec<u8> = Vec::with_capacity(len);
    result.set_len(len);

    let lower_a = _mm256_set1_epi8(b'a' as i8);
    let lower_z = _mm256_set1_epi8(b'z' as i8);
    let diff = _mm256_set1_epi8(32);

    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check if char is in range [a-z]
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(lower_a, _mm256_set1_epi8(1)));
        let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(lower_z, _mm256_set1_epi8(1)), chunk);
        let is_lower = _mm256_and_si256(ge_a, le_z);

        // Subtract 32 from lowercase letters
        let to_sub = _mm256_and_si256(is_lower, diff);
        let converted = _mm256_sub_epi8(chunk, to_sub);

        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut __m256i, converted);
        i += 32;
    }

    // Process remaining bytes
    while i < len {
        let b = bytes[i];
        result[i] = if b >= b'a' && b <= b'z' { b - 32 } else { b };
        i += 1;
    }

    String::from_utf8_unchecked(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn to_uppercase_sse2(bytes: &[u8]) -> String {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut result: Vec<u8> = Vec::with_capacity(len);
    result.set_len(len);

    let lower_a = _mm_set1_epi8(b'a' as i8);
    let lower_z = _mm_set1_epi8(b'z' as i8);
    let diff = _mm_set1_epi8(32);

    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Check if char is in range [a-z]
        let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(lower_a, _mm_set1_epi8(1)));
        let le_z = _mm_cmpgt_epi8(_mm_add_epi8(lower_z, _mm_set1_epi8(1)), chunk);
        let is_lower = _mm_and_si128(ge_a, le_z);

        // Subtract 32 from lowercase letters
        let to_sub = _mm_and_si128(is_lower, diff);
        let converted = _mm_sub_epi8(chunk, to_sub);

        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, converted);
        i += 16;
    }

    // Process remaining bytes
    while i < len {
        let b = bytes[i];
        result[i] = if b >= b'a' && b <= b'z' { b - 32 } else { b };
        i += 1;
    }

    String::from_utf8_unchecked(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn to_lowercase_avx2(bytes: &[u8]) -> String {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut result: Vec<u8> = Vec::with_capacity(len);
    result.set_len(len);

    let upper_a = _mm256_set1_epi8(b'A' as i8);
    let upper_z = _mm256_set1_epi8(b'Z' as i8);
    let diff = _mm256_set1_epi8(32);

    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check if char is in range [A-Z]
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(upper_a, _mm256_set1_epi8(1)));
        let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_z, _mm256_set1_epi8(1)), chunk);
        let is_upper = _mm256_and_si256(ge_a, le_z);

        // Add 32 to uppercase letters
        let to_add = _mm256_and_si256(is_upper, diff);
        let converted = _mm256_add_epi8(chunk, to_add);

        _mm256_storeu_si256(result.as_mut_ptr().add(i) as *mut __m256i, converted);
        i += 32;
    }

    // Process remaining bytes
    while i < len {
        let b = bytes[i];
        result[i] = if b >= b'A' && b <= b'Z' { b + 32 } else { b };
        i += 1;
    }

    String::from_utf8_unchecked(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn to_lowercase_sse2(bytes: &[u8]) -> String {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut result: Vec<u8> = Vec::with_capacity(len);
    result.set_len(len);

    let upper_a = _mm_set1_epi8(b'A' as i8);
    let upper_z = _mm_set1_epi8(b'Z' as i8);
    let diff = _mm_set1_epi8(32);

    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Check if char is in range [A-Z]
        let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(upper_a, _mm_set1_epi8(1)));
        let le_z = _mm_cmpgt_epi8(_mm_add_epi8(upper_z, _mm_set1_epi8(1)), chunk);
        let is_upper = _mm_and_si128(ge_a, le_z);

        // Add 32 to uppercase letters
        let to_add = _mm_and_si128(is_upper, diff);
        let converted = _mm_add_epi8(chunk, to_add);

        _mm_storeu_si128(result.as_mut_ptr().add(i) as *mut __m128i, converted);
        i += 16;
    }

    // Process remaining bytes
    while i < len {
        let b = bytes[i];
        result[i] = if b >= b'A' && b <= b'Z' { b + 32 } else { b };
        i += 1;
    }

    String::from_utf8_unchecked(result)
}

// ============================================================================
// Character Classification (Batch Operations)
// ============================================================================

/// Count ASCII digits in a string using SIMD
pub fn count_digits_simd(s: &str) -> usize {
    let bytes = s.as_bytes();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && bytes.len() >= 32 {
            return unsafe { count_digits_avx2(bytes) };
        }
        if has_sse2() && bytes.len() >= 16 {
            return unsafe { count_digits_sse2(bytes) };
        }
    }

    // Scalar fallback
    bytes.iter().filter(|&&b| b >= b'0' && b <= b'9').count()
}

/// Count ASCII alphabetic characters using SIMD
pub fn count_alpha_simd(s: &str) -> usize {
    let bytes = s.as_bytes();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && bytes.len() >= 32 {
            return unsafe { count_alpha_avx2(bytes) };
        }
        if has_sse2() && bytes.len() >= 16 {
            return unsafe { count_alpha_sse2(bytes) };
        }
    }

    // Scalar fallback
    bytes
        .iter()
        .filter(|&&b| (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z'))
        .count()
}

/// Count whitespace characters using SIMD
pub fn count_whitespace_simd(s: &str) -> usize {
    let bytes = s.as_bytes();

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && bytes.len() >= 32 {
            return unsafe { count_whitespace_avx2(bytes) };
        }
        if has_sse2() && bytes.len() >= 16 {
            return unsafe { count_whitespace_sse2(bytes) };
        }
    }

    // Scalar fallback
    bytes
        .iter()
        .filter(|&&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
        .count()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_digits_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let digit_0 = _mm256_set1_epi8(b'0' as i8);
    let digit_9 = _mm256_set1_epi8(b'9' as i8);

    // Process 32 bytes at a time
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check range [0-9]
        let ge_0 = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(digit_0, _mm256_set1_epi8(1)));
        let le_9 = _mm256_cmpgt_epi8(_mm256_add_epi8(digit_9, _mm256_set1_epi8(1)), chunk);
        let is_digit = _mm256_and_si256(ge_0, le_9);

        // Count set bits
        let mask = _mm256_movemask_epi8(is_digit) as u32;
        count += mask.count_ones() as usize;
        i += 32;
    }

    // Process remaining bytes
    while i < len {
        if bytes[i] >= b'0' && bytes[i] <= b'9' {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn count_digits_sse2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let digit_0 = _mm_set1_epi8(b'0' as i8);
    let digit_9 = _mm_set1_epi8(b'9' as i8);

    // Process 16 bytes at a time
    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Check range [0-9]
        let ge_0 = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(digit_0, _mm_set1_epi8(1)));
        let le_9 = _mm_cmpgt_epi8(_mm_add_epi8(digit_9, _mm_set1_epi8(1)), chunk);
        let is_digit = _mm_and_si128(ge_0, le_9);

        // Count set bits
        let mask = _mm_movemask_epi8(is_digit) as u32;
        count += mask.count_ones() as usize;
        i += 16;
    }

    // Process remaining bytes
    while i < len {
        if bytes[i] >= b'0' && bytes[i] <= b'9' {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_alpha_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let lower_a = _mm256_set1_epi8(b'a' as i8);
    let lower_z = _mm256_set1_epi8(b'z' as i8);
    let upper_a = _mm256_set1_epi8(b'A' as i8);
    let upper_z = _mm256_set1_epi8(b'Z' as i8);
    let one = _mm256_set1_epi8(1);

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check lowercase [a-z]
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(lower_a, one));
        let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(lower_z, one), chunk);
        let is_lower = _mm256_and_si256(ge_a, le_z);

        // Check uppercase [A-Z]
        let ge_upper = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(upper_a, one));
        let le_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_z, one), chunk);
        let is_upper = _mm256_and_si256(ge_upper, le_upper);

        // Combine
        let is_alpha = _mm256_or_si256(is_lower, is_upper);
        let mask = _mm256_movemask_epi8(is_alpha) as u32;
        count += mask.count_ones() as usize;
        i += 32;
    }

    // Scalar remainder
    while i < len {
        let b = bytes[i];
        if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn count_alpha_sse2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let lower_a = _mm_set1_epi8(b'a' as i8);
    let lower_z = _mm_set1_epi8(b'z' as i8);
    let upper_a = _mm_set1_epi8(b'A' as i8);
    let upper_z = _mm_set1_epi8(b'Z' as i8);
    let one = _mm_set1_epi8(1);

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Check lowercase [a-z]
        let ge_a = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(lower_a, one));
        let le_z = _mm_cmpgt_epi8(_mm_add_epi8(lower_z, one), chunk);
        let is_lower = _mm_and_si128(ge_a, le_z);

        // Check uppercase [A-Z]
        let ge_upper = _mm_cmpgt_epi8(chunk, _mm_sub_epi8(upper_a, one));
        let le_upper = _mm_cmpgt_epi8(_mm_add_epi8(upper_z, one), chunk);
        let is_upper = _mm_and_si128(ge_upper, le_upper);

        // Combine
        let is_alpha = _mm_or_si128(is_lower, is_upper);
        let mask = _mm_movemask_epi8(is_alpha) as u32;
        count += mask.count_ones() as usize;
        i += 16;
    }

    // Scalar remainder
    while i < len {
        let b = bytes[i];
        if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_whitespace_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let cr = _mm256_set1_epi8(b'\r' as i8);

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        let is_space = _mm256_cmpeq_epi8(chunk, space);
        let is_tab = _mm256_cmpeq_epi8(chunk, tab);
        let is_newline = _mm256_cmpeq_epi8(chunk, newline);
        let is_cr = _mm256_cmpeq_epi8(chunk, cr);

        let is_ws = _mm256_or_si256(
            _mm256_or_si256(is_space, is_tab),
            _mm256_or_si256(is_newline, is_cr),
        );

        let mask = _mm256_movemask_epi8(is_ws) as u32;
        count += mask.count_ones() as usize;
        i += 32;
    }

    // Scalar remainder
    while i < len {
        let b = bytes[i];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn count_whitespace_sse2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    let space = _mm_set1_epi8(b' ' as i8);
    let tab = _mm_set1_epi8(b'\t' as i8);
    let newline = _mm_set1_epi8(b'\n' as i8);
    let cr = _mm_set1_epi8(b'\r' as i8);

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        let is_space = _mm_cmpeq_epi8(chunk, space);
        let is_tab = _mm_cmpeq_epi8(chunk, tab);
        let is_newline = _mm_cmpeq_epi8(chunk, newline);
        let is_cr = _mm_cmpeq_epi8(chunk, cr);

        let is_ws = _mm_or_si128(
            _mm_or_si128(is_space, is_tab),
            _mm_or_si128(is_newline, is_cr),
        );

        let mask = _mm_movemask_epi8(is_ws) as u32;
        count += mask.count_ones() as usize;
        i += 16;
    }

    // Scalar remainder
    while i < len {
        let b = bytes[i];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            count += 1;
        }
        i += 1;
    }

    count
}

// ============================================================================
// Pattern Matching
// ============================================================================

/// Find first occurrence of a single byte using SIMD
pub fn find_byte_simd(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && haystack.len() >= 32 {
            return unsafe { find_byte_avx2(haystack, needle) };
        }
        if has_sse2() && haystack.len() >= 16 {
            return unsafe { find_byte_sse2(haystack, needle) };
        }
    }

    // Scalar fallback
    haystack.iter().position(|&b| b == needle)
}

/// Count occurrences of a single byte using SIMD
pub fn count_byte_simd(haystack: &[u8], needle: u8) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() && haystack.len() >= 32 {
            return unsafe { count_byte_avx2(haystack, needle) };
        }
        if has_sse2() && haystack.len() >= 16 {
            return unsafe { count_byte_sse2(haystack, needle) };
        }
    }

    // Scalar fallback
    haystack.iter().filter(|&&b| b == needle).count()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(haystack: &[u8], needle: u8) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    let mut i = 0;
    let needle_vec = _mm256_set1_epi8(needle as i8);

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 32;
    }

    // Scalar remainder
    while i < len {
        if haystack[i] == needle {
            return Some(i);
        }
        i += 1;
    }

    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn find_byte_sse2(haystack: &[u8], needle: u8) -> Option<usize> {
    use std::arch::x86_64::*;

    let len = haystack.len();
    let mut i = 0;
    let needle_vec = _mm_set1_epi8(needle as i8);

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(haystack.as_ptr().add(i) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm_movemask_epi8(cmp) as u32;

        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 16;
    }

    // Scalar remainder
    while i < len {
        if haystack[i] == needle {
            return Some(i);
        }
        i += 1;
    }

    None
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_byte_avx2(haystack: &[u8], needle: u8) -> usize {
    use std::arch::x86_64::*;

    let len = haystack.len();
    let mut count = 0usize;
    let mut i = 0;
    let needle_vec = _mm256_set1_epi8(needle as i8);

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(haystack.as_ptr().add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        count += mask.count_ones() as usize;
        i += 32;
    }

    // Scalar remainder
    while i < len {
        if haystack[i] == needle {
            count += 1;
        }
        i += 1;
    }

    count
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn count_byte_sse2(haystack: &[u8], needle: u8) -> usize {
    use std::arch::x86_64::*;

    let len = haystack.len();
    let mut count = 0usize;
    let mut i = 0;
    let needle_vec = _mm_set1_epi8(needle as i8);

    while i + 16 <= len {
        let chunk = _mm_loadu_si128(haystack.as_ptr().add(i) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm_movemask_epi8(cmp) as u32;
        count += mask.count_ones() as usize;
        i += 16;
    }

    // Scalar remainder
    while i < len {
        if haystack[i] == needle {
            count += 1;
        }
        i += 1;
    }

    count
}

// ============================================================================
// Batch Operations for Series
// ============================================================================

/// Batch uppercase conversion for a vector of strings
pub fn batch_uppercase(strings: &[String]) -> Vec<String> {
    strings.iter().map(|s| to_uppercase_simd(s)).collect()
}

/// Batch lowercase conversion for a vector of strings
pub fn batch_lowercase(strings: &[String]) -> Vec<String> {
    strings.iter().map(|s| to_lowercase_simd(s)).collect()
}

/// Batch ASCII check for a vector of strings
pub fn batch_is_ascii(strings: &[String]) -> Vec<bool> {
    strings.iter().map(|s| is_ascii_simd(s)).collect()
}

/// Batch digit count for a vector of strings
pub fn batch_count_digits(strings: &[String]) -> Vec<usize> {
    strings.iter().map(|s| count_digits_simd(s)).collect()
}

/// Batch alphabetic count for a vector of strings
pub fn batch_count_alpha(strings: &[String]) -> Vec<usize> {
    strings.iter().map(|s| count_alpha_simd(s)).collect()
}

/// Batch whitespace count for a vector of strings
pub fn batch_count_whitespace(strings: &[String]) -> Vec<usize> {
    strings.iter().map(|s| count_whitespace_simd(s)).collect()
}

// ============================================================================
// Parallel Batch Operations (using Rayon)
// ============================================================================

/// Parallel batch uppercase conversion
pub fn parallel_batch_uppercase(strings: &[String]) -> Vec<String> {
    use rayon::prelude::*;
    strings.par_iter().map(|s| to_uppercase_simd(s)).collect()
}

/// Parallel batch lowercase conversion
pub fn parallel_batch_lowercase(strings: &[String]) -> Vec<String> {
    use rayon::prelude::*;
    strings.par_iter().map(|s| to_lowercase_simd(s)).collect()
}

/// Parallel batch ASCII check
pub fn parallel_batch_is_ascii(strings: &[String]) -> Vec<bool> {
    use rayon::prelude::*;
    strings.par_iter().map(|s| is_ascii_simd(s)).collect()
}

// ============================================================================
// Statistics
// ============================================================================

/// SIMD string operation statistics
#[derive(Debug, Clone, Default)]
pub struct SimdStringStats {
    /// Whether AVX2 is available
    pub avx2_available: bool,
    /// Whether SSE2 is available
    pub sse2_available: bool,
    /// Number of strings processed with SIMD
    pub simd_operations: u64,
    /// Number of strings processed with scalar fallback
    pub scalar_operations: u64,
}

impl SimdStringStats {
    /// Create new stats with current feature detection
    pub fn new() -> Self {
        detect_features();
        Self {
            avx2_available: AVX2_SUPPORTED.load(Ordering::Relaxed),
            sse2_available: SSE2_SUPPORTED.load(Ordering::Relaxed),
            simd_operations: 0,
            scalar_operations: 0,
        }
    }

    /// Get the best available SIMD level
    pub fn simd_level(&self) -> &'static str {
        if self.avx2_available {
            "AVX2 (256-bit)"
        } else if self.sse2_available {
            "SSE2 (128-bit)"
        } else {
            "Scalar (no SIMD)"
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ascii_simd() {
        assert!(is_ascii_simd("Hello, World!"));
        assert!(is_ascii_simd("12345"));
        assert!(is_ascii_simd(""));
        assert!(!is_ascii_simd("Hello, 世界!"));
        assert!(!is_ascii_simd("café"));

        // Long string test
        let long_ascii = "a".repeat(1000);
        assert!(is_ascii_simd(&long_ascii));

        let long_mixed = format!("{}世界", "a".repeat(100));
        assert!(!is_ascii_simd(&long_mixed));
    }

    #[test]
    fn test_to_uppercase_simd() {
        assert_eq!(to_uppercase_simd("hello"), "HELLO");
        assert_eq!(to_uppercase_simd("Hello World"), "HELLO WORLD");
        assert_eq!(to_uppercase_simd("123abc"), "123ABC");
        assert_eq!(to_uppercase_simd(""), "");

        // Long string test
        let long = "hello ".repeat(100);
        let expected = "HELLO ".repeat(100);
        assert_eq!(to_uppercase_simd(&long), expected);
    }

    #[test]
    fn test_to_lowercase_simd() {
        assert_eq!(to_lowercase_simd("HELLO"), "hello");
        assert_eq!(to_lowercase_simd("Hello World"), "hello world");
        assert_eq!(to_lowercase_simd("123ABC"), "123abc");
        assert_eq!(to_lowercase_simd(""), "");

        // Long string test
        let long = "HELLO ".repeat(100);
        let expected = "hello ".repeat(100);
        assert_eq!(to_lowercase_simd(&long), expected);
    }

    #[test]
    fn test_count_digits_simd() {
        assert_eq!(count_digits_simd("abc123def456"), 6);
        assert_eq!(count_digits_simd("no digits"), 0);
        assert_eq!(count_digits_simd("12345678901234567890"), 20);
        assert_eq!(count_digits_simd(""), 0);

        // Long string test
        let long = "a1b2c3d4e5".repeat(50);
        assert_eq!(count_digits_simd(&long), 250);
    }

    #[test]
    fn test_count_alpha_simd() {
        assert_eq!(count_alpha_simd("abc123DEF"), 6);
        assert_eq!(count_alpha_simd("12345"), 0);
        assert_eq!(count_alpha_simd("AbCdEfGh"), 8);
        assert_eq!(count_alpha_simd(""), 0);

        // Long string test
        let long = "a1b2c3".repeat(100);
        assert_eq!(count_alpha_simd(&long), 300);
    }

    #[test]
    fn test_count_whitespace_simd() {
        assert_eq!(count_whitespace_simd("hello world"), 1);
        assert_eq!(count_whitespace_simd("a\tb\nc\rd"), 3);
        assert_eq!(count_whitespace_simd("no_whitespace"), 0);
        // "   \t\n\r   " = 3 spaces + tab + newline + cr + 3 spaces = 9
        assert_eq!(count_whitespace_simd("   \t\n\r   "), 9);

        // Long string test
        let long = "a b ".repeat(100);
        assert_eq!(count_whitespace_simd(&long), 200);
    }

    #[test]
    fn test_find_byte_simd() {
        assert_eq!(find_byte_simd(b"hello world", b'o'), Some(4));
        assert_eq!(find_byte_simd(b"hello world", b'x'), None);
        assert_eq!(find_byte_simd(b"", b'a'), None);

        // Long string test
        let long = "a".repeat(100) + "b";
        assert_eq!(find_byte_simd(long.as_bytes(), b'b'), Some(100));
    }

    #[test]
    fn test_count_byte_simd() {
        assert_eq!(count_byte_simd(b"hello world", b'o'), 2);
        assert_eq!(count_byte_simd(b"hello world", b'l'), 3);
        assert_eq!(count_byte_simd(b"hello world", b'x'), 0);

        // Long string test
        let long = "aba".repeat(100);
        assert_eq!(count_byte_simd(long.as_bytes(), b'a'), 200);
    }

    #[test]
    fn test_batch_operations() {
        let strings = vec![
            "hello".to_string(),
            "WORLD".to_string(),
            "Test123".to_string(),
        ];

        let upper = batch_uppercase(&strings);
        assert_eq!(upper, vec!["HELLO", "WORLD", "TEST123"]);

        let lower = batch_lowercase(&strings);
        assert_eq!(lower, vec!["hello", "world", "test123"]);

        let ascii = batch_is_ascii(&strings);
        assert_eq!(ascii, vec![true, true, true]);

        let digits = batch_count_digits(&strings);
        assert_eq!(digits, vec![0, 0, 3]);
    }

    #[test]
    fn test_simd_stats() {
        let stats = SimdStringStats::new();
        println!("SIMD Level: {}", stats.simd_level());
        println!("AVX2: {}", stats.avx2_available);
        println!("SSE2: {}", stats.sse2_available);

        // Should at least have scalar
        assert!(
            stats.avx2_available
                || stats.sse2_available
                || stats.simd_level() == "Scalar (no SIMD)"
        );
    }

    #[test]
    fn test_non_ascii_fallback() {
        // Test that non-ASCII strings fall back gracefully
        assert_eq!(to_uppercase_simd("café"), "CAFÉ");
        assert_eq!(to_lowercase_simd("CAFÉ"), "café");
        assert_eq!(to_uppercase_simd("日本語"), "日本語");
    }
}

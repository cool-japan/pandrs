//! JIT-compilable array operations for numerical data
//!
//! This module provides common numerical operations that can be JIT-compiled
//! for improved performance, especially on large datasets.

use super::jit_core::{JitCompilable, JitFunction, jit};

/// Create a JIT-compiled sum function
pub fn sum() -> impl JitCompilable<Vec<f64>, f64> {
    jit("sum", |values: Vec<f64>| -> f64 {
        values.iter().sum()
    })
}

/// Create a JIT-compiled mean function
pub fn mean() -> impl JitCompilable<Vec<f64>, f64> {
    jit("mean", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    })
}

/// Create a JIT-compiled standard deviation function
pub fn std(ddof: usize) -> impl JitCompilable<Vec<f64>, f64> {
    jit("std", move |values: Vec<f64>| -> f64 {
        if values.len() <= ddof {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (*v - mean).powi(2))
            .sum::<f64>() / (values.len() - ddof) as f64;
        
        variance.sqrt()
    })
}

/// Create a JIT-compiled variance function
pub fn var(ddof: usize) -> impl JitCompilable<Vec<f64>, f64> {
    jit("var", move |values: Vec<f64>| -> f64 {
        if values.len() <= ddof {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter()
            .map(|v| (*v - mean).powi(2))
            .sum::<f64>() / (values.len() - ddof) as f64
    })
}

/// Create a JIT-compiled minimum function
pub fn min() -> impl JitCompilable<Vec<f64>, f64> {
    jit("min", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let mut min = f64::INFINITY;
        for &val in &values {
            min = min.min(val);
        }
        min
    })
}

/// Create a JIT-compiled maximum function
pub fn max() -> impl JitCompilable<Vec<f64>, f64> {
    jit("max", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let mut max = f64::NEG_INFINITY;
        for &val in &values {
            max = max.max(val);
        }
        max
    })
}

/// Create a JIT-compiled median function
pub fn median() -> impl JitCompilable<Vec<f64>, f64> {
    jit("median", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    })
}

/// Create a JIT-compiled quantile function
pub fn quantile(q: f64) -> impl JitCompilable<Vec<f64>, f64> {
    jit("quantile", move |values: Vec<f64>| -> f64 {
        if values.is_empty() || q < 0.0 || q > 1.0 {
            return f64::NAN;
        }
        
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let idx = (sorted.len() as f64 - 1.0) * q;
        let idx_floor = idx.floor() as usize;
        let idx_ceil = idx.ceil() as usize;
        
        if idx_floor == idx_ceil {
            sorted[idx_floor]
        } else {
            let weight_ceil = idx - idx_floor as f64;
            let weight_floor = 1.0 - weight_ceil;
            
            sorted[idx_floor] * weight_floor + sorted[idx_ceil] * weight_ceil
        }
    })
}

/// Create a JIT-compiled count function
pub fn count() -> impl JitCompilable<Vec<f64>, f64> {
    jit("count", |values: Vec<f64>| -> f64 {
        values.len() as f64
    })
}

/// Create a JIT-compiled count non-NaN function
pub fn count_non_nan() -> impl JitCompilable<Vec<f64>, f64> {
    jit("count_non_nan", |values: Vec<f64>| -> f64 {
        values.iter().filter(|&x| !x.is_nan()).count() as f64
    })
}

/// Create a JIT-compiled product function
pub fn prod() -> impl JitCompilable<Vec<f64>, f64> {
    jit("prod", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return 1.0;
        }
        values.iter().fold(1.0, |acc, &x| acc * x)
    })
}

/// Create a JIT-compiled first element function
pub fn first() -> impl JitCompilable<Vec<f64>, f64> {
    jit("first", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            f64::NAN
        } else {
            values[0]
        }
    })
}

/// Create a JIT-compiled last element function
pub fn last() -> impl JitCompilable<Vec<f64>, f64> {
    jit("last", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            f64::NAN
        } else {
            *values.last().unwrap()
        }
    })
}

/// Create a JIT-compiled trimmed mean function
pub fn trimmed_mean(trim_fraction: f64) -> impl JitCompilable<Vec<f64>, f64> {
    jit("trimmed_mean", move |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        if trim_fraction <= 0.0 || trim_fraction >= 0.5 {
            // If trim is invalid, return regular mean
            return values.iter().sum::<f64>() / values.len() as f64;
        }
        
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let to_trim = (sorted.len() as f64 * trim_fraction).floor() as usize;
        let trimmed = &sorted[to_trim..sorted.len() - to_trim];
        
        if trimmed.is_empty() {
            return f64::NAN;
        }
        
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    })
}

/// Create a JIT-compiled skewness function
pub fn skew() -> impl JitCompilable<Vec<f64>, f64> {
    jit("skew", |values: Vec<f64>| -> f64 {
        if values.len() < 3 {
            return f64::NAN;
        }
        
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        
        let m2 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m3 = values.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
        
        if m2 == 0.0 {
            return 0.0;
        }
        
        // Adjust for sample bias
        let adj = (n * (n - 1.0)).sqrt() / (n - 2.0);
        adj * m3 / m2.powf(1.5)
    })
}

/// Create a JIT-compiled kurtosis function
pub fn kurt() -> impl JitCompilable<Vec<f64>, f64> {
    jit("kurt", |values: Vec<f64>| -> f64 {
        if values.len() < 4 {
            return f64::NAN;
        }
        
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        
        let m2 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let m4 = values.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
        
        if m2 == 0.0 {
            return 0.0;
        }
        
        // Convert to excess kurtosis (subtract 3.0 from result)
        // Adjust for sample bias
        let adj = (n - 1.0) * ((n + 1.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0)));
        adj * (m4 / m2.powi(2)) - 3.0
    })
}

/// Create a JIT-compiled absolute difference (max - min) function
pub fn abs_diff() -> impl JitCompilable<Vec<f64>, f64> {
    jit("abs_diff", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        
        for &val in &values {
            min = min.min(val);
            max = max.max(val);
        }
        
        max - min
    })
}

/// Create a JIT-compiled interquartile range function
pub fn iqr() -> impl JitCompilable<Vec<f64>, f64> {
    jit("iqr", |values: Vec<f64>| -> f64 {
        if values.len() < 4 {
            return f64::NAN;
        }
        
        let mut sorted = values;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate Q1 (25th percentile)
        let q1_idx = (sorted.len() as f64 * 0.25).floor() as usize;
        let q1 = sorted[q1_idx];
        
        // Calculate Q3 (75th percentile)
        let q3_idx = (sorted.len() as f64 * 0.75).floor() as usize;
        let q3 = sorted[q3_idx];
        
        q3 - q1
    })
}

/// Create a JIT-compiled weighted average function
pub fn weighted_avg() -> impl JitCompilable<Vec<f64>, f64> {
    jit("weighted_avg", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let n = values.len();
        let total_weight = (n * (n + 1)) / 2;
        
        let mut weighted_sum = 0.0;
        for (i, &val) in values.iter().enumerate() {
            weighted_sum += val * (i + 1) as f64;
        }
        
        weighted_sum / total_weight as f64
    })
}
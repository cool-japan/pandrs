//! # Just-In-Time (JIT) Compilation Module
//!
//! This module provides high-performance JIT compilation capabilities for DataFrame operations,
//! including SIMD vectorization, parallel processing, and optimized aggregations.

pub mod config;
pub mod core;
pub mod parallel;
pub mod simd;
pub mod groupby;

pub use config::{ParallelConfig, SIMDConfig, JITConfig};
pub use core::{JitCompilable, jit_f64, jit_i64, jit_string};
pub use parallel::{
    parallel_sum_f64, parallel_mean_f64, parallel_mean_f64_value, parallel_std_f64, parallel_var_f64,
    parallel_min_f64, parallel_max_f64, parallel_median_f64, parallel_custom
};
pub use simd::{
    simd_sum_f64, simd_mean_f64, simd_min_f64, simd_max_f64,
    simd_sum_i64, simd_mean_i64, simd_min_i64, simd_max_i64
};
pub use groupby::{GroupByJitExt, JitAggregation};

/// Re-export commonly used types
pub use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::*;

/// JIT compilation error types
#[derive(Debug, Clone)]
pub enum JitError {
    /// Compilation failed
    CompilationFailed(String),
    /// Runtime execution failed
    ExecutionFailed(String),
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::CompilationFailed(msg) => write!(f, "JIT compilation failed: {}", msg),
            JitError::ExecutionFailed(msg) => write!(f, "JIT execution failed: {}", msg),
            JitError::UnsupportedOperation(msg) => write!(f, "Unsupported JIT operation: {}", msg),
            JitError::InvalidConfig(msg) => write!(f, "Invalid JIT configuration: {}", msg),
        }
    }
}

impl std::error::Error for JitError {}

/// JIT Result type
pub type JitResult<T> = std::result::Result<T, JitError>;
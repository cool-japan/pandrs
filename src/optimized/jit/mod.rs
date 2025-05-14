//! JIT compilation support for performance-critical operations
//! 
//! This module provides Numba-like Just-In-Time compilation functionality for Rust,
//! allowing for efficient execution of data processing operations.
//!
//! The JIT system supports multiple numeric types (f64, f32, i64, i32) and provides
//! type-specific JIT functions for specialized performance with type safety.
//!
//! For maximum performance, SIMD (Single Instruction, Multiple Data) vectorization
//! is supported on compatible platforms, automatically utilizing CPU vector instructions.
//!
//! Additional performance is available through parallel execution on multi-core systems
//! using Rayon for multi-threading, with configurable chunk sizes and thread counts.
//! 
//! # Overview
//! 
//! The JIT compilation system in pandrs allows for:
//! 
//! - Accelerating performance-critical operations
//! - Creating custom aggregation functions that can be JIT-compiled
//! - Applying JIT functions to GroupBy operations
//! - Providing a fallback to native implementation when JIT is disabled
//! 
//! # Example
//! 
//! ```
//! use pandrs::optimized::jit::{jit, GroupByJitExt};
//! 
//! // Create a dataframe
//! let mut df = OptimizedDataFrame::new();
//! // ... add data ...
//! 
//! // Group by a column
//! let grouped = df.group_by(&["category"])?;
//! 
//! // Use a built-in JIT-compiled aggregation
//! let result1 = grouped.sum_jit("value", "sum_value")?;
//! 
//! // Create a custom JIT-compiled function
//! let weighted_mean = jit("weighted_mean", |values: Vec<f64>| -> f64 {
//!     if values.is_empty() {
//!         return 0.0;
//!     }
//!     
//!     let mut weighted_sum = 0.0;
//!     let mut weight_sum = 0.0;
//!     
//!     for (i, val) in values.iter().enumerate() {
//!         let weight = (i + 1) as f64;
//!         weighted_sum += val * weight;
//!         weight_sum += weight;
//!     }
//!     
//!     weighted_sum / weight_sum
//! });
//! 
//! // Use the custom JIT function
//! let result2 = grouped.aggregate_jit("value", weighted_mean, "weighted_mean")?;
//! ```

// Re-export important types and functions
pub mod jit_core;
pub mod array_ops;
pub mod groupby_jit;
pub mod types;
pub mod generic;
pub mod simd;
pub mod parallel;

// Core JIT functionality
pub use jit_core::{JitCompilable, GenericJitCompilable, JitFunction, jit, JitError, JitResult};

// Type system
pub use types::{JitType, JitNumeric, TypedVector, NumericValue};

// Generic JIT functions
pub use generic::{jit_f64, jit_f32, jit_i64, jit_i32, GenericJitFunction};

// SIMD vectorization
pub use simd::{SimdType, SimdJitFunction, simd_sum_f32, simd_sum_f64, simd_mean_f32, simd_mean_f64, auto_vectorize};

// Parallel execution
pub use parallel::{ParallelConfig, ParallelJitFunction, parallel_sum_f64, parallel_mean_f64, 
                     parallel_std_f64, parallel_min_f64, parallel_max_f64, parallel_custom};

// Pre-built JIT operations
pub use array_ops;

// GroupBy extension
pub use groupby_jit::GroupByJitExt;
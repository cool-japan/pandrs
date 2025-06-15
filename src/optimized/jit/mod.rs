//! # Enhanced Just-In-Time (JIT) Compilation Module
//!
//! This module provides high-performance JIT compilation capabilities for DataFrame operations,
//! including SIMD vectorization, parallel processing, optimized aggregations, adaptive optimization,
//! intelligent caching, and expression tree optimization.

// Core JIT modules
pub mod config;
pub mod core;
pub mod groupby;
pub mod parallel;
pub mod simd;
pub mod simd_column_ops;

// Enhanced JIT modules
pub mod adaptive_optimizer;
pub mod cache;
pub mod expression_tree;
pub mod jit_core;
pub mod performance_monitor;
pub mod types;

// Integration modules
// TODO: Fix complex recursive trait bounds for JIT DataFrame integration
// pub mod dataframe_integration;

// Re-export core types
pub use config::{JITConfig, ParallelConfig, SIMDConfig};
pub use core::{jit_f64, jit_i64, jit_string, JitCompilable};
pub use groupby::{GroupByJitExt, JitAggregation};
pub use parallel::{
    parallel_custom, parallel_max_f64, parallel_mean_f64, parallel_mean_f64_value,
    parallel_median_f64, parallel_min_f64, parallel_std_f64, parallel_sum_f64, parallel_var_f64,
};
pub use simd::{
    simd_max_f64, simd_max_i64, simd_mean_f64, simd_mean_i64, simd_min_f64, simd_min_i64,
    simd_sum_f64, simd_sum_i64,
};
pub use simd_column_ops::{
    simd_abs_f64, simd_abs_i64, simd_add_f64, simd_add_i64, simd_add_scalar_f64,
    simd_add_scalar_i64, simd_compare_f64, simd_compare_i64, simd_divide_f64, simd_multiply_f64,
    simd_multiply_i64, simd_multiply_scalar_f64, simd_sqrt_f64, simd_subtract_f64,
    simd_subtract_i64, ComparisonOp,
};

// Re-export enhanced types
pub use adaptive_optimizer::{AdaptiveOptimizer, OptimizationReport};
pub use cache::{CachedFunction, FunctionId, JitFunctionCache};
pub use expression_tree::{BinaryOperator, ExpressionNode, ExpressionTree, UnaryOperator};
pub use jit_core::{JitFunction, JitStats};
pub use performance_monitor::{
    FunctionPerformanceMetrics, JitPerformanceMonitor, OptimizationSuggestion,
};
pub use types::{JitNumeric, NumericValue, TypedVector};
// TODO: Re-enable after fixing trait bounds
// pub use dataframe_integration::{JitDataFrameOps, JitOptimizedDataFrame};

/// Re-export commonly used types
pub use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
pub use std::arch::x86_64::*;

/// Enhanced JIT compilation error types
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
    /// Cache operation failed
    CacheError(String),
    /// Performance monitoring error
    MonitoringError(String),
    /// Expression tree optimization error
    OptimizationError(String),
    /// Type system error
    TypeError(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::CompilationFailed(msg) => write!(f, "JIT compilation failed: {}", msg),
            JitError::ExecutionFailed(msg) => write!(f, "JIT execution failed: {}", msg),
            JitError::UnsupportedOperation(msg) => write!(f, "Unsupported JIT operation: {}", msg),
            JitError::InvalidConfig(msg) => write!(f, "Invalid JIT configuration: {}", msg),
            JitError::CacheError(msg) => write!(f, "JIT cache error: {}", msg),
            JitError::MonitoringError(msg) => write!(f, "JIT monitoring error: {}", msg),
            JitError::OptimizationError(msg) => write!(f, "JIT optimization error: {}", msg),
            JitError::TypeError(msg) => write!(f, "JIT type error: {}", msg),
        }
    }
}

impl std::error::Error for JitError {}

/// JIT Result type
pub type JitResult<T> = std::result::Result<T, JitError>;

/// Global JIT system initialization
pub fn initialize_jit_system(config: JITConfig) -> crate::core::error::Result<()> {
    // Initialize global cache
    cache::init_global_cache(128)?; // 128MB default cache size

    // Initialize global performance monitor
    performance_monitor::init_global_monitor(config.clone())?;

    Ok(())
}

/// Get global JIT system statistics
pub fn get_jit_system_stats() -> JitSystemStats {
    let cache_stats = cache::get_global_cache().get_stats();
    let monitor = performance_monitor::get_global_monitor();
    let system_metrics = monitor.get_system_metrics();

    JitSystemStats {
        cache_hit_rate: cache_stats.hit_rate,
        cache_utilization: cache_stats.utilization_percent(),
        active_functions: system_metrics.active_functions,
        jit_utilization: system_metrics.jit_utilization,
        total_compilations: system_metrics.total_compilations,
        failed_compilations: system_metrics.failed_compilations,
        avg_compilation_time_ns: system_metrics.avg_compilation_time_ns,
        uptime: system_metrics.uptime,
    }
}

/// JIT system statistics
#[derive(Debug, Clone)]
pub struct JitSystemStats {
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Cache utilization percentage
    pub cache_utilization: f64,
    /// Number of active JIT functions
    pub active_functions: usize,
    /// JIT system utilization (0.0 to 1.0)
    pub jit_utilization: f64,
    /// Total number of compilations
    pub total_compilations: u64,
    /// Number of failed compilations
    pub failed_compilations: u64,
    /// Average compilation time in nanoseconds
    pub avg_compilation_time_ns: f64,
    /// System uptime
    pub uptime: std::time::Duration,
}

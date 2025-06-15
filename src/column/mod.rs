mod boolean_column;
mod common;
mod float64_column;
mod int64_column;
mod string_column;
pub mod string_pool;
// mod zero_copy_string_column; // Temporarily disabled due to Send/Sync issues
mod simple_zero_copy_string_column;
pub mod simd_operations;

// Core column types (canonical location)
pub use crate::core::column::{BitMask, Column, ColumnTrait, ColumnType};

// Specific column implementations
pub use boolean_column::BooleanColumn;
pub use float64_column::Float64Column;
pub use int64_column::Int64Column;
pub use string_column::StringColumn;
pub use string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
pub use string_pool::StringPool;
// pub use zero_copy_string_column::{ZeroCopyStringColumn, ZeroCopyStringOps}; // Temporarily disabled
pub use simple_zero_copy_string_column::{SimpleZeroCopyStringColumn, SimpleZeroCopyStringOps};
pub use simd_operations::{SIMDFloat64Ops, SIMDInt64Ops, SIMDColumnArithmetic};

// Re-export column utility functions from core
pub use crate::core::column::utils;

// Expose internal implementation of string column (for benchmarking)
pub mod string_column_impl {
    pub use super::string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
}

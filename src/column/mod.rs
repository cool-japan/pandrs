mod boolean_column;
mod common;
mod float64_column;
mod int64_column;
mod string_column;
pub mod string_pool;
// mod zero_copy_string_column; // Temporarily disabled due to Send/Sync issues
pub mod simd_operations;
mod simple_zero_copy_string_column;

// Core column types (canonical location)
pub use crate::core::column::{BitMask, Column, ColumnTrait, ColumnType};

// Specific column implementations
pub use boolean_column::BooleanColumn;
pub use float64_column::Float64Column;
pub use int64_column::Int64Column;
pub use string_column::StringColumn;
pub use string_column::{
    default_optimization_mode, set_default_optimization_mode, StringColumnOptimizationMode,
};
pub use string_pool::StringPool;
// pub use zero_copy_string_column::{ZeroCopyStringColumn, ZeroCopyStringOps}; // Temporarily disabled
pub use simd_operations::{SIMDColumnArithmetic, SIMDFloat64Ops, SIMDInt64Ops};
pub use simple_zero_copy_string_column::{SimpleZeroCopyStringColumn, SimpleZeroCopyStringOps};

// Re-export column utility functions from core
pub use crate::core::column::utils;

// Expose internal implementation of string column (for benchmarking)
//
// NOTE: `DEFAULT_OPTIMIZATION_MODE` used to be re-exported here as a
// `pub static mut` (see `string_column.rs` for why that was unsound and was
// replaced with an atomic + accessor functions). Code that directly
// assigned to it -- `examples/string_optimization_benchmark.rs` was the
// only such caller in this repo -- has been switched to
// `set_default_optimization_mode(..)`; any downstream consumer of this
// crate doing the same will need the equivalent update.
pub mod string_column_impl {
    pub use super::string_column::{
        default_optimization_mode, set_default_optimization_mode, StringColumnOptimizationMode,
    };
}

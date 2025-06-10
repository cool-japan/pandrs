mod boolean_column;
mod common;
mod float64_column;
mod int64_column;
mod string_column;
pub mod string_pool;

pub use crate::core::column::BitMask;
pub use boolean_column::BooleanColumn;
pub use common::{Column, ColumnTrait, ColumnType};
pub use float64_column::Float64Column;
pub use int64_column::Int64Column;
pub use string_column::StringColumn;
pub use string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
pub use string_pool::StringPool;

// Expose internal implementation of string column (for benchmarking)
pub mod string_column_impl {
    pub use super::string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
}

// Re-export column utility functions
pub use common::utils;

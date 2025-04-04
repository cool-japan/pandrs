mod int64_column;
mod float64_column;
mod string_column;
mod boolean_column;
mod common;
mod string_pool;

pub use int64_column::Int64Column;
pub use float64_column::Float64Column;
pub use string_column::StringColumn;
pub use string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
pub use boolean_column::BooleanColumn;
pub use common::{Column, ColumnType, ColumnTrait, BitMask};
pub use string_pool::StringPool;

// 文字列列の内部実装を公開（ベンチマーク用）
pub mod string_column_impl {
    pub use super::string_column::{StringColumnOptimizationMode, DEFAULT_OPTIMIZATION_MODE};
}

// Re-export column utility functions
pub use common::utils;
//! Split implementation of OptimizedDataFrame

// Core module
pub mod core;

// Column operations
pub mod column_ops;

// Data operations
pub mod data_ops;

// Input/output
pub mod io;

// Join operations
pub mod join;

// Grouping and aggregation
pub mod group;

// Index operations
pub mod index;

// Row operations
pub mod row_ops;

// Function application
pub mod apply;

// Parallel processing
pub mod parallel;

// Selection operations
pub mod select;

// Aggregation operations
pub mod aggregate;

// Sort operations
pub mod sort;

// Serialization operations
pub mod serialize;

// ColumnView implementation
pub mod column_view;

// Statistical functions module
pub mod stats;

// GPU acceleration
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export
pub use core::{ColumnView, OptimizedDataFrame};

// Re-export I/O types
#[cfg(feature = "parquet")]
pub use io::ParquetCompression;
pub use serialize::JsonOrient;

// Re-export join types
pub use join::JoinType;

// Re-export grouping types
pub use group::{AggregateOp, GroupBy};

// Re-export stats types
pub use stats::{StatDescribe, StatResult};

//! Grouping and aggregation functionality for OptimizedDataFrame
//!
//! This module is organized into four sub-modules:
//! - `types`: Core types, enums, and structs
//! - `grouping`: Group creation logic and parallel grouping operations
//! - `aggregation`: Core aggregation implementations
//! - `operations`: Transform, filter, and convenience methods

pub mod aggregation;
pub mod grouping;
pub mod operations;
pub mod types;

// Re-export main types for convenience
pub use types::{AggregateFn, AggregateOp, CustomAggregation, FilterFn, GroupBy, TransformFn};

// The grouping methods are implemented directly on OptimizedDataFrame via the grouping module
// The aggregation and operations methods are implemented on GroupBy via their respective modules

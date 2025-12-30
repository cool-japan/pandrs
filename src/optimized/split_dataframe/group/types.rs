//! Core types, enums, and structs for grouping functionality

use std::collections::HashMap;
use std::sync::Arc;

use super::super::core::OptimizedDataFrame;
use crate::error::Result;

/// Enumeration representing aggregation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count
    Count,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Median
    Median,
    /// First
    First,
    /// Last
    Last,
    /// Custom (requires custom function)
    Custom,
}

/// Type for a filter function that determines if a group should be included in the result
pub type FilterFn = Arc<dyn Fn(&OptimizedDataFrame) -> bool + Send + Sync>;

/// Type for a transform function that transforms each group's data
pub type TransformFn = Arc<dyn Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync>;

/// Type for a custom aggregation function
pub type AggregateFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// Structure representing grouping results
pub struct GroupBy<'a> {
    /// Original DataFrame
    pub df: &'a OptimizedDataFrame,
    /// Grouping key columns
    pub group_by_columns: Vec<String>,
    /// Row indices for each group
    pub groups: HashMap<Vec<String>, Vec<usize>>,
    /// Whether to create multi-index for result
    pub create_multi_index: bool,
}

/// Structure to represent a custom aggregation operation
pub struct CustomAggregation {
    /// Column name to aggregate
    pub column: String,
    /// Aggregation operation to perform
    pub op: AggregateOp,
    /// Result column name
    pub result_name: String,
    /// Optional custom aggregation function (required for AggregateOp::Custom)
    pub custom_fn: Option<AggregateFn>,
}

//! Core types, enums, and structs for grouping functionality

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use super::super::core::OptimizedDataFrame;
use crate::column::Column;
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

/// One component of a composite group key, keeping the source column's type.
///
/// Typed key components avoid the `String` allocation per row per key column
/// that a stringified key requires, and they make distinct values that share a
/// textual rendering (or a separator character) impossible to conflate.
///
/// Missing values are represented by [`GroupKeyValue::Null`] rather than by a
/// literal `"NULL"` / `"NA"` string, so a genuine `"NULL"` cell can never be
/// merged into the missing-value group. `NaN` in a float column is mapped onto
/// `Null` as well, matching pandas where `NaN` *is* the float64 missing marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKeyValue<'a> {
    /// Missing (NA) key component
    Null,
    /// 64-bit integer key
    Int64(i64),
    /// Float key held as canonical bits (`-0.0` is normalised to `0.0`)
    Float64(u64),
    /// Boolean key
    Boolean(bool),
    /// String key borrowed from the source column (no allocation)
    String(&'a str),
}

impl<'a> GroupKeyValue<'a> {
    /// Extract the key component for `row` from `column`.
    pub fn from_column(column: &'a Column, row: usize) -> Self {
        match column {
            Column::Int64(col) => match col.get(row) {
                Ok(Some(value)) => GroupKeyValue::Int64(value),
                _ => GroupKeyValue::Null,
            },
            Column::Float64(col) => match col.get(row) {
                Ok(Some(value)) if !value.is_nan() => {
                    // Normalise -0.0 to 0.0 so the two do not form two groups.
                    let canonical = if value == 0.0 { 0.0 } else { value };
                    GroupKeyValue::Float64(canonical.to_bits())
                }
                _ => GroupKeyValue::Null,
            },
            Column::String(col) => match col.get(row) {
                Ok(Some(value)) => GroupKeyValue::String(value),
                _ => GroupKeyValue::Null,
            },
            Column::Boolean(col) => match col.get(row) {
                Ok(Some(value)) => GroupKeyValue::Boolean(value),
                _ => GroupKeyValue::Null,
            },
        }
    }

    /// Whether this component is a missing value.
    pub fn is_null(&self) -> bool {
        matches!(self, GroupKeyValue::Null)
    }

    /// Textual rendering of the key component, or `None` when it is missing.
    ///
    /// `None` is deliberately *not* rendered as a string: callers must decide
    /// how to represent NA in their output (a NULL cell, an explicit marker, …).
    pub fn to_value_string(&self) -> Option<String> {
        match self {
            GroupKeyValue::Null => None,
            GroupKeyValue::Int64(value) => Some(value.to_string()),
            GroupKeyValue::Float64(bits) => Some(f64::from_bits(*bits).to_string()),
            GroupKeyValue::Boolean(value) => Some(value.to_string()),
            GroupKeyValue::String(value) => Some((*value).to_string()),
        }
    }

    /// Variant rank used for the total order (missing values sort last).
    fn order_rank(&self) -> u8 {
        match self {
            GroupKeyValue::Int64(_) => 0,
            GroupKeyValue::Float64(_) => 1,
            GroupKeyValue::Boolean(_) => 2,
            GroupKeyValue::String(_) => 3,
            GroupKeyValue::Null => 4,
        }
    }
}

impl Ord for GroupKeyValue<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (GroupKeyValue::Null, GroupKeyValue::Null) => Ordering::Equal,
            (GroupKeyValue::Int64(a), GroupKeyValue::Int64(b)) => a.cmp(b),
            // Compare floats through the total order, never `partial_cmp`.
            (GroupKeyValue::Float64(a), GroupKeyValue::Float64(b)) => {
                f64::from_bits(*a).total_cmp(&f64::from_bits(*b))
            }
            (GroupKeyValue::Boolean(a), GroupKeyValue::Boolean(b)) => a.cmp(b),
            (GroupKeyValue::String(a), GroupKeyValue::String(b)) => a.cmp(b),
            // Mixed variants only occur across differently typed key columns.
            _ => self.order_rank().cmp(&other.order_rank()),
        }
    }
}

impl PartialOrd for GroupKeyValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Type for a filter function that determines if a group should be included in the result
pub type FilterFn = Arc<dyn Fn(&OptimizedDataFrame) -> bool + Send + Sync>;

/// Type for a transform function that transforms each group's data
pub type TransformFn = Arc<dyn Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync>;

/// Type for a custom aggregation function
pub type AggregateFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

/// A composite group key: one [`GroupKeyValue`] per grouping column.
pub type GroupKey<'a> = Vec<GroupKeyValue<'a>>;

/// Structure representing grouping results
pub struct GroupBy<'a> {
    /// Original DataFrame
    pub df: &'a OptimizedDataFrame,
    /// Grouping key columns
    pub group_by_columns: Vec<String>,
    /// Row indices for each group, keyed by the typed composite group key
    pub groups: HashMap<GroupKey<'a>, Vec<usize>>,
    /// Whether to create multi-index for result
    pub create_multi_index: bool,
    /// Whether rows with a missing key component were dropped
    /// (pandas `dropna=True` default)
    pub dropna: bool,
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

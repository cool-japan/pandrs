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

/// Enumeration representing join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join (only rows that exist in both tables)
    Inner,
    /// Left join (all rows from the left table and matching rows from the right table)
    Left,
    /// Right join (all rows from the right table and matching rows from the left table)
    Right,
    /// Outer join (all rows from both tables)
    Outer,
}

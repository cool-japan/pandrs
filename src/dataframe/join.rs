use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Enum for join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join (only rows that match in both tables)
    Inner,
    /// Left join (all rows from the left table and matching rows from the right table)
    Left,
    /// Right join (all rows from the right table and matching rows from the left table)
    Right,
    /// Outer join (all rows from both tables)
    Outer,
}

/// Join functionality for DataFrames
pub trait JoinExt {
    /// Join two DataFrames
    fn join(&self, other: &Self, on: &str, join_type: JoinType) -> Result<Self>
    where
        Self: Sized;

    /// Perform inner join
    fn inner_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform left join
    fn left_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform right join
    fn right_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform outer join
    fn outer_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;
}

/// Implementation of JoinExt for DataFrame
impl JoinExt for DataFrame {
    fn join(&self, other: &Self, on: &str, join_type: JoinType) -> Result<Self> {
        // Check if the join column exists
        if !self.contains_column(on) {
            return Err(Error::ColumnNotFound(format!(
                "Join column '{}' does not exist in the left DataFrame",
                on
            )));
        }

        if !other.contains_column(on) {
            return Err(Error::ColumnNotFound(format!(
                "Join column '{}' does not exist in the right DataFrame",
                on
            )));
        }

        match join_type {
            JoinType::Inner => self.inner_join(other, on),
            JoinType::Left => self.left_join(other, on),
            JoinType::Right => self.right_join(other, on),
            JoinType::Outer => self.outer_join(other, on),
        }
    }

    #[allow(unconditional_recursion)]
    fn inner_join(&self, other: &Self, on: &str) -> Result<Self> {
        // Forward to the legacy implementation for now
        // This would be replaced with a full implementation later
        let legacy_self = crate::dataframe::DataFrame::new();
        let legacy_other = crate::dataframe::DataFrame::new();

        let _ = legacy_self.inner_join(&legacy_other, on)?;

        Ok(DataFrame::new())
    }

    #[allow(unconditional_recursion)]
    fn left_join(&self, other: &Self, on: &str) -> Result<Self> {
        // Forward to the legacy implementation for now
        // This would be replaced with a full implementation later
        let legacy_self = crate::dataframe::DataFrame::new();
        let legacy_other = crate::dataframe::DataFrame::new();

        let _ = legacy_self.left_join(&legacy_other, on)?;

        Ok(DataFrame::new())
    }

    fn right_join(&self, other: &Self, on: &str) -> Result<Self> {
        // Right join is implemented as a left join with swapped arguments
        other.left_join(self, on)
    }

    #[allow(unconditional_recursion)]
    fn outer_join(&self, other: &Self, on: &str) -> Result<Self> {
        // Forward to the legacy implementation for now
        // This would be replaced with a full implementation later
        let legacy_self = crate::dataframe::DataFrame::new();
        let legacy_other = crate::dataframe::DataFrame::new();

        let _ = legacy_self.outer_join(&legacy_other, on)?;

        Ok(DataFrame::new())
    }
}

/// Re-export JoinType for backward compatibility
#[deprecated(since = "0.1.0-alpha.2", note = "Use crate::dataframe::join::JoinType")]
pub use crate::dataframe::join::JoinType as LegacyJoinType;

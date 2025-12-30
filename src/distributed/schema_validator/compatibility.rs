//! # Type Compatibility Utilities
//!
//! This module provides utilities for checking type compatibility for
//! various operations in distributed processing.

use crate::distributed::expr::ExprDataType;

/// Checks if two data types are compatible for joins
pub fn are_join_compatible(left_type: &ExprDataType, right_type: &ExprDataType) -> bool {
    match (left_type, right_type) {
        // Same types are always compatible
        (a, b) if a == b => true,

        // Integer and Float are compatible
        (ExprDataType::Integer, ExprDataType::Float)
        | (ExprDataType::Float, ExprDataType::Integer) => true,

        // Date and Timestamp are compatible
        (ExprDataType::Date, ExprDataType::Timestamp)
        | (ExprDataType::Timestamp, ExprDataType::Date) => true,

        // Other combinations are not compatible
        _ => false,
    }
}

//! # Expression Support for Distributed Processing
//!
//! This module provides support for complex expressions and user-defined functions
//! for distributed DataFrames, enabling more complex transformations and calculations.

use serde::{Deserialize, Serialize};
use std::fmt;

pub mod core;
pub mod projection;
pub mod schema;
pub mod validator;
// Re-export backward compatibility module
pub mod backward_compat;
pub use backward_compat::*;

/// Data types for expressions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExprDataType {
    /// Boolean type
    Boolean,
    /// Integer type
    Integer,
    /// Float type
    Float,
    /// String type
    String,
    /// Date type
    Date,
    /// Timestamp type
    Timestamp,
}

impl fmt::Display for ExprDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::Integer => write!(f, "BIGINT"),
            Self::Float => write!(f, "DOUBLE"),
            Self::String => write!(f, "VARCHAR"),
            Self::Date => write!(f, "DATE"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
        }
    }
}

// Re-exports for backward compatibility
pub use self::core::{BinaryOperator, Expr, Literal, UnaryOperator};
pub use self::projection::{ColumnProjection, ProjectionExt, UdfDefinition};
pub use self::schema::{ColumnMeta, ExprSchema};
pub use self::validator::{ExprValidator, InferredType};

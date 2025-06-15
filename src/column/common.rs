//! Column common types and utilities
//!
//! This module provides backward compatibility for column types.
//! All types have been moved to `crate::core::column` for better organization.

// Re-export core column types for backward compatibility
#[deprecated(
    since = "0.1.0-alpha.4",
    note = "Use crate::core::column types directly"
)]
pub use crate::core::column::*;

// Re-export core error types for backward compatibility
#[deprecated(
    since = "0.1.0-alpha.4",
    note = "Use crate::core::error types directly"
)]
pub use crate::core::error::{Error, Result};

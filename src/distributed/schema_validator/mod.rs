//! # Execution Schema Validation
//!
//! This module provides validation of execution plans against schemas,
//! ensuring type safety and preventing runtime errors.

pub mod backward_compat;
pub mod compatibility;
pub mod core;
pub mod validation;

// Re-export core types for easier access
pub use self::core::SchemaValidator;

// Re-export compatibility functions
pub use self::compatibility::are_join_compatible;

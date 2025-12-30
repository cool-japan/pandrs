//! # Query Plan Explanation
//!
//! This module provides functionality for explaining query plans, helping users
//! understand how their queries will be executed.

pub mod backward_compat;
pub mod conversion;
pub mod core;
pub mod format;
pub mod visualize;

// Re-export core types for easier access
pub use self::core::{ExplainFormat, ExplainOptions, PlanNode};

// Re-export conversion functions
pub use self::conversion::{execution_plan_to_plan_node, explain_plan};

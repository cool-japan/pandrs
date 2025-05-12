//! # Backward Compatibility Layer for Explain Module
//!
//! This module provides backward compatibility with code that was using the
//! previous organization of the explain.rs file.

// Re-export all types from the new modules
pub use crate::distributed::explain::{
    ExplainOptions, ExplainFormat, PlanNode, explain_plan
};

// The following is just to ensure documentation clarity in case someone is using
// the old path, but implementation is delegated to the new modules
#[deprecated(
    since = "0.1.0",
    note = "Use the specific modules instead: explain::core, explain::format, etc."
)]
pub type Deprecated = ();
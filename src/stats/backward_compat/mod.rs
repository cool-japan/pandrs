//! Backward compatibility module for statistics
//!
//! This module provides backward compatibility for code that uses the old
//! statistics module structure. It re-exports types and functions from
//! the new module structure with appropriate deprecation notices.

// Include the actual implementation modules
pub mod descriptive;
pub mod inference;
pub mod regression;
pub mod sampling;
pub mod categorical;
//! # Window Function Support for Distributed Processing
//!
//! This module provides window function capabilities for distributed DataFrames,
//! enabling advanced analytics like rolling calculations, cumulative aggregations,
//! and rank-based statistics.

pub mod backward_compat;
pub mod core;
pub mod functions;
pub mod operations;

// Re-export core types for easier access
pub use self::core::{WindowFrame, WindowFrameBoundary, WindowFrameType, WindowFunction};

// Re-export operations
pub use self::operations::WindowFunctionExt;

// Re-export functions from functions module
pub use self::functions::*;

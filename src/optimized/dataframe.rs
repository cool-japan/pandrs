//! Optimized DataFrame implementation
//!
//! This module provides a high-performance, columnar DataFrame implementation
//! optimized for data analysis and processing tasks.

pub mod core;
pub mod io;
pub mod operations;
pub mod stats;
pub mod transformations;

// Re-export the main public API
pub use core::{ColumnView, JsonOrient, OptimizedDataFrame};

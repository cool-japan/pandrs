//! Helper modules for PandasCompatExt trait implementation
//!
//! This module organizes the implementation logic into focused submodules
//! to keep each file under 2000 lines while maintaining all trait methods
//! in a single impl block as required by Rust.

pub mod aggregations;
pub mod comparison_ops;
pub mod indexing_ops;
pub mod math_ops;
pub mod reshape_ops;
pub mod string_ops;
pub mod window_ops;

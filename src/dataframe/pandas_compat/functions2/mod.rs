//! Auto-generated module structure
//!
//! Split implementation of PandasCompatExt trait across multiple files
//! to comply with <2000 lines policy

mod helpers;
pub mod assign_pipe;
pub mod stats_agg;
pub mod rolling_window;

// Re-export all implementations
pub use assign_pipe::*;
pub use stats_agg::*;
pub use rolling_window::*;

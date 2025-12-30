//! # Backward Compatibility Layer for Window Module
//!
//! This module provides backward compatibility with code that was using the
//! previous organization of the window.rs file.

// Re-export all types from the new modules
pub use crate::distributed::window::{
    functions, WindowFrame, WindowFrameBoundary, WindowFrameType, WindowFunction, WindowFunctionExt,
};

// The following is just to ensure documentation clarity in case someone is using
// the old path, but implementation is delegated to the new modules
#[deprecated(
    since = "0.1.0",
    note = "Use the specific modules instead: window::core, window::operations, etc."
)]
pub type Deprecated = ();

//! # Execution Engines for Distributed Processing
//!
//! This module provides implementations of execution engines for distributed processing.

// DataFusion engine
pub mod datafusion;

// Re-export engine implementations.
//
// The Ballista engine was removed: every method was `NotImplemented`, it was
// never constructed (both `ExecutorType::Ballista` sites now return
// `NotImplemented` instead of silently falling back to DataFusion), and it
// discarded its config on clone. DataFusion is the only real engine.
pub use datafusion::DataFusionEngine;

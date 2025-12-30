//! # DataFusion Integration
//!
//! This module provides integration with Apache Arrow DataFusion for distributed processing.

#[cfg(feature = "distributed")]
mod context;
#[cfg(feature = "distributed")]
mod conversion;
#[cfg(feature = "distributed")]
mod executor;
#[cfg(feature = "distributed")]
mod fault_tolerance;

#[cfg(feature = "distributed")]
pub use context::DataFusionContext;
#[cfg(feature = "distributed")]
pub use executor::DataFusionEngine;
#[cfg(feature = "distributed")]
pub use fault_tolerance::{FaultTolerantDataFusionContext, create_fault_tolerant_context};

/// Dummy implementation when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DataFusionEngine;

/// Dummy implementation when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
impl DataFusionEngine {
    /// Creates a new dummy engine
    pub fn new() -> Self {
        Self
    }
}
//! # Ballista Integration
//!
//! This module provides integration with Apache Ballista for distributed processing.

#[cfg(feature = "distributed")]
mod cluster;
#[cfg(feature = "distributed")]
mod executor;
#[cfg(feature = "distributed")]
mod scheduler;

#[cfg(feature = "distributed")]
pub use cluster::BallistaCluster;
#[cfg(feature = "distributed")]
pub use executor::BallistaEngine;

/// Dummy implementation when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct BallistaEngine;

/// Dummy implementation when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
impl BallistaEngine {
    /// Creates a new dummy engine
    pub fn new() -> Self {
        Self
    }
}
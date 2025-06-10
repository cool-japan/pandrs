//! # Core Components for Distributed Processing
//!
//! This module provides core components and types for distributed processing,
//! including configuration, DataFrame, and conversion traits.

#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};

// Re-export types from subdirectories
#[cfg(feature = "distributed")]
pub use self::config::{DistributedConfig, ExecutorType};
#[cfg(feature = "distributed")]
pub use self::context::DistributedContext;
#[cfg(feature = "distributed")]
pub use self::dataframe::DistributedDataFrame;
#[cfg(feature = "distributed")]
pub use self::partition::{Partition, PartitionSet, PartitionStrategy};
#[cfg(feature = "distributed")]
pub use self::statistics::{ColumnStatistics, ColumnValue, TableStatistics};

#[cfg(feature = "distributed")]
pub mod config;
#[cfg(feature = "distributed")]
pub mod context;
#[cfg(feature = "distributed")]
pub mod dataframe;
#[cfg(feature = "distributed")]
pub mod partition;
#[cfg(feature = "distributed")]
pub mod statistics;

/// Trait for converting a local DataFrame to a distributed DataFrame
#[cfg(feature = "distributed")]
pub trait ToDistributed {
    /// Converts a local DataFrame to a distributed DataFrame
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the distributed processing
    ///
    /// # Returns
    ///
    /// A `DistributedDataFrame` that can be used for distributed processing
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails
    fn to_distributed(&self, config: DistributedConfig) -> Result<DistributedDataFrame>;
}

/// Dummy implementation for when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedDataFrame;

/// Dummy configuration for when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedConfig;

/// Dummy implementation for the distributed module when the feature is not enabled
#[cfg(not(feature = "distributed"))]
impl DistributedConfig {
    /// Creates a new dummy configuration
    pub fn new() -> Self {
        Self
    }
}

/// Default implementation for DistributedConfig
#[cfg(not(feature = "distributed"))]
impl Default for DistributedConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait extension for error messages when distributed processing is not enabled
#[cfg(not(feature = "distributed"))]
pub trait ToDistributed {
    /// Returns an error indicating that the distributed feature is not enabled
    fn to_distributed(&self, _: DistributedConfig) -> Result<DistributedDataFrame> {
        Err(Error::FeatureNotAvailable(
            "Distributed processing is not enabled. Recompile with the 'distributed' feature flag."
                .to_string(),
        ))
    }
}

/// Implementation of ToDistributed for the DataFrame type
#[cfg(feature = "distributed")]
impl ToDistributed for DataFrame {
    fn to_distributed(&self, config: DistributedConfig) -> Result<DistributedDataFrame> {
        DistributedDataFrame::from_local(self, config)
    }
}

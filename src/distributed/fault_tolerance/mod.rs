//! # Fault Tolerance for Distributed Processing
//!
//! This module provides fault tolerance mechanisms for distributed query execution,
//! ensuring that queries can complete even when parts of the distributed system fail.

pub mod backward_compat;
pub mod checkpoint;
pub mod core;
pub mod recovery;

// Re-export core types for easier access
pub use self::core::{FailureInfo, FailureType, RecoveryStrategy, RetryPolicy};

// Re-export recovery types
pub use self::recovery::{
    FaultToleranceHandler, FaultTolerantContext, FaultTolerantExecutor, RecoveryAction,
};

// Re-export checkpoint types
pub use self::checkpoint::{CheckpointManager, ExecutionCheckpoint};

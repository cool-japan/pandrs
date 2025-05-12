//! # Backward Compatibility Layer for Fault Tolerance Module
//!
//! This module provides backward compatibility with code that was using the
//! previous organization of the fault_tolerance.rs file.

// Re-export all types from the new modules
pub use crate::distributed::fault_tolerance::{
    RetryPolicy, FailureType, FailureInfo, RecoveryStrategy,
    RecoveryAction, FaultToleranceHandler, FaultTolerantExecutor,
    FaultTolerantContext, ExecutionCheckpoint, CheckpointManager,
};

// The following is just to ensure documentation clarity in case someone is using
// the old path, but implementation is delegated to the new modules
#[deprecated(
    since = "0.1.0",
    note = "Use the specific modules instead: fault_tolerance::core, fault_tolerance::recovery, etc."
)]
pub type Deprecated = ();
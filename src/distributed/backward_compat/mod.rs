//! # Backward Compatibility Layer for Distributed Module
//!
//! This module provides backward compatibility with code that was using the
//! previous organization of the distributed module.

// Re-export types that were previously in this module directly
#[allow(deprecated)]
pub use crate::distributed::expr::backward_compat::*;

// Re-export config types for backward compatibility
#[deprecated(
    since = "0.1.0",
    note = "Use distributed::core::DistributedConfig instead"
)]
pub use crate::distributed::core::{DistributedConfig, ExecutorType, MemoryManager};

// Re-export context types for backward compatibility
#[deprecated(
    since = "0.1.0",
    note = "Use distributed::core::DistributedContext instead"
)]
pub use crate::distributed::core::{DistributedContext, ToDistributed};

// Re-export dataframe types for backward compatibility
#[deprecated(
    since = "0.1.0",
    note = "Use distributed::core::DistributedDataFrame instead"
)]
pub use crate::distributed::core::{
    DataFrameOperations, DistributedDataFrame, DistributedSeriesView,
};

// Re-export statistics types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::core::statistics instead")]
pub use crate::distributed::core::statistics::{ColumnStatistics, DistributedStatistics};

// Re-export execution types for backward compatibility
#[deprecated(since = "0.1.0", note = "Use distributed::execution instead")]
pub use crate::distributed::execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionMetrics, ExecutionPlan,
    ExecutionResult, JoinType, Operation, SortExpr,
};

// Re-export schema validator for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(
    since = "0.1.0",
    note = "Use distributed::expr::schema_validator instead"
)]
pub use crate::distributed::schema_validator::SchemaValidator;

// Re-export window operations for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::window instead")]
pub use crate::distributed::window::{
    functions as window_functions, WindowFrame, WindowFrameBoundary, WindowFrameType,
    WindowFunction, WindowFunctionExt,
};

// Re-export explain for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::explain instead")]
pub use crate::distributed::explain::{explain_plan, ExplainFormat, ExplainOptions, PlanNode};

// Re-export fault tolerance for backward compatibility
#[cfg(feature = "distributed")]
#[deprecated(since = "0.1.0", note = "Use distributed::fault_tolerance instead")]
pub use crate::distributed::fault_tolerance::{
    CheckpointManager, ExecutionCheckpoint, FailureInfo, FailureType, FaultToleranceHandler,
    FaultTolerantContext, RecoveryAction, RecoveryStrategy, RetryPolicy,
};

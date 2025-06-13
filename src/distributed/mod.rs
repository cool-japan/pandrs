//! # Distributed Processing Module
//!
//! This module provides distributed data processing capabilities for PandRS,
//! enabling it to handle datasets that exceed the memory capacity of a single machine.
//!
//! The module integrates with Arrow DataFusion for query execution and Ballista
//! for distributed computation.
//!
//! Enable with the "distributed" feature flag.

// Core module for essential components
pub mod core;

// Execution module for execution-related functionality
#[cfg(feature = "distributed")]
pub mod execution;

// Expression module for query expressions
#[cfg(feature = "distributed")]
pub mod expr;

// Engines module for different execution engines
#[cfg(feature = "distributed")]
pub mod engines;

// API module for high-level functionality (temporarily disabled)
// #[cfg(feature = "distributed")]
// pub mod api;

// Schema validator module
#[cfg(feature = "distributed")]
mod schema_validator;

// Window operations for time series (temporarily disabled)
// #[cfg(feature = "distributed")]
// pub mod window;

// Explain functionality (temporarily disabled)
// #[cfg(feature = "distributed")]
// mod explain;

// Fault tolerance functionality (temporarily disabled)
// #[cfg(feature = "distributed")]
// mod fault_tolerance;

// Backward compatibility module (temporarily disabled for compilation)
// #[cfg(feature = "distributed")]
// mod backward_compat;

// Re-exports for backward compatibility (temporarily disabled)
// #[allow(deprecated)]
// #[cfg(feature = "distributed")]
// pub use backward_compat::*;

// Re-exports for expr backward compatibility (temporarily disabled)
// #[allow(deprecated)]
// #[cfg(feature = "distributed")]
// pub use expr::backward_compat as expr_backward_compat;

// Re-exports from core
#[cfg(feature = "distributed")]
pub use core::{DistributedConfig, DistributedContext, DistributedDataFrame, ToDistributed};

// Re-exports from execution
#[cfg(feature = "distributed")]
pub use execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionMetrics, ExecutionPlan,
    ExecutionResult, JoinType, Operation, SortExpr,
};

// Re-exports from expr
#[cfg(feature = "distributed")]
pub use expr::{
    BinaryOperator, ColumnProjection, Expr, ExprDataType, Literal, ProjectionExt, UdfDefinition,
    UnaryOperator,
};

#[cfg(feature = "distributed")]
pub use expr::{ColumnMeta, ExprSchema, ExprValidator, InferredType};

// Re-exports from window (temporarily disabled)
// #[cfg(feature = "distributed")]
// pub use window::{
//     functions as window_functions, WindowFrame, WindowFrameBoundary, WindowFrameType,
//     WindowFunction, WindowFunctionExt,
// };

// Re-exports from schema_validator
#[cfg(feature = "distributed")]
pub use schema_validator::SchemaValidator;

// Re-exports from explain (temporarily disabled)
// #[cfg(feature = "distributed")]
// pub use explain::{explain_plan, ExplainFormat, ExplainOptions, PlanNode};

// Re-exports from fault_tolerance (temporarily disabled)
// #[cfg(feature = "distributed")]
// pub use fault_tolerance::{
//     CheckpointManager, ExecutionCheckpoint, FailureInfo, FailureType, FaultToleranceHandler,
//     FaultTolerantContext, RecoveryAction, RecoveryStrategy, RetryPolicy,
// };

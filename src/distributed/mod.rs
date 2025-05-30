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

// API module for high-level functionality
#[cfg(feature = "distributed")]
pub mod api;

// Schema validator module
#[cfg(feature = "distributed")]
mod schema_validator;

// Window operations for time series
#[cfg(feature = "distributed")]
pub mod window;

// Explain functionality
#[cfg(feature = "distributed")]
mod explain;

// Fault tolerance functionality
#[cfg(feature = "distributed")]
mod fault_tolerance;

// Backward compatibility module
#[cfg(feature = "distributed")]
mod backward_compat;

// Re-exports for backward compatibility
#[allow(deprecated)]
#[cfg(feature = "distributed")]
pub use backward_compat::*;

// Re-exports for expr backward compatibility
#[allow(deprecated)]
#[cfg(feature = "distributed")]
pub use expr::backward_compat;

// Re-exports from core
#[cfg(feature = "distributed")]
pub use core::{
    DistributedConfig,
    DistributedDataFrame,
    DistributedContext,
    ToDistributed,
};

// Re-exports from execution
#[cfg(feature = "distributed")]
pub use execution::{
    ExecutionEngine,
    ExecutionContext,
    ExecutionPlan,
    ExecutionResult,
    ExecutionMetrics,
    Operation,
    JoinType,
    AggregateExpr,
    SortExpr,
};

// Re-exports from expr
#[cfg(feature = "distributed")]
pub use expr::{
    Expr, Literal, BinaryOperator, UnaryOperator, ExprDataType,
    UdfDefinition, ColumnProjection, ProjectionExt,
};

#[cfg(feature = "distributed")]
pub use expr::{
    ExprSchema, ExprValidator, ColumnMeta, InferredType,
};

// Re-exports from window
#[cfg(feature = "distributed")]
pub use window::{
    WindowFrameBoundary, WindowFrameType, WindowFrame, WindowFunction, WindowFunctionExt,
    functions as window_functions,
};

// Re-exports from schema_validator
#[cfg(feature = "distributed")]
pub use schema_validator::SchemaValidator;

// Re-exports from explain
#[cfg(feature = "distributed")]
pub use explain::{ExplainOptions, ExplainFormat, PlanNode, explain_plan};

// Re-exports from fault_tolerance
#[cfg(feature = "distributed")]
pub use fault_tolerance::{
    FaultToleranceHandler, FaultTolerantContext, RetryPolicy, RecoveryStrategy,
    FailureType, FailureInfo, RecoveryAction, ExecutionCheckpoint, CheckpointManager,
};
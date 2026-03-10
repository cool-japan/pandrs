//! Lazy evaluation engine with query optimization for PandRS.
//!
//! This module provides:
//! - [`LazyFrame`] — a lazily-evaluated DataFrame wrapper
//! - [`plan`] — query plan AST (`LogicalPlan`, `Expr`, `AggExpr`, …)
//! - [`optimizer`] — optimization passes (`PredicatePushdown`, `ProjectionPushdown`, …)
//!
//! # Quick start
//!
//! ```rust,no_run
//! use pandrs::compute::lazy::{LazyFrame, Expr};
//! use pandrs::optimized::dataframe::OptimizedDataFrame;
//!
//! let df = OptimizedDataFrame::new();
//! let result = LazyFrame::scan(df)
//!     .filter(Expr::col("age").gt(Expr::lit_int(30)))
//!     .select(vec![Expr::col("name"), Expr::col("age")])
//!     .collect();
//! ```

pub mod frame;
pub mod optimizer;
pub mod plan;

// Re-export the primary public types
pub use frame::{GroupByBuilder, LazyFrame};
pub use optimizer::{
    ConstantFolding, DeadCodeElimination, Optimizer, OptimizerRule, PredicatePushdown,
    ProjectionPushdown,
};
pub use plan::{AggExpr, BinaryOp, CastType, Expr, LiteralValue, LogicalPlan, UnaryOp};

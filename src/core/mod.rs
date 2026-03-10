/// Core data structures and traits for PandRS.
///
/// This module contains the fundamental building blocks for all PandRS operations:
/// - Column types and operations
/// - Index implementations
/// - Data value representations
/// - Error types and contexts
/// - DataFrame trait definitions

/// Advanced multi-level indexing with hierarchical operations.
///
/// Provides sophisticated indexing capabilities for multi-dimensional data access.
pub mod advanced_multi_index;

/// Column types and trait implementations for type-safe data operations.
///
/// Core column type definitions and casting operations.
pub mod column;

/// Column-specific operations organized by type.
///
/// Type-specific implementations for numeric, string, boolean, datetime, and categorical columns.
pub mod column_ops;

/// Data value representation and conversions.
///
/// Unified value type for representing heterogeneous data.
pub mod data_value;

/// DataFrame trait definitions for polymorphic operations.
///
/// Defines the core traits that all DataFrame implementations must satisfy.
pub mod dataframe_traits;

/// Error types for the core module.
///
/// Defines all error conditions that can occur in core operations with context support.
pub mod error;

/// Error context and recovery utilities.
///
/// Enhanced error handling with contextual information and recovery strategies.
pub mod error_context;

/// Index trait and implementations.
///
/// Provides indexing capabilities for Series and DataFrames.
pub mod index;

/// Migration utilities for backward compatibility.
///
/// Tools for migrating between different versions of data structures.
pub mod migration;

/// Multi-index (hierarchical index) implementation.
///
/// Enables hierarchical indexing for advanced data organization.
pub mod multi_index;

/// Synchronization helpers for thread-safe operations.
///
/// Utilities for safe concurrent access to shared data structures.
pub mod sync_helpers;

// Re-exports for convenience
pub use advanced_multi_index::{
    AdvancedMultiIndex, CrossSectionResult, IndexValue, SelectionCriteria,
};
pub use column::{BitMask, Column, ColumnCast, ColumnTrait, ColumnType};
pub use column_ops::{
    BooleanColumn, BooleanColumnOps, CastErrorBehavior, CategoricalColumn, CategoricalColumnOps,
    ColumnFactory, ColumnOps, ColumnStorage, DateColumn, DateTimeColumn, DateTimeColumnOps,
    DefaultColumnFactory, DuplicateKeep, Float32Column, Float64Column, Int32Column, Int64Column,
    NumericColumnOps, PadSide, StringColumn, StringColumnOps, TimeColumn, TypedColumn,
};
pub use data_value::DataValue;
pub use dataframe_traits::{
    AggFunc, Axis, BooleanMask, ColIndexer, DataFrameAdvancedOps, DataFrameIO, DataFrameOps,
    DropNaHow, ExpandingWindow, FillMethod, GroupByOps, GroupKey, IndexingOps, JoinType,
    LabelIndexer, Resampler, RollingWindow, RowIndexer, StatisticalOps,
};
pub use error::{Error, PandRSError, Result};
pub use error_context::{
    ErrorContext, ErrorContextBuilder, ErrorRecovery, ErrorRecoveryHelper, ErrorSeverity,
};
pub use index::{Index, IndexTrait};
pub use migration::{
    BackupManager, BackupStrategy, BackwardCompatibilityLayer, MigrationExecutor, MigrationPlan,
    MigrationResult, MigrationRiskLevel, MigrationStep, Version,
};
pub use multi_index::MultiIndex;

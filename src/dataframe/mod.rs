/// DataFrame implementations module.
///
/// This module provides the core DataFrame data structure and associated operations.
/// A DataFrame is a two-dimensional, size-mutable, heterogeneous tabular data structure
/// with labeled axes (rows and columns).

/// Advanced indexing capabilities including categorical, datetime, interval, and period indexes.
///
/// Supports specialized index types for domain-specific operations.
pub mod advanced_indexing;

/// Apply functions to DataFrame rows, columns, or elements.
///
/// Enables custom transformations using user-defined functions.
pub mod apply;

/// Base DataFrame implementation with core operations.
///
/// The fundamental DataFrame structure with essential methods for data manipulation.
pub mod base;

/// Enhanced window operations including rolling, expanding, and exponentially weighted windows.
///
/// Provides sophisticated windowing functionality for time series analysis.
pub mod enhanced_window;

/// GroupBy operations for split-apply-combine workflows.
///
/// Efficient aggregation and transformation of grouped data.
pub mod groupby;

/// Window operations on grouped data.
///
/// Combines grouping with windowing for advanced analytics.
pub mod groupby_window;

/// Hierarchical groupby with multi-level aggregations.
///
/// Support for nested grouping operations and hierarchical aggregations.
pub mod hierarchical_agg_builder;
pub mod hierarchical_groupby;

/// Label-based and position-based indexing operations.
///
/// Implements loc, iloc, at, and iat indexers for flexible data access.
pub mod indexing;

/// JIT-compiled window operations for performance optimization.
///
/// Uses just-in-time compilation for faster window computations.
pub mod jit_window;

/// Join operations (inner, outer, left, right, cross).
///
/// Merge DataFrames using various join strategies.
pub mod join;

/// Multi-index cross-section operations.
///
/// Extract cross-sections from hierarchically indexed data.
pub mod multi_index_cross_section;

/// Multi-index result handling and utilities.
///
/// Tools for working with results from multi-index operations.
pub mod multi_index_results;

/// Optimized DataFrame operations using SIMD and vectorization.
///
/// High-performance implementations of common operations.
pub mod optimized;

/// pandas compatibility layer.
///
/// Provides pandas-like API and behavior for easy migration.
pub mod pandas_compat;

/// Plotting and visualization utilities.
///
/// Create charts and visualizations from DataFrame data.
pub mod plotting;

/// Query engine for SQL-like filtering.
///
/// Execute SQL-style queries on DataFrames.
pub mod query;

/// Serialization and deserialization support.
///
/// Convert DataFrames to/from various formats.
pub mod serialize;

/// Transform operations including melt, stack, unstack, and pivot.
///
/// Reshape DataFrames between long and wide formats.
pub mod transform;

/// DataFrame views for zero-copy slicing.
///
/// Efficient read-only views into DataFrame data.
pub mod view;

/// Window operations for rolling computations.
///
/// Base windowing functionality for time-based analysis.
pub mod window;

#[cfg(cuda_available)]
pub mod gpu;
#[cfg(cuda_available)]
pub mod gpu_window;

// Re-exports for convenience
pub use advanced_indexing::{
    AdvancedIndexingExt as SpecializedIndexingExt, CategoricalIndex, DatetimeIndex, Index,
    IndexOperations, IndexSetOps, IndexType, Interval, IntervalClosed, IntervalIndex, Period,
    PeriodFrequency, PeriodIndex,
};
pub use apply::{ApplyExt, Axis};
pub use base::DataFrame;
pub use enhanced_window::{
    DataFrameEWM, DataFrameEWMOps, DataFrameExpanding, DataFrameExpandingOps, DataFrameRolling,
    DataFrameRollingOps, DataFrameTimeRolling, DataFrameWindowExt as EnhancedDataFrameWindowExt,
};
pub use groupby::{AggFunc, ColumnAggBuilder, DataFrameGroupBy, GroupByExt, NamedAgg};
pub use groupby_window::{
    GroupWiseEWM, GroupWiseEWMOps, GroupWiseExpanding, GroupWiseExpandingOps, GroupWiseRolling,
    GroupWiseRollingOps, GroupWiseTimeRolling, GroupWiseTimeRollingOps, GroupWiseWindowExt,
};
pub use hierarchical_agg_builder::{utils as hierarchical_utils, HierarchicalAggBuilder};
pub use hierarchical_groupby::{
    GroupHierarchy, GroupNavigationContext, GroupNode, HierarchicalAgg,
    HierarchicalDataFrameGroupBy, HierarchicalGroupByExt, HierarchicalKey, HierarchyStatistics,
};
pub use indexing::{
    selectors, AdvancedIndexingExt, AlignmentStrategy, AtIndexer, ColumnSelector, IAtIndexer,
    ILocIndexer, IndexAligner, IndexRange, LocIndexer, MultiLevelIndex, RowSelector,
    SelectionBuilder,
};
pub use jit_window::{
    JitDataFrameEWM, JitDataFrameExpanding, JitDataFrameRolling, JitDataFrameRollingOps,
    JitDataFrameWindowExt, JitWindowContext, JitWindowStats, WindowFunctionKey, WindowOpType,
};
pub use join::{JoinExt, JoinType};
pub use multi_index_cross_section::{
    AggregationFunction, CrossSectionDataFrame as CrossSectionResult,
    MultiIndexDataFrame as CrossSectionDataFrame, MultiIndexGroupBy,
};
pub use multi_index_results::{
    utils as multi_index_utils, ColumnHierarchySummary, LevelSummary, MultiIndexColumn,
    MultiIndexDataFrame, MultiIndexDataFrameBuilder, MultiIndexMetadata, ToMultiIndex,
};
pub use pandas_compat::{
    Axis as PandasAxis, CorrelationMatrix, DescribeStats, PandasCompatExt, RankMethod, SeriesValue,
};
pub use plotting::{
    utils, ColorScheme, EnhancedPlotExt, FillStyle, GridStyle, InteractivePlot, PlotConfig,
    PlotFormat, PlotKind, PlotStyle, PlotTheme, StatPlotBuilder,
};
pub use query::{LiteralValue, QueryContext, QueryEngine, QueryExt};
pub use transform::{MeltOptions, StackOptions, TransformExt, UnstackOptions};
pub use window::DataFrameWindowExt;

// Optional feature re-exports
#[cfg(cuda_available)]
pub use gpu::DataFrameGpuExt;
#[cfg(cuda_available)]
pub use gpu_window::{
    GpuDataFrameRolling, GpuDataFrameWindowExt, GpuWindowContext, GpuWindowStats,
};

// Re-export from legacy module for backward compatibility
#[deprecated(
    since = "0.1.0",
    note = "Use new DataFrame implementation in crate::dataframe::base"
)]
pub use crate::dataframe::DataFrame as LegacyDataFrame;

#[deprecated(since = "0.1.0", note = "Use crate::dataframe::transform::MeltOptions")]
pub use crate::dataframe::transform::MeltOptions as LegacyMeltOptions;

#[deprecated(
    since = "0.1.0",
    note = "Use crate::dataframe::transform::StackOptions"
)]
pub use crate::dataframe::transform::StackOptions as LegacyStackOptions;

#[deprecated(
    since = "0.1.0",
    note = "Use crate::dataframe::transform::UnstackOptions"
)]
pub use crate::dataframe::transform::UnstackOptions as LegacyUnstackOptions;

#[deprecated(since = "0.1.0", note = "Use crate::dataframe::join::JoinType")]
pub use crate::dataframe::join::JoinType as LegacyJoinType;

#[deprecated(since = "0.1.0", note = "Use crate::dataframe::apply::Axis")]
pub use crate::dataframe::apply::Axis as LegacyAxis;

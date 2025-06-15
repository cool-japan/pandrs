// DataFrame implementations module
pub mod advanced_indexing;
pub mod apply;
pub mod base;
pub mod enhanced_window;
pub mod groupby;
pub mod groupby_window;
pub mod hierarchical_groupby;
pub mod indexing;
pub mod jit_window;
pub mod join;
pub mod multi_index_cross_section;
pub mod multi_index_results;
pub mod optimized;
pub mod plotting;
pub mod query;
pub mod serialize;
pub mod transform;
pub mod view;
pub mod window;

#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
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
pub use hierarchical_groupby::{
    utils as hierarchical_utils, GroupHierarchy, GroupNavigationContext, GroupNode,
    HierarchicalAgg, HierarchicalAggBuilder, HierarchicalDataFrameGroupBy, HierarchicalGroupByExt,
    HierarchicalKey, HierarchyStatistics,
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
pub use plotting::{
    utils, ColorScheme, EnhancedPlotExt, FillStyle, GridStyle, InteractivePlot, PlotConfig,
    PlotFormat, PlotKind, PlotStyle, PlotTheme, StatPlotBuilder,
};
pub use query::{LiteralValue, QueryContext, QueryEngine, QueryExt};
pub use transform::{MeltOptions, StackOptions, TransformExt, UnstackOptions};
pub use window::DataFrameWindowExt;

// Optional feature re-exports
#[cfg(feature = "cuda")]
pub use gpu::DataFrameGpuExt;
#[cfg(feature = "cuda")]
pub use gpu_window::{
    GpuDataFrameRolling, GpuDataFrameWindowExt, GpuWindowContext, GpuWindowStats,
};

// Re-export from legacy module for backward compatibility
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use new DataFrame implementation in crate::dataframe::base"
)]
pub use crate::dataframe::DataFrame as LegacyDataFrame;

#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use crate::dataframe::transform::MeltOptions"
)]
pub use crate::dataframe::transform::MeltOptions as LegacyMeltOptions;

#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use crate::dataframe::transform::StackOptions"
)]
pub use crate::dataframe::transform::StackOptions as LegacyStackOptions;

#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use crate::dataframe::transform::UnstackOptions"
)]
pub use crate::dataframe::transform::UnstackOptions as LegacyUnstackOptions;

#[deprecated(since = "0.1.0-alpha.2", note = "Use crate::dataframe::join::JoinType")]
pub use crate::dataframe::join::JoinType as LegacyJoinType;

#[deprecated(since = "0.1.0-alpha.2", note = "Use crate::dataframe::apply::Axis")]
pub use crate::dataframe::apply::Axis as LegacyAxis;

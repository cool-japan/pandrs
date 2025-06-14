// DataFrame implementations module
pub mod apply;
pub mod base;
pub mod groupby;
pub mod indexing;
pub mod join;
pub mod optimized;
pub mod plotting;
pub mod query;
pub mod serialize;
pub mod transform;
pub mod view;
pub mod window;
pub mod enhanced_window;
pub mod groupby_window;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports for convenience
pub use apply::{ApplyExt, Axis};
pub use base::DataFrame;
pub use groupby::{GroupByExt, DataFrameGroupBy, NamedAgg, ColumnAggBuilder, AggFunc};
pub use indexing::{
    AdvancedIndexingExt, ILocIndexer, LocIndexer, AtIndexer, IAtIndexer, 
    SelectionBuilder, IndexAligner, MultiLevelIndex, RowSelector, ColumnSelector, 
    IndexRange, AlignmentStrategy, selectors
};
pub use join::{JoinExt, JoinType};
pub use plotting::{
    EnhancedPlotExt, StatPlotBuilder, PlotConfig, PlotKind, PlotFormat, 
    ColorScheme, PlotStyle, FillStyle, GridStyle, InteractivePlot, PlotTheme, utils
};
pub use query::{QueryExt, QueryEngine, QueryContext, LiteralValue};
pub use transform::{MeltOptions, StackOptions, TransformExt, UnstackOptions};
pub use window::DataFrameWindowExt;
pub use enhanced_window::{
    DataFrameWindowExt as EnhancedDataFrameWindowExt, DataFrameRolling, DataFrameExpanding, 
    DataFrameEWM, DataFrameRollingOps, DataFrameExpandingOps, DataFrameEWMOps, DataFrameTimeRolling
};
pub use groupby_window::{
    GroupWiseWindowExt, GroupWiseRolling, GroupWiseExpanding, GroupWiseEWM, GroupWiseTimeRolling,
    GroupWiseRollingOps, GroupWiseExpandingOps, GroupWiseEWMOps, GroupWiseTimeRollingOps
};

// Optional feature re-exports
#[cfg(feature = "cuda")]
pub use gpu::DataFrameGpuExt;

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

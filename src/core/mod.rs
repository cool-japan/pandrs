// Core data structures and traits for PandRS
pub mod advanced_multi_index;
pub mod column;
pub mod column_ops;
pub mod data_value;
pub mod dataframe_traits;
pub mod error;
pub mod error_context;
pub mod index;
pub mod migration;
pub mod multi_index;

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

// Core data structures and traits for PandRS
pub mod column;
pub mod column_ops;
pub mod data_value;
pub mod dataframe_traits;
pub mod error;
pub mod error_context;
pub mod index;
pub mod multi_index;
pub mod advanced_multi_index;
pub mod migration;

// Re-exports for convenience
pub use column::{BitMask, Column, ColumnCast, ColumnTrait, ColumnType};
pub use column_ops::{
    ColumnOps, NumericColumnOps, StringColumnOps, DateTimeColumnOps, BooleanColumnOps, 
    CategoricalColumnOps, TypedColumn, ColumnStorage, ColumnFactory, DefaultColumnFactory,
    DuplicateKeep, PadSide, CastErrorBehavior,
    Int64Column, Int32Column, Float64Column, Float32Column, StringColumn, BooleanColumn,
    DateTimeColumn, DateColumn, TimeColumn, CategoricalColumn
};
pub use data_value::DataValue;
pub use dataframe_traits::{
    DataFrameOps, DataFrameAdvancedOps, GroupByOps, IndexingOps, StatisticalOps, DataFrameIO,
    Axis, JoinType, DropNaHow, FillMethod, AggFunc, RowIndexer, ColIndexer, LabelIndexer,
    BooleanMask, GroupKey, RollingWindow, ExpandingWindow, Resampler
};
pub use error::{Error, PandRSError, Result};
pub use error_context::{ErrorContext, ErrorContextBuilder, ErrorRecovery, ErrorRecoveryHelper, ErrorSeverity};
pub use migration::{
    Version, MigrationPlan, MigrationStep, MigrationExecutor, BackwardCompatibilityLayer,
    MigrationResult, MigrationRiskLevel, BackupManager, BackupStrategy
};
pub use index::{Index, IndexTrait};
pub use multi_index::MultiIndex;
pub use advanced_multi_index::{AdvancedMultiIndex, IndexValue, CrossSectionResult, SelectionCriteria};

// 特定の警告を無効化
#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(clippy::needless_return)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::let_and_return)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_lifetimes)]

pub mod column;
pub mod dataframe;
pub mod error;
pub mod groupby;
pub mod index;
pub mod io;
pub mod na;
pub mod optimized;
pub mod parallel;
pub mod pivot;
pub mod series;
pub mod stats;
pub mod temporal;
pub mod vis;
pub mod ml;

// Re-export commonly used types
pub use column::{Column, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
pub use dataframe::{DataFrame};
pub use error::PandRSError;
pub use groupby::GroupBy;
pub use index::{DataFrameIndex, Index, IndexTrait, MultiIndex, RangeIndex, StringIndex, StringMultiIndex};
pub use na::NA;
pub use optimized::{OptimizedDataFrame, LazyFrame, AggregateOp, JoinType};
pub use parallel::ParallelUtils;
pub use series::{Categorical, CategoricalOrder, NASeries, Series, StringCategorical};
pub use dataframe::{MeltOptions, StackOptions, UnstackOptions};
pub use stats::{DescriptiveStats, TTestResult, LinearRegressionResult};
pub use vis::{OutputFormat, PlotConfig, PlotType};
pub use ml::pipeline::Pipeline;
pub use ml::preprocessing::{StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, Binner, Imputer, ImputeStrategy, FeatureSelector};
pub use ml::models::{SupervisedModel, LinearRegression, LogisticRegression};
pub use ml::models::model_selection::{train_test_split, cross_val_score, k_fold_split, GridSearchCV};
pub use ml::models::model_persistence::ModelPersistence;
pub use ml::dimension_reduction::{PCA, TSNE, TSNEInit};

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

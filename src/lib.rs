// Disable specific warnings
#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(clippy::needless_return)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::let_and_return)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_lifetimes)]

// Crates that use macros
#[macro_use]
#[cfg(feature = "excel")]
extern crate simple_excel_writer;

// Core module with fundamental data structures and traits
pub mod core;

// Compute module for computation functionality
pub mod compute;

// Storage module for data storage engines
pub mod storage;

// Legacy modules (for backward compatibility)
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
pub mod large;
pub mod streaming;

// Internal utilities and compatibility layers
#[doc(hidden)]
pub mod utils;

#[cfg(feature = "wasm")]
pub mod web;

#[cfg(feature = "cuda")]
pub mod gpu;

#[cfg(feature = "distributed")]
pub mod distributed;

// Re-export core types (new organization)
pub use core::error::{Error, Result};
pub use core::column::{Column as CoreColumn, ColumnType as CoreColumnType, ColumnTrait, ColumnCast, BitMask as CoreBitMask};
pub use core::data_value::{DataValue, DataValueExt, DisplayExt};
pub use core::index::{Index as CoreIndex, IndexTrait};
pub use core::multi_index::MultiIndex as CoreMultiIndex;

// Re-export legacy types (for backward compatibility)
pub use column::{Column, ColumnType, Int64Column, Float64Column, StringColumn, BooleanColumn};
pub use dataframe::{DataFrame};
pub use error::PandRSError;
pub use groupby::GroupBy;
pub use index::{DataFrameIndex, Index, IndexTrait as LegacyIndexTrait, MultiIndex, RangeIndex, StringIndex, StringMultiIndex};
pub use na::NA;
pub use optimized::{OptimizedDataFrame, LazyFrame, AggregateOp, JoinType};
pub use parallel::ParallelUtils;
pub use series::{Categorical, CategoricalOrder, NASeries, Series, StringCategorical};
pub use dataframe::{MeltOptions, StackOptions, UnstackOptions};
pub use stats::{DescriptiveStats, TTestResult, LinearRegressionResult};
pub use vis::{OutputFormat, PlotConfig, PlotType};
// Machine learning features (new organization)
pub use ml::pipeline::{Pipeline, PipelineStage, PipelineTransformer};
pub use ml::preprocessing::{StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures, Binner, Imputer, ImputeStrategy, FeatureSelector};
pub use ml::metrics::regression::{mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error, explained_variance_score};
pub use ml::metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
pub use ml::models::{ModelMetrics, ModelEvaluator, SupervisedModel, UnsupervisedModel, CrossValidation, train_test_split};
pub use ml::models::linear::{LinearRegression, LogisticRegression};
pub use ml::dimension::{PCA, TSNE, TSNEInit};
pub use ml::clustering::{KMeans, AgglomerativeClustering, DBSCAN, Linkage, DistanceMetric};
pub use ml::anomaly::{IsolationForest, LocalOutlierFactor, OneClassSVM};

// Large data processing
pub use large::{DiskConfig, ChunkedDataFrame, DiskBasedDataFrame, DiskBasedOptimizedDataFrame};

// Streaming data processing
pub use streaming::{StreamConfig, DataStream, StreamRecord, StreamAggregator, StreamProcessor,
                   StreamConnector, RealTimeAnalytics, AggregationType, MetricType};

// WebAssembly and web visualization (when enabled)
#[cfg(feature = "wasm")]
pub use web::{WebVisualization, WebVisualizationConfig, ColorTheme, VisualizationType};

// Computation-related exports (new organization)
pub use compute::parallel::ParallelUtils as ComputeParallelUtils;
pub use compute::lazy::LazyFrame as ComputeLazyFrame;

// Storage-related exports (new organization)
pub use storage::string_pool::StringPool as StorageStringPool;
pub use storage::column_store::ColumnStore;
pub use storage::disk::DiskStorage;
pub use storage::memory_mapped::MemoryMappedFile;

// GPU acceleration (when enabled)
#[cfg(feature = "cuda")]
pub use compute::gpu::{GpuConfig, init_gpu, GpuDeviceStatus, GpuBenchmark};

// Legacy GPU exports (for backward compatibility)
#[cfg(feature = "cuda")]
pub use gpu::{GpuConfig, GpuManager, init_gpu, get_gpu_manager, GpuDeviceStatus};
#[cfg(feature = "cuda")]
pub use dataframe::gpu::DataFrameGpuExt;
#[cfg(feature = "cuda")]
pub use temporal::gpu::SeriesTimeGpuExt;
#[cfg(feature = "cuda")]
pub use gpu::benchmark::{GpuBenchmark, BenchmarkOperation, BenchmarkResult, BenchmarkSummary};

// Distributed processing (when enabled)
#[cfg(feature = "distributed")]
pub use distributed::core::{DistributedConfig, DistributedDataFrame, ToDistributed};
#[cfg(feature = "distributed")]
pub use distributed::execution::{ExecutionEngine, ExecutionContext, ExecutionPlan};
#[cfg(feature = "distributed")]
pub use distributed::expr::{Expr as DistributedExpr, UdfDefinition, ExprDataType};

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
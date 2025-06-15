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
pub mod jupyter;
pub mod large;
pub mod ml;
pub mod na;
pub mod optimized;
pub mod parallel;
pub mod pivot;
pub mod series;
pub mod stats;
pub mod streaming;
pub mod temporal;
pub mod vis;

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
pub use core::column::{
    BitMask as CoreBitMask, Column as CoreColumn, ColumnCast, ColumnTrait,
    ColumnType as CoreColumnType,
};
pub use core::data_value::{DataValue, DataValueExt, DisplayExt};
pub use core::error::{Error, Result};
pub use core::index::{Index as CoreIndex, IndexTrait};
pub use core::multi_index::MultiIndex as CoreMultiIndex;

// Re-export legacy types (for backward compatibility)
pub use column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
pub use dataframe::DataFrame;
pub use dataframe::{MeltOptions, StackOptions, UnstackOptions};
pub use error::PandRSError;
pub use groupby::GroupBy;
pub use index::{
    DataFrameIndex, Index, IndexTrait as LegacyIndexTrait, MultiIndex, RangeIndex, StringIndex,
    StringMultiIndex,
};
pub use na::NA;
pub use optimized::{AggregateOp, JoinType, LazyFrame, OptimizedDataFrame};
pub use parallel::ParallelUtils;
pub use series::{Categorical, CategoricalOrder, NASeries, Series, StringCategorical};
pub use stats::{DescriptiveStats, LinearRegressionResult, TTestResult};
pub use vis::{OutputFormat, PlotConfig, PlotType};

// Jupyter integration exports
pub use jupyter::{
    get_jupyter_config, init_jupyter, jupyter_dark_mode, jupyter_light_mode, set_jupyter_config,
    JupyterColorScheme, JupyterConfig, JupyterDisplay, JupyterMagics, TableStyle, TableWidth,
};
// Machine learning features (new organization)
pub use ml::anomaly::{IsolationForest, LocalOutlierFactor, OneClassSVM};
pub use ml::clustering::{AgglomerativeClustering, DistanceMetric, KMeans, Linkage, DBSCAN};
pub use ml::dimension::{TSNEInit, PCA, TSNE};
pub use ml::metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};
pub use ml::metrics::regression::{
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
    root_mean_squared_error,
};
pub use ml::models::linear::{LinearRegression, LogisticRegression};
pub use ml::models::{
    train_test_split, CrossValidation, ModelEvaluator, ModelMetrics, SupervisedModel,
    UnsupervisedModel,
};
pub use ml::pipeline::{Pipeline, PipelineStage, PipelineTransformer};
pub use ml::preprocessing::{
    Binner, FeatureSelector, ImputeStrategy, Imputer, MinMaxScaler, OneHotEncoder,
    PolynomialFeatures, StandardScaler,
};

// Large data processing
pub use large::{ChunkedDataFrame, DiskBasedDataFrame, DiskBasedOptimizedDataFrame, DiskConfig};

// Streaming data processing
pub use streaming::{
    AggregationType, DataStream, MetricType, RealTimeAnalytics, StreamAggregator, StreamConfig,
    StreamConnector, StreamProcessor, StreamRecord,
};

// WebAssembly and web visualization (when enabled)
#[cfg(feature = "wasm")]
pub use web::{ColorTheme, VisualizationType, WebVisualization, WebVisualizationConfig};

// Computation-related exports (new organization)
pub use compute::lazy::LazyFrame as ComputeLazyFrame;
pub use compute::parallel::ParallelUtils as ComputeParallelUtils;

// Storage-related exports (new organization)
pub use storage::column_store::ColumnStore;
pub use storage::disk::DiskStorage;
pub use storage::memory_mapped::MemoryMappedFile;
pub use storage::string_pool::StringPool as StorageStringPool;

// GPU acceleration (when enabled)
#[cfg(feature = "cuda")]
pub use compute::gpu::{init_gpu, GpuBenchmark, GpuConfig, GpuDeviceStatus};

// Legacy GPU exports (for backward compatibility)
#[cfg(feature = "cuda")]
pub use dataframe::gpu::DataFrameGpuExt;
#[cfg(feature = "cuda")]
pub use gpu::benchmark::{BenchmarkOperation, BenchmarkResult, BenchmarkSummary, GpuBenchmark};
#[cfg(feature = "cuda")]
pub use gpu::{get_gpu_manager, init_gpu, GpuConfig, GpuDeviceStatus, GpuManager};
#[cfg(feature = "cuda")]
pub use temporal::gpu::SeriesTimeGpuExt;

// Distributed processing (when enabled)
#[cfg(feature = "distributed")]
pub use distributed::core::{DistributedConfig, DistributedDataFrame, ToDistributed};
#[cfg(feature = "distributed")]
pub use distributed::execution::{ExecutionContext, ExecutionEngine, ExecutionPlan};
// #[cfg(feature = "distributed")]
// pub use distributed::expr::{Expr as DistributedExpr, ExprDataType, UdfDefinition}; // Temporarily disabled

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

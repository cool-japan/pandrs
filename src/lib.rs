//! # PandRS
//!
//! A high-performance DataFrame library for Rust, providing pandas-like API with advanced features
//! including SIMD optimization, parallel processing, and distributed computing capabilities.
//!
//! ## Overview
//!
//! PandRS brings the power and familiarity of pandas to the Rust ecosystem. Built with
//! performance, safety, and ease of use in mind, it provides:
//!
//! - **Type-safe operations** leveraging Rust's ownership system
//! - **High-performance computing** through SIMD vectorization and parallel processing
//! - **Memory-efficient design** with columnar storage and string pooling
//! - **Comprehensive functionality** matching pandas' core features
//! - **Seamless interoperability** with Python, Arrow, and various data formats
//!
//! ## Quick Start
//!
//! ```rust
//! use pandrs::{DataFrame, Series};
//!
//! // Create a DataFrame
//! let mut df = DataFrame::new();
//! df.add_column("name".to_string(),
//!     Series::new(vec!["Alice", "Bob", "Carol"], Some("name".to_string())).unwrap()).unwrap();
//! df.add_column("age".to_string(),
//!     Series::new(vec![30i64, 25, 35], Some("age".to_string())).unwrap()).unwrap();
//!
//! // Basic operations
//! let nrows = df.row_count();
//! let ncols = df.column_count();
//! ```
//!
//! ## Feature Flags
//!
//! PandRS supports various feature flags for optional functionality:
//!
//! - **Core features:**
//!   - `stable`: Recommended stable feature set
//!   - `optimized`: Performance optimizations and SIMD
//!   - `backward_compat`: Backward compatibility support
//!
//! - **Data formats:**
//!   - `parquet`: Apache Parquet file support
//!   - `excel`: Excel file support (read/write)
//!   - `sql`: Database connectivity (PostgreSQL, MySQL, SQLite)
//!
//! - **Advanced features:**
//!   - `distributed`: Distributed computing with DataFusion
//!   - `visualization`: Plotting capabilities
//!   - `streaming`: Real-time data processing
//!   - `serving`: Model serving and deployment
//!
//! - **Experimental:**
//!   - `cuda`: GPU acceleration (requires CUDA toolkit)
//!   - `wasm`: WebAssembly compilation support
//!   - `jit`: Just-in-time compilation
//!
//! ## Core Data Structures
//!
//! - [`Series`]: One-dimensional labeled array capable of holding any data type
//! - [`DataFrame`]: Two-dimensional, size-mutable, heterogeneous tabular data structure
//! - [`MultiIndex`]: Hierarchical indexing for advanced data organization
//! - [`Categorical`]: Memory-efficient representation for string data with limited cardinality
//!
//! ## Modules
//!
//! - [`dataframe`]: DataFrame operations and manipulation
//! - [`series`]: Series operations and manipulation
//! - [`stats`]: Statistical functions and analysis
//! - [`ml`]: Machine learning algorithms and utilities
//! - [`io`]: Input/output operations for various file formats
//! - [`streaming`]: Real-time streaming data processing
//! - [`time_series`]: Time series analysis and forecasting
//! - [`graph`]: Graph analytics and algorithms
//!
//! ## Version
//!
//! Current version: 0.1.0

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

// Arrow integration module for ecosystem compatibility
#[cfg(feature = "distributed")]
pub mod arrow_integration;

// Data connectors for databases and cloud storage
pub mod connectors;

// Configuration management for secure settings and credentials
pub mod config;

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
pub mod time_series;
pub mod vis;

// Graph analytics module
pub mod graph;

// Data versioning and lineage tracking module
pub mod versioning;

// Audit logging module
pub mod audit;

// Multi-tenancy support module
pub mod multitenancy;

// Enterprise authentication module (JWT, OAuth, API Keys)
pub mod auth;

// Real-time analytics dashboard module
pub mod analytics;

// Internal utilities and compatibility layers
#[doc(hidden)]
pub mod utils;

#[cfg(feature = "wasm")]
pub mod web;

#[cfg(cuda_available)]
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

// Configuration management exports
pub use config::credentials::{
    CredentialBuilder, CredentialMetadata, CredentialStore, CredentialStoreConfig, CredentialType,
    EncryptedCredential,
};
pub use config::{
    AccessControlConfig, AuditConfig, AwsConfig, AzureConfig, CachingConfig, CloudConfig,
    ConnectionPoolConfig, DatabaseConfig, EncryptionConfig, GcpConfig, GlobalCloudConfig,
    JitConfig, LogRotationConfig, LoggingConfig, MemoryConfig, PandRSConfig, PerformanceConfig,
    SecurityConfig, SslConfig, ThreadingConfig, TimeoutConfig,
};

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
pub use ml::models::ensemble::{
    GradientBoostingClassifier, GradientBoostingConfig, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestConfig, RandomForestRegressor,
};
pub use ml::models::linear::{LinearRegression, LogisticRegression};
pub use ml::models::neural::{
    Activation, LossFunction, MLPClassifier, MLPConfig, MLPConfigBuilder, MLPRegressor,
};
pub use ml::models::tree::{
    DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor, SplitCriterion,
};
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
    AggregationType,
    // Backpressure handling
    BackpressureBuffer,
    BackpressureChannel,
    BackpressureConfig,
    BackpressureConfigBuilder,
    BackpressureStats,
    BackpressureStrategy,
    DataStream,
    FlowController,
    MetricType,
    // Windowed aggregations
    MultiColumnAggregator,
    RealTimeAnalytics,
    StreamAggregator,
    StreamConfig,
    StreamConnector,
    StreamProcessor,
    StreamRecord,
    TimeWindow,
    WindowAggregation,
    WindowConfig,
    WindowConfigBuilder,
    WindowResult,
    WindowType,
    WindowedAggregator,
};

// Time series analysis and forecasting
pub use time_series::{
    ArimaForecaster,
    AugmentedDickeyFullerTest,
    // Advanced forecasting
    AutoArima,
    AutocorrelationAnalysis,
    ChangePointDetection,
    DateTimeIndex,
    DecompositionMethod,
    DecompositionResult,
    Differencing,
    ExponentialSmoothingForecaster,
    FeatureSet,
    ForecastMetrics,
    ForecastResult,
    Forecaster,
    Frequency,
    KwiatkowskiPhillipsSchmidtShinTest,
    LinearTrendForecaster,
    MissingValueStrategy,
    ModelSelectionCriterion,
    ModelSelectionResult,
    Normalization,
    OutlierDetection,
    SarimaForecaster,
    SeasonalDecomposition,
    SeasonalTest,
    SeasonalityAnalysis,
    SimpleMovingAverageForecaster,
    StationarityTest,
    StatisticalFeatures,
    TimePoint,
    TimeSeries,
    TimeSeriesBuilder,
    TimeSeriesFeatureExtractor,
    TimeSeriesPreprocessor,
    TimeSeriesStats,
    TrendAnalysis,
    WhiteNoiseTest,
    WindowFeatures,
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
#[cfg(cuda_available)]
pub use compute::gpu::{init_gpu, GpuBenchmark, GpuConfig, GpuDeviceStatus};

// Legacy GPU exports (for backward compatibility)
#[cfg(cuda_available)]
pub use dataframe::gpu::DataFrameGpuExt;
#[cfg(cuda_available)]
pub use gpu::benchmark::{BenchmarkOperation, BenchmarkResult, BenchmarkSummary};
#[cfg(cuda_available)]
pub use gpu::{get_gpu_manager, GpuManager};
#[cfg(cuda_available)]
pub use temporal::gpu::SeriesTimeGpuExt;

// Distributed processing (when enabled)
#[cfg(feature = "distributed")]
pub use distributed::core::{DistributedConfig, DistributedDataFrame, ToDistributed};
#[cfg(feature = "distributed")]
pub use distributed::execution::{ExecutionContext, ExecutionEngine, ExecutionPlan};
// #[cfg(feature = "distributed")]
// pub use distributed::expr::{Expr as DistributedExpr, ExprDataType, UdfDefinition}; // Temporarily disabled

// Graph analytics exports
pub use graph::{
    bellman_ford_default,
    betweenness_centrality,
    // Traversal algorithms
    bfs,
    closeness_centrality,
    // Component analysis
    connected_components,
    // Centrality metrics
    degree_centrality,
    dfs,
    // Path algorithms
    dijkstra,
    dijkstra_default,
    eigenvector_centrality_default,
    floyd_warshall_default,
    from_adjacency_matrix,
    // Graph construction
    from_edge_dataframe,
    has_cycle,
    hits_default,
    is_connected,
    label_propagation,
    louvain_default,
    modularity,
    pagerank,
    pagerank_default,
    shortest_path_bfs,
    strongly_connected_components,
    to_adjacency_matrix,
    to_edge_dataframe,
    topological_sort,
    AllPairsShortestPaths,
    BfsResult,
    ComponentResult,
    DfsResult,
    Edge,
    EdgeId,
    // Core types
    Graph,
    GraphBuilder,
    GraphError,
    GraphType,
    Node,
    NodeId,
    ShortestPathResult,
};

// Data versioning and lineage tracking exports
pub use versioning::{
    // DataFrame integration
    DataFrameVersioning,
    // Core types
    DataSchema,
    DataVersion,
    // Tracker types
    LineageConfig,
    LineageTracker,
    Operation,
    OperationType,
    SharedLineageTracker,
    TrackerStats,
    VersionDiff,
    VersionId,
    VersionedTransform,
    VersioningError,
};

// Audit logging exports
pub use audit::{
    // Global logger
    global_logger,
    init_global_logger,
    log_global,
    // Core types
    AuditConfig as AuditLogConfig,
    AuditConfigBuilder as AuditLogConfigBuilder,
    AuditEntry,
    AuditLogger,
    AuditStats,
    EventCategory,
    LogContext,
    LogDestination,
    LogLevel,
    SharedAuditLogger,
};

// Multi-tenancy exports
pub use multitenancy::{
    create_shared_manager, DatasetId, DatasetMetadata, IsolationContext, Permission, ResourceQuota,
    SharedTenantManager, TenantAuditEntry, TenantConfig, TenantId, TenantManager, TenantOperation,
    TenantUsage,
};

// Enterprise authentication exports
pub use auth::{
    create_shared_auth_manager,
    decode_jwt,
    encode_jwt,
    get_token_expiration,
    is_token_expired,
    verify_jwt,
    // API Key management
    ApiKeyInfo,
    ApiKeyManager,
    ApiKeyStats,
    AuthEvent,
    AuthEventType,
    // Core types
    AuthManager,
    AuthMethod,
    AuthResult,
    AuthorizationRequest,
    IntrospectionResponse,
    // JWT
    JwtConfig,
    OAuthClient,
    OAuthClientInfo,
    // OAuth 2.0
    OAuthConfig,
    OAuthGrantType,
    RefreshToken,
    ScopedApiKey,
    // Session management
    Session,
    SessionContext,
    SessionStore,
    SharedAuthManager,
    TokenClaims,
    TokenRequest,
    TokenResponse,
    UserInfo,
};

// Real-time analytics dashboard exports
pub use analytics::{
    create_default_rules,
    // Dashboard functions
    global_dashboard,
    init_global_dashboard,
    record_global,
    time_global,
    // Alerting
    ActiveAlert,
    AlertHandler,
    AlertManager,
    AlertMetric,
    AlertRule,
    AlertSeverity,
    // Core types
    Dashboard,
    DashboardConfig,
    DashboardSnapshot,
    LoggingAlertHandler,
    // Metrics
    Metric,
    MetricStats,
    MetricType as AnalyticsMetricType,
    MetricValue,
    MetricsCollector,
    OperationCategory,
    OperationRecord,
    RateCalculator,
    ResourceSnapshot,
    ScopedTimer,
    ThresholdOperator,
    TimeResolution,
};

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

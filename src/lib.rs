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
//!     Series::new(vec!["Alice", "Bob", "Carol"], Some("name".to_string())).expect("operation should succeed")).expect("operation should succeed");
//! df.add_column("age".to_string(),
//!     Series::new(vec![30i64, 25, 35], Some("age".to_string())).expect("operation should succeed")).expect("operation should succeed");
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

/// Core module with fundamental data structures and traits.
///
/// This module provides the foundational building blocks for PandRS, including:
/// - Column types and operations
/// - Index implementations
/// - Data value representations
/// - Error handling
pub mod core;

/// Compute module for computation functionality.
///
/// This module provides computational capabilities including:
/// - Lazy evaluation and query optimization
/// - Parallel processing utilities
/// - GPU acceleration (when feature enabled)
pub mod compute;

/// Arrow integration module for ecosystem compatibility.
///
/// Provides seamless integration with Apache Arrow for:
/// - Zero-copy data exchange
/// - Interoperability with Arrow-based tools
/// - Efficient distributed computing
#[cfg(feature = "distributed")]
pub mod arrow_integration;

/// Data connectors for databases and cloud storage.
///
/// Connect to external data sources:
/// - Relational databases (PostgreSQL, MySQL, SQLite)
/// - Cloud storage (AWS S3, Azure Blob, Google Cloud Storage)
pub mod connectors;

/// Configuration management for secure settings and credentials.
///
/// This module handles configuration for various aspects of PandRS:
/// - Database connection settings
/// - Cloud storage credentials
/// - Security and encryption settings
/// - Performance tuning parameters
pub mod config;

/// Storage module for data storage engines.
///
/// This module provides various storage backends and optimization techniques:
/// - Column-oriented storage
/// - Memory-mapped files
/// - Disk-based storage
/// - String pooling for memory efficiency
pub mod storage;

// Legacy modules (for backward compatibility)

/// Column implementations and column-level operations.
///
/// Provides strongly-typed column types (Int64, Float64, String, Boolean)
/// with support for vectorized operations and memory-efficient storage.
pub mod column;

/// DataFrame data structure and operations.
///
/// The core tabular data structure with support for:
/// - Row/column selection and manipulation
/// - Join, merge, and concatenation operations
/// - GroupBy aggregations
/// - Window functions
/// - Advanced indexing
pub mod dataframe;

/// Error types and error handling utilities.
///
/// Defines the error types used throughout PandRS for robust error propagation.
pub mod error;

/// GroupBy operations for split-apply-combine workflows.
///
/// Provides efficient grouped aggregations and transformations on DataFrames.
pub mod groupby;

/// Index types for DataFrame and Series labeling.
///
/// Supports various index types including:
/// - RangeIndex for integer-based indexing
/// - StringIndex for string-labeled indexing
/// - MultiIndex for hierarchical indexing
pub mod index;

/// Input/output operations for reading and writing data.
///
/// Supports multiple file formats:
/// - CSV (comma-separated values)
/// - JSON (JavaScript Object Notation)
/// - Parquet (columnar storage format)
/// - Excel (Microsoft Excel files)
/// - SQL databases (PostgreSQL, MySQL, SQLite)
///
/// # Examples
///
/// ```rust,no_run
/// use pandrs::{DataFrame, io};
/// use pandrs::io::json::JsonOrient;
///
/// // Read CSV file
/// let df = io::read_csv("data.csv", true).expect("Failed to read CSV");
///
/// // Write to JSON
/// io::write_json(&df, "output.json", JsonOrient::Records).expect("Failed to write JSON");
/// ```
pub mod io;

/// Jupyter notebook integration and display formatting.
///
/// Provides rich HTML formatting for DataFrames in Jupyter environments.
pub mod jupyter;

/// Large dataset processing with chunking and disk-based operations.
///
/// Enables processing of datasets larger than available RAM through:
/// - Chunked processing
/// - Disk-based storage
/// - Streaming operations
pub mod large;

/// Machine learning algorithms and utilities.
///
/// Comprehensive ML toolkit including:
/// - Supervised models (classification, regression)
/// - Unsupervised models (clustering, dimensionality reduction)
/// - Model evaluation and metrics
/// - Feature preprocessing
/// - ML pipelines
pub mod ml;

/// NA (Not Available) value handling and missing data operations.
///
/// Provides utilities for working with missing data across DataFrames and Series.
pub mod na;

/// Optimized implementations using SIMD and vectorization.
///
/// High-performance alternatives to standard operations leveraging:
/// - SIMD instructions
/// - Vectorized operations
/// - Cache-friendly algorithms
pub mod optimized;

/// Parallel processing utilities and thread pool management.
///
/// Enables parallel execution of operations across multiple cores.
pub mod parallel;

/// Pivot table operations for data reshaping.
///
/// Transform data from long to wide format and vice versa.
pub mod pivot;

/// Series data structure for one-dimensional labeled data.
///
/// The fundamental one-dimensional data structure with support for:
/// - Type-safe operations
/// - String and datetime accessors
/// - Categorical data
/// - Window operations
/// - NA handling
pub mod series;

/// Statistical functions and analysis tools.
///
/// Comprehensive statistical capabilities including:
/// - Descriptive statistics
/// - Correlation and covariance
/// - Hypothesis testing
/// - Regression analysis
pub mod stats;

/// Real-time streaming data processing.
///
/// Process continuous data streams with:
/// - Windowed aggregations
/// - Backpressure handling
/// - Stream connectors
pub mod streaming;

/// Temporal operations for time-based data.
///
/// Date/time functionality including:
/// - Date ranges
/// - Resampling
/// - Frequency conversions
/// - Time-based windowing
pub mod temporal;

/// Time series analysis and forecasting.
///
/// Advanced time series capabilities:
/// - ARIMA, SARIMA forecasting
/// - Seasonal decomposition
/// - Stationarity tests
/// - Feature extraction
/// - Trend analysis
///
/// # Examples
///
/// ```rust
/// use pandrs::time_series::{
///     TimeSeriesBuilder, SimpleMovingAverageForecaster, Forecaster,
/// };
/// use chrono::{Utc, TimeZone};
///
/// // Build a time series with the builder
/// let mut builder = TimeSeriesBuilder::new();
/// for i in 0..5 {
///     let ts_val = Utc.timestamp_opt(1640995200 + i * 86400, 0).single().expect("valid ts");
///     builder = builder.add_point(ts_val, (i + 1) as f64);
/// }
/// let ts = builder.build().expect("Failed to build time series");
///
/// // Create and fit a moving average forecaster
/// let mut forecaster = SimpleMovingAverageForecaster::new(3);
/// forecaster.fit(&ts).expect("Failed to fit");
///
/// // Forecast 2 steps ahead
/// let result = forecaster.forecast(2, 0.95).expect("Failed to forecast");
/// assert_eq!(result.forecast.len(), 2);
/// ```
pub mod time_series;

/// Visualization and plotting utilities.
///
/// Create charts and plots from DataFrames and Series.
pub mod vis;

/// Graph analytics module.
///
/// Comprehensive graph algorithms and analytics:
/// - Graph construction and manipulation
/// - Path algorithms (Dijkstra, BFS, DFS)
/// - Centrality metrics (PageRank, betweenness, closeness)
/// - Community detection (Louvain, label propagation)
/// - Component analysis
///
/// # Examples
///
/// ```rust
/// use pandrs::graph::{GraphBuilder, GraphType, pagerank};
///
/// let builder = GraphBuilder::<String, f64>::new(GraphType::Directed);
/// let builder = builder.add_node("A", "Node A".to_string());
/// let builder = builder.add_node("B", "Node B".to_string());
/// let builder = builder.add_node("C", "Node C".to_string());
/// let builder = builder.add_edge("A", "B", Some(1.0));
/// let builder = builder.add_edge("B", "C", Some(1.0));
/// let graph = builder.build();
///
/// let scores = pagerank(&graph, 0.85, 100, 1e-6);
/// assert!(!scores.is_empty());
/// ```
pub mod graph;

/// Data versioning and lineage tracking module.
///
/// Track data transformations and maintain version history:
/// - DataFrame versioning
/// - Operation lineage tracking
/// - Diff computation
/// - Rollback capabilities
pub mod versioning;

/// Schema evolution and migration tools.
///
/// Comprehensive tools for defining, versioning, and migrating DataFrame schemas:
/// - Schema definition with typed columns and constraints
/// - Semantic versioning of schemas
/// - Migration plans with ordered change sets
/// - Schema registry for version management and path-finding
/// - DataFrame migration and validation
/// - Schema inference from existing DataFrames
/// - Compatibility checking between schemas
/// - JSON/YAML serialization of schemas and migrations
pub mod schema_evolution;

/// Audit logging module.
///
/// Comprehensive audit trail for data operations:
/// - Event logging with metadata
/// - Configurable log destinations
/// - Query and analysis of audit logs
/// - Compliance support
pub mod audit;

/// Multi-tenancy support module.
///
/// Isolate and manage data across multiple tenants:
/// - Tenant isolation
/// - Resource quotas
/// - Access control
/// - Usage tracking
pub mod multitenancy;

/// Enterprise authentication module (JWT, OAuth, API Keys).
///
/// Comprehensive authentication and authorization:
/// - JWT token management
/// - OAuth 2.0 integration
/// - API key authentication
/// - Session management
/// - Role-based access control
///
/// # Examples
///
/// ```rust
/// use pandrs::auth::{AuthManager, JwtConfig};
///
/// let jwt_config = JwtConfig {
///     secret_key: b"your-secret-key".to_vec(),
///     issuer: "pandrs".to_string(),
///     audience: "pandrs-users".to_string(),
///     expiration_secs: 3600,
///     validate_exp: true,
///     validate_iss: true,
///     validate_aud: true,
///     leeway_secs: 30,
/// };
///
/// let auth = AuthManager::new(jwt_config);
/// ```
pub mod auth;

/// Real-time analytics dashboard module.
///
/// Monitor and analyze system metrics in real-time:
/// - Metrics collection and aggregation
/// - Dashboard visualization
/// - Alert management
/// - Performance monitoring
pub mod analytics;

/// Plugin system for custom data sources, transforms, and sinks.
///
/// Extend PandRS with custom data sources, data sinks, transforms,
/// aggregators, and validators. Includes built-in plugins for CSV and JSON.
pub mod plugins;

/// SciRS2 integration module for scientific computing capabilities.
///
/// Provides bridge between PandRS DataFrames/Series and the SciRS2 scientific
/// computing ecosystem. Enable with the `scirs2` feature flag.
pub mod scirs2_integration;

pub use scirs2_integration::dataframe_ext::SciRS2Ext;
#[cfg(feature = "scirs2")]
pub use scirs2_integration::{SciRS2LinAlg, SciRS2Stats};

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

// SVG/HTML visualization exports (pure Rust, always available)
pub use vis::svg::{
    BarChart as SvgBarChart, BarOrientation as SvgBarOrientation, Color as SvgColor,
    ColorScheme as SvgColorScheme, DrawStyle, HeatMap as SvgHeatMap, LegendPosition,
    LineChart as SvgLineChart, LineSeries, Margins as SvgMargins, MarkerShape, PathBuilder,
    PieChart as SvgPieChart, ScatterPlot as SvgScatterPlot, SvgCanvas, SvgChartConfig,
    SvgHistogram, SvgPlotType, SvgVisualize, Transform as SvgTransform,
};

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
pub use large::{external_sort, merge_sorted_chunks};
pub use large::{hash_join_out_of_core, OutOfCoreJoinType};
pub use large::{AggOp as OutOfCoreAggOp, OutOfCoreConfig, OutOfCoreReader, OutOfCoreWriter};
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

// Arrow Flight RPC support for distributed data transfer (requires "flight" feature)
#[cfg(feature = "flight")]
pub use distributed::flight::{PandRsFlightClient, PandRsFlightServer};

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

// Schema evolution exports
pub use schema_evolution::{
    BreakingChange, ColumnSchema, CompatibilityReport, DataFrameSchema, DefaultValue, Migration,
    MigrationBuilder, SchemaChange, SchemaConstraint, SchemaDataType, SchemaFormat, SchemaMigrator,
    SchemaRegistry, SchemaVersion, ValidationError, ValidationErrorType, ValidationReport,
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

/// The current version of the PandRS library.
///
/// This version string is automatically populated from the Cargo.toml package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

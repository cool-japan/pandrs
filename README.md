# PandRS

[![Crate](https://img.shields.io/crates/v/pandrs.svg)](https://crates.io/crates/pandrs)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://docs.rs/pandrs/badge.svg)](https://docs.rs/pandrs)
![Tests](https://img.shields.io/badge/tests-2817%20passing-brightgreen.svg)

A high-performance DataFrame library for Rust, providing pandas-like API with advanced features including SIMD optimization, parallel processing, and distributed computing capabilities.

> **Version 0.4.1**: Code-honesty release — real CPU fallbacks for non-CUDA GPU paths, real Python GPU bindings (PCA, k-means, linear regression, correlation), real Shapiro-Wilk W coefficients (Royston AS R94), fixed Column::from_any data-loss bug, path-traversal security enforcement, DataFusion 53 optimizer rule translation, Series<T> idiomatic traits (Index, IntoIterator, FromIterator, Extend, Display), lint hygiene (removed orphaned backward_compat code). **2817 tests passing** (`cargo nextest run --features all-safe`); see [CHANGELOG.md](CHANGELOG.md) for the full list.

## Code Quality Highlights

**Comprehensive Testing**: 2817 tests passing via `cargo nextest run --features all-safe`, plus 110 doc tests (`cargo test --doc --features all-safe`)
**Active Development**: Ongoing improvements to error handling and code quality across a large, actively-tested Rust codebase (run `tokei .` for current file/line counts — they change every release)
**Production-Ready Error Handling**: Established error handling patterns with descriptive messages

## Overview

PandRS is a comprehensive data manipulation library that brings the power and familiarity of pandas to the Rust ecosystem. Built with performance, safety, and ease of use in mind, it provides:

- **Type-safe operations** leveraging Rust's ownership system
- **High-performance computing** through SIMD vectorization and parallel processing
- **Memory-efficient design** with columnar storage and string pooling
- **Comprehensive functionality** matching pandas' core features
- **Seamless interoperability** with Python, Arrow, and various data formats

## Quick Start

```rust
use pandrs::{DataFrame, Series};
use pandrs::dataframe::{AggFunc, GroupByExt, NamedAgg};

// Create a DataFrame.
let mut df = DataFrame::new();
df.add_column(
    "name".to_string(),
    Series::new(
        vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
        Some("name".to_string()),
    )?,
)?;
df.add_column(
    "age".to_string(),
    Series::new(vec![30i64, 25, 35], Some("age".to_string()))?,
)?;
df.add_column(
    "department".to_string(),
    Series::new(
        vec![
            "Engineering".to_string(),
            "Engineering".to_string(),
            "Sales".to_string(),
        ],
        Some("department".to_string()),
    )?,
)?;
df.add_column(
    "salary".to_string(),
    Series::new(vec![75_000i64, 65_000, 85_000], Some("salary".to_string()))?,
)?;

// Column-level numeric summary.
let mean_salary = df.mean("salary")?;

// GroupBy + named aggregations, via the `GroupByExt` trait imported above.
let grouped = df.groupby(&["department"])?.agg(vec![
    NamedAgg::new("salary".to_string(), AggFunc::Mean, "salary_mean".to_string()),
    NamedAgg::new("salary".to_string(), AggFunc::Sum, "salary_sum".to_string()),
    NamedAgg::new("age".to_string(), AggFunc::Max, "age_max".to_string()),
])?;
```

## Core Features

### Data Structures

- **Series**: One-dimensional labeled array capable of holding any data type
- **DataFrame**: Two-dimensional, size-mutable, heterogeneous tabular data structure
- **MultiIndex**: Hierarchical indexing for advanced data organization
- **Categorical**: Memory-efficient representation for string data with limited cardinality

### Data Types

- **Columnar storage** (`OptimizedDataFrame`, recommended for performance): four primitive column types — `Int64`, `Float64`, `String` (with automatic string pooling), `Boolean`.
- **Generic `Series<T>`** (traditional `DataFrame`): any `T: Clone + Debug + 'static`, so `i32`/`u32`/`u64`/`f32`/chrono datetimes/etc. all work through this path, without the columnar/string-pool optimizations.
- Categorical: Efficient storage for repeated string values
- Missing Values: First-class `NA` support (`NASeries` / `Option<T>`); see the note on `Series` vs `NASeries` in [docs/API_GUIDE.md](docs/API_GUIDE.md)

### Operations

#### Data Manipulation
- Column addition, removal, and renaming
- Row and column selection with boolean indexing
- Sorting by single or multiple columns
- Duplicate detection and removal
- Data type conversion and casting

#### Aggregation & Grouping
- GroupBy operations with multiple aggregation functions
- Window functions (rolling, expanding, exponentially weighted)
- Pivot tables and cross-tabulation
- Custom aggregation functions

#### Joining & Merging
- Inner, left, right, and outer joins
- Merge on single or multiple keys
- Concat operations with axis control
- Append with automatic index alignment

#### Time Series
- DateTime indexing and slicing
- Resampling and frequency conversion
- Time zone handling and conversion
- Date range generation
- Business day calculations

### Performance Optimizations

#### SIMD Vectorization
- `x86_64`: SSE2 baseline, with an `#[target_feature(enable = "avx2")]`-gated AVX2 kernel selected at runtime via `is_x86_feature_detected!`. Scalar (still auto-vectorizable by the compiler) is the fallback on every other target, including `aarch64` — this crate ships no NEON or AVX-512 path.
- `OptimizedDataFrame` exposes four opt-in reduction methods on this fast path — `sum_simd`/`mean_simd`/`min_simd`/`max_simd` — but **only `sum` actually has a SIMD kernel**; `mean`/`min`/`max` there are scalar by design even though they're reachable through a `*_simd`-named method (their pandas `skipna` tie-break rules don't line up with a vector fold; see the module doc for why).
- **Implemented, public, real kernels — not yet called from a main DataFrame path:** element-wise column ops (add/sub/mul/div/abs/sqrt/compare, via the `SIMDFloat64Ops`/`SIMDInt64Ops` column traits), extended statistics (variance/std/covariance/correlation/skewness/kurtosis/dot product/L2 norm/weighted mean, `optimized::jit::simd_stats`), and SIMD string operations (`optimized::jit::simd_string`). Normal DataFrame/`Series` arithmetic, `.var()`/`.std()`, and the `.str` accessor do not call these yet — use them directly, or see `benches/`.
- See [`src/optimized/jit/simd.rs`](src/optimized/jit/simd.rs) and [`src/optimized/jit/simd_stats.rs`](src/optimized/jit/simd_stats.rs) module docs for the exact per-operation coverage and wiring status.

#### Parallel Processing
- Multi-threaded execution for large datasets
- Configurable thread pool sizing
- Parallel aggregations and transformations
- Load-balanced work distribution

#### Memory Efficiency
- Columnar storage format
- String interning with global string pool
- Memory-mapped file support
- Lazy evaluation for chain operations

### I/O Capabilities

#### File Formats
- **CSV**: Fast parallel CSV reader/writer
- **Parquet**: Apache Parquet with compression support
- **JSON**: Both records and columnar JSON formats
- **Excel**: XLSX (OOXML) read/write with multi-sheet support, Pure Rust (`excel` feature) — legacy binary `.xls` (BIFF) is not supported
- **Arrow**: Arrow interoperability, data-copying (not zero-copy) (`arrow_integration` module, requires the `distributed` feature)

#### Cloud Storage (`cloud-storage` feature)
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- MinIO (S3-compatible)

### Security Features

Enterprise-grade security features for data protection and access control:

#### Authentication & Authorization
- **JWT (JSON Web Tokens)**: Stateless authentication with token validation
- **OAuth 2.0**: Industry-standard authorization framework
- **API Key Management**: Secure API key generation and validation
- **Session Management**: User session tracking and lifecycle management

#### Access Control
- **Role-Based Access Control (RBAC)**: Fine-grained permission management
- **Multi-tenancy Support**: Isolated data access per tenant
- **Resource-level Permissions**: Control access to specific datasets and operations

#### Security Monitoring
- **Audit Logging**: Comprehensive tracking of data access and modifications
- **Security Events**: Real-time monitoring of authentication and authorization events
- **Compliance Support**: Features designed to meet security compliance requirements

See `examples/security_jwt_oauth_example.rs` and `examples/security_rbac_example.rs` for implementation details.

### Real-Time Analytics

Built-in analytics engine for monitoring and performance tracking:

#### Metrics Collection
- **Counters**: Track cumulative values and event counts
- **Gauges**: Monitor current values and resource levels
- **Histograms**: Measure distribution of values over time
- **Timers**: Track operation durations and performance

#### Operation Tracking
- **DataFrame Operations**: Monitor query execution and data transformations
- **Resource Monitoring**: Track memory usage, CPU utilization, and I/O operations
- **Performance Profiling**: Identify bottlenecks and optimization opportunities

#### Alert Management
- **Threshold-based Alerts**: Trigger notifications when metrics exceed limits
- **Custom Alert Rules**: Define complex alerting conditions
- **Alert History**: Track and analyze past alerts

#### Visualization
- **Real-time Dashboards**: Monitor system health and performance metrics
- **Metric Aggregation**: Combine and analyze metrics across dimensions
- **Export Capabilities**: Export metrics to external monitoring systems

See `examples/analytics_dashboard_example.rs` for comprehensive usage examples.

### Machine Learning

Advanced machine learning capabilities integrated with DataFrame operations:

#### Supervised Learning
- **Decision Trees**: Classification and regression with interpretable models
- **Random Forests**: Ensemble methods for improved accuracy
- **Gradient Boosting**: High-performance boosting algorithms
- **Neural Networks**: Deep learning with configurable architectures

#### Time Series Forecasting
- **ARIMA Models**: AutoRegressive Integrated Moving Average
- **Exponential Smoothing**: Trend and seasonality modeling
- **Feature Engineering**: Automatic lag features and date components

#### Model Pipeline
- **Feature Preprocessing**: Scaling, normalization, and encoding
- **Model Training**: Unified API for training various algorithms
- **Cross-validation**: K-fold and time series cross-validation
- **Hyperparameter Tuning**: Grid search and random search optimization

See `examples/ml_neural_network_example.rs`, `examples/ml_decision_tree_example.rs`,
`examples/ml_random_forest_example.rs`, `examples/ml_gradient_boosting_example.rs`,
and `examples/time_series_forecasting_example.rs` for detailed examples.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pandrs = "0.4.1"
```

### Feature Flags

Enable additional functionality with feature flags:

```toml
[dependencies]
pandrs = { version = "0.4.1", features = ["optimized"] }
```

Available features:
- **Core features:**
  - `optimized`: Performance optimizations and SIMD
  - `backward_compat`: Backward compatibility support
- **Data formats:**
  - `parquet`: Parquet file support
  - `excel`: Excel (XLSX) file support, Pure Rust
  - `cloud-storage`: S3 / GCS / Azure Blob / MinIO backends
- **Advanced features:**
  - `distributed`: Distributed computing with DataFusion
  - `flight`: Arrow Flight RPC for distributed data transfer (implies `distributed`)
  - `visualization`: Plotting capabilities
  - `streaming`: Real-time data processing
  - `serving`: Model serving and deployment
  - `resilience`: Retry / circuit-breaker patterns
  - `scirs2`: SciRS2 scientific computing integration
- **Experimental:**
  - `cuda`: GPU acceleration (requires the CUDA toolkit)
  - `wasm`: WebAssembly compilation support
  - `jit`: Just-in-time-style custom aggregations (see [docs/JIT_COMPILATION.md](docs/JIT_COMPILATION.md) for what this does and does not do today)
- **Bundles** (combine several of the above for convenience — see `Cargo.toml` for the exact members): `test-core`, `test-safe`, `all-safe` (excludes CUDA/WASM/distributed), `stable`

### Minimum Supported Rust Version (MSRV)

- **1.88** for the default feature set and `distributed`.
- **1.89** for `cloud-storage`, `all-safe`, and `stable` (these pull in a
  higher-floor transitive dependency — `crc-fast`, via `object_store`'s AWS
  backend).

The workspace `rust-version` in `Cargo.toml` is pinned at the 1.88 floor;
cargo enforces the higher per-dependency requirement automatically once a
1.89-requiring feature is enabled, so building `all-safe` on an
older-than-1.89 toolchain fails with a clear MSRV error at dependency
resolution rather than a confusing compile error.

## Performance

There is currently no reproducible, dated benchmark comparison against
pandas/Polars published in this README — a previous table here was
unverifiable (no commit/dataset/version pinned, and some rows had no
matching benchmark at all) and has been removed rather than kept as
unsubstantiated marketing numbers. To measure PandRS on your own workload,
run `cargo bench` (see [BENCHMARKING.md](BENCHMARKING.md)). Note that
`benches/pandas_comparison_benchmark.rs` benchmarks a Rust re-implementation
of pandas-equivalent logic, not actual pandas; `benches/pandas_benchmark.py`
and `benches/polars_benchmark.py` are separate standalone Python scripts
that do run real pandas/Polars — run them independently and compare numbers
yourself if you need a cross-library figure.

## Documentation

- [API Documentation](https://docs.rs/pandrs)
- [User Guide](docs/USER_GUIDE.md)
- [Examples](https://github.com/cool-japan/pandrs/tree/master/examples)
- [Migration from Pandas](docs/PANDAS_MIGRATION.md)

## Examples

The `examples/` directory contains comprehensive examples demonstrating all major features:

### Data Manipulation & Analysis
- **Basic Operations**: `transform_example.rs`, `pivot_example.rs`
- **GroupBy (DataFrame + `GroupByExt`, matching the Quick Start above)**: `groupby_named_agg_demo.rs`, `hierarchical_groupby_example.rs`
- **GroupBy (older `Series`-level `GroupBy` struct, a different/legacy API)**: `groupby_example.rs`
- **Time Series**: `time_series_example.rs`, `time_series_forecasting_example.rs`, `datetime_accessor_example.rs`
- **Window Operations**: `window_operations_example.rs`, `comprehensive_window_example.rs`, `dataframe_window_example.rs`
- **Multi-Index**: `multi_index_example.rs`, `hierarchical_groupby_example.rs`, `nested_group_operations_example.rs`
- **Categorical Data**: `categorical_example.rs`, `categorical_na_example.rs`

### Machine Learning
- **Neural Networks**: `ml_neural_network_example.rs`
- **Decision Trees**: `ml_decision_tree_example.rs`
- **Random Forests**: `ml_random_forest_example.rs`
- **Gradient Boosting**: `ml_gradient_boosting_example.rs`
- **ML Pipelines**: `optimized_ml_pipeline_example.rs`, `optimized_ml_feature_engineering_example.rs`
- **Specialized ML**: `optimized_ml_clustering_example.rs`, `optimized_ml_anomaly_detection_example.rs`, `optimized_ml_dimension_reduction_example.rs`

### Security & Authentication
- **JWT & OAuth 2.0**: `security_jwt_oauth_example.rs`
- **Role-Based Access Control**: `security_rbac_example.rs`

### Real-Time Analytics
- **Analytics Dashboard**: `analytics_dashboard_example.rs`

### I/O & Data Formats
- **CSV**: Examples integrated into basic operations
- **Parquet**: `parquet_example.rs`, `parquet_advanced_example.rs`, `parquet_advanced_features_example.rs`
- **Excel**: `excel_multisheet_example.rs`, `excel_advanced_features_example.rs`

### Performance & Optimization
- **SIMD & Parallel**: `parallel_example.rs`, `optimized_dataframe_example.rs`, `optimized_large_dataset_example.rs`
- **GPU Acceleration**: `gpu_dataframe_example.rs`, `gpu_ml_example.rs`, `gpu_benchmark_example.rs`
- **Distributed Computing**: `distributed_example.rs`, `distributed_window_example.rs`, `distributed_fault_tolerance_example.rs`
- **JIT Compilation**: `jit_parallel_example.rs`, `jit_window_operations_example.rs`
- **Streaming**: `streaming_example.rs`

### Visualization
- **Plotters Integration**: `visualization_plotters_example.rs`, `plotters_visualization_example.rs`, `enhanced_visualization_example.rs`

### Basic Data Analysis

```rust
use pandrs::{DataFrame, Series};
use pandrs::dataframe::{AggFunc, GroupByExt, NamedAgg};

// Build a DataFrame inline. (Use `pandrs::io::read_csv(path, has_header)?`
// for CSV ingestion; `DataFrame::read_csv` is reserved for future API work.)
let mut df = DataFrame::new();
df.add_column(
    "city".to_string(),
    Series::new(
        vec!["Tallinn".to_string(), "Tallinn".to_string(), "Tartu".to_string()],
        Some("city".to_string()),
    )?,
)?;
df.add_column(
    "occupation".to_string(),
    Series::new(
        vec!["Engineer".to_string(), "Engineer".to_string(), "Analyst".to_string()],
        Some("occupation".to_string()),
    )?,
)?;
df.add_column(
    "age".to_string(),
    Series::new(vec![21i64, 34, 40], Some("age".to_string()))?,
)?;
df.add_column(
    "income".to_string(),
    Series::new(vec![55_000i64, 72_000, 81_000], Some("income".to_string()))?,
)?;

// Grouped aggregation with explicit named aggregations.
let result = df.groupby(&["city", "occupation"])?.agg(vec![
    NamedAgg::new("income".to_string(), AggFunc::Mean, "income_mean".to_string()),
    NamedAgg::new("income".to_string(), AggFunc::Median, "income_median".to_string()),
    NamedAgg::new("income".to_string(), AggFunc::Std, "income_std".to_string()),
    NamedAgg::new("age".to_string(), AggFunc::Mean, "age_mean".to_string()),
])?;
```

> Note: the `Time Series Analysis` and `Machine Learning Pipeline` snippets
> below are illustrative of the target pandas-like API and still reference
> helpers (`fillna`, `resample`, `ewm`, `get_dummies`, `apply_columns`,
> `DataFrame::read_parquet`) that are not yet wired up on the stable
> `DataFrame`. They are being aligned with the real surface in a follow-up;
> see `examples/` for snippets that build and run today.

### Time Series Analysis

```rust,ignore
// NOTE: This snippet shows the target API. Some helpers (resample, ewm,
// DataFrame::read_csv on the base DataFrame) are not yet wired up on the
// stable DataFrame. See examples/time_series_example.rs for runnable code.
use pandrs::prelude::*;
use chrono::{Duration, Utc};

let mut df = DataFrame::read_csv("timeseries.csv", CsvReadOptions::default())?;
df.set_index("timestamp")?;

// Resample to daily frequency
let daily = df.resample("D")?.mean()?;

// Calculate rolling statistics
let rolling_stats = daily
    .rolling(RollingOptions {
        window: 7,
        min_periods: Some(1),
        center: false,
    })?
    .agg(HashMap::from([
        ("value".to_string(), vec!["mean", "std"]),
    ]))?;

// Exponentially weighted moving average
let ewm = daily.ewm(EwmOptions {
    span: Some(10.0),
    ..Default::default()
})?;
```

### Machine Learning Pipeline

```rust,ignore
// NOTE: This snippet shows the target API. Some helpers (read_parquet on the
// base DataFrame, fillna, get_dummies, apply_columns) are not yet wired up
// on the stable DataFrame. See examples/optimized_ml_pipeline_example.rs for
// runnable code.
use pandrs::prelude::*;

// Load and preprocess data
let df = DataFrame::read_parquet("features.parquet")?;

// Handle missing values
let df_filled = df.fillna(FillNaOptions::Forward)?;

// Encode categorical variables
let df_encoded = df_filled.get_dummies(vec!["category1", "category2"], None)?;

// Normalize numerical features
let features = vec!["feature1", "feature2", "feature3"];
let df_normalized = df_encoded.apply_columns(&features, |series| {
    let mean = series.mean()?;
    let std = series.std(1)?;
    series.sub_scalar(mean)?.div_scalar(std)
})?;

// Split features and target
let X = df_normalized.drop(vec!["target"])?;
let y = df_normalized.column("target")?;
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cool-japan/pandrs
cd pandrs

# Install development dependencies
cargo install cargo-nextest cargo-criterion

# Run tests
cargo nextest run

# Run benchmarks
cargo criterion

# Check code quality (clippy::correctness/suspicious/perf are enforced;
# style/complexity are intentionally allowed — see CONTRIBUTING.md for the
# exact policy and the handful of narrowly-scoped, justified allows)
cargo clippy --features all-safe -- -D warnings
cargo fmt -- --check
```

## Sponsorship

PandRS is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find PandRS useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or <http://www.apache.org/licenses/LICENSE-2.0>).

## Acknowledgments

PandRS is inspired by the excellent pandas library and incorporates ideas from:
- [Pandas](https://pandas.pydata.org/) - API design and functionality
- [Polars](https://www.pola.rs/) - Performance optimizations
- [Apache Arrow](https://arrow.apache.org/) - Columnar format
- [DataFusion](https://datafusion.apache.org/) - Query engine

## Support

- [Issue Tracker](https://github.com/cool-japan/pandrs/issues)
- [Discussions](https://github.com/cool-japan/pandrs/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pandrs)

---

PandRS is a COOLJAPAN project, bringing high-performance data analysis to the Rust ecosystem.
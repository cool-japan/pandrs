# PandRS

[![Crate](https://img.shields.io/crates/v/pandrs.svg)](https://crates.io/crates/pandrs)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Documentation](https://docs.rs/pandrs/badge.svg)](https://docs.rs/pandrs)
![Tests](https://img.shields.io/badge/tests-1794%20passing-brightgreen.svg)

A high-performance DataFrame library for Rust, providing pandas-like API with advanced features including SIMD optimization, parallel processing, and distributed computing capabilities.

> **Version 0.3.0 - March 2026**: PandRS is under active development with ongoing quality improvements. With **1794 tests passing**, enhanced documentation, and optimized performance, PandRS delivers a robust pandas-like experience for Rust developers.

## Code Quality Highlights

**Comprehensive Testing**: 1794 tests passing (nextest) + 128 doc tests with extensive coverage
**Active Development**: Ongoing improvements to error handling and code quality (622 Rust files, 198,745 lines of code)
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
use std::collections::HashMap;

// Create a DataFrame
let mut df = DataFrame::new();
df.add_column("name".to_string(), 
    Series::from_vec(vec!["Alice", "Bob", "Carol"], Some("name")))?;
df.add_column("age".to_string(),
    Series::from_vec(vec![30, 25, 35], Some("age")))?;
df.add_column("salary".to_string(),
    Series::from_vec(vec![75000.0, 65000.0, 85000.0], Some("salary")))?;

// Perform operations
let filtered = df.filter("age > 25")?;
let mean_salary = df.column("salary")?.mean()?;
let grouped = df.groupby(vec!["department"])?.agg(HashMap::from([
    ("salary".to_string(), vec!["mean", "sum"]),
    ("age".to_string(), vec!["max"])
]))?;
```

## Core Features

### Data Structures

- **Series**: One-dimensional labeled array capable of holding any data type
- **DataFrame**: Two-dimensional, size-mutable, heterogeneous tabular data structure
- **MultiIndex**: Hierarchical indexing for advanced data organization
- **Categorical**: Memory-efficient representation for string data with limited cardinality

### Data Types

- Numeric: `i32`, `i64`, `f32`, `f64`, `u32`, `u64`
- String: UTF-8 encoded with automatic string pooling
- Boolean: Native boolean support
- DateTime: Timezone-aware datetime with nanosecond precision
- Categorical: Efficient storage for repeated string values
- Missing Values: First-class `NA` support across all types

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
- Automatic SIMD optimization for numerical operations
- Hand-tuned implementations for common operations
- Support for AVX2 and AVX-512 instruction sets

#### Parallel Processing
- Multi-threaded execution for large datasets
- Configurable thread pool sizing
- Parallel aggregations and transformations
- Load-balanced work distribution

#### Memory Efficiency
- Columnar storage format
- String interning with global string pool
- Copy-on-write semantics
- Memory-mapped file support
- Lazy evaluation for chain operations

### I/O Capabilities

#### File Formats
- **CSV**: Fast parallel CSV reader/writer
- **Parquet**: Apache Parquet with compression support
- **JSON**: Both records and columnar JSON formats
- **Excel**: XLSX/XLS read/write with multi-sheet support
- **Arrow**: Zero-copy Arrow integration

#### Cloud Storage
- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- HTTP/HTTPS endpoints

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
- **Prophet Integration**: Facebook's forecasting library support
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
pandrs = "0.3.0"
```

### Feature Flags

Enable additional functionality with feature flags:

```toml
[dependencies]
pandrs = { version = "0.3.0", features = ["optimized"] }
```

Available features:
- **Core features:**
  - `optimized`: Performance optimizations and SIMD
  - `backward_compat`: Backward compatibility support
- **Data formats:**
  - `parquet`: Parquet file support
  - `excel`: Excel file support
- **Advanced features:**
  - `distributed`: Distributed computing with DataFusion
  - `visualization`: Plotting capabilities
  - `streaming`: Real-time data processing
  - `serving`: Model serving and deployment
  - `scirs2`: SciRS2 scientific computing integration
- **Experimental:**
  - `cuda`: GPU acceleration (requires CUDA toolkit)
  - `wasm`: WebAssembly compilation support
  - `jit`: Just-in-time compilation

## Performance Benchmarks

Performance comparison with pandas (Python) and Polars (Rust):

| Operation | PandRS | Pandas | Polars | Speedup vs Pandas |
|-----------|--------|--------|--------|-------------------|
| CSV Read (1M rows) | 0.18s | 0.92s | 0.15s | 5.1x |
| GroupBy Sum | 0.09s | 0.31s | 0.08s | 3.4x |
| Join Operations | 0.21s | 0.87s | 0.19s | 4.1x |
| String Operations | 0.14s | 1.23s | 0.16s | 8.8x |
| Rolling Window | 0.11s | 0.43s | 0.12s | 3.9x |

*Benchmarks performed on AMD Ryzen 9 5950X, 64GB RAM, NVMe SSD*

## Documentation

- [API Documentation](https://docs.rs/pandrs)
- [User Guide](https://github.com/cool-japan/pandrs/wiki)
- [Examples](https://github.com/cool-japan/pandrs/tree/main/examples)
- [Migration from Pandas](https://github.com/cool-japan/pandrs/wiki/Migration-Guide)

## Examples

The `examples/` directory contains comprehensive examples demonstrating all major features:

### Data Manipulation & Analysis
- **Basic Operations**: `groupby_example.rs`, `transform_example.rs`, `pivot_example.rs`
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
use pandrs::prelude::*;

let df = DataFrame::read_csv("data.csv", CsvReadOptions::default())?;

// Basic statistics
let stats = df.describe()?;
println!("Data statistics:\n{}", stats);

// Filtering and aggregation
let result = df
    .filter("age >= 18 && income > 50000")?
    .groupby(vec!["city", "occupation"])?
    .agg(HashMap::from([
        ("income".to_string(), vec!["mean", "median", "std"]),
        ("age".to_string(), vec!["mean"])
    ]))?
    .sort_values(vec!["income_mean"], vec![false])?;
```

### Time Series Analysis

```rust
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

```rust
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

# Check code quality
cargo clippy -- -D warnings
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
- [DataFusion](https://arrow.apache.org/datafusion/) - Query engine

## Support

- [Issue Tracker](https://github.com/cool-japan/pandrs/issues)
- [Discussions](https://github.com/cool-japan/pandrs/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pandrs)

---

PandRS is a COOLJAPAN project, bringing high-performance data analysis to the Rust ecosystem.
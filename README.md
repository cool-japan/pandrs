# PandRS

[![Rust CI](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml/badge.svg)](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Crate](https://img.shields.io/crates/v/pandrs.svg)](https://crates.io/crates/pandrs)

A high-performance DataFrame library for data analysis implemented in Rust. Inspired by Python's pandas library, PandRS combines fast data processing with type safety and distributed computing capabilities.

> **📢 Final Alpha Release (0.1.0-alpha.5)**: This is the final alpha release before beta. While feature-complete and extensively tested, please review the [Production Readiness Assessment](PRODUCTION_READINESS.md) for production deployment considerations.

## Quick Start

```rust
use pandrs::{DataFrame, Series};
use std::collections::HashMap;

// Create DataFrame
let mut df = DataFrame::new();
df.add_column("name".to_string(), 
    Series::from_vec(vec!["Alice", "Bob", "Carol"], Some("name")))?;
df.add_column("age".to_string(),
    Series::from_vec(vec![30, 25, 35], Some("age")))?;

// Rename columns
let mut rename_map = HashMap::new();
rename_map.insert("name".to_string(), "employee_name".to_string());
df.rename_columns(&rename_map)?;

// Basic operations
let filtered = df.filter("age > 25")?;
let grouped = df.groupby("department")?.sum(&["salary"])?;
```

## Key Features

- **🔥 High Performance**: Column-oriented storage with up to 5x faster aggregations
- **🧠 Memory Efficient**: String pool optimization reducing memory usage by up to 89%
- **⚡ Multi-core & GPU**: Parallel processing + CUDA acceleration (up to 20x speedup)
- **🌐 Distributed Computing**: DataFusion-powered distributed processing for large datasets
- **🐍 Python Integration**: Full PyO3 bindings with pandas interoperability
- **🔒 Type Safety**: Rust's ownership system ensuring memory safety and thread safety
- **📊 Rich Analytics**: Statistical functions, ML metrics, and categorical data analysis
- **💾 Flexible I/O**: Parquet, CSV, JSON, SQL, and Excel support

## Core Functionality

### DataFrame Operations
- Series (1-dimensional) and DataFrame (2-dimensional) data structures
- Missing value (NA) handling
- Multi-level indexes (hierarchical indexes)
- Grouping and aggregation operations
- Advanced filtering and sorting
- Join operations (inner, left, right, outer)

### Data Processing
- String accessor (.str) with 25+ methods for text processing
- DateTime accessor (.dt) with timezone support
- Advanced window operations (rolling, expanding, EWM)
- Statistical analysis and hypothesis testing
- Time series analysis and forecasting
- Categorical data types with memory optimization

### I/O Capabilities
- CSV, JSON, Parquet file formats
- Excel read/write with multi-sheet support
- Database connectivity (PostgreSQL, SQLite, MySQL)
- Cloud storage integration (AWS S3, Google Cloud, Azure)
- Streaming data support

### Performance Features
- Just-In-Time (JIT) compilation for mathematical operations
- SIMD vectorization support
- GPU acceleration (CUDA)
- Parallel processing with Rayon
- Distributed processing with DataFusion
- Zero-copy operations where possible

## Installation

Add to your Cargo.toml:

```toml
[dependencies]
pandrs = "0.1.0-alpha.5"
```

For GPU acceleration (requires CUDA toolkit):

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.5", features = ["cuda"] }
```

For distributed processing:

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.5", features = ["distributed"] }
```

Multiple features can be combined:

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.5", features = ["cuda", "distributed", "python"] }
```

## Examples

### Basic DataFrame Operations

```rust
use pandrs::{DataFrame, Series};

// Create and manipulate data
let ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;

let mut df = DataFrame::new();
df.add_column("age".to_string(), ages)?;
df.add_column("height".to_string(), heights)?;

// Statistical operations
let mean_age = df.column("age")?.mean()?;
let correlation = df.corr(&["age", "height"])?;

// Save and load
df.to_csv("data.csv")?;
let loaded_df = DataFrame::from_csv("data.csv", true)?;
```

### String and DateTime Processing

```rust
// String operations
let text_data = Series::new(vec!["Hello", "World", "PandRS"], Some("text"))?;
let uppercase = text_data.str().upper()?;
let contains_h = text_data.str().contains("H")?;

// DateTime operations
let dates = Series::new(vec!["2023-01-01", "2023-06-15", "2023-12-31"], Some("dates"))?;
let parsed_dates = dates.dt().parse("%Y-%m-%d")?;
let years = parsed_dates.dt().year()?;
let months = parsed_dates.dt().month()?;
```

### Window Operations

```rust
// Rolling operations
let values = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("values"))?;
let rolling_mean = values.rolling(3).mean()?;
let expanding_sum = values.expanding().sum()?;
let ewm_mean = values.ewm(span=2).mean()?;
```

### Statistical Analysis

```rust
use pandrs::stats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let stats_summary = stats::describe(&data)?;
println!("Mean: {}, Std: {}", stats_summary.mean, stats_summary.std);

// Hypothesis testing
let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let t_test_result = stats::ttest(&sample1, &sample2, 0.05, true)?;
println!("T-statistic: {}, P-value: {}", t_test_result.statistic, t_test_result.pvalue);
```

### Distributed Processing

```rust
use pandrs::distributed::{DistributedConfig, ToDistributed};

// Convert to distributed DataFrame
let config = DistributedConfig::new()
    .with_executor("datafusion")
    .with_concurrency(4);

let dist_df = df.to_distributed(config)?;

// Distributed operations
let result = dist_df
    .filter("value > 1000")?
    .groupby(&["category"])?
    .aggregate(&["value"], &["mean"])?;

// Execute and collect results
let final_df = result.execute()?.collect_to_local()?;
```

### Python Bindings

```python
import pandrs as pr
import pandas as pd

# Create DataFrame
df = pr.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Pandas interoperability
pd_df = df.to_pandas()  # Convert to pandas
pr_df = pr.DataFrame.from_pandas(pd_df)  # Convert from pandas

# GPU acceleration
pr.gpu.init_gpu()
gpu_df = df.gpu_accelerate()
corr_matrix = gpu_df.gpu_corr(['A', 'C'])
```

## Performance Benchmarks

| Operation | Traditional | PandRS | Speedup |
|-----------|-------------|--------|---------|
| DataFrame Creation | 198ms | 149ms | 1.33x |
| Filtering | 596ms | 162ms | 3.68x |
| Group Aggregation | 544ms | 108ms | 5.05x |
| Matrix Multiplication (GPU) | 233ms | 12ms | 20.2x |

## Testing

Run tests with:

```bash
# Core library tests
cargo test --lib

# With most features (excludes CUDA/WASM)
cargo test --features "test-safe"

# All features (requires CUDA toolkit)
cargo test --all-features
```

## Documentation

- [API Guide](docs/API_GUIDE.md)
- [Ecosystem Integration](docs/ECOSYSTEM_INTEGRATION_GUIDE.md)
- [Performance Optimization](docs/PERFORMANCE_PLAN.md)
- [Production Readiness Assessment](PRODUCTION_READINESS.md) 📋
- [GPU Acceleration Guide](docs/GPU_ACCELERATION_GUIDE.md)
- [JIT Compilation Guide](docs/JIT_COMPILATION.md)

## Contributing

We welcome contributions! Please see our contributing guidelines for development setup and code standards.

## License

PandRS is dual-licensed under either:

* MIT License (LICENSE-MIT or http://opensource.org/licenses/MIT)
* Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
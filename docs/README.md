# PandRS Documentation

Welcome to the comprehensive documentation for PandRS, a high-performance DataFrame library for Rust. This documentation covers everything from basic usage to advanced performance optimization.

## Getting Started

If you're new to PandRS, start with these essential guides:

### 📚 Core Documentation
- **[API Guide](API_GUIDE.md)** - Complete API reference and usage patterns
  - DataFrame and Series fundamentals
  - Data types and column management  
  - I/O operations and error handling
  - Best practices and examples

### 🌐 Integration & Ecosystem
- **[Ecosystem Integration Guide](ECOSYSTEM_INTEGRATION_GUIDE.md)** - Connect with external systems
  - Cloud storage integration (AWS S3, Google Cloud, Azure, MinIO)
  - Apache Arrow interoperability
  - Python bindings (`py_bindings/`) with pandas-compatible surface
  - *(There is no SQL/database feature — see the guide for what's actually there.)*

## Performance & Optimization

PandRS offers multiple performance optimization layers. Choose the features that match your use case:

### ⚡ Performance Features
- **[Performance Optimization Guide](PERFORMANCE_PLAN.md)** - Complete performance optimization strategies
  - Benchmarking tools and methodology
  - Memory optimization techniques
  - I/O performance best practices
  - Real-world performance examples

- **[JIT Compilation Guide](JIT_COMPILATION.md)** - Just-In-Time compilation for mathematical operations
  - Numba-like functionality for Rust
  - Custom aggregation functions
  - SIMD vectorization support
  - Parallel execution patterns

- **[GPU Acceleration Guide](GPU_ACCELERATION_GUIDE.md)** - CUDA-based GPU acceleration
  - Window operations optimization
  - Memory management strategies
  - Real-time data processing
  - Performance benchmarking

### 📊 Benchmarking
- **[Benchmarking Guide](../BENCHMARKING.md)** - Comprehensive benchmarking infrastructure
  - Performance regression detection
  - Realistic data generation
  - Throughput measurements
  - Memory profiling

## Documentation Index

### User Guides
| Guide | Description | Best For |
|-------|-------------|----------|
| [API Guide](API_GUIDE.md) | Core DataFrame/Series APIs | New users, reference |
| [Ecosystem Integration](ECOSYSTEM_INTEGRATION_GUIDE.md) | External system connectivity | Data engineers |
| [Performance Plan](PERFORMANCE_PLAN.md) | Optimization strategies | Performance-critical apps |
| [JIT Compilation](JIT_COMPILATION.md) | Runtime optimization | Custom aggregations |
| [GPU Acceleration](GPU_ACCELERATION_GUIDE.md) | CUDA acceleration | Large-scale analytics |

### Reference Documentation
| Resource | Description | Access |
|----------|-------------|---------|
| **API Reference** | Complete Rust API docs | `cargo doc --open` |
| **Examples** | Working code examples | [examples/](../examples/) directory |
| **Benchmarks** | Performance measurement | [benches/](../benches/) directory |
| **Tests** | Unit and integration tests | `cargo test` |

### Quick Reference

#### Installation
```toml
[dependencies]
# Basic usage
pandrs = "0.4.2"

# With performance features
pandrs = { version = "0.4.2", features = ["cuda", "distributed", "jit"] }

# Most features except CUDA/WASM/distributed (recommended for local dev)
pandrs = { version = "0.4.2", features = ["all-safe"] }
```

#### Feature Flags
| Feature | Description | When to Use |
|---------|-------------|-------------|
| `cuda` | GPU acceleration (needs the CUDA toolkit) | Large datasets, window operations |
| `distributed` | DataFusion distributed processing | Multi-node deployments |
| `jit` | Named-closure custom aggregations (see [JIT_COMPILATION.md](JIT_COMPILATION.md) for exactly what this does today) | Custom aggregations |
| `parquet` | Parquet file format support | Analytical workloads |
| `cloud-storage` | S3 / GCS / Azure / MinIO | Cloud-native pipelines |

*(There is no `python` Cargo feature — the Python bindings are a separate crate, `py_bindings/`, built independently with maturin; see the Ecosystem Integration Guide.)*

#### Basic Usage Patterns
```rust
use pandrs::optimized::OptimizedDataFrame;

// Create and populate DataFrame
let mut df = OptimizedDataFrame::new();
df.add_int_column("id", vec![1, 2, 3])?;
df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()])?;

// Basic operations
let mean_id = df.mean("id")?;

// I/O operations (round-trip through the *same* DataFrame type)
df.to_csv("output.csv", true)?;
let loaded_df = OptimizedDataFrame::from_csv("output.csv", true)?;
```

## Learning Path

### 1. **Beginner** (New to PandRS)
1. Read [API Guide](API_GUIDE.md) sections 1-3 (Core Concepts, DataFrame Types, Series Operations)
2. Try basic examples from [examples/](../examples/) directory
3. Practice with CSV I/O operations

### 2. **Intermediate** (Familiar with basics)
1. Explore [Ecosystem Integration](ECOSYSTEM_INTEGRATION_GUIDE.md) for external connectivity
2. Learn performance basics from [Performance Plan](PERFORMANCE_PLAN.md)
3. Try database and cloud storage examples

### 3. **Advanced** (Performance-focused)
1. Master [JIT Compilation](JIT_COMPILATION.md) for custom operations
2. Implement [GPU Acceleration](GPU_ACCELERATION_GUIDE.md) for large datasets
3. Set up [benchmarking](../BENCHMARKING.md) for your workloads

### 4. **Expert** (Production deployment)
1. Implement distributed processing
2. Optimize for specific hardware configurations
3. Contribute to the PandRS ecosystem

## Common Use Cases

### 📈 Financial Analytics
- High-frequency trading data processing
- Risk analytics and backtesting
- Technical indicator calculations
- Portfolio optimization

**Recommended features:** GPU acceleration, JIT compilation, streaming I/O

### 🔬 Scientific Computing
- Large-scale numerical analysis
- Statistical modeling and hypothesis testing  
- Time series analysis and forecasting
- Machine learning feature engineering

**Recommended features:** Distributed processing, Arrow integration, custom functions

### 📊 Business Intelligence
- ETL pipeline development
- Report generation and dashboards
- Data warehouse integration
- Real-time analytics

**Recommended features:** cloud storage, Python integration (no built-in database connectivity — see the Ecosystem Integration Guide)

### 🏭 Industrial IoT
- Sensor data processing
- Predictive maintenance analytics
- Quality control monitoring
- Production optimization

**Recommended features:** Streaming processing, edge computing, memory optimization

## Development and Contributing

### Building Documentation
```bash
# Build API documentation
cargo doc --all-features --open

# Build examples
cargo build --examples --all-features

# Run documentation tests
cargo test --doc
```

### Running Examples
```bash
# Basic DataFrame operations
cargo run --example optimized_dataframe_example

# Performance demonstrations
cargo run --example performance_demo --features jit

# GPU acceleration (requires CUDA)
cargo run --example gpu_window_operations_example --features cuda

# Ecosystem integration
cargo run --example ecosystem_integration_demo --features distributed
```

### Contributing to Documentation
1. **Improve existing guides** - Add examples, clarify explanations
2. **Create specialized guides** - Domain-specific usage patterns
3. **Add performance benchmarks** - Real-world performance data
4. **Update examples** - Keep code examples current and comprehensive

## Support and Community

### Getting Help
- 📖 **Documentation**: Start with this documentation
- 💬 **GitHub Issues**: Report bugs and request features  
- 🚀 **Examples**: Browse [examples/](../examples/) for working code
- 🧪 **Tests**: Check [tests/](../tests/) for usage patterns

### Performance Issues
1. **Read [Performance Plan](PERFORMANCE_PLAN.md)** for optimization strategies
2. **Run benchmarks** to identify bottlenecks: `cargo bench`
3. **Profile your workload** with system tools
4. **Enable appropriate features** for your use case

### API Questions
1. **Check [API Guide](API_GUIDE.md)** for comprehensive examples
2. **Browse API docs**: `cargo doc --open`
3. **Look at examples** in [examples/](../examples/) directory
4. **Search existing issues** on GitHub

## Version Information

- **Current Version**: 0.4.1 (pre-1.0 — see [docs/LTS_POLICY.md](LTS_POLICY.md) for what that means for compatibility)
- **Testing**: 2700+ tests passing via `cargo nextest run --features all-safe` (see [BENCHMARKING.md](../BENCHMARKING.md) and the top-level [README.md](../README.md) for current figures — they drift release to release, so this file doesn't pin an exact count)
- **Features**: DataFrame API with analytics, ML, GPU (CUDA, optional), and distributed (DataFusion, optional) capabilities

## External Resources

### Related Projects
- **[SciRS2](https://github.com/cool-japan/scirs)** - Rust-native SciPy equivalent
- **[NumRS2](https://github.com/cool-japan/numrs)** - NumPy-style arrays for Rust
- **[Apache Arrow](https://arrow.apache.org/)** - Columnar in-memory analytics
- **[DataFusion](https://datafusion.apache.org/)** - Distributed query engine

### Ecosystem
- **Python**: `py_bindings/` — a real pyo3 crate with a pandas-compatible surface (`DataFrame`, `Series`, `OptimizedDataFrame`, `LazyFrame`, GPU bindings)
- **Jupyter**: `src/jupyter` — HTML table rendering/styling (light/dark config), magics registration, `describe_to_json` (no built-in progress bars)
- **Cloud**: AWS S3, Google Cloud Storage, Azure Blob, MinIO (`cloud-storage` feature)
- *(No built-in database connectivity.)*

---

*For the latest updates and comprehensive examples, visit the [PandRS GitHub repository](https://github.com/cool-japan/pandrs).*
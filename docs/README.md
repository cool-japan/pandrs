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
  - Database connectivity (PostgreSQL, SQLite, MySQL)
  - Cloud storage integration (AWS S3, Google Cloud, Azure)
  - Apache Arrow interoperability
  - Python bindings with pandas compatibility

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
pandrs = "0.1.0"

# With performance features
pandrs = { version = "0.1.0", features = ["cuda", "distributed", "jit"] }

# All available features
pandrs = { version = "0.1.0", features = ["all-safe"] }
```

#### Feature Flags
| Feature | Description | When to Use |
|---------|-------------|-------------|
| `cuda` | GPU acceleration | Large datasets, window operations |
| `distributed` | DataFusion distributed processing | Multi-node deployments |
| `jit` | Just-In-Time compilation | Custom aggregations |
| `parquet` | Parquet file format support | Analytical workloads |
| `python` | Python bindings | Pandas interoperability |

#### Basic Usage Patterns
```rust
use pandrs::optimized::OptimizedDataFrame;

// Create and populate DataFrame
let mut df = OptimizedDataFrame::new();
df.add_int_column("id", vec![1, 2, 3])?;
df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()])?;

// Basic operations
let mean_id = df.mean("id")?;
let grouped = df.group_by(&["category"])?.sum(&["value"])?;

// I/O operations  
df.to_csv("output.csv", true)?;
let loaded_df = pandrs::io::read_csv("input.csv", true)?;
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

**Recommended features:** Database connectivity, cloud storage, Python integration

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
cargo run --example dataframe_basics

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

- **Current Version**: 0.1.0
- **API Stability**: Stable for 0.1.x releases
- **Performance**: Production-ready with 345+ passing tests
- **Features**: Complete DataFrame API with advanced analytics capabilities

## External Resources

### Related Projects
- **[SciRS](https://github.com/cool-japan/scirs)** - Rust-native SciPy equivalent
- **[NumRS](https://github.com/cool-japan/numrs)** - NumPy-style arrays for Rust
- **[Apache Arrow](https://arrow.apache.org/)** - Columnar in-memory analytics
- **[DataFusion](https://github.com/apache/arrow-datafusion)** - Distributed query engine

### Ecosystem
- **Python**: Seamless pandas interoperability
- **Jupyter**: Rich HTML displays and progress bars
- **Cloud**: Native AWS S3, Google Cloud, Azure support
- **Databases**: PostgreSQL, SQLite, MySQL connectivity

---

*For the latest updates and comprehensive examples, visit the [PandRS GitHub repository](https://github.com/cool-japan/pandrs).*
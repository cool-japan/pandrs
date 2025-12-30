# PandRS Performance Optimization Guide

This guide covers performance optimization strategies, benchmarking tools, and best practices for getting the best performance from PandRS.

## Performance Features Overview

PandRS provides multiple performance optimization layers:

- **Column-oriented storage** with type specialization
- **String pool optimization** for memory efficiency  
- **Just-In-Time (JIT) compilation** for mathematical operations
- **GPU acceleration** (CUDA) for large datasets
- **Parallel processing** with Rayon
- **SIMD vectorization** support
- **Distributed processing** with DataFusion
- **Zero-copy operations** where possible

## Quick Performance Tips

### Use OptimizedDataFrame

```rust
// Recommended: Use OptimizedDataFrame for performance
let mut df = OptimizedDataFrame::new();
df.add_int_column("data", vec![1, 2, 3])?;

// Avoid: Traditional DataFrame for large datasets
let mut df = DataFrame::new();
```

### Enable Performance Features

```toml
[dependencies]
pandrs = { version = "0.1.0", features = ["cuda", "distributed", "jit"] }
```

### Batch Operations

```rust
// Good: Prepare data in batches
let ids: Vec<i64> = (1..=10000).collect();
let names: Vec<String> = (1..=10000).map(|i| format!("User_{}", i)).collect();
df.add_int_column("id", ids)?;
df.add_string_column("name", names)?;

// Avoid: Row-by-row operations
```

## Performance Benchmarking

### Built-in Benchmarking

PandRS includes comprehensive benchmarking infrastructure:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suites
cargo bench --bench enhanced_comprehensive_benchmark
cargo bench --bench regression_benchmark  
cargo bench --bench profiling_benchmark
```

### Benchmark Categories

1. **Enhanced Comprehensive Benchmark**
   - Realistic data patterns (Zipfian distribution, seasonal data)
   - Multiple data sizes (1K to 1M rows)
   - Throughput measurements
   - Memory tracking

2. **Regression Detection Benchmark**
   - Automated performance regression detection
   - JSON-serialized performance baselines
   - Configurable threshold alerts (default: 10%)

3. **Profiling Benchmark**
   - Memory allocation tracking
   - Data pattern analysis
   - Cache performance insights

### Performance Monitoring

```rust
use pandrs::benchmark::*;

// Benchmark database operations
let benchmark = DatabaseBenchmark::new(&connector);
let results = benchmark.run_suite().await;
println!("Query latency p95: {}ms", results.query_latency_p95);
```

## JIT Compilation

### When to Use JIT

JIT compilation provides the most benefit for:
- Custom aggregation functions called repeatedly
- Complex mathematical operations
- Large datasets (>10K elements)
- Performance-critical inner loops

### JIT Usage Examples

```rust
use pandrs::optimized::jit::{jit, GroupByJitExt};

// Create custom JIT function
let custom_metric = jit("custom_metric", |values: Vec<f64>| -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean  // Coefficient of variation
});

// Use with GroupBy
let grouped = df.group_by(&["category"])?;
let result = grouped.aggregate_jit("value", custom_metric, "cv")?;
```

### Built-in JIT Operations

```rust
use pandrs::optimized::jit::array_ops;

// High-performance built-in operations
let sum_fn = array_ops::sum();
let mean_fn = array_ops::mean();
let std_fn = array_ops::std(1);  // Sample standard deviation
let quantile_fn = array_ops::quantile(0.95);
```

## GPU Acceleration

### Enabling GPU Support

```toml
[dependencies]
pandrs = { version = "0.1.0", features = ["cuda"] }
```

### GPU Window Operations

GPU acceleration provides significant speedups for window operations on large datasets:

```rust
use pandrs::dataframe::gpu_window::GpuWindowContext;

// Initialize GPU context
let gpu_context = GpuWindowContext::new()?;

// GPU-accelerated window operations
let rolling_mean = df.gpu_rolling(50, &gpu_context).mean()?;
let expanding_sum = df.gpu_expanding(&gpu_context).sum()?;
let ewm_mean = df.gpu_ewm(0.1, &gpu_context).mean()?;
```

### GPU Performance Thresholds

Default GPU activation thresholds:
- **Standard operations** (mean, sum): 50,000 elements
- **Complex operations** (std, var, EWM): 25,000 elements  
- **Memory-bound operations** (min, max): 100,000 elements

### Custom GPU Configuration

```rust
let gpu_config = GpuConfig::new()
    .with_memory_limit(1_000_000_000)  // 1GB limit
    .with_threshold(25_000)           // Lower threshold
    .with_device_id(0);               // Specific GPU device
```

## Parallel Processing

### Automatic Parallelization

PandRS automatically parallelizes operations when beneficial:

```rust
// Automatically parallelized for large datasets
let sum = df.sum("large_column")?;
let grouped = df.groupby("category")?.sum(&["value"])?;
```

### Explicit Parallel Operations

```rust
use pandrs::optimized::jit::{ParallelConfig, parallel_sum_f64};

// Configure parallel execution
let config = ParallelConfig::new()
    .with_min_chunk_size(10000)
    .with_max_threads(8);

let parallel_sum = parallel_sum_f64(Some(config));
let result = grouped.aggregate_jit("value", parallel_sum, "sum")?;
```

## Memory Optimization

### String Pool Benefits

OptimizedDataFrame automatically uses string pooling:

```rust
// High duplication = significant memory savings
let categories = vec!["A".to_string(); 100000];  // High duplication
df.add_string_column("category", categories)?;   // Memory efficient
```

### Memory Monitoring

```rust
// Monitor memory usage
let usage = df.memory_usage()?;
println!("Total memory: {} bytes", usage.total);
println!("String pool savings: {}%", usage.string_pool_efficiency);
```

## I/O Performance

### Format Selection

Choose appropriate file formats for your use case:

| Format | Best For | Read Speed | Write Speed | Size |
|--------|----------|------------|-------------|------|
| Parquet | Analytics, compression | Fast | Fast | Small |
| CSV | Human-readable, simple | Medium | Fast | Large |
| JSON | Nested data, APIs | Slow | Medium | Large |

### Parquet Optimization

```rust
use pandrs::io::{write_parquet, ParquetCompression};

// Use compression for better I/O performance
write_parquet(&df, "data.parquet", Some(ParquetCompression::Snappy))?;
write_parquet(&df, "data.parquet", Some(ParquetCompression::Zstd))?; // Higher compression
```

### Batch I/O Operations

```rust
// Read large files in chunks
let chunk_size = 100_000;
let mut reader = df.read_csv_chunked("large_file.csv", chunk_size)?;

while let Some(chunk) = reader.next()? {
    process_chunk(&chunk)?;
    // Process each chunk separately to manage memory
}
```

## Distributed Processing

### DataFusion Integration

```rust
use pandrs::distributed::{DistributedConfig, ToDistributed};

// Convert to distributed processing
let config = DistributedConfig::new()
    .with_executor("datafusion")
    .with_concurrency(8);

let dist_df = df.to_distributed(config)?;

// Distributed operations automatically parallelize
let result = dist_df
    .filter("amount > 1000")?
    .groupby(&["region"])?
    .aggregate(&["sales"], &["sum", "mean"])?
    .execute()?;
```

## Performance Tuning Guidelines

### Data Size Thresholds

Different optimizations activate at different data sizes:

- **Small datasets** (<1K rows): Standard CPU operations
- **Medium datasets** (1K-50K rows): JIT compilation beneficial
- **Large datasets** (50K-1M rows): GPU acceleration beneficial  
- **Very large datasets** (>1M rows): Distributed processing recommended

### Operation-Specific Optimizations

1. **Aggregations**: Use JIT for custom aggregations, GPU for large datasets
2. **Window Operations**: GPU acceleration for large windows or datasets
3. **String Operations**: Leverage string pool optimization
4. **Joins**: Use distributed processing for large joins
5. **I/O**: Use Parquet for repeated access, CSV for one-time exports

### CPU Architecture Considerations

```rust
// SIMD operations benefit from:
// - Aligned data access
// - Contiguous memory layout  
// - Appropriate chunk sizes

let simd_sum = simd_sum_f64();  // Automatically vectorized
let result = grouped.aggregate_jit("values", simd_sum, "sum")?;
```

## Performance Monitoring

### Built-in Metrics

```rust
use pandrs::metrics::*;

// Enable performance monitoring
let config = PandRSConfig::new()
    .with_metrics(MetricsConfig::enabled());

// Collect operation metrics
let timer = metrics::start_timer("complex_operation");
let result = df.complex_aggregation()?;
let duration = timer.observe_duration();

println!("Operation took: {}ms", duration.as_millis());
```

### Regression Detection

```bash
# Establish performance baseline
cargo test regression_benchmark::tests::test_baseline_creation

# Run regression detection  
cargo bench --bench regression_benchmark
```

Performance regressions are automatically detected:
```
⚠️  REGRESSION DETECTED in aggregation_sum: 15.3% slower
⚠️  REGRESSION DETECTED in parallel_groupby: 12.7% slower
```

## Real-World Performance Examples

### Financial Data Processing

```rust
// Process large financial dataset efficiently
let mut financial_df = OptimizedDataFrame::new();
financial_df.add_float_column("price", prices)?;      // 1M+ prices
financial_df.add_string_column("symbol", symbols)?;   // High duplication -> string pool

// GPU-accelerated technical indicators
let gpu_context = GpuWindowContext::new()?;
let sma_20 = financial_df.gpu_rolling(20, &gpu_context).mean()?;
let volatility = financial_df.gpu_rolling(252, &gpu_context).std()?;

// JIT-compiled custom metrics
let sharpe_ratio = jit("sharpe", |returns: Vec<f64>| -> f64 {
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_return = /* std calculation */;
    mean_return / std_return * (252.0_f64).sqrt()  // Annualized Sharpe
});

let grouped = financial_df.group_by(&["symbol"])?;
let metrics = grouped.aggregate_jit("returns", sharpe_ratio, "sharpe")?;
```

### Machine Learning Pipeline

```rust
// Efficient feature engineering pipeline
let features = df
    .gpu_rolling(50, &gpu_context).mean()?          // GPU window ops
    .parallel_groupby(&["category"])?               // Parallel grouping
    .aggregate_multi_jit(vec![                      // Multiple JIT aggregations
        ("value", array_ops::mean(), "mean_value"),
        ("value", array_ops::std(1), "std_value"),
        ("value", custom_skewness, "skew_value"),
    ])?;

// Efficient model training data preparation
features.to_parquet("training_features.parquet", Some(ParquetCompression::Snappy))?;
```

## Troubleshooting Performance Issues

### Common Performance Problems

1. **Using Traditional DataFrame**: Switch to OptimizedDataFrame
2. **Small dataset GPU usage**: GPU has overhead for small datasets
3. **Unoptimized string handling**: Use string pool with repeated values
4. **Row-by-row operations**: Use vectorized operations instead
5. **Wrong file format**: Use Parquet for analytical workloads

### Diagnostic Tools

```rust
use pandrs::diagnostics::*;

// Analyze performance bottlenecks
let analysis = PerformanceAnalysis::profile_operation(|| {
    df.complex_operation()
}).await;

println!("Bottlenecks: {:?}", analysis.bottlenecks);
println!("Memory usage: {} MB", analysis.peak_memory_mb);
println!("Recommendations: {:?}", analysis.recommendations);
```

### Performance Profiling

```bash
# Profile with system tools
cargo build --release
perf record --call-graph=dwarf ./target/release/examples/performance_demo
perf report

# Memory profiling
valgrind --tool=massif ./target/release/examples/performance_demo
```

## Best Practices Summary

1. **Use OptimizedDataFrame** for all performance-critical applications
2. **Enable appropriate features** (JIT, CUDA, distributed) based on workload
3. **Choose optimal file formats** (Parquet for analytics, CSV for exports)
4. **Leverage string pooling** for categorical data with high duplication
5. **Use batch operations** instead of row-by-row processing
6. **Monitor performance** with built-in metrics and regression detection
7. **Profile before optimizing** to identify actual bottlenecks
8. **Test at scale** - performance characteristics change with data size

## Benchmarking Your Workload

```rust
use std::time::Instant;

// Benchmark your specific operations
let start = Instant::now();
let result = your_operation(&df)?;
let duration = start.elapsed();

println!("Operation: {:?}", duration);
println!("Throughput: {:.2} MB/s", data_size_mb / duration.as_secs_f64());
```

For more detailed benchmarking, see [BENCHMARKING.md](../BENCHMARKING.md).

---

*For the latest performance optimization techniques and benchmarking results, visit the [PandRS GitHub repository](https://github.com/cool-japan/pandrs).*
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
pandrs = { version = "0.4.1", features = ["cuda", "distributed", "jit"] }
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

There is no `pandrs::benchmark` module and no built-in `DatabaseBenchmark`
type — PandRS has no database connectivity at all (see
[ECOSYSTEM_INTEGRATION_GUIDE.md](ECOSYSTEM_INTEGRATION_GUIDE.md)). To
benchmark your own code, use `std::time::Instant` (see "Benchmarking Your
Workload" near the end of this document) or write a Criterion bench under
`benches/` alongside the existing suites (see
[BENCHMARKING.md](../BENCHMARKING.md)).

## JIT Compilation

### When to Use JIT

JIT compilation provides the most benefit for:
- Custom aggregation functions called repeatedly
- Complex mathematical operations
- Large datasets (>10K elements)
- Performance-critical inner loops

### JIT Usage Examples

`pandrs::optimized::jit::jit_core::jit(name, closure)` and
`GroupByJitExt::aggregate_jit` use **different closure shapes**
(`Fn(Vec<f64>) -> f64` vs. `Fn(&[f64]) -> f64`) and do not compose directly
— use `jit_f64`, which is built for `aggregate_jit`:

```rust
use pandrs::optimized::jit::core::jit_f64;
use pandrs::optimized::jit::groupby::GroupByJitExt;

// Create a custom aggregation. Honest caveat: despite the module's name,
// this does not compile the closure via Cranelift at call time today — it
// wraps it as a named, ordinary Rust closure (see JIT_COMPILATION.md for
// what "JIT" currently means in this crate).
let cv = jit_f64("cv", |values: &[f64]| -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean  // Coefficient of variation
});

// GroupByJitExt::aggregate_jit is implemented on
// optimized::split_dataframe::group::GroupBy — see
// examples/jit_parallel_example.rs for a verified-working full chain from
// DataFrame construction through to aggregate_jit, since the exact
// group-by-construction call to reach that type is easy to get wrong from
// a doc snippet alone.
```

### Direct SIMD Statistics (no JIT machinery needed)

For common reductions you don't need `jit()` at all — call the SIMD-dispatched
functions directly on a slice:

```rust
use pandrs::optimized::jit::simd::simd_sum_f64;
use pandrs::optimized::jit::simd_stats::{simd_variance_f64, simd_correlation_f64};

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let total = simd_sum_f64(&data);          // AVX2 on x86_64, scalar elsewhere
let var = simd_variance_f64(&data, 1);    // ddof = 1 (sample variance)
```

*(There is no `pandrs::optimized::jit::array_ops` module — it was deleted;
don't `use` it.)*

## GPU Acceleration

### Enabling GPU Support

```toml
[dependencies]
pandrs = { version = "0.4.1", features = ["cuda"] }
```

Building with `cuda` requires the CUDA toolkit to be installed; this can't
be verified on every dev machine, so this guide points to verified example
files rather than hand-written GPU snippets — see
[GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) and
`examples/gpu_window_operations_example.rs`,
`examples/gpu_dataframe_example.rs`, `examples/gpu_benchmark_example.rs`.

### Custom GPU Configuration

`GpuConfig` (`pandrs::gpu::GpuConfig`) is a plain struct with public fields
and a `Default` impl — there is no builder (`GpuConfig::new().with_*(...)`
does not exist):

```rust
use pandrs::gpu::GpuConfig;

let gpu_config = GpuConfig {
    memory_limit: 1_000_000_000, // 1GB
    min_size_threshold: 25_000,
    device_id: 0,
    ..Default::default()
};
```

## Parallel Processing

### Automatic Parallelization

PandRS uses Rayon internally for large-dataset operations in several
modules (see `examples/parallel_example.rs`,
`examples/optimized_parallel_groupby.rs`).

### Explicit Parallel Operations

`ParallelConfig` (`pandrs::optimized::jit::config::ParallelConfig`) is a
plain struct — construct it with `ParallelConfig::new()` then its real
builder methods (`with_min_chunk_size`, `with_max_threads`, etc. — check
`src/optimized/jit/config.rs` for the current field/method list, it's a
small file). `parallel_sum_f64` lives in
`pandrs::optimized::jit::parallel`. See `examples/jit_parallel_example.rs`
for a verified, complete, compiling example that wires
`ParallelConfig` + the `parallel_*` functions + `aggregate_jit` together —
that's a more reliable reference than a hand-copied snippet here, since
this exact combination is easy to get subtly wrong (see the JIT Usage
Examples note above about `jit()` vs `jit_f64`).

## Memory Optimization

### String Pool Benefits

OptimizedDataFrame automatically uses string pooling:

```rust
// High duplication = significant memory savings
let categories = vec!["A".to_string(); 100000];  // High duplication
df.add_string_column("category", categories)?;   // Memory efficient
```

### Memory Monitoring

The public `pandrs::optimized::OptimizedDataFrame` (the type used throughout
this guide) does not have a `memory_usage()` method. Two *different*
internal types do, with two different return shapes — don't mix them up:

```rust
// Traditional DataFrame: pandrs::dataframe::DataFrame::memory_usage(&self) -> usize
let bytes = traditional_df.memory_usage();

// An internal SplitDataFrame (not the public OptimizedDataFrame):
// memory_usage(&self) -> std::collections::HashMap<String, usize>  (per-column bytes, no `?`)
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

There is no `read_csv_chunked` method. For chunked/streaming ingestion, use
the `streaming` feature's `DataStream` (`pandrs::streaming::DataStream`,
e.g. `DataStream::read_from_csv(...)` + `.process(...)`) — see
`examples/streaming_example.rs` for a verified working example.

## Distributed Processing

### DataFusion Integration

```rust
use pandrs::distributed::{DistributedConfig, ToDistributed};

// Convert to distributed processing
let config = DistributedConfig::new()
    .with_executor("datafusion")
    .with_concurrency(8);

let mut dist_df = df.to_distributed(config)?;

// DistributedDataFrame has no separate `.groupby()` — the group-by columns
// are the *first* argument to `.aggregate()`, and each aggregate is a
// (column, function, output_alias) triple:
let mut result = dist_df
    .filter("amount > 1000")?
    .aggregate(
        &["region"],
        &[("sales", "sum", "sales_sum"), ("sales", "mean", "sales_mean")],
    )?;
let execution_result = result.execute()?;
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

use pandrs::optimized::jit::simd::simd_sum_f64;

// `simd_sum_f64` takes the slice directly and returns the sum — it is not
// a zero-arg factory you pass into `aggregate_jit`:
let data = vec![1.0_f64, 2.0, 3.0, 4.0];
let result = simd_sum_f64(&data);  // AVX2-dispatched on x86_64, scalar elsewhere
```

## Performance Monitoring

There is no `pandrs::metrics` module, no `PandRSConfig`/`MetricsConfig`, and
no built-in timer/metrics-collection API. For ad-hoc timing, use
`std::time::Instant` directly (see "Benchmarking Your Workload" below). For
structured monitoring, `src/analytics` provides counters/gauges/histograms/timers
as a standalone facility you wire up yourself — it is not automatically
attached to DataFrame operations; see `examples/analytics_dashboard_example.rs`.

### Regression Detection

```bash
cargo bench --bench regression_benchmark
```

This prints regression warnings *if* a `benchmark_baseline.json` is present
and a tracked operation regresses >10%. See
[BENCHMARKING.md](../BENCHMARKING.md#establishing-baselines-currently-not-wired-to-a-runnable-command)
for the current, honest state of baseline creation — `cargo test
regression_benchmark::tests::test_baseline_creation` (seen in older
versions of this document) does not work.

## Real-World Performance Examples

Hand-written "realistic pipeline" snippets in this section previously
chained several APIs together in ways that don't compile against the
current crate (`array_ops` was deleted; `jit()`'s closure shape doesn't
match `aggregate_jit`'s; `aggregate_multi_jit`/`parallel_groupby` don't
exist), and one of the chains — GPU rolling windows piped into `.mean()` —
currently hits a real correctness bug in `src/dataframe/gpu_window.rs`
(the result comes back as string data, not the numeric aggregate you'd
expect), so showing it here as a trustworthy pattern would be actively
misleading rather than just wrong.

For working, verified full pipelines, use these example files directly —
they're compiled and spot-checked as part of the example suite:
- Financial-style windowed aggregation: `examples/gpu_window_operations_example.rs`, `examples/comprehensive_window_example.rs`
- JIT custom aggregations end-to-end: `examples/jit_parallel_example.rs`, `examples/integrated_jit_performance_showcase.rs`
- ML feature-engineering pipeline: `examples/optimized_ml_feature_engineering_example.rs`, `examples/optimized_ml_pipeline_example.rs`

## Troubleshooting Performance Issues

### Common Performance Problems

1. **Using Traditional DataFrame**: Switch to OptimizedDataFrame
2. **Small dataset GPU usage**: GPU has overhead for small datasets
3. **Unoptimized string handling**: Use string pool with repeated values
4. **Row-by-row operations**: Use vectorized operations instead
5. **Wrong file format**: Use Parquet for analytical workloads

### Diagnostic Tools

There is no `pandrs::diagnostics` module or `PerformanceAnalysis` type.
Use system-level profiling tools instead (see "Performance Profiling" below),
or time specific sections with `std::time::Instant`.

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
6. **Monitor performance** with `cargo bench --bench regression_benchmark` (once you have a baseline — see caveat above) or your own `Instant`-based timing
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
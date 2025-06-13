# PandRS Benchmarking Infrastructure

This document describes the comprehensive benchmarking infrastructure for PandRS alpha.4, designed to measure performance, detect regressions, and guide optimization efforts.

## Benchmark Suites

### 1. Enhanced Comprehensive Benchmark (`enhanced_comprehensive_benchmark.rs`)

A complete performance testing suite with realistic data patterns and comprehensive metrics.

**Features:**
- **Realistic Data Generation**: Uses Zipfian distribution for categorical data, seasonal patterns for numeric data, and configurable null percentages
- **Multiple Data Sizes**: Tests from 1K to 1M rows for scalability analysis
- **Throughput Measurement**: Reports elements processed per second
- **Memory-Aware Testing**: Configurable memory tracking capabilities

**Benchmark Categories:**
- DataFrame creation with different data patterns
- Aggregation operations (sum, mean, min, max) with size scaling
- GroupBy operations with varying cardinalities (10 to 10K categories)
- I/O operations (CSV and Parquet write performance)
- String operations with parallel processing
- Memory scalability testing
- SIMD operation comparisons
- Full analytics pipeline benchmarks

**Usage:**
```bash
cargo bench --bench enhanced_comprehensive_benchmark
```

### 2. Regression Detection Benchmark (`regression_benchmark.rs`)

Automated performance regression detection system with baseline comparison.

**Features:**
- **Performance Baselines**: JSON-serialized performance history
- **Regression Detection**: Configurable threshold-based regression alerts (default: 10%)
- **Deterministic Testing**: Seeded random number generation for reproducible results
- **Automated Alerts**: Console warnings when performance degrades

**Key Operations Monitored:**
- DataFrame creation performance
- Core aggregation operations (sum, mean, min, max)
- Parallel GroupBy operations
- SIMD vectorized operations
- I/O operation throughput

**Usage:**
```bash
# Run regression tests
cargo bench --bench regression_benchmark

# Establish new baseline (run after performance improvements)
cargo test regression_benchmark::tests::test_baseline_creation
```

### 3. Profiling Benchmark (`profiling_benchmark.rs`)

Detailed performance profiling with memory tracking and pattern analysis.

**Features:**
- **Memory Allocation Tracking**: Custom allocator for memory usage analysis
- **Data Pattern Analysis**: Tests different data distributions (sequential, random, sparse, strings)
- **Cache Performance**: Memory access pattern optimization insights
- **Throughput Analysis**: MB/sec calculations for I/O operations

**Profiling Categories:**
- DataFrame creation with different patterns
- Aggregation operations with memory efficiency metrics
- SIMD operations with data pattern sensitivity
- I/O operations with throughput measurement
- Memory-intensive operation patterns

**Usage:**
```bash
cargo bench --bench profiling_benchmark
```

### 4. Legacy DataFrame Benchmark (`legacy_dataframe_bench.rs`)

Compatibility benchmark for the original DataFrame API.

**Features:**
- **API Compatibility**: Tests original DataFrame interface
- **Baseline Comparison**: Reference performance for new optimized implementations
- **Simple Operations**: Focus on core DataFrame functionality

**Usage:**
```bash
cargo bench --bench legacy_dataframe_bench
```

### 5. Comprehensive Benchmark (`comprehensive_benchmark.rs`)

Original comprehensive benchmark suite with basic performance testing.

**Usage:**
```bash
cargo bench --bench comprehensive_benchmark
```

## Benchmark Configuration

### Feature Flags

Benchmarks automatically adapt to available features:

```toml
# Run with Parquet support
cargo bench --features parquet

# Run with all available features
cargo bench --features all-safe
```

### Environment Variables

```bash
# Set number of iterations
export CRITERION_SAMPLE_SIZE=50

# Generate HTML reports
export CRITERION_HTML=1
```

## Performance Baselines

### Establishing Baselines

```rust
// In regression_benchmark.rs
#[test]
fn test_baseline_creation() {
    establish_baseline();
}
```

Run this test after making performance improvements to update the baseline:

```bash
cargo test regression_benchmark::tests::test_baseline_creation
```

### Baseline Format

Baselines are stored in JSON format with the following structure:

```json
{
  "version": "0.1.0-alpha.4",
  "timestamp": 1640995200,
  "benchmarks": {
    "dataframe_creation": {
      "mean_time_ns": 1500000.0,
      "std_dev_ns": 50000.0,
      "throughput_ops_per_sec": 66666.67,
      "memory_usage_bytes": 1048576
    }
  }
}
```

## Performance Monitoring

### Regression Alerts

The system automatically detects performance regressions:

```
⚠️  REGRESSION DETECTED in aggregation_sum: 15.3% slower
⚠️  REGRESSION DETECTED in parallel_groupby: 12.7% slower
```

### Throughput Reporting

```
📊 Pattern: random, Size: 100000, Time: 45.2ms, Memory: 2048576 bytes, Peak: 3145728 bytes
⚡ simd_sum, Pattern: sequential, Throughput: 2.5M elements/sec
💾 CSV Write, Size: 50000, Throughput: 125.3 MB/sec
📦 Parquet Write, Size: 50000, Throughput: 89.7 MB/sec
```

## Optimization Guidelines

### Data Access Patterns

1. **Sequential Access**: Fastest for cache efficiency
2. **Random Access**: Test with realistic access patterns
3. **Sparse Data**: Handle null values efficiently

### Memory Efficiency

1. **Peak Memory Tracking**: Monitor memory usage spikes
2. **Allocation Patterns**: Minimize allocations in hot paths
3. **Cache Locality**: Structure data for CPU cache efficiency

### SIMD Utilization

1. **Data Alignment**: Ensure proper alignment for vectorization
2. **Chunk Sizes**: Optimize for SIMD register sizes
3. **Fallback Paths**: Provide scalar implementations

## Integration with CI/CD

### Automated Regression Detection

```yaml
# .github/workflows/benchmarks.yml
- name: Run Performance Benchmarks
  run: |
    cargo bench --bench regression_benchmark
    # Fail if regressions detected (exit code handling)
```

### Performance Tracking

```yaml
- name: Store Benchmark Results
  run: |
    cargo bench --bench enhanced_comprehensive_benchmark -- --output-format json
    # Upload results to performance monitoring service
```

## Troubleshooting

### Common Issues

1. **Memory Allocator**: Memory tracking requires custom allocator setup
2. **Feature Flags**: Some benchmarks require specific features (e.g., parquet)
3. **System Resources**: Large dataset benchmarks may require sufficient RAM

### Performance Tips

1. **System Isolation**: Run benchmarks on dedicated systems
2. **Thermal Throttling**: Monitor CPU temperature during long benchmarks
3. **Background Processes**: Minimize system load during benchmark runs

## Future Enhancements

### Planned Improvements

1. **GPU Benchmarks**: CUDA acceleration performance testing
2. **Distributed Benchmarks**: Multi-node performance evaluation
3. **Real-world Workloads**: Industry-specific benchmark scenarios
4. **Continuous Profiling**: Integration with profiling services

### Benchmark Additions

1. **Join Operations**: Cross-DataFrame operation performance
2. **Window Functions**: Time-series analysis benchmarks
3. **Machine Learning**: ML pipeline performance testing
4. **Streaming Operations**: Real-time data processing benchmarks

---

For more information, see the individual benchmark files in the `benches/` directory.
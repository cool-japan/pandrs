# PandRS Benchmarking Infrastructure

**Version**: 0.4.1

This document describes the benchmarking infrastructure for PandRS
(Criterion-based, `harness = false`, declared under `[[bench]]` in
`Cargo.toml`). There are **12 bench binaries** under `benches/`; this file
documents the 5 most commonly used ones in detail below, plus a full list of
all 12 further down. Sample numbers shown anywhere in this document
(throughput figures, regression-alert text, etc.) are illustrative output
*format*, not measured results — run the benchmarks yourself for real
numbers, and see the caveats under "Memory Tracking" and "Establishing
Baselines" below before trusting any single run.

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
cargo bench --bench regression_benchmark
```

*(There is no working "establish new baseline" command today — see
"Establishing Baselines" below.)*

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

### Other Benchmark Suites

The remaining 7 of the 12 `[[bench]]` targets in `Cargo.toml`, run the same
way (`cargo bench --bench <name>`):

| Bench | Covers |
|-------|--------|
| `comparison_benchmark` | DataFrame creation, GroupBy, filtering, sorting, joins, string ops, aggregation — sized to line up with the Python `pandas_benchmark.py` / `polars_benchmark.py` scripts (see [README.md](README.md#performance) for how those relate) |
| `pandas_comparison_benchmark` | A Rust re-implementation of pandas-equivalent operations, timed in isolation — it does **not** invoke real pandas; use `pandas_benchmark.py` for that |
| `comprehensive_features_benchmark` | DataFrame creation, column management, string-pool effects, I/O, distributed processing, series ops, memory usage, aggregation, type conversion, error handling |
| `ml_benchmarks` | Decision trees, ensemble methods, neural network training |
| `query_optimizer_benchmarks` | Query-plan optimization techniques |
| `multitenancy_benchmarks` | Tenant management and data-isolation operations |
| `simd_string_benchmarks` | SIMD-accelerated vs. scalar string operations |

## Benchmark Configuration

### Feature Flags

Benchmarks automatically adapt to available features:

```toml
# Run with Parquet support
cargo bench --features parquet

# Run with all available features
cargo bench --features all-safe
```

### Criterion Configuration

There are no `CRITERION_SAMPLE_SIZE` / `CRITERION_HTML` environment
variables — Criterion doesn't read either of those. What's actually true:

- **HTML reports**: PandRS depends on `criterion` with the `html_reports`
  feature enabled in `Cargo.toml`, so HTML output under
  `target/criterion/` is generated automatically on every `cargo bench` run
  — no configuration needed.
- **Sample size / other run parameters**: pass Criterion's own CLI flags
  after `--`, e.g. `cargo bench --bench comprehensive_benchmark -- --sample-size 50`.
  Run `cargo bench --bench <name> -- --help` for the full flag list for that
  binary.

## Performance Baselines

### Establishing Baselines (currently not wired to a runnable command)

`regression_benchmark.rs` contains an `establish_baseline()` helper and a
`#[test] fn test_baseline_creation()` that calls it, but this bench target
is declared `harness = false` in `Cargo.toml` — which means its own `main()`
(from `criterion_main!`) replaces the standard test harness, so `#[test]`
functions inside it are not runnable via `cargo test` the normal way.
`cargo test regression_benchmark::tests::test_baseline_creation` (a command
that appeared in earlier versions of this document) does not work. Until
this is wired up, treat `establish_baseline()` as source you'd invoke by
temporarily calling it from the bench's own `main`, not as a documented CLI
command.

### Baseline Format

Baselines are stored in JSON format with the following structure:

```json
{
  "version": "0.1.0",
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

`regression_benchmark.rs`'s 5 registered bench functions print a warning
(format below) when a `benchmark_baseline.json` file is present and the
current run is more than 10% slower than the recorded baseline for that
operation. This detection code runs for real on every
`cargo bench --bench regression_benchmark` invocation — but per the note
above, there is currently no clean, documented way to *create* that
baseline file in the first place, so treat this as "the mechanism exists in
source" rather than "there's a baseline checked in for you to compare
against."

```
⚠️  REGRESSION DETECTED in aggregation_sum: 15.3% slower
⚠️  REGRESSION DETECTED in parallel_groupby: 12.7% slower
```

*(Example output format — the specific percentages above are illustrative, not measured.)*

### Throughput Reporting

Example output format from the benches that print throughput lines
(illustrative, not measured):

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

**No CI workflow currently runs benchmarks.** The only workflow in
`.github/workflows/` (`pypi-publish.yml`) builds and publishes Python wheels
on version tags; it does not run `cargo bench`. The YAML below is an
illustrative sketch of what a benchmarks workflow *could* look like if one
is added later — it is not wired up today and referencing it as `.yml` is
not a claim that the file exists.

```yaml
# Illustrative only — not a workflow that exists in this repo today.
- name: Run Performance Benchmarks
  run: |
    cargo bench --bench regression_benchmark
    # Fail if regressions detected (exit code handling)
```

## Troubleshooting

### Common Issues

1. **Memory Allocator**: `profiling_benchmark.rs` defines a `TrackingAllocator`,
   but its `#[global_allocator]` registration is currently commented out in
   that file — so the memory-usage numbers that benchmark prints are not
   measuring real allocations right now. Treat any "Memory:"/"Peak:" output
   from `profiling_benchmark` as inert until that's re-enabled.
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
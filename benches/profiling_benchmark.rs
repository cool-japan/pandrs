use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pandrs::error::Result;
use pandrs::optimized::jit::*;
use pandrs::optimized::OptimizedDataFrame;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Memory tracking allocator for profiling
#[allow(dead_code)]
struct TrackingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static PEAK_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let size = layout.size();
            let current = ALLOCATED.fetch_add(size, Ordering::SeqCst) + size;
            let peak = PEAK_ALLOCATED.load(Ordering::SeqCst);
            if current > peak {
                PEAK_ALLOCATED.store(current, Ordering::SeqCst);
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
    }
}

// Uncomment to enable memory tracking (requires unstable features)
// #[global_allocator]
// static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Performance profiling metrics
#[derive(Debug, Clone)]
pub struct ProfilingMetrics {
    pub execution_time: Duration,
    pub memory_allocated: usize,
    pub memory_deallocated: usize,
    pub peak_memory: usize,
    pub cache_hits: Option<usize>,
    pub cache_misses: Option<usize>,
}

impl Default for ProfilingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingMetrics {
    pub fn new() -> Self {
        Self {
            execution_time: Duration::default(),
            memory_allocated: 0,
            memory_deallocated: 0,
            peak_memory: 0,
            cache_hits: None,
            cache_misses: None,
        }
    }

    pub fn reset_memory_counters() {
        ALLOCATED.store(0, Ordering::SeqCst);
        DEALLOCATED.store(0, Ordering::SeqCst);
        PEAK_ALLOCATED.store(0, Ordering::SeqCst);
    }

    pub fn capture_memory_stats(&mut self) {
        self.memory_allocated = ALLOCATED.load(Ordering::SeqCst);
        self.memory_deallocated = DEALLOCATED.load(Ordering::SeqCst);
        self.peak_memory = PEAK_ALLOCATED.load(Ordering::SeqCst);
    }

    pub fn net_memory_usage(&self) -> i64 {
        self.memory_allocated as i64 - self.memory_deallocated as i64
    }

    pub fn memory_efficiency(&self) -> f64 {
        if self.memory_allocated == 0 {
            return 1.0;
        }
        self.memory_deallocated as f64 / self.memory_allocated as f64
    }
}

/// Profiling-aware DataFrame creator with various patterns
#[allow(clippy::result_large_err)]
fn create_profiling_dataframe(size: usize, pattern: &str) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    match pattern {
        "sequential" => {
            let int_data: Vec<i64> = (0..size).map(|i| i as i64).collect();
            let float_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
            df.add_int_column("sequential_int", int_data)?;
            df.add_float_column("sequential_float", float_data)?;
        }
        "random" => {
            use rand::prelude::*;
            let mut rng = rand::rng();
            let int_data: Vec<i64> = (0..size).map(|_| rng.random_range(0..10000)).collect();
            let float_data: Vec<f64> = (0..size).map(|_| rng.random::<f64>() * 1000.0).collect();
            df.add_int_column("random_int", int_data)?;
            df.add_float_column("random_float", float_data)?;
        }
        "sparse" => {
            // Create sparse data with many nulls
            let int_data: Vec<i64> = (0..size)
                .map(|i| if i % 10 == 0 { i as i64 } else { 0 })
                .collect();
            let float_data: Vec<f64> = (0..size)
                .map(|i| if i % 5 == 0 { i as f64 * 0.1 } else { f64::NAN })
                .collect();
            df.add_int_column("sparse_int", int_data)?;
            df.add_float_column("sparse_float", float_data)?;
        }
        "strings" => {
            let string_data: Vec<String> = (0..size).map(|i| format!("String_{:08}", i)).collect();
            df.add_string_column("string_data", string_data)?;
        }
        _ => {
            return Err(pandrs::error::Error::InvalidValue(format!(
                "Unknown pattern: {}",
                pattern
            )));
        }
    }

    Ok(df)
}

/// Profile DataFrame creation with different patterns
fn profile_dataframe_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_dataframe_creation");
    group.measurement_time(Duration::from_secs(10));

    let patterns = ["sequential", "random", "sparse", "strings"];
    let sizes = [1_000, 10_000, 50_000];

    for pattern in &patterns {
        for &size in &sizes {
            group.throughput(Throughput::Elements(size as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("pattern_{}", pattern), size),
                &(size, *pattern),
                |b, &(size, pattern)| {
                    b.iter_custom(|iters| {
                        ProfilingMetrics::reset_memory_counters();
                        let start = Instant::now();

                        for _ in 0..iters {
                            let df = create_profiling_dataframe(size, pattern).unwrap();
                            black_box(df);
                        }

                        let elapsed = start.elapsed();

                        let mut metrics = ProfilingMetrics::new();
                        metrics.execution_time = elapsed;
                        metrics.capture_memory_stats();
                        println!(
                            "📊 Pattern: {}, Size: {}, Time: {:.2}ms, Memory: {} bytes, Peak: {} bytes",
                            pattern,
                            size,
                            elapsed.as_millis(),
                            metrics.net_memory_usage(),
                            metrics.peak_memory
                        );

                        elapsed
                    });
                },
            );
        }
    }

    group.finish();
}

/// Profile aggregation operations with detailed metrics
fn profile_aggregation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_aggregations");
    group.measurement_time(Duration::from_secs(15));

    let df = create_profiling_dataframe(100_000, "sequential").unwrap();
    group.throughput(Throughput::Elements(100_000));

    // Profile different aggregation methods
    let aggregations = [
        (
            "standard_sum",
            Box::new(|df: &OptimizedDataFrame| df.sum("sequential_int").unwrap())
                as Box<dyn Fn(&OptimizedDataFrame) -> f64>,
        ),
        (
            "standard_mean",
            Box::new(|df: &OptimizedDataFrame| df.mean("sequential_float").unwrap()),
        ),
    ];

    for (name, operation) in aggregations {
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                ProfilingMetrics::reset_memory_counters();
                let start = Instant::now();

                for _ in 0..iters {
                    let result = operation(&df);
                    black_box(result);
                }

                let elapsed = start.elapsed();

                let mut metrics = ProfilingMetrics::new();
                metrics.execution_time = elapsed;
                metrics.capture_memory_stats();

                println!(
                    "🔍 Operation: {}, Time: {:.2}μs/iter, Memory efficiency: {:.2}%",
                    name,
                    elapsed.as_micros() as f64 / iters as f64,
                    metrics.memory_efficiency() * 100.0
                );

                elapsed
            });
        });
    }

    group.finish();
}

/// Profile SIMD operations with different data patterns
fn profile_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_simd");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [1_000, 10_000, 100_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Test different data patterns
        let patterns = [
            (
                "sequential",
                (0..size).map(|i| i as f64).collect::<Vec<f64>>(),
            ),
            ("random", {
                use rand::prelude::*;
                let mut rng = rand::rng();
                (0..size).map(|_| rng.random::<f64>() * 1000.0).collect()
            }),
            (
                "alternating",
                (0..size)
                    .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
                    .collect(),
            ),
        ];

        for (pattern_name, data) in patterns {
            // Compare SIMD implementations
            let simd_ops = [
                (
                    "simd_sum",
                    Box::new(|data: &[f64]| simd_sum_f64(data)) as Box<dyn Fn(&[f64]) -> f64>,
                ),
                (
                    "parallel_sum",
                    Box::new(|data: &[f64]| parallel::immediate::sum(data, None)),
                ),
                // Note: simd_parallel module not available in scope
            ];

            for (op_name, operation) in simd_ops {
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_{}_size_{}", op_name, pattern_name, size), size),
                    &data,
                    |b, data| {
                        b.iter_custom(|iters| {
                            let start = Instant::now();

                            for _ in 0..iters {
                                let result = operation(data);
                                black_box(result);
                            }

                            let elapsed = start.elapsed();

                            // Calculate throughput
                            let elements_per_sec =
                                (size as f64 * iters as f64) / elapsed.as_secs_f64();

                            if size == 100_000 {
                                // Only print for largest size to reduce noise
                                println!(
                                    "⚡ {}, Pattern: {}, Throughput: {:.2}M elements/sec",
                                    op_name,
                                    pattern_name,
                                    elements_per_sec / 1_000_000.0
                                );
                            }

                            elapsed
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Profile I/O operations with different file sizes and formats
fn profile_io_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_io");
    group.measurement_time(Duration::from_secs(20));

    let sizes = [1_000, 10_000, 50_000];

    for &size in &sizes {
        let df = create_profiling_dataframe(size, "sequential").unwrap();
        group.throughput(Throughput::Elements(size as u64));

        // Profile CSV operations
        group.bench_with_input(BenchmarkId::new("csv_write", size), &df, |b, _df| {
            b.iter_custom(|iters| {
                ProfilingMetrics::reset_memory_counters();
                let start = Instant::now();

                for i in 0..iters {
                    let temp_path = format!("/tmp/profile_test_{}_{}.csv", size, i);
                    df.to_csv(&temp_path, true).unwrap();
                    std::fs::remove_file(&temp_path).ok();
                }

                let elapsed = start.elapsed();

                let file_size_estimate = size * 50; // Rough estimate
                let mb_per_sec = (file_size_estimate as f64 * iters as f64)
                    / (1024.0 * 1024.0 * elapsed.as_secs_f64());

                println!(
                    "💾 CSV Write, Size: {}, Throughput: {:.2} MB/sec",
                    size, mb_per_sec
                );

                elapsed
            });
        });

        // Profile Parquet operations
        group.bench_with_input(BenchmarkId::new("parquet_write", size), &df, |b, _df| {
            b.iter_custom(|iters| {
                let start = Instant::now();

                for _i in 0..iters {
                    #[cfg(feature = "parquet")]
                    {
                        use pandrs::io::{write_parquet, ParquetCompression};
                        let temp_path = format!("/tmp/profile_test_{}_{}.parquet", size, i);
                        write_parquet(df, &temp_path, Some(ParquetCompression::Snappy)).unwrap();
                        std::fs::remove_file(&temp_path).ok();
                    }
                    #[cfg(not(feature = "parquet"))]
                    {
                        // Skip parquet test when feature is not enabled
                    }
                }

                let elapsed = start.elapsed();

                let file_size_estimate = size * 30; // Parquet is typically more compact
                let mb_per_sec = (file_size_estimate as f64 * iters as f64)
                    / (1024.0 * 1024.0 * elapsed.as_secs_f64());

                println!(
                    "📦 Parquet Write, Size: {}, Throughput: {:.2} MB/sec",
                    size, mb_per_sec
                );

                elapsed
            });
        });
    }

    group.finish();
}

/// Profile memory usage patterns during complex operations
fn profile_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("profile_memory");
    group.measurement_time(Duration::from_secs(15));

    let sizes = [10_000, 50_000, 100_000];

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("memory_intensive_operations", size),
            &size,
            |b, &size| {
                b.iter_custom(|iters| {
                    ProfilingMetrics::reset_memory_counters();
                    let start = Instant::now();

                    for _ in 0..iters {
                        // Create multiple DataFrames and perform operations
                        let df1 = create_profiling_dataframe(size / 2, "sequential").unwrap();
                        let df2 = create_profiling_dataframe(size / 2, "random").unwrap();

                        // Perform memory-intensive operations
                        let sum1 = df1.sum("sequential_int").unwrap();
                        let sum2 = df2.sum("random_int").unwrap();
                        let grouped1 = df1.par_groupby(&["sequential_int"]).unwrap();

                        black_box((sum1, sum2, grouped1));

                        // DataFrames go out of scope here, triggering deallocation
                    }

                    let elapsed = start.elapsed();

                    let mut metrics = ProfilingMetrics::new();
                    metrics.execution_time = elapsed;
                    metrics.capture_memory_stats();

                    println!(
                        "🧠 Memory Pattern, Size: {}, Peak: {} KB, Efficiency: {:.2}%",
                        size,
                        metrics.peak_memory / 1024,
                        metrics.memory_efficiency() * 100.0
                    );

                    elapsed
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    profiling_benches,
    profile_dataframe_creation,
    profile_aggregation_operations,
    profile_simd_operations,
    profile_io_operations,
    profile_memory_patterns
);

criterion_main!(profiling_benches);

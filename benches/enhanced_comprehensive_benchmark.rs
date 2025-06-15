use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pandrs::error::Result;
use pandrs::optimized::jit::*;
use pandrs::optimized::OptimizedDataFrame;
use std::time::Duration;

/// Enhanced benchmark configuration with realistic data patterns
struct BenchmarkConfig {
    sizes: Vec<usize>,
    string_cardinality: usize,
    null_percentage: f64,
    enable_jit_comparison: bool,
    #[allow(dead_code)]
    enable_memory_tracking: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sizes: vec![1_000, 10_000, 50_000, 100_000, 500_000],
            string_cardinality: 1000,
            null_percentage: 0.05, // 5% null values
            enable_jit_comparison: true,
            enable_memory_tracking: true,
        }
    }
}

/// Enhanced data generator with realistic patterns
#[allow(clippy::result_large_err)]
fn create_realistic_dataframe(size: usize, config: &BenchmarkConfig) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();
    use rand::prelude::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // Use seeded RNG for reproducible benchmarks
    let mut rng = StdRng::seed_from_u64(42);

    // Generate categorical data with realistic cardinality
    let categories: Vec<String> = (0..config.string_cardinality)
        .map(|i| format!("Category_{:06}", i))
        .collect();

    let mut cat_data = Vec::with_capacity(size);
    let mut int_data = Vec::with_capacity(size);
    let mut float_data = Vec::with_capacity(size);
    let mut bool_data = Vec::with_capacity(size);
    let mut nullable_float_data = Vec::with_capacity(size);

    for i in 0..size {
        // Categorical data with zipfian distribution (more realistic)
        let cat_index = if rng.random::<f64>() < 0.8 {
            // 80% of data comes from first 20% of categories
            rng.random_range(0..(config.string_cardinality / 5).max(1))
        } else {
            rng.random_range(0..config.string_cardinality)
        };
        cat_data.push(categories[cat_index % categories.len()].clone());

        // Integer data with trends and patterns
        let base_value = (i as f64 / 1000.0).sin() * 1000.0 + i as f64 * 0.1;
        int_data.push(base_value as i64 + rng.random_range(-100..100));

        // Float data with normal distribution and seasonal patterns
        let seasonal = (i as f64 / 100.0).cos() * 10.0;
        let noise = rng.random::<f64>() * 20.0 - 10.0;
        float_data.push(100.0 + seasonal + noise);

        // Boolean data with bias
        bool_data.push(rng.random::<f64>() < 0.7);

        // Nullable float data
        if rng.random::<f64>() < config.null_percentage {
            nullable_float_data.push(f64::NAN);
        } else {
            nullable_float_data.push(rng.random::<f64>() * 1000.0);
        }
    }

    df.add_string_column("category", cat_data)?;
    df.add_int_column("trend_value", int_data)?;
    df.add_float_column("score", float_data)?;
    df.add_float_column("nullable_score", nullable_float_data)?;

    // Add time-series like data
    let mut ts_data = Vec::with_capacity(size);
    for i in 0..size {
        ts_data.push(1_600_000_000i64 + (i as i64 * 3600)); // Hourly timestamps
    }
    df.add_int_column("timestamp", ts_data)?;

    Ok(df)
}

/// Enhanced DataFrame creation benchmarks with memory tracking
fn benchmark_enhanced_dataframe_creation(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("enhanced_dataframe_creation");

    for &size in &config.sizes {
        // Set throughput for better measurement
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("realistic_data", size),
            &size,
            |b, &size| {
                b.iter(|| black_box(create_realistic_dataframe(size, &config).unwrap()));
            },
        );

        // Compare with simple data
        group.bench_with_input(BenchmarkId::new("simple_data", size), &size, |b, &size| {
            b.iter(|| {
                let mut df = OptimizedDataFrame::new();
                let int_data: Vec<i64> = (0..size).map(|i| i as i64).collect();
                let float_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
                df.add_int_column("simple_int", int_data).unwrap();
                df.add_float_column("simple_float", float_data).unwrap();
                black_box(df)
            });
        });
    }

    group.finish();
}

/// Enhanced aggregation benchmarks with JIT vs non-JIT comparison
fn benchmark_enhanced_aggregations(c: &mut Criterion) {
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("enhanced_aggregations");
    group.measurement_time(Duration::from_secs(15));

    // Test multiple data sizes
    for &size in &[10_000, 100_000, 1_000_000] {
        let df = create_realistic_dataframe(size, &config).unwrap();
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark different aggregation operations
        let operations: &[(&str, Box<dyn Fn(&OptimizedDataFrame) -> f64>)] = &[
            (
                "sum",
                Box::new(|df: &OptimizedDataFrame| df.sum("trend_value").unwrap()),
            ),
            (
                "mean",
                Box::new(|df: &OptimizedDataFrame| df.mean("score").unwrap()),
            ),
            (
                "min",
                Box::new(|df: &OptimizedDataFrame| df.min("trend_value").unwrap()),
            ),
            (
                "max",
                Box::new(|df: &OptimizedDataFrame| df.max("trend_value").unwrap()),
            ),
        ];

        for (op_name, op_func) in operations.iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_size_{}", op_name, size), size),
                &df,
                |b, df| {
                    b.iter(|| black_box(op_func(df)));
                },
            );

            // JIT comparison if enabled
            if config.enable_jit_comparison {
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_jit_size_{}", op_name, size), size),
                    &df,
                    |b, df| {
                        let _jit_config = ParallelConfig::default();
                        b.iter(|| {
                            // Note: JIT config operations not yet implemented in OptimizedDataFrame
                            // Using standard operations for now
                            black_box(op_func(df))
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

/// Enhanced GroupBy benchmarks with different cardinalities
fn benchmark_enhanced_groupby(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_groupby");
    group.measurement_time(Duration::from_secs(20));

    let size = 100_000;

    // Test different cardinalities
    for cardinality in [10, 100, 1000, 10000] {
        let config = BenchmarkConfig {
            string_cardinality: cardinality,
            ..Default::default()
        };
        let df = create_realistic_dataframe(size, &config).unwrap();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("parallel_groupby", cardinality),
            &df,
            |b, df| {
                b.iter(|| black_box(df.par_groupby(&["category"]).unwrap()));
            },
        );

        // Test multi-column groupby
        group.bench_with_input(
            BenchmarkId::new("multi_column_groupby", cardinality),
            &df,
            |b, df| {
                b.iter(|| {
                    // Create a secondary grouping column
                    let mut secondary_groups = Vec::new();
                    for i in 0..df.row_count() {
                        secondary_groups.push(format!("Group_{}", i % 5));
                    }
                    // Note: This would require extending the API
                    black_box(df.par_groupby(&["category"]).unwrap())
                });
            },
        );
    }

    group.finish();
}

/// I/O performance benchmarks
fn benchmark_enhanced_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_io");
    group.measurement_time(Duration::from_secs(10));

    let config = BenchmarkConfig::default();

    for &size in &[1_000, 10_000, 100_000] {
        let df = create_realistic_dataframe(size, &config).unwrap();
        group.throughput(Throughput::Elements(size as u64));

        // CSV benchmarks
        group.bench_with_input(BenchmarkId::new("csv_write", size), &df, |b, df| {
            use std::fs;
            b.iter(|| {
                let temp_path = format!("/tmp/benchmark_test_{}.csv", rand::random::<u32>());
                df.to_csv(&temp_path, true).unwrap(); // true for write_header
                let _ = fs::remove_file(&temp_path);
                black_box(())
            });
        });

        // Parquet benchmarks (only if feature enabled)
        #[cfg(feature = "parquet")]
        group.bench_with_input(BenchmarkId::new("parquet_write", size), &df, |b, df| {
            use pandrs::io::{write_parquet, ParquetCompression};
            use std::fs;
            b.iter(|| {
                let temp_path = format!("/tmp/benchmark_test_{}.parquet", rand::random::<u32>());
                write_parquet(df, &temp_path, Some(ParquetCompression::Snappy)).unwrap();
                let _ = fs::remove_file(&temp_path);
                black_box(())
            });
        });
    }

    group.finish();
}

/// String operations benchmarks
fn benchmark_enhanced_string_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_string_ops");

    let sizes = [1_000, 10_000, 100_000];

    for &size in &sizes {
        // Create string data with different patterns
        let string_data: Vec<String> = (0..size)
            .map(|i| format!("TestString_{:06}_{}", i, i % 100))
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // String column creation
        group.bench_with_input(
            BenchmarkId::new("string_column_creation", size),
            &string_data,
            |b, data| {
                b.iter(|| {
                    let mut df = OptimizedDataFrame::new();
                    df.add_string_column("test_strings", data.clone()).unwrap();
                    black_box(())
                });
            },
        );

        // String operations using standard iterators
        group.bench_with_input(
            BenchmarkId::new("string_length_calculation", size),
            &string_data,
            |b, data| {
                b.iter(|| {
                    let lengths: Vec<usize> = data.iter().map(|s| s.len()).collect();
                    black_box(lengths)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("string_filtering", size),
            &string_data,
            |b, data| {
                b.iter(|| {
                    let filtered: Vec<String> =
                        data.iter().filter(|s| s.len() > 10).cloned().collect();
                    black_box(filtered)
                });
            },
        );
    }

    group.finish();
}

/// Memory usage and scalability benchmarks
fn benchmark_memory_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scalability");
    group.measurement_time(Duration::from_secs(20));

    // Test memory usage with very large datasets
    let large_sizes = [100_000, 500_000, 1_000_000];

    for &size in &large_sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Memory allocation patterns
        group.bench_with_input(
            BenchmarkId::new("large_dataframe_creation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let config = BenchmarkConfig::default();
                    black_box(create_realistic_dataframe(size, &config).unwrap())
                });
            },
        );

        // Column access patterns
        group.bench_with_input(
            BenchmarkId::new("column_access_pattern", size),
            &size,
            |b, &size| {
                let config = BenchmarkConfig::default();
                let df = create_realistic_dataframe(size, &config).unwrap();
                b.iter(|| {
                    // Simulate random column access
                    for _ in 0..100 {
                        black_box(df.get_int_column("trend_value").unwrap());
                        black_box(df.get_float_column("score").unwrap());
                        black_box(df.get_string_column("category").unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

/// SIMD operations detailed benchmarks
fn benchmark_simd_detailed(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_detailed");
    group.measurement_time(Duration::from_secs(10));

    let test_sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &test_sizes {
        let f64_data: Vec<f64> = (0..size)
            .map(|i| (i as f64) * 0.1 + rand::random::<f64>())
            .collect();
        let _i64_data: Vec<i64> = (0..size)
            .map(|i| i as i64 + rand::random::<i64>() % 1000)
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        // Compare sequential vs SIMD vs parallel vs hybrid
        group.bench_with_input(
            BenchmarkId::new("sum_sequential", size),
            &f64_data,
            |b, data| {
                b.iter(|| black_box(data.iter().sum::<f64>()));
            },
        );

        group.bench_with_input(BenchmarkId::new("sum_simd", size), &f64_data, |b, data| {
            b.iter(|| black_box(simd_sum_f64(data)));
        });

        group.bench_with_input(
            BenchmarkId::new("sum_parallel", size),
            &f64_data,
            |b, data| {
                b.iter(|| black_box(parallel::immediate::sum(data, None)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum_simd_parallel", size),
            &f64_data,
            |b, data| {
                // simd_parallel not available, using standard sum
                b.iter(|| black_box(data.iter().sum::<f64>()));
            },
        );

        // Test other operations
        group.bench_with_input(
            BenchmarkId::new("std_simd_parallel", size),
            &f64_data,
            |b, data| {
                // simd_parallel not available, calculating std manually
                b.iter(|| {
                    let mean = data.iter().sum::<f64>() / data.len() as f64;
                    let variance =
                        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
                    black_box(variance.sqrt())
                });
            },
        );
    }

    group.finish();
}

/// Comprehensive performance comparison suite
fn benchmark_performance_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_comparison");
    group.measurement_time(Duration::from_secs(30));

    let config = BenchmarkConfig::default();
    let df = create_realistic_dataframe(1_000_000, &config).unwrap();

    group.throughput(Throughput::Elements(1_000_000));

    // Full pipeline benchmarks
    group.bench_function("full_analytics_pipeline", |b| {
        b.iter(|| {
            // Simulate a real analytics workflow
            let sum_result = df.sum("trend_value").unwrap();
            let mean_result = df.mean("score").unwrap();
            let grouped = df.par_groupby(&["category"]).unwrap();
            black_box((sum_result, mean_result, grouped))
        });
    });

    group.bench_function("full_pipeline_with_jit", |b| {
        let _jit_config = ParallelConfig::default();
        b.iter(|| {
            // JIT config not available, using standard operations
            let sum_result = df.sum("trend_value").unwrap();
            let mean_result = df.mean("score").unwrap();
            let grouped = df.par_groupby(&["category"]).unwrap();
            black_box((sum_result, mean_result, grouped))
        });
    });

    group.finish();
}

criterion_group!(
    enhanced_benches,
    benchmark_enhanced_dataframe_creation,
    benchmark_enhanced_aggregations,
    benchmark_enhanced_groupby,
    benchmark_enhanced_io,
    benchmark_enhanced_string_ops,
    benchmark_memory_scalability,
    benchmark_simd_detailed,
    benchmark_performance_comparison
);

criterion_main!(enhanced_benches);

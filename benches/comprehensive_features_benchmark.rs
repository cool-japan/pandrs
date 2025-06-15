use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::*;
use std::collections::HashMap;

/// Comprehensive benchmarking suite for PandRS Alpha.4
///
/// This benchmark validates all performance claims made in the alpha.4 documentation
/// and provides detailed performance metrics across all major features.
// Helper function to create test data
fn create_test_dataframe(size: usize) -> DataFrame {
    let mut df = DataFrame::new();

    // Create diverse test data
    let ids: Vec<i32> = (0..size).map(|i| i as i32).collect();
    let names: Vec<String> = (0..size).map(|i| format!("Name_{}", i % 100)).collect(); // High duplication
    let categories: Vec<String> = (0..size).map(|i| format!("Category_{}", i % 10)).collect(); // Very high duplication
    let values: Vec<f64> = (0..size).map(|i| (i as f64) * 1.5).collect();

    df.add_column(
        "id".to_string(),
        pandrs::series::Series::new(ids, Some("id".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "name".to_string(),
        pandrs::series::Series::new(names, Some("name".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "category".to_string(),
        pandrs::series::Series::new(categories, Some("category".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "value".to_string(),
        pandrs::series::Series::new(values, Some("value".to_string())).unwrap(),
    )
    .unwrap();

    df
}

fn create_optimized_dataframe(size: usize) -> OptimizedDataFrame {
    let mut df = OptimizedDataFrame::new();

    let ids: Vec<i64> = (0..size).map(|i| i as i64).collect();
    let names: Vec<String> = (0..size).map(|i| format!("Name_{}", i % 100)).collect();
    let categories: Vec<String> = (0..size).map(|i| format!("Category_{}", i % 10)).collect();
    let values: Vec<f64> = (0..size).map(|i| (i as f64) * 1.5).collect();
    let flags: Vec<bool> = (0..size).map(|i| i % 2 == 0).collect();

    df.add_column("id".to_string(), Column::Int64(Int64Column::new(ids)))
        .unwrap();
    df.add_column("name".to_string(), Column::String(StringColumn::new(names)))
        .unwrap();
    df.add_column(
        "category".to_string(),
        Column::String(StringColumn::new(categories)),
    )
    .unwrap();
    df.add_column(
        "value".to_string(),
        Column::Float64(Float64Column::new(values)),
    )
    .unwrap();
    df.add_column(
        "flag".to_string(),
        Column::Boolean(BooleanColumn::new(flags)),
    )
    .unwrap();

    df
}

/// Benchmark 1: DataFrame Creation Performance
fn bench_dataframe_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("DataFrame Creation");

    for size in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::new("Traditional", size), size, |b, &size| {
            b.iter(|| black_box(create_test_dataframe(size)))
        });

        group.bench_with_input(BenchmarkId::new("Optimized", size), size, |b, &size| {
            b.iter(|| black_box(create_optimized_dataframe(size)))
        });
    }

    group.finish();
}

/// Benchmark 2: Alpha.4 Column Management Operations
fn bench_column_management(c: &mut Criterion) {
    let mut group = c.benchmark_group("Alpha.4 Column Management");

    for size in [1_000, 10_000, 100_000].iter() {
        // Test rename_columns performance
        group.bench_with_input(
            BenchmarkId::new("rename_columns", size),
            size,
            |b, &size| {
                let mut df = create_test_dataframe(size);
                let mut rename_map = HashMap::new();
                rename_map.insert("name".to_string(), "employee_name".to_string());
                rename_map.insert("category".to_string(), "dept".to_string());

                b.iter(|| {
                    df.rename_columns(&rename_map).unwrap();
                    black_box(())
                })
            },
        );

        // Test set_column_names performance
        group.bench_with_input(
            BenchmarkId::new("set_column_names", size),
            size,
            |b, &size| {
                let mut df = create_test_dataframe(size);
                let new_names = vec![
                    "col1".to_string(),
                    "col2".to_string(),
                    "col3".to_string(),
                    "col4".to_string(),
                ];

                b.iter(|| {
                    df.set_column_names(new_names.clone()).unwrap();
                    black_box(())
                })
            },
        );

        // Test OptimizedDataFrame column operations
        group.bench_with_input(
            BenchmarkId::new("optimized_rename_columns", size),
            size,
            |b, &size| {
                let mut df = create_optimized_dataframe(size);
                let mut rename_map = HashMap::new();
                rename_map.insert("name".to_string(), "employee_name".to_string());
                rename_map.insert("category".to_string(), "dept".to_string());

                b.iter(|| {
                    df.rename_columns(&rename_map).unwrap();
                    black_box(())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 3: String Pool Optimization
fn bench_string_pool_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("String Pool Optimization");

    // Test high duplication scenario (1% unique strings)
    for size in [10_000, 100_000, 1_000_000].iter() {
        let unique_count = size / 100; // 1% unique

        group.bench_with_input(BenchmarkId::new("without_pool", size), size, |b, &size| {
            b.iter(|| {
                let data: Vec<String> = (0..size)
                    .map(|i| format!("String_{}", i % unique_count))
                    .collect();
                black_box(data)
            })
        });

        group.bench_with_input(BenchmarkId::new("with_pool", size), size, |b, &size| {
            b.iter(|| {
                let mut df = OptimizedDataFrame::new();
                let data: Vec<String> = (0..size)
                    .map(|i| format!("String_{}", i % unique_count))
                    .collect();
                df.add_column(
                    "strings".to_string(),
                    Column::String(StringColumn::new(data)),
                )
                .unwrap();
                black_box(df)
            })
        });
    }

    group.finish();
}

/// Benchmark 4: Enhanced I/O Operations
#[cfg(feature = "parquet")]
fn bench_enhanced_io(c: &mut Criterion) {
    use pandrs::io::parquet::{read_parquet, write_parquet, ParquetCompression};
    use tempfile::NamedTempFile;

    let mut group = c.benchmark_group("Enhanced I/O Operations");

    for size in [1_000, 10_000, 50_000].iter() {
        let df = create_optimized_dataframe(*size);

        // Benchmark Parquet writing with different compression
        group.bench_with_input(
            BenchmarkId::new("parquet_write_snappy", size),
            &df,
            |b, df| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    black_box(
                        write_parquet(df, temp_file.path(), Some(ParquetCompression::Snappy))
                            .unwrap(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parquet_write_gzip", size),
            &df,
            |b, df| {
                b.iter(|| {
                    let temp_file = NamedTempFile::new().unwrap();
                    black_box(
                        write_parquet(df, temp_file.path(), Some(ParquetCompression::Gzip))
                            .unwrap(),
                    )
                })
            },
        );

        // Benchmark Parquet reading
        let temp_file = NamedTempFile::new().unwrap();
        write_parquet(&df, temp_file.path(), Some(ParquetCompression::Snappy)).unwrap();

        group.bench_with_input(
            BenchmarkId::new("parquet_read", size),
            temp_file.path(),
            |b, path| b.iter(|| black_box(read_parquet(path).unwrap())),
        );
    }

    group.finish();
}

/// Benchmark 5: Distributed Processing Performance
#[cfg(feature = "distributed")]
fn bench_distributed_processing(c: &mut Criterion) {
    use pandrs::distributed::DistributedContext;

    let mut group = c.benchmark_group("Distributed Processing");

    for concurrency in [1, 2, 4, 8].iter() {
        for size in [10_000, 100_000].iter() {
            let df = create_test_dataframe(*size);

            group.bench_with_input(
                BenchmarkId::new(format!("sql_aggregation_{}threads", concurrency), size),
                &(df, *concurrency),
                |b, (df, concurrency)| {
                    b.iter(|| {
                        let mut context = DistributedContext::new_local(*concurrency).unwrap();
                        context.register_dataframe("test_data", df).unwrap();
                        let result = context.sql("SELECT category, AVG(CAST(value AS FLOAT)) as avg_value FROM test_data GROUP BY category").unwrap();
                        black_box(result)
                    })
                }
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("dataframe_aggregation_{}threads", concurrency),
                    size,
                ),
                &(df, *concurrency),
                |b, (df, concurrency)| {
                    b.iter(|| {
                        let mut context = DistributedContext::new_local(*concurrency).unwrap();
                        context.register_dataframe("test_data", df).unwrap();
                        let test_df = context.dataset("test_data").unwrap();
                        let result = test_df
                            .aggregate(&["category"], &[("value", "avg", "avg_value")])
                            .unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark 6: Series Operations Performance
fn bench_series_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Series Operations");

    for size in [1_000, 10_000, 100_000].iter() {
        let data: Vec<i32> = (0..*size).collect();

        // Benchmark series creation
        group.bench_with_input(
            BenchmarkId::new("series_creation", size),
            &data,
            |b, data| {
                b.iter(|| {
                    black_box(
                        pandrs::series::Series::new(data.clone(), Some("test".to_string()))
                            .unwrap(),
                    )
                })
            },
        );

        // Benchmark series name operations (Alpha.4 feature)
        group.bench_with_input(
            BenchmarkId::new("series_name_operations", size),
            &data,
            |b, data| {
                let mut series = pandrs::series::Series::new(data.clone(), None).unwrap();
                b.iter(|| {
                    series.set_name("test_name".to_string());
                    black_box(series.name().cloned())
                })
            },
        );

        // Benchmark series with_name (Alpha.4 fluent interface)
        group.bench_with_input(
            BenchmarkId::new("series_with_name", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let series = pandrs::series::Series::new(data.clone(), None)
                        .unwrap()
                        .with_name("fluent_name".to_string());
                    black_box(series)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 7: Memory Usage Comparison
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Usage");

    for size in [10_000, 100_000].iter() {
        // Traditional approach memory usage
        group.bench_with_input(
            BenchmarkId::new("traditional_memory", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut data = Vec::with_capacity(size);
                    for i in 0..size {
                        data.push(format!("Category_{}", i % 10)); // High duplication
                    }
                    black_box(data)
                })
            },
        );

        // Optimized approach with string pool
        group.bench_with_input(
            BenchmarkId::new("optimized_memory", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut df = OptimizedDataFrame::new();
                    let data: Vec<String> =
                        (0..size).map(|i| format!("Category_{}", i % 10)).collect();
                    df.add_column(
                        "category".to_string(),
                        Column::String(StringColumn::new(data)),
                    )
                    .unwrap();
                    black_box(df)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 8: Aggregation Performance
fn bench_aggregation_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Aggregation Performance");

    for size in [1_000, 10_000, 100_000].iter() {
        let df = create_test_dataframe(*size);

        // Traditional groupby operations
        group.bench_with_input(
            BenchmarkId::new("traditional_groupby", size),
            &df,
            |b, df| {
                b.iter(|| {
                    // Simulate traditional groupby aggregation
                    let categories = df.get_column_string_values("category").unwrap();
                    let values = df.get_column_string_values("value").unwrap();

                    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
                    for (cat, val) in categories.iter().zip(values.iter()) {
                        let val_f64: f64 = val.parse().unwrap_or(0.0);
                        groups.entry(cat.clone()).or_default().push(val_f64);
                    }

                    let mut results = HashMap::new();
                    for (cat, vals) in groups {
                        let sum: f64 = vals.iter().sum();
                        results.insert(cat, sum);
                    }

                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark 9: Type Conversion Performance
fn bench_type_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Type Conversion");

    for size in [1_000, 10_000, 100_000].iter() {
        let int_data: Vec<i32> = (0..*size).collect();
        let series = pandrs::series::Series::new(int_data, Some("test".to_string())).unwrap();

        // Alpha.4 enhanced type conversion
        group.bench_with_input(
            BenchmarkId::new("to_string_series", size),
            &series,
            |b, series: &pandrs::series::Series<i32>| {
                b.iter(|| black_box(series.to_string_series().unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark 10: Error Handling Performance
fn bench_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Error Handling");

    let mut df = create_test_dataframe(1000);

    // Test error handling performance for invalid operations
    group.bench_function("invalid_column_rename", |b| {
        b.iter(|| {
            let mut rename_map = HashMap::new();
            rename_map.insert("nonexistent".to_string(), "new_name".to_string());
            let result = df.rename_columns(&rename_map);
            black_box(result.is_err())
        })
    });

    group.bench_function("invalid_column_names", |b| {
        b.iter(|| {
            let result = df.set_column_names(vec![
                "too".to_string(),
                "many".to_string(),
                "names".to_string(),
                "here".to_string(),
                "extra".to_string(),
            ]);
            black_box(result.is_err())
        })
    });

    group.finish();
}

// Compile all benchmarks
criterion_group!(
    benches,
    bench_dataframe_creation,
    bench_column_management,
    bench_string_pool_optimization,
    bench_series_operations,
    bench_memory_usage,
    bench_aggregation_performance,
    bench_type_conversion,
    bench_error_handling
);

// Conditional benchmarks for optional features
#[cfg(feature = "parquet")]
criterion_group!(parquet_benches, bench_enhanced_io);

#[cfg(feature = "distributed")]
criterion_group!(distributed_benches, bench_distributed_processing);

// Main function to run all benchmarks
criterion_main!(benches);

//! Performance benchmarks comparing PandRS to pandas-equivalent operations
//!
//! This benchmark suite compares PandRS performance to typical pandas operations
//! by implementing equivalent functionality and measuring execution time, memory usage,
//! and throughput characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pandrs::core::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;
use std::time::{Duration, Instant};

/// Create test data equivalent to pandas DataFrame creation
fn create_pandas_equivalent_data(size: usize) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Equivalent to: pd.DataFrame({
    //     'id': range(size),
    //     'category': ['A', 'B', 'C', 'D'] * (size//4 + 1),
    //     'value': np.random.normal(100, 15, size),
    //     'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H')
    // })

    let categories = ["A", "B", "C", "D"];
    let mut id_data = Vec::with_capacity(size);
    let mut category_data = Vec::with_capacity(size);
    let mut value_data = Vec::with_capacity(size);

    for i in 0..size {
        id_data.push(format!("{}", i));
        category_data.push(categories[i % categories.len()].to_string());
        value_data.push(100.0 + (i as f64 * 0.15) + ((i % 100) as f64 - 50.0) * 0.3);
    }

    df.add_string_column("id", id_data)?;
    df.add_string_column("category", category_data)?;
    df.add_float_column("value", value_data)?;

    Ok(df)
}

fn create_optimized_pandas_data(size: usize) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    let categories = ["A", "B", "C", "D"];
    let mut id_data = Vec::with_capacity(size);
    let mut category_data = Vec::with_capacity(size);
    let mut value_data = Vec::with_capacity(size);

    for i in 0..size {
        id_data.push(i.to_string());
        category_data.push(categories[i % categories.len()].to_string());
        value_data.push(100.0 + (i as f64 * 0.15) + ((i % 100) as f64 - 50.0) * 0.3);
    }

    df.add_string_column("id", id_data)?;
    df.add_string_column("category", category_data)?;
    df.add_float_column("value", value_data)?;

    Ok(df)
}

/// Benchmark DataFrame creation (equivalent to pd.DataFrame construction)
fn benchmark_dataframe_creation_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_creation");
    group.throughput(Throughput::Elements(100_000));

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("pandrs_standard", size),
            size,
            |b, &size| {
                b.iter(|| black_box(create_pandas_equivalent_data(size).unwrap()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pandrs_optimized", size),
            size,
            |b, &size| {
                b.iter(|| black_box(create_optimized_pandas_data(size).unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark basic operations (equivalent to pandas column operations)
fn benchmark_basic_operations_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_basic_ops");
    group.measurement_time(Duration::from_secs(10));

    let sizes = [10_000, 100_000, 500_000];

    for size in sizes.iter() {
        let df_standard = create_pandas_equivalent_data(*size).unwrap();
        let df_optimized = create_optimized_pandas_data(*size).unwrap();

        // Column access (equivalent to df['column'])
        group.bench_with_input(
            BenchmarkId::new("column_access_standard", size),
            &df_standard,
            |b, df| {
                b.iter(|| {
                    black_box(df.get_float_column("value").unwrap());
                    black_box(df.get_string_column("category").unwrap());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("column_access_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| {
                    black_box(df.get_float_column("value").unwrap());
                    black_box(df.get_string_column("category").unwrap());
                });
            },
        );

        // Row count (equivalent to len(df))
        group.bench_with_input(
            BenchmarkId::new("row_count_standard", size),
            &df_standard,
            |b, df| {
                b.iter(|| black_box(df.row_count()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("row_count_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| black_box(df.row_count()));
            },
        );
    }

    group.finish();
}

/// Benchmark aggregation operations (equivalent to df.groupby().agg())
fn benchmark_aggregation_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_aggregation");
    group.measurement_time(Duration::from_secs(15));

    for size in [50_000, 200_000, 500_000].iter() {
        let df_standard = create_pandas_equivalent_data(*size).unwrap();
        let df_optimized = create_optimized_pandas_data(*size).unwrap();

        // Sum operation (equivalent to df['value'].sum())
        group.bench_with_input(
            BenchmarkId::new("sum_standard", size),
            &df_standard,
            |b, df| {
                b.iter(|| {
                    let value_col = df.get_float_column("value").unwrap();
                    let mut sum = 0.0;
                    for i in 0..value_col.len() {
                        if let Some(Some(val)) = value_col.get(i) {
                            sum += val;
                        }
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| black_box(df.sum("value").unwrap()));
            },
        );

        // Mean operation (equivalent to df['value'].mean())
        group.bench_with_input(
            BenchmarkId::new("mean_standard", size),
            &df_standard,
            |b, df| {
                b.iter(|| {
                    let value_col = df.get_float_column("value").unwrap();
                    let mut sum = 0.0;
                    let mut count = 0;
                    for i in 0..value_col.len() {
                        if let Some(Some(val)) = value_col.get(i) {
                            sum += val;
                            count += 1;
                        }
                    }
                    black_box(if count > 0 { sum / count as f64 } else { 0.0 });
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| black_box(df.mean("value").unwrap()));
            },
        );
    }

    group.finish();
}

/// Benchmark groupby operations (equivalent to df.groupby('category').agg())
fn benchmark_groupby_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_groupby");
    group.measurement_time(Duration::from_secs(20));

    for size in [10_000, 100_000, 500_000].iter() {
        let df_optimized = create_optimized_pandas_data(*size).unwrap();

        // GroupBy with sum (equivalent to df.groupby('category')['value'].sum())
        group.bench_with_input(
            BenchmarkId::new("groupby_sum_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| {
                    let grouped = df.groupby(&["category"]).unwrap();
                    black_box(grouped.sum("value").unwrap());
                });
            },
        );

        // GroupBy with mean (equivalent to df.groupby('category')['value'].mean())
        group.bench_with_input(
            BenchmarkId::new("groupby_mean_optimized", size),
            &df_optimized,
            |b, df| {
                b.iter(|| {
                    let grouped = df.groupby(&["category"]).unwrap();
                    black_box(grouped.mean("value").unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory operations (equivalent to pandas memory usage)
fn benchmark_memory_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_memory");
    group.measurement_time(Duration::from_secs(15));

    for size in [100_000, 500_000, 1_000_000].iter() {
        // DataFrame cloning (equivalent to df.copy())
        group.bench_with_input(
            BenchmarkId::new("clone_standard", size),
            size,
            |b, &size| {
                let df = create_pandas_equivalent_data(size).unwrap();
                b.iter(|| black_box(df.clone()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("clone_optimized", size),
            size,
            |b, &size| {
                let df = create_optimized_pandas_data(size).unwrap();
                b.iter(|| black_box(df.clone()));
            },
        );

        // Memory efficient column iteration
        group.bench_with_input(
            BenchmarkId::new("iteration_standard", size),
            size,
            |b, &size| {
                let df = create_pandas_equivalent_data(size).unwrap();
                b.iter(|| {
                    let value_col = df.get_float_column("value").unwrap();
                    let mut count = 0;
                    for i in 0..value_col.len() {
                        if value_col.get(i).is_some() {
                            count += 1;
                        }
                    }
                    black_box(count);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark I/O operations simulation (equivalent to pandas read/write operations)
fn benchmark_io_vs_pandas(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_io");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 50_000, 200_000].iter() {
        // CSV-like data creation (equivalent to df.to_csv() preparation)
        group.bench_with_input(
            BenchmarkId::new("csv_preparation", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let df = create_pandas_equivalent_data(size).unwrap();
                    // Simulate CSV preparation
                    let mut csv_data = Vec::new();
                    for i in 0..df.row_count() {
                        let id_col = df.get_string_column("id").unwrap();
                        let cat_col = df.get_string_column("category").unwrap();
                        let val_col = df.get_float_column("value").unwrap();

                        if let (Some(Some(id)), Some(Some(cat)), Some(Some(val))) =
                            (id_col.get(i), cat_col.get(i), val_col.get(i))
                        {
                            csv_data.push(format!("{},{},{}", id, cat, val));
                        }
                    }
                    black_box(csv_data);
                });
            },
        );
    }

    group.finish();
}

/// Performance comparison summary and analysis
fn benchmark_performance_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("pandas_comparison_analysis");

    // Comprehensive test with multiple operations
    let size = 100_000;
    let df_standard = create_pandas_equivalent_data(size).unwrap();
    let df_optimized = create_optimized_pandas_data(size).unwrap();

    group.bench_function("comprehensive_pandas_workflow_standard", |b| {
        b.iter(|| {
            // Equivalent to pandas workflow:
            // df = pd.DataFrame(data)
            // result = df.groupby('category')['value'].agg(['sum', 'mean', 'count'])
            // filtered = df[df['value'] > df['value'].mean()]

            let start = Instant::now();

            // 1. Get column access
            let value_col = df_standard.get_float_column("value").unwrap();
            let category_col = df_standard.get_string_column("category").unwrap();

            // 2. Calculate mean
            let mut sum = 0.0;
            let mut count = 0;
            for i in 0..value_col.len() {
                if let Some(Some(val)) = value_col.get(i) {
                    sum += val;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };

            // 3. Filter operation simulation
            let mut filtered_count = 0;
            for i in 0..value_col.len() {
                if let Some(Some(val)) = value_col.get(i) {
                    if *val > mean {
                        filtered_count += 1;
                    }
                }
            }

            let duration = start.elapsed();
            black_box((sum, mean, filtered_count, duration));
        });
    });

    group.bench_function("comprehensive_pandas_workflow_optimized", |b| {
        b.iter(|| {
            let start = Instant::now();

            // 1. Optimized operations
            let sum = df_optimized.sum("value").unwrap();
            let mean = df_optimized.mean("value").unwrap();

            // 2. GroupBy operation
            let grouped = df_optimized.groupby(&["category"]).unwrap();
            let group_sum = grouped.sum("value").unwrap();

            let duration = start.elapsed();
            black_box((sum, mean, group_sum, duration));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_dataframe_creation_vs_pandas,
    benchmark_basic_operations_vs_pandas,
    benchmark_aggregation_vs_pandas,
    benchmark_groupby_vs_pandas,
    benchmark_memory_vs_pandas,
    benchmark_io_vs_pandas,
    benchmark_performance_analysis
);

criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::error::Result;
use pandrs::optimized::jit::*;
use pandrs::optimized::OptimizedDataFrame;
use std::time::Duration;

fn create_test_dataframe(size: usize) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    // Create test data
    let categories = ["A", "B", "C", "D", "E"];
    let mut cat_data = Vec::with_capacity(size);
    let mut int_data = Vec::with_capacity(size);
    let mut float_data = Vec::with_capacity(size);

    for i in 0..size {
        cat_data.push(categories[i % categories.len()].to_string());
        int_data.push((i % 1000) as i64);
        float_data.push((i as f64) * 0.1 + (i % 100) as f64);
    }

    df.add_string_column("category", cat_data)?;
    df.add_int_column("value", int_data)?;
    df.add_float_column("score", float_data)?;

    Ok(df)
}

fn benchmark_dataframe_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_creation");

    for size in [1_000, 10_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_dataframe", size),
            size,
            |b, &size| {
                b.iter(|| black_box(create_test_dataframe(size).unwrap()));
            },
        );
    }

    group.finish();
}

fn benchmark_groupby_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("groupby_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 50_000, 100_000].iter() {
        let df = create_test_dataframe(*size).unwrap();

        group.bench_with_input(BenchmarkId::new("par_groupby", size), &df, |b, df| {
            b.iter(|| black_box(df.par_groupby(&["category"]).unwrap()));
        });
    }

    group.finish();
}

fn benchmark_aggregation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation_operations");
    group.measurement_time(Duration::from_secs(10));

    let df = create_test_dataframe(100_000).unwrap();

    group.bench_function("sum_standard", |b| {
        b.iter(|| black_box(df.sum("value").unwrap()));
    });

    group.bench_function("sum_with_jit_config", |b| {
        let config = ParallelConfig::default();
        b.iter(|| black_box(df.sum_with_config("value", Some(config.clone())).unwrap()));
    });

    group.bench_function("mean_standard", |b| {
        b.iter(|| black_box(df.mean("value").unwrap()));
    });

    group.bench_function("mean_with_jit_config", |b| {
        let config = ParallelConfig::default();
        b.iter(|| black_box(df.mean_with_config("value", Some(config.clone())).unwrap()));
    });

    group.bench_function("max_standard", |b| {
        b.iter(|| black_box(df.max("value").unwrap()));
    });

    group.bench_function("min_standard", |b| {
        b.iter(|| black_box(df.min("value").unwrap()));
    });

    group.finish();
}

fn benchmark_jit_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_operations");
    group.measurement_time(Duration::from_secs(10));

    let test_data: Vec<f64> = (0..100_000).map(|i| i as f64 * 0.1).collect();

    // Parallel operations
    group.bench_function("parallel_sum_f64", |b| {
        b.iter(|| {
            let sum_func = parallel_sum_f64(None);
            black_box(sum_func.execute(&test_data))
        });
    });

    group.bench_function("parallel_mean_f64", |b| {
        b.iter(|| black_box(parallel_mean_f64_value(&test_data, None)));
    });

    group.bench_function("parallel_min_f64", |b| {
        b.iter(|| {
            let min_func = parallel_min_f64(None);
            black_box(min_func.execute(&test_data))
        });
    });

    group.bench_function("parallel_max_f64", |b| {
        b.iter(|| {
            let max_func = parallel_max_f64(None);
            black_box(max_func.execute(&test_data))
        });
    });

    // SIMD operations
    group.bench_function("simd_sum_f64", |b| {
        b.iter(|| black_box(simd_sum_f64(&test_data)));
    });

    group.bench_function("simd_mean_f64", |b| {
        b.iter(|| black_box(simd_mean_f64(&test_data)));
    });

    group.bench_function("simd_min_f64", |b| {
        b.iter(|| black_box(simd_min_f64(&test_data)));
    });

    group.bench_function("simd_max_f64", |b| {
        b.iter(|| black_box(simd_max_f64(&test_data)));
    });

    group.finish();
}

fn benchmark_string_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    let string_data: Vec<String> = (0..10_000)
        .map(|i| format!("test_string_{}", i % 100))
        .collect();

    group.bench_function("string_column_creation", |b| {
        b.iter(|| {
            let mut df = OptimizedDataFrame::new();
            black_box(df.add_string_column("test", string_data.clone()).unwrap());
        });
    });

    group.finish();
}

fn benchmark_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    group.measurement_time(Duration::from_secs(15));

    for size in [50_000, 100_000, 200_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("dataframe_clone", size),
            size,
            |b, &size| {
                let df = create_test_dataframe(size).unwrap();
                b.iter(|| black_box(df.clone()));
            },
        );

        group.bench_with_input(BenchmarkId::new("column_access", size), size, |b, &size| {
            let df = create_test_dataframe(size).unwrap();
            b.iter(|| {
                black_box(df.get_int_column("value").unwrap());
                black_box(df.get_float_column("score").unwrap());
                black_box(df.get_string_column("category").unwrap());
            });
        });
    }

    group.finish();
}

fn benchmark_filter_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_operations");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 50_000, 100_000].iter() {
        let df = create_test_dataframe(*size).unwrap();

        group.bench_with_input(BenchmarkId::new("filter_by_value", size), &df, |b, df| {
            b.iter(|| {
                // Filter rows where value > 500
                let int_col = df.get_int_column("value").unwrap();
                let mask: Vec<bool> = (0..df.row_count())
                    .map(|i| int_col.get(i).unwrap_or(None).unwrap_or(0) > 500)
                    .collect();
                black_box(mask);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_dataframe_creation,
    benchmark_groupby_operations,
    benchmark_aggregation_operations,
    benchmark_jit_operations,
    benchmark_string_operations,
    benchmark_memory_operations,
    benchmark_filter_operations
);

criterion_main!(benches);

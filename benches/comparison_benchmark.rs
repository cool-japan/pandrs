//! Comprehensive benchmarks matching pandas and polars tests for direct comparison

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::join::JoinExt;
use pandrs::dataframe::pandas_compat::PandasCompatExt;
// Only the Parquet helpers come from the extension trait; CSV has inherent methods.
#[cfg(feature = "parquet")]
use pandrs::dataframe::serialize::SerializeExt;
use pandrs::series::Series;
use std::hint::black_box as bb;
use std::time::Duration;

const DATA_DIR: &str = "/tmp/benchmark_data";

// ===================
// I/O BENCHMARKS
// ===================

fn benchmark_csv_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_csv_read");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let df = DataFrame::from_csv(&csv_path, true).unwrap();
                bb(df)
            });
        });
    }

    group.finish();
}

fn benchmark_csv_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_csv_write");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();
        let output_path = format!("{}/pandrs_output_{}.csv", DATA_DIR, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                df.to_csv(&output_path).unwrap();
                bb(())
            });
        });
    }

    group.finish();
}

#[cfg(feature = "parquet")]
fn benchmark_parquet_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_parquet_read");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 100_000].iter() {
        let parquet_path = format!("{}/data_{}.parquet", DATA_DIR, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let df = DataFrame::from_parquet(&parquet_path).unwrap();
                bb(df)
            });
        });
    }

    group.finish();
}

#[cfg(feature = "parquet")]
fn benchmark_parquet_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("io_parquet_write");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();
        let output_path = format!("{}/pandrs_output_{}.parquet", DATA_DIR, size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                df.to_parquet(&output_path).unwrap();
                bb(())
            });
        });
    }

    group.finish();
}

// ===================
// DATAFRAME OPERATIONS
// ===================

fn benchmark_creation_from_vecs(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_creation");
    group.measurement_time(Duration::from_secs(10));

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut df = DataFrame::new();

                let ids: Vec<String> = (0..size).map(|i| i.to_string()).collect();
                let categories: Vec<String> = (0..size)
                    .map(|i| {
                        let cats = ["A", "B", "C", "D"];
                        cats[i % cats.len()].to_string()
                    })
                    .collect();
                let values: Vec<f64> = (0..size).map(|i| 100.0 + (i as f64 * 0.15)).collect();

                df.add_column(
                    "id".to_string(),
                    Series::new(ids, Some("id".to_string())).unwrap(),
                )
                .unwrap();
                df.add_column(
                    "category".to_string(),
                    Series::new(categories, Some("category".to_string())).unwrap(),
                )
                .unwrap();
                df.add_column(
                    "value".to_string(),
                    Series::new(values, Some("value".to_string())).unwrap(),
                )
                .unwrap();

                bb(df)
            });
        });
    }

    group.finish();
}

fn benchmark_column_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_column_selection");

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let _id_col = df.get_column::<i64>("id").unwrap();
                let _value_col = df.get_column::<f64>("value").unwrap();
                bb(())
            });
        });
    }

    group.finish();
}

fn benchmark_row_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_row_filtering");

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        // Calculate mean for filtering
        let value_col = df.get_column::<f64>("value").unwrap();
        let sum: f64 = (0..value_col.len()).filter_map(|i| value_col.get(i)).sum();
        let mean = sum / value_col.len() as f64;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let value_col = df.get_column::<f64>("value").unwrap();
                let mask: Vec<bool> = (0..value_col.len())
                    .map(|i| value_col.get(i).map(|v| *v > mean).unwrap_or(false))
                    .collect();
                bb(mask)
            });
        });
    }

    group.finish();
}

fn benchmark_sorting(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_sorting");
    group.measurement_time(Duration::from_secs(15));

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let sorted = df.sort_by_columns(&["value"], &[false]).unwrap();
                bb(sorted)
            });
        });
    }

    group.finish();
}

// ===================
// AGGREGATION BENCHMARKS
// ===================

fn benchmark_aggregations(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregations");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        // Sum
        group.bench_with_input(BenchmarkId::new("sum", size), size, |b, _| {
            b.iter(|| {
                let value_col = df.get_column::<f64>("value").unwrap();
                let sum: f64 = (0..value_col.len()).filter_map(|i| value_col.get(i)).sum();
                bb(sum)
            });
        });

        // Mean
        group.bench_with_input(BenchmarkId::new("mean", size), size, |b, _| {
            b.iter(|| {
                let value_col = df.get_column::<f64>("value").unwrap();
                let sum: f64 = (0..value_col.len()).filter_map(|i| value_col.get(i)).sum();
                let mean = sum / value_col.len() as f64;
                bb(mean)
            });
        });

        // Count
        group.bench_with_input(BenchmarkId::new("count", size), size, |b, _| {
            b.iter(|| {
                let count = df.row_count();
                bb(count)
            });
        });
    }

    group.finish();
}

// ===================
// GROUPBY BENCHMARKS
// ===================

fn benchmark_groupby(c: &mut Criterion) {
    let mut group = c.benchmark_group("groupby");
    group.measurement_time(Duration::from_secs(15));

    for size in [10_000, 100_000, 1_000_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        group.bench_with_input(BenchmarkId::new("single_key", size), size, |b, _| {
            b.iter(|| {
                let grouped = df.groupby_pivot("category").unwrap();
                bb(grouped)
            });
        });

        // Note: Standard DataFrame.groupby only supports single column
        // Skipping multi-key groupby for standard DataFrame
    }

    group.finish();
}

// ===================
// JOIN BENCHMARKS
// ===================

fn benchmark_joins(c: &mut Criterion) {
    let mut group = c.benchmark_group("joins");
    group.measurement_time(Duration::from_secs(20));

    for size in [10_000, 100_000].iter() {
        let left_path = format!("{}/join_left_{}.csv", DATA_DIR, size);
        let right_path = format!("{}/join_right_{}.csv", DATA_DIR, size);

        let left_df = DataFrame::from_csv(&left_path, true).unwrap();
        let right_df = DataFrame::from_csv(&right_path, true).unwrap();

        group.bench_with_input(BenchmarkId::new("inner", size), size, |b, _| {
            b.iter(|| {
                let joined = left_df.inner_join(&right_df, "key").unwrap();
                bb(joined)
            });
        });

        group.bench_with_input(BenchmarkId::new("left", size), size, |b, _| {
            b.iter(|| {
                let joined = left_df.left_join(&right_df, "key").unwrap();
                bb(joined)
            });
        });
    }

    group.finish();
}

// ===================
// STRING OPERATIONS
// ===================

fn benchmark_string_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_operations");

    for size in [10_000, 100_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        group.bench_with_input(BenchmarkId::new("upper", size), size, |b, _| {
            b.iter(|| {
                let string_col = df.get_column::<String>("string_data").unwrap();
                let upper: Vec<String> = (0..string_col.len())
                    .filter_map(|i| string_col.get(i).map(|s| s.to_uppercase()))
                    .collect();
                bb(upper)
            });
        });

        group.bench_with_input(BenchmarkId::new("lower", size), size, |b, _| {
            b.iter(|| {
                let string_col = df.get_column::<String>("string_data").unwrap();
                let lower: Vec<String> = (0..string_col.len())
                    .filter_map(|i| string_col.get(i).map(|s| s.to_lowercase()))
                    .collect();
                bb(lower)
            });
        });

        group.bench_with_input(BenchmarkId::new("contains", size), size, |b, _| {
            b.iter(|| {
                let string_col = df.get_column::<String>("string_data").unwrap();
                let matches: Vec<bool> = (0..string_col.len())
                    .map(|i| {
                        string_col
                            .get(i)
                            .map(|s| s.contains("string_5"))
                            .unwrap_or(false)
                    })
                    .collect();
                bb(matches)
            });
        });
    }

    group.finish();
}

// ===================
// STATISTICAL OPERATIONS
// ===================

fn benchmark_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistics");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let csv_path = format!("{}/data_{}.csv", DATA_DIR, size);
        let df = DataFrame::from_csv(&csv_path, true).unwrap();

        group.bench_with_input(BenchmarkId::new("describe", size), size, |b, _| {
            b.iter(|| {
                let stats = df.describe("value").unwrap();
                bb(stats)
            });
        });

        group.bench_with_input(BenchmarkId::new("value_counts", size), size, |b, _| {
            b.iter(|| {
                let value_counts = df.value_counts("category").unwrap();
                bb(value_counts)
            });
        });
    }

    group.finish();
}

// Conditional compilation for benchmarks based on features
#[cfg(feature = "parquet")]
criterion_group!(
    benches,
    benchmark_csv_read,
    benchmark_csv_write,
    benchmark_parquet_read,
    benchmark_parquet_write,
    benchmark_creation_from_vecs,
    benchmark_column_selection,
    benchmark_row_filtering,
    benchmark_sorting,
    benchmark_aggregations,
    benchmark_groupby,
    benchmark_joins,
    benchmark_string_ops,
    benchmark_statistics
);

#[cfg(not(feature = "parquet"))]
criterion_group!(
    benches,
    benchmark_csv_read,
    benchmark_csv_write,
    benchmark_creation_from_vecs,
    benchmark_column_selection,
    benchmark_row_filtering,
    benchmark_sorting,
    benchmark_aggregations,
    benchmark_groupby,
    benchmark_joins,
    benchmark_string_ops,
    benchmark_statistics
);

criterion_main!(benches);

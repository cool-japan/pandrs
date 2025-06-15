// Simple DataFrame benchmarks using Criterion
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pandrs::dataframe::serialize::SerializeExt;
use pandrs::{DataFrame, Series};
use std::collections::HashMap;

// 10-row DataFrame creation benchmark
fn bench_create_small_dataframe(c: &mut Criterion) {
    c.bench_function("create_small_dataframe", |b| {
        b.iter(|| {
            let col_a =
                Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
            let col_b = Series::new(
                vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
                Some("B".to_string()),
            )
            .unwrap();
            let col_c = Series::new(
                ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                Some("C".to_string()),
            )
            .unwrap();

            let mut df = DataFrame::new();
            df.add_column("A".to_string(), col_a).unwrap();
            df.add_column("B".to_string(), col_b).unwrap();
            df.add_column("C".to_string(), col_c).unwrap();
            black_box(df)
        });
    });
}

// 1,000-row DataFrame creation benchmark
fn bench_create_medium_dataframe(c: &mut Criterion) {
    c.bench_function("create_medium_dataframe", |b| {
        b.iter(|| {
            let col_a = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
            let col_b = Series::new(
                (0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
                Some("B".to_string()),
            )
            .unwrap();
            let col_c = Series::new(
                (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
                Some("C".to_string()),
            )
            .unwrap();

            let mut df = DataFrame::new();
            df.add_column("A".to_string(), col_a).unwrap();
            df.add_column("B".to_string(), col_b).unwrap();
            df.add_column("C".to_string(), col_c).unwrap();
            black_box(df)
        });
    });
}

// DataFrame creation using from_map method
fn bench_create_dataframe_from_map(c: &mut Criterion) {
    c.bench_function("create_dataframe_from_map", |b| {
        b.iter(|| {
            let mut data = HashMap::new();
            data.insert("A".to_string(), (0..1000).map(|n| n.to_string()).collect());
            data.insert(
                "B".to_string(),
                (0..1000)
                    .map(|n| format!("{:.1}", n as f64 * 0.5))
                    .collect(),
            );
            data.insert(
                "C".to_string(),
                (0..1000).map(|i| format!("val_{}", i)).collect(),
            );

            let df = DataFrame::from_map(data, None).unwrap();
            black_box(df)
        });
    });
}

// Column access benchmark
fn bench_column_access(c: &mut Criterion) {
    // Preparation
    let col_a = Series::new((0..10_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new(
        (0..10_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
        Some("B".to_string()),
    )
    .unwrap();
    let col_c = Series::new(
        (0..10_000)
            .map(|i| format!("val_{}", i))
            .collect::<Vec<_>>(),
        Some("C".to_string()),
    )
    .unwrap();

    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();

    c.bench_function("column_access", |b| {
        b.iter(|| {
            let _: &Series<i32> = df.get_column("A").unwrap();
            let _: &Series<f64> = df.get_column("B").unwrap();
            let _: &Series<String> = df.get_column("C").unwrap();
        });
    });
}

// CSV serialization (small)
fn bench_to_csv_small(c: &mut Criterion) {
    // Preparation
    let col_a = Series::new((0..100).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new(
        (0..100).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
        Some("B".to_string()),
    )
    .unwrap();
    let col_c = Series::new(
        (0..100).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
        Some("C".to_string()),
    )
    .unwrap();

    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();

    // Temporary file
    let temp_path = std::env::temp_dir().join("bench_test.csv");
    let path_str = temp_path.to_str().unwrap();

    c.bench_function("to_csv_small", |b| {
        b.iter(|| {
            df.to_csv(path_str).unwrap();
            black_box(())
        });
    });
}

// JSON serialization
fn bench_to_json_small(c: &mut Criterion) {
    // Preparation
    let col_a = Series::new((0..100).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
    let col_b = Series::new(
        (0..100).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
        Some("B".to_string()),
    )
    .unwrap();
    let col_c = Series::new(
        (0..100).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
        Some("C".to_string()),
    )
    .unwrap();

    let mut df = DataFrame::new();
    df.add_column("A".to_string(), col_a).unwrap();
    df.add_column("B".to_string(), col_b).unwrap();
    df.add_column("C".to_string(), col_c).unwrap();

    c.bench_function("to_json_small", |b| {
        b.iter(|| {
            let json = df.to_json().unwrap();
            black_box(json)
        });
    });
}

criterion_group!(
    legacy_benches,
    bench_create_small_dataframe,
    bench_create_medium_dataframe,
    bench_create_dataframe_from_map,
    bench_column_access,
    bench_to_csv_small,
    bench_to_json_small
);

criterion_main!(legacy_benches);

//! Query Optimizer Benchmarks
//!
//! Benchmarks for query plan optimization techniques.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::optimized::query_optimizer::{
    AggregateFunc, ColumnStats, FilterOp, OptimizationLevel, QueryOptimizer, QueryPlanBuilder,
};

/// Create a query plan with multiple filters for predicate pushdown testing
fn create_filter_heavy_plan(n_filters: usize) -> pandrs::optimized::query_optimizer::QueryPlan {
    let mut builder = QueryPlanBuilder::new(100_000);

    // Add column statistics
    for i in 0..n_filters {
        let mut stats = ColumnStats::new(format!("col_{}", i), 100_000);
        stats.distinct_count = 1000;
        stats.min_value = Some(0.0);
        stats.max_value = Some(1000.0);
        builder = builder.with_stats(stats);
    }

    // Add select
    let cols: Vec<String> = (0..n_filters).map(|i| format!("col_{}", i)).collect();
    builder = builder.select(cols);

    // Add filters
    for i in 0..n_filters {
        builder = builder.filter(format!("col_{}", i), FilterOp::GreaterThan(500.0));
    }

    // Add aggregation
    builder = builder.aggregate(
        vec!["col_0".to_string()],
        vec![("col_1".to_string(), AggregateFunc::Sum)],
    );

    builder.build()
}

/// Create a complex query plan with mixed operations
fn create_complex_plan() -> pandrs::optimized::query_optimizer::QueryPlan {
    let mut builder = QueryPlanBuilder::new(1_000_000);

    // Add column stats
    for col in ["id", "name", "value", "category", "date"] {
        let mut stats = ColumnStats::new(col.to_string(), 1_000_000);
        stats.distinct_count = if col == "category" { 100 } else { 500_000 };
        stats.min_value = Some(0.0);
        stats.max_value = Some(10000.0);
        builder = builder.with_stats(stats);
    }

    builder
        .select(vec![
            "id".to_string(),
            "name".to_string(),
            "value".to_string(),
            "category".to_string(),
        ])
        .filter("value".to_string(), FilterOp::GreaterThan(100.0))
        .filter("category".to_string(), FilterOp::Equals(5.0))
        .aggregate(
            vec!["category".to_string()],
            vec![
                ("value".to_string(), AggregateFunc::Sum),
                ("id".to_string(), AggregateFunc::Count),
            ],
        )
        .sort(vec!["value".to_string()], vec![false])
        .limit(100)
        .build()
}

fn bench_predicate_pushdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("Predicate Pushdown");

    for n_filters in [2, 5, 10, 20].iter() {
        let plan = create_filter_heavy_plan(*n_filters);

        group.bench_with_input(BenchmarkId::new("optimize", n_filters), &plan, |b, plan| {
            b.iter(|| {
                let mut optimizer = QueryOptimizer::new(OptimizationLevel::Basic);
                optimizer
                    .optimize(std::hint::black_box(plan.clone()))
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_full_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Optimization");

    let plan = create_complex_plan();

    for level in [
        OptimizationLevel::None,
        OptimizationLevel::Basic,
        OptimizationLevel::Standard,
        OptimizationLevel::Aggressive,
    ] {
        group.bench_with_input(
            BenchmarkId::new("level", format!("{:?}", level)),
            &plan,
            |b, plan| {
                b.iter(|| {
                    let mut optimizer = QueryOptimizer::new(level);
                    optimizer
                        .optimize(std::hint::black_box(plan.clone()))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_selectivity_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Selectivity Estimation");

    let mut stats = ColumnStats::new("test_col".to_string(), 1_000_000);
    stats.distinct_count = 10_000;
    stats.min_value = Some(0.0);
    stats.max_value = Some(100_000.0);

    let operations = vec![
        ("equals", FilterOp::Equals(500.0)),
        ("range", FilterOp::GreaterThan(50_000.0)),
        ("between", FilterOp::Between(25_000.0, 75_000.0)),
        ("in_list", FilterOp::In(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
    ];

    for (name, op) in operations {
        group.bench_function(name, |b| {
            b.iter(|| stats.estimate_selectivity(std::hint::black_box(&op)));
        });
    }

    group.finish();
}

fn bench_plan_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("Plan Construction");

    for n_operations in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("build", n_operations),
            n_operations,
            |b, &n| {
                b.iter(|| {
                    let mut builder = QueryPlanBuilder::new(100_000);
                    for i in 0..n {
                        builder =
                            builder.filter(format!("col_{}", i), FilterOp::GreaterThan(i as f64));
                    }
                    std::hint::black_box(builder.build())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_predicate_pushdown,
    bench_full_optimization,
    bench_selectivity_estimation,
    bench_plan_construction,
);

criterion_main!(benches);

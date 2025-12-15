//! Machine Learning Benchmarks
//!
//! Comprehensive benchmarks for ML algorithms including decision trees,
//! ensemble methods, and neural networks.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pandrs::dataframe::DataFrame;
use pandrs::ml::models::ensemble::{
    GradientBoostingClassifier, GradientBoostingConfigBuilder, RandomForestClassifier,
    RandomForestConfigBuilder,
};
use pandrs::ml::models::neural::{MLPClassifier, MLPConfigBuilder};
use pandrs::ml::models::tree::{DecisionTreeClassifier, DecisionTreeConfigBuilder};
use pandrs::ml::models::SupervisedModel;
use pandrs::series::Series;

/// Create a synthetic classification dataset
fn create_classification_dataset(n_samples: usize, n_features: usize) -> DataFrame {
    let mut df = DataFrame::new();

    // Simple LCG random generator for reproducibility
    let mut rng_state: u64 = 42;
    let rand_f64 = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as f64 / (u32::MAX as f64)
    };

    // Generate features
    for f in 0..n_features {
        let values: Vec<f64> = (0..n_samples).map(|_| rand_f64(&mut rng_state)).collect();
        let series = Series::new(values, Some(format!("feature_{}", f))).unwrap();
        df.add_column(format!("feature_{}", f), series).unwrap();
    }

    // Generate binary labels based on sum of first two features
    let f0 = df.get_column_numeric_values("feature_0").unwrap();
    let f1 = df.get_column_numeric_values("feature_1").unwrap();
    let labels: Vec<f64> = f0
        .iter()
        .zip(f1.iter())
        .map(|(a, b)| if a + b > 1.0 { 1.0 } else { 0.0 })
        .collect();
    let label_series = Series::new(labels, Some("label".to_string())).unwrap();
    df.add_column("label".to_string(), label_series).unwrap();

    df
}

/// Create a synthetic regression dataset
fn create_regression_dataset(n_samples: usize, n_features: usize) -> DataFrame {
    let mut df = DataFrame::new();

    let mut rng_state: u64 = 42;
    let rand_f64 = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 33) as f64 / (u32::MAX as f64)
    };

    for f in 0..n_features {
        let values: Vec<f64> = (0..n_samples).map(|_| rand_f64(&mut rng_state)).collect();
        let series = Series::new(values, Some(format!("feature_{}", f))).unwrap();
        df.add_column(format!("feature_{}", f), series).unwrap();
    }

    // Generate target as linear combination + noise
    let mut target = vec![0.0; n_samples];
    for f in 0..n_features.min(3) {
        let col = df
            .get_column_numeric_values(&format!("feature_{}", f))
            .unwrap();
        for (i, v) in col.iter().enumerate() {
            target[i] += v * (f as f64 + 1.0);
        }
    }
    // Add small noise
    for (i, t) in target.iter_mut().enumerate() {
        *t += rand_f64(&mut rng_state) * 0.1;
        let _ = i; // Suppress unused warning
    }

    let target_series = Series::new(target, Some("target".to_string())).unwrap();
    df.add_column("target".to_string(), target_series).unwrap();

    df
}

fn bench_decision_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decision Tree");

    for n_samples in [100, 500, 1000].iter() {
        let df = create_classification_dataset(*n_samples, 10);

        group.bench_with_input(BenchmarkId::new("fit", n_samples), &df, |b, df| {
            b.iter(|| {
                let config = DecisionTreeConfigBuilder::new().max_depth(5).build();
                let mut tree = DecisionTreeClassifier::new(config);
                tree.fit(std::hint::black_box(df), "label").unwrap();
            });
        });

        // Pre-fit a tree for prediction benchmark
        let config = DecisionTreeConfigBuilder::new().max_depth(5).build();
        let mut tree = DecisionTreeClassifier::new(config);
        tree.fit(&df, "label").unwrap();

        group.bench_with_input(
            BenchmarkId::new("predict", n_samples),
            &(&df, &tree),
            |b, (df, tree)| {
                b.iter(|| {
                    tree.predict(std::hint::black_box(*df)).unwrap();
                });
            },
        );
    }

    group.finish();
}

fn bench_random_forest(c: &mut Criterion) {
    let mut group = c.benchmark_group("Random Forest");
    group.sample_size(10); // Reduce sample size for slower benchmarks

    for n_estimators in [5, 10, 20].iter() {
        let df = create_classification_dataset(500, 10);

        group.bench_with_input(BenchmarkId::new("fit", n_estimators), &df, |b, df| {
            b.iter(|| {
                let config = RandomForestConfigBuilder::new()
                    .n_estimators(*n_estimators)
                    .max_depth(5)
                    .build();
                let mut rf = RandomForestClassifier::new(config);
                rf.fit(std::hint::black_box(df), "label").unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gradient_boosting(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gradient Boosting");
    group.sample_size(10);

    for n_estimators in [10, 25, 50].iter() {
        let df = create_classification_dataset(500, 10);

        group.bench_with_input(BenchmarkId::new("fit", n_estimators), &df, |b, df| {
            b.iter(|| {
                let config = GradientBoostingConfigBuilder::new()
                    .n_estimators(*n_estimators)
                    .max_depth(3)
                    .learning_rate(0.1)
                    .build();
                let mut gb = GradientBoostingClassifier::new(config);
                gb.fit(std::hint::black_box(df), "label").unwrap();
            });
        });
    }

    group.finish();
}

fn bench_neural_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("Neural Network");
    group.sample_size(10);

    for hidden_size in [16, 32, 64].iter() {
        let df = create_classification_dataset(200, 10);

        group.bench_with_input(BenchmarkId::new("fit", hidden_size), &df, |b, df| {
            b.iter(|| {
                let config = MLPConfigBuilder::new()
                    .hidden_layers(vec![*hidden_size])
                    .n_epochs(50)
                    .learning_rate(0.01)
                    .early_stopping_patience(None)
                    .build();
                let mut mlp = MLPClassifier::new(config);
                mlp.fit(std::hint::black_box(df), "label").unwrap();
            });
        });
    }

    group.finish();
}

fn bench_model_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Model Comparison");
    group.sample_size(10);

    let df = create_classification_dataset(500, 10);

    // Decision Tree
    group.bench_function("DecisionTree", |b| {
        b.iter(|| {
            let config = DecisionTreeConfigBuilder::new().max_depth(5).build();
            let mut tree = DecisionTreeClassifier::new(config);
            tree.fit(std::hint::black_box(&df), "label").unwrap();
        });
    });

    // Random Forest (small)
    group.bench_function("RandomForest_10", |b| {
        b.iter(|| {
            let config = RandomForestConfigBuilder::new()
                .n_estimators(10)
                .max_depth(5)
                .build();
            let mut rf = RandomForestClassifier::new(config);
            rf.fit(std::hint::black_box(&df), "label").unwrap();
        });
    });

    // Gradient Boosting (small)
    group.bench_function("GradientBoosting_20", |b| {
        b.iter(|| {
            let config = GradientBoostingConfigBuilder::new()
                .n_estimators(20)
                .max_depth(3)
                .build();
            let mut gb = GradientBoostingClassifier::new(config);
            gb.fit(std::hint::black_box(&df), "label").unwrap();
        });
    });

    // MLP (small)
    group.bench_function("MLP_32", |b| {
        b.iter(|| {
            let config = MLPConfigBuilder::new()
                .hidden_layers(vec![32])
                .n_epochs(50)
                .early_stopping_patience(None)
                .build();
            let mut mlp = MLPClassifier::new(config);
            mlp.fit(std::hint::black_box(&df), "label").unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_decision_tree,
    bench_random_forest,
    bench_gradient_boosting,
    bench_neural_network,
    bench_model_comparison,
);

criterion_main!(benches);

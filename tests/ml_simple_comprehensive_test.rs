#![allow(clippy::result_large_err)]
//! Simplified Comprehensive ML Tests
//!
//! Tests covering ML model creation, parameter setting, and basic functionality

use pandrs::ml::anomaly::IsolationForest;
use pandrs::ml::clustering::KMeans;
use pandrs::{DataFrame, PandRSError, Series};

// ============================================================================
// Anomaly Detection Tests (10 tests)
// ============================================================================

/// Test IsolationForest creation with default parameters
#[test]
fn test_isolation_forest_creation() {
    let iforest = IsolationForest::new();

    assert_eq!(iforest.n_estimators, 100);
    assert_eq!(iforest.contamination, 0.1);
    assert!(iforest.random_seed.is_none());
}

/// Test IsolationForest parameter configuration
#[test]
fn test_isolation_forest_parameters() {
    let iforest = IsolationForest::new()
        .n_estimators(50)
        .max_samples(100)
        .max_depth(10)
        .contamination(0.05)
        .random_seed(42);

    assert_eq!(iforest.n_estimators, 50);
    assert_eq!(iforest.max_samples, Some(100));
    assert_eq!(iforest.max_depth, Some(10));
    assert_eq!(iforest.contamination, 0.05);
    assert_eq!(iforest.random_seed, Some(42));
}

/// Test IsolationForest anomaly score retrieval
#[test]
fn test_isolation_forest_scores() {
    let iforest = IsolationForest::new();

    // Before fitting, scores should be empty
    let scores = iforest.anomaly_scores();
    assert_eq!(scores.len(), 0);

    // Labels should also be empty
    let labels = iforest.labels();
    assert_eq!(labels.len(), 0);
}

/// Test IsolationForest with extreme contamination values
#[test]
fn test_isolation_forest_extreme_contamination() {
    // Very low contamination
    let iforest_low = IsolationForest::new().contamination(0.001);
    assert_eq!(iforest_low.contamination, 0.001);

    // High contamination
    let iforest_high = IsolationForest::new().contamination(0.5);
    assert_eq!(iforest_high.contamination, 0.5);
}

/// Test IsolationForest column specification
#[test]
fn test_isolation_forest_column_spec() {
    let columns = vec!["feature1".to_string(), "feature2".to_string()];
    let iforest = IsolationForest::new().with_columns(columns.clone());

    assert_eq!(iforest.feature_columns, Some(columns));
}

/// Test IsolationForest reproducibility with seed
#[test]
fn test_isolation_forest_reproducibility() {
    let iforest1 = IsolationForest::new().random_seed(42);
    let iforest2 = IsolationForest::new().random_seed(42);

    assert_eq!(iforest1.random_seed, iforest2.random_seed);
    assert_eq!(iforest1.n_estimators, iforest2.n_estimators);
}

/// Test IsolationForest max_depth configuration
#[test]
fn test_isolation_forest_max_depth() {
    let iforest = IsolationForest::new().max_depth(5);

    assert_eq!(iforest.max_depth, Some(5));
}

/// Test IsolationForest max_samples configuration
#[test]
fn test_isolation_forest_max_samples() {
    let iforest = IsolationForest::new().max_samples(256);

    assert_eq!(iforest.max_samples, Some(256));
}

/// Test IsolationForest chaining multiple configurations
#[test]
fn test_isolation_forest_chaining() {
    let iforest = IsolationForest::new()
        .n_estimators(200)
        .max_samples(512)
        .max_depth(15)
        .contamination(0.15)
        .random_seed(123)
        .with_columns(vec!["x".to_string(), "y".to_string()]);

    assert_eq!(iforest.n_estimators, 200);
    assert_eq!(iforest.max_samples, Some(512));
    assert_eq!(iforest.max_depth, Some(15));
    assert_eq!(iforest.contamination, 0.15);
    assert_eq!(iforest.random_seed, Some(123));
    assert!(iforest.feature_columns.is_some());
}

/// Test IsolationForest clone
#[test]
fn test_isolation_forest_clone() {
    let iforest1 = IsolationForest::new().n_estimators(75).contamination(0.08);

    let iforest2 = iforest1.clone();

    assert_eq!(iforest1.n_estimators, iforest2.n_estimators);
    assert_eq!(iforest1.contamination, iforest2.contamination);
}

// ============================================================================
// Clustering Tests (10 tests)
// ============================================================================

/// Test KMeans creation
#[test]
fn test_kmeans_creation() {
    let kmeans = KMeans::new(3);

    assert_eq!(kmeans.n_clusters, 3);
    assert_eq!(kmeans.max_iter, 100);
    assert_eq!(kmeans.tol, 1e-4);
}

/// Test KMeans parameter configuration
#[test]
fn test_kmeans_parameters() {
    let kmeans = KMeans::new(5).max_iter(200).tol(1e-6).random_seed(42);

    assert_eq!(kmeans.n_clusters, 5);
    assert_eq!(kmeans.max_iter, 200);
    assert_eq!(kmeans.tol, 1e-6);
    assert_eq!(kmeans.random_seed, Some(42));
}

/// Test KMeans with single cluster
#[test]
fn test_kmeans_single_cluster() {
    let kmeans = KMeans::new(1);

    assert_eq!(kmeans.n_clusters, 1);
}

/// Test KMeans max iterations
#[test]
fn test_kmeans_max_iterations() {
    let kmeans = KMeans::new(3).max_iter(500);

    assert_eq!(kmeans.max_iter, 500);
}

/// Test KMeans tolerance
#[test]
fn test_kmeans_tolerance() {
    let kmeans = KMeans::new(3).tol(1e-8);

    assert_eq!(kmeans.tol, 1e-8);
}

/// Test KMeans random seed
#[test]
fn test_kmeans_random_seed() {
    let kmeans = KMeans::new(3).random_seed(999);

    assert_eq!(kmeans.random_seed, Some(999));
}

/// Test KMeans column specification
#[test]
fn test_kmeans_column_spec() {
    let columns = vec!["x".to_string(), "y".to_string()];
    let kmeans = KMeans::new(3).with_columns(columns.clone());

    assert_eq!(kmeans.feature_columns, Some(columns));
}

/// Test KMeans labels before fitting
#[test]
fn test_kmeans_labels_unfit() {
    let kmeans = KMeans::new(3);

    assert!(kmeans.labels.is_none());
    assert!(kmeans.centroids.is_none());
    assert!(kmeans.inertia.is_none());
}

/// Test KMeans chaining
#[test]
fn test_kmeans_chaining() {
    let kmeans = KMeans::new(4)
        .max_iter(300)
        .tol(1e-5)
        .random_seed(42)
        .with_columns(vec!["a".to_string(), "b".to_string()]);

    assert_eq!(kmeans.n_clusters, 4);
    assert_eq!(kmeans.max_iter, 300);
    assert_eq!(kmeans.tol, 1e-5);
    assert_eq!(kmeans.random_seed, Some(42));
}

/// Test KMeans clone
#[test]
fn test_kmeans_clone() {
    let kmeans1 = KMeans::new(5).max_iter(150);
    let kmeans2 = kmeans1.clone();

    assert_eq!(kmeans1.n_clusters, kmeans2.n_clusters);
    assert_eq!(kmeans1.max_iter, kmeans2.max_iter);
}

// ============================================================================
// Edge Case Tests (15 tests)
// ============================================================================

/// Test empty DataFrame handling
#[test]
fn test_empty_dataframe_handling() -> Result<(), PandRSError> {
    let df = DataFrame::new();

    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 0);

    Ok(())
}

/// Test single value Series
#[test]
fn test_single_value_series() -> Result<(), PandRSError> {
    let series = Series::new(vec![42.0], Some("single".to_string()))?;

    assert_eq!(series.len(), 1);

    Ok(())
}

/// Test large n_estimators
#[test]
fn test_large_n_estimators() {
    let iforest = IsolationForest::new().n_estimators(10000);

    assert_eq!(iforest.n_estimators, 10000);
}

/// Test very small contamination
#[test]
fn test_very_small_contamination() {
    let iforest = IsolationForest::new().contamination(0.0001);

    assert_eq!(iforest.contamination, 0.0001);
}

/// Test large number of clusters
#[test]
fn test_large_number_clusters() {
    let kmeans = KMeans::new(1000);

    assert_eq!(kmeans.n_clusters, 1000);
}

/// Test very large max_iterations
#[test]
fn test_very_large_max_iter() {
    let kmeans = KMeans::new(3).max_iter(1_000_000);

    assert_eq!(kmeans.max_iter, 1_000_000);
}

/// Test very small tolerance
#[test]
fn test_very_small_tolerance() {
    let kmeans = KMeans::new(3).tol(1e-15);

    assert_eq!(kmeans.tol, 1e-15);
}

/// Test parameter combination edge cases
#[test]
fn test_parameter_combinations() {
    let iforest = IsolationForest::new()
        .n_estimators(1)
        .max_samples(1)
        .max_depth(1)
        .contamination(0.99);

    assert_eq!(iforest.n_estimators, 1);
    assert_eq!(iforest.max_samples, Some(1));
    assert_eq!(iforest.max_depth, Some(1));
    assert_eq!(iforest.contamination, 0.99);
}

/// Test KMeans single iteration
#[test]
fn test_kmeans_single_iteration() {
    let kmeans = KMeans::new(3).max_iter(1);

    assert_eq!(kmeans.max_iter, 1);
}

/// Test zero tolerance
#[test]
fn test_zero_tolerance() {
    let kmeans = KMeans::new(3).tol(0.0);

    assert_eq!(kmeans.tol, 0.0);
}

/// Test multiple column specifications
#[test]
fn test_multiple_columns() {
    let cols = vec![
        "col1".to_string(),
        "col2".to_string(),
        "col3".to_string(),
        "col4".to_string(),
        "col5".to_string(),
    ];
    let iforest = IsolationForest::new().with_columns(cols.clone());

    assert_eq!(iforest.feature_columns, Some(cols));
}

/// Test no columns specified
#[test]
fn test_no_columns_specified() {
    let iforest = IsolationForest::new();

    assert!(iforest.feature_columns.is_none());
}

/// Test same seed produces same configuration
#[test]
fn test_same_seed_same_config() {
    let iforest1 = IsolationForest::new().random_seed(123);
    let iforest2 = IsolationForest::new().random_seed(123);

    assert_eq!(iforest1.random_seed, iforest2.random_seed);
}

/// Test different seeds produce different configurations
#[test]
fn test_different_seeds() {
    let iforest1 = IsolationForest::new().random_seed(123);
    let iforest2 = IsolationForest::new().random_seed(456);

    assert_ne!(iforest1.random_seed, iforest2.random_seed);
}

/// Test multiple parameter changes
#[test]
fn test_multiple_parameter_changes() {
    let mut iforest = IsolationForest::new();
    iforest = iforest.n_estimators(50);
    iforest = iforest.n_estimators(100);
    iforest = iforest.n_estimators(150);

    assert_eq!(iforest.n_estimators, 150);
}

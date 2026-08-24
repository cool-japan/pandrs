use super::*;
use crate::series::Series;

#[test]
fn test_auto_feature_engineer() {
    let mut engineer = AutoFeatureEngineer::new()
        .with_polynomial(2)
        .with_interactions(3)
        .with_scaling(ScalingMethod::StandardScaler);

    // Create test data
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");
    x.add_column(
        "feature2".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");

    // Fit and transform
    engineer
        .fit(&x, Some(&y))
        .expect("operation should succeed");
    let transformed = engineer.transform(&x).expect("operation should succeed");

    // Should have more features than original
    assert!(transformed.column_names().len() > x.column_names().len());
}

#[test]
fn test_standard_scaler() {
    let mut scaler = StandardScaler::new();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    scaler.fit(&data).expect("operation should succeed");
    let transformed = scaler.transform(&data).expect("operation should succeed");

    // Check that mean is approximately zero
    let mean = transformed.iter().sum::<f64>() / transformed.len() as f64;
    assert!((mean).abs() < 1e-10);

    // Check that std is approximately one
    let variance =
        transformed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / transformed.len() as f64;
    let std = variance.sqrt();
    assert!((std - 1.0).abs() < 1e-10);
}

#[test]
fn test_minmax_scaler() {
    let mut scaler = MinMaxScaler::new();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    scaler.fit(&data).expect("operation should succeed");
    let transformed = scaler.transform(&data).expect("operation should succeed");

    // Check range
    let min = transformed.iter().copied().fold(f64::INFINITY, f64::min);
    let max = transformed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!((min - 0.0).abs() < 1e-10);
    assert!((max - 1.0).abs() < 1e-10);
}

#[test]
fn test_aggregation_functions() {
    let engineer = AutoFeatureEngineer::new();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let mean = engineer
        .calculate_aggregation(&values, &AggregationFunction::Mean)
        .expect("operation should succeed");
    assert!((mean - 3.0).abs() < 1e-10);

    let sum = engineer
        .calculate_aggregation(&values, &AggregationFunction::Sum)
        .expect("operation should succeed");
    assert!((sum - 15.0).abs() < 1e-10);

    let min = engineer
        .calculate_aggregation(&values, &AggregationFunction::Min)
        .expect("operation should succeed");
    assert!((min - 1.0).abs() < 1e-10);

    let max = engineer
        .calculate_aggregation(&values, &AggregationFunction::Max)
        .expect("operation should succeed");
    assert!((max - 5.0).abs() < 1e-10);
}

/// Build a test DataFrame with named f64 columns
fn make_df(cols: &[(&str, Vec<f64>)]) -> DataFrame {
    let mut df = DataFrame::new();
    for (name, vals) in cols {
        df.add_column(
            name.to_string(),
            Series::new(vals.clone(), Some(name.to_string()))
                .expect("series creation should succeed"),
        )
        .expect("add_column should succeed");
    }
    df
}

#[test]
fn test_recursive_elimination_selects_k() {
    // feature0 = 2 * target (highly important), features 1-4 = noise-like constants/small
    let target: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let feat0: Vec<f64> = target.iter().map(|&t| 2.0 * t).collect();
    let feat1: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.01).sin()).collect();
    let feat2: Vec<f64> = vec![0.1; 20];
    let feat3: Vec<f64> = vec![0.2; 20];
    let feat4: Vec<f64> = vec![0.3; 20];

    let x = make_df(&[
        ("feat0", feat0),
        ("feat1", feat1),
        ("feat2", feat2),
        ("feat3", feat3),
        ("feat4", feat4),
    ]);
    let y = make_df(&[("target", target)]);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(FeatureSelectionMethod::RecursiveElimination, Some(2))
        .without_scaling();
    // Disable auto feature generation to keep the feature set small
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer.fit(&x, Some(&y)).expect("fit should succeed");
    let selected = engineer
        .get_selected_features()
        .expect("selected features should exist");

    // feature0 (index 0) should be among the top-2 selected
    assert!(
        selected.contains(&0),
        "feat0 (index 0) should be selected; got {:?}",
        selected
    );
    assert_eq!(selected.len(), 2, "should select exactly 2 features");
}

#[test]
fn test_l1_based_selects_k() {
    // All features are non-constant, non-collinear to avoid singular matrix
    // feat0 = 3 * target (dominant), others are orthogonal-ish patterns
    let target: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let feat0: Vec<f64> = target.iter().map(|&t| 3.0 * t).collect();
    let feat1: Vec<f64> = (1..=20)
        .map(|i| (i as f64 * 0.3).sin() + i as f64 * 0.001)
        .collect();
    let feat2: Vec<f64> = (1..=20)
        .map(|i| (i as f64 * 0.7).cos() * 0.1 + i as f64 * 0.002)
        .collect();
    let feat3: Vec<f64> = (1..=20)
        .map(|i| (i as f64 * 1.1).sin() * 0.05 - i as f64 * 0.0005)
        .collect();
    let feat4: Vec<f64> = (1..=20)
        .map(|i| (i as f64 * 1.5).cos() * 0.02 + i as f64 * 0.0001)
        .collect();

    let x = make_df(&[
        ("feat0", feat0),
        ("feat1", feat1),
        ("feat2", feat2),
        ("feat3", feat3),
        ("feat4", feat4),
    ]);
    let y = make_df(&[("target", target)]);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(FeatureSelectionMethod::L1Based, Some(2))
        .without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer.fit(&x, Some(&y)).expect("fit should succeed");
    let selected = engineer
        .get_selected_features()
        .expect("selected features should exist");

    assert!(
        selected.contains(&0),
        "feat0 should be selected by L1Based; got {:?}",
        selected
    );
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_robust_scaler_median_zero() {
    let mut scaler = RobustScaler::new();
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    scaler.fit(&data).expect("fit should succeed");
    let transformed = scaler.transform(&data).expect("transform should succeed");

    // median of [1,2,3,4,5] is 3; after subtracting median the middle element maps to 0
    let middle = transformed[2]; // was 3.0, should be (3-3)/IQR = 0.0
    assert!(
        middle.abs() < 1e-10,
        "median element should transform to 0.0, got {}",
        middle
    );

    // Round-trip: inverse_transform should recover original values
    let recovered = scaler
        .inverse_transform(&transformed)
        .expect("inverse_transform should succeed");
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!(
            (orig - rec).abs() < 1e-10,
            "inverse transform should recover {}, got {}",
            orig,
            rec
        );
    }
}

#[test]
fn test_mutual_info_selection() {
    // feat0 = constant (zero MI), feat1 = perfectly correlated with target (high MI)
    let target: Vec<f64> = (1..=30).map(|i| i as f64).collect();
    let feat0: Vec<f64> = vec![5.0; 30]; // constant → zero variance → zero MI
    let feat1: Vec<f64> = target.iter().map(|&t| t * 2.0 + 1.0).collect();

    let x = make_df(&[("constant", feat0), ("correlated", feat1)]);
    let y = make_df(&[("target", target)]);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(FeatureSelectionMethod::MutualInformation, Some(1))
        .without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer.fit(&x, Some(&y)).expect("fit should succeed");
    let selected = engineer
        .get_selected_features()
        .expect("selected features should exist");

    // The correlated feature (index 1) should be preferred over the constant (index 0)
    assert!(
        selected.contains(&1),
        "correlated feature (index 1) should be selected; got {:?}",
        selected
    );
    assert_eq!(selected.len(), 1);
}

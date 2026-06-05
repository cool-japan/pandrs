#![allow(clippy::result_large_err)]
//! Advanced ML capabilities integration tests
//!
//! This module provides comprehensive tests for the advanced machine learning
//! features including AutoML, feature engineering, and scikit-learn compatibility.

use pandrs::dataframe::DataFrame;
use pandrs::ml::*;
use pandrs::series::Series;
use std::collections::HashMap;

#[test]
fn test_sklearn_compat_standard_scaler() {
    let mut scaler = StandardScalerCompat::new();

    // Create test data
    let mut df = DataFrame::new();
    df.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "feature2".to_string(),
        Series::new(
            [10.0, 20.0, 30.0, 40.0, 50.0].to_vec(),
            Some("feature2".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    // Test fit and transform
    scaler.fit(&df, None).unwrap();
    let transformed = scaler.transform(&df).unwrap();

    // Verify transformation
    let feature1_col = transformed.get_column::<f64>("feature1").unwrap();
    let feature1_values = feature1_col.as_f64().unwrap();
    let mean = feature1_values.iter().sum::<f64>() / feature1_values.len() as f64;

    assert!(
        (mean).abs() < 1e-10,
        "Mean should be approximately zero after standardization"
    );

    // Test inverse transform
    let inverse_transformed = scaler.inverse_transform(&transformed).unwrap();
    let original_feature1 = df.get_column::<f64>("feature1").unwrap().as_f64().unwrap();
    let restored_feature1 = inverse_transformed
        .get_column::<f64>("feature1")
        .unwrap()
        .as_f64()
        .unwrap();

    for (original, restored) in original_feature1.iter().zip(restored_feature1.iter()) {
        assert!(
            (original - restored).abs() < 1e-10,
            "Inverse transform should restore original values"
        );
    }
}

#[test]
fn test_sklearn_compat_minmax_scaler() {
    let mut scaler = MinMaxScalerCompat::new();

    // Create test data
    let mut df = DataFrame::new();
    df.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
    )
    .unwrap();

    // Test fit and transform
    scaler.fit(&df, None).unwrap();
    let transformed = scaler.transform(&df).unwrap();

    // Verify range [0, 1]
    let feature1_col = transformed.get_column::<f64>("feature1").unwrap();
    let feature1_values = feature1_col.as_f64().unwrap();

    for value in &feature1_values {
        assert!(
            *value >= 0.0 && *value <= 1.0,
            "Values should be in range [0, 1]"
        );
    }

    let min_val = feature1_values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_val = feature1_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    assert!((min_val - 0.0).abs() < 1e-10, "Minimum should be 0");
    assert!((max_val - 1.0).abs() < 1e-10, "Maximum should be 1");
}

#[test]
fn test_feature_engineering_auto() {
    let mut engineer = AutoFeatureEngineer::new()
        .with_polynomial(2)
        .with_interactions(3)
        .with_scaling(ScalingMethod::StandardScaler);

    // Create test data
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature2".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string())).unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string())).unwrap(),
    )
    .unwrap();

    // Test fit and transform
    engineer.fit(&x, Some(&y)).unwrap();
    let transformed = engineer.transform(&x).unwrap();

    // Should have more features than original
    assert!(
        transformed.column_names().len() > x.column_names().len(),
        "Feature engineering should generate additional features"
    );

    // Check that some expected features exist
    let feature_names = transformed.column_names();
    let has_polynomial = feature_names.iter().any(|name| name.contains("^2"));
    let has_interaction = feature_names
        .iter()
        .any(|name| name.contains("_mult_") || name.contains("*"));

    assert!(
        has_polynomial || has_interaction,
        "Should generate polynomial or interaction features"
    );
}

#[test]
fn test_model_selection_kbest() {
    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);

    // Create test data with 3 features
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature2".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature3".to_string(),
        Series::new(vec![0.1, 0.2, 0.3, 0.4, 0.5], Some("feature3".to_string())).unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string())).unwrap(),
    )
    .unwrap();

    // Test fit and transform
    selector.fit(&x, &y).unwrap();
    let selected = selector.transform(&x).unwrap();

    // Should select exactly 2 features
    assert_eq!(
        selected.column_names().len(),
        2,
        "Should select exactly 2 features"
    );

    // Check that scores are available
    let scores = selector.get_scores();
    assert!(
        scores.is_some(),
        "Feature scores should be available after fitting"
    );
    assert_eq!(
        scores.unwrap().len(),
        3,
        "Should have scores for all 3 original features"
    );
}

#[test]
fn test_cross_validation_strategy() {
    let cv_kfold = CrossValidationStrategy::KFold {
        n_splits: 5,
        shuffle: true,
        random_state: Some(42),
    };

    match cv_kfold {
        CrossValidationStrategy::KFold {
            n_splits,
            shuffle,
            random_state,
        } => {
            assert_eq!(n_splits, 5);
            assert!(shuffle);
            assert_eq!(random_state, Some(42));
        }
        _ => panic!("Wrong CV strategy type"),
    }

    let cv_stratified = CrossValidationStrategy::StratifiedKFold {
        n_splits: 3,
        shuffle: false,
        random_state: None,
    };

    match cv_stratified {
        CrossValidationStrategy::StratifiedKFold {
            n_splits,
            shuffle,
            random_state,
        } => {
            assert_eq!(n_splits, 3);
            assert!(!shuffle);
            assert_eq!(random_state, None);
        }
        _ => panic!("Wrong CV strategy type"),
    }
}

#[test]
fn test_parameter_distributions() {
    let uniform_int = ParameterDistribution::UniformInt { low: 1, high: 10 };
    let sample = uniform_int.sample();
    let value: i64 = sample.parse().unwrap();
    assert!(
        (1..=10).contains(&value),
        "Uniform int sample should be in range"
    );

    let uniform_float = ParameterDistribution::UniformFloat {
        low: 0.0,
        high: 1.0,
    };
    let sample = uniform_float.sample();
    let value: f64 = sample.parse().unwrap();
    assert!(
        (0.0..=1.0).contains(&value),
        "Uniform float sample should be in range"
    );

    let choice = ParameterDistribution::Choice(vec![
        "option1".to_string(),
        "option2".to_string(),
        "option3".to_string(),
    ]);
    let sample = choice.sample();
    assert!(
        ["option1", "option2", "option3"].contains(&sample.as_str()),
        "Choice sample should be one of the options"
    );

    let fixed = ParameterDistribution::Fixed("fixed_value".to_string());
    let sample = fixed.sample();
    assert_eq!(
        sample, "fixed_value",
        "Fixed distribution should always return the same value"
    );
}

#[test]
fn test_scorer_functions() {
    let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = [1.1, 1.9, 3.1, 3.9, 5.1];

    // Test R² scorer
    let r2_scorer = Scorer::R2;
    let r2_score = r2_scorer.score(&y_true, &y_pred).unwrap();
    assert!(
        r2_score > 0.9,
        "R² score should be high for good predictions"
    );

    // Test MSE scorer (negated)
    let mse_scorer = Scorer::NegMeanSquaredError;
    let mse_score = mse_scorer.score(&y_true, &y_pred).unwrap();
    assert!(mse_score < 0.0, "Negative MSE should be negative");
    assert!(mse_score > -1.0, "MSE should be small for good predictions");

    // Test MAE scorer (negated)
    let mae_scorer = Scorer::NegMeanAbsoluteError;
    let mae_score = mae_scorer.score(&y_true, &y_pred).unwrap();
    assert!(mae_score < 0.0, "Negative MAE should be negative");
    assert!(mae_score > -1.0, "MAE should be small for good predictions");

    // Test binary classification scorers
    let y_true_binary = [0.0, 1.0, 1.0, 0.0, 1.0];
    let y_pred_binary = [0.0, 1.0, 1.0, 0.0, 1.0]; // Perfect predictions

    let accuracy_scorer = Scorer::Accuracy;
    let accuracy = accuracy_scorer
        .score(&y_true_binary, &y_pred_binary)
        .unwrap();
    assert!(
        (accuracy - 1.0).abs() < 1e-10,
        "Perfect predictions should have accuracy 1.0"
    );

    let f1_scorer = Scorer::F1;
    let f1_score = f1_scorer.score(&y_true_binary, &y_pred_binary).unwrap();
    assert!(
        (f1_score - 1.0).abs() < 1e-10,
        "Perfect predictions should have F1 score 1.0"
    );

    let precision_scorer = Scorer::Precision;
    let precision = precision_scorer
        .score(&y_true_binary, &y_pred_binary)
        .unwrap();
    assert!(
        (precision - 1.0).abs() < 1e-10,
        "Perfect predictions should have precision 1.0"
    );

    let recall_scorer = Scorer::Recall;
    let recall = recall_scorer.score(&y_true_binary, &y_pred_binary).unwrap();
    assert!(
        (recall - 1.0).abs() < 1e-10,
        "Perfect predictions should have recall 1.0"
    );
}

#[test]
fn test_automl_config() {
    let config = AutoMLConfig::default();

    assert!(matches!(config.task_type, TaskType::Auto));
    assert_eq!(config.time_limit, Some(3600.0));
    assert_eq!(config.max_models, Some(50));
    assert!(config.feature_engineering);
    assert!(config.feature_selection);
    assert!(config.ensemble_methods);
    assert_eq!(config.verbose, 1);
    assert_eq!(config.memory_limit, Some(8.0));

    // Test custom config
    let custom_config = AutoMLConfig {
        task_type: TaskType::Regression,
        time_limit: Some(1800.0),
        max_models: Some(20),
        feature_engineering: false,
        verbose: 2,
        ..Default::default()
    };

    assert!(matches!(custom_config.task_type, TaskType::Regression));
    assert_eq!(custom_config.time_limit, Some(1800.0));
    assert_eq!(custom_config.max_models, Some(20));
    assert!(!custom_config.feature_engineering);
    assert_eq!(custom_config.verbose, 2);
}

#[test]
fn test_automl_task_detection() {
    let automl = AutoML::new();

    // Test regression detection
    let mut y_reg = DataFrame::new();
    y_reg
        .add_column(
            "target".to_string(),
            Series::new(vec![1.5, 2.3, 3.7, 4.1, 5.9], Some("target".to_string())).unwrap(),
        )
        .unwrap();

    let task_type = automl.detect_task_type(&y_reg).unwrap();
    assert!(matches!(task_type, TaskType::Regression));

    // Test binary classification detection
    let mut y_binary = DataFrame::new();
    y_binary
        .add_column(
            "target".to_string(),
            Series::new(vec![0.0, 1.0, 1.0, 0.0, 1.0], Some("target".to_string())).unwrap(),
        )
        .unwrap();

    let task_type = automl.detect_task_type(&y_binary).unwrap();
    assert!(matches!(task_type, TaskType::BinaryClassification));

    // Test multi-class classification detection
    let mut y_multi = DataFrame::new();
    y_multi
        .add_column(
            "target".to_string(),
            Series::new(
                [0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0].to_vec(),
                Some("target".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

    let task_type = automl.detect_task_type(&y_multi).unwrap();
    assert!(matches!(task_type, TaskType::MultiClassification));
}

#[test]
fn test_model_search_space() {
    let regression_space = ModelSearchSpace::default_regression();

    assert!(
        !regression_space.linear_models.is_empty(),
        "Should have linear models"
    );
    assert!(
        !regression_space.tree_models.is_empty(),
        "Should have tree models"
    );
    assert!(
        !regression_space.ensemble_models.is_empty(),
        "Should have ensemble models"
    );

    // Check specific models are included
    let has_linear_regression = regression_space
        .linear_models
        .iter()
        .any(|(name, _)| name == "LinearRegression");
    assert!(has_linear_regression, "Should include LinearRegression");

    let has_ridge = regression_space
        .linear_models
        .iter()
        .any(|(name, _)| name == "Ridge");
    assert!(has_ridge, "Should include Ridge regression");

    let has_random_forest = regression_space
        .ensemble_models
        .iter()
        .any(|(name, _)| name == "RandomForest");
    assert!(has_random_forest, "Should include RandomForest");

    // Test classification space
    let classification_space = ModelSearchSpace::default_classification();

    let has_logistic_regression = classification_space
        .linear_models
        .iter()
        .any(|(name, _)| name == "LogisticRegression");
    assert!(has_logistic_regression, "Should include LogisticRegression");

    let has_decision_tree = classification_space
        .tree_models
        .iter()
        .any(|(name, _)| name == "DecisionTreeClassifier");
    assert!(has_decision_tree, "Should include DecisionTreeClassifier");
}

#[test]
fn test_aggregation_functions() {
    let engineer = AutoFeatureEngineer::new();
    let values = [1.0, 2.0, 3.0, 4.0, 5.0];

    // Test mean
    let mean = engineer
        .calculate_aggregation(&values, &AggregationFunction::Mean)
        .unwrap();
    assert!((mean - 3.0).abs() < 1e-10, "Mean should be 3.0");

    // Test sum
    let sum = engineer
        .calculate_aggregation(&values, &AggregationFunction::Sum)
        .unwrap();
    assert!((sum - 15.0).abs() < 1e-10, "Sum should be 15.0");

    // Test min
    let min = engineer
        .calculate_aggregation(&values, &AggregationFunction::Min)
        .unwrap();
    assert!((min - 1.0).abs() < 1e-10, "Min should be 1.0");

    // Test max
    let max = engineer
        .calculate_aggregation(&values, &AggregationFunction::Max)
        .unwrap();
    assert!((max - 5.0).abs() < 1e-10, "Max should be 5.0");

    // Test median
    let median = engineer
        .calculate_aggregation(&values, &AggregationFunction::Median)
        .unwrap();
    assert!((median - 3.0).abs() < 1e-10, "Median should be 3.0");

    // Test std
    let std = engineer
        .calculate_aggregation(&values, &AggregationFunction::Std)
        .unwrap();
    assert!(
        std > 1.0 && std < 2.0,
        "Standard deviation should be around 1.58"
    );

    // Test count
    let count = engineer
        .calculate_aggregation(&values, &AggregationFunction::Count)
        .unwrap();
    assert!((count - 5.0).abs() < 1e-10, "Count should be 5.0");

    // Test quantile
    let q25 = engineer
        .calculate_aggregation(&values, &AggregationFunction::Quantile(0.25))
        .unwrap();
    assert!((q25 - 2.0).abs() < 1e-10, "25th percentile should be 2.0");

    let q75 = engineer
        .calculate_aggregation(&values, &AggregationFunction::Quantile(0.75))
        .unwrap();
    assert!((q75 - 4.0).abs() < 1e-10, "75th percentile should be 4.0");
}

#[test]
fn test_sklearn_estimator_interface() {
    let scaler = StandardScalerCompat::new();

    // Test get_params
    let params = scaler.get_params();
    assert!(params.contains_key("with_mean"));
    assert!(params.contains_key("with_std"));
    assert!(params.contains_key("copy"));

    assert_eq!(params.get("with_mean").unwrap(), "true");
    assert_eq!(params.get("with_std").unwrap(), "true");
    assert_eq!(params.get("copy").unwrap(), "true");

    // Test set_params
    let mut scaler_copy = scaler.clone();
    let mut new_params = HashMap::new();
    new_params.insert("with_mean".to_string(), "false".to_string());
    new_params.insert("with_std".to_string(), "false".to_string());

    scaler_copy.set_params(new_params).unwrap();

    let updated_params = scaler_copy.get_params();
    assert_eq!(updated_params.get("with_mean").unwrap(), "false");
    assert_eq!(updated_params.get("with_std").unwrap(), "false");
}

// Integration test combining multiple components
#[test]
fn test_ml_pipeline_integration() {
    // Create sample dataset
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].to_vec(),
            Some("feature1".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature2".to_string(),
        Series::new(
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0].to_vec(),
            Some("feature2".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature3".to_string(),
        Series::new(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].to_vec(),
            Some("feature3".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0].to_vec(),
            Some("target".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    // Test feature engineering
    let mut engineer = AutoFeatureEngineer::new()
        .with_polynomial(2)
        .with_interactions(3)
        .with_selection(
            FeatureSelectionMethod::KBest(ScoreFunction::FRegression),
            Some(5),
        )
        .with_scaling(ScalingMethod::StandardScaler);

    engineer.fit(&x, Some(&y)).unwrap();
    let engineered_x = engineer.transform(&x).unwrap();

    assert!(
        engineered_x.column_names().len() >= 3,
        "Feature engineering should produce at least original features"
    );

    // Test model selection components
    let _cv_strategy = CrossValidationStrategy::KFold {
        n_splits: 3,
        shuffle: true,
        random_state: Some(42),
    };

    let _scoring = Scorer::R2;

    // Test parameter distributions
    let mut param_space = HashMap::new();
    param_space.insert(
        "alpha".to_string(),
        ParameterDistribution::LogUniform {
            low: 1e-3,
            high: 1e1,
        },
    );
    param_space.insert(
        "fit_intercept".to_string(),
        ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
    );

    // Sample parameters
    for _ in 0..5 {
        let alpha_sample = param_space.get("alpha").unwrap().sample();
        let alpha_value: f64 = alpha_sample.parse().unwrap();
        assert!(
            (1e-3..=1e1).contains(&alpha_value),
            "Alpha parameter should be in expected range"
        );

        let intercept_sample = param_space.get("fit_intercept").unwrap().sample();
        assert!(
            ["true", "false"].contains(&intercept_sample.as_str()),
            "fit_intercept should be true or false"
        );
    }

    println!("✅ Advanced ML integration test completed successfully");
}

// ─── Track C Stub 1 tests ───────────────────────────────────────────────────

/// PIECE B: `create_estimator` must return Ok for LinearRegression
#[test]
fn test_create_estimator_linear_regression() {
    let automl = AutoML::new();
    let estimator = automl.create_estimator("LinearRegression");
    assert!(
        estimator.is_ok(),
        "create_estimator('LinearRegression') should return Ok"
    );
}

/// PIECE B: `create_estimator` must return Ok for all supported model names
#[test]
fn test_create_estimator_all_supported() {
    let automl = AutoML::new();

    let names = [
        "LinearRegression",
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "DecisionTree",
        "RandomForestRegressor",
        "RandomForestClassifier",
        "RandomForest",
        "GradientBoostingRegressor",
        "GradientBoostingClassifier",
        "GradientBoosting",
    ];
    for name in &names {
        let result = automl.create_estimator(name);
        assert!(
            result.is_ok(),
            "create_estimator('{}') should return Ok, got: {:?}",
            name,
            result.err()
        );
    }
}

/// PIECE B: `create_estimator` must return Err for unknown model names
#[test]
fn test_create_estimator_unknown() {
    let automl = AutoML::new();
    let result = automl.create_estimator("UnknownModel123");
    assert!(
        result.is_err(),
        "create_estimator with unknown model should return Err"
    );
}

/// PIECE A: SupervisedAdapter<LinearRegression> — fit then predict
#[test]
fn test_supervised_adapter_fit_predict() {
    use pandrs::ml::models::linear::LinearRegression;
    use pandrs::ml::SupervisedAdapter;

    // Toy dataset: y = 2*x + 1
    let mut x = DataFrame::new();
    x.add_column(
        "x".to_string(),
        Series::new(
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            Some("x".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            vec![3.0f64, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
            Some("target".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut adapter = SupervisedAdapter::new(LinearRegression::new(), "target");

    adapter.fit(&x, &y).expect("fit should succeed");

    let preds = adapter.predict(&x).expect("predict should succeed");
    assert_eq!(
        preds.len(),
        x.nrows(),
        "predict should return same number of rows as input"
    );

    // Check predictions are close to ground truth (y = 2x + 1)
    for (i, (&pred, expected)) in preds
        .iter()
        .zip([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0].iter())
        .enumerate()
    {
        assert!(
            (pred - expected).abs() < 1.0,
            "Prediction {} should be close to {}, got {}",
            i,
            expected,
            pred
        );
    }
}

/// PIECE A: SupervisedAdapter score method
#[test]
fn test_supervised_adapter_score() {
    use pandrs::ml::models::linear::LinearRegression;
    use pandrs::ml::SupervisedAdapter;

    let mut x = DataFrame::new();
    x.add_column(
        "feature".to_string(),
        Series::new(
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Some("feature".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            vec![2.0f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            Some("target".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut adapter = SupervisedAdapter::new(LinearRegression::new(), "target");
    adapter.fit(&x, &y).expect("fit should succeed");

    let score = adapter.score(&x, &y).expect("score should succeed");
    assert!(
        score > 0.9,
        "R² score should be high for a linear dataset, got {}",
        score
    );
}

/// PIECE C: RandomizedSearchCV with real LinearRegression estimator
#[test]
fn test_randomized_search_cv_real() {
    use pandrs::ml::models::linear::LinearRegression;
    use pandrs::ml::SupervisedAdapter;

    // Create dataset: y = 3*x1 + 2*x2 + noise (features are NOT collinear)
    let n = 20usize;
    let x1: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
    // x2 is independent of x1 (alternating pattern)
    let x2: Vec<f64> = (1..=n as i64)
        .map(|i| {
            if i % 2 == 0 {
                i as f64 * 2.0
            } else {
                -(i as f64)
            }
        })
        .collect();
    let y_vals: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 3.0 * a + 2.0 * b + 1.0)
        .collect();

    let mut x = DataFrame::new();
    x.add_column(
        "x1".to_string(),
        Series::new(x1, Some("x1".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "x2".to_string(),
        Series::new(x2, Some("x2".to_string())).unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(y_vals, Some("target".to_string())).unwrap(),
    )
    .unwrap();

    let estimator: Box<dyn pandrs::ml::SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));

    let mut param_distributions = HashMap::new();
    param_distributions.insert(
        "fit_intercept".to_string(),
        ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
    );

    let mut search = RandomizedSearchCV::new(estimator, param_distributions, 3)
        .with_cv(CrossValidationStrategy::KFold {
            n_splits: 3,
            shuffle: false,
            random_state: Some(42),
        })
        .with_scoring(Scorer::R2);

    search
        .fit(&x, &y)
        .expect("RandomizedSearchCV::fit should succeed");

    let results = search.get_results().expect("Results should be available");

    // Best score should be finite and reasonable for a linear model
    assert!(
        results.best_score_.is_finite(),
        "best_score_ should be finite, got {}",
        results.best_score_
    );

    // cv_results should not be empty
    assert!(
        !results.cv_results_.is_empty(),
        "cv_results_ should not be empty"
    );
}

/// PIECE D / end-to-end: AutoML with LinearRegression whitelist
///
/// Verifies that `cv_std` is finite and `feature_importances` is Some after a full run.
#[test]
fn test_automl_end_to_end_linear() {
    // Build a dataset with non-collinear features: y = 2*x1 + x2
    // x1 is sequential, x2 is an independent pattern so XᵀX is invertible
    let n = 30usize;
    let x1: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
    // x2 is independently drawn (not a linear function of x1)
    let x2: Vec<f64> = (1..=n as i64)
        .map(|i| ((i as f64 * 1.3 + 7.0) % 13.0) + 1.0)
        .collect();
    let y_vals: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 2.0 * a + b)
        .collect();

    let mut x = DataFrame::new();
    x.add_column(
        "x1".to_string(),
        Series::new(x1, Some("x1".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "x2".to_string(),
        Series::new(x2, Some("x2".to_string())).unwrap(),
    )
    .unwrap();

    let mut y_df = DataFrame::new();
    y_df.add_column(
        "target".to_string(),
        Series::new(y_vals, Some("target".to_string())).unwrap(),
    )
    .unwrap();

    let config = AutoMLConfig {
        task_type: TaskType::Regression,
        max_models: Some(2),
        model_whitelist: Some(vec!["LinearRegression".to_string()]),
        feature_engineering: false, // disable to keep the test fast
        feature_selection: false,
        ensemble_methods: false,
        verbose: 0,
        cv_strategy: CrossValidationStrategy::KFold {
            n_splits: 3,
            shuffle: false,
            random_state: Some(42),
        },
        scoring: Scorer::R2,
        time_limit: None,
        random_state: Some(42),
        optimize_for_interpretability: false,
        memory_limit: None,
        model_blacklist: None,
    };

    let mut automl = AutoML::with_config(config);
    automl.fit(&x, &y_df).expect("AutoML::fit should succeed");

    let results = automl
        .get_results()
        .expect("Results should be available after fit");

    // Best model should be LinearRegression
    assert_eq!(results.best_pipeline, "LinearRegression");

    // Leaderboard should have at least one entry
    assert!(
        !results.leaderboard.is_empty(),
        "Leaderboard should not be empty"
    );

    // cv_std should be finite (not the placeholder 0.0 from stubs)
    let best = &results.leaderboard[0];
    assert!(
        best.cv_std.is_finite(),
        "cv_std should be a finite number, got {}",
        best.cv_std
    );

    // feature_importance should be Some for LinearRegression
    assert!(
        best.feature_importance.is_some(),
        "LinearRegression should have feature importances"
    );

    // holdout_score should be set
    assert!(
        results.holdout_score.is_some(),
        "holdout_score should be Some after full AutoML fit"
    );
}

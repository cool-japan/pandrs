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
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
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
            assert_eq!(shuffle, true);
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
            assert_eq!(shuffle, false);
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
        value >= 1 && value <= 10,
        "Uniform int sample should be in range"
    );

    let uniform_float = ParameterDistribution::UniformFloat {
        low: 0.0,
        high: 1.0,
    };
    let sample = uniform_float.sample();
    let value: f64 = sample.parse().unwrap();
    assert!(
        value >= 0.0 && value <= 1.0,
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
    let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

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
    let y_true_binary = vec![0.0, 1.0, 1.0, 0.0, 1.0];
    let y_pred_binary = vec![0.0, 1.0, 1.0, 0.0, 1.0]; // Perfect predictions

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
                vec![0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 2.0],
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
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

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
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Some("feature1".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature2".to_string(),
        Series::new(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            Some("feature2".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    x.add_column(
        "feature3".to_string(),
        Series::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            Some("feature3".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
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
            alpha_value >= 1e-3 && alpha_value <= 1e1,
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

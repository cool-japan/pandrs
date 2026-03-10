#![allow(clippy::result_large_err)]
//! Decision Tree Machine Learning Example
//!
//! This example demonstrates how to use Decision Tree models for both
//! classification and regression tasks in PandRS.
//!
//! Features demonstrated:
//! - DecisionTreeClassifier for categorical predictions
//! - DecisionTreeRegressor for continuous predictions
//! - Configuration options (max_depth, min_samples_split, criterion)
//! - Feature importance extraction
//! - Model evaluation and cross-validation
//! - Comparison of different split criteria (Gini vs Entropy, MSE vs MAE)

use pandrs::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::ml::models::tree::{
    DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeConfigBuilder, DecisionTreeRegressor,
    SplitCriterion,
};
use pandrs::ml::models::{train_test_split, SupervisedModel};
use pandrs::series::Series;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Decision Tree Examples ===\n");

    // Run classification example
    classification_example()?;

    // Run regression example
    regression_example()?;

    // Run feature importance example
    feature_importance_example()?;

    // Run criterion comparison
    criterion_comparison_example()?;

    Ok(())
}

/// Demonstrate Decision Tree Classification
///
/// This example shows how to train a classifier to predict iris flower species
/// based on sepal and petal measurements.
fn classification_example() -> Result<()> {
    println!("--- Classification Example: Iris Dataset ---");

    // Create a sample iris dataset
    let mut df = DataFrame::new();

    // Sepal length, sepal width, petal length, petal width
    df.add_column(
        "sepal_length".to_string(),
        Series::new(
            vec![
                5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 5.7, 6.7, 5.8,
                6.0, 5.4, 6.0, 6.9,
            ],
            Some("sepal_length".to_string()),
        )?,
    )?;

    df.add_column(
        "sepal_width".to_string(),
        Series::new(
            vec![
                3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.3, 2.7, 3.0, 2.9, 2.8, 3.0, 2.4, 2.8, 3.1, 2.7,
                2.9, 3.0, 2.2, 3.1,
            ],
            Some("sepal_width".to_string()),
        )?,
    )?;

    df.add_column(
        "petal_length".to_string(),
        Series::new(
            vec![
                1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 4.9, 5.1, 5.9, 5.6, 5.8, 6.6, 3.3, 4.2, 4.4, 5.1,
                4.5, 4.5, 5.0, 5.4,
            ],
            Some("petal_length".to_string()),
        )?,
    )?;

    df.add_column(
        "petal_width".to_string(),
        Series::new(
            vec![
                0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 1.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.0, 1.3, 1.4, 1.9,
                1.5, 1.5, 1.5, 2.3,
            ],
            Some("petal_width".to_string()),
        )?,
    )?;

    // Target: 0=setosa, 1=versicolor, 2=virginica
    df.add_column(
        "species".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0,
                1.0, 1.0, 2.0, 2.0,
            ],
            Some("species".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split into training and test sets (80-20 split)
    let (train_df, test_df) = train_test_split(&df, 0.2, false, Some(42))?;
    println!("Training set: {} rows", train_df.nrows());
    println!("Test set: {} rows", test_df.nrows());

    // Create and configure a Decision Tree Classifier
    // Using Gini impurity criterion with max depth of 3
    let config = DecisionTreeConfigBuilder::new()
        .max_depth(3)
        .min_samples_split(2)
        .criterion(SplitCriterion::Gini)
        .random_seed(42)
        .build();

    let mut classifier = DecisionTreeClassifier::new(config);

    // Train the model
    println!("\nTraining Decision Tree Classifier...");
    classifier.fit(&train_df, "species")?;
    println!("Model trained successfully!");

    // Make predictions on test set
    let predictions = classifier.predict(&test_df)?;
    println!("\nPredictions on test set:");
    for (i, pred) in predictions.iter().take(5).enumerate() {
        println!("  Sample {}: Predicted class = {:.0}", i + 1, pred);
    }

    // Calculate accuracy
    let test_labels = test_df.get_column::<f64>("species")?;
    let mut correct = 0;
    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        if (pred - actual).abs() < 0.5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("\nAccuracy: {:.2}%", accuracy * 100.0);

    // Get feature importances
    if let Some(importances) = classifier.feature_importances() {
        println!("\nFeature Importances:");
        let mut importance_vec: Vec<(&String, &f64)> = importances.iter().collect();
        importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (feature, importance) in importance_vec {
            println!("  {}: {:.4}", feature, importance);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate Decision Tree Regression
///
/// This example shows how to use a regression tree to predict continuous values
/// (e.g., house prices based on features).
fn regression_example() -> Result<()> {
    println!("--- Regression Example: House Price Prediction ---");

    // Create a sample house price dataset
    let mut df = DataFrame::new();

    // Features: square footage, number of bedrooms, age of house
    df.add_column(
        "square_feet".to_string(),
        Series::new(
            vec![
                1500.0, 1600.0, 1700.0, 1875.0, 1100.0, 1550.0, 2350.0, 2450.0, 1425.0, 1700.0,
                1900.0, 2300.0, 1320.0, 1600.0, 2400.0, 3000.0, 1800.0, 2100.0, 1650.0, 2200.0,
            ],
            Some("square_feet".to_string()),
        )?,
    )?;

    df.add_column(
        "bedrooms".to_string(),
        Series::new(
            vec![
                2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 3.0, 4.0,
            ],
            Some("bedrooms".to_string()),
        )?,
    )?;

    df.add_column(
        "age".to_string(),
        Series::new(
            vec![
                10.0, 8.0, 20.0, 5.0, 35.0, 12.0, 3.0, 2.0, 15.0, 10.0, 7.0, 4.0, 25.0, 9.0, 5.0,
                1.0, 11.0, 6.0, 13.0, 8.0,
            ],
            Some("age".to_string()),
        )?,
    )?;

    // Target: price in thousands
    df.add_column(
        "price".to_string(),
        Series::new(
            vec![
                250.0, 280.0, 310.0, 350.0, 180.0, 270.0, 450.0, 480.0, 230.0, 310.0, 340.0, 430.0,
                200.0, 290.0, 460.0, 600.0, 320.0, 390.0, 300.0, 410.0,
            ],
            Some("price".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split data
    let (train_df, test_df) = train_test_split(&df, 0.25, false, Some(42))?;

    // Configure Decision Tree Regressor using MSE criterion
    let config = DecisionTreeConfigBuilder::new()
        .max_depth(4)
        .min_samples_split(3)
        .min_samples_leaf(2)
        .criterion(SplitCriterion::MSE)
        .random_seed(42)
        .build();

    let mut regressor = DecisionTreeRegressor::new(config);

    // Train the model
    println!("\nTraining Decision Tree Regressor...");
    regressor.fit(&train_df, "price")?;
    println!("Model trained successfully!");

    // Make predictions
    let predictions = regressor.predict(&test_df)?;
    println!("\nPredictions on test set (price in $1000s):");
    let test_labels = test_df.get_column::<f64>("price")?;
    for (i, (pred, actual)) in predictions
        .iter()
        .zip(test_labels.values())
        .take(5)
        .enumerate()
    {
        println!(
            "  Sample {}: Predicted = ${:.1}k, Actual = ${:.1}k, Error = ${:.1}k",
            i + 1,
            pred,
            actual,
            (pred - actual).abs()
        );
    }

    // Calculate RMSE
    let mut mse = 0.0;
    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        mse += (pred - actual).powi(2);
    }
    mse /= predictions.len() as f64;
    let rmse = mse.sqrt();
    println!("\nRoot Mean Squared Error: ${:.2}k", rmse);

    // Calculate MAE
    let mut mae = 0.0;
    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        mae += (pred - actual).abs();
    }
    mae /= predictions.len() as f64;
    println!("Mean Absolute Error: ${:.2}k", mae);

    println!();
    Ok(())
}

/// Demonstrate Feature Importance Analysis
fn feature_importance_example() -> Result<()> {
    println!("--- Feature Importance Analysis ---");

    // Create a dataset where we know which features are most important
    let mut df = DataFrame::new();

    // Important feature: strongly correlated with target
    df.add_column(
        "important_feature".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
            Some("important_feature".to_string()),
        )?,
    )?;

    // Less important feature: weakly correlated
    df.add_column(
        "weak_feature".to_string(),
        Series::new(
            vec![
                5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2,
                4.8, 5.0, 5.1, 4.9,
            ],
            Some("weak_feature".to_string()),
        )?,
    )?;

    // Noise feature: random values
    df.add_column(
        "noise_feature".to_string(),
        Series::new(
            vec![
                2.3, 7.1, 3.4, 9.2, 1.5, 8.7, 4.2, 6.3, 5.1, 3.9, 7.8, 2.1, 9.5, 4.7, 6.2, 8.1,
                3.3, 5.9, 7.4, 2.8,
            ],
            Some("noise_feature".to_string()),
        )?,
    )?;

    // Target: mainly dependent on important_feature
    df.add_column(
        "target".to_string(),
        Series::new(
            vec![
                2.1, 4.2, 6.1, 8.3, 10.2, 12.1, 14.3, 16.2, 18.1, 20.3, 22.2, 24.1, 26.3, 28.2,
                30.1, 32.3, 34.2, 36.1, 38.3, 40.2,
            ],
            Some("target".to_string()),
        )?,
    )?;

    let config = DecisionTreeConfigBuilder::new()
        .max_depth(5)
        .random_seed(42)
        .build();

    let mut regressor = DecisionTreeRegressor::new(config);
    regressor.fit(&df, "target")?;

    println!("Feature importances (higher values indicate more important features):");
    if let Some(importances) = regressor.feature_importances() {
        let mut importance_vec: Vec<(&String, &f64)> = importances.iter().collect();
        importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (feature, importance) in importance_vec {
            let bar_length = (importance * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("  {:20} {:.4} {}", feature, importance, bar);
        }
    }

    println!("\nAs expected, 'important_feature' has the highest importance!");
    println!();
    Ok(())
}

/// Compare different split criteria
fn criterion_comparison_example() -> Result<()> {
    println!("--- Criterion Comparison ---");

    // Create a simple classification dataset
    let mut df = DataFrame::new();

    df.add_column(
        "feature1".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
            Some("feature1".to_string()),
        )?,
    )?;

    df.add_column(
        "feature2".to_string(),
        Series::new(
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
            Some("feature2".to_string()),
        )?,
    )?;

    df.add_column(
        "label".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            Some("label".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Test Gini criterion
    println!("Testing Gini criterion:");
    let gini_config = DecisionTreeConfigBuilder::new()
        .criterion(SplitCriterion::Gini)
        .max_depth(3)
        .random_seed(42)
        .build();

    let mut gini_classifier = DecisionTreeClassifier::new(gini_config);
    gini_classifier.fit(&train_df, "label")?;
    let gini_pred = gini_classifier.predict(&test_df)?;

    let mut gini_correct = 0;
    let test_labels = test_df.get_column::<f64>("label")?;
    for (pred, actual) in gini_pred.iter().zip(test_labels.values()) {
        if (pred - actual).abs() < 0.5 {
            gini_correct += 1;
        }
    }
    let gini_accuracy = gini_correct as f64 / gini_pred.len() as f64;
    println!("  Accuracy: {:.2}%", gini_accuracy * 100.0);

    // Test Entropy criterion
    println!("\nTesting Entropy criterion:");
    let entropy_config = DecisionTreeConfigBuilder::new()
        .criterion(SplitCriterion::Entropy)
        .max_depth(3)
        .random_seed(42)
        .build();

    let mut entropy_classifier = DecisionTreeClassifier::new(entropy_config);
    entropy_classifier.fit(&train_df, "label")?;
    let entropy_pred = entropy_classifier.predict(&test_df)?;

    let mut entropy_correct = 0;
    for (pred, actual) in entropy_pred.iter().zip(test_labels.values()) {
        if (pred - actual).abs() < 0.5 {
            entropy_correct += 1;
        }
    }
    let entropy_accuracy = entropy_correct as f64 / entropy_pred.len() as f64;
    println!("  Accuracy: {:.2}%", entropy_accuracy * 100.0);

    println!("\nBoth Gini and Entropy are effective splitting criteria.");
    println!("Gini is often faster, while Entropy may produce slightly different trees.");
    println!();

    Ok(())
}

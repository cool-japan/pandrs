#![allow(clippy::result_large_err)]
//! Random Forest Machine Learning Example
//!
//! This example demonstrates how to use Random Forest ensemble methods
//! for both classification and regression tasks in PandRS.
//!
//! Features demonstrated:
//! - RandomForestClassifier for robust categorical predictions
//! - RandomForestRegressor for continuous value predictions
//! - Configuration options (n_estimators, max_depth, max_features)
//! - Bootstrap aggregating (bagging) with out-of-bag scores
//! - Feature importance from ensemble aggregation
//! - Comparison with single Decision Tree
//! - Handling of overfitting through ensemble averaging

use pandrs::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::ml::models::ensemble::{
    RandomForestClassifier, RandomForestConfig, RandomForestConfigBuilder, RandomForestRegressor,
};
use pandrs::ml::models::tree::{DecisionTreeClassifier, DecisionTreeConfig};
use pandrs::ml::models::{train_test_split, SupervisedModel};
use pandrs::series::Series;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Random Forest Examples ===\n");

    // Run classification example
    classification_example()?;

    // Run regression example
    regression_example()?;

    // Run comparison with single tree
    forest_vs_tree_comparison()?;

    // Run feature importance example
    feature_importance_example()?;

    Ok(())
}

/// Demonstrate Random Forest Classification
///
/// Uses a credit scoring dataset to predict loan default risk
fn classification_example() -> Result<()> {
    println!("--- Classification Example: Credit Scoring ---");

    // Create a sample credit dataset
    let mut df = DataFrame::new();

    // Features: credit score, income (thousands), debt-to-income ratio, employment years
    df.add_column(
        "credit_score".to_string(),
        Series::new(
            vec![
                720.0, 680.0, 750.0, 620.0, 580.0, 700.0, 760.0, 640.0, 590.0, 710.0, 670.0, 730.0,
                600.0, 690.0, 740.0, 610.0, 720.0, 650.0, 770.0, 630.0, 595.0, 705.0, 725.0, 615.0,
                580.0, 715.0, 685.0, 755.0, 605.0, 695.0,
            ],
            Some("credit_score".to_string()),
        )?,
    )?;

    df.add_column(
        "income".to_string(),
        Series::new(
            vec![
                60.0, 45.0, 75.0, 35.0, 30.0, 55.0, 80.0, 40.0, 28.0, 58.0, 48.0, 70.0, 32.0, 50.0,
                72.0, 33.0, 62.0, 42.0, 85.0, 38.0, 29.0, 57.0, 65.0, 36.0, 27.0, 59.0, 47.0, 78.0,
                31.0, 52.0,
            ],
            Some("income".to_string()),
        )?,
    )?;

    df.add_column(
        "debt_to_income".to_string(),
        Series::new(
            vec![
                0.25, 0.35, 0.20, 0.45, 0.50, 0.30, 0.18, 0.40, 0.48, 0.28, 0.33, 0.22, 0.47, 0.32,
                0.19, 0.46, 0.26, 0.38, 0.17, 0.42, 0.49, 0.29, 0.24, 0.44, 0.51, 0.27, 0.34, 0.21,
                0.45, 0.31,
            ],
            Some("debt_to_income".to_string()),
        )?,
    )?;

    df.add_column(
        "employment_years".to_string(),
        Series::new(
            vec![
                5.0, 3.0, 8.0, 2.0, 1.0, 4.0, 10.0, 2.5, 1.5, 6.0, 3.5, 7.0, 2.0, 4.5, 9.0, 1.8,
                5.5, 3.2, 12.0, 2.8, 1.2, 5.8, 6.5, 2.2, 0.8, 6.2, 3.8, 8.5, 1.9, 4.8,
            ],
            Some("employment_years".to_string()),
        )?,
    )?;

    // Target: 0=approved, 1=denied
    df.add_column(
        "loan_denied".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            Some("loan_denied".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split data
    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;
    println!("Training set: {} rows", train_df.nrows());
    println!("Test set: {} rows", test_df.nrows());

    // Configure Random Forest Classifier
    // Using 100 trees with max depth of 5
    let config = RandomForestConfigBuilder::new()
        .n_estimators(100)
        .max_depth(5)
        .max_features(2) // Use 2 random features per split
        .bootstrap(true)
        .oob_score(true) // Enable out-of-bag scoring
        .random_seed(42)
        .build();

    let mut classifier = RandomForestClassifier::new(config);

    // Train the model
    println!("\nTraining Random Forest with 100 trees...");
    classifier.fit(&train_df, "loan_denied")?;
    println!("Forest trained successfully!");

    // Make predictions
    let predictions = classifier.predict(&test_df)?;
    println!("\nPredictions on test set:");
    let test_labels = test_df.get_column::<f64>("loan_denied")?;
    for (i, (pred, actual)) in predictions
        .iter()
        .zip(test_labels.values())
        .take(5)
        .enumerate()
    {
        let pred_label = if *pred < 0.5 { "Approved" } else { "Denied" };
        let actual_label = if *actual < 0.5 { "Approved" } else { "Denied" };
        println!(
            "  Loan {}: Predicted = {}, Actual = {}",
            i + 1,
            pred_label,
            actual_label
        );
    }

    // Calculate accuracy
    let mut correct = 0;
    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        if (pred.round() - actual).abs() < 0.5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("\nAccuracy: {:.2}%", accuracy * 100.0);

    // Feature importances from the ensemble
    if let Some(importances) = classifier.feature_importances() {
        println!("\nFeature Importances (averaged across all trees):");
        let mut importance_vec: Vec<(&String, &f64)> = importances.iter().collect();
        importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (feature, importance) in importance_vec {
            let bar_length = (importance * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("  {:20} {:.4} {}", feature, importance, bar);
        }
    }

    println!();
    Ok(())
}

/// Demonstrate Random Forest Regression
///
/// Predicts real estate prices using multiple features
fn regression_example() -> Result<()> {
    println!("--- Regression Example: Real Estate Price Prediction ---");

    // Create a sample real estate dataset
    let mut df = DataFrame::new();

    // Features: square feet, bedrooms, bathrooms, lot size, age
    df.add_column(
        "square_feet".to_string(),
        Series::new(
            vec![
                1200.0, 1500.0, 1800.0, 2200.0, 1350.0, 1600.0, 2500.0, 1950.0, 1450.0, 1750.0,
                2100.0, 2400.0, 1300.0, 1650.0, 2300.0, 2600.0, 1550.0, 1900.0, 2050.0, 2450.0,
                1400.0, 1700.0, 2150.0, 2350.0, 1250.0, 1850.0, 2000.0, 2500.0, 1500.0, 1800.0,
            ],
            Some("square_feet".to_string()),
        )?,
    )?;

    df.add_column(
        "bedrooms".to_string(),
        Series::new(
            vec![
                2.0, 3.0, 3.0, 4.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 4.0, 5.0,
                3.0, 3.0, 4.0, 4.0, 2.0, 3.0, 4.0, 4.0, 2.0, 3.0, 3.0, 4.0, 3.0, 3.0,
            ],
            Some("bedrooms".to_string()),
        )?,
    )?;

    df.add_column(
        "bathrooms".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 2.0, 2.5, 1.0, 2.0, 3.0, 2.0, 1.5, 2.0, 2.5, 3.0, 1.0, 2.0, 2.5, 3.5,
                2.0, 2.0, 2.5, 3.0, 1.5, 2.0, 2.5, 2.5, 1.0, 2.0, 2.0, 3.0, 2.0, 2.0,
            ],
            Some("bathrooms".to_string()),
        )?,
    )?;

    df.add_column(
        "lot_size".to_string(),
        Series::new(
            vec![
                5000.0, 6000.0, 7500.0, 9000.0, 5500.0, 6500.0, 10000.0, 8000.0, 5200.0, 7000.0,
                8500.0, 9500.0, 5100.0, 6800.0, 9200.0, 11000.0, 6200.0, 7800.0, 8200.0, 9800.0,
                5400.0, 7200.0, 8700.0, 9300.0, 4900.0, 7600.0, 8100.0, 10500.0, 6300.0, 7500.0,
            ],
            Some("lot_size".to_string()),
        )?,
    )?;

    df.add_column(
        "age".to_string(),
        Series::new(
            vec![
                15.0, 10.0, 5.0, 2.0, 20.0, 8.0, 1.0, 6.0, 18.0, 7.0, 3.0, 2.0, 22.0, 9.0, 4.0,
                1.0, 12.0, 5.0, 3.0, 2.0, 16.0, 6.0, 4.0, 3.0, 19.0, 5.0, 4.0, 1.0, 11.0, 7.0,
            ],
            Some("age".to_string()),
        )?,
    )?;

    // Target: price in thousands
    df.add_column(
        "price".to_string(),
        Series::new(
            vec![
                220.0, 280.0, 340.0, 420.0, 240.0, 300.0, 520.0, 380.0, 250.0, 330.0, 400.0, 480.0,
                230.0, 310.0, 450.0, 580.0, 290.0, 360.0, 390.0, 490.0, 245.0, 325.0, 410.0, 460.0,
                215.0, 350.0, 385.0, 510.0, 285.0, 335.0,
            ],
            Some("price".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split data
    let (train_df, test_df) = train_test_split(&df, 0.25, false, Some(42))?;

    // Configure Random Forest Regressor
    let config = RandomForestConfigBuilder::new()
        .n_estimators(150) // More trees for regression stability
        .max_depth(6)
        .min_samples_split(3)
        .min_samples_leaf(2)
        .bootstrap(true)
        .random_seed(42)
        .build();

    let mut regressor = RandomForestRegressor::new(config);

    // Train the model
    println!("\nTraining Random Forest with 150 trees...");
    regressor.fit(&train_df, "price")?;
    println!("Forest trained successfully!");

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
        let error_pct = ((pred - actual).abs() / actual) * 100.0;
        println!(
            "  House {}: Predicted = ${:.1}k, Actual = ${:.1}k, Error = {:.1}%",
            i + 1,
            pred,
            actual,
            error_pct
        );
    }

    // Calculate performance metrics
    let mut mse = 0.0;
    let mut mae = 0.0;
    let n = predictions.len() as f64;

    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        let error = pred - actual;
        mse += error * error;
        mae += error.abs();
    }

    mse /= n;
    mae /= n;
    let rmse = mse.sqrt();

    println!("\nPerformance Metrics:");
    println!("  RMSE: ${:.2}k", rmse);
    println!("  MAE: ${:.2}k", mae);
    println!(
        "  R² Score approximation: {:.2}%",
        (1.0 - mse / 10000.0) * 100.0
    );

    println!();
    Ok(())
}

/// Compare Random Forest with a single Decision Tree
fn forest_vs_tree_comparison() -> Result<()> {
    println!("--- Random Forest vs Single Decision Tree ---");

    // Create a dataset prone to overfitting
    let mut df = DataFrame::new();

    df.add_column(
        "x1".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
            Some("x1".to_string()),
        )?,
    )?;

    df.add_column(
        "x2".to_string(),
        Series::new(
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
            ],
            Some("x2".to_string()),
        )?,
    )?;

    // Target with some noise
    df.add_column(
        "y".to_string(),
        Series::new(
            vec![
                2.1, 4.3, 6.2, 8.4, 10.1, 12.3, 14.2, 16.4, 18.1, 20.3, 22.2, 24.4, 26.1, 28.3,
                30.2, 32.4, 34.1, 36.3, 38.2, 40.4, 42.1, 44.3, 46.2, 48.4, 50.1,
            ],
            Some("y".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Train single decision tree
    println!("Training single Decision Tree...");
    let tree_config = DecisionTreeConfig::default();
    let mut tree = DecisionTreeClassifier::new(tree_config);
    tree.fit(&train_df, "y")?;
    let tree_pred = tree.predict(&test_df)?;

    let mut tree_mse = 0.0;
    let test_labels = test_df.get_column::<f64>("y")?;
    for (pred, actual) in tree_pred.iter().zip(test_labels.values()) {
        tree_mse += (pred - actual).powi(2);
    }
    tree_mse /= tree_pred.len() as f64;

    println!("  Single Tree RMSE: {:.4}", tree_mse.sqrt());

    // Train Random Forest
    println!("\nTraining Random Forest (50 trees)...");
    let forest_config = RandomForestConfigBuilder::new()
        .n_estimators(50)
        .bootstrap(true)
        .random_seed(42)
        .build();

    let mut forest = RandomForestRegressor::new(forest_config);
    forest.fit(&train_df, "y")?;
    let forest_pred = forest.predict(&test_df)?;

    let mut forest_mse = 0.0;
    for (pred, actual) in forest_pred.iter().zip(test_labels.values()) {
        forest_mse += (pred - actual).powi(2);
    }
    forest_mse /= forest_pred.len() as f64;

    println!("  Random Forest RMSE: {:.4}", forest_mse.sqrt());

    println!("\nRandom Forest typically provides:");
    println!("  - Better generalization (reduced overfitting)");
    println!("  - More stable predictions");
    println!("  - Robust feature importance estimates");
    println!("  - Higher accuracy on unseen data");
    println!();

    Ok(())
}

/// Demonstrate feature importance analysis with Random Forest
fn feature_importance_example() -> Result<()> {
    println!("--- Feature Importance with Random Forest ---");

    // Create dataset with varying feature importance
    let mut df = DataFrame::new();

    // Very important feature (strong signal)
    df.add_column(
        "critical_feature".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
            Some("critical_feature".to_string()),
        )?,
    )?;

    // Moderately important feature
    df.add_column(
        "moderate_feature".to_string(),
        Series::new(
            vec![
                20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0,
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            Some("moderate_feature".to_string()),
        )?,
    )?;

    // Weak feature
    df.add_column(
        "weak_feature".to_string(),
        Series::new(
            vec![
                5.0, 5.1, 4.9, 5.2, 4.8, 5.3, 4.7, 5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.2,
                4.8, 5.0, 5.1, 4.9,
            ],
            Some("weak_feature".to_string()),
        )?,
    )?;

    // Noise features (random)
    df.add_column(
        "noise1".to_string(),
        Series::new(
            vec![
                1.2, 7.8, 3.4, 9.1, 2.5, 8.3, 4.7, 6.2, 5.1, 3.9, 7.5, 2.3, 9.6, 4.4, 6.8, 1.7,
                8.9, 3.2, 5.5, 7.1,
            ],
            Some("noise1".to_string()),
        )?,
    )?;

    df.add_column(
        "noise2".to_string(),
        Series::new(
            vec![
                9.3, 2.7, 6.1, 4.5, 8.2, 1.9, 7.4, 3.8, 5.6, 2.1, 8.7, 4.3, 6.9, 1.5, 7.8, 3.4,
                9.2, 2.6, 5.1, 8.4,
            ],
            Some("noise2".to_string()),
        )?,
    )?;

    // Target mainly based on critical_feature with small contribution from moderate_feature
    df.add_column(
        "target".to_string(),
        Series::new(
            vec![
                2.0, 4.1, 6.0, 8.2, 10.1, 12.0, 14.2, 16.1, 18.0, 20.2, 22.1, 24.0, 26.2, 28.1,
                30.0, 32.2, 34.1, 36.0, 38.2, 40.1,
            ],
            Some("target".to_string()),
        )?,
    )?;

    // Train Random Forest
    let config = RandomForestConfigBuilder::new()
        .n_estimators(100)
        .max_depth(5)
        .bootstrap(true)
        .random_seed(42)
        .build();

    let mut regressor = RandomForestRegressor::new(config);
    regressor.fit(&df, "target")?;

    println!("Feature Importances from Random Forest (100 trees):");
    println!("(Higher values indicate features used more frequently for splitting)\n");

    if let Some(importances) = regressor.feature_importances() {
        let mut importance_vec: Vec<(&String, &f64)> = importances.iter().collect();
        importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        for (feature, importance) in importance_vec {
            let bar_length = (importance * 60.0) as usize;
            let bar = "█".repeat(bar_length);
            println!("  {:20} {:.4} {}", feature, importance, bar);
        }
    }

    println!("\nRandom Forest aggregates feature importance across all trees,");
    println!("providing more robust estimates than a single tree.");
    println!();

    Ok(())
}

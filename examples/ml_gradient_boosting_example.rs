#![allow(clippy::result_large_err)]
//! Gradient Boosting Machine Learning Example
//!
//! This example demonstrates how to use Gradient Boosting ensemble methods
//! for both classification and regression tasks in PandRS.
//!
//! Features demonstrated:
//! - GradientBoostingClassifier for classification problems
//! - GradientBoostingRegressor for regression problems
//! - Sequential boosting process and how it differs from Random Forest
//! - Configuration options (n_estimators, learning_rate, max_depth)
//! - Subsample and feature subsampling
//! - Feature importance from gradient boosting
//! - Comparison with Random Forest
//! - Learning rate tuning and its impact

use pandrs::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::ml::models::ensemble::{
    GradientBoostingClassifier, GradientBoostingConfigBuilder, GradientBoostingRegressor,
    RandomForestConfigBuilder, RandomForestRegressor,
};
use pandrs::ml::models::{train_test_split, SupervisedModel};
use pandrs::series::Series;

fn main() -> Result<()> {
    println!("=== Gradient Boosting Examples ===\n");

    // Run classification example
    classification_example()?;

    // Run regression example
    regression_example()?;

    // Run learning rate comparison
    learning_rate_comparison()?;

    // Run comparison with Random Forest
    boosting_vs_bagging_comparison()?;

    Ok(())
}

/// Demonstrate Gradient Boosting Classification
///
/// Uses customer churn dataset to predict whether customers will leave
fn classification_example() -> Result<()> {
    println!("--- Classification Example: Customer Churn Prediction ---");

    // Create a sample customer churn dataset
    let mut df = DataFrame::new();

    // Features: tenure (months), monthly charges, total charges, contract type (0=month, 1=year, 2=two-year)
    df.add_column(
        "tenure".to_string(),
        Series::new(
            vec![
                2.0, 12.0, 24.0, 6.0, 36.0, 18.0, 48.0, 9.0, 3.0, 60.0, 15.0, 30.0, 4.0, 21.0,
                42.0, 7.0, 27.0, 14.0, 54.0, 8.0, 33.0, 16.0, 45.0, 10.0, 39.0, 20.0, 51.0, 5.0,
                25.0, 13.0,
            ],
            Some("tenure".to_string()),
        )?,
    )?;

    df.add_column(
        "monthly_charges".to_string(),
        Series::new(
            vec![
                85.0, 65.0, 55.0, 90.0, 50.0, 70.0, 45.0, 80.0, 95.0, 40.0, 75.0, 60.0, 92.0, 68.0,
                48.0, 88.0, 58.0, 72.0, 42.0, 82.0, 52.0, 73.0, 47.0, 78.0, 55.0, 67.0, 44.0, 89.0,
                63.0, 76.0,
            ],
            Some("monthly_charges".to_string()),
        )?,
    )?;

    df.add_column(
        "total_charges".to_string(),
        Series::new(
            vec![
                170.0, 780.0, 1320.0, 540.0, 1800.0, 1260.0, 2160.0, 720.0, 285.0, 2400.0, 1125.0,
                1800.0, 368.0, 1428.0, 2016.0, 616.0, 1566.0, 1008.0, 2268.0, 656.0, 1716.0,
                1168.0, 2115.0, 780.0, 2145.0, 1340.0, 2244.0, 445.0, 1575.0, 988.0,
            ],
            Some("total_charges".to_string()),
        )?,
    )?;

    df.add_column(
        "contract_type".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0,
            ],
            Some("contract_type".to_string()),
        )?,
    )?;

    // Target: 0=retained, 1=churned
    df.add_column(
        "churned".to_string(),
        Series::new(
            vec![
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
            Some("churned".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split data
    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;
    println!("Training set: {} rows", train_df.nrows());
    println!("Test set: {} rows", test_df.nrows());

    // Configure Gradient Boosting Classifier
    // Key parameters:
    // - learning_rate: 0.1 (slower learning = better generalization)
    // - n_estimators: 100 (number of boosting iterations)
    // - max_depth: 3 (shallow trees to prevent overfitting)
    // - subsample: 0.8 (use 80% of data per iteration for regularization)
    let config = GradientBoostingConfigBuilder::new()
        .n_estimators(100)
        .learning_rate(0.1)
        .max_depth(3)
        .subsample(0.8)
        .random_seed(42)
        .build();

    let mut classifier = GradientBoostingClassifier::new(config);

    // Train the model
    println!("\nTraining Gradient Boosting Classifier...");
    println!("Using sequential boosting: each tree corrects errors of previous trees");
    classifier.fit(&train_df, "churned")?;
    println!("Model trained successfully with 100 boosting iterations!");

    // Make predictions
    let predictions = classifier.predict(&test_df)?;
    println!("\nPredictions on test set:");
    let test_labels = test_df.get_column::<f64>("churned")?;
    for (i, (pred, actual)) in predictions
        .iter()
        .zip(test_labels.values())
        .take(5)
        .enumerate()
    {
        let pred_label = if *pred < 0.5 { "Retained" } else { "Churned" };
        let actual_label = if *actual < 0.5 { "Retained" } else { "Churned" };
        let confidence = if *pred < 0.5 { 1.0 - pred } else { *pred };
        println!(
            "  Customer {}: {} (confidence: {:.1}%), Actual: {}",
            i + 1,
            pred_label,
            confidence * 100.0,
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

    // Feature importances
    if let Some(importances) = classifier.feature_importances() {
        println!("\nFeature Importances (from boosting):");
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

/// Demonstrate Gradient Boosting Regression
///
/// Predicts product demand for inventory optimization
fn regression_example() -> Result<()> {
    println!("--- Regression Example: Demand Forecasting ---");

    // Create a sample demand forecasting dataset
    let mut df = DataFrame::new();

    // Features: day of week (0-6), promotional activity (0-1), price, competitor price, weather score
    df.add_column(
        "day_of_week".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0,
                3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 2.0,
            ],
            Some("day_of_week".to_string()),
        )?,
    )?;

    df.add_column(
        "promotion".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            ],
            Some("promotion".to_string()),
        )?,
    )?;

    df.add_column(
        "price".to_string(),
        Series::new(
            vec![
                29.99, 29.99, 24.99, 29.99, 24.99, 19.99, 19.99, 29.99, 29.99, 24.99, 29.99, 24.99,
                19.99, 19.99, 29.99, 29.99, 24.99, 29.99, 24.99, 19.99, 19.99, 29.99, 29.99, 24.99,
                29.99, 24.99, 19.99, 19.99, 29.99, 29.99,
            ],
            Some("price".to_string()),
        )?,
    )?;

    df.add_column(
        "competitor_price".to_string(),
        Series::new(
            vec![
                31.0, 30.0, 27.0, 29.0, 26.0, 25.0, 24.0, 32.0, 31.0, 28.0, 30.0, 27.0, 26.0, 25.0,
                33.0, 32.0, 29.0, 31.0, 28.0, 27.0, 26.0, 34.0, 33.0, 30.0, 32.0, 29.0, 28.0, 27.0,
                35.0, 34.0,
            ],
            Some("competitor_price".to_string()),
        )?,
    )?;

    df.add_column(
        "weather_score".to_string(),
        Series::new(
            vec![
                7.0, 8.0, 6.0, 9.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 6.0, 7.0,
                8.0, 9.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 6.0, 7.0,
            ],
            Some("weather_score".to_string()),
        )?,
    )?;

    // Target: units sold
    df.add_column(
        "demand".to_string(),
        Series::new(
            vec![
                150.0, 155.0, 220.0, 160.0, 230.0, 280.0, 290.0, 145.0, 150.0, 210.0, 155.0, 225.0,
                275.0, 285.0, 140.0, 145.0, 205.0, 150.0, 220.0, 270.0, 280.0, 135.0, 140.0, 200.0,
                145.0, 215.0, 265.0, 275.0, 130.0, 135.0,
            ],
            Some("demand".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    // Split data
    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Configure Gradient Boosting Regressor
    let config = GradientBoostingConfigBuilder::new()
        .n_estimators(150)
        .learning_rate(0.05) // Smaller learning rate for more stable predictions
        .max_depth(4)
        .subsample(0.8)
        .random_seed(42)
        .build();

    let mut regressor = GradientBoostingRegressor::new(config);

    // Train the model
    println!("\nTraining Gradient Boosting Regressor...");
    println!("Learning rate: 0.05 (slower but more accurate)");
    println!("Boosting iterations: 150");
    regressor.fit(&train_df, "demand")?;
    println!("Model trained successfully!");

    // Make predictions
    let predictions = regressor.predict(&test_df)?;
    println!("\nPredictions on test set (units):");
    let test_labels = test_df.get_column::<f64>("demand")?;
    for (i, (pred, actual)) in predictions
        .iter()
        .zip(test_labels.values())
        .take(5)
        .enumerate()
    {
        let error_pct = ((pred - actual).abs() / actual) * 100.0;
        println!(
            "  Day {}: Predicted = {:.0} units, Actual = {:.0} units, Error = {:.1}%",
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
    println!("  RMSE: {:.2} units", rmse);
    println!("  MAE: {:.2} units", mae);

    println!();
    Ok(())
}

/// Compare different learning rates
fn learning_rate_comparison() -> Result<()> {
    println!("--- Learning Rate Impact on Performance ---");

    // Create a simple dataset
    let mut df = DataFrame::new();

    df.add_column(
        "x1".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
            Some("x1".to_string()),
        )?,
    )?;

    df.add_column(
        "x2".to_string(),
        Series::new(
            vec![
                20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0,
                6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
            ],
            Some("x2".to_string()),
        )?,
    )?;

    df.add_column(
        "y".to_string(),
        Series::new(
            vec![
                3.0, 6.1, 9.0, 12.2, 15.1, 18.0, 21.2, 24.1, 27.0, 30.2, 33.1, 36.0, 39.2, 42.1,
                45.0, 48.2, 51.1, 54.0, 57.2, 60.1,
            ],
            Some("y".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    let learning_rates = vec![0.01, 0.1, 0.5];

    println!("Testing different learning rates with same number of estimators (50):\n");

    for lr in learning_rates {
        let config = GradientBoostingConfigBuilder::new()
            .n_estimators(50)
            .learning_rate(lr)
            .max_depth(3)
            .random_seed(42)
            .build();

        let mut regressor = GradientBoostingRegressor::new(config);
        regressor.fit(&train_df, "y")?;
        let predictions = regressor.predict(&test_df)?;

        let test_labels = test_df.get_column::<f64>("y")?;
        let mut mse = 0.0;
        for (pred, actual) in predictions.iter().zip(test_labels.values()) {
            mse += (pred - actual).powi(2);
        }
        mse /= predictions.len() as f64;

        println!("Learning Rate = {:.2}", lr);
        println!("  RMSE: {:.4}", mse.sqrt());
        println!("  Interpretation:");
        if lr < 0.05 {
            println!("    - Very slow learning, may need more iterations");
        } else if lr < 0.2 {
            println!("    - Good balance, recommended for most cases");
        } else {
            println!("    - Fast learning, risk of overshooting optimal solution");
        }
        println!();
    }

    println!("Lower learning rate + more iterations = better generalization");
    println!("Higher learning rate + fewer iterations = faster training but may overfit");
    println!();

    Ok(())
}

/// Compare Gradient Boosting with Random Forest
fn boosting_vs_bagging_comparison() -> Result<()> {
    println!("--- Gradient Boosting vs Random Forest ---");

    // Create a dataset
    let mut df = DataFrame::new();

    df.add_column(
        "feature1".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
            Some("feature1".to_string()),
        )?,
    )?;

    df.add_column(
        "feature2".to_string(),
        Series::new(
            vec![
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
            ],
            Some("feature2".to_string()),
        )?,
    )?;

    df.add_column(
        "target".to_string(),
        Series::new(
            vec![
                2.2, 4.3, 6.1, 8.4, 10.2, 12.1, 14.3, 16.2, 18.1, 20.3, 22.2, 24.1, 26.3, 28.2,
                30.1, 32.3, 34.2, 36.1, 38.3, 40.2, 42.1, 44.3, 46.2, 48.4, 50.1,
            ],
            Some("target".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Train Gradient Boosting
    println!("Training Gradient Boosting (sequential boosting)...");
    let gb_config = GradientBoostingConfigBuilder::new()
        .n_estimators(50)
        .learning_rate(0.1)
        .max_depth(3)
        .random_seed(42)
        .build();

    let mut gb_regressor = GradientBoostingRegressor::new(gb_config);
    gb_regressor.fit(&train_df, "target")?;
    let gb_predictions = gb_regressor.predict(&test_df)?;

    let test_labels = test_df.get_column::<f64>("target")?;
    let mut gb_mse = 0.0;
    for (pred, actual) in gb_predictions.iter().zip(test_labels.values()) {
        gb_mse += (pred - actual).powi(2);
    }
    gb_mse /= gb_predictions.len() as f64;

    println!("  RMSE: {:.4}", gb_mse.sqrt());

    // Train Random Forest
    println!("\nTraining Random Forest (parallel bagging)...");
    let rf_config = RandomForestConfigBuilder::new()
        .n_estimators(50)
        .max_depth(3)
        .bootstrap(true)
        .random_seed(42)
        .build();

    let mut rf_regressor = RandomForestRegressor::new(rf_config);
    rf_regressor.fit(&train_df, "target")?;
    let rf_predictions = rf_regressor.predict(&test_df)?;

    let mut rf_mse = 0.0;
    for (pred, actual) in rf_predictions.iter().zip(test_labels.values()) {
        rf_mse += (pred - actual).powi(2);
    }
    rf_mse /= rf_predictions.len() as f64;

    println!("  RMSE: {:.4}", rf_mse.sqrt());

    println!("\nKey Differences:");
    println!("  Gradient Boosting:");
    println!("    - Sequential: each tree corrects previous errors");
    println!("    - Often higher accuracy");
    println!("    - More prone to overfitting");
    println!("    - Sensitive to hyperparameters");
    println!("    - Cannot be parallelized easily");
    println!("\n  Random Forest:");
    println!("    - Parallel: trees are independent");
    println!("    - More robust to overfitting");
    println!("    - Less sensitive to hyperparameters");
    println!("    - Easily parallelizable");
    println!("    - Generally faster training");
    println!();

    Ok(())
}

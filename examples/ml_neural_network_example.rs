#![allow(clippy::result_large_err)]
//! Neural Network (Multi-Layer Perceptron) Machine Learning Example
//!
//! This example demonstrates how to use Multi-Layer Perceptron (MLP) neural networks
//! for both classification and regression tasks in PandRS.
//!
//! Features demonstrated:
//! - MLPClassifier for classification problems (binary and multi-class)
//! - MLPRegressor for regression problems
//! - Network architecture configuration (layers, neurons, activation functions)
//! - Training loop with backpropagation and early stopping
//! - Feature scaling and normalization
//! - Hyperparameter tuning (learning rate, layers, batch size)
//! - Model evaluation metrics
//! - Training loss visualization
//! - XOR problem (classic neural network example)
//! - Deep vs shallow network comparison

use pandrs::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::ml::models::neural::{Activation, MLPClassifier, MLPConfigBuilder, MLPRegressor};
use pandrs::ml::models::{train_test_split, ModelEvaluator, SupervisedModel};
use pandrs::series::Series;

fn main() -> Result<()> {
    println!("=== Neural Network (MLP) Examples ===\n");

    // Run binary classification example
    binary_classification_example()?;

    // Run multi-class classification example
    multiclass_classification_example()?;

    // Run regression example
    regression_example()?;

    // Run XOR problem (classic neural network example)
    xor_problem_example()?;

    // Run hyperparameter comparison
    hyperparameter_tuning_example()?;

    // Run deep vs shallow network comparison
    architecture_comparison()?;

    // Run feature scaling demonstration
    feature_scaling_example()?;

    Ok(())
}

/// Demonstrate Binary Classification with MLP
///
/// Predicts customer churn using neural network
fn binary_classification_example() -> Result<()> {
    println!("--- Binary Classification: Customer Churn Prediction ---");

    // Create customer churn dataset
    let mut df = DataFrame::new();

    // Features: account age (months), monthly spending, support tickets, satisfaction score
    df.add_column(
        "account_age".to_string(),
        Series::new(
            vec![
                3.0, 12.0, 24.0, 6.0, 36.0, 18.0, 48.0, 9.0, 2.0, 60.0, 15.0, 30.0, 5.0, 21.0,
                42.0, 8.0, 27.0, 14.0, 54.0, 7.0, 33.0, 16.0, 45.0, 11.0, 39.0, 20.0, 51.0, 4.0,
                25.0, 13.0, 37.0, 19.0, 46.0, 10.0, 29.0, 22.0, 53.0, 17.0, 41.0, 23.0,
            ],
            Some("account_age".to_string()),
        )?,
    )?;

    df.add_column(
        "monthly_spending".to_string(),
        Series::new(
            vec![
                45.0, 65.0, 85.0, 50.0, 95.0, 75.0, 105.0, 55.0, 40.0, 110.0, 70.0, 90.0, 48.0,
                80.0, 100.0, 52.0, 88.0, 72.0, 108.0, 54.0, 92.0, 74.0, 102.0, 68.0, 96.0, 82.0,
                106.0, 46.0, 86.0, 71.0, 94.0, 78.0, 104.0, 66.0, 89.0, 84.0, 109.0, 76.0, 98.0,
                83.0,
            ],
            Some("monthly_spending".to_string()),
        )?,
    )?;

    df.add_column(
        "support_tickets".to_string(),
        Series::new(
            vec![
                5.0, 2.0, 1.0, 4.0, 0.0, 1.0, 0.0, 3.0, 6.0, 0.0, 2.0, 1.0, 5.0, 1.0, 0.0, 4.0,
                1.0, 2.0, 0.0, 3.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 5.0, 1.0, 3.0, 0.0, 2.0,
                0.0, 3.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0,
            ],
            Some("support_tickets".to_string()),
        )?,
    )?;

    df.add_column(
        "satisfaction".to_string(),
        Series::new(
            vec![
                3.0, 7.0, 9.0, 4.0, 10.0, 8.0, 10.0, 5.0, 2.0, 10.0, 7.0, 9.0, 3.0, 8.0, 10.0, 4.0,
                8.0, 7.0, 10.0, 5.0, 9.0, 7.0, 10.0, 6.0, 9.0, 8.0, 10.0, 3.0, 8.0, 6.0, 10.0, 7.0,
                10.0, 6.0, 8.0, 8.0, 10.0, 7.0, 9.0, 8.0,
            ],
            Some("satisfaction".to_string()),
        )?,
    )?;

    // Target: 0=retained, 1=churned
    df.add_column(
        "churned".to_string(),
        Series::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            Some("churned".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());
    println!("Features: account_age, monthly_spending, support_tickets, satisfaction");

    // Split data (70% train, 30% test)
    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;
    println!("Training set: {} rows", train_df.nrows());
    println!("Test set: {} rows", test_df.nrows());

    // Configure MLP Classifier
    // Network architecture: 4 inputs -> 16 hidden (ReLU) -> 8 hidden (ReLU) -> 1 output (Sigmoid)
    println!("\nNetwork Architecture:");
    println!("  Input layer: 4 neurons (features)");
    println!("  Hidden layer 1: 16 neurons (ReLU activation)");
    println!("  Hidden layer 2: 8 neurons (ReLU activation)");
    println!("  Output layer: 1 neuron (Sigmoid activation for binary classification)");

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![16, 8])
        .hidden_activation(Activation::ReLU)
        .learning_rate(0.01)
        .n_epochs(500)
        .batch_size(8)
        .early_stopping_patience(Some(50))
        .random_seed(42)
        .verbose(false)
        .build();

    let mut classifier = MLPClassifier::new(config);

    // Train the model
    println!("\nTraining Neural Network...");
    println!("Learning rate: 0.01");
    println!("Batch size: 8");
    println!("Max epochs: 500 (with early stopping)");

    classifier.fit(&train_df, "churned")?;
    println!("Training completed!");

    // Show training progress
    let loss_history = classifier.training_loss_history();
    println!("\nTraining Loss History:");
    println!(
        "  Initial loss: {:.6}",
        loss_history.first().unwrap_or(&0.0)
    );
    println!("  Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));
    println!("  Epochs trained: {}", loss_history.len());
    println!(
        "  Loss reduction: {:.2}%",
        (1.0 - loss_history.last().unwrap_or(&1.0) / loss_history.first().unwrap_or(&1.0)) * 100.0
    );

    // Make predictions with probabilities
    let predictions = classifier.predict(&test_df)?;
    let probabilities = classifier.predict_proba(&test_df)?;

    println!("\nPredictions on test set:");
    let test_labels = test_df.get_column::<f64>("churned")?;
    for (i, ((pred, proba), actual)) in predictions
        .iter()
        .zip(probabilities.iter())
        .zip(test_labels.values())
        .take(8)
        .enumerate()
    {
        let pred_label = if *pred < 0.5 { "Retained" } else { "Churned" };
        let actual_label = if *actual < 0.5 { "Retained" } else { "Churned" };
        let confidence = proba[1]; // Probability of churn
        let status = if (pred.round() - actual).abs() < 0.5 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  Customer {:2}: {} (prob: {:.1}%) | Actual: {} {}",
            i + 1,
            pred_label,
            confidence * 100.0,
            actual_label,
            status
        );
    }

    // Evaluate model
    let metrics = classifier.evaluate(&test_df, "churned")?;
    if let Some(accuracy) = metrics.get_metric("accuracy") {
        println!("\nModel Performance:");
        println!("  Accuracy: {:.2}%", accuracy * 100.0);
    }

    // Calculate additional metrics
    let mut tp = 0; // True Positives
    let mut fp = 0; // False Positives
    let mut _tn = 0; // True Negatives
    let mut fn_count = 0; // False Negatives

    for (pred, actual) in predictions.iter().zip(test_labels.values()) {
        let pred_class = pred.round();
        if pred_class == 1.0 && *actual == 1.0 {
            tp += 1;
        } else if pred_class == 1.0 && *actual == 0.0 {
            fp += 1;
        } else if pred_class == 0.0 && *actual == 0.0 {
            _tn += 1;
        } else {
            fn_count += 1;
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("  Precision: {:.2}%", precision * 100.0);
    println!("  Recall: {:.2}%", recall * 100.0);
    println!("  F1-Score: {:.2}%", f1 * 100.0);
    println!();

    Ok(())
}

/// Demonstrate Multi-class Classification
///
/// Predicts customer segment (Bronze/Silver/Gold)
fn multiclass_classification_example() -> Result<()> {
    println!("--- Multi-class Classification: Customer Segmentation ---");

    let mut df = DataFrame::new();

    // Features representing customer behavior
    df.add_column(
        "purchase_frequency".to_string(),
        Series::new(
            vec![
                2.0, 8.0, 15.0, 3.0, 12.0, 5.0, 18.0, 7.0, 14.0, 4.0, 9.0, 16.0, 6.0, 11.0, 17.0,
                2.5, 8.5, 13.0, 4.5, 10.0, 19.0, 3.5, 7.5, 15.5, 5.5, 9.5, 16.5, 6.5, 12.5, 20.0,
            ],
            Some("purchase_frequency".to_string()),
        )?,
    )?;

    df.add_column(
        "avg_order_value".to_string(),
        Series::new(
            vec![
                25.0, 75.0, 150.0, 30.0, 95.0, 45.0, 180.0, 65.0, 120.0, 35.0, 80.0, 160.0, 50.0,
                90.0, 170.0, 28.0, 78.0, 110.0, 38.0, 85.0, 190.0, 32.0, 70.0, 140.0, 48.0, 88.0,
                165.0, 55.0, 100.0, 200.0,
            ],
            Some("avg_order_value".to_string()),
        )?,
    )?;

    df.add_column(
        "engagement_score".to_string(),
        Series::new(
            vec![
                3.0, 7.0, 9.5, 3.5, 8.0, 5.0, 10.0, 6.5, 8.5, 4.0, 7.5, 9.0, 5.5, 7.8, 9.8, 3.2,
                7.2, 8.2, 4.2, 7.6, 10.0, 3.8, 6.8, 9.2, 5.2, 7.7, 9.5, 6.0, 8.0, 10.0,
            ],
            Some("engagement_score".to_string()),
        )?,
    )?;

    // Target: 0=Bronze, 1=Silver, 2=Gold
    df.add_column(
        "segment".to_string(),
        Series::new(
            vec![
                0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0,
            ],
            Some("segment".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());
    println!("Target classes: 0=Bronze, 1=Silver, 2=Gold");

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Configure MLP for multi-class classification
    // Network uses Softmax activation in output layer for probability distribution
    println!("\nNetwork Architecture:");
    println!("  Input layer: 3 neurons");
    println!("  Hidden layer 1: 12 neurons (ReLU)");
    println!("  Hidden layer 2: 6 neurons (ReLU)");
    println!("  Output layer: 3 neurons (Softmax for probability distribution)");

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![12, 6])
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Softmax)
        .learning_rate(0.01)
        .n_epochs(400)
        .batch_size(6)
        .early_stopping_patience(Some(40))
        .random_seed(42)
        .build();

    let mut classifier = MLPClassifier::new(config);

    println!("\nTraining multi-class neural network...");
    classifier.fit(&train_df, "segment")?;

    let predictions = classifier.predict(&test_df)?;
    let probabilities = classifier.predict_proba(&test_df)?;

    println!("\nPredictions on test set:");
    let test_labels = test_df.get_column::<f64>("segment")?;
    let segment_names = ["Bronze", "Silver", "Gold"];

    for (i, ((pred, proba), actual)) in predictions
        .iter()
        .zip(probabilities.iter())
        .zip(test_labels.values())
        .take(6)
        .enumerate()
    {
        let pred_idx = pred.round() as usize;
        let actual_idx = *actual as usize;
        let status = if pred_idx == actual_idx { "✓" } else { "✗" };

        println!(
            "  Customer {}: {} (Bronze: {:.0}%, Silver: {:.0}%, Gold: {:.0}%) | Actual: {} {}",
            i + 1,
            segment_names.get(pred_idx).unwrap_or(&"Unknown"),
            proba.first().unwrap_or(&0.0) * 100.0,
            proba.get(1).unwrap_or(&0.0) * 100.0,
            proba.get(2).unwrap_or(&0.0) * 100.0,
            segment_names.get(actual_idx).unwrap_or(&"Unknown"),
            status
        );
    }

    let metrics = classifier.evaluate(&test_df, "segment")?;
    if let Some(accuracy) = metrics.get_metric("accuracy") {
        println!("\nAccuracy: {:.2}%", accuracy * 100.0);
    }

    println!();
    Ok(())
}

/// Demonstrate Regression with MLP
///
/// Predicts product demand based on various factors
fn regression_example() -> Result<()> {
    println!("--- Regression: Product Demand Forecasting ---");

    let mut df = DataFrame::new();

    // Features: price, advertising spend, competitor price, seasonality index
    df.add_column(
        "price".to_string(),
        Series::new(
            vec![
                29.99, 24.99, 19.99, 34.99, 27.99, 22.99, 32.99, 26.99, 21.99, 30.99, 25.99, 20.99,
                35.99, 28.99, 23.99, 31.99, 27.49, 22.49, 33.99, 26.49, 21.49, 29.49, 25.49, 24.49,
                28.49, 23.49, 32.49, 27.99, 22.99, 30.49,
            ],
            Some("price".to_string()),
        )?,
    )?;

    df.add_column(
        "advertising".to_string(),
        Series::new(
            vec![
                500.0, 1500.0, 2500.0, 300.0, 1200.0, 2000.0, 400.0, 1400.0, 2200.0, 600.0, 1600.0,
                2400.0, 350.0, 1300.0, 2100.0, 450.0, 1350.0, 2150.0, 380.0, 1380.0, 2300.0, 550.0,
                1550.0, 1700.0, 1250.0, 2050.0, 420.0, 1100.0, 2250.0, 520.0,
            ],
            Some("advertising".to_string()),
        )?,
    )?;

    df.add_column(
        "competitor_price".to_string(),
        Series::new(
            vec![
                32.0, 27.0, 22.0, 37.0, 30.0, 25.0, 35.0, 29.0, 24.0, 33.0, 28.0, 23.0, 38.0, 31.0,
                26.0, 34.0, 29.5, 24.5, 36.0, 28.5, 23.5, 31.5, 27.5, 26.5, 30.5, 25.5, 34.5, 30.0,
                25.0, 32.5,
            ],
            Some("competitor_price".to_string()),
        )?,
    )?;

    df.add_column(
        "seasonality".to_string(),
        Series::new(
            vec![
                0.8, 1.2, 1.5, 0.7, 1.0, 1.3, 0.75, 1.1, 1.4, 0.85, 1.15, 1.45, 0.72, 1.05, 1.35,
                0.78, 1.08, 1.38, 0.74, 1.12, 1.42, 0.82, 1.18, 1.25, 1.02, 1.32, 0.76, 0.95, 1.4,
                0.8,
            ],
            Some("seasonality".to_string()),
        )?,
    )?;

    // Target: demand (units sold)
    df.add_column(
        "demand".to_string(),
        Series::new(
            vec![
                150.0, 280.0, 420.0, 120.0, 250.0, 380.0, 140.0, 270.0, 400.0, 160.0, 290.0, 430.0,
                115.0, 260.0, 390.0, 145.0, 265.0, 395.0, 130.0, 275.0, 410.0, 155.0, 285.0, 320.0,
                255.0, 385.0, 135.0, 235.0, 415.0, 152.0,
            ],
            Some("demand".to_string()),
        )?,
    )?;

    println!("Dataset shape: {} rows, {} columns", df.nrows(), df.ncols());

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    // Configure MLP Regressor with deeper network for complex patterns
    println!("\nNetwork Architecture:");
    println!("  Input layer: 4 neurons");
    println!("  Hidden layer 1: 32 neurons (ReLU)");
    println!("  Hidden layer 2: 16 neurons (ReLU)");
    println!("  Hidden layer 3: 8 neurons (ReLU)");
    println!("  Output layer: 1 neuron (Linear activation for regression)");

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![32, 16, 8])
        .hidden_activation(Activation::ReLU)
        .learning_rate(0.001)
        .n_epochs(800)
        .batch_size(5)
        .early_stopping_patience(Some(100))
        .random_seed(42)
        .build();

    let mut regressor = MLPRegressor::new(config);

    println!("\nTraining neural network regressor...");
    regressor.fit(&train_df, "demand")?;

    let predictions = regressor.predict(&test_df)?;

    println!("\nDemand Predictions:");
    let test_labels = test_df.get_column::<f64>("demand")?;
    for (i, (pred, actual)) in predictions
        .iter()
        .zip(test_labels.values())
        .take(8)
        .enumerate()
    {
        let error = (pred - actual).abs();
        let error_pct = (error / actual) * 100.0;
        println!("  Day {:2}: Predicted = {:5.0} units | Actual = {:5.0} units | Error = {:5.1} ({:.1}%)",
                 i + 1, pred, actual, error, error_pct);
    }

    // Calculate metrics
    let metrics = regressor.evaluate(&test_df, "demand")?;
    println!("\nPerformance Metrics:");
    if let Some(rmse) = metrics.get_metric("rmse") {
        println!("  RMSE: {:.2} units", rmse);
    }
    if let Some(r2) = metrics.get_metric("r2") {
        println!("  R² Score: {:.4}", r2);
    }

    // Show training progress
    let loss_history = regressor.training_loss_history();
    println!("\nTraining Progress:");
    println!("  Epochs: {}", loss_history.len());
    println!(
        "  Initial loss: {:.4}",
        loss_history.first().unwrap_or(&0.0)
    );
    println!("  Final loss: {:.4}", loss_history.last().unwrap_or(&0.0));

    println!();
    Ok(())
}

/// Demonstrate XOR Problem
///
/// Classic neural network example showing non-linear decision boundaries
/// XOR is not linearly separable, requiring hidden layers
fn xor_problem_example() -> Result<()> {
    println!("--- XOR Problem: Classic Neural Network Example ---");
    println!("XOR is a non-linearly separable problem that requires hidden layers");
    println!("Truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0\n");

    let mut df = DataFrame::new();

    // XOR dataset with some noise to make it more realistic
    df.add_column(
        "x1".to_string(),
        Series::new(
            vec![
                0.0, 0.0, 1.0, 1.0, 0.1, 0.05, 0.9, 0.95, 0.02, 0.08, 0.92, 0.98, 0.0, 0.0, 1.0,
                1.0,
            ],
            Some("x1".to_string()),
        )?,
    )?;

    df.add_column(
        "x2".to_string(),
        Series::new(
            vec![
                0.0, 1.0, 0.0, 1.0, 0.1, 0.9, 0.05, 0.95, 0.05, 0.92, 0.08, 0.98, 0.02, 0.98, 0.02,
                0.97,
            ],
            Some("x2".to_string()),
        )?,
    )?;

    df.add_column(
        "y".to_string(),
        Series::new(
            vec![
                0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ],
            Some("y".to_string()),
        )?,
    )?;

    println!("Dataset: 16 samples with noise");

    // Network with hidden layer is essential for XOR
    println!("\nNetwork Architecture:");
    println!("  Input: 2 neurons (x1, x2)");
    println!("  Hidden: 8 neurons (ReLU) - essential for learning XOR");
    println!("  Output: 1 neuron (Sigmoid)");

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![8])
        .hidden_activation(Activation::ReLU)
        .learning_rate(0.1)
        .n_epochs(1000)
        .batch_size(4)
        .early_stopping_patience(None)
        .random_seed(42)
        .build();

    let mut classifier = MLPClassifier::new(config);

    println!("\nTraining on XOR problem...");
    classifier.fit(&df, "y")?;

    let predictions = classifier.predict(&df)?;

    println!("\nXOR Results:");
    let labels = df.get_column::<f64>("y")?;
    let x1_col = df.get_column::<f64>("x1")?;
    let x2_col = df.get_column::<f64>("x2")?;

    for (((pred, actual), x1), x2) in predictions
        .iter()
        .zip(labels.values())
        .zip(x1_col.values())
        .zip(x2_col.values())
        .take(8)
    {
        let pred_class = pred.round();
        let status = if (pred_class - actual).abs() < 0.5 {
            "✓"
        } else {
            "✗"
        };
        println!(
            "  ({:.1}, {:.1}) -> Predicted: {:.0} (confidence: {:.1}%) | Actual: {:.0} {}",
            x1,
            x2,
            pred_class,
            if pred_class == 1.0 {
                *pred
            } else {
                1.0 - *pred
            } * 100.0,
            actual,
            status
        );
    }

    let mut correct = 0;
    for (pred, actual) in predictions.iter().zip(labels.values()) {
        if (pred.round() - actual).abs() < 0.5 {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / predictions.len() as f64;
    println!("\nAccuracy: {:.1}%", accuracy * 100.0);
    println!("Note: Without hidden layers, this would be ~50% (random guessing)");
    println!();

    Ok(())
}

/// Compare different hyperparameters
fn hyperparameter_tuning_example() -> Result<()> {
    println!("--- Hyperparameter Tuning: Learning Rate Comparison ---");

    // Simple regression dataset
    let mut df = DataFrame::new();

    df.add_column(
        "x1".to_string(),
        Series::new((1..=30).map(|x| x as f64).collect(), Some("x1".to_string()))?,
    )?;
    df.add_column(
        "x2".to_string(),
        Series::new(
            (1..=30).map(|x| (31 - x) as f64).collect(),
            Some("x2".to_string()),
        )?,
    )?;
    df.add_column(
        "y".to_string(),
        Series::new(
            vec![
                5.2, 10.1, 15.3, 20.2, 25.0, 30.1, 35.3, 40.2, 45.1, 50.0, 55.2, 60.1, 65.0, 70.3,
                75.2, 80.1, 85.0, 90.2, 95.1, 100.3, 105.2, 110.0, 115.1, 120.3, 125.2, 130.1,
                135.0, 140.2, 145.1, 150.0,
            ],
            Some("y".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    let learning_rates = vec![0.001, 0.01, 0.1];

    println!("Testing learning rates: 0.001, 0.01, 0.1");
    println!("Network: [20, 10] hidden layers, 300 epochs\n");

    for lr in learning_rates {
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![20, 10])
            .learning_rate(lr)
            .n_epochs(300)
            .batch_size(6)
            .early_stopping_patience(Some(50))
            .random_seed(42)
            .build();

        let mut regressor = MLPRegressor::new(config);
        regressor.fit(&train_df, "y")?;

        let metrics = regressor.evaluate(&test_df, "y")?;
        let loss_history = regressor.training_loss_history();

        println!("Learning Rate: {}", lr);
        if let Some(rmse) = metrics.get_metric("rmse") {
            println!("  RMSE: {:.4}", rmse);
        }
        if let Some(r2) = metrics.get_metric("r2") {
            println!("  R²: {:.4}", r2);
        }
        println!("  Epochs trained: {}", loss_history.len());
        println!("  Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));

        if lr < 0.005 {
            println!("  Analysis: Very slow learning, may need more epochs");
        } else if lr < 0.05 {
            println!("  Analysis: Good balance, stable convergence");
        } else {
            println!("  Analysis: Fast learning, may be unstable");
        }
        println!();
    }

    Ok(())
}

/// Compare deep vs shallow networks
fn architecture_comparison() -> Result<()> {
    println!("--- Network Architecture: Deep vs Shallow ---");

    let mut df = DataFrame::new();

    // Complex non-linear pattern
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
                2.0, 4.0, 3.0, 5.0, 6.0, 8.0, 7.0, 9.0, 10.0, 12.0, 11.0, 13.0, 14.0, 16.0, 15.0,
                17.0, 18.0, 20.0, 19.0, 21.0,
            ],
            Some("x2".to_string()),
        )?,
    )?;

    df.add_column(
        "x3".to_string(),
        Series::new(
            vec![
                1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
                16.5, 17.5, 18.5, 19.5, 20.5,
            ],
            Some("x3".to_string()),
        )?,
    )?;

    df.add_column(
        "y".to_string(),
        Series::new(
            vec![
                5.0, 12.0, 18.0, 26.0, 35.0, 45.0, 54.0, 65.0, 77.0, 88.0, 100.0, 113.0, 125.0,
                139.0, 152.0, 166.0, 181.0, 195.0, 210.0, 225.0,
            ],
            Some("y".to_string()),
        )?,
    )?;

    let (train_df, test_df) = train_test_split(&df, 0.3, false, Some(42))?;

    println!("Comparing three architectures:\n");

    // Shallow network
    println!("1. Shallow Network: [50]");
    let shallow_config = MLPConfigBuilder::new()
        .hidden_layers(vec![50])
        .learning_rate(0.01)
        .n_epochs(400)
        .batch_size(5)
        .random_seed(42)
        .build();

    let mut shallow = MLPRegressor::new(shallow_config);
    shallow.fit(&train_df, "y")?;
    let shallow_metrics = shallow.evaluate(&test_df, "y")?;

    if let Some(rmse) = shallow_metrics.get_metric("rmse") {
        println!("   RMSE: {:.4}", rmse);
    }
    if let Some(r2) = shallow_metrics.get_metric("r2") {
        println!("   R²: {:.4}", r2);
    }

    // Medium network
    println!("\n2. Medium Network: [32, 16]");
    let medium_config = MLPConfigBuilder::new()
        .hidden_layers(vec![32, 16])
        .learning_rate(0.01)
        .n_epochs(400)
        .batch_size(5)
        .random_seed(42)
        .build();

    let mut medium = MLPRegressor::new(medium_config);
    medium.fit(&train_df, "y")?;
    let medium_metrics = medium.evaluate(&test_df, "y")?;

    if let Some(rmse) = medium_metrics.get_metric("rmse") {
        println!("   RMSE: {:.4}", rmse);
    }
    if let Some(r2) = medium_metrics.get_metric("r2") {
        println!("   R²: {:.4}", r2);
    }

    // Deep network
    println!("\n3. Deep Network: [64, 32, 16, 8]");
    let deep_config = MLPConfigBuilder::new()
        .hidden_layers(vec![64, 32, 16, 8])
        .learning_rate(0.01)
        .n_epochs(400)
        .batch_size(5)
        .random_seed(42)
        .build();

    let mut deep = MLPRegressor::new(deep_config);
    deep.fit(&train_df, "y")?;
    let deep_metrics = deep.evaluate(&test_df, "y")?;

    if let Some(rmse) = deep_metrics.get_metric("rmse") {
        println!("   RMSE: {:.4}", rmse);
    }
    if let Some(r2) = deep_metrics.get_metric("r2") {
        println!("   R²: {:.4}", r2);
    }

    println!("\nKey Insights:");
    println!("  - Shallow networks: Faster training, good for simple patterns");
    println!("  - Deep networks: Better for complex patterns, more parameters");
    println!("  - Trade-off: Complexity vs overfitting risk");
    println!();

    Ok(())
}

/// Demonstrate feature scaling importance
fn feature_scaling_example() -> Result<()> {
    println!("--- Feature Scaling: Impact on Neural Networks ---");
    println!("Neural networks are sensitive to feature scales");
    println!("Features should be normalized to similar ranges\n");

    // Dataset with different scales
    let mut df = DataFrame::new();

    // Feature 1: Small scale (0-10)
    df.add_column(
        "small_scale".to_string(),
        Series::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
                7.5, 8.5, 9.5, 10.0,
            ],
            Some("small_scale".to_string()),
        )?,
    )?;

    // Feature 2: Large scale (1000-10000)
    df.add_column(
        "large_scale".to_string(),
        Series::new(
            vec![
                1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0,
                1500.0, 2500.0, 3500.0, 4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 9500.0, 10000.0,
            ],
            Some("large_scale".to_string()),
        )?,
    )?;

    // Target depends on both features
    df.add_column(
        "target".to_string(),
        Series::new(
            vec![
                10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 15.0, 25.0, 35.0,
                45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 100.0,
            ],
            Some("target".to_string()),
        )?,
    )?;

    println!("Dataset features:");
    println!("  small_scale: range [1, 10]");
    println!("  large_scale: range [1000, 10000] <- 1000x larger!");

    // Create normalized version
    let mut df_normalized = DataFrame::new();

    // Min-max normalization: (x - min) / (max - min)
    let small_min = 1.0;
    let small_max = 10.0;
    let large_min = 1000.0;
    let large_max = 10000.0;

    let small_normalized: Vec<f64> = df
        .get_column::<f64>("small_scale")?
        .values()
        .iter()
        .map(|x| (x - small_min) / (small_max - small_min))
        .collect();

    let large_normalized: Vec<f64> = df
        .get_column::<f64>("large_scale")?
        .values()
        .iter()
        .map(|x| (x - large_min) / (large_max - large_min))
        .collect();

    df_normalized.add_column(
        "small_scale".to_string(),
        Series::new(small_normalized, Some("small_scale".to_string()))?,
    )?;
    df_normalized.add_column(
        "large_scale".to_string(),
        Series::new(large_normalized, Some("large_scale".to_string()))?,
    )?;

    let target_values = df.get_column::<f64>("target")?.values().to_vec();
    df_normalized.add_column(
        "target".to_string(),
        Series::new(target_values, Some("target".to_string()))?,
    )?;

    println!("\nAfter normalization: both features in range [0, 1]\n");

    let (train_norm, test_norm) = train_test_split(&df_normalized, 0.3, false, Some(42))?;

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![16, 8])
        .learning_rate(0.01)
        .n_epochs(300)
        .batch_size(5)
        .random_seed(42)
        .build();

    let mut regressor = MLPRegressor::new(config);
    regressor.fit(&train_norm, "target")?;

    let metrics = regressor.evaluate(&test_norm, "target")?;

    println!("Results with normalized features:");
    if let Some(rmse) = metrics.get_metric("rmse") {
        println!("  RMSE: {:.4}", rmse);
    }
    if let Some(r2) = metrics.get_metric("r2") {
        println!("  R²: {:.4}", r2);
    }

    let loss_history = regressor.training_loss_history();
    println!("  Training epochs: {}", loss_history.len());
    println!("  Final loss: {:.6}", loss_history.last().unwrap_or(&0.0));

    println!("\nBest Practices:");
    println!("  1. Min-Max Scaling: (x - min) / (max - min) -> [0, 1]");
    println!("  2. Standardization: (x - mean) / std -> zero mean, unit variance");
    println!("  3. Normalize all features to similar ranges");
    println!("  4. Apply same scaling to training and test data");
    println!("  5. Store scaling parameters for production deployment");
    println!();

    Ok(())
}

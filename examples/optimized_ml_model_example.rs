#[cfg(feature = "optimized")]
use pandrs::column::{Column, Float64Column, StringColumn};
#[cfg(feature = "optimized")]
use pandrs::error::{Error, Result};
#[cfg(feature = "optimized")]
use pandrs::ml::metrics::classification::{
    accuracy_score, f1_score, precision_score, recall_score,
};
#[cfg(feature = "optimized")]
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
#[cfg(feature = "optimized")]
use pandrs::ml::models::{LinearRegression, LogisticRegression};
// Model persistence has been removed in the reorganization
// use pandrs::ml::models::model_persistence::ModelPersistence;
#[cfg(feature = "optimized")]
use pandrs::OptimizedDataFrame;
#[cfg(feature = "optimized")]
use rand::Rng;

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_ml_model_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("Example of PandRS Model Training and Evaluation (OptimizedDataFrame version)");
    println!("==========================================");

    // Regression model example
    regression_example()?;

    // Classification model example
    classification_example()?;

    // Model selection and evaluation example
    model_selection_example()?;

    // Model persistence example - temporarily disabled due to API changes
    // model_persistence_example()?;

    Ok(())
}

#[cfg(feature = "optimized")]
// Regression model example
#[allow(clippy::result_large_err)]
fn regression_example() -> Result<()> {
    println!("\n==== Regression Model Example ====");

    // Generate regression data
    let reg_df = create_regression_data()?;
    println!("Regression Data Sample:");
    println!("{:?}", reg_df);

    // Split into training and test data (mock implementation)
    let train_size = (reg_df.row_count() as f64 * 0.7) as usize;
    let test_size = reg_df.row_count() - train_size;
    println!(
        "Training Data Size: {}, Test Data Size: {}",
        train_size, test_size
    );

    // Normally, the data would be split, but here we use the original data
    let _train_df = &reg_df;
    let test_df = &reg_df;

    // List of features
    let _features = ["feature1", "feature2", "feature3"];
    let target = "target";

    // Create and train a linear regression model
    let _model = LinearRegression::new();

    // For now, let's comment out or skip this example since the OptimizedDataFrame
    // doesn't directly convert to new DataFrame, and would need custom implementation
    println!("\nLinear Regression Model Results (skipped for now):");
    println!("Need to adapt OptimizedDataFrame to use with new DataFrame API");

    // Placeholder for predictions
    let predictions = vec![0.0; test_df.row_count()];

    // Model evaluation
    // Convert column view to f64 array
    let y_true: Vec<f64> = {
        // First, get the target column
        let target_col = test_df.column(target)?;
        let float_col = target_col
            .as_float64()
            .ok_or(Error::Type("Expected float column".to_string()))?;

        (0..test_df.row_count())
            .filter_map(|i| float_col.get(i).ok().flatten())
            .collect()
    };

    let mse = mean_squared_error(&y_true, &predictions)?;
    let r2 = r2_score(&y_true, &predictions)?;

    println!("\nModel Evaluation:");
    println!("MSE: {}", mse);
    println!("R^2: {}", r2);

    // Feature engineering and regression using pipeline
    println!("\nRegression using Feature Engineering Pipeline (skipped for now):");
    println!("Pipeline API has been updated and needs to be adapted for this example");

    // Placeholder for poly predictions
    let poly_predictions = vec![0.0; test_df.row_count()];

    // Placeholder for truth values
    let poly_y_true = vec![0.0; test_df.row_count()];

    let poly_mse = mean_squared_error(&poly_y_true, &poly_predictions)?;
    let poly_r2 = r2_score(&poly_y_true, &poly_predictions)?;

    println!("Evaluation of Linear Regression with Polynomial Features:");
    println!("MSE: {}", poly_mse);
    println!("R^2: {}", poly_r2);

    Ok(())
}

#[cfg(feature = "optimized")]
// Classification model example
#[allow(clippy::result_large_err)]
fn classification_example() -> Result<()> {
    println!("\n==== Classification Model Example ====");

    // Generate classification data
    let cls_df = create_classification_data()?;
    println!("Classification Data Sample:");
    println!("{:?}", cls_df);

    // Split into training and test data (mock implementation)
    let train_size = (cls_df.row_count() as f64 * 0.7) as usize;
    let test_size = cls_df.row_count() - train_size;
    println!(
        "Training Data Size: {}, Test Data Size: {}",
        train_size, test_size
    );

    // Normally, the data would be split, but here we use the original data
    let _train_df = &cls_df;
    let test_df = &cls_df;

    // List of features
    let _features = ["feature1", "feature2"];
    let target = "target";

    // Create and train a logistic regression model
    let _model = LogisticRegression::new();

    // For now, we'll skip the actual model training since we need to adapt the OptimizedDataFrame
    println!("\nLogistic Regression Model Results (skipped for now):");
    println!("Need to adapt OptimizedDataFrame to use with new DataFrame API");

    // Placeholder for predictions
    let predictions = vec![0.0; test_df.row_count()];

    // Model evaluation
    // Convert column view to bool array
    let y_true: Vec<bool> = {
        // First, get the target column
        let target_col = test_df.column(target)?;
        let string_col = target_col
            .as_string()
            .ok_or(Error::Type("Expected string column".to_string()))?;

        (0..test_df.row_count())
            .filter_map(|i| string_col.get(i).ok().flatten().map(|s| s == "1"))
            .collect()
    };

    // Convert predictions from f64 to bool
    let pred_bool: Vec<bool> = predictions.iter().map(|&val| val > 0.5).collect();

    let accuracy = accuracy_score(&y_true, &pred_bool)?;
    let precision = precision_score(&y_true, &pred_bool)?;
    let recall = recall_score(&y_true, &pred_bool)?;
    let f1 = f1_score(&y_true, &pred_bool)?;

    println!("\nModel Evaluation:");
    println!("Accuracy: {}", accuracy);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
    println!("F1 Score: {}", f1);

    // Probability predictions - skipped for now
    println!("\nProbability Predictions Sample (skipped for now):");
    println!("Probability prediction would go here, but API has changed");

    Ok(())
}

#[cfg(feature = "optimized")]
// Model selection and evaluation example
#[allow(clippy::result_large_err)]
fn model_selection_example() -> Result<()> {
    println!("\n==== Model Selection and Evaluation Example ====");

    // Generate regression data
    let _reg_df = create_regression_data()?;

    // Model evaluation using cross-validation
    println!("\nCross-Validation (5-fold) Results:");
    println!("Note: LinearRegression does not implement the Clone trait, so cross-validation cannot be performed");

    // The following code is disabled because the Clone trait is required
    // let model = LinearRegression::new();
    // let _features = vec!["feature1", "feature2", "feature3"];
    // let _target = "target";
    // let cv_scores = cross_val_score(&model, &_reg_df, _target, &_features, 5)?;
    // println!("Scores for each fold: {:?}", cv_scores);
    // println!("Average Score: {}", cv_scores.iter().sum::<f64>() / cv_scores.len() as f64);

    Ok(())
}

// Function no longer works with new API, temporarily commented out
/*
// Model persistence example
fn model_persistence_example() -> Result<(), PandRSError> {
    println!("\n==== Model Persistence Example ====");

    // Generate regression data
    let reg_df = create_regression_data()?;

    // Use the original DataFrame directly
    let df_to_use = &reg_df;

    // List of features
    let features = vec!["feature1", "feature2", "feature3"];
    let target = "target";

    // Train the model
    let mut model = LinearRegression::new();
    model.fit(df_to_use, target, &features)?;

    // Save the model
    let model_path = "/tmp/linear_regression_model.json";
    model.save_model(model_path)?;
    println!("Model saved: {}", model_path);

    // Load the model
    let loaded_model = LinearRegression::load_model(model_path)?;
    println!("Model loaded");

    // Verify the parameters of the loaded model
    println!("Loaded Model Coefficients: {:?}", loaded_model.coefficients());
    println!("Loaded Model Intercept: {}", loaded_model.intercept());

    // Verify predictions
    let orig_pred = model.predict(df_to_use)?;
    let loaded_pred = loaded_model.predict(df_to_use)?;

    // Extract only the first 5 elements
    let orig_pred_sample: Vec<&f64> = orig_pred.iter().take(5).collect();
    let loaded_pred_sample: Vec<&f64> = loaded_pred.iter().take(5).collect();

    println!("Original Model Predictions: {:?}", orig_pred_sample);
    println!("Loaded Model Predictions: {:?}", loaded_pred_sample);

    Ok(())
}
*/

#[cfg(feature = "optimized")]
// Generate regression data
#[allow(clippy::result_large_err)]
fn create_regression_data() -> Result<OptimizedDataFrame> {
    let mut rng = rand::rng();

    // Generate 100 rows of data
    let n = 100;

    // 3 features
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..100.0)).collect();
    let feature3: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..15.0)).collect();

    // Target variable with a linear relationship: y = 2*x1 + 0.5*x2 - 1.5*x3 + noise
    let target: Vec<f64> = feature1
        .iter()
        .zip(feature2.iter())
        .zip(feature3.iter())
        .map(|((x1, x2), x3)| 2.0 * x1 + 0.5 * x2 - 1.5 * x3 + rng.random_range(-5.0..5.0))
        .collect();

    // Create OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();

    // Add features
    let feature1_col = Float64Column::with_name(feature1, "feature1");
    df.add_column("feature1", Column::Float64(feature1_col))?;

    let feature2_col = Float64Column::with_name(feature2, "feature2");
    df.add_column("feature2", Column::Float64(feature2_col))?;

    let feature3_col = Float64Column::with_name(feature3, "feature3");
    df.add_column("feature3", Column::Float64(feature3_col))?;

    // Add target variable
    let target_col = Float64Column::with_name(target, "target");
    df.add_column("target", Column::Float64(target_col))?;

    Ok(df)
}

#[cfg(feature = "optimized")]
// Generate classification data
#[allow(clippy::result_large_err)]
fn create_classification_data() -> Result<OptimizedDataFrame> {
    let mut rng = rand::rng();

    // Generate 100 rows of data
    let n = 100;

    // 2 features
    let feature1: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();
    let feature2: Vec<f64> = (0..n).map(|_| rng.random_range(-5.0..5.0)).collect();

    // Binary classification using a logistic model
    // P(y=1) = sigmoid(1.5*x1 - 2*x2)
    let target: Vec<String> = feature1
        .iter()
        .zip(feature2.iter())
        .map(|(x1, x2)| {
            let z = 1.5 * x1 - 2.0 * x2;
            let p = 1.0 / (1.0 + (-z).exp());

            if rng.random::<f64>() < p {
                "1".to_string()
            } else {
                "0".to_string()
            }
        })
        .collect();

    // Create OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();

    // Add features
    let feature1_col = Float64Column::with_name(feature1, "feature1");
    df.add_column("feature1", Column::Float64(feature1_col))?;

    let feature2_col = Float64Column::with_name(feature2, "feature2");
    df.add_column("feature2", Column::Float64(feature2_col))?;

    // Add target variable
    let target_col = StringColumn::with_name(target, "target");
    df.add_column("target", Column::String(target_col))?;

    Ok(df)
}

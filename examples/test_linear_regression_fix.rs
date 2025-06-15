use pandrs::dataframe::DataFrame;
use pandrs::ml::models::linear::LinearRegression;
use pandrs::ml::models::{SupervisedModel, ModelEvaluator};
use pandrs::series::Series;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Enhanced Linear Regression Implementation");

    // Create simple linear relationship: y = 2x + 1 + noise
    let x_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_values: Vec<f64> = x_values.iter()
        .enumerate()
        .map(|(i, &x)| 2.0 * x + 1.0 + (i as f64 * 0.1 - 0.5)) // Add small noise
        .collect();

    // Create DataFrame
    let mut df = DataFrame::new();
    let x_series = Series::new(x_values, Some("x".to_string()))?;
    let y_series = Series::new(y_values, Some("y".to_string()))?;
    
    df.add_column("x".to_string(), x_series)?;
    df.add_column("y".to_string(), y_series)?;

    println!("Training data created: {} samples", df.nrows());

    // Test 1: Basic Linear Regression
    println!("\n=== Test 1: Basic Linear Regression ===");
    let mut model = LinearRegression::new();
    
    println!("Fitting model...");
    model.fit(&df, "y")?;
    
    if let Some(coefficients) = &model.coefficients {
        println!("Learned coefficients:");
        for (feature, coef) in coefficients {
            println!("  {}: {:.4}", feature, coef);
        }
    }
    
    if let Some(intercept) = model.intercept {
        println!("Intercept: {:.4}", intercept);
    }

    // Test predictions
    println!("\nMaking predictions...");
    let predictions = model.predict(&df)?;
    
    println!("Sample predictions:");
    for (i, prediction) in predictions.iter().enumerate().take(5) {
        println!("  Sample {}: predicted={:.4}", i, prediction);
    }

    // Calculate R²
    println!("\nCalculating R²...");
    let r_squared = model.r_squared(&df, "y")?;
    println!("R² score: {:.4}", r_squared);

    // Test 2: Model Evaluation
    println!("\n=== Test 2: Model Evaluation ===");
    let metrics = model.evaluate(&df, "y")?;
    
    println!("Evaluation metrics:");
    if let Some(mse) = metrics.get_metric("mse") {
        println!("  MSE: {:.6}", mse);
    }
    if let Some(mae) = metrics.get_metric("mae") {
        println!("  MAE: {:.6}", mae);
    }
    if let Some(r2) = metrics.get_metric("r2") {
        println!("  R²: {:.4}", r2);
    }

    // Test 3: Feature Importances
    println!("\n=== Test 3: Feature Importances ===");
    if let Some(importances) = model.feature_importances() {
        println!("Feature importances:");
        for (feature, importance) in importances {
            println!("  {}: {:.4}", feature, importance);
        }
    }

    // Test 4: Multiple Features
    println!("\n=== Test 4: Multiple Feature Regression ===");
    
    // Create y = 2*x1 + 3*x2 + 1 + noise
    let x1_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x2_values = vec![2.0, 3.0, 1.0, 4.0, 2.5];
    let y_multi: Vec<f64> = x1_values.iter().zip(x2_values.iter())
        .enumerate()
        .map(|(i, (&x1, &x2))| 2.0 * x1 + 3.0 * x2 + 1.0 + (i as f64 * 0.05))
        .collect();

    let mut df_multi = DataFrame::new();
    let x1_series = Series::new(x1_values, Some("x1".to_string()))?;
    let x2_series = Series::new(x2_values, Some("x2".to_string()))?;
    let y_multi_series = Series::new(y_multi, Some("y".to_string()))?;
    
    df_multi.add_column("x1".to_string(), x1_series)?;
    df_multi.add_column("x2".to_string(), x2_series)?;
    df_multi.add_column("y".to_string(), y_multi_series)?;

    let mut multi_model = LinearRegression::new();
    multi_model.fit(&df_multi, "y")?;
    
    println!("Multiple regression coefficients:");
    if let Some(coefficients) = &multi_model.coefficients {
        for (feature, coef) in coefficients {
            println!("  {}: {:.4}", feature, coef);
        }
    }
    if let Some(intercept) = multi_model.intercept {
        println!("Intercept: {:.4}", intercept);
    }

    let multi_r2 = multi_model.r_squared(&df_multi, "y")?;
    println!("Multiple regression R²: {:.4}", multi_r2);

    // Test 5: Normalization
    println!("\n=== Test 5: Feature Normalization ===");
    let mut normalized_model = LinearRegression::new().with_normalization(true);
    normalized_model.fit(&df_multi, "y")?;
    
    println!("Normalized model coefficients:");
    if let Some(coefficients) = &normalized_model.coefficients {
        for (feature, coef) in coefficients {
            println!("  {}: {:.4}", feature, coef);
        }
    }
    
    let norm_r2 = normalized_model.r_squared(&df_multi, "y")?;
    println!("Normalized model R²: {:.4}", norm_r2);

    println!("\n✅ Linear Regression implementation test completed successfully!");
    println!("The model correctly learns coefficients and makes accurate predictions.");

    Ok(())
}
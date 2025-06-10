#[cfg(feature = "optimized")]
use pandrs::ml::metrics::classification::{
    accuracy_score, f1_score, precision_score, recall_score,
};
#[cfg(feature = "optimized")]
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score, root_mean_squared_error};
#[cfg(feature = "optimized")]
use pandrs::stats;
#[cfg(feature = "optimized")]
use pandrs::*;
#[cfg(feature = "optimized")]
use rand::Rng;

// Example code to demonstrate a pipeline that only supports OptimizedDataFrame
// with a simple example using a regular DataFrame

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_ml_pipeline_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
fn main() -> Result<()> {
    println!("PandRS Machine Learning Features Example");
    println!("========================");

    // Create sample data
    let df = create_sample_data()?;

    println!("Original DataFrame:");
    println!("{:?}", df);

    // Linear regression (using the stats module)
    let x_columns = &["value1", "value2"];
    let y_column = "target";

    // Perform regression analysis
    let model = stats::linear_regression(&df, y_column, x_columns)?;

    println!("\nLinear Regression Results:");
    println!("Coefficients: {:?}", model.coefficients);
    println!("Intercept: {}", model.intercept);
    println!("R-squared: {}", model.r_squared);

    // Get values and compare actual and predicted values
    let x1 = 2.0;
    let x2 = 50.0;

    // Calculate predicted value (manually implemented)
    let predicted = model.intercept + model.coefficients[0] * x1 + model.coefficients[1] * x2;

    // Calculate actual value (without noise)
    let actual = 2.0 * x1 + 0.5 * x2;

    println!("\nPrediction Example:");
    println!("When x1 = {}, x2 = {}", x1, x2);
    println!("Predicted Value: {}", predicted);
    println!("Actual Value (without noise): {}", actual);
    println!("Error: {}", (predicted - actual).abs());

    // Regression metrics example
    let y_true = vec![3.0, 4.0, 5.0, 6.0, 7.0];
    let y_pred = vec![2.8, 4.2, 5.1, 5.8, 7.4];

    println!("\nRegression Metrics:");
    println!("MSE: {}", mean_squared_error(&y_true, &y_pred)?);
    println!("RMSE: {}", root_mean_squared_error(&y_true, &y_pred)?);
    println!("R2: {}", r2_score(&y_true, &y_pred)?);

    // Classification metrics example
    let y_true_class = vec![true, false, true, true, false];
    let y_pred_class = vec![true, false, false, true, true];

    println!("\nClassification Metrics:");
    println!(
        "Accuracy: {}",
        accuracy_score(&y_true_class, &y_pred_class)?
    );
    println!(
        "Precision: {}",
        precision_score(&y_true_class, &y_pred_class)?
    );
    println!("Recall: {}", recall_score(&y_true_class, &y_pred_class)?);
    println!("F1 Score: {}", f1_score(&y_true_class, &y_pred_class)?);

    // Introduction to machine learning pipeline using OptimizedDataFrame
    println!("\nMachine Learning Pipeline Features:");
    println!("- StandardScaler: Standardize numerical data");
    println!("- MinMaxScaler: Normalize numerical data to 0-1 range");
    println!("- OneHotEncoder: Convert categorical data to dummy variables");
    println!("- PolynomialFeatures: Generate polynomial features");
    println!("- Imputer: Fill missing values");
    println!("- FeatureSelector: Select features");
    println!("- Pipeline: Chain transformation steps");

    Ok(())
}

#[cfg(feature = "optimized")]
// Create sample data
fn create_sample_data() -> Result<DataFrame> {
    let mut rng = rand::rng();

    // Generate 10 rows of data
    let n = 10;

    // Categorical data
    let categories = vec!["A", "B", "C"];
    let cat_data: Vec<String> = (0..n)
        .map(|_| categories[rng.random_range(0..categories.len())].to_string())
        .collect();

    // Numerical data
    let value1: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
    let value2: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..100.0)).collect();

    // Target variable (linear relationship + noise)
    let target: Vec<f64> = value1
        .iter()
        .zip(value2.iter())
        .map(|(x, y)| 2.0 * x + 0.5 * y + rng.random_range(-5.0..5.0))
        .collect();

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "category".to_string(),
        Series::new(cat_data, Some("category".to_string()))?,
    )?;
    df.add_column(
        "value1".to_string(),
        Series::new(value1, Some("value1".to_string()))?,
    )?;
    df.add_column(
        "value2".to_string(),
        Series::new(value2, Some("value2".to_string()))?,
    )?;
    df.add_column(
        "target".to_string(),
        Series::new(target, Some("target".to_string()))?,
    )?;

    Ok(df)
}

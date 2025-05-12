// Minimal sample example (OptimizedDataFrame version)
#[cfg(feature = "optimized")]
use pandrs::column::ColumnTrait; // Add ColumnTrait import
#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::ml::preprocessing::{MinMaxScaler, StandardScaler};
#[cfg(feature = "optimized")]
use pandrs::ml::Transformer;
#[cfg(feature = "optimized")]
use pandrs::{Column, Float64Column, Int64Column, OptimizedDataFrame}; // Removed unused UnsupervisedModel import

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_ml_basic_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
fn main() -> Result<()> {
    println!("PandRS Machine Learning Module Basic Example (OptimizedDataFrame version)");
    println!("===================================================");

    // Create sample data
    let mut df = OptimizedDataFrame::new();

    // Feature 1: Float64 type
    let feature1 = Float64Column::with_name(vec![1.0, 2.0, 3.0, 4.0, 5.0], "feature1");
    df.add_column("feature1", Column::Float64(feature1))?;

    // Feature 2: Float64 type
    let feature2 = Float64Column::with_name(vec![10.0, 20.0, 30.0, 40.0, 50.0], "feature2");
    df.add_column("feature2", Column::Float64(feature2))?;

    // Feature 3: Int64 type
    let feature3 = Int64Column::with_name(vec![100, 200, 300, 400, 500], "feature3");
    df.add_column("feature3", Column::Int64(feature3))?;

    println!("Original DataFrame:");
    println!("{:?}", df);

    // Check basic information
    println!("\nBasic Information:");
    println!("Number of rows: {}", df.row_count());
    println!("Number of columns: {}", df.column_count());
    println!("Column names: {:?}", df.column_names());

    // Try using StandardScaler
    println!("\n=== Applying StandardScaler ===");
    let mut scaler = StandardScaler::new();
    scaler = scaler.with_columns(vec!["feature1".to_string(), "feature2".to_string()]);

    // Apply directly to the OptimizedDataFrame
    let scaled_df = scaler.fit_transform(&df)?;
    println!("Standardized DataFrame:");
    println!("{:?}", scaled_df);

    // The result is already an OptimizedDataFrame
    let opt_scaled_df = scaled_df;
    if let Ok(col) = opt_scaled_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nStandardized values of the feature1 column:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("Row {}: {:?}", i, val);
                }
            }
        }
    }

    // Try using MinMaxScaler
    println!("\n=== Applying MinMaxScaler ===");
    let mut minmax = MinMaxScaler::new();
    minmax = minmax.with_columns(vec!["feature1".to_string(), "feature2".to_string()]);
    minmax = minmax.with_range(0.0, 1.0);
    // Note: MinMaxScaler expects an OptimizedDataFrame, not a regular DataFrame
    let normalized_df = minmax.fit_transform(&df)?; // Using original OptimizedDataFrame directly
    println!("Normalized DataFrame:");
    println!("{:?}", normalized_df);

    // Since the result is already an OptimizedDataFrame, we don't need to convert
    let opt_normalized_df = normalized_df;
    if let Ok(col) = opt_normalized_df.column("feature1") {
        if let Some(float_col) = col.as_float64() {
            println!("\nNormalized values of the feature1 column:");
            for i in 0..float_col.len() {
                if let Ok(val) = float_col.get(i) {
                    println!("Row {}: {:?}", i, val);
                }
            }
        }
    }

    // Try using a pipeline
    println!("\n=== Using a Pipeline ===");

    // In the new API, we need to create a pipeline with individual transformers
    println!("Using individual transformers instead of Pipeline:");

    // First apply the StandardScaler to feature1
    let mut scaler1 = StandardScaler::new();
    scaler1 = scaler1.with_columns(vec!["feature1".to_string()]);
    let temp_df = scaler1.fit_transform(&df)?; // Using the original OptimizedDataFrame

    // Then apply the MinMaxScaler to feature2
    let mut scaler2 = MinMaxScaler::new();
    scaler2 = scaler2.with_columns(vec!["feature2".to_string()]);
    scaler2 = scaler2.with_range(0.0, 1.0);
    let result_df = scaler2.fit_transform(&temp_df)?;
    println!("DataFrame after applying the pipeline:");
    println!("{:?}", result_df);

    println!("\n === Sample Execution Complete ===");
    Ok(())
}

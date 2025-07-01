#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::{Column, Float64Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

// Provide a fallback main function when the optimized feature is not enabled
#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_basic_usage --features \"optimized\"");
}

// Translated Japanese comments and strings into English
#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS Optimized Basic Usage Example ===");

    // Create an optimized DataFrame
    println!("\n=== Creating a DataFrame ===");
    let mut df = OptimizedDataFrame::new();

    // Create and add an integer column
    let ages = Int64Column::new(vec![30, 25, 40]);
    df.add_column("age", Column::Int64(ages))?;

    // Create and add a floating-point column
    let heights = Float64Column::new(vec![180.0, 175.0, 182.0]);
    df.add_column("height", Column::Float64(heights))?;

    // Create and add a string column
    let names = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
    ]);
    df.add_column("name", Column::String(names))?;

    println!("DataFrame: {:?}", df);
    println!("Number of columns: {}", df.column_count());
    println!("Number of rows: {}", df.row_count());
    println!("Column names: {:?}", df.column_names());

    // Column operations
    println!("\n=== Statistics for the Age Column ===");
    let age_col = df.column("age")?;
    if let Some(int_col) = age_col.as_int64() {
        println!("Sum: {}", int_col.sum());
        println!("Mean: {:.2}", int_col.mean().unwrap_or(0.0));
        println!("Min: {}", int_col.min().unwrap_or(0));
        println!("Max: {}", int_col.max().unwrap_or(0));
    }

    // Test saving to and loading from CSV
    let file_path = "optimized_example_data.csv";
    df.to_csv(file_path, true)?;
    println!("\nSaved to CSV file: {}", file_path);

    // Load from CSV
    match OptimizedDataFrame::from_csv(file_path, true) {
        Ok(loaded_df) => {
            println!("DataFrame loaded from CSV: {:?}", loaded_df);
            println!("Number of columns: {}", loaded_df.column_count());
            println!("Number of rows: {}", loaded_df.row_count());
            println!("Column names: {:?}", loaded_df.column_names());
        }
        Err(e) => {
            println!("Failed to load CSV: {:?}", e);
        }
    }

    // Example of lazy evaluation
    println!("\n=== Example of Lazy Evaluation ===");
    let lazy_df = LazyFrame::new(df);

    // Select rows where the name is "Alice" or "Bob"
    let result = lazy_df.select(&["name", "age", "height"]).execute()?;

    println!("DataFrame with selected columns only: {:?}", result);

    println!("\n=== Sample Complete ===");
    Ok(())
}

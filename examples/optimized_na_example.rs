#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::{Column, Float64Column, Int64Column, OptimizedDataFrame, StringColumn};

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_na_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Optimized Version: Support for Missing Values ===\n");

    // Create an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // Create an integer column with missing values (OptimizedDataFrame handles missing values differently)
    // For Int64Column, missing values are handled using a null_mask
    let int_values = vec![10, 20, 0, 40, 0];
    let int_nulls = vec![false, false, true, false, true]; // Set indices 2 and 4 to NULL
    let int_data = Int64Column::with_nulls(int_values, int_nulls);

    df.add_column("numbers", Column::Int64(int_data))?;

    // Set missing values in a floating-point column
    let float_values = vec![1.1, 2.2, 0.0, 4.4, 0.0];
    let float_nulls = vec![false, false, true, false, true]; // Set indices 2 and 4 to NULL
    let float_data = Float64Column::with_nulls(float_values, float_nulls);

    df.add_column("floats", Column::Float64(float_data))?;

    // Set missing values in a string column
    let string_values = vec![
        "a".to_string(),
        "b".to_string(),
        "".to_string(),
        "d".to_string(),
        "".to_string(),
    ];
    let string_nulls = vec![false, false, true, false, true]; // Set indices 2 and 4 to NULL
    let string_data = StringColumn::with_nulls(string_values, string_nulls);

    df.add_column("strings", Column::String(string_data))?;

    println!("DataFrame with Missing Values:");
    println!("{:?}", df);

    // Operations on columns (integer column)
    println!("\n--- Missing Values in Integer Column ---");
    let age_col = df.column("numbers")?;
    if let Some(int64_col) = age_col.as_int64() {
        // Verification
        for i in 0..5 {
            let value_result = int64_col.get(i)?;
            let value_str = match value_result {
                Some(val) => val.to_string(),
                None => "NULL".to_string(),
            };
            println!("Position {}: {}", i, value_str);
        }

        // Aggregation functions (ignoring missing values)
        println!("Sum (ignoring NA): {}", int64_col.sum());
        println!("Mean (ignoring NA): {:.2}", int64_col.mean().unwrap_or(0.0));
        println!("Min (ignoring NA): {}", int64_col.min().unwrap_or(0));
        println!("Max (ignoring NA): {}", int64_col.max().unwrap_or(0));
    }

    // Operations on columns (floating-point column)
    println!("\n--- Missing Values in Floating-Point Column ---");
    let float_col = df.column("floats")?;
    if let Some(float64_col) = float_col.as_float64() {
        // Verification
        for i in 0..5 {
            let value_result = float64_col.get(i)?;
            let value_str = match value_result {
                Some(val) => format!("{:.1}", val),
                None => "NULL".to_string(),
            };
            println!("Position {}: {}", i, value_str);
        }

        // Aggregation functions (ignoring missing values)
        println!("Sum (ignoring NA): {:.1}", float64_col.sum());
        println!(
            "Mean (ignoring NA): {:.2}",
            float64_col.mean().unwrap_or(0.0)
        );
        println!("Min (ignoring NA): {:.1}", float64_col.min().unwrap_or(0.0));
        println!("Max (ignoring NA): {:.1}", float64_col.max().unwrap_or(0.0));
    }

    // Operations on columns (string column)
    println!("\n--- Missing Values in String Column ---");
    let string_col = df.column("strings")?;
    if let Some(str_col) = string_col.as_string() {
        // Verification
        for i in 0..5 {
            let value_result = str_col.get(i)?;
            let value_str = match value_result {
                Some(val) => val.to_string(),
                None => "NULL".to_string(),
            };
            println!("Position {}: {}", i, value_str);
        }
    }

    // Filtering - Create a boolean column to select rows without missing values
    println!("\n--- Filtering Missing Values ---");

    // Verification
    println!("In OptimizedDataFrame, filtering implementation may vary, but typically, null_mask is used to check for missing values.");

    println!("=== Sample Complete ===");
    Ok(())
}

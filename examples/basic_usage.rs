use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;
use pandrs::{DataFrame, Series};
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS Basic Usage Example (Alpha 4) ===");

    // Creating Series with new alpha.4 features
    let mut ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
    let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;
    let names = Series::new(
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
        Some("name".to_string()),
    )?;

    println!("Age Series: {:?}", ages);
    println!("Height Series: {:?}", heights);
    println!("Name Series: {:?}", names);

    // Demonstrate new Series name management (alpha.4)
    println!("\n=== Series Name Management (New in Alpha 4) ===");
    ages.set_name("person_age".to_string());
    println!("Updated age series name: {:?}", ages.name());

    // Create series with fluent API
    let weights = Series::new(vec![70, 80, 75], None)?.with_name("weight_kg".to_string());
    println!("Weight series with fluent API: {:?}", weights.name());

    // Statistics for numeric series
    println!("\n=== Statistics for Age Series ===");
    println!("Sum: {}", ages.sum());
    println!("Mean: {}", ages.mean()?);
    println!("Min: {}", ages.min()?);
    println!("Max: {}", ages.max()?);

    // Creating both traditional and optimized DataFrames
    println!("\n=== Creating a Traditional DataFrame ===");
    let mut df = DataFrame::new();
    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;
    df.add_column("height".to_string(), heights)?;

    println!("DataFrame: {:?}", df);
    println!("Number of Columns: {}", df.column_count());
    println!("Number of Rows: {}", df.row_count());
    println!("Column Names: {:?}", df.column_names());

    // Create an OptimizedDataFrame (recommended for performance)
    println!("\n=== Creating an OptimizedDataFrame (Recommended) ===");
    let mut opt_df = OptimizedDataFrame::new();
    opt_df.add_string_column(
        "name",
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
    )?;
    opt_df.add_int_column("age", vec![30, 25, 40])?;
    opt_df.add_int_column("height", vec![180, 175, 182])?;
    opt_df.add_float_column("weight", vec![70.5, 80.2, 75.8])?;

    println!(
        "OptimizedDataFrame created with {} rows and {} columns",
        opt_df.row_count(),
        opt_df.column_count()
    );
    println!("Column Names: {:?}", opt_df.column_names());

    // Demonstrate new column management features (alpha.4)
    println!("\n=== Column Management (New in Alpha 4) ===");

    // Rename specific columns
    let mut rename_map = HashMap::new();
    rename_map.insert("age".to_string(), "years_old".to_string());
    rename_map.insert("height".to_string(), "height_cm".to_string());
    opt_df.rename_columns(&rename_map)?;
    println!("After renaming columns: {:?}", opt_df.column_names());

    // Set all column names at once
    opt_df.set_column_names(vec![
        "full_name".to_string(),
        "person_age".to_string(),
        "height_measurement".to_string(),
        "body_weight".to_string(),
    ])?;
    println!(
        "After setting all column names: {:?}",
        opt_df.column_names()
    );

    // Testing I/O operations with enhanced error handling
    println!("\n=== I/O Operations ===");

    // Save traditional DataFrame to CSV
    let traditional_file = "traditional_data.csv";
    match df.to_csv(traditional_file) {
        Ok(_) => println!("✅ Saved traditional DataFrame to {}", traditional_file),
        Err(e) => println!("❌ Failed to save traditional DataFrame: {}", e),
    }

    // Save OptimizedDataFrame to CSV
    let optimized_file = "optimized_data.csv";
    match opt_df.to_csv(optimized_file, true) {
        Ok(_) => {
            println!("✅ Saved OptimizedDataFrame to {}", optimized_file);

            // Try to read it back using the I/O module
            match pandrs::io::read_csv(optimized_file, true) {
                Ok(loaded_df) => {
                    println!("✅ Successfully loaded DataFrame from CSV");
                    println!(
                        "   Loaded {} rows and {} columns",
                        loaded_df.row_count(),
                        loaded_df.column_count()
                    );
                    println!("   Column Names: {:?}", loaded_df.column_names());
                }
                Err(e) => println!("❌ Failed to load CSV: {}", e),
            }
        }
        Err(e) => println!("❌ Failed to save OptimizedDataFrame: {}", e),
    }

    // Demonstrate statistical operations on OptimizedDataFrame
    println!("\n=== Statistical Operations on OptimizedDataFrame ===");
    println!("Age statistics:");
    println!("  Sum: {:.2}", opt_df.sum("person_age")?);
    println!("  Mean: {:.2}", opt_df.mean("person_age")?);
    println!("  Min: {:.2}", opt_df.min("person_age")?);
    println!("  Max: {:.2}", opt_df.max("person_age")?);

    // Demonstrate type-safe column access
    println!("\n=== Type-Safe Column Access ===");
    let age_col_view = opt_df.column("person_age")?;
    if let Some(age_col) = age_col_view.as_int64() {
        println!("Successfully accessed age column as Int64");
        println!("First age value: {:?}", age_col.get(0)?);
    }

    let weight_col_view = opt_df.column("body_weight")?;
    if let Some(weight_col) = weight_col_view.as_float64() {
        println!("Successfully accessed weight column as Float64");
        println!("Average weight: {:?}", weight_col.mean());
    }

    // Type safety demonstration - this should return None
    let wrong_type = age_col_view.as_string();
    println!(
        "Accessing Int64 column as String: {:?}",
        wrong_type.is_some()
    );

    println!("\n🎉 Alpha 4 Basic Usage Example Complete! 🎉");
    Ok(())
}

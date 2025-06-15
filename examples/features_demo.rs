use pandrs::optimized::OptimizedDataFrame;
use pandrs::Series;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PandRS Alpha 4 New Features Demo");
    println!("=================================\n");

    // ========================================
    // Series Name Management (New in Alpha 4)
    // ========================================
    println!("1. Series Name Management");
    println!("-------------------------");

    // Create a series with no name initially
    let mut sales_data = Series::new(vec![100, 150, 200, 175, 300], None)?;
    println!("Initial series name: {:?}", sales_data.name());

    // Set name using the new set_name method
    sales_data.set_name("monthly_sales".to_string());
    println!("After set_name: {:?}", sales_data.name());

    // Create a series with fluent API using with_name
    let revenue_data = Series::new(vec![1000, 1500, 2000, 1750, 3000], None)?
        .with_name("monthly_revenue".to_string());
    println!("Revenue series name: {:?}", revenue_data.name());

    // Convert to string series while preserving functionality
    let sales_strings = sales_data.to_string_series()?;
    println!("String series name: {:?}", sales_strings.name());
    println!("String values: {:?}\n", sales_strings.values());

    // =============================================
    // DataFrame Column Management (New in Alpha 4)
    // =============================================
    println!("2. DataFrame Column Management");
    println!("------------------------------");

    // Create an OptimizedDataFrame with sample data
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("sales", vec![100, 150, 200, 175, 300])?;
    df.add_float_column("margin", vec![0.15, 0.20, 0.18, 0.22, 0.25])?;
    df.add_string_column(
        "region",
        vec![
            "North".to_string(),
            "South".to_string(),
            "East".to_string(),
            "West".to_string(),
            "Central".to_string(),
        ],
    )?;

    println!("Original column names: {:?}", df.column_names());

    // Rename specific columns using rename_columns
    let mut rename_map = HashMap::new();
    rename_map.insert("sales".to_string(), "total_sales".to_string());
    rename_map.insert("margin".to_string(), "profit_margin".to_string());
    df.rename_columns(&rename_map)?;

    println!("After rename_columns: {:?}", df.column_names());

    // Set all column names at once
    df.set_column_names(vec![
        "quarterly_sales".to_string(),
        "profit_percentage".to_string(),
        "sales_region".to_string(),
    ])?;

    println!("After set_column_names: {:?}", df.column_names());

    // ========================================
    // Enhanced Data Access and Type Safety
    // ========================================
    println!("\n3. Enhanced Data Access");
    println!("----------------------");

    // Safe column access with type checking
    let sales_col_view = df.column("quarterly_sales")?;
    if let Some(int_col) = sales_col_view.as_int64() {
        println!("Sales column type: Int64");
        println!("First sales value: {:?}", int_col.get(0)?);
        println!("Sum of sales: {}", int_col.sum());
    }

    let margin_col_view = df.column("profit_percentage")?;
    if let Some(float_col) = margin_col_view.as_float64() {
        println!("Margin column type: Float64");
        println!("Average margin: {:?}", float_col.mean());
    }

    let region_col_view = df.column("sales_region")?;
    if let Some(string_col) = region_col_view.as_string() {
        println!("Region column type: String");
        println!("First region: {:?}", string_col.get(0)?);
    }

    // ========================================
    // Error Handling and Edge Cases
    // ========================================
    println!("\n4. Robust Error Handling");
    println!("------------------------");

    // Demonstrate proper error handling for invalid operations
    match df.column("nonexistent_column") {
        Ok(_) => println!("Column found (unexpected)"),
        Err(e) => println!("Expected error for nonexistent column: {}", e),
    }

    // Type mismatch handling
    let test_col_view = df.column("quarterly_sales")?;
    match test_col_view.as_string() {
        Some(_) => println!("Type mismatch not detected (unexpected)"),
        None => println!("Correctly detected type mismatch: Int64 column accessed as String"),
    }

    // ========================================
    // Performance and Memory Efficiency
    // ========================================
    println!("\n5. Memory Efficiency Demo");
    println!("-------------------------");

    // Create a large dataset to demonstrate string pool efficiency
    let mut large_df = OptimizedDataFrame::new();

    // Create data with repeated string values (simulates categorical data)
    let regions: Vec<String> = (0..10000)
        .map(|i| format!("Region_{}", i % 5)) // Only 5 unique regions
        .collect();

    let sales: Vec<i64> = (0..10000).map(|i| 1000 + (i % 500) as i64).collect();

    large_df.add_string_column("region", regions)?;
    large_df.add_int_column("sales", sales)?;

    println!("Large DataFrame created:");
    println!("Rows: {}", large_df.row_count());
    println!("Columns: {}", large_df.column_count());
    println!("Column names: {:?}", large_df.column_names());

    // Demonstrate aggregation on large dataset
    let sales_sum = large_df.sum("sales")?;
    println!("Total sales: {}", sales_sum);

    // ========================================
    // I/O Operations with Error Handling
    // ========================================
    println!("\n6. Enhanced I/O Operations");
    println!("--------------------------");

    // Save to CSV with proper error handling
    match large_df.to_csv("alpha4_demo_output.csv", true) {
        Ok(_) => {
            println!("Successfully saved DataFrame to CSV");

            // Read it back
            match pandrs::io::read_csv("alpha4_demo_output.csv", true) {
                Ok(loaded_df) => {
                    println!("Successfully loaded DataFrame from CSV");
                    println!(
                        "Loaded {} rows and {} columns",
                        loaded_df.row_count(),
                        loaded_df.column_count()
                    );
                }
                Err(e) => println!("Error loading CSV: {}", e),
            }
        }
        Err(e) => println!("Error saving CSV: {}", e),
    }

    // ========================================
    // Advanced Operations
    // ========================================
    println!("\n7. Advanced Operations");
    println!("----------------------");

    // Groupby operations with the improved API
    match large_df.par_groupby(&["region"]) {
        Ok(grouped) => {
            println!("Successfully grouped by region");
            println!("Number of groups: {}", grouped.len());
        }
        Err(e) => println!("Groupby error: {}", e),
    }

    // Statistical operations
    println!("Statistical summary:");
    println!("Sales mean: {:.2}", large_df.mean("sales")?);
    println!("Sales min: {:.2}", large_df.min("sales")?);
    println!("Sales max: {:.2}", large_df.max("sales")?);

    println!("\n✅ Alpha 4 features demonstration completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha4_features() {
        // Test the main functionality without I/O operations
        let mut sales_data = Series::new(vec![100, 150, 200], None).unwrap();
        sales_data.set_name("test_sales".to_string());
        assert_eq!(sales_data.name(), Some(&"test_sales".to_string()));

        let mut df = OptimizedDataFrame::new();
        df.add_int_column("values", vec![1, 2, 3]).unwrap();

        let mut rename_map = HashMap::new();
        rename_map.insert("values".to_string(), "new_values".to_string());
        df.rename_columns(&rename_map).unwrap();

        assert_eq!(df.column_names(), &["new_values"]);
    }

    #[test]
    fn test_error_handling() {
        let df = OptimizedDataFrame::new();

        // Test accessing nonexistent column
        assert!(df.column("nonexistent").is_err());

        // Test type safety
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("int_col", vec![1, 2, 3]).unwrap();

        let col_view = df.column("int_col").unwrap();
        assert!(col_view.as_int64().is_some());
        assert!(col_view.as_string().is_none());
    }
}

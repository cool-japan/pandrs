//! Comprehensive PandRS Tutorial
//!
//! This example demonstrates common data analysis workflows using PandRS,
//! including data loading, cleaning, transformation, analysis, and visualization.

//use pandrs::{DataFrame, Series}; // Currently unused
use pandrs::optimized::OptimizedDataFrame;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PandRS Comprehensive Tutorial");
    println!("============================\n");

    // Step 1: Create and populate a DataFrame with sales data
    tutorial_step_1_data_creation()?;

    // Step 2: Data cleaning and preparation
    tutorial_step_2_data_cleaning()?;

    // Step 3: Data transformation and feature engineering
    tutorial_step_3_transformation()?;

    // Step 4: Statistical analysis
    tutorial_step_4_analysis()?;

    // Step 5: Grouping and aggregation
    tutorial_step_5_grouping()?;

    // Step 6: Data export and persistence
    tutorial_step_6_persistence()?;

    println!("\n🎉 Tutorial completed successfully!");
    Ok(())
}

/// Step 1: Creating and populating DataFrames
fn tutorial_step_1_data_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 1: Data Creation and Basic Operations");
    println!("==========================================");

    // Create an OptimizedDataFrame (recommended for performance)
    let mut sales_df = OptimizedDataFrame::new();

    // Add columns with different data types
    sales_df.add_string_column(
        "product",
        vec![
            "Laptop".to_string(),
            "Phone".to_string(),
            "Tablet".to_string(),
            "Monitor".to_string(),
            "Keyboard".to_string(),
            "Mouse".to_string(),
        ],
    )?;

    sales_df.add_int_column("quantity", vec![10, 25, 15, 8, 50, 30])?;

    sales_df.add_float_column("price", vec![999.99, 599.99, 399.99, 249.99, 79.99, 29.99])?;

    sales_df.add_string_column(
        "category",
        vec![
            "Electronics".to_string(),
            "Electronics".to_string(),
            "Electronics".to_string(),
            "Electronics".to_string(),
            "Accessories".to_string(),
            "Accessories".to_string(),
        ],
    )?;

    sales_df.add_string_column(
        "region",
        vec![
            "North".to_string(),
            "South".to_string(),
            "East".to_string(),
            "West".to_string(),
            "North".to_string(),
            "South".to_string(),
        ],
    )?;

    // Display basic information about the DataFrame
    println!(
        "DataFrame created with {} rows and {} columns",
        sales_df.row_count(),
        sales_df.column_count()
    );
    println!("Column names: {:?}", sales_df.column_names());

    // Calculate total revenue for each product
    let mut revenue_values = Vec::new();
    let quantity_binding = sales_df.column("quantity")?;
    let quantity_col = quantity_binding.as_int64().unwrap();
    let price_binding = sales_df.column("price")?;
    let price_col = price_binding.as_float64().unwrap();

    for i in 0..sales_df.row_count() {
        let qty = quantity_col.get(i)?.unwrap() as f64;
        let price = price_col.get(i)?.unwrap();
        revenue_values.push(qty * price);
    }

    sales_df.add_float_column("revenue", revenue_values)?;

    println!("Added revenue column");
    println!("Sample data preview (first 3 rows):");
    for i in 0..3.min(sales_df.row_count()) {
        let product_binding = sales_df.column("product")?;
        let product = product_binding.as_string().unwrap().get(i)?.unwrap();
        let quantity_binding = sales_df.column("quantity")?;
        let quantity = quantity_binding.as_int64().unwrap().get(i)?.unwrap();
        let price_binding = sales_df.column("price")?;
        let price = price_binding.as_float64().unwrap().get(i)?.unwrap();
        let revenue_binding = sales_df.column("revenue")?;
        let revenue = revenue_binding.as_float64().unwrap().get(i)?.unwrap();
        println!(
            "  {} | Qty: {} | Price: ${:.2} | Revenue: ${:.2}",
            product, quantity, price, revenue
        );
    }
    println!();

    Ok(())
}

/// Step 2: Data cleaning and validation
fn tutorial_step_2_data_cleaning() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 2: Data Cleaning and Validation");
    println!("====================================");

    let mut df = OptimizedDataFrame::new();

    // Create data with some edge cases to demonstrate cleaning
    df.add_string_column(
        "customer_name",
        vec![
            "John Doe".to_string(),
            "jane smith".to_string(), // lowercase
            "".to_string(),           // empty string
            "Bob Johnson".to_string(),
            "ALICE WILLIAMS".to_string(), // uppercase
        ],
    )?;

    df.add_float_column("score", vec![85.5, 92.0, -1.0, 78.5, 95.0])?; // -1.0 as invalid score
    df.add_int_column("age", vec![25, 35, 0, 42, 29])?; // 0 as potentially invalid age

    println!("Original data:");
    for i in 0..df.row_count() {
        let name_binding = df.column("customer_name")?;
        let name = name_binding.as_string().unwrap().get(i)?.unwrap();
        let score_binding = df.column("score")?;
        let score = score_binding.as_float64().unwrap().get(i)?.unwrap();
        let age_binding = df.column("age")?;
        let age = age_binding.as_int64().unwrap().get(i)?.unwrap();
        println!("  '{}' | Score: {} | Age: {}", name, score, age);
    }

    // Data cleaning steps:
    // 1. Normalize customer names (this would typically be done with a dedicated function)
    // 2. Validate score ranges
    // 3. Check for reasonable age values

    let score_binding = df.column("score")?;
    let score_col = score_binding.as_float64().unwrap();
    let mut valid_scores = 0;
    let mut total_score = 0.0;

    for i in 0..df.row_count() {
        let score = score_col.get(i)?.unwrap();
        if (0.0..=100.0).contains(&score) {
            valid_scores += 1;
            total_score += score;
        }
    }

    println!("\nData quality check:");
    println!("Valid scores: {}/{}", valid_scores, df.row_count());
    println!(
        "Average valid score: {:.2}",
        total_score / valid_scores as f64
    );

    // Check for empty names
    let name_binding = df.column("customer_name")?;
    let name_col = name_binding.as_string().unwrap();
    let mut empty_names = 0;
    for i in 0..df.row_count() {
        let name = name_col.get(i)?.unwrap();
        if name.trim().is_empty() {
            empty_names += 1;
        }
    }
    println!("Empty names found: {}", empty_names);
    println!();

    Ok(())
}

/// Step 3: Data transformation and feature engineering
fn tutorial_step_3_transformation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 3: Data Transformation and Feature Engineering");
    println!("===================================================");

    let mut df = OptimizedDataFrame::new();

    // Create time series-like data
    df.add_int_column("month", vec![1, 2, 3, 4, 5, 6])?;
    df.add_float_column(
        "sales",
        vec![1000.0, 1200.0, 1100.0, 1400.0, 1350.0, 1500.0],
    )?;
    df.add_float_column("costs", vec![600.0, 720.0, 660.0, 840.0, 810.0, 900.0])?;

    // Feature engineering: Calculate profit and profit margin
    let sales_binding = df.column("sales")?;
    let sales_col = sales_binding.as_float64().unwrap();
    let costs_binding = df.column("costs")?;
    let costs_col = costs_binding.as_float64().unwrap();

    let mut profit_values = Vec::new();
    let mut margin_values = Vec::new();

    for i in 0..df.row_count() {
        let sales = sales_col.get(i)?.unwrap();
        let costs = costs_col.get(i)?.unwrap();
        let profit = sales - costs;
        let margin = (profit / sales) * 100.0;

        profit_values.push(profit);
        margin_values.push(margin);
    }

    df.add_float_column("profit", profit_values)?;
    df.add_float_column("profit_margin", margin_values)?;

    // Calculate moving averages (3-month window)
    let mut moving_avg_sales = Vec::new();
    let sales_binding = df.column("sales")?;
    let sales_col = sales_binding.as_float64().unwrap();

    for i in 0..df.row_count() {
        let start = if i >= 2 { i - 2 } else { 0 };
        let end = i + 1;
        let window_size = end - start;

        let mut sum = 0.0;
        for j in start..end {
            sum += sales_col.get(j)?.unwrap();
        }
        moving_avg_sales.push(sum / window_size as f64);
    }

    df.add_float_column("sales_3ma", moving_avg_sales)?;

    // Categorize months by season
    let month_binding = df.column("month")?;
    let month_col = month_binding.as_int64().unwrap();
    let mut seasons = Vec::new();

    for i in 0..df.row_count() {
        let month = month_col.get(i)?.unwrap();
        let season = match month {
            1..=2 => "Winter",
            3..=5 => "Spring",
            6..=8 => "Summer",
            9..=11 => "Fall",
            12 => "Winter",
            _ => "Unknown",
        };
        seasons.push(season.to_string());
    }

    df.add_string_column("season", seasons)?;

    println!("Transformed data with new features:");
    for i in 0..df.row_count() {
        let month_binding = df.column("month")?;
        let month = month_binding.as_int64().unwrap().get(i)?.unwrap();
        let sales_binding = df.column("sales")?;
        let sales = sales_binding.as_float64().unwrap().get(i)?.unwrap();
        let profit_binding = df.column("profit")?;
        let profit = profit_binding.as_float64().unwrap().get(i)?.unwrap();
        let margin_binding = df.column("profit_margin")?;
        let margin = margin_binding.as_float64().unwrap().get(i)?.unwrap();
        let ma_binding = df.column("sales_3ma")?;
        let ma = ma_binding.as_float64().unwrap().get(i)?.unwrap();
        let season_binding = df.column("season")?;
        let season = season_binding.as_string().unwrap().get(i)?.unwrap();

        println!(
            "  Month {}: Sales ${:.0} | Profit ${:.0} | Margin {:.1}% | 3MA ${:.0} | {}",
            month, sales, profit, margin, ma, season
        );
    }
    println!();

    Ok(())
}

/// Step 4: Statistical analysis
fn tutorial_step_4_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 4: Statistical Analysis");
    println!("============================");

    let mut df = OptimizedDataFrame::new();

    // Create sample dataset for analysis
    let values = vec![15.0, 23.0, 18.0, 31.0, 27.0, 19.0, 22.0, 25.0, 29.0, 21.0];
    df.add_float_column("values", values)?;

    // Basic statistical measures
    let values_binding = df.column("values")?;
    let values_col = values_binding.as_float64().unwrap();

    // Calculate statistics manually to demonstrate the process
    let mut sum = 0.0;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for i in 0..df.row_count() {
        let val = values_col.get(i)?.unwrap();
        sum += val;
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    let mean = sum / df.row_count() as f64;

    // Calculate standard deviation
    let mut variance_sum = 0.0;
    for i in 0..df.row_count() {
        let val = values_col.get(i)?.unwrap();
        variance_sum += (val - mean).powi(2);
    }
    let variance = variance_sum / (df.row_count() - 1) as f64;
    let std_dev = variance.sqrt();

    // Using built-in statistical functions
    let built_in_sum = df.sum("values")?;
    let built_in_mean = df.mean("values")?;
    let built_in_min = df.min("values")?;
    let built_in_max = df.max("values")?;

    println!("Statistical Analysis Results:");
    println!("Manual calculations:");
    println!("  Sum: {:.2}", sum);
    println!("  Mean: {:.2}", mean);
    println!("  Min: {:.2}", min_val);
    println!("  Max: {:.2}", max_val);
    println!("  Std Dev: {:.2}", std_dev);

    println!("Built-in functions:");
    println!("  Sum: {:.2}", built_in_sum);
    println!("  Mean: {:.2}", built_in_mean);
    println!("  Min: {:.2}", built_in_min);
    println!("  Max: {:.2}", built_in_max);

    // Percentile calculations (simplified)
    let mut sorted_values: Vec<f64> = Vec::new();
    for i in 0..df.row_count() {
        sorted_values.push(values_col.get(i)?.unwrap());
    }
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if sorted_values.len() % 2 == 0 {
        (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };

    println!("  Median: {:.2}", median);
    println!();

    Ok(())
}

/// Step 5: Grouping and aggregation
fn tutorial_step_5_grouping() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 5: Grouping and Aggregation");
    println!("=================================");

    let mut df = OptimizedDataFrame::new();

    // Create sales data by region and product category
    df.add_string_column(
        "region",
        vec![
            "North".to_string(),
            "North".to_string(),
            "South".to_string(),
            "South".to_string(),
            "East".to_string(),
            "East".to_string(),
            "West".to_string(),
            "West".to_string(),
        ],
    )?;

    df.add_string_column(
        "category",
        vec![
            "Electronics".to_string(),
            "Accessories".to_string(),
            "Electronics".to_string(),
            "Accessories".to_string(),
            "Electronics".to_string(),
            "Accessories".to_string(),
            "Electronics".to_string(),
            "Accessories".to_string(),
        ],
    )?;

    df.add_float_column(
        "sales",
        vec![1000.0, 300.0, 1200.0, 350.0, 800.0, 250.0, 900.0, 280.0],
    )?;

    println!("Original sales data:");
    for i in 0..df.row_count() {
        let region_binding = df.column("region")?;
        let region = region_binding.as_string().unwrap().get(i)?.unwrap();
        let category_binding = df.column("category")?;
        let category = category_binding.as_string().unwrap().get(i)?.unwrap();
        let sales_binding = df.column("sales")?;
        let sales = sales_binding.as_float64().unwrap().get(i)?.unwrap();
        println!("  {} | {} | ${:.0}", region, category, sales);
    }

    // Manual grouping by region to demonstrate the concept
    let mut region_totals: HashMap<String, f64> = HashMap::new();
    let region_binding = df.column("region")?;
    let region_col = region_binding.as_string().unwrap();
    let sales_binding = df.column("sales")?;
    let sales_col = sales_binding.as_float64().unwrap();

    for i in 0..df.row_count() {
        let region = region_col.get(i)?.unwrap().to_string();
        let sales = sales_col.get(i)?.unwrap();
        *region_totals.entry(region).or_insert(0.0) += sales;
    }

    println!("\nSales by region (manual calculation):");
    for (region, total) in &region_totals {
        println!("  {}: ${:.0}", region, total);
    }

    // Category totals
    let mut category_totals: HashMap<String, f64> = HashMap::new();
    let category_binding = df.column("category")?;
    let category_col = category_binding.as_string().unwrap();

    for i in 0..df.row_count() {
        let category = category_col.get(i)?.unwrap().to_string();
        let sales = sales_col.get(i)?.unwrap();
        *category_totals.entry(category).or_insert(0.0) += sales;
    }

    println!("\nSales by category:");
    for (category, total) in &category_totals {
        println!("  {}: ${:.0}", category, total);
    }

    // Cross-tabulation (region x category)
    let mut cross_tab: HashMap<(String, String), f64> = HashMap::new();

    for i in 0..df.row_count() {
        let region = region_col.get(i)?.unwrap().to_string();
        let category = category_col.get(i)?.unwrap().to_string();
        let sales = sales_col.get(i)?.unwrap();
        *cross_tab.entry((region, category)).or_insert(0.0) += sales;
    }

    println!("\nCross-tabulation (Region x Category):");
    for ((region, category), total) in &cross_tab {
        println!("  {} x {}: ${:.0}", region, category, total);
    }
    println!();

    Ok(())
}

/// Step 6: Data persistence and export
fn tutorial_step_6_persistence() -> Result<(), Box<dyn std::error::Error>> {
    println!("Step 6: Data Persistence and Export");
    println!("===================================");

    let mut df = OptimizedDataFrame::new();

    // Create a summary dataset for export
    df.add_string_column(
        "metric",
        vec![
            "Total Sales".to_string(),
            "Total Customers".to_string(),
            "Avg Order Value".to_string(),
            "Conversion Rate".to_string(),
        ],
    )?;

    df.add_float_column("value", vec![125000.0, 450.0, 277.78, 0.034])?;

    df.add_string_column(
        "unit",
        vec![
            "USD".to_string(),
            "Count".to_string(),
            "USD".to_string(),
            "Percentage".to_string(),
        ],
    )?;

    println!("Summary report data:");
    for i in 0..df.row_count() {
        let metric_binding = df.column("metric")?;
        let metric = metric_binding.as_string().unwrap().get(i)?.unwrap();
        let value_binding = df.column("value")?;
        let value = value_binding.as_float64().unwrap().get(i)?.unwrap();
        let unit_binding = df.column("unit")?;
        let unit = unit_binding.as_string().unwrap().get(i)?.unwrap();
        println!("  {}: {:.2} {}", metric, value, unit);
    }

    // Export to CSV
    match df.to_csv("tutorial_summary_report.csv", true) {
        Ok(_) => {
            println!("\n✅ Successfully exported data to tutorial_summary_report.csv");

            // Verify by reading it back
            match pandrs::io::read_csv("tutorial_summary_report.csv", true) {
                Ok(loaded_df) => {
                    println!("✅ Successfully verified CSV export/import");
                    println!(
                        "   Loaded {} rows and {} columns",
                        loaded_df.row_count(),
                        loaded_df.column_count()
                    );
                }
                Err(e) => println!("❌ Error verifying CSV: {}", e),
            }
        }
        Err(e) => println!("❌ Error exporting to CSV: {}", e),
    }

    // Example of conditional exports based on data
    let total_sales_binding = df.column("value")?;
    let total_sales = total_sales_binding.as_float64().unwrap().get(0)?.unwrap();
    if total_sales > 100000.0 {
        println!("📊 High sales volume detected - consider generating detailed report");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tutorial_steps() {
        // Test that all tutorial steps can run without panicking
        assert!(tutorial_step_1_data_creation().is_ok());
        assert!(tutorial_step_2_data_cleaning().is_ok());
        assert!(tutorial_step_3_transformation().is_ok());
        assert!(tutorial_step_4_analysis().is_ok());
        assert!(tutorial_step_5_grouping().is_ok());
        // Skip step 6 in tests to avoid file I/O
    }

    #[test]
    fn test_basic_dataframe_operations() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("test", vec![1, 2, 3]).unwrap();

        assert_eq!(df.row_count(), 3);
        assert_eq!(df.column_count(), 1);
        assert_eq!(df.sum("test").unwrap(), 6.0);
        assert_eq!(df.mean("test").unwrap(), 2.0);
    }
}

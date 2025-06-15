#![allow(clippy::result_large_err)]
#![allow(unused_mut)]

use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::groupby::AggFunc;
use pandrs::dataframe::hierarchical_groupby::{
    utils as hierarchical_utils, HierarchicalAggBuilder, HierarchicalGroupByExt,
};
use pandrs::error::Result;
use pandrs::series::base::Series;

fn main() -> Result<()> {
    println!("=== Hierarchical GroupBy Example for PandRS ===\n");
    println!("This example demonstrates advanced hierarchical grouping capabilities");
    println!("with multi-level group structures, navigation, and nested operations.\n");

    // Create a comprehensive sales dataset with natural hierarchy
    println!("1. Creating Sales Dataset with Natural Hierarchy:");
    let df = create_sales_dataset()?;
    println!(
        "   • Dataset size: {} rows × {} columns",
        df.row_count(),
        df.column_count()
    );
    println!("   • Columns: {}", df.column_names().join(", "));
    println!("   • Hierarchy: Region → Department → Product");

    // Display sample data
    println!("\nSample data (first 10 rows):");
    display_sample_data(&df, 10)?;

    println!("\n=== Hierarchical GroupBy Operations ===\n");

    // Test 1: Basic hierarchical grouping
    println!("2. Basic Hierarchical Grouping:");
    let hierarchical_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;

    let stats = hierarchical_gb.hierarchy_stats();
    println!("   Hierarchy Statistics:");
    println!("     • Total levels: {}", stats.total_levels);
    println!("     • Total groups: {}", stats.total_groups);
    println!("     • Leaf groups: {}", stats.leaf_groups);
    println!("     • Groups per level: {:?}", stats.groups_per_level);
    println!("     • Max group size: {}", stats.max_group_size);
    println!("     • Min group size: {}", stats.min_group_size);
    println!("     • Avg group size: {:.2}", stats.avg_group_size);

    // Test 2: Group sizes at different levels
    println!("\n3. Group Sizes Analysis:");

    // All groups (default - leaf level)
    println!("   a) All Groups (Leaf Level):");
    let all_sizes = hierarchical_gb.size()?;
    display_dataframe_sample(&all_sizes, "All Groups Sizes", 8)?;

    // Level 0 groups (regions)
    println!("   b) Level 0 Groups (Regions):");
    let mut level0_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let level0_sizes = level0_gb.level(0).with_children().size()?;
    display_dataframe_sample(&level0_sizes, "Regional Totals", 5)?;

    // Level 1 groups (departments within regions)
    println!("   c) Level 1 Groups (Departments within Regions):");
    let mut level1_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let level1_sizes = level1_gb.level(1).with_children().size()?;
    display_dataframe_sample(&level1_sizes, "Department Totals", 8)?;

    // Test 3: Hierarchical aggregations
    println!("\n4. Hierarchical Aggregations:");

    // Simple aggregations at different levels
    println!("   a) Sales Sum by Level:");
    let sales_agg = HierarchicalAggBuilder::new("sales".to_string())
        .at_level(0, AggFunc::Sum, "region_total_sales".to_string())
        .at_level(1, AggFunc::Sum, "dept_total_sales".to_string())
        .at_level(2, AggFunc::Sum, "product_total_sales".to_string())
        .with_propagation()
        .build();

    let mut sales_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let sales_result = sales_gb.agg_hierarchical(vec![sales_agg])?;
    display_dataframe_sample(&sales_result, "Sales Totals by Level", 10)?;

    // Multiple aggregations
    println!("   b) Multiple Aggregations (Sales & Quantity):");
    let multi_aggs = vec![
        HierarchicalAggBuilder::new("sales".to_string())
            .at_level(0, AggFunc::Mean, "avg_sales_region".to_string())
            .at_level(1, AggFunc::Mean, "avg_sales_dept".to_string())
            .at_level(2, AggFunc::Mean, "avg_sales_product".to_string())
            .build(),
        HierarchicalAggBuilder::new("quantity".to_string())
            .at_level(0, AggFunc::Sum, "total_qty_region".to_string())
            .at_level(1, AggFunc::Sum, "total_qty_dept".to_string())
            .at_level(2, AggFunc::Sum, "total_qty_product".to_string())
            .build(),
    ];

    let mut multi_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let multi_result = multi_gb.agg_hierarchical(multi_aggs)?;
    display_dataframe_sample(&multi_result, "Multiple Aggregations", 12)?;

    // Test 4: Navigation and specific group analysis
    println!("\n5. Group Navigation and Analysis:");

    // Focus on specific region
    println!("   a) North Region Analysis:");
    let mut north_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let north_analysis = north_gb
        .group_path(vec!["North".to_string()])
        .with_children()
        .size()?;
    display_dataframe_sample(&north_analysis, "North Region Breakdown", 10)?;

    // Focus on specific department across all regions
    println!("   b) Electronics Department Across All Regions:");
    let mut electronics_groups =
        df.hierarchical_groupby(vec!["region".to_string(), "department".to_string()])?;

    let electronics_agg = HierarchicalAggBuilder::new("sales".to_string())
        .at_level(1, AggFunc::Sum, "electronics_sales".to_string())
        .build();

    let electronics_result = electronics_groups.agg_hierarchical(vec![electronics_agg])?;
    display_dataframe_sample(&electronics_result, "Electronics by Region", 8)?;

    // Test 5: Custom aggregation functions
    println!("\n6. Custom Aggregation Functions:");

    let custom_agg = HierarchicalAggBuilder::new("sales".to_string())
        .at_level(
            0,
            AggFunc::Custom,
            "sales_coefficient_of_variation".to_string(),
        )
        .with_custom(|values: &[f64]| {
            if values.len() < 2 {
                return 0.0;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            let std_dev = variance.sqrt();
            if mean != 0.0 {
                std_dev / mean
            } else {
                0.0
            }
        })
        .build();

    let mut custom_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let custom_result = custom_gb.level(0).agg_hierarchical(vec![custom_agg])?;
    display_dataframe_sample(&custom_result, "Custom CV Analysis", 5)?;

    // Test 6: Utility functions demonstration
    println!("\n7. Utility Functions:");

    println!("   a) Simple aggregation utility:");
    let simple_agg = hierarchical_utils::simple_hierarchical_agg("quantity", AggFunc::Count, 1);
    let mut simple_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let simple_result = simple_gb.level(1).agg_hierarchical(vec![simple_agg])?;
    display_dataframe_sample(&simple_result, "Simple Count by Department", 8)?;

    println!("   b) Comprehensive aggregation utility:");
    let comprehensive_aggs = hierarchical_utils::comprehensive_agg("sales", 0);
    let mut comprehensive_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let comprehensive_result = comprehensive_gb
        .level(0)
        .agg_hierarchical(comprehensive_aggs)?;
    display_dataframe_sample(&comprehensive_result, "Comprehensive Analysis", 5)?;

    println!("\n=== Performance and Scalability Analysis ===");

    // Test 7: Performance analysis
    println!("\n8. Performance Analysis:");
    let large_df = create_large_sales_dataset(1000)?;
    println!(
        "   • Created large dataset with {} rows",
        large_df.row_count()
    );

    let start = std::time::Instant::now();
    let large_hierarchical = large_df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let hierarchy_build_time = start.elapsed();

    let large_stats = large_hierarchical.hierarchy_stats();
    println!("   • Hierarchy build time: {:?}", hierarchy_build_time);
    println!(
        "   • Large dataset stats: {} total groups, {} leaf groups",
        large_stats.total_groups, large_stats.leaf_groups
    );

    let start = std::time::Instant::now();
    let large_agg = HierarchicalAggBuilder::new("sales".to_string())
        .at_level(0, AggFunc::Sum, "total_sales".to_string())
        .build();
    let _large_result = large_hierarchical.agg_hierarchical(vec![large_agg])?;
    let aggregation_time = start.elapsed();
    println!("   • Aggregation time: {:?}", aggregation_time);

    println!("\n=== Advanced Features Demonstration ===");

    // Test 8: Complex hierarchy navigation
    println!("\n9. Complex Navigation Patterns:");

    // Multi-level navigation with parent context
    let mut complex_nav_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
    ])?;
    let nav_result = complex_nav_gb
        .group_path(vec!["North".to_string(), "Electronics".to_string()])
        .with_parents()
        .with_children()
        .size()?;
    display_dataframe_sample(&nav_result, "Complex Navigation", 10)?;

    println!("\n=== Hierarchical GroupBy Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("✓ Multi-level hierarchical grouping with tree-based structure");
    println!("✓ Efficient group navigation and context switching");
    println!("✓ Hierarchical aggregations with level-specific operations");
    println!("✓ Custom aggregation functions with hierarchical propagation");
    println!("✓ Group size analysis and statistics at different levels");
    println!("✓ Performance optimization for large datasets");
    println!("✓ Complex navigation patterns with parent/child context");
    println!("✓ Utility functions for common hierarchical operations");
    println!("✓ Production-ready implementation with comprehensive error handling");

    Ok(())
}

/// Create a sales dataset with natural hierarchy (Region → Department → Product)
fn create_sales_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Create hierarchical data
    let regions = vec![
        "North", "North", "North", "North", "North", "North", "South", "South", "South", "South",
        "South", "South", "East", "East", "East", "East", "East", "East", "West", "West", "West",
        "West", "West", "West",
    ];

    let departments = vec![
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Home",
        "Home",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Home",
        "Home",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Home",
        "Home",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Home",
        "Home",
    ];

    let products = vec![
        "Laptop", "Phone", "Shirt", "Pants", "Chair", "Table", "Laptop", "Phone", "Shirt", "Pants",
        "Chair", "Table", "Laptop", "Phone", "Shirt", "Pants", "Chair", "Table", "Laptop", "Phone",
        "Shirt", "Pants", "Chair", "Table",
    ];

    let sales = [
        15000.0, 8000.0, 1200.0, 800.0, 2500.0, 3500.0, 12000.0, 7500.0, 1100.0, 750.0, 2200.0,
        3200.0, 18000.0, 9500.0, 1350.0, 950.0, 2800.0, 4000.0, 14000.0, 8200.0, 1250.0, 850.0,
        2600.0, 3700.0,
    ];

    let quantities = [
        150, 200, 60, 40, 25, 35, 120, 190, 55, 38, 22, 32, 180, 240, 68, 48, 28, 40, 140, 210, 63,
        43, 26, 37,
    ];

    // Add columns to DataFrame
    df.add_column(
        "region".to_string(),
        Series::new(
            regions.iter().map(|s| s.to_string()).collect(),
            Some("region".to_string()),
        )?,
    )?;

    df.add_column(
        "department".to_string(),
        Series::new(
            departments.iter().map(|s| s.to_string()).collect(),
            Some("department".to_string()),
        )?,
    )?;

    df.add_column(
        "product".to_string(),
        Series::new(
            products.iter().map(|s| s.to_string()).collect(),
            Some("product".to_string()),
        )?,
    )?;

    df.add_column(
        "sales".to_string(),
        Series::new(
            sales.iter().map(|&s| s.to_string()).collect(),
            Some("sales".to_string()),
        )?,
    )?;

    df.add_column(
        "quantity".to_string(),
        Series::new(
            quantities.iter().map(|&q| q.to_string()).collect(),
            Some("quantity".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Create a larger sales dataset for performance testing
fn create_large_sales_dataset(multiplier: usize) -> Result<DataFrame> {
    let base_df = create_sales_dataset()?;
    let mut large_df = DataFrame::new();

    // Get base data
    let base_regions = base_df.get_column_string_values("region")?;
    let base_departments = base_df.get_column_string_values("department")?;
    let base_products = base_df.get_column_string_values("product")?;
    let base_sales = base_df.get_column_string_values("sales")?;
    let base_quantities = base_df.get_column_string_values("quantity")?;

    // Multiply the data
    let mut regions = Vec::new();
    let mut departments = Vec::new();
    let mut products = Vec::new();
    let mut sales = Vec::new();
    let mut quantities = Vec::new();

    for i in 0..multiplier {
        for j in 0..base_regions.len() {
            regions.push(format!("{}_{}", base_regions[j], i % 10));
            departments.push(base_departments[j].clone());
            products.push(format!("{}_{}", base_products[j], i % 20));

            // Add some variation to the numbers
            let sales_base: f64 = base_sales[j].parse().unwrap_or(0.0);
            let qty_base: i32 = base_quantities[j].parse().unwrap_or(0);

            sales.push((sales_base * (1.0 + (i as f64 * 0.1) % 0.5)).to_string());
            quantities.push((qty_base + (i as i32 * 5) % 50).to_string());
        }
    }

    // Add columns
    large_df.add_column(
        "region".to_string(),
        Series::new(regions, Some("region".to_string()))?,
    )?;
    large_df.add_column(
        "department".to_string(),
        Series::new(departments, Some("department".to_string()))?,
    )?;
    large_df.add_column(
        "product".to_string(),
        Series::new(products, Some("product".to_string()))?,
    )?;
    large_df.add_column(
        "sales".to_string(),
        Series::new(sales, Some("sales".to_string()))?,
    )?;
    large_df.add_column(
        "quantity".to_string(),
        Series::new(quantities, Some("quantity".to_string()))?,
    )?;

    Ok(large_df)
}

/// Display sample data from DataFrame
fn display_sample_data(df: &DataFrame, rows: usize) -> Result<()> {
    let display_rows = rows.min(df.row_count());

    for i in 0..display_rows {
        let row_data: Vec<String> = df
            .column_names()
            .iter()
            .map(|col| {
                df.get_column_string_values(col)
                    .map(|values| values.get(i).cloned().unwrap_or_else(|| "NULL".to_string()))
                    .unwrap_or_else(|_| "ERROR".to_string())
            })
            .collect();

        println!("   Row {}: {}", i, row_data.join(" | "));
    }

    Ok(())
}

/// Display a sample of DataFrame with title
fn display_dataframe_sample(df: &DataFrame, title: &str, max_rows: usize) -> Result<()> {
    println!("   {}:", title);
    if df.row_count() == 0 {
        println!("     (No data)");
        return Ok(());
    }

    // Display column headers
    let headers = df.column_names();
    println!("     Headers: {}", headers.join(" | "));

    // Display data rows
    let display_rows = max_rows.min(df.row_count());
    for i in 0..display_rows {
        let row_data: Vec<String> = headers
            .iter()
            .map(|col| {
                df.get_column_string_values(col)
                    .map(|values| values.get(i).cloned().unwrap_or_else(|| "NULL".to_string()))
                    .unwrap_or_else(|_| "ERROR".to_string())
            })
            .collect();

        println!("     Row {}: {}", i, row_data.join(" | "));
    }

    if df.row_count() > display_rows {
        println!("     ... and {} more rows", df.row_count() - display_rows);
    }

    println!();
    Ok(())
}

#![allow(clippy::result_large_err)]
#![allow(unused_mut)]

use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::groupby::AggFunc;
use pandrs::dataframe::hierarchical_groupby::HierarchicalGroupByExt;
use pandrs::error::Result;
use pandrs::series::base::Series;

fn main() -> Result<()> {
    println!("=== Nested Group Operations Example for PandRS ===\n");
    println!("This example demonstrates advanced nested group operations across hierarchy levels:");
    println!("• Cross-level aggregations");
    println!("• Nested transformations within groups");
    println!("• Inter-level operations and ratios");
    println!("• Hierarchical filtering");
    println!("• Cross-level comparisons");
    println!("• Nested rollup operations\n");

    // Create a comprehensive business dataset with sales hierarchy
    println!("1. Creating Business Dataset with Complex Hierarchy:");
    let df = create_business_dataset()?;
    println!(
        "   • Dataset size: {} rows × {} columns",
        df.row_count(),
        df.column_count()
    );
    println!("   • Columns: {}", df.column_names().join(", "));
    println!("   • Hierarchy: Region → Department → Product → Quarter");

    // Display sample data
    println!("\nSample data (first 12 rows):");
    display_sample_data(&df, 12)?;

    println!("\n=== Nested Group Operations ===\n");

    // Test 1: Cross-level aggregations
    println!("2. Cross-Level Aggregations:");
    println!("   Aggregating sales from product level (3) to department level (1)");

    let hierarchical_gb = df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
        "quarter".to_string(),
    ])?;

    let cross_level_result = hierarchical_gb.cross_level_agg("sales", AggFunc::Sum, 3, 1)?;
    display_dataframe_sample(&cross_level_result, "Cross-Level Sales Aggregation", 8)?;

    // Test 2: Nested transformations
    println!("3. Nested Transformations:");
    println!("   Applying rank transformation within each department");

    let nested_transform_result = hierarchical_gb.nested_transform(
        "sales",
        |values: &[f64]| {
            // Rank transformation within group
            let mut indexed_values: Vec<(usize, f64)> =
                values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_values
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut ranks = vec![0.0; values.len()];
            for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
                ranks[*original_idx] = (rank + 1) as f64;
            }
            ranks
        },
        1, // Department level
    )?;

    println!("   Transformed data sample (first 10 rows):");
    display_sample_data(&nested_transform_result, 10)?;

    // Test 3: Inter-level ratios
    println!("\n4. Inter-Level Operations (Parent-Child Ratios):");
    println!("   Computing product-to-department sales ratios");

    let inter_level_result = hierarchical_gb.inter_level_ratio("sales", 1, 2)?;
    display_dataframe_sample(&inter_level_result, "Product-to-Department Ratios", 12)?;

    // Test 4: Hierarchical filtering
    println!("5. Hierarchical Filtering:");
    println!("   Filtering departments with total sales > 50000");

    let filtered_gb = hierarchical_gb.hierarchical_filter(
        "sales",
        1, // Department level
        |total_sales| total_sales > 50000.0,
    )?;

    let filtered_stats = filtered_gb.hierarchy_stats();
    println!("   Filtered hierarchy statistics:");
    println!(
        "     • Total groups after filtering: {}",
        filtered_stats.total_groups
    );
    println!(
        "     • Leaf groups after filtering: {}",
        filtered_stats.leaf_groups
    );
    println!(
        "     • Groups per level: {:?}",
        filtered_stats.groups_per_level
    );

    let filtered_sizes = filtered_gb.size()?;
    display_dataframe_sample(&filtered_sizes, "Filtered Group Sizes", 10)?;

    // Test 5: Cross-level comparisons
    println!("6. Cross-Level Comparisons:");
    println!("   Comparing sales across region (0), department (1), and product (2) levels");

    let comparison_result = hierarchical_gb.cross_level_comparison(
        "sales",
        1,          // Base level: departments
        vec![0, 2], // Compare with regions and products
    )?;
    display_dataframe_sample(&comparison_result, "Cross-Level Sales Comparison", 8)?;

    // Test 6: Nested rollup
    println!("7. Nested Rollup Operations:");
    println!("   Creating rollup aggregations across all hierarchy levels");

    let rollup_result = hierarchical_gb.nested_rollup("sales", AggFunc::Sum)?;
    display_dataframe_sample(&rollup_result, "Nested Rollup Results", 15)?;

    // Test 7: Advanced cross-level analysis
    println!("8. Advanced Cross-Level Analysis:");
    println!("   Analyzing quarterly performance aggregated to regional level");

    let quarterly_regional = hierarchical_gb.cross_level_agg("quantity", AggFunc::Mean, 3, 0)?;
    display_dataframe_sample(&quarterly_regional, "Quarterly-to-Regional Analysis", 6)?;

    // Test 8: Complex nested transformation
    println!("9. Complex Nested Transformation:");
    println!("   Normalizing sales within each region (z-score transformation)");

    let normalized_result = hierarchical_gb.nested_transform(
        "sales",
        |values: &[f64]| {
            if values.len() <= 1 {
                return values.to_vec();
            }

            // Calculate mean and standard deviation
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            let std_dev = variance.sqrt();

            // Apply z-score normalization
            if std_dev > 0.0 {
                values.iter().map(|&x| (x - mean) / std_dev).collect()
            } else {
                vec![0.0; values.len()]
            }
        },
        0, // Regional level
    )?;

    println!("   Normalized data sample (first 10 rows):");
    display_sample_data(&normalized_result, 10)?;

    // Test 9: Multi-level ratio analysis
    println!("\n10. Multi-Level Ratio Analysis:");
    println!("    Computing quarter-to-product and quarter-to-department ratios");

    let quarter_product_ratios = hierarchical_gb.inter_level_ratio("sales", 2, 3)?;
    display_dataframe_sample(&quarter_product_ratios, "Quarter-to-Product Ratios", 12)?;

    let quarter_dept_ratios = hierarchical_gb.inter_level_ratio("sales", 1, 3)?;
    display_dataframe_sample(&quarter_dept_ratios, "Quarter-to-Department Ratios", 12)?;

    println!("\n=== Performance and Scalability Analysis ===");

    // Test 10: Performance with larger dataset
    println!("\n11. Performance Analysis:");
    let large_df = create_large_business_dataset(500)?;
    println!(
        "   • Created large dataset with {} rows",
        large_df.row_count()
    );

    let start = std::time::Instant::now();
    let large_hierarchical = large_df.hierarchical_groupby(vec![
        "region".to_string(),
        "department".to_string(),
        "product".to_string(),
        "quarter".to_string(),
    ])?;
    let hierarchy_build_time = start.elapsed();

    let start = std::time::Instant::now();
    let _large_cross_level = large_hierarchical.cross_level_agg("sales", AggFunc::Sum, 3, 0)?;
    let cross_level_time = start.elapsed();

    let start = std::time::Instant::now();
    let _large_rollup = large_hierarchical.nested_rollup("sales", AggFunc::Mean)?;
    let rollup_time = start.elapsed();

    println!("   • Hierarchy build time: {:?}", hierarchy_build_time);
    println!("   • Cross-level aggregation time: {:?}", cross_level_time);
    println!("   • Nested rollup time: {:?}", rollup_time);

    println!("\n=== Nested Group Operations Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("✓ Cross-level aggregations for flexible data analysis");
    println!("✓ Nested transformations within hierarchical groups");
    println!("✓ Inter-level ratio calculations for relative analysis");
    println!("✓ Hierarchical filtering with group-level criteria");
    println!("✓ Cross-level comparisons across multiple hierarchy levels");
    println!("✓ Nested rollup operations for comprehensive aggregation");
    println!("✓ Complex transformations (ranking, normalization) within groups");
    println!("✓ Multi-level ratio analysis for detailed insights");
    println!("✓ Performance optimization for large hierarchical datasets");
    println!("✓ Production-ready implementations with error handling");

    Ok(())
}

/// Create a business dataset with comprehensive hierarchy (Region → Department → Product → Quarter)
fn create_business_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Create hierarchical business data
    let regions = vec![
        "North", "North", "North", "North", "North", "North", "North", "North", "South", "South",
        "South", "South", "South", "South", "South", "South", "East", "East", "East", "East",
        "East", "East", "East", "East", "West", "West", "West", "West", "West", "West", "West",
        "West",
    ];

    let departments = vec![
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Clothing",
        "Clothing",
        "Clothing",
        "Clothing",
    ];

    let products = vec![
        "Laptop", "Laptop", "Phone", "Phone", "Shirt", "Shirt", "Pants", "Pants", "Laptop",
        "Laptop", "Phone", "Phone", "Shirt", "Shirt", "Pants", "Pants", "Laptop", "Laptop",
        "Phone", "Phone", "Shirt", "Shirt", "Pants", "Pants", "Laptop", "Laptop", "Phone", "Phone",
        "Shirt", "Shirt", "Pants", "Pants",
    ];

    let quarters = vec![
        "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1",
        "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2", "Q1", "Q2",
        "Q1", "Q2",
    ];

    let sales = vec![
        25000.0, 28000.0, 15000.0, 16500.0, 2000.0, 2200.0, 1500.0, 1650.0, 22000.0, 25000.0,
        13000.0, 14500.0, 1800.0, 2000.0, 1300.0, 1450.0, 30000.0, 33000.0, 18000.0, 19500.0,
        2400.0, 2600.0, 1800.0, 2000.0, 27000.0, 30000.0, 16000.0, 17500.0, 2100.0, 2300.0, 1600.0,
        1750.0,
    ];

    let quantities = [
        100, 110, 150, 165, 80, 88, 75, 83, 90, 100, 130, 145, 72, 80, 65, 73, 120, 132, 180, 195,
        96, 104, 90, 100, 108, 120, 160, 175, 84, 92, 80, 88,
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
        "quarter".to_string(),
        Series::new(
            quarters.iter().map(|s| s.to_string()).collect(),
            Some("quarter".to_string()),
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

/// Create a larger business dataset for performance testing
fn create_large_business_dataset(multiplier: usize) -> Result<DataFrame> {
    let base_df = create_business_dataset()?;
    let mut large_df = DataFrame::new();

    // Get base data
    let base_regions = base_df.get_column_string_values("region")?;
    let base_departments = base_df.get_column_string_values("department")?;
    let base_products = base_df.get_column_string_values("product")?;
    let base_quarters = base_df.get_column_string_values("quarter")?;
    let base_sales = base_df.get_column_string_values("sales")?;
    let base_quantities = base_df.get_column_string_values("quantity")?;

    // Multiply the data
    let mut regions = Vec::new();
    let mut departments = Vec::new();
    let mut products = Vec::new();
    let mut quarters = Vec::new();
    let mut sales = Vec::new();
    let mut quantities = Vec::new();

    for i in 0..multiplier {
        for j in 0..base_regions.len() {
            regions.push(format!("{}_{}", base_regions[j], i % 8));
            departments.push(base_departments[j].clone());
            products.push(format!("{}_{}", base_products[j], i % 12));
            quarters.push(base_quarters[j].clone());

            // Add some variation to the numbers
            let sales_base: f64 = base_sales[j].parse().unwrap_or(0.0);
            let qty_base: i32 = base_quantities[j].parse().unwrap_or(0);

            sales.push((sales_base * (1.0 + (i as f64 * 0.15) % 0.6)).to_string());
            quantities.push((qty_base + (i as i32 * 7) % 60).to_string());
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
        "quarter".to_string(),
        Series::new(quarters, Some("quarter".to_string()))?,
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

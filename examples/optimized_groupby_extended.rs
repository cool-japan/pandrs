use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS GroupBy Extended Example");
    println!("===============================");

    // Create a sample DataFrame with various data
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories = vec!["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    df.add_string_column("category", categories)?;

    let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
    df.add_int_column("value", values)?;

    let prices = vec![
        110.5, 225.2, 115.8, 130.4, 222.1, 118.9, 324.5, 112.3, 156.7, 120.9,
    ];
    df.add_float_column("price", prices)?;

    // Display the DataFrame
    println!("Original DataFrame:");
    println!("{:?}", df);

    // Use the existing par_groupby method which returns a HashMap
    println!("\n1. Group by category using par_groupby");
    println!("------------------------------------");

    let grouped_dfs = df.par_groupby(&["category"])?;

    // Display each group
    for (key, group_df) in &grouped_dfs {
        println!("\nGroup: {}", key);
        println!("{:?}", group_df);

        // Calculate some statistics for each group
        let value_col = group_df.get_int_column("value")?;
        let sum: i64 = value_col.iter().filter_map(|v| *v).sum();
        let count = value_col.iter().filter_map(|v| *v).count();
        let mean = if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        };

        println!("  Sum: {}, Count: {}, Mean: {:.2}", sum, count, mean);
    }

    // Demonstrate manual aggregation
    println!("\n2. Manual aggregation calculation");
    println!("-------------------------------");

    // Calculate aggregated results manually
    let mut category_stats = std::collections::HashMap::new();

    for (key, group_df) in &grouped_dfs {
        let value_col = group_df.get_int_column("value")?;
        let price_col = group_df.get_float_column("price")?;

        // Calculate statistics
        let value_sum: i64 = value_col.iter().filter_map(|v| *v).sum();
        let value_count = value_col.iter().filter_map(|v| *v).count() as i64;
        let value_mean = if value_count > 0 {
            value_sum as f64 / value_count as f64
        } else {
            0.0
        };

        let price_sum: f64 = price_col.iter().sum();
        let price_max = price_col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        category_stats.insert(key.clone(), (value_mean, price_sum, price_max));
    }

    println!("Category statistics:");
    for (category, (value_mean, price_sum, price_max)) in &category_stats {
        println!(
            "  {}: value_mean={:.2}, price_sum={:.2}, price_max={:.2}",
            category, value_mean, price_sum, price_max
        );
    }

    // Create a summary DataFrame
    println!("\n3. Creating summary DataFrame");
    println!("----------------------------");

    let mut summary_df = OptimizedDataFrame::new();

    let mut categories_vec = Vec::new();
    let mut value_means = Vec::new();
    let mut price_sums = Vec::new();
    let mut price_maxes = Vec::new();

    for (category, (value_mean, price_sum, price_max)) in category_stats {
        categories_vec.push(category);
        value_means.push(value_mean);
        price_sums.push(price_sum);
        price_maxes.push(price_max);
    }

    summary_df.add_string_column("category", categories_vec)?;
    summary_df.add_float_column("value_mean", value_means)?;
    summary_df.add_float_column("price_sum", price_sums)?;
    summary_df.add_float_column("price_max", price_maxes)?;

    println!("Summary DataFrame:");
    println!("{:?}", summary_df);

    println!("\nExample completed successfully!");

    Ok(())
}

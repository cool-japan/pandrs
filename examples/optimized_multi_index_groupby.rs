use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;

fn main() -> Result<()> {
    println!("=== Multi-Index GroupBy Example ===");

    // Create a sample DataFrame
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories = vec!["A", "A", "B", "B", "A", "B", "A", "B", "C", "C"];
    let regions = vec![
        "East", "West", "East", "West", "East", "East", "West", "West", "East", "West",
    ];
    let values = vec![10, 15, 20, 25, 12, 22, 18, 24, 30, 35];
    let scores = vec![85.5, 92.3, 77.8, 88.9, 90.2, 82.5, 94.7, 79.3, 88.1, 91.6];

    df.add_string_column(
        "category",
        categories.iter().map(|s| s.to_string()).collect(),
    )?;
    df.add_string_column("region", regions.iter().map(|s| s.to_string()).collect())?;
    df.add_int_column("value", values)?;
    df.add_float_column("score", scores)?;

    // Display original DataFrame
    println!("\nOriginal DataFrame:");
    println!("{:?}", df);

    // Group by single column
    println!("\n=== Group by 'category' ===");
    let grouped_by_category = df.par_groupby(&["category"])?;

    println!("\nResult of groupby with single column:");
    for (key, group_df) in &grouped_by_category {
        println!("Group: {}", key);
        println!("  Rows: {}", group_df.row_count());

        // Calculate some simple statistics for each group
        if let (Ok(value_col), Ok(score_col)) = (
            group_df.get_int_column("value"),
            group_df.get_float_column("score"),
        ) {
            let value_sum: i64 = value_col.iter().filter_map(|v| *v).sum();
            let score_avg: f64 = score_col.iter().sum::<f64>() / score_col.len() as f64;
            println!("  Value sum: {}, Score avg: {:.2}", value_sum, score_avg);
        }
    }

    // Group by multiple columns
    println!("\n=== Group by 'category' and 'region' ===");
    let grouped_with_multi = df.par_groupby(&["category", "region"])?;

    println!("\nResult of multi-column groupby:");
    for (key, group_df) in &grouped_with_multi {
        println!("Group: {}", key);
        println!("  Rows: {}", group_df.row_count());
    }

    println!("\nTotal groups: {}", grouped_with_multi.len());

    // Show group filtering example
    println!("\n=== Filtered groups (groups with more than 1 row) ===");
    let large_groups: Vec<_> = grouped_with_multi
        .iter()
        .filter(|(_, group)| group.row_count() > 1)
        .collect();

    println!("Large groups found: {}", large_groups.len());
    for (key, group_df) in large_groups {
        println!("Group: {} has {} rows", key, group_df.row_count());
    }

    println!("\nExample completed successfully!");

    Ok(())
}

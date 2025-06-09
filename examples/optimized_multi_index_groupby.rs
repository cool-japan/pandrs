use pandrs::error::Result;
use pandrs::optimized::split_dataframe::group::AggregateOp;
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
    println!("{}", df);

    // Group by single column (without multi-index)
    println!("\n=== Group by 'category' (without multi-index) ===");
    let grouped_by_category = df.group_by(["category"])?.aggregate(vec![
        (
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        ),
        (
            "score".to_string(),
            AggregateOp::Mean,
            "score_mean".to_string(),
        ),
    ])?;

    println!("\nResult of groupby with single column:");
    println!("{}", grouped_by_category);

    // Group by multiple columns with multi-index
    println!("\n=== Group by 'category' and 'region' with multi-index ===");
    let grouped_with_multi_index = df
        .group_by_with_options(["category", "region"], true)?
        .aggregate(vec![
            (
                "value".to_string(),
                AggregateOp::Sum,
                "value_sum".to_string(),
            ),
            (
                "score".to_string(),
                AggregateOp::Mean,
                "score_mean".to_string(),
            ),
        ])?;

    println!("\nResult of groupby with multi-index:");
    println!("{}", grouped_with_multi_index);

    // Demonstrate parallel operation with multi-index
    println!("\n=== Parallel group by with multi-index ===");
    let parallel_grouped = df
        .group_by_with_options(["category", "region"], true)?
        .par_aggregate(vec![
            (
                "value".to_string(),
                AggregateOp::Sum,
                "value_sum".to_string(),
            ),
            (
                "score".to_string(),
                AggregateOp::Mean,
                "score_mean".to_string(),
            ),
            (
                "value".to_string(),
                AggregateOp::Max,
                "value_max".to_string(),
            ),
            (
                "score".to_string(),
                AggregateOp::Min,
                "score_min".to_string(),
            ),
        ])?;

    println!("\nResult of parallel groupby with multi-index:");
    println!("{}", parallel_grouped);

    // Custom aggregation with multi-index
    println!("\n=== Custom aggregation with multi-index ===");

    // Define a custom coefficient of variation function
    let cv = |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let sum: f64 = values.iter().sum();
        let count = values.len() as f64;
        let mean = sum / count;

        if mean == 0.0 {
            return 0.0; // Avoid division by zero
        }

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count;

        let std_dev = variance.sqrt();

        // Coefficient of variation = (std_dev / mean) * 100
        (std_dev / mean) * 100.0
    };

    let custom_result = df
        .group_by_with_options(["category", "region"], true)?
        .custom("score", "score_cv", cv)?;

    println!("\nResult of custom aggregation with multi-index:");
    println!("{}", custom_result);

    // Parallel custom aggregation with multi-index
    println!("\n=== Parallel custom aggregation with multi-index ===");

    let par_custom_result = df
        .group_by_with_options(["category", "region"], true)?
        .par_custom("score", "score_cv", cv)?;

    println!("\nResult of parallel custom aggregation with multi-index:");
    println!("{}", par_custom_result);

    // Compare with multiple columns without multi-index
    println!("\n=== Group by multiple columns without multi-index ===");
    let grouped_flat = df
        .group_by_with_options(["category", "region"], false)?
        .aggregate(vec![
            (
                "value".to_string(),
                AggregateOp::Sum,
                "value_sum".to_string(),
            ),
            (
                "score".to_string(),
                AggregateOp::Mean,
                "score_mean".to_string(),
            ),
        ])?;

    println!("\nResult of groupby with multiple columns (flat index):");
    println!("{}", grouped_flat);

    Ok(())
}

#[cfg(feature = "optimized")]
use pandrs::error::Error;
#[cfg(feature = "optimized")]
use pandrs::{AggregateOp, Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_groupby_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
fn main() -> Result<(), Error> {
    println!("=== Sample of Optimized Group Operations ===");

    // Prepare data
    let mut df = OptimizedDataFrame::new();

    // Column of values
    let values = Int64Column::new(vec![10, 20, 15, 30, 25, 15]);
    df.add_column("values", Column::Int64(values))?;

    // Keys for grouping
    let categories = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
        "B".to_string(),
        "A".to_string(),
    ]);
    df.add_column("category", Column::String(categories))?;

    println!("Original Data:");
    println!("{:?}", df);

    // Calculate group sizes
    let sizes = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Count, "size".to_string())],
        )
        .execute()?;

    println!("\n--- Group Sizes ---");
    println!("{:?}", sizes);

    // Calculate sum for each group
    let sums = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Sum, "sum".to_string())],
        )
        .execute()?;

    println!("\n--- Sum by Group ---");
    println!("{:?}", sums);

    // Calculate mean for each group
    let means = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![("values".to_string(), AggregateOp::Mean, "mean".to_string())],
        )
        .execute()?;

    println!("\n--- Mean by Group ---");
    println!("{:?}", means);

    // Calculate multiple statistics at once
    let all_stats = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![
                (
                    "values".to_string(),
                    AggregateOp::Count,
                    "count".to_string(),
                ),
                ("values".to_string(), AggregateOp::Sum, "sum".to_string()),
                ("values".to_string(), AggregateOp::Mean, "mean".to_string()),
                ("values".to_string(), AggregateOp::Min, "min".to_string()),
                ("values".to_string(), AggregateOp::Max, "max".to_string()),
            ],
        )
        .execute()?;

    println!("\n--- Multiple Statistics by Group ---");
    println!("{:?}", all_stats);

    // Grouping with different data types
    println!("\n--- Grouping with Different Data Types ---");

    let mut age_df = OptimizedDataFrame::new();
    let ages = Int64Column::new(vec![25, 30, 25, 40, 30, 25]);
    let values = Int64Column::new(vec![10, 20, 15, 30, 25, 15]);

    age_df.add_column("age", Column::Int64(ages))?;
    age_df.add_column("values", Column::Int64(values))?;

    let age_means = LazyFrame::new(age_df)
        .aggregate(
            vec!["age".to_string()],
            vec![("values".to_string(), AggregateOp::Mean, "mean".to_string())],
        )
        .execute()?;

    println!("Mean by Age:");
    println!("{:?}", age_means);

    println!("=== Group Operations Sample Complete ===");
    Ok(())
}

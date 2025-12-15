//! Example of Distributed Processing with PandRS
//!
//! This example demonstrates how to use the distributed processing module
//! to process a dataset across multiple executors.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::Series;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Distributed Processing Example");
    println!("-------------------------------------");

    // Create a sample DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("id".to_string()))?,
    )?;
    df.add_column(
        "value".to_string(),
        Series::new(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            Some("value".to_string()),
        )?,
    )?;
    df.add_column(
        "category".to_string(),
        Series::new(
            vec!["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            Some("category".to_string()),
        )?,
    )?;

    println!("\nOriginal DataFrame:");
    println!("{:?}\n", df);

    // Configure distributed processing
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_optimization(true);

    println!("Distributed Configuration:");
    println!("  Executor: datafusion");
    println!("  Concurrency: 2");
    println!("  Optimization: enabled");

    // Create distributed context
    let mut ctx = DistributedContext::new(config)?;

    // Register the DataFrame
    ctx.register_dataframe("sample_data", &df)?;

    println!("\nDataFrame registered in distributed context.");
    println!("Available operations:");
    println!("  - SQL queries via sql_to_dataframe()");
    println!("  - DataFrame operations via get_dataset()");
    println!("  - Join, filter, aggregate operations");

    println!("\n=== Distributed Processing Example Completed ===");
    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("PandRS Distributed Processing Example");
    println!("-------------------------------------");
    println!();
    println!("This example requires the 'distributed' feature flag.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_example --features distributed");
}

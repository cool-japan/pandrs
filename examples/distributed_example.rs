//! Example of Distributed Processing with PandRS
//!
//! This example demonstrates how to use the distributed processing module
//! to process a large dataset across multiple executors.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::distributed::datafusion::DataFusionContext;
#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, ToDistributed};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::optimized::OptimizedDataFrame;

#[cfg(feature = "distributed")]
fn main() -> Result<()> {
    println!("PandRS Distributed Processing Example");

    // Create a data frame
    let mut df = create_test_data()?;
    println!("Original DataFrame:\n{:?}\n", df);

    // Configure distributed processing
    let config = DistributedConfig::default()
        .with_executor_count(2)
        .with_partition_size(5);

    println!("Distributed Configuration:");
    println!("- Executor Count: {}", config.executor_count);
    println!("- Partition Size: {}", config.partition_size);

    // Create a distributed context
    let ctx = DataFusionContext::new(config);
    println!("\nCreated DataFusion Context");

    // Convert DataFrame to a distributed DataFrame
    let dist_df = df.to_distributed(&ctx)?;
    println!("Converted to Distributed DataFrame");
    println!("- Partitions: {}", dist_df.partition_count());

    // Simple transformation: filter rows
    println!("\nPerforming filter operation...");
    let filtered = dist_df.filter("value > 50")?;

    // Execute and collect results
    println!("Executing and collecting results...");
    let result = filtered.collect()?;

    println!("\nFiltered Result DataFrame (value > 50):");
    println!("{:?}", result);

    // Perform aggregation
    println!("\nPerforming aggregation operation...");
    let agg = dist_df
        .group_by("category")
        .aggregate("value", "max")
        .aggregate("id", "count")?;

    // Execute and collect results
    println!("Executing and collecting results...");
    let agg_result = agg.collect()?;

    println!("\nAggregation Result DataFrame:");
    println!("{:?}", agg_result);

    Ok(())
}

#[cfg(feature = "distributed")]
/// Create a test DataFrame for the example
fn create_test_data() -> Result<OptimizedDataFrame> {
    use pandrs::column::{Column, Float64Column, Int64Column, StringColumn};
    use pandrs::optimized::OptimizedDataFrame;

    // Create a test DataFrame
    let mut df = OptimizedDataFrame::new();

    // Add ID column
    let ids = Int64Column::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    df.add_column("id".to_string(), Column::Int64(ids))?;

    // Add Value column
    let values = Float64Column::new(vec![
        20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,
    ]);
    df.add_column("value".to_string(), Column::Float64(values))?;

    // Add Category column
    let categories = StringColumn::new(vec![
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "C".to_string(),
        "C".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "A".to_string(),
    ]);
    df.add_column("category".to_string(), Column::String(categories))?;

    Ok(df)
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!(
        "Please recompile with 'cargo run --example distributed_example --features distributed'"
    );
}

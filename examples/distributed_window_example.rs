//! Example of using Window Functions with Distributed Processing
//!
//! This example demonstrates the concept of window functions in the
//! distributed processing framework.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::series::Series;
#[cfg(feature = "distributed")]
use pandrs::DataFrame;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS Distributed Window Functions Example ===\n");

    // Create a data frame
    let df = create_test_data()?;
    println!("Created test DataFrame with {} rows", df.row_count());
    println!("Columns: {:?}\n", df.column_names());

    // Configure distributed processing
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2);

    // Create a context and register the data
    let mut context = DistributedContext::new(config)?;
    context.register_dataframe("test_data", &df)?;

    // Verify registration
    if context.get_dataset("test_data").is_some() {
        println!("Successfully registered DataFrame in distributed context");
    }

    // Describe window function concepts
    println!("\n--- Window Functions Overview ---");
    println!("Window functions perform calculations across related rows:");
    println!("  - ROW_NUMBER: Sequential row numbering");
    println!("  - RANK: Ranking with gaps for ties");
    println!("  - DENSE_RANK: Ranking without gaps");
    println!("  - SUM/AVG/COUNT: Aggregate over window");
    println!("  - LAG/LEAD: Access values from other rows");
    println!("  - FIRST_VALUE/LAST_VALUE: First or last value in window");

    println!("\n--- Window Frame Types ---");
    println!("  - ROWS BETWEEN: Physical row-based frame");
    println!("  - RANGE BETWEEN: Logical value-based frame");
    println!("  - UNBOUNDED PRECEDING: From start of partition");
    println!("  - CURRENT ROW: Current row");
    println!("  - UNBOUNDED FOLLOWING: To end of partition");

    println!("\n=== Distributed Window Example Complete ===");

    Ok(())
}

#[cfg(feature = "distributed")]
/// Create a test DataFrame for the example
#[allow(clippy::result_large_err)]
fn create_test_data() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Add ID column
    df.add_column(
        "id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("id".to_string()))?,
    )?;

    // Add Value column
    df.add_column(
        "value".to_string(),
        Series::new(
            vec![55.0, 30.0, 40.0, 85.0, 60.0, 75.0, 45.0, 90.0, 25.0, 50.0],
            Some("value".to_string()),
        )?,
    )?;

    // Add Category column
    df.add_column(
        "category".to_string(),
        Series::new(
            vec!["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            Some("category".to_string()),
        )?,
    )?;

    Ok(df)
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_window_example --features distributed");
}

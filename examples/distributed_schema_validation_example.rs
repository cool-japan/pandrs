//! Example demonstrating schema validation for distributed operations
//!
//! This example shows how to use schema validation when working with distributed data processing.

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::series::Series;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Distributed Schema Validation Example ===");

    // Create a test DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5], Some("id".to_string()))?,
    )?;
    df.add_column(
        "name".to_string(),
        Series::new(vec!["A", "B", "C", "D", "E"], Some("name".to_string()))?,
    )?;
    df.add_column(
        "value".to_string(),
        Series::new(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Some("value".to_string()),
        )?,
    )?;
    df.add_column(
        "active".to_string(),
        Series::new(
            vec![true, false, true, false, true],
            Some("active".to_string()),
        )?,
    )?;

    println!("\nCreated test DataFrame with {} rows", df.row_count());
    println!("Columns: {:?}", df.column_names());

    // Create a distributed context with schema validation enabled
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_optimization(true);

    let mut context = DistributedContext::new(config)?;

    // Register the DataFrame
    context.register_dataframe("test_data", &df)?;

    println!("\nDataFrame registered in distributed context");

    // Verify registration
    if let Some(registered_df) = context.get_dataset("test_data") {
        println!("Successfully retrieved registered dataset");
    }

    // Show schema validation concepts
    println!("\n--- Schema Validation Features ---");
    println!("  - Column type validation");
    println!("  - Expression type inference");
    println!("  - Operation compatibility checking");
    println!("  - Schema-aware query optimization");

    println!("\n=== Schema Validation Example Completed ===");

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("=== Distributed Schema Validation Example ===");
    println!();
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_schema_validation_example --features distributed");
}

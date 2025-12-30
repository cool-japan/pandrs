//! Example of using the DistributedContext API
//!
//! This example demonstrates how to use the DistributedContext API to
//! manage datasets in a distributed context.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::series::Series;

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_context_example --features distributed");
}

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS Distributed Context Example ===\n");

    // Create configuration
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(4)
        .with_memory_limit_str("1GB");

    // Create distributed context
    let mut context = DistributedContext::new(config)?;

    println!("Created distributed context");

    // Create first DataFrame for customers
    let mut customers = DataFrame::new();
    customers.add_column(
        "customer_id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5], Some("customer_id".to_string()))?,
    )?;
    customers.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice", "Bob", "Carol", "Dave", "Eve"],
            Some("name".to_string()),
        )?,
    )?;
    customers.add_column(
        "city".to_string(),
        Series::new(
            vec!["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
            Some("city".to_string()),
        )?,
    )?;

    println!(
        "Created customers DataFrame with {} rows",
        customers.row_count()
    );

    // Create second DataFrame for orders
    let mut orders = DataFrame::new();
    orders.add_column(
        "order_id".to_string(),
        Series::new(
            vec![101, 102, 103, 104, 105, 106, 107],
            Some("order_id".to_string()),
        )?,
    )?;
    orders.add_column(
        "customer_id".to_string(),
        Series::new(vec![1, 2, 1, 3, 2, 4, 1], Some("customer_id".to_string()))?,
    )?;
    orders.add_column(
        "amount".to_string(),
        Series::new(
            vec![100.0, 150.0, 50.0, 200.0, 75.0, 225.0, 80.0],
            Some("amount".to_string()),
        )?,
    )?;

    println!("Created orders DataFrame with {} rows", orders.row_count());

    // Register DataFrames with the context
    println!("\nRegistering 'customers' and 'orders' datasets...");
    context.register_dataframe("customers", &customers)?;
    context.register_dataframe("orders", &orders)?;

    // Verify registration by retrieving datasets
    if context.get_dataset("customers").is_some() {
        println!("Successfully retrieved 'customers' dataset");
    }

    if context.get_dataset("orders").is_some() {
        println!("Successfully retrieved 'orders' dataset");
    }

    // Show what distributed features are available
    println!("\n--- Distributed Context Features ---");
    println!("  - Dataset registration and management");
    println!("  - Memory-limited execution");
    println!("  - Concurrent processing with configurable parallelism");
    println!("  - DataFusion query engine integration");

    println!("\n=== Distributed Context Example Complete ===");

    Ok(())
}

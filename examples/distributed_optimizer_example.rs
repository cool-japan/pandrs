//! Example demonstrating query optimization and plan explanation for distributed processing

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext, ToDistributed};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::series::Series;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Create test data
    let mut orders_df = DataFrame::new();
    orders_df.add_column(
        "order_id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5], Some("order_id".to_string()))?,
    )?;
    orders_df.add_column(
        "customer_id".to_string(),
        Series::new(
            vec![101, 102, 101, 103, 102],
            Some("customer_id".to_string()),
        )?,
    )?;
    orders_df.add_column(
        "amount".to_string(),
        Series::new(
            vec![100.0, 200.0, 150.0, 300.0, 250.0],
            Some("amount".to_string()),
        )?,
    )?;
    orders_df.add_column(
        "date".to_string(),
        Series::new(
            vec![
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            Some("date".to_string()),
        )?,
    )?;

    let mut customers_df = DataFrame::new();
    customers_df.add_column(
        "customer_id".to_string(),
        Series::new(vec![101, 102, 103, 104], Some("customer_id".to_string()))?,
    )?;
    customers_df.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice", "Bob", "Charlie", "David"],
            Some("name".to_string()),
        )?,
    )?;
    customers_df.add_column(
        "region".to_string(),
        Series::new(
            vec!["North", "South", "East", "West"],
            Some("region".to_string()),
        )?,
    )?;

    println!("Orders data:");
    println!("{:?}", orders_df);
    println!("\nCustomers data:");
    println!("{:?}", customers_df);

    // 1. Run with optimization disabled
    println!("\n\n=== Running with optimization DISABLED ===\n");

    let config_no_opt = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_optimization(false);

    let mut context_no_opt = DistributedContext::new(config_no_opt)?;
    context_no_opt.register_dataframe("orders", &orders_df)?;
    context_no_opt.register_dataframe("customers", &customers_df)?;

    // Check registered datasets
    if let Some(_orders_dataset) = context_no_opt.get_dataset("orders") {
        println!("Orders dataset registered successfully");
    }

    println!("Distributed context created with optimization disabled");

    // 2. Run with optimization enabled
    println!("\n\n=== Running with optimization ENABLED ===\n");

    let config_opt = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_optimization(true)
        // Configure specific optimizations
        .with_optimizer_rule("filter_pushdown", true)
        .with_optimizer_rule("join_reordering", true);

    let mut context_opt = DistributedContext::new(config_opt)?;
    context_opt.register_dataframe("orders", &orders_df)?;
    context_opt.register_dataframe("customers", &customers_df)?;

    println!("Distributed context created with optimization enabled");
    println!("  - Filter pushdown: enabled");
    println!("  - Join reordering: enabled");

    println!("\n=== Distributed query optimization example completed ===");

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_optimizer_example --features distributed");
}

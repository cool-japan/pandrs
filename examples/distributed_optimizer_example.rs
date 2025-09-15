//! Example demonstrating query optimization and plan explanation for distributed processing

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::execution::JoinType;
#[cfg(feature = "distributed")]
use pandrs::distributed::{
    DistributedConfig, DistributedContext, ExplainFormat, ExplainOptions, ToDistributed,
};
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
        Series::new(vec![1, 2, 3, 4, 5], Some("order_id".to_string())),
    )?;
    orders_df.add_column(
        "customer_id".to_string(),
        Series::new(
            vec![101, 102, 101, 103, 102],
            Some("customer_id".to_string()),
        ),
    )?;
    orders_df.add_column(
        "amount".to_string(),
        Series::new(
            vec![100.0, 200.0, 150.0, 300.0, 250.0],
            Some("amount".to_string()),
        ),
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
        ),
    )?;

    let mut customers_df = DataFrame::new();
    customers_df.add_column(
        "customer_id".to_string(),
        Series::new(vec![101, 102, 103, 104], Some("customer_id".to_string())),
    )?;
    customers_df.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice", "Bob", "Charlie", "David"],
            Some("name".to_string()),
        ),
    )?;
    customers_df.add_column(
        "region".to_string(),
        Series::new(
            vec!["North", "South", "East", "West"],
            Some("region".to_string()),
        ),
    )?;

    println!("Orders data:");
    println!("{}", orders_df);
    println!("\nCustomers data:");
    println!("{}", customers_df);

    // 1. Run with optimization disabled
    println!("\n\n=== Running with optimization DISABLED ===\n");

    let config_no_opt = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_optimization(false);

    let mut context_no_opt = DistributedContext::new(config_no_opt)?;
    context_no_opt.register_dataframe("orders", &orders_df)?;
    context_no_opt.register_dataframe("customers", &customers_df)?;

    // Create a query with join, filter, and aggregation
    let query = context_no_opt
        .dataset("orders")?
        .join(
            "customers",
            JoinType::Inner,
            &["customer_id"],
            &["customer_id"],
        )?
        .filter("amount > 100.0")?
        .groupby(&["region"])?
        .aggregate(&["amount"], &["sum"])?;

    // Explain the query plan
    println!("Unoptimized Query Plan:");
    println!("{}\n", query.explain(false)?);

    // Execute and display results
    println!("Query Results (optimization disabled):");
    let results_no_opt = query.collect()?;
    println!("{}", results_no_opt);

    if let Some(metrics) = query.execution_metrics() {
        println!("\nExecution Metrics (optimization disabled):");
        println!("Execution time: {}ms", metrics.execution_time_ms());
        println!("Rows processed: {}", metrics.rows_processed());
    }

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

    // Create the same query
    let query_opt = context_opt
        .dataset("orders")?
        .join(
            "customers",
            JoinType::Inner,
            &["customer_id"],
            &["customer_id"],
        )?
        .filter("amount > 100.0")?
        .groupby(&["region"])?
        .aggregate(&["amount"], &["sum"])?;

    // Explain the query plan with statistics
    println!("Optimized Query Plan (with statistics):");
    println!("{}\n", query_opt.explain(true)?);

    // Execute and display results
    println!("Query Results (optimization enabled):");
    let results_opt = query_opt.collect()?;
    println!("{}", results_opt);

    if let Some(metrics) = query_opt.execution_metrics() {
        println!("\nExecution Metrics (optimization enabled):");
        println!("Execution time: {}ms", metrics.execution_time_ms());
        println!("Rows processed: {}", metrics.rows_processed());
        println!("Partitions processed: {}", metrics.partitions_processed());
        println!("Bytes processed: {} bytes", metrics.bytes_processed());
    }

    // 3. Demonstrate SQL query with optimization
    println!("\n\n=== SQL Query Execution with Optimization ===\n");

    let sql_result = context_opt.sql_to_dataframe(
        "SELECT c.region, SUM(o.amount) as total_amount
         FROM orders o
         JOIN customers c ON o.customer_id = c.customer_id
         WHERE o.amount > 100.0
         GROUP BY c.region
         ORDER BY total_amount DESC",
    )?;

    println!("SQL Query Results:");
    println!("{}", sql_result);

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_optimizer_example --features distributed");
}

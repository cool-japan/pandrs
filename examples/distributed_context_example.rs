//! Example of using the DistributedContext API for SQL-like operations
//!
//! This example demonstrates how to use the DistributedContext API to
//! manage multiple datasets and execute SQL queries against them.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::{DistributedConfig, DistributedContext};
#[cfg(feature = "distributed")]
use pandrs::error::Result;

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_context_example --features distributed");
}

#[cfg(feature = "distributed")]
fn main() -> Result<()> {
    println!("PandRS Distributed Context Example");
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
    customers.add_column("customer_id".to_string(), vec![1, 2, 3, 4, 5])?;
    customers.add_column(
        "name".to_string(),
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Carol".to_string(),
            "Dave".to_string(),
            "Eve".to_string(),
        ],
    )?;
    customers.add_column(
        "city".to_string(),
        vec![
            "New York".to_string(),
            "Los Angeles".to_string(),
            "Chicago".to_string(),
            "Houston".to_string(),
            "Phoenix".to_string(),
        ],
    )?;

    // Create second DataFrame for orders
    let mut orders = DataFrame::new();
    orders.add_column(
        "order_id".to_string(),
        vec![101, 102, 103, 104, 105, 106, 107],
    )?;
    orders.add_column("customer_id".to_string(), vec![1, 2, 1, 3, 2, 4, 1])?;
    orders.add_column(
        "amount".to_string(),
        vec![100.0, 150.0, 50.0, 200.0, 75.0, 225.0, 80.0],
    )?;
    orders.add_column(
        "date".to_string(),
        vec![
            "2023-01-15".to_string(),
            "2023-01-20".to_string(),
            "2023-02-05".to_string(),
            "2023-02-10".to_string(),
            "2023-02-15".to_string(),
            "2023-03-01".to_string(),
            "2023-03-10".to_string(),
        ],
    )?;

    // Register DataFrames with the context
    println!("Registering 'customers' and 'orders' datasets");
    context.register_dataframe("customers", &customers)?;
    context.register_dataframe("orders", &orders)?;

    // Get the list of registered datasets
    let dataset_names = context.dataset_names();
    println!("Registered datasets: {:?}", dataset_names);

    // Execute a SQL query to list all customers
    println!("\nQuery 1: List all customers");
    let result = context.sql_to_dataframe("SELECT * FROM customers ORDER BY customer_id")?;
    println!("{}", result);

    // Execute a SQL query to get order details for each customer
    println!("\nQuery 2: Order details for each customer");
    let result = context.sql_to_dataframe(
        "
        SELECT c.customer_id, c.name, o.order_id, o.amount, o.date
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        ORDER BY c.customer_id, o.order_id
    ",
    )?;
    println!("{}", result);

    // Execute a SQL query to get total order amount by customer
    println!("\nQuery 3: Total order amount by customer");
    let query = "
        SELECT c.customer_id, c.name, COUNT(o.order_id) as order_count, SUM(o.amount) as total_amount
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name
        ORDER BY total_amount DESC
    ";

    let start = std::time::Instant::now();
    let exec_result = context.sql(query)?;
    let elapsed = start.elapsed();

    // Show execution metrics
    println!("Query executed in {:.2?}", elapsed);
    println!("\nExecution Metrics:");
    println!("{}", exec_result.metrics().format());

    // Convert to DataFrame and display
    let result = exec_result.collect_to_local()?;
    println!("\nResult:");
    println!("{}", result);

    // Execute a complex analytical query
    println!("\nQuery 4: Monthly sales analysis");
    let query = "
        SELECT
            SUBSTR(o.date, 1, 7) as month,
            COUNT(DISTINCT o.customer_id) as unique_customers,
            COUNT(o.order_id) as order_count,
            SUM(o.amount) as total_sales,
            AVG(o.amount) as avg_order_value
        FROM orders o
        GROUP BY SUBSTR(o.date, 1, 7)
        ORDER BY month
    ";

    let result = context.sql_to_dataframe(query)?;
    println!("{}", result);

    // Save query results to parquet file
    println!("\nSaving customer order summary to Parquet");
    let query = "
        SELECT c.customer_id, c.name, c.city, COUNT(o.order_id) as order_count, SUM(o.amount) as total_spent
        FROM customers c
        LEFT JOIN orders o ON c.customer_id = o.customer_id
        GROUP BY c.customer_id, c.name, c.city
        ORDER BY total_spent DESC
    ";

    // Use temporary file for example
    let temp_file = std::env::temp_dir().join("customer_summary.parquet");
    let temp_path = temp_file.to_str().unwrap();

    let metrics = context.sql_to_parquet(query, temp_path)?;
    println!("Data saved to {} with metrics:", temp_path);
    println!("{}", metrics.format());

    Ok(())
}

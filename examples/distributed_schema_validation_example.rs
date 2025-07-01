//! Example demonstrating schema validation for distributed operations

#[cfg(feature = "distributed")]
use pandrs::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use pandrs::distributed::{
    ColumnMeta, ColumnProjection, DistributedConfig, DistributedContext, Expr, ExprDataType,
    ExprSchema, ToDistributed,
};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::series::Series;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Create a test DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5], Some("id".to_string())),
    )?;
    df.add_column(
        "name".to_string(),
        Series::new(vec!["A", "B", "C", "D", "E"], Some("name".to_string())),
    )?;
    df.add_column(
        "value".to_string(),
        Series::new(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Some("value".to_string()),
        ),
    )?;
    df.add_column(
        "active".to_string(),
        Series::new(
            vec![true, false, true, false, true],
            Some("active".to_string()),
        ),
    )?;

    println!("Created DataFrame:");
    println!("{}", df);

    // Create a distributed context
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2);

    let mut context = DistributedContext::new(config)?;

    // Register the DataFrame with the context
    context.register_dataframe("test_data", &df)?;

    // Get the distributed DataFrame
    let dist_df = context.dataset("test_data")?;

    println!("\nPerforming valid operations (should pass validation):");

    // Valid operation: Select existing columns
    println!("\n1. Selecting existing columns (id, name):");
    let result = dist_df.select(&["id", "name"])?.collect()?;
    println!("{}", result);

    // Valid operation: Filter with valid column
    println!("\n2. Filtering with valid column (id > 2):");
    let result = dist_df.filter("id > 2")?.collect()?;
    println!("{}", result);

    // Valid operation: Using expressions with compatible types
    println!("\n3. Using expressions with compatible types (value * 2):");
    let result = dist_df
        .select_expr(&[
            ColumnProjection::column("id"),
            ColumnProjection::column("name"),
            ColumnProjection::with_alias(Expr::col("value").mul(Expr::lit(2.0)), "doubled_value"),
        ])?
        .collect()?;
    println!("{}", result);

    println!("\nPerforming invalid operations (should fail validation):");

    // Invalid operation: Select non-existent column
    println!("\n4. Attempting to select non-existent column (should fail):");
    match dist_df.select(&["id", "nonexistent"])?.collect() {
        Ok(_) => println!("❌ Operation succeeded but should have failed!"),
        Err(e) => println!("✅ Operation failed as expected: {}", e),
    }

    // Invalid operation: Filter with invalid column
    println!("\n5. Attempting to filter with invalid column (should fail):");
    match dist_df.filter("nonexistent > 0")?.collect() {
        Ok(_) => println!("❌ Operation succeeded but should have failed!"),
        Err(e) => println!("✅ Operation failed as expected: {}", e),
    }

    // Invalid operation: Expression with incompatible types
    println!("\n6. Using expressions with incompatible types (should fail):");
    match dist_df
        .select_expr(&[ColumnProjection::with_alias(
            Expr::col("name").mul(Expr::col("value")), // String * Float is invalid
            "invalid_expr",
        )])?
        .collect()
    {
        Ok(_) => println!("❌ Operation succeeded but should have failed!"),
        Err(e) => println!("✅ Operation failed as expected: {}", e),
    }

    // Try disabling validation
    println!("\nDisabling schema validation:");
    let config_no_validation = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(2)
        .with_skip_validation(true);

    let mut context_no_validation = DistributedContext::new(config_no_validation)?;
    context_no_validation.register_dataframe("test_data", &df)?;
    let dist_df_no_validation = context_no_validation.dataset("test_data")?;

    // This would normally fail validation, but should pass now
    println!("\n7. Attempting operation with validation disabled:");
    match dist_df_no_validation
        .select(&["id", "nonexistent"])?
        .collect()
    {
        Ok(_) => {
            println!("Note: The operation was allowed to proceed without validation,");
            println!("      but will likely fail at runtime with DataFusion/Arrow errors");
        }
        Err(e) => println!("Failed anyway due to runtime errors: {}", e),
    }

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_schema_validation_example --features distributed");
}

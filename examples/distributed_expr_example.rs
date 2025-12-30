//! Example demonstrating the use of expressions in distributed processing

#[cfg(feature = "distributed")]
use pandrs::distributed::expr::{ColumnProjection, Expr, ExprDataType, UdfDefinition};
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
    println!("=== Distributed Expression Example ===\n");

    // Create a distributed context with configuration
    let config = DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(4);

    let mut context = DistributedContext::new(config)?;

    // Create and register a test DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "region".to_string(),
        Series::new(
            vec!["East", "West", "North", "South"],
            Some("region".to_string()),
        )?,
    )?;
    df.add_column(
        "sales".to_string(),
        Series::new(
            vec![1000.0, 1500.0, 1200.0, 800.0],
            Some("sales".to_string()),
        )?,
    )?;
    df.add_column(
        "profit".to_string(),
        Series::new(vec![200.0, 350.0, 280.0, 150.0], Some("profit".to_string()))?,
    )?;

    context.register_dataframe("sales", &df)?;
    println!("Registered sales DataFrame with {} rows", df.row_count());

    // Example 1: Basic column expressions
    println!("\n--- Example 1: Column Expressions ---");
    let col_region = Expr::col("region");
    let col_sales = Expr::col("sales");
    println!("Created column expression for 'region': {:?}", col_region);
    println!("Created column expression for 'sales': {:?}", col_sales);

    // Example 2: Calculated expressions
    println!("\n--- Example 2: Calculated Expressions ---");
    let sales_bonus = Expr::col("sales").mul(Expr::lit(1.1));
    println!("Created calculation: sales * 1.1: {:?}", sales_bonus);

    let profit_margin = Expr::col("profit")
        .div(Expr::col("sales"))
        .mul(Expr::lit(100.0));
    println!(
        "Created calculation: (profit / sales) * 100: {:?}",
        profit_margin
    );

    // Example 3: Filter expressions
    println!("\n--- Example 3: Filter Expressions ---");
    let high_sales = Expr::col("sales").gt(Expr::lit(1000.0));
    println!("Created filter: sales > 1000: {:?}", high_sales);

    let region_east = Expr::col("region").eq(Expr::lit("East"));
    println!("Created filter: region == 'East': {:?}", region_east);

    // Example 4: Column projections
    println!("\n--- Example 4: Column Projections ---");
    let simple_projection = ColumnProjection::column("region");
    let aliased_projection =
        ColumnProjection::with_alias(Expr::col("sales").mul(Expr::lit(1.1)), "sales_with_bonus");
    println!("Simple projection: {:?}", simple_projection);
    println!("Aliased projection: {:?}", aliased_projection);

    // Example 5: UDF definition
    println!("\n--- Example 5: User Defined Functions ---");
    let multiply_udf = UdfDefinition::new(
        "calculate_margin",
        ExprDataType::Float,
        vec![ExprDataType::Float, ExprDataType::Float],
        "(param0 / param1) * 100",
    );
    println!("Created UDF: {:?}", multiply_udf);

    // Verify context has the data
    if context.get_dataset("sales").is_some() {
        println!("\nVerified: Dataset 'sales' is registered in distributed context");
    }

    println!("\n=== Distributed Expression Example Complete ===");

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("Distributed Expression Example");
    println!("===============================");
    println!();
    println!("This example requires the 'distributed' feature flag.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_expr_example --features distributed");
}

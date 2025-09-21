//! Example demonstrating the use of expressions and user-defined functions in distributed processing

#[cfg(feature = "distributed")]
use pandrs::distributed::expr::{ColumnProjection, Expr, ExprDataType, UdfDefinition};
#[cfg(feature = "distributed")]
use pandrs::distributed::DistributedContext;
#[cfg(feature = "distributed")]
use pandrs::error::Result;

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Create a new distributed context
    let mut context = DistributedContext::new_local(4)?;

    // Register a CSV file as a dataset
    context.register_csv("sales", "examples/data/sales.csv")?;

    // Get the dataset as a distributed DataFrame
    let sales_df = context.dataset("sales")?;

    println!("Original DataFrame:");
    let result = sales_df.collect()?;
    println!("{}", result);

    // Example 1: Basic column expressions
    println!("\n--- Example 1: Column Selection ---");

    // Select specific columns with expressions
    let selected = sales_df.select_expr(&[
        ColumnProjection::column("region"),
        ColumnProjection::column("sales"),
        // Calculate a new column
        ColumnProjection::with_alias(Expr::col("sales").mul(Expr::lit(1.1)), "sales_with_bonus"),
    ])?;

    println!("Selected columns with expression:");
    let result = selected.collect()?;
    println!("{}", result);

    // Example 2: Complex expressions with calculations
    println!("\n--- Example 2: Complex Calculations ---");

    // Add calculated columns
    let with_calcs = sales_df.with_column(
        "profit_margin",
        Expr::col("profit")
            .div(Expr::col("sales"))
            .mul(Expr::lit(100.0)),
    )?;

    println!("With calculated columns:");
    let result = with_calcs.collect()?;
    println!("{}", result);

    // Example 3: Filtering with expressions
    println!("\n--- Example 3: Expression-based Filtering ---");

    // Filter using complex expressions
    let high_margin = sales_df.filter_expr(
        Expr::col("profit")
            .div(Expr::col("sales"))
            .mul(Expr::lit(100.0))
            .gt(Expr::lit(15.0)),
    )?;

    println!("High margin sales (> 15%):");
    let result = high_margin.collect()?;
    println!("{}", result);

    // Example 4: User-defined functions
    println!("\n--- Example 4: User Defined Functions ---");

    // Define a UDF for calculating a commission
    let commission_udf = UdfDefinition::new(
        "calculate_commission",
        ExprDataType::Float,
        vec![ExprDataType::Float, ExprDataType::Float],
        "CASE
            WHEN param1 / param0 > 0.2 THEN param0 * 0.05
            WHEN param1 / param0 > 0.1 THEN param0 * 0.03
            ELSE param0 * 0.01
         END",
    );

    // Register the UDF
    let with_udf = sales_df.create_udf(&[commission_udf])?;

    // Use the UDF in a query
    let with_commission = with_udf.select_expr(&[
        ColumnProjection::column("region"),
        ColumnProjection::column("sales"),
        ColumnProjection::column("profit"),
        ColumnProjection::with_alias(
            Expr::call(
                "calculate_commission",
                vec![Expr::col("sales"), Expr::col("profit")],
            ),
            "commission",
        ),
    ])?;

    println!("Sales with calculated commission:");
    let result = with_commission.collect()?;
    println!("{}", result);

    // Example 5: Combining multiple operations
    println!("\n--- Example 5: Combining Operations ---");

    // Chain operations together
    let final_analysis = sales_df
        // Add calculated columns
        .with_column("profit_pct", 
            Expr::col("profit").div(Expr::col("sales")).mul(Expr::lit(100.0))
        )?
        // Filter for high-profit regions
        .filter_expr(
            Expr::col("profit_pct").gt(Expr::lit(12.0))
        )?
        // Project final columns with calculations
        .select_expr(&[
            ColumnProjection::column("region"),
            ColumnProjection::column("product"),
            ColumnProjection::column("sales"),
            ColumnProjection::column("profit"),
            ColumnProjection::column("profit_pct"),
            // Calculate a bonus amount
            ColumnProjection::with_alias(
                Expr::col("profit")
                    .mul(Expr::lit(0.1))
                    .add(Expr::col("profit_pct").mul(Expr::lit(5.0))),
                "bonus"
            ),
        ])?;

    println!("Final analysis with combined operations:");
    let result = final_analysis.collect()?;
    println!("{}", result);

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example distributed_expr_example --features distributed");
}

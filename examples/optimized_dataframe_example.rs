#[cfg(feature = "optimized")]
use pandrs::{
    AggregateOp, Column, Float64Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn,
};
#[cfg(feature = "optimized")]
use std::error::Error;

// Translated Japanese comments and strings into English
#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_dataframe_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Sample of Optimized DataFrame and Lazy Evaluation ===\n");

    // Create an optimized DataFrame
    println!("1. Creating an Optimized DataFrame");
    let mut df = OptimizedDataFrame::new();

    // Add an integer column
    let id_data = Int64Column::new(vec![1, 2, 3, 4, 5]);
    df.add_column("id", Column::Int64(id_data))?;

    // Add a floating-point column
    let value_data = Float64Column::new(vec![10.1, 20.2, 30.3, 40.4, 50.5]);
    df.add_column("value", Column::Float64(value_data))?;

    // Add a string column
    let category_data = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
        "B".to_string(),
    ]);
    df.add_column("category", Column::String(category_data))?;

    // Display the DataFrame
    println!("\n{:?}\n", df);

    // Retrieve and manipulate columns
    println!("2. Retrieving and Manipulating Columns");
    let value_col = df.column("value")?;
    if let Some(float_col) = value_col.as_float64() {
        let sum = float_col.sum();
        let mean = float_col.mean().unwrap_or(0.0);
        println!("Sum of 'value' column: {}", sum);
        println!("Mean of 'value' column: {:.2}", mean);
    }

    // Filtering
    println!("\n3. Selecting Rows Where Category is 'A'");
    // First, create a boolean column
    let category_col = df.column("category")?;
    let mut is_a = vec![false; df.row_count()];
    if let Some(str_col) = category_col.as_string() {
        for (i, is_a_val) in is_a.iter_mut().enumerate().take(df.row_count()) {
            if let Ok(Some(val)) = str_col.get(i) {
                *is_a_val = val == "A";
            }
        }
    }

    // Add the boolean column to the DataFrame
    let bool_data = pandrs::BooleanColumn::new(is_a);
    df.add_column("is_a", Column::Boolean(bool_data))?;

    // Execute filtering
    let filtered_df = df.filter("is_a")?;
    println!("\n{:?}\n", filtered_df);

    // Data processing using lazy evaluation
    println!("4. Data Processing Using Lazy Evaluation");

    let lazy_df = LazyFrame::new(df.clone());

    // Define the processing (not executed yet)
    let result_lazy = lazy_df
        .select(&["id", "value", "category", "is_a"])
        .filter("is_a");

    // Explain the execution plan
    println!("\nExecution Plan:");
    println!("{}", result_lazy.explain());

    // Execute the computation
    println!("\nResult of Lazy Evaluation:");
    let result_df = result_lazy.execute()?;
    println!("{:?}\n", result_df);

    // Example of grouping and aggregation
    println!("5. Grouping and Aggregation");

    // Create a new DataFrame
    let mut sales_df = OptimizedDataFrame::new();

    // Product category column
    let category_data = StringColumn::new(vec![
        "Electronics".to_string(),
        "Furniture".to_string(),
        "Electronics".to_string(),
        "Furniture".to_string(),
        "Electronics".to_string(),
        "Food".to_string(),
        "Food".to_string(),
        "Electronics".to_string(),
        "Furniture".to_string(),
    ]);
    sales_df.add_column("category", Column::String(category_data))?;

    // Sales amount column
    let amount_data = Float64Column::new(vec![
        150.0, 230.5, 120.0, 450.5, 300.0, 50.0, 75.5, 200.0, 175.0,
    ]);
    sales_df.add_column("sales", Column::Float64(amount_data))?;

    println!("\nOriginal Sales Data:");
    println!("{:?}\n", sales_df);

    // Perform grouping and aggregation using lazy evaluation
    let lazy_sales = LazyFrame::new(sales_df);
    let agg_result = lazy_sales
        .aggregate(
            vec!["category".to_string()],
            vec![
                (
                    "sales".to_string(),
                    AggregateOp::Sum,
                    "total_sales".to_string(),
                ),
                (
                    "sales".to_string(),
                    AggregateOp::Mean,
                    "average_sales".to_string(),
                ),
                (
                    "sales".to_string(),
                    AggregateOp::Count,
                    "sales_count".to_string(),
                ),
            ],
        )
        .execute()?;

    println!("Category-wise Aggregation Results:");
    println!("{:?}\n", agg_result);

    Ok(())
}

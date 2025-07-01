#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::{AggregateOp, Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_transform_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Optimized Data Transformation ===\n");

    // Create sample data frame
    let mut df = OptimizedDataFrame::new();

    // ID column
    let id_col = StringColumn::new(vec!["1".to_string(), "2".to_string(), "3".to_string()]);
    df.add_column("id", Column::String(id_col))?;

    // Product category
    let category_col = StringColumn::new(vec![
        "Food".to_string(),
        "Electronics".to_string(),
        "Clothing".to_string(),
    ]);
    df.add_column("category", Column::String(category_col))?;

    // Monthly sales data
    let jan_col = Int64Column::new(vec![1000, 1500, 800]);
    df.add_column("January", Column::Int64(jan_col))?;

    let feb_col = Int64Column::new(vec![1200, 1300, 1100]);
    df.add_column("February", Column::Int64(feb_col))?;

    let mar_col = Int64Column::new(vec![900, 1800, 1400]);
    df.add_column("March", Column::Int64(mar_col))?;

    println!("Original DataFrame:");
    println!("{:?}", df);

    // Melt operation - Convert from wide format to long format
    println!("\n----- Melt Operation (Convert from Wide Format to Long Format) -----");

    let melted_df = df.melt(
        &["id", "category"],
        Some(&["January", "February", "March"]),
        Some("Month"),
        Some("Sales"),
    )?;

    println!("{:?}", melted_df);

    // DataFrame concatenation
    println!("\n----- DataFrame Concatenation -----");

    // Create additional data frame
    let mut df2 = OptimizedDataFrame::new();

    let id_col2 = StringColumn::new(vec!["4".to_string(), "5".to_string()]);
    df2.add_column("id", Column::String(id_col2))?;

    let category_col2 = StringColumn::new(vec!["Stationery".to_string(), "Furniture".to_string()]);
    df2.add_column("category", Column::String(category_col2))?;

    let jan_col2 = Int64Column::new(vec![500, 2000]);
    df2.add_column("January", Column::Int64(jan_col2))?;

    let feb_col2 = Int64Column::new(vec![600, 2200]);
    df2.add_column("February", Column::Int64(feb_col2))?;

    let mar_col2 = Int64Column::new(vec![700, 1900]);
    df2.add_column("March", Column::Int64(mar_col2))?;

    println!("Additional DataFrame:");
    println!("{:?}", df2);

    // Concatenate data frames vertically
    // Note: Here we assume the existence of the concat method in OptimizedDataFrame
    // Actual implementation may be required
    let concat_df = df.append(&df2)?;
    println!("Concatenated DataFrame:");
    println!("{:?}", concat_df);

    // Conditional aggregation - Using LazyFrame and filtering
    println!("\n----- Conditional Aggregation (Using LazyFrame) -----");

    // Condition: Calculate the total sales in March by category for rows where February sales are 1000 or more
    // First, create a boolean column for filtering rows that meet the condition
    let feb_sales = df.column("February")?;
    let mut is_high_sales = vec![false; df.row_count()];

    if let Some(int_col) = feb_sales.as_int64() {
        #[allow(clippy::needless_range_loop)]
        for i in 0..df.row_count() {
            if let Ok(Some(value)) = int_col.get(i) {
                is_high_sales[i] = value >= 1000;
            }
        }
    }

    // Add boolean column to DataFrame
    let bool_col = pandrs::BooleanColumn::new(is_high_sales);
    let mut filtered_df = df.clone();
    filtered_df.add_column("is_high_sales", Column::Boolean(bool_col))?;

    // Filtering and group aggregation
    let result = LazyFrame::new(filtered_df)
        .filter("is_high_sales")
        .aggregate(
            vec!["category".to_string()],
            vec![(
                "March".to_string(),
                AggregateOp::Sum,
                "sum_march".to_string(),
            )],
        )
        .execute()?;

    println!("Conditional Aggregation Result (Total March Sales by Category for Rows with February Sales of 1000 or More):");
    println!("{:?}", result);

    println!("\n=== Optimized Data Transformation Example Complete ===");

    Ok(())
}

#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::{Column, Float64Column, LazyFrame, OptimizedDataFrame, StringColumn};

// Translated Japanese comments and strings into English
#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_multi_index_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
fn main() -> Result<()> {
    println!("=== Optimized MultiIndex Example ===\n");

    // =========================================
    // Simulating Multi-Level Index
    // =========================================

    println!("--- Simulating Multi-Level Index with Multiple Columns ---");

    // Create an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // Level 1 column (Product Category)
    let category = vec![
        "Electronics".to_string(),
        "Electronics".to_string(),
        "Electronics".to_string(),
        "Electronics".to_string(),
        "Furniture".to_string(),
        "Furniture".to_string(),
        "Furniture".to_string(),
        "Furniture".to_string(),
    ];
    let category_col = StringColumn::new(category);
    df.add_column("category", Column::String(category_col))?;

    // Level 2 column (Product Name)
    let product = vec![
        "TV".to_string(),
        "TV".to_string(),
        "Computer".to_string(),
        "Computer".to_string(),
        "Table".to_string(),
        "Table".to_string(),
        "Chair".to_string(),
        "Chair".to_string(),
    ];
    let product_col = StringColumn::new(product);
    df.add_column("product", Column::String(product_col))?;

    // Level 3 column (Year)
    let year = vec![
        "2022".to_string(),
        "2023".to_string(),
        "2022".to_string(),
        "2023".to_string(),
        "2022".to_string(),
        "2023".to_string(),
        "2022".to_string(),
        "2023".to_string(),
    ];
    let year_col = StringColumn::new(year);
    df.add_column("year", Column::String(year_col))?;

    // Value column (Sales)
    let sales = vec![1000.0, 1200.0, 1500.0, 1800.0, 800.0, 900.0, 600.0, 700.0];
    let sales_col = Float64Column::new(sales);
    df.add_column("sales", Column::Float64(sales_col))?;

    println!("DataFrame with Multi-Level Index:");
    println!("{:?}", df);

    // =========================================
    // Aggregation Using Multi-Level Index
    // =========================================

    println!("\n--- Aggregation Using Multi-Level Index ---");

    // Aggregate sales by category and product
    let result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string(), "product".to_string()],
            vec![(
                "sales".to_string(),
                pandrs::AggregateOp::Sum,
                "total_sales".to_string(),
            )],
        )
        .execute()?;

    println!("Sales by Category and Product:");
    println!("{:?}", result);

    // Aggregate sales by category
    let category_result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["category".to_string()],
            vec![(
                "sales".to_string(),
                pandrs::AggregateOp::Sum,
                "total_sales".to_string(),
            )],
        )
        .execute()?;

    println!("\nSales by Category:");
    println!("{:?}", category_result);

    // Aggregate sales by year
    let year_result = LazyFrame::new(df)
        .aggregate(
            vec!["year".to_string()],
            vec![(
                "sales".to_string(),
                pandrs::AggregateOp::Sum,
                "total_sales".to_string(),
            )],
        )
        .execute()?;

    println!("\nSales by Year:");
    println!("{:?}", year_result);

    println!("\n=== MultiIndex Example Complete ===");
    Ok(())
}

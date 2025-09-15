#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::{Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_categorical_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Optimized Categorical Data ===\n");

    // ===========================================================
    // Create basic categorical data
    // ===========================================================

    println!("--- Create basic categorical data ---");

    // Create an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // City data (categorical)
    let cities = vec![
        "Tokyo".to_string(),
        "Osaka".to_string(),
        "Tokyo".to_string(),
        "Nagoya".to_string(),
        "Osaka".to_string(),
        "Tokyo".to_string(),
    ];

    // Use string pool to efficiently store categorical data in the optimized version
    let city_col = StringColumn::new(cities);
    df.add_column("city", Column::String(city_col))?;

    // Add population data
    let population = vec![1350, 980, 1380, 550, 990, 1360];
    let pop_col = Int64Column::new(population);
    df.add_column("population", Column::Int64(pop_col))?;

    println!("DataFrame with categorical data:");
    println!("{:?}", df);

    // ===========================================================
    // Analyze categorical data
    // ===========================================================

    println!("\n--- Analyze categorical data ---");

    // Aggregate by category
    let result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["city".to_string()],
            vec![
                (
                    "population".to_string(),
                    pandrs::AggregateOp::Count,
                    "count".to_string(),
                ),
                (
                    "population".to_string(),
                    pandrs::AggregateOp::Sum,
                    "total_population".to_string(),
                ),
                (
                    "population".to_string(),
                    pandrs::AggregateOp::Mean,
                    "avg_population".to_string(),
                ),
            ],
        )
        .execute()?;

    println!("Aggregation results by category:");
    println!("{:?}", result);

    // ===========================================================
    // Example of filtering categorical data
    // ===========================================================

    println!("\n--- Filter categorical data ---");

    // Example of filtering data for "Tokyo"
    // Note: Actual filtering requires creating a boolean column

    // Create a boolean column for Tokyo
    let mut is_tokyo = vec![false; df.row_count()];
    let city_view = df.column("city")?;

    if let Some(str_col) = city_view.as_string() {
        for (i, item) in is_tokyo.iter_mut().enumerate().take(df.row_count()) {
            if let Ok(Some(city)) = str_col.get(i) {
                *item = city == "Tokyo";
            }
        }
    }

    // Add boolean column to DataFrame
    let bool_col = pandrs::BooleanColumn::new(is_tokyo);
    let mut filtered_df = df.clone();
    filtered_df.add_column("is_tokyo", Column::Boolean(bool_col))?;

    // Execute filtering
    let tokyo_df = filtered_df.filter("is_tokyo")?;

    println!("Data for 'Tokyo' only:");
    println!("{:?}", tokyo_df);

    println!("\n=== Optimized Categorical Data Example Complete ===");
    Ok(())
}

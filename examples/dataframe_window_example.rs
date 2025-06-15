use pandrs::dataframe::{DataFrame, DataFrameWindowExt};
use pandrs::error::Result;
use pandrs::series::Series;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of DataFrame Window Operations ===\n");

    // Create sample data
    let df = create_sample_dataframe()?;

    // Display the contents of the DataFrame
    println!("Original DataFrame:");
    println!("{:?}\n", df);

    // 1. Example of rolling mean
    println!("1. Example of rolling mean");
    let df_rolling = df.rolling(3, "Price", "mean", None)?;
    println!("{:?}\n", df_rolling);

    // 2. Example of rolling sum
    println!("2. Example of rolling sum");
    let df_sum = df.rolling(3, "Quantity", "sum", Some("Quantity_RollingSum_3"))?;
    println!("{:?}\n", df_sum);

    // 3. Example of rolling standard deviation
    println!("3. Example of rolling standard deviation");
    let df_std = df.rolling(3, "Price", "std", None)?;
    println!("{:?}\n", df_std);

    // 4. Example of rolling minimum
    println!("4. Example of rolling minimum");
    let df_min = df.rolling(3, "Price", "min", None)?;
    println!("{:?}\n", df_min);

    // 5. Example of rolling maximum
    println!("5. Example of rolling maximum");
    let df_max = df.rolling(3, "Price", "max", None)?;
    println!("{:?}\n", df_max);

    // 6. Example of expanding mean
    println!("6. Example of expanding mean");
    let df_expanding = df.expanding(2, "Price", "mean", None)?;
    println!("{:?}\n", df_expanding);

    // 7. Example of Exponential Weighted Moving Average (EWM)
    println!("7. Example of Exponential Weighted Moving Average (EWM)");

    // When span is specified (span = 3)
    let df_ewm_span = df.ewm("Price", "mean", Some(3), None, Some("Price_ewm_span3"))?;
    println!("7.1 When span=3 is specified:");
    println!("{:?}\n", df_ewm_span);

    // When alpha is specified (alpha = 0.5)
    let df_ewm_alpha = df.ewm("Price", "mean", None, Some(0.5), Some("Price_ewm_alpha0.5"))?;
    println!("7.2 When alpha=0.5 is specified:");
    println!("{:?}\n", df_ewm_alpha);

    // 8. Example of applying multiple operations at once
    println!("8. Example of combining multiple operations:");

    // Apply multiple operations in sequence
    let mut result_df = df.clone();

    // Add rolling mean
    result_df = result_df.rolling(3, "Price", "mean", None)?;

    // Add expanding mean
    result_df = result_df.expanding(2, "Price", "mean", None)?;

    // Add Exponential Weighted Moving Average
    result_df = result_df.ewm("Price", "mean", Some(3), None, None)?;

    println!("{:?}\n", result_df);

    println!("=== DataFrame Window Operations Example Complete ===");
    Ok(())
}

// Helper function to create a sample DataFrame
#[allow(clippy::result_large_err)]
fn create_sample_dataframe() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Add date column
    let dates = vec![
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
        "2023-01-08",
        "2023-01-09",
        "2023-01-10",
    ];
    let date_series = Series::new(dates, Some("Date".to_string()))?;
    df.add_column("Date".to_string(), date_series)?;

    // Add product column
    let products = vec![
        "ProductA", "ProductB", "ProductA", "ProductC", "ProductB", "ProductA", "ProductC",
        "ProductA", "ProductB", "ProductC",
    ];
    let product_series = Series::new(products, Some("Product".to_string()))?;
    df.add_column("Product".to_string(), product_series)?;

    // Add price column
    let prices = vec![
        "100", "150", "110", "200", "160", "120", "210", "115", "165", "220",
    ];
    let price_series = Series::new(prices, Some("Price".to_string()))?;
    df.add_column("Price".to_string(), price_series)?;

    // Add quantity column
    let quantities = vec!["5", "3", "6", "2", "4", "7", "3", "8", "5", "4"];
    let quantity_series = Series::new(quantities, Some("Quantity".to_string()))?;
    df.add_column("Quantity".to_string(), quantity_series)?;

    Ok(df)
}

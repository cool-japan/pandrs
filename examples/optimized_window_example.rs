#[cfg(feature = "optimized")]
use chrono::NaiveDate;
#[cfg(feature = "optimized")]
use pandrs::error::{Error, Result};
#[cfg(feature = "optimized")]
use pandrs::{Column, Float64Column, OptimizedDataFrame, StringColumn};
#[cfg(feature = "optimized")]
use std::str::FromStr;

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_window_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Optimized Window Operations ===\n");

    // Create an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // Prepare date data
    let mut dates = Vec::new();
    let mut values = Vec::new();

    // Start date and end date
    let start_date =
        NaiveDate::from_str("2023-01-01").map_err(|e| Error::InvalidInput(e.to_string()))?;

    // Generate data for 20 days
    for i in 0..20 {
        let date = start_date
            .checked_add_days(chrono::Days::new(i as u64))
            .unwrap();
        dates.push(date.format("%Y-%m-%d").to_string());

        // Values are a simple linear trend + noise
        let value = 100.0 + i as f64 * 2.0 + (i as f64 * 0.5).sin() * 5.0;
        values.push(value);
    }

    // Add date column
    let date_col = StringColumn::new(dates);
    df.add_column("date", Column::String(date_col))?;

    // Add value column
    let value_col = Float64Column::new(values);
    df.add_column("value", Column::Float64(value_col))?;

    // Display data
    println!("=== Original Data ===");
    println!("{:?}", df);

    // Simulate window operations
    // Note: Actual window operations need to be implemented in OptimizedDataFrame
    println!("\n=== Simulation of Window Operations ===");
    println!("Window operations for OptimizedDataFrame are not yet implemented, but the following features are needed:");
    println!("1. Rolling Window - Aggregation over a fixed-size moving window");
    println!("2. Expanding Window - Aggregation over all historical data");
    println!("3. Exponentially Weighted Window (EWM) - Aggregation with exponentially decaying weights on past data");

    // Example implementation of window operations (pseudo-code)
    println!("\n=== Example Implementation of Window Operations (Pseudo-code) ===");
    println!("df.rolling_window(\"value\", 3, \"mean\") → 3-day moving average");
    println!("df.expanding_window(\"value\", \"sum\") → Cumulative sum");
    println!("df.ewm_window(\"value\", 0.5, \"mean\") → Exponentially weighted moving average (alpha=0.5)");

    println!("\n=== Optimized Window Operations Example Complete ===");
    Ok(())
}

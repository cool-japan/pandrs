use chrono::NaiveDate;
use pandrs::error::{Error, Result};
use pandrs::temporal::{date_range, Frequency, TimeSeries};
use pandrs::NA;
use std::str::FromStr;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Time Series Data Example ===");

    // Create date range
    let start_date = NaiveDate::from_str("2023-01-01").map_err(|e| Error::Format(e.to_string()))?;
    let end_date = NaiveDate::from_str("2023-01-31").map_err(|e| Error::Format(e.to_string()))?;

    // Generate daily date range
    let dates = date_range(start_date, end_date, Frequency::Daily, true)?;
    println!("Date range length: {}", dates.len());
    println!("First date: {}", dates[0]);
    println!("Last date: {}", dates[dates.len() - 1]);

    // Generate time series data
    let mut values = Vec::with_capacity(dates.len());
    for i in 0..dates.len() {
        // Generate sample data (simple sine wave)
        let day = i as f64;
        let value = 100.0 + 10.0 * (day * 0.2).sin();

        // Include missing values (every 5 days)
        if i % 5 == 0 {
            values.push(NA::NA);
        } else {
            values.push(NA::Value(value));
        }
    }

    // Create TimeSeries
    let time_series = TimeSeries::new(values, dates, Some("daily_values".to_string()))?;

    println!("\n=== Time Series Basic Information ===");
    println!("Length: {}", time_series.len());

    // Time filtering
    let start_filter =
        NaiveDate::from_str("2023-01-10").map_err(|e| Error::Format(e.to_string()))?;
    let end_filter = NaiveDate::from_str("2023-01-20").map_err(|e| Error::Format(e.to_string()))?;
    let filtered = time_series.filter_by_time(&start_filter, &end_filter)?;

    println!("\n=== Time Filtering Results ===");
    println!("Original time series length: {}", time_series.len());
    println!("Filtered length: {}", filtered.len());

    // Calculate moving average
    let window_size = 3;
    let moving_avg = time_series.rolling_mean(window_size)?;

    println!("\n=== Moving Average (Window Size: {}) ===", window_size);
    println!("Moving average length: {}", moving_avg.len());

    // Display first few values
    println!("\n=== Comparison of Original Data and Moving Average (First 10 Rows) ===");
    println!("Date\t\tOriginal\tMoving Average");
    for i in 0..10.min(time_series.len()) {
        let date = time_series.timestamps()[i];
        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        let ma = match moving_avg.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}\t\t{}", date, original, ma);
    }

    // Resample to weekly
    let weekly = time_series.resample(Frequency::Weekly).mean()?;

    println!("\n=== Weekly Resampling ===");
    println!("Original time series length: {}", time_series.len());
    println!("Resampled length: {}", weekly.len());

    // Display weekly data
    println!("\n=== Weekly Data ===");
    println!("Date\t\tValue");
    for i in 0..weekly.len() {
        let date = weekly.timestamps()[i];
        let value = match weekly.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}", date, value);
    }

    println!("\n=== Time Series Sample Completed ===");
    Ok(())
}

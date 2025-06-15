use chrono::NaiveDate;
use pandrs::error::{Error, Result};
use pandrs::temporal::{date_range, Frequency, TimeSeries};
use pandrs::NA;
use std::str::FromStr;

// Translated Japanese comments and strings into English
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Window Operations ===\n");

    // Create date range
    let start_date = NaiveDate::from_str("2023-01-01").map_err(|e| Error::Format(e.to_string()))?;
    let end_date = NaiveDate::from_str("2023-01-20").map_err(|e| Error::Format(e.to_string()))?;

    // Generate daily date range
    let dates = date_range(start_date, end_date, Frequency::Daily, true)?;

    // Generate time series data (simple linear + noise)
    let mut values = Vec::with_capacity(dates.len());
    for i in 0..dates.len() {
        let value = 100.0 + i as f64 * 2.0 + (i as f64 * 0.5).sin() * 5.0;

        // Set some values to missing (every 7 days)
        if i % 7 == 0 {
            values.push(NA::NA);
        } else {
            values.push(NA::Value(value));
        }
    }

    // Create TimeSeries
    let time_series = TimeSeries::new(values, dates, Some("sample_data".to_string()))?;

    // Display data
    println!("=== Original Data ===");
    println!("Date\t\tValue");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];
        let value = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };
        println!("{}\t{}", date, value);
    }

    // 1. Fixed-length window operations (Rolling)
    println!("\n=== Fixed-length Window Operations (Window Size: 3) ===");

    // 1.1 Moving average
    let window_size = 3;
    let rolling_mean = time_series.rolling(window_size)?.mean()?;

    // 1.2 Moving sum
    let rolling_sum = time_series.rolling(window_size)?.sum()?;

    // 1.3 Moving standard deviation
    let rolling_std = time_series.rolling(window_size)?.std(1)?;

    // 1.4 Moving minimum
    let rolling_min = time_series.rolling(window_size)?.min()?;

    // 1.5 Moving maximum
    let rolling_max = time_series.rolling(window_size)?.max()?;

    // Display results
    println!(
        "Date\t\tOriginal Data\tMoving Avg\tMoving Sum\tMoving Std Dev\tMoving Min\tMoving Max"
    );
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];

        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let mean = match rolling_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let sum = match rolling_sum.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let std = match rolling_std.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let min = match rolling_min.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let max = match rolling_max.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        println!(
            "{}\t{}\t{}\t{}\t{}\\t{}\t{}",
            date, original, mean, sum, std, min, max
        );
    }

    // 2. Expanding window operations
    println!("\n=== Expanding Window Operations (Minimum Periods: 3) ===");

    // 2.1 Expanding average
    let min_periods = 3; // At least 3 data points required
    let expanding_mean = time_series.expanding(min_periods)?.mean()?;

    // 2.2 Expanding sum
    let expanding_sum = time_series.expanding(min_periods)?.sum()?;

    // Display results
    println!("Date\t\tOriginal Data\tExpanding Avg\tExpanding Sum");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];

        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let exp_mean = match expanding_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let exp_sum = match expanding_sum.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        println!("{}\t{}\t{}\t{}", date, original, exp_mean, exp_sum);
    }

    // 3. Exponentially weighted moving operations (EWM)
    println!("\n=== Exponentially Weighted Moving Operations (span: 5) ===");

    // 3.1 EWM average (specify span)
    let span = 5; // Half-life
    let ewm_mean = time_series.ewm(Some(span), None, false)?.mean()?;

    // 3.2 EWM standard deviation
    let ewm_std = time_series.ewm(Some(span), None, false)?.std(1)?;

    // 3.3 EWM with a different alpha
    let alpha = 0.3; // Specify alpha value directly
    let ewm_mean_alpha = time_series.ewm(None, Some(alpha), false)?.mean()?;

    // Display results
    println!("Date\t\tOriginal Data\tEWM Avg (span=5)\tEWM Std Dev\tEWM Avg (alpha=0.3)");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];

        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let ewm = match ewm_mean.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let ewm_s = match ewm_std.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let ewm_a = match ewm_mean_alpha.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        println!("{}\t{}\\t{}\\t{}\\t{}", date, original, ewm, ewm_s, ewm_a);
    }

    // 4. Example of using custom aggregation functions
    println!("\n=== Example of Custom Aggregation Function (Median) ===");

    // Custom function to calculate median
    let median = |values: &[f64]| -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    // Apply custom aggregation function
    let rolling_median = time_series
        .rolling(window_size)?
        .aggregate(median, Some(1))?;

    // Display results
    println!("Date\t\tOriginal Data\tMoving Median");
    for i in 0..time_series.len() {
        let date = time_series.timestamps()[i];

        let original = match time_series.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        let med = match rolling_median.values()[i] {
            NA::Value(v) => format!("{:.2}", v),
            NA::NA => "NA".to_string(),
        };

        println!("{}\t{}\t{}", date, original, med);
    }

    println!("\n=== Example of Window Operations Complete ===");
    Ok(())
}

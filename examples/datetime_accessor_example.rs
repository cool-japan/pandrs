use chrono::NaiveDate;
use pandrs::error::Result;
use pandrs::series::datetime_accessor::datetime_constructors;
use pandrs::{DataFrame, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== DateTime Accessor Example ===");

    // Create datetime data
    let dt1 = NaiveDate::from_ymd_opt(2023, 12, 25)
        .unwrap()
        .and_hms_opt(14, 30, 45)
        .unwrap();
    let dt2 = NaiveDate::from_ymd_opt(2024, 6, 15)
        .unwrap()
        .and_hms_opt(9, 15, 30)
        .unwrap();
    let dt3 = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let dt4 = NaiveDate::from_ymd_opt(2023, 12, 23)
        .unwrap()
        .and_hms_opt(10, 20, 30)
        .unwrap(); // Saturday

    let datetime_data = vec![dt1, dt2, dt3, dt4];
    let datetime_series = Series::new(datetime_data, Some("timestamps".to_string()))?;

    println!("Original DateTime Series:");
    for (i, dt) in datetime_series.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    // Test datetime accessor
    let dt_accessor = datetime_series.dt()?;

    // Extract year
    let years = dt_accessor.year()?;
    println!("\nYears: {:?}", years.values());

    // Extract month
    let months = dt_accessor.month()?;
    println!("Months: {:?}", months.values());

    // Extract day
    let days = dt_accessor.day()?;
    println!("Days: {:?}", days.values());

    // Extract hour
    let hours = dt_accessor.hour()?;
    println!("Hours: {:?}", hours.values());

    // Extract minute
    let minutes = dt_accessor.minute()?;
    println!("Minutes: {:?}", minutes.values());

    // Extract second
    let seconds = dt_accessor.second()?;
    println!("Seconds: {:?}", seconds.values());

    // Weekday
    let weekdays = dt_accessor.weekday()?;
    println!("Weekdays (0=Mon, 6=Sun): {:?}", weekdays.values());

    // Day of year
    let dayofyear = dt_accessor.dayofyear()?;
    println!("Day of year: {:?}", dayofyear.values());

    // Quarter
    let quarters = dt_accessor.quarter()?;
    println!("Quarters: {:?}", quarters.values());

    // Weekend check
    let is_weekend = dt_accessor.is_weekend()?;
    println!("Is weekend: {:?}", is_weekend.values());

    // Format as string
    let formatted = dt_accessor.strftime("%Y-%m-%d %H:%M:%S")?;
    println!("Formatted: {:?}", formatted.values());

    // Get timestamp
    let timestamps = dt_accessor.timestamp()?;
    println!("Timestamps: {:?}", timestamps.values());

    // Date arithmetic
    let plus_days = dt_accessor.add_days(7)?;
    println!("\nAfter adding 7 days:");
    for (i, dt) in plus_days.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    let plus_hours = dt_accessor.add_hours(5)?;
    println!("\nAfter adding 5 hours:");
    for (i, dt) in plus_hours.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    // Normalize to start of day
    let normalized = dt_accessor.normalize()?;
    println!("\nNormalized to start of day:");
    for (i, dt) in normalized.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    // Round to hour
    let rounded = dt_accessor.round("H")?;
    println!("\nRounded to hour:");
    for (i, dt) in rounded.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    println!("\n=== Parse DateTime from Strings ===");

    // Parse datetime from strings
    let datetime_strings = vec![
        "2023-12-25 14:30:45".to_string(),
        "2024-06-15 09:15:30".to_string(),
        "2024-01-01 00:00:00".to_string(),
    ];

    let parsed_series = datetime_constructors::parse_datetime_series(
        datetime_strings,
        Some("%Y-%m-%d %H:%M:%S"),
        Some("parsed_datetimes".to_string()),
    )?;

    println!("Parsed from strings:");
    for (i, dt) in parsed_series.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    println!("\n=== Date Range ===");

    // Create date range
    let start_date = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let end_date = NaiveDate::from_ymd_opt(2024, 1, 7).unwrap();
    let date_range = datetime_constructors::date_range(start_date, end_date, "D")?;

    println!("Daily date range:");
    for (i, dt) in date_range.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt);
    }

    println!("\n=== DataFrame with DateTime Column ===");

    // Create DataFrame with datetime column
    let mut df = DataFrame::new();
    df.add_column("datetime".to_string(), datetime_series)?;

    println!("DataFrame created successfully with datetime column!");

    Ok(())
}

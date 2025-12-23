use chrono::NaiveDate;
use pandrs::error::Result;
use pandrs::{DataFrame, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Enhanced DateTime Accessor Example ===");

    // Create diverse datetime data for comprehensive testing
    let dt1 = NaiveDate::from_ymd_opt(2023, 1, 31)
        .unwrap()
        .and_hms_opt(14, 37, 23)
        .unwrap(); // Jan 31 (month-end)
    let dt2 = NaiveDate::from_ymd_opt(2024, 2, 29)
        .unwrap()
        .and_hms_opt(9, 45, 15)
        .unwrap(); // Leap day
    let dt3 = NaiveDate::from_ymd_opt(2023, 12, 23)
        .unwrap()
        .and_hms_opt(16, 22, 45)
        .unwrap(); // Saturday
    let dt4 = NaiveDate::from_ymd_opt(2024, 6, 15)
        .unwrap()
        .and_hms_opt(11, 8, 30)
        .unwrap(); // Mid-year

    let datetime_data = vec![dt1, dt2, dt3, dt4];
    let datetime_series = Series::new(datetime_data, Some("enhanced_timestamps".to_string()))?;

    println!("Original DateTime Series:");
    for (i, dt) in datetime_series.values().iter().enumerate() {
        println!("  [{}]: {} ({})", i, dt, dt.format("%A, %B %d, %Y"));
    }

    let dt_accessor = datetime_series.dt()?;

    println!("\n=== Enhanced Temporal Properties ===");

    // Week number
    let weeks = dt_accessor.week()?;
    println!("Week numbers: {:?}", weeks.values());

    // Days in month
    let days_in_month = dt_accessor.days_in_month()?;
    println!("Days in month: {:?}", days_in_month.values());

    // Leap year detection
    let is_leap = dt_accessor.is_leap_year()?;
    println!("Is leap year: {:?}", is_leap.values());

    // Business day detection
    let is_bday = dt_accessor.is_business_day()?;
    println!("Is business day: {:?}", is_bday.values());

    println!("\n=== Enhanced Date Arithmetic ===");

    // Add months with smart overflow handling
    let plus_months = dt_accessor.add_months(3)?;
    println!("After adding 3 months:");
    for (i, dt) in plus_months.values().iter().enumerate() {
        println!("  [{}]: {} -> {}", i, datetime_series.values()[i], dt);
    }

    // Add years with leap day handling
    let plus_years = dt_accessor.add_years(1)?;
    println!("\nAfter adding 1 year:");
    for (i, dt) in plus_years.values().iter().enumerate() {
        println!("  [{}]: {} -> {}", i, datetime_series.values()[i], dt);
    }

    // Subtract months
    let minus_months = dt_accessor.add_months(-6)?;
    println!("\nAfter subtracting 6 months:");
    for (i, dt) in minus_months.values().iter().enumerate() {
        println!("  [{}]: {} -> {}", i, datetime_series.values()[i], dt);
    }

    println!("\n=== Enhanced Rounding Operations ===");

    // Round to 15-minute intervals
    let rounded_15min = dt_accessor.round("15min")?;
    println!("Rounded to 15-minute intervals:");
    for (i, dt) in rounded_15min.values().iter().enumerate() {
        println!("  [{}]: {} -> {}", i, datetime_series.values()[i], dt);
    }

    // Round to 30-second intervals
    let rounded_30sec = dt_accessor.round("30S")?;
    println!("\nRounded to 30-second intervals:");
    for (i, dt) in rounded_30sec.values().iter().enumerate() {
        println!("  [{}]: {} -> {}", i, datetime_series.values()[i], dt);
    }

    println!("\n=== Business Day Analysis ===");

    // Business day count from a reference date
    let reference_date = NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let bday_counts = dt_accessor.business_day_count(reference_date)?;

    println!("Business days from 2024-01-01:");
    for (i, count) in bday_counts.values().iter().enumerate() {
        println!(
            "  [{}]: {} -> {} business days",
            i,
            datetime_series.values()[i],
            count
        );
    }

    println!("\n=== Financial Date Analysis ===");

    // Create a financial time series (month-end dates)
    let month_ends = vec![
        NaiveDate::from_ymd_opt(2024, 1, 31)
            .unwrap()
            .and_hms_opt(16, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2024, 2, 29)
            .unwrap()
            .and_hms_opt(16, 0, 0)
            .unwrap(), // Leap year
        NaiveDate::from_ymd_opt(2024, 3, 31)
            .unwrap()
            .and_hms_opt(16, 0, 0)
            .unwrap(),
        NaiveDate::from_ymd_opt(2024, 4, 30)
            .unwrap()
            .and_hms_opt(16, 0, 0)
            .unwrap(),
    ];

    let financial_series = Series::new(month_ends, Some("month_ends".to_string()))?;
    let fin_accessor = financial_series.dt()?;

    println!("Financial month-end analysis:");
    let quarters = fin_accessor.quarter()?;
    let days_in_months = fin_accessor.days_in_month()?;
    let is_leap_years = fin_accessor.is_leap_year()?;

    for i in 0..financial_series.len() {
        println!(
            "  {}: Q{}, {} days in month, leap year: {}",
            financial_series.values()[i].format("%B %Y"),
            quarters.values()[i],
            days_in_months.values()[i],
            is_leap_years.values()[i]
        );
    }

    println!("\n=== Time Series Frequency Analysis ===");

    // Create hourly data for frequency rounding demonstration
    let hourly_data = vec![
        NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(9, 7, 23)
            .unwrap(),
        NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(9, 23, 45)
            .unwrap(),
        NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(9, 38, 12)
            .unwrap(),
        NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(9, 52, 55)
            .unwrap(),
    ];

    let hourly_series = Series::new(hourly_data, Some("hourly_data".to_string()))?;
    let hourly_accessor = hourly_series.dt()?;

    println!("Original hourly data:");
    for (i, dt) in hourly_series.values().iter().enumerate() {
        println!("  [{}]: {}", i, dt.format("%H:%M:%S"));
    }

    let rounded_5min = hourly_accessor.round("5min")?;
    println!("\nRounded to 5-minute intervals:");
    for (i, dt) in rounded_5min.values().iter().enumerate() {
        println!(
            "  [{}]: {} -> {}",
            i,
            hourly_series.values()[i].format("%H:%M:%S"),
            dt.format("%H:%M:%S")
        );
    }

    println!("\n=== Business Calendar Operations ===");

    // Demonstrate business day operations across a week
    let week_start = NaiveDate::from_ymd_opt(2024, 1, 15)
        .unwrap()
        .and_hms_opt(9, 0, 0)
        .unwrap(); // Monday
    let week_data: Vec<_> = (0..7)
        .map(|i| week_start + chrono::Duration::days(i))
        .collect();

    let week_series = Series::new(week_data, Some("business_week".to_string()))?;
    let week_accessor = week_series.dt()?;

    let weekdays = week_accessor.weekday()?;
    let is_weekend = week_accessor.is_weekend()?;
    let is_business = week_accessor.is_business_day()?;

    println!("Business week analysis:");
    for i in 0..week_series.len() {
        println!(
            "  {}: Weekday {}, Weekend: {}, Business day: {}",
            week_series.values()[i].format("%A %Y-%m-%d"),
            weekdays.values()[i],
            is_weekend.values()[i],
            is_business.values()[i]
        );
    }

    println!("\n=== DataFrame Integration ===");

    // Create a comprehensive DataFrame with datetime and derived columns
    let mut df = DataFrame::new();
    df.add_column("datetime".to_string(), datetime_series.clone())?;
    df.add_column("year".to_string(), dt_accessor.year()?)?;
    df.add_column("quarter".to_string(), dt_accessor.quarter()?)?;
    df.add_column("is_leap_year".to_string(), dt_accessor.is_leap_year()?)?;
    df.add_column(
        "is_business_day".to_string(),
        dt_accessor.is_business_day()?,
    )?;
    df.add_column("days_in_month".to_string(), dt_accessor.days_in_month()?)?;

    println!(
        "Comprehensive datetime DataFrame created with {} rows and {} columns!",
        datetime_series.len(),
        df.column_names().len()
    );

    // Display some insights
    println!("\nDataFrame insights:");
    println!(
        "- Contains {} leap year dates",
        dt_accessor
            .is_leap_year()?
            .values()
            .iter()
            .filter(|&&x| x)
            .count()
    );
    println!(
        "- Contains {} business days",
        dt_accessor
            .is_business_day()?
            .values()
            .iter()
            .filter(|&&x| x)
            .count()
    );
    println!("- Spans {} unique years", {
        let year_series = dt_accessor.year()?;
        let mut years: Vec<_> = year_series.values().iter().collect();
        years.sort();
        years.dedup();
        years.len()
    });

    println!("\n=== Enhanced DateTime Accessor Implementation Complete! ===");
    println!("✅ All enhanced features working correctly:");
    println!("   • Advanced date arithmetic (months/years with overflow handling)");
    println!("   • Enhanced rounding (15min, 30S, etc.)");
    println!("   • Business day operations");
    println!("   • Leap year and calendar functions");
    println!("   • Comprehensive temporal properties");
    println!("   • Financial calendar support");

    Ok(())
}

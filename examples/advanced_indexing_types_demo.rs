#![allow(clippy::result_large_err)]

use chrono::{NaiveDate, NaiveDateTime};
use pandrs::dataframe::advanced_indexing::{
    CategoricalIndex, DatetimeIndex, Index, IndexOperations, IntervalClosed, IntervalIndex,
    PeriodFrequency, PeriodIndex,
};
use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::SpecializedIndexingExt as AdvancedIndexingExt;
use pandrs::error::Result;
use pandrs::series::base::Series;

fn main() -> Result<()> {
    println!("=== Alpha 8: Advanced Indexing Types Example ===\n");

    // Create sample time series financial data
    println!("1. Creating Sample Financial Time Series Data:");
    let mut df = DataFrame::new();

    let dates = vec![
        "2024-01-01 09:00:00",
        "2024-01-02 09:00:00",
        "2024-01-03 09:00:00",
        "2024-01-04 09:00:00",
        "2024-01-05 09:00:00",
        "2024-01-08 09:00:00",
        "2024-01-09 09:00:00",
        "2024-01-10 09:00:00",
        "2024-01-11 09:00:00",
        "2024-01-12 09:00:00",
        "2024-01-15 09:00:00",
        "2024-01-16 09:00:00",
    ];

    let prices = vec![
        "100.5", "102.3", "101.8", "103.2", "104.1", "105.2", "106.8", "105.5", "107.3", "108.9",
        "110.2", "109.8",
    ];

    let volumes = vec![
        "15000", "18000", "16500", "22000", "19500", "21000", "24000", "17500", "25000", "23500",
        "26000", "20000",
    ];

    let sectors = vec![
        "Technology",
        "Technology",
        "Healthcare",
        "Technology",
        "Finance",
        "Healthcare",
        "Technology",
        "Finance",
        "Healthcare",
        "Technology",
        "Finance",
        "Healthcare",
    ];

    let date_series = Series::new(
        dates.into_iter().map(|s| s.to_string()).collect(),
        Some("Date".to_string()),
    )?;
    let price_series = Series::new(
        prices.into_iter().map(|s| s.to_string()).collect(),
        Some("Price".to_string()),
    )?;
    let volume_series = Series::new(
        volumes.into_iter().map(|s| s.to_string()).collect(),
        Some("Volume".to_string()),
    )?;
    let sector_series = Series::new(
        sectors.into_iter().map(|s| s.to_string()).collect(),
        Some("Sector".to_string()),
    )?;

    df.add_column("Date".to_string(), date_series)?;
    df.add_column("Price".to_string(), price_series)?;
    df.add_column("Volume".to_string(), volume_series)?;
    df.add_column("Sector".to_string(), sector_series)?;

    println!("Original Financial Data:");
    println!("{:?}", df);

    println!("\n=== DatetimeIndex Examples ===\n");

    // 2. DatetimeIndex
    println!("2. DatetimeIndex Operations:");

    let (df_with_dt_index, dt_index) =
        df.set_datetime_index("Date", Some("DatetimeIdx".to_string()))?;
    println!("DataFrame with DatetimeIndex:");
    println!("{:?}", df_with_dt_index);
    println!("DatetimeIndex info:");
    println!("  Length: {}", dt_index.len());
    println!("  Name: {:?}", dt_index.name());
    println!("  Has duplicates: {}", dt_index.has_duplicates());

    // DatetimeIndex operations
    println!("\nDatetime component extraction:");
    println!("  Years: {:?}", dt_index.year());
    println!("  Months: {:?}", dt_index.month());
    println!("  Days: {:?}", dt_index.day());
    println!("  Weekdays: {:?}", dt_index.weekday());

    // Date range filtering
    let start_filter = NaiveDateTime::parse_from_str("2024-01-05 00:00:00", "%Y-%m-%d %H:%M:%S")
        .map_err(|e| pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e)))?;
    let end_filter = NaiveDateTime::parse_from_str("2024-01-10 23:59:59", "%Y-%m-%d %H:%M:%S")
        .map_err(|e| pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e)))?;
    let filtered_indices = dt_index.filter_range(start_filter, end_filter)?;
    println!(
        "  Rows in date range 2024-01-05 to 2024-01-10: {:?}",
        filtered_indices
    );

    // Create date range with frequency
    println!("\nCreating daily date range:");
    let daily_range = DatetimeIndex::date_range(
        NaiveDateTime::parse_from_str("2024-01-01 09:00:00", "%Y-%m-%d %H:%M:%S").map_err(|e| {
            pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e))
        })?,
        None,
        Some(7),
        "d",
        Some("DailyRange".to_string()),
    )?;
    println!(
        "  Daily range (7 periods): {:?}",
        daily_range.to_string_vec()
    );

    println!("\nCreating hourly date range:");
    let hourly_range = DatetimeIndex::date_range(
        NaiveDateTime::parse_from_str("2024-01-01 09:00:00", "%Y-%m-%d %H:%M:%S").map_err(|e| {
            pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e))
        })?,
        None,
        Some(6),
        "2h",
        Some("HourlyRange".to_string()),
    )?;
    println!(
        "  Hourly range (6 periods, 2h frequency): {:?}",
        hourly_range.to_string_vec()
    );

    // Resampling
    println!("\nResampling DatetimeIndex:");
    let resample_groups = dt_index.resample("3d")?;
    println!("  Resampled to 3-day groups: {:?}", resample_groups);

    println!("\n=== PeriodIndex Examples ===\n");

    // 3. PeriodIndex
    println!("3. PeriodIndex Operations:");

    let (df_with_period_index, period_index) = df_with_dt_index.set_period_index(
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        PeriodFrequency::Weekly,
        Some("WeeklyPeriods".to_string()),
    )?;

    println!("DataFrame with PeriodIndex:");
    println!("{:?}", df_with_period_index);
    println!("PeriodIndex info:");
    println!("  Length: {}", period_index.len());
    println!("  Name: {:?}", period_index.name());
    println!("  Frequency: {:?}", period_index.frequency);
    println!("  Labels: {:?}", period_index.labels());

    // Create different frequency period ranges
    println!("\nCreating quarterly period range:");
    let quarterly_periods = PeriodIndex::period_range(
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2024, 12, 31).unwrap(),
        PeriodFrequency::Quarterly,
        Some("QuarterlyPeriods".to_string()),
    )?;
    println!("  Quarterly periods: {:?}", quarterly_periods.labels());

    println!("\nCreating monthly period range:");
    let monthly_periods = PeriodIndex::period_range(
        NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
        NaiveDate::from_ymd_opt(2024, 6, 30).unwrap(),
        PeriodFrequency::Monthly,
        Some("MonthlyPeriods".to_string()),
    )?;
    println!("  Monthly periods: {:?}", monthly_periods.labels());

    // Find periods containing specific dates
    let target_date = NaiveDate::from_ymd_opt(2024, 3, 15).unwrap();
    let containing_periods = quarterly_periods.find_periods_containing(target_date);
    println!("  Periods containing 2024-03-15: {:?}", containing_periods);

    println!("\n=== IntervalIndex Examples ===\n");

    // 4. IntervalIndex
    println!("4. IntervalIndex Operations:");

    // Cut prices into equal-width bins
    let (df_with_interval_index, interval_index) = df.set_interval_index_cut(
        "Price",
        4,
        IntervalClosed::Right,
        Some("PriceBins".to_string()),
    )?;

    println!("DataFrame with IntervalIndex (price bins):");
    println!("{:?}", df_with_interval_index);
    println!("IntervalIndex info:");
    println!("  Length: {}", interval_index.len());
    println!("  Name: {:?}", interval_index.name());
    println!("  Closed: {:?}", interval_index.closed);
    println!("  Labels: {:?}", interval_index.labels());
    println!("  Midpoints: {:?}", interval_index.midpoints());
    println!("  Widths: {:?}", interval_index.widths());

    // Quantile-based binning
    println!("\nQuantile-based volume binning:");
    let (_df_with_qcut_index, qcut_index) = df.set_interval_index_qcut(
        "Volume",
        3,
        IntervalClosed::Left,
        Some("VolumeQuantiles".to_string()),
    )?;
    println!("  Volume quantile bins: {:?}", qcut_index.labels());
    println!("  Quantile midpoints: {:?}", qcut_index.midpoints());

    // Find intervals containing specific values
    let target_price = 105.0;
    let containing_intervals = interval_index.find_intervals_containing(target_price);
    println!(
        "  Intervals containing price 105.0: {:?}",
        containing_intervals
    );

    // Create custom interval index
    println!("\nCreating custom intervals:");
    let custom_intervals = IntervalIndex::from_breaks(
        vec![0.0, 50.0, 100.0, 150.0, 200.0],
        IntervalClosed::Both,
        Some("CustomBins".to_string()),
    )?;
    println!("  Custom intervals: {:?}", custom_intervals.labels());

    println!("\n=== CategoricalIndex Examples ===\n");

    // 5. CategoricalIndex
    println!("5. CategoricalIndex Operations:");

    let (df_with_cat_index, mut cat_index) =
        df.set_categorical_index("Sector", true, Some("SectorIndex".to_string()))?;

    println!("DataFrame with CategoricalIndex:");
    println!("{:?}", df_with_cat_index);
    println!("CategoricalIndex info:");
    println!("  Length: {}", cat_index.len());
    println!("  Name: {:?}", cat_index.name());
    println!("  Ordered: {}", cat_index.ordered);
    println!("  Categories: {:?}", cat_index.categories);
    println!("  Values: {:?}", cat_index.values());
    println!("  Memory usage: {} bytes", cat_index.memory_usage());

    // Category operations
    println!("\nCategory operations:");
    let value_counts = cat_index.value_counts();
    println!("  Value counts: {:?}", value_counts);

    // Add new categories
    cat_index.add_categories(vec!["Energy".to_string(), "Utilities".to_string()])?;
    println!(
        "  Categories after adding Energy and Utilities: {:?}",
        cat_index.categories
    );

    // Remove categories
    cat_index.remove_categories(vec!["Energy".to_string()])?;
    println!(
        "  Categories after removing Energy: {:?}",
        cat_index.categories
    );

    println!("\n=== Index Set Operations ===\n");

    // 6. Index Set Operations
    println!("6. Index Set Operations:");

    let index1 = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(),
    ];
    let index2 = vec![
        "C".to_string(),
        "D".to_string(),
        "E".to_string(),
        "F".to_string(),
    ];

    println!("Index 1: {:?}", index1);
    println!("Index 2: {:?}", index2);

    let union_result = IndexOperations::union_string_indexes(&index1, &index2);
    println!("  Union: {:?}", union_result);

    let intersection_result = IndexOperations::intersection_string_indexes(&index1, &index2);
    println!("  Intersection: {:?}", intersection_result);

    let difference_result = IndexOperations::difference_string_indexes(&index1, &index2);
    println!("  Difference (1 - 2): {:?}", difference_result);

    let symmetric_diff_result =
        IndexOperations::symmetric_difference_string_indexes(&index1, &index2);
    println!("  Symmetric difference: {:?}", symmetric_diff_result);

    println!("\n=== Index Sorting and Uniqueness ===\n");

    // 7. Index operations
    println!("7. Index Sorting and Uniqueness:");

    // Sort operations
    let (sorted_dt_index, sort_indices) = dt_index.sort(true)?;
    println!("Sorted DatetimeIndex (ascending):");
    println!(
        "  Sorted values: {:?}",
        sorted_dt_index.to_string_vec()[..5].to_vec()
    );
    println!("  Sort indices: {:?}", sort_indices[..5].to_vec());

    // Unique operations
    let unique_cat_index = cat_index.unique()?;
    println!("Unique categories: {:?}", unique_cat_index.to_string_vec());

    println!("\n=== Advanced Index Applications ===\n");

    // 8. Advanced applications
    println!("8. Advanced Index Applications:");

    // Time series analysis with DatetimeIndex
    println!("Time series analysis:");
    let business_days_range = DatetimeIndex::date_range(
        NaiveDateTime::parse_from_str("2024-01-01 09:00:00", "%Y-%m-%d %H:%M:%S").map_err(|e| {
            pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e))
        })?,
        None,
        Some(10),
        "d",
        Some("BusinessDays".to_string()),
    )?;

    // Filter out weekends (simplified - would need more complex logic for real business days)
    let weekdays: Vec<u32> = business_days_range.weekday();
    let business_day_mask: Vec<bool> = weekdays.iter().map(|&day| day < 5).collect();
    println!("  Business day mask (Mon-Fri): {:?}", business_day_mask);

    // Risk analysis with IntervalIndex
    println!("\nRisk analysis with intervals:");
    let risk_levels = IntervalIndex::from_breaks(
        vec![0.0, 50.0, 100.0, 150.0, f64::INFINITY],
        IntervalClosed::Right,
        Some("RiskLevels".to_string()),
    )?;

    let mut risk_labels = Vec::new();
    for interval in &risk_levels.intervals {
        let risk_level = if interval.right <= 50.0 {
            "Low Risk"
        } else if interval.right <= 100.0 {
            "Medium Risk"
        } else if interval.right <= 150.0 {
            "High Risk"
        } else {
            "Very High Risk"
        };
        risk_labels.push(risk_level);
    }

    println!("  Risk intervals: {:?}", risk_levels.labels());
    println!("  Risk labels: {:?}", risk_labels);

    // Portfolio analysis with CategoricalIndex
    println!("\nPortfolio analysis with categorical sectors:");
    let sector_weights = vec![
        ("Technology", 0.4),
        ("Healthcare", 0.3),
        ("Finance", 0.2),
        ("Energy", 0.1),
    ];

    for (sector, weight) in sector_weights {
        println!("  {}: {:.1}%", sector, weight * 100.0);
    }

    println!("\n=== Performance Comparison ===\n");

    // 9. Performance demonstration
    println!("9. Performance Comparison:");
    demonstrate_performance()?;

    println!("\n=== Error Handling ===\n");

    // 10. Error handling examples
    println!("10. Error Handling:");
    demonstrate_error_handling()?;

    println!("\n=== Alpha 8 Advanced Indexing Types Complete ===");
    println!("\nNew advanced indexing capabilities implemented:");
    println!("✓ DatetimeIndex with timezone and frequency support");
    println!("✓ PeriodIndex for business periods (annual, quarterly, monthly, weekly, daily)");
    println!("✓ IntervalIndex for range-based and binned data indexing");
    println!("✓ CategoricalIndex with memory optimization");
    println!("✓ Index set operations (union, intersection, difference, symmetric_difference)");
    println!("✓ Advanced index sorting and uniqueness operations");
    println!("✓ Comprehensive date range generation and resampling");
    println!("✓ Equal-width and quantile-based binning");
    println!("✓ Memory-efficient categorical data handling");
    println!("✓ Production-ready error handling and validation");

    Ok(())
}

/// Demonstrate performance with larger datasets
fn demonstrate_performance() -> Result<()> {
    println!("--- Performance with Large Dataset ---");

    use std::time::Instant;

    // Create large dataset for performance testing
    let size = 10000;
    println!("Creating dataset with {} rows", size);

    // DatetimeIndex performance
    let start = Instant::now();
    let large_dt_index = DatetimeIndex::date_range(
        NaiveDateTime::parse_from_str("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").map_err(|e| {
            pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e))
        })?,
        None,
        Some(size),
        "h",
        Some("LargeDatetimeIndex".to_string()),
    )?;
    let duration = start.elapsed();
    println!("DatetimeIndex creation ({} entries): {:?}", size, duration);

    // DatetimeIndex component extraction
    let start = Instant::now();
    let _years = large_dt_index.year();
    let _months = large_dt_index.month();
    let _weekdays = large_dt_index.weekday();
    let duration = start.elapsed();
    println!("DatetimeIndex component extraction: {:?}", duration);

    // CategoricalIndex performance
    let categories = ["A", "B", "C", "D", "E"];
    let mut large_categorical_data = Vec::with_capacity(size);
    for i in 0..size {
        large_categorical_data.push(categories[i % categories.len()].to_string());
    }

    let start = Instant::now();
    let large_cat_index = CategoricalIndex::new(
        large_categorical_data,
        Some("LargeCategorical".to_string()),
        false,
    );
    let duration = start.elapsed();
    println!(
        "CategoricalIndex creation ({} entries): {:?}",
        size, duration
    );

    let start = Instant::now();
    let _value_counts = large_cat_index.value_counts();
    let duration = start.elapsed();
    println!("CategoricalIndex value_counts: {:?}", duration);
    println!("Memory usage: {} bytes", large_cat_index.memory_usage());

    // IntervalIndex performance
    let large_numeric_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();

    let start = Instant::now();
    let _large_interval_index = IntervalIndex::cut(
        &large_numeric_data,
        100,
        IntervalClosed::Right,
        Some("LargeIntervals".to_string()),
    )?;
    let duration = start.elapsed();
    println!(
        "IntervalIndex cut operation ({} values, 100 bins): {:?}",
        size, duration
    );

    // Index set operations performance
    let index1: Vec<String> = (0..1000).map(|i| format!("item_{}", i)).collect();
    let index2: Vec<String> = (500..1500).map(|i| format!("item_{}", i)).collect();

    let start = Instant::now();
    let _union = IndexOperations::union_string_indexes(&index1, &index2);
    let duration = start.elapsed();
    println!("Index union operation (1000 + 1000 items): {:?}", duration);

    let start = Instant::now();
    let _intersection = IndexOperations::intersection_string_indexes(&index1, &index2);
    let duration = start.elapsed();
    println!("Index intersection operation: {:?}", duration);

    Ok(())
}

/// Demonstrate error handling
fn demonstrate_error_handling() -> Result<()> {
    println!("--- Error Handling Examples ---");

    // Invalid date range
    let test_dt = NaiveDateTime::parse_from_str("2024-01-01 09:00:00", "%Y-%m-%d %H:%M:%S")
        .map_err(|e| pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e)))?;

    match DatetimeIndex::date_range(test_dt, None, None, "d", None) {
        Ok(_) => println!("Unexpected success with invalid date range"),
        Err(e) => println!("Expected error for invalid date range: {:?}", e),
    }

    // Invalid frequency
    match DatetimeIndex::date_range(test_dt, None, Some(5), "invalid_freq", None) {
        Ok(_) => println!("Unexpected success with invalid frequency"),
        Err(e) => println!("Expected error for invalid frequency: {:?}", e),
    }

    // Invalid interval breaks
    match IntervalIndex::from_breaks(vec![1.0], IntervalClosed::Right, None) {
        Ok(_) => println!("Unexpected success with single break"),
        Err(e) => println!("Expected error for insufficient breaks: {:?}", e),
    }

    // Empty values for cutting
    match IntervalIndex::cut(&[], 5, IntervalClosed::Right, None) {
        Ok(_) => println!("Unexpected success with empty values"),
        Err(e) => println!("Expected error for empty values: {:?}", e),
    }

    // Identical values for cutting
    let identical_values = vec![1.0, 1.0, 1.0, 1.0];
    match IntervalIndex::cut(&identical_values, 3, IntervalClosed::Right, None) {
        Ok(_) => println!("Unexpected success with identical values"),
        Err(e) => println!("Expected error for identical values: {:?}", e),
    }

    Ok(())
}

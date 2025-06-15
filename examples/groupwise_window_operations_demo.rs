use chrono::{Duration, NaiveDateTime};
use pandrs::dataframe::{DataFrame, EnhancedDataFrameWindowExt, GroupWiseWindowExt};
use pandrs::error::Result;
use pandrs::series::window::WindowClosed;
use pandrs::series::Series;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Alpha.7: Group-wise Window Operations Example");
    println!("===================================================");

    // Create sample multi-asset financial data with groups
    let symbols = vec![
        "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "AAPL", "GOOGL", "GOOGL", "GOOGL", "GOOGL",
        "GOOGL", "GOOGL", "MSFT", "MSFT", "MSFT", "MSFT", "MSFT", "MSFT",
    ];

    let dates = vec![
        "2024-01-01 09:00:00",
        "2024-01-01 09:15:00",
        "2024-01-01 09:30:00",
        "2024-01-01 09:45:00",
        "2024-01-01 10:00:00",
        "2024-01-01 10:15:00",
        "2024-01-01 09:00:00",
        "2024-01-01 09:15:00",
        "2024-01-01 09:30:00",
        "2024-01-01 09:45:00",
        "2024-01-01 10:00:00",
        "2024-01-01 10:15:00",
        "2024-01-01 09:00:00",
        "2024-01-01 09:15:00",
        "2024-01-01 09:30:00",
        "2024-01-01 09:45:00",
        "2024-01-01 10:00:00",
        "2024-01-01 10:15:00",
    ];

    let prices = [
        150.0, 152.5, 151.8, 153.2, 154.1, 152.9, // AAPL
        2800.0, 2825.0, 2810.0, 2835.0, 2845.0, 2820.0, // GOOGL
        400.0, 402.5, 401.8, 403.2, 404.1, 402.9, // MSFT
    ];

    let volumes = [
        1000, 1500, 1200, 1800, 2200, 1600, // AAPL
        500, 750, 600, 900, 1100, 800, // GOOGL
        800, 1200, 960, 1440, 1760, 1280, // MSFT
    ];

    // Parse datetime strings
    let datetime_series: Vec<NaiveDateTime> = dates
        .iter()
        .map(|date_str| NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S"))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| pandrs::error::Error::InvalidValue(format!("Date parsing error: {}", e)))?;

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "symbol".to_string(),
        Series::new(
            symbols.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some("symbol".to_string()),
        )?,
    )?;
    df.add_column(
        "datetime".to_string(),
        Series::new(datetime_series, Some("datetime".to_string()))?,
    )?;
    df.add_column(
        "price".to_string(),
        Series::new(
            prices.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
            Some("price".to_string()),
        )?,
    )?;
    df.add_column(
        "volume".to_string(),
        Series::new(
            volumes.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
            Some("volume".to_string()),
        )?,
    )?;

    println!("Original Multi-Asset Data:");
    println!("{:?}", df);
    println!();

    // Example 1: Group-wise Rolling Operations
    println!("1. Group-wise Rolling Window Operations");
    println!("======================================");

    // Create group-wise rolling configuration
    let groupwise_rolling = df
        .rolling_by_group(vec!["symbol".to_string()], 3)
        .min_periods(2)
        .center(false)
        .closed(WindowClosed::Right)
        .columns(vec!["price".to_string(), "volume".to_string()]);

    let rolling_ops = df.apply_rolling_by_group(&groupwise_rolling)?;
    let rolling_mean = rolling_ops.mean()?;

    println!("Group-wise Rolling Mean (window=3, by symbol):");
    println!("{:?}", rolling_mean);
    println!();

    // Group-wise rolling standard deviation
    let rolling_std = rolling_ops.std(1)?;
    println!("Group-wise Rolling Standard Deviation:");
    println!("{:?}", rolling_std);
    println!();

    // Example 2: Group-wise Expanding Operations
    println!("2. Group-wise Expanding Window Operations");
    println!("=========================================");

    let groupwise_expanding = df
        .expanding_by_group(vec!["symbol".to_string()], 2)
        .columns(vec!["price".to_string(), "volume".to_string()]);

    let expanding_ops = df.apply_expanding_by_group(&groupwise_expanding)?;
    let expanding_mean = expanding_ops.mean()?;
    let expanding_max = expanding_ops.max()?;

    println!("Group-wise Expanding Mean (min_periods=2, by symbol):");
    println!("{:?}", expanding_mean);
    println!();

    println!("Group-wise Expanding Maximum:");
    println!("{:?}", expanding_max);
    println!();

    // Example 3: Group-wise EWM Operations
    println!("3. Group-wise Exponentially Weighted Moving Operations");
    println!("======================================================");

    let groupwise_ewm = df
        .ewm_by_group(vec!["symbol".to_string()])
        .span(4)
        .adjust(true)
        .columns(vec!["price".to_string()]);

    let ewm_ops = df.apply_ewm_by_group(&groupwise_ewm)?;
    let ewm_mean = ewm_ops.mean()?;

    println!("Group-wise EWM Mean (span=4, by symbol):");
    println!("{:?}", ewm_mean);
    println!();

    // Example 4: Group-wise Time-based Rolling
    println!("4. Group-wise Time-based Rolling Operations");
    println!("===========================================");

    let groupwise_time_rolling = df
        .rolling_time_by_group(
            vec!["symbol".to_string()],
            Duration::minutes(30),
            "datetime".to_string(),
        )
        .columns(vec!["price".to_string(), "volume".to_string()]);

    let time_rolling_ops = df.apply_rolling_time_by_group(&groupwise_time_rolling)?;
    let time_mean = time_rolling_ops.mean()?;
    let time_count = time_rolling_ops.count()?;

    println!("Group-wise Time-based Rolling Mean (30-min window, by symbol):");
    println!("{:?}", time_mean);
    println!();

    println!("Group-wise Time-based Rolling Count:");
    println!("{:?}", time_count);
    println!();

    // Example 5: Advanced Multi-column Operations
    println!("5. Advanced Multi-column Window Operations");
    println!("==========================================");

    // Apply operations to all numeric columns automatically
    let multi_rolling = df
        .rolling_by_group(vec!["symbol".to_string()], 3)
        .min_periods(2);

    let multi_ops = df.apply_rolling_by_group(&multi_rolling)?;
    let multi_mean = multi_ops.mean()?;
    let multi_quantile = multi_ops.quantile(0.75)?;

    println!("Multi-column Group-wise Rolling Mean (all numeric columns):");
    println!("{:?}", multi_mean);
    println!();

    println!("Multi-column Group-wise Rolling 75th Percentile:");
    println!("{:?}", multi_quantile);
    println!();

    // Example 6: Custom Aggregation Functions within Groups
    println!("6. Custom Aggregation Functions within Groups");
    println!("==============================================");

    let custom_rolling = df
        .rolling_by_group(vec!["symbol".to_string()], 3)
        .columns(vec!["price".to_string()]);

    let custom_ops = df.apply_rolling_by_group(&custom_rolling)?;

    // Custom function: Price range (max - min) within window
    let price_range = custom_ops.apply(|window| {
        if window.is_empty() {
            0.0
        } else {
            let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            max_val - min_val
        }
    })?;

    println!("Group-wise Price Range (max-min in 3-period window):");
    println!("{:?}", price_range);
    println!();

    // Example 7: Regular Window Operations (non-grouped) for comparison
    println!("7. Regular Window Operations (Non-grouped) for Comparison");
    println!("=========================================================");

    // Regular rolling operations without grouping
    let regular_rolling = df.rolling(3).min_periods(2);
    let regular_ops = df.apply_rolling(&regular_rolling);
    let regular_mean = regular_ops.mean()?;

    println!("Regular Rolling Mean (no grouping - across all symbols):");
    println!("{:?}", regular_mean);
    println!();

    // Example 8: Combined Window and Time-based Operations
    println!("8. Combined Window and Time-based Operations");
    println!("============================================");

    // Combine different window types for comprehensive analysis
    let combined_analysis = df
        .rolling_by_group(vec!["symbol".to_string()], 2)
        .columns(vec!["price".to_string()]);

    let combined_ops = df.apply_rolling_by_group(&combined_analysis)?;
    let rolling_min = combined_ops.min()?;
    let rolling_max = combined_ops.max()?;

    println!("Combined Analysis - Rolling Min per Symbol:");
    println!("{:?}", rolling_min);
    println!();

    println!("Combined Analysis - Rolling Max per Symbol:");
    println!("{:?}", rolling_max);
    println!();

    println!("Group-wise Window Operations Example Completed Successfully!");
    println!("============================================================");

    Ok(())
}

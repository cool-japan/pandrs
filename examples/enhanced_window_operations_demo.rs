use chrono::{Duration, NaiveDateTime};
use pandrs::dataframe::{DataFrame, EnhancedDataFrameWindowExt};
use pandrs::error::Result;
use pandrs::series::window::WindowClosed;
use pandrs::series::Series;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Alpha.7: Enhanced Window Operations Example");
    println!("==================================================");

    // Create sample financial time series data
    let dates = [
        "2024-01-01 09:00:00",
        "2024-01-01 09:15:00",
        "2024-01-01 09:30:00",
        "2024-01-01 09:45:00",
        "2024-01-01 10:00:00",
        "2024-01-01 10:15:00",
        "2024-01-01 10:30:00",
        "2024-01-01 10:45:00",
        "2024-01-01 11:00:00",
        "2024-01-01 11:15:00",
        "2024-01-01 11:30:00",
        "2024-01-01 11:45:00",
    ];

    let prices = [
        100.0, 102.5, 101.8, 103.2, 104.1, 102.9, 105.6, 107.2, 106.8, 108.1, 109.3, 107.9,
    ];
    let volumes = [
        1000, 1500, 1200, 1800, 2200, 1600, 2400, 2100, 1900, 2300, 2000, 1700,
    ];
    let returns = [
        0.0, 0.025, -0.007, 0.014, 0.009, -0.012, 0.026, 0.015, -0.004, 0.012, 0.011, -0.013,
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
    df.add_column(
        "returns".to_string(),
        Series::new(
            returns.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
            Some("returns".to_string()),
        )?,
    )?;

    println!("Original Financial Data:");
    println!("{:?}", df);
    println!();

    // Example 1: Enhanced Rolling Window Operations
    println!("1. Enhanced Rolling Window Operations");
    println!("=====================================");

    // Create rolling window configuration with advanced parameters
    let rolling_config = df
        .rolling(3)
        .min_periods(2)
        .center(false)
        .closed(WindowClosed::Right)
        .columns(vec!["price".to_string(), "volume".to_string()]);

    // Apply multiple rolling operations
    let rolling_ops = df.apply_rolling(&rolling_config);
    let rolling_mean = rolling_ops.mean()?;

    println!("Rolling Mean (window=3, min_periods=2):");
    println!("{:?}", rolling_mean);
    println!();

    // Rolling standard deviation with custom ddof
    let rolling_std = rolling_ops.std(1)?;
    println!("Rolling Standard Deviation (ddof=1):");
    println!("{:?}", rolling_std);
    println!();

    // Rolling quantiles
    let rolling_quantile = rolling_ops.quantile(0.75)?;
    println!("Rolling 75th Percentile:");
    println!("{:?}", rolling_quantile);
    println!();

    // Example 2: Enhanced Expanding Window Operations
    println!("2. Enhanced Expanding Window Operations");
    println!("======================================");

    let expanding_config = df
        .expanding(2)
        .columns(vec!["price".to_string(), "returns".to_string()]);

    let expanding_ops = df.apply_expanding(&expanding_config);
    let expanding_mean = expanding_ops.mean()?;
    let expanding_std = expanding_ops.std(1)?;

    println!("Expanding Mean (min_periods=2):");
    println!("{:?}", expanding_mean);
    println!();

    println!("Expanding Standard Deviation:");
    println!("{:?}", expanding_std);
    println!();

    // Example 3: Enhanced EWM Operations
    println!("3. Enhanced Exponentially Weighted Moving Operations");
    println!("====================================================");

    // EWM with span
    let ewm_config_span = df
        .ewm()
        .span(4)
        .adjust(true)
        .columns(vec!["price".to_string(), "returns".to_string()]);

    let ewm_ops = df.apply_ewm(&ewm_config_span)?;
    let ewm_mean = ewm_ops.mean()?;

    println!("EWM Mean (span=4, adjust=true):");
    println!("{:?}", ewm_mean);
    println!();

    // EWM with alpha
    let ewm_config_alpha = df
        .ewm()
        .alpha(0.3)
        .unwrap()
        .adjust(false)
        .columns(vec!["price".to_string()]);

    let ewm_ops_alpha = df.apply_ewm(&ewm_config_alpha)?;
    let ewm_var = ewm_ops_alpha.var(1)?;

    println!("EWM Variance (alpha=0.3, adjust=false):");
    println!("{:?}", ewm_var);
    println!();

    // Example 4: Time-based Rolling Windows
    println!("4. Time-based Rolling Windows");
    println!("=============================");

    // 30-minute rolling window based on datetime
    let time_rolling = df.rolling_time(Duration::minutes(30), "datetime")?;
    let time_mean = time_rolling.mean()?;
    let time_count = time_rolling.count()?;

    println!("Time-based Rolling Mean (30-minute window):");
    println!("{:?}", time_mean);
    println!();

    println!("Time-based Rolling Count (30-minute window):");
    println!("{:?}", time_count);
    println!();

    // Example 5: Multi-column Operations
    println!("5. Multi-column Rolling Operations");
    println!("==================================");

    // Apply rolling operations to all numeric columns
    let multi_rolling = df.rolling(4).min_periods(3);
    let multi_ops = df.apply_rolling(&multi_rolling);

    let multi_mean = multi_ops.mean()?;
    let multi_max = multi_ops.max()?;
    let multi_min = multi_ops.min()?;

    println!("Multi-column Rolling Mean (all numeric columns):");
    println!("{:?}", multi_mean);
    println!();

    println!("Multi-column Rolling Max:");
    println!("{:?}", multi_max);
    println!();

    println!("Multi-column Rolling Min:");
    println!("{:?}", multi_min);
    println!();

    // Example 6: Custom Aggregation Functions
    println!("6. Custom Aggregation Functions");
    println!("===============================");

    // Custom function: calculate price momentum (difference between max and min in window)
    let custom_rolling = df.rolling(3).columns(vec!["price".to_string()]);
    let custom_ops = df.apply_rolling(&custom_rolling);

    let price_momentum = custom_ops.apply(|window| {
        if window.is_empty() {
            0.0
        } else {
            let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            max_val - min_val
        }
    })?;

    println!("Price Momentum (max-min in 3-period window):");
    println!("{:?}", price_momentum);
    println!();

    // Example 7: Advanced Window Parameters
    println!("7. Advanced Window Parameters");
    println!("=============================");

    // Centered rolling window
    let centered_rolling = df
        .rolling(5)
        .center(true)
        .min_periods(3)
        .closed(WindowClosed::Both)
        .columns(vec!["price".to_string()]);

    let centered_ops = df.apply_rolling(&centered_rolling);
    let centered_mean = centered_ops.mean()?;

    println!("Centered Rolling Mean (window=5, center=true):");
    println!("{:?}", centered_mean);
    println!();

    println!("Enhanced Window Operations Example Completed Successfully!");
    println!("=========================================================");

    Ok(())
}

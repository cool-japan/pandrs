use pandrs::dataframe::DataFrameWindowExt;
use pandrs::error::Result;
use pandrs::series::{WindowExt, WindowOps};
use pandrs::{DataFrame, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Comprehensive Window Operations Example ===\n");

    // Create sample time series data
    println!("1. Creating Sample Data:");
    let values = vec![
        100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 109.0,
    ];
    let dates = vec![
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    ];

    // Create DataFrame
    let mut df = DataFrame::new();
    let date_series = Series::new(
        dates.into_iter().map(|s| s.to_string()).collect(),
        Some("Date".to_string()),
    )?;
    let value_series = Series::new(values.clone(), Some("Price".to_string()))?;
    let price_series = value_series.to_string_series()?;

    df.add_column("Date".to_string(), date_series)?;
    df.add_column("Price".to_string(), price_series)?;

    println!("Original Data:");
    println!("{:?}", df);

    println!("\n=== Series-Level Window Operations ===\n");

    // 2. Series Rolling Window Operations
    println!("2. Rolling Window Operations (window=3):");
    let value_series = Series::new(values.clone(), Some("Price".to_string()))?;

    // Rolling mean
    let rolling_mean = value_series.rolling(3)?.mean()?;
    println!("Rolling Mean: {:?}", rolling_mean.values());

    // Rolling sum
    let rolling_sum = value_series.rolling(3)?.sum()?;
    println!("Rolling Sum: {:?}", rolling_sum.values());

    // Rolling standard deviation
    let rolling_std = value_series.rolling(3)?.std(1)?;
    println!("Rolling Std: {:?}", rolling_std.values());

    // Rolling min/max
    let rolling_min = value_series.rolling(3)?.min()?;
    let rolling_max = value_series.rolling(3)?.max()?;
    println!("Rolling Min: {:?}", rolling_min.values());
    println!("Rolling Max: {:?}", rolling_max.values());

    // Rolling median
    let rolling_median = value_series.rolling(3)?.median()?;
    println!("Rolling Median: {:?}", rolling_median.values());

    // Rolling quantile (75th percentile)
    let rolling_q75 = value_series.rolling(3)?.quantile(0.75)?;
    println!("Rolling 75th Percentile: {:?}", rolling_q75.values());

    // 3. Expanding Window Operations
    println!("\n3. Expanding Window Operations (min_periods=2):");

    let expanding_mean = value_series.expanding(2)?.mean()?;
    println!("Expanding Mean: {:?}", expanding_mean.values());

    let expanding_std = value_series.expanding(2)?.std(1)?;
    println!("Expanding Std: {:?}", expanding_std.values());

    let expanding_min = value_series.expanding(2)?.min()?;
    let expanding_max = value_series.expanding(2)?.max()?;
    println!("Expanding Min: {:?}", expanding_min.values());
    println!("Expanding Max: {:?}", expanding_max.values());

    // 4. Exponentially Weighted Moving Operations
    println!("\n4. Exponentially Weighted Moving Operations:");

    // EWM with span
    let ewm_mean_span = value_series.ewm().span(3).mean()?;
    println!("EWM Mean (span=3): {:?}", ewm_mean_span.values());

    // EWM with alpha
    let ewm_mean_alpha = value_series.ewm().alpha(0.5)?.mean()?;
    println!("EWM Mean (alpha=0.5): {:?}", ewm_mean_alpha.values());

    // EWM standard deviation
    let ewm_std = value_series.ewm().span(5).std(1)?;
    println!("EWM Std (span=5): {:?}", ewm_std.values());

    // 5. Custom Aggregation Functions
    println!("\n5. Custom Aggregation Functions:");

    // Custom function: Range (max - min)
    let rolling_range = value_series.rolling(3)?.apply(|window| {
        let min_val = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        max_val - min_val
    })?;
    println!("Rolling Range: {:?}", rolling_range.values());

    // Custom function: Geometric mean
    let rolling_geomean = value_series.rolling(3)?.apply(|window| {
        let product: f64 = window.iter().product();
        product.powf(1.0 / window.len() as f64)
    })?;
    println!("Rolling Geometric Mean: {:?}", rolling_geomean.values());

    println!("\n=== DataFrame-Level Window Operations ===\n");

    // 6. DataFrame Window Operations
    println!("6. DataFrame Rolling Operations:");

    // DataFrame rolling mean
    let df_rolling_mean = df.rolling(3, "Price", "mean", Some("Price_Rolling_Mean"))?;
    println!("DataFrame with Rolling Mean:");
    println!("{:?}", df_rolling_mean);

    // DataFrame rolling std
    let df_rolling_std = df.rolling(3, "Price", "std", Some("Price_Rolling_Std"))?;
    println!("\nDataFrame with Rolling Std:");
    println!("{:?}", df_rolling_std);

    // 7. DataFrame Expanding Operations
    println!("\n7. DataFrame Expanding Operations:");

    let df_expanding = df.expanding(2, "Price", "mean", Some("Price_Expanding_Mean"))?;
    println!("DataFrame with Expanding Mean:");
    println!("{:?}", df_expanding);

    // 8. DataFrame EWM Operations
    println!("\n8. DataFrame EWM Operations:");

    let df_ewm = df.ewm("Price", "mean", Some(3), None, Some("Price_EWM_Mean"))?;
    println!("DataFrame with EWM Mean:");
    println!("{:?}", df_ewm);

    // 9. Advanced Window Configurations
    println!("\n=== Advanced Window Configurations ===\n");

    println!("9. Advanced Rolling Configurations:");

    // Rolling with minimum periods
    let rolling_min_periods = value_series.rolling(5)?.min_periods(3).mean()?;
    println!(
        "Rolling Mean (window=5, min_periods=3): {:?}",
        rolling_min_periods.values()
    );

    // Centered rolling window
    let rolling_centered = value_series.rolling(5)?.center(true).mean()?;
    println!(
        "Centered Rolling Mean (window=5): {:?}",
        rolling_centered.values()
    );

    // 10. Performance Comparison
    println!("\n10. Performance Demonstration:");
    demonstrate_performance_scenarios()?;

    // 11. Edge Cases and Error Handling
    println!("\n11. Edge Cases and Error Handling:");
    demonstrate_edge_cases()?;

    println!("\n=== Comprehensive Window Operations Complete ===");
    println!("\nNew capabilities implemented:");
    println!("✓ Rolling windows with configurable parameters");
    println!("✓ Expanding windows with minimum periods");
    println!("✓ Exponentially weighted moving operations");
    println!("✓ Custom aggregation functions");
    println!("✓ DataFrame-level window operations");
    println!("✓ Advanced window configurations");
    println!("✓ Comprehensive statistical functions");
    println!("✓ Performance optimizations");
    println!("✓ Robust error handling");

    Ok(())
}

/// Demonstrate performance scenarios
#[allow(clippy::result_large_err)]
fn demonstrate_performance_scenarios() -> Result<()> {
    println!("--- Performance Scenarios ---");

    // Large dataset simulation
    let large_data: Vec<f64> = (0..1000)
        .map(|i| 100.0 + (i as f64 * 0.1) + ((i as f64 * 0.01).sin() * 10.0))
        .collect();

    let large_series = Series::new(large_data, Some("LargeData".to_string()))?;

    println!("Processing 1000 data points:");

    // Efficient rolling operations
    let start = std::time::Instant::now();
    let _rolling_mean = large_series.rolling(50)?.mean()?;
    println!("  Rolling mean (window=50): {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let _expanding_mean = large_series.expanding(10)?.mean()?;
    println!("  Expanding mean (min_periods=10): {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let _ewm_mean = large_series.ewm().span(20).mean()?;
    println!("  EWM mean (span=20): {:?}", start.elapsed());

    Ok(())
}

/// Demonstrate edge cases and error handling
#[allow(clippy::result_large_err)]
fn demonstrate_edge_cases() -> Result<()> {
    println!("--- Edge Cases ---");

    // Empty series
    let empty_series: Series<f64> = Series::new(vec![], Some("Empty".to_string()))?;
    match empty_series.rolling(3) {
        Ok(rolling) => match rolling.mean() {
            Ok(result) => println!("Empty series rolling mean: {} values", result.len()),
            Err(e) => println!("Empty series rolling mean error: {:?}", e),
        },
        Err(e) => println!("Empty series rolling error: {:?}", e),
    }

    // Single value
    let single_series = Series::new(vec![42.0], Some("Single".to_string()))?;
    match single_series.rolling(3) {
        Ok(rolling) => {
            let result = rolling.mean()?;
            println!("Single value rolling mean: {:?}", result.values());
        }
        Err(e) => println!("Single value rolling error: {:?}", e),
    }

    // Invalid parameters
    let test_series = Series::new(vec![1.0, 2.0, 3.0], Some("Test".to_string()))?;

    // Zero window size
    match test_series.rolling(0) {
        Ok(_) => println!("Zero window size: Unexpectedly succeeded"),
        Err(e) => println!("Zero window size error (expected): {:?}", e),
    }

    // Invalid quantile
    match test_series.rolling(2) {
        Ok(rolling) => match rolling.quantile(1.5) {
            Ok(_) => println!("Invalid quantile: Unexpectedly succeeded"),
            Err(e) => println!("Invalid quantile error (expected): {:?}", e),
        },
        Err(e) => println!("Rolling creation error: {:?}", e),
    }

    // Invalid EWM alpha
    match test_series.ewm().alpha(1.5) {
        Ok(_) => println!("Invalid alpha: Unexpectedly succeeded"),
        Err(e) => println!("Invalid alpha error (expected): {:?}", e),
    }

    Ok(())
}

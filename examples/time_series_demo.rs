//! Time Series Analysis Demo
//!
//! This example demonstrates the comprehensive time series analysis capabilities
//! including data creation, decomposition, and feature extraction.

use chrono::{TimeZone, Utc};
use pandrs::time_series::core::{Frequency, TimeSeriesBuilder};
use pandrs::time_series::decomposition::{DecompositionMethod, SeasonalDecomposition};
use pandrs::time_series::features::TimeSeriesFeatureExtractor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 PandRS Time Series Analysis Demo");
    println!("=====================================");

    // 1. Create a synthetic time series with trend and seasonality
    println!("\n📊 1. Creating synthetic time series data...");
    let mut builder = TimeSeriesBuilder::new();

    for i in 0..100 {
        let timestamp = Utc
            .timestamp_opt(1640995200 + (i * 86400) as i64, 0)
            .unwrap(); // Daily data
        let trend = i as f64 * 0.1; // Linear trend
        let seasonal = (2.0 * std::f64::consts::PI * i as f64 / 7.0).sin() * 2.0; // Weekly seasonality
        let noise = (i as f64 % 3.0 - 1.0) * 0.1; // Small noise
        let value = 10.0 + trend + seasonal + noise;

        builder = builder.add_point(timestamp, value);
    }

    let ts = builder
        .name("Example Time Series".to_string())
        .frequency(Frequency::Daily)
        .build()?;

    println!("✅ Created time series with {} data points", ts.len());
    println!("   Start: {:?}", ts.index.start());
    println!("   End: {:?}", ts.index.end());
    println!("   Frequency: {:?}", ts.index.frequency);

    // 2. Perform seasonal decomposition
    println!("\n📈 2. Performing seasonal decomposition...");
    let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7); // Weekly seasonality

    let decomp_result = decomposer.decompose(&ts)?;

    println!("✅ Decomposition completed:");
    println!("   Method: {:?}", decomp_result.method);
    println!("   Period: {}", decomp_result.period);
    println!(
        "   Trend variance ratio: {:.3}",
        decomp_result.metrics.trend_variance_ratio
    );
    println!(
        "   Seasonal variance ratio: {:.3}",
        decomp_result.metrics.seasonal_variance_ratio
    );
    println!("   Quality score: {:.3}", decomp_result.quality_score());

    // 3. Extract time series features
    println!("\n🔍 3. Extracting time series features...");
    let feature_extractor = TimeSeriesFeatureExtractor::new()
        .with_window_sizes(vec![3, 7, 14])
        .with_ema_alphas(vec![0.1, 0.3, 0.5])
        .with_frequency_features(true)
        .with_complexity_features(true);

    let features = feature_extractor.extract_features(&ts)?;

    println!("✅ Feature extraction completed:");
    println!(
        "   Statistical features: mean={:.3}, std={:.3}",
        features.statistical.mean, features.statistical.std
    );
    println!(
        "   Window features: {} moving averages",
        features.window.moving_averages.len()
    );
    println!(
        "   EMA features: {} alpha values",
        features.window.ema_features.len()
    );
    println!(
        "   Frequency features: dominant freq={:.3}",
        features.frequency.dominant_frequency
    );

    // 4. Demonstrate basic time series operations
    println!("\n⚙️  4. Time series operations...");

    // Rolling mean
    let _rolling_mean = ts.rolling_mean(7)?;
    println!("✅ 7-day rolling mean calculated");

    // Differencing
    let _diff = ts.diff(1)?;
    println!("✅ First-order differencing applied");

    // Percentage change
    let _pct_change = ts.pct_change(1)?;
    println!("✅ Percentage change calculated");

    // 5. Summary statistics
    println!("\n📋 5. Summary statistics:");
    if let (Some(first_val), Some(last_val)) = (ts.get(0), ts.get(ts.len() - 1)) {
        println!("   First value: {:.3} at {:?}", first_val.1, first_val.0);
        println!("   Last value: {:.3} at {:?}", last_val.1, last_val.0);
    }

    println!("   Data points: {}", ts.len());
    println!(
        "   Mean statistical feature: {:.3}",
        features.statistical.mean
    );
    println!("   Std deviation: {:.3}", features.statistical.std);
    println!("   Min value: {:.3}", features.statistical.min);
    println!("   Max value: {:.3}", features.statistical.max);

    println!("\n🎉 Time Series Analysis Demo completed successfully!");
    println!("   All time series modules working correctly.");

    Ok(())
}

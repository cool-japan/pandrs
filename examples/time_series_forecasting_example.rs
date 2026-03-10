#![allow(clippy::result_large_err)]
//! Comprehensive Advanced Time Series Forecasting Example
//!
//! This example demonstrates the full capabilities of PandRS for time series analysis
//! and forecasting, including:
//!
//! 1. Time series preprocessing (normalization, differencing, outlier treatment)
//! 2. Feature engineering (lag features, rolling statistics)
//! 3. Seasonal decomposition (additive and multiplicative)
//! 4. Statistical tests (stationarity, seasonality, autocorrelation)
//! 5. Multiple forecasting models (ARIMA, SARIMA, Exponential Smoothing, etc.)
//! 6. Model evaluation (RMSE, MAE, MAPE with cross-validation)
//! 7. Confidence intervals and uncertainty quantification
//! 8. Real-world use cases (sales, stocks, energy, weather)
//!
//! Run this example with:
//! ```bash
//! cargo run --example time_series_forecasting_example
//! ```

use chrono::{DateTime, Duration, TimeZone, Utc};
use pandrs::core::error::Result;
use pandrs::time_series::advanced_forecasting::{
    AutoArima, ModelSelectionCriterion, SarimaForecaster,
};
use pandrs::time_series::core::{Frequency, TimeSeries, TimeSeriesBuilder, TimeSeriesData};
use pandrs::time_series::decomposition::{DecompositionMethod, SeasonalDecomposition};
use pandrs::time_series::features::TimeSeriesFeatureExtractor;
use pandrs::time_series::forecasting::{
    ArimaForecaster, ExponentialSmoothingForecaster, ForecastMetrics, Forecaster,
    LinearTrendForecaster, SimpleMovingAverageForecaster,
};
use pandrs::time_series::preprocessing::{
    Differencing, MissingValueStrategy, Normalization, OutlierDetection, TimeSeriesPreprocessor,
};
use pandrs::time_series::stats::TimeSeriesStats;
use std::collections::HashMap;

/// Generate synthetic sales data with weekly seasonality and trend
fn generate_sales_data(n_points: usize) -> Result<TimeSeries> {
    println!("\n=== Generating Synthetic Sales Data ===");
    println!(
        "Creating {} data points with weekly seasonality...",
        n_points
    );

    let mut builder = TimeSeriesBuilder::new();
    let start_date = Utc
        .timestamp_opt(1640995200, 0)
        .single()
        .ok_or_else(|| pandrs::core::error::Error::InvalidInput("Invalid timestamp".to_string()))?;

    for i in 0..n_points {
        let timestamp = start_date + Duration::days(i as i64);

        // Base level
        let base = 1000.0;

        // Upward trend
        let trend = i as f64 * 2.5;

        // Weekly seasonality (higher sales on weekends)
        let day_of_week = i % 7;
        let seasonal = match day_of_week {
            5 | 6 => 300.0, // Weekend boost
            _ => 0.0,
        };

        // Monthly seasonality
        let monthly_seasonal = 150.0 * (2.0 * std::f64::consts::PI * i as f64 / 30.0).sin();

        // Random noise
        let noise = (i as f64 * 0.1).sin() * 50.0 + (i as f64 * 0.3).cos() * 30.0;

        let value = base + trend + seasonal + monthly_seasonal + noise;
        builder = builder.add_point(timestamp, value);
    }

    let ts = builder.frequency(Frequency::Daily).build()?;
    println!(
        "Generated {} sales records from {} to {}",
        ts.len(),
        ts.index
            .start()
            .ok_or_else(|| pandrs::core::error::Error::InvalidInput("No start".to_string()))?,
        ts.index
            .end()
            .ok_or_else(|| pandrs::core::error::Error::InvalidInput("No end".to_string()))?
    );

    Ok(ts)
}

/// Generate synthetic stock price data with trend and volatility
fn generate_stock_data(n_points: usize) -> Result<TimeSeries> {
    println!("\n=== Generating Synthetic Stock Price Data ===");

    let mut builder = TimeSeriesBuilder::new();
    let start_date = Utc
        .timestamp_opt(1609459200, 0)
        .single()
        .ok_or_else(|| pandrs::core::error::Error::InvalidInput("Invalid timestamp".to_string()))?;

    let mut price = 100.0;

    for i in 0..n_points {
        let timestamp = start_date + Duration::days(i as i64);

        // Random walk with drift
        let drift = 0.05;
        let volatility = 2.0;
        let change = drift + volatility * ((i as f64 * 0.1).sin() + (i as f64 * 0.3).cos()) * 0.3;

        price += change;
        price = price.max(50.0); // Floor price

        builder = builder.add_point(timestamp, price);
    }

    let ts = builder.frequency(Frequency::Daily).build()?;
    println!("Generated {} stock price records", ts.len());

    Ok(ts)
}

/// Generate synthetic energy consumption data with daily and weekly patterns
fn generate_energy_data(n_points: usize) -> Result<TimeSeries> {
    println!("\n=== Generating Synthetic Energy Consumption Data ===");

    let mut builder = TimeSeriesBuilder::new();
    let start_date = Utc
        .timestamp_opt(1640995200, 0)
        .single()
        .ok_or_else(|| pandrs::core::error::Error::InvalidInput("Invalid timestamp".to_string()))?;

    for i in 0..n_points {
        let timestamp = start_date + Duration::hours(i as i64);

        // Base consumption
        let base = 1000.0;

        // Daily pattern (higher during day, lower at night)
        let hour = i % 24;
        let daily_pattern = match hour {
            0..=5 => -200.0,  // Night
            6..=8 => 100.0,   // Morning peak
            9..=16 => 50.0,   // Day
            17..=20 => 150.0, // Evening peak
            _ => -100.0,      // Late evening
        };

        // Weekly pattern (lower on weekends)
        let day = (i / 24) % 7;
        let weekly_pattern = if day >= 5 { -100.0 } else { 0.0 };

        // Seasonal (summer vs winter)
        let seasonal = 200.0 * (2.0 * std::f64::consts::PI * i as f64 / (24.0 * 365.0)).sin();

        // Noise
        let noise = (i as f64 * 0.05).sin() * 30.0;

        let value = base + daily_pattern + weekly_pattern + seasonal + noise;
        builder = builder.add_point(timestamp, value.max(0.0));
    }

    let ts = builder.frequency(Frequency::Hour).build()?;
    println!("Generated {} hourly energy consumption records", ts.len());

    Ok(ts)
}

/// Demonstrate preprocessing pipeline
fn demonstrate_preprocessing(ts: &TimeSeries) -> Result<()> {
    println!("\n=== Time Series Preprocessing ===");

    // Create preprocessor with comprehensive settings
    let preprocessor = TimeSeriesPreprocessor::new()
        .with_missing_value_strategy(MissingValueStrategy::LinearInterpolation)
        .with_outlier_detection(OutlierDetection::ModifiedZScore { threshold: 3.5 })
        .with_normalization(Normalization::ZScore);

    let result = preprocessor.preprocess(ts)?;

    println!("Preprocessing Results:");
    println!("  Original length: {}", result.statistics.original_length);
    println!("  Final length: {}", result.statistics.final_length);
    println!(
        "  Missing values handled: {}",
        result.statistics.missing_values_handled
    );
    println!(
        "  Outliers detected: {}",
        result.statistics.outliers_detected
    );
    println!(
        "  Original mean: {:.2}, std: {:.2}",
        result.statistics.mean_before_after.0, result.statistics.std_before_after.0
    );
    println!(
        "  Processed mean: {:.2}, std: {:.2}",
        result.statistics.mean_before_after.1, result.statistics.std_before_after.1
    );

    println!("\nTransformations applied:");
    for (i, transform) in result.transformations.iter().enumerate() {
        println!(
            "  {}. {} (affected {} values)",
            i + 1,
            transform.transformation_type,
            transform.affected_values
        );
    }

    Ok(())
}

/// Demonstrate feature engineering
fn demonstrate_feature_engineering(ts: &TimeSeries) -> Result<()> {
    println!("\n=== Feature Engineering ===");

    let extractor = TimeSeriesFeatureExtractor::new()
        .with_window_sizes(vec![7, 14, 30])
        .with_ema_alphas(vec![0.1, 0.3, 0.5])
        .with_frequency_features(true)
        .with_complexity_features(false); // Disable for speed

    let features = extractor.extract_features(ts)?;

    println!("\nStatistical Features:");
    println!("  Mean: {:.2}", features.statistical.mean);
    println!("  Std Dev: {:.2}", features.statistical.std);
    println!("  Skewness: {:.2}", features.statistical.skewness);
    println!("  Kurtosis: {:.2}", features.statistical.kurtosis);
    println!(
        "  Min: {:.2}, Max: {:.2}",
        features.statistical.min, features.statistical.max
    );
    println!("  Zero crossings: {}", features.statistical.zero_crossings);
    println!(
        "  Peaks: {}, Valleys: {}",
        features.statistical.peaks, features.statistical.valleys
    );

    println!("\nWindow-based Features:");
    for (window_size, _values) in &features.window.moving_averages {
        println!(
            "  Moving Average (window={}): {} values",
            window_size,
            _values.len()
        );
    }

    println!("\nBollinger Bands:");
    println!(
        "  Percentage within bands: {:.1}%",
        features.window.bollinger_bands.pct_within_bands * 100.0
    );

    if features.frequency.dominant_frequency > 0.0 {
        println!("\nFrequency Domain Features:");
        println!(
            "  Dominant frequency: {:.4}",
            features.frequency.dominant_frequency
        );
        println!(
            "  Spectral centroid: {:.4}",
            features.frequency.spectral_centroid
        );
        println!(
            "  Spectral bandwidth: {:.4}",
            features.frequency.spectral_bandwidth
        );
    }

    Ok(())
}

/// Demonstrate seasonal decomposition
fn demonstrate_decomposition(ts: &TimeSeries) -> Result<()> {
    println!("\n=== Seasonal Decomposition ===");

    // Additive decomposition
    let decomposer = SeasonalDecomposition::new(DecompositionMethod::Additive).with_period(7);

    let result = decomposer.decompose(ts)?;

    println!("Decomposition Method: {:?}", result.method);
    println!("Seasonal Period: {}", result.period);

    println!("\nDecomposition Metrics:");
    println!(
        "  Trend variance ratio: {:.2}%",
        result.metrics.trend_variance_ratio * 100.0
    );
    println!(
        "  Seasonal variance ratio: {:.2}%",
        result.metrics.seasonal_variance_ratio * 100.0
    );
    println!(
        "  Residual variance ratio: {:.2}%",
        result.metrics.residual_variance_ratio * 100.0
    );
    println!("  Trend strength: {:.2}", result.metrics.trend_strength);
    println!(
        "  Seasonality strength: {:.2}",
        result.metrics.seasonality_strength
    );
    println!(
        "  Signal-to-noise ratio: {:.2}",
        result.metrics.signal_to_noise_ratio
    );
    println!("  Quality score: {:.2}%", result.quality_score() * 100.0);

    // Get seasonal indices
    let seasonal_indices = result.get_seasonal_indices();
    println!(
        "\nSeasonal Indices (first {} periods):",
        result.period.min(5)
    );
    for i in 0..result.period.min(5) {
        if let Some(&value) = seasonal_indices.get(&i) {
            println!("  Period {}: {:.2}", i, value);
        }
    }

    Ok(())
}

/// Demonstrate statistical tests
fn demonstrate_statistical_tests(ts: &TimeSeries) -> Result<()> {
    println!("\n=== Statistical Tests ===");

    let stats = TimeSeriesStats::compute(ts)?;

    println!("\nDescriptive Statistics:");
    println!("  Count: {}", stats.descriptive.count);
    println!(
        "  Mean: {:.2}, Median: {:.2}",
        stats.descriptive.mean, stats.descriptive.median
    );
    println!("  Std Dev: {:.2}", stats.descriptive.std);
    println!(
        "  Skewness: {:.2}, Kurtosis: {:.2}",
        stats.descriptive.skewness, stats.descriptive.kurtosis
    );
    println!("  IQR: {:.2}", stats.descriptive.iqr);

    println!("\nStationarity Tests:");
    println!("  ADF Test:");
    println!(
        "    Statistic: {:.4}",
        stats.stationarity_tests.adf_test.statistic
    );
    println!(
        "    P-value: {:.4}",
        stats.stationarity_tests.adf_test.p_value
    );
    println!(
        "    Is stationary: {}",
        stats.stationarity_tests.adf_test.is_stationary
    );
    println!("  KPSS Test:");
    println!(
        "    Statistic: {:.4}",
        stats.stationarity_tests.kpss_test.statistic
    );
    println!(
        "    Is stationary: {}",
        stats.stationarity_tests.kpss_test.is_stationary
    );
    println!(
        "  Overall: {}",
        if stats.stationarity_tests.is_stationary {
            "Stationary"
        } else {
            "Non-stationary"
        }
    );

    if !stats.stationarity_tests.is_stationary {
        println!(
            "  Recommended differencing: d={}, seasonal_d={}",
            stats
                .stationarity_tests
                .differencing_recommendation
                .recommended_d,
            stats
                .stationarity_tests
                .differencing_recommendation
                .recommended_seasonal_d
        );
    }

    println!("\nSeasonality Tests:");
    println!(
        "  Has seasonality: {}",
        stats.seasonality_tests.has_seasonality
    );
    if !stats.seasonality_tests.seasonal_periods.is_empty() {
        println!(
            "  Detected periods: {:?}",
            stats.seasonality_tests.seasonal_periods
        );
    }
    println!(
        "  Seasonal strength: {:.2}",
        stats.seasonality_tests.seasonal_test.seasonal_strength
    );

    println!("\nAutocorrelation Tests:");
    println!("  Ljung-Box test:");
    println!(
        "    Statistic: {:.4}",
        stats.autocorrelation_tests.ljung_box_test.statistic
    );
    println!(
        "    Has autocorrelation: {}",
        stats
            .autocorrelation_tests
            .ljung_box_test
            .has_autocorrelation
    );
    println!("  Durbin-Watson test:");
    println!(
        "    Statistic: {:.4}",
        stats.autocorrelation_tests.durbin_watson_test.statistic
    );
    println!(
        "    Result: {}",
        stats.autocorrelation_tests.durbin_watson_test.result
    );

    println!("\nNormality Tests:");
    println!("  Jarque-Bera test:");
    println!(
        "    Statistic: {:.4}",
        stats.normality_tests.jarque_bera_test.statistic
    );
    println!(
        "    Is normal: {}",
        stats.normality_tests.jarque_bera_test.is_normal
    );

    println!("\nOutlier Detection:");
    println!(
        "  Outlier percentage: {:.2}%",
        stats.outlier_tests.outlier_percentage
    );
    println!(
        "  Number of outliers: {}",
        stats.outlier_tests.outlier_indices.len()
    );

    Ok(())
}

/// Split time series into train and test sets
fn train_test_split(ts: &TimeSeries, train_ratio: f64) -> Result<(TimeSeries, TimeSeries)> {
    let split_point = (ts.len() as f64 * train_ratio) as usize;

    let train_dates: Vec<DateTime<Utc>> = (0..split_point)
        .filter_map(|i| ts.index.get(i).copied())
        .collect();
    let train_values: Vec<f64> = (0..split_point)
        .filter_map(|i| ts.values.get_f64(i))
        .collect();

    let test_dates: Vec<DateTime<Utc>> = (split_point..ts.len())
        .filter_map(|i| ts.index.get(i).copied())
        .collect();
    let test_values: Vec<f64> = (split_point..ts.len())
        .filter_map(|i| ts.values.get_f64(i))
        .collect();

    let train = TimeSeries::from_vecs(train_dates, train_values)?;
    let test = TimeSeries::from_vecs(test_dates, test_values)?;

    Ok((train, test))
}

/// Calculate forecast evaluation metrics
fn calculate_metrics(actual: &[f64], predicted: &[f64]) -> Result<ForecastMetrics> {
    if actual.len() != predicted.len() {
        return Err(pandrs::core::error::Error::DimensionMismatch(
            "Actual and predicted must have same length".to_string(),
        ));
    }

    let n = actual.len() as f64;

    // MAE
    let mae = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / n;

    // MSE and RMSE
    let mse = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / n;
    let rmse = mse.sqrt();

    // MAPE
    let mape = actual
        .iter()
        .zip(predicted.iter())
        .filter(|(a, _)| **a != 0.0)
        .map(|(a, p)| ((a - p) / a).abs())
        .sum::<f64>()
        / n
        * 100.0;

    // SMAPE
    let smape = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| {
            let denominator = (a.abs() + p.abs()) / 2.0;
            if denominator != 0.0 {
                (a - p).abs() / denominator
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / n
        * 100.0;

    Ok(ForecastMetrics {
        mae: Some(mae),
        mse: Some(mse),
        rmse: Some(rmse),
        mape: Some(mape),
        smape: Some(smape),
        aic: None,
        bic: None,
        log_likelihood: None,
    })
}

/// Demonstrate multiple forecasting models
fn demonstrate_forecasting_models(ts: &TimeSeries) -> Result<()> {
    println!("\n=== Forecasting Models Comparison ===");

    let (train, test) = train_test_split(ts, 0.8)?;
    let forecast_horizon = test.len();

    println!("Train size: {}, Test size: {}", train.len(), test.len());

    // Collect actual test values for evaluation
    let actual_values: Vec<f64> = (0..test.len())
        .filter_map(|i| test.values.get_f64(i))
        .collect();

    // 1. Simple Moving Average
    println!("\n1. Simple Moving Average (window=7)");
    let mut sma = SimpleMovingAverageForecaster::new(7);
    sma.fit(&train)?;
    let sma_result = sma.forecast(forecast_horizon, 0.95)?;
    let sma_predictions: Vec<f64> = (0..sma_result.forecast.len())
        .filter_map(|i| sma_result.forecast.values.get_f64(i))
        .collect();
    let sma_metrics = calculate_metrics(&actual_values, &sma_predictions)?;
    println!("  RMSE: {:.2}", sma_metrics.rmse.unwrap_or(0.0));
    println!("  MAE: {:.2}", sma_metrics.mae.unwrap_or(0.0));
    println!("  MAPE: {:.2}%", sma_metrics.mape.unwrap_or(0.0));

    // 2. Linear Trend
    println!("\n2. Linear Trend");
    let mut linear = LinearTrendForecaster::new();
    linear.fit(&train)?;
    let linear_result = linear.forecast(forecast_horizon, 0.95)?;
    let linear_predictions: Vec<f64> = (0..linear_result.forecast.len())
        .filter_map(|i| linear_result.forecast.values.get_f64(i))
        .collect();
    let linear_metrics = calculate_metrics(&actual_values, &linear_predictions)?;
    println!("  RMSE: {:.2}", linear_metrics.rmse.unwrap_or(0.0));
    println!("  MAE: {:.2}", linear_metrics.mae.unwrap_or(0.0));
    println!("  MAPE: {:.2}%", linear_metrics.mape.unwrap_or(0.0));
    println!(
        "  Parameters: slope={:.4}, intercept={:.2}",
        linear.parameters().get("slope").unwrap_or(&0.0),
        linear.parameters().get("intercept").unwrap_or(&0.0)
    );

    // 3. Exponential Smoothing (Simple)
    println!("\n3. Simple Exponential Smoothing (alpha=0.3)");
    let mut ses = ExponentialSmoothingForecaster::simple(0.3);
    ses.fit(&train)?;
    let ses_result = ses.forecast(forecast_horizon, 0.95)?;
    let ses_predictions: Vec<f64> = (0..ses_result.forecast.len())
        .filter_map(|i| ses_result.forecast.values.get_f64(i))
        .collect();
    let ses_metrics = calculate_metrics(&actual_values, &ses_predictions)?;
    println!("  RMSE: {:.2}", ses_metrics.rmse.unwrap_or(0.0));
    println!("  MAE: {:.2}", ses_metrics.mae.unwrap_or(0.0));
    println!("  MAPE: {:.2}%", ses_metrics.mape.unwrap_or(0.0));

    // 4. Double Exponential Smoothing (Holt's method)
    println!("\n4. Double Exponential Smoothing (alpha=0.3, beta=0.1)");
    let mut des = ExponentialSmoothingForecaster::double(0.3, 0.1);
    des.fit(&train)?;
    let des_result = des.forecast(forecast_horizon, 0.95)?;
    let des_predictions: Vec<f64> = (0..des_result.forecast.len())
        .filter_map(|i| des_result.forecast.values.get_f64(i))
        .collect();
    let des_metrics = calculate_metrics(&actual_values, &des_predictions)?;
    println!("  RMSE: {:.2}", des_metrics.rmse.unwrap_or(0.0));
    println!("  MAE: {:.2}", des_metrics.mae.unwrap_or(0.0));
    println!("  MAPE: {:.2}%", des_metrics.mape.unwrap_or(0.0));

    // 5. Triple Exponential Smoothing (Holt-Winters)
    if train.len() >= 14 {
        println!("\n5. Triple Exponential Smoothing (Holt-Winters, period=7)");
        let mut tes = ExponentialSmoothingForecaster::triple(0.3, 0.1, 0.1, 7);
        if let Ok(()) = tes.fit(&train) {
            let tes_result = tes.forecast(forecast_horizon, 0.95)?;
            let tes_predictions: Vec<f64> = (0..tes_result.forecast.len())
                .filter_map(|i| tes_result.forecast.values.get_f64(i))
                .collect();
            let tes_metrics = calculate_metrics(&actual_values, &tes_predictions)?;
            println!("  RMSE: {:.2}", tes_metrics.rmse.unwrap_or(0.0));
            println!("  MAE: {:.2}", tes_metrics.mae.unwrap_or(0.0));
            println!("  MAPE: {:.2}%", tes_metrics.mape.unwrap_or(0.0));
        }
    }

    // 6. ARIMA
    if train.len() >= 30 {
        println!("\n6. ARIMA(1,1,1)");
        let mut arima = ArimaForecaster::new(1, 1, 1);
        if let Ok(()) = arima.fit(&train) {
            let arima_result = arima.forecast(forecast_horizon, 0.95)?;
            let arima_predictions: Vec<f64> = (0..arima_result.forecast.len())
                .filter_map(|i| arima_result.forecast.values.get_f64(i))
                .collect();
            let arima_metrics = calculate_metrics(&actual_values, &arima_predictions)?;
            println!("  RMSE: {:.2}", arima_metrics.rmse.unwrap_or(0.0));
            println!("  MAE: {:.2}", arima_metrics.mae.unwrap_or(0.0));
            println!("  MAPE: {:.2}%", arima_metrics.mape.unwrap_or(0.0));
        }
    }

    // 7. SARIMA
    if train.len() >= 50 {
        println!("\n7. SARIMA(1,1,1)(1,0,1)[7]");
        let mut sarima = SarimaForecaster::new(1, 1, 1, 1, 0, 1, 7);
        if let Ok(()) = sarima.fit(&train) {
            let sarima_result = sarima.forecast(forecast_horizon, 0.95)?;
            let sarima_predictions: Vec<f64> = (0..sarima_result.forecast.len())
                .filter_map(|i| sarima_result.forecast.values.get_f64(i))
                .collect();
            let sarima_metrics = calculate_metrics(&actual_values, &sarima_predictions)?;
            println!("  RMSE: {:.2}", sarima_metrics.rmse.unwrap_or(0.0));
            println!("  MAE: {:.2}", sarima_metrics.mae.unwrap_or(0.0));
            println!("  MAPE: {:.2}%", sarima_metrics.mape.unwrap_or(0.0));
            if let Some(aic) = sarima.aic() {
                println!("  AIC: {:.2}", aic);
            }
            if let Some(bic) = sarima.bic(train.len()) {
                println!("  BIC: {:.2}", bic);
            }
        }
    }

    // 8. Auto ARIMA
    if train.len() >= 50 {
        println!("\n8. Auto ARIMA (automatic model selection)");
        let mut auto_arima = AutoArima::new()
            .max_p(2)
            .max_d(2)
            .max_q(2)
            .seasonal(7)
            .max_seasonal_p(1)
            .max_seasonal_d(1)
            .max_seasonal_q(1)
            .criterion(ModelSelectionCriterion::AICc);

        if let Ok(()) = auto_arima.fit(&train) {
            println!("{}", auto_arima.summary());

            let auto_result = auto_arima.forecast(forecast_horizon, 0.95)?;
            let auto_predictions: Vec<f64> = (0..auto_result.forecast.len())
                .filter_map(|i| auto_result.forecast.values.get_f64(i))
                .collect();
            let auto_metrics = calculate_metrics(&actual_values, &auto_predictions)?;
            println!("  RMSE: {:.2}", auto_metrics.rmse.unwrap_or(0.0));
            println!("  MAE: {:.2}", auto_metrics.mae.unwrap_or(0.0));
            println!("  MAPE: {:.2}%", auto_metrics.mape.unwrap_or(0.0));
        }
    }

    Ok(())
}

/// Use case 1: Sales Forecasting
fn sales_forecasting_use_case() -> Result<()> {
    println!("\n");
    println!("================================================================================");
    println!("                    USE CASE 1: SALES FORECASTING                              ");
    println!("================================================================================");
    println!("Goal: Forecast daily sales for the next 2 weeks using historical data");
    println!("Pattern: Weekly seasonality (higher sales on weekends) + upward trend");

    let ts = generate_sales_data(90)?;

    demonstrate_statistical_tests(&ts)?;
    demonstrate_decomposition(&ts)?;

    let (train, _test) = train_test_split(&ts, 0.85)?;

    println!("\n--- Forecasting Next 14 Days ---");

    // Use Holt-Winters for seasonal data with trend
    let mut forecaster = ExponentialSmoothingForecaster::triple(0.3, 0.1, 0.1, 7);
    forecaster.fit(&train)?;

    let result = forecaster.forecast(14, 0.95)?;

    println!("\nForecast Results:");
    for i in 0..result.forecast.len().min(7) {
        let forecast = result.forecast.values.get_f64(i).unwrap_or(0.0);
        let lower = result.lower_ci.values.get_f64(i).unwrap_or(0.0);
        let upper = result.upper_ci.values.get_f64(i).unwrap_or(0.0);
        let date = result
            .forecast
            .index
            .get(i)
            .ok_or_else(|| pandrs::core::error::Error::InvalidInput("No date".to_string()))?;

        println!(
            "  Day {}: {:.0} units (95% CI: [{:.0}, {:.0}])",
            date.format("%Y-%m-%d"),
            forecast,
            lower,
            upper
        );
    }
    println!("  ... (and 7 more days)");

    Ok(())
}

/// Use case 2: Stock Price Prediction
fn stock_price_prediction_use_case() -> Result<()> {
    println!("\n");
    println!("================================================================================");
    println!("                   USE CASE 2: STOCK PRICE PREDICTION                          ");
    println!("================================================================================");
    println!("Goal: Predict stock prices for next 5 trading days");
    println!("Pattern: Random walk with drift + volatility");

    let ts = generate_stock_data(100)?;

    demonstrate_statistical_tests(&ts)?;
    demonstrate_feature_engineering(&ts)?;

    let (train, test) = train_test_split(&ts, 0.9)?;

    println!("\n--- Predicting Next 5 Trading Days ---");

    // Use ARIMA for financial time series
    let mut forecaster = ArimaForecaster::new(1, 1, 1);
    forecaster.fit(&train)?;

    let result = forecaster.forecast(5, 0.95)?;

    let test_values: Vec<f64> = (0..test.len())
        .filter_map(|i| test.values.get_f64(i))
        .collect();
    let predicted_values: Vec<f64> = (0..result.forecast.len())
        .filter_map(|i| result.forecast.values.get_f64(i))
        .collect();

    if !predicted_values.is_empty() && predicted_values.len() <= test_values.len() {
        let metrics = calculate_metrics(&test_values[..predicted_values.len()], &predicted_values)?;

        println!("\nPrediction Accuracy:");
        println!("  RMSE: ${:.2}", metrics.rmse.unwrap_or(0.0));
        println!("  MAE: ${:.2}", metrics.mae.unwrap_or(0.0));
        println!("  MAPE: {:.2}%", metrics.mape.unwrap_or(0.0));
    }

    println!("\nForecasted Prices:");
    for i in 0..result.forecast.len() {
        let forecast = result.forecast.values.get_f64(i).unwrap_or(0.0);
        let lower = result.lower_ci.values.get_f64(i).unwrap_or(0.0);
        let upper = result.upper_ci.values.get_f64(i).unwrap_or(0.0);

        println!(
            "  Day {}: ${:.2} (95% CI: [${:.2}, ${:.2}])",
            i + 1,
            forecast,
            lower,
            upper
        );
    }

    Ok(())
}

/// Use case 3: Energy Consumption Forecasting
fn energy_forecasting_use_case() -> Result<()> {
    println!("\n");
    println!("================================================================================");
    println!("                USE CASE 3: ENERGY CONSUMPTION FORECASTING                     ");
    println!("================================================================================");
    println!("Goal: Forecast hourly energy consumption for next 24 hours");
    println!("Pattern: Daily pattern + weekly pattern + seasonal component");

    let ts = generate_energy_data(24 * 14)?; // 2 weeks of hourly data

    demonstrate_decomposition(&ts)?;

    let (train, _test) = train_test_split(&ts, 0.9)?;

    println!("\n--- Forecasting Next 24 Hours ---");

    // Use SARIMA for multiple seasonal patterns
    let mut forecaster = SarimaForecaster::new(1, 0, 1, 1, 0, 1, 24);
    forecaster.fit(&train)?;

    let result = forecaster.forecast(24, 0.95)?;

    println!("\nForecasted Energy Consumption:");
    println!("Peak hours prediction:");

    // Show only peak hours
    for hour in [0, 6, 9, 12, 17, 20, 23] {
        if hour < result.forecast.len() {
            let forecast = result.forecast.values.get_f64(hour).unwrap_or(0.0);
            let lower = result.lower_ci.values.get_f64(hour).unwrap_or(0.0);
            let upper = result.upper_ci.values.get_f64(hour).unwrap_or(0.0);

            println!(
                "  Hour {:02}:00: {:.0} kWh (95% CI: [{:.0}, {:.0}])",
                hour, forecast, lower, upper
            );
        }
    }

    Ok(())
}

/// Use case 4: Weather Data Prediction
fn weather_prediction_use_case() -> Result<()> {
    println!("\n");
    println!("================================================================================");
    println!("                   USE CASE 4: WEATHER DATA PREDICTION                         ");
    println!("================================================================================");
    println!("Goal: Predict temperature for next 7 days");
    println!("Pattern: Seasonal variation + daily fluctuations");

    // Generate synthetic temperature data
    let mut builder = TimeSeriesBuilder::new();
    let start_date = Utc
        .timestamp_opt(1640995200, 0)
        .single()
        .ok_or_else(|| pandrs::core::error::Error::InvalidInput("Invalid timestamp".to_string()))?;

    for i in 0..60 {
        let timestamp = start_date + Duration::days(i as i64);

        let base_temp = 15.0;
        let seasonal = 10.0 * (2.0 * std::f64::consts::PI * i as f64 / 365.0).sin();
        let daily_variation = 3.0 * (i as f64 * 0.5).sin();
        let noise = (i as f64 * 0.3).cos() * 2.0;

        let temp = base_temp + seasonal + daily_variation + noise;
        builder = builder.add_point(timestamp, temp);
    }

    let ts = builder.frequency(Frequency::Daily).build()?;

    demonstrate_preprocessing(&ts)?;

    let (train, test) = train_test_split(&ts, 0.85)?;

    println!("\n--- Predicting Next 7 Days ---");

    // Use Auto ARIMA for best model selection
    let mut forecaster = AutoArima::new().max_p(3).max_d(1).max_q(3);

    forecaster.fit(&train)?;

    let result = forecaster.forecast(7, 0.95)?;

    let test_values: Vec<f64> = (0..test.len())
        .filter_map(|i| test.values.get_f64(i))
        .collect();
    let predicted_values: Vec<f64> = (0..result.forecast.len())
        .filter_map(|i| result.forecast.values.get_f64(i))
        .collect();

    if !predicted_values.is_empty() && predicted_values.len() <= test_values.len() {
        let metrics = calculate_metrics(&test_values[..predicted_values.len()], &predicted_values)?;

        println!("\nPrediction Accuracy:");
        println!("  RMSE: {:.2}°C", metrics.rmse.unwrap_or(0.0));
        println!("  MAE: {:.2}°C", metrics.mae.unwrap_or(0.0));
    }

    println!("\nTemperature Forecast:");
    for i in 0..result.forecast.len() {
        let forecast = result.forecast.values.get_f64(i).unwrap_or(0.0);
        let lower = result.lower_ci.values.get_f64(i).unwrap_or(0.0);
        let upper = result.upper_ci.values.get_f64(i).unwrap_or(0.0);
        let date = result
            .forecast
            .index
            .get(i)
            .ok_or_else(|| pandrs::core::error::Error::InvalidInput("No date".to_string()))?;

        println!(
            "  {}: {:.1}°C (95% CI: [{:.1}°C, {:.1}°C])",
            date.format("%Y-%m-%d"),
            forecast,
            lower,
            upper
        );
    }

    Ok(())
}

/// Demonstrate best practices
fn demonstrate_best_practices() -> Result<()> {
    println!("\n");
    println!("================================================================================");
    println!("                         BEST PRACTICES GUIDE                                  ");
    println!("================================================================================");

    println!("\n1. MODEL SELECTION");
    println!("   - Use Simple Moving Average for: Stable series, quick baseline");
    println!("   - Use Linear Trend for: Series with clear linear trend, no seasonality");
    println!("   - Use Exponential Smoothing for: Recent data matters more, smooth trends");
    println!("     • Simple: Level only (no trend, no seasonality)");
    println!("     • Double (Holt): Level + trend (no seasonality)");
    println!("     • Triple (Holt-Winters): Level + trend + seasonality");
    println!("   - Use ARIMA for: Complex patterns, stationarity after differencing");
    println!("   - Use SARIMA for: Multiple seasonal patterns, complex data");
    println!("   - Use Auto ARIMA for: Automatic model selection, unknown patterns");

    println!("\n2. DATA PREPROCESSING");
    println!("   - Always check for stationarity first (ADF/KPSS tests)");
    println!("   - Apply differencing if non-stationary");
    println!("   - Handle missing values appropriately:");
    println!("     • Linear interpolation: Good for smooth data");
    println!("     • Forward/backward fill: When neighboring values are similar");
    println!("     • Mean/median fill: When data is stationary");
    println!("   - Detect and treat outliers (Modified Z-score or IQR methods)");
    println!("   - Consider normalization for ML-based methods");

    println!("\n3. FEATURE ENGINEERING");
    println!("   - Create lag features for autoregressive patterns");
    println!("   - Use rolling statistics for trend information");
    println!("   - Extract seasonal features (day of week, month, etc.)");
    println!("   - Consider frequency domain features for periodic data");

    println!("\n4. MODEL EVALUATION");
    println!("   - Always use time series cross-validation (not random splits)");
    println!(
        "   - Use multiple metrics: RMSE (penalizes large errors), MAE (robust), MAPE (percentage)"
    );
    println!("   - Check residuals: Should be white noise (no pattern)");
    println!("   - Use information criteria (AIC/BIC) for model selection");
    println!("   - Always provide confidence intervals");

    println!("\n5. COMMON PITFALLS TO AVOID");
    println!("   - ✗ Using future information in training (look-ahead bias)");
    println!("   - ✗ Ignoring seasonality in data");
    println!("   - ✗ Not checking for stationarity");
    println!("   - ✗ Using too many parameters (overfitting)");
    println!("   - ✗ Extrapolating too far into the future");
    println!("   - ✗ Ignoring outliers and missing values");

    println!("\n6. VALIDATION STRATEGIES");
    println!("   - Use rolling window cross-validation");
    println!("   - Hold out at least 10-20% of data for testing");
    println!("   - Check forecast accuracy at different horizons");
    println!("   - Monitor model performance over time");
    println!("   - Re-train periodically with new data");

    Ok(())
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!("      PANDRS - COMPREHENSIVE ADVANCED TIME SERIES FORECASTING EXAMPLE         ");
    println!("================================================================================");
    println!("\nThis example demonstrates the complete workflow for time series analysis");
    println!("and forecasting using PandRS, covering:");
    println!("  • Data preprocessing and outlier detection");
    println!("  • Feature engineering and extraction");
    println!("  • Seasonal decomposition");
    println!("  • Statistical tests (stationarity, seasonality, autocorrelation)");
    println!("  • Multiple forecasting models (ARIMA, SARIMA, Exponential Smoothing, etc.)");
    println!("  • Model evaluation with confidence intervals");
    println!("  • Real-world use cases");

    // Generate sample data and demonstrate core functionality
    let sales_ts = generate_sales_data(90)?;

    println!("\n");
    println!("================================================================================");
    println!("                    CORE FUNCTIONALITY DEMONSTRATIONS                          ");
    println!("================================================================================");

    demonstrate_preprocessing(&sales_ts)?;
    demonstrate_feature_engineering(&sales_ts)?;
    demonstrate_decomposition(&sales_ts)?;
    demonstrate_statistical_tests(&sales_ts)?;
    demonstrate_forecasting_models(&sales_ts)?;

    // Real-world use cases
    sales_forecasting_use_case()?;
    stock_price_prediction_use_case()?;
    energy_forecasting_use_case()?;
    weather_prediction_use_case()?;

    // Best practices
    demonstrate_best_practices()?;

    println!("\n");
    println!("================================================================================");
    println!("                              EXAMPLE COMPLETE                                 ");
    println!("================================================================================");
    println!("\nFor more information, see the documentation:");
    println!("  - forecasting.rs: Basic forecasting models");
    println!("  - advanced_forecasting.rs: SARIMA and Auto ARIMA");
    println!("  - preprocessing.rs: Data preprocessing pipeline");
    println!("  - decomposition.rs: Seasonal decomposition methods");
    println!("  - features.rs: Feature extraction");
    println!("  - stats.rs: Statistical tests");

    Ok(())
}

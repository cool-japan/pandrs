//! Extended ML Pipeline Example
//!
//! This example demonstrates the advanced ML pipeline features in PandRS,
//! including feature engineering, polynomial features, interaction features,
//! binning, rolling window operations, and custom transformations.

use pandrs::error::Result;
use pandrs::ml::{AdvancedPipeline, BinningStrategy, FeatureEngineeringStage, WindowOperation};
use pandrs::optimized::OptimizedDataFrame;

fn main() -> Result<()> {
    println!("=== Extended ML Pipeline Example ===\n");

    // Create sample dataset for financial analysis
    let mut df = OptimizedDataFrame::new();
    df.add_float_column(
        "price",
        vec![
            100.0, 102.0, 98.0, 105.0, 103.0, 107.0, 101.0, 110.0, 108.0, 112.0,
        ],
    )?;
    df.add_float_column(
        "volume",
        vec![
            1000.0, 1200.0, 800.0, 1500.0, 1100.0, 1300.0, 900.0, 1600.0, 1400.0, 1700.0,
        ],
    )?;
    df.add_float_column(
        "market_cap",
        vec![
            50000.0, 52000.0, 49000.0, 55000.0, 53000.0, 57000.0, 51000.0, 60000.0, 58000.0,
            62000.0,
        ],
    )?;

    println!("Original Dataset:");
    println!("{:?}", df);
    println!();

    // Example 1: Basic Feature Engineering Pipeline
    println!("=== Example 1: Basic Feature Engineering ===");
    basic_feature_engineering_example(&df)?;

    // Example 2: Advanced Pipeline with Multiple Operations
    println!("\n=== Example 2: Advanced Pipeline with Multiple Operations ===");
    advanced_pipeline_example(&df)?;

    // Example 3: Custom Transformation Pipeline
    println!("\n=== Example 3: Custom Transformation Pipeline ===");
    custom_transformation_example(&df)?;

    // Example 4: Financial Analysis Pipeline
    println!("\n=== Example 4: Financial Analysis Pipeline ===");
    financial_analysis_pipeline(&df)?;

    Ok(())
}

fn basic_feature_engineering_example(df: &OptimizedDataFrame) -> Result<()> {
    // Create feature engineering stage with polynomial features
    let feature_stage = FeatureEngineeringStage::new()
        .with_polynomial_features(vec!["price".to_string()], 2)
        .with_interaction_features(vec![("price".to_string(), "volume".to_string())]);

    // Create and execute pipeline
    let mut pipeline = AdvancedPipeline::new()
        .add_stage(Box::new(feature_stage))
        .with_monitoring(true);

    let result = pipeline.execute(df.clone())?;

    println!("Result with polynomial and interaction features:");
    println!("{:?}", result);

    // Show execution summary
    let summary = pipeline.execution_summary();
    println!("Execution Summary:");
    println!("- Total stages: {}", summary.total_stages);
    println!("- Total duration: {:?}", summary.total_duration);
    println!("- Peak memory usage: {} bytes", summary.peak_memory_usage);

    Ok(())
}

fn advanced_pipeline_example(df: &OptimizedDataFrame) -> Result<()> {
    // Create comprehensive feature engineering stage
    let feature_stage = FeatureEngineeringStage::new()
        .with_polynomial_features(vec!["price".to_string()], 3)
        .with_interaction_features(vec![
            ("price".to_string(), "volume".to_string()),
            ("price".to_string(), "market_cap".to_string()),
        ])
        .with_binning("price".to_string(), 5, BinningStrategy::EqualWidth)
        .with_rolling_window("price".to_string(), 3, WindowOperation::Mean)
        .with_rolling_window("volume".to_string(), 3, WindowOperation::Std);

    // Create pipeline with monitoring
    let mut pipeline = AdvancedPipeline::new()
        .add_stage(Box::new(feature_stage))
        .with_monitoring(true);

    let result = pipeline.execute(df.clone())?;

    println!("Result with comprehensive feature engineering:");
    println!("Columns: {:?}", result.column_names());
    println!(
        "Shape: {} rows × {} columns",
        result.row_count(),
        result.column_count()
    );

    // Show detailed execution metrics
    let summary = pipeline.execution_summary();
    println!("\nDetailed Execution Metrics:");
    for stage_exec in &summary.stage_details {
        println!("Stage: {}", stage_exec.stage_name);
        println!("  Duration: {:?}", stage_exec.duration);
        println!(
            "  Input rows: {}, Output rows: {}",
            stage_exec.input_rows, stage_exec.output_rows
        );
        println!("  Memory usage: {} bytes", stage_exec.memory_usage);
    }

    Ok(())
}

fn custom_transformation_example(df: &OptimizedDataFrame) -> Result<()> {
    // Create feature engineering stage with custom transformation
    let feature_stage = FeatureEngineeringStage::new()
        .with_polynomial_features(vec!["price".to_string()], 2)
        .with_custom_transform(
            "price_volatility_indicator".to_string(),
            |df: &OptimizedDataFrame| -> Result<OptimizedDataFrame> {
                let mut result = df.clone();

                // Get price column and calculate volatility indicator
                if let Ok(price_values) = df.get_float_column("price") {
                    let volatility: Vec<f64> = price_values
                        .windows(2)
                        .map(|window| ((window[1] - window[0]) / window[0] * 100.0).abs())
                        .collect();

                    // Pad with first value to maintain length
                    let mut full_volatility = vec![0.0];
                    full_volatility.extend(volatility);

                    result.add_float_column("price_volatility", full_volatility)?;
                }

                Ok(result)
            },
        );

    let mut pipeline = AdvancedPipeline::new().add_stage(Box::new(feature_stage));

    let result = pipeline.execute(df.clone())?;

    println!("Result with custom volatility transformation:");
    println!("{:?}", result);

    Ok(())
}

fn financial_analysis_pipeline(df: &OptimizedDataFrame) -> Result<()> {
    // Multi-stage pipeline for financial analysis

    // Stage 1: Technical indicators
    let technical_stage = FeatureEngineeringStage::new()
        .with_rolling_window("price".to_string(), 3, WindowOperation::Mean)  // Simple Moving Average
        .with_rolling_window("price".to_string(), 3, WindowOperation::Std)   // Price volatility
        .with_rolling_window("volume".to_string(), 3, WindowOperation::Sum)  // Volume sum
        .with_custom_transform(
            "momentum_indicator".to_string(),
            |df: &OptimizedDataFrame| -> Result<OptimizedDataFrame> {
                let mut result = df.clone();

                if let Ok(price_values) = df.get_float_column("price") {
                    let momentum: Vec<f64> = price_values.windows(3)
                        .map(|window| {
                            if window.len() >= 3 {
                                window[2] - window[0]  // Price momentum over 3 periods
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    // Pad to maintain length
                    let mut full_momentum = vec![0.0, 0.0];
                    full_momentum.extend(momentum);

                    result.add_float_column("momentum", full_momentum)?;
                }

                Ok(result)
            }
        );

    // Stage 2: Risk indicators
    let risk_stage = FeatureEngineeringStage::new()
        .with_binning("price".to_string(), 4, BinningStrategy::EqualFrequency)
        .with_binning("volume".to_string(), 3, BinningStrategy::EqualWidth)
        .with_custom_transform(
            "risk_score".to_string(),
            |df: &OptimizedDataFrame| -> Result<OptimizedDataFrame> {
                let mut result = df.clone();

                // Calculate simple risk score based on price volatility
                if let Ok(prices) = df.get_float_column("price") {
                    let risk_scores: Vec<f64> = prices
                        .windows(3)
                        .map(|window| {
                            if window.len() >= 3 {
                                // Calculate volatility as std deviation of window
                                let mean = window.iter().sum::<f64>() / window.len() as f64;
                                let variance =
                                    window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                                        / window.len() as f64;
                                variance.sqrt()
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    // Pad to maintain length
                    let mut full_risk_scores = vec![0.0, 0.0];
                    full_risk_scores.extend(risk_scores);

                    result.add_float_column("risk_score", full_risk_scores)?;
                }

                Ok(result)
            },
        );

    // Create multi-stage pipeline
    let mut pipeline = AdvancedPipeline::new()
        .add_stage(Box::new(technical_stage))
        .add_stage(Box::new(risk_stage))
        .with_monitoring(true);

    let result = pipeline.execute(df.clone())?;

    println!("Financial Analysis Pipeline Results:");
    println!(
        "Generated {} features from {} original features",
        result.column_count(),
        df.column_count()
    );
    println!("\nFinal columns: {:?}", result.column_names());

    // Show key metrics
    if let Ok(risk_scores) = result.get_float_column("risk_score") {
        let avg_risk: f64 = risk_scores.iter().sum::<f64>() / risk_scores.len() as f64;
        let max_risk = risk_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_risk = risk_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        println!("\nRisk Analysis:");
        println!("- Average risk score: {:.3}", avg_risk);
        println!("- Maximum risk score: {:.3}", max_risk);
        println!("- Minimum risk score: {:.3}", min_risk);
    }

    // Pipeline execution summary
    let summary = pipeline.execution_summary();
    println!("\nPipeline Performance:");
    println!("- Total execution time: {:?}", summary.total_duration);
    println!("- Stages executed: {}", summary.total_stages);
    println!("- Peak memory usage: {} bytes", summary.peak_memory_usage);

    // Per-stage breakdown
    println!("\nPer-stage breakdown:");
    for (i, stage_exec) in summary.stage_details.iter().enumerate() {
        println!(
            "Stage {}: {} ({:?})",
            i + 1,
            stage_exec.stage_name,
            stage_exec.duration
        );
    }

    Ok(())
}

//! Visualization Example using Plotters Backend
//!
//! This example demonstrates how to create visualizations using PandRS with the plotters backend.
//!
//! To run this example:
//!   cargo run --example visualization_plotters_example --features visualization

#[cfg(feature = "visualization")]
use pandrs::vis::direct::{DataFramePlotExt, SeriesPlotExt};
#[cfg(feature = "visualization")]
use pandrs::{DataFrame, Series};
#[cfg(feature = "visualization")]
use rand::{rng, Rng};

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example visualization_plotters_example --features \"visualization\"");
}

#[cfg(feature = "visualization")]
#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PandRS Visualization Example with Plotters");
    println!("==========================================");

    // Generate random data
    let mut rng = rng();
    let y1: Vec<f64> = (0..100)
        .map(|i| i as f64 + rng.random_range(-5.0..5.0))
        .collect();
    let y2: Vec<f64> = (0..100)
        .map(|i| i as f64 * 0.8 + 10.0 + rng.random_range(-3.0..3.0))
        .collect();

    // 1. Single series line chart
    println!("\n1. Creating line chart...");
    let series1 = Series::new(y1.clone(), Some("Data1".to_string()))?;
    series1.line_plot("line_chart.png", Some("Line Chart"))?;
    println!("   -> Saved to line_chart.png");

    // 2. Histogram
    println!("\n2. Creating histogram...");
    let hist_data: Vec<f64> = (0..1000).map(|_| rng.random_range(-50.0..50.0)).collect();
    let hist_series = Series::new(hist_data, Some("Distribution".to_string()))?;
    hist_series.histogram("histogram.png", Some(20), Some("Histogram"))?;
    println!("   -> Saved to histogram.png");

    // 3. DataFrame visualization
    println!("\n3. Creating DataFrame plots...");

    // Create a sample DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new((0..100).collect::<Vec<i32>>(), Some("x".to_string()))?,
    )?;
    df.add_column(
        "Data1".to_string(),
        Series::new(y1, Some("Data1".to_string()))?,
    )?;
    df.add_column(
        "Data2".to_string(),
        Series::new(y2, Some("Data2".to_string()))?,
    )?;

    // Scatter plot
    df.scatter_xy("x", "Data1", "scatter_chart.png", Some("Scatter Chart"))?;
    println!("   -> Saved to scatter_chart.png");

    // Multi-line plot
    df.multi_line_plot(
        &["Data1", "Data2"],
        "multi_line_chart.png",
        Some("Multi-Line Chart"),
    )?;
    println!("   -> Saved to multi_line_chart.png");

    // Bar chart
    let bar_series = Series::new(
        vec![25.0, 45.0, 30.0, 55.0, 40.0],
        Some("Values".to_string()),
    )?;
    bar_series.bar_plot("bar_chart.png", Some("Bar Chart"))?;
    println!("   -> Saved to bar_chart.png");

    println!("\n=== Visualization Example Completed ===");
    println!("\nGenerated files:");
    println!("  - line_chart.png");
    println!("  - histogram.png");
    println!("  - scatter_chart.png");
    println!("  - multi_line_chart.png");
    println!("  - bar_chart.png");

    Ok(())
}

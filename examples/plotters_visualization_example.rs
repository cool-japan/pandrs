//! Example of visualization features using Plotters
//!
//! This sample demonstrates the basic usage of high-quality visualization features with Plotters.

#[cfg(feature = "visualization")]
use pandrs::error::Result;
#[cfg(feature = "visualization")]
use pandrs::vis::direct::{DataFramePlotExt, SeriesPlotExt};
#[cfg(feature = "visualization")]
use pandrs::vis::PlotKind;
#[cfg(feature = "visualization")]
use pandrs::{DataFrame, Series};

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example plotters_visualization_example --features visualization");
}

#[cfg(feature = "visualization")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // 1. Example of plotting a single Series
    println!("Sample 1: Creating a plot for a single Series");
    let values = vec![15.0, 23.5, 18.2, 29.8, 32.1, 28.5, 19.2, 22.3, 25.6, 21.9];

    // Line chart creation
    let series = Series::new(values.clone(), Some("Temperature Change".to_string()))?;
    series.line_plot(
        "examples/temp_line.png",
        Some("Temperature Change Over Time"),
    )?;
    println!("  ✓ Line chart generated: examples/temp_line.png");

    // Bar chart creation
    series.bar_plot(
        "examples/temp_bar.png",
        Some("Temperature Change Over Time"),
    )?;
    println!("  ✓ Bar chart generated: examples/temp_bar.png");

    // Scatter plot in SVG format
    series.plot_svg(
        "examples/temp_scatter.svg",
        PlotKind::Scatter,
        Some("Temperature Change Over Time"),
    )?;
    println!("  ✓ Scatter plot (SVG) generated: examples/temp_scatter.svg");

    // Histogram creation
    series.histogram(
        "examples/temp_histogram.png",
        Some(5),
        Some("Temperature Distribution Histogram"),
    )?;
    println!("  ✓ Histogram generated: examples/temp_histogram.png");

    // 2. Visualization using DataFrame
    println!("\nSample 2: Visualization using DataFrame");
    let mut df = DataFrame::new();

    // Prepare data
    let days = Series::new(vec![1, 2, 3, 4, 5, 6, 7], Some("Day".to_string()))?;
    let temp = Series::new(
        vec![22.5, 25.1, 23.8, 27.2, 26.5, 24.9, 29.1],
        Some("Temperature".to_string()),
    )?;
    let humidity = Series::new(
        vec![67.0, 72.3, 69.5, 58.2, 62.1, 71.5, 55.8],
        Some("Humidity".to_string()),
    )?;
    let pressure = Series::new(
        vec![1013.2, 1010.5, 1009.8, 1014.5, 1018.2, 1015.7, 1011.3],
        Some("Pressure".to_string()),
    )?;

    // Add columns to DataFrame
    df.add_column("Day".to_string(), days)?;
    df.add_column("Temperature".to_string(), temp)?;
    df.add_column("Humidity".to_string(), humidity)?;
    df.add_column("Pressure".to_string(), pressure)?;

    // Create XY scatter plot (relationship between temperature and humidity)
    df.scatter_xy(
        "Temperature",
        "Humidity",
        "examples/temp_humidity.png",
        Some("Relationship Between Temperature and Humidity"),
    )?;
    println!("  ✓ Scatter plot (Temperature vs Humidity) generated: examples/temp_humidity.png");

    // Comparison of multiple series
    df.multi_line_plot(
        &["Temperature", "Humidity", "Pressure"],
        "examples/weather_multi.png",
        Some("Weather Data Trends"),
    )?;
    println!("  ✓ Multi-series plot generated: examples/weather_multi.png");

    println!("\nAll samples executed successfully. Please check the generated files.");
    Ok(())
}

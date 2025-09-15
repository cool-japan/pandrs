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
    println!("  cargo run --example visualization_example --features visualization");
}

#[cfg(feature = "visualization")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Visualization Example ===\n");

    // Create sample data
    let x = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string()))?;
    let y = Series::new(vec![2.0, 3.5, 4.2, 4.8, 7.0], Some("y".to_string()))?;
    let z = Series::new(vec![1.5, 2.2, 3.1, 5.3, 8.5], Some("z".to_string()))?;

    // Clone before adding to DataFrame
    let y_for_plot = y.clone();

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), x)?;
    df.add_column("y".to_string(), y)?;
    df.add_column("z".to_string(), z)?;

    // Plot Series using terminal
    println!("Plotting y series (check file 'y_series.png'):");
    y_for_plot.line_plot("y_series.png", Some("Sample Series Plot"))?;

    // Plot Scatter XY
    println!("\nPlotting XY Scatter Plot (check file 'xy_scatter.png'):");
    df.scatter_xy("x", "y", "xy_scatter.png", Some("X vs Y Scatter Plot"))?;

    // Single Series Line Plot
    println!("\nPlotting Line Graph (check file 'z_line.png'):");
    df.line_plot("z", "z_line.png", Some("Z Line Plot"))?;

    // Multiple Series Line Plot
    println!("\nPlotting Multiple Line Graph (check file 'multi_line.png'):");
    df.multi_line_plot(
        &["x", "y", "z"],
        "multi_line.png",
        Some("Multi-Column Plot"),
    )?;

    // Save as SVG format
    println!("\nSaving plot to SVG (check file 'y_plot.svg'):");
    let svg_path = "y_plot.svg";
    df.plot_svg("y", svg_path, PlotKind::Line, Some("Y Series SVG Plot"))?;
    println!("SVG Plot saved: {}", svg_path);

    println!("\n=== Visualization Example Complete ===");
    Ok(())
}

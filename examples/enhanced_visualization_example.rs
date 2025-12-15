#[cfg(feature = "visualization")]
use pandrs::error::Result;
#[cfg(feature = "visualization")]
use pandrs::vis::config::PlotSettings;
#[cfg(feature = "visualization")]
use pandrs::vis::direct::{DataFramePlotExt, SeriesPlotExt};
#[cfg(feature = "visualization")]
use pandrs::vis::plotters_ext::{OutputType, PlotKind as PlottersPlotKind};
#[cfg(feature = "visualization")]
use pandrs::vis::PlotKind;
#[cfg(feature = "visualization")]
use pandrs::{DataFrame, Series};

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example enhanced_visualization_example --features visualization");
}

#[cfg(feature = "visualization")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("Enhanced Visualization Example");
    println!("=============================");

    // Create sample data for Series
    let s = create_sample_series()?;

    // Create sample DataFrame
    let df = create_sample_dataframe()?;

    // Example 1: Direct Series plotting
    example_series_plots(&s)?;

    // Example 2: Direct DataFrame plotting
    example_dataframe_plots(&df)?;

    // Example 3: Multi-series plots
    example_multi_series_plots(&df)?;

    // Example 4: Box plots and categorical data visualization
    example_box_plots(&df)?;

    Ok(())
}

#[cfg(feature = "visualization")]
// Helper function to create a sample Series
#[allow(clippy::result_large_err)]
fn create_sample_series() -> Result<Series<f64>> {
    let values = vec![2.5, 3.1, 4.8, 6.3, 5.2, 7.5, 8.1, 7.9, 6.2, 5.5, 4.3, 3.8];
    Series::new(values, Some("Monthly Data".to_string()))
}

#[cfg(feature = "visualization")]
// Helper function to create a sample DataFrame
#[allow(clippy::result_large_err)]
fn create_sample_dataframe() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Add data columns
    let months = vec![
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    let temperature = vec![
        2.5, 3.1, 4.8, 6.3, 8.2, 13.5, 15.1, 14.9, 11.2, 8.5, 5.3, 3.8,
    ];
    let rainfall = vec![
        48.2, 38.1, 35.3, 30.6, 28.5, 24.1, 20.5, 22.8, 31.9, 40.2, 45.5, 50.1,
    ];
    let visitors = vec![15, 18, 25, 32, 38, 42, 50, 48, 36, 28, 22, 17];

    // Each location is a category
    let location = vec![
        "Urban", "Urban", "Urban", "Urban", "Suburban", "Suburban", "Suburban", "Suburban",
        "Rural", "Rural", "Rural", "Rural",
    ];

    // Add columns to DataFrame
    df.add_column(
        "month".to_string(),
        Series::new(months.clone(), Some("Month".to_string()))?,
    )?;
    df.add_column(
        "temperature".to_string(),
        Series::new(temperature.clone(), Some("Temperature (°C)".to_string()))?,
    )?;
    df.add_column(
        "rainfall".to_string(),
        Series::new(rainfall.clone(), Some("Rainfall (mm)".to_string()))?,
    )?;
    df.add_column(
        "visitors".to_string(),
        Series::new(visitors.clone(), Some("Visitors (thousands)".to_string()))?,
    )?;
    df.add_column(
        "location".to_string(),
        Series::new(location.clone(), Some("Location".to_string()))?,
    )?;

    Ok(df)
}

#[cfg(feature = "visualization")]
// Example of direct Series plots
#[allow(clippy::result_large_err)]
fn example_series_plots(s: &Series<f64>) -> Result<()> {
    println!("\nDirect Series Plotting Examples:");

    // Basic plot with minimal configuration
    s.plot_to("examples/output/series_basic.png", None)?;
    println!("Created basic plot: examples/output/series_basic.png");

    // Line plot
    s.line_plot("examples/output/series_line.png", Some("Monthly Trend"))?;
    println!("Created line plot: examples/output/series_line.png");

    // Bar plot
    s.bar_plot(
        "examples/output/series_bar.png",
        Some("Monthly Distribution"),
    )?;
    println!("Created bar plot: examples/output/series_bar.png");

    // Histogram
    s.histogram(
        "examples/output/series_histogram.png",
        Some(6),
        Some("Histogram"),
    )?;
    println!("Created histogram: examples/output/series_histogram.png");

    // Area plot
    s.area_plot(
        "examples/output/series_area.png",
        Some("Area Visualization"),
    )?;
    println!("Created area plot: examples/output/series_area.png");

    // SVG output
    s.plot_svg(
        "examples/output/series_svg.svg",
        PlottersPlotKind::Line,
        Some("SVG Line Plot"),
    )?;
    println!("Created SVG plot: examples/output/series_svg.svg");

    Ok(())
}

#[cfg(feature = "visualization")]
// Example of direct DataFrame plots
#[allow(clippy::result_large_err)]
fn example_dataframe_plots(df: &DataFrame) -> Result<()> {
    println!("\nDirect DataFrame Plotting Examples:");

    // Basic column plot
    df.plot_column("temperature", "examples/output/df_temperature.png", None)?;
    println!("Created column plot: examples/output/df_temperature.png");

    // Line plot of rainfall
    df.line_plot(
        "rainfall",
        "examples/output/df_rainfall_line.png",
        Some("Monthly Rainfall"),
    )?;
    println!("Created line plot: examples/output/df_rainfall_line.png");

    // Bar plot of visitors
    df.bar_plot(
        "visitors",
        "examples/output/df_visitors_bar.png",
        Some("Monthly Visitors"),
    )?;
    println!("Created bar plot: examples/output/df_visitors_bar.png");

    // Scatter plot comparing temperature and rainfall
    df.scatter_xy(
        "temperature",
        "rainfall",
        "examples/output/df_temp_vs_rain.png",
        Some("Temperature vs Rainfall"),
    )?;
    println!("Created scatter plot: examples/output/df_temp_vs_rain.png");

    // Area plot
    df.area_plot(
        "visitors",
        "examples/output/df_visitors_area.png",
        Some("Visitors Over Time"),
    )?;
    println!("Created area plot: examples/output/df_visitors_area.png");

    // SVG output
    df.plot_svg(
        "rainfall",
        "examples/output/df_rainfall.svg",
        PlottersPlotKind::Bar,
        Some("Rainfall (SVG)"),
    )?;
    println!("Created SVG plot: examples/output/df_rainfall.svg");

    Ok(())
}

#[cfg(feature = "visualization")]
// Example of multi-series plots
#[allow(clippy::result_large_err)]
fn example_multi_series_plots(df: &DataFrame) -> Result<()> {
    println!("\nMulti-Series Plotting Examples:");

    // Multiple line plot (temperature, rainfall, visitors)
    df.multi_line_plot(
        &["temperature", "rainfall", "visitors"],
        "examples/output/multi_series.png",
        Some("Multiple Metrics Over Time"),
    )?;
    println!("Created multi-series plot: examples/output/multi_series.png");

    // Create a custom setting for multi-series
    let mut settings = PlotSettings::default();
    settings.plot_kind = PlotKind::Line;
    settings.title = "Weather and Visitors Trends".to_string();
    settings.width = 1200;
    settings.height = 800;
    settings.show_grid = true;
    settings.show_legend = true;

    // Plot with custom settings
    // Use the multi_line_plot method instead of the deprecated plotters_plot_columns
    df.multi_line_plot(
        &["temperature", "rainfall"],
        "examples/output/custom_multi_series.png",
        Some(&settings.title),
    )?;
    println!("Created custom multi-series plot: examples/output/custom_multi_series.png");

    Ok(())
}

#[cfg(feature = "visualization")]
// Example of box plots and categorical visualization
#[allow(clippy::result_large_err)]
fn example_box_plots(df: &DataFrame) -> Result<()> {
    println!("\nBox Plot and Categorical Data Examples:");

    // Box plot of temperature by location
    df.box_plot(
        "temperature",
        "location",
        "examples/output/temperature_by_location.png",
        Some("Temperature Distribution by Location"),
    )?;
    println!("Created box plot: examples/output/temperature_by_location.png");

    // Box plot of rainfall by location
    df.box_plot(
        "rainfall",
        "location",
        "examples/output/rainfall_by_location.png",
        Some("Rainfall Distribution by Location"),
    )?;
    println!("Created box plot: examples/output/rainfall_by_location.png");

    // Box plot of visitors by location
    df.box_plot(
        "visitors",
        "location",
        "examples/output/visitors_by_location.png",
        Some("Visitor Distribution by Location"),
    )?;
    println!("Created box plot: examples/output/visitors_by_location.png");

    Ok(())
}

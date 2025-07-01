#[cfg(feature = "visualization")]
use pandrs::vis::{DataFramePlotExt, SeriesPlotExt};
#[cfg(feature = "visualization")]
use pandrs::vis::{OutputType, PlotKind, PlotSettings};
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
    // Generate random data
    let mut rng = rng();
    let x: Vec<i32> = (0..100).collect();
    let y1: Vec<f64> = (0..100)
        .map(|i| i as f64 + rng.random_range(-5.0..5.0))
        .collect();
    let y2: Vec<f64> = (0..100)
        .map(|i| i as f64 * 0.8 + 10.0 + rng.random_range(-3.0..3.0))
        .collect();
    let y3: Vec<f64> = (0..100)
        .map(|i| 50.0 + 30.0 * (i as f64 * 0.1).sin())
        .collect();

    // Single series line chart
    let series1 = Series::new(y1.clone(), Some("Data1".to_string()))?;

    let line_settings = PlotSettings {
        title: "Line Chart".to_string(),
        x_label: "X Axis".to_string(),
        y_label: "Y Value".to_string(),
        plot_kind: PlotKind::Line,
        ..PlotSettings::default()
    };

    println!("Creating line chart...");
    series1.line_plot("line_chart.png", line_settings)?;
    println!("-> Saved to line_chart.png");

    // Histogram
    let hist_data: Vec<f64> = (0..1000).map(|_| rng.random_range(-50.0..50.0)).collect();
    let hist_series = Series::new(hist_data, Some("Distribution".to_string()))?;

    let hist_settings = PlotSettings {
        title: "Histogram".to_string(),
        x_label: "Value".to_string(),
        y_label: "Frequency".to_string(),
        plot_kind: PlotKind::Histogram,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };

    println!("Creating histogram...");
    hist_series.histogram_plot("histogram.png", 20, hist_settings)?;
    println!("-> Saved to histogram.png");

    // Use DataFrame for multiple series chart
    let mut df = DataFrame::new();
    df.add_column("X".to_string(), Series::new(x, Some("X".to_string()))?)?;
    df.add_column(
        "Data1".to_string(),
        Series::new(y1, Some("Data1".to_string()))?,
    )?;
    df.add_column(
        "Data2".to_string(),
        Series::new(y2, Some("Data2".to_string()))?,
    )?;
    df.add_column(
        "Data3".to_string(),
        Series::new(y3.clone(), Some("Data3".to_string()))?,
    )?;

    // Scatter plot
    let scatter_settings = PlotSettings {
        title: "Scatter Plot".to_string(),
        x_label: "X Value".to_string(),
        y_label: "Y Value".to_string(),
        plot_kind: PlotKind::Scatter,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };

    println!("Creating scatter plot...");
    df.scatter_xy("X", "Data1", "scatter_chart.png", scatter_settings)?;
    println!("-> Saved to scatter_chart.png");

    // Multiple series line chart
    let multi_line_settings = PlotSettings {
        title: "Multiple Series Line Chart".to_string(),
        x_label: "X Value".to_string(),
        y_label: "Y Value".to_string(),
        plot_kind: PlotKind::Line,
        output_type: OutputType::SVG, // Save as SVG
        ..PlotSettings::default()
    };

    println!("Creating multiple series line chart...");
    df.multi_line_plot(
        &["Data1", "Data2", "Data3"],
        "multi_line_chart.svg",
        multi_line_settings,
    )?;
    println!("-> Saved to multi_line_chart.svg");

    // Bar chart
    let bar_values = vec![15, 30, 25, 40, 20];
    let categories = vec!["A", "B", "C", "D", "E"];

    let mut bar_df = DataFrame::new();
    bar_df.add_column(
        "Category".to_string(),
        Series::new(categories, Some("Category".to_string()))?,
    )?;
    bar_df.add_column(
        "Value".to_string(),
        Series::new(bar_values.clone(), Some("Value".to_string()))?,
    )?;

    let bar_settings = PlotSettings {
        title: "Bar Chart".to_string(),
        x_label: "Category".to_string(),
        y_label: "Value".to_string(),
        plot_kind: PlotKind::Bar,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };

    println!("Creating bar chart...");
    // Create bar chart using index
    let bar_series = Series::new(bar_values, Some("Value".to_string()))?;
    bar_series.plot("bar_chart.png", bar_settings)?;
    println!("-> Saved to bar_chart.png");

    // Area chart
    let area_settings = PlotSettings {
        title: "Area Chart".to_string(),
        x_label: "Time".to_string(),
        y_label: "Value".to_string(),
        plot_kind: PlotKind::Area,
        output_type: OutputType::PNG,
        ..PlotSettings::default()
    };

    println!("Creating area chart...");
    let area_series = Series::new(y3, Some("Waveform".to_string()))?;
    area_series.plot("area_chart.png", area_settings)?;
    println!("-> Saved to area_chart.png");

    println!("All charts have been generated.");
    Ok(())
}

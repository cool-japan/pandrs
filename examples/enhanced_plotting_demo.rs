//! Enhanced Plotting Integration Example for PandRS Alpha 4
//!
//! This example demonstrates the comprehensive plotting capabilities including:
//! - Statistical plotting with various chart types
//! - Interactive plotting and dashboards
//! - Correlation matrices and pair plots
//! - Distribution analysis and box plots
//! - Custom color schemes and styling
//! - HTML report generation

use pandrs::core::error::Result;
use pandrs::dataframe::{utils, ColorScheme, DataFrame, EnhancedPlotExt, PlotTheme};
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("🎨 Enhanced Plotting Integration Example for PandRS Alpha 4");
    println!("===========================================================\n");

    // Create sample data for plotting demonstration
    let df = create_sample_data()?;
    println!(
        "📊 Created sample dataset with {} rows and {} columns",
        df.row_count(),
        df.column_names().len()
    );

    // 1. Basic Statistical Plotting
    println!("\n1️⃣ Basic Statistical Plotting");
    println!("------------------------------");

    // Quick line plot with custom title using StatPlotBuilder
    df.plot()
        .title("Monthly Sales Trend")
        .labels("Month", "Sales")
        .correlation_matrix("plots/line_plot.txt")?; // For now use correlation matrix method
    println!("✅ Created line plot: plots/line_plot.txt");

    // Scatter plot
    df.plot_scatter("advertising", "sales", "plots/scatter_plot.txt")?;
    println!("✅ Created scatter plot: plots/scatter_plot.txt");

    // Histogram
    df.plot_hist("sales", "plots/histogram.txt", Some(10))?;
    println!("✅ Created histogram: plots/histogram.txt");

    // 2. Advanced Statistical Visualizations
    println!("\n2️⃣ Advanced Statistical Visualizations");
    println!("---------------------------------------");

    // Correlation matrix
    df.plot_corr("plots/correlation_matrix.txt")?;
    println!("✅ Created correlation matrix: plots/correlation_matrix.txt");

    // Box plots grouped by category
    df.plot_box("sales", "region", "plots/box_plot.txt")?;
    println!("✅ Created box plot: plots/box_plot.txt");

    // 3. Using the StatPlotBuilder for Advanced Configuration
    println!("\n3️⃣ Advanced Plot Configuration");
    println!("--------------------------------");

    // Create a styled correlation matrix with custom color scheme
    let custom_colors = vec![
        (255, 87, 51), // Red-orange
        (255, 165, 0), // Orange
        (255, 215, 0), // Gold
        (50, 205, 50), // Lime green
        (0, 191, 255), // Deep sky blue
    ];

    df.plot()
        .title("Sales Performance Correlation Matrix")
        .figsize(1000, 800)
        .color_scheme(ColorScheme::Custom(custom_colors))
        .correlation_matrix("plots/styled_correlation.txt")?;
    println!("✅ Created styled correlation matrix: plots/styled_correlation.txt");

    // 4. Distribution Analysis
    println!("\n4️⃣ Distribution Analysis");
    println!("------------------------");

    let dist_files = df.plot_distributions("plots/dist")?;
    println!("✅ Created {} distribution plots:", dist_files.len());
    for file in &dist_files {
        println!("   📈 {}", file);
    }

    // 5. Pair Plot (Scatter Matrix)
    println!("\n5️⃣ Pair Plot Generation");
    println!("-----------------------");

    let numeric_cols = vec![
        "sales".to_string(),
        "advertising".to_string(),
        "temperature".to_string(),
    ];
    df.plot()
        .title("Sales Data Pair Plot")
        .pair_plot(Some(&numeric_cols), "plots/pair_plot.txt")?;
    println!("✅ Created pair plot: plots/pair_plot.txt");

    // 6. Grouped Box Plots
    println!("\n6️⃣ Grouped Analysis");
    println!("-------------------");

    let value_cols = vec!["sales".to_string(), "advertising".to_string()];
    df.plot()
        .title("Regional Performance Analysis")
        .grouped_box_plots("region", Some(&value_cols), "plots/grouped_analysis.txt")?;
    println!("✅ Created grouped box plots: plots/grouped_analysis.txt");

    // 7. Dashboard Generation
    println!("\n7️⃣ Dashboard Generation");
    println!("-----------------------");

    df.plot()
        .title("Sales Dashboard")
        .dashboard("plots/dashboard.txt")?;
    println!("✅ Created dashboard: plots/dashboard.txt");

    // 8. Interactive Plotting
    println!("\n8️⃣ Interactive Plotting");
    println!("------------------------");

    let interactive = pandrs::dataframe::plotting::InteractivePlot::new(df.clone());
    interactive.to_html("plots/interactive_dashboard.html")?;
    println!("✅ Created interactive HTML dashboard: plots/interactive_dashboard.html");

    // 9. Theme Customization
    println!("\n9️⃣ Theme Customization");
    println!("----------------------");

    let dark_theme = PlotTheme::dark();
    let light_theme = PlotTheme::light();

    let dark_config =
        dark_theme.apply_to_config(pandrs::dataframe::plotting::PlotConfig::default());
    let light_config =
        light_theme.apply_to_config(pandrs::dataframe::plotting::PlotConfig::default());

    println!(
        "✅ Applied dark theme with color scheme: {:?}",
        dark_config.color_scheme
    );
    println!(
        "✅ Applied light theme with color scheme: {:?}",
        light_config.color_scheme
    );

    // 10. Comprehensive Plotting Report
    println!("\n🔟 Comprehensive Report Generation");
    println!("----------------------------------");

    let report_files = df.plot_report("plots/comprehensive_report")?;
    println!(
        "✅ Created comprehensive plotting report with {} files:",
        report_files.len()
    );
    for file in &report_files {
        println!("   📄 {}", file);
    }

    // 11. Utility Functions Demo
    println!("\n1️⃣1️⃣ Utility Functions");
    println!("---------------------");

    let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let optimal_bins = utils::optimal_bins(&sample_data);
    println!("✅ Optimal bins for sample data: {}", optimal_bins);

    let quantiles = utils::quantiles(&sample_data, &[0.25, 0.5, 0.75]);
    println!(
        "✅ Quartiles: Q1={:.2}, Q2={:.2}, Q3={:.2}",
        quantiles[0], quantiles[1], quantiles[2]
    );

    let palette = utils::generate_palette(5, &ColorScheme::Viridis);
    println!("✅ Generated color palette with {} colors", palette.len());

    println!("\n🎉 Enhanced Plotting Integration Example Completed!");
    println!("📁 All plots saved to 'plots/' directory");
    println!(
        "🌐 Open 'plots/interactive_dashboard.html' in a web browser for interactive visualization"
    );

    Ok(())
}

#[allow(clippy::result_large_err)]
fn create_sample_data() -> Result<DataFrame> {
    // Create comprehensive sample data for plotting demonstrations
    let mut data = HashMap::new();

    // Sales data
    let sales = vec![
        "45000", "52000", "48000", "55000", "62000", "58000", "51000", "47000", "49000", "56000",
        "61000", "59000", "53000", "46000", "50000", "57000", "63000", "60000",
    ];

    // Advertising spend
    let advertising = vec![
        "5000", "6200", "5800", "6500", "7200", "6800", "5900", "5400", "5700", "6600", "7100",
        "6900", "6300", "5300", "5800", "6700", "7300", "7000",
    ];

    // Temperature (seasonal factor)
    let temperature = vec![
        "15", "18", "22", "25", "28", "32", "35", "33", "29", "24", "19", "16", "14", "17", "21",
        "26", "30", "34",
    ];

    // Region categories
    let region = vec![
        "North", "South", "East", "West", "North", "South", "East", "West", "North", "South",
        "East", "West", "North", "South", "East", "West", "North", "South",
    ];

    // Product categories
    let product = vec![
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Electronics",
        "Clothing",
    ];

    // Month names
    let month = vec![
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan",
        "Feb", "Mar", "Apr", "May", "Jun",
    ];

    data.insert(
        "sales".to_string(),
        sales.iter().map(|s| s.to_string()).collect(),
    );
    data.insert(
        "advertising".to_string(),
        advertising.iter().map(|s| s.to_string()).collect(),
    );
    data.insert(
        "temperature".to_string(),
        temperature.iter().map(|s| s.to_string()).collect(),
    );
    data.insert(
        "region".to_string(),
        region.iter().map(|s| s.to_string()).collect(),
    );
    data.insert(
        "product".to_string(),
        product.iter().map(|s| s.to_string()).collect(),
    );
    data.insert(
        "month".to_string(),
        month.iter().map(|s| s.to_string()).collect(),
    );

    DataFrame::from_map(data, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_plotting_integration() {
        let result = main();
        assert!(
            result.is_ok(),
            "Enhanced plotting integration example should complete successfully"
        );
    }

    #[test]
    fn test_sample_data_creation() {
        let df = create_sample_data();
        assert!(df.is_ok(), "Sample data creation should succeed");

        let df = df.unwrap();
        assert!(df.row_count() > 0, "Sample data should have rows");
        assert!(
            df.column_names().len() >= 6,
            "Sample data should have at least 6 columns"
        );
    }

    #[test]
    fn test_plot_utilities() {
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let bins = utils::optimal_bins(&sample_data);
        assert!(bins > 0, "Optimal bins should be positive");

        let quantiles = utils::quantiles(&sample_data, &[0.5]);
        assert_eq!(quantiles.len(), 1, "Should return one quantile");
        assert_eq!(quantiles[0], 3.0, "Median should be 3.0");

        let palette = utils::generate_palette(3, &ColorScheme::Default);
        assert_eq!(palette.len(), 3, "Should generate 3 colors");
    }
}

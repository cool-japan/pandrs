//! Text-based visualization module for PandRS
//!
//! Provides ASCII/Unicode charts for quick data exploration in terminal environments.
//! Supports histograms, bar charts, line plots, and sparklines.

mod charts;
mod dataframe_ext;
mod sparkline;

pub use charts::{
    BarChart, BarChartConfig, BarOrientation, Histogram, HistogramConfig, LinePlot, LinePlotConfig,
    ScatterPlot, ScatterPlotConfig,
};
pub use dataframe_ext::{quick as viz_quick, DataFrameVizExt};
pub use sparkline::{Sparkline, SparklineStyle};

/// Chart rendering trait
pub trait Chart {
    /// Render the chart to a string
    fn render(&self) -> String;

    /// Render to stdout
    fn display(&self) {
        println!("{}", self.render());
    }
}

/// Common chart configuration
#[derive(Debug, Clone)]
pub struct ChartConfig {
    /// Chart width in characters
    pub width: usize,
    /// Chart height in characters
    pub height: usize,
    /// Show axis labels
    pub show_labels: bool,
    /// Show grid lines
    pub show_grid: bool,
    /// Title for the chart
    pub title: Option<String>,
    /// X-axis label
    pub x_label: Option<String>,
    /// Y-axis label
    pub y_label: Option<String>,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: 60,
            height: 20,
            show_labels: true,
            show_grid: false,
            title: None,
            x_label: None,
            y_label: None,
        }
    }
}

/// Chart style options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartStyle {
    /// Simple ASCII characters
    Ascii,
    /// Unicode block characters
    Unicode,
    /// Unicode braille characters (high resolution)
    Braille,
}

impl Default for ChartStyle {
    fn default() -> Self {
        Self::Unicode
    }
}

/// Quick visualization functions for Series/DataFrame data
pub mod quick {
    use super::*;

    /// Create a quick histogram from numeric data
    pub fn histogram(data: &[f64], bins: usize) -> String {
        let hist = Histogram::new(data, bins);
        hist.render()
    }

    /// Create a quick bar chart from labeled data
    pub fn bar_chart(labels: &[&str], values: &[f64]) -> String {
        let chart = BarChart::new(labels, values);
        chart.render()
    }

    /// Create a quick horizontal bar chart
    pub fn hbar_chart(labels: &[&str], values: &[f64]) -> String {
        let chart = BarChart::horizontal(labels, values);
        chart.render()
    }

    /// Create a quick line plot
    pub fn line_plot(data: &[f64]) -> String {
        let plot = LinePlot::new(data);
        plot.render()
    }

    /// Create a sparkline (inline mini chart)
    pub fn sparkline(data: &[f64]) -> String {
        let spark = Sparkline::new(data);
        spark.render()
    }

    /// Create a scatter plot
    pub fn scatter(x: &[f64], y: &[f64]) -> String {
        let plot = ScatterPlot::new(x, y);
        plot.render()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_config_default() {
        let config = ChartConfig::default();
        assert_eq!(config.width, 60);
        assert_eq!(config.height, 20);
        assert!(config.show_labels);
    }

    #[test]
    fn test_quick_histogram() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let result = quick::histogram(&data, 5);
        assert!(!result.is_empty());
        assert!(result.contains("█") || result.contains("#"));
    }

    #[test]
    fn test_quick_sparkline() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = quick::sparkline(&data);
        assert!(!result.is_empty());
    }
}

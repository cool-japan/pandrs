//! Configuration for visualization functionality
//!
//! This module provides configuration structures for both text-based and
//! high-quality visualization features in the PandRS library.

/// Plot types
#[derive(Debug, Clone, Copy)]
pub enum PlotType {
    /// Line graph
    Line,
    /// Scatter plot
    Scatter,
    /// Point plot
    Points,
}

/// Plot output formats
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Terminal output
    Terminal,
    /// File output (text format)
    TextFile,
}

/// Plot configuration for text-based plots
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Width (characters)
    pub width: u32,
    /// Height (lines)
    pub height: u32,
    /// Plot type
    pub plot_type: PlotType,
    /// Output format
    pub format: OutputFormat,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 80,
            height: 25,
            plot_type: PlotType::Line,
            format: OutputFormat::Terminal,
        }
    }
}

/// Plot types (high-quality visualization)
#[derive(Debug, Clone, Copy)]
pub enum PlotKind {
    /// Line graph
    Line,
    /// Scatter plot
    Scatter,
    /// Bar chart
    Bar,
    /// Histogram
    Histogram,
    /// Box plot
    BoxPlot,
    /// Area chart
    Area,
}

/// Plot output formats (high-quality visualization)
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    /// PNG image
    PNG,
    /// SVG format
    SVG,
}

/// Extended plot settings for high-quality visualization
#[derive(Debug, Clone)]
pub struct PlotSettings {
    /// Title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Width of the graph (pixels)
    pub width: u32,
    /// Height of the graph (pixels)
    pub height: u32,
    /// Plot type
    pub plot_kind: PlotKind,
    /// Output format
    pub output_type: OutputType,
    /// Show legend
    pub show_legend: bool,
    /// Show grid
    pub show_grid: bool,
    /// Color palette
    pub color_palette: Vec<(u8, u8, u8)>,
}

impl Default for PlotSettings {
    fn default() -> Self {
        PlotSettings {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 800,
            height: 600,
            plot_kind: PlotKind::Line,
            output_type: OutputType::PNG,
            show_legend: true,
            show_grid: true,
            color_palette: vec![
                (0, 123, 255),  // Blue
                (255, 99, 71),  // Red
                (46, 204, 113), // Green
                (255, 193, 7),  // Yellow
                (142, 68, 173), // Purple
                (52, 152, 219), // Cyan
                (243, 156, 18), // Orange
                (211, 84, 0),   // Brown
            ],
        }
    }
}

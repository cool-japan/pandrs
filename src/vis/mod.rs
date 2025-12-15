//! Module providing data visualization functionality
//!
//! This module includes both text-based (textplots) and high-quality visualization (plotters)
//! capabilities, as well as direct plotting methods on DataFrame and Series objects.
//!
//! ## ASCII Visualization
//!
//! The `ascii` submodule provides standalone ASCII/Unicode chart rendering without
//! external dependencies:
//!
//! - Histograms for distribution visualization
//! - Bar charts (horizontal and vertical)
//! - Line plots for time series
//! - Scatter plots for correlation
//! - Sparklines for inline mini charts

// Module structure
pub mod config;
pub mod direct;
pub mod plotters;
pub mod text;

// ASCII visualization (no external dependencies)
pub mod ascii;

// Backward compatibility layer
mod backward_compat;

// Re-export public items
pub use self::config::{OutputFormat, OutputType, PlotConfig, PlotKind, PlotSettings, PlotType};
pub use self::direct::{DataFramePlotExt, SeriesPlotExt};
pub use self::plotters::backend::{
    plot_boxplot_png, plot_boxplot_svg, plot_histogram_png, plot_histogram_svg,
    plot_multi_series_png, plot_multi_series_svg, plot_series_xy_png, plot_series_xy_svg,
};
pub use self::text::plot_xy;

// Re-export ASCII visualization types
pub use self::ascii::{
    viz_quick, BarChart, BarChartConfig, BarOrientation, Chart, ChartConfig, ChartStyle,
    DataFrameVizExt, Histogram, HistogramConfig, LinePlot, LinePlotConfig, ScatterPlot,
    ScatterPlotConfig, Sparkline, SparklineStyle,
};

// For backward compatibility, re-export the backward-compatible functionality
#[allow(deprecated)]
pub use backward_compat::*;

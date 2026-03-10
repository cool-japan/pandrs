//! SVG/HTML visualization module for PandRS
//!
//! Provides a pure-Rust SVG chart engine with no external dependencies.
//!
//! ## Chart Types
//!
//! - [`charts::BarChart`] — vertical and horizontal bar charts
//! - [`charts::LineChart`] — line chart with optional markers and area fill
//! - [`charts::ScatterPlot`] — scatter plot with configurable marker shape
//! - [`charts::SvgHistogram`] — frequency histogram
//! - [`charts::HeatMap`] — 2D heatmap with color gradient
//! - [`charts::PieChart`] — pie and donut charts
//!
//! ## DataFrame Integration
//!
//! The [`dataframe_ext::SvgVisualize`] trait is implemented for `DataFrame`,
//! providing convenient `plot_*_svg()` methods.
//!
//! ## Example
//!
//! ```rust,no_run
//! use pandrs::vis::svg::{SvgChartConfig, SvgVisualize};
//! use pandrs::DataFrame;
//!
//! // df.plot_bar_svg("category", "value", None)
//! ```

pub mod charts;
pub mod colors;
pub mod dataframe_ext;
pub mod engine;

// Re-export commonly used types
pub use charts::{
    BarChart, BarOrientation, HeatMap, LegendPosition, LineChart, LineSeries, Margins, MarkerShape,
    PieChart, ScatterPlot, SvgChartConfig, SvgHistogram,
};
pub use colors::{Color, ColorGradient, ColorScheme};
pub use dataframe_ext::{SvgPlotType, SvgVisualize};
pub use engine::{DrawStyle, PathBuilder, SvgCanvas, SvgDefs, SvgGroup, Transform};

//! Module providing data visualization functionality
//!
//! This module includes both text-based (textplots) and high-quality visualization (plotters)
//! capabilities, as well as direct plotting methods on DataFrame and Series objects.

// Module structure
pub mod config;
pub mod direct;
pub mod plotters;
pub mod text;

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

// For backward compatibility, re-export the backward-compatible functionality
#[allow(deprecated)]
pub use backward_compat::*;

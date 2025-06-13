//! Backward compatibility module for visualization functionality
//!
//! This module provides deprecated re-exports of the old visualization API for backward compatibility.

#[cfg(feature = "visualization")]
pub mod direct_plot;
#[cfg(feature = "visualization")]
pub mod plotters_ext;

use crate::error::Result;
use crate::vis::config::{OutputFormat, PlotConfig, PlotType};
use crate::DataFrame;
use crate::Series;
use std::path::Path;

// Re-export text-based plotting functionality for backward compatibility
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use `pandrs::vis::text::plot_xy` instead"
)]
pub use crate::vis::text::plot_xy;

// Implement backward-compatible Series visualization methods
impl<T> Series<T>
where
    T: Clone + Copy + Into<f32> + std::fmt::Debug,
{
    /// Plot Series and save to file or display in terminal
    ///
    /// Note: This implementation is kept for backward compatibility.
    /// New code should use the `plot_to`, `line_plot`, etc. methods instead.
    #[allow(deprecated)]
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `Series.plot_to()` or `Series.line_plot()` instead"
    )]
    pub fn plot<P: AsRef<Path>>(&self, path: P, config: PlotConfig) -> Result<()> {
        let values: Vec<f32> = self.values().iter().map(|v| (*v).into()).collect();
        let indices: Vec<f32> = (0..values.len()).map(|i| i as f32).collect();

        plot_xy(&indices, &values, path, config)
    }
}

// Implement backward-compatible DataFrame visualization methods
impl DataFrame {
    /// Plot two columns as XY coordinates
    ///
    /// Note: This implementation is kept for backward compatibility.
    /// New code should use the `scatter_xy` method instead.
    #[allow(deprecated)]
    #[deprecated(since = "0.1.0-alpha.2", note = "Use `DataFrame.scatter_xy()` instead")]
    pub fn plot_xy<P: AsRef<Path>>(
        &self,
        x_col: &str,
        y_col: &str,
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // Use the old implementation that calls plot_xy
        #[allow(deprecated)]
        {
            let x_values = self.get_column_numeric_values(x_col)?;
            let y_values = self.get_column_numeric_values(y_col)?;

            let x_f32: Vec<f32> = x_values.iter().map(|&v| v as f32).collect();
            let y_f32: Vec<f32> = y_values.iter().map(|&v| v as f32).collect();

            plot_xy(&x_f32, &y_f32, path, config)
        }
    }

    /// Draw line graphs for multiple columns
    ///
    /// Note: This implementation is kept for backward compatibility.
    /// New code should use the `multi_line_plot` method instead.
    #[allow(deprecated)]
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `DataFrame.multi_line_plot()` instead"
    )]
    pub fn plot_lines<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        config: PlotConfig,
    ) -> Result<()> {
        // Use the old implementation that calls plot_xy
        #[allow(deprecated)]
        {
            // Use only the first column (textplots has limitations for displaying multiple series)
            if let Some(&first_col) = columns.first() {
                let values = self.get_column_numeric_values(first_col)?;
                let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

                let indices: Vec<f32> = (0..self.row_count()).map(|i| i as f32).collect();

                let mut custom_config = config;
                custom_config.title = format!("{} ({})", custom_config.title, first_col);

                return plot_xy(&indices, &values_f32, path, custom_config);
            }
        }

        Err(crate::error::PandRSError::Empty(
            "No columns to plot".to_string(),
        ))
    }
}

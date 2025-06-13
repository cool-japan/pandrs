//! Direct plotting functionality for DataFrame and Series
//!
//! This module adds direct plotting methods to DataFrame and Series objects, making it
//! easier to create visualizations with less code.

use crate::error::Result;
#[cfg(feature = "visualization")]
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::vis::config::PlotKind;
#[cfg(feature = "visualization")]
use crate::vis::config::{OutputType, PlotSettings};
// Import the plotters extension methods
#[cfg(feature = "visualization")]
use crate::vis::backward_compat::plotters_ext::*;
use crate::DataFrame;
use crate::Series;
use std::path::Path;

/// Extension trait for Series to add direct plotting methods
pub trait SeriesPlotExt<T> {
    /// Plot this Series with minimal configuration
    fn plot_to<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()>;

    /// Create a line plot from this Series
    fn line_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()>;

    /// Create a scatter plot from this Series
    fn scatter_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()>;

    /// Create a bar plot from this Series
    fn bar_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()>;

    /// Create a histogram from this Series
    fn histogram<P: AsRef<Path>>(
        &self,
        path: P,
        bins: Option<usize>,
        title: Option<&str>,
    ) -> Result<()>;

    /// Create an area plot from this Series
    fn area_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()>;

    /// Save the plot as SVG format
    fn plot_svg<P: AsRef<Path>>(
        &self,
        path: P,
        plot_kind: PlotKind,
        title: Option<&str>,
    ) -> Result<()>;
}

/// Extension trait for DataFrame to add direct plotting methods
pub trait DataFramePlotExt {
    /// Plot a column from this DataFrame with minimal configuration
    fn plot_column<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>)
        -> Result<()>;

    /// Create a line plot for a column
    fn line_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()>;

    /// Create a scatter plot for a column
    fn scatter_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()>;

    /// Create a bar plot for a column
    fn bar_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()>;

    /// Create an area plot for a column
    fn area_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()>;

    /// Create a box plot for a column grouped by a category
    fn box_plot<P: AsRef<Path>>(
        &self,
        value_column: &str,
        category_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()>;

    /// Create a scatter plot between two columns
    fn scatter_xy<P: AsRef<Path>>(
        &self,
        x_column: &str,
        y_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()>;

    /// Create a line plot for multiple columns
    fn multi_line_plot<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        title: Option<&str>,
    ) -> Result<()>;

    /// Save the plot as SVG format
    fn plot_svg<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        plot_kind: PlotKind,
        title: Option<&str>,
    ) -> Result<()>;
}

#[cfg(feature = "visualization")]
impl<T> SeriesPlotExt<T> for Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    fn plot_to<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();

        // Use title if provided, otherwise use series name
        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    fn line_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Line;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Line Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    fn scatter_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Scatter;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Scatter Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    fn bar_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Bar;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Bar Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    fn histogram<P: AsRef<Path>>(
        &self,
        path: P,
        bins: Option<usize>,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Histogram;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Histogram", name);
        }

        self.plotters_histogram(path, bins.unwrap_or(10), settings)
    }

    fn area_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Area;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Area Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        path: P,
        plot_kind: PlotKind,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = plot_kind;
        settings.output_type = OutputType::SVG;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Plot", name);
        }

        self.plotters_plot(path, settings)
    }
}

#[cfg(feature = "visualization")]
impl DataFramePlotExt for DataFrame {
    fn plot_column<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();

        // Use title if provided, otherwise use column name
        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Plot", column);
        }

        // Use column name as y-axis label
        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }

    fn line_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Line;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Line Plot", column);
        }

        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }

    fn scatter_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Scatter;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Scatter Plot", column);
        }

        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }

    fn bar_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Bar;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Bar Plot", column);
        }

        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }

    fn area_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Area;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Area Plot", column);
        }

        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }

    fn box_plot<P: AsRef<Path>>(
        &self,
        value_column: &str,
        category_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::BoxPlot;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} by {}", value_column, category_column);
        }

        settings.x_label = category_column.to_string();
        settings.y_label = value_column.to_string();

        self.plotters_boxplot(category_column, value_column, path, settings)
    }

    fn scatter_xy<P: AsRef<Path>>(
        &self,
        x_column: &str,
        y_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Scatter;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} vs {}", y_column, x_column);
        }

        settings.x_label = x_column.to_string();
        settings.y_label = y_column.to_string();

        self.plotters_scatter(x_column, y_column, path, settings)
    }

    fn multi_line_plot<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Line;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = "Multi Column Plot".to_string();
        }

        self.plotters_plot_columns(columns, path, settings)
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        plot_kind: PlotKind,
        title: Option<&str>,
    ) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = plot_kind;
        settings.output_type = OutputType::SVG;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else {
            settings.title = format!("{} Plot", column);
        }

        settings.y_label = column.to_string();

        self.plotters_plot_column(column, path, settings)
    }
}

#[cfg(feature = "visualization")]
impl DataFramePlotExt for OptimizedDataFrame {
    fn plot_column<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        // Convert to regular DataFrame first
        let df = self.to_dataframe()?;
        df.plot_column(column, path, title)
    }

    fn line_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let df = self.to_dataframe()?;
        df.line_plot(column, path, title)
    }

    fn scatter_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = self.to_dataframe()?;
        df.scatter_plot(column, path, title)
    }

    fn bar_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let df = self.to_dataframe()?;
        df.bar_plot(column, path, title)
    }

    fn area_plot<P: AsRef<Path>>(&self, column: &str, path: P, title: Option<&str>) -> Result<()> {
        let df = self.to_dataframe()?;
        df.area_plot(column, path, title)
    }

    fn box_plot<P: AsRef<Path>>(
        &self,
        value_column: &str,
        category_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = self.to_dataframe()?;
        df.box_plot(value_column, category_column, path, title)
    }

    fn scatter_xy<P: AsRef<Path>>(
        &self,
        x_column: &str,
        y_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = self.to_dataframe()?;
        df.scatter_xy(x_column, y_column, path, title)
    }

    fn multi_line_plot<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = self.to_dataframe()?;
        df.multi_line_plot(columns, path, title)
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        plot_kind: PlotKind,
        title: Option<&str>,
    ) -> Result<()> {
        let df = self.to_dataframe()?;
        df.plot_svg(column, path, plot_kind, title)
    }
}

// Fallback implementations when visualization is not enabled
#[cfg(not(feature = "visualization"))]
impl<T> SeriesPlotExt<T> for Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    fn plot_to<P: AsRef<Path>>(&self, _path: P, _title: Option<&str>) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn line_plot<P: AsRef<Path>>(&self, _path: P, _title: Option<&str>) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn scatter_plot<P: AsRef<Path>>(&self, _path: P, _title: Option<&str>) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn bar_plot<P: AsRef<Path>>(&self, _path: P, _title: Option<&str>) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn histogram<P: AsRef<Path>>(
        &self,
        _path: P,
        _bins: Option<usize>,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn area_plot<P: AsRef<Path>>(&self, _path: P, _title: Option<&str>) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        _path: P,
        _plot_kind: crate::vis::config::PlotKind,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "visualization"))]
impl DataFramePlotExt for DataFrame {
    fn plot_column<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn line_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn scatter_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn bar_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn area_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn box_plot<P: AsRef<Path>>(
        &self,
        _value_column: &str,
        _category_column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn scatter_xy<P: AsRef<Path>>(
        &self,
        _x_column: &str,
        _y_column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn multi_line_plot<P: AsRef<Path>>(
        &self,
        _columns: &[&str],
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _plot_kind: crate::vis::config::PlotKind,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "visualization"))]
impl DataFramePlotExt for crate::optimized::dataframe::OptimizedDataFrame {
    fn plot_column<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn line_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn scatter_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn bar_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn area_plot<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn box_plot<P: AsRef<Path>>(
        &self,
        _value_column: &str,
        _category_column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn scatter_xy<P: AsRef<Path>>(
        &self,
        _x_column: &str,
        _y_column: &str,
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn multi_line_plot<P: AsRef<Path>>(
        &self,
        _columns: &[&str],
        _path: P,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }

    fn plot_svg<P: AsRef<Path>>(
        &self,
        _column: &str,
        _path: P,
        _plot_kind: crate::vis::config::PlotKind,
        _title: Option<&str>,
    ) -> Result<()> {
        Err(crate::error::PandRSError::FeatureNotAvailable(
            "Visualization feature is not enabled. Recompile with --feature visualization"
                .to_string(),
        ))
    }
}

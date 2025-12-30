//! Direct plotting functionality for DataFrame and Series
//!
//! This module adds direct plotting methods to DataFrame and Series objects, making it
//! easier to create visualizations with less code.

use crate::error::Result;
use crate::optimized::convert::standard_dataframe;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::vis::backward_compat::plotters_ext::{OutputType, PlotKind, PlotSettings};
use crate::DataFrame;
use crate::Series;
use std::path::Path;

// Add direct plotting methods to Series
impl<T> Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    /// Directly plot this Series with minimal configuration
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional - will use series name if not provided)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::Series;
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let series = Series::new(data, Some("values".to_string())).unwrap();
    /// series.plot_to("series_plot.png", None).unwrap();
    /// ```
    pub fn plot_to<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();

        // Use title if provided, otherwise use series name
        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    /// Directly create a line plot from this Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn line_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Line;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Line Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    /// Directly create a scatter plot from this Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn scatter_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Scatter;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Scatter Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    /// Directly create a bar plot from this Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn bar_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Bar;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Bar Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    /// Directly create a histogram from this Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `bins` - Number of bins (default: 10)
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn histogram<P: AsRef<Path>>(
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

    /// Directly create an area plot from this Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn area_plot<P: AsRef<Path>>(&self, path: P, title: Option<&str>) -> Result<()> {
        let mut settings = PlotSettings::default();
        settings.plot_kind = PlotKind::Area;

        if let Some(title_str) = title {
            settings.title = title_str.to_string();
        } else if let Some(name) = self.name() {
            settings.title = format!("{} Area Plot", name);
        }

        self.plotters_plot(path, settings)
    }

    /// Save the plot as SVG format
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the SVG file
    /// * `plot_kind` - Type of plot to create
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn plot_svg<P: AsRef<Path>>(
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

// Add direct plotting methods to DataFrame
impl DataFrame {
    /// Directly plot a column from this DataFrame with minimal configuration
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional - will use column name if not provided)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::DataFrame;
    ///
    /// let mut df = DataFrame::new();
    /// // Add data to the DataFrame...
    /// df.plot_column("value", "column_plot.png", None).unwrap();
    /// ```
    pub fn plot_column<P: AsRef<Path>>(
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

    /// Directly create a line plot for a column
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn line_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
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

    /// Directly create a scatter plot for a column
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn scatter_plot<P: AsRef<Path>>(
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

    /// Directly create a bar plot for a column
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn bar_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
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

    /// Directly create an area plot for a column
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn area_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
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

    /// Directly create a box plot for a column grouped by a category
    ///
    /// # Arguments
    ///
    /// * `value_column` - Name of the column with values
    /// * `category_column` - Name of the column with categories
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn box_plot<P: AsRef<Path>>(
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

    /// Directly create a scatter plot between two columns
    ///
    /// # Arguments
    ///
    /// * `x_column` - Name of the X-axis column
    /// * `y_column` - Name of the Y-axis column
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn scatter_xy<P: AsRef<Path>>(
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

    /// Directly create a line plot for multiple columns
    ///
    /// # Arguments
    ///
    /// * `columns` - Names of the columns to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn multi_line_plot<P: AsRef<Path>>(
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

    /// Save the plot as SVG format
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the SVG file
    /// * `plot_kind` - Type of plot to create
    /// * `title` - Title of the plot (optional)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn plot_svg<P: AsRef<Path>>(
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

// Add direct plotting methods to OptimizedDataFrame as well
impl OptimizedDataFrame {
    /// Directly plot a column from this OptimizedDataFrame with minimal configuration
    ///
    /// # Arguments
    ///
    /// * `column` - Name of the column to plot
    /// * `path` - Path to save the plot
    /// * `title` - Title of the plot (optional - will use column name if not provided)
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn plot_column<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        // Convert to regular DataFrame first
        let df = standard_dataframe(self)?;
        df.plot_column(column, path, title)
    }

    /// Directly create a line plot for a column
    pub fn line_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.line_plot(column, path, title)
    }

    /// Directly create a scatter plot for a column
    pub fn scatter_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.scatter_plot(column, path, title)
    }

    /// Directly create a bar plot for a column
    pub fn bar_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.bar_plot(column, path, title)
    }

    /// Directly create an area plot for a column
    pub fn area_plot<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.area_plot(column, path, title)
    }

    /// Directly create a box plot for a column grouped by a category
    pub fn box_plot<P: AsRef<Path>>(
        &self,
        value_column: &str,
        category_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.box_plot(value_column, category_column, path, title)
    }

    /// Directly create a scatter plot between two columns
    pub fn scatter_xy<P: AsRef<Path>>(
        &self,
        x_column: &str,
        y_column: &str,
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.scatter_xy(x_column, y_column, path, title)
    }

    /// Directly create a line plot for multiple columns
    pub fn multi_line_plot<P: AsRef<Path>>(
        &self,
        columns: &[&str],
        path: P,
        title: Option<&str>,
    ) -> Result<()> {
        let df = standard_dataframe(self)?;
        df.multi_line_plot(columns, path, title)
    }
}

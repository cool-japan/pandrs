//! Implementation of advanced visualization features using Plotters
//!
//! This module provides Plotters-based visualization functionality for DataFrame and Series.
//! In addition to text-based visualization (textplots), it can generate higher quality graphs and visualizations.

use crate::error::{PandRSError, Result};
use crate::DataFrame;
use crate::Series;
use plotters::prelude::*;
use std::path::Path;

/// Plot types (extended version)
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

/// Plot output formats (extended version)
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    /// PNG image
    PNG,
    /// SVG format
    SVG,
}

/// Extended plot settings
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

/// Extended functionality for Series type
impl<T> Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    /// Output Series as a high-quality graph
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{Series, vis::plotters_ext::{PlotSettings, PlotKind, OutputType}};
    ///
    /// let series = Series::new(vec![1, 2, 3, 4, 5], Some("data".to_string())).expect("operation should succeed");
    /// let settings = PlotSettings {
    ///     title: "My Plot".to_string(),
    ///     plot_kind: PlotKind::Line,
    ///     ..PlotSettings::default()
    /// };
    /// series.plotters_plot("my_plot.png", settings).expect("operation should succeed");
    /// ```
    pub fn plotters_plot<P: AsRef<Path>>(&self, path: P, mut settings: PlotSettings) -> Result<()> {
        let values: Vec<f64> = self.values().iter().map(|v| (*v).into()).collect();
        let indices: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        // Reflect series name in the title (if not set)
        if settings.title == "Plot" {
            if let Some(name) = self.name() {
                settings.title = format!("{} Plot", name);
            }
        }

        // Get series name (for legend)
        let series_name = self
            .name()
            .map_or_else(|| "Series".to_string(), |s| s.clone());

        match settings.output_type {
            OutputType::PNG => plot_series_xy_png(&indices, &values, path, &settings, &series_name),
            OutputType::SVG => plot_series_xy_svg(&indices, &values, path, &settings, &series_name),
        }
    }

    /// Create a histogram from Series
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    /// * `bins` - Number of bins
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{Series, vis::plotters_ext::{PlotSettings, OutputType}};
    ///
    /// let series = Series::new(vec![1, 2, 3, 4, 5, 1, 2, 3, 2, 1], Some("data".to_string())).expect("operation should succeed");
    /// let settings = PlotSettings {
    ///     title: "Histogram".to_string(),
    ///     ..PlotSettings::default()
    /// };
    /// series.plotters_histogram("histogram.png", 5, settings).expect("operation should succeed");
    /// ```
    pub fn plotters_histogram<P: AsRef<Path>>(
        &self,
        path: P,
        bins: usize,
        mut settings: PlotSettings,
    ) -> Result<()> {
        let values: Vec<f64> = self.values().iter().map(|v| (*v).into()).collect();

        // Reflect series name in the title (if not set)
        if settings.title == "Plot" {
            if let Some(name) = self.name() {
                settings.title = format!("{} Histogram", name);
            } else {
                settings.title = "Histogram".to_string();
            }
        }

        // Get series name (for legend)
        let series_name = self
            .name()
            .map_or_else(|| "Series".to_string(), |s| s.clone());

        match settings.output_type {
            OutputType::PNG => plot_histogram_png(&values, bins, path, &settings, &series_name),
            OutputType::SVG => plot_histogram_svg(&values, bins, path, &settings, &series_name),
        }
    }
}

/// Extended functionality for DataFrame - High-quality graphs
impl DataFrame {
    /// Generate plot from DataFrame (specified column)
    ///
    /// # Arguments
    ///
    /// * `col_name` - Name of the column to plot
    /// * `path` - Path to the output file
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// // ... Add data to the DataFrame ...
    /// let settings = PlotSettings {
    ///     title: "Column Plot".to_string(),
    ///     plot_kind: PlotKind::Line,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_plot_column("value", "column_plot.png", settings).expect("operation should succeed");
    /// ```
    pub fn plotters_plot_column<P: AsRef<Path>>(
        &self,
        col_name: &str,
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        // Check if the column exists
        if !self.contains_column(col_name) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                col_name
            )));
        }

        // Get numeric data
        let column: &Series<f64> = self.get_column(col_name)?;
        let values = column.as_f64()?;

        // Values are already f64, no need to handle NA conversion

        let indices: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        // Set title
        if settings.title == "Plot" {
            settings.title = format!("{} Plot", col_name);
        }
        if settings.y_label == "Y" {
            settings.y_label = col_name.to_string();
        }

        // Use column name as legend
        let series_name = col_name.to_string();

        match settings.output_type {
            OutputType::PNG => plot_series_xy_png(&indices, &values, path, &settings, &series_name),
            OutputType::SVG => plot_series_xy_svg(&indices, &values, path, &settings, &series_name),
        }
    }

    /// Generate plot from multiple columns in DataFrame
    ///
    /// # Arguments
    ///
    /// * `col_names` - Names of the columns to plot (multiple)
    /// * `path` - Path to the output file
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// // ... Add data to the DataFrame ...
    /// let settings = PlotSettings {
    ///     title: "Multi Column Plot".to_string(),
    ///     plot_kind: PlotKind::Line,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_plot_columns(&["value1", "value2"], "multi_plot.png", settings).expect("operation should succeed");
    /// ```
    pub fn plotters_plot_columns<P: AsRef<Path>>(
        &self,
        col_names: &[&str],
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        if col_names.is_empty() {
            return Err(PandRSError::Empty(
                "No columns specified for plotting".to_string(),
            ));
        }

        // Check if the columns exist
        for &col_name in col_names.iter() {
            if !self.contains_column(col_name) {
                return Err(PandRSError::Column(format!(
                    "Column '{}' does not exist",
                    col_name
                )));
            }
        }

        // Set title
        if settings.title == "Plot" {
            settings.title = "Multi Column Plot".to_string();
        }

        // Prepare data for each column
        let mut series_data = Vec::new();
        for (i, &col_name) in col_names.iter().enumerate() {
            let column: &Series<f64> = self.get_column(col_name)?;
            let values = column.as_f64()?;

            // Values are already f64, no need to handle NA conversion

            let indices: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

            let color_idx = i % settings.color_palette.len();
            let color = settings.color_palette[color_idx];

            series_data.push((col_name.to_string(), indices, values, color));
        }

        match settings.output_type {
            OutputType::PNG => plot_multi_series_png(series_data, path, &settings),
            OutputType::SVG => plot_multi_series_svg(series_data, path, &settings),
        }
    }

    /// Create XY scatter plot from DataFrame
    ///
    /// # Arguments
    ///
    /// * `x_col` - Name of the X-axis column
    /// * `y_col` - Name of the Y-axis column
    /// * `path` - Path to the output file
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// // ... Add data to the DataFrame ...
    /// let settings = PlotSettings {
    ///     title: "Scatter Plot".to_string(),
    ///     plot_kind: PlotKind::Scatter,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_scatter("x_col", "y_col", "scatter_plot.png", settings).expect("operation should succeed");
    /// ```
    pub fn plotters_scatter<P: AsRef<Path>>(
        &self,
        x_col: &str,
        y_col: &str,
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        // Check if the columns exist
        if !self.contains_column(x_col) {
            return Err(PandRSError::Column(format!(
                "X column '{}' does not exist",
                x_col
            )));
        }
        if !self.contains_column(y_col) {
            return Err(PandRSError::Column(format!(
                "Y column '{}' does not exist",
                y_col
            )));
        }

        // Get numeric data
        let x_column: &Series<f64> = self.get_column(x_col)?;
        let y_column: &Series<f64> = self.get_column(y_col)?;
        let x_values = x_column.as_f64()?;
        let y_values = y_column.as_f64()?;

        // Values are already f64, no need to handle NA conversion

        // Check data length
        if x_values.len() != y_values.len() {
            return Err(PandRSError::Consistency(
                "X and Y column lengths do not match".to_string(),
            ));
        }

        // Set title and labels
        if settings.title == "Plot" {
            settings.title = format!("{} vs {}", y_col, x_col);
        }
        if settings.x_label == "X" {
            settings.x_label = x_col.to_string();
        }
        if settings.y_label == "Y" {
            settings.y_label = y_col.to_string();
        }

        // Use names for legend
        let series_name = format!("{} vs {}", y_col, x_col);

        match settings.output_type {
            OutputType::PNG => {
                plot_series_xy_png(&x_values, &y_values, path, &settings, &series_name)
            }
            OutputType::SVG => {
                plot_series_xy_svg(&x_values, &y_values, path, &settings, &series_name)
            }
        }
    }

    /// Create box plot
    ///
    /// # Arguments
    ///
    /// * `category_col` - Name of the category column
    /// * `value_col` - Name of the value column
    /// * `path` - Path to the output file
    /// * `settings` - Plot settings
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pandrs::{DataFrame, vis::plotters_ext::{PlotSettings, PlotKind}};
    ///
    /// let mut df = DataFrame::new();
    /// // ... Add data to the DataFrame ...
    /// let settings = PlotSettings {
    ///     title: "Box Plot by Category".to_string(),
    ///     plot_kind: PlotKind::BoxPlot,
    ///     ..PlotSettings::default()
    /// };
    /// df.plotters_boxplot("category", "value", "boxplot.png", settings).expect("operation should succeed");
    /// ```
    pub fn plotters_boxplot<P: AsRef<Path>>(
        &self,
        category_col: &str,
        value_col: &str,
        path: P,
        mut settings: PlotSettings,
    ) -> Result<()> {
        // Check if the columns exist
        if !self.contains_column(category_col) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                category_col
            )));
        }
        if !self.contains_column(value_col) {
            return Err(PandRSError::Column(format!(
                "Column '{}' does not exist",
                value_col
            )));
        }

        // Create mapping of categories and their values
        let cat_column: &Series<String> = self.get_column(category_col)?;
        let val_column: &Series<f64> = self.get_column(value_col)?;
        let categories = cat_column
            .values()
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>();
        let numeric_values = val_column.as_f64()?;

        // Values are already f64, no need to handle NA conversion

        // Group values by category
        let mut category_map: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for (cat, val) in categories.iter().zip(numeric_values.iter()) {
            let entry = category_map.entry(cat.clone()).or_insert_with(Vec::new);
            entry.push(*val);
        }

        // Set title and labels
        if settings.title == "Plot" {
            settings.title = format!("{} by {}", value_col, category_col);
        }
        if settings.x_label == "X" {
            settings.x_label = category_col.to_string();
        }
        if settings.y_label == "Y" {
            settings.y_label = value_col.to_string();
        }

        // Box plot implementation
        match settings.output_type {
            OutputType::PNG => plot_boxplot_png(&category_map, path, &settings),
            OutputType::SVG => plot_boxplot_svg(&category_map, path, &settings),
        }
    }
}

/// Box plot implementation in PNG format
///
/// Statistics and the box/whisker/median/outlier drawing are delegated to
/// [`crate::vis::plotters::boxplot_stats`] and
/// [`crate::vis::plotters::draw_boxplot_series`], the same shared
/// implementation used by `crate::vis::plotters::backend`'s box plot, so
/// this no longer carries its own divergent quartile/whisker math.
fn plot_boxplot_png<P: AsRef<Path>>(
    category_map: &std::collections::HashMap<String, Vec<f64>>,
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    let root =
        BitMapBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_boxplot(&root, category_map, settings)
}

/// Box plot implementation in SVG format (see [`plot_boxplot_png`]).
fn plot_boxplot_svg<P: AsRef<Path>>(
    category_map: &std::collections::HashMap<String, Vec<f64>>,
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    let root =
        SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_boxplot(&root, category_map, settings)
}

fn draw_boxplot<DB: plotters::prelude::DrawingBackend>(
    root: &plotters::drawing::DrawingArea<DB, plotters::coord::Shift>,
    category_map: &std::collections::HashMap<String, Vec<f64>>,
    settings: &PlotSettings,
) -> Result<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    // Get and sort list of categories
    let mut categories: Vec<String> = category_map
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(k, _)| k.clone())
        .collect();
    categories.sort();

    if categories.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    let all_stats: Vec<crate::vis::plotters::BoxPlotStats> = categories
        .iter()
        .filter_map(|c| category_map.get(c))
        .filter_map(|v| crate::vis::plotters::boxplot_stats(v))
        .collect();
    if all_stats.is_empty() {
        return Err(PandRSError::Empty("No finite data to plot".to_string()));
    }

    let y_min = all_stats
        .iter()
        .map(|s| s.min.min(s.whisker_low))
        .fold(f64::INFINITY, f64::min);
    let y_max = all_stats
        .iter()
        .map(|s| s.max.max(s.whisker_high))
        .fold(f64::NEG_INFINITY, f64::max);
    let y_range = if (y_max - y_min).abs() < f64::EPSILON {
        1.0
    } else {
        y_max - y_min
    };
    let y_margin = y_range * 0.1;

    // Build chart
    let mut chart = ChartBuilder::on(root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            -0.5f64..(categories.len() as f64 - 0.5),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // Set labels
    chart
        .configure_mesh()
        .x_labels(categories.len())
        .x_label_formatter(&|x| {
            let i = x.round() as isize;
            if i >= 0 && (i as usize) < categories.len() {
                categories[i as usize].clone()
            } else {
                String::new()
            }
        })
        .x_desc(&settings.x_label)
        .y_desc(&settings.y_label)
        .draw()?;

    crate::vis::plotters::draw_boxplot_series(
        &mut chart,
        &categories,
        category_map,
        &settings.color_palette,
    )
}

/// Series XY plot implementation in PNG format
fn plot_series_xy_png<P: AsRef<Path>>(
    x: &[f64],
    y: &[f64],
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "X and Y lengths do not match".to_string(),
        ));
    }

    if x.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    // Calculate min and max values of the data
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Calculate margin
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // Create PNG backend
    let root =
        BitMapBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .disable_mesh()
            .draw()?;
    }

    // Plot data points according to plot type
    let (r, g, b) = settings.color_palette[0];
    let rgb = (r, g, b);
    let color = RGBColor(r, g, b);
    let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)).collect();

    match settings.plot_kind {
        PlotKind::Line => {
            let series = LineSeries::new(points.iter().map(|&(x, y)| (x, y)), color);
            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                });
        }
        PlotKind::Scatter => {
            let series = points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, color.filled()));
            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Circle::new((x + 10, y), 3, RGBColor(rgb.0, rgb.1, rgb.2).filled())
                });
        }
        PlotKind::Bar => {
            let y_baseline = 0.0f64.max(y_min - y_margin);
            let bar_width = if x.len() <= 1 {
                0.5f64
            } else {
                let mut min_diff = f64::INFINITY;
                for i in 1..x.len() {
                    let diff = (x[i] - x[i - 1]).abs();
                    if diff < min_diff {
                        min_diff = diff;
                    }
                }
                min_diff * 0.8
            };

            let series = points.iter().map(|&(x, y)| {
                Rectangle::new(
                    [(x - bar_width / 2.0, y_baseline), (x + bar_width / 2.0, y)],
                    color.filled(),
                )
            });

            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 20, y + 5)],
                        RGBColor(rgb.0, rgb.1, rgb.2).filled(),
                    )
                });
        }
        PlotKind::Area => {
            let baseline = y_min.min(0.0);
            let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);

            let series = AreaSeries::new(points.iter().map(|&(x, y)| (x, y)), baseline, area_color);

            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                });
        }
        _ => {
            return Err(PandRSError::NotImplemented(
                "The specified plot type is not supported by this function".to_string(),
            ));
        }
    }

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// Series XY plot implementation in SVG format
fn plot_series_xy_svg<P: AsRef<Path>>(
    x: &[f64],
    y: &[f64],
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "X and Y lengths do not match".to_string(),
        ));
    }

    if x.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    // Calculate min and max values of the data
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Calculate margin
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // Create SVG backend
    let root =
        SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .disable_mesh()
            .draw()?;
    }

    // Plot data points according to plot type
    let (r, g, b) = settings.color_palette[0];
    let rgb = (r, g, b);
    let color = RGBColor(r, g, b);
    let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)).collect();

    match settings.plot_kind {
        PlotKind::Line => {
            let series = LineSeries::new(points.iter().map(|&(x, y)| (x, y)), color);
            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                });
        }
        PlotKind::Scatter => {
            let series = points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 3, color.filled()));
            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Circle::new((x + 10, y), 3, RGBColor(rgb.0, rgb.1, rgb.2).filled())
                });
        }
        PlotKind::Bar => {
            let y_baseline = 0.0f64.max(y_min - y_margin);
            let bar_width = if x.len() <= 1 {
                0.5f64
            } else {
                let mut min_diff = f64::INFINITY;
                for i in 1..x.len() {
                    let diff = (x[i] - x[i - 1]).abs();
                    if diff < min_diff {
                        min_diff = diff;
                    }
                }
                min_diff * 0.8
            };

            let series = points.iter().map(|&(x, y)| {
                Rectangle::new(
                    [(x - bar_width / 2.0, y_baseline), (x + bar_width / 2.0, y)],
                    color.filled(),
                )
            });

            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    Rectangle::new(
                        [(x, y - 5), (x + 20, y + 5)],
                        RGBColor(rgb.0, rgb.1, rgb.2).filled(),
                    )
                });
        }
        PlotKind::Area => {
            let baseline = y_min.min(0.0);
            let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);

            let series = AreaSeries::new(points.iter().map(|&(x, y)| (x, y)), baseline, area_color);

            chart
                .draw_series(series)?
                .label(series_name.to_owned())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                });
        }
        _ => {
            return Err(PandRSError::NotImplemented(
                "The specified plot type is not supported by this function".to_string(),
            ));
        }
    }

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// PNG histogram implementation
fn plot_histogram_png<P: AsRef<Path>>(
    values: &[f64],
    bins: usize,
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if values.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    if bins == 0 {
        return Err(PandRSError::InvalidInput(
            "Histogram: bins must be greater than 0".to_string(),
        ));
    }

    // Calculate histogram bins
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // A constant (or single-point) series has max_val == min_val, which
    // would otherwise divide by a zero bin_width; fall back to a single
    // unit-width bin holding every value instead.
    let bin_width = if (max_val - min_val).abs() < f64::EPSILON {
        1.0
    } else {
        (max_val - min_val) / (bins as f64)
    };
    let mut histogram = vec![0; bins];

    for &val in values {
        let bin_idx = ((val - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        histogram[bin_idx] += 1;
    }

    // Find maximum frequency
    let max_freq = *histogram.iter().max().unwrap_or(&0);

    // Create PNG backend
    let root =
        BitMapBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            min_val..(max_val + bin_width * 0.1),
            0.0..((max_freq as f64) * 1.1),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .disable_mesh()
            .draw()?;
    }

    // Draw histogram
    let (r, g, b) = settings.color_palette[0];
    let rgb = (r, g, b);
    let color = RGBColor(r, g, b);

    let bars = histogram.iter().enumerate().map(|(i, &count)| {
        let x0 = min_val + (i as f64) * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0.0), (x1, count as f64)], color.mix(0.7).filled())
    });

    chart
        .draw_series(bars)?
        .label(series_name.to_owned())
        .legend(move |(x, y)| {
            Rectangle::new(
                [(x, y - 5), (x + 20, y + 5)],
                RGBColor(rgb.0, rgb.1, rgb.2).mix(0.7).filled(),
            )
        });

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// SVG histogram implementation
fn plot_histogram_svg<P: AsRef<Path>>(
    values: &[f64],
    bins: usize,
    path: P,
    settings: &PlotSettings,
    series_name: &str,
) -> Result<()> {
    if values.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }
    if bins == 0 {
        return Err(PandRSError::InvalidInput(
            "Histogram: bins must be greater than 0".to_string(),
        ));
    }

    // Calculate histogram bins
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // See plot_histogram_png: a constant series has max_val == min_val,
    // which would otherwise divide by a zero bin_width.
    let bin_width = if (max_val - min_val).abs() < f64::EPSILON {
        1.0
    } else {
        (max_val - min_val) / (bins as f64)
    };
    let mut histogram = vec![0; bins];

    for &val in values {
        let bin_idx = ((val - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        histogram[bin_idx] += 1;
    }

    // Find maximum frequency
    let max_freq = *histogram.iter().max().unwrap_or(&0);

    // Create SVG backend
    let root =
        SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            min_val..(max_val + bin_width * 0.1),
            0.0..((max_freq as f64) * 1.1),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc("Frequency")
            .disable_mesh()
            .draw()?;
    }

    // Draw histogram
    let (r, g, b) = settings.color_palette[0];
    let rgb = (r, g, b);
    let color = RGBColor(r, g, b);

    let bars = histogram.iter().enumerate().map(|(i, &count)| {
        let x0 = min_val + (i as f64) * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0.0), (x1, count as f64)], color.mix(0.7).filled())
    });

    chart
        .draw_series(bars)?
        .label(series_name.to_owned())
        .legend(move |(x, y)| {
            Rectangle::new(
                [(x, y - 5), (x + 20, y + 5)],
                RGBColor(rgb.0, rgb.1, rgb.2).mix(0.7).filled(),
            )
        });

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// Multi-series PNG plot implementation
fn plot_multi_series_png<P: AsRef<Path>>(
    series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>,
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    if series_data.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    // Calculate min and max values of all data
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (_, x, y, _) in &series_data {
        if x.is_empty() {
            continue;
        }

        let x_min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min_val = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max_val = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        x_min = x_min.min(x_min_val);
        x_max = x_max.max(x_max_val);
        y_min = y_min.min(y_min_val);
        y_max = y_max.max(y_max_val);
    }

    // Calculate margin
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // Create PNG backend
    let root =
        BitMapBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .disable_mesh()
            .draw()?;
    }

    // Plot each series according to plot type
    for (name, x, y, rgb) in series_data {
        let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)).collect();
        let color = RGBColor(rgb.0, rgb.1, rgb.2);

        match settings.plot_kind {
            PlotKind::Line => {
                let series = LineSeries::new(points.iter().map(|&(x, y)| (x, y)), color);
                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                    });
            }
            PlotKind::Scatter => {
                let series = points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, color.filled()));
                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        Circle::new((x + 10, y), 3, RGBColor(rgb.0, rgb.1, rgb.2).filled())
                    });
            }
            PlotKind::Bar => {
                return Err(PandRSError::NotImplemented(
                    "Multi-series bar charts are not yet supported".to_string(),
                ));
            }
            PlotKind::Area => {
                let baseline = y_min.min(0.0);
                let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);

                let series =
                    AreaSeries::new(points.iter().map(|&(x, y)| (x, y)), baseline, area_color);

                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                    });
            }
            _ => {
                return Err(PandRSError::NotImplemented(
                    "The specified plot type is not supported by this function".to_string(),
                ));
            }
        }
    }

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

/// Multi-series SVG plot implementation
fn plot_multi_series_svg<P: AsRef<Path>>(
    series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>,
    path: P,
    settings: &PlotSettings,
) -> Result<()> {
    if series_data.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    // Calculate min and max values of all data
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for (_, x, y, _) in &series_data {
        if x.is_empty() {
            continue;
        }

        let x_min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min_val = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max_val = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        x_min = x_min.min(x_min_val);
        x_max = x_max.max(x_max_val);
        y_min = y_min.min(y_min_val);
        y_max = y_max.max(y_max_val);
    }

    // Calculate margin
    let x_margin = (x_max - x_min) * 0.05;
    let y_margin = (y_max - y_min) * 0.05;

    // Create SVG backend
    let root =
        SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(&settings.title, ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (x_min - x_margin)..(x_max + x_margin),
            (y_min - y_margin)..(y_max + y_margin),
        )?;

    // Add grid lines
    if settings.show_grid {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .draw()?;
    } else {
        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .x_desc(&settings.x_label)
            .y_desc(&settings.y_label)
            .disable_mesh()
            .draw()?;
    }

    // Plot each series according to plot type
    for (name, x, y, rgb) in series_data {
        let points: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(x, y)| (*x, *y)).collect();
        let color = RGBColor(rgb.0, rgb.1, rgb.2);

        match settings.plot_kind {
            PlotKind::Line => {
                let series = LineSeries::new(points.iter().map(|&(x, y)| (x, y)), color);
                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                    });
            }
            PlotKind::Scatter => {
                let series = points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 3, color.filled()));
                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        Circle::new((x + 10, y), 3, RGBColor(rgb.0, rgb.1, rgb.2).filled())
                    });
            }
            PlotKind::Bar => {
                return Err(PandRSError::NotImplemented(
                    "Multi-series bar charts are not yet supported".to_string(),
                ));
            }
            PlotKind::Area => {
                let baseline = y_min.min(0.0);
                let area_color = RGBColor(rgb.0, rgb.1, rgb.2).mix(0.2);

                let series =
                    AreaSeries::new(points.iter().map(|&(x, y)| (x, y)), baseline, area_color);

                chart
                    .draw_series(series)?
                    .label(name)
                    .legend(move |(x, y)| {
                        PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(rgb.0, rgb.1, rgb.2))
                    });
            }
            _ => {
                return Err(PandRSError::NotImplemented(
                    "The specified plot type is not supported by this function".to_string(),
                ));
            }
        }
    }

    // Show legend
    if settings.show_legend {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root.present()?;
    Ok(())
}

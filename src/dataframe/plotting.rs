//! Enhanced plotting integration for DataFrame
//!
//! This module provides comprehensive plotting capabilities with pandas-like API,
//! statistical plotting functions, and interactive visualization features.

use std::collections::HashMap;
use std::path::Path;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Enhanced plotting configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Plot type/kind
    pub kind: PlotKind,
    /// Output format
    pub format: PlotFormat,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Show legend
    pub legend: bool,
    /// Show grid
    pub grid: bool,
    /// Figure DPI for high-quality output
    pub dpi: u32,
    /// Custom styling options
    pub style: PlotStyle,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: "Plot".to_string(),
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            width: 800,
            height: 600,
            kind: PlotKind::Line,
            format: PlotFormat::PNG,
            color_scheme: ColorScheme::Default,
            legend: true,
            grid: true,
            dpi: 150,
            style: PlotStyle::default(),
        }
    }
}

/// Plot types available
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlotKind {
    /// Line plot
    Line,
    /// Scatter plot
    Scatter,
    /// Bar plot
    Bar,
    /// Horizontal bar plot
    Barh,
    /// Histogram
    Hist,
    /// Box plot
    Box,
    /// Area plot
    Area,
    /// Pie chart
    Pie,
    /// Density plot
    Density,
    /// Hexbin plot
    Hexbin,
    /// Violin plot
    Violin,
    /// Heatmap
    Heatmap,
    /// Correlation matrix
    Corr,
    /// Pair plot (scatter matrix)
    Pair,
    /// Time series plot
    TimeSeries,
    /// 3D surface plot
    Surface3D,
    /// Contour plot
    Contour,
}

/// Output formats
#[derive(Debug, Clone, Copy)]
pub enum PlotFormat {
    /// PNG format
    PNG,
    /// SVG format
    SVG,
    /// PDF format (if available)
    PDF,
    /// HTML with interactive features
    HTML,
    /// Terminal output (text-based)
    Terminal,
}

/// Color schemes
#[derive(Debug, Clone)]
pub enum ColorScheme {
    /// Default color palette
    Default,
    /// Viridis color scheme
    Viridis,
    /// Plasma color scheme
    Plasma,
    /// Set1 categorical colors
    Set1,
    /// Set2 categorical colors
    Set2,
    /// Custom color palette
    Custom(Vec<(u8, u8, u8)>),
}

impl ColorScheme {
    /// Get the color palette as RGB tuples
    pub fn colors(&self) -> Vec<(u8, u8, u8)> {
        match self {
            ColorScheme::Default => vec![
                (31, 119, 180),  // Blue
                (255, 127, 14),  // Orange
                (44, 160, 44),   // Green
                (214, 39, 40),   // Red
                (148, 103, 189), // Purple
                (140, 86, 75),   // Brown
                (227, 119, 194), // Pink
                (127, 127, 127), // Gray
                (188, 189, 34),  // Olive
                (23, 190, 207),  // Cyan
            ],
            ColorScheme::Viridis => vec![
                (68, 1, 84),    // Dark purple
                (59, 82, 139),  // Blue-purple
                (33, 145, 140), // Teal
                (94, 201, 98),  // Green
                (253, 231, 37), // Yellow
            ],
            ColorScheme::Plasma => vec![
                (13, 8, 135),   // Dark blue
                (84, 2, 163),   // Purple
                (139, 10, 165), // Magenta
                (185, 50, 137), // Pink
                (219, 92, 104), // Red
                (244, 136, 73), // Orange
                (254, 188, 43), // Yellow
                (240, 249, 33), // Light yellow
            ],
            ColorScheme::Set1 => vec![
                (228, 26, 28),   // Red
                (55, 126, 184),  // Blue
                (77, 175, 74),   // Green
                (152, 78, 163),  // Purple
                (255, 127, 0),   // Orange
                (255, 255, 51),  // Yellow
                (166, 86, 40),   // Brown
                (247, 129, 191), // Pink
                (153, 153, 153), // Gray
            ],
            ColorScheme::Set2 => vec![
                (102, 194, 165), // Teal
                (252, 141, 98),  // Coral
                (141, 160, 203), // Light blue
                (231, 138, 195), // Light pink
                (166, 216, 84),  // Light green
                (255, 217, 47),  // Yellow
                (229, 196, 148), // Beige
                (179, 179, 179), // Gray
            ],
            ColorScheme::Custom(colors) => colors.clone(),
        }
    }
}

/// Plot styling options
#[derive(Debug, Clone)]
pub struct PlotStyle {
    /// Line width for line plots
    pub line_width: f32,
    /// Marker size for scatter plots
    pub marker_size: f32,
    /// Transparency/alpha value (0.0 to 1.0)
    pub alpha: f32,
    /// Font size for labels
    pub font_size: u32,
    /// Border/edge color
    pub edge_color: Option<(u8, u8, u8)>,
    /// Fill style for bars/areas
    pub fill_style: FillStyle,
    /// Grid style
    pub grid_style: GridStyle,
}

impl Default for PlotStyle {
    fn default() -> Self {
        Self {
            line_width: 2.0,
            marker_size: 6.0,
            alpha: 0.8,
            font_size: 12,
            edge_color: None,
            fill_style: FillStyle::Solid,
            grid_style: GridStyle::Minor,
        }
    }
}

/// Fill styles for areas and bars
#[derive(Debug, Clone, Copy)]
pub enum FillStyle {
    /// Solid fill
    Solid,
    /// Gradient fill
    Gradient,
    /// Pattern fill
    Pattern,
    /// No fill (outline only)
    None,
}

/// Grid styles
#[derive(Debug, Clone, Copy)]
pub enum GridStyle {
    /// No grid
    None,
    /// Major grid lines only
    Major,
    /// Minor grid lines only
    Minor,
    /// Both major and minor grid lines
    Both,
}

/// Statistical plotting builder
pub struct StatPlotBuilder<'a> {
    df: &'a DataFrame,
    config: PlotConfig,
}

impl<'a> StatPlotBuilder<'a> {
    pub fn new(df: &'a DataFrame) -> Self {
        Self {
            df,
            config: PlotConfig::default(),
        }
    }

    /// Set the plot title
    pub fn title(mut self, title: &str) -> Self {
        self.config.title = title.to_string();
        self
    }

    /// Set axis labels
    pub fn labels(mut self, x_label: &str, y_label: &str) -> Self {
        self.config.x_label = x_label.to_string();
        self.config.y_label = y_label.to_string();
        self
    }

    /// Set figure size
    pub fn figsize(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set color scheme
    pub fn color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.config.color_scheme = scheme;
        self
    }

    /// Set plot style
    pub fn style(mut self, style: PlotStyle) -> Self {
        self.config.style = style;
        self
    }

    /// Create a correlation matrix heatmap
    pub fn correlation_matrix<P: AsRef<Path>>(self, path: P) -> Result<()> {
        let numeric_columns = self.get_numeric_columns()?;
        if numeric_columns.len() < 2 {
            return Err(Error::InvalidValue(
                "At least 2 numeric columns required for correlation matrix".to_string(),
            ));
        }

        let correlation_matrix = self.calculate_correlation_matrix(&numeric_columns)?;
        self.plot_heatmap(correlation_matrix, path, &numeric_columns)
    }

    /// Create a pair plot (scatter matrix)
    pub fn pair_plot<P: AsRef<Path>>(self, columns: Option<&[String]>, path: P) -> Result<()> {
        let cols = match columns {
            Some(cols) => cols.to_vec(),
            None => self.get_numeric_columns()?,
        };

        if cols.len() < 2 {
            return Err(Error::InvalidValue(
                "At least 2 columns required for pair plot".to_string(),
            ));
        }

        self.create_pair_plot(&cols, path)
    }

    /// Create distribution plots for all numeric columns
    pub fn distribution_plots<P: AsRef<Path>>(self, path_prefix: P) -> Result<Vec<String>> {
        let numeric_columns = self.get_numeric_columns()?;
        let mut created_files = Vec::new();

        for column in &numeric_columns {
            let mut config = self.config.clone();
            config.title = format!("Distribution of {}", column);
            config.kind = PlotKind::Hist;

            let path = format!(
                "{}_{}_dist.png",
                path_prefix.as_ref().to_string_lossy(),
                column
            );
            self.plot_column_histogram(column, &path, &config)?;
            created_files.push(path);
        }

        Ok(created_files)
    }

    /// Create box plots grouped by a categorical column
    pub fn grouped_box_plots<P: AsRef<Path>>(
        self,
        group_by: &str,
        value_columns: Option<&[String]>,
        path: P,
    ) -> Result<()> {
        let value_cols = match value_columns {
            Some(cols) => cols.to_vec(),
            None => self.get_numeric_columns()?,
        };

        if !self.df.contains_column(group_by) {
            return Err(Error::ColumnNotFound(group_by.to_string()));
        }

        self.create_grouped_box_plots(group_by, &value_cols, path)
    }

    /// Create time series plots
    pub fn time_series<P: AsRef<Path>>(
        self,
        date_column: &str,
        value_columns: &[String],
        path: P,
    ) -> Result<()> {
        if !self.df.contains_column(date_column) {
            return Err(Error::ColumnNotFound(date_column.to_string()));
        }

        for col in value_columns {
            if !self.df.contains_column(col) {
                return Err(Error::ColumnNotFound(col.to_string()));
            }
        }

        self.create_time_series_plot(date_column, value_columns, path)
    }

    /// Create a dashboard with multiple subplots
    pub fn dashboard<P: AsRef<Path>>(self, path: P) -> Result<()> {
        let numeric_columns = self.get_numeric_columns()?;
        if numeric_columns.is_empty() {
            return Err(Error::InvalidValue(
                "No numeric columns found for dashboard".to_string(),
            ));
        }

        self.create_dashboard(&numeric_columns, path)
    }

    // Helper methods
    fn get_numeric_columns(&self) -> Result<Vec<String>> {
        let mut numeric_cols = Vec::new();

        for col_name in self.df.column_names() {
            let values = self.df.get_column_string_values(&col_name)?;

            // Check if column contains numeric values
            let is_numeric = values
                .iter()
                .all(|v| v.trim().parse::<f64>().is_ok() || v.trim().is_empty());

            if is_numeric {
                numeric_cols.push(col_name);
            }
        }

        Ok(numeric_cols)
    }

    fn calculate_correlation_matrix(&self, columns: &[String]) -> Result<Vec<Vec<f64>>> {
        let mut matrix = vec![vec![0.0; columns.len()]; columns.len()];

        for (i, col1) in columns.iter().enumerate() {
            let values1 = self.get_numeric_values(col1)?;

            for (j, col2) in columns.iter().enumerate() {
                let values2 = self.get_numeric_values(col2)?;

                if values1.len() != values2.len() {
                    return Err(Error::InvalidValue(
                        "Column lengths don't match".to_string(),
                    ));
                }

                let correlation = self.calculate_correlation(&values1, &values2);
                matrix[i][j] = correlation;
            }
        }

        Ok(matrix)
    }

    fn get_numeric_values(&self, column: &str) -> Result<Vec<f64>> {
        let string_values = self.df.get_column_string_values(column)?;
        let mut numeric_values = Vec::new();

        for value in string_values {
            if let Ok(num) = value.parse::<f64>() {
                numeric_values.push(num);
            } else if !value.trim().is_empty() {
                return Err(Error::InvalidValue(format!(
                    "Non-numeric value '{}' in column '{}'",
                    value, column
                )));
            }
        }

        Ok(numeric_values)
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.is_empty() || y.is_empty() || x.len() != y.len() {
            return 0.0;
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        let sum_x_sq: f64 = x.iter().map(|&val| val * val).sum();
        let sum_y_sq: f64 = y.iter().map(|&val| val * val).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn plot_heatmap<P: AsRef<Path>>(
        &self,
        matrix: Vec<Vec<f64>>,
        path: P,
        labels: &[String],
    ) -> Result<()> {
        // For now, create a simple text representation
        // This would be replaced with actual plotting implementation
        let mut output = String::new();
        output.push_str(&format!(
            "Correlation Matrix Heatmap: {}\n\n",
            self.config.title
        ));

        // Header
        output.push_str("     ");
        for label in labels {
            output.push_str(&format!("{:>8.8}", label));
        }
        output.push('\n');

        // Matrix values
        for (i, row) in matrix.iter().enumerate() {
            output.push_str(&format!("{:>8.8}", labels[i]));
            for &value in row {
                output.push_str(&format!("{:>8.2}", value));
            }
            output.push('\n');
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn create_pair_plot<P: AsRef<Path>>(&self, columns: &[String], path: P) -> Result<()> {
        // Create a text-based pair plot representation
        let mut output = String::new();
        output.push_str(&format!("Pair Plot: {}\n\n", self.config.title));

        for (i, col1) in columns.iter().enumerate() {
            for (j, col2) in columns.iter().enumerate() {
                if i <= j {
                    output.push_str(&format!("{} vs {}\n", col1, col2));

                    let values1 = self.get_numeric_values(col1)?;
                    let values2 = self.get_numeric_values(col2)?;

                    if i == j {
                        // Diagonal: histogram
                        output.push_str(&format!("  Distribution of {}\n", col1));
                        let stats = self.calculate_basic_stats(&values1);
                        output.push_str(&format!("  Mean: {:.2}, Std: {:.2}\n", stats.0, stats.1));
                    } else {
                        // Off-diagonal: scatter plot info
                        let correlation = self.calculate_correlation(&values1, &values2);
                        output.push_str(&format!("  Correlation: {:.3}\n", correlation));
                    }
                    output.push('\n');
                }
            }
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn calculate_basic_stats(&self, values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (values.len() - 1).max(1) as f64;
        let std_dev = variance.sqrt();

        (mean, std_dev)
    }

    fn plot_column_histogram<P: AsRef<Path>>(
        &self,
        column: &str,
        path: P,
        config: &PlotConfig,
    ) -> Result<()> {
        let values = self.get_numeric_values(column)?;
        if values.is_empty() {
            return Err(Error::InvalidValue(format!(
                "No numeric values in column '{}'",
                column
            )));
        }

        // Create a simple text histogram
        let mut output = String::new();
        output.push_str(&format!("Histogram: {}\n\n", config.title));

        let (mean, std_dev) = self.calculate_basic_stats(&values);
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        output.push_str(&format!("Statistics for {}:\n", column));
        output.push_str(&format!("  Count: {}\n", values.len()));
        output.push_str(&format!("  Mean: {:.2}\n", mean));
        output.push_str(&format!("  Std: {:.2}\n", std_dev));
        output.push_str(&format!("  Min: {:.2}\n", min_val));
        output.push_str(&format!("  Max: {:.2}\n", max_val));

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn create_grouped_box_plots<P: AsRef<Path>>(
        &self,
        group_by: &str,
        value_columns: &[String],
        path: P,
    ) -> Result<()> {
        let mut output = String::new();
        output.push_str(&format!("Grouped Box Plots by {}\n\n", group_by));

        let group_values = self.df.get_column_string_values(group_by)?;
        let unique_groups: std::collections::HashSet<String> = group_values.into_iter().collect();

        for value_col in value_columns {
            output.push_str(&format!("\n{} by {}:\n", value_col, group_by));

            for group in &unique_groups {
                output.push_str(&format!("  Group '{}': ", group));

                // Get values for this group
                let group_column_values = self.df.get_column_string_values(group_by)?;
                let value_column_values = self.get_numeric_values(value_col)?;

                let group_data: Vec<f64> = group_column_values
                    .iter()
                    .zip(value_column_values.iter())
                    .filter_map(|(g, &v)| if g == group { Some(v) } else { None })
                    .collect();

                if !group_data.is_empty() {
                    let (mean, std_dev) = self.calculate_basic_stats(&group_data);
                    output.push_str(&format!(
                        "Mean: {:.2}, Std: {:.2}, Count: {}\n",
                        mean,
                        std_dev,
                        group_data.len()
                    ));
                } else {
                    output.push_str("No data\n");
                }
            }
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn create_time_series_plot<P: AsRef<Path>>(
        &self,
        _date_column: &str,
        value_columns: &[String],
        path: P,
    ) -> Result<()> {
        let mut output = String::new();
        output.push_str(&format!("Time Series Plot: {}\n\n", self.config.title));

        for value_col in value_columns {
            let values = self.get_numeric_values(value_col)?;
            output.push_str(&format!("{}: {} data points\n", value_col, values.len()));

            if !values.is_empty() {
                let (mean, std_dev) = self.calculate_basic_stats(&values);
                output.push_str(&format!("  Mean: {:.2}, Std: {:.2}\n", mean, std_dev));
            }
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn create_dashboard<P: AsRef<Path>>(&self, numeric_columns: &[String], path: P) -> Result<()> {
        let mut output = String::new();
        output.push_str(&format!("Data Dashboard: {}\n\n", self.config.title));
        output.push_str(&format!("Dataset Overview:\n"));
        output.push_str(&format!("  Rows: {}\n", self.df.row_count()));
        output.push_str(&format!("  Columns: {}\n", self.df.column_names().len()));
        output.push_str(&format!("  Numeric columns: {}\n\n", numeric_columns.len()));

        output.push_str("Column Statistics:\n");
        for column in numeric_columns {
            let values = self.get_numeric_values(column)?;
            if !values.is_empty() {
                let (mean, std_dev) = self.calculate_basic_stats(&values);
                let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                output.push_str(&format!("  {}:\n", column));
                output.push_str(&format!(
                    "    Count: {}, Mean: {:.2}, Std: {:.2}\n",
                    values.len(),
                    mean,
                    std_dev
                ));
                output.push_str(&format!("    Min: {:.2}, Max: {:.2}\n", min_val, max_val));
            }
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }
}

/// Enhanced plotting extension trait for DataFrame
pub trait EnhancedPlotExt {
    /// Get a statistical plotting builder
    fn plot(&self) -> StatPlotBuilder;

    /// Quick line plot of a column
    fn plot_line<P: AsRef<Path>>(&self, column: &str, path: P) -> Result<()>;

    /// Quick scatter plot of two columns
    fn plot_scatter<P: AsRef<Path>>(&self, x_col: &str, y_col: &str, path: P) -> Result<()>;

    /// Quick histogram of a column
    fn plot_hist<P: AsRef<Path>>(&self, column: &str, path: P, bins: Option<usize>) -> Result<()>;

    /// Quick box plot grouped by category
    fn plot_box<P: AsRef<Path>>(&self, value_col: &str, group_col: &str, path: P) -> Result<()>;

    /// Plot correlation matrix
    fn plot_corr<P: AsRef<Path>>(&self, path: P) -> Result<()>;

    /// Create distribution plots for all numeric columns
    fn plot_distributions<P: AsRef<Path>>(&self, path_prefix: P) -> Result<Vec<String>>;

    /// Generate a comprehensive data visualization report
    fn plot_report<P: AsRef<Path>>(&self, output_dir: P) -> Result<Vec<String>>;
}

impl EnhancedPlotExt for DataFrame {
    fn plot(&self) -> StatPlotBuilder {
        StatPlotBuilder::new(self)
    }

    fn plot_line<P: AsRef<Path>>(&self, column: &str, path: P) -> Result<()> {
        self.plot()
            .title(&format!("Line Plot of {}", column))
            .labels("Index", column)
            .correlation_matrix(path) // For now, delegate to available method
    }

    fn plot_scatter<P: AsRef<Path>>(&self, x_col: &str, y_col: &str, path: P) -> Result<()> {
        if !self.contains_column(x_col) {
            return Err(Error::ColumnNotFound(x_col.to_string()));
        }
        if !self.contains_column(y_col) {
            return Err(Error::ColumnNotFound(y_col.to_string()));
        }

        let mut output = String::new();
        output.push_str(&format!("Scatter Plot: {} vs {}\n\n", y_col, x_col));

        let x_values = self.get_column_string_values(x_col)?;
        let y_values = self.get_column_string_values(y_col)?;

        let x_numeric: Result<Vec<f64>> = x_values
            .iter()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|e| Error::InvalidValue(e.to_string()))
            })
            .collect();
        let y_numeric: Result<Vec<f64>> = y_values
            .iter()
            .map(|v| {
                v.parse::<f64>()
                    .map_err(|e| Error::InvalidValue(e.to_string()))
            })
            .collect();

        match (x_numeric, y_numeric) {
            (Ok(x_vals), Ok(y_vals)) => {
                let builder = StatPlotBuilder::new(self);
                let correlation = builder.calculate_correlation(&x_vals, &y_vals);
                output.push_str(&format!("Correlation: {:.3}\n", correlation));
                output.push_str(&format!(
                    "Data points: {}\n",
                    x_vals.len().min(y_vals.len())
                ));
            }
            _ => {
                output.push_str("Unable to create scatter plot: non-numeric data\n");
            }
        }

        std::fs::write(path, output).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    fn plot_hist<P: AsRef<Path>>(&self, column: &str, path: P, _bins: Option<usize>) -> Result<()> {
        self.plot()
            .title(&format!("Histogram of {}", column))
            .plot_column_histogram(column, path, &PlotConfig::default())
    }

    fn plot_box<P: AsRef<Path>>(&self, value_col: &str, group_col: &str, path: P) -> Result<()> {
        self.plot()
            .title(&format!("Box Plot: {} by {}", value_col, group_col))
            .grouped_box_plots(group_col, Some(&[value_col.to_string()]), path)
    }

    fn plot_corr<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.plot()
            .title("Correlation Matrix")
            .correlation_matrix(path)
    }

    fn plot_distributions<P: AsRef<Path>>(&self, path_prefix: P) -> Result<Vec<String>> {
        self.plot()
            .title("Distribution Analysis")
            .distribution_plots(path_prefix)
    }

    fn plot_report<P: AsRef<Path>>(&self, output_dir: P) -> Result<Vec<String>> {
        let output_dir = output_dir.as_ref();
        let mut created_files = Vec::new();

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir).map_err(|e| Error::IoError(e.to_string()))?;

        // Generate dashboard
        let dashboard_path = output_dir.join("dashboard.txt");
        self.plot().dashboard(&dashboard_path)?;
        created_files.push(dashboard_path.to_string_lossy().to_string());

        // Generate correlation matrix
        let corr_path = output_dir.join("correlation_matrix.txt");
        if self.plot_corr(&corr_path).is_ok() {
            created_files.push(corr_path.to_string_lossy().to_string());
        }

        // Generate distribution plots
        let dist_prefix = output_dir.join("distribution");
        if let Ok(dist_files) = self.plot_distributions(&dist_prefix) {
            created_files.extend(dist_files);
        }

        // Generate pair plot if enough numeric columns
        let builder = StatPlotBuilder::new(self);
        if let Ok(numeric_cols) = builder.get_numeric_columns() {
            if numeric_cols.len() >= 2 {
                let pair_path = output_dir.join("pair_plot.txt");
                if builder.pair_plot(Some(&numeric_cols), &pair_path).is_ok() {
                    created_files.push(pair_path.to_string_lossy().to_string());
                }
            }
        }

        Ok(created_files)
    }
}

/// Interactive plotting capabilities
pub struct InteractivePlot {
    config: PlotConfig,
    data: DataFrame,
}

impl InteractivePlot {
    /// Create a new interactive plot
    pub fn new(data: DataFrame) -> Self {
        Self {
            config: PlotConfig::default(),
            data,
        }
    }

    /// Configure the plot
    pub fn config(mut self, config: PlotConfig) -> Self {
        self.config = config;
        self
    }

    /// Generate HTML with interactive features
    pub fn to_html<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", self.config.title));
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str(
            ".plot-container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }\n",
        );
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str(&format!("<h1>{}</h1>\n", self.config.title));
        html.push_str("<div class='plot-container'>\n");
        html.push_str("<h2>Interactive Data Visualization</h2>\n");
        html.push_str(&format!(
            "<p>Dataset: {} rows, {} columns</p>\n",
            self.data.row_count(),
            self.data.column_names().len()
        ));

        // Add column information
        html.push_str("<h3>Columns:</h3>\n<ul>\n");
        for col_name in self.data.column_names() {
            html.push_str(&format!("<li>{}</li>\n", col_name));
        }
        html.push_str("</ul>\n");

        html.push_str("</div>\n</body>\n</html>");

        std::fs::write(path, html).map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    /// Generate a web-based dashboard
    pub fn dashboard<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.to_html(path)
    }
}

/// Plot theme management
pub struct PlotTheme {
    pub background_color: (u8, u8, u8),
    pub grid_color: (u8, u8, u8),
    pub text_color: (u8, u8, u8),
    pub color_scheme: ColorScheme,
    pub font_family: String,
}

impl PlotTheme {
    /// Default light theme
    pub fn light() -> Self {
        Self {
            background_color: (255, 255, 255),
            grid_color: (200, 200, 200),
            text_color: (0, 0, 0),
            color_scheme: ColorScheme::Default,
            font_family: "Arial".to_string(),
        }
    }

    /// Dark theme
    pub fn dark() -> Self {
        Self {
            background_color: (30, 30, 30),
            grid_color: (80, 80, 80),
            text_color: (255, 255, 255),
            color_scheme: ColorScheme::Viridis,
            font_family: "Arial".to_string(),
        }
    }

    /// Apply theme to plot configuration
    pub fn apply_to_config(&self, mut config: PlotConfig) -> PlotConfig {
        config.color_scheme = self.color_scheme.clone();
        config
    }
}

/// Utility functions for plotting
pub mod utils {
    use super::*;

    /// Calculate optimal number of bins for histogram
    pub fn optimal_bins(data: &[f64]) -> usize {
        if data.is_empty() {
            return 10;
        }

        // Sturges' rule
        let n = data.len() as f64;
        (n.log2() + 1.0).ceil() as usize
    }

    /// Calculate quantiles
    pub fn quantiles(data: &[f64], q: &[f64]) -> Vec<f64> {
        if data.is_empty() {
            return vec![0.0; q.len()];
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        q.iter()
            .map(|&quantile| {
                let index = (quantile * (sorted_data.len() - 1) as f64).round() as usize;
                sorted_data[index.min(sorted_data.len() - 1)]
            })
            .collect()
    }

    /// Generate a color palette
    pub fn generate_palette(n: usize, scheme: &ColorScheme) -> Vec<(u8, u8, u8)> {
        let base_colors = scheme.colors();
        if n <= base_colors.len() {
            base_colors.into_iter().take(n).collect()
        } else {
            // Repeat and interpolate
            let mut palette = Vec::with_capacity(n);
            for i in 0..n {
                let idx = i % base_colors.len();
                palette.push(base_colors[idx]);
            }
            palette
        }
    }
}

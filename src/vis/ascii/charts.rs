//! Chart implementations for text-based visualization

use super::{Chart, ChartConfig, ChartStyle};

// ============================================================================
// Histogram
// ============================================================================

/// Configuration for histogram
#[derive(Debug, Clone)]
pub struct HistogramConfig {
    /// Base chart config
    pub base: ChartConfig,
    /// Chart style
    pub style: ChartStyle,
    /// Number of bins
    pub bins: usize,
    /// Show bin counts
    pub show_counts: bool,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            base: ChartConfig::default(),
            style: ChartStyle::Unicode,
            bins: 10,
            show_counts: true,
        }
    }
}

/// Histogram chart for distribution visualization
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin edges
    bin_edges: Vec<f64>,
    /// Bin counts
    counts: Vec<usize>,
    /// Configuration
    config: HistogramConfig,
}

impl Histogram {
    /// Create a new histogram from data
    pub fn new(data: &[f64], bins: usize) -> Self {
        let config = HistogramConfig {
            bins,
            ..Default::default()
        };
        Self::with_config(data, config)
    }

    /// Create histogram with custom configuration
    pub fn with_config(data: &[f64], config: HistogramConfig) -> Self {
        let (bin_edges, counts) = Self::compute_bins(data, config.bins);
        Self {
            bin_edges,
            counts,
            config,
        }
    }

    fn compute_bins(data: &[f64], bins: usize) -> (Vec<f64>, Vec<usize>) {
        if data.is_empty() || bins == 0 {
            return (vec![], vec![]);
        }

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < f64::EPSILON {
            return (vec![min, max], vec![data.len()]);
        }

        let bin_width = (max - min) / bins as f64;
        let mut edges = Vec::with_capacity(bins + 1);
        let mut counts = vec![0; bins];

        for i in 0..=bins {
            edges.push(min + i as f64 * bin_width);
        }

        for &value in data {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1;
        }

        (edges, counts)
    }

    fn get_bar_char(&self) -> char {
        match self.config.style {
            ChartStyle::Ascii => '#',
            ChartStyle::Unicode | ChartStyle::Braille => '█',
        }
    }

    fn get_partial_chars(&self) -> &[char] {
        match self.config.style {
            ChartStyle::Ascii => &['#'],
            ChartStyle::Unicode | ChartStyle::Braille => &['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'],
        }
    }
}

impl Chart for Histogram {
    fn render(&self) -> String {
        if self.counts.is_empty() {
            return String::from("No data to display");
        }

        let mut output = String::new();
        let max_count = *self.counts.iter().max().unwrap_or(&1);
        let bar_width = self.config.base.width.saturating_sub(15);
        let bar_char = self.get_bar_char();

        // Title
        if let Some(ref title) = self.config.base.title {
            output.push_str(&format!(
                "{:^width$}\n\n",
                title,
                width = self.config.base.width
            ));
        }

        // Render bars
        for (i, &count) in self.counts.iter().enumerate() {
            let bar_len = if max_count > 0 {
                (count as f64 / max_count as f64 * bar_width as f64).round() as usize
            } else {
                0
            };

            let bar: String = std::iter::repeat(bar_char).take(bar_len).collect();
            let edge_start = self.bin_edges[i];
            let edge_end = self.bin_edges[i + 1];

            if self.config.show_counts {
                output.push_str(&format!(
                    "{:>6.1}-{:<6.1} │{:<width$}│ {}\n",
                    edge_start,
                    edge_end,
                    bar,
                    count,
                    width = bar_width
                ));
            } else {
                output.push_str(&format!(
                    "{:>6.1}-{:<6.1} │{:<width$}│\n",
                    edge_start,
                    edge_end,
                    bar,
                    width = bar_width
                ));
            }
        }

        output
    }
}

// ============================================================================
// Bar Chart
// ============================================================================

/// Bar orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarOrientation {
    /// Vertical bars
    Vertical,
    /// Horizontal bars
    Horizontal,
}

impl Default for BarOrientation {
    fn default() -> Self {
        Self::Horizontal
    }
}

/// Configuration for bar chart
#[derive(Debug, Clone)]
pub struct BarChartConfig {
    /// Base chart config
    pub base: ChartConfig,
    /// Chart style
    pub style: ChartStyle,
    /// Bar orientation
    pub orientation: BarOrientation,
    /// Show values on bars
    pub show_values: bool,
    /// Max label width
    pub label_width: usize,
}

impl Default for BarChartConfig {
    fn default() -> Self {
        Self {
            base: ChartConfig::default(),
            style: ChartStyle::Unicode,
            orientation: BarOrientation::Horizontal,
            show_values: true,
            label_width: 12,
        }
    }
}

/// Bar chart for categorical data
#[derive(Debug, Clone)]
pub struct BarChart {
    /// Labels for each bar
    labels: Vec<String>,
    /// Values for each bar
    values: Vec<f64>,
    /// Configuration
    config: BarChartConfig,
}

impl BarChart {
    /// Create a new bar chart
    pub fn new(labels: &[&str], values: &[f64]) -> Self {
        let config = BarChartConfig::default();
        Self::with_config(labels, values, config)
    }

    /// Create a horizontal bar chart
    pub fn horizontal(labels: &[&str], values: &[f64]) -> Self {
        let config = BarChartConfig {
            orientation: BarOrientation::Horizontal,
            ..Default::default()
        };
        Self::with_config(labels, values, config)
    }

    /// Create a vertical bar chart
    pub fn vertical(labels: &[&str], values: &[f64]) -> Self {
        let config = BarChartConfig {
            orientation: BarOrientation::Vertical,
            ..Default::default()
        };
        Self::with_config(labels, values, config)
    }

    /// Create with custom configuration
    pub fn with_config(labels: &[&str], values: &[f64], config: BarChartConfig) -> Self {
        Self {
            labels: labels.iter().map(|s| s.to_string()).collect(),
            values: values.to_vec(),
            config,
        }
    }

    fn get_bar_char(&self) -> char {
        match self.config.style {
            ChartStyle::Ascii => '#',
            ChartStyle::Unicode | ChartStyle::Braille => '█',
        }
    }
}

impl Chart for BarChart {
    fn render(&self) -> String {
        if self.values.is_empty() {
            return String::from("No data to display");
        }

        match self.config.orientation {
            BarOrientation::Horizontal => self.render_horizontal(),
            BarOrientation::Vertical => self.render_vertical(),
        }
    }
}

impl BarChart {
    fn render_horizontal(&self) -> String {
        let mut output = String::new();
        let max_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let bar_width = self
            .config
            .base
            .width
            .saturating_sub(self.config.label_width + 10);
        let bar_char = self.get_bar_char();

        // Title
        if let Some(ref title) = self.config.base.title {
            output.push_str(&format!(
                "{:^width$}\n\n",
                title,
                width = self.config.base.width
            ));
        }

        for (label, &value) in self.labels.iter().zip(self.values.iter()) {
            let bar_len = if max_val > 0.0 {
                (value / max_val * bar_width as f64).round() as usize
            } else {
                0
            };

            let bar: String = std::iter::repeat(bar_char).take(bar_len).collect();
            let truncated_label: String = label.chars().take(self.config.label_width).collect();

            if self.config.show_values {
                output.push_str(&format!(
                    "{:>label_width$} │{:<bar_width$}│ {:.2}\n",
                    truncated_label,
                    bar,
                    value,
                    label_width = self.config.label_width,
                    bar_width = bar_width
                ));
            } else {
                output.push_str(&format!(
                    "{:>label_width$} │{:<bar_width$}│\n",
                    truncated_label,
                    bar,
                    label_width = self.config.label_width,
                    bar_width = bar_width
                ));
            }
        }

        output
    }

    fn render_vertical(&self) -> String {
        let mut output = String::new();
        let max_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let height = self.config.base.height.min(20);
        let bar_char = self.get_bar_char();

        // Title
        if let Some(ref title) = self.config.base.title {
            output.push_str(&format!(
                "{:^width$}\n\n",
                title,
                width = self.labels.len() * 4
            ));
        }

        // Normalize values to height
        let normalized: Vec<usize> = self
            .values
            .iter()
            .map(|&v| {
                if max_val > 0.0 {
                    (v / max_val * height as f64).round() as usize
                } else {
                    0
                }
            })
            .collect();

        // Render from top to bottom
        for row in (0..height).rev() {
            for &bar_height in &normalized {
                if bar_height > row {
                    output.push_str(&format!(" {} ", bar_char));
                } else {
                    output.push_str("   ");
                }
            }
            output.push('\n');
        }

        // Baseline
        for _ in 0..self.labels.len() {
            output.push_str("───");
        }
        output.push('\n');

        // Labels (abbreviated)
        for label in &self.labels {
            let abbrev: String = label.chars().take(2).collect();
            output.push_str(&format!(" {} ", abbrev));
        }
        output.push('\n');

        output
    }
}

// ============================================================================
// Line Plot
// ============================================================================

/// Configuration for line plot
#[derive(Debug, Clone)]
pub struct LinePlotConfig {
    /// Base chart config
    pub base: ChartConfig,
    /// Chart style
    pub style: ChartStyle,
    /// Show data points
    pub show_points: bool,
    /// Point character
    pub point_char: char,
}

impl Default for LinePlotConfig {
    fn default() -> Self {
        Self {
            base: ChartConfig {
                height: 10,
                ..Default::default()
            },
            style: ChartStyle::Unicode,
            show_points: true,
            point_char: '●',
        }
    }
}

/// Line plot for time series or sequential data
#[derive(Debug, Clone)]
pub struct LinePlot {
    /// Data values
    values: Vec<f64>,
    /// Configuration
    config: LinePlotConfig,
}

impl LinePlot {
    /// Create a new line plot
    pub fn new(values: &[f64]) -> Self {
        Self::with_config(values, LinePlotConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(values: &[f64], config: LinePlotConfig) -> Self {
        Self {
            values: values.to_vec(),
            config,
        }
    }
}

impl Chart for LinePlot {
    fn render(&self) -> String {
        if self.values.is_empty() {
            return String::from("No data to display");
        }

        let mut output = String::new();
        let height = self.config.base.height;
        let width = self.config.base.width.min(self.values.len());

        let min_val = self.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let range = if (max_val - min_val).abs() < f64::EPSILON {
            1.0
        } else {
            max_val - min_val
        };

        // Title
        if let Some(ref title) = self.config.base.title {
            output.push_str(&format!("{:^width$}\n\n", title, width = width));
        }

        // Sample data to fit width
        let step = self.values.len() as f64 / width as f64;
        let sampled: Vec<usize> = (0..width)
            .map(|i| {
                let idx = (i as f64 * step).floor() as usize;
                let val = self.values[idx.min(self.values.len() - 1)];
                ((val - min_val) / range * (height - 1) as f64).round() as usize
            })
            .collect();

        // Render from top to bottom
        for row in (0..height).rev() {
            // Y-axis label
            if self.config.base.show_labels {
                let y_val = min_val + (row as f64 / (height - 1) as f64) * range;
                output.push_str(&format!("{:>6.1} │", y_val));
            }

            for &y in &sampled {
                if y == row {
                    output.push(self.config.point_char);
                } else {
                    output.push(' ');
                }
            }
            output.push('\n');
        }

        // X-axis
        if self.config.base.show_labels {
            output.push_str("       └");
            for _ in 0..width {
                output.push('─');
            }
            output.push('\n');
        }

        output
    }
}

// ============================================================================
// Scatter Plot
// ============================================================================

/// Configuration for scatter plot
#[derive(Debug, Clone)]
pub struct ScatterPlotConfig {
    /// Base chart config
    pub base: ChartConfig,
    /// Chart style
    pub style: ChartStyle,
    /// Point character
    pub point_char: char,
}

impl Default for ScatterPlotConfig {
    fn default() -> Self {
        Self {
            base: ChartConfig {
                height: 15,
                width: 40,
                ..Default::default()
            },
            style: ChartStyle::Unicode,
            point_char: '●',
        }
    }
}

/// Scatter plot for two-dimensional data
#[derive(Debug, Clone)]
pub struct ScatterPlot {
    /// X values
    x: Vec<f64>,
    /// Y values
    y: Vec<f64>,
    /// Configuration
    config: ScatterPlotConfig,
}

impl ScatterPlot {
    /// Create a new scatter plot
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        Self::with_config(x, y, ScatterPlotConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(x: &[f64], y: &[f64], config: ScatterPlotConfig) -> Self {
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            config,
        }
    }
}

impl Chart for ScatterPlot {
    fn render(&self) -> String {
        if self.x.is_empty() || self.y.is_empty() {
            return String::from("No data to display");
        }

        let len = self.x.len().min(self.y.len());
        let height = self.config.base.height;
        let width = self.config.base.width;

        let x_min = self.x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = self.x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = self.y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = if (x_max - x_min).abs() < f64::EPSILON {
            1.0
        } else {
            x_max - x_min
        };
        let y_range = if (y_max - y_min).abs() < f64::EPSILON {
            1.0
        } else {
            y_max - y_min
        };

        // Create grid
        let mut grid = vec![vec![' '; width]; height];

        // Plot points
        for i in 0..len {
            let px = ((self.x[i] - x_min) / x_range * (width - 1) as f64).round() as usize;
            let py = ((self.y[i] - y_min) / y_range * (height - 1) as f64).round() as usize;
            let px = px.min(width - 1);
            let py = py.min(height - 1);
            grid[py][px] = self.config.point_char;
        }

        // Render
        let mut output = String::new();

        // Title
        if let Some(ref title) = self.config.base.title {
            output.push_str(&format!("{:^width$}\n\n", title, width = width + 8));
        }

        for row in (0..height).rev() {
            if self.config.base.show_labels {
                let y_val = y_min + (row as f64 / (height - 1) as f64) * y_range;
                output.push_str(&format!("{:>6.1} │", y_val));
            }
            for col in 0..width {
                output.push(grid[row][col]);
            }
            output.push('\n');
        }

        // X-axis
        if self.config.base.show_labels {
            output.push_str("       └");
            for _ in 0..width {
                output.push('─');
            }
            output.push('\n');
            output.push_str(&format!(
                "        {:<width$.1}{:>8.1}\n",
                x_min,
                x_max,
                width = width - 8
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_creation() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let hist = Histogram::new(&data, 5);
        let output = hist.render();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_histogram_empty() {
        let data: Vec<f64> = vec![];
        let hist = Histogram::new(&data, 5);
        let output = hist.render();
        assert!(output.contains("No data"));
    }

    #[test]
    fn test_bar_chart_horizontal() {
        let labels = vec!["A", "B", "C"];
        let values = vec![10.0, 20.0, 15.0];
        let chart = BarChart::horizontal(&labels, &values);
        let output = chart.render();
        assert!(output.contains("A"));
        assert!(output.contains("B"));
        assert!(output.contains("C"));
    }

    #[test]
    fn test_bar_chart_vertical() {
        let labels = vec!["A", "B", "C"];
        let values = vec![10.0, 20.0, 15.0];
        let chart = BarChart::vertical(&labels, &values);
        let output = chart.render();
        assert!(!output.is_empty());
    }

    #[test]
    fn test_line_plot() {
        let data = vec![1.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0];
        let plot = LinePlot::new(&data);
        let output = plot.render();
        assert!(!output.is_empty());
        assert!(output.contains('●'));
    }

    #[test]
    fn test_scatter_plot() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        let plot = ScatterPlot::new(&x, &y);
        let output = plot.render();
        assert!(!output.is_empty());
        assert!(output.contains('●'));
    }

    #[test]
    fn test_bar_chart_with_title() {
        let labels = vec!["A", "B"];
        let values = vec![10.0, 20.0];
        let config = BarChartConfig {
            base: ChartConfig {
                title: Some("Test Chart".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let chart = BarChart::with_config(&labels, &values, config);
        let output = chart.render();
        assert!(output.contains("Test Chart"));
    }
}

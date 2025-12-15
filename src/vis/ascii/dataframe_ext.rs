//! DataFrame visualization extension trait
//!
//! Provides quick visualization methods for DataFrame columns using ASCII charts.

use super::{BarChart, Chart, Histogram, LinePlot, ScatterPlot, Sparkline};

/// Extension trait for DataFrame-like structures to add quick visualization
pub trait DataFrameVizExt {
    /// Get numeric column data for visualization
    fn get_numeric_column(&self, name: &str) -> Option<Vec<f64>>;

    /// Get all numeric column names
    fn numeric_columns(&self) -> Vec<String>;

    /// Get string column data for labels
    fn get_string_column(&self, name: &str) -> Option<Vec<String>>;

    /// Create a histogram of a numeric column
    fn histogram(&self, column: &str, bins: usize) -> Option<String> {
        self.get_numeric_column(column)
            .map(|data| Histogram::new(&data, bins).render())
    }

    /// Create a sparkline of a numeric column
    fn sparkline(&self, column: &str) -> Option<String> {
        self.get_numeric_column(column)
            .map(|data| Sparkline::new(&data).to_string_with_stats())
    }

    /// Create a line plot of a numeric column
    fn line_plot(&self, column: &str) -> Option<String> {
        self.get_numeric_column(column)
            .map(|data| LinePlot::new(&data).render())
    }

    /// Create a scatter plot of two numeric columns
    fn scatter_plot(&self, x_column: &str, y_column: &str) -> Option<String> {
        let x = self.get_numeric_column(x_column)?;
        let y = self.get_numeric_column(y_column)?;
        Some(ScatterPlot::new(&x, &y).render())
    }

    /// Create a bar chart with string labels and numeric values
    fn bar_chart(&self, label_column: &str, value_column: &str) -> Option<String> {
        let labels = self.get_string_column(label_column)?;
        let values = self.get_numeric_column(value_column)?;
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        Some(BarChart::horizontal(&label_refs, &values).render())
    }

    /// Generate a quick summary visualization of all numeric columns
    fn summary_sparklines(&self) -> String {
        let columns = self.numeric_columns();
        if columns.is_empty() {
            return String::from("No numeric columns to visualize");
        }

        let max_name_len = columns.iter().map(|n| n.len()).max().unwrap_or(10);
        let mut output = String::new();
        output.push_str("=== Column Summary ===\n\n");

        for col_name in columns {
            if let Some(data) = self.get_numeric_column(&col_name) {
                let spark = Sparkline::new(&data);
                output.push_str(&format!(
                    "{:>width$}: {}\n",
                    col_name,
                    spark.to_string_with_stats(),
                    width = max_name_len
                ));
            }
        }

        output
    }
}

/// Quick visualization helper functions that work with raw data
pub mod quick {
    use super::*;

    /// Create histogram from numeric slice
    pub fn histogram(data: &[f64], bins: usize) -> String {
        Histogram::new(data, bins).render()
    }

    /// Create sparkline from numeric slice
    pub fn sparkline(data: &[f64]) -> String {
        Sparkline::new(data).render()
    }

    /// Create sparkline with statistics
    pub fn sparkline_stats(data: &[f64]) -> String {
        Sparkline::new(data).to_string_with_stats()
    }

    /// Create line plot from numeric slice
    pub fn line_plot(data: &[f64]) -> String {
        LinePlot::new(data).render()
    }

    /// Create scatter plot from two numeric slices
    pub fn scatter(x: &[f64], y: &[f64]) -> String {
        ScatterPlot::new(x, y).render()
    }

    /// Create horizontal bar chart from labels and values
    pub fn bar_chart(labels: &[&str], values: &[f64]) -> String {
        BarChart::horizontal(labels, values).render()
    }

    /// Create vertical bar chart from labels and values
    pub fn vbar_chart(labels: &[&str], values: &[f64]) -> String {
        BarChart::vertical(labels, values).render()
    }

    /// Generate summary statistics string
    pub fn summary(data: &[f64]) -> String {
        if data.is_empty() {
            return String::from("No data");
        }

        let n = data.len();
        let sum: f64 = data.iter().sum();
        let mean = sum / n as f64;
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Variance and std dev
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Sorted for percentiles
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p25 = sorted[(n as f64 * 0.25) as usize];
        let p50 = sorted[(n as f64 * 0.50) as usize];
        let p75 = sorted[(n as f64 * 0.75) as usize];

        format!(
            "Count: {}\nMean:  {:.4}\nStd:   {:.4}\nMin:   {:.4}\n25%:   {:.4}\n50%:   {:.4}\n75%:   {:.4}\nMax:   {:.4}",
            n, mean, std_dev, min, p25, p50, p75, max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_histogram() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0];
        let result = quick::histogram(&data, 5);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_sparkline() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = quick::sparkline(&data);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_sparkline_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = quick::sparkline_stats(&data);
        assert!(result.contains("min:"));
        assert!(result.contains("max:"));
    }

    #[test]
    fn test_quick_line_plot() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = quick::line_plot(&data);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_scatter() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        let result = quick::scatter(&x, &y);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_bar_chart() {
        let labels = vec!["A", "B", "C"];
        let values = vec![10.0, 20.0, 15.0];
        let result = quick::bar_chart(&labels, &values);
        assert!(result.contains("A"));
        assert!(result.contains("B"));
    }

    #[test]
    fn test_quick_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = quick::summary(&data);
        assert!(result.contains("Count: 10"));
        assert!(result.contains("Mean:"));
        assert!(result.contains("Std:"));
    }

    #[test]
    fn test_quick_summary_empty() {
        let data: Vec<f64> = vec![];
        let result = quick::summary(&data);
        assert_eq!(result, "No data");
    }
}

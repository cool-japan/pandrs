//! DataFrame integration for SVG visualization
//!
//! Implements `SvgVisualize` for `DataFrame`, enabling direct plotting methods.

use std::io::Write as IoWrite;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::vis::svg::charts::{
    BarChart, BarOrientation, HeatMap, LineChart, LineSeries, PieChart, ScatterPlot,
    SvgChartConfig, SvgHistogram,
};

/// Plot type selector for save_svg / save_html convenience methods
#[derive(Debug, Clone)]
pub enum SvgPlotType {
    Bar,
    BarHorizontal,
    Line {
        x_col: String,
        y_cols: Vec<String>,
    },
    Scatter {
        x_col: String,
        y_col: String,
    },
    Histogram {
        col: String,
        bins: usize,
    },
    Heatmap,
    Pie {
        label_col: String,
        value_col: String,
    },
}

/// Extension trait providing SVG plotting capabilities on DataFrame
pub trait SvgVisualize {
    /// Plot a bar chart from label and value columns, returns SVG string
    fn plot_bar_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Plot a horizontal bar chart from label and value columns
    fn plot_bar_horizontal_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Plot a line chart for one or more numeric y columns against an x column
    fn plot_line_svg(
        &self,
        x_col: &str,
        y_cols: &[&str],
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Plot a scatter plot of two numeric columns
    fn plot_scatter_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Plot a histogram of a numeric column
    fn plot_histogram_svg(
        &self,
        col: &str,
        bins: usize,
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Plot a 2D heatmap of all numeric columns (rows = row index, columns = column names)
    fn plot_heatmap_svg(&self, config: Option<SvgChartConfig>) -> Result<String>;

    /// Plot a pie chart using label_col for labels and value_col for slice sizes
    fn plot_pie_svg(
        &self,
        label_col: &str,
        value_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String>;

    /// Save SVG output to a file
    fn save_svg(
        &self,
        path: &str,
        plot_type: SvgPlotType,
        config: Option<SvgChartConfig>,
    ) -> Result<()>;

    /// Save HTML (with embedded SVG) to a file
    fn save_html(
        &self,
        path: &str,
        plot_type: SvgPlotType,
        config: Option<SvgChartConfig>,
    ) -> Result<()>;
}

// ============================================================================
// Helper: extract numeric f64 data from DataFrame column
// ============================================================================

fn extract_f64_column(df: &DataFrame, col: &str) -> Result<Vec<f64>> {
    // Try i64 first, then f64, then f32
    if let Ok(series) = df.get_column::<i64>(col) {
        let values: Vec<f64> = series.values().iter().map(|v| *v as f64).collect();
        return Ok(values);
    }
    if let Ok(series) = df.get_column::<f64>(col) {
        let values: Vec<f64> = series.values().to_vec();
        return Ok(values);
    }
    if let Ok(series) = df.get_column::<f32>(col) {
        let values: Vec<f64> = series.values().iter().map(|v| *v as f64).collect();
        return Ok(values);
    }
    Err(Error::ColumnNotFound(format!(
        "Column '{}' not found or not a numeric type",
        col
    )))
}

fn extract_string_column(df: &DataFrame, col: &str) -> Result<Vec<String>> {
    if let Ok(series) = df.get_column::<String>(col) {
        let values: Vec<String> = series.values().to_vec();
        return Ok(values);
    }
    // Fall back: try to stringify numeric columns
    if let Ok(values) = extract_f64_column(df, col) {
        return Ok(values.into_iter().map(|v| format!("{}", v)).collect());
    }
    Err(Error::ColumnNotFound(format!("Column '{}' not found", col)))
}

// ============================================================================
// Impl
// ============================================================================

impl SvgVisualize for DataFrame {
    fn plot_bar_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let labels = extract_string_column(self, x_col)?;
        let values = extract_f64_column(self, y_col)?;
        if labels.len() != values.len() {
            return Err(Error::LengthMismatch {
                expected: labels.len(),
                actual: values.len(),
            });
        }
        let cfg = config.unwrap_or_default();
        let chart = BarChart::new(labels, values, BarOrientation::Vertical, cfg);
        chart.render()
    }

    fn plot_bar_horizontal_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let labels = extract_string_column(self, x_col)?;
        let values = extract_f64_column(self, y_col)?;
        if labels.len() != values.len() {
            return Err(Error::LengthMismatch {
                expected: labels.len(),
                actual: values.len(),
            });
        }
        let cfg = config.unwrap_or_default();
        let chart = BarChart::new(labels, values, BarOrientation::Horizontal, cfg);
        chart.render()
    }

    fn plot_line_svg(
        &self,
        x_col: &str,
        y_cols: &[&str],
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let x_values = extract_f64_column(self, x_col)?;
        if y_cols.is_empty() {
            return Err(Error::InvalidInput(
                "plot_line_svg: y_cols must not be empty".to_string(),
            ));
        }
        let mut series_vec = Vec::with_capacity(y_cols.len());
        for &col in y_cols {
            let values = extract_f64_column(self, col)?;
            series_vec.push(LineSeries::new(col, values));
        }
        let cfg = config.unwrap_or_default();
        let chart = LineChart::new(x_values, series_vec, cfg);
        chart.render()
    }

    fn plot_scatter_svg(
        &self,
        x_col: &str,
        y_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let x_values = extract_f64_column(self, x_col)?;
        let y_values = extract_f64_column(self, y_col)?;
        let cfg = config.unwrap_or_default();
        let chart = ScatterPlot::new(x_values, y_values, cfg);
        chart.render()
    }

    fn plot_histogram_svg(
        &self,
        col: &str,
        bins: usize,
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let data = extract_f64_column(self, col)?;
        let cfg = config.unwrap_or_default();
        let chart = SvgHistogram::new(data, bins, cfg);
        chart.render()
    }

    fn plot_heatmap_svg(&self, config: Option<SvgChartConfig>) -> Result<String> {
        let col_names = self.column_names();
        // Collect all numeric columns
        let numeric_cols: Vec<String> = col_names
            .iter()
            .filter(|c| extract_f64_column(self, c).is_ok())
            .cloned()
            .collect();
        if numeric_cols.is_empty() {
            return Err(Error::EmptyData(
                "plot_heatmap_svg: no numeric columns".to_string(),
            ));
        }
        let nrows = self.row_count();
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            data.push(Vec::with_capacity(numeric_cols.len()));
        }
        for col in &numeric_cols {
            let values = extract_f64_column(self, col)?;
            for (r, v) in values.into_iter().enumerate() {
                if r < data.len() {
                    data[r].push(v);
                }
            }
        }
        let row_labels: Vec<String> = (0..nrows).map(|i| i.to_string()).collect();
        let cfg = config.unwrap_or_default();
        let chart = HeatMap::new(data, row_labels, numeric_cols, cfg);
        chart.render()
    }

    fn plot_pie_svg(
        &self,
        label_col: &str,
        value_col: &str,
        config: Option<SvgChartConfig>,
    ) -> Result<String> {
        let labels = extract_string_column(self, label_col)?;
        let values = extract_f64_column(self, value_col)?;
        let cfg = config.unwrap_or_default();
        let chart = PieChart::new(labels, values, cfg);
        chart.render()
    }

    fn save_svg(
        &self,
        path: &str,
        plot_type: SvgPlotType,
        config: Option<SvgChartConfig>,
    ) -> Result<()> {
        let svg = generate_svg(self, plot_type, config)?;
        write_to_file(path, svg.as_bytes())
    }

    fn save_html(
        &self,
        path: &str,
        plot_type: SvgPlotType,
        config: Option<SvgChartConfig>,
    ) -> Result<()> {
        let svg = generate_svg(self, plot_type.clone(), config.clone())?;
        let title = "PandRS Chart";
        let html = format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ margin: 0; padding: 20px; background: #f5f5f5; font-family: Arial, sans-serif; }}
  .chart-container {{ background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); display: inline-block; padding: 10px; }}
</style>
</head>
<body>
<div class="chart-container">
{svg}
</div>
</body>
</html>"#
        );
        write_to_file(path, html.as_bytes())
    }
}

fn generate_svg(
    df: &DataFrame,
    plot_type: SvgPlotType,
    config: Option<SvgChartConfig>,
) -> Result<String> {
    match plot_type {
        SvgPlotType::Bar => {
            // For Bar without specified columns, use first string col as labels, first numeric as values
            let cols = df.column_names();
            let label_col = cols
                .first()
                .ok_or_else(|| Error::EmptyData("DataFrame is empty".to_string()))?
                .clone();
            let value_col = cols
                .get(1)
                .ok_or_else(|| Error::EmptyData("DataFrame needs at least 2 columns".to_string()))?
                .clone();
            df.plot_bar_svg(&label_col, &value_col, config)
        }
        SvgPlotType::BarHorizontal => {
            let cols = df.column_names();
            let label_col = cols
                .first()
                .ok_or_else(|| Error::EmptyData("DataFrame is empty".to_string()))?
                .clone();
            let value_col = cols
                .get(1)
                .ok_or_else(|| Error::EmptyData("DataFrame needs at least 2 columns".to_string()))?
                .clone();
            df.plot_bar_horizontal_svg(&label_col, &value_col, config)
        }
        SvgPlotType::Line { x_col, y_cols } => {
            let y_refs: Vec<&str> = y_cols.iter().map(|s| s.as_str()).collect();
            df.plot_line_svg(&x_col, &y_refs, config)
        }
        SvgPlotType::Scatter { x_col, y_col } => df.plot_scatter_svg(&x_col, &y_col, config),
        SvgPlotType::Histogram { col, bins } => df.plot_histogram_svg(&col, bins, config),
        SvgPlotType::Heatmap => df.plot_heatmap_svg(config),
        SvgPlotType::Pie {
            label_col,
            value_col,
        } => df.plot_pie_svg(&label_col, &value_col, config),
    }
}

fn write_to_file(path: &str, content: &[u8]) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .map_err(|e| Error::IoError(format!("Failed to create file '{}': {}", path, e)))?;
    file.write_all(content)
        .map_err(|e| Error::IoError(format!("Failed to write file '{}': {}", path, e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;

    fn make_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "label".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string(), "C".to_string()],
                Some("label".to_string()),
            )
            .expect("series"),
        )
        .expect("add column");
        df.add_column(
            "value".to_string(),
            Series::new(vec![10i64, 25, 15], Some("value".to_string())).expect("series"),
        )
        .expect("add column");
        df
    }

    #[test]
    fn test_plot_bar_svg() {
        let df = make_test_df();
        let svg = df.plot_bar_svg("label", "value", None).expect("bar svg");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_plot_scatter_svg() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0f64, 2.0, 3.0, 4.0], None).expect("series"),
        )
        .expect("add column");
        df.add_column(
            "y".to_string(),
            Series::new(vec![2.0f64, 4.0, 1.0, 3.0], None).expect("series"),
        )
        .expect("add column");
        let svg = df.plot_scatter_svg("x", "y", None).expect("scatter svg");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_plot_histogram_svg() {
        let mut df = DataFrame::new();
        df.add_column(
            "data".to_string(),
            Series::new(vec![1.0f64, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0], None).expect("series"),
        )
        .expect("add column");
        let svg = df
            .plot_histogram_svg("data", 5, None)
            .expect("histogram svg");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_plot_line_svg() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![0.0f64, 1.0, 2.0, 3.0], None).expect("series"),
        )
        .expect("add column");
        df.add_column(
            "y1".to_string(),
            Series::new(vec![1.0f64, 3.0, 2.0, 4.0], None).expect("series"),
        )
        .expect("add column");
        let svg = df.plot_line_svg("x", &["y1"], None).expect("line svg");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_save_svg_to_temp() {
        let df = make_test_df();
        let mut path = std::env::temp_dir();
        path.push("pandrs_test_bar.svg");
        let path_str = path.to_str().expect("path str").to_string();
        df.save_svg(&path_str, SvgPlotType::Bar, None)
            .expect("save svg");
        let content = std::fs::read_to_string(&path_str).expect("read file");
        assert!(content.contains("<svg"));
        // Cleanup
        let _ = std::fs::remove_file(&path_str);
    }

    #[test]
    fn test_save_html_to_temp() {
        let df = make_test_df();
        let mut path = std::env::temp_dir();
        path.push("pandrs_test_bar.html");
        let path_str = path.to_str().expect("path str").to_string();
        df.save_html(&path_str, SvgPlotType::Bar, None)
            .expect("save html");
        let content = std::fs::read_to_string(&path_str).expect("read file");
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("<svg"));
        // Cleanup
        let _ = std::fs::remove_file(&path_str);
    }
}

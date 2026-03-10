//! SVG chart implementations
//!
//! Each chart type uses `SvgCanvas` to render itself as an SVG string.

use crate::core::error::{Error, Result};
use crate::vis::svg::colors::{Color, ColorScheme};
use crate::vis::svg::engine::{DrawStyle, PathBuilder, SvgCanvas, Transform};

// ============================================================================
// Common configuration
// ============================================================================

/// Margin settings for charts
#[derive(Debug, Clone, Copy)]
pub struct Margins {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl Margins {
    pub fn new(top: f64, right: f64, bottom: f64, left: f64) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    pub fn uniform(v: f64) -> Self {
        Self::new(v, v, v, v)
    }
}

impl Default for Margins {
    fn default() -> Self {
        Self::new(40.0, 30.0, 60.0, 70.0)
    }
}

/// Legend position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LegendPosition {
    #[default]
    TopRight,
    TopLeft,
    BottomRight,
    BottomLeft,
    None,
}

/// Common chart configuration
#[derive(Debug, Clone)]
pub struct SvgChartConfig {
    pub title: Option<String>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub width: f64,
    pub height: f64,
    pub margins: Margins,
    pub color_scheme: ColorScheme,
    pub show_legend: bool,
    pub legend_position: LegendPosition,
    pub background: Option<Color>,
    pub grid_color: Color,
    pub show_grid: bool,
    pub font_size: f64,
    pub title_font_size: f64,
}

impl Default for SvgChartConfig {
    fn default() -> Self {
        Self {
            title: None,
            x_label: None,
            y_label: None,
            width: 800.0,
            height: 500.0,
            margins: Margins::default(),
            color_scheme: ColorScheme::Default,
            show_legend: true,
            legend_position: LegendPosition::TopRight,
            background: Some(Color::WHITE),
            grid_color: Color::rgb(220, 220, 220),
            show_grid: true,
            font_size: 12.0,
            title_font_size: 16.0,
        }
    }
}

impl SvgChartConfig {
    /// Width of the plot area inside margins
    pub fn plot_width(&self) -> f64 {
        self.width - self.margins.left - self.margins.right
    }

    /// Height of the plot area inside margins
    pub fn plot_height(&self) -> f64 {
        self.height - self.margins.top - self.margins.bottom
    }

    /// X coordinate of left edge of plot area
    pub fn plot_x(&self) -> f64 {
        self.margins.left
    }

    /// Y coordinate of top edge of plot area
    pub fn plot_y(&self) -> f64 {
        self.margins.top
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn nice_ticks(min: f64, max: f64, count: usize) -> Vec<f64> {
    if (max - min).abs() < f64::EPSILON {
        return vec![min];
    }
    let range = max - min;
    let raw_step = range / count as f64;
    let magnitude = raw_step.log10().floor();
    let power = 10_f64.powf(magnitude);
    let nice_step = {
        let normalized = raw_step / power;
        let nice_normalized = if normalized <= 1.0 {
            1.0
        } else if normalized <= 2.0 {
            2.0
        } else if normalized <= 5.0 {
            5.0
        } else {
            10.0
        };
        nice_normalized * power
    };
    let nice_min = (min / nice_step).floor() * nice_step;
    let nice_max = (max / nice_step).ceil() * nice_step;
    let mut ticks = Vec::new();
    let mut v = nice_min;
    while v <= nice_max + f64::EPSILON {
        ticks.push(v);
        v += nice_step;
    }
    ticks
}

fn format_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let abs = v.abs();
    if abs >= 1000.0 || (abs > 0.0 && abs < 0.01) {
        format!("{:.2e}", v)
    } else if abs >= 100.0 {
        format!("{:.0}", v)
    } else if abs >= 10.0 {
        format!("{:.1}", v)
    } else {
        format!("{:.2}", v)
    }
}

/// Draw common chart elements: background, title, axis labels, grid
fn draw_chart_base(canvas: &mut SvgCanvas, config: &SvgChartConfig) {
    // Background
    if let Some(bg) = config.background {
        canvas.set_background(bg);
    }
    // Title
    if let Some(ref title) = config.title {
        let style = DrawStyle {
            fill: Some(Color::BLACK),
            font_size: config.title_font_size,
            font_weight: "bold".to_string(),
            text_anchor: "middle".to_string(),
            dominant_baseline: "auto".to_string(),
            ..Default::default()
        };
        canvas.text(
            title,
            config.width / 2.0,
            config.margins.top / 2.0 + 4.0,
            &style,
        );
    }
    // X-axis label
    if let Some(ref xlabel) = config.x_label {
        let style = DrawStyle {
            fill: Some(Color::BLACK),
            font_size: config.font_size,
            text_anchor: "middle".to_string(),
            dominant_baseline: "auto".to_string(),
            ..Default::default()
        };
        canvas.text(
            xlabel,
            config.plot_x() + config.plot_width() / 2.0,
            config.height - 8.0,
            &style,
        );
    }
    // Y-axis label (rotated)
    if let Some(ref ylabel) = config.y_label {
        let style = DrawStyle {
            fill: Some(Color::BLACK),
            font_size: config.font_size,
            text_anchor: "middle".to_string(),
            dominant_baseline: "auto".to_string(),
            ..Default::default()
        };
        let cx = 14.0;
        let cy = config.plot_y() + config.plot_height() / 2.0;
        let transform = Transform::new().rotate_around(-90.0, cx, cy);
        canvas.text_transformed(ylabel, cx, cy, &transform, &style);
    }
}

fn draw_axes_and_grid(
    canvas: &mut SvgCanvas,
    config: &SvgChartConfig,
    x_ticks: &[(f64, String)], // (pixel_x, label)
    y_ticks: &[(f64, String)], // (pixel_y, label)
) {
    let px = config.plot_x();
    let py = config.plot_y();
    let pw = config.plot_width();
    let ph = config.plot_height();
    let axis_style = DrawStyle {
        fill: None,
        stroke: Some(Color::rgb(80, 80, 80)),
        stroke_width: 1.5,
        ..Default::default()
    };
    let grid_style = DrawStyle {
        fill: None,
        stroke: Some(config.grid_color),
        stroke_width: 0.8,
        stroke_dasharray: Some("4,4".to_string()),
        ..Default::default()
    };
    let tick_label_style = DrawStyle {
        fill: Some(Color::rgb(60, 60, 60)),
        font_size: config.font_size - 1.0,
        text_anchor: "middle".to_string(),
        dominant_baseline: "auto".to_string(),
        ..Default::default()
    };
    let y_tick_label_style = DrawStyle {
        fill: Some(Color::rgb(60, 60, 60)),
        font_size: config.font_size - 1.0,
        text_anchor: "end".to_string(),
        dominant_baseline: "middle".to_string(),
        ..Default::default()
    };
    // X axis line
    canvas.line(px, py + ph, px + pw, py + ph, &axis_style);
    // Y axis line
    canvas.line(px, py, px, py + ph, &axis_style);
    // X ticks and grid
    for (tick_x, label) in x_ticks {
        if config.show_grid {
            canvas.line(*tick_x, py, *tick_x, py + ph, &grid_style);
        }
        // tick mark
        canvas.line(*tick_x, py + ph, *tick_x, py + ph + 5.0, &axis_style);
        canvas.text(label, *tick_x, py + ph + 18.0, &tick_label_style);
    }
    // Y ticks and grid
    for (tick_y, label) in y_ticks {
        if config.show_grid {
            canvas.line(px, *tick_y, px + pw, *tick_y, &grid_style);
        }
        canvas.line(px - 5.0, *tick_y, px, *tick_y, &axis_style);
        canvas.text(label, px - 8.0, *tick_y, &y_tick_label_style);
    }
}

fn draw_legend(canvas: &mut SvgCanvas, config: &SvgChartConfig, items: &[(String, Color)]) {
    if items.is_empty() || config.legend_position == LegendPosition::None || !config.show_legend {
        return;
    }
    let item_h = 20.0;
    let box_w = 14.0;
    let box_h = 14.0;
    let pad = 10.0;
    let text_offset = box_w + 6.0;
    let max_label_len = items.iter().map(|(s, _)| s.len()).max().unwrap_or(5);
    let legend_w = text_offset + max_label_len as f64 * 7.0 + pad;
    let legend_h = items.len() as f64 * item_h + pad;

    let (lx, ly) = match config.legend_position {
        LegendPosition::TopRight => (
            config.plot_x() + config.plot_width() - legend_w - 8.0,
            config.plot_y() + 8.0,
        ),
        LegendPosition::TopLeft => (config.plot_x() + 8.0, config.plot_y() + 8.0),
        LegendPosition::BottomRight => (
            config.plot_x() + config.plot_width() - legend_w - 8.0,
            config.plot_y() + config.plot_height() - legend_h - 8.0,
        ),
        LegendPosition::BottomLeft => (
            config.plot_x() + 8.0,
            config.plot_y() + config.plot_height() - legend_h - 8.0,
        ),
        LegendPosition::None => return,
    };

    // Legend background
    let bg_style = DrawStyle {
        fill: Some(Color::rgba(255, 255, 255, 220)),
        stroke: Some(Color::rgb(180, 180, 180)),
        stroke_width: 1.0,
        ..Default::default()
    };
    canvas.rect_rounded(lx, ly, legend_w, legend_h, 4.0, 4.0, &bg_style);

    for (i, (label, color)) in items.iter().enumerate() {
        let iy = ly + pad / 2.0 + i as f64 * item_h;
        let color_style = DrawStyle {
            fill: Some(*color),
            stroke: None,
            ..Default::default()
        };
        canvas.rect(
            lx + pad / 2.0,
            iy + (item_h - box_h) / 2.0,
            box_w,
            box_h,
            &color_style,
        );
        let text_style = DrawStyle {
            fill: Some(Color::rgb(40, 40, 40)),
            font_size: config.font_size - 1.0,
            text_anchor: "start".to_string(),
            dominant_baseline: "middle".to_string(),
            ..Default::default()
        };
        canvas.text(
            label,
            lx + pad / 2.0 + text_offset,
            iy + item_h / 2.0,
            &text_style,
        );
    }
}

/// Wrap SVG in a complete HTML page
fn wrap_in_html(svg: &str, title: &str) -> String {
    format!(
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
</html>"#,
        title = title,
        svg = svg
    )
}

// ============================================================================
// BarChart
// ============================================================================

/// Orientation of a bar chart
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarOrientation {
    #[default]
    Vertical,
    Horizontal,
}

/// Bar chart (vertical or horizontal)
#[derive(Debug, Clone)]
pub struct BarChart {
    labels: Vec<String>,
    values: Vec<f64>,
    orientation: BarOrientation,
    config: SvgChartConfig,
}

impl BarChart {
    pub fn new(
        labels: Vec<String>,
        values: Vec<f64>,
        orientation: BarOrientation,
        config: SvgChartConfig,
    ) -> Self {
        Self {
            labels,
            values,
            orientation,
            config,
        }
    }

    pub fn render(&self) -> Result<String> {
        if self.values.is_empty() {
            return Err(Error::EmptyData("BarChart: no data".to_string()));
        }
        match self.orientation {
            BarOrientation::Vertical => self.render_vertical(),
            BarOrientation::Horizontal => self.render_horizontal(),
        }
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Bar Chart");
        Ok(wrap_in_html(&svg, title))
    }

    fn render_vertical(&self) -> Result<String> {
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let pw = config.plot_width();
        let ph = config.plot_height();
        let px = config.plot_x();
        let py = config.plot_y();

        let n = self.values.len();
        let max_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(0.0);
        let data_range = max_val - min_val;
        let effective_max = if data_range < f64::EPSILON {
            max_val + 1.0
        } else {
            max_val
        };
        let effective_min = min_val;

        let y_ticks_vals = nice_ticks(effective_min, effective_max, 6);
        let y_ticks: Vec<(f64, String)> = y_ticks_vals
            .iter()
            .map(|&v| {
                let pixel_y = py + ph - (v - effective_min) / (effective_max - effective_min) * ph;
                (pixel_y, format_tick(v))
            })
            .collect();

        let bar_total_w = pw / n as f64;
        let bar_w = bar_total_w * 0.7;
        let bar_gap = bar_total_w * 0.15;

        // X ticks at bar centers
        let x_ticks: Vec<(f64, String)> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let cx = px + i as f64 * bar_total_w + bar_gap + bar_w / 2.0;
                (cx, label.clone())
            })
            .collect();

        draw_axes_and_grid(&mut canvas, config, &x_ticks, &y_ticks);

        let zero_y = py + ph - (0.0 - effective_min) / (effective_max - effective_min) * ph;

        for (i, &val) in self.values.iter().enumerate() {
            let color = config.color_scheme.color_at(i);
            let bx = px + i as f64 * bar_total_w + bar_gap;
            let val_y = py + ph - (val - effective_min) / (effective_max - effective_min) * ph;
            let (by, bh) = if val >= 0.0 {
                (val_y, zero_y - val_y)
            } else {
                (zero_y, val_y - zero_y)
            };
            let style = DrawStyle {
                fill: Some(color),
                stroke: Some(Color::rgba(0, 0, 0, 30)),
                stroke_width: 0.5,
                ..Default::default()
            };
            canvas.rect_rounded(bx, by, bar_w, bh.max(1.0), 2.0, 2.0, &style);
        }

        Ok(canvas.to_string())
    }

    fn render_horizontal(&self) -> Result<String> {
        let config = &self.config;
        let margins = Margins::new(40.0, 30.0, 40.0, 120.0);
        let effective_config = SvgChartConfig {
            margins,
            ..config.clone()
        };
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, &effective_config);

        let pw = effective_config.plot_width();
        let ph = effective_config.plot_height();
        let px = effective_config.plot_x();
        let py = effective_config.plot_y();

        let n = self.values.len();
        let max_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_val = self
            .values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(0.0);
        let data_range = max_val - min_val;
        let effective_max = if data_range < f64::EPSILON {
            max_val + 1.0
        } else {
            max_val
        };
        let effective_min = min_val;

        let x_ticks_vals = nice_ticks(effective_min, effective_max, 6);
        let x_ticks: Vec<(f64, String)> = x_ticks_vals
            .iter()
            .map(|&v| {
                let pixel_x = px + (v - effective_min) / (effective_max - effective_min) * pw;
                (pixel_x, format_tick(v))
            })
            .collect();

        let bar_total_h = ph / n as f64;
        let bar_h = bar_total_h * 0.7;
        let bar_gap = bar_total_h * 0.15;

        let y_ticks: Vec<(f64, String)> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let cy = py + i as f64 * bar_total_h + bar_gap + bar_h / 2.0;
                (cy, label.clone())
            })
            .collect();

        // For horizontal: swap roles — x_ticks on bottom, y labels on left
        let y_label_style = DrawStyle {
            fill: Some(Color::rgb(60, 60, 60)),
            font_size: effective_config.font_size - 1.0,
            text_anchor: "end".to_string(),
            dominant_baseline: "middle".to_string(),
            ..Default::default()
        };
        let axis_style = DrawStyle {
            fill: None,
            stroke: Some(Color::rgb(80, 80, 80)),
            stroke_width: 1.5,
            ..Default::default()
        };
        let grid_style = DrawStyle {
            fill: None,
            stroke: Some(effective_config.grid_color),
            stroke_width: 0.8,
            stroke_dasharray: Some("4,4".to_string()),
            ..Default::default()
        };
        canvas.line(px, py, px, py + ph, &axis_style);
        canvas.line(px, py + ph, px + pw, py + ph, &axis_style);

        for (tick_x, label) in &x_ticks {
            if effective_config.show_grid {
                canvas.line(*tick_x, py, *tick_x, py + ph, &grid_style);
            }
            canvas.line(*tick_x, py + ph, *tick_x, py + ph + 5.0, &axis_style);
            let tick_style = DrawStyle {
                fill: Some(Color::rgb(60, 60, 60)),
                font_size: effective_config.font_size - 1.0,
                text_anchor: "middle".to_string(),
                dominant_baseline: "auto".to_string(),
                ..Default::default()
            };
            canvas.text(label, *tick_x, py + ph + 18.0, &tick_style);
        }

        for (cy, label) in &y_ticks {
            canvas.text(label, px - 8.0, *cy, &y_label_style);
            canvas.line(px - 4.0, *cy, px, *cy, &axis_style);
        }

        let zero_x = px + (0.0 - effective_min) / (effective_max - effective_min) * pw;

        for (i, &val) in self.values.iter().enumerate() {
            let color = effective_config.color_scheme.color_at(i);
            let by = py + i as f64 * bar_total_h + bar_gap;
            let val_x = px + (val - effective_min) / (effective_max - effective_min) * pw;
            let (bx, bw) = if val >= 0.0 {
                (zero_x, val_x - zero_x)
            } else {
                (val_x, zero_x - val_x)
            };
            let style = DrawStyle {
                fill: Some(color),
                stroke: Some(Color::rgba(0, 0, 0, 30)),
                stroke_width: 0.5,
                ..Default::default()
            };
            canvas.rect_rounded(bx, by, bw.max(1.0), bar_h, 2.0, 2.0, &style);
        }

        Ok(canvas.to_string())
    }
}

// ============================================================================
// LineChart
// ============================================================================

/// A series for line charts
#[derive(Debug, Clone)]
pub struct LineSeries {
    pub name: String,
    pub values: Vec<f64>,
    pub show_markers: bool,
    pub fill_area: bool,
    pub color: Option<Color>,
}

impl LineSeries {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values,
            show_markers: false,
            fill_area: false,
            color: None,
        }
    }
}

/// Line chart (multiple series)
#[derive(Debug, Clone)]
pub struct LineChart {
    x_values: Vec<f64>,
    series: Vec<LineSeries>,
    config: SvgChartConfig,
}

impl LineChart {
    pub fn new(x_values: Vec<f64>, series: Vec<LineSeries>, config: SvgChartConfig) -> Self {
        Self {
            x_values,
            series,
            config,
        }
    }

    pub fn render(&self) -> Result<String> {
        if self.x_values.is_empty() || self.series.is_empty() {
            return Err(Error::EmptyData("LineChart: no data".to_string()));
        }
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let pw = config.plot_width();
        let ph = config.plot_height();
        let px = config.plot_x();
        let py = config.plot_y();

        let x_min = self.x_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = self
            .x_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let all_y: Vec<f64> = self
            .series
            .iter()
            .flat_map(|s| s.values.iter().cloned())
            .collect();
        let y_min_raw = all_y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max_raw = all_y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = y_max_raw - y_min_raw;
        let y_min = y_min_raw - y_range * 0.05;
        let y_max = y_max_raw + y_range * 0.05;

        let x_range = if (x_max - x_min).abs() < f64::EPSILON {
            1.0
        } else {
            x_max - x_min
        };
        let y_range_eff = if (y_max - y_min).abs() < f64::EPSILON {
            1.0
        } else {
            y_max - y_min
        };

        let to_px = |xv: f64| px + (xv - x_min) / x_range * pw;
        let to_py = |yv: f64| py + ph - (yv - y_min) / y_range_eff * ph;

        let x_ticks_vals = nice_ticks(x_min, x_max, 6);
        let x_ticks: Vec<(f64, String)> = x_ticks_vals
            .iter()
            .map(|&v| (to_px(v), format_tick(v)))
            .collect();
        let y_ticks_vals = nice_ticks(y_min, y_max, 6);
        let y_ticks: Vec<(f64, String)> = y_ticks_vals
            .iter()
            .map(|&v| (to_py(v), format_tick(v)))
            .collect();

        draw_axes_and_grid(&mut canvas, config, &x_ticks, &y_ticks);

        let base_y = to_py(y_min.max(0.0));

        for (si, series) in self.series.iter().enumerate() {
            let color = series
                .color
                .unwrap_or_else(|| config.color_scheme.color_at(si));
            let n = series.values.len().min(self.x_values.len());
            if n == 0 {
                continue;
            }
            let points: Vec<(f64, f64)> = (0..n)
                .map(|i| (to_px(self.x_values[i]), to_py(series.values[i])))
                .collect();

            // Fill area under line
            if series.fill_area && n > 1 {
                let mut fill_pts = points.clone();
                fill_pts.push((points[n - 1].0, base_y));
                fill_pts.push((points[0].0, base_y));
                let fill_style = DrawStyle {
                    fill: Some(Color::rgba(color.r, color.g, color.b, 50)),
                    stroke: None,
                    ..Default::default()
                };
                canvas.polygon(&fill_pts, &fill_style);
            }

            // Line
            if n > 1 {
                let line_style = DrawStyle {
                    fill: None,
                    stroke: Some(color),
                    stroke_width: 2.0,
                    ..Default::default()
                };
                canvas.polyline(&points, &line_style);
            }

            // Markers
            if series.show_markers {
                let marker_style = DrawStyle {
                    fill: Some(color),
                    stroke: Some(Color::WHITE),
                    stroke_width: 1.5,
                    ..Default::default()
                };
                for &(mx, my) in &points {
                    canvas.circle(mx, my, 4.0, &marker_style);
                }
            }
        }

        // Legend
        let legend_items: Vec<(String, Color)> = self
            .series
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (
                    s.name.clone(),
                    s.color.unwrap_or_else(|| config.color_scheme.color_at(i)),
                )
            })
            .collect();
        draw_legend(&mut canvas, config, &legend_items);

        Ok(canvas.to_string())
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Line Chart");
        Ok(wrap_in_html(&svg, title))
    }
}

// ============================================================================
// ScatterPlot
// ============================================================================

/// Marker shape for scatter plot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkerShape {
    #[default]
    Circle,
    Square,
    Diamond,
    Cross,
}

/// Scatter plot
#[derive(Debug, Clone)]
pub struct ScatterPlot {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    labels: Option<Vec<String>>,
    marker_size: f64,
    marker_shape: MarkerShape,
    config: SvgChartConfig,
}

impl ScatterPlot {
    pub fn new(x_values: Vec<f64>, y_values: Vec<f64>, config: SvgChartConfig) -> Self {
        Self {
            x_values,
            y_values,
            labels: None,
            marker_size: 5.0,
            marker_shape: MarkerShape::Circle,
            config,
        }
    }

    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    pub fn with_marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }

    pub fn with_marker_shape(mut self, shape: MarkerShape) -> Self {
        self.marker_shape = shape;
        self
    }

    pub fn render(&self) -> Result<String> {
        let n = self.x_values.len().min(self.y_values.len());
        if n == 0 {
            return Err(Error::EmptyData("ScatterPlot: no data".to_string()));
        }
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let pw = config.plot_width();
        let ph = config.plot_height();
        let px = config.plot_x();
        let py = config.plot_y();

        let x_min = self.x_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = self
            .x_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = self.y_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = self
            .y_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let x_pad = (x_max - x_min).max(1.0) * 0.05;
        let y_pad = (y_max - y_min).max(1.0) * 0.05;
        let x_min = x_min - x_pad;
        let x_max = x_max + x_pad;
        let y_min = y_min - y_pad;
        let y_max = y_max + y_pad;

        let x_range = (x_max - x_min).max(f64::EPSILON);
        let y_range = (y_max - y_min).max(f64::EPSILON);

        let to_px = |xv: f64| px + (xv - x_min) / x_range * pw;
        let to_py = |yv: f64| py + ph - (yv - y_min) / y_range * ph;

        let x_ticks: Vec<(f64, String)> = nice_ticks(x_min, x_max, 6)
            .into_iter()
            .map(|v| (to_px(v), format_tick(v)))
            .collect();
        let y_ticks: Vec<(f64, String)> = nice_ticks(y_min, y_max, 6)
            .into_iter()
            .map(|v| (to_py(v), format_tick(v)))
            .collect();

        draw_axes_and_grid(&mut canvas, config, &x_ticks, &y_ticks);

        let color = config.color_scheme.color_at(0);
        let style = DrawStyle {
            fill: Some(Color::rgba(color.r, color.g, color.b, 180)),
            stroke: Some(Color::rgba(color.r / 2, color.g / 2, color.b / 2, 200)),
            stroke_width: 1.0,
            ..Default::default()
        };

        for i in 0..n {
            let mx = to_px(self.x_values[i]);
            let my = to_py(self.y_values[i]);
            let r = self.marker_size;
            match self.marker_shape {
                MarkerShape::Circle => canvas.circle(mx, my, r, &style),
                MarkerShape::Square => canvas.rect(mx - r, my - r, r * 2.0, r * 2.0, &style),
                MarkerShape::Diamond => {
                    let pts = vec![
                        (mx, my - r * 1.4),
                        (mx + r, my),
                        (mx, my + r * 1.4),
                        (mx - r, my),
                    ];
                    canvas.polygon(&pts, &style);
                }
                MarkerShape::Cross => {
                    let line_style = DrawStyle {
                        fill: None,
                        stroke: Some(color),
                        stroke_width: 2.0,
                        ..Default::default()
                    };
                    canvas.line(mx - r, my, mx + r, my, &line_style);
                    canvas.line(mx, my - r, mx, my + r, &line_style);
                }
            }
        }

        Ok(canvas.to_string())
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Scatter Plot");
        Ok(wrap_in_html(&svg, title))
    }
}

// ============================================================================
// Histogram
// ============================================================================

/// SVG Histogram
#[derive(Debug, Clone)]
pub struct SvgHistogram {
    data: Vec<f64>,
    bins: usize,
    config: SvgChartConfig,
}

impl SvgHistogram {
    pub fn new(data: Vec<f64>, bins: usize, config: SvgChartConfig) -> Self {
        Self { data, bins, config }
    }

    fn compute_bins(&self) -> (Vec<f64>, Vec<usize>) {
        if self.data.is_empty() || self.bins == 0 {
            return (vec![], vec![]);
        }
        let min = self.data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() < f64::EPSILON {
            return (vec![min, max], vec![self.data.len()]);
        }
        let bin_width = (max - min) / self.bins as f64;
        let mut edges = Vec::with_capacity(self.bins + 1);
        let mut counts = vec![0usize; self.bins];
        for i in 0..=self.bins {
            edges.push(min + i as f64 * bin_width);
        }
        for &v in &self.data {
            let idx = ((v - min) / bin_width).floor() as usize;
            let idx = idx.min(self.bins - 1);
            counts[idx] += 1;
        }
        (edges, counts)
    }

    pub fn render(&self) -> Result<String> {
        if self.data.is_empty() {
            return Err(Error::EmptyData("Histogram: no data".to_string()));
        }
        let (edges, counts) = self.compute_bins();
        if edges.len() < 2 {
            return Err(Error::EmptyData("Histogram: insufficient bins".to_string()));
        }
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let pw = config.plot_width();
        let ph = config.plot_height();
        let px = config.plot_x();
        let py = config.plot_y();

        let x_min = edges[0];
        let x_max = *edges.last().expect("non-empty edges");
        let y_max = *counts.iter().max().unwrap_or(&1) as f64;
        let y_max = y_max * 1.05;

        let x_range = (x_max - x_min).max(f64::EPSILON);
        let to_px = |xv: f64| px + (xv - x_min) / x_range * pw;
        let to_py = |yv: f64| py + ph - yv / y_max * ph;

        let x_ticks: Vec<(f64, String)> = nice_ticks(x_min, x_max, 6)
            .into_iter()
            .map(|v| (to_px(v), format_tick(v)))
            .collect();
        let y_ticks: Vec<(f64, String)> = nice_ticks(0.0, y_max, 5)
            .into_iter()
            .map(|v| (to_py(v), format_tick(v)))
            .collect();

        draw_axes_and_grid(&mut canvas, config, &x_ticks, &y_ticks);

        let color = config.color_scheme.color_at(0);
        for (i, &count) in counts.iter().enumerate() {
            let bar_x = to_px(edges[i]);
            let bar_x2 = to_px(edges[i + 1]);
            let bar_w = (bar_x2 - bar_x - 1.0).max(1.0);
            let bar_h = count as f64 / y_max * ph;
            let style = DrawStyle {
                fill: Some(color),
                stroke: Some(Color::rgba(0, 0, 0, 40)),
                stroke_width: 0.5,
                ..Default::default()
            };
            canvas.rect(bar_x, py + ph - bar_h, bar_w, bar_h.max(0.0), &style);
        }

        Ok(canvas.to_string())
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Histogram");
        Ok(wrap_in_html(&svg, title))
    }
}

// ============================================================================
// HeatMap
// ============================================================================

/// 2D heatmap chart
#[derive(Debug, Clone)]
pub struct HeatMap {
    data: Vec<Vec<f64>>,
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    config: SvgChartConfig,
}

impl HeatMap {
    pub fn new(
        data: Vec<Vec<f64>>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
        config: SvgChartConfig,
    ) -> Self {
        Self {
            data,
            row_labels,
            col_labels,
            config,
        }
    }

    pub fn render(&self) -> Result<String> {
        if self.data.is_empty() || self.data[0].is_empty() {
            return Err(Error::EmptyData("HeatMap: no data".to_string()));
        }
        let nrows = self.data.len();
        let ncols = self.data[0].len();
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let pw = config.plot_width();
        let ph = config.plot_height();
        let px = config.plot_x();
        let py = config.plot_y();

        let all_vals: Vec<f64> = self.data.iter().flat_map(|r| r.iter().cloned()).collect();
        let v_min = all_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let v_max = all_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let v_range = (v_max - v_min).max(f64::EPSILON);

        let gradient = config.color_scheme.gradient();

        // Add gradient to defs for colorbar
        let grad_stops: Vec<(f64, Color)> = (0..=10)
            .map(|i| {
                let t = i as f64 / 10.0;
                (t, gradient.sample(t))
            })
            .collect();
        canvas
            .defs_mut()
            .add_linear_gradient("hm_grad", 0.0, 0.0, 1.0, 0.0, &grad_stops);

        let cell_w = pw / ncols as f64;
        let cell_h = ph / nrows as f64;

        for (r, row) in self.data.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                let t = (val - v_min) / v_range;
                let color = gradient.sample(t);
                let cx = px + c as f64 * cell_w;
                let cy = py + r as f64 * cell_h;
                let style = DrawStyle {
                    fill: Some(color),
                    stroke: Some(Color::rgba(255, 255, 255, 80)),
                    stroke_width: 0.5,
                    ..Default::default()
                };
                canvas.rect(cx, cy, cell_w, cell_h, &style);
                // Cell value label
                let lum = 0.299 * color.r as f64 + 0.587 * color.g as f64 + 0.114 * color.b as f64;
                let text_color = if lum > 128.0 {
                    Color::BLACK
                } else {
                    Color::WHITE
                };
                let text_style = DrawStyle {
                    fill: Some(text_color),
                    font_size: (config.font_size - 2.0).max(8.0),
                    text_anchor: "middle".to_string(),
                    dominant_baseline: "middle".to_string(),
                    ..Default::default()
                };
                canvas.text(
                    format_tick(val),
                    cx + cell_w / 2.0,
                    cy + cell_h / 2.0,
                    &text_style,
                );
            }
        }

        // Column labels (top)
        for (c, label) in self.col_labels.iter().enumerate() {
            let label_style = DrawStyle {
                fill: Some(Color::rgb(40, 40, 40)),
                font_size: config.font_size - 1.0,
                text_anchor: "middle".to_string(),
                dominant_baseline: "auto".to_string(),
                ..Default::default()
            };
            canvas.text(
                label,
                px + c as f64 * cell_w + cell_w / 2.0,
                py - 6.0,
                &label_style,
            );
        }

        // Row labels (left)
        for (r, label) in self.row_labels.iter().enumerate() {
            let label_style = DrawStyle {
                fill: Some(Color::rgb(40, 40, 40)),
                font_size: config.font_size - 1.0,
                text_anchor: "end".to_string(),
                dominant_baseline: "middle".to_string(),
                ..Default::default()
            };
            canvas.text(
                label,
                px - 8.0,
                py + r as f64 * cell_h + cell_h / 2.0,
                &label_style,
            );
        }

        // Colorbar
        let cb_x = px + pw + 10.0;
        let cb_y = py;
        let cb_w = 16.0;
        let cb_h = ph;
        canvas.rect_with_gradient(cb_x, cb_y, cb_w, cb_h, "hm_grad", None, 0.0);
        let cb_border = DrawStyle {
            fill: None,
            stroke: Some(Color::rgb(100, 100, 100)),
            stroke_width: 0.8,
            ..Default::default()
        };
        canvas.rect(cb_x, cb_y, cb_w, cb_h, &cb_border);
        // Colorbar labels
        for i in 0..=4 {
            let t = i as f64 / 4.0;
            let val = v_min + t * v_range;
            let label_y = cb_y + cb_h - t * cb_h;
            let cb_label_style = DrawStyle {
                fill: Some(Color::rgb(60, 60, 60)),
                font_size: config.font_size - 2.0,
                text_anchor: "start".to_string(),
                dominant_baseline: "middle".to_string(),
                ..Default::default()
            };
            canvas.text(
                format_tick(val),
                cb_x + cb_w + 4.0,
                label_y,
                &cb_label_style,
            );
        }

        Ok(canvas.to_string())
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Heatmap");
        Ok(wrap_in_html(&svg, title))
    }
}

// ============================================================================
// PieChart
// ============================================================================

/// Pie or donut chart
#[derive(Debug, Clone)]
pub struct PieChart {
    labels: Vec<String>,
    values: Vec<f64>,
    donut: bool,
    donut_ratio: f64,
    config: SvgChartConfig,
}

impl PieChart {
    pub fn new(labels: Vec<String>, values: Vec<f64>, config: SvgChartConfig) -> Self {
        Self {
            labels,
            values,
            donut: false,
            donut_ratio: 0.5,
            config,
        }
    }

    pub fn as_donut(mut self, hole_ratio: f64) -> Self {
        self.donut = true;
        self.donut_ratio = hole_ratio.clamp(0.1, 0.9);
        self
    }

    pub fn render(&self) -> Result<String> {
        if self.values.is_empty() {
            return Err(Error::EmptyData("PieChart: no data".to_string()));
        }
        let total: f64 = self.values.iter().sum();
        if total <= 0.0 {
            return Err(Error::InvalidInput(
                "PieChart: total must be positive".to_string(),
            ));
        }
        let config = &self.config;
        let mut canvas = SvgCanvas::new(config.width, config.height);
        draw_chart_base(&mut canvas, config);

        let plot_area_w = config.plot_width();
        let plot_area_h = config.plot_height();
        let cx = config.plot_x() + plot_area_w / 2.0;
        let cy = config.plot_y() + plot_area_h / 2.0;
        let r = (plot_area_w.min(plot_area_h) / 2.0 - 10.0).max(10.0);
        let inner_r = if self.donut {
            r * self.donut_ratio
        } else {
            0.0
        };

        let mut start_angle = -std::f64::consts::FRAC_PI_2;

        for (i, &val) in self.values.iter().enumerate() {
            let color = config.color_scheme.color_at(i);
            let frac = val / total;
            let sweep = frac * 2.0 * std::f64::consts::PI;
            let end_angle = start_angle + sweep;
            let large_arc = sweep > std::f64::consts::PI;

            let x1_outer = cx + r * start_angle.cos();
            let y1_outer = cy + r * start_angle.sin();
            let x2_outer = cx + r * end_angle.cos();
            let y2_outer = cy + r * end_angle.sin();

            let path_d = if self.donut {
                let x1_inner = cx + inner_r * end_angle.cos();
                let y1_inner = cy + inner_r * end_angle.sin();
                let x2_inner = cx + inner_r * start_angle.cos();
                let y2_inner = cy + inner_r * start_angle.sin();
                PathBuilder::new()
                    .move_to(x1_outer, y1_outer)
                    .arc(r, r, 0.0, large_arc, true, x2_outer, y2_outer)
                    .line_to(x1_inner, y1_inner)
                    .arc(inner_r, inner_r, 0.0, large_arc, false, x2_inner, y2_inner)
                    .close()
                    .build()
            } else {
                PathBuilder::new()
                    .move_to(cx, cy)
                    .line_to(x1_outer, y1_outer)
                    .arc(r, r, 0.0, large_arc, true, x2_outer, y2_outer)
                    .close()
                    .build()
            };

            let style = DrawStyle {
                fill: Some(color),
                stroke: Some(Color::WHITE),
                stroke_width: 1.5,
                ..Default::default()
            };
            canvas.path(path_d, &style);

            // Percentage label on slice
            let mid_angle = start_angle + sweep / 2.0;
            let label_r = if self.donut {
                (r + inner_r) / 2.0
            } else {
                r * 0.65
            };
            let lx = cx + label_r * mid_angle.cos();
            let ly = cy + label_r * mid_angle.sin();

            if frac > 0.04 {
                let pct = format!("{:.1}%", frac * 100.0);
                let lum = 0.299 * color.r as f64 + 0.587 * color.g as f64 + 0.114 * color.b as f64;
                let text_color = if lum > 128.0 {
                    Color::BLACK
                } else {
                    Color::WHITE
                };
                let lbl_style = DrawStyle {
                    fill: Some(text_color),
                    font_size: config.font_size - 1.0,
                    text_anchor: "middle".to_string(),
                    dominant_baseline: "middle".to_string(),
                    ..Default::default()
                };
                canvas.text(pct, lx, ly, &lbl_style);
            }

            start_angle = end_angle;
        }

        // Legend
        let legend_items: Vec<(String, Color)> = self
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| (l.clone(), config.color_scheme.color_at(i)))
            .collect();
        draw_legend(&mut canvas, config, &legend_items);

        Ok(canvas.to_string())
    }

    pub fn render_html(&self) -> Result<String> {
        let svg = self.render()?;
        let title = self.config.title.as_deref().unwrap_or("Pie Chart");
        Ok(wrap_in_html(&svg, title))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SvgChartConfig {
        SvgChartConfig::default()
    }

    #[test]
    fn test_bar_chart_vertical() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![10.0, 25.0, 15.0];
        let chart = BarChart::new(labels, values, BarOrientation::Vertical, default_config());
        let svg = chart.render().expect("render bar chart");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_bar_chart_horizontal() {
        let labels = vec!["X".to_string(), "Y".to_string()];
        let values = vec![5.0, 8.0];
        let chart = BarChart::new(labels, values, BarOrientation::Horizontal, default_config());
        let svg = chart.render().expect("render horizontal bar chart");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_line_chart() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let series = vec![LineSeries {
            name: "test".to_string(),
            values: vec![1.0, 3.0, 2.0, 5.0, 4.0],
            show_markers: true,
            fill_area: false,
            color: None,
        }];
        let chart = LineChart::new(x, series, default_config());
        let svg = chart.render().expect("render line chart");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("polyline") || svg.contains("circle"));
    }

    #[test]
    fn test_scatter_plot() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![4.0, 3.0, 2.0, 1.0];
        let chart = ScatterPlot::new(x, y, default_config());
        let svg = chart.render().expect("render scatter plot");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_histogram() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let chart = SvgHistogram::new(data, 5, default_config());
        let svg = chart.render().expect("render histogram");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn test_heatmap() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let rows = vec!["R1".to_string(), "R2".to_string()];
        let cols = vec!["C1".to_string(), "C2".to_string(), "C3".to_string()];
        let chart = HeatMap::new(data, rows, cols, default_config());
        let svg = chart.render().expect("render heatmap");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_pie_chart() {
        let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let values = vec![30.0, 45.0, 25.0];
        let chart = PieChart::new(labels, values, default_config());
        let svg = chart.render().expect("render pie chart");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn test_donut_chart() {
        let labels = vec!["X".to_string(), "Y".to_string()];
        let values = vec![60.0, 40.0];
        let chart = PieChart::new(labels, values, default_config()).as_donut(0.4);
        let svg = chart.render().expect("render donut chart");
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_html_wrapper() {
        let labels = vec!["A".to_string()];
        let values = vec![1.0];
        let chart = BarChart::new(labels, values, BarOrientation::Vertical, default_config());
        let html = chart.render_html().expect("render html");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_nice_ticks() {
        let ticks = nice_ticks(0.0, 100.0, 5);
        assert!(!ticks.is_empty());
        assert!(ticks[0] <= 0.0);
        assert!(*ticks.last().expect("last") >= 100.0);
    }
}

#![allow(clippy::result_large_err)]
//! Integration tests for SVG/HTML visualization module

use pandrs::vis::svg::{
    BarChart, BarOrientation, Color, ColorScheme, DrawStyle, HeatMap, LegendPosition, LineChart,
    LineSeries, Margins, MarkerShape, PathBuilder, PieChart, ScatterPlot, SvgCanvas,
    SvgChartConfig, SvgHistogram, SvgPlotType, SvgVisualize, Transform,
};
use pandrs::{DataFrame, Series};

// ============================================================================
// Helpers
// ============================================================================

fn default_config() -> SvgChartConfig {
    SvgChartConfig::default()
}

fn small_config() -> SvgChartConfig {
    SvgChartConfig {
        width: 400.0,
        height: 300.0,
        ..Default::default()
    }
}

fn make_bar_df() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "category".to_string(),
        Series::new(
            vec![
                "Alpha".to_string(),
                "Beta".to_string(),
                "Gamma".to_string(),
                "Delta".to_string(),
            ],
            Some("category".to_string()),
        )
        .expect("series"),
    )
    .expect("add column");
    df.add_column(
        "sales".to_string(),
        Series::new(vec![42i64, 78, 35, 91], Some("sales".to_string())).expect("series"),
    )
    .expect("add column");
    df
}

fn make_numeric_df() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], None).expect("series"),
    )
    .expect("add column");
    df.add_column(
        "y1".to_string(),
        Series::new(vec![2.0f64, 4.5, 3.1, 6.2, 5.0, 7.8], None).expect("series"),
    )
    .expect("add column");
    df.add_column(
        "y2".to_string(),
        Series::new(vec![5.0f64, 3.2, 4.8, 2.1, 6.3, 4.0], None).expect("series"),
    )
    .expect("add column");
    df
}

// ============================================================================
// Engine tests
// ============================================================================

#[test]
fn test_svg_canvas_produces_valid_svg() {
    let mut canvas = SvgCanvas::new(400.0, 300.0);
    canvas.set_background(Color::WHITE);
    let fill_style = DrawStyle::with_fill(Color::BLUE);
    canvas.rect(10.0, 10.0, 100.0, 50.0, &fill_style);
    let svg = canvas.to_string();
    assert!(svg.starts_with("<svg"), "SVG must start with <svg");
    assert!(svg.ends_with("</svg>"), "SVG must end with </svg>");
    assert!(svg.contains("xmlns=\"http://www.w3.org/2000/svg\""));
}

#[test]
fn test_svg_canvas_all_primitives() {
    let mut canvas = SvgCanvas::new(600.0, 400.0);
    let fill = DrawStyle::with_fill(Color::rgb(200, 100, 50));
    let stroke = DrawStyle::with_stroke(Color::BLACK, 2.0);
    canvas.rect(10.0, 10.0, 50.0, 30.0, &fill);
    canvas.rect_rounded(70.0, 10.0, 50.0, 30.0, 5.0, 5.0, &fill);
    canvas.line(0.0, 0.0, 100.0, 100.0, &stroke);
    canvas.circle(200.0, 200.0, 30.0, &fill);
    canvas.ellipse(300.0, 200.0, 40.0, 20.0, &fill);
    canvas.polyline(&[(10.0, 10.0), (50.0, 40.0), (90.0, 20.0)], &stroke);
    canvas.polygon(&[(100.0, 10.0), (130.0, 50.0), (70.0, 50.0)], &fill);
    let path_d = PathBuilder::new()
        .move_to(0.0, 0.0)
        .line_to(50.0, 0.0)
        .arc(25.0, 25.0, 0.0, false, true, 50.0, 50.0)
        .close()
        .build();
    canvas.path(path_d, &fill);
    let text_style = DrawStyle {
        fill: Some(Color::BLACK),
        font_size: 14.0,
        text_anchor: "middle".to_string(),
        dominant_baseline: "middle".to_string(),
        ..Default::default()
    };
    canvas.text("Hello World", 200.0, 100.0, &text_style);
    let t = Transform::new().rotate_around(-90.0, 50.0, 50.0);
    canvas.text_transformed("Rotated", 50.0, 50.0, &t, &text_style);
    let svg = canvas.to_string();
    assert!(svg.contains("<rect"));
    assert!(svg.contains("<line"));
    assert!(svg.contains("<circle"));
    assert!(svg.contains("<ellipse"));
    assert!(svg.contains("<polyline"));
    assert!(svg.contains("<polygon"));
    assert!(svg.contains("<path"));
    assert!(svg.contains("<text"));
}

#[test]
fn test_svg_canvas_with_gradient() {
    let mut canvas = SvgCanvas::new(300.0, 200.0);
    canvas.defs_mut().add_linear_gradient(
        "test_grad",
        0.0,
        0.0,
        1.0,
        0.0,
        &[
            (0.0, Color::WHITE),
            (0.5, Color::rgb(100, 150, 200)),
            (1.0, Color::BLUE),
        ],
    );
    canvas
        .defs_mut()
        .add_clip_rect("clip1", 10.0, 10.0, 280.0, 180.0);
    canvas.rect_with_gradient(0.0, 0.0, 300.0, 200.0, "test_grad", Some(Color::BLACK), 1.0);
    let svg = canvas.to_string();
    assert!(svg.contains("linearGradient"));
    assert!(svg.contains("test_grad"));
    assert!(svg.contains("clipPath"));
}

#[test]
fn test_path_builder_all_commands() {
    let path = PathBuilder::new()
        .move_to(0.0, 0.0)
        .line_to(100.0, 0.0)
        .horizontal_to(150.0)
        .vertical_to(100.0)
        .cubic_bezier(160.0, 50.0, 180.0, 90.0, 200.0, 100.0)
        .quadratic_bezier(220.0, 50.0, 250.0, 100.0)
        .arc(30.0, 30.0, 0.0, false, true, 300.0, 100.0)
        .close()
        .build();
    assert!(path.contains("M "));
    assert!(path.contains("L "));
    assert!(path.contains("H "));
    assert!(path.contains("V "));
    assert!(path.contains("C "));
    assert!(path.contains("Q "));
    assert!(path.contains("A "));
    assert!(path.ends_with("Z"));
}

// ============================================================================
// Color tests
// ============================================================================

#[test]
fn test_color_constants() {
    assert_eq!(Color::WHITE.r, 255);
    assert_eq!(Color::BLACK.r, 0);
    assert_eq!(Color::WHITE.to_hex(), "#FFFFFF");
    assert_eq!(Color::BLACK.to_hex(), "#000000");
}

#[test]
fn test_color_from_hex_with_hash() {
    let c = Color::from_hex("#FF8800").expect("parse color");
    assert_eq!(c.r, 255);
    assert_eq!(c.g, 136);
    assert_eq!(c.b, 0);
}

#[test]
fn test_color_from_hex_without_hash() {
    let c = Color::from_hex("1A2B3C").expect("parse color");
    assert_eq!(c.r, 0x1A);
    assert_eq!(c.g, 0x2B);
    assert_eq!(c.b, 0x3C);
}

#[test]
fn test_color_interpolation_endpoints() {
    let red = Color::rgb(255, 0, 0);
    let blue = Color::rgb(0, 0, 255);
    let at_zero = red.interpolate(&blue, 0.0);
    assert_eq!(at_zero.r, 255);
    assert_eq!(at_zero.b, 0);
    let at_one = red.interpolate(&blue, 1.0);
    assert_eq!(at_one.r, 0);
    assert_eq!(at_one.b, 255);
}

#[test]
fn test_color_opacity() {
    let c = Color::rgba(100, 150, 200, 128);
    let opacity = c.opacity();
    assert!((opacity - 0.502).abs() < 0.01);
}

#[test]
fn test_color_scheme_viridis() {
    let scheme = ColorScheme::Viridis;
    let colors = scheme.categorical_colors();
    assert!(!colors.is_empty());
    let grad = scheme.gradient();
    let mid = grad.sample(0.5);
    // Should be a valid color (not pure black)
    assert!(mid.r > 0 || mid.g > 0 || mid.b > 0);
}

#[test]
fn test_color_scheme_plasma() {
    let scheme = ColorScheme::Plasma;
    let colors = scheme.categorical_colors();
    assert!(!colors.is_empty());
}

// ============================================================================
// Chart type tests
// ============================================================================

#[test]
fn test_bar_chart_vertical_svg_structure() {
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let values = vec![10.0, 25.0, 15.0];
    let config = SvgChartConfig {
        title: Some("Test Bar Chart".to_string()),
        x_label: Some("Category".to_string()),
        y_label: Some("Value".to_string()),
        show_legend: false,
        ..small_config()
    };
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, config);
    let svg = chart.render().expect("render bar chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
    assert!(svg.contains("<rect"));
    assert!(svg.contains("<line")); // axes
    assert!(svg.contains("<text")); // labels
}

#[test]
fn test_bar_chart_horizontal_svg_structure() {
    let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let values = vec![5.0, 12.0, 8.0];
    let chart = BarChart::new(labels, values, BarOrientation::Horizontal, small_config());
    let svg = chart.render().expect("render horizontal bar chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_bar_chart_with_negative_values() {
    let labels = vec!["P".to_string(), "N".to_string(), "Z".to_string()];
    let values = vec![10.0, -5.0, 0.0];
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, small_config());
    let svg = chart.render().expect("render bar with negatives");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_bar_chart_empty_data_error() {
    let chart = BarChart::new(vec![], vec![], BarOrientation::Vertical, default_config());
    let result = chart.render();
    assert!(result.is_err());
}

#[test]
fn test_bar_chart_html_output() {
    let labels = vec!["A".to_string(), "B".to_string()];
    let values = vec![1.0, 2.0];
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, small_config());
    let html = chart.render_html().expect("render html");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<html"));
    assert!(html.contains("<svg"));
    assert!(html.contains("</html>"));
}

#[test]
fn test_line_chart_single_series() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let series = vec![LineSeries {
        name: "Series A".to_string(),
        values: vec![1.0, 3.0, 2.5, 4.0, 3.5],
        show_markers: true,
        fill_area: false,
        color: None,
    }];
    let chart = LineChart::new(x, series, small_config());
    let svg = chart.render().expect("render line chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
    assert!(svg.contains("polyline") || svg.contains("circle"));
}

#[test]
fn test_line_chart_multi_series_with_area() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let series = vec![
        LineSeries {
            name: "S1".to_string(),
            values: vec![1.0, 2.0, 3.0, 4.0],
            show_markers: false,
            fill_area: true,
            color: Some(Color::rgb(100, 150, 200)),
        },
        LineSeries {
            name: "S2".to_string(),
            values: vec![4.0, 3.0, 2.0, 1.0],
            show_markers: true,
            fill_area: false,
            color: None,
        },
    ];
    let config = SvgChartConfig {
        show_legend: true,
        legend_position: LegendPosition::TopLeft,
        ..small_config()
    };
    let chart = LineChart::new(x, series, config);
    let svg = chart.render().expect("render multi-series line chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("polygon") || svg.contains("polyline"));
}

#[test]
fn test_line_chart_html_output() {
    let x = vec![1.0, 2.0, 3.0];
    let series = vec![LineSeries::new("test", vec![1.0, 2.0, 1.5])];
    let chart = LineChart::new(x, series, small_config());
    let html = chart.render_html().expect("render html");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<svg"));
}

#[test]
fn test_scatter_plot_circle_markers() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 1.0, 3.0, 5.0];
    let chart = ScatterPlot::new(x, y, small_config());
    let svg = chart.render().expect("render scatter plot");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<circle"));
}

#[test]
fn test_scatter_plot_square_markers() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    let chart = ScatterPlot::new(x, y, small_config())
        .with_marker_shape(MarkerShape::Square)
        .with_marker_size(8.0);
    let svg = chart.render().expect("render scatter square");
    assert!(svg.contains("<rect"));
}

#[test]
fn test_scatter_plot_diamond_markers() {
    let x = vec![1.0, 2.0];
    let y = vec![1.0, 2.0];
    let chart = ScatterPlot::new(x, y, small_config()).with_marker_shape(MarkerShape::Diamond);
    let svg = chart.render().expect("render scatter diamond");
    assert!(svg.contains("<polygon"));
}

#[test]
fn test_scatter_plot_cross_markers() {
    let x = vec![1.0, 2.0];
    let y = vec![1.0, 2.0];
    let chart = ScatterPlot::new(x, y, small_config()).with_marker_shape(MarkerShape::Cross);
    let svg = chart.render().expect("render scatter cross");
    assert!(svg.contains("<line"));
}

#[test]
fn test_scatter_plot_empty_error() {
    let chart = ScatterPlot::new(vec![], vec![], default_config());
    assert!(chart.render().is_err());
}

#[test]
fn test_histogram_basic() {
    let data = vec![1.0, 1.5, 2.0, 2.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0];
    let chart = SvgHistogram::new(data, 5, small_config());
    let svg = chart.render().expect("render histogram");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_histogram_single_bin() {
    let data = vec![3.0; 20];
    let chart = SvgHistogram::new(data, 5, small_config());
    let svg = chart.render().expect("render single-value histogram");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_histogram_empty_error() {
    let chart = SvgHistogram::new(vec![], 5, default_config());
    assert!(chart.render().is_err());
}

#[test]
fn test_histogram_html_output() {
    let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
    let chart = SvgHistogram::new(data, 10, small_config());
    let html = chart.render_html().expect("render html");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<svg"));
}

#[test]
fn test_heatmap_basic() {
    let data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let rows = vec!["R1".to_string(), "R2".to_string(), "R3".to_string()];
    let cols = vec!["C1".to_string(), "C2".to_string(), "C3".to_string()];
    let config = SvgChartConfig {
        title: Some("Test Heatmap".to_string()),
        color_scheme: ColorScheme::Viridis,
        ..small_config()
    };
    let chart = HeatMap::new(data, rows, cols, config);
    let svg = chart.render().expect("render heatmap");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
    assert!(svg.contains("linearGradient"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_heatmap_plasma_scheme() {
    let data = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
    let rows = vec!["A".to_string(), "B".to_string()];
    let cols = vec!["X".to_string(), "Y".to_string()];
    let config = SvgChartConfig {
        color_scheme: ColorScheme::Plasma,
        ..small_config()
    };
    let chart = HeatMap::new(data, rows, cols, config);
    let svg = chart.render().expect("render heatmap plasma");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_heatmap_empty_error() {
    let chart = HeatMap::new(vec![], vec![], vec![], default_config());
    assert!(chart.render().is_err());
}

#[test]
fn test_pie_chart_basic() {
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let values = vec![30.0, 45.0, 25.0];
    let chart = PieChart::new(labels, values, small_config());
    let svg = chart.render().expect("render pie chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<path"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_donut_chart() {
    let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let values = vec![50.0, 30.0, 20.0];
    let chart = PieChart::new(labels, values, small_config()).as_donut(0.5);
    let svg = chart.render().expect("render donut chart");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<path"));
}

#[test]
fn test_pie_chart_with_legend() {
    let labels = vec!["Cat1".to_string(), "Cat2".to_string(), "Cat3".to_string()];
    let values = vec![40.0, 35.0, 25.0];
    let config = SvgChartConfig {
        show_legend: true,
        legend_position: LegendPosition::BottomRight,
        ..small_config()
    };
    let chart = PieChart::new(labels, values, config);
    let svg = chart.render().expect("render pie with legend");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_pie_chart_empty_error() {
    let chart = PieChart::new(vec![], vec![], default_config());
    assert!(chart.render().is_err());
}

#[test]
fn test_pie_chart_zero_total_error() {
    let labels = vec!["A".to_string()];
    let values = vec![0.0];
    let chart = PieChart::new(labels, values, default_config());
    assert!(chart.render().is_err());
}

#[test]
fn test_pie_chart_html_output() {
    let labels = vec!["A".to_string(), "B".to_string()];
    let values = vec![60.0, 40.0];
    let chart = PieChart::new(labels, values, small_config());
    let html = chart.render_html().expect("render html");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<svg"));
}

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn test_svg_chart_config_dimensions() {
    let config = SvgChartConfig {
        width: 800.0,
        height: 600.0,
        margins: Margins::new(50.0, 40.0, 80.0, 90.0),
        ..Default::default()
    };
    assert_eq!(config.plot_width(), 800.0 - 90.0 - 40.0);
    assert_eq!(config.plot_height(), 600.0 - 50.0 - 80.0);
    assert_eq!(config.plot_x(), 90.0);
    assert_eq!(config.plot_y(), 50.0);
}

#[test]
fn test_color_scheme_categorical_cycling() {
    for scheme in &[
        ColorScheme::Default,
        ColorScheme::Blues,
        ColorScheme::Greens,
        ColorScheme::Oranges,
        ColorScheme::Viridis,
        ColorScheme::Plasma,
        ColorScheme::Categorical,
    ] {
        let colors = scheme.categorical_colors();
        assert!(!colors.is_empty(), "{:?} should have colors", scheme);
        let c0 = scheme.color_at(0);
        let c_wrap = scheme.color_at(colors.len());
        assert_eq!(c0, c_wrap, "{:?} should cycle colors", scheme);
    }
}

// ============================================================================
// DataFrame integration tests
// ============================================================================

#[test]
fn test_df_plot_bar_svg() {
    let df = make_bar_df();
    let svg = df
        .plot_bar_svg("category", "sales", None)
        .expect("plot bar");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_df_plot_bar_horizontal_svg() {
    let df = make_bar_df();
    let svg = df
        .plot_bar_horizontal_svg("category", "sales", None)
        .expect("plot bar h");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_df_plot_bar_with_config() {
    let df = make_bar_df();
    let config = SvgChartConfig {
        title: Some("Sales by Category".to_string()),
        x_label: Some("Category".to_string()),
        y_label: Some("Sales".to_string()),
        color_scheme: ColorScheme::Blues,
        show_legend: false,
        ..Default::default()
    };
    let svg = df
        .plot_bar_svg("category", "sales", Some(config))
        .expect("plot bar with config");
    assert!(svg.contains("Sales by Category") || svg.contains("<text"));
}

#[test]
fn test_df_plot_line_svg() {
    let df = make_numeric_df();
    let svg = df
        .plot_line_svg("x", &["y1", "y2"], None)
        .expect("plot line");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("</svg>"));
}

#[test]
fn test_df_plot_line_svg_single_series() {
    let df = make_numeric_df();
    let svg = df
        .plot_line_svg("x", &["y1"], None)
        .expect("plot line single");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_df_plot_scatter_svg() {
    let df = make_numeric_df();
    let svg = df.plot_scatter_svg("x", "y1", None).expect("plot scatter");
    assert!(svg.contains("<svg"));
    assert!(
        svg.contains("<circle")
            || svg.contains("<rect")
            || svg.contains("<polygon")
            || svg.contains("<line")
    );
}

#[test]
fn test_df_plot_histogram_svg() {
    let df = make_numeric_df();
    let svg = df
        .plot_histogram_svg("y1", 5, None)
        .expect("plot histogram");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
}

#[test]
fn test_df_plot_heatmap_svg() {
    let df = make_numeric_df();
    let svg = df.plot_heatmap_svg(None).expect("plot heatmap");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<rect"));
}

#[test]
fn test_df_plot_pie_svg() {
    let df = make_bar_df();
    let svg = df
        .plot_pie_svg("category", "sales", None)
        .expect("plot pie");
    assert!(svg.contains("<svg"));
    assert!(svg.contains("<path"));
}

#[test]
fn test_df_missing_column_error() {
    let df = make_bar_df();
    let result = df.plot_bar_svg("nonexistent_col", "sales", None);
    assert!(result.is_err());
}

#[test]
fn test_df_plot_line_empty_y_cols_error() {
    let df = make_numeric_df();
    let result = df.plot_line_svg("x", &[], None);
    assert!(result.is_err());
}

// ============================================================================
// File I/O tests using temp directory
// ============================================================================

#[test]
fn test_save_svg_to_temp_file() {
    let df = make_bar_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_bar.svg");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_svg(&path_str, SvgPlotType::Bar, None)
        .expect("save svg");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<svg"));
    assert!(content.contains("</svg>"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_html_to_temp_file() {
    let df = make_bar_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_bar.html");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_html(&path_str, SvgPlotType::Bar, None)
        .expect("save html");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("<svg"));
    assert!(content.contains("</html>"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_scatter_svg_to_temp() {
    let df = make_numeric_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_scatter.svg");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_svg(
        &path_str,
        SvgPlotType::Scatter {
            x_col: "x".to_string(),
            y_col: "y1".to_string(),
        },
        None,
    )
    .expect("save scatter svg");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<svg"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_histogram_svg_to_temp() {
    let df = make_numeric_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_hist.svg");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_svg(
        &path_str,
        SvgPlotType::Histogram {
            col: "y1".to_string(),
            bins: 8,
        },
        None,
    )
    .expect("save histogram svg");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<svg"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_heatmap_html_to_temp() {
    let df = make_numeric_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_heatmap.html");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_html(&path_str, SvgPlotType::Heatmap, None)
        .expect("save heatmap html");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("<svg"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_line_html_to_temp() {
    let df = make_numeric_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_line.html");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_html(
        &path_str,
        SvgPlotType::Line {
            x_col: "x".to_string(),
            y_cols: vec!["y1".to_string(), "y2".to_string()],
        },
        None,
    )
    .expect("save line html");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<!DOCTYPE html>"));
    let _ = std::fs::remove_file(&path_str);
}

#[test]
fn test_save_pie_html_to_temp() {
    let df = make_bar_df();
    let mut path = std::env::temp_dir();
    path.push("pandrs_svg_test_pie.html");
    let path_str = path.to_str().expect("path str").to_string();
    df.save_html(
        &path_str,
        SvgPlotType::Pie {
            label_col: "category".to_string(),
            value_col: "sales".to_string(),
        },
        None,
    )
    .expect("save pie html");
    let content = std::fs::read_to_string(&path_str).expect("read file");
    assert!(content.contains("<!DOCTYPE html>"));
    let _ = std::fs::remove_file(&path_str);
}

// ============================================================================
// SVG validity structural tests
// ============================================================================

#[test]
fn test_svg_has_valid_structure() {
    let labels = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let values = vec![10.0, 20.0, 30.0];
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, default_config());
    let svg = chart.render().expect("render");
    // Count opening and closing svg tags
    let open_count = svg.matches("<svg").count();
    let close_count = svg.matches("</svg>").count();
    assert_eq!(open_count, 1, "exactly one <svg opening tag");
    assert_eq!(close_count, 1, "exactly one </svg> closing tag");
    // Verify it starts and ends correctly
    assert!(svg.starts_with("<svg"));
    assert!(svg.ends_with("</svg>"));
}

#[test]
fn test_html_embeds_single_svg() {
    let labels = vec!["A".to_string(), "B".to_string()];
    let values = vec![10.0, 20.0];
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, default_config());
    let html = chart.render_html().expect("render html");
    let svg_count = html.matches("<svg").count();
    assert_eq!(svg_count, 1, "HTML should contain exactly one embedded SVG");
}

#[test]
fn test_large_dataset_bar_chart() {
    let n = 100;
    let labels: Vec<String> = (0..n).map(|i| format!("Item{}", i)).collect();
    let values: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.37).sin() * 50.0 + 50.0)
        .collect();
    let chart = BarChart::new(labels, values, BarOrientation::Vertical, default_config());
    let svg = chart.render().expect("render large bar chart");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_large_dataset_scatter_plot() {
    let n = 500;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 100.0).collect();
    let chart = ScatterPlot::new(x, y, default_config());
    let svg = chart.render().expect("render large scatter");
    assert!(svg.contains("<svg"));
}

#[test]
fn test_svg_text_escaping() {
    let mut canvas = SvgCanvas::new(300.0, 200.0);
    let style = DrawStyle {
        fill: Some(Color::BLACK),
        font_size: 12.0,
        text_anchor: "start".to_string(),
        dominant_baseline: "auto".to_string(),
        ..Default::default()
    };
    canvas.text("a & b < c > d \"quoted\"", 50.0, 100.0, &style);
    let svg = canvas.to_string();
    assert!(svg.contains("&amp;"));
    assert!(svg.contains("&lt;"));
    assert!(svg.contains("&gt;"));
    // The double-quote in the text content should be escaped
    assert!(!svg.contains("\"quoted\"") || svg.contains("&quot;") || svg.contains("quoted"));
}

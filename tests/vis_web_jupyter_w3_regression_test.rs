//! Wave 3 regression tests for `pandrs::vis` (svg/ascii charts), the
//! `web` WASM canvas backend's shared drawing helpers, and
//! `pandrs::jupyter`.
//!
//! See `/private/tmp/.../scratchpad/reports/distributed_gpu_full.md`
//! section 8 for the full audit this fixes. Highlights covered here:
//!
//! - `nice_ticks` (svg/charts.rs): an accumulator loop that could spin
//!   forever and exhaust memory on huge, narrow axis ranges (e.g.
//!   nanosecond timestamps) — MUST-fix, reproduced at a 2^60-scale range.
//! - All-negative bar charts drew entirely off-canvas.
//! - `Color`'s alpha channel was silently discarded by every shape draw.
//! - A pie chart with a single (or totally dominant) slice rendered
//!   nothing at all, per the SVG spec's handling of a coterminous arc.
//! - A NaN sample serialized as the literal text "NaN" into an SVG
//!   `points` attribute, corrupting the whole polyline/polygon.
//! - Jupyter's null-value count was a hardcoded 0 regardless of data.

use pandrs::{
    DataFrame, JupyterConfig, JupyterDisplay, LineSeries, Series, SvgBarChart, SvgBarOrientation,
    SvgLineChart, SvgPieChart,
};
use std::sync::mpsc;
use std::time::Duration;

/// Run `f` on a background thread and fail the test if it does not
/// finish within `timeout`. Used for the `nice_ticks` MUST-fix so that a
/// regression (an infinite loop reintroduced into the tick generator)
/// fails this test instead of hanging the whole test binary.
fn assert_completes_within<F: FnOnce() + Send + 'static>(timeout: Duration, f: F) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    rx.recv_timeout(timeout)
        .expect("operation did not complete within the timeout — possible infinite loop");
}

// ---------------------------------------------------------------------
// 1. nice_ticks infinite loop (MUST fix)
// ---------------------------------------------------------------------

/// `nice_ticks` is a private helper inside `vis::svg::charts`, so it is
/// exercised here through the public `SvgLineChart`/`SvgBarChart` API
/// with an axis domain around 2^60: at that magnitude the ULP is 256,
/// well above a "nice" step like 50, so the old `v += step` accumulator
/// loop was a silent no-op that pushed ticks forever.
#[test]
fn nice_ticks_terminates_on_huge_narrow_range() {
    assert_completes_within(Duration::from_secs(10), || {
        let base = 2f64.powi(60);
        let x = vec![base, base + 64.0, base + 128.0, base + 192.0, base + 256.0];
        let series = vec![LineSeries::new("s", vec![1.0, 2.0, 3.0, 2.0, 1.0])];
        let chart = SvgLineChart::new(x, series, Default::default());
        let svg = chart
            .render()
            .expect("line chart with a huge x-domain must render");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    });
}

/// Same reproduction directly from the audit report:
/// `nice_ticks(2^60, 2^60 + 256, 6)`, reached via a bar chart's y-axis
/// this time (a different call site, same helper).
#[test]
fn nice_ticks_terminates_on_reported_repro_case() {
    assert_completes_within(Duration::from_secs(10), || {
        let base = 2f64.powi(60);
        let labels = vec!["a".to_string(), "b".to_string()];
        // Values chosen so effective_min/effective_max span exactly the
        // reported [2^60, 2^60 + 256] range.
        let values = vec![base, base + 256.0];
        let chart = SvgBarChart::new(
            labels,
            values,
            SvgBarOrientation::Vertical,
            Default::default(),
        );
        let svg = chart
            .render()
            .expect("bar chart with a huge y-domain must render");
        assert!(svg.contains("<svg"));
    });
}

// ---------------------------------------------------------------------
// 2. All-negative bar chart in-bounds
// ---------------------------------------------------------------------

/// Extract every numeric value following `y="` in an SVG document. A
/// crude but effective way to check "did anything get drawn far outside
/// the canvas" without pulling in a full SVG/XML parser.
fn extract_y_values(svg: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut rest = svg;
    while let Some(pos) = rest.find("y=\"") {
        rest = &rest[pos + 3..];
        if let Some(end) = rest.find('"') {
            if let Ok(v) = rest[..end].parse::<f64>() {
                out.push(v);
            }
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    out
}

#[test]
fn all_negative_bar_chart_vertical_stays_in_bounds() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let values = vec![-20.0, -5.0, -10.0];
    let config = pandrs::SvgChartConfig::default();
    let height = config.height;
    let chart = SvgBarChart::new(labels, values, SvgBarOrientation::Vertical, config);
    let svg = chart.render().expect("all-negative bar chart must render");

    let ys = extract_y_values(&svg);
    assert!(
        !ys.is_empty(),
        "expected at least one y-coordinate in the SVG"
    );
    for y in ys {
        assert!(
            y >= -5.0 && y <= height + 5.0,
            "bar coordinate y={y} fell outside the canvas (height={height}); \
             all-negative data must not push the zero baseline off-canvas"
        );
    }
}

#[test]
fn all_negative_bar_chart_horizontal_stays_in_bounds() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let values = vec![-30.0, -12.0];
    let config = pandrs::SvgChartConfig::default();
    let width = config.width;
    let chart = SvgBarChart::new(labels, values, SvgBarOrientation::Horizontal, config);
    let svg = chart.render().expect("all-negative bar chart must render");

    // Horizontal bars vary in x, not y; check both so the test is
    // meaningful regardless of orientation-specific internals.
    let mut xs = Vec::new();
    let mut rest = svg.as_str();
    while let Some(pos) = rest.find("x=\"") {
        rest = &rest[pos + 3..];
        if let Some(end) = rest.find('"') {
            if let Ok(v) = rest[..end].parse::<f64>() {
                xs.push(v);
            }
            rest = &rest[end + 1..];
        } else {
            break;
        }
    }
    assert!(!xs.is_empty());
    for x in xs {
        assert!(
            x >= -5.0 && x <= width + 5.0,
            "bar coordinate x={x} fell outside the canvas (width={width})"
        );
    }
}

// ---------------------------------------------------------------------
// 3. Color alpha honored
// ---------------------------------------------------------------------

#[test]
fn line_chart_fill_area_honors_alpha() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let mut series = LineSeries::new("s", vec![1.0, 3.0, 2.0, 4.0]);
    series.fill_area = true;
    let chart = SvgLineChart::new(x, vec![series], Default::default());
    let svg = chart.render().expect("filled line chart must render");

    // The fill area uses Color::rgba(r, g, b, 50) (~20% opacity). Before
    // the fix, `DrawStyle::apply_to_attrs` only ever looked at the
    // separate `fill_opacity` style field (always left at its default
    // 1.0 by every caller in this module) and never consulted the
    // color's own alpha channel, so a translucent fill rendered fully
    // opaque and no `fill-opacity` attribute ever appeared anywhere in
    // this crate's SVG output.
    assert!(
        svg.contains("fill-opacity"),
        "translucent fill area did not produce a fill-opacity attribute:\n{svg}"
    );
}

// ---------------------------------------------------------------------
// 4. Single-slice pie chart
// ---------------------------------------------------------------------

#[test]
fn single_slice_pie_chart_emits_a_circle() {
    let labels = vec!["only".to_string()];
    let values = vec![42.0];
    let chart = SvgPieChart::new(labels, values, Default::default());
    let svg = chart.render().expect("single-slice pie chart must render");

    // A single slice sweeps the full 2*PI, making the arc's start and
    // end points coterminous; per the SVG spec such an arc renders
    // nothing. The only correct way to show 100% of a pie is a <circle>.
    assert!(
        svg.contains("<circle"),
        "single-slice pie chart did not emit a <circle>:\n{svg}"
    );
}

#[test]
fn dominant_slice_pie_chart_emits_a_circle() {
    // Not a literal single-element Vec this time: every other slice is
    // exactly zero, which is the more general form of the same bug
    // (frac == 1.0 for the one nonzero slice).
    let labels = vec![
        "all".to_string(),
        "none".to_string(),
        "also-none".to_string(),
    ];
    let values = vec![100.0, 0.0, 0.0];
    let chart = SvgPieChart::new(labels, values, Default::default());
    let svg = chart.render().expect("pie chart must render");
    assert!(
        svg.contains("<circle"),
        "dominant-slice pie chart did not emit a <circle>"
    );
}

// ---------------------------------------------------------------------
// 5. NaN filtered out of SVG polylines
// ---------------------------------------------------------------------

#[test]
fn line_chart_with_nan_sample_does_not_corrupt_svg() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let series = LineSeries::new("s", vec![1.0, f64::NAN, 3.0, 4.0, 5.0]);
    let chart = SvgLineChart::new(x, vec![series], Default::default());
    let svg = chart
        .render()
        .expect("line chart with one NaN sample must still render");

    // An unfiltered NaN coordinate serializes as the literal text "NaN"
    // inside the `points` attribute, which is invalid SVG; browsers
    // respond by dropping the entire element. The fixed renderer must
    // never emit that token anywhere.
    assert!(
        !svg.contains("NaN"),
        "SVG output contains the literal text \"NaN\", which corrupts point lists:\n{svg}"
    );
    // The other four (finite) points must still have been drawn as one
    // or more line segments, not silently dropped along with the NaN.
    assert!(
        svg.contains("<polyline") || svg.contains("<circle"),
        "no line segments were drawn for the finite points around the gap"
    );
}

#[test]
fn line_chart_all_nan_series_renders_without_polyline_corruption() {
    let x = vec![0.0, 1.0, 2.0];
    let series = LineSeries::new("s", vec![f64::NAN, f64::NAN, f64::NAN]);
    let chart = SvgLineChart::new(x, vec![series], Default::default());
    let svg = chart
        .render()
        .expect("an all-NaN series must still produce valid SVG");
    assert!(!svg.contains("NaN"));
    assert!(svg.contains("</svg>"));
}

// ---------------------------------------------------------------------
// 6. Real (not fabricated) Jupyter null counts
// ---------------------------------------------------------------------

#[test]
fn jupyter_summary_reports_real_null_count() {
    let mut df = DataFrame::new();
    // Two of five values are NaN — the missing-value sentinel this
    // DataFrame representation uses for floating-point columns.
    df.add_column(
        "value".to_string(),
        Series::new(
            vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
            Some("value".to_string()),
        )
        .expect("series"),
    )
    .expect("add column");

    let config = JupyterConfig::default();
    let html = df.to_summary_html(&config).expect("summary html");

    // Previously `estimate_null_count` always returned 0 regardless of
    // the data, so this string never appeared for any DataFrame with
    // real missing values.
    assert!(
        html.contains("Null values:</strong> 2"),
        "expected the real NaN count (2) to appear in the summary, got:\n{html}"
    );
    assert!(
        !html.contains("Null values:</strong> 0"),
        "null count must not be fabricated as 0 when the data has real NaNs"
    );
}

#[test]
fn jupyter_summary_reports_zero_nulls_when_data_is_actually_complete() {
    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        Series::new(vec![1.0, 2.0, 3.0], Some("value".to_string())).expect("series"),
    )
    .expect("add column");

    let config = JupyterConfig::default();
    let html = df.to_summary_html(&config).expect("summary html");
    assert!(html.contains("Null values:</strong> 0"));
    assert!(html.contains("Non-null values:</strong> 3"));
}

// ---------------------------------------------------------------------
// Bonus coverage for the rest of the audit fixes in this ownership area
// ---------------------------------------------------------------------

#[test]
fn jupyter_describe_to_json_reports_real_statistics() {
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string())).expect("series"),
    )
    .expect("add column");

    let json_text = df.describe_to_json().expect("describe_to_json");
    let parsed: serde_json::Value =
        serde_json::from_str(&json_text).expect("describe_to_json must produce valid JSON");
    // mean of [1,2,3,4,5] is exactly 3.0 — a real computed value, not a
    // placeholder string like "Summary statistics would be displayed
    // here...".
    let mean = parsed["x"]["mean"]
        .as_f64()
        .expect("expected a numeric mean for column 'x'");
    assert!((mean - 3.0).abs() < 1e-9, "expected mean 3.0, got {mean}");
    let count = parsed["x"]["count"]
        .as_u64()
        .expect("expected a numeric count");
    assert_eq!(count, 5);
}

#[test]
fn jupyter_widgets_get_distinct_dom_ids_for_same_shaped_frames() {
    // Two DataFrames with the *same* row count previously collided on
    // `id="pandrs-widget-<row_count>"`.
    let build = || {
        let mut df = DataFrame::new();
        df.add_column(
            "v".to_string(),
            Series::new(vec![1.0, 2.0], Some("v".to_string())).expect("series"),
        )
        .expect("add column");
        df
    };
    let df1 = build();
    let df2 = build();
    let config = JupyterConfig::default();
    let html1 = df1
        .to_interactive_widget(&config)
        .expect("interactive widget 1");
    let html2 = df2
        .to_interactive_widget(&config)
        .expect("interactive widget 2");

    let id1 = html1.find("pandrs-widget-").map(|i| &html1[i..i + 30]);
    let id2 = html2.find("pandrs-widget-").map(|i| &html2[i..i + 30]);
    assert_ne!(
        id1, id2,
        "two same-shaped DataFrames must not share a widget DOM id"
    );
}

#[test]
fn ascii_negative_bar_chart_shows_a_baseline_relative_bar() {
    use pandrs::vis::ascii::BarChart as AsciiBarChart;
    use pandrs::vis::Chart as _;

    let labels = ["a", "b"];
    let values = [-8.0, 4.0];
    let chart = AsciiBarChart::horizontal(&labels, &values);
    let rendered = chart.render();

    // Before the fix, a negative value's bar length was computed as
    // `(value / max_val * width).round() as usize`, and casting a
    // negative float to `usize` saturates to 0 — so the bar for "a" was
    // entirely empty while its printed value still read "-8.00". After
    // the fix, negative values draw a real (nonzero-width) bar on the
    // opposite side of a zero baseline from positive ones. Check the
    // rendered line for "a" contains at least one fill character
    // between the label and its printed value.
    let line_a = rendered
        .lines()
        .find(|l| l.trim_start().starts_with('a'))
        .expect("line for label 'a'");
    let bar_char = ['█', '#'];
    assert!(
        line_a.chars().any(|c| bar_char.contains(&c)),
        "negative-value bar must not render as an empty bar: {line_a:?}"
    );
}

#[test]
fn sparkline_dot_style_places_higher_values_visually_higher() {
    use pandrs::vis::{Sparkline, SparklineStyle};

    // A monotonically rising series.
    let spark = Sparkline::new(&[0.0, 1.0, 2.0, 3.0]).with_style(SparklineStyle::Dot);
    let rendered: String = spark.to_string_compact();
    let chars: Vec<char> = rendered.chars().collect();
    assert_eq!(chars.len(), 4);

    // Braille dot 1 (top-left, U+2801) must correspond to the highest
    // value and dot 7 (bottom-left, U+2840) to the lowest. Before the
    // fix this mapping was inverted, so a rising series rendered with
    // dots moving visually downward.
    assert_eq!(
        chars[0], '\u{2840}',
        "lowest value should use the bottom dot"
    );
    assert_eq!(chars[3], '\u{2801}', "highest value should use the top dot");
}

#[test]
fn sparkline_nan_is_distinguishable_from_the_minimum() {
    use pandrs::vis::{Sparkline, SparklineStyle};

    let spark = Sparkline::new(&[5.0, f64::NAN, 1.0]).with_style(SparklineStyle::Block);
    let rendered: String = spark.to_string_compact();
    let chars: Vec<char> = rendered.chars().collect();
    assert_eq!(chars.len(), 3);
    // The real minimum (1.0) gets the lowest block glyph; NaN must not
    // render identically to it.
    assert_ne!(
        chars[1], chars[2],
        "a NaN sample must not render identically to the series minimum"
    );
}

#[test]
fn ascii_histogram_ignores_nan_and_infinite_values() {
    use pandrs::vis::Chart as _;
    use pandrs::vis::Histogram as AsciiHistogram;

    let data = [1.0, 2.0, 3.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let hist = AsciiHistogram::new(&data, 3);
    // Must not panic (the old bin-index computation saturated NaN/-inf
    // into bin 0 and +inf into the last bin via a saturating cast) and
    // must still produce real output for the finite values.
    let rendered = hist.render();
    assert!(!rendered.is_empty());
    assert!(!rendered.contains("No data"));
}

#[cfg(feature = "visualization")]
#[test]
fn text_plot_xy_honors_path_under_default_terminal_format() {
    use pandrs::vis::plot_xy;
    use pandrs::{OutputFormat, PlotConfig};

    let mut path = std::env::temp_dir();
    path.push(format!("pandrs_w3_plot_xy_{}.txt", std::process::id()));
    let config = PlotConfig::default(); // OutputFormat::Terminal by default
    assert!(matches!(config.format, OutputFormat::Terminal));

    plot_xy(&[0.0f32, 1.0, 2.0], &[1.0f32, 2.0, 1.0], &path, config)
        .expect("plot_xy under Terminal format must still honor the path");

    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("plot_xy did not write to the path it was given (Terminal format): {e}")
    });
    assert!(!content.is_empty());
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "visualization")]
#[test]
fn shared_boxplot_stats_computes_real_quartiles_and_flags_outliers() {
    use pandrs::vis::plotters::boxplot_stats;

    // 0..=9 plus one extreme outlier.
    let mut values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    values.push(1000.0);
    let stats = boxplot_stats(&values).expect("non-empty finite data must produce stats");

    assert!(stats.q1 < stats.median);
    assert!(stats.median < stats.q3);
    assert!(
        stats.outliers.contains(&1000.0),
        "the planted extreme value must be reported as an outlier, not silently clipped"
    );
    assert!(
        stats.whisker_high < 1000.0,
        "the upper whisker must stay within the bulk of the data, not stretch to the outlier"
    );
}

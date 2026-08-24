//! Wave 3b regression tests for `pandrs::vis` (svg charts), extending
//! `tests/vis_web_jupyter_w3_regression_test.rs` with cases that file did
//! not cover.
//!
//! Context: `line_chart_all_nan_series_renders_without_polyline_corruption`
//! in the `_w3_` file was still failing at the start of this wave. Root
//! cause (see audit report section 8, finding 8): when every sample in a
//! series is NaN, `f64::min`/`f64::max` treat a NaN operand as absent and
//! return the *other* operand, so folding an all-NaN slice over the usual
//! `(f64::INFINITY, f64::min)` / `(f64::NEG_INFINITY, f64::max)` seeds
//! left the axis domain stuck at `(+inf, -inf)` instead of producing NaN
//! directly. That pair of infinities was just as toxic once it reached
//! `nice_ticks`: `(-inf).log10()` is NaN, every "nice" step derived from
//! it is NaN, and every generated tick inherited that NaN and printed the
//! literal text "NaN" into an axis `<text>` label — not into the
//! `points`/`d` attribute the existing `_w3_` tests' doc comments
//! describe, which a prior wave had already filtered correctly.
//!
//! The fix (`finite_domain` in `src/vis/svg/charts.rs`, plus a matching
//! guard in `nice_ticks` and finite-coordinate guards on
//! `SvgCanvas::circle`/`SvgCanvas::line` in `src/vis/svg/engine.rs`)
//! applies to `LineChart` and `ScatterPlot` alike. These tests cover:
//!
//! - The *positive* side of the `LineChart` fix: an all-NaN series must
//!   still draw a real, finite axis (not just suppress all output to
//!   satisfy a "no NaN text" check), while drawing no polyline for the
//!   phantom data.
//! - `ScatterPlot`, which the `_w3_` file does not exercise at all under
//!   NaN input. This matters independently of `LineChart`: `ScatterPlot`
//!   draws one marker per data point directly from `self.x_values`/
//!   `self.y_values` with no gap-splitting/segment logic, and its
//!   `Square` marker shape draws through `SvgCanvas::rect`, which —
//!   unlike `polyline`/`polygon`/`circle`/`line` — has no
//!   finite-coordinate guard of its own. Without a per-point skip in
//!   `ScatterPlot::render` itself, a NaN sample would still leak into the
//!   SVG for that one marker shape even with every other fix in place.

use pandrs::{MarkerShape, SvgChartConfig, SvgLineChart, SvgScatterPlot};

// ---------------------------------------------------------------------
// LineChart: all-NaN series must still render a real, finite axis.
// ---------------------------------------------------------------------

#[test]
fn line_chart_all_nan_series_still_draws_a_real_finite_axis() {
    use pandrs::LineSeries;

    // Both x and y are all-NaN here (not just y, as in the `_w3_` file's
    // `line_chart_all_nan_series_renders_without_polyline_corruption`):
    // with real, finite x-values present, that test's x-axis alone would
    // supply several finite tick labels regardless of whether the y-axis
    // fallback worked at all, which would let a weakened/removed y-axis
    // fallback slip through undetected. Making x-values NaN too means
    // *every* tick label on *either* axis, if any appear, must have come
    // from `finite_domain`'s fallback — there is no real data left to
    // supply one by coincidence.
    let x = vec![f64::NAN, f64::NAN, f64::NAN];
    let series = LineSeries::new("s", vec![f64::NAN, f64::NAN, f64::NAN]);
    let chart = SvgLineChart::new(x, vec![series], Default::default());
    let svg = chart
        .render()
        .expect("an all-NaN chart must still produce valid SVG");

    // The narrow regression check (no literal "NaN" text anywhere) is
    // already covered by the `_w3_` file. That check alone is
    // satisfiable by silently emitting nothing at all, which would not
    // actually be correct: a chart whose data is all-missing should
    // still draw *something* useful — real axis lines and finite tick
    // labels — rather than a blank plot area. Confirm the positive side
    // explicitly: more than just the legend's `<text>` (the series name)
    // was drawn, which is only possible if axis-tick labels rendered
    // using the finite fallback domain `finite_domain` substitutes for
    // the all-NaN x-values and y-values.
    let text_count = svg.matches("<text").count();
    assert!(
        text_count > 1,
        "expected real (finite) axis-tick labels to be drawn from the \
         fallback domain even though every x- and y-value was NaN, not \
         just the legend rendered on its own: found {text_count} <text> \
         element(s):\n{svg}"
    );
    // `finite_domain`'s fallback is `(0.0, 1.0)`; both axes' tick sets
    // include the value 0.0 exactly (`nice_ticks(0.0, 1.0, ...)` for x,
    // `nice_ticks(-0.05, 1.05, ...)` for the 5%-padded y), and
    // `format_tick` renders an exact-zero tick as the bare digit "0" via
    // its `v == 0.0` special case, not "0.00". Pin that concrete label
    // so this test fails loudly (not just on a vague count) if the
    // fallback domain ever changes to something that isn't finite.
    assert!(
        svg.contains(">0</text>"),
        "expected a tick label of exactly \"0\" from the finite fallback \
         domain:\n{svg}"
    );

    // And no polyline should be drawn for the all-NaN series itself: the
    // series has no finite points to connect, so a "line" of any kind
    // (even a fallback/placeholder one) would be fabricated data.
    assert!(
        !svg.contains("<polyline"),
        "an all-NaN series has no finite points to connect and must not \
         draw a polyline:\n{svg}"
    );
    assert!(svg.contains("</svg>"));
}

// ---------------------------------------------------------------------
// ScatterPlot: NaN points must be skipped, not leaked as literal "NaN".
// ---------------------------------------------------------------------

#[test]
fn scatter_plot_circle_markers_skip_nan_points_without_leaking_nan_text() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![1.0, f64::NAN, 3.0, 4.0];
    let chart = SvgScatterPlot::new(x, y, SvgChartConfig::default());
    let svg = chart
        .render()
        .expect("a scatter plot with one NaN point must still render");

    assert!(
        !svg.contains("NaN"),
        "SVG output contains the literal text \"NaN\":\n{svg}"
    );
    assert!(
        svg.contains("<circle"),
        "the three finite points must still be drawn as circle markers:\n{svg}"
    );
}

#[test]
fn scatter_plot_square_markers_skip_nan_points_without_leaking_nan_text() {
    // `Square` markers draw through `SvgCanvas::rect`, which — unlike
    // `polyline`/`polygon`/`circle`/`line` — carries no finite-coordinate
    // guard of its own. This case only passes if `ScatterPlot::render`
    // itself skips the non-finite point *before* calling any drawing
    // primitive, not merely because the primitive happens to be safe.
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![1.0, f64::NAN, 3.0, 4.0];
    let chart =
        SvgScatterPlot::new(x, y, SvgChartConfig::default()).with_marker_shape(MarkerShape::Square);
    let svg = chart
        .render()
        .expect("a scatter plot with one NaN point must still render");

    assert!(
        !svg.contains("NaN"),
        "SVG output contains the literal text \"NaN\" (Square markers draw \
         through SvgCanvas::rect, which has no finite-coordinate guard of \
         its own, so a NaN point must be filtered out by ScatterPlot \
         itself):\n{svg}"
    );
    assert!(
        svg.contains("<rect"),
        "the three finite points must still be drawn as square (rect) markers:\n{svg}"
    );
}

#[test]
fn scatter_plot_all_nan_data_renders_without_nan_leak_or_markers() {
    let x = vec![f64::NAN, f64::NAN, f64::NAN];
    let y = vec![f64::NAN, f64::NAN, f64::NAN];
    let chart = SvgScatterPlot::new(x, y, SvgChartConfig::default());
    let svg = chart
        .render()
        .expect("an all-NaN scatter plot must still produce valid SVG");

    assert!(
        !svg.contains("NaN"),
        "SVG output contains the literal text \"NaN\":\n{svg}"
    );
    // `ScatterPlot::render` draws no legend, so the only possible source
    // of a `<circle>` element is a data marker; with zero finite points,
    // none should be drawn at all.
    assert!(
        !svg.contains("<circle"),
        "no marker should be drawn for data with no finite points at all:\n{svg}"
    );
    assert!(svg.contains("</svg>"));
}

//! High-quality visualization using Plotters
//!
//! This module provides Plotters-based visualization functionality for DataFrame and Series.
//! It can generate high-quality graphs and visualizations in various formats.

#[cfg(feature = "visualization")]
use plotters::prelude::*;
#[cfg(feature = "visualization")]
use std::collections::HashMap;
#[cfg(feature = "visualization")]
use std::path::Path;

use crate::error::{PandRSError, Result};
#[cfg(feature = "visualization")]
use crate::vis::config::{PlotKind, PlotSettings};

/// Statistics for one Tukey-style box plot, computed once and shared by
/// every box-plot renderer in this crate (PNG/SVG file output, the web
/// canvas backend, and the backward-compatible `plotters_ext` module) so
/// they all agree on quartiles, whiskers, and outliers instead of each
/// carrying its own divergent — and in one case outlier-hiding — copy.
#[cfg(feature = "visualization")]
#[derive(Debug, Clone)]
pub struct BoxPlotStats {
    pub min: f64,
    pub q1: f64,
    pub median: f64,
    pub q3: f64,
    pub max: f64,
    /// Lower whisker end: the smallest data point >= Q1 - 1.5*IQR.
    pub whisker_low: f64,
    /// Upper whisker end: the largest data point <= Q3 + 1.5*IQR.
    pub whisker_high: f64,
    /// Points beyond the whiskers (Tukey's outlier rule), reported
    /// rather than silently clipped out of the whisker range.
    pub outliers: Vec<f64>,
}

/// Linear-interpolation quantile: the value at fractional rank
/// `q * (n-1)` of `sorted`, interpolating between the two nearest ranks.
/// This is the convention NumPy/pandas use by default (`interpolation="linear"`).
#[cfg(feature = "visualization")]
fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            if lo == hi {
                sorted[lo]
            } else {
                let frac = pos - lo as f64;
                sorted[lo] + (sorted[hi] - sorted[lo]) * frac
            }
        }
    }
}

/// Compute Tukey box-plot statistics for one category's values.
///
/// Returns `None` if there is no finite data. Quartiles use linear
/// interpolation; whiskers extend to the most extreme finite data point
/// within 1.5*IQR of the corresponding quartile, and anything further
/// out is returned as an outlier instead of being clipped away.
#[cfg(feature = "visualization")]
pub fn boxplot_stats(values: &[f64]) -> Option<BoxPlotStats> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let q1 = quantile_linear(&sorted, 0.25);
    let median = quantile_linear(&sorted, 0.5);
    let q3 = quantile_linear(&sorted, 0.75);
    let iqr = q3 - q1;
    let lower_fence = q1 - 1.5 * iqr;
    let upper_fence = q3 + 1.5 * iqr;

    let whisker_low = sorted
        .iter()
        .copied()
        .find(|&v| v >= lower_fence)
        .unwrap_or(min);
    let whisker_high = sorted
        .iter()
        .rev()
        .copied()
        .find(|&v| v <= upper_fence)
        .unwrap_or(max);

    let outliers: Vec<f64> = sorted
        .iter()
        .copied()
        .filter(|&v| v < whisker_low || v > whisker_high)
        .collect();

    Some(BoxPlotStats {
        min,
        q1,
        median,
        q3,
        max,
        whisker_low,
        whisker_high,
        outliers,
    })
}

/// Draw one box plot per category onto an already-configured cartesian
/// chart with a continuous `x in [0, categories.len())` axis (category
/// `i` centered at `x = i`).
///
/// Shared by every box-plot backend (PNG/SVG files, the web canvas, and
/// the backward-compatible `plotters_ext` module) so a real box width,
/// a real (non-zero-length) median line, and correctly centered whisker
/// caps only have to be implemented once.
#[cfg(feature = "visualization")]
pub fn draw_boxplot_series<DB: plotters::prelude::DrawingBackend>(
    chart: &mut plotters::chart::ChartContext<
        DB,
        plotters::coord::cartesian::Cartesian2d<
            plotters::coord::types::RangedCoordf64,
            plotters::coord::types::RangedCoordf64,
        >,
    >,
    categories: &[String],
    category_map: &std::collections::HashMap<String, Vec<f64>>,
    color_palette: &[(u8, u8, u8)],
) -> Result<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    use plotters::prelude::*;

    const BOX_WIDTH: f64 = 0.6;
    let cap_half = BOX_WIDTH / 4.0;

    for (i, category) in categories.iter().enumerate() {
        let values = match category_map.get(category) {
            Some(v) => v,
            None => continue,
        };
        let stats = match boxplot_stats(values) {
            Some(s) => s,
            None => continue,
        };
        let x = i as f64;
        let palette_len = color_palette.len().max(1);
        let (r, g, b) = color_palette
            .get(i % palette_len)
            .copied()
            .unwrap_or((70, 130, 180));
        let color = RGBColor(r, g, b);

        // Box: a real rectangle spanning [Q1, Q3], not a zero-width sliver.
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (x - BOX_WIDTH / 2.0, stats.q1),
                (x + BOX_WIDTH / 2.0, stats.q3),
            ],
            color.mix(0.25).filled(),
        )))?;

        // Median: two distinct endpoints spanning the box width, not a
        // zero-length segment collapsed onto a single point.
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (x - BOX_WIDTH / 2.0, stats.median),
                (x + BOX_WIDTH / 2.0, stats.median),
            ],
            color.stroke_width(2),
        )))?;

        // Whiskers
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, stats.q3), (x, stats.whisker_high)],
            color.stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, stats.q1), (x, stats.whisker_low)],
            color.stroke_width(1),
        )))?;

        // Whisker caps, centered on this category's own column (not
        // shifted into the neighboring category by a truncating
        // float-to-usize cast).
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (x - cap_half, stats.whisker_low),
                (x + cap_half, stats.whisker_low),
            ],
            color.stroke_width(1),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (x - cap_half, stats.whisker_high),
                (x + cap_half, stats.whisker_high),
            ],
            color.stroke_width(1),
        )))?;

        // Outliers beyond the whiskers, shown instead of silently
        // vanishing when the whiskers are drawn IQR-bounded.
        if !stats.outliers.is_empty() {
            chart.draw_series(
                stats
                    .outliers
                    .iter()
                    .map(|&v| Circle::new((x, v), 3, color.filled())),
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "visualization")]
pub use self::backend::plot_boxplot_png;
#[cfg(feature = "visualization")]
pub use self::backend::plot_boxplot_svg;
#[cfg(feature = "visualization")]
pub use self::backend::plot_histogram_png;
#[cfg(feature = "visualization")]
pub use self::backend::plot_histogram_svg;
#[cfg(feature = "visualization")]
pub use self::backend::plot_multi_series_png;
#[cfg(feature = "visualization")]
pub use self::backend::plot_multi_series_svg;
#[cfg(feature = "visualization")]
pub use self::backend::plot_series_xy_png;
#[cfg(feature = "visualization")]
pub use self::backend::plot_series_xy_svg;

/// Backend module for implementing plotters-based visualization
#[cfg(feature = "visualization")]
pub mod backend {
    use super::*;

    /// Plot XY data to PNG using Plotters
    pub fn plot_series_xy_png<P: AsRef<Path>>(
        x: &[f64],
        y: &[f64],
        path: P,
        settings: &PlotSettings,
        series_name: &str,
    ) -> Result<()> {
        // Implementation using Plotters backend
        let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Determine min/max values for both axes
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Add margins
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let x_min = x_min - x_range * 0.05;
        let x_max = x_max + x_range * 0.05;
        let y_min = y_min - y_range * 0.05;
        let y_max = y_max + y_range * 0.05;

        // Create chart context
        let mut chart = ChartBuilder::on(&root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        // Add grid if specified
        if settings.show_grid {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .disable_mesh()
                .draw()?;
        }

        // Define color
        let color = RGBColor(
            settings.color_palette[0].0,
            settings.color_palette[0].1,
            settings.color_palette[0].2,
        );

        // Draw series based on plot type
        match settings.plot_kind {
            PlotKind::Line => {
                let line_series =
                    LineSeries::new(x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)), color);

                if settings.show_legend {
                    chart
                        .draw_series(line_series)?
                        .label(series_name)
                        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
                } else {
                    chart.draw_series(line_series)?;
                }
            }
            PlotKind::Scatter => {
                let scatter_series = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 3, color.filled()));

                if settings.show_legend {
                    chart
                        .draw_series(scatter_series)?
                        .label(series_name)
                        .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
                } else {
                    chart.draw_series(scatter_series)?;
                }
            }
            PlotKind::Bar => {
                // For a bar chart, we use indices as x-values
                let num_bars = x.len() as f64;
                let bars = x
                    .iter()
                    .zip(y.iter())
                    .enumerate()
                    .map(|(_i, (&x_val, &y))| {
                        let bar_width = x_range / num_bars * 0.8;
                        let x0 = x_val - bar_width / 2.0;
                        let x1 = x_val + bar_width / 2.0;

                        Rectangle::new([(x0, 0.0), (x1, y)], color.filled())
                    });

                if settings.show_legend {
                    chart
                        .draw_series(bars)?
                        .label(series_name)
                        .legend(move |(x, y)| {
                            Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
                        });
                } else {
                    chart.draw_series(bars)?;
                }
            }
            PlotKind::Area => {
                let area_series = AreaSeries::new(
                    x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)),
                    0.0,
                    color.mix(0.2),
                )
                .border_style(color);

                if settings.show_legend {
                    chart
                        .draw_series(area_series)?
                        .label(series_name)
                        .legend(move |(x, y)| {
                            Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.2).filled())
                        });
                } else {
                    chart.draw_series(area_series)?;
                }
            }
            _ => {
                return Err(PandRSError::NotImplemented(format!(
                    "Plot kind {:?} not supported for this function",
                    settings.plot_kind
                )));
            }
        }

        // Add legend if specified
        if settings.show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;

        Ok(())
    }

    /// Plot XY data to SVG using Plotters
    pub fn plot_series_xy_svg<P: AsRef<Path>>(
        x: &[f64],
        y: &[f64],
        path: P,
        settings: &PlotSettings,
        series_name: &str,
    ) -> Result<()> {
        // SVG implementation (very similar to PNG but with different backend)
        let root =
            SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();

        root.fill(&WHITE)?;

        // Determine min/max values for both axes
        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Add margins
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let x_min = x_min - x_range * 0.05;
        let x_max = x_max + x_range * 0.05;
        let y_min = y_min - y_range * 0.05;
        let y_max = y_max + y_range * 0.05;

        // Similar implementation as PNG but using SVG backend
        // Create chart context
        let mut chart = ChartBuilder::on(&root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        // Add grid if specified
        if settings.show_grid {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .disable_mesh()
                .draw()?;
        }

        // Define color
        let color = RGBColor(
            settings.color_palette[0].0,
            settings.color_palette[0].1,
            settings.color_palette[0].2,
        );

        // Draw series based on plot type
        match settings.plot_kind {
            PlotKind::Line => {
                let line_series =
                    LineSeries::new(x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)), color);

                if settings.show_legend {
                    chart
                        .draw_series(line_series)?
                        .label(series_name)
                        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
                } else {
                    chart.draw_series(line_series)?;
                }
            }
            PlotKind::Scatter => {
                let scatter_series = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 3, color.filled()));

                if settings.show_legend {
                    chart
                        .draw_series(scatter_series)?
                        .label(series_name)
                        .legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
                } else {
                    chart.draw_series(scatter_series)?;
                }
            }
            // Other plot types would follow the same pattern
            _ => {
                return Err(PandRSError::NotImplemented(format!(
                    "Plot kind {:?} not supported for this function in SVG format",
                    settings.plot_kind
                )));
            }
        }

        // Add legend if specified
        if settings.show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;

        Ok(())
    }

    /// Plot multiple series to PNG using Plotters
    pub fn plot_multi_series_png<P: AsRef<Path>>(
        series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Implementation for multiple series
        // Similar to plot_series_xy_png but handles multiple series
        let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Determine global min/max values across all series
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for (_, x, y, _) in &series_data {
            let x_min_local = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let x_max_local = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let y_min_local = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let y_max_local = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            x_min = x_min.min(x_min_local);
            x_max = x_max.max(x_max_local);
            y_min = y_min.min(y_min_local);
            y_max = y_max.max(y_max_local);
        }

        // Add margins
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let x_min = x_min - x_range * 0.05;
        let x_max = x_max + x_range * 0.05;
        let y_min = y_min - y_range * 0.05;
        let y_max = y_max + y_range * 0.05;

        // Create chart context
        let mut chart = ChartBuilder::on(&root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        // Add grid if specified
        if settings.show_grid {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc(&settings.y_label)
                .disable_mesh()
                .draw()?;
        }

        // Draw each series
        for (name, x, y, rgb) in series_data {
            let color = RGBColor(rgb.0, rgb.1, rgb.2);

            match settings.plot_kind {
                PlotKind::Line => {
                    let line_series =
                        LineSeries::new(x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)), color);

                    if settings.show_legend {
                        chart
                            .draw_series(line_series)?
                            .label(&name)
                            .legend(move |(x, y)| {
                                PathElement::new(vec![(x, y), (x + 20, y)], color)
                            });
                    } else {
                        chart.draw_series(line_series)?;
                    }
                }
                // Add support for other plot types as needed
                _ => {
                    return Err(PandRSError::NotImplemented(format!(
                        "Plot kind {:?} not supported for multiple series",
                        settings.plot_kind
                    )));
                }
            }
        }

        // Add legend if specified
        if settings.show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;

        Ok(())
    }

    /// Plot multiple series to SVG using Plotters
    pub fn plot_multi_series_svg<P: AsRef<Path>>(
        _series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Similar to plot_multi_series_png but with SVG backend
        let _root =
            SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();

        // Rest of implementation would be similar to plot_multi_series_png
        // with SVG-specific adaptations

        // For brevity, we'll just return a not implemented error for now
        Err(PandRSError::NotImplemented(
            "SVG multi-series plotting not fully implemented yet".to_string(),
        ))
    }

    /// Plot histogram to PNG using Plotters
    pub fn plot_histogram_png<P: AsRef<Path>>(
        values: &[f64],
        bins: usize,
        path: P,
        settings: &PlotSettings,
        series_name: &str,
    ) -> Result<()> {
        // Histogram implementation
        // We need to bin the data first
        if values.is_empty() {
            return Err(PandRSError::Empty("No data to plot".to_string()));
        }
        if bins == 0 {
            return Err(PandRSError::InvalidInput(
                "Histogram: bins must be greater than 0".to_string(),
            ));
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // A constant series has max_val == min_val; fall back to a
        // single unit-width bin instead of dividing by zero.
        let bin_width = if (max_val - min_val).abs() < f64::EPSILON {
            1.0
        } else {
            (max_val - min_val) / bins as f64
        };
        let mut histogram = vec![0; bins];

        for &value in values {
            let bin = ((value - min_val) / bin_width).floor() as usize;
            let bin = bin.min(bins - 1); // Ensure within bounds
            histogram[bin] += 1;
        }

        // Now plot the histogram
        let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Find max bin height for y-axis scaling
        let max_height = *histogram.iter().max().unwrap_or(&1) as f64;

        // Create chart context
        let mut chart = ChartBuilder::on(&root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                (min_val - bin_width * 0.1)..(max_val + bin_width * 0.1),
                0.0..(max_height * 1.1),
            )?;

        // Add grid if specified
        if settings.show_grid {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc("Frequency")
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .x_desc(&settings.x_label)
                .y_desc("Frequency")
                .disable_mesh()
                .draw()?;
        }

        // Define color
        let color = RGBColor(
            settings.color_palette[0].0,
            settings.color_palette[0].1,
            settings.color_palette[0].2,
        );

        // Draw histogram bars
        let bars = histogram.iter().enumerate().map(|(i, &count)| {
            let x0 = min_val + i as f64 * bin_width;
            let x1 = x0 + bin_width * 0.8; // Slight space between bars
            let y = count as f64;

            Rectangle::new([(x0, 0.0), (x1, y)], color.filled())
        });

        if settings.show_legend {
            chart
                .draw_series(bars)?
                .label(series_name)
                .legend(move |(x, y)| {
                    Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled())
                });
        } else {
            chart.draw_series(bars)?;
        }

        // Add legend if specified
        if settings.show_legend {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }

        root.present()?;

        Ok(())
    }

    /// Plot histogram to SVG using Plotters
    pub fn plot_histogram_svg<P: AsRef<Path>>(
        _values: &[f64],
        _bins: usize,
        _path: P,
        _settings: &PlotSettings,
        _series_name: &str,
    ) -> Result<()> {
        // Similar to PNG implementation but with SVG backend
        Err(PandRSError::NotImplemented(
            "SVG histogram plotting not fully implemented yet".to_string(),
        ))
    }

    /// Plot box plot to PNG using Plotters
    pub fn plot_boxplot_png<P: AsRef<Path>>(
        category_map: &HashMap<String, Vec<f64>>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
            .into_drawing_area();
        root.fill(&WHITE)?;
        draw_boxplot_chart(&root, category_map, settings)
    }

    /// Plot box plot to SVG using Plotters
    pub fn plot_boxplot_svg<P: AsRef<Path>>(
        category_map: &HashMap<String, Vec<f64>>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        let root =
            SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();
        root.fill(&WHITE)?;
        draw_boxplot_chart(&root, category_map, settings)
    }

    /// Shared box-plot chart setup (axis, mesh, category labels) for any
    /// drawing backend; the per-category box/whisker/median/outlier
    /// drawing itself is delegated to [`super::draw_boxplot_series`] so
    /// PNG, SVG, and the web canvas backend all render identically.
    fn draw_boxplot_chart<DB: DrawingBackend>(
        root: &DrawingArea<DB, plotters::coord::Shift>,
        category_map: &HashMap<String, Vec<f64>>,
        settings: &PlotSettings,
    ) -> Result<()>
    where
        DB::ErrorType: std::error::Error + Send + Sync + 'static,
    {
        if category_map.is_empty() {
            return Err(PandRSError::Empty("No data to plot".to_string()));
        }

        let mut categories: Vec<String> = category_map
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, _)| k.clone())
            .collect();
        categories.sort();
        if categories.is_empty() {
            return Err(PandRSError::Empty("No data to plot".to_string()));
        }

        let all_stats: Vec<super::BoxPlotStats> = categories
            .iter()
            .filter_map(|c| category_map.get(c))
            .filter_map(|v| super::boxplot_stats(v))
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
        let y_min = y_min - y_range * 0.1;
        let y_max = y_max + y_range * 0.1;

        let mut chart = ChartBuilder::on(root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(-0.5f64..(categories.len() as f64 - 0.5), y_min..y_max)?;

        let label_formatter = |x: &f64| {
            let i = x.round() as isize;
            if i >= 0 && (i as usize) < categories.len() {
                categories[i as usize].clone()
            } else {
                String::new()
            }
        };
        let mut mesh = chart.configure_mesh();
        mesh.x_labels(categories.len())
            .x_label_formatter(&label_formatter)
            .y_desc(&settings.y_label);
        if !settings.show_grid {
            mesh.disable_mesh();
        }
        mesh.draw()?;

        super::draw_boxplot_series(
            &mut chart,
            &categories,
            category_map,
            &settings.color_palette,
        )
    }
}

/// Fallback implementations when visualization is not enabled
#[cfg(not(feature = "visualization"))]
pub mod backend {
    use super::*;
    use std::collections::HashMap;
    use std::path::Path;

    macro_rules! visualization_not_enabled {
        ($name:ident, $($arg:ident: $type:ty),*) => {
            pub fn $name<P: AsRef<Path>>($($arg: $type,)* _path: P) -> Result<()> {
                Err(PandRSError::FeatureNotAvailable("Visualization feature is not enabled. Recompile with --feature visualization".to_string()))
            }
        };
    }

    visualization_not_enabled!(plot_series_xy_png, _x: &[f64], _y: &[f64], _settings: &crate::vis::config::PlotSettings, _series_name: &str);
    visualization_not_enabled!(plot_series_xy_svg, _x: &[f64], _y: &[f64], _settings: &crate::vis::config::PlotSettings, _series_name: &str);
    visualization_not_enabled!(plot_multi_series_png, _series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>, _settings: &crate::vis::config::PlotSettings);
    visualization_not_enabled!(plot_multi_series_svg, _series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>, _settings: &crate::vis::config::PlotSettings);
    visualization_not_enabled!(plot_histogram_png, _values: &[f64], _bins: usize, _settings: &crate::vis::config::PlotSettings, _series_name: &str);
    visualization_not_enabled!(plot_histogram_svg, _values: &[f64], _bins: usize, _settings: &crate::vis::config::PlotSettings, _series_name: &str);
    visualization_not_enabled!(plot_boxplot_png, _category_map: &HashMap<String, Vec<f64>>, _settings: &crate::vis::config::PlotSettings);
    visualization_not_enabled!(plot_boxplot_svg, _category_map: &HashMap<String, Vec<f64>>, _settings: &crate::vis::config::PlotSettings);
}

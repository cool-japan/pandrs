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
use crate::vis::config::{OutputType, PlotKind, PlotSettings};

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
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
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
                        .legend(|(x, y)| Circle::new((x, y), 3, color.filled()));
                } else {
                    chart.draw_series(scatter_series)?;
                }
            }
            PlotKind::Bar => {
                // For a bar chart, we use indices as x-values
                let bars = x.iter().zip(y.iter()).enumerate().map(|(i, (&x, &y))| {
                    let bar_width = x_range / x.len() as f64 * 0.8;
                    let x0 = x - bar_width / 2.0;
                    let x1 = x + bar_width / 2.0;

                    Rectangle::new([(x0, 0.0), (x1, y)], color.filled())
                });

                if settings.show_legend {
                    chart
                        .draw_series(bars)?
                        .label(series_name)
                        .legend(|(x, y)| {
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
                        .legend(|(x, y)| {
                            Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.2).filled())
                        });
                } else {
                    chart.draw_series(area_series)?;
                }
            }
            _ => {
                return Err(PandRSError::Feature(format!(
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
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
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
                        .legend(|(x, y)| Circle::new((x, y), 3, color.filled()));
                } else {
                    chart.draw_series(scatter_series)?;
                }
            }
            // Other plot types would follow the same pattern
            _ => {
                return Err(PandRSError::Feature(format!(
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
                            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
                    } else {
                        chart.draw_series(line_series)?;
                    }
                }
                // Add support for other plot types as needed
                _ => {
                    return Err(PandRSError::Feature(format!(
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
        series_data: Vec<(String, Vec<f64>, Vec<f64>, (u8, u8, u8))>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Similar to plot_multi_series_png but with SVG backend
        let root =
            SVGBackend::new(path.as_ref(), (settings.width, settings.height)).into_drawing_area();

        // Rest of implementation would be similar to plot_multi_series_png
        // with SVG-specific adaptations

        // For brevity, we'll just return a not implemented error for now
        Err(PandRSError::Feature(
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

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let bin_width = (max_val - min_val) / bins as f64;
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
                .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.filled()));
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
        values: &[f64],
        bins: usize,
        path: P,
        settings: &PlotSettings,
        series_name: &str,
    ) -> Result<()> {
        // Similar to PNG implementation but with SVG backend
        Err(PandRSError::Feature(
            "SVG histogram plotting not fully implemented yet".to_string(),
        ))
    }

    /// Plot box plot to PNG using Plotters
    pub fn plot_boxplot_png<P: AsRef<Path>>(
        category_map: &HashMap<String, Vec<f64>>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Box plot implementation
        if category_map.is_empty() {
            return Err(PandRSError::Empty("No data to plot".to_string()));
        }

        // Calculate statistics for each category
        let mut categories = Vec::new();
        let mut stats = Vec::new();

        for (cat, values) in category_map {
            if values.is_empty() {
                continue;
            }

            categories.push(cat.clone());

            // Sort values for quantile calculation
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate statistics
            let len = sorted_values.len();
            let median = if len % 2 == 0 {
                (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
            } else {
                sorted_values[len / 2]
            };

            let q1_idx = len / 4;
            let q3_idx = len * 3 / 4;
            let q1 = sorted_values[q1_idx];
            let q3 = sorted_values[q3_idx];

            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;

            let min = *sorted_values
                .iter()
                .find(|&&x| x >= lower_bound)
                .unwrap_or(&sorted_values[0]);
            let max = *sorted_values
                .iter()
                .rev()
                .find(|&&x| x <= upper_bound)
                .unwrap_or(&sorted_values[len - 1]);

            stats.push((min, q1, median, q3, max));
        }

        // Now plot the box plot
        let root = BitMapBackend::new(path.as_ref(), (settings.width, settings.height))
            .into_drawing_area();

        root.fill(&WHITE)?;

        // Determine global min/max for y-axis
        let y_min = stats
            .iter()
            .map(|&(min, _, _, _, _)| min)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let y_max = stats
            .iter()
            .map(|&(_, _, _, _, max)| max)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        // Add margin
        let y_range = y_max - y_min;
        let y_min = y_min - y_range * 0.1;
        let y_max = y_max + y_range * 0.1;

        // Create chart context
        let mut chart = ChartBuilder::on(&root)
            .caption(&settings.title, ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d((0..categories.len()).into_segmented(), y_min..y_max)?;

        // Add grid if specified
        if settings.show_grid {
            chart
                .configure_mesh()
                .disable_x_mesh()
                .x_labels(categories.len())
                .x_label_formatter(&|idx| {
                    if *idx < categories.len() {
                        categories[*idx].to_string()
                    } else {
                        "".to_string()
                    }
                })
                .y_desc(&settings.y_label)
                .draw()?;
        } else {
            chart
                .configure_mesh()
                .disable_mesh()
                .x_labels(categories.len())
                .x_label_formatter(&|idx| {
                    if *idx < categories.len() {
                        categories[*idx].to_string()
                    } else {
                        "".to_string()
                    }
                })
                .y_desc(&settings.y_label)
                .draw()?;
        }

        // Define color rotation function
        let category_color = |idx: usize| {
            let color_idx = idx % settings.color_palette.len();
            let (r, g, b) = settings.color_palette[color_idx];
            RGBColor(r, g, b)
        };

        // Draw box plots
        for (i, &(min, q1, median, q3, max)) in stats.iter().enumerate() {
            let color = category_color(i);

            // Draw box
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (SegmentValue::CenterOf(i), q1),
                    (SegmentValue::CenterOf(i), q3),
                ],
                color.mix(0.3).filled(),
            )))?;

            // Draw median line
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (SegmentValue::CenterOf(i), median),
                    (SegmentValue::CenterOf(i), median),
                ],
                color.stroke_width(3),
            )))?;

            // Draw whiskers
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (SegmentValue::CenterOf(i), min),
                    (SegmentValue::CenterOf(i), q1),
                ],
                color.stroke_width(1),
            )))?;

            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (SegmentValue::CenterOf(i), q3),
                    (SegmentValue::CenterOf(i), max),
                ],
                color.stroke_width(1),
            )))?;

            // Draw caps
            let width = 0.2;
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (SegmentValue::Exact(i as i32) - width, min),
                    (SegmentValue::Exact(i as i32) + width, min),
                ],
                color.stroke_width(1),
            )))?;

            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (SegmentValue::Exact(i as i32) - width, max),
                    (SegmentValue::Exact(i as i32) + width, max),
                ],
                color.stroke_width(1),
            )))?;
        }

        root.present()?;

        Ok(())
    }

    /// Plot box plot to SVG using Plotters
    pub fn plot_boxplot_svg<P: AsRef<Path>>(
        category_map: &HashMap<String, Vec<f64>>,
        path: P,
        settings: &PlotSettings,
    ) -> Result<()> {
        // Similar to PNG implementation but with SVG backend
        Err(PandRSError::Feature(
            "SVG box plot not fully implemented yet".to_string(),
        ))
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

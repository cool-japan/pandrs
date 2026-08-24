//! Text-based visualization functionality
//!
//! This module provides text-based visualization capabilities using the textplots library.
//! These visualizations are lightweight and can be displayed directly in the terminal.

#[cfg(feature = "visualization")]
use std::fs::File;
#[cfg(feature = "visualization")]
use std::io::Write;
use std::path::Path;
#[cfg(feature = "visualization")]
use textplots::{Chart, Plot, Shape};

use crate::error::{PandRSError, Result};
#[cfg(feature = "visualization")]
use crate::vis::config::{OutputFormat, PlotConfig, PlotType};

/// Basic plot function for XY coordinates (text-based)
#[cfg(feature = "visualization")]
pub fn plot_xy<P: AsRef<Path>>(x: &[f32], y: &[f32], path: P, config: PlotConfig) -> Result<()> {
    if x.len() != y.len() {
        return Err(PandRSError::Consistency(
            "X and Y lengths do not match".to_string(),
        ));
    }

    // Do nothing if data is empty
    if x.is_empty() {
        return Err(PandRSError::Empty("No data to plot".to_string()));
    }

    // Create points
    let points: Vec<(f32, f32)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

    // Domain from the actual min/max of x, not the first/last elements:
    // callers are not required to pass x in sorted order (e.g. a scatter
    // plot), and using x[0]/x[x.len()-1] as the axis bounds in that case
    // silently gives a reversed or truncated domain that omits part of
    // the data.
    let x_min = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Create chart
    let mut chart_string = String::new();
    chart_string.push_str(&format!("=== {} ===\n", config.title));
    chart_string.push_str(&format!(
        "X-axis: {}, Y-axis: {}\n\n",
        config.x_label, config.y_label
    ));

    // Draw plot
    let chart_result = match config.plot_type {
        PlotType::Line => Chart::new(config.width as u32, config.height as u32, x_min, x_max)
            .lineplot(&Shape::Lines(&points))
            .to_string(),
        PlotType::Scatter | PlotType::Points => {
            Chart::new(config.width as u32, config.height as u32, x_min, x_max)
                .lineplot(&Shape::Points(&points))
                .to_string()
        }
    };

    chart_string.push_str(&chart_result);

    // Output. `path` is honored regardless of `format`: `Terminal` is the
    // default `OutputFormat`, and silently ignoring a caller-supplied
    // path just because that default was in effect meant `plot_xy(x, y,
    // "chart.txt", PlotConfig::default())` printed to stdout and never
    // touched "chart.txt", contradicting the function's own signature.
    let mut file = File::create(path).map_err(PandRSError::Io)?;
    file.write_all(chart_string.as_bytes())
        .map_err(PandRSError::Io)?;
    if let OutputFormat::Terminal = config.format {
        println!("{}", chart_string);
    }
    Ok(())
}

/// Fallback implementation when visualization is not available
#[cfg(not(feature = "visualization"))]
pub fn plot_xy<P: AsRef<Path>>(
    _x: &[f32],
    _y: &[f32],
    _path: P,
    _config: crate::vis::config::PlotConfig,
) -> Result<()> {
    Err(PandRSError::FeatureNotAvailable(
        "Visualization feature is not enabled. Recompile with --feature visualization".to_string(),
    ))
}

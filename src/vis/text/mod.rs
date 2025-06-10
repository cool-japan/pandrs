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

    // Create chart
    let mut chart_string = String::new();
    chart_string.push_str(&format!("=== {} ===\n", config.title));
    chart_string.push_str(&format!(
        "X-axis: {}, Y-axis: {}\n\n",
        config.x_label, config.y_label
    ));

    // Draw plot
    let chart_result = match config.plot_type {
        PlotType::Line => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Lines(&points))
        .to_string(),
        PlotType::Scatter | PlotType::Points => Chart::new(
            config.width as u32,
            config.height as u32,
            x[0],
            x[x.len() - 1],
        )
        .lineplot(&Shape::Points(&points))
        .to_string(),
    };

    chart_string.push_str(&chart_result);

    // Output
    match config.format {
        OutputFormat::Terminal => {
            println!("{}", chart_string);
            Ok(())
        }
        OutputFormat::TextFile => {
            let mut file = File::create(path).map_err(PandRSError::Io)?;
            file.write_all(chart_string.as_bytes())
                .map_err(PandRSError::Io)?;
            Ok(())
        }
    }
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

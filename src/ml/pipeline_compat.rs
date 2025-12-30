//! Compatibility layer for old Pipeline API
//!
//! This module provides backward compatibility for the old Pipeline API.
//! It allows code that used the old Transformer trait to continue working.

use crate::column::ColumnTrait;
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::preprocessing::{MinMaxScaler, StandardScaler};
use crate::optimized::OptimizedDataFrame;
use std::collections::HashMap;

/// Trait for data transformers (backward compatibility version)
pub trait Transformer: std::fmt::Debug {
    /// Transform data
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame>;

    /// Learn from data and then transform it
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame>;

    /// Learn from data
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()>;
}

// We don't need an explicit Debug implementation since Transformer requires
// Debug and both StandardScaler and MinMaxScaler already implement Debug

// Implement Transformer for StandardScaler
impl Transformer for StandardScaler {
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // Create a new OptimizedDataFrame to hold the result
        let mut result = OptimizedDataFrame::new();

        // For each column in the original DataFrame
        let column_names: Vec<String> = df.column_names().to_vec();
        for col_name in &column_names {
            // Check if this column should be scaled
            let should_scale = if let Some(columns) = &self.columns {
                columns.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // If this column should be scaled
            if should_scale && column_view.as_float64().is_some() {
                // Get the column data
                let float_values = column_view.as_float64().unwrap();

                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                // Get the mean and std
                let mean = if let Some(means) = &self.means {
                    means.get(col_name.as_str()).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let std_dev = if let Some(stds) = &self.stds {
                    stds.get(col_name.as_str()).copied().unwrap_or(1.0)
                } else {
                    1.0
                };

                // Scale the values
                let scaled_values = if std_dev > 1e-10 {
                    values.iter().map(|&x| (x - mean) / std_dev).collect()
                } else {
                    vec![0.0; values.len()]
                };

                // Create a new column
                let scaled_column =
                    crate::column::Float64Column::with_name(scaled_values, col_name.to_string());
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                // Create a column copy
                if let Some(float_values) = column_view.as_float64() {
                    // Handle float column
                    let mut values = Vec::new();
                    for i in 0..float_values.len() {
                        if let Ok(Some(val)) = float_values.get(i) {
                            values.push(val);
                        } else {
                            values.push(0.0); // Default value
                        }
                    }
                    let col = crate::column::Float64Column::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::Float64(col))?;
                } else if let Some(string_values) = column_view.as_string() {
                    // Handle string column
                    let mut values = Vec::new();
                    for i in 0..string_values.len() {
                        if let Ok(Some(val)) = string_values.get(i) {
                            values.push(val.to_string());
                        } else {
                            values.push(String::new()); // Default value
                        }
                    }
                    let col = crate::column::StringColumn::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::String(col))?;
                }
                // Skip other column types for simplicity
            }
        }

        Ok(result)
    }

    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // Instead of calling normal fit and transform, we'll implement directly

        // First, fit
        let column_names: Vec<String> = match &self.columns {
            Some(cols) => cols.clone(),
            None => df.column_names().to_vec(),
        };

        let mut means = HashMap::new();
        let mut stds = HashMap::new();

        for col_name in &column_names {
            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                if values.is_empty() {
                    continue;
                }

                // Calculate mean
                let sum: f64 = values.iter().sum();
                let mean = sum / values.len() as f64;
                means.insert(col_name.clone(), mean);

                // Calculate standard deviation
                let var_sum: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
                let variance = var_sum / values.len() as f64;
                let std_dev = variance.sqrt();
                stds.insert(col_name.clone(), std_dev);
            }
        }

        self.means = Some(means);
        self.stds = Some(stds);

        // Then transform
        // Create a new OptimizedDataFrame to hold the result
        let mut result = OptimizedDataFrame::new();

        for col_name in &column_names {
            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // Check if this column should be scaled
            let should_scale = if let Some(cols) = &self.columns {
                cols.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            // If this column should be scaled
            if should_scale && column_view.as_float64().is_some() {
                // Get the column data
                let float_values = column_view.as_float64().unwrap();

                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                // Get the mean and std
                let mean = if let Some(means) = &self.means {
                    means.get(col_name.as_str()).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let std_dev = if let Some(stds) = &self.stds {
                    stds.get(col_name.as_str()).copied().unwrap_or(1.0)
                } else {
                    1.0
                };

                // Scale the values
                let scaled_values = if std_dev > 1e-10 {
                    values.iter().map(|&x| (x - mean) / std_dev).collect()
                } else {
                    vec![0.0; values.len()]
                };

                // Create a new column
                let scaled_column =
                    crate::column::Float64Column::with_name(scaled_values, col_name.to_string());
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                // Create a column copy
                if let Some(float_values) = column_view.as_float64() {
                    // Handle float column
                    let mut values = Vec::new();
                    for i in 0..float_values.len() {
                        if let Ok(Some(val)) = float_values.get(i) {
                            values.push(val);
                        } else {
                            values.push(0.0); // Default value
                        }
                    }
                    let col = crate::column::Float64Column::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::Float64(col))?;
                } else if let Some(string_values) = column_view.as_string() {
                    // Handle string column
                    let mut values = Vec::new();
                    for i in 0..string_values.len() {
                        if let Ok(Some(val)) = string_values.get(i) {
                            values.push(val.to_string());
                        } else {
                            values.push(String::new()); // Default value
                        }
                    }
                    let col = crate::column::StringColumn::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::String(col))?;
                }
                // Skip other column types for simplicity
            }
        }

        Ok(result)
    }

    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Get the columns to process
        let column_names: Vec<String> = match &self.columns {
            Some(cols) => cols.clone(),
            None => df.column_names().to_vec(),
        };

        let mut means = HashMap::new();
        let mut stds = HashMap::new();

        for col_name in column_names {
            // Check if column exists
            if let Err(_) = df.column(&col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(&col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                if values.is_empty() {
                    continue;
                }

                // Calculate mean
                let sum: f64 = values.iter().sum();
                let mean = sum / values.len() as f64;
                means.insert(col_name.clone(), mean);

                // Calculate standard deviation
                let var_sum: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
                let variance = var_sum / values.len() as f64;
                let std_dev = variance.sqrt();
                stds.insert(col_name.clone(), std_dev);
            }
        }

        self.means = Some(means);
        self.stds = Some(stds);

        Ok(())
    }
}

// Implement Transformer for MinMaxScaler
impl Transformer for MinMaxScaler {
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // Create a new OptimizedDataFrame to hold the result
        let mut result = OptimizedDataFrame::new();

        // For each column in the original DataFrame
        let column_names: Vec<String> = df.column_names().to_vec();
        for col_name in &column_names {
            // Check if this column should be scaled
            let should_scale = if let Some(columns) = &self.columns {
                columns.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // If this column should be scaled
            if should_scale && column_view.as_float64().is_some() {
                // Get the column data
                let float_values = column_view.as_float64().unwrap();

                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                // Get the min and max
                let min_val = if let Some(mins) = &self.min_values {
                    mins.get(col_name.as_str()).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let max_val = if let Some(maxs) = &self.max_values {
                    maxs.get(col_name.as_str()).copied().unwrap_or(1.0)
                } else {
                    1.0
                };

                // Get feature range
                let (feature_min, feature_max) = self.feature_range;

                // Scale the values
                let scaled_values = if (max_val - min_val).abs() > 1e-10 {
                    values
                        .iter()
                        .map(|&x| {
                            let scaled = (x - min_val) / (max_val - min_val);
                            scaled * (feature_max - feature_min) + feature_min
                        })
                        .collect()
                } else {
                    vec![feature_min; values.len()]
                };

                // Create a new column
                let scaled_column =
                    crate::column::Float64Column::with_name(scaled_values, col_name.to_string());
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                // Create a column copy
                if let Some(float_values) = column_view.as_float64() {
                    // Handle float column
                    let mut values = Vec::new();
                    for i in 0..float_values.len() {
                        if let Ok(Some(val)) = float_values.get(i) {
                            values.push(val);
                        } else {
                            values.push(0.0); // Default value
                        }
                    }
                    let col = crate::column::Float64Column::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::Float64(col))?;
                } else if let Some(string_values) = column_view.as_string() {
                    // Handle string column
                    let mut values = Vec::new();
                    for i in 0..string_values.len() {
                        if let Ok(Some(val)) = string_values.get(i) {
                            values.push(val.to_string());
                        } else {
                            values.push(String::new()); // Default value
                        }
                    }
                    let col = crate::column::StringColumn::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::String(col))?;
                }
                // Skip other column types for simplicity
            }
        }

        Ok(result)
    }

    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // Instead of calling normal fit and transform, we'll implement directly

        // First, fit
        let column_names: Vec<String> = match &self.columns {
            Some(cols) => cols.clone(),
            None => df.column_names().to_vec(),
        };

        let mut min_values = HashMap::new();
        let mut max_values = HashMap::new();

        for col_name in &column_names {
            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                if values.is_empty() {
                    continue;
                }

                // Calculate min and max
                let min_val = *values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let max_val = *values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();

                min_values.insert(col_name.clone(), min_val);
                max_values.insert(col_name.clone(), max_val);
            }
        }

        self.min_values = Some(min_values);
        self.max_values = Some(max_values);

        // Then transform
        // Create a new OptimizedDataFrame to hold the result
        let mut result = OptimizedDataFrame::new();

        for col_name in &column_names {
            // Check if column exists
            if let Err(_) = df.column(col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(col_name)?;

            // Check if this column should be scaled
            let should_scale = if let Some(cols) = &self.columns {
                cols.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            // If this column should be scaled
            if should_scale && column_view.as_float64().is_some() {
                // Get the column data
                let float_values = column_view.as_float64().unwrap();

                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                // Get the min and max
                let min_val = if let Some(mins) = &self.min_values {
                    mins.get(col_name.as_str()).copied().unwrap_or(0.0)
                } else {
                    0.0
                };

                let max_val = if let Some(maxs) = &self.max_values {
                    maxs.get(col_name.as_str()).copied().unwrap_or(1.0)
                } else {
                    1.0
                };

                // Get feature range
                let (feature_min, feature_max) = self.feature_range;

                // Scale the values
                let scaled_values = if (max_val - min_val).abs() > 1e-10 {
                    values
                        .iter()
                        .map(|&x| {
                            let scaled = (x - min_val) / (max_val - min_val);
                            scaled * (feature_max - feature_min) + feature_min
                        })
                        .collect()
                } else {
                    vec![feature_min; values.len()]
                };

                // Create a new column
                let scaled_column =
                    crate::column::Float64Column::with_name(scaled_values, col_name.to_string());
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                // Create a column copy
                if let Some(float_values) = column_view.as_float64() {
                    // Handle float column
                    let mut values = Vec::new();
                    for i in 0..float_values.len() {
                        if let Ok(Some(val)) = float_values.get(i) {
                            values.push(val);
                        } else {
                            values.push(0.0); // Default value
                        }
                    }
                    let col = crate::column::Float64Column::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::Float64(col))?;
                } else if let Some(string_values) = column_view.as_string() {
                    // Handle string column
                    let mut values = Vec::new();
                    for i in 0..string_values.len() {
                        if let Ok(Some(val)) = string_values.get(i) {
                            values.push(val.to_string());
                        } else {
                            values.push(String::new()); // Default value
                        }
                    }
                    let col = crate::column::StringColumn::with_name(values, col_name.to_string());
                    result.add_column(col_name.to_string(), crate::column::Column::String(col))?;
                }
                // Skip other column types for simplicity
            }
        }

        Ok(result)
    }

    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Get the columns to process
        let column_names: Vec<String> = match &self.columns {
            Some(cols) => cols.clone(),
            None => df.column_names().to_vec(),
        };

        let mut min_values = HashMap::new();
        let mut max_values = HashMap::new();

        for col_name in column_names {
            // Check if column exists
            if let Err(_) = df.column(&col_name) {
                continue;
            }

            // Get the column
            let column_view = df.column(&col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values
                let mut values: Vec<f64> = Vec::new();
                for i in 0..float_values.len() {
                    if let Ok(Some(val)) = float_values.get(i) {
                        values.push(val);
                    }
                }

                if values.is_empty() {
                    continue;
                }

                // Calculate min and max
                let min_val = *values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let max_val = *values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();

                min_values.insert(col_name.clone(), min_val);
                max_values.insert(col_name.clone(), max_val);
            }
        }

        self.min_values = Some(min_values);
        self.max_values = Some(max_values);

        Ok(())
    }
}

// Re-export transformer trait for backwards compatibility
pub use self::Transformer as PipelineTransformer;

/// Pipeline for chaining multiple data transformation steps
pub struct Pipeline {
    /// Pipeline stages
    pub stages: Vec<Box<dyn Transformer>>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("stages_count", &self.stages.len())
            .finish()
    }
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Pipeline { stages: Vec::new() }
    }

    /// Add a stage to the pipeline
    pub fn add_stage<T: Transformer + 'static>(&mut self, stage: T) -> &mut Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Fit the pipeline to the data
    pub fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        let mut current_df = df.clone();

        for stage in &mut self.stages {
            stage.fit(&current_df)?;
            current_df = stage.transform(&current_df)?;
        }

        Ok(())
    }

    /// Transform data using the fitted pipeline
    pub fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut current_df = df.clone();

        for stage in &self.stages {
            current_df = stage.transform(&current_df)?;
        }

        Ok(current_df)
    }

    /// Fit the pipeline and transform data in one step
    pub fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

//! Compatibility layer for old Pipeline API
//!
//! This module provides backward compatibility for the old Pipeline API.
//! It allows code that used the old Transformer trait to continue working.

use crate::column::ColumnTrait;
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

/// Read every value of a float column, preserving row alignment: a null
/// input value contributes `f64::NAN` and is flagged in the returned mask,
/// rather than being dropped -- dropping nulls (the previous behavior)
/// shortened the output column relative to every other column in the same
/// result frame, so `add_column` would either reject the length mismatch
/// or silently misalign rows.
fn read_f64_preserving_nulls(col: &crate::column::Float64Column) -> (Vec<f64>, Vec<bool>) {
    let n = col.len();
    let mut values = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for i in 0..n {
        match col.get(i) {
            Ok(Some(v)) => {
                values.push(v);
                nulls.push(false);
            }
            _ => {
                values.push(f64::NAN);
                nulls.push(true);
            }
        }
    }
    (values, nulls)
}

/// Read every value of a string column, preserving row alignment (see
/// [`read_f64_preserving_nulls`]).
fn read_string_preserving_nulls(col: &crate::column::StringColumn) -> (Vec<String>, Vec<bool>) {
    let n = col.len();
    let mut values = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for i in 0..n {
        match col.get(i) {
            Ok(Some(v)) => {
                values.push(v.to_string());
                nulls.push(false);
            }
            _ => {
                values.push(String::new());
                nulls.push(true);
            }
        }
    }
    (values, nulls)
}

/// Build a `Float64Column`, attaching a real null bitmask (via
/// `with_nulls`, the same constructor `optimized::convert` and the xlsx
/// reader use for null-aware columns elsewhere in this crate) only when at
/// least one value is actually null.
fn make_f64_column(values: Vec<f64>, nulls: Vec<bool>, name: &str) -> crate::column::Float64Column {
    if nulls.iter().any(|&n| n) {
        crate::column::Float64Column::with_nulls(values, nulls)
    } else {
        crate::column::Float64Column::with_name(values, name.to_string())
    }
}

/// Build a `StringColumn` with a null bitmask when needed (see
/// [`make_f64_column`]).
fn make_string_column(
    values: Vec<String>,
    nulls: Vec<bool>,
    name: &str,
) -> crate::column::StringColumn {
    if nulls.iter().any(|&n| n) {
        crate::column::StringColumn::with_nulls(values, nulls)
    } else {
        crate::column::StringColumn::with_name(values, name.to_string())
    }
}

/// Copy a column into `result` unchanged, preserving null positions
/// (rather than substituting `0.0`/`""` for them, which fabricates data
/// that was never actually observed).
fn copy_column_passthrough(
    result: &mut OptimizedDataFrame,
    col_name: &str,
    column_view: &crate::optimized::ColumnView,
) -> Result<()> {
    if let Some(float_values) = column_view.as_float64() {
        let (values, nulls) = read_f64_preserving_nulls(float_values);
        let col = make_f64_column(values, nulls, col_name);
        result.add_column(col_name.to_string(), crate::column::Column::Float64(col))?;
    } else if let Some(string_values) = column_view.as_string() {
        let (values, nulls) = read_string_preserving_nulls(string_values);
        let col = make_string_column(values, nulls, col_name);
        result.add_column(col_name.to_string(), crate::column::Column::String(col))?;
    }
    // Other column types fall outside this compatibility layer's
    // pre-existing scope (float/string columns only).
    Ok(())
}

// Implement Transformer for StandardScaler
impl Transformer for StandardScaler {
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // An unfitted scaler previously fell back to mean=0/std=1 per
        // column, i.e. silently returned the input unchanged. Report it
        // instead of pretending to have scaled anything.
        let means = self.means.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "StandardScaler::transform called before fit() (or fit_transform()); \
                 no fitted means available"
                    .into(),
            )
        })?;
        let stds = self.stds.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "StandardScaler::transform called before fit() (or fit_transform()); \
                 no fitted standard deviations available"
                    .into(),
            )
        })?;

        let mut result = OptimizedDataFrame::new();

        let column_names: Vec<String> = df.column_names().to_vec();
        for col_name in &column_names {
            let should_scale = if let Some(columns) = &self.columns {
                columns.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            let column_view = match df.column(col_name) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if should_scale && column_view.as_float64().is_some() {
                let float_values = column_view.as_float64().ok_or_else(|| {
                    Error::TypeMismatch("column type check failed for Float64".into())
                })?;

                let mean = *means.get(col_name.as_str()).ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "StandardScaler: no fitted mean for column '{}'; it was not \
                         present (or was entirely null) when fit() ran",
                        col_name
                    ))
                })?;
                let std_dev = *stds.get(col_name.as_str()).ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "StandardScaler: no fitted standard deviation for column '{}'; \
                         it was not present (or was entirely null) when fit() ran",
                        col_name
                    ))
                })?;

                let (raw_values, nulls) = read_f64_preserving_nulls(float_values);
                let scaled_values: Vec<f64> = raw_values
                    .iter()
                    .map(|&x| {
                        if std_dev > 1e-10 {
                            (x - mean) / std_dev
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let scaled_column = make_f64_column(scaled_values, nulls, col_name);
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                copy_column_passthrough(&mut result, col_name, &column_view)?;
            }
        }

        Ok(result)
    }

    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // `StandardScaler` also has its own INHERENT `fit`/`transform`
        // methods (operating on `crate::dataframe::DataFrame`, defined in
        // `ml::preprocessing`) that share these method names. Inherent
        // methods always win plain `self.fit(df)` dot-call resolution over
        // trait methods of the same name, so calling that way here would
        // try to typecheck `df: &OptimizedDataFrame` against the inherent
        // method's `&DataFrame` parameter and fail to compile. The
        // fully-qualified form below unambiguously selects this trait's
        // `OptimizedDataFrame`-based methods instead.
        <Self as Transformer>::fit(self, df)?;
        <Self as Transformer>::transform(self, df)
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
            if df.column(&col_name).is_err() {
                continue;
            }

            // Get the column
            let column_view = df.column(&col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values, skipping nulls -- correct for a summary
                // statistic (mean/std), unlike `transform`'s row-aligned
                // output columns which must preserve every row.
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
        // An unfitted scaler previously fell back to min=0/max=1 per
        // column, i.e. silently returned the input unchanged. Report it
        // instead of pretending to have scaled anything.
        let mins = self.min_values.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "MinMaxScaler::transform called before fit() (or fit_transform()); \
                 no fitted minimums available"
                    .into(),
            )
        })?;
        let maxs = self.max_values.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "MinMaxScaler::transform called before fit() (or fit_transform()); \
                 no fitted maximums available"
                    .into(),
            )
        })?;

        let mut result = OptimizedDataFrame::new();

        let column_names: Vec<String> = df.column_names().to_vec();
        for col_name in &column_names {
            let should_scale = if let Some(columns) = &self.columns {
                columns.contains(col_name)
            } else {
                true // Scale all columns if not specified
            };

            let column_view = match df.column(col_name) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if should_scale && column_view.as_float64().is_some() {
                let float_values = column_view.as_float64().ok_or_else(|| {
                    Error::TypeMismatch("column type check failed for Float64".into())
                })?;

                let min_val = *mins.get(col_name.as_str()).ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "MinMaxScaler: no fitted minimum for column '{}'; it was not \
                         present (or was entirely null) when fit() ran",
                        col_name
                    ))
                })?;
                let max_val = *maxs.get(col_name.as_str()).ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "MinMaxScaler: no fitted maximum for column '{}'; it was not \
                         present (or was entirely null) when fit() ran",
                        col_name
                    ))
                })?;

                let (feature_min, feature_max) = self.feature_range;

                let (raw_values, nulls) = read_f64_preserving_nulls(float_values);
                let scaled_values: Vec<f64> = if (max_val - min_val).abs() > 1e-10 {
                    raw_values
                        .iter()
                        .map(|&x| {
                            let scaled = (x - min_val) / (max_val - min_val);
                            scaled * (feature_max - feature_min) + feature_min
                        })
                        .collect()
                } else {
                    vec![feature_min; raw_values.len()]
                };

                let scaled_column = make_f64_column(scaled_values, nulls, col_name);
                result.add_column(
                    col_name.to_string(),
                    crate::column::Column::Float64(scaled_column),
                )?;
            } else {
                copy_column_passthrough(&mut result, col_name, &column_view)?;
            }
        }

        Ok(result)
    }

    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        // See the identical comment on `StandardScaler::fit_transform`:
        // `MinMaxScaler` also has inherent `fit`/`transform` methods (on
        // `DataFrame`, in `ml::preprocessing`) sharing these names, so the
        // fully-qualified form is required to select this trait's
        // `OptimizedDataFrame`-based methods rather than failing to
        // typecheck against the inherent ones.
        <Self as Transformer>::fit(self, df)?;
        <Self as Transformer>::transform(self, df)
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
            if df.column(&col_name).is_err() {
                continue;
            }

            // Get the column
            let column_view = df.column(&col_name)?;

            // If this is a float column
            if let Some(float_values) = column_view.as_float64() {
                // Extract values, skipping nulls -- correct for a summary
                // statistic (min/max), unlike `transform`'s row-aligned
                // output columns which must preserve every row.
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
                let min_val = values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Cannot compute min of empty values".into())
                    })?;
                let max_val = values
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Cannot compute max of empty values".into())
                    })?;

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

//! Machine learning pipelines
//!
//! This module provides functionality for creating pipelines of data
//! transformations and machine learning models.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;
use std::collections::HashMap;

/// Trait for pipeline stages that transform DataFrames
pub trait PipelineTransformer {
    /// Transform DataFrame according to this pipeline stage
    fn transform(&self, df: &DataFrame) -> Result<DataFrame>;

    /// Fit this transformer to the data (if needed)
    fn fit(&mut self, df: &DataFrame) -> Result<()>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// Enum for different types of pipeline stages
#[derive(Debug)]
pub enum PipelineStage {
    /// Standard scaler
    StandardScaler {
        /// Columns to scale
        columns: Option<Vec<String>>,
        /// Internal storage for fit parameters
        _means: Option<HashMap<String, f64>>,
        _stds: Option<HashMap<String, f64>>,
    },

    /// Min-max scaler
    MinMaxScaler {
        /// Columns to scale
        columns: Option<Vec<String>>,
        /// Feature range (min, max)
        feature_range: (f64, f64),
        /// Internal storage for fit parameters
        _min_values: Option<HashMap<String, f64>>,
        _max_values: Option<HashMap<String, f64>>,
    },

    /// One-hot encoder
    OneHotEncoder {
        /// Columns to encode
        columns: Option<Vec<String>>,
        /// Whether to drop the first category
        drop_first: bool,
        /// Prefix for new column names
        prefix: Option<String>,
        /// Internal storage for fit parameters
        _categories: Option<HashMap<String, Vec<String>>>,
    },

    /// Imputer for missing values
    Imputer {
        /// Columns to impute
        columns: Option<Vec<String>>,
        /// Imputation strategy
        strategy: String,
        /// Constant value for constant strategy
        fill_value: Option<f64>,
        /// Internal storage for fit parameters
        _fill_values: Option<HashMap<String, f64>>,
    },

    /// Feature selector
    FeatureSelector {
        /// Columns to select
        columns: Vec<String>,
    },
    // Custom transformer (commented out since we can't implement Debug for it)
    /*
    Custom {
        // Custom transformation function
        transform_fn: Box<dyn Fn(&DataFrame) -> Result<DataFrame>>,
    },
    */
}

impl PipelineTransformer for PipelineStage {
    fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        match self {
            PipelineStage::StandardScaler { _means, _stds, .. } => {
                let means = _means.as_ref().ok_or_else(not_fitted("StandardScaler"))?;
                let stds = _stds.as_ref().ok_or_else(not_fitted("StandardScaler"))?;

                let mut result = DataFrame::new();
                for col_name in df.column_names() {
                    if let Some(&mean) = means.get(col_name.as_str()) {
                        let std = stds.get(col_name.as_str()).copied().unwrap_or(1.0);
                        let values = df.get_column_numeric_values(col_name.as_str())?;
                        let scaled: Vec<f64> = values
                            .iter()
                            .map(|&v| {
                                if std > 1e-10 {
                                    (v - mean) / std
                                } else {
                                    v - mean
                                }
                            })
                            .collect();
                        result.add_column(
                            col_name.clone(),
                            Series::new(scaled, Some(col_name.clone()))?,
                        )?;
                    } else {
                        passthrough_column(df, &mut result, col_name.as_str())?;
                    }
                }
                Ok(result)
            }

            PipelineStage::MinMaxScaler {
                feature_range,
                _min_values,
                _max_values,
                ..
            } => {
                let mins = _min_values
                    .as_ref()
                    .ok_or_else(not_fitted("MinMaxScaler"))?;
                let maxs = _max_values
                    .as_ref()
                    .ok_or_else(not_fitted("MinMaxScaler"))?;
                let (out_min, out_max) = *feature_range;

                let mut result = DataFrame::new();
                for col_name in df.column_names() {
                    if let (Some(&min_v), Some(&max_v)) =
                        (mins.get(col_name.as_str()), maxs.get(col_name.as_str()))
                    {
                        let values = df.get_column_numeric_values(col_name.as_str())?;
                        let span = max_v - min_v;
                        let scaled: Vec<f64> = values
                            .iter()
                            .map(|&v| {
                                if span.abs() < f64::EPSILON {
                                    // sklearn's MinMaxScaler special-cases a
                                    // degenerate (constant) column to
                                    // `feature_range[0]`, not the midpoint
                                    // of the range.
                                    out_min
                                } else {
                                    out_min + (out_max - out_min) * (v - min_v) / span
                                }
                            })
                            .collect();
                        result.add_column(
                            col_name.clone(),
                            Series::new(scaled, Some(col_name.clone()))?,
                        )?;
                    } else {
                        passthrough_column(df, &mut result, col_name.as_str())?;
                    }
                }
                Ok(result)
            }

            PipelineStage::OneHotEncoder {
                drop_first,
                prefix,
                _categories,
                ..
            } => {
                let categories = _categories
                    .as_ref()
                    .ok_or_else(not_fitted("OneHotEncoder"))?;

                let mut result = DataFrame::new();
                for col_name in df.column_names() {
                    if let Some(cats) = categories.get(col_name.as_str()) {
                        let values = df.get_column_string_values(col_name.as_str())?;
                        let start = if *drop_first { 1 } else { 0 };
                        let base = prefix.clone().unwrap_or_else(|| col_name.clone());
                        for cat in cats.iter().skip(start) {
                            let dummy: Vec<f64> = values
                                .iter()
                                .map(|v| if v == cat { 1.0 } else { 0.0 })
                                .collect();
                            let new_name = format!("{}_{}", base, cat);
                            result.add_column(
                                new_name.clone(),
                                Series::new(dummy, Some(new_name))?,
                            )?;
                        }
                    } else {
                        passthrough_column(df, &mut result, col_name.as_str())?;
                    }
                }
                Ok(result)
            }

            PipelineStage::Imputer { _fill_values, .. } => {
                let fills = _fill_values.as_ref().ok_or_else(not_fitted("Imputer"))?;

                let mut result = DataFrame::new();
                for col_name in df.column_names() {
                    if let Some(&fill) = fills.get(col_name.as_str()) {
                        let values = df.get_column_numeric_values(col_name.as_str())?;
                        let imputed: Vec<f64> = values
                            .iter()
                            .map(|&v| if v.is_nan() { fill } else { v })
                            .collect();
                        result.add_column(
                            col_name.clone(),
                            Series::new(imputed, Some(col_name.clone()))?,
                        )?;
                    } else {
                        passthrough_column(df, &mut result, col_name.as_str())?;
                    }
                }
                Ok(result)
            }

            PipelineStage::FeatureSelector { columns } => {
                for col_name in columns {
                    if !df.contains_column(col_name) {
                        return Err(Error::InvalidValue(format!(
                            "Column '{}' not found",
                            col_name
                        )));
                    }
                }
                let refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
                df.select_columns(&refs)
            }
        }
    }

    fn fit(&mut self, df: &DataFrame) -> Result<()> {
        match self {
            PipelineStage::StandardScaler {
                columns,
                _means,
                _stds,
            } => {
                let targets = resolve_numeric_targets(df, columns);
                let mut means = HashMap::new();
                let mut stds = HashMap::new();

                for col_name in &targets {
                    let values = df.get_column_numeric_values(col_name)?;
                    if values.is_empty() {
                        continue;
                    }
                    let n = values.len() as f64;
                    let mean = values.iter().sum::<f64>() / n;
                    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
                    means.insert(col_name.clone(), mean);
                    stds.insert(col_name.clone(), variance.sqrt());
                }

                *_means = Some(means);
                *_stds = Some(stds);
                Ok(())
            }

            PipelineStage::MinMaxScaler {
                columns,
                _min_values,
                _max_values,
                ..
            } => {
                let targets = resolve_numeric_targets(df, columns);
                let mut mins = HashMap::new();
                let mut maxs = HashMap::new();

                for col_name in &targets {
                    let values = df.get_column_numeric_values(col_name)?;
                    if values.is_empty() {
                        continue;
                    }
                    let min_v = values.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    mins.insert(col_name.clone(), min_v);
                    maxs.insert(col_name.clone(), max_v);
                }

                *_min_values = Some(mins);
                *_max_values = Some(maxs);
                Ok(())
            }

            PipelineStage::OneHotEncoder {
                columns,
                _categories,
                ..
            } => {
                let targets = resolve_categorical_targets(df, columns);
                let mut categories = HashMap::new();

                for col_name in &targets {
                    let values = df.get_column_string_values(col_name)?;
                    // BTreeSet yields sorted, de-duplicated categories for stable column ordering.
                    let unique: Vec<String> = values
                        .into_iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    categories.insert(col_name.clone(), unique);
                }

                *_categories = Some(categories);
                Ok(())
            }

            PipelineStage::Imputer {
                columns,
                strategy,
                fill_value,
                _fill_values,
            } => {
                let targets = resolve_numeric_targets(df, columns);
                let mut fills = HashMap::new();

                for col_name in &targets {
                    let values = df.get_column_numeric_values(col_name)?;
                    let present: Vec<f64> =
                        values.iter().cloned().filter(|v| !v.is_nan()).collect();

                    // An all-missing column has no observed values to
                    // derive a mean/median/mode from; fabricating 0.0 (as
                    // this used to for "mean", and as `median`/`mode`'s
                    // own empty-input fallback silently did) would
                    // substitute a made-up number for data that was never
                    // observed, indistinguishable downstream from a
                    // genuine 0.0 reading.
                    if present.is_empty()
                        && matches!(
                            strategy.to_lowercase().as_str(),
                            "mean" | "median" | "most_frequent" | "mode"
                        )
                    {
                        return Err(Error::InvalidValue(format!(
                            "Imputer: column '{}' has no non-missing values; cannot compute \
                             a '{}' fill value from it",
                            col_name, strategy
                        )));
                    }

                    let fill = match strategy.to_lowercase().as_str() {
                        "mean" => present.iter().sum::<f64>() / present.len() as f64,
                        "median" => median(&present),
                        "most_frequent" | "mode" => mode(&present),
                        "constant" => fill_value.ok_or_else(|| {
                            Error::InvalidValue(
                                "Imputer: strategy 'constant' requires fill_value to be set"
                                    .to_string(),
                            )
                        })?,
                        other => {
                            return Err(Error::InvalidValue(format!(
                                "Unknown imputer strategy '{}'",
                                other
                            )))
                        }
                    };
                    fills.insert(col_name.clone(), fill);
                }

                *_fill_values = Some(fills);
                Ok(())
            }

            PipelineStage::FeatureSelector { .. } => Ok(()),
        }
    }
}

/// Build a closure that produces a "not fitted" error for the named stage.
fn not_fitted(stage: &'static str) -> impl Fn() -> Error {
    move || Error::InvalidOperation(format!("{} must be fitted before transform", stage))
}

/// Whether `name` names a column whose CONCRETE storage type is numeric
/// (`f64`, `f32`, `i64`, `i32`, or `bool`), checked via `get_column::<T>`'s
/// exact-type downcast rather than string parseability.
///
/// `get_column_numeric_values` also successfully parses a `Series<String>`
/// column whose text happens to look numeric (e.g. digit-coded category
/// labels like `"1"`, `"2"`), so using it to decide "is this numeric" put
/// such categorical columns through `StandardScaler`/`MinMaxScaler`/
/// `Imputer` as if they were continuous features, while simultaneously
/// excluding them from `resolve_categorical_targets` below.
fn is_concrete_numeric_column(df: &DataFrame, name: &str) -> bool {
    df.get_column::<f64>(name).is_ok()
        || df.get_column::<f32>(name).is_ok()
        || df.get_column::<i64>(name).is_ok()
        || df.get_column::<i32>(name).is_ok()
        || df.get_column::<bool>(name).is_ok()
}

/// Whether `name` names a column whose concrete storage type is `String`.
fn is_concrete_string_column(df: &DataFrame, name: &str) -> bool {
    df.get_column::<String>(name).is_ok()
}

/// Resolve the numeric target columns for a scaler/imputer: the explicit list, or every
/// column whose CONCRETE type is numeric when none was specified.
fn resolve_numeric_targets(df: &DataFrame, columns: &Option<Vec<String>>) -> Vec<String> {
    match columns {
        Some(cols) => cols.clone(),
        None => df
            .column_names()
            .iter()
            .filter(|name| is_concrete_numeric_column(df, name.as_str()))
            .cloned()
            .collect(),
    }
}

/// Resolve categorical target columns for one-hot encoding: the explicit list, or every
/// column whose CONCRETE type is `String` when none was specified.
fn resolve_categorical_targets(df: &DataFrame, columns: &Option<Vec<String>>) -> Vec<String> {
    match columns {
        Some(cols) => cols.clone(),
        None => df
            .column_names()
            .iter()
            .filter(|name| is_concrete_string_column(df, name.as_str()))
            .cloned()
            .collect(),
    }
}

/// Copy a column unchanged into `result`, preserving numeric vs. string representation.
///
/// Routes by the column's CONCRETE storage type, via the same
/// [`is_concrete_string_column`] check `resolve_categorical_targets` uses to decide
/// what is categorical -- not by which read happens to parse successfully. A
/// digit-coded categorical `Series<String>` column (e.g. `"1"`, `"2"`) parses
/// successfully through `get_column_numeric_values` (see the doc comment on
/// `is_concrete_numeric_column`), so trying that read first -- as this previously
/// did -- silently rewrote such a column to `Series<f64>` on every pass-through.
/// That is invisible to whichever stage produced `result`, but breaks any *later*
/// stage in the same `Pipeline` (e.g. `OneHotEncoder`) that also decides "is this
/// categorical?" from the concrete type: it would see an f64 column and skip it,
/// even though `resolve_categorical_targets` correctly identified it as
/// categorical when this same column was still the pipeline's input.
fn passthrough_column(df: &DataFrame, result: &mut DataFrame, name: &str) -> Result<()> {
    if is_concrete_string_column(df, name) {
        let values = df.get_column_string_values(name)?;
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
    } else if let Ok(values) = df.get_column_numeric_values(name) {
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
    } else if let Ok(values) = df.get_column_string_values(name) {
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
    } else {
        return Err(Error::InvalidValue(format!(
            "Unable to determine type of column '{}' for pass-through",
            name
        )));
    }
    Ok(())
}

/// Median of a slice (returns 0.0 for an empty slice).
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Most frequent value (mode). Ties are broken by the smaller value; empty input yields 0.0.
fn mode(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let mut best_bits = values[0].to_bits();
    let mut best_count = 0usize;
    for (&bits, &count) in &counts {
        let value = f64::from_bits(bits);
        if count > best_count || (count == best_count && value < f64::from_bits(best_bits)) {
            best_count = count;
            best_bits = bits;
        }
    }
    f64::from_bits(best_bits)
}

/// Pipeline for chaining multiple data transformation steps
#[derive(Debug)]
pub struct Pipeline {
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new empty pipeline
    pub fn new() -> Self {
        Pipeline { stages: Vec::new() }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Fit the pipeline to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let mut current_df = df.clone();

        for stage in &mut self.stages {
            stage.fit(&current_df)?;
            current_df = stage.transform(&current_df)?;
        }

        Ok(())
    }

    /// Transform data using the fitted pipeline
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut current_df = df.clone();

        for stage in &self.stages {
            current_df = stage.transform(&current_df)?;
        }

        Ok(current_df)
    }

    /// Fit the pipeline and transform data in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        let mut current_df = df.clone();

        for stage in &mut self.stages {
            stage.fit(&current_df)?;
            current_df = stage.transform(&current_df)?;
        }

        Ok(current_df)
    }
}

// No need for re-exports here as the types are already defined in this module

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_standard_scaler_real() {
        let df = numeric_df();
        let mut stage = PipelineStage::StandardScaler {
            columns: None,
            _means: None,
            _stds: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();

        let a = out.get_column_numeric_values("a").unwrap();
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let var: f64 = a.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(
            mean.abs() < 1e-9,
            "standardized mean should be ~0, got {mean}"
        );
        assert!(
            (var - 1.0).abs() < 1e-6,
            "standardized variance should be ~1, got {var}"
        );
    }

    #[test]
    fn test_min_max_scaler_real() {
        let df = numeric_df();
        let mut stage = PipelineStage::MinMaxScaler {
            columns: None,
            feature_range: (0.0, 1.0),
            _min_values: None,
            _max_values: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();

        let a = out.get_column_numeric_values("a").unwrap();
        assert!((a[0] - 0.0).abs() < 1e-9);
        assert!((a[4] - 1.0).abs() < 1e-9);
        assert!((a[2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_one_hot_encoder_real() {
        let mut df = DataFrame::new();
        df.add_column(
            "color".to_string(),
            Series::new(
                vec!["red".to_string(), "blue".to_string(), "red".to_string()],
                Some("color".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let mut stage = PipelineStage::OneHotEncoder {
            columns: Some(vec!["color".to_string()]),
            drop_first: false,
            prefix: None,
            _categories: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();

        // Categories are sorted -> blue, red.
        assert!(out.contains_column("color_blue"));
        assert!(out.contains_column("color_red"));
        assert_eq!(
            out.get_column_numeric_values("color_red").unwrap(),
            vec![1.0, 0.0, 1.0]
        );
        assert_eq!(
            out.get_column_numeric_values("color_blue").unwrap(),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_one_hot_encoder_drop_first() {
        let mut df = DataFrame::new();
        df.add_column(
            "color".to_string(),
            Series::new(
                vec!["red".to_string(), "blue".to_string(), "green".to_string()],
                Some("color".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let mut stage = PipelineStage::OneHotEncoder {
            columns: Some(vec!["color".to_string()]),
            drop_first: true,
            prefix: None,
            _categories: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();

        // Sorted categories blue, green, red -> first ("blue") dropped.
        assert!(!out.contains_column("color_blue"));
        assert!(out.contains_column("color_green"));
        assert!(out.contains_column("color_red"));
    }

    #[test]
    fn test_imputer_mean_real() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("x".to_string())).unwrap(),
        )
        .unwrap();

        let mut stage = PipelineStage::Imputer {
            columns: Some(vec!["x".to_string()]),
            strategy: "mean".to_string(),
            fill_value: None,
            _fill_values: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();

        // Mean of present values {1, 3} = 2.0
        assert_eq!(
            out.get_column_numeric_values("x").unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_imputer_constant_real() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![f64::NAN, 5.0, f64::NAN], Some("x".to_string())).unwrap(),
        )
        .unwrap();

        let mut stage = PipelineStage::Imputer {
            columns: Some(vec!["x".to_string()]),
            strategy: "constant".to_string(),
            fill_value: Some(-1.0),
            _fill_values: None,
        };
        stage.fit(&df).unwrap();
        let out = stage.transform(&df).unwrap();
        assert_eq!(
            out.get_column_numeric_values("x").unwrap(),
            vec![-1.0, 5.0, -1.0]
        );
    }

    #[test]
    fn test_feature_selector_numeric_columns() {
        // Regression test: the old implementation read columns as Series<String>
        // and errored on real numeric data. Selection must now work on f64 columns.
        let df = numeric_df();
        let stage = PipelineStage::FeatureSelector {
            columns: vec!["a".to_string()],
        };
        let out = stage.transform(&df).unwrap();
        assert_eq!(out.column_names(), vec!["a".to_string()]);
        assert_eq!(
            out.get_column_numeric_values("a").unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_unfitted_transform_errors() {
        let df = numeric_df();
        let stage = PipelineStage::StandardScaler {
            columns: None,
            _means: None,
            _stds: None,
        };
        // Transforming before fit must error, not silently pass through.
        assert!(stage.transform(&df).is_err());
    }

    #[test]
    fn test_pipeline_fit_transform_end_to_end() {
        let df = numeric_df();
        let mut pipeline = Pipeline::new();
        pipeline.add_stage(PipelineStage::StandardScaler {
            columns: None,
            _means: None,
            _stds: None,
        });
        let out = pipeline.fit_transform(&df).unwrap();

        for col in ["a", "b"] {
            let values = out.get_column_numeric_values(col).unwrap();
            let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
            assert!(mean.abs() < 1e-9, "column {col} should be centered");
        }
    }
}

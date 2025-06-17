//! Scikit-learn compatibility layer for PandRS ML
//!
//! This module provides comprehensive scikit-learn compatible interfaces,
//! allowing PandRS to be used as a drop-in replacement for many sklearn workflows.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel, UnsupervisedModel};
use crate::ml::preprocessing::*;
use crate::series::Series;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Trait for all scikit-learn compatible estimators
pub trait SklearnEstimator: fmt::Debug {
    /// Get parameters of the estimator
    fn get_params(&self) -> HashMap<String, String>;

    /// Set parameters of the estimator
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()>;

    /// Get feature names output by this estimator
    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>>;
}

/// Trait for transformers that fit to data and transform it
pub trait SklearnTransformer: SklearnEstimator {
    /// Fit transformer to training data
    fn fit(&mut self, x: &DataFrame, y: Option<&DataFrame>) -> Result<()>;

    /// Transform data using fitted transformer
    fn transform(&self, x: &DataFrame) -> Result<DataFrame>;

    /// Fit to data, then transform it
    fn fit_transform(&mut self, x: &DataFrame, y: Option<&DataFrame>) -> Result<DataFrame> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Inverse transform data (if supported)
    fn inverse_transform(&self, x: &DataFrame) -> Result<DataFrame> {
        Err(Error::NotImplemented(
            "inverse_transform not supported".into(),
        ))
    }

    /// Create a clone of this transformer for cross-validation
    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync>;
}

/// Trait for predictors (classifiers and regressors)
pub trait SklearnPredictor: SklearnEstimator {
    /// Fit predictor to training data
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()>;

    /// Make predictions on data
    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>>;

    /// Get prediction confidence scores (if supported)
    fn predict_proba(&self, x: &DataFrame) -> Result<Vec<Vec<f64>>> {
        Err(Error::NotImplemented("predict_proba not supported".into()))
    }

    /// Score the model on test data
    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64>;

    /// Create a clone of this predictor for cross-validation
    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync>;
}

/// Enhanced StandardScaler with full scikit-learn compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScalerCompat {
    /// Whether to center the data at 0
    pub with_mean: bool,
    /// Whether to scale the data to unit variance
    pub with_std: bool,
    /// Whether to remove a copy of the data
    pub copy: bool,
    /// Mean values for each feature (fitted)
    mean_: Option<HashMap<String, f64>>,
    /// Scale values for each feature (fitted)
    scale_: Option<HashMap<String, f64>>,
    /// Variance values for each feature (fitted)
    var_: Option<HashMap<String, f64>>,
    /// Number of samples seen during fit
    n_samples_seen_: Option<usize>,
    /// Feature names seen during fit
    feature_names_in_: Option<Vec<String>>,
    /// Number of features seen during fit
    n_features_in_: Option<usize>,
}

impl StandardScalerCompat {
    /// Create new StandardScaler with default parameters
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            copy: true,
            mean_: None,
            scale_: None,
            var_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }

    /// Create StandardScaler with custom parameters
    pub fn with_params(with_mean: bool, with_std: bool) -> Self {
        Self {
            with_mean,
            with_std,
            copy: true,
            mean_: None,
            scale_: None,
            var_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }
}

impl SklearnEstimator for StandardScalerCompat {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("with_mean".to_string(), self.with_mean.to_string());
        params.insert("with_std".to_string(), self.with_std.to_string());
        params.insert("copy".to_string(), self.copy.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "with_mean" => {
                    self.with_mean = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid boolean value for with_mean: {}",
                            value
                        ))
                    })?
                }
                "with_std" => {
                    self.with_std = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid boolean value for with_std: {}",
                            value
                        ))
                    })?
                }
                "copy" => {
                    self.copy = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for copy: {}", value))
                    })?
                }
                _ => return Err(Error::InvalidValue(format!("Unknown parameter: {}", key))),
            }
        }
        Ok(())
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        if let Some(features) = input_features {
            Some(features.to_vec())
        } else {
            self.feature_names_in_.clone()
        }
    }
}

impl SklearnTransformer for StandardScalerCompat {
    fn fit(&mut self, x: &DataFrame, _y: Option<&DataFrame>) -> Result<()> {
        let feature_names: Vec<String> = x.column_names();
        let n_features = feature_names.len();
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(Error::InvalidValue("Cannot fit on empty dataset".into()));
        }

        let mut means = HashMap::new();
        let mut vars = HashMap::new();
        let mut scales = HashMap::new();

        for feature_name in &feature_names {
            // Try to get numeric column
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                continue;
            }

            // Calculate mean
            let mean = if self.with_mean {
                values.iter().sum::<f64>() / values.len() as f64
            } else {
                0.0
            };

            // Calculate variance
            let variance = if self.with_std {
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
            } else {
                1.0
            };

            // Calculate scale (standard deviation)
            let scale = if self.with_std && variance > 1e-10 {
                variance.sqrt()
            } else {
                1.0
            };

            means.insert(feature_name.clone(), mean);
            vars.insert(feature_name.clone(), variance);
            scales.insert(feature_name.clone(), scale);
        }

        self.mean_ = Some(means);
        self.var_ = Some(vars);
        self.scale_ = Some(scales);
        self.n_samples_seen_ = Some(n_samples);
        self.feature_names_in_ = Some(feature_names);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let means = self.mean_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("StandardScaler must be fitted before transform".into())
        })?;
        let scales = self.scale_.as_ref().unwrap();

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(&feature_name)?;
            let values = col.as_f64()?;

            let mean = means.get(&feature_name).copied().unwrap_or(0.0);
            let scale = scales.get(&feature_name).copied().unwrap_or(1.0);

            let transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let centered = if self.with_mean { val - mean } else { val };
                    if self.with_std && scale > 1e-10 {
                        centered / scale
                    } else {
                        centered
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(transformed_values, Some(feature_name))?,
            )?;
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let means = self.mean_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("StandardScaler must be fitted before inverse_transform".into())
        })?;
        let scales = self.scale_.as_ref().unwrap();

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(&feature_name)?;
            let values = col.as_f64()?;

            let mean = means.get(&feature_name).copied().unwrap_or(0.0);
            let scale = scales.get(&feature_name).copied().unwrap_or(1.0);

            let inverse_transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let scaled = if self.with_std && scale > 1e-10 {
                        val * scale
                    } else {
                        val
                    };
                    if self.with_mean {
                        scaled + mean
                    } else {
                        scaled
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(inverse_transformed_values, Some(feature_name))?,
            )?;
        }

        Ok(result)
    }

    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Enhanced MinMaxScaler with full scikit-learn compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScalerCompat {
    /// Desired range of transformed data
    pub feature_range: (f64, f64),
    /// Whether to remove a copy of the data
    pub copy: bool,
    /// Whether to clip transformed values to feature_range
    pub clip: bool,
    /// Minimum values for each feature (fitted)
    data_min_: Option<HashMap<String, f64>>,
    /// Maximum values for each feature (fitted)
    data_max_: Option<HashMap<String, f64>>,
    /// Range of each feature (fitted)
    data_range_: Option<HashMap<String, f64>>,
    /// Scaling factor for each feature (fitted)
    scale_: Option<HashMap<String, f64>>,
    /// Minimum bound for each feature (fitted)
    min_: Option<HashMap<String, f64>>,
    /// Number of samples seen during fit
    n_samples_seen_: Option<usize>,
    /// Feature names seen during fit
    feature_names_in_: Option<Vec<String>>,
    /// Number of features seen during fit
    n_features_in_: Option<usize>,
}

impl MinMaxScalerCompat {
    /// Create new MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            copy: true,
            clip: false,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            scale_: None,
            min_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }

    /// Create MinMaxScaler with custom range
    pub fn with_range(min: f64, max: f64) -> Self {
        Self {
            feature_range: (min, max),
            copy: true,
            clip: false,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            scale_: None,
            min_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }
}

impl SklearnEstimator for MinMaxScalerCompat {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "feature_range_min".to_string(),
            self.feature_range.0.to_string(),
        );
        params.insert(
            "feature_range_max".to_string(),
            self.feature_range.1.to_string(),
        );
        params.insert("copy".to_string(), self.copy.to_string());
        params.insert("clip".to_string(), self.clip.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "feature_range_min" => {
                    let min_val: f64 = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid float value for feature_range_min: {}",
                            value
                        ))
                    })?;
                    self.feature_range.0 = min_val;
                }
                "feature_range_max" => {
                    let max_val: f64 = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid float value for feature_range_max: {}",
                            value
                        ))
                    })?;
                    self.feature_range.1 = max_val;
                }
                "copy" => {
                    self.copy = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for copy: {}", value))
                    })?
                }
                "clip" => {
                    self.clip = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for clip: {}", value))
                    })?
                }
                _ => return Err(Error::InvalidValue(format!("Unknown parameter: {}", key))),
            }
        }
        Ok(())
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        if let Some(features) = input_features {
            Some(features.to_vec())
        } else {
            self.feature_names_in_.clone()
        }
    }
}

impl SklearnTransformer for MinMaxScalerCompat {
    fn fit(&mut self, x: &DataFrame, _y: Option<&DataFrame>) -> Result<()> {
        let feature_names: Vec<String> = x.column_names();
        let n_features = feature_names.len();
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(Error::InvalidValue("Cannot fit on empty dataset".into()));
        }

        let mut data_mins = HashMap::new();
        let mut data_maxs = HashMap::new();
        let mut data_ranges = HashMap::new();
        let mut scales = HashMap::new();
        let mut mins = HashMap::new();

        let (feature_min, feature_max) = self.feature_range;
        let feature_range = feature_max - feature_min;

        for feature_name in &feature_names {
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                continue;
            }

            let data_min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let data_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let data_range = data_max - data_min;

            let scale = if data_range > 1e-10 {
                feature_range / data_range
            } else {
                1.0
            };

            let min = feature_min - data_min * scale;

            data_mins.insert(feature_name.clone(), data_min);
            data_maxs.insert(feature_name.clone(), data_max);
            data_ranges.insert(feature_name.clone(), data_range);
            scales.insert(feature_name.clone(), scale);
            mins.insert(feature_name.clone(), min);
        }

        self.data_min_ = Some(data_mins);
        self.data_max_ = Some(data_maxs);
        self.data_range_ = Some(data_ranges);
        self.scale_ = Some(scales);
        self.min_ = Some(mins);
        self.n_samples_seen_ = Some(n_samples);
        self.feature_names_in_ = Some(feature_names);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let scales = self.scale_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("MinMaxScaler must be fitted before transform".into())
        })?;
        let mins = self.min_.as_ref().unwrap();

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(&feature_name)?;
            let values = col.as_f64()?;

            let scale = scales.get(&feature_name).copied().unwrap_or(1.0);
            let min = mins.get(&feature_name).copied().unwrap_or(0.0);

            let transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let transformed = val * scale + min;
                    if self.clip {
                        transformed
                            .max(self.feature_range.0)
                            .min(self.feature_range.1)
                    } else {
                        transformed
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(transformed_values, Some(feature_name))?,
            )?;
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let scales = self.scale_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("MinMaxScaler must be fitted before inverse_transform".into())
        })?;
        let mins = self.min_.as_ref().unwrap();

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(&feature_name)?;
            let values = col.as_f64()?;

            let scale = scales.get(&feature_name).copied().unwrap_or(1.0);
            let min = mins.get(&feature_name).copied().unwrap_or(0.0);

            let inverse_transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    if scale > 1e-10 {
                        (val - min) / scale
                    } else {
                        val
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(inverse_transformed_values, Some(feature_name))?,
            )?;
        }

        Ok(result)
    }

    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Enhanced Pipeline with full scikit-learn compatibility
#[derive(Debug)]
pub struct Pipeline {
    /// List of pipeline steps (name, transformer/estimator)
    pub steps: Vec<(String, PipelineStep)>,
    /// Whether to cache intermediate results
    pub memory: Option<String>,
    /// Verbose output
    pub verbose: bool,
}

/// A step in the pipeline that can be either a transformer or predictor
#[derive(Debug)]
pub enum PipelineStep {
    /// A transformer step
    Transformer(Box<dyn SklearnTransformer + Send + Sync>),
    /// A predictor step (must be the last step)
    Predictor(Box<dyn SklearnPredictor + Send + Sync>),
}

impl Clone for PipelineStep {
    fn clone(&self) -> Self {
        match self {
            PipelineStep::Transformer(transformer) => {
                PipelineStep::Transformer(transformer.clone_transformer())
            }
            PipelineStep::Predictor(predictor) => {
                PipelineStep::Predictor(predictor.clone_predictor())
            }
        }
    }
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new(steps: Vec<(String, PipelineStep)>) -> Self {
        Self {
            steps,
            memory: None,
            verbose: false,
        }
    }

    /// Add a step to the pipeline
    pub fn add_step(&mut self, name: String, step: PipelineStep) {
        self.steps.push((name, step));
    }

    /// Get a step by name
    pub fn get_step(&self, name: &str) -> Option<&PipelineStep> {
        self.steps
            .iter()
            .find(|(step_name, _)| step_name == name)
            .map(|(_, step)| step)
    }

    /// Get step names
    pub fn get_step_names(&self) -> Vec<&String> {
        self.steps.iter().map(|(name, _)| name).collect()
    }

    /// Set pipeline parameters
    pub fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            if key == "verbose" {
                self.verbose = value.parse().map_err(|_| {
                    Error::InvalidValue(format!("Invalid boolean value for verbose: {}", value))
                })?;
            } else if key.starts_with("memory") {
                self.memory = Some(value);
            } else if let Some(param_sep) = key.find("__") {
                // Step-specific parameter (e.g., "scaler__with_mean")
                let step_name = &key[..param_sep];
                let param_name = &key[param_sep + 2..];

                // Find the step and set its parameter
                for (name, step) in &mut self.steps {
                    if name == step_name {
                        let mut step_params = HashMap::new();
                        step_params.insert(param_name.to_string(), value);

                        match step {
                            PipelineStep::Transformer(transformer) => {
                                transformer.set_params(step_params)?;
                            }
                            PipelineStep::Predictor(predictor) => {
                                predictor.set_params(step_params)?;
                            }
                        }
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

impl SklearnEstimator for Pipeline {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("verbose".to_string(), self.verbose.to_string());

        if let Some(memory) = &self.memory {
            params.insert("memory".to_string(), memory.clone());
        }

        // Add step-specific parameters
        for (step_name, step) in &self.steps {
            let step_params = match step {
                PipelineStep::Transformer(transformer) => transformer.get_params(),
                PipelineStep::Predictor(predictor) => predictor.get_params(),
            };

            for (param_name, param_value) in step_params {
                params.insert(format!("{}__{}", step_name, param_name), param_value);
            }
        }

        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        self.set_params(params)
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        let mut current_features = input_features.map(|f| f.to_vec());

        for (_, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_features = transformer
                        .get_feature_names_out(current_features.as_ref().map(|f| f.as_slice()));
                }
                PipelineStep::Predictor(predictor) => {
                    current_features = predictor
                        .get_feature_names_out(current_features.as_ref().map(|f| f.as_slice()));
                }
            }
        }

        current_features
    }
}

impl SklearnPredictor for Pipeline {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let mut current_x = x.clone();
        let steps_len = self.steps.len();

        for (i, (step_name, step)) in self.steps.iter_mut().enumerate() {
            if self.verbose {
                println!("Fitting step {}: {}", i, step_name);
            }

            match step {
                PipelineStep::Transformer(transformer) => {
                    transformer.fit(&current_x, Some(y))?;
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    // Predictor should be the last step
                    if i != steps_len - 1 {
                        return Err(Error::InvalidOperation(
                            "Predictor must be the last step in pipeline".into(),
                        ));
                    }
                    predictor.fit(&current_x, y)?;
                }
            }
        }

        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let mut current_x = x.clone();

        for (i, (step_name, step)) in self.steps.iter().enumerate() {
            if self.verbose {
                println!("Transforming step {}: {}", i, step_name);
            }

            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.predict(&current_x);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let mut current_x = x.clone();

        for (step_name, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.predict_proba(&current_x);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64> {
        let mut current_x = x.clone();

        for (step_name, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.score(&current_x, y);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        let cloned_steps = self
            .steps
            .iter()
            .map(|(name, step)| (name.clone(), step.clone()))
            .collect();

        Box::new(Pipeline {
            steps: cloned_steps,
            memory: self.memory.clone(),
            verbose: self.verbose,
        })
    }
}

/// Helper functions for creating common pipeline configurations
pub mod pipeline_builders {
    use super::*;

    /// Create a standard preprocessing pipeline
    pub fn standard_preprocessing_pipeline() -> Pipeline {
        let steps = vec![(
            "scaler".to_string(),
            PipelineStep::Transformer(Box::new(StandardScalerCompat::new())),
        )];

        Pipeline::new(steps)
    }

    /// Create a minmax preprocessing pipeline
    pub fn minmax_preprocessing_pipeline() -> Pipeline {
        let steps = vec![(
            "scaler".to_string(),
            PipelineStep::Transformer(Box::new(MinMaxScalerCompat::new())),
        )];

        Pipeline::new(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_standard_scaler_compat() {
        let mut scaler = StandardScalerCompat::new();

        // Create test data
        let mut df = DataFrame::new();
        df.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "feature2".to_string(),
            Series::new(
                vec![10.0, 20.0, 30.0, 40.0, 50.0],
                Some("feature2".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // Fit and transform
        scaler.fit(&df, None).unwrap();
        let transformed = scaler.transform(&df).unwrap();

        // Check that means are approximately zero
        let feature1_col = transformed.get_column::<f64>("feature1").unwrap();
        let feature1_values = feature1_col.as_f64().unwrap();
        let feature1_mean = feature1_values.iter().sum::<f64>() / feature1_values.len() as f64;

        assert!(
            (feature1_mean).abs() < 1e-10,
            "Mean should be approximately zero"
        );

        // Test inverse transform
        let inverse_transformed = scaler.inverse_transform(&transformed).unwrap();
        let original_feature1 = df.get_column::<f64>("feature1").unwrap().as_f64().unwrap();
        let restored_feature1 = inverse_transformed
            .get_column::<f64>("feature1")
            .unwrap()
            .as_f64()
            .unwrap();

        for (original, restored) in original_feature1.iter().zip(restored_feature1.iter()) {
            assert!(
                (original - restored).abs() < 1e-10,
                "Inverse transform should restore original values"
            );
        }
    }

    #[test]
    fn test_minmax_scaler_compat() {
        let mut scaler = MinMaxScalerCompat::new();

        // Create test data
        let mut df = DataFrame::new();
        df.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
        )
        .unwrap();

        // Fit and transform
        scaler.fit(&df, None).unwrap();
        let transformed = scaler.transform(&df).unwrap();

        // Check that values are in range [0, 1]
        let feature1_col = transformed.get_column::<f64>("feature1").unwrap();
        let feature1_values = feature1_col.as_f64().unwrap();

        for &value in &feature1_values {
            assert!(
                value >= 0.0 && value <= 1.0,
                "Values should be in range [0, 1]"
            );
        }

        // Check that min is 0 and max is 1
        let min_val = feature1_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_val = feature1_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!((min_val - 0.0).abs() < 1e-10, "Minimum should be 0");
        assert!((max_val - 1.0).abs() < 1e-10, "Maximum should be 1");
    }

    #[test]
    fn test_pipeline_parameters() {
        let pipeline = pipeline_builders::standard_preprocessing_pipeline();
        let params = pipeline.get_params();

        assert!(params.contains_key("verbose"));
        assert!(params.contains_key("scaler__with_mean"));
        assert!(params.contains_key("scaler__with_std"));
    }
}

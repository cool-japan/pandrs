//! Machine learning models
//!
//! This module provides interfaces and implementations for machine learning models,
//! including regression, classification, and utility functions for model evaluation
//! and cross-validation.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use std::collections::HashMap;

/// Metrics from model evaluation
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Specific metrics for the model (varies by model type)
    pub metrics: HashMap<String, f64>,
    /// Training time in seconds
    pub training_time: f64,
    /// Prediction time in seconds
    pub prediction_time: Option<f64>,
}

impl ModelMetrics {
    /// Create a new empty ModelMetrics instance
    pub fn new() -> Self {
        ModelMetrics {
            metrics: HashMap::new(),
            training_time: 0.0,
            prediction_time: None,
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<&f64> {
        self.metrics.get(name)
    }

    /// Set training time
    pub fn set_training_time(&mut self, time: f64) {
        self.training_time = time;
    }

    /// Set prediction time
    pub fn set_prediction_time(&mut self, time: f64) {
        self.prediction_time = Some(time);
    }
}

/// Trait for evaluating models
pub trait ModelEvaluator {
    /// Evaluate a model using test data
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics>;

    /// Cross-validate a model
    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>>;
}

/// Trait for supervised machine learning models
pub trait SupervisedModel: ModelEvaluator {
    /// Fit model to training data
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()>;

    /// Predict using the fitted model
    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>>;

    /// Get feature importances (if applicable)
    fn feature_importances(&self) -> Option<HashMap<String, f64>>;
}

/// Trait for unsupervised machine learning models
pub trait UnsupervisedModel: ModelEvaluator {
    /// Fit model to training data
    fn fit(&mut self, data: &DataFrame) -> Result<()>;

    /// Transform data using the fitted model
    fn transform(&self, data: &DataFrame) -> Result<DataFrame>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &DataFrame) -> Result<DataFrame> {
        self.fit(data)?;
        self.transform(data)
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidation {
    /// Number of folds
    pub n_folds: usize,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for CrossValidation {
    fn default() -> Self {
        CrossValidation {
            n_folds: 5,
            shuffle: true,
            random_seed: None,
        }
    }
}

/// Split data into training and test sets
///
/// # Arguments
/// * `data` - DataFrame to split
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `shuffle` - Whether to shuffle data before splitting
/// * `random_seed` - Optional random seed for reproducibility
///
/// # Returns
/// * Tuple of (train_data, test_data)
pub fn train_test_split(
    data: &DataFrame,
    test_size: f64,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<(DataFrame, DataFrame)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(Error::InvalidInput(
            "test_size must be between 0 and 1".into(),
        ));
    }

    let n_rows = data.nrows();
    let n_test = (n_rows as f64 * test_size).round() as usize;

    if n_test == 0 || n_test == n_rows {
        return Err(Error::InvalidInput(format!(
            "test_size {} would result in empty training or test set",
            test_size
        )));
    }

    // For now, implement a simple sequential split
    // In a real implementation, this would support shuffling based on the parameters

    let train_indices: Vec<usize> = (0..(n_rows - n_test)).collect();
    let test_indices: Vec<usize> = ((n_rows - n_test)..n_rows).collect();

    let train_data = data.sample(&train_indices)?;
    let test_data = data.sample(&test_indices)?;

    Ok((train_data, test_data))
}

/// Split OptimizedDataFrame into training and test sets
///
/// # Arguments
/// * `data` - OptimizedDataFrame to split
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `random_seed` - Optional random seed for reproducibility
///
/// # Returns
/// * Tuple of (train_data, test_data)
pub fn train_test_split_opt(
    data: &OptimizedDataFrame,
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(Error::InvalidInput(
            "test_size must be between 0 and 1".into(),
        ));
    }

    let n_rows = data.row_count();
    let n_test = (n_rows as f64 * test_size).round() as usize;

    if n_test == 0 || n_test == n_rows {
        return Err(Error::InvalidInput(format!(
            "test_size {} would result in empty training or test set",
            test_size
        )));
    }

    // Generate indices for training and test sets
    let train_indices: Vec<usize> = (0..(n_rows - n_test)).collect();
    let test_indices: Vec<usize> = ((n_rows - n_test)..n_rows).collect();

    // Sample rows to create train and test sets
    let train_data = data.sample_rows(&train_indices)?;
    let test_data = data.sample_rows(&test_indices)?;

    Ok((train_data, test_data))
}

pub mod evaluation;
pub mod linear;
pub mod selection;

// Re-export commonly used model types and functions
pub use evaluation::{cross_val_score, learning_curve, validation_curve};
pub use linear::{LinearRegression, LogisticRegression};
pub use selection::{GridSearchCV, HyperparameterGrid, RandomizedSearchCV};

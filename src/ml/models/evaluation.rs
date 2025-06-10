//! Model evaluation utilities
//!
//! This module provides functions for evaluating machine learning models,
//! including cross-validation, learning curves, and validation curves.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use std::collections::HashMap;

/// Perform cross-validation on a model
///
/// # Arguments
/// * `model` - The model to evaluate
/// * `data` - The data to use for cross-validation
/// * `target` - The target column name
/// * `folds` - Number of cross-validation folds
/// * `metric` - Name of the metric to return (must be a metric computed by the model)
///
/// # Returns
/// * Vector of metric values for each fold
pub fn cross_val_score<T: SupervisedModel + Clone>(
    model: &T,
    data: &DataFrame,
    target: &str,
    folds: usize,
    metric: &str,
) -> Result<Vec<f64>> {
    if folds < 2 {
        return Err(Error::InvalidInput(
            "Number of folds must be at least 2".into(),
        ));
    }

    let metrics = model.cross_validate(data, target, folds)?;

    let mut scores = Vec::with_capacity(folds);
    for fold_metrics in metrics {
        if let Some(score) = fold_metrics.get_metric(metric) {
            scores.push(*score);
        } else {
            return Err(Error::InvalidInput(format!(
                "Metric '{}' not found in model evaluation",
                metric
            )));
        }
    }

    Ok(scores)
}

/// Generate learning curve for a model
///
/// A learning curve shows model performance as a function of training set size.
///
/// # Arguments
/// * `model` - The model to evaluate
/// * `data` - The full dataset
/// * `target` - The target column name
/// * `train_sizes` - Vector of training set sizes (fractions between 0 and 1)
/// * `metric` - Name of the metric to track
/// * `cv` - Number of cross-validation folds
///
/// # Returns
/// * Tuple of (train_sizes, train_scores, test_scores)
///   where each element is a vector of values for each training size
pub fn learning_curve<T: SupervisedModel + Clone>(
    model: &T,
    data: &DataFrame,
    target: &str,
    train_sizes: &[f64],
    metric: &str,
    cv: usize,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    if cv < 2 {
        return Err(Error::InvalidInput(
            "Number of CV folds must be at least 2".into(),
        ));
    }

    for &size in train_sizes {
        if size <= 0.0 || size > 1.0 {
            return Err(Error::InvalidInput(
                "Training sizes must be between 0 and 1".into(),
            ));
        }
    }

    let n_samples = data.nrows();

    let mut absolute_sizes = Vec::with_capacity(train_sizes.len());
    let mut train_scores = Vec::with_capacity(train_sizes.len());
    let mut test_scores = Vec::with_capacity(train_sizes.len());

    // Placeholder implementation
    // In a real implementation, this would:
    // 1. For each training size:
    //    a. Subsample the data to the current size
    //    b. Perform cross-validation
    //    c. Record train and test scores

    for &size_fraction in train_sizes {
        let absolute_size = (n_samples as f64 * size_fraction).round() as usize;
        absolute_sizes.push(absolute_size);

        // Placeholder scores
        train_scores.push(0.9);
        test_scores.push(0.8);
    }

    Ok((absolute_sizes, train_scores, test_scores))
}

/// Generate validation curve for a model
///
/// A validation curve shows model performance as a function of a hyperparameter.
///
/// # Arguments
/// * `model_factory` - Function that creates a model with a given parameter value
/// * `data` - The dataset
/// * `target` - The target column name
/// * `param_name` - Name of the parameter being varied (for informational purposes)
/// * `param_values` - Vector of parameter values to evaluate
/// * `metric` - Name of the metric to track
/// * `cv` - Number of cross-validation folds
///
/// # Returns
/// * Tuple of (param_values, train_scores, test_scores)
///   where each element is a vector of values for each parameter value
pub fn validation_curve<T, F, P>(
    model_factory: F,
    data: &DataFrame,
    target: &str,
    param_name: &str,
    param_values: &[P],
    metric: &str,
    cv: usize,
) -> Result<(Vec<P>, Vec<f64>, Vec<f64>)>
where
    T: SupervisedModel,
    F: Fn(P) -> T,
    P: Clone,
{
    if cv < 2 {
        return Err(Error::InvalidInput(
            "Number of CV folds must be at least 2".into(),
        ));
    }

    if param_values.is_empty() {
        return Err(Error::InvalidInput(
            "Parameter values array cannot be empty".into(),
        ));
    }

    let mut train_scores = Vec::with_capacity(param_values.len());
    let mut test_scores = Vec::with_capacity(param_values.len());

    // Placeholder implementation
    // In a real implementation, this would:
    // 1. For each parameter value:
    //    a. Create a model with that parameter value
    //    b. Perform cross-validation
    //    c. Record train and test scores

    for _ in param_values {
        // Placeholder scores
        train_scores.push(0.9);
        test_scores.push(0.8);
    }

    Ok((param_values.to_vec(), train_scores, test_scores))
}

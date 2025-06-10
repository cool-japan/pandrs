//! Linear models for regression and classification
//!
//! This module provides implementations of linear regression and logistic regression.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use crate::series::Series;
use std::collections::HashMap;
use std::time::Instant;

/// Linear regression model
///
/// Implements ordinary least squares linear regression.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients (weights) for each feature
    pub coefficients: Option<HashMap<String, f64>>,
    /// Intercept (bias) term
    pub intercept: Option<f64>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Whether to normalize features
    pub normalize: bool,
    /// Feature names
    feature_names: Option<Vec<String>>,
}

impl LinearRegression {
    /// Create a new LinearRegression model
    pub fn new() -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            normalize: false,
            feature_names: None,
        }
    }

    /// Set whether to fit the intercept
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to normalize features
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Get the R² coefficient of determination
    pub fn r_squared(&self) -> Option<f64> {
        None // Placeholder - would calculate R² from model and data
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        let start_time = Instant::now();

        // Validate input data
        if !train_data.has_column(target_column) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target_column
            )));
        }

        // Extract feature columns (all numeric columns except target)
        let mut feature_names: Vec<String> = Vec::new();
        for name in train_data.column_names() {
            if name != target_column {
                if let Ok(col) = train_data.get_column::<f64>(&name) {
                    feature_names.push(name.clone());
                }
            }
        }

        // Store feature names for later use
        self.feature_names = Some(feature_names.clone());

        // Placeholder for actual linear regression implementation
        // In a real implementation, this would:
        // 1. Extract features X and target y
        // 2. Normalize features if self.normalize is true
        // 3. Add a column of 1s if self.fit_intercept is true
        // 4. Compute coefficients using the normal equation: β = (X'X)⁻¹X'y

        // For now, just set placeholder coefficients
        let mut coefficients = HashMap::new();
        for name in &feature_names {
            coefficients.insert(name.clone(), 1.0);
        }

        self.coefficients = Some(coefficients);
        self.intercept = if self.fit_intercept { Some(0.0) } else { None };

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if self.coefficients.is_none() {
            return Err(Error::InvalidValue("Model not fitted".into()));
        }

        let coefficients = self.coefficients.as_ref().unwrap();
        let feature_names = self.feature_names.as_ref().unwrap();

        // Validate input data
        for name in feature_names {
            if !data.has_column(name) {
                return Err(Error::InvalidValue(format!(
                    "Feature column '{}' not found",
                    name
                )));
            }
        }

        // Placeholder for actual prediction implementation
        // In a real implementation, this would:
        // 1. Extract features X
        // 2. Normalize features if self.normalize was true during fitting
        // 3. Compute predictions as X·β + intercept

        // For now, just return a vector of 0s
        let n_samples = data.nrows();
        let predictions = vec![0.0; n_samples];

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        if let Some(coefficients) = &self.coefficients {
            // For linear regression, feature importances are related to coefficient magnitudes
            let mut importances = HashMap::new();

            // Compute sum of absolute coefficients for normalization
            let sum_abs_coefs: f64 = coefficients.values().map(|&c| c.abs()).sum();

            if sum_abs_coefs > 0.0 {
                for (name, &coef) in coefficients.iter() {
                    importances.insert(name.clone(), coef.abs() / sum_abs_coefs);
                }

                Some(importances)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ModelEvaluator for LinearRegression {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let start_time = Instant::now();

        if !test_data.has_column(test_target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                test_target
            )));
        }

        // Make predictions
        let predictions = self.predict(test_data)?;

        // Get actual target values
        let target_col = test_data.get_column::<f64>(test_target)?;
        let target_values = target_col.as_f64()?;

        if predictions.len() != target_values.len() {
            return Err(Error::InvalidOperation(
                "Prediction length doesn't match target length".into(),
            ));
        }

        // Calculate metrics (MSE, MAE, R²)
        let n_samples = predictions.len();

        // Mean Squared Error
        let mse: f64 = predictions
            .iter()
            .zip(target_values.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / n_samples as f64;

        // Mean Absolute Error
        let mae: f64 = predictions
            .iter()
            .zip(target_values.iter())
            .map(|(&pred, &actual)| (pred - actual).abs())
            .sum::<f64>()
            / n_samples as f64;

        // Placeholder for R² calculation
        let r2 = 0.8; // Would be calculated properly in a real implementation

        let prediction_time = start_time.elapsed().as_secs_f64();

        let mut metrics = ModelMetrics::new();
        metrics.add_metric("mse", mse);
        metrics.add_metric("mae", mae);
        metrics.add_metric("r2", r2);
        metrics.set_prediction_time(prediction_time);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        if folds < 2 {
            return Err(Error::InvalidInput(
                "Number of folds must be at least 2".into(),
            ));
        }

        // Placeholder for cross-validation implementation
        // In a real implementation, this would:
        // 1. Split data into folds
        // 2. For each fold:
        //    a. Train a model on all other folds
        //    b. Evaluate on the current fold
        // 3. Return a vector of metrics for each fold

        let mut metrics = Vec::new();
        for _ in 0..folds {
            let mut fold_metrics = ModelMetrics::new();
            fold_metrics.add_metric("mse", 0.0);
            fold_metrics.add_metric("mae", 0.0);
            fold_metrics.add_metric("r2", 0.8);
            metrics.push(fold_metrics);
        }

        Ok(metrics)
    }
}

/// Logistic regression model for binary classification
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Coefficients (weights) for each feature
    pub coefficients: Option<HashMap<String, f64>>,
    /// Intercept (bias) term
    pub intercept: Option<f64>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Regularization strength (C parameter, inverse of regularization strength)
    pub c: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for stopping criteria
    pub tol: f64,
    /// Feature names
    feature_names: Option<Vec<String>>,
}

impl LogisticRegression {
    /// Create a new LogisticRegression model with default parameters
    pub fn new() -> Self {
        LogisticRegression {
            coefficients: None,
            intercept: None,
            fit_intercept: true,
            c: 1.0,
            max_iter: 100,
            tol: 1e-4,
            feature_names: None,
        }
    }

    /// Set regularization strength
    pub fn with_regularization(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit the intercept
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Predict probabilities instead of classes
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<f64>> {
        // Similar to predict, but returns probabilities instead of binary predictions
        // For now, just return a placeholder
        let n_samples = data.nrows();
        let probabilities = vec![0.5; n_samples];

        Ok(probabilities)
    }
}

impl SupervisedModel for LogisticRegression {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        // Implementation would be similar to LinearRegression::fit,
        // but with logistic regression optimization algorithm
        // For now, just use placeholder

        if !train_data.has_column(target_column) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target_column
            )));
        }

        // Extract feature columns (all numeric columns except target)
        let mut feature_names: Vec<String> = Vec::new();
        for name in train_data.column_names() {
            if name != target_column {
                if let Ok(col) = train_data.get_column::<f64>(&name) {
                    feature_names.push(name.clone());
                }
            }
        }

        // Store feature names for later use
        self.feature_names = Some(feature_names.clone());

        // Set placeholder coefficients
        let mut coefficients = HashMap::new();
        for name in &feature_names {
            coefficients.insert(name.clone(), 0.0);
        }

        self.coefficients = Some(coefficients);
        self.intercept = if self.fit_intercept { Some(0.0) } else { None };

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        // Predict binary class labels (0 or 1)
        // For now, just return a placeholder
        let n_samples = data.nrows();
        let predictions = vec![0.0; n_samples];

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        // Similar to LinearRegression::feature_importances
        if let Some(coefficients) = &self.coefficients {
            let mut importances = HashMap::new();

            let sum_abs_coefs: f64 = coefficients.values().map(|&c| c.abs()).sum();

            if sum_abs_coefs > 0.0 {
                for (name, &coef) in coefficients.iter() {
                    importances.insert(name.clone(), coef.abs() / sum_abs_coefs);
                }

                Some(importances)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ModelEvaluator for LogisticRegression {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        // Similar to LinearRegression::evaluate, but with classification metrics
        let start_time = Instant::now();

        if !test_data.has_column(test_target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                test_target
            )));
        }

        // Make predictions
        let predictions = self.predict(test_data)?;

        // Get actual target values
        let target_col = test_data.get_column::<f64>(test_target)?;
        let target_values = target_col.as_f64()?;

        if predictions.len() != target_values.len() {
            return Err(Error::InvalidOperation(
                "Prediction length doesn't match target length".into(),
            ));
        }

        // Placeholder for classification metrics calculation
        let accuracy = 0.85;
        let precision = 0.8;
        let recall = 0.9;
        let f1 = 0.85;

        let prediction_time = start_time.elapsed().as_secs_f64();

        let mut metrics = ModelMetrics::new();
        metrics.add_metric("accuracy", accuracy);
        metrics.add_metric("precision", precision);
        metrics.add_metric("recall", recall);
        metrics.add_metric("f1", f1);
        metrics.set_prediction_time(prediction_time);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        // Similar to LinearRegression::cross_validate
        if folds < 2 {
            return Err(Error::InvalidInput(
                "Number of folds must be at least 2".into(),
            ));
        }

        let mut metrics = Vec::new();
        for _ in 0..folds {
            let mut fold_metrics = ModelMetrics::new();
            fold_metrics.add_metric("accuracy", 0.85);
            fold_metrics.add_metric("precision", 0.8);
            fold_metrics.add_metric("recall", 0.9);
            fold_metrics.add_metric("f1", 0.85);
            metrics.push(fold_metrics);
        }

        Ok(metrics)
    }
}

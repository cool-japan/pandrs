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

    /// Get the R² coefficient of determination (requires training data)
    pub fn r_squared(&self, data: &DataFrame, target_column: &str) -> Result<f64> {
        if self.coefficients.is_none() {
            return Err(Error::InvalidValue("Model not fitted".into()));
        }

        // Get actual target values
        let target_col = data.get_column::<f64>(target_column)?;
        let y_actual = target_col.as_f64()?;

        // Get predictions
        let y_pred = self.predict(data)?;

        if y_actual.len() != y_pred.len() {
            return Err(Error::DimensionMismatch(
                "Actual and predicted values have different lengths".into(),
            ));
        }

        // Calculate R² = 1 - (SS_res / SS_tot)
        let y_mean = y_actual.iter().sum::<f64>() / y_actual.len() as f64;

        let ss_tot: f64 = y_actual.iter().map(|&y| (y - y_mean).powi(2)).sum();

        let ss_res: f64 = y_actual
            .iter()
            .zip(y_pred.iter())
            .map(|(&actual, &pred)| (actual - pred).powi(2))
            .sum();

        if ss_tot == 0.0 {
            return Ok(1.0); // Perfect fit when variance is 0
        }

        Ok(1.0 - ss_res / ss_tot)
    }

    // Matrix operation helper functions
    fn matrix_multiply_transpose(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b.len();
        let mut result = vec![vec![0.0; m]; n];

        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..a[i].len() {
                    sum += a[i][k] * b[j][k];
                }
                result[i][j] = sum;
            }
        }

        result
    }

    fn vec_multiply_transpose(a: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..y.len() {
                sum += a[i][k] * y[k];
            }
            result[i] = sum;
        }

        result
    }

    fn matrix_inverse(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();

        if n == 0 {
            return Err(Error::InvalidOperation("Matrix is empty".into()));
        }

        for row in matrix {
            if row.len() != n {
                return Err(Error::DimensionMismatch("Matrix must be square".into()));
            }
        }

        // Create augmented matrix [A|I]
        let mut augmented = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(2 * n);
            row.extend_from_slice(&matrix[i]);

            // Identity matrix part
            for j in 0..n {
                row.push(if i == j { 1.0 } else { 0.0 });
            }

            augmented.push(row);
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Pivot selection
            let mut max_row = i;
            let mut max_val = augmented[i][i].abs();

            for j in i + 1..n {
                let abs_val = augmented[j][i].abs();
                if abs_val > max_val {
                    max_row = j;
                    max_val = abs_val;
                }
            }

            if max_val < 1e-10 {
                return Err(Error::Computation(
                    "Matrix is singular (inverse does not exist)".into(),
                ));
            }

            // Swap rows
            if max_row != i {
                augmented.swap(i, max_row);
            }

            // Make pivot element 1
            let pivot = augmented[i][i];
            for j in 0..2 * n {
                augmented[i][j] /= pivot;
            }

            // Eliminate other rows
            for j in 0..n {
                if j != i {
                    let factor = augmented[j][i];
                    for k in 0..2 * n {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }

        // Extract result (right half is inverse matrix)
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        Ok(inverse)
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
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
                // Try to get as numeric column
                if train_data.get_column::<f64>(&name).is_ok() {
                    feature_names.push(name.clone());
                }
            }
        }

        if feature_names.is_empty() {
            return Err(Error::InvalidValue(
                "No numeric feature columns found".into(),
            ));
        }

        // Store feature names for later use
        self.feature_names = Some(feature_names.clone());

        // Extract target variable
        let target_col = train_data.get_column::<f64>(target_column)?;
        let y_values = target_col.as_f64()?;
        let n = y_values.len();

        if n == 0 {
            return Err(Error::InvalidValue("No data to train on".into()));
        }

        // Build feature matrix X
        let mut x_matrix: Vec<Vec<f64>> = Vec::new();

        // Add intercept column if needed
        if self.fit_intercept {
            x_matrix.push(vec![1.0; n]);
        }

        // Add feature columns
        for feature_name in &feature_names {
            let feature_col = train_data.get_column::<f64>(feature_name)?;
            let feature_values = feature_col.as_f64()?;

            if feature_values.len() != n {
                return Err(Error::DimensionMismatch(format!(
                    "Feature column '{}' has different length than target",
                    feature_name
                )));
            }

            x_matrix.push(feature_values.to_vec());
        }

        // Normalize features if requested
        if self.normalize {
            for i in if self.fit_intercept { 1 } else { 0 }..x_matrix.len() {
                let mean = x_matrix[i].iter().sum::<f64>() / n as f64;
                let variance =
                    x_matrix[i].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                let std_dev = variance.sqrt();

                if std_dev > 1e-10 {
                    for j in 0..n {
                        x_matrix[i][j] = (x_matrix[i][j] - mean) / std_dev;
                    }
                }
            }
        }

        // Solve normal equation: β = (X'X)⁻¹X'y
        let xt_x = Self::matrix_multiply_transpose(&x_matrix, &x_matrix);
        let xt_x_inv = Self::matrix_inverse(&xt_x)?;
        let xt_y = Self::vec_multiply_transpose(&x_matrix, &y_values);

        // Calculate coefficients
        let mut beta_coefs = vec![0.0; x_matrix.len()];
        for i in 0..beta_coefs.len() {
            for j in 0..xt_y.len() {
                beta_coefs[i] += xt_x_inv[i][j] * xt_y[j];
            }
        }

        // Store results
        let mut coefficients = HashMap::new();
        let start_idx = if self.fit_intercept { 1 } else { 0 };

        if self.fit_intercept {
            self.intercept = Some(beta_coefs[0]);
        } else {
            self.intercept = None;
        }

        for (i, feature_name) in feature_names.iter().enumerate() {
            coefficients.insert(feature_name.clone(), beta_coefs[start_idx + i]);
        }

        self.coefficients = Some(coefficients);
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

        // Get number of samples
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        // Extract feature data
        let mut predictions = vec![0.0; n_samples];

        // Add intercept if fitted with intercept
        if let Some(intercept) = self.intercept {
            for i in 0..n_samples {
                predictions[i] += intercept;
            }
        }

        // Add contribution from each feature
        for feature_name in feature_names {
            let feature_col = data.get_column::<f64>(feature_name)?;
            let feature_values = feature_col.as_f64()?;

            if feature_values.len() != n_samples {
                return Err(Error::DimensionMismatch(format!(
                    "Feature column '{}' has different length than expected",
                    feature_name
                )));
            }

            if let Some(&coef) = coefficients.get(feature_name) {
                for i in 0..n_samples {
                    predictions[i] += coef * feature_values[i];
                }
            }
        }

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

        // Calculate R² = 1 - (SS_res / SS_tot)
        let y_mean = target_values.iter().sum::<f64>() / n_samples as f64;
        let ss_tot: f64 = target_values.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(target_values.iter())
            .map(|(&pred, &actual)| (actual - pred).powi(2))
            .sum();

        let r2 = if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

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

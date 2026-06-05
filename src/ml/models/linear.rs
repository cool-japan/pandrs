//! Linear models for regression and classification
//!
//! This module provides implementations of linear regression and logistic regression.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use crate::series::Series;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Module-level matrix helpers (used by both LinearRegression and
// LogisticRegression)
// ---------------------------------------------------------------------------

/// Compute A * B^T where result[i][j] = dot(A[i], B[j]).
/// Both `a` and `b` are stored as column vectors (column-major): a[col][row].
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

/// Compute A * y where result[i] = dot(A[i], y).
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

/// Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
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

        if max_row != i {
            augmented.swap(i, max_row);
        }

        let pivot = augmented[i][i];
        for j in 0..2 * n {
            augmented[i][j] /= pivot;
        }

        for j in 0..n {
            if j != i {
                let factor = augmented[j][i];
                for k in 0..2 * n {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    // Extract right half (inverse matrix)
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = augmented[i][j + n];
        }
    }

    Ok(inverse)
}

/// Sigmoid function σ(x) = 1 / (1 + e^{-x}).
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Convert a column-major matrix (x_cols[col][row]) to row-major (result[row][col]).
fn transpose_to_row_major(x_cols: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let p = x_cols.len();
    let mut x_rows = vec![vec![0.0_f64; p]; n];
    for (col_idx, col) in x_cols.iter().enumerate() {
        for (row_idx, &val) in col.iter().enumerate() {
            x_rows[row_idx][col_idx] = val;
        }
    }
    x_rows
}

/// Compute linear predictor η = Xβ given row-major X and coefficient vector β.
fn linear_predictor(x_rows: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    x_rows
        .iter()
        .map(|row| row.iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum())
        .collect()
}

// ---------------------------------------------------------------------------
// LinearRegression
// ---------------------------------------------------------------------------

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

        let target_col = data.get_column::<f64>(target_column)?;
        let y_actual = target_col.as_f64()?;
        let y_pred = self.predict(data)?;

        if y_actual.len() != y_pred.len() {
            return Err(Error::DimensionMismatch(
                "Actual and predicted values have different lengths".into(),
            ));
        }

        let y_mean = y_actual.iter().sum::<f64>() / y_actual.len() as f64;
        let ss_tot: f64 = y_actual.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = y_actual
            .iter()
            .zip(y_pred.iter())
            .map(|(&actual, &pred)| (actual - pred).powi(2))
            .sum();

        if ss_tot == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        if !train_data.has_column(target_column) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target_column
            )));
        }

        // Collect numeric feature columns (excluding target)
        let mut feature_names: Vec<String> = Vec::new();
        for name in train_data.column_names() {
            if name != target_column && train_data.get_column::<f64>(&name).is_ok() {
                feature_names.push(name.clone());
            }
        }

        if feature_names.is_empty() {
            return Err(Error::InvalidValue(
                "No numeric feature columns found".into(),
            ));
        }

        self.feature_names = Some(feature_names.clone());

        let target_col = train_data.get_column::<f64>(target_column)?;
        let y_values = target_col.as_f64()?;
        let n = y_values.len();

        if n == 0 {
            return Err(Error::InvalidValue("No data to train on".into()));
        }

        // Build column-major feature matrix
        let mut x_matrix: Vec<Vec<f64>> = Vec::new();
        if self.fit_intercept {
            x_matrix.push(vec![1.0; n]);
        }
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

        // Optional feature normalisation
        if self.normalize {
            let start = if self.fit_intercept { 1 } else { 0 };
            for col in x_matrix[start..].iter_mut() {
                let mean = col.iter().sum::<f64>() / n as f64;
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                let std_dev = variance.sqrt();
                if std_dev > 1e-10 {
                    for v in col.iter_mut() {
                        *v = (*v - mean) / std_dev;
                    }
                }
            }
        }

        // Normal equation: β = (X'X)^{-1} X'y
        let xt_x = matrix_multiply_transpose(&x_matrix, &x_matrix);
        let xt_x_inv = matrix_inverse(&xt_x)?;
        let xt_y = vec_multiply_transpose(&x_matrix, &y_values);

        let mut beta_coefs = vec![0.0; x_matrix.len()];
        for i in 0..beta_coefs.len() {
            for j in 0..xt_y.len() {
                beta_coefs[i] += xt_x_inv[i][j] * xt_y[j];
            }
        }

        let start_idx = if self.fit_intercept { 1 } else { 0 };

        if self.fit_intercept {
            self.intercept = Some(beta_coefs[0]);
        } else {
            self.intercept = None;
        }

        let mut coefficients = HashMap::new();
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

        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;
        let feature_names = self
            .feature_names
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        for name in feature_names {
            if !data.has_column(name) {
                return Err(Error::InvalidValue(format!(
                    "Feature column '{}' not found",
                    name
                )));
            }
        }

        let n_samples = data.nrows();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let mut predictions = vec![0.0; n_samples];

        if let Some(intercept) = self.intercept {
            for pred in predictions.iter_mut() {
                *pred += intercept;
            }
        }

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

impl ModelEvaluator for LinearRegression {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let start_time = Instant::now();

        if !test_data.has_column(test_target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                test_target
            )));
        }

        let predictions = self.predict(test_data)?;

        let target_col = test_data.get_column::<f64>(test_target)?;
        let target_values = target_col.as_f64()?;

        if predictions.len() != target_values.len() {
            return Err(Error::InvalidOperation(
                "Prediction length doesn't match target length".into(),
            ));
        }

        let n_samples = predictions.len();

        let mse: f64 = predictions
            .iter()
            .zip(target_values.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / n_samples as f64;

        let mae: f64 = predictions
            .iter()
            .zip(target_values.iter())
            .map(|(&pred, &actual)| (pred - actual).abs())
            .sum::<f64>()
            / n_samples as f64;

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

        let n = data.nrows();
        if n < folds {
            return Err(Error::InvalidInput(
                "Number of samples must be at least equal to the number of folds".into(),
            ));
        }

        let fold_size = n / folds;

        // Compute fold boundaries; last fold absorbs the remainder
        let fold_starts: Vec<usize> = (0..folds)
            .map(|i| if i == 0 { 0 } else { i * fold_size })
            .collect();
        let fold_ends: Vec<usize> = (0..folds)
            .map(|i| {
                if i == folds - 1 {
                    n
                } else {
                    (i + 1) * fold_size
                }
            })
            .collect();

        let mut all_metrics: Vec<ModelMetrics> = Vec::with_capacity(folds);

        for fold_idx in 0..folds {
            let test_start = fold_starts[fold_idx];
            let test_end = fold_ends[fold_idx];

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();

            if train_indices.is_empty() || test_indices.is_empty() {
                return Err(Error::InvalidInput(
                    "A fold resulted in empty train or test set".into(),
                ));
            }

            let train_df = data.sample(&train_indices)?;
            let test_df = data.sample(&test_indices)?;

            let mut model = self.clone();
            model.fit(&train_df, target)?;
            let fold_metrics = model.evaluate(&test_df, target)?;
            all_metrics.push(fold_metrics);
        }

        Ok(all_metrics)
    }
}

// ---------------------------------------------------------------------------
// LogisticRegression
// ---------------------------------------------------------------------------

/// Logistic regression model for binary classification.
///
/// Uses Iteratively Reweighted Least Squares (IRLS) for fitting with optional
/// L2 (Ridge) regularisation controlled by the `c` parameter (inverse of λ).
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
    /// Maximum number of IRLS iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Feature names stored during fitting
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

    /// Set regularization strength (C = 1/λ; larger C = less regularisation)
    pub fn with_regularization(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set maximum number of IRLS iterations
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

    // ------------------------------------------------------------------
    // Internal: IRLS solver
    // ------------------------------------------------------------------

    /// Run IRLS on a row-major feature matrix `x_rows` (n×p) and binary
    /// target vector `y` (length n).  Returns the fitted β vector of length p.
    /// When `fit_intercept` is true, β[0] is the intercept and β[1..] are the
    /// per-feature coefficients.
    fn irls_fit(&self, x_rows: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>> {
        let n = x_rows.len();
        let p = if n > 0 { x_rows[0].len() } else { 0 };

        // Initialise β = 0
        let mut beta = vec![0.0_f64; p];

        for _iter in 0..self.max_iter {
            // η = Xβ
            let eta = linear_predictor(x_rows, &beta);

            // μ = σ(η)
            let mu: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();

            // W_ii = max(μ_i * (1 - μ_i), 1e-10)
            let w_diag: Vec<f64> = mu.iter().map(|&m| (m * (1.0 - m)).max(1e-10)).collect();

            // Working response z_i = η_i + (y_i - μ_i) / W_ii
            let z: Vec<f64> = (0..n)
                .map(|i| eta[i] + (y[i] - mu[i]) / w_diag[i])
                .collect();

            // Build weighted (column-major) design matrix and weighted response
            //   X_w[col][row] = sqrt(W[row]) * X[row][col]
            //   z_w[row]       = sqrt(W[row]) * z[row]
            let w_sqrt: Vec<f64> = w_diag.iter().map(|&w| w.sqrt()).collect();

            let mut xw_cols: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; p];
            for i in 0..n {
                for j in 0..p {
                    xw_cols[j][i] = w_sqrt[i] * x_rows[i][j];
                }
            }
            let zw: Vec<f64> = (0..n).map(|i| w_sqrt[i] * z[i]).collect();

            // (Xw'Xw) and its inverse
            let xwt_xw = matrix_multiply_transpose(&xw_cols, &xw_cols);
            let xwt_xw_inv = matrix_inverse(&xwt_xw).map_err(|_| {
                Error::Computation(
                    "IRLS: X'WX is singular; try increasing regularisation (lower C) \
                     or reducing feature dimensionality"
                        .into(),
                )
            })?;

            // Xw' zw
            let xwt_zw = vec_multiply_transpose(&xw_cols, &zw);

            // β_new = (Xw'Xw)^{-1} Xw'zw
            let mut beta_new = vec![0.0_f64; p];
            for i in 0..p {
                for j in 0..p {
                    beta_new[i] += xwt_xw_inv[i][j] * xwt_zw[j];
                }
            }

            // L2 regularisation: shrink non-intercept coefficients by factor
            //   1 / (1 + 1/(C·n))
            let intercept_offset = if self.fit_intercept { 1 } else { 0 };
            let reg_factor = 1.0 / (1.0 + 1.0 / (self.c * (n as f64)));
            for j in intercept_offset..p {
                beta_new[j] *= reg_factor;
            }

            // Convergence: relative change in β
            let delta: Vec<f64> = beta_new
                .iter()
                .zip(beta.iter())
                .map(|(&bn, &b)| bn - b)
                .collect();
            let rel_change = l2_norm(&delta) / (1.0 + l2_norm(&beta));
            beta = beta_new;
            if rel_change < self.tol {
                break;
            }
        }

        Ok(beta)
    }

    // ------------------------------------------------------------------
    // Public probability prediction
    // ------------------------------------------------------------------

    /// Compute raw sigmoid probabilities P(y=1|x) for each row of `data`.
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if self.coefficients.is_none() {
            return Err(Error::InvalidValue("Model not fitted".into()));
        }

        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;
        let feature_names = self
            .feature_names
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        let n_samples = data.nrows();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        // Start with intercept
        let mut eta = vec![self.intercept.unwrap_or(0.0); n_samples];

        for fname in feature_names {
            if !data.has_column(fname) {
                return Err(Error::InvalidValue(format!(
                    "Feature column '{}' not found",
                    fname
                )));
            }
            let col = data.get_column::<f64>(fname)?;
            let vals = col.as_f64()?;
            if vals.len() != n_samples {
                return Err(Error::DimensionMismatch(format!(
                    "Feature column '{}' has unexpected length",
                    fname
                )));
            }
            if let Some(&coef) = coefficients.get(fname) {
                for i in 0..n_samples {
                    eta[i] += coef * vals[i];
                }
            }
        }

        Ok(eta.iter().map(|&e| sigmoid(e)).collect())
    }
}

// ---------------------------------------------------------------------------
// SupervisedModel for LogisticRegression
// ---------------------------------------------------------------------------

impl SupervisedModel for LogisticRegression {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        if !train_data.has_column(target_column) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target_column
            )));
        }

        // Collect numeric feature columns (excluding target)
        let mut feature_names: Vec<String> = Vec::new();
        for name in train_data.column_names() {
            if name != target_column && train_data.get_column::<f64>(&name).is_ok() {
                feature_names.push(name.clone());
            }
        }

        if feature_names.is_empty() {
            return Err(Error::InvalidValue(
                "No numeric feature columns found".into(),
            ));
        }

        let target_col = train_data.get_column::<f64>(target_column)?;
        let y_raw = target_col.as_f64()?;
        let n = y_raw.len();

        if n == 0 {
            return Err(Error::InvalidValue("No data to train on".into()));
        }

        // Clamp labels to [0, 1]
        let y: Vec<f64> = y_raw.iter().map(|&v| v.clamp(0.0, 1.0)).collect();

        // Build column-major design matrix (intercept column first when applicable)
        let mut x_cols: Vec<Vec<f64>> = Vec::new();
        if self.fit_intercept {
            x_cols.push(vec![1.0; n]);
        }
        for fname in &feature_names {
            let col = train_data.get_column::<f64>(fname)?;
            let vals = col.as_f64()?;
            if vals.len() != n {
                return Err(Error::DimensionMismatch(format!(
                    "Feature column '{}' has different length than target",
                    fname
                )));
            }
            x_cols.push(vals.to_vec());
        }

        // Convert to row-major for IRLS
        let x_rows = transpose_to_row_major(&x_cols, n);

        // Run IRLS
        let beta = self.irls_fit(&x_rows, &y)?;

        // Unpack results
        let intercept_offset = if self.fit_intercept { 1 } else { 0 };

        if self.fit_intercept {
            self.intercept = Some(beta[0]);
        } else {
            self.intercept = None;
        }

        let mut coefficients = HashMap::new();
        for (i, fname) in feature_names.iter().enumerate() {
            coefficients.insert(fname.clone(), beta[intercept_offset + i]);
        }

        self.coefficients = Some(coefficients);
        self.feature_names = Some(feature_names);

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if self.coefficients.is_none() {
            return Err(Error::InvalidValue("Model not fitted".into()));
        }

        let probas = self.predict_proba(data)?;
        Ok(probas
            .iter()
            .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
            .collect())
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
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

// ---------------------------------------------------------------------------
// ModelEvaluator for LogisticRegression
// ---------------------------------------------------------------------------

impl ModelEvaluator for LogisticRegression {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let start_time = Instant::now();

        if self.coefficients.is_none() {
            return Err(Error::InvalidValue("Model not fitted".into()));
        }

        if !test_data.has_column(test_target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                test_target
            )));
        }

        let predictions = self.predict(test_data)?;

        let target_col = test_data.get_column::<f64>(test_target)?;
        let target_values = target_col.as_f64()?;

        if predictions.len() != target_values.len() {
            return Err(Error::InvalidOperation(
                "Prediction length doesn't match target length".into(),
            ));
        }

        let n = predictions.len();

        // Build confusion counts
        let mut tp = 0.0_f64;
        let mut tn = 0.0_f64;
        let mut fp = 0.0_f64;
        let mut fn_ = 0.0_f64;

        for i in 0..n {
            let pred_pos = predictions[i] >= 0.5;
            let actual_pos = target_values[i] >= 0.5;
            match (pred_pos, actual_pos) {
                (true, true) => tp += 1.0,
                (true, false) => fp += 1.0,
                (false, true) => fn_ += 1.0,
                (false, false) => tn += 1.0,
            }
        }

        let accuracy = (tp + tn) / n as f64;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

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
        if folds < 2 {
            return Err(Error::InvalidInput(
                "Number of folds must be at least 2".into(),
            ));
        }

        let n = data.nrows();
        if n < folds {
            return Err(Error::InvalidInput(
                "Number of samples must be at least equal to the number of folds".into(),
            ));
        }

        let fold_size = n / folds;

        let fold_starts: Vec<usize> = (0..folds)
            .map(|i| if i == 0 { 0 } else { i * fold_size })
            .collect();
        let fold_ends: Vec<usize> = (0..folds)
            .map(|i| {
                if i == folds - 1 {
                    n
                } else {
                    (i + 1) * fold_size
                }
            })
            .collect();

        let mut all_metrics: Vec<ModelMetrics> = Vec::with_capacity(folds);

        for fold_idx in 0..folds {
            let test_start = fold_starts[fold_idx];
            let test_end = fold_ends[fold_idx];

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();

            if train_indices.is_empty() || test_indices.is_empty() {
                return Err(Error::InvalidInput(
                    "A fold resulted in empty train or test set".into(),
                ));
            }

            let train_df = data.sample(&train_indices)?;
            let test_df = data.sample(&test_indices)?;

            let mut model = self.clone();
            model.fit(&train_df, target)?;
            let fold_metrics = model.evaluate(&test_df, target)?;
            all_metrics.push(fold_metrics);
        }

        Ok(all_metrics)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Linearly separable dataset: 5 samples near 0.0 (class 0), 5 near 5.0 (class 1).
    fn make_separable_df() -> DataFrame {
        let features: Vec<f64> = vec![0.0, 0.1, -0.1, 0.05, -0.05, 5.0, 4.9, 5.1, 4.95, 5.05];
        let labels: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(features, Some("x".to_string())).expect("series creation"),
        )
        .expect("add column");
        df.add_column(
            "y".to_string(),
            Series::new(labels, Some("y".to_string())).expect("series creation"),
        )
        .expect("add column");
        df
    }

    /// Perfect linear dataset: y = 2*x + 1 for x in 0..10.
    fn make_linear_df() -> DataFrame {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(x, Some("x".to_string())).expect("series creation"),
        )
        .expect("add column");
        df.add_column(
            "y".to_string(),
            Series::new(y, Some("y".to_string())).expect("series creation"),
        )
        .expect("add column");
        df
    }

    #[test]
    fn test_logistic_regression_linearly_separable() {
        let df = make_separable_df();
        let mut model = LogisticRegression::new();
        model.fit(&df, "y").expect("fit should succeed");
        let predictions = model.predict(&df).expect("predict should succeed");
        assert_eq!(predictions.len(), 10);

        let ground_truth = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let correct: usize = predictions
            .iter()
            .zip(ground_truth.iter())
            .filter(|(&pred, &actual)| (pred - actual).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / 10.0;
        assert!(
            accuracy >= 0.9,
            "Expected accuracy >= 0.9, got {}",
            accuracy
        );
    }

    #[test]
    fn test_logistic_regression_predict_proba() {
        let df = make_separable_df();
        let mut model = LogisticRegression::new();
        model.fit(&df, "y").expect("fit should succeed");
        let probas = model
            .predict_proba(&df)
            .expect("predict_proba should succeed");
        assert_eq!(probas.len(), 10);

        for &p in &probas {
            assert!(p >= 0.0 && p <= 1.0, "Probability {} not in [0, 1]", p);
        }

        let low_avg = probas[..5].iter().sum::<f64>() / 5.0;
        let high_avg = probas[5..].iter().sum::<f64>() / 5.0;
        assert!(
            low_avg < 0.5,
            "Average probability for class-0 samples should be < 0.5, got {}",
            low_avg
        );
        assert!(
            high_avg > 0.5,
            "Average probability for class-1 samples should be > 0.5, got {}",
            high_avg
        );
    }

    #[test]
    fn test_logistic_regression_evaluate() {
        let df = make_separable_df();
        let mut model = LogisticRegression::new();
        model.fit(&df, "y").expect("fit should succeed");
        let metrics = model.evaluate(&df, "y").expect("evaluate should succeed");
        let acc = metrics.get_metric("accuracy").copied().unwrap_or(0.0);
        assert!(acc >= 0.9, "Expected accuracy >= 0.9, got {}", acc);
    }

    #[test]
    fn test_linear_regression_cross_validate() {
        let df = make_linear_df();
        let model = LinearRegression::new();
        let fold_metrics = model
            .cross_validate(&df, "y", 3)
            .expect("cross_validate should succeed");
        assert_eq!(fold_metrics.len(), 3);
        for (fold_idx, fm) in fold_metrics.iter().enumerate() {
            let r2 = fm.get_metric("r2").copied().unwrap_or(0.0);
            assert!(
                r2 >= 0.9,
                "Expected fold {} r2 >= 0.9, got {}",
                fold_idx,
                r2
            );
        }
    }

    #[test]
    fn test_logistic_unfitted_errors() {
        let df = make_separable_df();
        let model = LogisticRegression::new();

        let pred_result = model.predict(&df);
        assert!(
            pred_result.is_err(),
            "predict on unfitted model should return Err"
        );

        let eval_result = model.evaluate(&df, "y");
        assert!(
            eval_result.is_err(),
            "evaluate on unfitted model should return Err"
        );
    }
}

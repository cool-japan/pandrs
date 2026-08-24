//! Linear models for regression and classification
//!
//! This module provides implementations of linear regression and logistic regression.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::selection::{
    err_on_unknown_params, parse_param_bool, parse_param_f64, parse_param_usize, TunableModel,
};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
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

/// Solve the square linear system `A x = b` by Gaussian elimination with partial
/// pivoting and back substitution.
///
/// Solving the normal equations directly is both cheaper and numerically better
/// conditioned than forming `A^{-1}` and multiplying by it, which is why both
/// `LinearRegression::fit` and the IRLS loop below call this instead of inverting.
///
/// The singularity test is *relative* to the magnitude of the system
/// (`n · ε · max|A_ij|`) rather than an absolute `1e-10`: an absolute threshold
/// rejects perfectly well-conditioned systems whose entries are simply small
/// (e.g. features measured in millivolts) and accepts singular ones whose entries
/// are large.
fn solve_linear_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = matrix.len();

    if n == 0 {
        return Err(Error::InvalidOperation("Matrix is empty".into()));
    }

    for row in matrix {
        if row.len() != n {
            return Err(Error::DimensionMismatch("Matrix must be square".into()));
        }
    }

    if rhs.len() != n {
        return Err(Error::DimensionMismatch(format!(
            "Right-hand side has length {} but the matrix is {}x{}",
            rhs.len(),
            n,
            n
        )));
    }

    // Scale of the system, used for a relative singularity threshold.
    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let tol = (n as f64) * f64::EPSILON * scale.max(f64::MIN_POSITIVE);

    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, row) in matrix.iter().enumerate() {
        let mut augmented_row = Vec::with_capacity(n + 1);
        augmented_row.extend_from_slice(row);
        augmented_row.push(rhs[i]);
        aug.push(augmented_row);
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = aug[col][col].abs();
        for r in col + 1..n {
            let candidate = aug[r][col].abs();
            if candidate > pivot_val {
                pivot_row = r;
                pivot_val = candidate;
            }
        }

        if pivot_val <= tol {
            return Err(Error::Computation(
                "Linear system is singular (or numerically indistinguishable from \
                 singular); the design matrix likely has collinear or constant columns"
                    .into(),
            ));
        }

        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        let pivot = aug[col][col];
        for r in col + 1..n {
            let factor = aug[r][col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for k in col..=n {
                aug[r][k] -= factor * aug[col][k];
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut acc = aug[i][n];
        for j in i + 1..n {
            acc -= aug[i][j] * solution[j];
        }
        solution[i] = acc / aug[i][i];
    }

    Ok(solution)
}

/// Sigmoid function σ(x) = 1 / (1 + e^{-x}), evaluated without overflow.
///
/// The naive `1 / (1 + exp(-x))` overflows `exp` for large negative `x`
/// (`exp(800)` = `inf`, so the result becomes exactly `0.0` and its logarithm
/// `-inf`). Branching on the sign keeps the exponent argument non-positive in
/// both halves, which is exactly representable.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
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
///
/// # Feature normalisation
///
/// With `normalize = true` the design matrix is standardised (zero mean, unit
/// variance) *for the solve only*; the fitted coefficients are then transformed
/// back into the original feature units, so [`coefficients`](Self::coefficients),
/// [`intercept`](Self::intercept) and [`SupervisedModel::predict`] all speak the
/// same units as the raw input data. (Previously the scaling was applied during
/// `fit` and never at predict time, so a normalised model predicted from raw
/// features with standardised coefficients.) Normalisation requires
/// `fit_intercept = true`: the mean shift it introduces can only be absorbed by an
/// intercept term.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients (weights) for each feature, always in the original feature units
    pub coefficients: Option<HashMap<String, f64>>,
    /// Intercept (bias) term
    pub intercept: Option<f64>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Whether to standardise features for the solve (see the type-level docs)
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

        // A constant target has no variance to explain. Reporting R² = 1.0
        // regardless of the residuals credited a model that predicts the wrong
        // constant with a perfect score; mirror `metrics::regression::r2_score`
        // and only call it perfect when the residuals really are zero.
        if ss_tot == 0.0 {
            return Ok(if ss_res == 0.0 { 1.0 } else { 0.0 });
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

        let start_idx = if self.fit_intercept { 1 } else { 0 };

        // Optional feature standardisation. The (mean, std) pair actually applied to
        // each feature column is recorded so the solved coefficients can be mapped
        // back into the original feature units below; a column left untouched
        // (constant, i.e. zero standard deviation) records the identity transform
        // (0, 1) so the back-transform is exact for it too.
        let mut scaling: Vec<(f64, f64)> = Vec::with_capacity(feature_names.len());
        if self.normalize {
            if !self.fit_intercept {
                return Err(Error::InvalidInput(
                    "normalize = true requires fit_intercept = true: centring the features \
                     shifts the response by a constant that only an intercept term can absorb"
                        .into(),
                ));
            }

            for col in x_matrix[start_idx..].iter_mut() {
                let mean = col.iter().sum::<f64>() / n as f64;
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                let std_dev = variance.sqrt();
                if std_dev > 1e-10 {
                    for v in col.iter_mut() {
                        *v = (*v - mean) / std_dev;
                    }
                    scaling.push((mean, std_dev));
                } else {
                    scaling.push((0.0, 1.0));
                }
            }
        }

        // Normal equations: (X'X) β = X'y, solved directly rather than by forming
        // (X'X)^{-1} explicitly.
        let xt_x = matrix_multiply_transpose(&x_matrix, &x_matrix);
        let xt_y = vec_multiply_transpose(&x_matrix, &y_values);
        let beta_coefs = solve_linear_system(&xt_x, &xt_y)?;

        let mut coefficients = HashMap::new();

        if self.normalize {
            // Back-transform: ŷ = b₀ + Σ b_j (x_j - m_j)/s_j
            //                   = (b₀ - Σ (b_j/s_j)·m_j) + Σ (b_j/s_j)·x_j
            let mut intercept = beta_coefs[0];
            for (i, feature_name) in feature_names.iter().enumerate() {
                let (mean, std_dev) = scaling[i];
                let coef = beta_coefs[start_idx + i] / std_dev;
                intercept -= coef * mean;
                coefficients.insert(feature_name.clone(), coef);
            }
            self.intercept = Some(intercept);
        } else {
            if self.fit_intercept {
                self.intercept = Some(beta_coefs[0]);
            } else {
                self.intercept = None;
            }
            for (i, feature_name) in feature_names.iter().enumerate() {
                coefficients.insert(feature_name.clone(), beta_coefs[start_idx + i]);
            }
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

        // Constant target: only a residual-free prediction earns R² = 1.0 (mirrors
        // `metrics::regression::r2_score`). Returning 1.0 unconditionally scored a
        // model that missed a constant target as perfect.
        let r2 = if ss_tot == 0.0 {
            if ss_res == 0.0 {
                1.0
            } else {
                0.0
            }
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

    /// K-fold cross-validation over *contiguous* folds, matching
    /// `sklearn.model_selection.KFold`'s default (`shuffle=False`). Callers whose rows
    /// are ordered by the target should shuffle the DataFrame before calling this.
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

impl TunableModel for LinearRegression {
    /// Apply a hyperparameter combination from a grid search.
    ///
    /// Recognised keys: `fit_intercept`, `normalize`. Values are applied on top of the
    /// model's current configuration; anything else is an error (see
    /// [`TunableModel`]'s contract).
    fn set_params(&mut self, params: &HashMap<String, String>) -> Result<()> {
        let mut unknown = Vec::new();
        for (key, value) in params {
            match key.as_str() {
                "fit_intercept" => self.fit_intercept = parse_param_bool(key, value)?,
                "normalize" => self.normalize = parse_param_bool(key, value)?,
                _ => unknown.push(key.clone()),
            }
        }
        err_on_unknown_params("LinearRegression", unknown)
    }
}

// ---------------------------------------------------------------------------
// LogisticRegression
// ---------------------------------------------------------------------------

/// Logistic regression model for binary classification.
///
/// Uses Iteratively Reweighted Least Squares (IRLS) for fitting with optional
/// L2 (Ridge) regularisation controlled by the `c` parameter.
///
/// # Regularisation convention
///
/// `c` is scikit-learn's inverse regularisation strength: the fitted objective is
///
/// ```text
///   minimise   Σ_i −log p(y_i | x_i)  +  (1 / (2·C)) · ‖w‖²
/// ```
///
/// i.e. the penalty weight is `λ = 1/C`, applied to the *feature* coefficients only —
/// the intercept is never penalised. Concretely, `λ` is added to the diagonal of the
/// IRLS normal equations `(X'WX + λD) β = X'Wz` with `D = diag(0, 1, …, 1)`, which is
/// the exact Newton step for the penalised objective. Set `c = f64::INFINITY` (or call
/// [`without_regularization`](Self::without_regularization)) for an unpenalised fit.
///
/// This replaces an earlier post-hoc uniform shrinkage of the unpenalised solution,
/// which was not L2 regularisation at all and whose effect vanished as `n` grew.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// Coefficients (weights) for each feature
    pub coefficients: Option<HashMap<String, f64>>,
    /// Intercept (bias) term
    pub intercept: Option<f64>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Inverse L2 regularisation strength (`λ = 1/c`); `f64::INFINITY` disables the penalty
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

    /// Disable L2 regularisation entirely (equivalent to `with_regularization(f64::INFINITY)`)
    pub fn without_regularization(mut self) -> Self {
        self.c = f64::INFINITY;
        self
    }

    /// L2 penalty weight `λ = 1/C` used in the IRLS normal equations.
    ///
    /// `c = +∞` means "no penalty" (`λ = 0`). Non-positive or NaN values are rejected
    /// rather than silently producing a negative (anti-)penalty.
    fn regularization_lambda(&self) -> Result<f64> {
        if self.c.is_nan() || self.c <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "LogisticRegression: C must be a positive number (or f64::INFINITY for an \
                 unregularised fit), got {}",
                self.c
            )));
        }
        if self.c.is_infinite() {
            Ok(0.0)
        } else {
            Ok(1.0 / self.c)
        }
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
    ///
    /// Each iteration solves the penalised normal equations
    /// `(X'WX + λD) β_new = X'W z` with working response `z = η + W⁻¹(y − μ)`,
    /// `λ = 1/C` and `D = diag(0, 1, …, 1)` (intercept unpenalised) — the exact
    /// Newton step for the L2-penalised log-likelihood.
    fn irls_fit(&self, x_rows: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>> {
        let n = x_rows.len();
        let p = if n > 0 { x_rows[0].len() } else { 0 };

        let lambda = self.regularization_lambda()?;
        let intercept_offset = if self.fit_intercept { 1 } else { 0 };

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

            // X'WX, with the L2 penalty added to the diagonal of the feature block
            // (the intercept column, index 0 when fitted, stays unpenalised).
            let mut xwt_xw = matrix_multiply_transpose(&xw_cols, &xw_cols);
            for (j, row) in xwt_xw.iter_mut().enumerate().skip(intercept_offset) {
                row[j] += lambda;
            }

            // Xw' zw
            let xwt_zw = vec_multiply_transpose(&xw_cols, &zw);

            // β_new solves (X'WX + λD) β_new = X'W z
            let beta_new = solve_linear_system(&xwt_xw, &xwt_zw).map_err(|_| {
                Error::Computation(
                    "IRLS: X'WX + λI is singular; try increasing regularisation (lower C) \
                     or reducing feature dimensionality"
                        .into(),
                )
            })?;

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

        // Validate that the target really is binary {0, 1}.
        //
        // Clamping to [0, 1] (what this used to do) silently turned a three-class
        // target {0, 1, 2} into {0, 1} and a continuous target into a mixture of
        // saturated ends, then reported classification metrics for a problem the
        // caller never posed. Encode labels explicitly before fitting instead.
        let mut has_zero = false;
        let mut has_one = false;
        for (idx, &value) in y_raw.iter().enumerate() {
            if value == 0.0 {
                has_zero = true;
            } else if value == 1.0 {
                has_one = true;
            } else {
                return Err(Error::InvalidValue(format!(
                    "LogisticRegression requires a binary target encoded as 0.0/1.0; \
                     column '{}' contains {} at row {}",
                    target_column, value, idx
                )));
            }
        }
        if !(has_zero && has_one) {
            return Err(Error::InvalidValue(format!(
                "LogisticRegression requires both classes to be present in the training \
                 target; column '{}' contains only {}",
                target_column,
                if has_one { "1.0" } else { "0.0" }
            )));
        }
        let y: Vec<f64> = y_raw.to_vec();

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

impl TunableModel for LogisticRegression {
    /// Apply a hyperparameter combination from a grid search.
    ///
    /// Recognised keys: `C` (or `c`), `fit_intercept`, `max_iter`, `tol`. Values are
    /// applied on top of the model's current configuration; anything else is an error.
    fn set_params(&mut self, params: &HashMap<String, String>) -> Result<()> {
        let mut unknown = Vec::new();
        for (key, value) in params {
            match key.as_str() {
                // scikit-learn spells the inverse regularisation strength "C"; the
                // lowercase form matches this struct's own field name.
                "C" | "c" => self.c = parse_param_f64(key, value)?,
                "fit_intercept" => self.fit_intercept = parse_param_bool(key, value)?,
                "max_iter" => self.max_iter = parse_param_usize(key, value)?,
                "tol" => self.tol = parse_param_f64(key, value)?,
                _ => unknown.push(key.clone()),
            }
        }
        err_on_unknown_params("LogisticRegression", unknown)
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
    use crate::series::Series;

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

//! Dimensionality reduction algorithms
//!
//! This module provides real implementations of PCA (via Jacobi
//! eigendecomposition) and t-SNE (full O(N²) gradient-descent algorithm).

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::models::UnsupervisedModel;

// ──────────────────────────────────────────────────────────────────────────────
// Private helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Jacobi eigendecomposition of a real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[k]` is the k-th
/// eigenvector (column) corresponding to `eigenvalues[k]`.
///
/// The classic Jacobi iteration is applied until the maximum off-diagonal
/// element drops below 1e-10 or 1 000 sweeps are exhausted.
fn jacobi_eigen_symmetric(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Working copy of the matrix.
    let mut a: Vec<Vec<f64>> = matrix.iter().map(|row| row.to_vec()).collect();

    // V accumulates the accumulated orthogonal transformation (starts as identity).
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0_f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _sweep in 0..1_000 {
        // Find (p, q) with maximum |a[p][q]|, p < q.
        let mut max_val = 0.0_f64;
        let mut p_idx = 0usize;
        let mut q_idx = 1usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p_idx = i;
                    q_idx = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        let p = p_idx;
        let q = q_idx;

        // Compute the Jacobi rotation parameters.
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        let tau = s / (1.0 + c);

        // Save diagonal and off-diagonal before update.
        let a_pp = a[p][p];
        let a_qq = a[q][q];
        let a_pq = a[p][q];

        // Update diagonal and zero the (p,q) / (q,p) entries.
        a[p][p] = a_pp - t * a_pq;
        a[q][q] = a_qq + t * a_pq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Update the remaining rows/columns.
        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let a_rp = a[r][p];
            let a_rq = a[r][q];
            let new_rp = a_rp - s * (a_rq + tau * a_rp);
            let new_rq = a_rq + s * (a_rp - tau * a_rq);
            a[r][p] = new_rp;
            a[p][r] = new_rp;
            a[r][q] = new_rq;
            a[q][r] = new_rq;
        }

        // Accumulate the rotation into the eigenvector matrix V.
        for r in 0..n {
            let v_rp = v[r][p];
            let v_rq = v[r][q];
            v[r][p] = v_rp - s * (v_rq + tau * v_rp);
            v[r][q] = v_rq + s * (v_rp - tau * v_rq);
        }
    }

    // Diagonal of A holds the eigenvalues; columns of V are the eigenvectors.
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n).map(|k| (0..n).map(|r| v[r][k]).collect()).collect();

    (eigenvalues, eigenvectors)
}

/// Extract all f64-typed columns from `data`.
///
/// Returns `(row_major_matrix, column_names)` where the matrix has shape
/// `[n_rows][n_features]` and the columns are in the same order as
/// `data.column_names()`.
fn extract_float_columns(data: &DataFrame) -> (Vec<Vec<f64>>, Vec<String>) {
    let col_names = data.column_names();
    let n_rows = data.nrows();

    let mut feature_cols: Vec<String> = Vec::new();
    // Temporary column-major storage before transposing.
    let mut col_major: Vec<Vec<f64>> = Vec::new();

    for col_name in &col_names {
        if let Ok(col) = data.get_column::<f64>(col_name) {
            let col_vals: Vec<f64> = col.values().to_vec();
            if col_vals.len() == n_rows {
                feature_cols.push(col_name.clone());
                col_major.push(col_vals);
            }
        }
    }

    let n_features = feature_cols.len();
    // Transpose to row-major.
    let mut row_major: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n_rows];
    for (col_idx, col_data) in col_major.iter().enumerate() {
        for (row_idx, &val) in col_data.iter().enumerate() {
            row_major[row_idx][col_idx] = val;
        }
    }

    (row_major, feature_cols)
}

// ──────────────────────────────────────────────────────────────────────────────
// PCA
// ──────────────────────────────────────────────────────────────────────────────

/// Principal Component Analysis (PCA) via Jacobi eigendecomposition.
///
/// Steps performed by `fit`:
/// 1. Extract f64 numeric columns.
/// 2. Compute column means; optionally compute std devs.
/// 3. Centre (and optionally z-score) the data.
/// 4. Compute the sample covariance matrix.
/// 5. Eigen-decompose via Jacobi rotation.
/// 6. Sort by descending eigenvalue, keep top `n_components`.
#[derive(Debug, Clone)]
pub struct PCA {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to standardize (z-score) features before PCA
    pub standardize: bool,
    /// Component vectors (eigenvectors), each of length `n_features`
    pub components: Option<Vec<Vec<f64>>>,
    /// Explained variance ratio for each kept component
    pub explained_variance_ratio: Option<Vec<f64>>,
    /// Per-feature column mean stored during `fit`
    mean_values: Option<Vec<f64>>,
    /// Per-feature column std-dev stored during `fit` (only when `standardize`)
    std_values: Option<Vec<f64>>,
    /// Column names seen during `fit` (preserves ordering for `transform`)
    feature_columns: Option<Vec<String>>,
}

impl PCA {
    /// Create a new PCA instance.
    pub fn new(n_components: usize, standardize: bool) -> Self {
        PCA {
            n_components,
            standardize,
            components: None,
            explained_variance_ratio: None,
            mean_values: None,
            std_values: None,
            feature_columns: None,
        }
    }

    /// Sum of explained variance ratios across all kept components.
    pub fn total_explained_variance(&self) -> Option<f64> {
        self.explained_variance_ratio
            .as_ref()
            .map(|ratios| ratios.iter().sum())
    }

    /// Centre (and optionally scale) one data row using the stored statistics.
    fn center_row(&self, row: &[f64]) -> Vec<f64> {
        let means = self.mean_values.as_ref().expect("PCA not fitted");
        match &self.std_values {
            Some(stds) => row
                .iter()
                .zip(means.iter())
                .zip(stds.iter())
                .map(|((&x, &m), &s)| if s > 1e-12 { (x - m) / s } else { x - m })
                .collect(),
            None => row.iter().zip(means.iter()).map(|(&x, &m)| x - m).collect(),
        }
    }

    /// Compute mean squared reconstruction error (project → inverse-project → residual).
    fn reconstruction_mse(&self, data: &DataFrame) -> Result<f64> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("PCA has not been fitted".into()))?;
        let feature_columns = self
            .feature_columns
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("PCA has not been fitted".into()))?;

        let n_rows = data.nrows();
        let n_features = feature_columns.len();
        let n_comp = components.len();
        let mut mse = 0.0_f64;

        for row_idx in 0..n_rows {
            // Re-extract the same feature columns used during fit.
            let mut raw_row = Vec::with_capacity(n_features);
            for col_name in feature_columns {
                let col = data.get_column::<f64>(col_name)?;
                raw_row.push(*col.get(row_idx).ok_or_else(|| {
                    Error::InvalidValue(format!("Missing value at row {}", row_idx))
                })?);
            }
            let centered = self.center_row(&raw_row);

            // Scores: project onto each component.
            let scores: Vec<f64> = components
                .iter()
                .map(|comp| centered.iter().zip(comp.iter()).map(|(&x, &w)| x * w).sum())
                .collect();

            // Reconstruct from the low-rank approximation.
            let mut x_approx = vec![0.0_f64; n_features];
            for k in 0..n_comp {
                for j in 0..n_features {
                    x_approx[j] += scores[k] * components[k][j];
                }
            }

            for (orig, approx) in centered.iter().zip(x_approx.iter()) {
                let diff = orig - approx;
                mse += diff * diff;
            }
        }

        let total_elements = (n_rows * n_features) as f64;
        Ok(if total_elements > 0.0 {
            mse / total_elements
        } else {
            0.0
        })
    }
}

impl UnsupervisedModel for PCA {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        // 1. Extract numeric columns.
        let (x, feature_columns) = extract_float_columns(data);
        let n_samples = x.len();
        let n_features = feature_columns.len();

        if n_features == 0 {
            return Err(Error::InvalidOperation(
                "DataFrame contains no f64 columns for PCA".into(),
            ));
        }
        if n_samples < 2 {
            return Err(Error::InvalidOperation(
                "PCA requires at least 2 samples".into(),
            ));
        }

        // Clamp n_components to what's possible given the data dimensions.
        let n_components = self.n_components.min(n_features).min(n_samples - 1).max(1);
        self.n_components = n_components;
        self.feature_columns = Some(feature_columns.clone());

        // 2. Column means.
        let mut means = vec![0.0_f64; n_features];
        for row in &x {
            for (j, &val) in row.iter().enumerate() {
                means[j] += val;
            }
        }
        for m in &mut means {
            *m /= n_samples as f64;
        }
        self.mean_values = Some(means.clone());

        // 3. Column standard deviations (if standardize is requested).
        let stds_opt: Option<Vec<f64>> = if self.standardize {
            let mut var_sum = vec![0.0_f64; n_features];
            for row in &x {
                for (j, &val) in row.iter().enumerate() {
                    let diff = val - means[j];
                    var_sum[j] += diff * diff;
                }
            }
            let stds_vec: Vec<f64> = var_sum
                .iter()
                .map(|&ss| {
                    let var = ss / (n_samples as f64 - 1.0);
                    if var > 1e-24 {
                        var.sqrt()
                    } else {
                        1.0
                    }
                })
                .collect();
            self.std_values = Some(stds_vec.clone());
            Some(stds_vec)
        } else {
            self.std_values = None;
            None
        };

        // 4. Centre (and optionally scale) the data.
        let x_centered: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, &val)| {
                        let centered = val - means[j];
                        match &stds_opt {
                            Some(s) => {
                                if s[j] > 1e-12 {
                                    centered / s[j]
                                } else {
                                    centered
                                }
                            }
                            None => centered,
                        }
                    })
                    .collect()
            })
            .collect();

        // 5. Sample covariance matrix C = X'X / (n-1).
        let mut cov = vec![vec![0.0_f64; n_features]; n_features];
        for row in &x_centered {
            for i in 0..n_features {
                for j in 0..n_features {
                    cov[i][j] += row[i] * row[j];
                }
            }
        }
        let denom = (n_samples as f64) - 1.0;
        for row in &mut cov {
            for v in row.iter_mut() {
                *v /= denom;
            }
        }

        // 6. Jacobi eigendecomposition.
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(&cov);

        // 7. Sort by descending eigenvalue.
        let mut pairs: Vec<(f64, Vec<f64>)> = eigenvalues.into_iter().zip(eigenvectors).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Total variance = sum of all eigenvalues (clamp negatives to 0).
        let total_variance: f64 = pairs.iter().map(|(ev, _)| ev.max(0.0)).sum();

        // 8. Keep top n_components.
        let top: Vec<(f64, Vec<f64>)> = pairs.into_iter().take(n_components).collect();

        self.components = Some(top.iter().map(|(_, v)| v.clone()).collect());
        self.explained_variance_ratio = Some(
            top.iter()
                .map(|(ev, _)| {
                    if total_variance > 1e-24 {
                        ev.max(0.0) / total_variance
                    } else {
                        0.0
                    }
                })
                .collect(),
        );

        Ok(())
    }

    fn transform(&self, data: &DataFrame) -> Result<DataFrame> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("PCA has not been fitted".into()))?;
        let feature_columns = self
            .feature_columns
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("PCA has not been fitted".into()))?;

        let n_rows = data.nrows();
        let n_features = feature_columns.len();
        let n_comp = components.len();

        // Allocate per-component score vectors.
        let mut pc_data: Vec<Vec<f64>> = vec![vec![0.0_f64; n_rows]; n_comp];

        for row_idx in 0..n_rows {
            // Reconstruct the raw row in feature order.
            let mut raw_row = Vec::with_capacity(n_features);
            for col_name in feature_columns {
                let col = data.get_column::<f64>(col_name)?;
                raw_row.push(*col.get(row_idx).ok_or_else(|| {
                    Error::InvalidValue(format!("Missing value at row {}", row_idx))
                })?);
            }
            let centered = self.center_row(&raw_row);

            // Project onto each principal component.
            for (k, comp) in components.iter().enumerate() {
                let proj: f64 = centered.iter().zip(comp.iter()).map(|(&x, &w)| x * w).sum();
                pc_data[k][row_idx] = proj;
            }
        }

        // Build result DataFrame with columns "PC_1" … "PC_n".
        let mut result = DataFrame::new();
        for k in 0..n_comp {
            let col_name = format!("PC_{}", k + 1);
            result.add_column(
                col_name.clone(),
                crate::series::Series::new(pc_data[k].clone(), Some(col_name))?,
            )?;
        }

        Ok(result)
    }
}

impl crate::ml::models::ModelEvaluator for PCA {
    fn evaluate(
        &self,
        test_data: &DataFrame,
        _test_target: &str,
    ) -> Result<crate::ml::models::ModelMetrics> {
        let mut metrics = crate::ml::models::ModelMetrics::new();

        let mse = self.reconstruction_mse(test_data)?;
        metrics.add_metric("reconstruction_error", mse);

        if let Some(ratio) = self.total_explained_variance() {
            metrics.add_metric("explained_variance_ratio", ratio);
        }

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<crate::ml::models::ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for PCA".into(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// t-SNE
// ──────────────────────────────────────────────────────────────────────────────

/// Initialisation strategy for the t-SNE embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TSNEInit {
    /// Small Gaussian noise (σ = 1e-4).
    Random,
    /// PCA projections scaled by 1e-4.
    PCA,
}

/// t-Distributed Stochastic Neighbor Embedding (t-SNE).
///
/// This is the original O(N²) algorithm — correct for N ≲ 1 000 samples.
/// For very large N a Barnes-Hut approximation should be preferred.
#[derive(Debug, Clone)]
pub struct TSNE {
    /// Dimensionality of the low-dimensional space (usually 2 or 3)
    pub n_components: usize,
    /// Perplexity — effective number of nearest neighbours
    pub perplexity: f64,
    /// Number of gradient-descent iterations
    pub n_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Initialisation strategy
    pub init: TSNEInit,
    /// Optional RNG seed for reproducibility
    pub random_seed: Option<u64>,
    /// Embedding result — `embedding[i]` is the position of sample i
    pub embedding: Option<Vec<Vec<f64>>>,
}

impl TSNE {
    /// Create a new t-SNE instance with sensible defaults.
    pub fn new() -> Self {
        TSNE {
            n_components: 2,
            perplexity: 30.0,
            n_iter: 1000,
            learning_rate: 200.0,
            init: TSNEInit::PCA,
            random_seed: None,
            embedding: None,
        }
    }

    /// Create a t-SNE instance with custom hyperparameters.
    pub fn with_params(
        n_components: usize,
        perplexity: f64,
        n_iter: usize,
        learning_rate: f64,
        init: TSNEInit,
    ) -> Self {
        TSNE {
            n_components,
            perplexity,
            n_iter,
            learning_rate,
            init,
            random_seed: None,
            embedding: None,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    #[inline]
    fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }

    /// Compute the symmetric joint probability matrix P with per-point bandwidth
    /// tuned via binary search to match `target_perplexity`.
    fn compute_p_matrix(data: &[Vec<f64>], target_perplexity: f64) -> Vec<Vec<f64>> {
        let n = data.len();
        let target_log_perp = target_perplexity.ln();

        // p[i][j] = p_{j|i}
        let mut p = vec![vec![0.0_f64; n]; n];

        for i in 0..n {
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;
            let mut beta = 1.0_f64;

            for _bin_iter in 0..50 {
                let mut sum_pi = 0.0_f64;
                for j in 0..n {
                    if i == j {
                        p[i][j] = 0.0;
                        continue;
                    }
                    let d = Self::sq_dist(&data[i], &data[j]);
                    p[i][j] = (-beta * d).exp();
                    sum_pi += p[i][j];
                }

                let inv_sum = if sum_pi > 1e-200 { 1.0 / sum_pi } else { 0.0 };
                for j in 0..n {
                    if i != j {
                        p[i][j] *= inv_sum;
                    }
                }

                // Shannon entropy of the conditional distribution.
                let mut entropy = 0.0_f64;
                for j in 0..n {
                    if i != j && p[i][j] > 1e-300 {
                        entropy -= p[i][j] * p[i][j].ln();
                    }
                }

                let entropy_diff = entropy - target_log_perp;
                if entropy_diff.abs() < 1e-5 {
                    break;
                }

                if entropy_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_finite() {
                        (beta + beta_max) * 0.5
                    } else {
                        beta * 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_finite() {
                        (beta + beta_min) * 0.5
                    } else {
                        beta * 0.5
                    };
                }
            }
        }

        // Symmetrise: P_{ij} = (p_{j|i} + p_{i|j}) / (2N).
        let inv_2n = 0.5 / (n as f64);
        let mut sym_p = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                sym_p[i][j] = (p[i][j] + p[j][i]) * inv_2n;
            }
        }

        sym_p
    }

    /// Compute the Student-t Q matrix and the normalisation constant Z.
    /// Q_{ij} = w_{ij} / Z where w_{ij} = (1 + ||y_i - y_j||²)⁻¹.
    fn compute_q_matrix(embedding: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
        let n = embedding.len();
        let mut q = vec![vec![0.0_f64; n]; n];
        let mut z = 0.0_f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let w = 1.0 / (1.0 + Self::sq_dist(&embedding[i], &embedding[j]));
                q[i][j] = w;
                q[j][i] = w;
                z += 2.0 * w;
            }
        }

        let inv_z = if z > 1e-300 { 1.0 / z } else { 0.0 };
        for row in &mut q {
            for v in row.iter_mut() {
                *v *= inv_z;
            }
        }

        (q, z)
    }

    /// t-SNE gradient: dC/dy_i = 4 Σ_j (P_{ij} - Q_{ij}) * (y_i - y_j) * w_{ij}
    /// where w_{ij} = (1 + ||y_i - y_j||²)⁻¹ = Q_{ij} * Z.
    fn compute_gradient(p: &[Vec<f64>], q: &[Vec<f64>], embedding: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = embedding.len();
        let dim = embedding[0].len();
        let mut grad = vec![vec![0.0_f64; dim]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // w_{ij} = (1 + ||y_i - y_j||²)⁻¹  (unnormalised Q numerator)
                let inv_dist = 1.0 / (1.0 + Self::sq_dist(&embedding[i], &embedding[j]));
                let factor = 4.0 * (p[i][j] - q[i][j]) * inv_dist;
                for d in 0..dim {
                    grad[i][d] += factor * (embedding[i][d] - embedding[j][d]);
                }
            }
        }

        grad
    }
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsupervisedModel for TSNE {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::Rng;
        use scirs2_core::random::RngExt;
        use scirs2_core::random::SeedableRng;

        // 1. Extract numeric columns.
        let (x, _cols) = extract_float_columns(data);
        let n_samples = x.len();

        if n_samples < 3 {
            return Err(Error::InvalidOperation(
                "t-SNE requires at least 3 samples".into(),
            ));
        }
        if x[0].is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame contains no f64 columns for t-SNE".into(),
            ));
        }

        // 2. High-dimensional affinities with early exaggeration.
        let perplexity = self.perplexity.min((n_samples as f64 - 1.0) / 3.0);
        let mut p = Self::compute_p_matrix(&x, perplexity);
        let exaggeration = 4.0_f64;
        let exaggeration_end = 100usize;

        // Apply early exaggeration.
        for row in &mut p {
            for v in row.iter_mut() {
                *v *= exaggeration;
            }
        }

        // 3. Initialise the embedding.
        let dim = self.n_components;

        let mut rng: StdRng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut seed_bytes = [0u8; 32];
                scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
                StdRng::from_seed(seed_bytes)
            }
        };

        /// Box-Muller standard normal sample using pre-created StdRng.
        fn box_muller(rng: &mut StdRng) -> f64 {
            // u1 must be in (0, 1) to avoid ln(0); clamp to [epsilon, 1).
            let u1: f64 = rng.random_range(1e-300_f64..1.0_f64);
            let u2: f64 = rng.random::<f64>();
            let z: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            z
        }

        let mut embedding: Vec<Vec<f64>> = match self.init {
            TSNEInit::Random => {
                // Gaussian noise with σ = 1e-4.
                (0..n_samples)
                    .map(|_| (0..dim).map(|_| box_muller(&mut rng) * 1e-4).collect())
                    .collect()
            }

            TSNEInit::PCA => {
                let mut pca = PCA::new(dim, false);
                pca.fit(data)?;
                let pca_df = pca.transform(data)?;
                let actual_comp = pca.n_components;

                let mut init_emb = vec![vec![0.0_f64; dim]; n_samples];
                for comp_idx in 0..actual_comp.min(dim) {
                    let col_name = format!("PC_{}", comp_idx + 1);
                    if let Ok(col) = pca_df.get_column::<f64>(&col_name) {
                        for row_idx in 0..n_samples {
                            if let Some(&val) = col.get(row_idx) {
                                init_emb[row_idx][comp_idx] = val * 1e-4;
                            }
                        }
                    }
                }
                // Fill any remaining dimensions with Gaussian noise.
                for row in &mut init_emb {
                    for d in actual_comp..dim {
                        row[d] = box_muller(&mut rng) * 1e-4;
                    }
                }
                init_emb
            }
        };

        // 4. Gradient descent with momentum.
        let mut velocities = vec![vec![0.0_f64; dim]; n_samples];

        for iter in 0..self.n_iter {
            // Remove early exaggeration at iter == exaggeration_end.
            if iter == exaggeration_end {
                for row in &mut p {
                    for v in row.iter_mut() {
                        *v /= exaggeration;
                    }
                }
            }

            let (q_mat, _z) = Self::compute_q_matrix(&embedding);
            let grad = Self::compute_gradient(&p, &q_mat, &embedding);

            // Standard t-SNE momentum schedule.
            let momentum = if iter < 250 { 0.5_f64 } else { 0.8_f64 };

            for i in 0..n_samples {
                for d in 0..dim {
                    velocities[i][d] =
                        momentum * velocities[i][d] - self.learning_rate * grad[i][d];
                    embedding[i][d] += velocities[i][d];
                }
            }

            // Centre embedding to prevent numerical drift.
            let mut mean_emb = vec![0.0_f64; dim];
            for row in &embedding {
                for d in 0..dim {
                    mean_emb[d] += row[d];
                }
            }
            for m in &mut mean_emb {
                *m /= n_samples as f64;
            }
            for row in &mut embedding {
                for d in 0..dim {
                    row[d] -= mean_emb[d];
                }
            }
        }

        self.embedding = Some(embedding);
        Ok(())
    }

    fn transform(&self, _data: &DataFrame) -> Result<DataFrame> {
        // t-SNE does not learn a mapping to unseen points.
        Err(Error::InvalidOperation(
            "t-SNE does not support transform on new data; use fit_transform".into(),
        ))
    }

    fn fit_transform(&mut self, data: &DataFrame) -> Result<DataFrame> {
        self.fit(data)?;

        let n_samples = data.nrows();
        let embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| Error::InvalidValue("t-SNE embedding not computed".into()))?;

        let mut result = DataFrame::new();
        for c in 0..self.n_components {
            let col_name = format!("Component_{}", c + 1);
            let column_data: Vec<f64> = (0..n_samples).map(|i| embedding[i][c]).collect();
            result.add_column(
                col_name.clone(),
                crate::series::Series::new(column_data, Some(col_name))?,
            )?;
        }

        Ok(result)
    }
}

impl crate::ml::models::ModelEvaluator for TSNE {
    fn evaluate(
        &self,
        _test_data: &DataFrame,
        _test_target: &str,
    ) -> Result<crate::ml::models::ModelMetrics> {
        let mut metrics = crate::ml::models::ModelMetrics::new();
        // KL divergence requires re-computing P and Q; report 0 as placeholder.
        metrics.add_metric("kl_divergence", 0.0);
        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<crate::ml::models::ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for t-SNE".into(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;

    /// Build a DataFrame from column-major `(name, values)` pairs.
    fn make_df(cols: &[(&str, Vec<f64>)]) -> DataFrame {
        let mut df = DataFrame::new();
        for (name, data) in cols {
            df.add_column(
                name.to_string(),
                Series::new(data.clone(), Some(name.to_string())).expect("Series::new"),
            )
            .expect("add_column");
        }
        df
    }

    // ── PCA ──────────────────────────────────────────────────────────────────

    /// A near-planar dataset: PCA(2) should capture ≈ 100 % of variance and
    /// the result must have exactly columns "PC_1" and "PC_2".
    #[test]
    fn test_pca_reconstruction() {
        let col1: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let col2: Vec<f64> = (0..10).map(|i| (i * 2) as f64).collect();
        // col3 ≈ col1 + col2 with negligible noise.
        let col3: Vec<f64> = (0..10)
            .map(|i| col1[i] + col2[i] + 0.001 * i as f64)
            .collect();

        let df = make_df(&[("a", col1), ("b", col2), ("c", col3)]);

        let mut pca = PCA::new(2, false);
        pca.fit(&df).expect("fit");

        let components = pca.components.as_ref().expect("components");
        assert_eq!(components.len(), 2);

        let transformed = pca.transform(&df).expect("transform");
        let names = transformed.column_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"PC_1".to_string()));
        assert!(names.contains(&"PC_2".to_string()));

        let total = pca.total_explained_variance().expect("total_ev");
        assert!(
            total > 0.99,
            "expected explained variance > 0.99, got {}",
            total
        );
    }

    /// After projecting centred data, PC_1 must have mean ≈ 0.
    #[test]
    fn test_pca_centering() {
        let col1: Vec<f64> = (0..20).map(|i| 3.0 + i as f64 * 1.5).collect();
        let col2: Vec<f64> = (0..20)
            .map(|i| -7.0 + i as f64 * 0.5 + 0.1 * (i % 3) as f64)
            .collect();
        let df = make_df(&[("x", col1), ("y", col2)]);

        let mut pca = PCA::new(2, false);
        pca.fit(&df).expect("fit");
        let transformed = pca.transform(&df).expect("transform");

        let pc1 = transformed.get_column::<f64>("PC_1").expect("PC_1");
        let mean_pc1: f64 = pc1.values().iter().sum::<f64>() / pc1.values().len() as f64;
        assert!(
            mean_pc1.abs() < 1e-10,
            "Mean of PC_1 should be ≈ 0; got {}",
            mean_pc1
        );
    }

    /// Exact linear dependence (col3 = col1 + col2) ⟹ 2-component PCA
    /// captures > 99 % of variance.
    #[test]
    fn test_pca_low_rank() {
        let col1: Vec<f64> = (0..15).map(|i| (i as f64).sin() * 10.0).collect();
        let col2: Vec<f64> = (0..15).map(|i| (i as f64).cos() * 5.0).collect();
        let col3: Vec<f64> = col1.iter().zip(col2.iter()).map(|(a, b)| a + b).collect();

        let df = make_df(&[("a", col1), ("b", col2), ("c", col3)]);

        let mut pca = PCA::new(2, false);
        pca.fit(&df).expect("fit");

        let ratios = pca.explained_variance_ratio.as_ref().expect("ratios");
        let sum_ratio: f64 = ratios.iter().sum();
        assert!(
            sum_ratio > 0.99,
            "expected 2-component variance ratio > 0.99, got {}",
            sum_ratio
        );
    }

    // ── t-SNE ────────────────────────────────────────────────────────────────

    /// fit_transform on 20 samples × 4 features must return 2 columns × 20 rows
    /// with all finite values.
    #[test]
    fn test_tsne_output_shape() {
        let n = 20usize;
        let cols: Vec<(&'static str, Vec<f64>)> = (0..4usize)
            .map(|c| {
                let data: Vec<f64> = (0..n).map(|r| (r as f64) * (c + 1) as f64).collect();
                let name: &'static str = Box::leak(format!("f{}", c).into_boxed_str());
                (name, data)
            })
            .collect();

        let df = make_df(&cols);

        let mut tsne = TSNE::with_params(2, 5.0, 100, 50.0, TSNEInit::Random);
        tsne.random_seed = Some(42);

        let result = tsne.fit_transform(&df).expect("fit_transform");

        assert_eq!(result.ncols(), 2);
        assert_eq!(result.nrows(), 20);

        for col_name in result.column_names() {
            let col = result.get_column::<f64>(&col_name).expect("get col");
            for (i, &v) in col.values().iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Non-finite value at row {} column {}: {}",
                    i,
                    col_name,
                    v
                );
            }
        }
    }

    /// Two well-separated clusters: after t-SNE, average within-cluster
    /// distance should be smaller than average between-cluster distance.
    #[test]
    fn test_tsne_preserves_clusters() {
        let n_per = 10usize;
        let scale = 0.1_f64;

        let mut fa = Vec::new();
        let mut fb = Vec::new();
        let mut fc = Vec::new();
        let mut fd = Vec::new();

        for i in 0..n_per {
            let noise = i as f64 * scale;
            fa.push(noise);
            fb.push(noise * 0.5);
            fc.push(noise * 0.7);
            fd.push(noise * 0.3);
        }
        for i in 0..n_per {
            let noise = i as f64 * scale;
            fa.push(100.0 + noise);
            fb.push(100.0 + noise * 0.5);
            fc.push(100.0 + noise * 0.7);
            fd.push(100.0 + noise * 0.3);
        }

        let df = make_df(&[("a", fa), ("b", fb), ("c", fc), ("d", fd)]);

        let mut tsne = TSNE::with_params(2, 3.0, 300, 100.0, TSNEInit::Random);
        tsne.random_seed = Some(7);

        let result = tsne.fit_transform(&df).expect("fit_transform");

        let c1_col = result.get_column::<f64>("Component_1").expect("c1");
        let c2_col = result.get_column::<f64>("Component_2").expect("c2");
        let n_total = 2 * n_per;

        let mut within_sum = 0.0_f64;
        let mut within_count = 0usize;
        let mut between_sum = 0.0_f64;
        let mut between_count = 0usize;

        for i in 0..n_total {
            for j in (i + 1)..n_total {
                let dy1 = c1_col.values()[i] - c1_col.values()[j];
                let dy2 = c2_col.values()[i] - c2_col.values()[j];
                let dist = (dy1 * dy1 + dy2 * dy2).sqrt();
                if (i < n_per) == (j < n_per) {
                    within_sum += dist;
                    within_count += 1;
                } else {
                    between_sum += dist;
                    between_count += 1;
                }
            }
        }

        let avg_within = if within_count > 0 {
            within_sum / within_count as f64
        } else {
            0.0
        };
        let avg_between = if between_count > 0 {
            between_sum / between_count as f64
        } else {
            f64::MAX
        };

        assert!(
            avg_within < avg_between,
            "Within-cluster dist ({}) should be < between-cluster dist ({})",
            avg_within,
            avg_between
        );
    }

    // ── Jacobi helper ────────────────────────────────────────────────────────

    /// [[3,1],[1,3]] → eigenvalues {4, 2}.
    #[test]
    fn test_jacobi_eigen_2x2() {
        let matrix = vec![vec![3.0_f64, 1.0], vec![1.0_f64, 3.0]];
        let (evals, evecs) = jacobi_eigen_symmetric(&matrix);

        let mut pairs: Vec<(f64, Vec<f64>)> = evals.into_iter().zip(evecs).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        assert!((pairs[0].0 - 4.0).abs() < 1e-8, "ev0 = {}", pairs[0].0);
        assert!((pairs[1].0 - 2.0).abs() < 1e-8, "ev1 = {}", pairs[1].0);

        // Verify A v = λ v.
        let v = &pairs[0].1;
        let lam = pairs[0].0;
        let av0 = matrix[0][0] * v[0] + matrix[0][1] * v[1];
        let av1 = matrix[1][0] * v[0] + matrix[1][1] * v[1];
        assert!((av0 - lam * v[0]).abs() < 1e-8);
        assert!((av1 - lam * v[1]).abs() < 1e-8);
    }

    /// Diagonal matrix: eigenvalues should equal the diagonal entries.
    #[test]
    fn test_jacobi_diagonal_matrix() {
        let matrix = vec![
            vec![5.0_f64, 0.0, 0.0],
            vec![0.0_f64, 3.0, 0.0],
            vec![0.0_f64, 0.0, 7.0],
        ];
        let (mut evals, _) = jacobi_eigen_symmetric(&matrix);
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap());

        assert!((evals[0] - 7.0).abs() < 1e-8);
        assert!((evals[1] - 5.0).abs() < 1e-8);
        assert!((evals[2] - 3.0).abs() < 1e-8);
    }
}

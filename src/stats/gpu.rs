//! Statistical functions with optional GPU dispatch
//!
//! This module is intended to provide GPU-accelerated statistical functions.
//! However, cudarc 0.19.x does not expose the cuBLAS/cuSOLVER routines required
//! for true GPU kernels, so the numerical work below is currently performed on
//! the **CPU** (via `scirs2_core::ndarray`). The "GPU" code paths exist so a real
//! CUDA implementation can be slotted in later; they are documented honestly as
//! CPU computations and never fabricate results.

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{get_gpu_manager, init_gpu};
use crate::stats::DescriptiveStats;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Compute the correlation matrix.
///
/// Dispatches between two code paths based on data size, but both currently
/// compute on the CPU (cudarc exposes no CUDA kernel here). The result is real.
pub fn correlation_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    let gpu_manager = get_gpu_manager()?;
    let use_gpu = gpu_manager.is_available()
        && data.len() >= gpu_manager.context().config().min_size_threshold;

    if use_gpu {
        // Use GPU implementation
        correlation_matrix_gpu(data)
    } else {
        // Use CPU implementation
        correlation_matrix_cpu(data)
    }
}

/// "GPU" correlation matrix path. Currently computes on the CPU via ndarray (no
/// CUDA kernel is dispatched); kept separate so a real GPU path can be added.
fn correlation_matrix_gpu(data: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = data.dim();

    // Calculate column means
    let mut means = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Center the data (subtract mean from each column)
    let mut centered_data = data.clone();
    for col_idx in 0..n_cols {
        let col_mean = means[col_idx];
        for row_idx in 0..n_rows {
            centered_data[[row_idx, col_idx]] -= col_mean;
        }
    }

    // Create a GPU matrix for the centered data
    let gpu_centered = GpuMatrix::new(centered_data);

    // Compute covariance matrix: X'X / (n-1).
    // NOTE: this uses ndarray's CPU matrix multiplication; no CUDA kernel is
    // dispatched despite the surrounding GpuMatrix wrappers.
    let cov_matrix = gpu_centered.data.t().dot(&gpu_centered.data) / (n_rows - 1) as f64;

    // Convert covariance matrix to correlation matrix
    let mut corr_matrix = Array2::zeros((n_cols, n_cols));
    for i in 0..n_cols {
        for j in 0..n_cols {
            if i == j {
                corr_matrix[[i, j]] = 1.0;
            } else {
                let cov_ij = cov_matrix[[i, j]];
                let var_i = cov_matrix[[i, i]];
                let var_j = cov_matrix[[j, j]];
                corr_matrix[[i, j]] = cov_ij / (var_i.sqrt() * var_j.sqrt());
            }
        }
    }

    Ok(corr_matrix)
}

/// CPU implementation of correlation matrix computation (fallback)
fn correlation_matrix_cpu(data: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = data.dim();

    // Calculate means
    let mut means = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Initialize correlation matrix
    let mut corr_matrix = Array2::zeros((n_cols, n_cols));

    // Compute correlation coefficients
    for i in 0..n_cols {
        // Diagonal elements are always 1.0
        corr_matrix[[i, i]] = 1.0;

        for j in (i + 1)..n_cols {
            // Calculate correlation coefficient
            let mut cov_sum = 0.0;
            let mut var_i_sum = 0.0;
            let mut var_j_sum = 0.0;

            for row_idx in 0..n_rows {
                let x_i = data[[row_idx, i]] - means[i];
                let x_j = data[[row_idx, j]] - means[j];

                cov_sum += x_i * x_j;
                var_i_sum += x_i * x_i;
                var_j_sum += x_j * x_j;
            }

            // Calculate correlation coefficient
            let corr_ij = cov_sum / (var_i_sum.sqrt() * var_j_sum.sqrt());

            // Fill in correlation matrix (symmetric)
            corr_matrix[[i, j]] = corr_ij;
            corr_matrix[[j, i]] = corr_ij;
        }
    }

    Ok(corr_matrix)
}

/// Compute the covariance matrix.
///
/// Dispatches between two code paths based on data size, but both currently
/// compute on the CPU (cudarc exposes no CUDA kernel here). The result is real.
pub fn covariance_matrix(data: &Array2<f64>) -> Result<Array2<f64>> {
    let gpu_manager = get_gpu_manager()?;
    let use_gpu = gpu_manager.is_available()
        && data.len() >= gpu_manager.context().config().min_size_threshold;

    if use_gpu {
        // Use GPU implementation
        covariance_matrix_gpu(data)
    } else {
        // Use CPU implementation
        covariance_matrix_cpu(data)
    }
}

/// "GPU" covariance matrix path. Currently computes on the CPU via ndarray (no
/// CUDA kernel is dispatched); kept separate so a real GPU path can be added.
fn covariance_matrix_gpu(data: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = data.dim();

    // Calculate column means
    let mut means = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Center the data (subtract mean from each column)
    let mut centered_data = data.clone();
    for col_idx in 0..n_cols {
        let col_mean = means[col_idx];
        for row_idx in 0..n_rows {
            centered_data[[row_idx, col_idx]] -= col_mean;
        }
    }

    // Create a GPU matrix for the centered data
    let gpu_centered = GpuMatrix::new(centered_data);

    // Compute covariance matrix: X'X / (n-1).
    // NOTE: this uses ndarray's CPU matrix multiplication; no CUDA kernel is
    // dispatched despite the surrounding GpuMatrix wrappers.
    let cov_matrix = gpu_centered.data.t().dot(&gpu_centered.data) / (n_rows - 1) as f64;

    Ok(cov_matrix)
}

/// CPU implementation of covariance matrix computation (fallback)
fn covariance_matrix_cpu(data: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = data.dim();

    // Calculate means
    let mut means = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Initialize covariance matrix
    let mut cov_matrix = Array2::zeros((n_cols, n_cols));

    // Compute covariance coefficients
    for i in 0..n_cols {
        for j in i..n_cols {
            // Calculate covariance
            let mut cov_sum = 0.0;

            for row_idx in 0..n_rows {
                let x_i = data[[row_idx, i]] - means[i];
                let x_j = data[[row_idx, j]] - means[j];

                cov_sum += x_i * x_j;
            }

            // Calculate covariance
            let cov_ij = cov_sum / (n_rows - 1) as f64;

            // Fill in covariance matrix (symmetric)
            cov_matrix[[i, j]] = cov_ij;
            cov_matrix[[j, i]] = cov_ij;
        }
    }

    Ok(cov_matrix)
}

/// Compute principal component analysis (PCA).
///
/// The covariance matrix and its eigendecomposition are both computed on the
/// CPU; cudarc does not expose cuSOLVER, so no GPU eigensolver is available.
/// This computes **real** principal components (replacing a previous version
/// that returned identity/ones placeholders).
///
/// Returns `(components, explained_variance)` where `components` has shape
/// `n_components x n_features` (one principal component per row, ordered by
/// decreasing explained variance) and `explained_variance` holds the
/// corresponding eigenvalues of the covariance matrix.
pub fn pca(data: &Array2<f64>, n_components: usize) -> Result<(Array2<f64>, Array1<f64>)> {
    // Center the data.
    let (n_rows, n_cols) = data.dim();
    let mut centered_data = data.clone();

    for col_idx in 0..n_cols {
        let col_mean = data.column(col_idx).mean().unwrap_or(0.0);
        for row_idx in 0..n_rows {
            centered_data[[row_idx, col_idx]] -= col_mean;
        }
    }

    // Covariance matrix (n_cols x n_cols, symmetric).
    let cov_matrix = covariance_matrix(&centered_data)?;

    // Real eigendecomposition of the symmetric covariance matrix using the
    // cyclic Jacobi method (shared with `crate::gpu::advanced_ops`). Eigenvalues
    // are returned in descending order with eigenvectors as columns.
    let (eigenvalues, eigenvectors) =
        crate::gpu::advanced_ops::jacobi_symmetric_eigen(&cov_matrix)?;

    let n_components = n_components.min(n_cols);

    // Explained variance = the top eigenvalues.
    let mut explained_variance = Array1::zeros(n_components);
    for c in 0..n_components {
        explained_variance[c] = eigenvalues[c];
    }

    // Principal components: the top eigenvectors, one component per row.
    let mut components = Array2::zeros((n_components, n_cols));
    for c in 0..n_components {
        components.row_mut(c).assign(&eigenvectors.column(c));
    }

    Ok((components, explained_variance))
}

/// Descriptive statistics (CPU computation).
///
/// Named `*_gpu` for historical reasons, but the reductions run on the CPU via
/// ndarray; no CUDA kernel is dispatched. The result is real.
pub fn describe_gpu(data: &[f64]) -> Result<DescriptiveStats> {
    if data.is_empty() {
        return Err(Error::EmptyData("Input data is empty".into()));
    }

    let gpu_manager = get_gpu_manager()?;
    let use_gpu = gpu_manager.is_available()
        && data.len() >= gpu_manager.context().config().min_size_threshold;

    if use_gpu {
        // Convert the slice to an Array1 for GPU processing
        let data_array = Array1::from_vec(data.to_vec());
        let gpu_data = GpuVector::new(data_array);

        // NOTE: these reductions use ndarray's CPU implementation; the
        // GpuVector wrapper does not currently dispatch a CUDA kernel.
        let count = data.len();

        // Sum (CPU).
        let sum = gpu_data.data.sum();

        // Mean (CPU).
        let mean = sum / count as f64;

        // Find min and max
        // These simple operations are often faster on CPU for small arrays
        let mut min = data[0];
        let mut max = data[0];
        for &val in data.iter().skip(1) {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }

        // Sum of squared differences for the variance (CPU reduction).
        let mut var_data = Vec::with_capacity(count);
        for &val in data.iter() {
            var_data.push((val - mean).powi(2));
        }
        let gpu_var_data = GpuVector::new(Array1::from_vec(var_data));
        let sum_squared_diff = gpu_var_data.data.sum();

        // Variance and standard deviation
        let var = sum_squared_diff / (count - 1) as f64; // using n-1 for sample variance
        let std_dev = var.sqrt();

        // For simplicity, calculate quartiles on CPU
        // A full implementation would have GPU-accelerated quantile computation
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = count / 2;
        let median = if count % 2 == 0 {
            (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        } else {
            sorted_data[mid]
        };

        let q1_pos = count / 4;
        let q3_pos = count * 3 / 4;
        let q1 = sorted_data[q1_pos];
        let q3 = sorted_data[q3_pos];

        Ok(DescriptiveStats {
            count,
            mean,
            std: std_dev,
            min,
            q1,
            median,
            q3,
            max,
        })
    } else {
        // Fall back to CPU implementation
        // In a real implementation, this would call the regular CPU function

        let count = data.len();
        let sum: f64 = data.iter().sum();
        let mean = sum / count as f64;

        let mut min = data[0];
        let mut max = data[0];
        for &val in data.iter().skip(1) {
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
        }

        // Variance and standard deviation
        let sum_squared_diff: f64 = data.iter().map(|&val| (val - mean).powi(2)).sum();
        let var = sum_squared_diff / (count - 1) as f64; // using n-1 for sample variance
        let std_dev = var.sqrt();

        // Calculate quartiles
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = count / 2;
        let median = if count % 2 == 0 {
            (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        } else {
            sorted_data[mid]
        };

        let q1_pos = count / 4;
        let q3_pos = count * 3 / 4;
        let q1 = sorted_data[q1_pos];
        let q3 = sorted_data[q3_pos];

        Ok(DescriptiveStats {
            count,
            mean,
            std: std_dev,
            min,
            q1,
            median,
            q3,
            max,
        })
    }
}

/// Checks if GPU is available and initialized
fn ensure_gpu_available() -> Result<()> {
    let status = init_gpu()?;
    if !status.available {
        return Err(Error::Computation(
            "GPU is not available or initialized".into(),
        ));
    }
    Ok(())
}

/// Perform ordinary least-squares linear regression.
///
/// # Description
/// Computes real linear regression coefficients and statistics by solving the
/// normal equations. The linear algebra runs on the CPU; cudarc exposes no
/// cuSOLVER/cuBLAS routines, so no CUDA kernel is dispatched. `p_values` are
/// reported as NaN (not computed) rather than fabricated. Requires a GPU device
/// to be present (see `ensure_gpu_available`).
///
/// # Example
/// ```
/// use pandrs::stats::gpu;
/// use pandrs::gpu::init_gpu;
/// use scirs2_core::ndarray::Array2;
///
/// // Initialize GPU
/// init_gpu().expect("operation should succeed");
///
/// // Create input data matrix (features) and target vector
/// let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("operation should succeed");
/// let y = vec![2.0, 4.0, 5.0, 4.0, 6.0];
///
/// let result = gpu::linear_regression(&x, &y).expect("operation should succeed");
/// println!("Intercept: {}", result.intercept);
/// println!("Coefficient: {}", result.coefficients[0]);
/// println!("R-squared: {}", result.r_squared);
/// ```
pub fn linear_regression(
    x: &Array2<f64>,
    y: &[f64],
) -> Result<crate::stats::LinearRegressionResult> {
    // Ensure GPU is available
    ensure_gpu_available()?;

    let (n_rows, n_cols) = x.dim();

    if y.len() != n_rows {
        return Err(Error::DimensionMismatch(format!(
            "Target variable has length {}, expected {}",
            y.len(),
            n_rows
        )));
    }

    // Create design matrix X (with intercept column)
    let mut design_matrix = Array2::ones((n_rows, n_cols + 1));
    for i in 0..n_rows {
        for j in 0..n_cols {
            design_matrix[[i, j + 1]] = x[[i, j]];
        }
    }

    // Wrap the data in GPU matrix types. Note: the matrix products below use
    // ndarray's CPU `dot` (no CUDA kernel is dispatched); the wrappers exist so
    // a real GPU path can be added later.
    let gpu_x = GpuMatrix::new(design_matrix);
    let gpu_y = GpuVector::new(Array1::from_vec(y.to_vec()));

    // Calculate Xᵀ·X (on the CPU).
    let xtx = gpu_x.data.t().dot(&gpu_x.data);

    // Calculate Xᵀ·y (on the CPU).
    let xty = gpu_x.data.t().dot(&gpu_y.data);

    // Solve the normal equations (XᵀX)·beta = Xᵀy for the regression
    // coefficients. The linear algebra runs on the CPU (cudarc exposes no
    // cuSOLVER); this is a real least-squares solution for both the univariate
    // and multivariate cases, replacing the previous fabricated placeholders.
    let xtx_inv = invert_matrix(&xtx)?;
    let beta = xtx_inv.dot(&xty);
    let coefficients: Vec<f64> = beta.to_vec();

    // Extract intercept and feature coefficients
    let intercept = coefficients[0];
    let feature_coeffs = coefficients[1..].to_vec();

    // Calculate fitted values and residuals
    let mut fitted_values = vec![0.0; n_rows];
    let mut residuals = vec![0.0; n_rows];

    for i in 0..n_rows {
        fitted_values[i] = intercept;
        for j in 0..n_cols {
            fitted_values[i] += feature_coeffs[j] * x[[i, j]];
        }
        residuals[i] = y[i] - fitted_values[i];
    }

    // Calculate R-squared
    let y_mean = y.iter().sum::<f64>() / n_rows as f64;
    let total_sum_squares: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let residual_sum_squares: f64 = residuals.iter().map(|&r| r.powi(2)).sum();
    let r_squared = 1.0 - (residual_sum_squares / total_sum_squares);

    // Calculate adjusted R-squared
    let adj_r_squared =
        1.0 - ((1.0 - r_squared) * (n_rows as f64 - 1.0) / (n_rows as f64 - n_cols as f64 - 1.0));

    // p-values require a Student's t-distribution CDF, which is not available in
    // this build (scirs2-stats is behind a separate feature). Rather than
    // fabricating a constant significance level, report NaN to make explicit
    // that the p-values are not computed here.
    let p_values = vec![f64::NAN; n_cols + 1];

    Ok(crate::stats::LinearRegressionResult {
        intercept,
        coefficients: feature_coeffs,
        r_squared,
        adj_r_squared,
        p_values,
        fitted_values,
        residuals,
    })
}

/// Calculate feature importance from correlation with the target.
///
/// # Description
/// Estimates feature importance as the absolute Pearson correlation between each
/// feature and the target. This is a real computation performed on the CPU; no
/// CUDA kernel is dispatched despite the module name.
///
/// # Example
/// ```
/// use pandrs::stats::gpu;
/// use pandrs::gpu::init_gpu;
/// use scirs2_core::ndarray::Array2;
///
/// // Initialize GPU
/// init_gpu().expect("operation should succeed");
///
/// // Create input data matrix (features) and target vector
/// let x = Array2::from_shape_vec((5, 3), vec![
///     1.0, 2.0, 3.0,
///     2.0, 3.0, 1.0,
///     3.0, 1.0, 0.0,
///     4.0, 2.0, 1.0,
///     5.0, 0.0, 2.0
/// ]).expect("operation should succeed");
/// let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
///
/// let importance = gpu::feature_importance(&x, &y).expect("operation should succeed");
/// println!("Feature importance: {:?}", importance);
/// ```
pub fn feature_importance(x: &Array2<f64>, y: &[f64]) -> Result<HashMap<usize, f64>> {
    // Ensure GPU is available
    ensure_gpu_available()?;

    let (n_rows, n_cols) = x.dim();

    if y.len() != n_rows {
        return Err(Error::DimensionMismatch(format!(
            "Target variable has length {}, expected {}",
            y.len(),
            n_rows
        )));
    }

    // Calculate importance based on correlation with target
    let mut importance = HashMap::new();
    let y_array = Array1::from_vec(y.to_vec());

    for j in 0..n_cols {
        let feature_col = x.column(j).to_owned();

        // Calculate correlation between feature and target
        let mut cov_sum = 0.0;
        let mut var_x_sum = 0.0;
        let mut var_y_sum = 0.0;

        let feature_mean = feature_col.mean().unwrap_or(0.0);
        let y_mean = y_array.mean().unwrap_or(0.0);

        for i in 0..n_rows {
            let x_i = feature_col[i] - feature_mean;
            let y_i = y_array[i] - y_mean;

            cov_sum += x_i * y_i;
            var_x_sum += x_i * x_i;
            var_y_sum += y_i * y_i;
        }

        // Calculate correlation coefficient
        let corr = cov_sum / (var_x_sum.sqrt() * var_y_sum.sqrt());

        // Use absolute correlation as importance
        importance.insert(j, corr.abs());
    }

    // Normalize importance scores
    let max_importance = importance.values().cloned().fold(0.0, f64::max);

    if max_importance > 0.0 {
        for (_, value) in importance.iter_mut() {
            *value /= max_importance;
        }
    }

    Ok(importance)
}

/// Perform k-means clustering.
///
/// # Description
/// Implements Lloyd's k-means clustering algorithm. The computation runs on the
/// CPU (cudarc exposes no kernel for this); despite the module name no GPU work
/// is dispatched. The result is a real clustering.
///
/// # Example
/// ```
/// use pandrs::stats::gpu;
/// use pandrs::gpu::init_gpu;
/// use scirs2_core::ndarray::Array2;
///
/// // Initialize GPU
/// init_gpu().expect("operation should succeed");
///
/// // Create input data matrix
/// let data = Array2::from_shape_vec((8, 2), vec![
///     1.0, 2.0,
///     1.5, 1.8,
///     1.2, 2.2,
///     8.0, 7.0,
///     8.5, 8.0,
///     9.0, 7.5,
///     2.0, 1.5,
///     7.5, 8.5
/// ]).expect("operation should succeed");
///
/// let k = 2;
/// let max_iter = 100;
/// let (centroids, labels, inertia) = gpu::kmeans(&data, k, max_iter).expect("operation should succeed");
/// println!("Cluster labels: {:?}", labels);
/// ```
pub fn kmeans(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
) -> Result<(Array2<f64>, Vec<usize>, f64)> {
    // Ensure GPU is available
    ensure_gpu_available()?;

    let (n_rows, n_cols) = data.dim();

    if k == 0 || k > n_rows {
        return Err(Error::InvalidInput(format!(
            "Number of clusters must be between 1 and number of samples ({})",
            n_rows
        )));
    }

    // For simplicity, initialize centroids randomly
    let mut centroids = Array2::zeros((k, n_cols));

    // Select k random rows from data as initial centroids
    use scirs2_core::rand_prelude::IndexedRandom;
    let mut rng = scirs2_core::random::rng();
    let all_indices: Vec<usize> = (0..n_rows).collect();

    // Sample without replacement
    let indices: Vec<usize> = all_indices
        .as_slice()
        .sample(&mut rng, k)
        .cloned()
        .collect();

    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_cols {
            centroids[[i, j]] = data[[idx, j]];
        }
    }

    // Main k-means loop (real CPU implementation; no GPU kernel is dispatched).
    let mut labels = vec![0; n_rows];
    let mut inertia = 0.0;

    for _ in 0..max_iter {
        // Assign points to nearest centroid
        inertia = 0.0;

        for i in 0..n_rows {
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;

            for c in 0..k {
                let mut dist = 0.0;
                for j in 0..n_cols {
                    let diff = data[[i, j]] - centroids[[c, j]];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = c;
                }
            }

            labels[i] = min_cluster;
            inertia += min_dist;
        }

        // Update centroids
        let mut new_centroids = Array2::zeros((k, n_cols));
        let mut counts = vec![0; k];

        for i in 0..n_rows {
            let cluster = labels[i];
            counts[cluster] += 1;

            for j in 0..n_cols {
                new_centroids[[cluster, j]] += data[[i, j]];
            }
        }

        // Calculate new centroids as mean of points in each cluster
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..n_cols {
                    new_centroids[[c, j]] /= counts[c] as f64;
                }
            } else {
                // Handle empty cluster by assigning a random point
                let random_idx = (scirs2_core::random::random::<f64>() * n_rows as f64) as usize;
                for j in 0..n_cols {
                    new_centroids[[c, j]] = data[[random_idx, j]];
                }
            }
        }

        // Check for convergence (simplified)
        let mut has_changed = false;
        for c in 0..k {
            for j in 0..n_cols {
                if (centroids[[c, j]] - new_centroids[[c, j]]).abs() > 1e-6 {
                    has_changed = true;
                    break;
                }
            }
            if has_changed {
                break;
            }
        }

        if !has_changed {
            break;
        }

        centroids = new_centroids;
    }

    Ok((centroids, labels, inertia))
}

/// Invert a square matrix on the CPU using Gauss-Jordan elimination with
/// partial pivoting. Returns an error if the matrix is singular.
fn invert_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = matrix.dim();
    if n != m {
        return Err(Error::DimensionMismatch(format!(
            "Matrix must be square for inversion, got {:?}",
            (n, m)
        )));
    }

    // Build the augmented matrix [A | I].
    let mut augmented = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, i + n]] = 1.0;
    }

    for i in 0..n {
        // Partial pivoting: find the row with the largest pivot magnitude.
        let mut max_val = augmented[[i, i]].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > max_val {
                max_val = augmented[[k, i]].abs();
                max_row = k;
            }
        }

        if max_val < 1e-12 {
            return Err(Error::Computation(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        if max_row != i {
            for j in 0..(2 * n) {
                augmented.swap([i, j], [max_row, j]);
            }
        }

        // Scale the pivot row.
        let pivot = augmented[[i, i]];
        for j in 0..(2 * n) {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate the pivot column from every other row.
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..(2 * n) {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract the right half, which now holds A⁻¹.
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(cuda_available)]
    fn test_describe_gpu() {
        // Test GPU-accelerated descriptive statistics
        // This is just a placeholder for actual tests
    }
}

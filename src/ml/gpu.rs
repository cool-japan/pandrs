//! Machine learning algorithms with optional GPU dispatch
//!
//! This module is intended to provide GPU-accelerated machine learning
//! algorithms. However, cudarc 0.19.x does not expose the cuBLAS/cuSOLVER
//! routines required for real GPU kernels, so the `*_gpu` code paths below
//! currently perform the **same CPU computation** as their `*_cpu` counterparts.
//! They are documented honestly and never fabricate results; the separate
//! entry points remain so a real CUDA implementation can be added later.

use crate::error::{Error, Result};
use crate::gpu::get_gpu_manager;
use crate::ml::metrics::regression::r2_score;
use crate::stats::LinearRegressionResult;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::RngExt;

/// Ordinary least-squares linear regression (with optional GPU dispatch).
///
/// Both dispatch targets currently compute on the CPU; cudarc exposes no GPU
/// least-squares routine. The result is a real least-squares solution.
pub fn linear_regression(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
) -> Result<LinearRegressionResult> {
    // Check if GPU is available
    let gpu_manager = get_gpu_manager()?;
    let use_gpu = gpu_manager.is_available()
        && x_data.len() >= gpu_manager.context().config().min_size_threshold;

    if use_gpu {
        linear_regression_gpu(x_data, y_data)
    } else {
        linear_regression_cpu(x_data, y_data)
    }
}

/// "GPU" linear regression path.
///
/// cudarc does not provide the cuBLAS/cuSOLVER routines needed for a real GPU
/// least-squares solve, so this performs the **same CPU computation** as
/// [`linear_regression_cpu`]. It is kept as a separate entry point for when a
/// real GPU implementation is added.
fn linear_regression_gpu(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
) -> Result<LinearRegressionResult> {
    linear_regression_cpu(x_data, y_data)
}

/// CPU implementation of linear regression (fallback)
fn linear_regression_cpu(
    x_data: &Array2<f64>,
    y_data: &Array1<f64>,
) -> Result<LinearRegressionResult> {
    let (n_samples, n_features) = x_data.dim();

    if n_samples != y_data.len() {
        return Err(Error::DimensionMismatch(format!(
            "X samples ({}) must match y length ({})",
            n_samples,
            y_data.len()
        )));
    }

    // Add intercept column (all ones) to X
    let mut x_with_intercept = Array2::ones((n_samples, n_features + 1));
    for i in 0..n_samples {
        for j in 0..n_features {
            x_with_intercept[[i, j + 1]] = x_data[[i, j]];
        }
    }

    // Compute: beta = (X^T X)^(-1) X^T y
    let x_t_x = x_with_intercept.t().dot(&x_with_intercept);
    let x_t_x_inv = invert_matrix(&x_t_x)?;
    let x_t_y = x_with_intercept.t().dot(y_data);
    let coefficients = x_t_x_inv.dot(&x_t_y);

    // Extract intercept and coefficients
    let intercept = coefficients[0];
    let feature_coefficients = coefficients.slice(s![1..]).to_vec();

    // Compute fitted values
    let fitted_values = x_with_intercept.dot(&coefficients).to_vec();

    // Compute residuals
    let residuals: Vec<f64> = y_data
        .iter()
        .zip(fitted_values.iter())
        .map(|(&y, &y_hat)| y - y_hat)
        .collect();

    // Compute R^2
    let r_squared = r2_score(&y_data.to_vec(), &fitted_values)?;

    // Compute adjusted R^2. Cast to `f64` before subtracting: `n_samples -
    // n_features - 1` as `usize` arithmetic underflows (panics in debug,
    // wraps to a huge value in release) whenever `n_features + 1 >=
    // n_samples` — more features than residual degrees of freedom, e.g. an
    // exactly- or under-determined fit — and the statistic is mathematically
    // undefined there (division by <= 0 degrees of freedom) rather than
    // merely large, so report `NaN` instead of a nonsensical or panicking
    // computation.
    let residual_dof = n_samples as f64 - n_features as f64 - 1.0;
    let adj_r_squared = if residual_dof > 0.0 {
        1.0 - (1.0 - r_squared) * (n_samples as f64 - 1.0) / residual_dof
    } else {
        f64::NAN
    };

    // p-values require a Student's t-distribution CDF, which is not available in
    // this build. Report NaN ("not computed") instead of a fabricated constant.
    let p_values = vec![f64::NAN; n_features + 1];

    Ok(LinearRegressionResult {
        intercept,
        coefficients: feature_coefficients,
        r_squared,
        adj_r_squared,
        p_values: p_values[1..].to_vec(), // Skip intercept p-value
        fitted_values,
        residuals,
    })
}

/// k-means clustering (with optional GPU dispatch)
pub fn kmeans(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Array2<f64>, Array1<usize>, f64)> {
    // Check if GPU is available
    let gpu_manager = get_gpu_manager()?;
    let use_gpu = gpu_manager.is_available()
        && data.len() >= gpu_manager.context().config().min_size_threshold;

    if use_gpu {
        kmeans_gpu(data, k, max_iter, tol)
    } else {
        kmeans_cpu(data, k, max_iter, tol)
    }
}

/// "GPU" k-means path.
///
/// cudarc does not provide a kernel for k-means, so this performs the **same
/// CPU computation** as [`kmeans_cpu`]. It is kept as a separate entry point for
/// when a real GPU implementation is added.
fn kmeans_gpu(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Array2<f64>, Array1<usize>, f64)> {
    kmeans_cpu(data, k, max_iter, tol)
}

/// CPU implementation of k-means clustering
fn kmeans_cpu(
    data: &Array2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Result<(Array2<f64>, Array1<usize>, f64)> {
    let (n_samples, n_features) = data.dim();

    if n_samples < k {
        return Err(Error::InsufficientData(format!(
            "Number of samples ({}) must be greater than number of clusters ({})",
            n_samples, k
        )));
    }

    // Initialize centroids randomly
    let mut centroids = Array2::zeros((k, n_features));
    for i in 0..k {
        let sample_idx = i * (n_samples / k); // Simple initialization for example
        for j in 0..n_features {
            centroids[[i, j]] = data[[sample_idx, j]];
        }
    }

    let mut labels = Array1::zeros(n_samples);
    let mut inertia = 0.0;
    let mut old_inertia = std::f64::MAX;

    // Iterate until convergence or max iterations
    for _ in 0..max_iter {
        // Assign points to clusters
        let mut new_labels = Array1::zeros(n_samples);
        let mut new_inertia = 0.0;

        // For each point, find the nearest centroid
        for i in 0..n_samples {
            let point = data.row(i).to_owned();
            let mut min_dist = std::f64::MAX;
            let mut min_idx = 0;

            for c in 0..k {
                let centroid = centroids.row(c).to_owned();
                let dist_squared: f64 = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&p, &c)| (p - c).powi(2))
                    .sum();

                if dist_squared < min_dist {
                    min_dist = dist_squared;
                    min_idx = c;
                }
            }

            new_labels[i] = min_idx;
            new_inertia += min_dist;
        }

        // Update centroids
        let mut new_centroids = Array2::zeros((k, n_features));
        let mut counts = vec![0; k];

        for i in 0..n_samples {
            let cluster = new_labels[i];
            counts[cluster] += 1;

            for j in 0..n_features {
                new_centroids[[cluster, j]] += data[[i, j]];
            }
        }

        // Normalize by cluster sizes
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..n_features {
                    new_centroids[[c, j]] /= counts[c] as f64;
                }
            } else {
                // If a cluster is empty, reinitialize its centroid to a
                // uniformly random row. `random::<f64>() as usize` truncates
                // any value in `[0, 1)` straight to `0`, so the previous
                // `% n_samples` always picked row 0 — every empty cluster
                // was reseeded to the exact same point instead of a random
                // one, regardless of how many clusters emptied out.
                let random_idx = scirs2_core::random::rng().random_range(0..n_samples);
                for j in 0..n_features {
                    new_centroids[[c, j]] = data[[random_idx, j]];
                }
            }
        }

        // Check for convergence
        if (old_inertia - new_inertia).abs() < tol * old_inertia {
            inertia = new_inertia;
            labels = new_labels;
            centroids = new_centroids;
            break;
        }

        inertia = new_inertia;
        old_inertia = new_inertia;
        labels = new_labels;
        centroids = new_centroids;
    }

    Ok((centroids, labels, inertia))
}

/// Principal component analysis (PCA).
///
/// Computes **real** principal components on the CPU by delegating the
/// covariance eigendecomposition to [`crate::stats::gpu::pca`] (cudarc exposes
/// no GPU eigensolver, so there is no separate GPU path). The previous version
/// returned identity/ones/zeros placeholders for the non-GPU branch.
///
/// Returns `(components, explained_variance, transformed)` where `components`
/// has shape `n_components x n_features` and `transformed` is the input
/// projected onto those components (`n_samples x n_components`).
pub fn pca(
    data: &Array2<f64>,
    n_components: usize,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    // Real principal components and explained variance. `stats::gpu::pca`
    // centers its own internal copy of `data` before eigendecomposing the
    // covariance matrix, so `components` are directions relative to
    // `data`'s column means.
    let (components, explained_variance) = crate::stats::gpu::pca(data, n_components)?;

    // Project the *centered* data onto those directions. PCA scores are
    // defined relative to the training mean (`score = (x - mean) ·
    // components^T`); projecting the raw, uncentered `data` here instead
    // (as the previous version did) added a fixed offset — the projection
    // of the mean itself, `mean · components^T` — to every single sample's
    // scores, which is generally nonzero whenever the data isn't already
    // centered at the origin.
    let (n_rows, n_cols) = data.dim();
    let mut centered = data.clone();
    for col_idx in 0..n_cols {
        let col_mean = data.column(col_idx).mean().unwrap_or(0.0);
        for row_idx in 0..n_rows {
            centered[[row_idx, col_idx]] -= col_mean;
        }
    }

    // `components` is (n_components x n_features), so `components.t()` is
    // (n_features x n_components).
    let transformed = centered.dot(&components.t());

    Ok((components, explained_variance, transformed))
}

// Helper functions

/// Invert a matrix using simple Gaussian elimination
/// Note: In a real implementation, this would use a more robust algorithm
fn invert_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let (n, m) = matrix.dim();

    if n != m {
        return Err(Error::DimensionMismatch(format!(
            "Matrix must be square for inversion, got {:?}",
            (n, m)
        )));
    }

    // Create augmented matrix [A|I]
    let mut augmented = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, i + n]] = 1.0;
    }

    // Gaussian elimination
    for i in 0..n {
        // Find pivot
        let mut max_val = augmented[[i, i]].abs();
        let mut max_row = i;

        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > max_val {
                max_val = augmented[[k, i]].abs();
                max_row = k;
            }
        }

        // Check if matrix is singular
        if max_val < 1e-10 {
            return Err(Error::Computation(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = temp;
            }
        }

        // Scale row i
        let pivot = augmented[[i, i]];
        for j in 0..(2 * n) {
            augmented[[i, j]] /= pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = augmented[[k, i]];
                for j in 0..(2 * n) {
                    augmented[[k, j]] -= factor * augmented[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = augmented[[i, j + n]];
        }
    }

    Ok(inverse)
}

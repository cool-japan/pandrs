//! GPU acceleration integration for OptimizedDataFrame
//!
//! This module provides GPU acceleration capabilities for the OptimizedDataFrame
//! implementation, enabling high-performance computation for large datasets.

use scirs2_core::ndarray::Array2;

use crate::column::Column;
use crate::error::{Error, Result};
use crate::gpu::get_gpu_manager;
use crate::gpu::operations::{GpuAccelerated, GpuMatrix};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl GpuAccelerated for OptimizedDataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // There is no real GPU kernel behind this entry point (see
        // `crate::gpu`'s module-level honesty notes): GPU dispatch happens
        // per-operation, in `matrix_multiply`/`corr_matrix` below, not as a
        // one-shot transform of the whole frame. Returning `Ok(self.clone())`
        // presented a full deep copy of the data as the result of
        // "acceleration" when nothing was accelerated (or even attempted);
        // report that honestly instead.
        Err(Error::NotImplemented(
            "GPU acceleration for OptimizedDataFrame not implemented (no real CUDA kernel); use \
             matrix_multiply/corr_matrix for real (CPU-fallback) per-operation dispatch"
                .into(),
        ))
    }

    fn is_gpu_acceleratable(&self) -> bool {
        self.row_count() >= 10_000 // Only accelerate large datasets
    }
}

impl OptimizedDataFrame {
    /// Perform a matrix multiplication operation with GPU acceleration if available
    pub fn matrix_multiply(&self, columns1: &[&str], columns2: &[&str]) -> Result<Array2<f64>> {
        let gpu_manager = get_gpu_manager()?;
        let use_gpu = gpu_manager.is_available()
            && self.row_count() >= gpu_manager.context().config().min_size_threshold;

        // Extract columns into matrices
        let matrix1 = self.to_matrix(columns1)?;
        let matrix2 = self.to_matrix(columns2)?;

        if use_gpu {
            // Use GPU acceleration
            let gpu_matrix1 = GpuMatrix::new(matrix1.clone());
            let gpu_matrix2 = GpuMatrix::new(matrix2.clone());

            match gpu_matrix1.dot(&gpu_matrix2) {
                Ok(result) => Ok(result.data),
                Err(e) => {
                    // If GPU fails and fallback is enabled, try CPU
                    if gpu_manager.context().config().fallback_to_cpu {
                        let result = matrix1.dot(&matrix2);
                        Ok(result)
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            // Use CPU implementation
            let result = matrix1.dot(&matrix2);
            Ok(result)
        }
    }

    /// Convert selected columns to a matrix.
    ///
    /// A null/missing cell becomes `f64::NAN`, not `0.0`: silently treating
    /// "missing" as "zero" would fabricate a data point that was never
    /// observed and bias every downstream sum/mean/dot-product that touches
    /// it. `NaN` instead poisons exactly the computations that depend on the
    /// missing cell (per IEEE 754 propagation), which is the honest
    /// behavior — the caller can see something is missing rather than
    /// silently getting a slightly-wrong number back. `col.get` errors
    /// (distinct from a null value) are propagated via `?` rather than also
    /// being folded into a default.
    fn to_matrix(&self, columns: &[&str]) -> Result<Array2<f64>> {
        let n_rows = self.row_count();
        let n_cols = columns.len();

        let mut matrix = Array2::zeros((n_rows, n_cols));

        for (col_idx, col_name) in columns.iter().enumerate() {
            let col_view = self.column(*col_name)?;
            match &col_view.column {
                Column::Float64(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] = col.get(row_idx)?.unwrap_or(f64::NAN);
                    }
                }
                Column::Int64(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] =
                            col.get(row_idx)?.map(|v| v as f64).unwrap_or(f64::NAN);
                    }
                }
                Column::Boolean(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] = match col.get(row_idx)? {
                            Some(true) => 1.0,
                            Some(false) => 0.0,
                            None => f64::NAN,
                        };
                    }
                }
                Column::String(_) => {
                    return Err(Error::Type(format!(
                        "Cannot convert string column '{}' to numeric matrix",
                        col_name
                    )));
                }
            }
        }

        Ok(matrix)
    }

    /// Create a correlation matrix with GPU acceleration if available
    pub fn corr_matrix(&self, columns: &[&str]) -> Result<Array2<f64>> {
        let gpu_manager = get_gpu_manager()?;
        let use_gpu = gpu_manager.is_available()
            && self.row_count() >= gpu_manager.context().config().min_size_threshold;

        // Extract columns into a matrix
        let data_matrix = self.to_matrix(columns)?;
        let n_cols = columns.len();

        if use_gpu {
            // Use GPU acceleration
            // Center the columns (subtract mean)
            let mut centered_data = data_matrix.clone();
            for col_idx in 0..n_cols {
                let col_mean = data_matrix.column(col_idx).mean().unwrap_or(0.0);
                for row_idx in 0..self.row_count() {
                    centered_data[[row_idx, col_idx]] -= col_mean;
                }
            }

            let gpu_centered = GpuMatrix::new(centered_data);

            // Compute covariance matrix: X'X / (n-1)
            let cov_matrix =
                gpu_centered.data.t().dot(&gpu_centered.data) / (self.row_count() - 1) as f64;

            // Convert covariance to correlation
            let mut corr_matrix = Array2::zeros((n_cols, n_cols));
            for i in 0..n_cols {
                for j in 0..n_cols {
                    if i == j {
                        corr_matrix[[i, j]] = 1.0;
                    } else {
                        let cov_ij = cov_matrix[[i, j]];
                        let var_i = cov_matrix[[i, i]];
                        let var_j = cov_matrix[[j, j]];
                        let denominator = var_i.sqrt() * var_j.sqrt();
                        // A constant column has zero variance, making the
                        // correlation with any other column mathematically
                        // undefined (0/0). Report it as 0 (no linear
                        // relationship can be observed), matching the
                        // zero-variance convention used everywhere else in
                        // the crate (`stats::descriptive::pearson_correlation`,
                        // `gpu::advanced_ops::correlation_with_pvalues`)
                        // rather than leaving it as an unguarded NaN.
                        corr_matrix[[i, j]] = if denominator > 1e-10 {
                            cov_ij / denominator
                        } else {
                            0.0
                        };
                    }
                }
            }

            Ok(corr_matrix)
        } else {
            // Use CPU implementation
            compute_corr_matrix_cpu(&data_matrix)
        }
    }
}

/// Compute correlation matrix using CPU implementation
fn compute_corr_matrix_cpu(data_matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let n_rows = data_matrix.shape()[0];
    let n_cols = data_matrix.shape()[1];

    // Calculate means
    let mut means = Vec::with_capacity(n_cols);
    for col_idx in 0..n_cols {
        means.push(data_matrix.column(col_idx).mean().unwrap_or(0.0));
    }

    // Initialize correlation matrix
    let mut corr_matrix = Array2::zeros((n_cols, n_cols));

    // Compute correlation coefficients
    for i in 0..n_cols {
        // Diagonal elements are always 1
        corr_matrix[[i, i]] = 1.0;

        for j in (i + 1)..n_cols {
            // Calculate correlation coefficient
            let mut cov_sum = 0.0;
            let mut var_i_sum = 0.0;
            let mut var_j_sum = 0.0;

            for row_idx in 0..n_rows {
                let x_i = data_matrix[[row_idx, i]] - means[i];
                let x_j = data_matrix[[row_idx, j]] - means[j];

                cov_sum += x_i * x_j;
                var_i_sum += x_i * x_i;
                var_j_sum += x_j * x_j;
            }

            // Calculate correlation coefficient. A constant column (zero
            // variance) makes this mathematically undefined (0/0); report 0
            // rather than an unguarded NaN, matching the zero-variance
            // convention used everywhere else in the crate.
            let denominator = var_i_sum.sqrt() * var_j_sum.sqrt();
            let corr_ij = if denominator > 1e-10 {
                cov_sum / denominator
            } else {
                0.0
            };

            // Store in correlation matrix (symmetric)
            corr_matrix[[i, j]] = corr_ij;
            corr_matrix[[j, i]] = corr_ij;
        }
    }

    Ok(corr_matrix)
}

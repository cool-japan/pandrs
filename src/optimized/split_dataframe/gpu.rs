//! GPU acceleration integration for OptimizedDataFrame
//!
//! This module provides GPU acceleration capabilities for the OptimizedDataFrame
//! implementation, enabling high-performance computation for large datasets.

use ndarray::{Array1, Array2};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuAccelerated, GpuMatrix, GpuVector};
use crate::gpu::{get_gpu_manager, GpuError, GpuManager};
use crate::optimized::split_dataframe::column_ops::ColumnView;
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl GpuAccelerated for OptimizedDataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // For the actual DataFrame, we just return a clone since GPU
        // acceleration is applied at operation time, not on the structure itself
        Ok(self.clone())
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
            && self.row_count() >= gpu_manager.context().config.min_size_threshold;

        // Extract columns into matrices
        let matrix1 = self.to_matrix(columns1)?;
        let matrix2 = self.to_matrix(columns2)?;

        if use_gpu {
            // Use GPU acceleration
            let gpu_matrix1 = GpuMatrix::new(matrix1);
            let gpu_matrix2 = GpuMatrix::new(matrix2);

            match gpu_matrix1.dot(&gpu_matrix2) {
                Ok(result) => Ok(result.data),
                Err(e) => {
                    // If GPU fails and fallback is enabled, try CPU
                    if gpu_manager.context().config.fallback_to_cpu {
                        let mut result = Array2::zeros((matrix1.shape()[0], matrix2.shape()[1]));
                        result = matrix1.dot(&matrix2);
                        Ok(result)
                    } else {
                        Err(e)
                    }
                }
            }
        } else {
            // Use CPU implementation
            let mut result = Array2::zeros((matrix1.shape()[0], matrix2.shape()[1]));
            result = matrix1.dot(&matrix2);
            Ok(result)
        }
    }

    /// Convert selected columns to a matrix
    fn to_matrix(&self, columns: &[&str]) -> Result<Array2<f64>> {
        let n_rows = self.row_count();
        let n_cols = columns.len();

        let mut matrix = Array2::zeros((n_rows, n_cols));

        for (col_idx, col_name) in columns.iter().enumerate() {
            match self.column(*col_name)? {
                ColumnView::Float(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] = col.get(row_idx);
                    }
                }
                ColumnView::Int(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] = col.get(row_idx) as f64;
                    }
                }
                ColumnView::Bool(col) => {
                    for row_idx in 0..n_rows {
                        matrix[[row_idx, col_idx]] = if col.get(row_idx) { 1.0 } else { 0.0 };
                    }
                }
                ColumnView::String(_) => {
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
            && self.row_count() >= gpu_manager.context().config.min_size_threshold;

        // Extract columns into a matrix
        let data_matrix = self.to_matrix(columns)?;
        let n_cols = columns.len();

        if use_gpu {
            // Use GPU acceleration
            let gpu_data = GpuMatrix::new(data_matrix.clone());

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
            let cov_matrix = match gpu_centered.data.t().dot(&gpu_centered.data) {
                Ok(result) => result / (self.row_count() - 1) as f64,
                Err(e) => {
                    // If GPU fails and fallback is enabled, compute using CPU
                    if gpu_manager.context().config.fallback_to_cpu {
                        compute_corr_matrix_cpu(&data_matrix)
                    } else {
                        return Err(e);
                    }
                }
            };

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
                        corr_matrix[[i, j]] = cov_ij / (var_i.sqrt() * var_j.sqrt());
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

            // Calculate correlation coefficient
            let corr_ij = cov_sum / (var_i_sum.sqrt() * var_j_sum.sqrt());

            // Store in correlation matrix (symmetric)
            corr_matrix[[i, j]] = corr_ij;
            corr_matrix[[j, i]] = corr_ij;
        }
    }

    Ok(corr_matrix)
}

//! Common GPU operations implementation
//!
//! This module provides the core logic for GPU-accelerated operations, with
//! implementations that can use CUDA when available or fall back to CPU processing.

use ndarray::{Array, Array1, Array2, Axis};
use rayon::prelude::*;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gpu::{GpuConfig, GpuError, GpuManager, GpuOperationType};
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::DataFrame;
use crate::Series;

/// Trait for GPU-accelerated operations
pub trait GpuAccelerated {
    /// Apply GPU acceleration to supported operations
    fn gpu_accelerate(&self) -> Result<Self>
    where
        Self: Sized;

    /// Check if this object can be GPU-accelerated
    fn is_gpu_acceleratable(&self) -> bool;
}

/// Trait for operations that can be executed on GPU
pub trait GpuExecutable {
    /// Execute operation on GPU
    fn execute_on_gpu(&self, manager: &GpuManager) -> Result<Self>
    where
        Self: Sized;

    /// Get operation type
    fn operation_type(&self) -> GpuOperationType;

    /// Get operation size (for deciding whether to use GPU)
    fn size(&self) -> usize;
}

/// GPU-accelerated matrix operations
pub struct GpuMatrix {
    /// The data matrix
    pub data: Array2<f64>,
    /// Whether the matrix is already on GPU
    pub on_gpu: bool,
}

impl GpuMatrix {
    /// Create a new GPU matrix from an ndarray matrix
    pub fn new(data: Array2<f64>) -> Self {
        GpuMatrix {
            data,
            on_gpu: false,
        }
    }

    /// Multiply this matrix by another matrix (CPU fallback)
    pub fn dot_cpu(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Check if dimensions are compatible
        if self.data.shape()[1] != other.data.shape()[0] {
            return Err(Error::Dimension(format!(
                "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
                self.data.shape(),
                other.data.shape()
            )));
        }

        // Perform matrix multiplication using ndarray
        let result = self.data.dot(&other.data);

        Ok(GpuMatrix {
            data: result,
            on_gpu: false,
        })
    }

    /// Matrix multiplication (with possible GPU acceleration)
    pub fn dot(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::matrix_multiply(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!("Warning: GPU matrix multiplication failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.dot_cpu(other)
    }

    /// Element-wise operation (CPU fallback)
    fn elementwise_operation_cpu(
        &self,
        other: &GpuMatrix,
        op: fn(f64, f64) -> f64,
    ) -> Result<GpuMatrix> {
        // Check if dimensions match
        if self.data.shape() != other.data.shape() {
            return Err(Error::Dimension(format!(
                "Incompatible dimensions for element-wise operation: {:?} and {:?}",
                self.data.shape(),
                other.data.shape()
            )));
        }

        // Perform element-wise operation
        let result = Array::from_shape_fn(self.data.dim(), |(i, j)| {
            op(self.data[[i, j]], other.data[[i, j]])
        });

        Ok(GpuMatrix {
            data: result,
            on_gpu: false,
        })
    }

    /// Element-wise addition (with possible GPU acceleration)
    pub fn add(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_add(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!("Warning: GPU addition failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.elementwise_operation_cpu(other, |a, b| a + b)
    }

    /// Element-wise subtraction (with possible GPU acceleration)
    pub fn subtract(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_subtract(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!(
                                "Warning: GPU subtraction failed ({}). Falling back to CPU.",
                                e
                            );
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.elementwise_operation_cpu(other, |a, b| a - b)
    }

    /// Element-wise multiplication (with possible GPU acceleration)
    pub fn multiply(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_multiply(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!(
                                "Warning: GPU multiplication failed ({}). Falling back to CPU.",
                                e
                            );
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.elementwise_operation_cpu(other, |a, b| a * b)
    }

    /// Element-wise division (with possible GPU acceleration)
    pub fn divide(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_divide(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!("Warning: GPU division failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.elementwise_operation_cpu(other, |a, b| if b != 0.0 { a / b } else { f64::NAN })
    }

    /// Sum of all elements (with possible GPU acceleration)
    pub fn sum(&self) -> Result<f64> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrix is large enough
            if manager.is_available() && manager.context().should_use_gpu(self.data.len()) {
                match crate::gpu::cuda::matrix_sum(self, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!("Warning: GPU sum failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        let sum = self.data.sum();
        Ok(sum)
    }

    /// Mean of all elements (with possible GPU acceleration)
    pub fn mean(&self) -> Result<f64> {
        let sum = self.sum()?;
        let len = self.data.len();

        if len > 0 {
            Ok(sum / (len as f64))
        } else {
            Ok(f64::NAN)
        }
    }

    /// Sort matrix rows (with possible GPU acceleration)
    pub fn sort_rows(&self) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrix is large enough
            if manager.is_available() && manager.context().should_use_gpu(self.data.len()) {
                match crate::gpu::cuda::sort_matrix_rows(self, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!("Warning: GPU sort failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation using rayon for parallelism
        let shape = self.data.dim();
        let mut result = self.data.clone();

        // Sort each row in parallel
        result
            .axis_iter_mut(Axis(0))
            .par_bridge()
            .for_each(|mut row| {
                let mut row_vec: Vec<f64> = row.iter().cloned().collect();
                row_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                for (i, val) in row_vec.iter().enumerate() {
                    row[i] = *val;
                }
            });

        Ok(GpuMatrix {
            data: result,
            on_gpu: false,
        })
    }
}

/// Add GPU acceleration to Series
impl<T> GpuAccelerated for Series<T>
where
    T: Clone + Copy + Into<f64> + std::fmt::Debug,
{
    fn gpu_accelerate(&self) -> Result<Self> {
        // If the series is large enough, use GPU acceleration
        if self.len() < 10_000 {
            return Ok(self.clone());
        }

        // Actual acceleration happens when operations are performed
        Ok(self.clone())
    }

    fn is_gpu_acceleratable(&self) -> bool {
        // Simple numeric series are generally acceleratable
        self.len() >= 10_000
    }
}

/// Add GPU acceleration to DataFrame
impl GpuAccelerated for DataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // For now, just return a clone - actual acceleration happens when operations are performed
        Ok(self.clone())
    }

    fn is_gpu_acceleratable(&self) -> bool {
        // Check if DataFrame has numeric columns and is large enough
        let mut has_numeric = false;

        for col_name in self.column_names() {
            if self.is_numeric_column(col_name) {
                has_numeric = true;
                break;
            }
        }

        has_numeric && self.row_count() >= 10_000
    }
}

/// Add GPU acceleration to OptimizedDataFrame
impl GpuAccelerated for OptimizedDataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // For now, just return a clone - actual acceleration happens when operations are performed
        Ok(self.clone())
    }

    fn is_gpu_acceleratable(&self) -> bool {
        // OptimizedDataFrame is already optimized, but can benefit from GPU acceleration
        self.row_count() >= 10_000
    }
}

/// GPU vector operations
pub struct GpuVector {
    /// The data vector
    pub data: Array1<f64>,
    /// Whether the vector is already on GPU
    pub on_gpu: bool,
}

impl GpuVector {
    /// Create a new GPU vector from an ndarray vector
    pub fn new(data: Array1<f64>) -> Self {
        GpuVector {
            data,
            on_gpu: false,
        }
    }

    /// Dot product (with possible GPU acceleration)
    pub fn dot(&self, other: &GpuVector) -> Result<f64> {
        // Check if dimensions are compatible
        if self.data.len() != other.data.len() {
            return Err(Error::Dimension(format!(
                "Incompatible dimensions for dot product: {} and {}",
                self.data.len(),
                other.data.len()
            )));
        }

        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the vectors are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::vector_dot_product(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!(
                                "Warning: GPU dot product failed ({}). Falling back to CPU.",
                                e
                            );
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        let result = self.data.dot(&other.data);
        Ok(result)
    }

    /// Element-wise operation (CPU fallback)
    fn elementwise_operation_cpu(
        &self,
        other: &GpuVector,
        op: fn(f64, f64) -> f64,
    ) -> Result<GpuVector> {
        // Check if dimensions match
        if self.data.len() != other.data.len() {
            return Err(Error::Dimension(format!(
                "Incompatible dimensions for element-wise operation: {} and {}",
                self.data.len(),
                other.data.len()
            )));
        }

        // Perform element-wise operation
        let result = Array::from_iter(
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| op(a, b)),
        );

        Ok(GpuVector {
            data: result,
            on_gpu: false,
        })
    }

    /// Element-wise addition (with possible GPU acceleration)
    pub fn add(&self, other: &GpuVector) -> Result<GpuVector> {
        // Try GPU acceleration if available
        #[cfg(feature = "cuda")]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the vectors are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::vector_add(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config.fallback_to_cpu {
                            println!(
                                "Warning: GPU vector addition failed ({}). Falling back to CPU.",
                                e
                            );
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation
        self.elementwise_operation_cpu(other, |a, b| a + b)
    }
}

/// A trait for operations that can be GPU-accelerated
pub trait GpuOperation {
    /// Get the operation type
    fn operation_type(&self) -> GpuOperationType;

    /// Execute the operation
    fn execute(&self) -> Result<Box<dyn GpuOperation>>;

    /// Execute the operation on GPU
    fn execute_on_gpu(&self, manager: &GpuManager) -> Result<Box<dyn GpuOperation>>;

    /// Get the size of the operation (in elements)
    fn size(&self) -> usize;
}

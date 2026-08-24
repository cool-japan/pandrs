//! Common GPU operations implementation
//!
//! This module provides the core logic for GPU-accelerated operations, with
//! implementations that can use CUDA when available or fall back to CPU processing.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array, Array1, Array2, Axis};

use crate::error::{Error, Result};
use crate::gpu::{GpuManager, GpuOperationType};
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

    /// Get the CPU data
    pub fn to_cpu(&self) -> Result<Array2<f64>> {
        Ok(self.data.clone())
    }

    /// Create from a row-major flat `Vec<f64>` plus shape.
    ///
    /// Used by Python bindings to bridge the ndarray version boundary: the bindings
    /// link against ndarray 0.16 (required by numpy 0.25), while the root crate uses
    /// ndarray 0.17. `Vec<f64>` is version-agnostic and is the safe handoff point.
    pub fn from_raw_parts(data: Vec<f64>, nrows: usize, ncols: usize) -> Result<Self> {
        let arr = Array2::from_shape_vec((nrows, ncols), data)
            .map_err(|e| Error::InvalidValue(e.to_string()))?;
        Ok(Self::new(arr))
    }

    /// Export to a row-major flat `Vec<f64>` plus shape `(nrows, ncols)`.
    ///
    /// Used by Python bindings for the same ndarray version bridge as `from_raw_parts`.
    /// The iteration order is always logical row-major regardless of the underlying
    /// memory layout, so a subsequent `Array2::from_shape_vec` with the returned
    /// shape will reproduce the original matrix exactly.
    pub fn to_raw_parts(&self) -> (Vec<f64>, usize, usize) {
        let nrows = self.data.nrows();
        let ncols = self.data.ncols();
        let flat: Vec<f64> = self.data.iter().cloned().collect();
        (flat, nrows, ncols)
    }

    /// Multiply this matrix by another matrix (CPU fallback)
    pub fn dot_cpu(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Check if dimensions are compatible
        if self.data.shape()[1] != other.data.shape()[0] {
            return Err(Error::DimensionMismatch(format!(
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::matrix_multiply(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!("Warning: GPU matrix multiplication failed ({}). Falling back to CPU.", e);
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
            return Err(Error::DimensionMismatch(format!(
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_add(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
                                "Warning: GPU addition failed ({}). Falling back to CPU.",
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

    /// Element-wise subtraction (with possible GPU acceleration)
    pub fn subtract(&self, other: &GpuMatrix) -> Result<GpuMatrix> {
        // Try GPU acceleration if available
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_subtract(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_multiply(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrices are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::elementwise_divide(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
                                "Warning: GPU division failed ({}). Falling back to CPU.",
                                e
                            );
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation. Plain IEEE-754 division: a nonzero
        // numerator over zero is +-infinity (not NaN), and 0.0 / 0.0 is NaN;
        // the previous `if b != 0.0 { a / b } else { f64::NAN }` collapsed
        // both cases to NaN, discarding the sign/magnitude information a
        // real division-by-zero carries and diverging from the plain `/`
        // used by every other numeric path in the crate (and from the CUDA
        // division kernel text that used to live in `cuda.rs`, which itself
        // forced NaN on any zero divisor regardless of the numerator).
        self.elementwise_operation_cpu(other, |a, b| a / b)
    }

    /// Sum of all elements (with possible GPU acceleration)
    pub fn sum(&self) -> Result<f64> {
        // Try GPU acceleration if available
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrix is large enough
            if manager.is_available() && manager.context().should_use_gpu(self.data.len()) {
                match crate::gpu::cuda::matrix_sum(self, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!("Warning: GPU sum failed ({}). Falling back to CPU.", e);
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the matrix is large enough
            if manager.is_available() && manager.context().should_use_gpu(self.data.len()) {
                match crate::gpu::cuda::sort_matrix_rows(self, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!("Warning: GPU sort failed ({}). Falling back to CPU.", e);
                        } else {
                            return Err(e.into());
                        }
                    }
                }
            }
        }

        // Fallback to CPU implementation using rayon for parallelism
        let mut result = self.data.clone();

        // Sort each row in parallel
        result
            .axis_iter_mut(Axis(0))
            .par_bridge()
            .for_each(|mut row| {
                let mut row_vec: Vec<f64> = row.iter().cloned().collect();
                // `total_cmp` gives a real total order over all `f64` values
                // (NaNs sort as greatest, matching IEEE 754-2019's
                // `totalOrder` predicate) instead of mapping every NaN
                // comparison to `Equal`. The latter is not transitive (it
                // can report `x < NaN`, `NaN == NaN`, and `NaN < x` for the
                // same `x`), and Rust's sort implementations (pattern-
                // defeating quicksort, in use since the 1.81 stable-sort
                // rewrite) detect that kind of inconsistency at runtime and
                // panic with "user-provided comparison function does not
                // correctly implement a total order" instead of silently
                // producing a badly-ordered result.
                row_vec.sort_by(|a, b| a.total_cmp(b));

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
        // There is no real GPU kernel behind this entry point (see the
        // module-level honesty notes throughout `crate::gpu`): GPU
        // acceleration for element-wise/matrix operations happens per-call
        // in `GpuMatrix`/`GpuVector`, not as a one-shot transformation of a
        // `Series`. Returning `Ok(self.clone())` here — the same value
        // regardless of `is_gpu_acceleratable()` — presented a full clone of
        // the data as the result of "acceleration" when nothing was
        // accelerated (and nothing was even attempted); report that
        // honestly instead of paying for a clone that accomplishes nothing.
        Err(Error::NotImplemented(
            "GPU acceleration for Series not implemented (no real CUDA kernel); use GpuVector \
             for real (CPU-fallback) per-operation dispatch"
                .into(),
        ))
    }

    fn is_gpu_acceleratable(&self) -> bool {
        // Simple numeric series are generally acceleratable
        self.len() >= 10_000
    }
}

/// Add GPU acceleration to DataFrame
impl GpuAccelerated for DataFrame {
    fn gpu_accelerate(&self) -> Result<Self> {
        // See `Series::gpu_accelerate` above: no real GPU kernel exists to
        // run here, so this reports that honestly instead of returning an
        // unmodified deep copy under an "acceleration" label.
        Err(Error::NotImplemented(
            "GPU acceleration for DataFrame not implemented (no real CUDA kernel); use \
             GpuMatrix for real (CPU-fallback) per-operation dispatch"
                .into(),
        ))
    }

    fn is_gpu_acceleratable(&self) -> bool {
        // Check if DataFrame has numeric columns and is large enough
        let mut has_numeric = false;

        for col_name in self.column_names() {
            if self.is_numeric_column(&col_name) {
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
        // See `Series::gpu_accelerate` above: no real GPU kernel exists to
        // run here, so this reports that honestly instead of returning an
        // unmodified deep copy under an "acceleration" label.
        Err(Error::NotImplemented(
            "GPU acceleration for OptimizedDataFrame not implemented (no real CUDA kernel); use \
             GpuMatrix for real (CPU-fallback) per-operation dispatch"
                .into(),
        ))
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
            return Err(Error::DimensionMismatch(format!(
                "Incompatible dimensions for dot product: {} and {}",
                self.data.len(),
                other.data.len()
            )));
        }

        // Try GPU acceleration if available
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the vectors are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::vector_dot_product(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
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
            return Err(Error::DimensionMismatch(format!(
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
        #[cfg(cuda_available)]
        {
            let manager = crate::gpu::get_gpu_manager()?;

            // Check if GPU is available and the vectors are large enough
            let total_elements = self.data.len() + other.data.len();
            if manager.is_available() && manager.context().should_use_gpu(total_elements) {
                match crate::gpu::cuda::vector_add(self, other, &manager) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        // If configured to fallback to CPU, do so
                        if manager.context().config().fallback_to_cpu {
                            log::debug!(
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

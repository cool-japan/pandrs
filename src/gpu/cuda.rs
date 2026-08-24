//! CUDA-specific GPU operations
//!
//! This module provides CUDA implementations of GPU operations used in the operations module.
//! It's only compiled when the 'cuda' feature is enabled.

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{GpuError, GpuManager};

// Import CUDA-specific dependencies when the feature is enabled. `Arc` is
// grouped with these (rather than kept as an unconditional top-level import)
// because every one of its uses below lives inside `PandrsGpuContext`, which
// is itself entirely `#[cfg(cuda_available)]`-gated: left unconditional, this
// import is unused (and would warn) whenever `cuda_available` is off, which
// on this crate's own build-script logic is unconditionally true on macOS
// regardless of feature flags.
#[cfg(cuda_available)]
use cudarc::cublas::CudaBlas;
#[cfg(cuda_available)]
use cudarc::driver::CudaFunction;
#[cfg(cuda_available)]
use cudarc::driver::{CudaContext as CudarcContext, CudaStream};
#[cfg(cuda_available)]
use std::sync::Arc;

/// CUDA context wrapper for managing CUDA resources
#[cfg(cuda_available)]
pub struct PandrsGpuContext {
    /// CUDA context (cudarc 0.18+ renamed CudaDevice to CudaContext)
    context: Arc<CudarcContext>,
    /// CUDA stream for operations
    stream: Arc<CudaStream>,
    /// cuBLAS handle
    cublas: Arc<CudaBlas>,
    /// Whether the device supports compute capability 7.0+ (Volta or later)
    supports_tensor_cores: bool,
}

#[cfg(cuda_available)]
impl PandrsGpuContext {
    /// Create a new CUDA context
    pub fn new(device_id: i32) -> Result<Self> {
        // Initialize CUDA context (cudarc 0.18+ uses CudaContext instead of CudaDevice)
        let context = match CudarcContext::new(device_id as usize) {
            Ok(ctx) => ctx,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to initialize CUDA context: {}",
                    e
                ))))
            }
        };

        // Get default stream
        let stream = context.default_stream();

        // Initialize cuBLAS
        let cublas = match CudaBlas::new(stream.clone()) {
            Ok(cublas) => Arc::new(cublas),
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to initialize cuBLAS: {}",
                    e
                ))))
            }
        };

        // Query the device's real compute capability to determine Tensor Core
        // support. Tensor Cores were introduced with the Volta architecture
        // (compute capability 7.0); cudarc 0.19 exposes this via
        // `CudaContext::compute_capability()`. If the query fails we
        // conservatively report no Tensor Core support rather than assuming it.
        let supports_tensor_cores = match context.compute_capability() {
            Ok((major, _minor)) => major >= 7,
            Err(_) => false,
        };

        Ok(PandrsGpuContext {
            context,
            stream,
            cublas,
            supports_tensor_cores,
        })
    }

    /// Get CUDA context
    pub fn context(&self) -> Arc<CudarcContext> {
        self.context.clone()
    }

    /// Get CUDA stream
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    /// Get cuBLAS handle
    pub fn cublas(&self) -> Arc<CudaBlas> {
        self.cublas.clone()
    }

    /// Check if tensor cores are supported
    pub fn supports_tensor_cores(&self) -> bool {
        self.supports_tensor_cores
    }

    /// Load a CUDA kernel from PTX
    pub fn load_kernel(&self, name: &str, _ptx: &str) -> Result<CudaFunction> {
        // In cudarc 0.19.x, kernel loading requires load_module() first.
        // Dynamic PTX loading is not implemented here, so report it honestly
        // instead of returning a fabricated kernel handle.
        Err(Error::from(GpuError::DeviceError(format!(
            "Kernel '{}' not found. PTX loading requires module loading in cudarc 0.19.x.",
            name
        ))))
    }
}

/// Matrix multiplication using CUDA
pub fn matrix_multiply(a: &GpuMatrix, b: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    // Check if dimensions are compatible
    if a.data.shape()[1] != b.data.shape()[0] {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(cuda_available)]
    {
        // A full GEMM here would require cuBLAS bindings cudarc 0.19.x does
        // not expose (see `PandrsGpuContext::cublas`/`load_kernel`). Report
        // that honestly *before* touching the device: transferring both
        // operands and allocating a result buffer only to then unconditionally
        // error would waste a real H2D copy and a real allocation on a call
        // that can never succeed.
        let _ = manager;
        return Err(Error::from(GpuError::DeviceError(
            "GPU matrix multiplication not implemented for cudarc 0.19.x (no cuBLAS GEMM binding)"
                .to_string(),
        )));
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = a.data.dot(&b.data);

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Element-wise binary operation via CUDA (currently unavailable).
///
/// A real implementation would load a compiled kernel, allocate device
/// buffers, transfer both operands, launch the kernel, and copy the result
/// back. `PandrsGpuContext::load_kernel` already reports honestly that
/// cudarc 0.19.x exposes no path to load a kernel from PTX, so calling it
/// here first and then unconditionally erroring after also performing the
/// (real, but pointless) H2D transfers and result allocation would just
/// waste device work on a call that can never succeed. This returns the
/// same honest error immediately instead, after only the dimension check.
///
/// The previous version additionally took an unused `F: Fn(f64, f64) ->
/// f64` type parameter (satisfied only via a `::<fn(f64, f64) -> f64>`
/// turbofish at every call site) and a `ptx_code: &str` argument, neither of
/// which was ever read: `load_kernel`'s PTX parameter is itself discarded
/// (see its doc comment), so the ~160 lines of literal PTX assembly text
/// that were being built and passed in by each caller were dead code.
#[cfg(cuda_available)]
fn elementwise_op(a: &GpuMatrix, b: &GpuMatrix, op_type: &str) -> Result<GpuMatrix> {
    if a.data.shape() != b.data.shape() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for element-wise {}: {:?} and {:?}",
            op_type,
            a.data.shape(),
            b.data.shape()
        )));
    }

    Err(Error::from(GpuError::DeviceError(format!(
        "GPU kernel launch not implemented for cudarc 0.19.x (element-wise {})",
        op_type
    ))))
}

/// Element-wise addition of matrices using CUDA
pub fn elementwise_add(a: &GpuMatrix, b: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for element-wise addition: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(cuda_available)]
    {
        let _ = manager;
        return elementwise_op(a, b, "addition");
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = &a.data + &b.data;

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Element-wise subtraction of matrices using CUDA
pub fn elementwise_subtract(
    a: &GpuMatrix,
    b: &GpuMatrix,
    manager: &GpuManager,
) -> Result<GpuMatrix> {
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for element-wise subtraction: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(cuda_available)]
    {
        let _ = manager;
        return elementwise_op(a, b, "subtraction");
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = &a.data - &b.data;

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Element-wise multiplication of matrices using CUDA
pub fn elementwise_multiply(
    a: &GpuMatrix,
    b: &GpuMatrix,
    manager: &GpuManager,
) -> Result<GpuMatrix> {
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for element-wise multiplication: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(cuda_available)]
    {
        let _ = manager;
        return elementwise_op(a, b, "multiplication");
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = &a.data * &b.data;

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Element-wise division of matrices using CUDA
pub fn elementwise_divide(a: &GpuMatrix, b: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for element-wise division: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(cuda_available)]
    {
        let _ = manager;
        return elementwise_op(a, b, "division");
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = &a.data / &b.data;

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Sum of all matrix elements using CUDA
pub fn matrix_sum(a: &GpuMatrix, manager: &GpuManager) -> Result<f64> {
    #[cfg(cuda_available)]
    {
        // Computing the sum via a cuBLAS dot-with-ones would require bindings
        // cudarc 0.19.x does not expose. Report that honestly before
        // transferring the matrix and a ones-vector to the device and
        // allocating a result buffer, all of which would otherwise be wasted
        // work on a call that can never succeed. `a` is only read on the
        // `#[cfg(not(cuda_available))]` path below, so it is otherwise unused
        // here.
        let _ = (a, manager);
        return Err(Error::from(GpuError::DeviceError(
            "GPU sum not implemented for cudarc 0.19.x (no cuBLAS binding)".to_string(),
        )));
    }

    #[cfg(not(cuda_available))]
    {
        Ok(a.data.sum())
    }
}

/// Sort matrix rows on the GPU.
///
/// A real GPU bitonic-sort kernel is not implemented. The previous version
/// returned the *unsorted* input matrix (or a zero matrix) while presenting it
/// as a sorted result, which is incorrect. This now reports the missing kernel
/// honestly; callers such as `GpuMatrix::sort_rows` fall back to the real CPU
/// sort when this returns `NotImplemented`.
pub fn sort_matrix_rows(_a: &GpuMatrix, _manager: &GpuManager) -> Result<GpuMatrix> {
    Err(Error::NotImplemented(
        "GPU matrix row sort not implemented (no real CUDA kernel)".into(),
    ))
}

/// Vector dot product using CUDA
pub fn vector_dot_product(a: &GpuVector, b: &GpuVector, manager: &GpuManager) -> Result<f64> {
    // Check if dimensions match
    if a.data.len() != b.data.len() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for dot product: {} and {}",
            a.data.len(),
            b.data.len()
        )));
    }

    #[cfg(cuda_available)]
    {
        // A real dot product would need a cuBLAS binding cudarc 0.19.x does
        // not expose. Report that honestly before transferring both vectors
        // and allocating a result buffer, all of which would otherwise be
        // wasted work on a call that can never succeed.
        let _ = manager;
        return Err(Error::from(GpuError::DeviceError(
            "GPU dot product not implemented for cudarc 0.19.x (no cuBLAS binding)".to_string(),
        )));
    }

    #[cfg(not(cuda_available))]
    {
        let result = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum();
        Ok(result)
    }
}

/// Vector addition using CUDA
pub fn vector_add(a: &GpuVector, b: &GpuVector, manager: &GpuManager) -> Result<GpuVector> {
    // Check if dimensions match
    if a.data.len() != b.data.len() {
        return Err(Error::DimensionMismatch(format!(
            "Incompatible dimensions for vector addition: {} and {}",
            a.data.len(),
            b.data.len()
        )));
    }

    #[cfg(cuda_available)]
    {
        // A real vector addition (cuBLAS axpy) would need a binding cudarc
        // 0.19.x does not expose. Report that honestly before transferring
        // both vectors and allocating a result buffer, all of which would
        // otherwise be wasted work on a call that can never succeed.
        let _ = manager;
        return Err(Error::from(GpuError::DeviceError(
            "GPU vector addition not implemented for cudarc 0.19.x (no cuBLAS binding)".to_string(),
        )));
    }

    #[cfg(not(cuda_available))]
    {
        let result_data = &a.data + &b.data;

        Ok(GpuVector {
            data: result_data,
            on_gpu: false,
        })
    }
}

// NOTE: this module used to keep `to_gpu`/`to_cpu` helpers (real H2D/D2H
// transfer wrappers around `cudarc`'s `clone_htod`/`clone_dtoh`) here,
// `#[allow(dead_code)]`-suppressed because nothing called them. They are
// gone rather than merely un-suppressed: every kernel entry point above
// (`matrix_multiply`, `elementwise_op`, `matrix_sum`, `vector_dot_product`,
// `vector_add`) deliberately returns its honest "not implemented" error
// *before* transferring anything, precisely to avoid paying for a real but
// pointless H2D copy ahead of a call that can never succeed (see each
// function's doc comment) — so wiring these helpers into that path would
// undo that fix, and nothing else in the crate has a use for a bare device
// buffer with no kernel to run on it. Real transfer helpers belong here
// again once a real kernel exists to consume their output.

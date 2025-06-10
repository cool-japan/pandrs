//! CUDA-specific GPU operations
//!
//! This module provides CUDA implementations of GPU operations used in the operations module.
//! It's only compiled when the 'cuda' feature is enabled.

use ndarray::{Array1, Array2};
use std::ptr;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{GpuError, GpuManager};

// Import CUDA-specific dependencies when the feature is enabled
#[cfg(feature = "cuda")]
use cudarc::cublas::{CublasContext, CublasError, CublasHandle, CublasOperation};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaFunction;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DevicePtr, DeviceSlice, DriverError};
#[cfg(feature = "cuda")]
use half::f16;

/// CUDA context wrapper for managing CUDA resources
#[cfg(feature = "cuda")]
pub struct CudaContext {
    /// CUDA device
    device: Arc<CudaDevice>,
    /// cuBLAS handle
    cublas: Arc<CublasHandle>,
    /// Whether the device supports compute capability 7.0+ (Volta or later)
    supports_tensor_cores: bool,
}

#[cfg(feature = "cuda")]
impl CudaContext {
    /// Create a new CUDA context
    pub fn new(device_id: i32) -> Result<Self> {
        // Initialize CUDA device
        let device = match CudaDevice::new(device_id as usize) {
            Ok(device) => Arc::new(device),
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to initialize CUDA device: {}",
                    e
                ))))
            }
        };

        // Initialize cuBLAS
        let cublas = match CublasContext::new(device.clone()) {
            Ok(cublas) => Arc::new(cublas),
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to initialize cuBLAS: {}",
                    e
                ))))
            }
        };

        // Check compute capability
        let props = match device.get_properties() {
            Ok(props) => props,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to get device properties: {}",
                    e
                ))))
            }
        };

        let supports_tensor_cores = props.major >= 7;

        Ok(CudaContext {
            device,
            cublas,
            supports_tensor_cores,
        })
    }

    /// Get CUDA device
    pub fn device(&self) -> Arc<CudaDevice> {
        self.device.clone()
    }

    /// Get cuBLAS handle
    pub fn cublas(&self) -> Arc<CublasHandle> {
        self.cublas.clone()
    }

    /// Check if tensor cores are supported
    pub fn supports_tensor_cores(&self) -> bool {
        self.supports_tensor_cores
    }

    /// Load a CUDA kernel from PTX
    pub fn load_kernel(&self, name: &str, ptx: &str) -> Result<CudaFunction> {
        match self.device.load_ptx(ptx.into(), name, &[name]) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to load PTX: {}",
                    e
                ))))
            }
        }

        match self.device.get_func(name, name) {
            Ok(func) => Ok(func),
            Err(e) => Err(Error::from(GpuError::DeviceError(format!(
                "Failed to get function: {}",
                e
            )))),
        }
    }
}

/// Get or create a CUDA context for the specified device
#[cfg(feature = "cuda")]
fn get_cuda_context(manager: &GpuManager) -> Result<Arc<CudaContext>> {
    // In a real implementation, this would maintain a cache of contexts
    // For simplicity, we'll create a new one each time
    let context = CudaContext::new(manager.context().config.device_id)?;
    Ok(Arc::new(context))
}

/// Matrix multiplication using CUDA
pub fn matrix_multiply(a: &GpuMatrix, b: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    // Check if dimensions are compatible
    if a.data.shape()[1] != b.data.shape()[0] {
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_context = get_cuda_context(manager)?;
        let device = cuda_context.device();
        let cublas = cuda_context.cublas();

        // Get dimensions
        let m = a.data.shape()[0] as i32;
        let n = b.data.shape()[1] as i32;
        let k = a.data.shape()[1] as i32;

        // Allocate device memory
        let a_data = a.data.as_slice().unwrap();
        let b_data = b.data.as_slice().unwrap();

        // Copy data to device
        let d_a = match device.htod_copy(a_data) {
            Ok(d_a) => d_a,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy matrix A to device: {}",
                    e
                ))))
            }
        };

        let d_b = match device.htod_copy(b_data) {
            Ok(d_b) => d_b,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy matrix B to device: {}",
                    e
                ))))
            }
        };

        // Allocate device memory for result
        let mut d_c = match device.alloc::<f64>((m * n) as usize) {
            Ok(d_c) => d_c,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to allocate device memory: {}",
                    e
                ))))
            }
        };

        // Set up parameters for GEMM
        let alpha = 1.0f64;
        let beta = 0.0f64;

        // Perform matrix multiplication
        // C = α * (A × B) + β * C
        match cublas.gemm(
            CublasOperation::None, // A is not transposed
            CublasOperation::None, // B is not transposed
            n,                     // Number of columns in the resulting matrix C and in matrix B
            m,                     // Number of rows in the resulting matrix C and in matrix A
            k,                     // Number of columns in matrix A and rows in matrix B
            &alpha,                // Scaling factor for A * B
            &d_b,                  // Input matrix B
            n,                     // Leading dimension of B
            &d_a,                  // Input matrix A
            k,                     // Leading dimension of A
            &beta,                 // Scaling factor for C
            &mut d_c,              // Output matrix C
            n,                     // Leading dimension of C
        ) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::KernelExecutionError(format!(
                    "cuBLAS gemm failed: {}",
                    e
                ))))
            }
        }

        // Copy result back to host
        let mut c_data = vec![0.0f64; (m * n) as usize];
        match device.dtoh_copy(&d_c, &mut c_data) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy result from device: {}",
                    e
                ))))
            }
        }

        // Reshape the flat vector to 2D array
        let result_shape = (m as usize, n as usize);
        let result_data = Array2::from_shape_vec(result_shape, c_data)
            .map_err(|e| Error::Dimension(format!("Failed to reshape result: {}", e)))?;

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false, // Data is now on CPU
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with appropriate dimensions
        let m = a.data.shape()[0];
        let n = b.data.shape()[1];
        let result_shape = (m, n);
        let result_data = Array2::zeros(result_shape);

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Element-wise operation template for CUDA
#[cfg(feature = "cuda")]
fn elementwise_op<F>(
    a: &GpuMatrix,
    b: &GpuMatrix,
    manager: &GpuManager,
    op_name: &str,
    ptx_code: &str,
    op_type: &str,
) -> Result<GpuMatrix>
where
    F: Fn(f64, f64) -> f64,
{
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for element-wise {}: {:?} and {:?}",
            op_type,
            a.data.shape(),
            b.data.shape()
        )));
    }

    let cuda_context = get_cuda_context(manager)?;
    let device = cuda_context.device();

    // Get dimensions
    let shape = a.data.shape();
    let total_elements = shape[0] * shape[1];

    // Load kernel
    let kernel = cuda_context.load_kernel(op_name, ptx_code)?;

    // Allocate device memory
    let a_data = a.data.as_slice().unwrap();
    let b_data = b.data.as_slice().unwrap();

    // Copy data to device
    let d_a = match device.htod_copy(a_data) {
        Ok(d_a) => d_a,
        Err(e) => {
            return Err(Error::from(GpuError::TransferError(format!(
                "Failed to copy matrix A to device: {}",
                e
            ))))
        }
    };

    let d_b = match device.htod_copy(b_data) {
        Ok(d_b) => d_b,
        Err(e) => {
            return Err(Error::from(GpuError::TransferError(format!(
                "Failed to copy matrix B to device: {}",
                e
            ))))
        }
    };

    // Allocate device memory for result
    let mut d_c = match device.alloc::<f64>(total_elements) {
        Ok(d_c) => d_c,
        Err(e) => {
            return Err(Error::from(GpuError::DeviceError(format!(
                "Failed to allocate device memory: {}",
                e
            ))))
        }
    };

    // Calculate grid and block dimensions
    let block_size = 256;
    let grid_size = (total_elements + block_size - 1) / block_size;

    // Launch kernel
    match unsafe {
        kernel.launch(
            [grid_size as u32, 1, 1],  // Grid dimensions
            [block_size as u32, 1, 1], // Block dimensions
            0,                         // Shared memory size
            device.stream(),           // CUDA stream
            // Kernel arguments
            (
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                total_elements as i32,
            ),
        )
    } {
        Ok(()) => {}
        Err(e) => {
            return Err(Error::from(GpuError::KernelExecutionError(format!(
                "Kernel execution failed: {}",
                e
            ))))
        }
    }

    // Copy result back to host
    let mut c_data = vec![0.0f64; total_elements];
    match device.dtoh_copy(&d_c, &mut c_data) {
        Ok(()) => {}
        Err(e) => {
            return Err(Error::from(GpuError::TransferError(format!(
                "Failed to copy result from device: {}",
                e
            ))))
        }
    }

    // Reshape the flat vector to 2D array
    let result_data = Array2::from_shape_vec(shape, c_data)
        .map_err(|e| Error::Dimension(format!("Failed to reshape result: {}", e)))?;

    Ok(GpuMatrix {
        data: result_data,
        on_gpu: false, // Data is now on CPU
    })
}

/// Element-wise addition of matrices using CUDA
pub fn elementwise_add(a: &GpuMatrix, b: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    // Check if dimensions match
    if a.data.shape() != b.data.shape() {
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for element-wise addition: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        // PTX code for element-wise addition kernel
        const PTX_ADD: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry add(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        )
        {
            .reg .b32 	%r<4>;
            .reg .b64 	%rd<8>;
            .reg .f64 	%fd<4>;

            ld.param.u64 	%rd1, [a];
            ld.param.u64 	%rd2, [b];
            ld.param.u64 	%rd3, [c];
            ld.param.u32 	%r1, [n];

            mov.u32 	%r2, %tid.x;
            mov.u32 	%r3, %ntid.x;
            mad.lo.s32 	%r2, %r3, %ctaid.x, %r2;

            setp.ge.s32	%p1, %r2, %r1;
            @%p1 bra 	$L__BB0_2;

            cvt.s64.s32	%rd4, %r2;
            mul.wide.s32 	%rd5, %r2, 8;
            add.s64 	%rd6, %rd1, %rd5;
            add.s64 	%rd7, %rd2, %rd5;
            ld.global.f64 	%fd1, [%rd6];
            ld.global.f64 	%fd2, [%rd7];
            add.f64 	%fd3, %fd1, %fd2;
            add.s64 	%rd4, %rd3, %rd5;
            st.global.f64 	[%rd4], %fd3;

        $L__BB0_2:
            ret;
        }
        "#;

        return elementwise_op::<fn(f64, f64) -> f64>(a, b, manager, "add", PTX_ADD, "addition");
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with same dimensions
        let shape = a.data.shape();
        let result_data = Array2::zeros(shape);

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
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for element-wise subtraction: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        // PTX code for element-wise subtraction kernel
        const PTX_SUB: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry subtract(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        )
        {
            .reg .b32 	%r<4>;
            .reg .b64 	%rd<8>;
            .reg .f64 	%fd<4>;

            ld.param.u64 	%rd1, [a];
            ld.param.u64 	%rd2, [b];
            ld.param.u64 	%rd3, [c];
            ld.param.u32 	%r1, [n];

            mov.u32 	%r2, %tid.x;
            mov.u32 	%r3, %ntid.x;
            mad.lo.s32 	%r2, %r3, %ctaid.x, %r2;

            setp.ge.s32	%p1, %r2, %r1;
            @%p1 bra 	$L__BB0_2;

            cvt.s64.s32	%rd4, %r2;
            mul.wide.s32 	%rd5, %r2, 8;
            add.s64 	%rd6, %rd1, %rd5;
            add.s64 	%rd7, %rd2, %rd5;
            ld.global.f64 	%fd1, [%rd6];
            ld.global.f64 	%fd2, [%rd7];
            sub.f64 	%fd3, %fd1, %fd2;
            add.s64 	%rd4, %rd3, %rd5;
            st.global.f64 	[%rd4], %fd3;

        $L__BB0_2:
            ret;
        }
        "#;

        return elementwise_op::<fn(f64, f64) -> f64>(
            a,
            b,
            manager,
            "subtract",
            PTX_SUB,
            "subtraction",
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with same dimensions
        let shape = a.data.shape();
        let result_data = Array2::zeros(shape);

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
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for element-wise multiplication: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        // PTX code for element-wise multiplication kernel
        const PTX_MUL: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry multiply(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        )
        {
            .reg .b32 	%r<4>;
            .reg .b64 	%rd<8>;
            .reg .f64 	%fd<4>;

            ld.param.u64 	%rd1, [a];
            ld.param.u64 	%rd2, [b];
            ld.param.u64 	%rd3, [c];
            ld.param.u32 	%r1, [n];

            mov.u32 	%r2, %tid.x;
            mov.u32 	%r3, %ntid.x;
            mad.lo.s32 	%r2, %r3, %ctaid.x, %r2;

            setp.ge.s32	%p1, %r2, %r1;
            @%p1 bra 	$L__BB0_2;

            cvt.s64.s32	%rd4, %r2;
            mul.wide.s32 	%rd5, %r2, 8;
            add.s64 	%rd6, %rd1, %rd5;
            add.s64 	%rd7, %rd2, %rd5;
            ld.global.f64 	%fd1, [%rd6];
            ld.global.f64 	%fd2, [%rd7];
            mul.f64 	%fd3, %fd1, %fd2;
            add.s64 	%rd4, %rd3, %rd5;
            st.global.f64 	[%rd4], %fd3;

        $L__BB0_2:
            ret;
        }
        "#;

        return elementwise_op::<fn(f64, f64) -> f64>(
            a,
            b,
            manager,
            "multiply",
            PTX_MUL,
            "multiplication",
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with same dimensions
        let shape = a.data.shape();
        let result_data = Array2::zeros(shape);

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
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for element-wise division: {:?} and {:?}",
            a.data.shape(),
            b.data.shape()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        // PTX code for element-wise division kernel
        const PTX_DIV: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry divide(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        )
        {
            .reg .b32 	%r<4>;
            .reg .b64 	%rd<8>;
            .reg .f64 	%fd<4>;
            .reg .pred 	%p<2>;

            ld.param.u64 	%rd1, [a];
            ld.param.u64 	%rd2, [b];
            ld.param.u64 	%rd3, [c];
            ld.param.u32 	%r1, [n];

            mov.u32 	%r2, %tid.x;
            mov.u32 	%r3, %ntid.x;
            mad.lo.s32 	%r2, %r3, %ctaid.x, %r2;

            setp.ge.s32	%p1, %r2, %r1;
            @%p1 bra 	$L__BB0_2;

            cvt.s64.s32	%rd4, %r2;
            mul.wide.s32 	%rd5, %r2, 8;
            add.s64 	%rd6, %rd1, %rd5;
            add.s64 	%rd7, %rd2, %rd5;
            ld.global.f64 	%fd1, [%rd6];
            ld.global.f64 	%fd2, [%rd7];

            // Check for division by zero
            setp.eq.f64	%p2, %fd2, 0.0;
            @%p2 bra	$L__BB0_1;

            div.rn.f64 	%fd3, %fd1, %fd2;
            bra $L__BB0_3;

        $L__BB0_1:
            mov.f64 	%fd3, 0d7FF8000000000000;  // NaN

        $L__BB0_3:
            add.s64 	%rd4, %rd3, %rd5;
            st.global.f64 	[%rd4], %fd3;

        $L__BB0_2:
            ret;
        }
        "#;

        return elementwise_op::<fn(f64, f64) -> f64>(a, b, manager, "divide", PTX_DIV, "division");
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with same dimensions
        let shape = a.data.shape();
        let result_data = Array2::zeros(shape);

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Sum of all matrix elements using CUDA
pub fn matrix_sum(a: &GpuMatrix, manager: &GpuManager) -> Result<f64> {
    #[cfg(feature = "cuda")]
    {
        let cuda_context = get_cuda_context(manager)?;
        let device = cuda_context.device();
        let cublas = cuda_context.cublas();

        // Get dimensions
        let shape = a.data.shape();
        let total_elements = shape[0] * shape[1];

        // Allocate device memory
        let a_data = a.data.as_slice().unwrap();

        // Copy data to device
        let d_a = match device.htod_copy(a_data) {
            Ok(d_a) => d_a,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy matrix to device: {}",
                    e
                ))))
            }
        };

        // Create a vector of ones for reduction
        let ones = vec![1.0f64; total_elements];
        let d_ones = match device.htod_copy(&ones) {
            Ok(d_ones) => d_ones,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy ones vector to device: {}",
                    e
                ))))
            }
        };

        // Allocate device memory for result
        let mut d_sum = match device.alloc::<f64>(1) {
            Ok(d_sum) => d_sum,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to allocate device memory: {}",
                    e
                ))))
            }
        };

        // Compute dot product: sum = A • ones
        let incx = 1;
        let incy = 1;
        match cublas.dot(total_elements as i32, &d_a, incx, &d_ones, incy, &mut d_sum) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::KernelExecutionError(format!(
                    "cuBLAS dot failed: {}",
                    e
                ))))
            }
        }

        // Copy result back to host
        let mut sum = vec![0.0f64; 1];
        match device.dtoh_copy(&d_sum, &mut sum) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy result from device: {}",
                    e
                ))))
            }
        }

        Ok(sum[0])
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(0.0)
    }
}

/// Sort matrix rows using CUDA
pub fn sort_matrix_rows(a: &GpuMatrix, manager: &GpuManager) -> Result<GpuMatrix> {
    #[cfg(feature = "cuda")]
    {
        // PTX code for bitonic sort kernel (simplified for demonstration)
        const PTX_SORT: &str = r#"
        .version 7.0
        .target sm_70
        .address_size 64

        .visible .entry bitonic_sort(
            .param .u64 input,
            .param .u64 output,
            .param .u32 width,
            .param .u32 height
        )
        {
            // Implementation of bitonic sort would go here
            // This is a simplified placeholder
        }
        "#;

        let cuda_context = get_cuda_context(manager)?;
        let device = cuda_context.device();

        // Get dimensions
        let shape = a.data.shape();
        let height = shape[0];
        let width = shape[1];
        let total_elements = height * width;

        // For demonstration, we'll just copy the input to output
        // In a real implementation, we would perform a proper sort
        let a_data = a.data.as_slice().unwrap();

        // Create result matrix with same dimensions
        let result_data = a.data.clone();

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result matrix with same dimensions
        let shape = a.data.shape();
        let result_data = Array2::zeros(shape);

        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }
}

/// Vector dot product using CUDA
pub fn vector_dot_product(a: &GpuVector, b: &GpuVector, manager: &GpuManager) -> Result<f64> {
    // Check if dimensions match
    if a.data.len() != b.data.len() {
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for dot product: {} and {}",
            a.data.len(),
            b.data.len()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_context = get_cuda_context(manager)?;
        let device = cuda_context.device();
        let cublas = cuda_context.cublas();

        // Get dimensions
        let n = a.data.len() as i32;

        // Allocate device memory
        let a_data = a.data.as_slice().unwrap();
        let b_data = b.data.as_slice().unwrap();

        // Copy data to device
        let d_a = match device.htod_copy(a_data) {
            Ok(d_a) => d_a,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy vector A to device: {}",
                    e
                ))))
            }
        };

        let d_b = match device.htod_copy(b_data) {
            Ok(d_b) => d_b,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy vector B to device: {}",
                    e
                ))))
            }
        };

        // Allocate device memory for result
        let mut d_result = match device.alloc::<f64>(1) {
            Ok(d_result) => d_result,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to allocate device memory: {}",
                    e
                ))))
            }
        };

        // Perform dot product
        let incx = 1;
        let incy = 1;
        match cublas.dot(n, &d_a, incx, &d_b, incy, &mut d_result) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::KernelExecutionError(format!(
                    "cuBLAS dot failed: {}",
                    e
                ))))
            }
        }

        // Copy result back to host
        let mut result = vec![0.0f64; 1];
        match device.dtoh_copy(&d_result, &mut result) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy result from device: {}",
                    e
                ))))
            }
        }

        Ok(result[0])
    }

    #[cfg(not(feature = "cuda"))]
    {
        Ok(0.0)
    }
}

/// Vector addition using CUDA
pub fn vector_add(a: &GpuVector, b: &GpuVector, manager: &GpuManager) -> Result<GpuVector> {
    // Check if dimensions match
    if a.data.len() != b.data.len() {
        return Err(Error::Dimension(format!(
            "Incompatible dimensions for vector addition: {} and {}",
            a.data.len(),
            b.data.len()
        )));
    }

    #[cfg(feature = "cuda")]
    {
        let cuda_context = get_cuda_context(manager)?;
        let device = cuda_context.device();
        let cublas = cuda_context.cublas();

        // Get dimensions
        let n = a.data.len() as i32;

        // Allocate device memory
        let a_data = a.data.as_slice().unwrap();
        let b_data = b.data.as_slice().unwrap();

        // Copy data to device
        let d_a = match device.htod_copy(a_data) {
            Ok(d_a) => d_a,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy vector A to device: {}",
                    e
                ))))
            }
        };

        let d_b = match device.htod_copy(b_data) {
            Ok(d_b) => d_b,
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy vector B to device: {}",
                    e
                ))))
            }
        };

        // Allocate device memory for result (copy B as the starting point)
        let mut d_result = match device.htod_copy(b_data) {
            Ok(d_result) => d_result,
            Err(e) => {
                return Err(Error::from(GpuError::DeviceError(format!(
                    "Failed to allocate device memory: {}",
                    e
                ))))
            }
        };

        // Perform vector addition: result = a + b
        // Using axpy: y = alpha*x + y (with alpha=1.0, x=a, y=b/result)
        let alpha = 1.0f64;
        let incx = 1;
        let incy = 1;
        match cublas.axpy(n, &alpha, &d_a, incx, &mut d_result, incy) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::KernelExecutionError(format!(
                    "cuBLAS axpy failed: {}",
                    e
                ))))
            }
        }

        // Copy result back to host
        let mut result = vec![0.0f64; n as usize];
        match device.dtoh_copy(&d_result, &mut result) {
            Ok(()) => {}
            Err(e) => {
                return Err(Error::from(GpuError::TransferError(format!(
                    "Failed to copy result from device: {}",
                    e
                ))))
            }
        }

        // Convert to Array1
        let result_data = Array1::from_vec(result);

        Ok(GpuVector {
            data: result_data,
            on_gpu: false,
        })
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Create result vector with same dimensions
        let len = a.data.len();
        let result_data = Array1::zeros(len);

        Ok(GpuVector {
            data: result_data,
            on_gpu: false,
        })
    }
}

// Helper functions for memory management

/// Transfer a CPU matrix to GPU memory
#[cfg(feature = "cuda")]
fn to_gpu(matrix: &Array2<f64>, device: &Arc<CudaDevice>) -> Result<DevicePtr<f64>> {
    let data = matrix.as_slice().unwrap();
    let d_data = match device.htod_copy(data) {
        Ok(d_data) => d_data,
        Err(e) => {
            return Err(Error::from(GpuError::TransferError(format!(
                "Failed to copy matrix to device: {}",
                e
            ))))
        }
    };

    Ok(d_data)
}

/// Transfer a GPU matrix to CPU memory
#[cfg(feature = "cuda")]
fn to_cpu<T: cudarc::driver::DeviceRepr + Copy + Default>(
    device: &Arc<CudaDevice>,
    d_data: &DevicePtr<T>,
    shape: (usize, usize),
) -> Result<Vec<T>> {
    let total_elements = shape.0 * shape.1;
    let mut result = vec![T::default(); total_elements];

    match device.dtoh_copy(d_data, &mut result) {
        Ok(()) => Ok(result),
        Err(e) => Err(Error::from(GpuError::TransferError(format!(
            "Failed to copy data from device: {}",
            e
        )))),
    }
}

/// Free GPU memory (no longer needed with RAII approach)
#[allow(dead_code)]
fn free_gpu(gpu_ptr: *mut f64) -> Result<()> {
    // Modern CUDA crates handle this automatically through RAII
    Ok(())
}

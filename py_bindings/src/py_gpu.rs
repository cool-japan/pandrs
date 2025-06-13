//! Python bindings for GPU acceleration
//!
//! This module provides Python bindings for the GPU acceleration features of PandRS.
//! It allows Python users to leverage GPU acceleration for large-scale data operations.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, IntoPyArray};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::py_optimized::PyOptimizedDataFrame;

#[pyclass(name = "GpuConfig")]
#[derive(Clone)]
/// Configuration for GPU acceleration
pub struct PyGpuConfig {
    #[pyo3(get, set)]
    /// Whether GPU acceleration is enabled
    pub enabled: bool,
    
    #[pyo3(get, set)]
    /// Memory limit for GPU operations in bytes
    pub memory_limit: usize,
    
    #[pyo3(get, set)]
    /// Device ID to use (for multi-GPU systems)
    pub device_id: i32,
    
    #[pyo3(get, set)]
    /// Whether to fall back to CPU if GPU operation fails
    pub fallback_to_cpu: bool,
    
    #[pyo3(get, set)]
    /// Whether to use pinned memory for faster transfers
    pub use_pinned_memory: bool,
    
    #[pyo3(get, set)]
    /// Minimum size threshold for offloading to GPU
    pub min_size_threshold: usize,
}

#[pymethods]
impl PyGpuConfig {
    #[new]
    fn new(
        enabled: Option<bool>,
        memory_limit: Option<usize>,
        device_id: Option<i32>,
        fallback_to_cpu: Option<bool>,
        use_pinned_memory: Option<bool>,
        min_size_threshold: Option<usize>
    ) -> Self {
        let config = ::pandrs::gpu::GpuConfig {
            enabled: enabled.unwrap_or(true),
            memory_limit: memory_limit.unwrap_or(1024 * 1024 * 1024), // 1GB default
            device_id: device_id.unwrap_or(0),
            fallback_to_cpu: fallback_to_cpu.unwrap_or(true),
            use_pinned_memory: use_pinned_memory.unwrap_or(true),
            min_size_threshold: min_size_threshold.unwrap_or(10_000),
        };
        
        PyGpuConfig {
            enabled: config.enabled,
            memory_limit: config.memory_limit,
            device_id: config.device_id,
            fallback_to_cpu: config.fallback_to_cpu,
            use_pinned_memory: config.use_pinned_memory,
            min_size_threshold: config.min_size_threshold,
        }
    }
    
    /// Convert to a string representation
    fn __repr__(&self) -> String {
        format!(
            "GpuConfig(enabled={}, memory_limit={}, device_id={}, fallback_to_cpu={}, use_pinned_memory={}, min_size_threshold={})",
            self.enabled, self.memory_limit, self.device_id, self.fallback_to_cpu, self.use_pinned_memory, self.min_size_threshold
        )
    }
}

impl From<PyGpuConfig> for ::pandrs::gpu::GpuConfig {
    fn from(py_config: PyGpuConfig) -> Self {
        ::pandrs::gpu::GpuConfig {
            enabled: py_config.enabled,
            memory_limit: py_config.memory_limit,
            device_id: py_config.device_id,
            fallback_to_cpu: py_config.fallback_to_cpu,
            use_pinned_memory: py_config.use_pinned_memory,
            min_size_threshold: py_config.min_size_threshold,
        }
    }
}

#[pyclass(name = "GpuDeviceStatus")]
#[derive(Clone)]
/// Status of GPU device
pub struct PyGpuDeviceStatus {
    #[pyo3(get)]
    /// Whether a CUDA-compatible GPU is available
    pub available: bool,
    
    #[pyo3(get)]
    /// CUDA version
    pub cuda_version: Option<String>,
    
    #[pyo3(get)]
    /// Device name
    pub device_name: Option<String>,
    
    #[pyo3(get)]
    /// Total device memory in bytes
    pub total_memory: Option<usize>,
    
    #[pyo3(get)]
    /// Free device memory in bytes
    pub free_memory: Option<usize>,
    
    #[pyo3(get)]
    /// Number of CUDA cores
    pub core_count: Option<usize>,
}

impl From<::pandrs::gpu::GpuDeviceStatus> for PyGpuDeviceStatus {
    fn from(status: ::pandrs::gpu::GpuDeviceStatus) -> Self {
        PyGpuDeviceStatus {
            available: status.available,
            cuda_version: status.cuda_version,
            device_name: status.device_name,
            total_memory: status.total_memory,
            free_memory: status.free_memory,
            core_count: status.core_count,
        }
    }
}

#[pymethods]
impl PyGpuDeviceStatus {
    /// Convert to a string representation
    fn __repr__(&self) -> String {
        format!(
            "GpuDeviceStatus(available={}, device_name={}, cuda_version={}, total_memory={}, free_memory={})",
            self.available,
            self.device_name.as_ref().map_or("None", |s| s.as_str()),
            self.cuda_version.as_ref().map_or("None", |s| s.as_str()),
            self.total_memory.map_or("None".to_string(), |m| format!("{} bytes", m)),
            self.free_memory.map_or("None".to_string(), |m| format!("{} bytes", m)),
        )
    }
}

/// Initialize GPU with default configuration
#[pyfunction]
pub fn init_gpu() -> PyResult<PyGpuDeviceStatus> {
    match ::pandrs::gpu::init_gpu() {
        Ok(status) => Ok(status.into()),
        Err(e) => Err(PyValueError::new_err(format!("GPU initialization failed: {}", e))),
    }
}

/// Initialize GPU with custom configuration
#[pyfunction]
pub fn init_gpu_with_config(config: PyGpuConfig) -> PyResult<PyGpuDeviceStatus> {
    match ::pandrs::gpu::init_gpu_with_config(config.into()) {
        Ok(status) => Ok(status.into()),
        Err(e) => Err(PyValueError::new_err(format!("GPU initialization failed: {}", e))),
    }
}

#[pyclass(name = "GpuMatrix")]
/// GPU-accelerated matrix operations
pub struct PyGpuMatrix {
    matrix: ::pandrs::gpu::operations::GpuMatrix,
}

#[pymethods]
impl PyGpuMatrix {
    #[new]
    fn new(array: &PyArray2<f64>) -> PyResult<Self> {
        // Convert PyArray2 to ndarray::Array2
        let array_owned = unsafe { array.as_array().to_owned() };
        
        Ok(PyGpuMatrix {
            matrix: ::pandrs::gpu::operations::GpuMatrix::new(array_owned),
        })
    }
    
    /// Perform matrix multiplication
    fn dot(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.dot(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix multiplication failed: {}", e))),
        }
    }
    
    /// Perform element-wise addition
    fn add(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.add(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix addition failed: {}", e))),
        }
    }
    
    /// Perform element-wise subtraction
    fn subtract(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.subtract(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix subtraction failed: {}", e))),
        }
    }
    
    /// Perform element-wise multiplication
    fn multiply(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.multiply(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix multiplication failed: {}", e))),
        }
    }
    
    /// Perform element-wise division
    fn divide(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.divide(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix division failed: {}", e))),
        }
    }
    
    /// Calculate sum of all elements
    fn sum(&self) -> PyResult<f64> {
        match self.matrix.sum() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!("Matrix sum failed: {}", e))),
        }
    }
    
    /// Calculate mean of all elements
    fn mean(&self) -> PyResult<f64> {
        match self.matrix.mean() {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!("Matrix mean failed: {}", e))),
        }
    }
    
    /// Sort matrix rows
    fn sort_rows(&self) -> PyResult<PyGpuMatrix> {
        match self.matrix.sort_rows() {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!("Matrix sort failed: {}", e))),
        }
    }
    
    /// Convert to NumPy array
    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.matrix.data.clone().into_pyarray(py)
    }
}

/// Add GPU acceleration methods to PyOptimizedDataFrame
#[pymethods]
impl PyOptimizedDataFrame {
    /// Enable GPU acceleration for this DataFrame
    fn gpu_accelerate(&self) -> PyResult<Self> {
        // For now, just return a copy since gpu_accelerate isn't implemented
        // TODO: Implement GPU acceleration when the underlying API is available
        Ok(PyOptimizedDataFrame { inner: self.inner.clone() })
    }
    
    /// Compute correlation matrix with GPU acceleration
    fn gpu_corr<'py>(&self, py: Python<'py>, columns: &PyList) -> PyResult<&'py PyArray2<f64>> {
        // Convert Python list to Rust vector of strings
        let columns: Vec<String> = columns.iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        
        // For now, return a dummy correlation matrix
        // TODO: Implement GPU correlation when the underlying API is available
        let n_cols = columns.len();
        let mut corr_data = vec![0.0; n_cols * n_cols];
        
        // Set diagonal to 1.0 (perfect correlation with self)
        for i in 0..n_cols {
            corr_data[i * n_cols + i] = 1.0;
        }
        
        let corr_matrix = Array2::from_shape_vec((n_cols, n_cols), corr_data)
            .map_err(|e| PyValueError::new_err(format!("Failed to create correlation matrix: {}", e)))?;
            
        Ok(corr_matrix.into_pyarray(py))
    }
    
    /// Perform PCA with GPU acceleration
    fn gpu_pca<'py>(&self, py: Python<'py>, columns: &PyList, n_components: usize) -> PyResult<(PyOptimizedDataFrame, &'py PyArray1<f64>)> {
        // Convert Python list to Rust vector of strings
        let columns: Vec<String> = columns.iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        
        // For now, return dummy PCA results
        // TODO: Implement GPU PCA when the underlying API is available
        let mut result_df = ::pandrs::OptimizedDataFrame::new();
        
        // Add dummy principal components
        for i in 0..n_components {
            let col_name = format!("PC{}", i + 1);
            let col_data: Vec<f64> = vec![0.0; self.inner.row_count()];
            result_df.add_column(col_name, ::pandrs::column::Column::Float64(
                ::pandrs::column::Float64Column::new(col_data)
            )).map_err(|e| PyValueError::new_err(format!("Failed to add PCA column: {}", e)))?;
        }
        
        // Return dummy explained variance
        let explained_variance = vec![0.0; n_components];
        
        Ok((
            PyOptimizedDataFrame { inner: result_df },
            Array1::from_vec(explained_variance).into_pyarray(py)
        ))
    }
    
    /// Perform k-means clustering with GPU acceleration
    fn gpu_kmeans<'py>(&self, py: Python<'py>, columns: &PyList, k: usize, _max_iter: usize) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<usize>, f64)> {
        // Convert Python list to Rust vector of strings
        let columns: Vec<String> = columns.iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        
        // For now, return dummy k-means results
        // TODO: Implement GPU k-means when the underlying API is available
        let n_rows = self.inner.row_count();
        let n_cols = columns.len();
        
        // Create dummy centroids
        let centroids = Array2::zeros((k, n_cols));
        
        // Create dummy labels (assign all points to cluster 0)
        let labels = Array1::zeros(n_rows);
        
        // Dummy inertia
        let inertia = 0.0;
        
        Ok((
            centroids.into_pyarray(py),
            labels.into_pyarray(py),
            inertia
        ))
    }
    
    /// Perform linear regression with GPU acceleration
    fn gpu_linear_regression<'py>(&self, py: Python<'py>, _y_column: &str, x_columns: &PyList) -> PyResult<&'py PyDict> {
        // Convert Python list to Rust vector of strings
        let x_columns: Vec<String> = x_columns.iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        
        // For now, return dummy linear regression results
        // TODO: Implement GPU linear regression when the underlying API is available
        let result_dict = PyDict::new(py);
        
        result_dict.set_item("intercept", 0.0)?;
        
        let coefficients = PyDict::new(py);
        for col in x_columns.iter() {
            coefficients.set_item(col, 0.0)?;
        }
        result_dict.set_item("coefficients", coefficients)?;
        
        result_dict.set_item("r_squared", 0.0)?;
        result_dict.set_item("adj_r_squared", 0.0)?;
        result_dict.set_item("fitted_values", Vec::<f64>::new())?;
        result_dict.set_item("residuals", Vec::<f64>::new())?;
        
        Ok(result_dict)
    }
}

/// Register GPU-related functions and classes
pub fn register(py: Python, m: &PyModule) -> PyResult<()> {
    let gpu = PyModule::new(py, "gpu")?;
    
    gpu.add_class::<PyGpuConfig>()?;
    gpu.add_class::<PyGpuDeviceStatus>()?;
    gpu.add_function(wrap_pyfunction!(init_gpu, gpu)?)?;
    gpu.add_function(wrap_pyfunction!(init_gpu_with_config, gpu)?)?;
    
    gpu.add_class::<PyGpuMatrix>()?;
    
    m.add_submodule(gpu)?;
    
    Ok(())
}
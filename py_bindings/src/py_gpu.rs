//! Python bindings for GPU acceleration
//!
//! This module provides Python bindings for the GPU acceleration features of PandRS.
//! It allows Python users to leverage GPU acceleration for large-scale data operations.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyListMethods, PyModule, PyModuleMethods};

use crate::py_optimized::PyOptimizedDataFrame;

// `init_gpu_with_config` below takes a bare `PyGpuConfig` parameter, which relies
// on the (formerly implicit, now opt-in) `FromPyObject` impl for `Clone`-able
// `#[pyclass]` types to extract it from a Python `GpuConfig` argument.
#[pyclass(name = "GpuConfig", from_py_object)]
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
        min_size_threshold: Option<usize>,
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

// Only ever used as a return type (never a bare `#[pyfunction]`/`#[pymethods]`
// parameter), so `FromPyObject` is never needed; opt out of the deprecated
// implicit by-value blanket impl for `Clone`-able `#[pyclass]` types.
#[pyclass(name = "GpuDeviceStatus", skip_from_py_object)]
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
        Err(e) => Err(PyValueError::new_err(format!(
            "GPU initialization failed: {}",
            e
        ))),
    }
}

/// Initialize GPU with custom configuration
#[pyfunction]
pub fn init_gpu_with_config(config: PyGpuConfig) -> PyResult<PyGpuDeviceStatus> {
    match ::pandrs::gpu::init_gpu_with_config(config.into()) {
        Ok(status) => Ok(status.into()),
        Err(e) => Err(PyValueError::new_err(format!(
            "GPU initialization failed: {}",
            e
        ))),
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
    fn new<'py>(array: &Bound<'py, PyArray2<f64>>) -> PyResult<Self> {
        // Bridge through Vec<f64> to avoid the ndarray version incompatibility:
        // py_bindings links ndarray 0.16 (required by numpy 0.25), while the root
        // crate uses ndarray 0.17.  The two Array2 types are ABI-incompatible, but
        // a plain Vec<f64> is shared safely across the boundary.
        let ro = array.readonly();
        let view = ro.as_array();
        let shape = view.shape();
        let (nrows, ncols) = (shape[0], shape[1]);
        let flat: Vec<f64> = view.iter().cloned().collect();
        let matrix = ::pandrs::gpu::operations::GpuMatrix::from_raw_parts(flat, nrows, ncols)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyGpuMatrix { matrix })
    }

    /// Perform matrix multiplication
    fn dot(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.dot(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Matrix multiplication failed: {}",
                e
            ))),
        }
    }

    /// Perform element-wise addition
    fn add(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.add(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Matrix addition failed: {}",
                e
            ))),
        }
    }

    /// Perform element-wise subtraction
    fn subtract(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.subtract(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Matrix subtraction failed: {}",
                e
            ))),
        }
    }

    /// Perform element-wise multiplication
    fn multiply(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.multiply(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Matrix multiplication failed: {}",
                e
            ))),
        }
    }

    /// Perform element-wise division
    fn divide(&self, other: &PyGpuMatrix) -> PyResult<PyGpuMatrix> {
        match self.matrix.divide(&other.matrix) {
            Ok(result) => Ok(PyGpuMatrix { matrix: result }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Matrix division failed: {}",
                e
            ))),
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
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Bridge back through Vec<f64>: self.matrix.data is a ndarray 0.17 Array2,
        // but into_pyarray (from numpy 0.25) is only implemented for ndarray 0.16
        // types.  Reconstruct as a local (0.16) Array2 before calling into_pyarray.
        let (flat, nrows, ncols) = self.matrix.to_raw_parts();
        let arr = Array2::from_shape_vec((nrows, ncols), flat).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to reconstruct numpy array from GPU matrix: {}",
                e
            ))
        })?;
        Ok(arr.into_pyarray(py))
    }
}

/// Private helper methods for PyOptimizedDataFrame (not exposed to Python).
impl PyOptimizedDataFrame {
    fn to_standard_df(&self) -> PyResult<::pandrs::dataframe::DataFrame> {
        ::pandrs::optimized::standard_dataframe(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("Conversion failed: {}", e)))
    }
}

/// Add GPU acceleration methods to PyOptimizedDataFrame
#[pymethods]
impl PyOptimizedDataFrame {
    /// Enable GPU acceleration for this DataFrame.
    ///
    /// Returns a clone of this DataFrame. In pandrs, GPU acceleration is applied
    /// lazily per-operation (e.g. in matrix ops) rather than eagerly at the
    /// DataFrame level; cloning here preserves the same optimized layout.
    fn gpu_accelerate(&self) -> PyResult<Self> {
        Ok(PyOptimizedDataFrame {
            inner: self.inner.clone(),
        })
    }

    /// Compute correlation matrix using Pearson correlation, with listwise
    /// deletion of rows that have a null in any of the requested columns.
    fn gpu_corr<'py>(
        &self,
        py: Python<'py>,
        columns: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let columns: Vec<String> = columns
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;

        let n = columns.len();

        // Extract each column's values, preserving null positions as `None`
        // (rather than dropping them immediately) so the shared
        // listwise-deletion helper below can align rows *across* columns
        // correctly. The previous version collected only the non-null
        // values of each column independently
        // (`.filter_map(|i| float_col.get(i).ok().flatten())`), which drops
        // nulls one column at a time: whenever two columns didn't have
        // nulls at exactly the same row positions, the surviving values
        // ended up misaligned and the reported correlation silently
        // measured the relationship between the wrong pairs of numbers.
        let mut col_data: Vec<Vec<Option<f64>>> = Vec::with_capacity(n);
        for col_name in &columns {
            let view = self.inner.column(col_name).map_err(|e| {
                PyValueError::new_err(format!("Column '{}' not found: {}", col_name, e))
            })?;
            let vals: Vec<Option<f64>> = if let Some(float_col) = view.as_float64() {
                (0..self.inner.row_count())
                    .map(|i| float_col.get(i).ok().flatten())
                    .collect()
            } else if let Some(int_col) = view.as_int64() {
                (0..self.inner.row_count())
                    .map(|i| int_col.get(i).ok().flatten().map(|v| v as f64))
                    .collect()
            } else {
                return Err(PyValueError::new_err(format!(
                    "Column '{}' is not numeric",
                    col_name
                )));
            };
            col_data.push(vals);
        }

        let corr_matrix =
            ::pandrs::gpu::cpu_math::correlation_matrix_listwise(&col_data).map_err(|e| {
                PyValueError::new_err(format!("Failed to compute correlation matrix: {}", e))
            })?;

        // Bridge across the ndarray-version boundary via a flat row-major
        // `Vec<f64>`, the same pattern `GpuMatrix::to_raw_parts` uses: the
        // root crate's `correlation_matrix_listwise` returns a
        // `scirs2_core::ndarray::Array2` (ndarray 0.17), while this crate's
        // `into_pyarray` needs the locally-linked `ndarray` 0.16 `Array2`.
        let flat: Vec<f64> = corr_matrix.iter().cloned().collect();
        let corr_matrix = Array2::from_shape_vec((n, n), flat).map_err(|e| {
            PyValueError::new_err(format!("Failed to create correlation matrix: {}", e))
        })?;

        Ok(corr_matrix.into_pyarray(py))
    }

    /// Perform PCA using the pandrs ML implementation
    fn gpu_pca<'py>(
        &self,
        py: Python<'py>,
        columns: &Bound<'py, PyList>,
        n_components: usize,
    ) -> PyResult<(PyOptimizedDataFrame, Bound<'py, PyArray1<f64>>)> {
        use ::pandrs::ml::UnsupervisedModel;

        let requested_cols: Vec<String> = columns
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;

        // Convert the inner OptimizedDataFrame to a standard DataFrame,
        // then build a subset containing only the requested columns.
        let full_std = self.to_standard_df()?;
        let mut subset = ::pandrs::dataframe::DataFrame::new();
        for col_name in &requested_cols {
            let col = full_std
                .get_column::<f64>(col_name)
                .map_err(|e| PyValueError::new_err(format!("Column '{}': {}", col_name, e)))?;
            let series =
                ::pandrs::series::Series::new(col.values().to_vec(), Some(col_name.clone()))
                    .map_err(|e| PyValueError::new_err(format!("Series error: {}", e)))?;
            subset
                .add_column(col_name.clone(), series)
                .map_err(|e| PyValueError::new_err(format!("Add column error: {}", e)))?;
        }

        let mut pca = ::pandrs::ml::PCA::new(n_components, false);
        pca.fit(&subset)
            .map_err(|e| PyValueError::new_err(format!("PCA fit failed: {}", e)))?;
        let transformed = pca
            .transform(&subset)
            .map_err(|e| PyValueError::new_err(format!("PCA transform failed: {}", e)))?;

        // Convert transformed DataFrame back to OptimizedDataFrame.
        // NOTE: PCA column names are "PC_1", "PC_2", etc. (underscore-separated).
        let result_opt = ::pandrs::optimized::optimize_dataframe(&transformed)
            .map_err(|e| PyValueError::new_err(format!("Optimize failed: {}", e)))?;

        let ratios = pca
            .explained_variance_ratio
            .unwrap_or_else(|| vec![0.0; n_components]);

        Ok((
            PyOptimizedDataFrame { inner: result_opt },
            Array1::from_vec(ratios).into_pyarray(py),
        ))
    }

    /// Perform k-means clustering using the pandrs ML implementation
    fn gpu_kmeans<'py>(
        &self,
        py: Python<'py>,
        columns: &Bound<'py, PyList>,
        k: usize,
        max_iter: usize,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<usize>>, f64)> {
        use ::pandrs::ml::UnsupervisedModel;

        let cols_vec: Vec<String> = columns
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;

        let n_features = cols_vec.len();

        // Build standard DataFrame from the full inner frame.
        let full_std = self.to_standard_df()?;

        let mut km = ::pandrs::ml::KMeans::new(k)
            .max_iter(max_iter)
            .with_columns(cols_vec);

        km.fit(&full_std)
            .map_err(|e| PyValueError::new_err(format!("KMeans fit failed: {}", e)))?;

        let centroids_nested = km
            .centroids
            .ok_or_else(|| PyValueError::new_err("KMeans did not produce centroids"))?;
        let labels = km
            .labels
            .ok_or_else(|| PyValueError::new_err("KMeans did not produce labels"))?;
        let inertia = km.inertia.unwrap_or(0.0);

        // Flatten centroids to row-major Vec<f64>.
        let flat_centroids: Vec<f64> = centroids_nested.into_iter().flatten().collect();
        let centroids_arr = Array2::from_shape_vec((k, n_features), flat_centroids)
            .map_err(|e| PyValueError::new_err(format!("Centroids shape error: {}", e)))?;

        Ok((
            centroids_arr.into_pyarray(py),
            Array1::from_vec(labels).into_pyarray(py),
            inertia,
        ))
    }

    /// Perform linear regression using the pandrs OptimizedDataFrame implementation
    fn gpu_linear_regression<'py>(
        &self,
        py: Python<'py>,
        y_column: &str,
        x_columns: &Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x_col_names: Vec<String> = x_columns
            .iter()
            .map(|item| item.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;

        let x_refs: Vec<&str> = x_col_names.iter().map(|s| s.as_str()).collect();

        let result = self
            .inner
            .linear_regression(y_column, &x_refs)
            .map_err(|e| PyValueError::new_err(format!("Linear regression failed: {}", e)))?;

        let result_dict = PyDict::new(py);
        result_dict.set_item("intercept", result.intercept)?;

        // Map column names to their coefficients (parallel to x_refs order).
        let coefficients = PyDict::new(py);
        for (col_name, &coeff) in x_col_names.iter().zip(result.coefficients.iter()) {
            coefficients.set_item(col_name, coeff)?;
        }
        result_dict.set_item("coefficients", &coefficients)?;
        result_dict.set_item("r_squared", result.r_squared)?;
        result_dict.set_item("adj_r_squared", result.adj_r_squared)?;
        result_dict.set_item("fitted_values", result.fitted_values)?;
        result_dict.set_item("residuals", result.residuals)?;

        Ok(result_dict)
    }
}

/// Register GPU-related functions and classes
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let gpu = PyModule::new(m.py(), "gpu")?;

    gpu.add_class::<PyGpuConfig>()?;
    gpu.add_class::<PyGpuDeviceStatus>()?;
    gpu.add_function(wrap_pyfunction!(init_gpu, &gpu)?)?;
    gpu.add_function(wrap_pyfunction!(init_gpu_with_config, &gpu)?)?;

    gpu.add_class::<PyGpuMatrix>()?;

    m.add_submodule(&gpu)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Test that pearson_correlation produces 1.0 on the diagonal
    /// and a symmetric matrix for two perfectly correlated columns.
    #[test]
    fn test_pearson_correlation_properties() {
        let col_a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let col_b: Vec<f64> = (0..10).map(|i| (i * 2) as f64).collect();

        // Diagonal must be 1.0 (self-correlation).
        let self_corr = ::pandrs::stats::descriptive::pearson_correlation(&col_a, &col_a)
            .expect("self correlation");
        assert!((self_corr - 1.0).abs() < 1e-10, "self-corr = {}", self_corr);

        // col_a and col_b are perfectly linearly correlated -> r = 1.0.
        let cross = ::pandrs::stats::descriptive::pearson_correlation(&col_a, &col_b)
            .expect("cross correlation");
        assert!((cross - 1.0).abs() < 1e-10, "cross-corr = {}", cross);

        // Symmetry: pearson(a,b) == pearson(b,a).
        let rev = ::pandrs::stats::descriptive::pearson_correlation(&col_b, &col_a)
            .expect("rev correlation");
        assert!(
            (cross - rev).abs() < 1e-15,
            "not symmetric: {} vs {}",
            cross,
            rev
        );
    }

    /// Test that linear_regression recovers intercept=1.0, slope=2.0 for y=2x+1.
    #[test]
    fn test_linear_regression_recovery() {
        use ::pandrs::column::{Column, Float64Column};
        use ::pandrs::OptimizedDataFrame;

        let x_vals: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_vals: Vec<f64> = x_vals.iter().map(|&x| 2.0 * x + 1.0).collect();

        let mut df = OptimizedDataFrame::new();
        df.add_column("x".to_string(), Column::Float64(Float64Column::new(x_vals)))
            .expect("add x");
        df.add_column("y".to_string(), Column::Float64(Float64Column::new(y_vals)))
            .expect("add y");

        let result = df
            .linear_regression("y", &["x"])
            .expect("linear_regression");

        assert!(
            (result.intercept - 1.0).abs() < 1e-6,
            "intercept = {}",
            result.intercept
        );
        assert_eq!(result.coefficients.len(), 1);
        assert!(
            (result.coefficients[0] - 2.0).abs() < 1e-6,
            "slope = {}",
            result.coefficients[0]
        );
        assert!(
            (result.r_squared - 1.0).abs() < 1e-6,
            "r_squared = {}",
            result.r_squared
        );
    }
}

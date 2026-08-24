use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
// Import parent crate (using explicit path)
use ::pandrs::dataframe::serialize::SerializeExt;
use ::pandrs::{DataFrame, NASeries, Series, NA};
use std::collections::HashMap;

// Optimized Python integration module
mod py_optimized;

// Pandas compatibility layer
mod pandas_compat;

// GPU acceleration module
#[cfg(cuda_available)]
mod py_gpu;

/// A Rust-powered DataFrame implementation with pandas-like API
#[pymodule]
fn pandrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register traditional classes
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyNASeries>()?;

    // Register optimized classes
    py_optimized::register_optimized_types(m)?;

    // Register pandas compatibility layer
    pandas_compat::register_pandas_types(m)?;

    // Register GPU acceleration classes if CUDA is enabled
    #[cfg(cuda_available)]
    py_gpu::register(m)?;

    // Add module version. Sourced from the crate's own `Cargo.toml` at compile
    // time so it can never drift from the published package version.
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// General-purpose, string-typed DataFrame (the top-level `pandrs.DataFrame`).
///
/// Every column is stored as text, backed by the real pandrs `DataFrame`, so
/// CSV/JSON I/O, column/shape access, positional `iloc`, and pandas interop are
/// genuine while numeric work goes through string parsing. For fully typed,
/// high-performance columns use `OptimizedDataFrame`; for a pandas-shaped method
/// surface use `PandasDataFrame`.
// `FromPyObject` is never needed for this type (it's never used as a bare
// `#[pymethods]`/`#[pyfunction]` parameter type -- only via `Py<T>`/`PyRef<T>`
// or constructed directly in Rust), so opt out of the deprecated implicit
// by-value `FromPyObject` blanket impl for `Clone`-able `#[pyclass]` types.
#[pyclass(name = "DataFrame", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDataFrame {
    pub(crate) inner: DataFrame,
}

impl PyDataFrame {
    /// Number of rows (Rust-side helper; not exposed to Python).
    pub(crate) fn rs_len(&self) -> usize {
        self.inner.row_count()
    }

    /// Column names as owned `Vec<String>` (Rust-side helper).
    pub(crate) fn rs_column_names(&self) -> Vec<String> {
        self.inner.column_names().to_vec()
    }

    /// Borrow the underlying real `DataFrame` (Rust-side helper) so sibling
    /// modules (e.g. the pandas-compatibility layer) can drive the genuine
    /// pandrs implementations instead of re-deriving data.
    pub(crate) fn rs_df(&self) -> &DataFrame {
        &self.inner
    }

    /// Wrap an already-built real `DataFrame` (Rust-side helper).
    pub(crate) fn from_df(inner: DataFrame) -> Self {
        PyDataFrame { inner }
    }

    /// Replace every column name (Rust-side helper), validating the count and
    /// delegating to the real `DataFrame::set_column_names`.
    pub(crate) fn rs_set_column_names(&mut self, columns: Vec<String>) -> PyResult<()> {
        if columns.len() != self.inner.column_names().len() {
            return Err(PyValueError::new_err(
                "Length of new columns doesn't match the number of columns in DataFrame",
            ));
        }
        self.inner
            .set_column_names(columns)
            .map_err(|e| PyValueError::new_err(format!("Failed to set columns: {}", e)))
    }
}

#[pymethods]
impl PyDataFrame {
    /// Create a new DataFrame from a dictionary of lists/arrays
    #[new]
    #[pyo3(signature = (data=None))]
    fn new(py: Python<'_>, data: Option<Py<PyAny>>) -> PyResult<Self> {
        if let Some(data_obj) = data {
            let data_dict = data_obj.cast_bound::<PyDict>(py)?;
            let mut columns = Vec::new();
            let mut data_values: HashMap<String, Vec<String>> = HashMap::new();

            for item in data_dict.items() {
                let (key, value) = (item.get_item(0)?, item.get_item(1)?);
                let key_str = key.extract::<String>()?;
                columns.push(key_str.clone());

                let values_vec = if let Ok(list) = value.cast::<PyList>() {
                    let mut result = Vec::with_capacity(list.len());
                    for i in 0..list.len() {
                        let item = list.get_item(i)?;
                        if item.is_none() {
                            result.push(NA::<String>::NA.to_string());
                        } else {
                            result.push(item.to_string());
                        }
                    }
                    result
                } else if let Ok(array) = value.cast::<PyArray1<f64>>() {
                    let array_ref = unsafe { array.as_array() };
                    array_ref.iter().map(|v| v.to_string()).collect()
                } else {
                    return Err(PyValueError::new_err("Unsupported data type"));
                };

                data_values.insert(key_str, values_vec);
            }

            // Create DataFrame from the data
            match DataFrame::from_map(data_values, None) {
                Ok(df) => Ok(PyDataFrame { inner: df }),
                Err(e) => Err(PyValueError::new_err(format!(
                    "Failed to create DataFrame: {}",
                    e
                ))),
            }
        } else {
            // Create an empty DataFrame
            Ok(PyDataFrame {
                inner: DataFrame::new(),
            })
        }
    }

    /// Convert DataFrame to a Python dictionary
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        for col in self.inner.column_names() {
            if let Ok(values) = self.inner.get_column::<String>(col) {
                let values_vec = values.values().to_vec();
                let python_list = PyList::new(py, &values_vec)?;
                dict.set_item(col, python_list)?;
            }
        }

        Ok(dict.into())
    }

    /// Get column names
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let cols = self.inner.column_names();
        let python_list = PyList::new(py, cols)?;
        Ok(python_list.into())
    }

    /// Set column names
    #[setter]
    fn set_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        if columns.len() != self.inner.column_names().len() {
            return Err(PyValueError::new_err(
                "Length of new columns doesn't match the number of columns in DataFrame",
            ));
        }

        // Use set_column_names to update all column names
        match self.inner.set_column_names(columns) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to set columns: {}",
                e
            ))),
        }
    }

    /// Rename columns using a dictionary mapping old names to new names
    fn rename_columns(&mut self, columns: HashMap<String, String>) -> PyResult<()> {
        match self.inner.rename_columns(&columns) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to rename columns: {}",
                e
            ))),
        }
    }

    /// Get a single column as Series
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<PySeries> {
        let key_str = key.extract::<String>(py)?;
        match self.inner.get_column::<String>(&key_str) {
            Ok(col) => {
                // Get column data
                let values = col.values().to_vec();
                // Create a new Series
                match Series::new(values, Some(key_str.to_string())) {
                    Ok(series) => Ok(PySeries { inner: series }),
                    Err(e) => Err(PyValueError::new_err(format!(
                        "Failed to create Series: {}",
                        e
                    ))),
                }
            }
            Err(_) => Err(PyValueError::new_err(format!(
                "Column '{}' not found",
                key_str
            ))),
        }
    }

    /// Get the shape of the DataFrame (rows, columns)
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        Ok((self.inner.row_count(), self.inner.column_names().len()))
    }

    /// Get a string representation of the DataFrame
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    /// Get a string representation of the DataFrame
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    /// Convert to a pandas DataFrame (requires pandas)
    fn to_pandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pandas = py.import("pandas")?;
        let pd_df = pandas.getattr("DataFrame")?;

        let dict = self.to_dict(py)?;

        // Use dict directly as an argument
        let pd_df_obj = pd_df.call1((dict,))?;
        Ok(pd_df_obj.into())
    }

    /// Create a DataFrame from a pandas DataFrame
    #[staticmethod]
    fn from_pandas(py: Python<'_>, pandas_df: Py<PyAny>) -> PyResult<Self> {
        // Get columns and prepare data
        let pd_obj = pandas_df.bind(py);
        let columns = pd_obj.getattr("columns")?;
        let _columns_vec = columns.extract::<Vec<String>>()?;

        // Get the data as a dictionary
        let to_dict = pd_obj.getattr("to_dict")?;
        let dict_result = to_dict.call1(("list",))?;
        let dict = dict_result.cast::<PyDict>()?;

        // Convert to our DataFrame format
        let py_obj = dict.clone().into();
        PyDataFrame::new(py, Some(py_obj))
    }

    /// Return a new DataFrame containing the rows at the given positional
    /// indices, in the requested order.
    ///
    /// Positions are validated against the row count (an out-of-range position
    /// is a hard error, never silently skipped or padded). Because this
    /// `DataFrame` wrapper stores every column as strings, the selection is
    /// materialised column-by-column from the real backing columns; the result
    /// carries a fresh default positional index (`0..len`).
    fn iloc(&self, indices: Vec<usize>) -> PyResult<Self> {
        // Reconstruct the selected rows column-by-column.
        let mut data: HashMap<String, Vec<String>> = HashMap::new();

        // Maximum allowed index
        let max_idx = self.inner.row_count();

        // Validate indices up front so an out-of-range request fails loudly.
        for idx in &indices {
            if *idx >= max_idx {
                return Err(PyValueError::new_err(format!(
                    "Index {} out of bounds (max: {})",
                    idx,
                    max_idx.saturating_sub(1)
                )));
            }
        }

        for col_name in self.inner.column_names() {
            let all_values = self.inner.get_column_string_values(col_name).map_err(|e| {
                PyValueError::new_err(format!("Failed to read column '{}': {}", col_name, e))
            })?;
            let selected_values: Vec<String> =
                indices.iter().map(|&idx| all_values[idx].clone()).collect();
            data.insert(col_name.clone(), selected_values);
        }

        match DataFrame::from_map(data, None) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to create DataFrame: {}",
                e
            ))),
        }
    }

    /// Save the DataFrame to a CSV file
    fn to_csv(&self, path: &str) -> PyResult<()> {
        match self.inner.to_csv(path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to save DataFrame to CSV: {}",
                e
            ))),
        }
    }

    /// Load a DataFrame from a CSV file
    #[staticmethod]
    fn read_csv(path: &str) -> PyResult<Self> {
        match DataFrame::from_csv(path, true) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to read CSV: {}", e))),
        }
    }

    /// Convert DataFrame to JSON string
    fn to_json(&self) -> PyResult<String> {
        match self.inner.to_json() {
            Ok(json) => Ok(json),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to convert to JSON: {}",
                e
            ))),
        }
    }

    /// Load a DataFrame from a JSON string
    #[staticmethod]
    fn read_json(json: &str) -> PyResult<Self> {
        match DataFrame::from_json(json) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to read JSON: {}", e))),
        }
    }
}

/// Python wrapper for pandrs Series
#[pyclass(name = "Series")]
struct PySeries {
    inner: Series<String>,
}

#[pymethods]
impl PySeries {
    #[new]
    fn new(name: String, data: Vec<String>) -> PyResult<Self> {
        let inner = Series::new(data, Some(name))
            .map_err(|e| PyValueError::new_err(format!("Failed to create Series: {}", e)))?;
        Ok(PySeries { inner })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    /// Convert to NumPy array
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let data = self.inner.values();
        // Try to convert to numeric values if possible
        let mut values: Vec<f64> = Vec::with_capacity(data.len());

        for s in data {
            if let Ok(num) = s.parse::<f64>() {
                values.push(num);
            } else {
                // If can't parse as number, convert to string representation
                let str_array = PyList::new(py, data)?;
                return Ok(str_array.into());
            }
        }

        // All values successfully parsed as numbers
        let np_array = values.into_pyarray(py);
        Ok(np_array.into())
    }

    #[getter]
    fn values(&self) -> Vec<String> {
        self.inner.values().to_vec()
    }

    /// Shift values by `periods`, filling exposed slots with NA.
    ///
    /// Returns an `NASeries` because shifting introduces missing values at the
    /// padded end of the result. Matches pandas' `Series.shift(periods)` with
    /// the default `fill_value` of NA.
    fn shift(&self, periods: i64) -> PyResult<PyNASeries> {
        let shifted = self
            .inner
            .shift(periods)
            .map_err(|e| PyValueError::new_err(format!("Failed to shift Series: {}", e)))?;
        Ok(PyNASeries { inner: shifted })
    }

    #[getter]
    fn name(&self) -> String {
        match self.inner.name() {
            Some(name) => name.clone(),
            None => "".to_string(),
        }
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.set_name(name);
    }
}

/// Python wrapper for pandrs NASeries
#[pyclass(name = "NASeries")]
struct PyNASeries {
    inner: NASeries<String>,
}

#[pymethods]
impl PyNASeries {
    #[new]
    fn new(name: String, data: Vec<Option<String>>) -> PyResult<Self> {
        let processed_data: Vec<String> = data
            .iter()
            .map(|opt| match opt {
                Some(s) => s.clone(),
                None => NA::<String>::NA.to_string(),
            })
            .collect();

        let inner = NASeries::<String>::from_strings(processed_data, Some(name))
            .map_err(|e| PyValueError::new_err(format!("Failed to create NASeries: {}", e)))?;
        Ok(PyNASeries { inner })
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    /// Find NA values in the series
    fn isna(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let is_na = self.inner.is_na();
        let np_array = is_na.into_pyarray(py);
        Ok(np_array.into())
    }

    /// Drop NA values from the series
    fn dropna(&self) -> PyResult<Self> {
        match self.inner.dropna() {
            Ok(new_series) => Ok(PyNASeries { inner: new_series }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to drop NA values: {}",
                e
            ))),
        }
    }

    /// Fill NA values with a specific value
    fn fillna(&self, value: String) -> PyResult<Self> {
        match self.inner.fillna(value) {
            Ok(new_series) => Ok(PyNASeries { inner: new_series }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fill NA values: {}",
                e
            ))),
        }
    }

    /// Shift values by `periods`, filling exposed slots with NA.
    ///
    /// Positive `periods` shifts values toward larger indices (NA is inserted
    /// at the front); negative `periods` shifts toward smaller indices (NA is
    /// inserted at the end). Matches pandas' `Series.shift(periods)` with the
    /// default `fill_value` of NA.
    fn shift(&self, periods: i64) -> PyResult<Self> {
        match self.inner.shift(periods) {
            Ok(new_series) => Ok(PyNASeries { inner: new_series }),
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to shift Series: {}",
                e
            ))),
        }
    }

    #[getter]
    fn name(&self) -> String {
        match self.inner.name() {
            Some(name) => name.clone(),
            None => "".to_string(),
        }
    }

    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.set_name(name);
    }
}

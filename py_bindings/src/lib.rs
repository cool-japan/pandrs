use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pandrs::{DataFrame, Series, NA, NASeries};
use std::collections::HashMap;

/// A Rust-powered DataFrame implementation with pandas-like API
#[pymodule]
fn pandrs_python(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyNASeries>()?;
    
    // Add module version
    m.add("__version__", pandrs::VERSION)?;
    
    Ok(())
}

/// Python wrapper for pandrs DataFrame
#[pyclass(name = "DataFrame")]
struct PyDataFrame {
    inner: DataFrame,
}

#[pymethods]
impl PyDataFrame {
    /// Create a new DataFrame from a dictionary of lists/arrays
    #[new]
    fn new(data: Option<&PyDict>, index: Option<Vec<String>>) -> PyResult<Self> {
        if let Some(data_dict) = data {
            let mut columns = Vec::new();
            let mut data_values: HashMap<String, Vec<String>> = HashMap::new();
            
            for (key, value) in data_dict.iter() {
                let key_str = key.extract::<String>()?;
                columns.push(key_str.clone());
                
                let values_vec = if let Ok(list) = value.downcast::<PyList>() {
                    let mut result = Vec::with_capacity(list.len());
                    for item in list.iter() {
                        if item.is_none() {
                            result.push(NA::new().to_string());
                        } else {
                            result.push(item.to_string());
                        }
                    }
                    result
                } else if let Ok(array) = value.extract::<&PyArray1<f64>>() {
                    let array_ref = unsafe { array.as_array() };
                    array_ref.iter().map(|v| v.to_string()).collect()
                } else {
                    return Err(PyValueError::new_err("Unsupported data type"));
                };
                
                data_values.insert(key_str, values_vec);
            }
            
            // Create DataFrame from the data
            match DataFrame::from_map(data_values, index) {
                Ok(df) => Ok(PyDataFrame { inner: df }),
                Err(e) => Err(PyValueError::new_err(format!("Failed to create DataFrame: {}", e))),
            }
        } else {
            // Create an empty DataFrame
            Ok(PyDataFrame { inner: DataFrame::empty() })
        }
    }
    
    /// Convert DataFrame to a Python dictionary
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        
        for col in self.inner.columns() {
            let values = self.inner.get_column(&col).unwrap_or_default();
            let python_list = PyList::new(py, &values);
            dict.set_item(col, python_list)?;
        }
        
        Ok(dict.into())
    }
    
    /// Get column names
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cols = self.inner.columns();
        let python_list = PyList::new(py, &cols);
        Ok(python_list.into())
    }
    
    /// Set column names
    #[setter]
    fn set_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        if columns.len() != self.inner.columns().len() {
            return Err(PyValueError::new_err(
                "Length of new columns doesn't match the number of columns in DataFrame"
            ));
        }
        
        let mut column_map = HashMap::new();
        for (i, col) in columns.iter().enumerate() {
            column_map.insert(self.inner.columns()[i].clone(), col.clone());
        }
        
        match self.inner.rename_columns(&column_map) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to set columns: {}", e))),
        }
    }
    
    /// Get a single column as Series
    fn __getitem__(&self, key: &PyString) -> PyResult<PySeries> {
        let key_str = key.to_str()?;
        match self.inner.get_column(key_str) {
            Some(col) => {
                let series = Series::new(key_str.to_string(), col);
                Ok(PySeries { inner: series })
            },
            None => Err(PyValueError::new_err(format!("Column '{}' not found", key_str))),
        }
    }
    
    /// Get the shape of the DataFrame (rows, columns)
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        Ok((self.inner.len(), self.inner.columns().len()))
    }
    
    /// Get a string representation of the DataFrame
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    /// Get a string representation of the DataFrame
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    /// Convert to a pandas DataFrame (requires pandas)
    fn to_pandas(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let pd_df = pandas.getattr("DataFrame")?;
        
        let dict = self.to_dict(py)?;
        let args = PyTuple::new(py, &[]);
        let kwargs = PyDict::new(py);
        kwargs.set_item("data", dict)?;
        
        if let Some(index) = &self.inner.index().string_values() {
            kwargs.set_item("index", index)?;
        }
        
        Ok(pd_df.call(args, Some(kwargs))?.into())
    }
    
    /// Create a DataFrame from a pandas DataFrame
    #[staticmethod]
    fn from_pandas(pandas_df: &PyAny) -> PyResult<Self> {
        let py = pandas_df.py();
        
        // Get the columns
        let columns = pandas_df.getattr("columns")?.extract::<Vec<String>>()?;
        
        // Get the data as a dictionary
        let to_dict = pandas_df.getattr("to_dict")?;
        let dict = to_dict.call1(("list",))?.extract::<&PyDict>()?;
        
        // Convert to our DataFrame format
        PyDataFrame::new(Some(dict), None)
    }
    
    /// Return a new DataFrame by selecting rows with the given indices
    fn iloc(&self, indices: Vec<usize>) -> PyResult<Self> {
        let mut rows = Vec::new();
        for idx in indices {
            if idx < self.inner.len() {
                if let Some(row) = self.inner.get_row(idx) {
                    rows.push(row);
                }
            } else {
                return Err(PyValueError::new_err(format!("Index {} out of bounds", idx)));
            }
        }
        
        // Create a new DataFrame from the rows
        let cols = self.inner.columns();
        let mut data: HashMap<String, Vec<String>> = HashMap::new();
        
        for col in cols {
            let mut values = Vec::new();
            for row in &rows {
                if let Some(val) = row.get(&col) {
                    values.push(val.clone());
                } else {
                    values.push(NA::new().to_string());
                }
            }
            data.insert(col.clone(), values);
        }
        
        match DataFrame::from_map(data, None) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create DataFrame: {}", e))),
        }
    }
    
    /// Save the DataFrame to a CSV file
    fn to_csv(&self, path: &str) -> PyResult<()> {
        match self.inner.to_csv(path) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to save DataFrame to CSV: {}", e))),
        }
    }
    
    /// Load a DataFrame from a CSV file
    #[staticmethod]
    fn read_csv(path: &str) -> PyResult<Self> {
        match DataFrame::from_csv(path) {
            Ok(df) => Ok(PyDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to read CSV: {}", e))),
        }
    }
    
    /// Convert DataFrame to JSON string
    fn to_json(&self) -> PyResult<String> {
        match self.inner.to_json() {
            Ok(json) => Ok(json),
            Err(e) => Err(PyValueError::new_err(format!("Failed to convert to JSON: {}", e))),
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
    inner: Series,
}

#[pymethods]
impl PySeries {
    #[new]
    fn new(name: String, data: Vec<String>) -> Self {
        PySeries {
            inner: Series::new(name, data),
        }
    }
    
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    /// Convert to NumPy array
    fn to_numpy(&self, py: Python<'_>) -> PyResult<PyObject> {
        let data = self.inner.data();
        let numeric_data: Result<Vec<f64>, _> = data.iter()
            .map(|s| s.parse::<f64>())
            .collect();
        
        match numeric_data {
            Ok(values) => {
                let np_array = values.to_pyarray(py);
                Ok(np_array.to_object(py))
            },
            Err(_) => {
                // Return as string array if can't convert to float
                let np_array = PyArray1::from_vec(py, data.clone());
                Ok(np_array.to_object(py))
            }
        }
    }
    
    #[getter]
    fn values(&self) -> Vec<String> {
        self.inner.data().clone()
    }
    
    #[getter]
    fn name(&self) -> String {
        self.inner.name().clone()
    }
    
    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.set_name(name);
    }
}

/// Python wrapper for pandrs NASeries
#[pyclass(name = "NASeries")]
struct PyNASeries {
    inner: NASeries,
}

#[pymethods]
impl PyNASeries {
    #[new]
    fn new(name: String, data: Vec<Option<String>>) -> Self {
        let processed_data: Vec<String> = data.iter()
            .map(|opt| match opt {
                Some(s) => s.clone(),
                None => NA::new().to_string(),
            })
            .collect();
        
        PyNASeries {
            inner: NASeries::new(name, processed_data),
        }
    }
    
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }
    
    /// Find NA values in the series
    fn isna(&self, py: Python<'_>) -> PyResult<PyObject> {
        let is_na = self.inner.is_na();
        let np_array = is_na.to_pyarray(py);
        Ok(np_array.to_object(py))
    }
    
    /// Drop NA values from the series
    fn dropna(&self) -> PyResult<Self> {
        let new_series = self.inner.dropna().clone();
        Ok(PyNASeries { inner: new_series })
    }
    
    /// Fill NA values with a specific value
    fn fillna(&self, value: String) -> PyResult<Self> {
        let new_series = self.inner.fillna(&value).clone();
        Ok(PyNASeries { inner: new_series })
    }
    
    #[getter]
    fn name(&self) -> String {
        self.inner.name().clone()
    }
    
    #[setter]
    fn set_name(&mut self, name: String) {
        self.inner.set_name(name);
    }
}
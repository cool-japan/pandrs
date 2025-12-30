use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
// Import the parent crate (using explicit paths)
use ::pandrs::{
    OptimizedDataFrame, LazyFrame, AggregateOp, 
    Column, Int64Column, Float64Column, StringColumn, BooleanColumn
};
use ::pandrs::column::ColumnTrait;

// Import the string pool implementation
pub mod py_string_pool;
use self::py_string_pool::{get_or_init_global_pool, py_string_list_to_indices, indices_to_py_string_list};

/// Python wrapper for optimized pandrs DataFrame
#[pyclass(name = "OptimizedDataFrame")]
pub struct PyOptimizedDataFrame {
    inner: OptimizedDataFrame,
}

#[pymethods]
impl PyOptimizedDataFrame {
    /// Create a new optimized DataFrame
    #[new]
    fn new() -> Self {
        PyOptimizedDataFrame {
            inner: OptimizedDataFrame::new()
        }
    }

    /// Add a column to the DataFrame - specialized for numeric data
    fn add_int_column(&mut self, name: String, data: Vec<i64>) -> PyResult<()> {
        let column = Int64Column::new(data);
        match self.inner.add_column(name, Column::Int64(column)) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add column: {}", e))),
        }
    }

    /// Add a column to the DataFrame - specialized for float data
    fn add_float_column(&mut self, name: String, data: Vec<f64>) -> PyResult<()> {
        let column = Float64Column::new(data);
        match self.inner.add_column(name, Column::Float64(column)) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add column: {}", e))),
        }
    }

    /// Add a column to the DataFrame - specialized for string data
    /// Uses string pool optimization for memory efficiency
    fn add_string_column(&mut self, name: String, data: Vec<String>) -> PyResult<()> {
        // String pool optimization (sharing duplicate strings)
        let pool = get_or_init_global_pool();
        let mut string_indices = Vec::with_capacity(data.len());
        
        // Add each string to the pool and get its index
        for s in data {
            let idx = match pool.lock() {
                Ok(mut pool_guard) => pool_guard.add(s),
                Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
            };
            string_indices.push(idx);
        }
        
        // Create a StringColumn from the indices (stores indices instead of actual strings)
        let interned_strings: Vec<String> = string_indices.iter()
            .map(|&idx| idx.to_string())
            .collect();
        
        // Create a string column that retains the original index information
        let column = StringColumn::new(interned_strings);
        match self.inner.add_column(name, Column::String(column)) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add column: {}", e))),
        }
    }
    
    /// Add a column to the DataFrame directly from a Python list
    fn add_string_column_from_pylist(&mut self, py: Python<'_>, name: String, data: PyObject) -> PyResult<()> {
        // Downcast to a list
        let list_obj = data.downcast_bound::<PyList>(py)?;
        
        // Efficiently convert using the string pool
        let indices = py_string_list_to_indices(py, &list_obj)?;
        
        // Create a StringColumn from the indices
        let interned_strings: Vec<String> = indices.iter()
            .map(|&idx| idx.to_string())
            .collect();
        
        let column = StringColumn::new(interned_strings);
        match self.inner.add_column(name, Column::String(column)) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add column: {}", e))),
        }
    }

    /// Add a column to the DataFrame - specialized for boolean data
    fn add_boolean_column(&mut self, name: String, data: Vec<bool>) -> PyResult<()> {
        let column = BooleanColumn::new(data);
        match self.inner.add_column(name, Column::Boolean(column)) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to add column: {}", e))),
        }
    }

    /// Get column names
    #[getter]
    fn column_names(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cols = self.inner.column_names();
        let python_list = PyList::new(py, cols)?;
        Ok(python_list.into())
    }

    /// Get the shape of the DataFrame (rows, columns)
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        Ok((self.inner.row_count(), self.inner.column_count()))
    }

    /// Rename columns using a dictionary mapping old names to new names
    fn rename_columns(&mut self, columns: PyObject, py: Python<'_>) -> PyResult<()> {
        // Convert Python dict to Rust HashMap
        let dict = columns.downcast_bound::<PyDict>(py)?;
        let mut column_map = HashMap::new();
        
        for item in dict.iter() {
            let (key, value) = item;
            let old_name: String = key.extract()?;
            let new_name: String = value.extract()?;
            column_map.insert(old_name, new_name);
        }
        
        match self.inner.rename_columns(&column_map) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to rename columns: {}", e))),
        }
    }

    /// Set all column names in the DataFrame
    fn set_column_names(&mut self, names: Vec<String>) -> PyResult<()> {
        match self.inner.set_column_names(names) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to set column names: {}", e))),
        }
    }

    /// Filter rows by a boolean column
    fn filter(&self, column: String) -> PyResult<Self> {
        match self.inner.filter(&column) {
            Ok(filtered) => Ok(PyOptimizedDataFrame { inner: filtered }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to filter: {}", e))),
        }
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
    fn to_pandas(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        let pd_df = pandas.getattr("DataFrame")?;
        
        // Convert to dictionary
        let dict = PyDict::new(py);
        
        // Add each column to the dictionary with optimized type conversion
        for name in self.inner.column_names() {
            let col_view = match self.inner.column(name) {
                Ok(view) => view,
                Err(e) => return Err(PyValueError::new_err(format!("Failed to get column: {}", e))),
            };
            
            // Type-specific conversion to NumPy arrays for better performance
            if let Some(int_col) = col_view.as_int64() {
                let mut values = Vec::with_capacity(int_col.len());
                for i in 0..int_col.len() {
                    if let Ok(Some(val)) = int_col.get(i) {
                        values.push(val as f64);  // NumPy uses float64 as default
                    } else {
                        values.push(f64::NAN);  // Use NaN for null values
                    }
                }
                let np_array = values.into_pyarray(py);
                dict.set_item(name, np_array)?;
            } else if let Some(float_col) = col_view.as_float64() {
                let mut values = Vec::with_capacity(float_col.len());
                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        values.push(val);
                    } else {
                        values.push(f64::NAN);
                    }
                }
                let np_array = values.into_pyarray(py);
                dict.set_item(name, np_array)?;
            } else if let Some(string_col) = col_view.as_string() {
                // Efficiently convert using the string pool
                let pool = get_or_init_global_pool();
                let mut string_indices = Vec::with_capacity(string_col.len());
                
                // Restore values from indices in the string column
                for i in 0..string_col.len() {
                    if let Ok(Some(val)) = string_col.get(i) {
                        // Get the index of the string and convert
                        if let Ok(idx) = val.parse::<usize>() {
                            string_indices.push(idx);
                        } else {
                            // If not an index, process as a normal string
                            match pool.lock() {
                                Ok(mut pool_guard) => {
                                    let idx = pool_guard.add(val.to_string());
                                    string_indices.push(idx);
                                },
                                Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
                            }
                        }
                    } else {
                        // Add an index for an empty string
                        match pool.lock() {
                            Ok(mut pool_guard) => {
                                let idx = pool_guard.add(String::new());
                                string_indices.push(idx);
                            },
                            Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
                        }
                    }
                }
                
                // Convert indices to a Python string list
                let py_string_list = indices_to_py_string_list(py, &string_indices)?;
                dict.set_item(name, py_string_list)?;
            } else if let Some(bool_col) = col_view.as_boolean() {
                let mut values = Vec::with_capacity(bool_col.len());
                for i in 0..bool_col.len() {
                    if let Ok(Some(val)) = bool_col.get(i) {
                        values.push(val);
                    } else {
                        values.push(false);
                    }
                }
                let py_list = PyList::new(py, &values)?;
                dict.set_item(name, py_list)?;
            }
        }
        
        // Create pandas DataFrame
        let pd_df_obj = pd_df.call1((dict,))?;
        Ok(pd_df_obj.into())
    }
    
    /// Create an optimized DataFrame from a pandas DataFrame
    #[staticmethod]
    fn from_pandas(py: Python<'_>, pandas_df: PyObject) -> PyResult<Self> {
        // Get columns and prepare data
        let pd_obj = pandas_df.bind(py);
        let columns = pd_obj.getattr("columns")?;
        let columns_vec = columns.extract::<Vec<String>>()?;
        let mut df = OptimizedDataFrame::new();
        
        // Efficiently process each column
        for col_name in &columns_vec {
            // Access column using pandas' __getitem__
            let get_item = pd_obj.getattr("__getitem__")?;
            let pd_col = get_item.call1((col_name,))?;
            
            // Get dtype info to determine the best column type
            let dtype = pd_col.getattr("dtype")?;
            let dtype_str = dtype.str()?.to_string();
            
            // Specialized processing based on data type
            if dtype_str.contains("int") {
                // Use numpy's to_list to get values as Python list
                let values = pd_col.call_method0("to_list")?;
                let int_values: Vec<i64> = values.extract()?;
                let column = Int64Column::new(int_values);
                match df.add_column(col_name.clone(), Column::Int64(column)) {
                    Ok(_) => {},
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to add int column: {}", e))),
                }
            } else if dtype_str.contains("float") {
                let values = pd_col.call_method0("to_list")?;
                let float_values: Vec<f64> = values.extract()?;
                let column = Float64Column::new(float_values);
                match df.add_column(col_name.clone(), Column::Float64(column)) {
                    Ok(_) => {},
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to add float column: {}", e))),
                }
            } else if dtype_str.contains("bool") {
                let values = pd_col.call_method0("to_list")?;
                let bool_values: Vec<bool> = values.extract()?;
                let column = BooleanColumn::new(bool_values);
                match df.add_column(col_name.clone(), Column::Boolean(column)) {
                    Ok(_) => {},
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to add bool column: {}", e))),
                }
            } else {
                // Default to string for anything else - use string pool for efficiency
                let values = pd_col.call_method0("to_list")?;
                let py_list = values.downcast::<PyList>()?;
                
                // Efficiently convert using the string pool
                let indices = py_string_list_to_indices(py, &py_list)?;
                
                // Create a string column from the indices
                let interned_strings: Vec<String> = indices.iter()
                    .map(|&idx| idx.to_string())
                    .collect();
                
                let column = StringColumn::new(interned_strings);
                match df.add_column(col_name.clone(), Column::String(column)) {
                    Ok(_) => {},
                    Err(e) => return Err(PyValueError::new_err(format!("Failed to add string column: {}", e))),
                }
            }
        }
        
        Ok(PyOptimizedDataFrame { inner: df })
    }

    /// Write DataFrame to Parquet file with optional compression
    #[cfg(feature = "parquet")]
    fn to_parquet(&self, path: &str, compression: Option<&str>) -> PyResult<()> {
        use ::pandrs::io::parquet::{write_parquet, ParquetCompression};
        
        let compression_type = compression.map(|comp| match comp {
            "snappy" => ParquetCompression::Snappy,
            "gzip" => ParquetCompression::Gzip,
            "brotli" => ParquetCompression::Brotli,
            "lz4" => ParquetCompression::Lz4,
            "zstd" => ParquetCompression::Zstd,
            _ => ParquetCompression::Snappy,
        });
        
        match write_parquet(&self.inner, path, compression_type) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to write Parquet: {}", e))),
        }
    }

    /// Read DataFrame from Parquet file  
    #[cfg(feature = "parquet")]
    #[classmethod]
    fn from_parquet(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        use ::pandrs::io::parquet::read_parquet;
        
        match read_parquet(path) {
            Ok(df) => Ok(PyOptimizedDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to read Parquet: {}", e))),
        }
    }

    /// Write DataFrame to SQL database
    #[cfg(feature = "sql")]
    fn to_sql(&self, table_name: &str, db_path: &str, if_exists: &str) -> PyResult<()> {
        use ::pandrs::io::sql::write_to_sql;
        
        match write_to_sql(&self.inner, table_name, db_path, if_exists) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to write to SQL: {}", e))),
        }
    }

    /// Read DataFrame from SQL query
    #[cfg(feature = "sql")]
    #[classmethod]  
    fn from_sql(_cls: &Bound<'_, PyType>, query: &str, db_path: &str) -> PyResult<Self> {
        use ::pandrs::io::sql::read_sql;
        
        match read_sql(query, db_path) {
            Ok(df) => {
                // Convert regular DataFrame to OptimizedDataFrame
                let mut opt_df = OptimizedDataFrame::new();
                
                // Add each column from the regular DataFrame
                for col_name in df.column_names() {
                    // This is a placeholder - we'd need proper conversion logic here
                    // For now, return error suggesting to use regular DataFrame for SQL
                    return Err(PyValueError::new_err(
                        "SQL operations currently only supported for regular DataFrame. Use DataFrame.from_sql() instead."
                    ));
                }
                
                Ok(PyOptimizedDataFrame { inner: opt_df })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to read from SQL: {}", e))),
        }
    }
}

/// Python wrapper for LazyFrame
#[pyclass(name = "LazyFrame")]
pub struct PyLazyFrame {
    inner: LazyFrame,
}

#[pymethods]
impl PyLazyFrame {
    /// Create a new LazyFrame from an optimized DataFrame
    #[new]
    fn new(df: &PyOptimizedDataFrame) -> Self {
        PyLazyFrame {
            inner: LazyFrame::new(df.inner.clone())
        }
    }
    
    /// Filter rows by a boolean column
    fn filter(&self, column: String) -> PyResult<Self> {
        let filtered = self.inner.clone().filter(&column);
        Ok(PyLazyFrame { inner: filtered })
    }
    
    /// Select columns to keep
    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        // Convert the list of strings to a slice
        let columns_str: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let selected = self.inner.clone().select(&columns_str);
        Ok(PyLazyFrame { inner: selected })
    }
    
    /// Perform aggregate operations
    fn aggregate(&self, group_by: Vec<String>, agg_list: Vec<(String, String, String)>) -> PyResult<Self> {
        // Convert string operation names to AggregateOp enum
        let agg_ops: Result<Vec<(String, AggregateOp, String)>, PyErr> = agg_list.into_iter()
            .map(|(col, op_str, new_name)| {
                match op_str.as_str() {
                    "sum" => Ok((col, AggregateOp::Sum, new_name)),
                    "mean" | "avg" | "average" => Ok((col, AggregateOp::Mean, new_name)),
                    "min" => Ok((col, AggregateOp::Min, new_name)),
                    "max" => Ok((col, AggregateOp::Max, new_name)),
                    "count" => Ok((col, AggregateOp::Count, new_name)),
                    _ => Err(PyValueError::new_err(format!("Unsupported aggregate operation: {}", op_str))),
                }
            })
            .collect();
            
        let aggs = agg_ops?;
        let aggregated = self.inner.clone().aggregate(group_by, aggs);
        Ok(PyLazyFrame { inner: aggregated })
    }
    
    /// Execute all the lazy operations and return a materialized DataFrame
    fn execute(&self) -> PyResult<PyOptimizedDataFrame> {
        match self.inner.clone().execute() {
            Ok(df) => Ok(PyOptimizedDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to execute: {}", e))),
        }
    }
}

/// Register the optimized types in the Python module
pub fn register_optimized_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the basic optimized classes
    m.add_class::<PyOptimizedDataFrame>()?;
    m.add_class::<PyLazyFrame>()?;
    
    // Register the string pool optimization classes
    py_string_pool::register_string_pool_types(m)?;
    
    Ok(())
}
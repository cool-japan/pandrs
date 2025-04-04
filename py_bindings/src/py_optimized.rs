use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

// 親クレートをインポート（明示的パスを使用）
use ::pandrs::{
    DataFrame, Series, NA, NASeries,
    OptimizedDataFrame, LazyFrame, AggregateOp, 
    Column, Int64Column, Float64Column, StringColumn, BooleanColumn
};
use ::pandrs::column::ColumnTrait;

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
    fn add_string_column(&mut self, name: String, data: Vec<String>) -> PyResult<()> {
        let column = StringColumn::new(data);
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
        let python_list = PyList::new(py, cols);
        Ok(python_list.into())
    }

    /// Get the shape of the DataFrame (rows, columns)
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        Ok((self.inner.row_count(), self.inner.column_count()))
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
                dict.set_item(name, values.to_pyarray(py))?;
            } else if let Some(float_col) = col_view.as_float64() {
                let mut values = Vec::with_capacity(float_col.len());
                for i in 0..float_col.len() {
                    if let Ok(Some(val)) = float_col.get(i) {
                        values.push(val);
                    } else {
                        values.push(f64::NAN);
                    }
                }
                dict.set_item(name, values.to_pyarray(py))?;
            } else if let Some(string_col) = col_view.as_string() {
                let mut values = Vec::with_capacity(string_col.len());
                for i in 0..string_col.len() {
                    if let Ok(Some(val)) = string_col.get(i) {
                        values.push(val.to_string());
                    } else {
                        values.push(String::new());
                    }
                }
                dict.set_item(name, PyList::new(py, &values))?;
            } else if let Some(bool_col) = col_view.as_boolean() {
                let mut values = Vec::with_capacity(bool_col.len());
                for i in 0..bool_col.len() {
                    if let Ok(Some(val)) = bool_col.get(i) {
                        values.push(val);
                    } else {
                        values.push(false);
                    }
                }
                dict.set_item(name, PyList::new(py, &values))?;
            }
        }
        
        // Create pandas DataFrame
        let args = PyTuple::new(py, &[] as &[PyObject]);
        let kwargs = PyDict::new(py);
        kwargs.set_item("data", dict)?;
        
        Ok(pd_df.call(args, Some(kwargs))?.into())
    }
    
    /// Create an optimized DataFrame from a pandas DataFrame
    #[staticmethod]
    fn from_pandas(pandas_df: &PyAny) -> PyResult<Self> {
        let py = pandas_df.py();
        
        // Get columns and prepare data
        let columns = pandas_df.getattr("columns")?.extract::<Vec<String>>()?;
        let mut df = OptimizedDataFrame::new();
        
        // Efficiently process each column
        for col_name in &columns {
            // Access column using pandas' __getitem__
            let pd_col = pandas_df.getattr("__getitem__")?.call1((col_name,))?;
            
            // Get dtype info to determine the best column type
            let dtype = pd_col.getattr("dtype")?.str()?.to_str()?;
            
            // Specialized processing based on data type
            if dtype.contains("int") {
                // Use numpy's to_list to get values as Python list
                let values = pd_col.call_method0("to_list")?;
                let int_values: Vec<i64> = values.extract()?;
                let column = Int64Column::new(int_values);
                df.add_column(col_name.clone(), Column::Int64(column))?;
            } else if dtype.contains("float") {
                let values = pd_col.call_method0("to_list")?;
                let float_values: Vec<f64> = values.extract()?;
                let column = Float64Column::new(float_values);
                df.add_column(col_name.clone(), Column::Float64(column))?;
            } else if dtype.contains("bool") {
                let values = pd_col.call_method0("to_list")?;
                let bool_values: Vec<bool> = values.extract()?;
                let column = BooleanColumn::new(bool_values);
                df.add_column(col_name.clone(), Column::Boolean(column))?;
            } else {
                // Default to string for anything else
                let values = pd_col.call_method0("to_list")?;
                let string_values: Vec<String> = values.extract()?;
                let column = StringColumn::new(string_values);
                df.add_column(col_name.clone(), Column::String(column))?;
            }
        }
        
        Ok(PyOptimizedDataFrame { inner: df })
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
        match self.inner.filter(column) {
            Ok(filtered) => Ok(PyLazyFrame { inner: filtered }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to filter: {}", e))),
        }
    }
    
    /// Select columns to keep
    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        match self.inner.select(columns) {
            Ok(selected) => Ok(PyLazyFrame { inner: selected }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to select columns: {}", e))),
        }
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
            
        match self.inner.aggregate(group_by, aggs) {
            Ok(aggregated) => Ok(PyLazyFrame { inner: aggregated }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to aggregate: {}", e))),
        }
    }
    
    /// Execute all the lazy operations and return a materialized DataFrame
    fn execute(&self) -> PyResult<PyOptimizedDataFrame> {
        match self.inner.execute() {
            Ok(df) => Ok(PyOptimizedDataFrame { inner: df }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to execute: {}", e))),
        }
    }
}

/// Register the optimized types in the Python module
pub fn register_optimized_types(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOptimizedDataFrame>()?;
    m.add_class::<PyLazyFrame>()?;
    Ok(())
}
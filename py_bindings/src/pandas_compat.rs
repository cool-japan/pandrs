//! # Pandas Compatibility Layer
//!
//! This module provides a pandas-compatible API for seamless migration
//! from pandas to PandRS, enabling users to use familiar pandas syntax.

use crate::PyDataFrame;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyStringMethods, PyType};
use std::collections::HashMap;

/// Pandas-compatible DataFrame implementation
#[pyclass(name = "PandasDataFrame")]
#[derive(Clone)]
pub struct PandasCompatibleDataFrame {
    inner: PyDataFrame,
}

#[pymethods]
impl PandasCompatibleDataFrame {
    /// Create DataFrame from dictionary (pandas-style constructor)
    #[new]
    #[pyo3(signature = (data=None, index=None, columns=None, dtype=None))]
    #[allow(unused_variables)]
    fn new(
        py: Python<'_>,
        data: Option<PyObject>,
        index: Option<PyObject>,
        columns: Option<PyObject>,
        dtype: Option<PyObject>,
    ) -> PyResult<Self> {
        let inner = PyDataFrame::new(py, data)?;
        Ok(Self { inner })
    }

    /// Pandas-style head() method
    #[pyo3(signature = (n=5))]
    #[allow(unused_variables)]
    fn head(&self, py: Python<'_>, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        // In a real implementation, you'd slice the DataFrame
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Pandas-style tail() method
    #[pyo3(signature = (n=5))]
    #[allow(unused_variables)]
    fn tail(&self, py: Python<'_>, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        // In a real implementation, you'd slice the DataFrame
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Pandas-style info() method
    fn info(&self, py: Python<'_>) -> PyResult<()> {
        let row_count = self.inner.rs_len();
        let info_str = format!(
            "<class 'pandrs.PandasDataFrame'>\n\
            Data columns (total {} columns):\n\
            #   Column  Non-Null Count  Dtype\n\
            Memory usage: {} KB",
            row_count,
            row_count * 8 / 1024 // Rough estimate
        );

        // Route through Python's print so redirects / Jupyter capture still work.
        py.import("builtins")?
            .getattr("print")?
            .call1((info_str,))?;
        Ok(())
    }

    /// Pandas-style describe() method
    fn describe(&self, py: Python<'_>) -> PyResult<Self> {
        // In a real implementation, you'd compute statistical summaries
        let mut stats_data = HashMap::new();

        // Mock statistical data
        stats_data.insert("count".to_string(), vec!["100".to_string()]);
        stats_data.insert("mean".to_string(), vec!["50.5".to_string()]);
        stats_data.insert("std".to_string(), vec!["28.9".to_string()]);
        stats_data.insert("min".to_string(), vec!["1.0".to_string()]);
        stats_data.insert("25%".to_string(), vec!["25.75".to_string()]);
        stats_data.insert("50%".to_string(), vec!["50.5".to_string()]);
        stats_data.insert("75%".to_string(), vec!["75.25".to_string()]);
        stats_data.insert("max".to_string(), vec!["100.0".to_string()]);

        let stats_dict = PyDict::new(py);
        for (key, values) in stats_data {
            let py_list = PyList::new(py, values)?;
            stats_dict.set_item(key, py_list)?;
        }

        let inner = PyDataFrame::new(py, Some(stats_dict.into()))?;
        Ok(Self { inner })
    }

    /// Pandas-style iloc indexer
    #[allow(unused_variables)]
    fn iloc(&self, py: Python<'_>) -> PyResult<ILocIndexer> {
        Ok(ILocIndexer {
            dataframe: self.clone(),
        })
    }

    /// Pandas-style loc indexer
    #[allow(unused_variables)]
    fn loc(&self, py: Python<'_>) -> PyResult<LocIndexer> {
        Ok(LocIndexer {
            dataframe: self.clone(),
        })
    }

    /// Pandas-style column access via __getitem__
    fn __getitem__(&self, py: Python<'_>, key: PyObject) -> PyResult<PyObject> {
        if let Ok(column_name) = key.downcast_bound::<PyString>(py) {
            // Single column access
            let column_str = column_name.to_cow()?.into_owned();
            self.get_column(py, &column_str)
        } else if let Ok(column_list) = key.downcast_bound::<PyList>(py) {
            // Multiple column access
            let mut column_names: Vec<String> = Vec::with_capacity(column_list.len());
            for item in column_list.iter() {
                let py_str = item
                    .downcast::<PyString>()
                    .map_err(|_| PyKeyError::new_err("Invalid column selection"))?;
                column_names.push(py_str.to_cow()?.into_owned());
            }
            self.select_columns(py, column_names)
        } else {
            Err(PyKeyError::new_err("Unsupported key type"))
        }
    }

    /// Pandas-style query() method
    #[pyo3(signature = (expr, inplace=false))]
    #[allow(unused_variables)]
    fn query(&self, py: Python<'_>, expr: &str, inplace: Option<bool>) -> PyResult<Self> {
        // In a real implementation, you'd parse and execute the query
        println!("Executing query: {}", expr);

        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Pandas-style groupby() method
    fn groupby(&self, py: Python<'_>, by: PyObject) -> PyResult<GroupBy> {
        if let Ok(column_name) = by.downcast_bound::<PyString>(py) {
            Ok(GroupBy {
                dataframe: self.clone(),
                group_keys: vec![column_name.to_cow()?.into_owned()],
            })
        } else {
            Err(PyValueError::new_err("Invalid groupby key"))
        }
    }

    /// Pandas-style drop() method
    #[pyo3(signature = (labels=None, axis=0, index=None, columns=None, inplace=false))]
    #[allow(unused_variables)]
    fn drop(
        &self,
        py: Python<'_>,
        labels: Option<PyObject>,
        axis: Option<i32>,
        index: Option<PyObject>,
        columns: Option<PyObject>,
        inplace: Option<bool>,
    ) -> PyResult<Self> {
        // In a real implementation, you'd handle the drop logic
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Pandas-style merge() method
    #[pyo3(signature = (
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=false,
        right_index=false,
        sort=false,
        suffixes=("_x".to_string(), "_y".to_string())
    ))]
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn merge(
        &self,
        py: Python<'_>,
        right: &Self,
        how: Option<&str>,
        on: Option<PyObject>,
        left_on: Option<PyObject>,
        right_on: Option<PyObject>,
        left_index: Option<bool>,
        right_index: Option<bool>,
        sort: Option<bool>,
        suffixes: Option<(String, String)>,
    ) -> PyResult<Self> {
        // In a real implementation, you'd perform the merge operation
        Ok(Self {
            inner: self.inner.clone(),
        })
    }

    /// Pandas-style to_csv() method
    #[pyo3(signature = (path_or_buf=None, sep=",", na_rep="", index=true, header=true))]
    #[allow(unused_variables)]
    fn to_csv(
        &self,
        py: Python<'_>,
        path_or_buf: Option<&str>,
        sep: Option<&str>,
        na_rep: Option<&str>,
        index: Option<bool>,
        header: Option<bool>,
    ) -> PyResult<Option<String>> {
        let csv_content = "mock,csv,content\n1,2,3\n4,5,6".to_string();

        if let Some(path) = path_or_buf {
            // Write to file
            std::fs::write(path, &csv_content)
                .map_err(|e| PyValueError::new_err(format!("Failed to write CSV: {}", e)))?;
            Ok(None)
        } else {
            // Return as string
            Ok(Some(csv_content))
        }
    }

    /// Pandas-style read_csv() class method
    #[classmethod]
    #[pyo3(signature = (filepath_or_buffer, sep=",", header="infer", names=None, index_col=None))]
    #[allow(unused_variables)]
    fn read_csv(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        filepath_or_buffer: &str,
        sep: Option<&str>,
        header: Option<&str>,
        names: Option<PyObject>,
        index_col: Option<PyObject>,
    ) -> PyResult<Self> {
        // In a real implementation, you'd parse the CSV file
        let mock_data = PyDict::new(py);
        mock_data.set_item("col1", PyList::new(py, ["1", "2", "3"])?)?;
        mock_data.set_item("col2", PyList::new(py, ["a", "b", "c"])?)?;

        let inner = PyDataFrame::new(py, Some(mock_data.into()))?;
        Ok(Self { inner })
    }

    /// Get DataFrame shape (pandas-style property)
    #[getter]
    #[allow(unused_variables)]
    fn shape(&self, py: Python<'_>) -> PyResult<(usize, usize)> {
        let rows = self.inner.rs_len();
        let cols = self.inner.rs_column_names().len();
        Ok((rows, cols))
    }

    /// Get DataFrame columns (pandas-style property)
    #[getter]
    #[allow(unused_variables)]
    fn columns(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        Ok(self.inner.rs_column_names())
    }

    /// Set DataFrame columns (pandas-style property)
    #[setter]
    #[allow(unused_variables)]
    fn set_columns(&mut self, py: Python<'_>, columns: Vec<String>) -> PyResult<()> {
        // In a real implementation, you'd update the column names
        Ok(())
    }

    /// Get DataFrame index (pandas-style property)
    #[getter]
    #[allow(unused_variables)]
    fn index(&self, py: Python<'_>) -> PyResult<Vec<usize>> {
        let rows = self.inner.rs_len();
        Ok((0..rows).collect())
    }

    /// Helper method for column access
    fn get_column(&self, py: Python<'_>, column_name: &str) -> PyResult<PyObject> {
        // In a real implementation, you'd return the actual column data
        let mock_series = PandasSeries::new(
            py,
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
            Some(column_name.to_string()),
        )?;
        Ok(Py::new(py, mock_series)?.into_any())
    }

    /// Helper method for multi-column selection
    #[allow(unused_variables)]
    fn select_columns(&self, py: Python<'_>, column_names: Vec<String>) -> PyResult<PyObject> {
        // In a real implementation, you'd return a DataFrame with selected columns
        Ok(Py::new(py, self.clone())?.into_any())
    }
}

/// Pandas-compatible Series implementation
#[pyclass(name = "PandasSeries")]
pub struct PandasSeries {
    data: Vec<String>,
    name: Option<String>,
}

#[pymethods]
impl PandasSeries {
    #[new]
    #[allow(unused_variables)]
    #[pyo3(signature = (data, name=None))]
    fn new(py: Python<'_>, data: Vec<String>, name: Option<String>) -> PyResult<Self> {
        Ok(Self { data, name })
    }

    /// Pandas-style head() method
    #[pyo3(signature = (n=5))]
    fn head(&self, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        let head_data = self.data.iter().take(n).cloned().collect();
        Ok(Self {
            data: head_data,
            name: self.name.clone(),
        })
    }

    /// Pandas-style value_counts() method
    #[allow(unused_variables)]
    fn value_counts(&self, py: Python<'_>) -> PyResult<Self> {
        let mut counts = HashMap::new();
        for value in &self.data {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }

        let mut count_data = Vec::new();
        for (_value, count) in counts {
            count_data.push(count.to_string());
        }

        Ok(Self {
            data: count_data,
            name: Some("count".to_string()),
        })
    }

    /// Get Series length
    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Series indexing
    fn __getitem__(&self, index: usize) -> PyResult<String> {
        self.data
            .get(index)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("Index out of bounds"))
    }

    /// String representation
    fn __str__(&self) -> String {
        format!("PandasSeries({:?}, name='{:?}')", self.data, self.name)
    }
}

/// iloc indexer for position-based selection
#[pyclass(name = "ILocIndexer")]
pub struct ILocIndexer {
    dataframe: PandasCompatibleDataFrame,
}

#[pymethods]
impl ILocIndexer {
    /// Position-based indexing
    #[allow(unused_variables)]
    fn __getitem__(&self, py: Python<'_>, key: PyObject) -> PyResult<PyObject> {
        // In a real implementation, you'd handle various iloc patterns
        // For now, return the original DataFrame
        Ok(Py::new(py, self.dataframe.clone())?.into_any())
    }
}

/// loc indexer for label-based selection
#[pyclass(name = "LocIndexer")]
pub struct LocIndexer {
    dataframe: PandasCompatibleDataFrame,
}

#[pymethods]
impl LocIndexer {
    /// Label-based indexing
    #[allow(unused_variables)]
    fn __getitem__(&self, py: Python<'_>, key: PyObject) -> PyResult<PyObject> {
        // In a real implementation, you'd handle various loc patterns
        // For now, return the original DataFrame
        Ok(Py::new(py, self.dataframe.clone())?.into_any())
    }
}

/// GroupBy object for pandas-style groupby operations
#[pyclass(name = "GroupBy")]
pub struct GroupBy {
    dataframe: PandasCompatibleDataFrame,
    group_keys: Vec<String>,
}

#[pymethods]
impl GroupBy {
    /// Compute group sizes
    fn size(&self, py: Python<'_>) -> PyResult<PandasSeries> {
        PandasSeries::new(
            py,
            vec!["10".to_string(), "15".to_string(), "8".to_string()],
            Some("size".to_string()),
        )
    }

    /// Compute group means
    #[allow(unused_variables)]
    fn mean(&self, py: Python<'_>) -> PyResult<PandasCompatibleDataFrame> {
        Ok(PandasCompatibleDataFrame {
            inner: self.dataframe.inner.clone(),
        })
    }

    /// Compute group sums
    #[allow(unused_variables)]
    fn sum(&self, py: Python<'_>) -> PyResult<PandasCompatibleDataFrame> {
        Ok(PandasCompatibleDataFrame {
            inner: self.dataframe.inner.clone(),
        })
    }

    /// Aggregate with custom functions
    #[allow(unused_variables)]
    fn agg(&self, py: Python<'_>, func: PyObject) -> PyResult<PandasCompatibleDataFrame> {
        // In a real implementation, you'd apply the aggregation functions
        Ok(PandasCompatibleDataFrame {
            inner: self.dataframe.inner.clone(),
        })
    }
}

/// Register pandas compatibility types with the Python module
pub fn register_pandas_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PandasCompatibleDataFrame>()?;
    m.add_class::<PandasSeries>()?;
    m.add_class::<ILocIndexer>()?;
    m.add_class::<LocIndexer>()?;
    m.add_class::<GroupBy>()?;

    // Add convenience functions
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;

    Ok(())
}

/// Pandas-style read_csv function
#[pyfunction]
#[pyo3(signature = (filepath_or_buffer, **kwargs))]
#[allow(unused_variables)]
fn read_csv(
    py: Python<'_>,
    filepath_or_buffer: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PandasCompatibleDataFrame> {
    PandasCompatibleDataFrame::read_csv(
        &py.get_type::<PandasCompatibleDataFrame>(),
        py,
        filepath_or_buffer,
        None,
        None,
        None,
        None,
    )
}

/// Pandas-style concat function
#[pyfunction]
#[pyo3(signature = (objs, axis=0, ignore_index=false))]
#[allow(unused_variables)]
fn concat(
    py: Python<'_>,
    objs: Vec<Py<PandasCompatibleDataFrame>>,
    axis: Option<i32>,
    ignore_index: Option<bool>,
) -> PyResult<PandasCompatibleDataFrame> {
    // In a real implementation, you'd concatenate the DataFrames.
    // Preserve previous semantics: return a clone of the first element.
    let first = objs
        .first()
        .ok_or_else(|| PyValueError::new_err("No objects to concatenate"))?;
    let first_ref: PyRef<'_, PandasCompatibleDataFrame> = first.bind(py).borrow();
    Ok(PandasCompatibleDataFrame::clone(&first_ref))
}

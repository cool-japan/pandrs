use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// String pool optimization for Python
/// 
/// Module for efficient sharing and conversion of string data.
/// Reduces memory usage and conversion overhead in string data
/// conversion between Python and Rust.

/// Pool for string internalization
#[pyclass(name = "StringPool")]
pub struct PyStringPool {
    /// Internal pool implementation
    inner: Arc<Mutex<StringPoolInner>>,
}

/// Internal string pool implementation
pub struct StringPoolInner {
    /// String hash map
    string_map: HashMap<StringRef, usize>,
    /// String storage
    strings: Vec<Arc<String>>,
    /// Statistics
    stats: StringPoolStats,
}

/// String pool statistics
#[derive(Clone, Debug, Default)]
struct StringPoolStats {
    /// Total number of stored strings
    total_strings: usize,
    /// Memory saved through sharing
    bytes_saved: usize,
    /// Number of unique strings stored
    unique_strings: usize,
}

/// Safe reference to a string
#[derive(Clone)]
struct StringRef(Arc<String>);

impl PartialEq for StringRef {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_str() == other.0.as_str()
    }
}

impl Eq for StringRef {}

impl Hash for StringRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl StringPoolInner {
    /// Create a new string pool
    pub fn new() -> Self {
        Self {
            string_map: HashMap::new(),
            strings: Vec::new(),
            stats: StringPoolStats::default(),
        }
    }

    /// Add a string and return its index
    pub fn add(&mut self, s: String) -> usize {
        let string_arc = Arc::new(s);
        let string_ref = StringRef(string_arc.clone());
        
        // Return the index if it already exists
        if let Some(&idx) = self.string_map.get(&string_ref) {
            // Record memory saved by detecting duplicate strings
            self.stats.bytes_saved += string_ref.0.len();
            self.stats.total_strings += 1;
            return idx;
        }
        
        // Add new string
        let idx = self.strings.len();
        self.strings.push(string_arc);
        self.string_map.insert(string_ref, idx);
        self.stats.unique_strings += 1;
        self.stats.total_strings += 1;
        
        idx
    }

    /// Get a string from its index
    pub fn get(&self, idx: usize) -> Option<Arc<String>> {
        self.strings.get(idx).cloned()
    }

    /// Look up a string in the pool
    #[allow(dead_code)]
    pub fn lookup(&self, s: &str) -> Option<usize> {
        let temp = StringRef(Arc::new(s.to_string()));
        self.string_map.get(&temp).copied()
    }
}

#[pymethods]
impl PyStringPool {
    /// Create a new string pool
    #[new]
    fn new() -> Self {
        PyStringPool {
            inner: Arc::new(Mutex::new(StringPoolInner::new())),
        }
    }

    /// Add a string to the pool
    fn add(&self, s: String) -> PyResult<usize> {
        match self.inner.lock() {
            Ok(mut pool) => Ok(pool.add(s)),
            Err(_) => Err(PyValueError::new_err("Failed to lock string pool")),
        }
    }

    /// Add strings from a Python list to the pool
    fn add_list(&self, py: Python<'_>, strings: PyObject) -> PyResult<Vec<usize>> {
        let list_obj = strings.downcast_bound::<PyList>(py)?;
        let mut indices = Vec::new();
        
        for item in list_obj.iter() {
            if let Ok(s) = item.extract::<String>() {
                let idx = match self.inner.lock() {
                    Ok(mut pool) => pool.add(s),
                    Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
                };
                indices.push(idx);
            } else {
                return Err(PyValueError::new_err("List must contain only strings"));
            }
        }
        
        Ok(indices)
    }

    /// Get a string from an index
    fn get(&self, idx: usize) -> PyResult<String> {
        match self.inner.lock() {
            Ok(pool) => {
                match pool.get(idx) {
                    Some(s) => Ok(s.to_string()),
                    None => Err(PyValueError::new_err(format!("No string at index {}", idx))),
                }
            },
            Err(_) => Err(PyValueError::new_err("Failed to lock string pool")),
        }
    }

    /// Get a list of strings from a list of indices
    fn get_list(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<PyObject> {
        let pool = match self.inner.lock() {
            Ok(p) => p,
            Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
        };
        
        let strings: Result<Vec<_>, _> = indices.iter()
            .map(|&idx| {
                pool.get(idx)
                    .map(|s| s.to_string())
                    .ok_or_else(|| PyValueError::new_err(format!("No string at index {}", idx)))
            })
            .collect();
        
        let string_vec = strings?;
        
        // Create a Python list directly
        let py_list_temp = PyList::new(py, &string_vec)?;
        Ok(py_list_temp.into())
    }

    /// Get pool statistics
    fn get_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let pool = match self.inner.lock() {
            Ok(p) => p,
            Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
        };
        
        let stats = &pool.stats;
        let dict = PyDict::new(py);
        
        dict.set_item("total_strings", stats.total_strings)?;
        dict.set_item("unique_strings", stats.unique_strings)?;
        dict.set_item("bytes_saved", stats.bytes_saved)?;
        dict.set_item("duplicated_strings", stats.total_strings - stats.unique_strings)?;
        dict.set_item("duplicate_ratio", if stats.total_strings > 0 {
            1.0 - (stats.unique_strings as f64 / stats.total_strings as f64)
        } else {
            0.0
        })?;
        
        Ok(dict.into())
    }
}

/// Global string pool instance for Python bindings
#[allow(static_mut_refs)]
static mut GLOBAL_STRING_POOL: Option<Arc<Mutex<StringPoolInner>>> = None;

/// Access to the global string pool
pub fn get_or_init_global_pool() -> Arc<Mutex<StringPoolInner>> {
    unsafe {
        if GLOBAL_STRING_POOL.is_none() {
            GLOBAL_STRING_POOL = Some(Arc::new(Mutex::new(StringPoolInner::new())));
        }
        GLOBAL_STRING_POOL.as_ref().unwrap().clone()
    }
}

/// String conversion utility function
pub fn py_string_list_to_indices(_py: Python<'_>, list: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
    let pool = get_or_init_global_pool();
    let mut indices = Vec::new();
    
    for item in list.iter() {
        if let Ok(s) = item.extract::<String>() {
            let idx = match pool.lock() {
                Ok(mut inner_pool) => inner_pool.add(s),
                Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
            };
            indices.push(idx);
        } else {
            return Err(PyValueError::new_err("List must contain only strings"));
        }
    }
    
    Ok(indices)
}

/// Convert index list to Python string list
pub fn indices_to_py_string_list(py: Python<'_>, indices: &[usize]) -> PyResult<PyObject> {
    let pool = get_or_init_global_pool();
    let pool_guard = match pool.lock() {
        Ok(guard) => guard,
        Err(_) => return Err(PyValueError::new_err("Failed to lock string pool")),
    };
    
    let mut strings = Vec::with_capacity(indices.len());
    for &idx in indices {
        if let Some(s) = pool_guard.get(idx) {
            strings.push(s.to_string());
        } else {
            return Err(PyValueError::new_err(format!("No string at index {}", idx)));
        }
    }
    
    let py_list = PyList::new(py, &strings)?;
    Ok(py_list.into())
}

/// Register with Python module
pub fn register_string_pool_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStringPool>()?;
    Ok(())
}
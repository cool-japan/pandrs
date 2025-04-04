use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Python向け文字列プール最適化
/// 
/// 文字列データの効率的な共有と変換を実現するためのモジュール。
/// Python<->Rust間の文字列データ変換において、メモリ使用量と
/// 変換オーバーヘッドを削減する。

/// 文字列インターン化のためのプール
#[pyclass(name = "StringPool")]
pub struct PyStringPool {
    /// 内部プール実装
    inner: Arc<Mutex<StringPoolInner>>,
}

/// 内部文字列プール実装
pub struct StringPoolInner {
    /// 文字列のハッシュマップ
    string_map: HashMap<StringRef, usize>,
    /// 文字列ストレージ
    strings: Vec<Arc<String>>,
    /// 統計情報
    stats: StringPoolStats,
}

/// 文字列プールの統計情報
#[derive(Clone, Debug, Default)]
struct StringPoolStats {
    /// 保存された文字列の総数
    total_strings: usize,
    /// 共有によって節約されたメモリ
    bytes_saved: usize,
    /// 格納されている一意の文字列数
    unique_strings: usize,
}

/// 文字列への安全な参照
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
    /// 新しい文字列プールを作成
    pub fn new() -> Self {
        Self {
            string_map: HashMap::new(),
            strings: Vec::new(),
            stats: StringPoolStats::default(),
        }
    }

    /// 文字列を追加し、そのインデックスを返す
    pub fn add(&mut self, s: String) -> usize {
        let string_arc = Arc::new(s);
        let string_ref = StringRef(string_arc.clone());
        
        // すでに存在する場合はそのインデックスを返す
        if let Some(&idx) = self.string_map.get(&string_ref) {
            // 重複文字列の検出で節約したメモリを記録
            self.stats.bytes_saved += string_ref.0.len();
            self.stats.total_strings += 1;
            return idx;
        }
        
        // 新しい文字列を追加
        let idx = self.strings.len();
        self.strings.push(string_arc);
        self.string_map.insert(string_ref, idx);
        self.stats.unique_strings += 1;
        self.stats.total_strings += 1;
        
        idx
    }

    /// インデックスから文字列を取得
    pub fn get(&self, idx: usize) -> Option<Arc<String>> {
        self.strings.get(idx).cloned()
    }

    /// 文字列をプールから検索
    pub fn lookup(&self, s: &str) -> Option<usize> {
        let temp = StringRef(Arc::new(s.to_string()));
        self.string_map.get(&temp).copied()
    }
}

#[pymethods]
impl PyStringPool {
    /// 新しい文字列プールを作成
    #[new]
    fn new() -> Self {
        PyStringPool {
            inner: Arc::new(Mutex::new(StringPoolInner::new())),
        }
    }

    /// 文字列をプールに追加
    fn add(&self, s: String) -> PyResult<usize> {
        match self.inner.lock() {
            Ok(mut pool) => Ok(pool.add(s)),
            Err(_) => Err(PyValueError::new_err("Failed to lock string pool")),
        }
    }

    /// Pythonリストの文字列をプールに追加
    fn add_list(&self, _py: Python<'_>, strings: &PyList) -> PyResult<Vec<usize>> {
        let mut indices = Vec::with_capacity(strings.len());
        
        for item in strings.iter() {
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

    /// インデックスから文字列を取得
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

    /// インデックスのリストから文字列のリストを取得
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
        
        match strings {
            Ok(s) => Ok(PyList::new(py, &s).into()),
            Err(e) => Err(e),
        }
    }

    /// プールの統計情報を取得
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

/// Pythonバインディングで使用する文字列プールグローバルインスタンス
static mut GLOBAL_STRING_POOL: Option<Arc<Mutex<StringPoolInner>>> = None;

/// グローバル文字列プールへのアクセス
pub fn get_or_init_global_pool() -> Arc<Mutex<StringPoolInner>> {
    unsafe {
        if GLOBAL_STRING_POOL.is_none() {
            GLOBAL_STRING_POOL = Some(Arc::new(Mutex::new(StringPoolInner::new())));
        }
        GLOBAL_STRING_POOL.as_ref().unwrap().clone()
    }
}

/// 文字列変換ユーティリティ関数
pub fn py_string_list_to_indices(_py: Python<'_>, list: &PyList) -> PyResult<Vec<usize>> {
    let pool = get_or_init_global_pool();
    let mut indices = Vec::with_capacity(list.len());
    
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

/// インデックスリストからPython文字列リストへの変換
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
    
    Ok(PyList::new(py, &strings).into())
}

/// Python モジュールへの登録
pub fn register_string_pool_types(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyStringPool>()?;
    Ok(())
}
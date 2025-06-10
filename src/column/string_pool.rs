use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once, RwLock};

// Singleton instance of global string pool
lazy_static! {
    pub static ref GLOBAL_STRING_POOL: GlobalStringPool = GlobalStringPool::new();
}

/// Global string pool (singleton)
#[derive(Debug)]
pub struct GlobalStringPool {
    pool: RwLock<StringPoolMut>,
}

impl GlobalStringPool {
    /// Create a new global string pool
    pub fn new() -> Self {
        Self {
            pool: RwLock::new(StringPoolMut {
                strings: Vec::new(),
                hash_map: HashMap::new(),
            }),
        }
    }

    /// Add a string to the pool and return its index
    pub fn get_or_insert(&self, s: &str) -> u32 {
        // Try with read lock
        if let Ok(read_pool) = self.pool.read() {
            if let Some(&idx) = read_pool.hash_map.get(s) {
                return idx;
            }
        }

        // If not found, add with write lock
        if let Ok(mut write_pool) = self.pool.write() {
            // Recheck (another thread might have added it)
            if let Some(&idx) = write_pool.hash_map.get(s) {
                return idx;
            }

            // Assign a new index
            let idx = write_pool.strings.len() as u32;
            let arc_str: Arc<str> = Arc::from(s.to_owned());
            write_pool.strings.push(arc_str.clone());
            write_pool.hash_map.insert(arc_str, idx);
            idx
        } else {
            // Fallback in case of lock failure (should be error handled in practice)
            0
        }
    }

    /// Get a string by its index
    pub fn get(&self, index: u32) -> Option<String> {
        if let Ok(pool) = self.pool.read() {
            pool.strings.get(index as usize).map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Return the number of registered strings
    pub fn len(&self) -> usize {
        if let Ok(pool) = self.pool.read() {
            pool.strings.len()
        } else {
            0
        }
    }

    /// Add a vector of strings to the global pool and return a vector of indices
    pub fn add_strings(&self, strings: &[String]) -> Vec<u32> {
        strings.iter().map(|s| self.get_or_insert(s)).collect()
    }
}

/// String pool for efficiently managing string data
#[derive(Debug, Clone)]
pub struct StringPool {
    strings: Arc<Vec<Arc<str>>>,
    hash_map: Arc<HashMap<Arc<str>, u32>>,
}

impl StringPool {
    /// Create a new empty string pool
    pub fn new() -> Self {
        Self {
            strings: Arc::new(Vec::new()),
            hash_map: Arc::new(HashMap::new()),
        }
    }

    /// Create a new string pool from a vector of strings (optimized version using global pool)
    pub fn from_strings(strings: Vec<String>) -> Self {
        // Add optimization using global pool
        let indices = GLOBAL_STRING_POOL.add_strings(&strings);

        // Create actual pool here, but can utilize references from the global pool
        let mut pool = Self::new_mut();

        for (idx, s) in indices.iter().zip(strings.iter()) {
            let arc_str: Arc<str> = Arc::from(s.to_owned());
            pool.hash_map.insert(arc_str.clone(), *idx);
            if idx >= &(pool.strings.len() as u32) {
                // Expand size as needed (could be optimized in practice)
                while pool.strings.len() <= *idx as usize {
                    pool.strings.push(arc_str.clone());
                }
            } else {
                pool.strings[*idx as usize] = arc_str;
            }
        }

        pool.freeze()
    }

    /// Create a new string pool from a vector of strings (original implementation)
    pub fn from_strings_legacy(strings: Vec<String>) -> Self {
        let mut pool = Self::new_mut();

        for s in strings {
            pool.get_or_insert(&s);
        }

        pool.freeze()
    }

    /// Create a mutable string pool (used internally)
    fn new_mut() -> StringPoolMut {
        StringPoolMut {
            strings: Vec::new(),
            hash_map: HashMap::new(),
        }
    }

    /// Return the number of strings
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Return whether the string pool is empty
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// Get a string by its index
    pub fn get(&self, index: u32) -> Option<&str> {
        self.strings.get(index as usize).map(|s| s.as_ref())
    }

    /// Search for a string and return its index (None if not found)
    pub fn find(&self, s: &str) -> Option<u32> {
        self.hash_map.get(s).copied()
    }

    /// Get all strings as a vector
    pub fn all_strings(&self) -> Vec<&str> {
        self.strings.iter().map(|s| s.as_ref()).collect()
    }

    /// Build a vector of strings from a string pool and indices
    pub fn indices_to_strings(&self, indices: &[u32]) -> Vec<String> {
        indices
            .iter()
            .map(|&idx| self.get(idx).unwrap_or("").to_string())
            .collect()
    }

    /// Merge two string pools
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = Self::new_mut();

        // Add strings from this pool
        for s in self.all_strings() {
            merged.get_or_insert(s);
        }

        // Add strings from the other pool
        for s in other.all_strings() {
            merged.get_or_insert(s);
        }

        merged.freeze()
    }
}

/// Mutable string pool (used only during construction)
#[derive(Debug)]
struct StringPoolMut {
    strings: Vec<Arc<str>>,
    hash_map: HashMap<Arc<str>, u32>,
}

impl StringPoolMut {
    /// Add a string to the pool and return its index
    fn get_or_insert(&mut self, s: &str) -> u32 {
        // Convert string to Arc<str>
        let arc_str: Arc<str> = s.into();

        // If already exists, return its index
        if let Some(&index) = self.hash_map.get(&arc_str) {
            return index;
        }

        // Assign a new index
        let index = self.strings.len() as u32;
        self.strings.push(arc_str.clone());
        self.hash_map.insert(arc_str, index);

        index
    }

    /// Convert mutable pool to immutable pool
    fn freeze(self) -> StringPool {
        StringPool {
            strings: Arc::new(self.strings),
            hash_map: Arc::new(self.hash_map),
        }
    }
}

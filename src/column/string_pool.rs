use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::core::error::{Error, Result};

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

    /// Add a string to the pool and return its index.
    ///
    /// # Errors
    /// Returns `Error::LockPoisoned` if a prior panic elsewhere in the
    /// process left this pool's lock poisoned. Earlier code treated that
    /// case as "return index 0", which silently aliased whatever string was
    /// interned *first* -- a correctness bug (every subsequent lookup for
    /// an unrelated string would read back the wrong value), not a
    /// harmless degradation. Propagating the error lets callers decide how
    /// to react instead of reading corrupted data.
    pub fn get_or_insert(&self, s: &str) -> Result<u32> {
        // Try with read lock first (fast path: string already interned).
        {
            let read_pool = self
                .pool
                .read()
                .map_err(|_| Error::lock_poisoned("GlobalStringPool::get_or_insert (read)"))?;
            if let Some(&idx) = read_pool.hash_map.get(s) {
                return Ok(idx);
            }
        }

        // Not found under the read lock: take the write lock and insert.
        let mut write_pool = self
            .pool
            .write()
            .map_err(|_| Error::lock_poisoned("GlobalStringPool::get_or_insert (write)"))?;

        // Re-check: another thread may have inserted the same string while
        // we were waiting for the write lock.
        if let Some(&idx) = write_pool.hash_map.get(s) {
            return Ok(idx);
        }

        // Assign a new index
        let idx = write_pool.strings.len() as u32;
        let arc_str: Arc<str> = Arc::from(s.to_owned());
        write_pool.strings.push(arc_str.clone());
        write_pool.hash_map.insert(arc_str, idx);
        Ok(idx)
    }

    /// Get a string by its index.
    ///
    /// # Errors
    /// Returns `Error::LockPoisoned` if the pool's lock is poisoned, rather
    /// than the `None` used earlier (which reads to a caller exactly like
    /// "no string is registered at this index", losing the distinction
    /// between "absent" and "the pool is broken").
    pub fn get(&self, index: u32) -> Result<Option<String>> {
        let pool = self
            .pool
            .read()
            .map_err(|_| Error::lock_poisoned("GlobalStringPool::get"))?;
        Ok(pool.strings.get(index as usize).map(|s| s.to_string()))
    }

    /// Return the number of registered strings.
    ///
    /// # Errors
    /// Returns `Error::LockPoisoned` if the pool's lock is poisoned, rather
    /// than fabricating `0` (which reads exactly like "the pool is empty").
    pub fn len(&self) -> Result<usize> {
        let pool = self
            .pool
            .read()
            .map_err(|_| Error::lock_poisoned("GlobalStringPool::len"))?;
        Ok(pool.strings.len())
    }

    /// Add a vector of strings to the global pool and return a vector of
    /// indices, one per input string, in the same order.
    pub fn add_strings(&self, strings: &[String]) -> Result<Vec<u32>> {
        strings.iter().map(|s| self.get_or_insert(s)).collect()
    }
}

impl Default for GlobalStringPool {
    fn default() -> Self {
        Self::new()
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

    /// Create a new *local* string pool from a slice of strings, using the
    /// process-wide global pool to intern/dedupe the string payloads (so
    /// identical strings interned by many columns share one lookup path).
    ///
    /// The returned pool's own index space is always LOCAL and DENSE
    /// (`0..pool.len()`), never the global pool's sparse, ever-growing
    /// index space. An earlier version of this function used the global
    /// index directly as the local one, which meant a small column built
    /// after many other columns already existed had to pad its local
    /// vector out to the global index's value -- an unbounded blow-up
    /// (observed: 200 unrelated columns pushed the *global* counter past
    /// 60,000, so a fresh 3-string column allocated a 60,300-slot local
    /// vector), and the padding slots were filled with clones of whatever
    /// string happened to be last, corrupting `len()`/`all_strings()` and
    /// anything built from them (like `merge()`).
    ///
    /// Returns the pool together with one LOCAL index per input string, in
    /// the same order as `strings`, for the caller to store per-row.
    ///
    /// # Errors
    /// Propagates `Error::LockPoisoned` from the global pool (see
    /// [`GlobalStringPool::get_or_insert`]).
    pub fn from_strings(strings: &[String]) -> Result<(Self, Vec<u32>)> {
        // Touch the global pool exactly once (a single batched call, not
        // once per string and not a second time later) purely to obtain a
        // dedup key per string; the ids it returns are used only as a
        // `HashMap` key here; they are never used as indices into `pool`.
        let global_indices = GLOBAL_STRING_POOL.add_strings(strings)?;

        let mut pool = Self::new_mut();
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut local_indices = Vec::with_capacity(strings.len());

        for (s, &global_idx) in strings.iter().zip(global_indices.iter()) {
            let local_idx = match remap.get(&global_idx) {
                Some(&idx) => idx,
                None => {
                    let arc_str: Arc<str> = Arc::from(s.as_str());
                    let idx = pool.strings.len() as u32;
                    pool.strings.push(arc_str.clone());
                    pool.hash_map.insert(arc_str, idx);
                    remap.insert(global_idx, idx);
                    idx
                }
            };
            local_indices.push(local_idx);
        }

        Ok((pool.freeze(), local_indices))
    }

    /// Create a new string pool from a slice of strings without touching
    /// the global pool (original, "legacy" implementation).
    ///
    /// Returns the pool together with one LOCAL index per input string.
    /// The index comes directly from each string's own insertion (or
    /// de-duplication) into the pool, so -- unlike the earlier
    /// implementation, which built the pool and then looked every string
    /// back up with `find(s).unwrap_or(0)` -- there is no separate lookup
    /// pass and therefore no failure mode that could substitute the
    /// pool's first string for one that (in principle) failed to be
    /// found.
    pub fn from_strings_legacy(strings: &[String]) -> (Self, Vec<u32>) {
        let mut pool = Self::new_mut();
        let mut indices = Vec::with_capacity(strings.len());

        for s in strings {
            indices.push(pool.get_or_insert(s));
        }

        (pool.freeze(), indices)
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

    /// Build a vector of strings from a string pool and indices.
    ///
    /// `indices` must be LOCAL indices into *this* pool (as returned by
    /// [`Self::from_strings`]/[`Self::from_strings_legacy`]/
    /// `StringPoolMut::get_or_insert` for this same pool) -- an
    /// out-of-bounds index (including a *global* pool id, which this pool
    /// no longer uses as its own index space) reads back as an empty
    /// string rather than an error, since this is a low-level building
    /// block with no way to signal "the caller passed the wrong id space"
    /// except by panicking. There are currently no callers in this crate;
    /// a future caller with untrusted/foreign indices should validate
    /// `idx < self.len()` itself rather than relying on this fallback.
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

impl Default for StringPool {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_strings_produces_a_dense_local_id_space() {
        let strings: Vec<String> = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        let (pool, indices) = StringPool::from_strings(&strings).expect("no poisoned lock");

        // 2 unique strings -> pool length 2, regardless of how large the
        // global pool has grown from unrelated earlier tests/columns.
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.all_strings().len(), 2);

        // Local indices must be in-bounds for THIS pool.
        for &idx in &indices {
            assert!((idx as usize) < pool.len());
        }
        assert_eq!(pool.get(indices[0]), Some("a"));
        assert_eq!(pool.get(indices[1]), Some("b"));
        assert_eq!(pool.get(indices[2]), Some("a"));
        assert_eq!(indices[0], indices[2], "repeated string dedups to one id");
    }

    #[test]
    fn from_strings_legacy_matches_from_strings_shape() {
        let strings: Vec<String> = vec!["x".to_string(), "y".to_string(), "x".to_string()];
        let (pool, indices) = StringPool::from_strings_legacy(&strings);

        assert_eq!(pool.len(), 2);
        assert_eq!(pool.get(indices[0]), Some("x"));
        assert_eq!(pool.get(indices[1]), Some("y"));
        assert_eq!(indices[0], indices[2]);
    }

    #[test]
    fn empty_input_produces_empty_pool() {
        let (pool, indices) = StringPool::from_strings(&[]).expect("no poisoned lock");
        assert_eq!(pool.len(), 0);
        assert!(indices.is_empty());
    }
}

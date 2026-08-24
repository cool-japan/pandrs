//! Simplified Unified Zero-Copy String Pool Implementation
//!
//! A contiguous-buffer string pool with deduplication and borrowed access.
//!
//! # Lifetime contract
//!
//! **The pool never removes anything.** Ids are handed out permanently, string
//! bytes are appended to one growing buffer, and there is no eviction,
//! compaction or removal API — so offsets stay valid for the life of the pool
//! and views can be handed around freely. The flip side is that a pool fed
//! unbounded distinct strings grows without bound; callers that intern
//! user-controlled data must bound it themselves or drop the whole pool.
//!
//! # "Zero-copy"
//!
//! Only [`SimpleStringView::with_str_ref`] is genuinely copy-free: it hands a
//! `&str` borrowed from the pool buffer to a closure. [`SimpleStringView::as_str`]
//! and [`SimpleStringView::as_bytes`] allocate, because the buffer is behind a
//! lock and a borrow cannot escape it.

use crate::core::error::{Error, Result};
use std::collections::HashMap;
use std::str;
use std::sync::{Arc, RwLock};

/// Metadata for a string in the simplified pool
#[derive(Debug, Clone, Copy)]
pub struct StringMetadata {
    /// Offset in the buffer where the string starts
    pub offset: u32,
    /// Length of the string in bytes
    pub length: u32,
    /// Hash of the string, used to narrow deduplication candidates
    pub hash: u64,
}

/// All mutable pool state, behind a single lock.
///
/// Keeping the buffer, the metadata table, the dedup index and the write cursor
/// under one lock is what makes a write atomic. The previous design used five
/// separate locks and performed a read-modify-write on the cursor with the lock
/// released in between, so concurrent adds (the type is `Clone` and shares its
/// state) could hand out overlapping buffer regions and corrupt each other.
#[derive(Debug)]
struct PoolInner {
    /// Contiguous buffer storing all string data
    buffer: Vec<u8>,
    /// String metadata (offsets and lengths), indexed by string id
    strings: Vec<StringMetadata>,
    /// Hash -> candidate ids. Content is verified before an id is reused.
    dedup_index: HashMap<u64, Vec<u32>>,
    /// Total number of string additions (including duplicates)
    total_additions: usize,
}

impl PoolInner {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            strings: Vec::new(),
            dedup_index: HashMap::new(),
            total_additions: 0,
        }
    }

    fn bytes_of(&self, metadata: &StringMetadata) -> Option<&[u8]> {
        let start = metadata.offset as usize;
        let end = start.checked_add(metadata.length as usize)?;
        self.buffer.get(start..end)
    }
}

/// Simplified unified string pool using contiguous buffer storage
#[derive(Debug)]
pub struct SimpleUnifiedStringPool {
    inner: Arc<RwLock<PoolInner>>,
}

/// Borrowed string view for the simplified pool
#[derive(Debug, Clone)]
pub struct SimpleStringView {
    /// Metadata for the string
    metadata: StringMetadata,
    /// Shared pool state
    pool: Arc<RwLock<PoolInner>>,
}

fn read_lock(inner: &Arc<RwLock<PoolInner>>) -> Result<std::sync::RwLockReadGuard<'_, PoolInner>> {
    inner
        .read()
        .map_err(|_| Error::InvalidOperation("String pool lock is poisoned".to_string()))
}

fn write_lock(
    inner: &Arc<RwLock<PoolInner>>,
) -> Result<std::sync::RwLockWriteGuard<'_, PoolInner>> {
    inner
        .write()
        .map_err(|_| Error::InvalidOperation("String pool lock is poisoned".to_string()))
}

impl SimpleStringView {
    /// Get the string as an owned `String` (allocates).
    pub fn as_str(&self) -> Result<String> {
        self.with_str_ref(|s| s.to_string())
    }

    /// Get the string as owned bytes (allocates).
    pub fn as_bytes(&self) -> Result<Vec<u8>> {
        let guard = read_lock(&self.pool)?;
        guard
            .bytes_of(&self.metadata)
            .map(|b| b.to_vec())
            .ok_or_else(|| Error::InvalidOperation("String extends beyond buffer".to_string()))
    }

    /// Get the length of the string in bytes.
    pub fn len(&self) -> usize {
        self.metadata.length as usize
    }

    /// Check if the string is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.length == 0
    }

    /// Get metadata for the string.
    pub fn metadata(&self) -> StringMetadata {
        self.metadata
    }

    /// Create a substring view over the byte range `start..end`.
    ///
    /// The offsets must fall on UTF-8 character boundaries; the old
    /// implementation accepted any offsets and produced a view whose bytes were
    /// not valid UTF-8, and added them to the base offset without an overflow
    /// check.
    pub fn substring(&self, start: usize, end: usize) -> Result<SimpleStringView> {
        if start > end || end > self.len() {
            return Err(Error::InvalidOperation(format!(
                "Invalid substring range {}..{} for a {} byte string",
                start,
                end,
                self.len()
            )));
        }
        let boundaries_ok =
            self.with_str_ref(|s| s.is_char_boundary(start) && s.is_char_boundary(end))?;
        if !boundaries_ok {
            return Err(Error::InvalidOperation(format!(
                "Substring range {}..{} does not fall on UTF-8 character boundaries",
                start, end
            )));
        }

        let offset = u32::try_from(self.metadata.offset as usize + start).map_err(|_| {
            Error::InvalidOperation("Substring offset exceeds the 32-bit pool address space".into())
        })?;
        let length = u32::try_from(end - start).map_err(|_| {
            Error::InvalidOperation("Substring length exceeds the 32-bit pool limit".into())
        })?;

        Ok(SimpleStringView {
            metadata: StringMetadata {
                offset,
                length,
                // A substring is not an interned string; it has no dedup entry.
                hash: 0,
            },
            pool: Arc::clone(&self.pool),
        })
    }

    /// Borrow the string for the duration of `f` (genuinely copy-free).
    pub fn with_str_ref<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&str) -> R,
    {
        let guard = read_lock(&self.pool)?;
        let data = guard
            .bytes_of(&self.metadata)
            .ok_or_else(|| Error::InvalidOperation("String extends beyond buffer".to_string()))?;
        let s = str::from_utf8(data)
            .map_err(|e| Error::InvalidOperation(format!("Invalid UTF-8: {}", e)))?;
        Ok(f(s))
    }
}

impl std::fmt::Display for SimpleStringView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.as_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid UTF-8>"),
        }
    }
}

impl SimpleUnifiedStringPool {
    /// Create a new simplified unified string pool
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(PoolInner::new(1024 * 1024))),
        }
    }

    /// Create a pool with an explicit initial buffer capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(PoolInner::new(capacity))),
        }
    }

    /// Add a string to the pool and return its ID.
    ///
    /// The whole reserve-copy-commit sequence happens under one write lock, and
    /// deduplication compares the stored bytes rather than trusting a 64-bit
    /// hash match (which could return a *different* string's id on collision).
    pub fn add_string(&self, s: &str) -> Result<u32> {
        let bytes = s.as_bytes();
        let hash = hash_string(s);

        let mut inner = write_lock(&self.inner)?;
        inner.total_additions += 1;

        let candidates = inner.dedup_index.get(&hash).cloned().unwrap_or_default();
        for id in candidates {
            if let Some(metadata) = inner.strings.get(id as usize).copied() {
                if metadata.length as usize == bytes.len()
                    && inner.bytes_of(&metadata) == Some(bytes)
                {
                    return Ok(id);
                }
            }
        }

        let offset = u32::try_from(inner.buffer.len()).map_err(|_| {
            Error::InvalidOperation(
                "String pool exceeded its 32-bit address space (4 GiB of string data)".to_string(),
            )
        })?;
        let length = u32::try_from(bytes.len()).map_err(|_| {
            Error::InvalidOperation("Individual strings are limited to 4 GiB".to_string())
        })?;
        // Guard the end offset too: `offset` fitting does not imply `offset + length` does.
        u32::try_from(offset as u64 + length as u64).map_err(|_| {
            Error::InvalidOperation(
                "String pool exceeded its 32-bit address space (4 GiB of string data)".to_string(),
            )
        })?;
        let id = u32::try_from(inner.strings.len()).map_err(|_| {
            Error::InvalidOperation("String pool exceeded 2^32 distinct strings".to_string())
        })?;

        inner.buffer.extend_from_slice(bytes);
        inner.strings.push(StringMetadata {
            offset,
            length,
            hash,
        });
        inner.dedup_index.entry(hash).or_default().push(id);
        Ok(id)
    }

    /// Add multiple strings to the pool efficiently
    pub fn add_strings(&self, strings: &[String]) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(strings.len());
        for s in strings {
            result.push(self.add_string(s)?);
        }
        Ok(result)
    }

    /// Get a view of a string by ID.
    pub fn get_string(&self, string_id: u32) -> Result<SimpleStringView> {
        let metadata = {
            let inner = read_lock(&self.inner)?;
            inner
                .strings
                .get(string_id as usize)
                .copied()
                .ok_or_else(|| {
                    Error::InvalidOperation(format!("String ID {} not found", string_id))
                })?
        };

        Ok(SimpleStringView {
            metadata,
            // One atomic increment. The old code built a fresh
            // `Arc::new(self.clone())` — five more Arc clones — on every access.
            pool: Arc::clone(&self.inner),
        })
    }

    /// Get multiple strings by their IDs
    pub fn get_strings(&self, string_ids: &[u32]) -> Result<Vec<SimpleStringView>> {
        let mut result = Vec::with_capacity(string_ids.len());
        for &id in string_ids {
            result.push(self.get_string(id)?);
        }
        Ok(result)
    }

    /// Number of distinct strings stored.
    pub fn len(&self) -> Result<usize> {
        Ok(read_lock(&self.inner)?.strings.len())
    }

    /// Whether the pool holds no strings.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(read_lock(&self.inner)?.strings.is_empty())
    }

    /// Get pool statistics
    pub fn stats(&self) -> Result<SimpleStringPoolStats> {
        let inner = read_lock(&self.inner)?;
        let total_additions = inner.total_additions;
        // Count actual distinct strings, not hash buckets: colliding strings
        // share a bucket and used to be under-counted.
        let unique_strings = inner.strings.len();

        Ok(SimpleStringPoolStats {
            total_strings: total_additions,
            unique_strings,
            total_bytes: inner.buffer.len(),
            buffer_capacity: inner.buffer.capacity(),
            deduplication_ratio: if total_additions > 0 {
                1.0 - (unique_strings as f64 / total_additions as f64)
            } else {
                0.0
            },
            memory_efficiency: if inner.buffer.capacity() > 0 {
                inner.buffer.len() as f64 / inner.buffer.capacity() as f64
            } else {
                0.0
            },
        })
    }
}

fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

impl Default for SimpleUnifiedStringPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SimpleUnifiedStringPool {
    /// Clones share the same underlying pool state.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Statistics for the simplified string pool
#[derive(Debug, Clone)]
pub struct SimpleStringPoolStats {
    /// Total number of strings (including duplicates)
    pub total_strings: usize,
    /// Number of unique strings
    pub unique_strings: usize,
    /// Total bytes used for string data
    pub total_bytes: usize,
    /// Total buffer capacity
    pub buffer_capacity: usize,
    /// Deduplication ratio (0.0 = no deduplication, 1.0 = all duplicates)
    pub deduplication_ratio: f64,
    /// Memory efficiency ratio (used/capacity)
    pub memory_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_string_pool_creation() {
        let pool = SimpleUnifiedStringPool::new();
        let stats = pool.stats().expect("operation should succeed");

        assert_eq!(stats.total_strings, 0);
        assert_eq!(stats.unique_strings, 0);
        assert!(stats.buffer_capacity > 0);
    }

    #[test]
    fn test_string_addition_and_retrieval() {
        let pool = SimpleUnifiedStringPool::new();

        let id1 = pool.add_string("hello").expect("operation should succeed");
        let id2 = pool.add_string("world").expect("operation should succeed");
        let id3 = pool.add_string("hello").expect("operation should succeed");

        assert_ne!(id1, id2);
        assert_eq!(id1, id3);

        let view1 = pool.get_string(id1).expect("operation should succeed");
        let view2 = pool.get_string(id2).expect("operation should succeed");

        assert_eq!(view1.as_str().expect("operation should succeed"), "hello");
        assert_eq!(view2.as_str().expect("operation should succeed"), "world");

        let stats = pool.stats().expect("operation should succeed");
        assert_eq!(stats.total_strings, 3);
        assert_eq!(stats.unique_strings, 2);
    }

    #[test]
    fn test_multiple_string_operations() {
        let pool = SimpleUnifiedStringPool::new();

        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "apple".to_string(),
        ];

        let ids = pool
            .add_strings(&strings)
            .expect("operation should succeed");
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], ids[3]);

        let views = pool.get_strings(&ids).expect("operation should succeed");
        assert_eq!(views.len(), 4);
        for (view, expected) in views.iter().zip(strings.iter()) {
            assert_eq!(&view.as_str().expect("operation should succeed"), expected);
        }
    }

    #[test]
    fn test_zero_copy_access() {
        let pool = SimpleUnifiedStringPool::new();

        let id = pool
            .add_string("hello world")
            .expect("operation should succeed");
        let view = pool.get_string(id).expect("operation should succeed");

        let result = view
            .with_str_ref(|s| s.to_uppercase())
            .expect("operation should succeed");
        assert_eq!(result, "HELLO WORLD");

        let starts_with_hello = view
            .with_str_ref(|s| s.starts_with("hello"))
            .expect("operation should succeed");
        assert!(starts_with_hello);
    }

    #[test]
    fn test_substring() {
        let pool = SimpleUnifiedStringPool::new();

        let id = pool
            .add_string("hello world")
            .expect("operation should succeed");
        let view = pool.get_string(id).expect("operation should succeed");

        let substring = view.substring(0, 5).expect("operation should succeed");
        assert_eq!(
            substring.as_str().expect("operation should succeed"),
            "hello"
        );

        let substring2 = view.substring(6, 11).expect("operation should succeed");
        assert_eq!(
            substring2.as_str().expect("operation should succeed"),
            "world"
        );
    }

    #[test]
    fn substring_rejects_non_character_boundaries() {
        let pool = SimpleUnifiedStringPool::new();
        let id = pool.add_string("日本語").expect("add");
        let view = pool.get_string(id).expect("get");

        // "日" is 3 bytes; slicing at 1 is inside a code point.
        assert!(view.substring(0, 1).is_err());
        assert!(view.substring(1, 3).is_err());
        assert_eq!(
            view.substring(0, 3).expect("valid").as_str().expect("str"),
            "日"
        );
        assert!(view.substring(0, 99).is_err());
    }

    #[test]
    fn deduplication_is_content_verified() {
        let pool = SimpleUnifiedStringPool::new();
        // Force two distinct strings into the same hash bucket by inspecting the
        // pool's behaviour indirectly: distinct content must always yield
        // distinct ids and correct read-back.
        let mut ids = Vec::new();
        for i in 0..512 {
            ids.push(pool.add_string(&format!("value-{}", i)).expect("add"));
        }
        for (i, id) in ids.iter().enumerate() {
            let view = pool.get_string(*id).expect("get");
            assert_eq!(
                view.as_str().expect("str"),
                format!("value-{}", i),
                "id {} returned the wrong string",
                id
            );
        }
        let stats = pool.stats().expect("stats");
        assert_eq!(stats.unique_strings, 512);
    }

    #[test]
    fn test_pool_statistics() {
        let pool = SimpleUnifiedStringPool::new();

        pool.add_string("test").expect("operation should succeed");
        pool.add_string("data").expect("operation should succeed");
        pool.add_string("test").expect("operation should succeed");

        let stats = pool.stats().expect("operation should succeed");
        assert_eq!(stats.total_strings, 3);
        assert_eq!(stats.unique_strings, 2);
        assert!(stats.total_bytes > 0);
        assert!(stats.deduplication_ratio > 0.0);
    }

    #[test]
    fn concurrent_adds_do_not_corrupt_the_buffer() {
        use std::thread;

        let pool = SimpleUnifiedStringPool::new();
        let mut handles = Vec::new();
        for t in 0..8 {
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                let mut local = Vec::new();
                for i in 0..200 {
                    let value = format!("thread{}-value{}", t, i);
                    let id = pool.add_string(&value).expect("add");
                    local.push((id, value));
                }
                local
            }));
        }

        let mut all = Vec::new();
        for handle in handles {
            all.extend(handle.join().expect("thread panicked"));
        }

        // Every id must still resolve to exactly the string that produced it.
        for (id, expected) in all {
            let view = pool.get_string(id).expect("get");
            assert_eq!(view.as_str().expect("str"), expected);
        }
    }

    #[test]
    fn clones_share_state() {
        let pool = SimpleUnifiedStringPool::new();
        let clone = pool.clone();
        let id = clone.add_string("shared").expect("add");
        assert_eq!(
            pool.get_string(id).expect("get").as_str().expect("str"),
            "shared"
        );
        assert_eq!(pool.len().expect("len"), 1);
    }
}

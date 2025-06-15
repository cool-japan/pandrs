//! Simplified Unified Zero-Copy String Pool Implementation
//!
//! This module provides a simplified version of the unified string pool that focuses
//! on the core zero-copy string optimization without depending on the complex
//! zero-copy infrastructure that has Send/Sync issues.

use crate::core::error::{Error, Result};
use std::collections::HashMap;
use std::str;
use std::sync::{Arc, RwLock};

/// Simplified unified string pool using contiguous buffer storage
#[derive(Debug)]
pub struct SimpleUnifiedStringPool {
    /// Contiguous buffer storing all string data
    buffer: Arc<RwLock<Vec<u8>>>,
    /// String metadata (offsets and lengths)
    strings: Arc<RwLock<Vec<StringMetadata>>>,
    /// Hash table for string deduplication (hash -> string_id)
    dedup_index: Arc<RwLock<HashMap<u64, u32>>>,
    /// Current buffer position
    current_pos: Arc<RwLock<usize>>,
    /// Total number of string additions (including duplicates)
    total_additions: Arc<RwLock<usize>>,
}

/// Metadata for a string in the simplified pool
#[derive(Debug, Clone, Copy)]
pub struct StringMetadata {
    /// Offset in the buffer where string starts
    pub offset: u32,
    /// Length of the string in bytes
    pub length: u32,
    /// Hash of the string for deduplication
    pub hash: u64,
}

/// Zero-copy string view for the simplified pool
#[derive(Debug, Clone)]
pub struct SimpleStringView {
    /// Metadata for the string
    metadata: StringMetadata,
    /// Pool reference for data access
    pool_ref: Arc<SimpleUnifiedStringPool>,
}

impl SimpleStringView {
    /// Get the string as a String (allocates)
    pub fn as_str(&self) -> Result<String> {
        let buffer =
            self.pool_ref.buffer.read().map_err(|_| {
                Error::InvalidOperation("Failed to acquire buffer lock".to_string())
            })?;

        let start = self.metadata.offset as usize;
        let end = start + self.metadata.length as usize;

        if end > buffer.len() {
            return Err(Error::InvalidOperation(
                "String extends beyond buffer".to_string(),
            ));
        }

        let data = &buffer[start..end];
        let s = str::from_utf8(data)
            .map_err(|e| Error::InvalidOperation(format!("Invalid UTF-8: {}", e)))?;

        Ok(s.to_string())
    }

    /// Get the string as bytes (allocates)
    pub fn as_bytes(&self) -> Result<Vec<u8>> {
        let buffer =
            self.pool_ref.buffer.read().map_err(|_| {
                Error::InvalidOperation("Failed to acquire buffer lock".to_string())
            })?;

        let start = self.metadata.offset as usize;
        let end = start + self.metadata.length as usize;

        if end > buffer.len() {
            return Err(Error::InvalidOperation(
                "String extends beyond buffer".to_string(),
            ));
        }

        Ok(buffer[start..end].to_vec())
    }

    /// Get the length of the string
    pub fn len(&self) -> usize {
        self.metadata.length as usize
    }

    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.metadata.length == 0
    }

    /// Get metadata for the string
    pub fn metadata(&self) -> StringMetadata {
        self.metadata
    }

    /// Create a substring view
    pub fn substring(&self, start: usize, end: usize) -> Result<SimpleStringView> {
        if start > self.len() || end > self.len() || start > end {
            return Err(Error::InvalidOperation(
                "Invalid substring range".to_string(),
            ));
        }

        let sub_metadata = StringMetadata {
            offset: self.metadata.offset + start as u32,
            length: (end - start) as u32,
            hash: 0, // Will be computed if needed
        };

        Ok(SimpleStringView {
            metadata: sub_metadata,
            pool_ref: Arc::clone(&self.pool_ref),
        })
    }

    /// Get a string reference with a specific lifetime (for temporary use)
    pub fn with_str_ref<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&str) -> R,
    {
        let buffer =
            self.pool_ref.buffer.read().map_err(|_| {
                Error::InvalidOperation("Failed to acquire buffer lock".to_string())
            })?;

        let start = self.metadata.offset as usize;
        let end = start + self.metadata.length as usize;

        if end > buffer.len() {
            return Err(Error::InvalidOperation(
                "String extends beyond buffer".to_string(),
            ));
        }

        let data = &buffer[start..end];
        let s = str::from_utf8(data)
            .map_err(|e| Error::InvalidOperation(format!("Invalid UTF-8: {}", e)))?;

        Ok(f(s))
    }
}

impl std::fmt::Display for SimpleStringView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.as_str()
                .unwrap_or_else(|_| "<invalid UTF-8>".to_string())
        )
    }
}

impl SimpleUnifiedStringPool {
    /// Create a new simplified unified string pool
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(RwLock::new(Vec::with_capacity(1024 * 1024))), // 1MB initial capacity
            strings: Arc::new(RwLock::new(Vec::new())),
            dedup_index: Arc::new(RwLock::new(HashMap::new())),
            current_pos: Arc::new(RwLock::new(0)),
            total_additions: Arc::new(RwLock::new(0)),
        }
    }

    /// Add a string to the pool and return its ID
    pub fn add_string(&self, s: &str) -> Result<u32> {
        let bytes = s.as_bytes();
        let hash = self.hash_string(s);

        // Increment total additions count
        {
            let mut total_additions = self.total_additions.write().map_err(|_| {
                Error::InvalidOperation("Failed to acquire total_additions lock".to_string())
            })?;
            *total_additions += 1;
        }

        // Check for existing string
        {
            let dedup_index = self.dedup_index.read().map_err(|_| {
                Error::InvalidOperation("Failed to acquire dedup index lock".to_string())
            })?;

            if let Some(&existing_id) = dedup_index.get(&hash) {
                return Ok(existing_id);
            }
        }

        // Add new string to pool
        self.add_new_string(bytes, hash)
    }

    /// Add multiple strings to the pool efficiently
    pub fn add_strings(&self, strings: &[String]) -> Result<Vec<u32>> {
        let mut result = Vec::with_capacity(strings.len());

        for s in strings {
            result.push(self.add_string(s)?);
        }

        Ok(result)
    }

    /// Get a zero-copy view of a string by ID
    pub fn get_string(&self, string_id: u32) -> Result<SimpleStringView> {
        let strings = self
            .strings
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire strings lock".to_string()))?;

        let metadata = strings
            .get(string_id as usize)
            .ok_or_else(|| Error::InvalidOperation(format!("String ID {} not found", string_id)))?
            .clone();

        Ok(SimpleStringView {
            metadata,
            pool_ref: Arc::new(self.clone()),
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

    /// Get pool statistics
    pub fn stats(&self) -> Result<SimpleStringPoolStats> {
        let strings = self
            .strings
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire strings lock".to_string()))?;

        let dedup_index = self.dedup_index.read().map_err(|_| {
            Error::InvalidOperation("Failed to acquire dedup index lock".to_string())
        })?;

        let current_pos = self.current_pos.read().map_err(|_| {
            Error::InvalidOperation("Failed to acquire current_pos lock".to_string())
        })?;

        let buffer = self
            .buffer
            .read()
            .map_err(|_| Error::InvalidOperation("Failed to acquire buffer lock".to_string()))?;

        let total_additions = self.total_additions.read().map_err(|_| {
            Error::InvalidOperation("Failed to acquire total_additions lock".to_string())
        })?;

        Ok(SimpleStringPoolStats {
            total_strings: *total_additions, // Total number of string addition calls
            unique_strings: dedup_index.len(), // Number of unique strings stored
            total_bytes: *current_pos,
            buffer_capacity: buffer.capacity(),
            deduplication_ratio: if *total_additions > 0 {
                1.0 - (dedup_index.len() as f64 / *total_additions as f64)
            } else {
                0.0
            },
            memory_efficiency: if buffer.capacity() > 0 {
                *current_pos as f64 / buffer.capacity() as f64
            } else {
                0.0
            },
        })
    }

    /// Private method to add a new string to the pool
    fn add_new_string(&self, bytes: &[u8], hash: u64) -> Result<u32> {
        // Get current position
        let offset = {
            let current_pos = self.current_pos.read().map_err(|_| {
                Error::InvalidOperation("Failed to acquire current_pos lock".to_string())
            })?;
            *current_pos
        };

        // Expand buffer if needed
        {
            let mut buffer = self.buffer.write().map_err(|_| {
                Error::InvalidOperation("Failed to acquire buffer write lock".to_string())
            })?;

            // Ensure we have enough capacity
            let required_capacity = offset + bytes.len();
            if buffer.len() < required_capacity {
                buffer.resize(required_capacity, 0);
            }

            // Copy string data to buffer
            buffer[offset..offset + bytes.len()].copy_from_slice(bytes);
        }

        // Create metadata
        let metadata = StringMetadata {
            offset: offset as u32,
            length: bytes.len() as u32,
            hash,
        };

        // Add to strings vector
        let string_id = {
            let mut strings = self.strings.write().map_err(|_| {
                Error::InvalidOperation("Failed to acquire strings write lock".to_string())
            })?;

            let id = strings.len() as u32;
            strings.push(metadata);
            id
        };

        // Add to deduplication index
        {
            let mut dedup_index = self.dedup_index.write().map_err(|_| {
                Error::InvalidOperation("Failed to acquire dedup index write lock".to_string())
            })?;

            dedup_index.insert(hash, string_id);
        }

        // Update current position
        {
            let mut current_pos = self.current_pos.write().map_err(|_| {
                Error::InvalidOperation("Failed to acquire current_pos write lock".to_string())
            })?;

            *current_pos = offset + bytes.len();
        }

        Ok(string_id)
    }

    /// Hash a string for deduplication
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

// Implement Clone manually
impl Clone for SimpleUnifiedStringPool {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            strings: Arc::clone(&self.strings),
            dedup_index: Arc::clone(&self.dedup_index),
            current_pos: Arc::clone(&self.current_pos),
            total_additions: Arc::clone(&self.total_additions),
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
        let stats = pool.stats().unwrap();

        assert_eq!(stats.total_strings, 0);
        assert_eq!(stats.unique_strings, 0);
        assert!(stats.buffer_capacity > 0);
    }

    #[test]
    fn test_string_addition_and_retrieval() {
        let pool = SimpleUnifiedStringPool::new();

        let id1 = pool.add_string("hello").unwrap();
        let id2 = pool.add_string("world").unwrap();
        let id3 = pool.add_string("hello").unwrap(); // Duplicate

        assert_ne!(id1, id2);
        assert_eq!(id1, id3); // Should be deduplicated

        let view1 = pool.get_string(id1).unwrap();
        let view2 = pool.get_string(id2).unwrap();

        assert_eq!(view1.as_str().unwrap(), "hello");
        assert_eq!(view2.as_str().unwrap(), "world");

        let stats = pool.stats().unwrap();
        assert_eq!(stats.total_strings, 3); // Total additions including duplicates
        assert_eq!(stats.unique_strings, 2); // Only unique strings counted
    }

    #[test]
    fn test_multiple_string_operations() {
        let pool = SimpleUnifiedStringPool::new();

        let strings = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "apple".to_string(), // Duplicate
        ];

        let ids = pool.add_strings(&strings).unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], ids[3]); // Duplicates should have same ID

        let views = pool.get_strings(&ids).unwrap();
        assert_eq!(views.len(), 4);
        assert_eq!(views[0].as_str().unwrap(), "apple");
        assert_eq!(views[1].as_str().unwrap(), "banana");
        assert_eq!(views[2].as_str().unwrap(), "cherry");
        assert_eq!(views[3].as_str().unwrap(), "apple");
    }

    #[test]
    fn test_zero_copy_access() {
        let pool = SimpleUnifiedStringPool::new();

        let id = pool.add_string("hello world").unwrap();
        let view = pool.get_string(id).unwrap();

        // Test with_str_ref for zero-copy access
        let result = view.with_str_ref(|s| s.to_uppercase()).unwrap();
        assert_eq!(result, "HELLO WORLD");

        let starts_with_hello = view.with_str_ref(|s| s.starts_with("hello")).unwrap();
        assert!(starts_with_hello);
    }

    #[test]
    fn test_substring() {
        let pool = SimpleUnifiedStringPool::new();

        let id = pool.add_string("hello world").unwrap();
        let view = pool.get_string(id).unwrap();

        let substring = view.substring(0, 5).unwrap();
        assert_eq!(substring.as_str().unwrap(), "hello");

        let substring2 = view.substring(6, 11).unwrap();
        assert_eq!(substring2.as_str().unwrap(), "world");
    }

    #[test]
    fn test_pool_statistics() {
        let pool = SimpleUnifiedStringPool::new();

        pool.add_string("test").unwrap();
        pool.add_string("data").unwrap();
        pool.add_string("test").unwrap(); // Duplicate

        let stats = pool.stats().unwrap();
        assert_eq!(stats.total_strings, 3); // 3 total additions (including 1 duplicate)
        assert_eq!(stats.unique_strings, 2); // 2 unique strings
        assert!(stats.total_bytes > 0);
        assert!(stats.deduplication_ratio > 0.0); // Should have some deduplication
    }
}

//! Zero-Copy String Column Implementation
//!
//! This module provides a string column implementation that leverages the unified
//! zero-copy string pool for optimal memory efficiency and performance.
//!
//! Key features:
//! - Zero-copy string access and manipulation
//! - Unified memory management with cache-aware allocation
//! - Automatic string deduplication and interning
//! - Memory-efficient storage with offset-based indexing
//! - Thread-safe concurrent access

use std::any::Any;
use std::sync::Arc;

use crate::core::column::{Column, ColumnTrait, ColumnType};
use crate::core::error::{Error, Result};
use crate::storage::unified_string_pool::{UnifiedStringPool, UnifiedStringView, UnifiedStringPoolConfig, StringMetadata};

/// Zero-copy string column that uses the unified string pool
#[derive(Debug, Clone)]
pub struct ZeroCopyStringColumn {
    /// Reference to the unified string pool
    pool: Arc<UnifiedStringPool>,
    /// String IDs for each row
    string_ids: Arc<[u32]>,
    /// Null mask for handling NULL values
    null_mask: Option<Arc<[u8]>>,
    /// Column name
    name: Option<String>,
}

impl ZeroCopyStringColumn {
    /// Create a new zero-copy string column from strings
    pub fn new(data: Vec<String>) -> Result<Self> {
        let pool = Arc::new(UnifiedStringPool::with_default_config()?);
        let string_ids = pool.add_strings(&data)?;
        
        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask: None,
            name: None,
        })
    }
    
    /// Create a zero-copy string column with custom pool configuration
    pub fn with_config(data: Vec<String>, config: UnifiedStringPoolConfig) -> Result<Self> {
        let pool = Arc::new(UnifiedStringPool::new(config)?);
        let string_ids = pool.add_strings(&data)?;
        
        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask: None,
            name: None,
        })
    }
    
    /// Create a zero-copy string column with a shared pool
    pub fn with_shared_pool(data: Vec<String>, pool: Arc<UnifiedStringPool>) -> Result<Self> {
        let string_ids = pool.add_strings(&data)?;
        
        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask: None,
            name: None,
        })
    }
    
    /// Create a zero-copy string column with name
    pub fn with_name(data: Vec<String>, name: impl Into<String>) -> Result<Self> {
        let mut column = Self::new(data)?;
        column.name = Some(name.into());
        Ok(column)
    }
    
    /// Create a zero-copy string column with null values
    pub fn with_nulls(data: Vec<String>, nulls: Vec<bool>) -> Result<Self> {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };
        
        let mut column = Self::new(data)?;
        column.null_mask = null_mask;
        Ok(column)
    }
    
    /// Set the column name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }
    
    /// Get the column name
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    /// Get a zero-copy string view at the specified index
    pub fn get_view(&self, index: usize) -> Result<Option<UnifiedStringView>> {
        if index >= self.string_ids.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.string_ids.len(),
            });
        }
        
        // Check for NULL value
        if let Some(ref mask) = self.null_mask {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            if byte_idx < mask.len() && (mask[byte_idx] & (1 << bit_idx)) != 0 {
                return Ok(None);
            }
        }
        
        let string_id = self.string_ids[index];
        let view = self.pool.get_string(string_id)?;
        Ok(Some(view))
    }
    
    /// Get string at the specified index (allocates a new String)
    pub fn get(&self, index: usize) -> Result<Option<String>> {
        match self.get_view(index)? {
            Some(view) => Ok(Some(view.as_str()?.to_string())),
            None => Ok(None),
        }
    }
    
    /// Get multiple zero-copy string views
    pub fn get_views(&self, indices: &[usize]) -> Result<Vec<Option<UnifiedStringView>>> {
        let mut result = Vec::with_capacity(indices.len());
        
        for &index in indices {
            result.push(self.get_view(index)?);
        }
        
        Ok(result)
    }
    
    /// Convert to a vector of strings (allocates new strings)
    pub fn to_strings(&self) -> Result<Vec<Option<String>>> {
        let mut result = Vec::with_capacity(self.string_ids.len());
        
        for i in 0..self.string_ids.len() {
            result.push(self.get(i)?);
        }
        
        Ok(result)
    }
    
    /// Get all zero-copy views (for efficient iteration)
    pub fn iter_views(&self) -> Result<ZeroCopyStringColumnIterator> {
        Ok(ZeroCopyStringColumnIterator {
            column: self,
            current_index: 0,
        })
    }
    
    /// Apply a function to each string view (zero-copy)
    pub fn map_views<F, R>(&self, mut f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<UnifiedStringView>) -> R,
    {
        let mut result = Vec::with_capacity(self.string_ids.len());
        
        for i in 0..self.string_ids.len() {
            let view = self.get_view(i)?;
            result.push(f(view));
        }
        
        Ok(result)
    }
    
    /// Filter strings using a zero-copy predicate
    pub fn filter_views<F>(&self, mut predicate: F) -> Result<Vec<usize>>
    where
        F: FnMut(&str) -> bool,
    {
        let mut result = Vec::new();
        
        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                if predicate(view.as_str()?) {
                    result.push(i);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Check if a string exists in the column (using zero-copy)
    pub fn contains(&self, target: &str) -> Result<bool> {
        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                if view.as_str()? == target {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
    
    /// Count occurrences of a string in the column
    pub fn count_occurrences(&self, target: &str) -> Result<usize> {
        let mut count = 0;
        
        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                if view.as_str()? == target {
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
    
    /// Get unique strings (using zero-copy)
    pub fn unique_views(&self) -> Result<Vec<UnifiedStringView>> {
        let mut unique_views = Vec::new();
        let mut seen_hashes = std::collections::HashSet::new();
        
        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                let metadata = view.metadata();
                if seen_hashes.insert(metadata.hash) {
                    unique_views.push(view);
                }
            }
        }
        
        Ok(unique_views)
    }
    
    /// Get string lengths efficiently
    pub fn string_lengths(&self) -> Result<Vec<Option<usize>>> {
        let mut lengths = Vec::with_capacity(self.string_ids.len());
        
        for i in 0..self.string_ids.len() {
            match self.get_view(i)? {
                Some(view) => lengths.push(Some(view.len())),
                None => lengths.push(None),
            }
        }
        
        Ok(lengths)
    }
    
    /// Get the underlying string pool
    pub fn pool(&self) -> &Arc<UnifiedStringPool> {
        &self.pool
    }
    
    /// Get pool statistics
    pub fn pool_stats(&self) -> Result<crate::storage::unified_string_pool::UnifiedStringPoolStats> {
        self.pool.stats()
    }
    
    /// Create a new column with the same pool but different string IDs
    pub fn with_string_ids(&self, string_ids: Vec<u32>) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            string_ids: string_ids.into(),
            null_mask: self.null_mask.clone(),
            name: self.name.clone(),
        }
    }
    
    /// Concatenate strings with another column (zero-copy where possible)
    pub fn concat_with(&self, other: &ZeroCopyStringColumn, separator: &str) -> Result<ZeroCopyStringColumn> {
        if self.string_ids.len() != other.string_ids.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.string_ids.len(),
                found: other.string_ids.len(),
            });
        }
        
        let mut new_strings = Vec::with_capacity(self.string_ids.len());
        
        for i in 0..self.string_ids.len() {
            let left = self.get_view(i)?;
            let right = other.get_view(i)?;
            
            match (left, right) {
                (Some(left_view), Some(right_view)) => {
                    let concatenated = format!("{}{}{}", 
                        left_view.as_str()?, 
                        separator, 
                        right_view.as_str()?
                    );
                    new_strings.push(concatenated);
                },
                (Some(left_view), None) => {
                    new_strings.push(left_view.as_str()?.to_string());
                },
                (None, Some(right_view)) => {
                    new_strings.push(right_view.as_str()?.to_string());
                },
                (None, None) => {
                    new_strings.push(String::new()); // Empty string for NULL + NULL
                },
            }
        }
        
        ZeroCopyStringColumn::with_shared_pool(new_strings, Arc::clone(&self.pool))
    }
}

impl ColumnTrait for ZeroCopyStringColumn {
    fn len(&self) -> usize {
        self.string_ids.len()
    }
    
    fn is_empty(&self) -> bool {
        self.string_ids.is_empty()
    }
    
    fn column_type(&self) -> ColumnType {
        ColumnType::String
    }
    
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    
    fn clone_column(&self) -> Column {
        // For now, convert to regular StringColumn for compatibility
        // In a full implementation, we'd add ZeroCopyStringColumn to the Column enum
        let strings = self.to_strings().unwrap_or_default();
        let string_values: Vec<String> = strings.into_iter()
            .map(|opt| opt.unwrap_or_default())
            .collect();
        Column::String(crate::column::StringColumn::new(string_values))
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Iterator for zero-copy string column views
pub struct ZeroCopyStringColumnIterator<'a> {
    column: &'a ZeroCopyStringColumn,
    current_index: usize,
}

impl<'a> Iterator for ZeroCopyStringColumnIterator<'a> {
    type Item = Result<Option<UnifiedStringView>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.column.string_ids.len() {
            return None;
        }
        
        let result = self.column.get_view(self.current_index);
        self.current_index += 1;
        
        Some(result)
    }
}

/// Zero-copy string operations trait
pub trait ZeroCopyStringOps {
    /// Transform strings using zero-copy views
    fn transform_zero_copy<F, R>(&self, f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<UnifiedStringView>) -> R;
    
    /// Get substring views (zero-copy)
    fn substring_views(&self, start: usize, end: usize) -> Result<ZeroCopyStringColumn>;
    
    /// Case conversion with minimal allocation
    fn to_lowercase_optimized(&self) -> Result<ZeroCopyStringColumn>;
    
    /// Case conversion with minimal allocation
    fn to_uppercase_optimized(&self) -> Result<ZeroCopyStringColumn>;
}

impl ZeroCopyStringOps for ZeroCopyStringColumn {
    fn transform_zero_copy<F, R>(&self, f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<UnifiedStringView>) -> R,
    {
        self.map_views(f)
    }
    
    fn substring_views(&self, start: usize, end: usize) -> Result<ZeroCopyStringColumn> {
        let mut new_strings = Vec::with_capacity(self.string_ids.len());
        
        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                let str_len = view.len();
                let actual_start = start.min(str_len);
                let actual_end = end.min(str_len);
                
                if actual_start < actual_end {
                    let substring = view.substring(actual_start..actual_end)?;
                    new_strings.push(substring.as_str()?.to_string());
                } else {
                    new_strings.push(String::new());
                }
            } else {
                new_strings.push(String::new());
            }
        }
        
        ZeroCopyStringColumn::with_shared_pool(new_strings, Arc::clone(&self.pool))
    }
    
    fn to_lowercase_optimized(&self) -> Result<ZeroCopyStringColumn> {
        let new_strings = self.map_views(|view_opt| {
            match view_opt {
                Some(view) => view.as_str().unwrap_or("").to_lowercase(),
                None => String::new(),
            }
        })?;
        
        ZeroCopyStringColumn::with_shared_pool(new_strings, Arc::clone(&self.pool))
    }
    
    fn to_uppercase_optimized(&self) -> Result<ZeroCopyStringColumn> {
        let new_strings = self.map_views(|view_opt| {
            match view_opt {
                Some(view) => view.as_str().unwrap_or("").to_uppercase(),
                None => String::new(),
            }
        })?;
        
        ZeroCopyStringColumn::with_shared_pool(new_strings, Arc::clone(&self.pool))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_copy_string_column_creation() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];
        
        let column = ZeroCopyStringColumn::new(data.clone()).unwrap();
        assert_eq!(column.len(), 3);
        assert!(!column.is_empty());
        assert_eq!(column.column_type(), ColumnType::String);
        
        // Test data retrieval
        assert_eq!(column.get(0).unwrap().unwrap(), "hello");
        assert_eq!(column.get(1).unwrap().unwrap(), "world");
        assert_eq!(column.get(2).unwrap().unwrap(), "test");
    }
    
    #[test]
    fn test_zero_copy_views() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
        ];
        
        let column = ZeroCopyStringColumn::new(data).unwrap();
        
        let view1 = column.get_view(0).unwrap().unwrap();
        let view2 = column.get_view(1).unwrap().unwrap();
        
        assert_eq!(view1.as_str().unwrap(), "hello");
        assert_eq!(view2.as_str().unwrap(), "world");
        assert_eq!(view1.len(), 5);
        assert_eq!(view2.len(), 5);
    }
    
    #[test]
    fn test_string_deduplication() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "hello".to_string(), // Duplicate
            "test".to_string(),
            "world".to_string(), // Another duplicate
        ];
        
        let column = ZeroCopyStringColumn::new(data).unwrap();
        let stats = column.pool_stats().unwrap();
        
        // Should have deduplication
        assert!(stats.unique_strings < 5);
        assert!(stats.deduplication_ratio > 0.0);
        
        // Verify correctness
        assert_eq!(column.get(0).unwrap().unwrap(), "hello");
        assert_eq!(column.get(1).unwrap().unwrap(), "world");
        assert_eq!(column.get(2).unwrap().unwrap(), "hello");
        assert_eq!(column.get(3).unwrap().unwrap(), "test");
        assert_eq!(column.get(4).unwrap().unwrap(), "world");
    }
    
    #[test]
    fn test_zero_copy_operations() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];
        
        let column = ZeroCopyStringColumn::new(data).unwrap();
        
        // Test contains
        assert!(column.contains("hello").unwrap());
        assert!(column.contains("world").unwrap());
        assert!(!column.contains("missing").unwrap());
        
        // Test count occurrences
        assert_eq!(column.count_occurrences("hello").unwrap(), 1);
        assert_eq!(column.count_occurrences("missing").unwrap(), 0);
        
        // Test string lengths
        let lengths = column.string_lengths().unwrap();
        assert_eq!(lengths, vec![Some(5), Some(5), Some(4)]);
    }
    
    #[test]
    fn test_zero_copy_filtering() {
        let data = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "apricot".to_string(),
        ];
        
        let column = ZeroCopyStringColumn::new(data).unwrap();
        
        // Filter strings starting with 'a'
        let indices = column.filter_views(|s| s.starts_with('a')).unwrap();
        assert_eq!(indices, vec![0, 3]); // "apple" and "apricot"
        
        // Filter by length
        let long_indices = column.filter_views(|s| s.len() > 5).unwrap();
        assert_eq!(long_indices, vec![1, 2, 3]); // "banana", "cherry", "apricot"
    }
    
    #[test]
    fn test_zero_copy_transformations() {
        let data = vec![
            "Hello".to_string(),
            "WORLD".to_string(),
            "Test".to_string(),
        ];
        
        let column = ZeroCopyStringColumn::new(data).unwrap();
        
        // Test lowercase conversion
        let lowercase = column.to_lowercase_optimized().unwrap();
        assert_eq!(lowercase.get(0).unwrap().unwrap(), "hello");
        assert_eq!(lowercase.get(1).unwrap().unwrap(), "world");
        assert_eq!(lowercase.get(2).unwrap().unwrap(), "test");
        
        // Test uppercase conversion
        let uppercase = column.to_uppercase_optimized().unwrap();
        assert_eq!(uppercase.get(0).unwrap().unwrap(), "HELLO");
        assert_eq!(uppercase.get(1).unwrap().unwrap(), "WORLD");
        assert_eq!(uppercase.get(2).unwrap().unwrap(), "TEST");
    }
    
    #[test]
    fn test_shared_pool() {
        let pool = Arc::new(UnifiedStringPool::with_default_config().unwrap());
        
        let data1 = vec!["shared".to_string(), "pool".to_string()];
        let data2 = vec!["test".to_string(), "shared".to_string()]; // "shared" is repeated
        
        let column1 = ZeroCopyStringColumn::with_shared_pool(data1, Arc::clone(&pool)).unwrap();
        let column2 = ZeroCopyStringColumn::with_shared_pool(data2, Arc::clone(&pool)).unwrap();
        
        // Both columns should share the same pool
        let stats = pool.stats().unwrap();
        assert_eq!(stats.unique_strings, 3); // "shared", "pool", "test"
        
        // Verify data integrity
        assert_eq!(column1.get(0).unwrap().unwrap(), "shared");
        assert_eq!(column1.get(1).unwrap().unwrap(), "pool");
        assert_eq!(column2.get(0).unwrap().unwrap(), "test");
        assert_eq!(column2.get(1).unwrap().unwrap(), "shared");
    }
    
    #[test]
    fn test_concatenation() {
        let data1 = vec!["hello".to_string(), "world".to_string()];
        let data2 = vec!["there".to_string(), "test".to_string()];
        
        let column1 = ZeroCopyStringColumn::new(data1).unwrap();
        let column2 = ZeroCopyStringColumn::new(data2).unwrap();
        
        let concatenated = column1.concat_with(&column2, " ").unwrap();
        
        assert_eq!(concatenated.get(0).unwrap().unwrap(), "hello there");
        assert_eq!(concatenated.get(1).unwrap().unwrap(), "world test");
    }
    
    #[test]
    fn test_with_nulls() {
        let data = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
        ];
        let nulls = vec![false, true, false]; // world is null
        
        let column = ZeroCopyStringColumn::with_nulls(data, nulls).unwrap();
        
        assert_eq!(column.get(0).unwrap().unwrap(), "hello");
        assert!(column.get(1).unwrap().is_none()); // null
        assert_eq!(column.get(2).unwrap().unwrap(), "test");
    }
}
//! Simplified Zero-Copy String Column Implementation
//!
//! This module provides a string column implementation that leverages the simplified
//! zero-copy string pool for optimal memory efficiency and performance.

use std::any::Any;
use std::sync::Arc;

use crate::core::column::{Column, ColumnTrait, ColumnType};
use crate::core::error::{Error, Result};
use crate::storage::simple_unified_string_pool::{
    SimpleStringPoolStats, SimpleStringView, SimpleUnifiedStringPool,
};

/// Simplified zero-copy string column
#[derive(Debug, Clone)]
pub struct SimpleZeroCopyStringColumn {
    /// Reference to the unified string pool
    pool: Arc<SimpleUnifiedStringPool>,
    /// String IDs for each row
    string_ids: Arc<[u32]>,
    /// Null mask for handling NULL values
    null_mask: Option<Arc<[u8]>>,
    /// Column name
    name: Option<String>,
}

impl SimpleZeroCopyStringColumn {
    /// Create a new zero-copy string column from strings
    pub fn new(data: Vec<String>) -> Result<Self> {
        let pool = Arc::new(SimpleUnifiedStringPool::new());
        let string_ids = pool.add_strings(&data)?;

        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask: None,
            name: None,
        })
    }

    /// Create a zero-copy string column with a shared pool
    pub fn with_shared_pool(data: Vec<String>, pool: Arc<SimpleUnifiedStringPool>) -> Result<Self> {
        let string_ids = pool.add_strings(&data)?;

        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask: None,
            name: None,
        })
    }

    /// Create a zero-copy string column with a shared pool AND explicit
    /// NULL positions.
    ///
    /// This is what derived-column operations (`to_lowercase_optimized`,
    /// `to_uppercase_optimized`, `concat_with`, `substring_views`) build on
    /// top of: transforming a column that has NULLs needs a way to carry
    /// those NULL positions into the new column. [`Self::with_shared_pool`]
    /// has no `null_mask` parameter at all, so those operations used to
    /// hardcode `null_mask: None` on their output -- meaning a source-NULL
    /// cell (represented as a `String::new()` placeholder so the pool has
    /// *something* to store) silently read back as a real, non-NULL empty
    /// string instead of NULL.
    pub fn with_shared_pool_and_nulls(
        data: Vec<String>,
        pool: Arc<SimpleUnifiedStringPool>,
        nulls: Vec<bool>,
    ) -> Result<Self> {
        let string_ids = pool.add_strings(&data)?;
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };

        Ok(Self {
            pool,
            string_ids: string_ids.into(),
            null_mask,
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
    pub fn get_view(&self, index: usize) -> Result<Option<SimpleStringView>> {
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
            Some(view) => Ok(Some(view.as_str()?)),
            None => Ok(None),
        }
    }

    /// Get multiple zero-copy string views
    pub fn get_views(&self, indices: &[usize]) -> Result<Vec<Option<SimpleStringView>>> {
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

    /// Apply a function to each string view (zero-copy)
    pub fn map_views<F, R>(&self, mut f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<SimpleStringView>) -> R,
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
                // Use with_str_ref for zero-copy access
                let matches = view.with_str_ref(&mut predicate)?;
                if matches {
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
                let is_match = view.with_str_ref(|s| s == target)?;
                if is_match {
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
                let is_match = view.with_str_ref(|s| s == target)?;
                if is_match {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Get unique strings (using zero-copy)
    pub fn unique_views(&self) -> Result<Vec<SimpleStringView>> {
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
    pub fn pool(&self) -> &Arc<SimpleUnifiedStringPool> {
        &self.pool
    }

    /// Get pool statistics
    pub fn pool_stats(&self) -> Result<SimpleStringPoolStats> {
        self.pool.stats()
    }

    /// Fallible counterpart of [`ColumnTrait::clone_column`].
    ///
    /// `clone_column` must return a bare `Column` (that's the trait's
    /// signature), so it cannot propagate a lookup failure -- see its
    /// implementation below for how it now handles that without corrupting
    /// row count. This method exists for callers that *can* handle a
    /// `Result` and would rather see the real error (a mismatched
    /// `string_ids`/`pool` pairing, only reachable via
    /// [`Self::with_string_ids`], is the only way `get` can fail here).
    pub fn try_clone_column(&self) -> Result<Column> {
        let strings = self.to_strings()?;
        let nulls: Vec<bool> = strings.iter().map(|s| s.is_none()).collect();
        let string_values: Vec<String> =
            strings.into_iter().map(|s| s.unwrap_or_default()).collect();

        if nulls.iter().any(|&is_null| is_null) {
            Ok(Column::String(crate::column::StringColumn::with_nulls(
                string_values,
                nulls,
            )))
        } else {
            Ok(Column::String(crate::column::StringColumn::new(
                string_values,
            )))
        }
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

    /// Case conversion with minimal allocation
    pub fn to_lowercase_optimized(&self) -> Result<SimpleZeroCopyStringColumn> {
        let mut nulls = Vec::with_capacity(self.string_ids.len());
        let new_strings = self.map_views(|view_opt| match view_opt {
            Some(view) => match view.as_str() {
                Ok(s) => {
                    nulls.push(false);
                    s.to_lowercase()
                }
                Err(_) => {
                    // A pool lookup/UTF-8 failure is an error, not a
                    // legitimate empty string -- report it as NULL rather
                    // than fabricating a non-NULL "".
                    nulls.push(true);
                    String::new()
                }
            },
            None => {
                nulls.push(true);
                String::new()
            }
        })?;

        SimpleZeroCopyStringColumn::with_shared_pool_and_nulls(
            new_strings,
            Arc::clone(&self.pool),
            nulls,
        )
    }

    /// Case conversion with minimal allocation
    pub fn to_uppercase_optimized(&self) -> Result<SimpleZeroCopyStringColumn> {
        let mut nulls = Vec::with_capacity(self.string_ids.len());
        let new_strings = self.map_views(|view_opt| match view_opt {
            Some(view) => match view.as_str() {
                Ok(s) => {
                    nulls.push(false);
                    s.to_uppercase()
                }
                Err(_) => {
                    nulls.push(true);
                    String::new()
                }
            },
            None => {
                nulls.push(true);
                String::new()
            }
        })?;

        SimpleZeroCopyStringColumn::with_shared_pool_and_nulls(
            new_strings,
            Arc::clone(&self.pool),
            nulls,
        )
    }

    /// Concatenate strings with another column (zero-copy where possible)
    pub fn concat_with(
        &self,
        other: &SimpleZeroCopyStringColumn,
        separator: &str,
    ) -> Result<SimpleZeroCopyStringColumn> {
        if self.string_ids.len() != other.string_ids.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.string_ids.len(),
                found: other.string_ids.len(),
            });
        }

        let mut new_strings = Vec::with_capacity(self.string_ids.len());
        let mut nulls = Vec::with_capacity(self.string_ids.len());

        for i in 0..self.string_ids.len() {
            let left = self.get_view(i)?;
            let right = other.get_view(i)?;

            // Match pandas `Series.str.cat(other, sep=..)` with its default
            // `na_rep=None`: the result is NULL wherever EITHER operand is
            // NULL. The previous implementation instead coalesced a
            // single-sided NULL to the non-NULL side's raw value, which
            // silently hid the missingness from the caller instead of
            // propagating it.
            match (left, right) {
                (Some(left_view), Some(right_view)) => {
                    let concatenated = format!(
                        "{}{}{}",
                        left_view.as_str()?,
                        separator,
                        right_view.as_str()?
                    );
                    new_strings.push(concatenated);
                    nulls.push(false);
                }
                _ => {
                    new_strings.push(String::new());
                    nulls.push(true);
                }
            }
        }

        SimpleZeroCopyStringColumn::with_shared_pool_and_nulls(
            new_strings,
            Arc::clone(&self.pool),
            nulls,
        )
    }
}

impl ColumnTrait for SimpleZeroCopyStringColumn {
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
        // Convert to regular StringColumn for compatibility.
        //
        // This used to call `self.to_strings().unwrap_or_default()`:
        // `to_strings` returns ONE `Result` wrapping the whole row vector,
        // so a single failed per-row pool lookup made the WHOLE conversion
        // fail, and `unwrap_or_default()` silently turned that `Err` into
        // an *empty* `Vec` -- collapsing an N-row column to 0 rows instead
        // of surfacing (or even just isolating) the failure. Iterating
        // per-cell instead means one bad cell degrades to a NULL in that
        // one cell, preserving `len() == self.len()` unconditionally. It
        // also carries this column's own NULLs through, which the old
        // `StringColumn::new(..)`-only path dropped entirely.
        let mut string_values = Vec::with_capacity(self.string_ids.len());
        let mut nulls = Vec::with_capacity(self.string_ids.len());

        for i in 0..self.string_ids.len() {
            match self.get(i) {
                Ok(Some(s)) => {
                    string_values.push(s);
                    nulls.push(false);
                }
                Ok(None) | Err(_) => {
                    string_values.push(String::new());
                    nulls.push(true);
                }
            }
        }

        if nulls.iter().any(|&is_null| is_null) {
            Column::String(crate::column::StringColumn::with_nulls(
                string_values,
                nulls,
            ))
        } else {
            Column::String(crate::column::StringColumn::new(string_values))
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Zero-copy string operations trait for the simplified column
pub trait SimpleZeroCopyStringOps {
    /// Transform strings using zero-copy views
    fn transform_zero_copy<F, R>(&self, f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<SimpleStringView>) -> R;

    /// Get substring views (zero-copy)
    fn substring_views(&self, start: usize, end: usize) -> Result<SimpleZeroCopyStringColumn>;
}

impl SimpleZeroCopyStringOps for SimpleZeroCopyStringColumn {
    fn transform_zero_copy<F, R>(&self, f: F) -> Result<Vec<R>>
    where
        F: FnMut(Option<SimpleStringView>) -> R,
    {
        self.map_views(f)
    }

    fn substring_views(&self, start: usize, end: usize) -> Result<SimpleZeroCopyStringColumn> {
        let mut new_strings = Vec::with_capacity(self.string_ids.len());
        let mut nulls = Vec::with_capacity(self.string_ids.len());

        for i in 0..self.string_ids.len() {
            if let Some(view) = self.get_view(i)? {
                let str_len = view.len();
                let actual_start = start.min(str_len);
                let actual_end = end.min(str_len);

                if actual_start < actual_end {
                    let substring = view.substring(actual_start, actual_end)?;
                    new_strings.push(substring.as_str()?);
                } else {
                    // Slicing past the end (or an empty range) is a
                    // legitimate empty string, not a missing value --
                    // matches pandas `str.slice`.
                    new_strings.push(String::new());
                }
                nulls.push(false);
            } else {
                // Source row was NULL -- keep it NULL in the derived
                // column instead of turning it into a real empty string.
                new_strings.push(String::new());
                nulls.push(true);
            }
        }

        SimpleZeroCopyStringColumn::with_shared_pool_and_nulls(
            new_strings,
            Arc::clone(&self.pool),
            nulls,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_zero_copy_string_column_creation() {
        let data = vec!["hello".to_string(), "world".to_string(), "test".to_string()];

        let column =
            SimpleZeroCopyStringColumn::new(data.clone()).expect("operation should succeed");
        assert_eq!(column.len(), 3);
        assert!(!column.is_empty());
        assert_eq!(column.column_type(), ColumnType::String);

        // Test data retrieval
        assert_eq!(
            column
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello"
        );
        assert_eq!(
            column
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "world"
        );
        assert_eq!(
            column
                .get(2)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "test"
        );
    }

    #[test]
    fn test_zero_copy_views() {
        let data = vec!["hello".to_string(), "world".to_string()];

        let column = SimpleZeroCopyStringColumn::new(data).expect("operation should succeed");

        let view1 = column
            .get_view(0)
            .expect("operation should succeed")
            .expect("operation should succeed");
        let view2 = column
            .get_view(1)
            .expect("operation should succeed")
            .expect("operation should succeed");

        assert_eq!(view1.as_str().expect("operation should succeed"), "hello");
        assert_eq!(view2.as_str().expect("operation should succeed"), "world");
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

        let column = SimpleZeroCopyStringColumn::new(data).expect("operation should succeed");
        let stats = column.pool_stats().expect("operation should succeed");

        // Should have deduplication: we have 5 total strings but only 3 unique
        assert_eq!(stats.total_strings, 5); // 5 total string addition calls
        assert_eq!(stats.unique_strings, 3); // "hello", "world", "test"
                                             // Deduplication ratio should be positive (we saved 2 duplicates out of 5 total)
                                             // Expected: 1.0 - (3/5) = 0.4 (40% deduplication)
        assert!(
            stats.deduplication_ratio > 0.0,
            "Deduplication ratio: {}",
            stats.deduplication_ratio
        );
        assert!(
            (stats.deduplication_ratio - 0.4).abs() < 0.001,
            "Expected ~0.4, got {}",
            stats.deduplication_ratio
        );

        // Verify correctness
        assert_eq!(
            column
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello"
        );
        assert_eq!(
            column
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "world"
        );
        assert_eq!(
            column
                .get(2)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello"
        );
        assert_eq!(
            column
                .get(3)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "test"
        );
        assert_eq!(
            column
                .get(4)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "world"
        );
    }

    #[test]
    fn test_zero_copy_operations() {
        let data = vec!["hello".to_string(), "world".to_string(), "test".to_string()];

        let column = SimpleZeroCopyStringColumn::new(data).expect("operation should succeed");

        // Test contains
        assert!(column.contains("hello").expect("operation should succeed"));
        assert!(column.contains("world").expect("operation should succeed"));
        assert!(!column
            .contains("missing")
            .expect("operation should succeed"));

        // Test count occurrences
        assert_eq!(
            column
                .count_occurrences("hello")
                .expect("operation should succeed"),
            1
        );
        assert_eq!(
            column
                .count_occurrences("missing")
                .expect("operation should succeed"),
            0
        );

        // Test string lengths
        let lengths = column.string_lengths().expect("operation should succeed");
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

        let column = SimpleZeroCopyStringColumn::new(data).expect("operation should succeed");

        // Filter strings starting with 'a'
        let indices = column
            .filter_views(|s| s.starts_with('a'))
            .expect("operation should succeed");
        assert_eq!(indices, vec![0, 3]); // "apple" and "apricot"

        // Filter by length
        let long_indices = column
            .filter_views(|s| s.len() > 5)
            .expect("operation should succeed");
        assert_eq!(long_indices, vec![1, 2, 3]); // "banana", "cherry", "apricot"
    }

    #[test]
    fn test_zero_copy_transformations() {
        let data = vec!["Hello".to_string(), "WORLD".to_string(), "Test".to_string()];

        let column = SimpleZeroCopyStringColumn::new(data).expect("operation should succeed");

        // Test lowercase conversion
        let lowercase = column
            .to_lowercase_optimized()
            .expect("operation should succeed");
        assert_eq!(
            lowercase
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello"
        );
        assert_eq!(
            lowercase
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "world"
        );
        assert_eq!(
            lowercase
                .get(2)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "test"
        );

        // Test uppercase conversion
        let uppercase = column
            .to_uppercase_optimized()
            .expect("operation should succeed");
        assert_eq!(
            uppercase
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "HELLO"
        );
        assert_eq!(
            uppercase
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "WORLD"
        );
        assert_eq!(
            uppercase
                .get(2)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "TEST"
        );
    }

    #[test]
    fn test_shared_pool() {
        let pool = Arc::new(SimpleUnifiedStringPool::new());

        let data1 = vec!["shared".to_string(), "pool".to_string()];
        let data2 = vec!["test".to_string(), "shared".to_string()]; // "shared" is repeated

        let column1 = SimpleZeroCopyStringColumn::with_shared_pool(data1, Arc::clone(&pool))
            .expect("operation should succeed");
        let column2 = SimpleZeroCopyStringColumn::with_shared_pool(data2, Arc::clone(&pool))
            .expect("operation should succeed");

        // Both columns should share the same pool
        let stats = pool.stats().expect("operation should succeed");
        assert_eq!(stats.unique_strings, 3); // "shared", "pool", "test"

        // Verify data integrity
        assert_eq!(
            column1
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "shared"
        );
        assert_eq!(
            column1
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "pool"
        );
        assert_eq!(
            column2
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "test"
        );
        assert_eq!(
            column2
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "shared"
        );
    }

    #[test]
    fn test_concatenation() {
        let data1 = vec!["hello".to_string(), "world".to_string()];
        let data2 = vec!["there".to_string(), "test".to_string()];

        let column1 = SimpleZeroCopyStringColumn::new(data1).expect("operation should succeed");
        let column2 = SimpleZeroCopyStringColumn::new(data2).expect("operation should succeed");

        let concatenated = column1
            .concat_with(&column2, " ")
            .expect("operation should succeed");

        assert_eq!(
            concatenated
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello there"
        );
        assert_eq!(
            concatenated
                .get(1)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "world test"
        );
    }

    #[test]
    fn test_with_nulls() {
        let data = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let nulls = vec![false, true, false]; // world is null

        let column =
            SimpleZeroCopyStringColumn::with_nulls(data, nulls).expect("operation should succeed");

        assert_eq!(
            column
                .get(0)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "hello"
        );
        assert!(column.get(1).expect("operation should succeed").is_none()); // null
        assert_eq!(
            column
                .get(2)
                .expect("operation should succeed")
                .expect("operation should succeed"),
            "test"
        );
    }

    #[test]
    fn null_mask_survives_case_conversion() {
        let data = vec!["Hello".to_string(), "World".to_string(), "Test".to_string()];
        let nulls = vec![false, true, false]; // "World" is NULL
        let column =
            SimpleZeroCopyStringColumn::with_nulls(data, nulls).expect("operation should succeed");

        let lower = column
            .to_lowercase_optimized()
            .expect("operation should succeed");
        assert_eq!(lower.get(0).expect("ok"), Some("hello".to_string()));
        assert_eq!(
            lower.get(1).expect("ok"),
            None,
            "NULL must stay NULL, not become an empty string"
        );
        assert_eq!(lower.get(2).expect("ok"), Some("test".to_string()));

        let upper = column
            .to_uppercase_optimized()
            .expect("operation should succeed");
        assert_eq!(upper.get(0).expect("ok"), Some("HELLO".to_string()));
        assert_eq!(upper.get(1).expect("ok"), None);
        assert_eq!(upper.get(2).expect("ok"), Some("TEST".to_string()));
    }

    #[test]
    fn concat_with_propagates_null_like_pandas_str_cat() {
        let left = SimpleZeroCopyStringColumn::with_nulls(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![false, true, false],
        )
        .expect("ok");
        let right = SimpleZeroCopyStringColumn::with_nulls(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            vec![false, false, true],
        )
        .expect("ok");

        let result = left.concat_with(&right, "-").expect("ok");

        assert_eq!(result.get(0).expect("ok"), Some("a-x".to_string()));
        assert_eq!(result.get(1).expect("ok"), None, "left NULL must propagate");
        assert_eq!(
            result.get(2).expect("ok"),
            None,
            "right NULL must propagate"
        );
    }

    #[test]
    fn substring_views_carries_null_but_keeps_legitimate_empty_strings() {
        let column = SimpleZeroCopyStringColumn::with_nulls(
            vec!["hello".to_string(), "hi".to_string(), "world".to_string()],
            vec![false, true, false],
        )
        .expect("ok");

        // Slice past the end of "hi" (len 2): a legitimate empty string,
        // not NULL.
        let sliced = column.substring_views(10, 20).expect("ok");
        assert_eq!(
            sliced.get(0).expect("ok"),
            Some(String::new()),
            "out-of-range slice is an empty string, not NULL"
        );
        assert_eq!(
            sliced.get(1).expect("ok"),
            None,
            "a NULL source row must stay NULL, not become an empty string"
        );
        assert_eq!(sliced.get(2).expect("ok"), Some(String::new()));
    }

    #[test]
    fn clone_column_preserves_row_count_and_nulls() {
        let data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let nulls = vec![false, true, false];
        let column =
            SimpleZeroCopyStringColumn::with_nulls(data, nulls).expect("operation should succeed");

        let cloned = column.clone_column();
        assert_eq!(cloned.len(), 3, "clone_column must never change row count");

        if let Column::String(string_col) = &cloned {
            assert_eq!(string_col.get(0).expect("ok"), Some("a"));
            assert_eq!(string_col.get(1).expect("ok"), None);
            assert_eq!(string_col.get(2).expect("ok"), Some("c"));
        } else {
            panic!("clone_column of a string column must produce Column::String");
        }

        let via_try = column
            .try_clone_column()
            .expect("no lookup failure expected here");
        assert_eq!(via_try.len(), 3);
    }
}

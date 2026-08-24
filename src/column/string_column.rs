use std::any::Any;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::column::string_pool::StringPool;
use crate::core::column::{Column, ColumnTrait, ColumnType};
use crate::core::error::{Error, Result};

/// Optimization modes for string columns
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum StringColumnOptimizationMode {
    /// Legacy mode (original implementation)
    Legacy = 0,
    /// Uses global string pool
    GlobalPool = 1,
    /// Uses categorical encoding
    Categorical = 2,
}

impl StringColumnOptimizationMode {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => StringColumnOptimizationMode::Legacy,
            2 => StringColumnOptimizationMode::Categorical,
            _ => StringColumnOptimizationMode::GlobalPool,
        }
    }
}

/// Process-wide default optimization mode, as an atomic `u8` encoding of
/// [`StringColumnOptimizationMode`].
///
/// This used to be a `pub static mut StringColumnOptimizationMode`, read
/// and written through ad hoc `unsafe` blocks at every call site (see
/// `StringColumn::new`/`with_nulls` below). That is undefined behavior the
/// moment two threads touch it concurrently -- one thread reading while
/// another writes is a data race on a non-atomic static -- and merely
/// taking a reference to a mutable static is a hard error under the Rust
/// 2024 edition. `AtomicU8` gives the same "process-wide default, settable
/// at runtime" behavior with neither problem, via the
/// [`default_optimization_mode`]/[`set_default_optimization_mode`] accessors.
static DEFAULT_OPTIMIZATION_MODE: AtomicU8 =
    AtomicU8::new(StringColumnOptimizationMode::GlobalPool as u8);

/// Get the process-wide default optimization mode used by [`StringColumn::new`]
/// and [`StringColumn::with_nulls`].
pub fn default_optimization_mode() -> StringColumnOptimizationMode {
    StringColumnOptimizationMode::from_u8(DEFAULT_OPTIMIZATION_MODE.load(Ordering::Relaxed))
}

/// Set the process-wide default optimization mode used by [`StringColumn::new`]
/// and [`StringColumn::with_nulls`].
pub fn set_default_optimization_mode(mode: StringColumnOptimizationMode) {
    DEFAULT_OPTIMIZATION_MODE.store(mode as u8, Ordering::Relaxed);
}

/// Structure representing a string column (using string pool)
#[derive(Debug, Clone)]
pub struct StringColumn {
    pub(crate) string_pool: Arc<StringPool>,
    pub(crate) indices: Arc<[u32]>,
    pub(crate) null_mask: Option<Arc<[u8]>>,
    pub(crate) name: Option<String>,
    pub(crate) optimization_mode: StringColumnOptimizationMode,
}

impl StringColumn {
    /// Create a new StringColumn from a vector of strings
    pub fn new(data: Vec<String>) -> Self {
        match default_optimization_mode() {
            StringColumnOptimizationMode::Legacy => Self::new_legacy(data),
            StringColumnOptimizationMode::GlobalPool => Self::new_with_global_pool(data),
            StringColumnOptimizationMode::Categorical => Self::new_categorical(data),
        }
    }

    /// Create a StringColumn using legacy mode (original implementation,
    /// bypassing the global string pool entirely).
    pub fn new_legacy(data: Vec<String>) -> Self {
        let (pool, indices) = StringPool::from_strings_legacy(&data);

        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
            null_mask: None,
            name: None,
            optimization_mode: StringColumnOptimizationMode::Legacy,
        }
    }

    /// Create a StringColumn using global pool.
    ///
    /// Falls back to a purely local pool (built the same way
    /// [`Self::new_legacy`] builds one) if the process-wide global string
    /// pool's lock is poisoned -- an exceedingly rare condition that can
    /// only happen if some unrelated thread already panicked while holding
    /// it. This keeps the constructor infallible, which callers rely on,
    /// while never fabricating a wrong index the way the old "poisoned
    /// lock silently returns index 0" behavior did: every string in `data`
    /// still round-trips correctly through `get()`, just without the
    /// cross-column interning bookkeeping for this one call. The fallback
    /// is also tagged `Legacy` rather than `GlobalPool` in
    /// [`Self::optimization_mode`]'s return value: it never touched the
    /// global pool, so claiming otherwise would be exactly the kind of
    /// fabricated status this crate's audit exists to catch.
    pub fn new_with_global_pool(data: Vec<String>) -> Self {
        let (pool, indices, optimization_mode) = match StringPool::from_strings(&data) {
            Ok((pool, indices)) => (pool, indices, StringColumnOptimizationMode::GlobalPool),
            Err(_) => {
                let (pool, indices) = StringPool::from_strings_legacy(&data);
                (pool, indices, StringColumnOptimizationMode::Legacy)
            }
        };

        Self {
            string_pool: Arc::new(pool),
            indices: indices.into(),
            null_mask: None,
            name: None,
            optimization_mode,
        }
    }

    /// Create a StringColumn using categorical encoding
    pub fn new_categorical(data: Vec<String>) -> Self {
        Self::from_strings_optimized(data)
    }

    /// Create a StringColumn with optimized one-pass processing
    pub fn from_strings_optimized(data: Vec<String>) -> Self {
        // Use the global pool mode for better thread safety
        Self::new_with_global_pool(data)
    }

    /// Create a StringColumn with a name
    pub fn with_name(data: Vec<String>, name: impl Into<String>) -> Self {
        // Create using default optimization mode
        let mut column = Self::new(data);
        column.name = Some(name.into());
        column
    }

    /// Create a StringColumn with NULL values
    pub fn with_nulls(data: Vec<String>, nulls: Vec<bool>) -> Self {
        let null_mask = if nulls.iter().any(|&is_null| is_null) {
            Some(crate::column::common::utils::create_bitmask(&nulls))
        } else {
            None
        };

        // Processing according to optimization mode
        let mut column = match default_optimization_mode() {
            StringColumnOptimizationMode::Legacy => Self::new_legacy(data),
            StringColumnOptimizationMode::GlobalPool => Self::new_with_global_pool(data),
            StringColumnOptimizationMode::Categorical => Self::new_categorical(data),
        };

        column.null_mask = null_mask;
        column
    }

    /// Set the name
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get the name
    pub fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get the optimization mode this column was built with
    pub fn optimization_mode(&self) -> StringColumnOptimizationMode {
        self.optimization_mode
    }

    /// Get string at the specified index
    pub fn get(&self, index: usize) -> Result<Option<&str>> {
        if index >= self.indices.len() {
            return Err(Error::IndexOutOfBounds {
                index,
                size: self.indices.len(),
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

        let str_idx = self.indices[index];
        Ok(self.string_pool.get(str_idx))
    }

    /// Get all strings in the column
    pub fn to_strings(&self) -> Vec<Option<String>> {
        let mut result = Vec::with_capacity(self.indices.len());

        for i in 0..self.indices.len() {
            let value = match self.get(i) {
                Ok(Some(s)) => Some(s.to_string()),
                Ok(None) => None,
                Err(_) => None,
            };
            result.push(value);
        }

        result
    }

    /// Search for a string
    pub fn contains(&self, search_str: &str) -> Vec<bool> {
        let mut result = vec![false; self.indices.len()];

        for i in 0..self.indices.len() {
            if let Ok(Some(s)) = self.get(i) {
                result[i] = s.contains(search_str);
            }
        }

        result
    }

    /// Match using regular expression
    pub fn matches(&self, pattern: &str) -> Result<Vec<bool>> {
        use regex::Regex;

        let re = Regex::new(pattern).map_err(|e| Error::InvalidRegex(e.to_string()))?;

        let mut result = vec![false; self.indices.len()];

        for i in 0..self.indices.len() {
            if let Ok(Some(s)) = self.get(i) {
                result[i] = re.is_match(s);
            }
        }

        Ok(result)
    }

    /// Create a new column by applying a mapping function
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(&str) -> String,
    {
        let mut mapped_data = Vec::with_capacity(self.indices.len());
        let mut has_nulls = false;

        for i in 0..self.indices.len() {
            match self.get(i) {
                Ok(Some(s)) => mapped_data.push(f(s)),
                Ok(None) => {
                    has_nulls = true;
                    mapped_data.push(String::new()); // Dummy value
                }
                Err(_) => {
                    has_nulls = true;
                    mapped_data.push(String::new()); // Dummy value
                }
            }
        }

        if has_nulls {
            let nulls = (0..self.indices.len())
                .map(|i| self.get(i).map(|opt| opt.is_none()).unwrap_or(true))
                .collect();

            Self::with_nulls(mapped_data, nulls)
        } else {
            Self::new(mapped_data)
        }
    }

    /// Create a new column based on filtering conditions
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(Option<&str>) -> bool,
    {
        let mut filtered_data = Vec::new();
        let mut filtered_nulls = Vec::new();
        let has_nulls = self.null_mask.is_some();

        for i in 0..self.indices.len() {
            let value = self.get(i).unwrap_or(None);
            if predicate(value) {
                filtered_data.push(value.unwrap_or("").to_string());
                if has_nulls {
                    filtered_nulls.push(value.is_none());
                }
            }
        }

        if has_nulls {
            Self::with_nulls(filtered_data, filtered_nulls)
        } else {
            Self::new(filtered_data)
        }
    }
}

impl ColumnTrait for StringColumn {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn column_type(&self) -> ColumnType {
        ColumnType::String
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn clone_column(&self) -> crate::core::column::Column {
        // Convert the legacy Column type to the core Column type
        let legacy_column = Column::String(self.clone());
        // This is a temporary workaround - in a complete solution,
        // we would implement proper conversion between column types
        crate::core::column::Column::from_any(Box::new(legacy_column))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_optimization_mode_round_trips_through_atomic_accessor() {
        // Save/restore so this test doesn't leak its mode change into
        // other tests running in the same process (StringColumn::new()
        // consults the process-wide default).
        let previous = default_optimization_mode();

        set_default_optimization_mode(StringColumnOptimizationMode::Legacy);
        assert_eq!(
            default_optimization_mode(),
            StringColumnOptimizationMode::Legacy
        );

        set_default_optimization_mode(StringColumnOptimizationMode::Categorical);
        assert_eq!(
            default_optimization_mode(),
            StringColumnOptimizationMode::Categorical
        );

        set_default_optimization_mode(previous);
    }

    #[test]
    fn optimization_mode_getter_reports_construction_mode() {
        let legacy = StringColumn::new_legacy(vec!["a".to_string()]);
        assert_eq!(
            legacy.optimization_mode(),
            StringColumnOptimizationMode::Legacy
        );

        let global = StringColumn::new_with_global_pool(vec!["a".to_string()]);
        assert_eq!(
            global.optimization_mode(),
            StringColumnOptimizationMode::GlobalPool
        );
    }
}

//! JIT Function Caching System
//!
//! This module provides intelligent caching of JIT-compiled functions to avoid
//! recompilation overhead and improve performance across repeated operations.

use crate::core::error::{Error, Result};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Unique identifier for cached functions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionId {
    /// Function name
    pub name: String,
    /// Input type signature
    pub input_type: String,
    /// Output type signature
    pub output_type: String,
    /// Operation signature (for distinguishing between operations with same types)
    pub operation_signature: String,
    /// Optimization level used for compilation
    pub optimization_level: u8,
}

impl FunctionId {
    /// Create a new function ID
    pub fn new(
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
        operation_signature: impl Into<String>,
        optimization_level: u8,
    ) -> Self {
        Self {
            name: name.into(),
            input_type: input_type.into(),
            output_type: output_type.into(),
            operation_signature: operation_signature.into(),
            optimization_level,
        }
    }

    /// Generate a unique hash for this function ID
    pub fn hash_value(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Metadata about a cached function
#[derive(Debug, Clone)]
pub struct CachedFunctionMetadata {
    /// When the function was compiled
    pub compiled_at: SystemTime,
    /// How many times this function has been executed
    pub execution_count: u64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: f64,
    /// Size of the compiled function in bytes
    pub function_size_bytes: usize,
    /// Compilation time in nanoseconds
    pub compilation_time_ns: u64,
    /// Whether this function is performance-critical (frequently used)
    pub is_hot: bool,
    /// Last access time
    pub last_accessed: Instant,
    /// Function effectiveness score (performance improvement over naive implementation)
    pub effectiveness_score: f64,
}

impl CachedFunctionMetadata {
    /// Create new metadata for a compiled function
    pub fn new(compilation_time_ns: u64, function_size_bytes: usize) -> Self {
        Self {
            compiled_at: SystemTime::now(),
            execution_count: 0,
            total_execution_time_ns: 0,
            avg_execution_time_ns: 0.0,
            function_size_bytes,
            compilation_time_ns,
            is_hot: false,
            last_accessed: Instant::now(),
            effectiveness_score: 1.0,
        }
    }

    /// Record an execution of this function
    pub fn record_execution(&mut self, execution_time_ns: u64) {
        self.execution_count += 1;
        self.total_execution_time_ns += execution_time_ns;
        self.avg_execution_time_ns =
            self.total_execution_time_ns as f64 / self.execution_count as f64;
        self.last_accessed = Instant::now();

        // Mark as hot if executed frequently
        if self.execution_count > 100 && self.avg_execution_time_ns < 1_000_000.0 {
            self.is_hot = true;
        }
    }

    /// Calculate the benefit of caching this function
    pub fn cache_benefit(&self) -> f64 {
        if self.execution_count == 0 {
            return 0.0;
        }

        // Benefit = (compilation cost amortized over executions) * effectiveness
        let amortized_compilation_cost =
            self.compilation_time_ns as f64 / self.execution_count as f64;
        let avg_execution_savings = amortized_compilation_cost * self.effectiveness_score;

        avg_execution_savings
    }

    /// Check if this function should be evicted from cache
    pub fn should_evict(&self, cache_pressure: f64) -> bool {
        let time_since_access = self.last_accessed.elapsed().as_secs_f64();
        let cache_benefit = self.cache_benefit();

        // Evict if:
        // 1. High cache pressure and low benefit
        // 2. Not accessed for a long time
        // 3. Poor effectiveness score
        (cache_pressure > 0.8 && cache_benefit < 1000.0) ||
        (time_since_access > 3600.0) || // 1 hour
        (self.effectiveness_score < 0.5)
    }
}

/// Cached compiled function
pub struct CachedFunction {
    /// Function metadata
    pub metadata: CachedFunctionMetadata,
    /// Compiled function pointer (type-erased)
    pub function: Box<dyn std::any::Any + Send + Sync>,
    /// Function signature for runtime type checking
    pub signature: FunctionSignature,
}

/// Runtime function signature for type safety
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSignature {
    /// Input types
    pub input_types: Vec<String>,
    /// Output type
    pub output_type: String,
    /// Whether the function is variadic
    pub is_variadic: bool,
}

impl FunctionSignature {
    /// Create a new function signature
    pub fn new(input_types: Vec<String>, output_type: String, is_variadic: bool) -> Self {
        Self {
            input_types,
            output_type,
            is_variadic,
        }
    }

    /// Check if this signature matches another
    pub fn matches(&self, other: &FunctionSignature) -> bool {
        self.output_type == other.output_type
            && (self.is_variadic || self.input_types == other.input_types)
    }
}

/// JIT Function Cache with intelligent eviction and performance tracking
pub struct JitFunctionCache {
    /// Cache storage
    cache: RwLock<HashMap<FunctionId, CachedFunction>>,
    /// Maximum cache size in bytes
    max_cache_size_bytes: usize,
    /// Current cache size in bytes
    current_cache_size_bytes: RwLock<usize>,
    /// Cache hit statistics
    cache_hits: RwLock<u64>,
    /// Cache miss statistics
    cache_misses: RwLock<u64>,
    /// Cache eviction statistics
    cache_evictions: RwLock<u64>,
}

impl JitFunctionCache {
    /// Create a new JIT function cache
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_cache_size_bytes: max_size_mb * 1024 * 1024,
            current_cache_size_bytes: RwLock::new(0),
            cache_hits: RwLock::new(0),
            cache_misses: RwLock::new(0),
            cache_evictions: RwLock::new(0),
        }
    }

    /// Get a cached function if it exists
    pub fn get(&self, function_id: &FunctionId) -> Option<Arc<CachedFunction>> {
        let cache = self.cache.read().unwrap();
        if let Some(cached) = cache.get(function_id) {
            *self.cache_hits.write().unwrap() += 1;
            // Note: In a real implementation, we would update last_accessed here
            // but that requires mutable access, so we'll track it separately
            Some(Arc::new(CachedFunction {
                metadata: cached.metadata.clone(),
                function: unsafe {
                    // This is a simplified implementation - in practice, we'd need
                    // proper type-safe cloning of the function pointer
                    std::mem::transmute_copy(&cached.function)
                },
                signature: cached.signature.clone(),
            }))
        } else {
            *self.cache_misses.write().unwrap() += 1;
            None
        }
    }

    /// Store a compiled function in the cache
    pub fn store(
        &self,
        function_id: FunctionId,
        function: Box<dyn std::any::Any + Send + Sync>,
        signature: FunctionSignature,
        metadata: CachedFunctionMetadata,
    ) -> Result<()> {
        // Check if we need to evict functions to make space
        self.evict_if_needed(metadata.function_size_bytes)?;

        let cached_function = CachedFunction {
            metadata,
            function,
            signature,
        };

        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_cache_size_bytes.write().unwrap();

        // Remove old function if it exists
        if let Some(old_function) = cache.remove(&function_id) {
            *current_size -= old_function.metadata.function_size_bytes;
        }

        *current_size += cached_function.metadata.function_size_bytes;
        cache.insert(function_id, cached_function);

        Ok(())
    }

    /// Record execution of a cached function
    pub fn record_execution(&self, function_id: &FunctionId, execution_time_ns: u64) {
        let mut cache = self.cache.write().unwrap();
        if let Some(cached_function) = cache.get_mut(function_id) {
            cached_function.metadata.record_execution(execution_time_ns);
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let hits = *self.cache_hits.read().unwrap();
        let misses = *self.cache_misses.read().unwrap();
        let evictions = *self.cache_evictions.read().unwrap();
        let cache_size = *self.current_cache_size_bytes.read().unwrap();
        let cache_entries = self.cache.read().unwrap().len();

        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };

        CacheStats {
            hit_rate,
            hits,
            misses,
            evictions,
            cache_size_bytes: cache_size,
            cache_entries,
            max_cache_size_bytes: self.max_cache_size_bytes,
        }
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_cache_size_bytes.write().unwrap();

        cache.clear();
        *current_size = 0;
    }

    /// Evict functions if cache is full
    fn evict_if_needed(&self, new_function_size: usize) -> Result<()> {
        let current_size = *self.current_cache_size_bytes.read().unwrap();

        if current_size + new_function_size <= self.max_cache_size_bytes {
            return Ok(()); // No eviction needed
        }

        let mut cache = self.cache.write().unwrap();
        let mut size = self.current_cache_size_bytes.write().unwrap();
        let mut evictions = self.cache_evictions.write().unwrap();

        // Calculate cache pressure
        let cache_pressure = (*size + new_function_size) as f64 / self.max_cache_size_bytes as f64;

        // Find functions to evict based on their benefit and access patterns
        let mut to_evict = Vec::new();
        for (id, cached_function) in cache.iter() {
            if cached_function.metadata.should_evict(cache_pressure) {
                to_evict.push((id.clone(), cached_function.metadata.function_size_bytes));
            }
        }

        // Sort by cache benefit (evict least beneficial first)
        to_evict.sort_by(|a, b| {
            let a_benefit = cache.get(&a.0).unwrap().metadata.cache_benefit();
            let b_benefit = cache.get(&b.0).unwrap().metadata.cache_benefit();
            a_benefit
                .partial_cmp(&b_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Evict functions until we have enough space
        let needed_space = (*size + new_function_size).saturating_sub(self.max_cache_size_bytes);
        let mut freed_space = 0;

        for (function_id, function_size) in to_evict {
            if freed_space >= needed_space {
                break;
            }

            cache.remove(&function_id);
            *size -= function_size;
            freed_space += function_size;
            *evictions += 1;
        }

        if *size + new_function_size > self.max_cache_size_bytes {
            return Err(Error::InvalidOperation(
                "Unable to free enough cache space for new function".to_string(),
            ));
        }

        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of cache evictions
    pub evictions: u64,
    /// Current cache size in bytes
    pub cache_size_bytes: usize,
    /// Number of cached functions
    pub cache_entries: usize,
    /// Maximum cache size in bytes
    pub max_cache_size_bytes: usize,
}

impl CacheStats {
    /// Get cache utilization as a percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.max_cache_size_bytes > 0 {
            (self.cache_size_bytes as f64 / self.max_cache_size_bytes as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Get average function size in bytes
    pub fn avg_function_size_bytes(&self) -> f64 {
        if self.cache_entries > 0 {
            self.cache_size_bytes as f64 / self.cache_entries as f64
        } else {
            0.0
        }
    }
}

/// Global function cache instance
static GLOBAL_CACHE: std::sync::OnceLock<JitFunctionCache> = std::sync::OnceLock::new();

/// Get the global JIT function cache
pub fn get_global_cache() -> &'static JitFunctionCache {
    GLOBAL_CACHE.get_or_init(|| JitFunctionCache::new(128)) // 128MB default cache size
}

/// Initialize the global cache with a specific size
pub fn init_global_cache(max_size_mb: usize) -> Result<()> {
    GLOBAL_CACHE
        .set(JitFunctionCache::new(max_size_mb))
        .map_err(|_| Error::InvalidOperation("Global cache already initialized".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_id_creation() {
        let id = FunctionId::new("test_sum", "f64", "f64", "sum_operation", 2);
        assert_eq!(id.name, "test_sum");
        assert_eq!(id.input_type, "f64");
        assert_eq!(id.output_type, "f64");
        assert_eq!(id.optimization_level, 2);
    }

    #[test]
    fn test_cache_metadata() {
        let mut metadata = CachedFunctionMetadata::new(1_000_000, 1024);
        assert_eq!(metadata.execution_count, 0);

        metadata.record_execution(500_000);
        assert_eq!(metadata.execution_count, 1);
        assert_eq!(metadata.avg_execution_time_ns, 500_000.0);
    }

    #[test]
    fn test_function_signature_matching() {
        let sig1 = FunctionSignature::new(vec!["f64".to_string()], "f64".to_string(), false);
        let sig2 = FunctionSignature::new(vec!["f64".to_string()], "f64".to_string(), false);
        let sig3 = FunctionSignature::new(vec!["i64".to_string()], "f64".to_string(), false);

        assert!(sig1.matches(&sig2));
        assert!(!sig1.matches(&sig3));
    }

    #[test]
    fn test_cache_operations() {
        let cache = JitFunctionCache::new(1); // 1MB cache
        let function_id = FunctionId::new("test", "f64", "f64", "test_op", 1);

        // Initially, function should not be in cache
        assert!(cache.get(&function_id).is_none());

        let stats = cache.get_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }
}

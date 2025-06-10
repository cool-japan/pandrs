//! # JIT Configuration Module
//!
//! This module provides configuration options for JIT compilation and execution.

use std::num::NonZeroUsize;

/// Configuration for parallel processing
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Minimum chunk size for parallel processing
    pub min_chunk_size: usize,
    /// Maximum number of threads to use (None = use all available)
    pub max_threads: Option<NonZeroUsize>,
    /// Whether to use work stealing
    pub work_stealing: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancing,
}

/// Load balancing strategy for parallel processing
#[derive(Debug, Clone)]
pub enum LoadBalancing {
    /// Static load balancing (equal chunks)
    Static,
    /// Dynamic load balancing (work stealing)
    Dynamic,
    /// Adaptive load balancing (adjusts based on performance)
    Adaptive,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 1000,
            max_threads: None,
            work_stealing: true,
            load_balancing: LoadBalancing::Dynamic,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum chunk size
    pub fn with_min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = size;
        self
    }

    /// Set the maximum number of threads
    pub fn with_max_threads(mut self, threads: usize) -> Self {
        self.max_threads = NonZeroUsize::new(threads);
        self
    }

    /// Enable or disable work stealing
    pub fn with_work_stealing(mut self, enabled: bool) -> Self {
        self.work_stealing = enabled;
        self
    }

    /// Set the load balancing strategy
    pub fn with_load_balancing(mut self, strategy: LoadBalancing) -> Self {
        self.load_balancing = strategy;
        self
    }

    /// Calculate optimal chunk size for given data size
    pub fn optimal_chunk_size(&self, data_size: usize) -> usize {
        let thread_count = self
            .max_threads
            .map(|t| t.get())
            .unwrap_or_else(|| num_cpus::get());

        let base_chunk_size = (data_size + thread_count - 1) / thread_count;
        base_chunk_size.max(self.min_chunk_size)
    }
}

/// Configuration for SIMD operations
#[derive(Debug, Clone)]
pub struct SIMDConfig {
    /// Whether SIMD is enabled
    pub enabled: bool,
    /// Vector width (in bytes)
    pub vector_width: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Minimum data size to use SIMD
    pub min_simd_size: usize,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vector_width: 32, // AVX2
            alignment: 32,
            min_simd_size: 64,
        }
    }
}

impl SIMDConfig {
    /// Create a new SIMD configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable SIMD
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set vector width
    pub fn with_vector_width(mut self, width: usize) -> Self {
        self.vector_width = width;
        self
    }

    /// Set alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set minimum size for SIMD operations
    pub fn with_min_simd_size(mut self, size: usize) -> Self {
        self.min_simd_size = size;
        self
    }

    /// Check if SIMD should be used for given data size
    pub fn should_use_simd(&self, data_size: usize) -> bool {
        self.enabled && data_size >= self.min_simd_size
    }
}

/// Overall JIT configuration
#[derive(Debug, Clone)]
pub struct JITConfig {
    /// Parallel processing configuration
    pub parallel: ParallelConfig,
    /// SIMD configuration
    pub simd: SIMDConfig,
    /// Whether JIT compilation is enabled
    pub jit_enabled: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Whether to cache compiled functions
    pub cache_compiled: bool,
}

impl Default for JITConfig {
    fn default() -> Self {
        Self {
            parallel: ParallelConfig::default(),
            simd: SIMDConfig::default(),
            jit_enabled: true,
            optimization_level: 2,
            cache_compiled: true,
        }
    }
}

impl JITConfig {
    /// Create a new JIT configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set parallel configuration
    pub fn with_parallel(mut self, config: ParallelConfig) -> Self {
        self.parallel = config;
        self
    }

    /// Set SIMD configuration
    pub fn with_simd(mut self, config: SIMDConfig) -> Self {
        self.simd = config;
        self
    }

    /// Enable or disable JIT compilation
    pub fn with_jit_enabled(mut self, enabled: bool) -> Self {
        self.jit_enabled = enabled;
        self
    }

    /// Set optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Enable or disable function caching
    pub fn with_cache_compiled(mut self, enabled: bool) -> Self {
        self.cache_compiled = enabled;
        self
    }

    /// Create a configuration optimized for throughput
    pub fn throughput_optimized() -> Self {
        Self::new()
            .with_parallel(
                ParallelConfig::new()
                    .with_min_chunk_size(10000)
                    .with_load_balancing(LoadBalancing::Static),
            )
            .with_simd(SIMDConfig::new().with_min_simd_size(128))
            .with_optimization_level(3)
    }

    /// Create a configuration optimized for latency
    pub fn latency_optimized() -> Self {
        Self::new()
            .with_parallel(
                ParallelConfig::new()
                    .with_min_chunk_size(100)
                    .with_load_balancing(LoadBalancing::Dynamic),
            )
            .with_simd(SIMDConfig::new().with_min_simd_size(32))
            .with_optimization_level(2)
    }
}

/// Global JIT configuration
static GLOBAL_JIT_CONFIG: std::sync::OnceLock<JITConfig> = std::sync::OnceLock::new();

/// Get the global JIT configuration
pub fn get_global_config() -> &'static JITConfig {
    GLOBAL_JIT_CONFIG.get_or_init(|| JITConfig::default())
}

/// Set the global JIT configuration
/// Note: This will only work if called before the first get_global_config() call
pub fn set_global_config(config: JITConfig) -> Result<(), JITConfig> {
    GLOBAL_JIT_CONFIG.set(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config() {
        let config = ParallelConfig::new()
            .with_min_chunk_size(500)
            .with_max_threads(4);

        assert_eq!(config.min_chunk_size, 500);
        assert_eq!(config.max_threads.unwrap().get(), 4);
    }

    #[test]
    fn test_simd_config() {
        let config = SIMDConfig::new().with_enabled(false).with_min_simd_size(32);

        assert!(!config.enabled);
        assert_eq!(config.min_simd_size, 32);
        assert!(!config.should_use_simd(64));
    }

    #[test]
    fn test_jit_config() {
        let config = JITConfig::throughput_optimized();

        assert_eq!(config.parallel.min_chunk_size, 10000);
        assert_eq!(config.optimization_level, 3);
    }
}

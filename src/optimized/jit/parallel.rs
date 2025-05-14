//! Parallel JIT execution
//!
//! This module provides support for parallel execution of JIT-compiled functions
//! using Rayon for multi-threading, allowing for improved performance on multi-core CPUs.

use std::sync::Arc;
use std::marker::PhantomData;

use rayon::prelude::*;

use super::jit_core::{JitCompilable, GenericJitCompilable, JitResult};
use super::types::{JitType, JitNumeric, TypedVector, NumericValue};

/// Configuration for parallel execution
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Minimum chunk size for parallel processing
    pub min_chunk_size: usize,
    /// Maximum number of threads to use (None for auto-detection)
    pub max_threads: Option<usize>,
    /// Whether to use thread-local storage for intermediate results
    pub use_thread_local: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 1000,
            max_threads: None,
            use_thread_local: true,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the minimum chunk size
    pub fn with_min_chunk_size(mut self, min_chunk_size: usize) -> Self {
        self.min_chunk_size = min_chunk_size;
        self
    }
    
    /// Set the maximum number of threads
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = Some(max_threads);
        self
    }
    
    /// Set whether to use thread-local storage
    pub fn with_thread_local(mut self, use_thread_local: bool) -> Self {
        self.use_thread_local = use_thread_local;
        self
    }
    
    /// Get the number of threads to use
    pub fn threads(&self) -> usize {
        match self.max_threads {
            Some(n) => n,
            None => rayon::current_num_threads(),
        }
    }
    
    /// Calculate chunk size based on input size
    pub fn chunk_size(&self, input_size: usize) -> usize {
        let threads = self.threads();
        if input_size <= self.min_chunk_size {
            // Too small to parallelize effectively
            input_size
        } else {
            // Divide input into chunks, but ensure each chunk is at least min_chunk_size
            std::cmp::max(
                self.min_chunk_size,
                (input_size + threads - 1) / threads,
            )
        }
    }
    
    /// Determine if parallelization should be used
    pub fn should_parallelize(&self, input_size: usize) -> bool {
        input_size > self.min_chunk_size && self.threads() > 1
    }
}

/// A JIT function that uses parallel execution for improved performance
#[derive(Clone)]
pub struct ParallelJitFunction<T, F, R>
where
    T: JitType + Send + Sync,
    F: Fn(Vec<T>) -> R + Send + Sync,
    R: Send + Sync,
{
    /// Function name
    name: String,
    /// Native implementation for fallback
    native_fn: Arc<F>,
    /// Parallel implementation
    parallel_fn: Arc<dyn Fn(&[T]) -> R + Send + Sync>,
    /// Parallel configuration
    config: ParallelConfig,
    /// Phantom data for type parameters
    _marker: PhantomData<(T, R)>,
}

impl<T, F, R> ParallelJitFunction<T, F, R>
where
    T: JitType + Send + Sync + 'static,
    F: Fn(Vec<T>) -> R + Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    /// Create a new parallel JIT function
    pub fn new(
        name: impl Into<String>,
        native_fn: F,
        parallel_fn: impl Fn(&[T]) -> R + Send + Sync + 'static,
        config: ParallelConfig,
    ) -> Self {
        Self {
            name: name.into(),
            native_fn: Arc::new(native_fn),
            parallel_fn: Arc::new(parallel_fn),
            config,
            _marker: PhantomData,
        }
    }
    
    /// Create a new parallel JIT function with default configuration
    pub fn with_default_config(
        name: impl Into<String>,
        native_fn: F,
        parallel_fn: impl Fn(&[T]) -> R + Send + Sync + 'static,
    ) -> Self {
        Self::new(name, native_fn, parallel_fn, ParallelConfig::default())
    }
    
    #[cfg(feature = "jit")]
    /// Compile with JIT (placeholder for now)
    pub fn with_jit(self) -> JitResult<Self> {
        // In a real implementation, this would compile the parallel function
        Ok(self)
    }
}

impl<T, F> JitCompilable<Vec<T>, T> for ParallelJitFunction<T, F, T>
where
    T: JitType + Send + Sync + std::iter::Sum,
    F: Fn(Vec<T>) -> T + Send + Sync,
{
    fn execute(&self, args: Vec<T>) -> T {
        // Use parallel implementation if appropriate
        if self.config.should_parallelize(args.len()) {
            (self.parallel_fn)(&args)
        } else {
            // Fall back to native implementation for small inputs
            (self.native_fn)(args)
        }
    }
}

// Helper functions for common parallel operations

/// Create a parallel sum function for f64 values
pub fn parallel_sum_f64(config: Option<ParallelConfig>) -> impl JitCompilable<Vec<f64>, f64> {
    let config = config.unwrap_or_default();
    
    // Native implementation
    let native_fn = |values: Vec<f64>| -> f64 {
        values.iter().sum()
    };
    
    // Parallel implementation
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let chunk_size = config.chunk_size(values.len());
        
        // Use Rayon's parallel iterator to process chunks in parallel
        values
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum()
    };
    
    ParallelJitFunction::new("parallel_sum_f64", native_fn, parallel_fn, config)
}

/// Create a parallel mean function for f64 values
pub fn parallel_mean_f64(config: Option<ParallelConfig>) -> impl JitCompilable<Vec<f64>, f64> {
    let config = config.unwrap_or_default();
    
    // Native implementation
    let native_fn = |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    };
    
    // Parallel implementation
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let n = values.len();
        let chunk_size = config.chunk_size(n);
        
        // Use Rayon's parallel iterator to compute partial sums
        let sum = values
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum::<f64>();
        
        sum / n as f64
    };
    
    ParallelJitFunction::new("parallel_mean_f64", native_fn, parallel_fn, config)
}

/// Create a parallel standard deviation function for f64 values
pub fn parallel_std_f64(config: Option<ParallelConfig>) -> impl JitCompilable<Vec<f64>, f64> {
    let config = config.unwrap_or_default();
    
    // Native implementation
    let native_fn = |values: Vec<f64>| -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    };
    
    // Parallel implementation using a two-pass algorithm
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        
        let n = values.len();
        let chunk_size = config.chunk_size(n);
        
        // First pass: compute the mean in parallel
        let mean = values
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum::<f64>() / n as f64;
        
        // Second pass: compute the variance in parallel
        let variance = values
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter()
                    .map(|&v| (v - mean).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>() / (n - 1) as f64;
        
        variance.sqrt()
    };
    
    ParallelJitFunction::new("parallel_std_f64", native_fn, parallel_fn, config)
}

/// Create a parallel min function for f64 values
pub fn parallel_min_f64(config: Option<ParallelConfig>) -> impl JitCompilable<Vec<f64>, f64> {
    let config = config.unwrap_or_default();
    
    // Native implementation
    let native_fn = |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        *values.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap()
    };
    
    // Parallel implementation
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let chunk_size = config.chunk_size(values.len());
        
        // Use Rayon's parallel iterator to find the minimum in each chunk
        values
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(f64::INFINITY)
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::NAN)
    };
    
    ParallelJitFunction::new("parallel_min_f64", native_fn, parallel_fn, config)
}

/// Create a parallel max function for f64 values
pub fn parallel_max_f64(config: Option<ParallelConfig>) -> impl JitCompilable<Vec<f64>, f64> {
    let config = config.unwrap_or_default();
    
    // Native implementation
    let native_fn = |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        *values.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap()
    };
    
    // Parallel implementation
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        
        let chunk_size = config.chunk_size(values.len());
        
        // Use Rayon's parallel iterator to find the maximum in each chunk
        values
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .copied()
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::NAN)
    };
    
    ParallelJitFunction::new("parallel_max_f64", native_fn, parallel_fn, config)
}

/// Create a parallel custom function
pub fn parallel_custom<F, M, R>(
    name: impl Into<String>,
    native_fn: F,
    map_fn: M,
    reduce_fn: R,
    config: Option<ParallelConfig>,
) -> impl JitCompilable<Vec<f64>, f64>
where
    F: Fn(Vec<f64>) -> f64 + Send + Sync + 'static,
    M: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
    R: Fn(Vec<f64>) -> f64 + Send + Sync + Clone + 'static,
{
    let config = config.unwrap_or_default();
    
    // Parallel implementation using map-reduce pattern
    let parallel_fn = move |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let chunk_size = config.chunk_size(values.len());
        let map_fn_clone = map_fn.clone();
        let reduce_fn_clone = reduce_fn.clone();
        
        // Map phase: process chunks in parallel
        let mapped = values
            .par_chunks(chunk_size)
            .map(|chunk| map_fn_clone(chunk))
            .collect::<Vec<_>>();
        
        // Reduce phase: combine results
        reduce_fn_clone(mapped)
    };
    
    ParallelJitFunction::new(name, native_fn, parallel_fn, config)
}
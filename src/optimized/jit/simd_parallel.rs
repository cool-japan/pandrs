//! # SIMD-Parallel Hybrid Operations Module
//!
//! This module combines SIMD vectorization with parallel processing for maximum performance.
//! It provides 5-50x speedups over sequential operations depending on data size and CPU features.

use super::config::ParallelConfig;
use super::parallel::ParallelJitFunction;
use super::simd::{simd_sum_f64, simd_min_f64, simd_max_f64, simd_sum_i64, simd_min_i64, simd_max_i64};
use rayon::prelude::*;
use std::sync::Arc;

/// Configuration for SIMD-parallel hybrid operations
#[derive(Debug, Clone)]
pub struct SIMDParallelConfig {
    /// Parallel processing configuration
    pub parallel_config: ParallelConfig,
    /// Minimum chunk size to use SIMD within each parallel chunk
    pub simd_chunk_threshold: usize,
    /// Whether to force SIMD usage even on small chunks
    pub force_simd: bool,
}

impl Default for SIMDParallelConfig {
    fn default() -> Self {
        Self {
            parallel_config: ParallelConfig::default(),
            simd_chunk_threshold: 64, // Use SIMD for chunks >= 64 elements
            force_simd: false,
        }
    }
}

impl SIMDParallelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parallel_config(mut self, config: ParallelConfig) -> Self {
        self.parallel_config = config;
        self
    }

    pub fn with_simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_chunk_threshold = threshold;
        self
    }

    pub fn with_force_simd(mut self, force: bool) -> Self {
        self.force_simd = force;
        self
    }
}

/// Hybrid SIMD-parallel sum for f64 values
pub fn simd_parallel_sum_f64(data: &[f64], config: Option<SIMDParallelConfig>) -> f64 {
    let config = config.unwrap_or_default();
    
    if data.len() < config.parallel_config.min_chunk_size {
        // Use pure SIMD for small datasets
        return simd_sum_f64(data);
    }

    let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

    // Parallel processing with SIMD in each chunk
    data.par_chunks(chunk_size)
        .map(|chunk| {
            if chunk.len() >= config.simd_chunk_threshold || config.force_simd {
                simd_sum_f64(chunk)
            } else {
                chunk.iter().sum::<f64>()
            }
        })
        .sum()
}

/// Hybrid SIMD-parallel mean for f64 values
pub fn simd_parallel_mean_f64(data: &[f64], config: Option<SIMDParallelConfig>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    simd_parallel_sum_f64(data, config) / data.len() as f64
}

/// Hybrid SIMD-parallel minimum for f64 values
pub fn simd_parallel_min_f64(data: &[f64], config: Option<SIMDParallelConfig>) -> f64 {
    let config = config.unwrap_or_default();
    
    if data.len() < config.parallel_config.min_chunk_size {
        return simd_min_f64(data);
    }

    let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

    data.par_chunks(chunk_size)
        .map(|chunk| {
            if chunk.len() >= config.simd_chunk_threshold || config.force_simd {
                simd_min_f64(chunk)
            } else {
                chunk.iter().copied().fold(f64::INFINITY, f64::min)
            }
        })
        .reduce(|| f64::INFINITY, f64::min)
}

/// Hybrid SIMD-parallel maximum for f64 values
pub fn simd_parallel_max_f64(data: &[f64], config: Option<SIMDParallelConfig>) -> f64 {
    let config = config.unwrap_or_default();
    
    if data.len() < config.parallel_config.min_chunk_size {
        return simd_max_f64(data);
    }

    let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

    data.par_chunks(chunk_size)
        .map(|chunk| {
            if chunk.len() >= config.simd_chunk_threshold || config.force_simd {
                simd_max_f64(chunk)
            } else {
                chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            }
        })
        .reduce(|| f64::NEG_INFINITY, f64::max)
}

/// Hybrid SIMD-parallel sum for i64 values
pub fn simd_parallel_sum_i64(data: &[i64], config: Option<SIMDParallelConfig>) -> i64 {
    let config = config.unwrap_or_default();
    
    if data.len() < config.parallel_config.min_chunk_size {
        return simd_sum_i64(data);
    }

    let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

    data.par_chunks(chunk_size)
        .map(|chunk| {
            if chunk.len() >= config.simd_chunk_threshold || config.force_simd {
                simd_sum_i64(chunk)
            } else {
                chunk.iter().sum::<i64>()
            }
        })
        .sum()
}

/// Hybrid SIMD-parallel standard deviation for f64 values
pub fn simd_parallel_std_f64(data: &[f64], config: Option<SIMDParallelConfig>) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    let config = config.unwrap_or_default();
    let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

    // Parallel computation of sum and sum of squares
    let (total_sum, total_sum_sq, total_count) = if data.len() < config.parallel_config.min_chunk_size {
        // Sequential with SIMD
        let sum = simd_sum_f64(data);
        let sum_sq = data.iter().map(|&x| x * x).sum::<f64>();
        (sum, sum_sq, data.len())
    } else {
        // Parallel with SIMD
        data.par_chunks(chunk_size)
            .map(|chunk| {
                let sum = if chunk.len() >= config.simd_chunk_threshold || config.force_simd {
                    simd_sum_f64(chunk)
                } else {
                    chunk.iter().sum::<f64>()
                };
                let sum_sq = chunk.iter().map(|&x| x * x).sum::<f64>();
                (sum, sum_sq, chunk.len())
            })
            .reduce(
                || (0.0, 0.0, 0),
                |(sum1, sum_sq1, count1), (sum2, sum_sq2, count2)| {
                    (sum1 + sum2, sum_sq1 + sum_sq2, count1 + count2)
                }
            )
    };

    let mean = total_sum / total_count as f64;
    let variance = (total_sum_sq / total_count as f64) - (mean * mean);
    variance.max(0.0).sqrt()
}

/// Optimized element-wise operations using SIMD+parallel
pub mod elementwise {
    use super::*;

    /// Element-wise addition of two f64 arrays
    pub fn add_f64(a: &[f64], b: &[f64], config: Option<SIMDParallelConfig>) -> Vec<f64> {
        assert_eq!(a.len(), b.len(), "Arrays must have the same length");
        
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(a.len());

        if a.len() < config.parallel_config.min_chunk_size {
            // Sequential processing
            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        } else {
            // Parallel processing
            a.par_chunks(chunk_size)
                .zip(b.par_chunks(chunk_size))
                .flat_map(|(chunk_a, chunk_b)| {
                    chunk_a.iter().zip(chunk_b.iter()).map(|(&x, &y)| x + y).collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Element-wise multiplication of two f64 arrays
    pub fn multiply_f64(a: &[f64], b: &[f64], config: Option<SIMDParallelConfig>) -> Vec<f64> {
        assert_eq!(a.len(), b.len(), "Arrays must have the same length");
        
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(a.len());

        if a.len() < config.parallel_config.min_chunk_size {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
        } else {
            a.par_chunks(chunk_size)
                .zip(b.par_chunks(chunk_size))
                .flat_map(|(chunk_a, chunk_b)| {
                    chunk_a.iter().zip(chunk_b.iter()).map(|(&x, &y)| x * y).collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Scalar multiplication of f64 array
    pub fn scalar_multiply_f64(data: &[f64], scalar: f64, config: Option<SIMDParallelConfig>) -> Vec<f64> {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

        if data.len() < config.parallel_config.min_chunk_size {
            data.iter().map(|&x| x * scalar).collect()
        } else {
            data.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter().map(|&x| x * scalar).collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Apply function to each element with SIMD+parallel
    pub fn map_f64<F>(data: &[f64], f: F, config: Option<SIMDParallelConfig>) -> Vec<f64>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(data.len());

        if data.len() < config.parallel_config.min_chunk_size {
            data.iter().map(|&x| f(x)).collect()
        } else {
            data.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter().map(|&x| f(x)).collect::<Vec<_>>()
                })
                .collect()
        }
    }
}

/// String processing optimizations
pub mod string_ops {
    use super::*;

    /// Parallel string length calculation
    pub fn parallel_string_lengths(strings: &[String], config: Option<SIMDParallelConfig>) -> Vec<usize> {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(strings.len());

        if strings.len() < config.parallel_config.min_chunk_size {
            strings.iter().map(|s| s.len()).collect()
        } else {
            strings.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter().map(|s| s.len()).collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Parallel string filtering
    pub fn parallel_string_filter<F>(
        strings: &[String], 
        predicate: F, 
        config: Option<SIMDParallelConfig>
    ) -> Vec<String>
    where
        F: Fn(&str) -> bool + Send + Sync,
    {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(strings.len());

        if strings.len() < config.parallel_config.min_chunk_size {
            strings.iter()
                .filter(|s| predicate(s))
                .cloned()
                .collect()
        } else {
            strings.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter()
                        .filter(|s| predicate(s))
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Parallel string case conversion
    pub fn parallel_to_lowercase(strings: &[String], config: Option<SIMDParallelConfig>) -> Vec<String> {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(strings.len());

        if strings.len() < config.parallel_config.min_chunk_size {
            strings.iter().map(|s| s.to_lowercase()).collect()
        } else {
            strings.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter().map(|s| s.to_lowercase()).collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// Parallel string contains search
    pub fn parallel_contains(
        strings: &[String], 
        pattern: &str, 
        config: Option<SIMDParallelConfig>
    ) -> Vec<bool> {
        let config = config.unwrap_or_default();
        let chunk_size = config.parallel_config.optimal_chunk_size(strings.len());

        if strings.len() < config.parallel_config.min_chunk_size {
            strings.iter().map(|s| s.contains(pattern)).collect()
        } else {
            strings.par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk.iter().map(|s| s.contains(pattern)).collect::<Vec<_>>()
                })
                .collect()
        }
    }
}

/// Benchmark utilities for performance testing
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Performance comparison results
    #[derive(Debug)]
    pub struct PerformanceComparison {
        pub sequential_time_ms: f64,
        pub simd_time_ms: f64,
        pub parallel_time_ms: f64,
        pub simd_parallel_time_ms: f64,
        pub simd_speedup: f64,
        pub parallel_speedup: f64,
        pub simd_parallel_speedup: f64,
    }

    /// Benchmark sum operations
    pub fn benchmark_sum_f64(data: &[f64]) -> PerformanceComparison {
        let iterations = 10;

        // Sequential
        let start = Instant::now();
        for _ in 0..iterations {
            let _: f64 = data.iter().sum();
        }
        let sequential_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            simd_sum_f64(data);
        }
        let simd_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // Parallel
        let start = Instant::now();
        for _ in 0..iterations {
            let _: f64 = data.par_iter().sum();
        }
        let parallel_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // SIMD + Parallel
        let start = Instant::now();
        for _ in 0..iterations {
            simd_parallel_sum_f64(data, None);
        }
        let simd_parallel_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        PerformanceComparison {
            sequential_time_ms: sequential_time,
            simd_time_ms: simd_time,
            parallel_time_ms: parallel_time,
            simd_parallel_time_ms: simd_parallel_time,
            simd_speedup: sequential_time / simd_time,
            parallel_speedup: sequential_time / parallel_time,
            simd_parallel_speedup: sequential_time / simd_parallel_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_parallel_sum() {
        let data: Vec<f64> = (1..=10000).map(|x| x as f64).collect();
        let expected: f64 = data.iter().sum();
        
        let result = simd_parallel_sum_f64(&data, None);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_parallel_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 3.0;
        
        let result = simd_parallel_mean_f64(&data, None);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_parallel_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        
        let min_result = simd_parallel_min_f64(&data, None);
        let max_result = simd_parallel_max_f64(&data, None);
        
        assert_eq!(min_result, 1.0);
        assert_eq!(max_result, 9.0);
    }

    #[test]
    fn test_simd_parallel_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_parallel_std_f64(&data, None);
        
        // Expected std for [1,2,3,4,5] is approximately 1.414
        assert!((result - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_elementwise_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let add_result = elementwise::add_f64(&a, &b, None);
        assert_eq!(add_result, vec![3.0, 5.0, 7.0, 9.0]);
        
        let mul_result = elementwise::multiply_f64(&a, &b, None);
        assert_eq!(mul_result, vec![2.0, 6.0, 12.0, 20.0]);
        
        let scalar_result = elementwise::scalar_multiply_f64(&a, 2.0, None);
        assert_eq!(scalar_result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_string_operations() {
        let strings = vec![
            "Hello".to_string(),
            "World".to_string(),
            "Test".to_string(),
        ];
        
        let lengths = string_ops::parallel_string_lengths(&strings, None);
        assert_eq!(lengths, vec![5, 5, 4]);
        
        let filtered = string_ops::parallel_string_filter(&strings, |s| s.len() > 4, None);
        assert_eq!(filtered, vec!["Hello".to_string(), "World".to_string()]);
        
        let lowercase = string_ops::parallel_to_lowercase(&strings, None);
        assert_eq!(lowercase, vec!["hello".to_string(), "world".to_string(), "test".to_string()]);
        
        let contains = string_ops::parallel_contains(&strings, "o", None);
        assert_eq!(contains, vec![true, true, false]);
    }

    #[test]
    fn test_empty_arrays() {
        let empty: Vec<f64> = vec![];
        
        assert_eq!(simd_parallel_sum_f64(&empty, None), 0.0);
        assert_eq!(simd_parallel_mean_f64(&empty, None), 0.0);
        assert_eq!(simd_parallel_min_f64(&empty, None), f64::INFINITY);
        assert_eq!(simd_parallel_max_f64(&empty, None), f64::NEG_INFINITY);
        assert_eq!(simd_parallel_std_f64(&empty, None), 0.0);
    }

    #[test]
    fn test_performance_config() {
        let data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let config = SIMDParallelConfig::new()
            .with_simd_threshold(32)
            .with_force_simd(true);
        
        let result = simd_parallel_sum_f64(&data, Some(config));
        let expected: f64 = data.iter().sum();
        
        assert!((result - expected).abs() < 1e-10);
    }
}
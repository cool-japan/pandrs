//! # Parallel JIT Operations Module
//!
//! This module provides parallel implementations of common aggregation operations.

use super::config::ParallelConfig;
use super::core::{JitCompilable, JitFunction};
use super::{JitError, JitResult};
use rayon::prelude::*;
use std::sync::Arc;

/// A parallel JIT function that can execute operations across multiple threads
pub struct ParallelJitFunction<F, T, R> {
    /// The function to execute on each chunk
    pub map_fn: F,
    /// The function to reduce results from multiple chunks
    pub reduce_fn: Box<dyn Fn(Vec<R>) -> R + Send + Sync>,
    /// Function name for debugging
    pub name: String,
    /// Parallel configuration
    pub config: ParallelConfig,
    /// Marker for types
    _phantom: std::marker::PhantomData<(T, R)>,
}

impl<F, T, R> ParallelJitFunction<F, T, R>
where
    F: Fn(&[T]) -> R + Send + Sync + Clone,
    T: Send + Sync,
    R: Send + Sync,
{
    /// Create a new parallel JIT function
    pub fn new<RF>(
        name: impl Into<String>,
        map_fn: F,
        reduce_fn: RF,
        config: Option<ParallelConfig>,
    ) -> Self
    where
        RF: Fn(Vec<R>) -> R + Send + Sync + 'static,
    {
        Self {
            map_fn,
            reduce_fn: Box::new(reduce_fn),
            name: name.into(),
            config: config.unwrap_or_default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Execute the parallel function
    pub fn execute(&self, data: &[T]) -> R {
        if data.len() < self.config.min_chunk_size {
            // Use sequential execution for small datasets
            return (self.map_fn)(data);
        }

        let chunk_size = self.config.optimal_chunk_size(data.len());

        // Execute in parallel chunks
        let results: Vec<R> = data
            .par_chunks(chunk_size)
            .map(|chunk| (self.map_fn)(chunk))
            .collect();

        // Reduce the results
        (self.reduce_fn)(results)
    }
}

/// Parallel sum for f64 values
pub fn parallel_sum_f64(
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<impl Fn(&[f64]) -> f64 + Send + Sync + Clone, f64, f64> {
    ParallelJitFunction::new(
        "parallel_sum_f64",
        |chunk: &[f64]| -> f64 {
            // Use Kahan summation for numerical stability
            let mut sum = 0.0;
            let mut c = 0.0;
            for &value in chunk {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            sum
        },
        |partial_sums: Vec<f64>| -> f64 {
            // Combine partial sums with Kahan summation
            let mut total = 0.0;
            let mut c = 0.0;
            for sum in partial_sums {
                let y = sum - c;
                let t = total + y;
                c = (t - total) - y;
                total = t;
            }
            total
        },
        config,
    )
}

/// Parallel mean for f64 values
pub fn parallel_mean_f64(
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<impl Fn(&[f64]) -> (f64, usize) + Send + Sync + Clone, f64, (f64, usize)> {
    ParallelJitFunction::new(
        "parallel_mean_f64",
        |chunk: &[f64]| -> (f64, usize) {
            if chunk.is_empty() {
                return (0.0, 0);
            }
            let mut sum = 0.0;
            let mut c = 0.0;
            for &value in chunk {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            (sum, chunk.len())
        },
        |partial_results: Vec<(f64, usize)>| -> (f64, usize) {
            let mut total_sum = 0.0;
            let mut total_count = 0;
            let mut c = 0.0;

            for (sum, count) in partial_results {
                let y = sum - c;
                let t = total_sum + y;
                c = (t - total_sum) - y;
                total_sum = t;
                total_count += count;
            }

            (total_sum, total_count)
        },
        config,
    )
}

/// Execute parallel mean and return just the mean value
pub fn parallel_mean_f64_value(data: &[f64], config: Option<ParallelConfig>) -> f64 {
    let mean_func = parallel_mean_f64(config);
    let (sum, count) = mean_func.execute(data);
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Parallel standard deviation for f64 values
pub fn parallel_std_f64(
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<
    impl Fn(&[f64]) -> (f64, f64, usize) + Send + Sync + Clone,
    f64,
    (f64, f64, usize),
> {
    ParallelJitFunction::new(
        "parallel_std_f64",
        |chunk: &[f64]| -> (f64, f64, usize) {
            if chunk.is_empty() {
                return (0.0, 0.0, 0);
            }

            // Calculate sum and sum of squares
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut c1 = 0.0;
            let mut c2 = 0.0;

            for &value in chunk {
                // Sum with Kahan summation
                let y1 = value - c1;
                let t1 = sum + y1;
                c1 = (t1 - sum) - y1;
                sum = t1;

                // Sum of squares with Kahan summation
                let value_sq = value * value;
                let y2 = value_sq - c2;
                let t2 = sum_sq + y2;
                c2 = (t2 - sum_sq) - y2;
                sum_sq = t2;
            }

            (sum, sum_sq, chunk.len())
        },
        |partial_results: Vec<(f64, f64, usize)>| -> (f64, f64, usize) {
            let mut total_sum = 0.0;
            let mut total_sum_sq = 0.0;
            let mut total_count = 0;
            let mut c1 = 0.0;
            let mut c2 = 0.0;

            for (sum, sum_sq, count) in partial_results {
                // Combine sums
                let y1 = sum - c1;
                let t1 = total_sum + y1;
                c1 = (t1 - total_sum) - y1;
                total_sum = t1;

                // Combine sum of squares
                let y2 = sum_sq - c2;
                let t2 = total_sum_sq + y2;
                c2 = (t2 - total_sum_sq) - y2;
                total_sum_sq = t2;

                total_count += count;
            }

            (total_sum, total_sum_sq, total_count)
        },
        config,
    )
}

/// Execute parallel standard deviation and return just the std value
pub fn parallel_std_f64_value(data: &[f64], config: Option<ParallelConfig>) -> f64 {
    let std_func = parallel_std_f64(config);
    let (sum, sum_sq, count) = std_func.execute(data);

    if count <= 1 {
        return 0.0;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);
    variance.max(0.0).sqrt() // Ensure non-negative due to floating point errors
}

/// Parallel variance for f64 values
pub fn parallel_var_f64(data: &[f64], config: Option<ParallelConfig>) -> f64 {
    let std_func = parallel_std_f64(config);
    let (sum, sum_sq, count) = std_func.execute(data);

    if count <= 1 {
        return 0.0;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);
    variance.max(0.0)
}

/// Parallel minimum for f64 values
pub fn parallel_min_f64(
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<impl Fn(&[f64]) -> f64 + Send + Sync + Clone, f64, f64> {
    ParallelJitFunction::new(
        "parallel_min_f64",
        |chunk: &[f64]| -> f64 { chunk.iter().copied().fold(f64::INFINITY, f64::min) },
        |partial_mins: Vec<f64>| -> f64 { partial_mins.into_iter().fold(f64::INFINITY, f64::min) },
        config,
    )
}

/// Parallel maximum for f64 values
pub fn parallel_max_f64(
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<impl Fn(&[f64]) -> f64 + Send + Sync + Clone, f64, f64> {
    ParallelJitFunction::new(
        "parallel_max_f64",
        |chunk: &[f64]| -> f64 { chunk.iter().copied().fold(f64::NEG_INFINITY, f64::max) },
        |partial_maxs: Vec<f64>| -> f64 {
            partial_maxs.into_iter().fold(f64::NEG_INFINITY, f64::max)
        },
        config,
    )
}

/// Parallel median for f64 values (approximate for large datasets)
pub fn parallel_median_f64(data: &[f64], config: Option<ParallelConfig>) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted_data = data.to_vec();

    // Use parallel sort for large datasets
    if data.len() > config.as_ref().map(|c| c.min_chunk_size).unwrap_or(1000) {
        sorted_data.par_sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let len = sorted_data.len();
    if len % 2 == 0 {
        (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
    } else {
        sorted_data[len / 2]
    }
}

/// Create a custom parallel function
pub fn parallel_custom<F, R, RF>(
    name: impl Into<String>,
    _sequential_fn: F,
    map_fn: F,
    reduce_fn: RF,
    config: Option<ParallelConfig>,
) -> ParallelJitFunction<F, f64, R>
where
    F: Fn(&[f64]) -> R + Send + Sync + Clone,
    R: Send + Sync,
    RF: Fn(Vec<R>) -> R + Send + Sync + 'static,
{
    ParallelJitFunction::new(name, map_fn, reduce_fn, config)
}

/// Convenience functions that execute immediately
pub mod immediate {
    use super::*;

    /// Execute parallel sum immediately
    pub fn sum(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_sum_f64(config).execute(data)
    }

    /// Execute parallel mean immediately
    pub fn mean(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_mean_f64_value(data, config)
    }

    /// Execute parallel std immediately
    pub fn std(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_std_f64_value(data, config)
    }

    /// Execute parallel min immediately
    pub fn min(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_min_f64(config).execute(data)
    }

    /// Execute parallel max immediately
    pub fn max(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_max_f64(config).execute(data)
    }

    /// Execute parallel median immediately
    pub fn median(data: &[f64], config: Option<ParallelConfig>) -> f64 {
        parallel_median_f64(data, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sum() {
        let data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let expected = data.iter().sum::<f64>();

        let sum_func = parallel_sum_f64(None);
        let result = sum_func.execute(&data);

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_mean() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let expected = 50.5; // Mean of 1..=100

        let result = parallel_mean_f64_value(&data, None);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_std() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_std_f64_value(&data, None);

        // Expected std for [1,2,3,4,5] is approximately 1.414
        assert!((result - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];

        let min_func = parallel_min_f64(None);
        let max_func = parallel_max_f64(None);

        assert_eq!(min_func.execute(&data), 1.0);
        assert_eq!(max_func.execute(&data), 9.0);
    }

    #[test]
    fn test_parallel_median() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = parallel_median_f64(&data, None);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_small_dataset_sequential() {
        let data = vec![1.0, 2.0, 3.0]; // Below min_chunk_size
        let config = Some(ParallelConfig::new().with_min_chunk_size(10));

        let sum_func = parallel_sum_f64(config);
        let result = sum_func.execute(&data);
        assert_eq!(result, 6.0);
    }
}

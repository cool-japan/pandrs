//! # GroupBy JIT Extensions Module
//!
//! This module provides JIT-accelerated extensions for GroupBy operations.

use super::config::ParallelConfig;
use super::core::{JitCompilable, JitFunction};
use super::parallel::{
    parallel_max_f64, parallel_mean_f64_value, parallel_min_f64, parallel_std_f64_value,
    parallel_sum_f64,
};
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame;
use crate::optimized::split_dataframe::group::{AggregateOp, CustomAggregation, GroupBy};
use std::sync::Arc;

/// Extension trait for GroupBy operations with JIT acceleration
pub trait GroupByJitExt<'a> {
    /// Apply JIT-accelerated sum operation
    fn sum_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame>;

    /// Apply JIT-accelerated mean operation
    fn mean_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame>;

    /// Apply JIT-accelerated standard deviation operation
    fn std_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame>;

    /// Apply JIT-accelerated minimum operation
    fn min_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame>;

    /// Apply JIT-accelerated maximum operation
    fn max_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame>;

    /// Apply parallel JIT-accelerated sum operation
    fn parallel_sum_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame>;

    /// Apply parallel JIT-accelerated mean operation
    fn parallel_mean_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame>;

    /// Apply parallel JIT-accelerated standard deviation operation
    fn parallel_std_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame>;

    /// Apply custom JIT-compiled aggregation function
    fn aggregate_jit<F>(
        &self,
        column: &str,
        func: JitFunction<F>,
        result_name: &str,
    ) -> Result<OptimizedDataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static;
}

impl<'a> GroupByJitExt<'a> for GroupBy<'a> {
    fn sum_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame> {
        // Create a JIT-optimized sum function
        let sum_func = super::core::jit_f64("jit_sum", |data: &[f64]| -> f64 {
            // Use Kahan summation for numerical stability
            let mut sum = 0.0;
            let mut c = 0.0;
            for &value in data {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            sum
        });

        // Compile the JIT function to get the optimized version
        let compiled_func = sum_func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn mean_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame> {
        let mean_func = super::core::jit_f64("jit_mean", |data: &[f64]| -> f64 {
            if data.is_empty() {
                return 0.0;
            }

            // Use Kahan summation for numerical stability
            let mut sum = 0.0;
            let mut c = 0.0;
            for &value in data {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }

            sum / data.len() as f64
        });

        let compiled_func = mean_func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn std_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame> {
        let std_func = super::core::jit_f64("jit_std", |data: &[f64]| -> f64 {
            if data.len() <= 1 {
                return 0.0;
            }

            // Calculate mean first
            let mut sum = 0.0;
            let mut c = 0.0;
            for &value in data {
                let y = value - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            let mean = sum / data.len() as f64;

            // Calculate variance
            let mut var_sum = 0.0;
            let mut var_c = 0.0;
            for &value in data {
                let diff = value - mean;
                let sq_diff = diff * diff;
                let y = sq_diff - var_c;
                let t = var_sum + y;
                var_c = (t - var_sum) - y;
                var_sum = t;
            }

            let variance = var_sum / (data.len() - 1) as f64;
            variance.sqrt()
        });

        let compiled_func = std_func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn min_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame> {
        let min_func = super::core::jit_f64("jit_min", |data: &[f64]| -> f64 {
            data.iter().copied().fold(f64::INFINITY, f64::min)
        });

        let compiled_func = min_func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn max_jit(&self, column: &str, result_name: &str) -> Result<OptimizedDataFrame> {
        let max_func = super::core::jit_f64("jit_max", |data: &[f64]| -> f64 {
            data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        });

        let compiled_func = max_func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn parallel_sum_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame> {
        let sum_func = parallel_sum_f64(config);
        let custom_fn = Arc::new(move |values: &[f64]| sum_func.execute(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn parallel_mean_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame> {
        let custom_fn =
            Arc::new(move |values: &[f64]| parallel_mean_f64_value(values, config.clone()));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn parallel_std_jit(
        &self,
        column: &str,
        result_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<OptimizedDataFrame> {
        let custom_fn =
            Arc::new(move |values: &[f64]| parallel_std_f64_value(values, config.clone()));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }

    fn aggregate_jit<F>(
        &self,
        column: &str,
        func: JitFunction<F>,
        result_name: &str,
    ) -> Result<OptimizedDataFrame>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
    {
        let compiled_func = func.compile()?;
        let custom_fn = Arc::new(move |values: &[f64]| compiled_func(values));

        let custom_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            result_name: result_name.to_string(),
            custom_fn: Some(custom_fn),
        };

        self.aggregate_custom(vec![custom_agg])
    }
}

/// JIT aggregation function wrapper
pub struct JitAggregation<F> {
    /// The JIT function
    pub func: JitFunction<F>,
    /// Function name for debugging
    pub name: String,
}

impl<F> JitAggregation<F>
where
    F: Fn(&[f64]) -> f64 + Send + Sync + Clone + 'static,
{
    /// Create a new JIT aggregation
    pub fn new(name: impl Into<String>, func: F) -> Self {
        let name = name.into();
        let jit_func = super::core::jit_f64(name.clone(), func).without_jit(); // Disable JIT for custom aggregations
        Self {
            func: jit_func,
            name,
        }
    }

    /// Create a new JIT aggregation with parallel execution
    pub fn parallel(name: impl Into<String>, func: F, config: Option<ParallelConfig>) -> Self {
        let name_str = name.into();
        let jit_func = super::core::jit_f64(name_str.clone(), func).without_jit(); // Disable JIT for custom aggregations
        Self {
            func: jit_func,
            name: name_str,
        }
    }

    /// Execute the aggregation
    pub fn execute(&self, data: &[f64]) -> f64 {
        self.func.execute(data)
    }
}

/// Create pre-defined JIT aggregations
pub mod aggregations {
    use super::*;

    /// Weighted mean aggregation where weight = position in group
    pub fn weighted_mean() -> JitAggregation<impl Fn(&[f64]) -> f64 + Send + Sync + Clone> {
        JitAggregation::new("weighted_mean", |values: &[f64]| -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (i, &val) in values.iter().enumerate() {
                let weight = (i + 1) as f64;
                weighted_sum += val * weight;
                weight_sum += weight;
            }

            if weight_sum == 0.0 {
                0.0
            } else {
                weighted_sum / weight_sum
            }
        })
    }

    /// Geometric mean aggregation
    pub fn geometric_mean() -> JitAggregation<impl Fn(&[f64]) -> f64 + Send + Sync + Clone> {
        JitAggregation::new("geometric_mean", |values: &[f64]| -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            let mut log_sum = 0.0;
            let mut count = 0;

            for &val in values {
                if val > 0.0 {
                    log_sum += val.ln();
                    count += 1;
                }
            }

            if count == 0 {
                0.0
            } else {
                (log_sum / count as f64).exp()
            }
        })
    }

    /// Harmonic mean aggregation
    pub fn harmonic_mean() -> JitAggregation<impl Fn(&[f64]) -> f64 + Send + Sync + Clone> {
        JitAggregation::new("harmonic_mean", |values: &[f64]| -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            let mut reciprocal_sum = 0.0;
            let mut count = 0;

            for &val in values {
                if val != 0.0 {
                    reciprocal_sum += 1.0 / val;
                    count += 1;
                }
            }

            if count == 0 {
                0.0
            } else {
                count as f64 / reciprocal_sum
            }
        })
    }

    /// Range (max - min) aggregation
    pub fn range() -> JitAggregation<impl Fn(&[f64]) -> f64 + Send + Sync + Clone> {
        JitAggregation::new("range", |values: &[f64]| -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            max - min
        })
    }

    /// Coefficient of variation (std/mean) aggregation
    pub fn coefficient_of_variation() -> JitAggregation<impl Fn(&[f64]) -> f64 + Send + Sync + Clone>
    {
        JitAggregation::new("coefficient_of_variation", |values: &[f64]| -> f64 {
            if values.len() <= 1 {
                return 0.0;
            }

            // Calculate mean
            let sum: f64 = values.iter().sum();
            let mean = sum / values.len() as f64;

            if mean == 0.0 {
                return 0.0;
            }

            // Calculate standard deviation
            let variance: f64 =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

            let std_dev = variance.sqrt();

            std_dev / mean.abs()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::split_dataframe::core::OptimizedDataFrame;

    #[test]
    fn test_jit_aggregation() {
        let weighted_mean = aggregations::weighted_mean();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Manual calculation: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (1 + 2 + 3 + 4 + 5) = 55 / 15 = 3.67
        let result = weighted_mean.execute(&data);
        let expected = 55.0 / 15.0;
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_geometric_mean() {
        let geo_mean = aggregations::geometric_mean();
        let data = vec![1.0, 2.0, 4.0, 8.0];

        // Geometric mean of [1, 2, 4, 8] = (1 * 2 * 4 * 8)^(1/4) = 64^(1/4) = 2.83
        let result = geo_mean.execute(&data);
        let expected = (64.0_f64).powf(0.25); // More accurate calculation
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_harmonic_mean() {
        let harm_mean = aggregations::harmonic_mean();
        let data = vec![1.0, 2.0, 4.0];

        // Harmonic mean of [1, 2, 4] = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 = 1.714
        let result = harm_mean.execute(&data);
        let expected = 3.0 / (1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0); // More accurate calculation
        assert!(
            (result - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_range() {
        let range_agg = aggregations::range();
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];

        // Range = max - min = 9 - 1 = 8
        let result = range_agg.execute(&data);
        assert_eq!(result, 8.0);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let cv = aggregations::coefficient_of_variation();
        let data = vec![10.0, 12.0, 14.0, 16.0, 18.0];

        // Mean = 14, std ≈ 3.16, CV ≈ 0.226
        let result = cv.execute(&data);
        assert!((result - 0.22587697572631278).abs() < 1e-10);
    }
}

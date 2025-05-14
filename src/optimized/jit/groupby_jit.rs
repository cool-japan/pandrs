//! JIT support for GroupBy operations
//!
//! This module provides JIT compilation capabilities for GroupBy operations,
//! allowing for accelerated custom aggregations similar to pandas with Numba.
//!
//! Both single-threaded and parallel JIT functions are supported, with
//! parallel functions automatically used when appropriate based on data size.

use std::sync::Arc;
use crate::core::error::Result;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::split_dataframe::group::{AggregateOp, GroupBy};

use super::jit_core::{JitCompilable, JitFunction};
use super::parallel::{ParallelConfig, parallel_sum_f64, parallel_mean_f64, 
                      parallel_std_f64, parallel_min_f64, parallel_max_f64};

/// Extension trait for JIT-enabled GroupBy operations
pub trait GroupByJitExt<'a> {
    /// Aggregate using a JIT-compiled function
    ///
    /// # Arguments
    /// * `column` - The column name to aggregate
    /// * `jit_fn` - The JIT-compilable function to apply
    /// * `alias` - The name for the resulting column
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    fn aggregate_jit<J>(
        &self,
        column: &str,
        jit_fn: J,
        alias: &str,
    ) -> Result<OptimizedDataFrame>
    where
        J: JitCompilable<Vec<f64>, f64> + 'static;
        
    /// Aggregate with sum using JIT compilation
    fn sum_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with mean using JIT compilation
    fn mean_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with standard deviation using JIT compilation
    fn std_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with variance using JIT compilation
    fn var_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with min using JIT compilation
    fn min_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with max using JIT compilation
    fn max_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with median using JIT compilation
    fn median_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Apply multiple JIT-compiled aggregations at once
    ///
    /// # Arguments
    /// * `aggregations` - List of (column, JIT function, alias) tuples
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    fn aggregate_multi_jit<I, J>(
        &self,
        aggregations: I,
    ) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, J, String)>,
        J: JitCompilable<Vec<f64>, f64> + 'static;
        
    /// Aggregate with parallel sum using JIT compilation
    /// 
    /// This uses multi-threading for improved performance on large groups.
    fn parallel_sum_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with parallel mean using JIT compilation
    /// 
    /// This uses multi-threading for improved performance on large groups.
    fn parallel_mean_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with parallel standard deviation using JIT compilation
    /// 
    /// This uses multi-threading for improved performance on large groups.
    fn parallel_std_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with parallel min using JIT compilation
    /// 
    /// This uses multi-threading for improved performance on large groups.
    fn parallel_min_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with parallel max using JIT compilation
    /// 
    /// This uses multi-threading for improved performance on large groups.
    fn parallel_max_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame>;
}

/// Implement the GroupByJitExt trait for GroupBy
impl<'a> GroupByJitExt<'a> for GroupBy<'a> {
    fn aggregate_jit<J>(
        &self,
        column: &str,
        jit_fn: J,
        alias: &str,
    ) -> Result<OptimizedDataFrame>
    where
        J: JitCompilable<Vec<f64>, f64> + 'static,
    {
        // Create a custom aggregation that uses the JIT function
        let jit_agg = CustomAggregation {
            column: column.to_string(),
            op: AggregateOp::Custom,
            alias: alias.to_string(),
            custom_fn: Some(Arc::new(move |values: &[f64]| -> f64 {
                // Convert slice to Vec for the JIT function
                let vec_values = values.to_vec();
                jit_fn.execute(vec_values)
            })),
        };
        
        // Use the existing aggregate_custom method
        self.aggregate_custom(vec![jit_agg])
    }
    
    fn sum_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        self.aggregate_jit(column, array_ops::sum(), alias)
    }
    
    fn mean_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        self.aggregate_jit(column, array_ops::mean(), alias)
    }
    
    fn std_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        // Use 1 degree of freedom (n-1) for sample standard deviation
        self.aggregate_jit(column, array_ops::std(1), alias)
    }
    
    fn var_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        // Use 1 degree of freedom (n-1) for sample variance
        self.aggregate_jit(column, array_ops::var(1), alias)
    }
    
    fn min_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        self.aggregate_jit(column, array_ops::min(), alias)
    }
    
    fn max_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        self.aggregate_jit(column, array_ops::max(), alias)
    }
    
    fn median_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        use super::array_ops;
        self.aggregate_jit(column, array_ops::median(), alias)
    }
    
    fn aggregate_multi_jit<I, J>(
        &self,
        aggregations: I,
    ) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, J, String)>,
        J: JitCompilable<Vec<f64>, f64> + 'static,
    {
        let custom_aggs = aggregations
            .into_iter()
            .map(|(col, jit_fn, alias)| {
                let jit_function = jit_fn;
                CustomAggregation {
                    column: col,
                    op: AggregateOp::Custom,
                    alias,
                    custom_fn: Some(Arc::new(move |values: &[f64]| -> f64 {
                        let vec_values = values.to_vec();
                        jit_function.execute(vec_values)
                    })),
                }
            })
            .collect::<Vec<_>>();
        
        self.aggregate_custom(custom_aggs)
    }
    
    // Parallel implementation of sum
    fn parallel_sum_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame> {
        let parallel_sum = parallel_sum_f64(config);
        self.aggregate_jit(column, parallel_sum, alias)
    }
    
    // Parallel implementation of mean
    fn parallel_mean_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame> {
        let parallel_mean = parallel_mean_f64(config);
        self.aggregate_jit(column, parallel_mean, alias)
    }
    
    // Parallel implementation of standard deviation
    fn parallel_std_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame> {
        let parallel_std = parallel_std_f64(config);
        self.aggregate_jit(column, parallel_std, alias)
    }
    
    // Parallel implementation of min
    fn parallel_min_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame> {
        let parallel_min = parallel_min_f64(config);
        self.aggregate_jit(column, parallel_min, alias)
    }
    
    // Parallel implementation of max
    fn parallel_max_jit(
        &self, 
        column: &str, 
        alias: &str,
        config: Option<ParallelConfig>
    ) -> Result<OptimizedDataFrame> {
        let parallel_max = parallel_max_f64(config);
        self.aggregate_jit(column, parallel_max, alias)
    }
}

// This struct is imported from the existing codebase
// It's duplicated here for documentation purposes
struct CustomAggregation {
    column: String,
    op: AggregateOp,
    alias: String,
    custom_fn: Option<Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>>,
}
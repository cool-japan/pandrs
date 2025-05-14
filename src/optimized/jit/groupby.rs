//! JIT integration for GroupBy operations

use crate::core::error::Result;
use crate::optimized::jit::{JitCompilable, JitFunction};
use crate::optimized::split_dataframe::core::Column;
use crate::optimized::split_dataframe::group::{AggregateOp, GroupBy};
use crate::optimized::dataframe::OptimizedDataFrame;
use std::sync::Arc;

#[cfg(feature = "jit")]
use super::aggregation;

/// Extension trait for JIT-enabled GroupBy operations
#[cfg(feature = "jit")]
pub trait GroupByJitExt<'a> {
    /// Aggregate using a JIT-compiled function
    fn aggregate_jit<F>(
        &self,
        column: &str,
        jit_fn: F,
        alias: &str,
    ) -> Result<OptimizedDataFrame>
    where
        F: JitCompilable<&[f64], f64> + Send + Sync + 'static;
        
    /// Aggregate with sum using JIT compilation
    fn sum_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with mean using JIT compilation
    fn mean_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Aggregate with standard deviation using JIT compilation
    fn std_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame>;
    
    /// Apply multiple JIT-compiled aggregations at once
    fn aggregate_multi_jit<I, F>(
        &self,
        aggregations: I,
    ) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, F, String)>,
        F: JitCompilable<&[f64], f64> + Send + Sync + 'static;
}

/// Implement the GroupByJitExt trait for GroupBy
#[cfg(feature = "jit")]
impl<'a> GroupByJitExt<'a> for GroupBy<'a> {
    /// Aggregate using a JIT-compiled function
    ///
    /// # Arguments
    /// * `column` - The column name to aggregate
    /// * `jit_fn` - The JIT-compiled function to apply
    /// * `alias` - The name for the resulting column
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn aggregate_jit<F>(
        &self,
        column: &str,
        jit_fn: F,
        alias: &str,
    ) -> Result<OptimizedDataFrame>
    where
        F: JitCompilable<&[f64], f64> + Send + Sync + 'static,
    {
        let jit_fn = Arc::new(jit_fn);
        self.aggregate(vec![(
            column.to_string(),
            AggregateOp::JitFunc(jit_fn),
            alias.to_string(),
        )])
    }

    /// Aggregate with sum using JIT compilation
    pub fn sum_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        self.aggregate_jit(column, aggregation::sum(), alias)
    }

    /// Aggregate with mean using JIT compilation
    pub fn mean_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        self.aggregate_jit(column, aggregation::mean(), alias)
    }

    /// Aggregate with standard deviation using JIT compilation
    pub fn std_jit(&self, column: &str, alias: &str) -> Result<OptimizedDataFrame> {
        self.aggregate_jit(column, aggregation::std(), alias)
    }

    /// Apply multiple JIT-compiled aggregations at once
    ///
    /// # Arguments
    /// * `aggregations` - List of (column, JIT function, alias) tuples
    ///
    /// # Returns
    /// * `Result<OptimizedDataFrame>` - DataFrame containing aggregation results
    pub fn aggregate_multi_jit<I, F>(
        &self,
        aggregations: I,
    ) -> Result<OptimizedDataFrame>
    where
        I: IntoIterator<Item = (String, F, String)>,
        F: JitCompilable<&[f64], f64> + Send + Sync + 'static,
    {
        let aggs = aggregations
            .into_iter()
            .map(|(col, jit_fn, alias)| {
                (
                    col,
                    AggregateOp::JitFunc(Arc::new(jit_fn)),
                    alias,
                )
            })
            .collect::<Vec<_>>();

        self.aggregate(aggs)
    }
}
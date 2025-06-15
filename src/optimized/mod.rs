pub mod convert;
pub mod dataframe;
pub mod direct_aggregations;
pub mod jit;
pub mod lazy;
pub mod operations;
pub mod split_dataframe;

pub use convert::{optimize_dataframe, standard_dataframe};
pub use dataframe::{ColumnView, OptimizedDataFrame};
pub use jit::core::{JitCompilable, JitFunction};
pub use jit::{
    parallel_max_f64, parallel_mean_f64, parallel_mean_f64_value, parallel_min_f64,
    parallel_std_f64, parallel_sum_f64, simd_max_f64, simd_mean_f64, simd_min_f64, simd_sum_f64,
    GroupByJitExt, JITConfig, JitAggregation, ParallelConfig, SIMDConfig,
};
pub use lazy::{LazyFrame, Operation};
pub use operations::{AggregateOp, JoinType};
pub use split_dataframe::group::CustomAggregation;

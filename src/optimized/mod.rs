pub mod dataframe;
pub mod operations;
pub mod lazy;
pub mod split_dataframe;
pub mod convert;
pub mod jit;

pub use dataframe::{OptimizedDataFrame, ColumnView};
pub use operations::{AggregateOp, JoinType};
pub use lazy::{LazyFrame, Operation};
pub use convert::{optimize_dataframe, standard_dataframe};
pub use split_dataframe::group::CustomAggregation;
pub use jit::{
    ParallelConfig, SIMDConfig, JITConfig,
    parallel_sum_f64, parallel_mean_f64, parallel_mean_f64_value, parallel_std_f64, parallel_min_f64, parallel_max_f64,
    simd_sum_f64, simd_mean_f64, simd_min_f64, simd_max_f64,
    GroupByJitExt, JitAggregation
};
pub use jit::core::{JitFunction, JitCompilable};
// Computation functionality module
pub mod lazy;
pub mod parallel;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports
pub use lazy::LazyFrame;
pub use parallel::ParallelUtils;

#[cfg(feature = "cuda")]
pub use gpu::{init_gpu, GpuBenchmark, GpuConfig, GpuDeviceStatus};

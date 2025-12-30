// Computation functionality module
pub mod lazy;
pub mod parallel;

#[cfg(cuda_available)]
pub mod gpu;

// Re-exports
pub use lazy::LazyFrame;
pub use parallel::ParallelUtils;

#[cfg(cuda_available)]
pub use gpu::{init_gpu, GpuBenchmark, GpuConfig, GpuDeviceStatus};

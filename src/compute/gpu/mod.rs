#[cfg(cuda_available)]
// Re-export from legacy module for now
#[deprecated(since = "0.1.0", note = "Use crate::compute::gpu instead")]
pub use crate::gpu::{init_gpu, GpuConfig, GpuDeviceStatus};

#[cfg(cuda_available)]
// Re-export benchmark module
#[deprecated(since = "0.1.0", note = "Use crate::compute::gpu::benchmark instead")]
pub use crate::gpu::benchmark::GpuBenchmark;

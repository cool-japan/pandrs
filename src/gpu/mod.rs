//! GPU acceleration module for PandRS
//!
//! This module provides GPU-accelerated operations using CUDA/cuBLAS for large-scale
//! data processing. It allows PandRS to leverage the massive parallelism offered by
//! GPU hardware for significant performance improvements on compatible operations.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::lock_safe;

/// Configuration options for GPU operations
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Whether to enable GPU acceleration when available
    pub enabled: bool,
    /// Maximum memory usage for GPU operations (in bytes)
    pub memory_limit: usize,
    /// Device ID to use (for multi-GPU systems)
    pub device_id: i32,
    /// Whether to fallback to CPU if GPU operation fails
    pub fallback_to_cpu: bool,
    /// Whether to use pinned memory for faster transfers
    pub use_pinned_memory: bool,
    /// Minimum size threshold for offloading to GPU (in elements)
    pub min_size_threshold: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        GpuConfig {
            enabled: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB default
            device_id: 0,
            fallback_to_cpu: true,
            use_pinned_memory: true,
            min_size_threshold: 10_000, // Only use GPU for operations on at least 10K elements
        }
    }
}

/// GPU operation context
pub struct GpuContext {
    /// GPU device status
    device_status: Arc<Mutex<GpuDeviceStatus>>,
    /// Configuration
    config: GpuConfig,
}

/// GPU device status
#[derive(Debug, Clone)]
pub struct GpuDeviceStatus {
    /// Whether a CUDA-compatible GPU is available
    pub available: bool,
    /// CUDA version
    pub cuda_version: Option<String>,
    /// Device name
    pub device_name: Option<String>,
    /// Total device memory (in bytes)
    pub total_memory: Option<usize>,
    /// Free device memory (in bytes)
    pub free_memory: Option<usize>,
    /// Number of CUDA cores
    pub core_count: Option<usize>,
}

impl Default for GpuDeviceStatus {
    fn default() -> Self {
        GpuDeviceStatus {
            available: false,
            cuda_version: None,
            device_name: None,
            total_memory: None,
            free_memory: None,
            core_count: None,
        }
    }
}

impl GpuContext {
    /// Create a new GPU context with the given configuration
    pub fn new(config: GpuConfig) -> Self {
        let device_status = Arc::new(Mutex::new(Self::detect_device(&config)));

        GpuContext {
            device_status,
            config,
        }
    }

    /// Detect available GPU devices and their capabilities
    fn detect_device(_config: &GpuConfig) -> GpuDeviceStatus {
        #[cfg(cuda_available)]
        {
            // Use cudarc (0.19.x) to detect CUDA availability and query real
            // device properties instead of reporting hardcoded placeholders.
            match cudarc::driver::CudaContext::new(0) {
                Ok(context) => {
                    // Real device name (None if the query fails).
                    let device_name = context.name().ok();

                    // Real total/free device memory. `mem_get_info` returns
                    // `(free, total)`.
                    let (free_memory, total_memory) = match context.mem_get_info() {
                        Ok((free, total)) => (Some(free), Some(total)),
                        Err(_) => (None, None),
                    };

                    // cudarc's safe API does not expose the CUDA runtime/driver
                    // version, so report the real device compute capability
                    // rather than fabricating a version string.
                    let cuda_version = context
                        .compute_capability()
                        .ok()
                        .map(|(major, minor)| format!("compute capability {}.{}", major, minor));

                    return GpuDeviceStatus {
                        available: true,
                        cuda_version,
                        device_name,
                        total_memory,
                        free_memory,
                        // The streaming-multiprocessor/core count is not queried
                        // here; report None rather than a fabricated value.
                        core_count: None,
                    };
                }
                Err(_) => {
                    // CUDA not available or context creation failed
                }
            }
        }

        // Default to unavailable when CUDA is not enabled or not available
        GpuDeviceStatus::default()
    }

    /// Check if GPU acceleration is available and enabled
    pub fn is_available(&self) -> bool {
        if !self.config.enabled {
            return false;
        }

        lock_safe!(self.device_status, "gpu device status lock")
            .map(|status| status.available)
            .unwrap_or(false)
    }

    /// Get device status
    pub fn get_device_status(&self) -> GpuDeviceStatus {
        lock_safe!(self.device_status, "gpu device status lock")
            .map(|status| status.clone())
            .unwrap_or_default()
    }

    /// Update device status
    pub fn update_device_status(&self) {
        let new_status = Self::detect_device(&self.config);
        if let Ok(mut status) = lock_safe!(self.device_status, "gpu device status lock") {
            *status = new_status;
        }
    }

    /// Check if operation should be offloaded to GPU based on size threshold
    pub fn should_use_gpu(&self, size: usize) -> bool {
        self.is_available() && size >= self.config.min_size_threshold
    }

    /// Get the GPU configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
}

/// Types of GPU operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuOperationType {
    /// Matrix multiplication
    MatrixMultiply,
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Subtract,
    /// Element-wise multiplication
    Multiply,
    /// Element-wise division
    Divide,
    /// Aggregation (sum, mean, etc.)
    Aggregate,
    /// Sorting
    Sort,
    /// Filtering
    Filter,
}

/// Custom error type for GPU operations
#[derive(Debug)]
pub enum GpuError {
    /// CUDA driver not available
    DriverNotAvailable,
    /// Insufficient GPU memory
    InsufficientMemory,
    /// Incompatible data type
    IncompatibleDataType,
    /// Device error
    DeviceError(String),
    /// Kernel execution error
    KernelExecutionError(String),
    /// Transfer error
    TransferError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuError::DriverNotAvailable => write!(f, "CUDA driver not available"),
            GpuError::InsufficientMemory => write!(f, "Insufficient GPU memory"),
            GpuError::IncompatibleDataType => write!(f, "Incompatible data type for GPU operation"),
            GpuError::DeviceError(msg) => write!(f, "GPU device error: {}", msg),
            GpuError::KernelExecutionError(msg) => write!(f, "GPU kernel execution error: {}", msg),
            GpuError::TransferError(msg) => write!(f, "GPU data transfer error: {}", msg),
        }
    }
}

impl StdError for GpuError {}

impl From<GpuError> for Error {
    fn from(error: GpuError) -> Self {
        Error::Computation(error.to_string())
    }
}

/// Struct for managing GPU operations across the application
#[derive(Clone)]
pub struct GpuManager {
    /// The global GPU context
    context: Arc<GpuContext>,
}

impl GpuManager {
    /// Create a new GPU manager with default configuration
    pub fn new() -> Self {
        Self::with_config(GpuConfig::default())
    }

    /// Create a new GPU manager with the given configuration
    pub fn with_config(config: GpuConfig) -> Self {
        GpuManager {
            context: Arc::new(GpuContext::new(config)),
        }
    }

    /// Get GPU context
    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    /// Check if GPU acceleration is available
    pub fn is_available(&self) -> bool {
        self.context.is_available()
    }

    /// Get detailed GPU device information
    pub fn device_info(&self) -> GpuDeviceStatus {
        self.context.get_device_status()
    }
}

// Global GPU manager instance initialized lazily
lazy_static::lazy_static! {
    static ref GPU_MANAGER: Mutex<Option<GpuManager>> = Mutex::new(None);
}

/// Initialize global GPU manager with default configuration
pub fn init_gpu() -> Result<GpuDeviceStatus> {
    init_gpu_with_config(GpuConfig::default())
}

/// Initialize global GPU manager with custom configuration
pub fn init_gpu_with_config(config: GpuConfig) -> Result<GpuDeviceStatus> {
    let manager = GpuManager::with_config(config);
    let status = manager.device_info();

    let mut global = lock_safe!(GPU_MANAGER, "global gpu manager lock")?;
    *global = Some(manager);

    Ok(status)
}

/// Get the global GPU manager instance
pub fn get_gpu_manager() -> Result<GpuManager> {
    let global = lock_safe!(GPU_MANAGER, "global gpu manager lock")?;

    match &*global {
        Some(manager) => Ok(manager.clone()),
        None => {
            // Initialize with default config if not already initialized
            drop(global); // Release lock before recursive call
            init_gpu()?;
            get_gpu_manager()
        }
    }
}

// Include appropriate implementation modules based on features
#[cfg(cuda_available)]
pub mod cuda;

// Include this module regardless of CUDA availability to provide fallback
pub mod operations;

// Benchmarking module
pub mod benchmark;

// Advanced GPU operations (matrix decompositions, advanced stats)
pub mod advanced_ops;

// Multi-GPU support for distributed computation
pub mod multi_gpu;

// GPU memory pooling for efficient memory management
pub mod memory_pool;

// Real CPU math shared by the GPU-dispatch entry points (Rust extension
// traits and the Python bindings alike) — see its module doc for why.
pub mod cpu_math;

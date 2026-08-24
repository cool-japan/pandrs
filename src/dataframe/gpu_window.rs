//! Window operations for DataFrame with a GPU-dispatch path
//!
//! This module provides the scaffolding for GPU-accelerated window operations
//! and integrates with the existing JIT window operations, including a seamless
//! fallback to CPU.
//!
//! IMPORTANT (honesty note): cudarc 0.19.x does not expose the kernels needed to
//! implement these window operations on the device, so the `gpu_*` window
//! functions below currently compute their results on the **CPU**. The
//! `GpuWindowStats` counters therefore track the GPU-dispatch decisions and the
//! real wall-clock timing of that path; they never report a fabricated
//! GPU-vs-CPU speedup. The structure remains so that real CUDA kernels can be
//! slotted in later for:
//! - Large datasets (> 50,000 elements)
//! - Computationally intensive operations (std, var, quantiles)
//! - Repeated operations on similar data patterns
//! - Operations that can leverage massive parallelism

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::jit_window::{
    JitWindowContext, JitWindowStats, WindowFunctionKey, WindowOpType,
};
use crate::gpu::{get_gpu_manager, GpuManager};
use crate::lock_safe;
use crate::series::Series;

/// Statistics for the GPU-dispatch window-operation path.
///
/// NOTE: the "GPU" window kernels in this module currently execute on the CPU
/// (no real CUDA kernel is implemented). These counters therefore track how
/// often the GPU-dispatch path was taken and its real wall-clock timing; they do
/// not report a fabricated GPU-vs-CPU speedup.
#[derive(Debug, Clone, Default)]
pub struct GpuWindowStats {
    /// Number of operations routed through the GPU-dispatch path
    pub gpu_executions: u64,
    /// Number of operations that fell back to CPU
    pub cpu_fallbacks: u64,
    /// Total GPU memory allocated (bytes)
    pub total_gpu_memory_allocated: u64,
    /// Total data transfer time (nanoseconds)
    pub total_transfer_time_ns: u64,
    /// Total wall-clock execution time of the GPU-dispatch path (nanoseconds)
    pub total_kernel_time_ns: u64,
    /// GPU memory transfer efficiency (bytes/ns)
    pub transfer_efficiency: f64,
    /// Number of successful GPU memory allocations
    pub successful_allocations: u64,
    /// Number of failed GPU memory allocations
    pub failed_allocations: u64,
}

impl GpuWindowStats {
    /// Create new empty GPU statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one execution routed through the GPU-dispatch path.
    ///
    /// `execution_time_ns` is the real measured wall-clock time. No GPU-vs-CPU
    /// speedup is recorded: the window kernels currently run on the CPU, so any
    /// speedup figure would be fabricated.
    pub fn record_gpu_execution(&mut self, execution_time_ns: u64) {
        self.gpu_executions += 1;
        self.total_kernel_time_ns += execution_time_ns;
    }

    /// Record a CPU fallback
    pub fn record_cpu_fallback(&mut self) {
        self.cpu_fallbacks += 1;
    }

    /// Record memory allocation
    pub fn record_memory_allocation(&mut self, bytes: u64, success: bool) {
        if success {
            self.successful_allocations += 1;
            self.total_gpu_memory_allocated += bytes;
        } else {
            self.failed_allocations += 1;
        }
    }

    /// Record data transfer
    pub fn record_data_transfer(&mut self, transfer_time_ns: u64, bytes: u64) {
        self.total_transfer_time_ns += transfer_time_ns;
        if transfer_time_ns > 0 {
            self.transfer_efficiency = bytes as f64 / transfer_time_ns as f64;
        }
    }

    /// Calculate GPU usage ratio
    pub fn gpu_usage_ratio(&self) -> f64 {
        let total_ops = self.gpu_executions + self.cpu_fallbacks;
        if total_ops > 0 {
            self.gpu_executions as f64 / total_ops as f64
        } else {
            0.0
        }
    }

    /// Calculate memory allocation success rate
    pub fn allocation_success_rate(&self) -> f64 {
        let total_allocations = self.successful_allocations + self.failed_allocations;
        if total_allocations > 0 {
            self.successful_allocations as f64 / total_allocations as f64
        } else {
            1.0
        }
    }
}

/// GPU-enhanced window operation context
pub struct GpuWindowContext {
    /// Base JIT context
    jit_context: JitWindowContext,
    /// GPU manager for device operations
    gpu_manager: GpuManager,
    /// GPU-specific statistics
    gpu_stats: Arc<Mutex<GpuWindowStats>>,
    /// Minimum dataset size for GPU acceleration
    gpu_threshold_size: usize,
    /// Memory allocation cache
    memory_cache: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    /// Enable/disable GPU acceleration
    gpu_enabled: bool,
}

impl GpuWindowContext {
    /// Create a new GPU-enhanced window context
    pub fn new() -> Result<Self> {
        let jit_context = JitWindowContext::new();
        let gpu_manager = get_gpu_manager()?;

        Ok(Self {
            jit_context,
            gpu_manager,
            gpu_stats: Arc::new(Mutex::new(GpuWindowStats::new())),
            gpu_threshold_size: 50_000, // Use GPU for datasets > 50K elements
            memory_cache: Arc::new(Mutex::new(HashMap::new())),
            gpu_enabled: true,
        })
    }

    /// Create a new GPU context with custom configuration
    pub fn with_config(
        jit_enabled: bool,
        jit_threshold: u64,
        gpu_enabled: bool,
        gpu_threshold_size: usize,
    ) -> Result<Self> {
        let jit_context = JitWindowContext::with_settings(jit_enabled, jit_threshold);
        let gpu_manager = get_gpu_manager()?;

        Ok(Self {
            jit_context,
            gpu_manager,
            gpu_stats: Arc::new(Mutex::new(GpuWindowStats::new())),
            gpu_threshold_size,
            memory_cache: Arc::new(Mutex::new(HashMap::new())),
            gpu_enabled,
        })
    }

    /// Check if GPU acceleration should be used for the given operation
    pub fn should_use_gpu(&self, data_size: usize, op_type: &WindowOpType) -> bool {
        if !self.gpu_enabled || !self.gpu_manager.is_available() {
            return false;
        }

        // Size threshold check
        if data_size < self.gpu_threshold_size {
            return false;
        }

        // Operation-specific checks
        match op_type {
            // These operations benefit most from GPU acceleration
            WindowOpType::RollingMean
            | WindowOpType::RollingSum
            | WindowOpType::ExpandingMean
            | WindowOpType::ExpandingSum => data_size >= self.gpu_threshold_size,
            // These operations benefit significantly from GPU parallelism
            WindowOpType::RollingStd
            | WindowOpType::RollingVar
            | WindowOpType::EWMMean
            | WindowOpType::EWMStd
            | WindowOpType::EWMVar => {
                data_size >= self.gpu_threshold_size / 2 // Lower threshold for complex ops
            }
            // These operations are memory-bound and may not benefit as much
            WindowOpType::RollingMin | WindowOpType::RollingMax => {
                data_size >= self.gpu_threshold_size * 2 // Higher threshold
            }
            // Sorting-heavy operations benefit from GPU for very large datasets
            WindowOpType::RollingMedian | WindowOpType::RollingQuantile(_) => {
                data_size >= self.gpu_threshold_size * 3 // Much higher threshold
            }
            _ => false,
        }
    }

    /// Execute a GPU-accelerated window operation
    pub fn execute_gpu_operation(
        &self,
        key: &WindowFunctionKey,
        data: &[f64],
        window_size: usize,
    ) -> Result<Vec<f64>> {
        let start_time = Instant::now();

        // Check GPU availability and memory requirements
        let data_bytes = data.len() * std::mem::size_of::<f64>();
        let result_bytes = data.len() * std::mem::size_of::<f64>();
        let total_memory_required =
            data_bytes + result_bytes + (window_size * std::mem::size_of::<f64>());

        let device_status = self.gpu_manager.device_info();
        if let Some(free_memory) = device_status.free_memory {
            if total_memory_required > free_memory / 2 {
                // Use only half of available memory
                let mut stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?;
                stats.record_cpu_fallback();
                return Err(Error::InvalidOperation(
                    "Insufficient GPU memory for operation".to_string(),
                ));
            }
        }

        // Execute the GPU kernel based on operation type
        let result = match &key.operation {
            WindowOpType::RollingMean => self.gpu_rolling_mean(data, window_size),
            WindowOpType::RollingSum => self.gpu_rolling_sum(data, window_size),
            WindowOpType::RollingStd => self.gpu_rolling_std(data, window_size),
            WindowOpType::RollingVar => self.gpu_rolling_var(data, window_size),
            WindowOpType::ExpandingMean => self.gpu_expanding_mean(data),
            WindowOpType::ExpandingSum => self.gpu_expanding_sum(data),
            WindowOpType::EWMMean => {
                self.gpu_ewm_mean(data, 0.1) // Default alpha
            }
            _ => {
                // Fallback to JIT implementation for unsupported operations
                let mut stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?;
                stats.record_cpu_fallback();
                return Err(Error::InvalidOperation(format!(
                    "GPU kernel not implemented for {:?}",
                    key.operation
                )));
            }
        };

        match result {
            Ok(gpu_result) => {
                let execution_time = start_time.elapsed().as_nanos() as u64;

                // Record the GPU-dispatch path execution with its real measured
                // wall-clock time (no fabricated speedup).
                let mut stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?;
                stats.record_gpu_execution(execution_time);
                stats.record_memory_allocation(total_memory_required as u64, true);

                Ok(gpu_result)
            }
            Err(e) => {
                // Record fallback
                let mut stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?;
                stats.record_cpu_fallback();
                Err(e)
            }
        }
    }

    /// GPU-accelerated rolling mean implementation
    fn gpu_rolling_mean(&self, data: &[f64], window_size: usize) -> Result<Vec<f64>> {
        if window_size == 0 {
            // `window_size - 1` below would underflow `usize` and panic.
            return Ok(vec![f64::NAN; data.len()]);
        }

        #[cfg(cuda_available)]
        {
            // A real CUDA implementation would allocate device memory, transfer
            // the data, launch a parallel rolling-mean kernel, and copy the
            // result back. cudarc 0.19.x exposes no such kernel here, so the
            // result is computed on the CPU below (correct, just not on-device).
            //
            // No transfer/allocation stats are recorded here (the previous
            // version recorded a fabricated H2D+D2H byte count derived from
            // `data.len()` even though the computation above never touches
            // the device at all — exactly the "fabricated speedup/metric"
            // this module's own honesty note disclaims).
            let mut result = vec![f64::NAN; data.len()];

            for i in window_size - 1..data.len() {
                let window_start = i + 1 - window_size;
                let window_end = i + 1;
                let window_data = &data[window_start..window_end];
                result[i] = window_data.iter().sum::<f64>() / window_data.len() as f64;
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            // CPU fallback implementation
            let mut result = vec![f64::NAN; data.len()];
            for i in window_size - 1..data.len() {
                let window_start = i + 1 - window_size;
                let window_end = i + 1;
                let window_data = &data[window_start..window_end];
                result[i] = window_data.iter().sum::<f64>() / window_data.len() as f64;
            }
            Ok(result)
        }
    }

    /// GPU-accelerated rolling sum implementation
    fn gpu_rolling_sum(&self, data: &[f64], window_size: usize) -> Result<Vec<f64>> {
        if window_size == 0 {
            // `window_size - 1` below would underflow `usize` and panic.
            return Ok(vec![f64::NAN; data.len()]);
        }

        #[cfg(cuda_available)]
        {
            let mut result = vec![f64::NAN; data.len()];

            // Use cumulative sum approach for efficiency
            if !data.is_empty() {
                let mut cumsum = vec![0.0; data.len() + 1];
                for i in 0..data.len() {
                    cumsum[i + 1] = cumsum[i] + data[i];
                }

                for i in window_size - 1..data.len() {
                    let window_start = i + 1 - window_size;
                    result[i] = cumsum[i + 1] - cumsum[window_start];
                }
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            let mut result = vec![f64::NAN; data.len()];
            for i in window_size - 1..data.len() {
                let window_start = i + 1 - window_size;
                let window_end = i + 1;
                result[i] = data[window_start..window_end].iter().sum::<f64>();
            }
            Ok(result)
        }
    }

    /// GPU-accelerated rolling standard deviation implementation.
    ///
    /// NOTE: always uses `ddof = 1` (sample standard deviation); unlike
    /// [`GpuDataFrameRolling::std`]'s CPU/JIT-fallback path
    /// (`GpuDataFrameRolling::cpu_rolling_std`), this GPU-dispatch path has
    /// no way to receive a caller-specified `ddof` (the
    /// `WindowFunctionKey`/`execute_gpu_operation` dispatch this is reached
    /// through carries no such parameter). Only reachable for datasets at
    /// least `GpuWindowContext::should_use_gpu`'s size threshold, and — per
    /// this module's honesty note — computes on the CPU regardless.
    fn gpu_rolling_std(&self, data: &[f64], window_size: usize) -> Result<Vec<f64>> {
        let variance = self.gpu_rolling_var(data, window_size)?;
        Ok(variance
            .into_iter()
            .map(|v| if v.is_nan() { f64::NAN } else { v.sqrt() })
            .collect())
    }

    /// GPU-accelerated rolling variance implementation.
    ///
    /// NOTE: always uses `ddof = 1`; see the caveat on
    /// [`Self::gpu_rolling_std`].
    fn gpu_rolling_var(&self, data: &[f64], window_size: usize) -> Result<Vec<f64>> {
        if window_size == 0 {
            // `window_size - 1` below would underflow `usize` and panic.
            return Ok(vec![f64::NAN; data.len()]);
        }

        #[cfg(cuda_available)]
        {
            let mut result = vec![f64::NAN; data.len()];

            // Two-pass algorithm for numerical stability
            for i in window_size - 1..data.len() {
                let window_start = i + 1 - window_size;
                let window_end = i + 1;
                let window_data = &data[window_start..window_end];

                if window_data.len() <= 1 {
                    result[i] = f64::NAN;
                    continue;
                }

                // Calculate mean
                let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;

                // Calculate variance
                let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (window_data.len() - 1) as f64;

                result[i] = variance;
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            let mut result = vec![f64::NAN; data.len()];
            for i in window_size - 1..data.len() {
                let window_start = i + 1 - window_size;
                let window_end = i + 1;
                let window_data = &data[window_start..window_end];

                if window_data.len() <= 1 {
                    result[i] = f64::NAN;
                    continue;
                }

                let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
                let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (window_data.len() - 1) as f64;

                result[i] = variance;
            }
            Ok(result)
        }
    }

    /// GPU-accelerated expanding mean implementation
    fn gpu_expanding_mean(&self, data: &[f64]) -> Result<Vec<f64>> {
        #[cfg(cuda_available)]
        {
            let mut result = vec![f64::NAN; data.len()];
            let mut cumsum = 0.0;

            for i in 0..data.len() {
                cumsum += data[i];
                result[i] = cumsum / (i + 1) as f64;
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            let mut result = vec![f64::NAN; data.len()];
            let mut cumsum = 0.0;

            for i in 0..data.len() {
                cumsum += data[i];
                result[i] = cumsum / (i + 1) as f64;
            }

            Ok(result)
        }
    }

    /// GPU-accelerated expanding sum implementation
    fn gpu_expanding_sum(&self, data: &[f64]) -> Result<Vec<f64>> {
        #[cfg(cuda_available)]
        {
            let mut result = vec![0.0; data.len()];
            let mut cumsum = 0.0;

            for i in 0..data.len() {
                cumsum += data[i];
                result[i] = cumsum;
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            let mut result = vec![0.0; data.len()];
            let mut cumsum = 0.0;

            for i in 0..data.len() {
                cumsum += data[i];
                result[i] = cumsum;
            }

            Ok(result)
        }
    }

    /// GPU-accelerated exponentially weighted moving mean implementation
    fn gpu_ewm_mean(&self, data: &[f64], alpha: f64) -> Result<Vec<f64>> {
        #[cfg(cuda_available)]
        {
            let mut result = vec![f64::NAN; data.len()];

            if !data.is_empty() {
                result[0] = data[0];

                for i in 1..data.len() {
                    result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
                }
            }

            Ok(result)
        }

        #[cfg(not(cuda_available))]
        {
            let mut result = vec![f64::NAN; data.len()];

            if !data.is_empty() {
                result[0] = data[0];

                for i in 1..data.len() {
                    result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
                }
            }

            Ok(result)
        }
    }

    /// Get combined JIT and GPU statistics
    pub fn combined_stats(&self) -> Result<(JitWindowStats, GpuWindowStats)> {
        let jit_stats = self.jit_context.stats()?;
        let gpu_stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?.clone();
        Ok((jit_stats, gpu_stats))
    }

    /// Clear all caches (JIT and GPU memory)
    pub fn clear_caches(&self) -> Result<()> {
        let _ = self.jit_context.clear_cache();
        let mut cache = lock_safe!(self.memory_cache, "gpu window memory cache lock")?;
        cache.clear();
        Ok(())
    }

    /// Enable or disable GPU acceleration
    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.gpu_enabled = enabled;
    }

    /// Set GPU threshold size
    pub fn set_gpu_threshold_size(&mut self, threshold: usize) {
        self.gpu_threshold_size = threshold;
    }

    /// Get GPU usage statistics summary
    pub fn gpu_summary(&self) -> Result<String> {
        let stats = lock_safe!(self.gpu_stats, "gpu window stats lock")?;
        Ok(format!(
            "GPU Window Operations Summary (kernels currently execute on CPU):\n\
             • GPU-dispatch Executions: {}\n\
             • CPU Fallbacks: {}\n\
             • GPU-dispatch Ratio: {:.2}%\n\
             • Memory Allocation Success Rate: {:.2}%\n\
             • Total GPU Memory Used: {:.2} MB\n\
             • Transfer Efficiency: {:.2} GB/s",
            stats.gpu_executions,
            stats.cpu_fallbacks,
            stats.gpu_usage_ratio() * 100.0,
            stats.allocation_success_rate() * 100.0,
            stats.total_gpu_memory_allocated as f64 / (1024.0 * 1024.0),
            stats.transfer_efficiency * 1e9 / (1024.0 * 1024.0 * 1024.0)
        ))
    }
}

impl Default for GpuWindowContext {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            // Fallback to JIT-only context if GPU initialization fails
            let jit_context = JitWindowContext::new();
            Self {
                jit_context,
                gpu_manager: GpuManager::new(), // Use default GPU manager
                gpu_stats: Arc::new(Mutex::new(GpuWindowStats::new())),
                gpu_threshold_size: 50_000,
                memory_cache: Arc::new(Mutex::new(HashMap::new())),
                gpu_enabled: false, // Disable GPU if initialization failed
            }
        })
    }
}

/// GPU-enhanced rolling operations for DataFrames
pub struct GpuDataFrameRolling<'a> {
    dataframe: &'a DataFrame,
    window_size: usize,
    gpu_context: &'a GpuWindowContext,
    min_periods: Option<usize>,
    center: bool,
    columns: Option<Vec<String>>,
}

impl<'a> GpuDataFrameRolling<'a> {
    /// Create new GPU-enhanced rolling operations
    pub fn new(
        dataframe: &'a DataFrame,
        window_size: usize,
        gpu_context: &'a GpuWindowContext,
    ) -> Self {
        Self {
            dataframe,
            window_size,
            gpu_context,
            min_periods: None,
            center: false,
            columns: None,
        }
    }

    /// Set minimum periods
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    /// Set center alignment
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set specific columns to operate on
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Execute GPU-enhanced rolling mean
    pub fn mean(self) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingMean, 1)
    }

    /// Execute GPU-enhanced rolling sum
    pub fn sum(self) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingSum, 1)
    }

    /// Execute GPU-enhanced rolling standard deviation.
    ///
    /// `ddof` (delta degrees of freedom) is honored on the CPU/JIT-fallback
    /// path (`ddof = 1` for the sample standard deviation, `ddof = 0` for
    /// the population standard deviation) — the previous version accepted
    /// `ddof` but silently ignored it (`_ddof`), always computing `ddof =
    /// 1` regardless of what was requested. The GPU-dispatch path (only
    /// reachable for datasets at least [`GpuWindowContext::should_use_gpu`]'s
    /// threshold in size, and identical CPU computation under the hood; see
    /// this module's honesty note) still hardcodes `ddof = 1`; see
    /// [`GpuWindowContext::gpu_rolling_var`].
    pub fn std(self, ddof: usize) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingStd, ddof)
    }

    /// Execute GPU-enhanced rolling variance. See [`Self::std`] for the
    /// `ddof` caveat on the GPU-dispatch path.
    pub fn var(self, ddof: usize) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingVar, ddof)
    }

    /// Execute GPU-enhanced rolling minimum
    pub fn min(self) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingMin, 1)
    }

    /// Execute GPU-enhanced rolling maximum
    pub fn max(self) -> Result<DataFrame> {
        self.execute_rolling_operation(WindowOpType::RollingMax, 1)
    }

    /// Execute a rolling operation with GPU acceleration when beneficial.
    /// `ddof` is only consulted for `RollingVar`/`RollingStd` on the
    /// CPU/JIT-fallback path; other operations ignore it.
    fn execute_rolling_operation(self, op_type: WindowOpType, ddof: usize) -> Result<DataFrame> {
        let mut result_df = DataFrame::new();

        // Determine which columns to process
        let target_columns = if let Some(ref cols) = self.columns {
            cols.clone()
        } else {
            // Get numeric columns
            self.dataframe
                .column_names()
                .iter()
                .filter(|col_name| {
                    // Try to get as f64 to check if numeric
                    self.dataframe.get_column::<f64>(col_name).is_ok()
                })
                .map(|s| s.clone())
                .collect()
        };

        // Process each column
        for col_name in target_columns {
            if let Ok(series) = self.dataframe.get_column::<f64>(&col_name) {
                let data = series.values();
                let data_size = data.len();

                // Create function key for caching
                let key = WindowFunctionKey::new(
                    op_type.clone(),
                    Some(self.window_size),
                    "f64".to_string(),
                );

                // Decide between GPU, JIT, or standard implementation
                let processed_data = if self.gpu_context.should_use_gpu(data_size, &op_type) {
                    // Try GPU acceleration first
                    match self
                        .gpu_context
                        .execute_gpu_operation(&key, data, self.window_size)
                    {
                        Ok(gpu_result) => gpu_result,
                        Err(_) => {
                            // Fallback to JIT implementation
                            self.fallback_to_jit_operation(&op_type, data, ddof)?
                        }
                    }
                } else {
                    // Use JIT or standard implementation based on threshold
                    self.fallback_to_jit_operation(&op_type, data, ddof)?
                };

                // Create result series. `processed_data` is already
                // `Vec<f64>`: stringifying it (the previous
                // `.map(|v| v.to_string())`) turned every numeric window
                // result -- including `.mean()`, whose whole job is to
                // produce a number -- into a `Series<String>`, silently
                // breaking any further numeric use (arithmetic, plotting,
                // more window ops) of the result.
                let result_series = Series::new(processed_data, Some(col_name.clone()))?;

                result_df.add_column(col_name, result_series)?;
            }
        }

        Ok(result_df)
    }

    /// Fallback to JIT implementation when GPU is not suitable
    fn fallback_to_jit_operation(
        &self,
        op_type: &WindowOpType,
        data: &[f64],
        ddof: usize,
    ) -> Result<Vec<f64>> {
        // This would integrate with the existing JIT window operations
        // For now, implement a simple CPU version
        match op_type {
            WindowOpType::RollingMean => self.cpu_rolling_mean(data),
            WindowOpType::RollingSum => self.cpu_rolling_sum(data),
            WindowOpType::RollingStd => self.cpu_rolling_std(data, ddof),
            WindowOpType::RollingVar => self.cpu_rolling_var(data, ddof),
            WindowOpType::RollingMin => self.cpu_rolling_min(data),
            WindowOpType::RollingMax => self.cpu_rolling_max(data),
            _ => Err(Error::InvalidOperation(format!(
                "Fallback not implemented for {:?}",
                op_type
            ))),
        }
    }

    /// CPU implementation of rolling mean
    fn cpu_rolling_mean(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![f64::NAN; data.len()];
        if self.window_size == 0 || self.window_size > data.len() {
            return Ok(result);
        }
        for i in self.window_size - 1..data.len() {
            // Use i + 1 - window_size to avoid usize underflow
            let window_start = i + 1 - self.window_size;
            let window_end = i + 1;
            let window_data = &data[window_start..window_end];
            result[i] = window_data.iter().sum::<f64>() / window_data.len() as f64;
        }
        Ok(result)
    }

    /// CPU implementation of rolling sum
    fn cpu_rolling_sum(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![f64::NAN; data.len()];
        if self.window_size == 0 || self.window_size > data.len() {
            return Ok(result);
        }
        for i in self.window_size - 1..data.len() {
            // Use i + 1 - window_size to avoid usize underflow
            let window_start = i + 1 - self.window_size;
            let window_end = i + 1;
            result[i] = data[window_start..window_end].iter().sum::<f64>();
        }
        Ok(result)
    }

    /// CPU implementation of rolling standard deviation. `ddof` (delta
    /// degrees of freedom) is forwarded to [`Self::cpu_rolling_var`]; see
    /// its doc comment.
    fn cpu_rolling_std(&self, data: &[f64], ddof: usize) -> Result<Vec<f64>> {
        let variance = self.cpu_rolling_var(data, ddof)?;
        Ok(variance
            .into_iter()
            .map(|v| if v.is_nan() { f64::NAN } else { v.sqrt() })
            .collect())
    }

    /// CPU implementation of rolling variance.
    ///
    /// `ddof` is the delta degrees of freedom (`1` for the sample variance,
    /// `0` for the population variance) — this used to be silently ignored
    /// (`std`/`var`'s public parameter was named `_ddof` and the divisor
    /// here was hardcoded to `window_data.len() - 1`), so a caller
    /// requesting the population variance (`ddof = 0`) got the sample
    /// variance instead.
    fn cpu_rolling_var(&self, data: &[f64], ddof: usize) -> Result<Vec<f64>> {
        let mut result = vec![f64::NAN; data.len()];
        if self.window_size == 0 || self.window_size > data.len() {
            return Ok(result);
        }
        for i in self.window_size - 1..data.len() {
            // Use i + 1 - window_size to avoid usize underflow
            let window_start = i + 1 - self.window_size;
            let window_end = i + 1;
            let window_data = &data[window_start..window_end];

            if window_data.len() <= ddof {
                // Too few observations for this many degrees of freedom
                // (e.g. `ddof = 1` needs at least 2 points): the divisor
                // below would underflow `usize` (or, for `ddof = 0` and an
                // empty window, divide by zero).
                result[i] = f64::NAN;
                continue;
            }

            let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
            let variance = window_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (window_data.len() - ddof) as f64;

            result[i] = variance;
        }
        Ok(result)
    }

    /// CPU implementation of rolling minimum.
    ///
    /// `GpuDataFrameRolling::min` (and the `RollingMin` GPU-dispatch case,
    /// which always falls back here since `GpuWindowContext::
    /// execute_gpu_operation` has no kernel for it either) used to have no
    /// fallback case at all in `fallback_to_jit_operation`, so calling
    /// `.min()` unconditionally returned `Err("Fallback not implemented for
    /// RollingMin")` regardless of GPU availability.
    fn cpu_rolling_min(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![f64::NAN; data.len()];
        if self.window_size == 0 || self.window_size > data.len() {
            return Ok(result);
        }
        for i in self.window_size - 1..data.len() {
            let window_start = i + 1 - self.window_size;
            let window_end = i + 1;
            result[i] = data[window_start..window_end]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
        }
        Ok(result)
    }

    /// CPU implementation of rolling maximum. See `cpu_rolling_min` for why
    /// this fallback needed to exist at all (`.max()` previously always
    /// returned `Err`).
    fn cpu_rolling_max(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![f64::NAN; data.len()];
        if self.window_size == 0 || self.window_size > data.len() {
            return Ok(result);
        }
        for i in self.window_size - 1..data.len() {
            let window_start = i + 1 - self.window_size;
            let window_end = i + 1;
            result[i] = data[window_start..window_end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
        }
        Ok(result)
    }
}

/// GPU-enhanced window extension trait for DataFrame
pub trait GpuDataFrameWindowExt {
    /// Create GPU-enhanced rolling operations
    fn gpu_rolling<'a>(
        &'a self,
        window_size: usize,
        gpu_context: &'a GpuWindowContext,
    ) -> GpuDataFrameRolling<'a>;
}

impl GpuDataFrameWindowExt for DataFrame {
    fn gpu_rolling<'a>(
        &'a self,
        window_size: usize,
        gpu_context: &'a GpuWindowContext,
    ) -> GpuDataFrameRolling<'a> {
        GpuDataFrameRolling::new(self, window_size, gpu_context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_gpu_context_creation() {
        // Test that GPU context can be created (may fallback to CPU-only)
        let context = GpuWindowContext::default();
        assert!(context.gpu_threshold_size > 0);
    }

    #[test]
    fn test_gpu_threshold_logic() {
        let context = GpuWindowContext::default();

        // Small datasets should not use GPU
        assert!(!context.should_use_gpu(1000, &WindowOpType::RollingMean));

        // Large datasets should potentially use GPU (if available)
        let should_use = context.should_use_gpu(100_000, &WindowOpType::RollingMean);
        // Result depends on GPU availability, but logic should work
        assert!(should_use || !context.gpu_enabled || !context.gpu_manager.is_available());
    }

    #[test]
    fn test_gpu_rolling_mean_fallback() -> Result<()> {
        let mut df = DataFrame::new();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let series = Series::new(data, Some("test_col".to_string()))?;
        df.add_column("test_col".to_string(), series)?;

        let context = GpuWindowContext::default();
        let result = df.gpu_rolling(3, &context).mean()?;

        // Check that result has correct structure
        assert!(result.column_names().contains(&"test_col".to_string()));
        assert_eq!(result.row_count(), 10);

        Ok(())
    }

    #[test]
    fn test_gpu_stats_tracking() {
        let mut stats = GpuWindowStats::new();

        // Test recording executions (real wall-clock times; no fabricated speedup)
        stats.record_gpu_execution(1000);
        stats.record_gpu_execution(800);
        stats.record_cpu_fallback();

        assert_eq!(stats.gpu_executions, 2);
        assert_eq!(stats.cpu_fallbacks, 1);
        assert_eq!(stats.gpu_usage_ratio(), 2.0 / 3.0);
        assert_eq!(stats.total_kernel_time_ns, 1800);
    }
}

//! Multi-GPU coordination scaffolding
//!
//! This module provides the structure for distributing computations across
//! multiple GPU devices (data/model/pipeline-parallel splitting, result
//! collection, and load-balancing bookkeeping).
//!
//! HONESTY NOTE: cudarc 0.19.x does not expose the per-device kernels needed to
//! actually run these computations on the GPU, so the numerical work (e.g.
//! `matmul_on_device`) currently executes on the **CPU**, and there is no real
//! peer-to-peer transfer or device synchronization. The matrix splitting and
//! result collection are real CPU operations; the device-level steps are
//! documented honestly as not-yet-implemented rather than being faked.

use scirs2_core::ndarray::{s, Array2};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};
use crate::gpu::operations::GpuMatrix;
use crate::gpu::{GpuConfig, GpuDeviceStatus, GpuError, GpuManager};

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// List of device IDs to use
    pub device_ids: Vec<i32>,
    /// Strategy for distributing work across devices
    pub distribution_strategy: DistributionStrategy,
    /// Whether to enable peer-to-peer memory access between devices.
    ///
    /// Currently inert: [`MultiGpuManager::is_p2p_available`] always reports
    /// `false` (see [`MultiGpuManager::check_p2p_support`]) regardless of
    /// this flag, because there is no real device-to-device transfer path
    /// in this crate to enable in the first place (the compute path runs on
    /// the CPU; see the module-level honesty note above). This field is
    /// kept for forward compatibility with a real P2P implementation.
    pub enable_p2p: bool,
    /// Memory limit per device (in bytes)
    pub memory_limit_per_device: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0], // Single GPU by default
            distribution_strategy: DistributionStrategy::DataParallel,
            enable_p2p: true,
            memory_limit_per_device: 1024 * 1024 * 1024, // 1GB per device
        }
    }
}

/// Strategy for distributing computation across multiple GPUs
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionStrategy {
    /// Data parallel - split data across devices, same operation on each
    DataParallel,
    /// Model parallel - split model/computation across devices
    ModelParallel,
    /// Pipeline parallel - different stages on different devices
    PipelineParallel,
    /// Custom distribution based on operation type
    Custom,
}

/// Multi-GPU manager for coordinating operations across multiple devices
#[derive(Clone)]
pub struct MultiGpuManager {
    /// Configuration
    config: MultiGpuConfig,
    /// Individual GPU managers for each device
    device_managers: HashMap<i32, GpuManager>,
    /// Device capabilities and status
    device_statuses: HashMap<i32, GpuDeviceStatus>,
    /// Whether peer-to-peer transfers are available
    p2p_available: bool,
}

impl MultiGpuManager {
    /// Create a new multi-GPU manager
    pub fn new(config: MultiGpuConfig) -> Result<Self> {
        if config.device_ids.is_empty() {
            return Err(Error::InvalidValue(
                "MultiGpuConfig::device_ids must list at least one device".to_string(),
            ));
        }

        let mut device_managers = HashMap::new();
        let mut device_statuses = HashMap::new();

        // Initialize GPU managers for each device
        for &device_id in &config.device_ids {
            let device_config = GpuConfig {
                device_id,
                memory_limit: config.memory_limit_per_device,
                ..GpuConfig::default()
            };

            let manager = GpuManager::with_config(device_config);
            let status = manager.device_info();

            device_managers.insert(device_id, manager);
            device_statuses.insert(device_id, status);
        }

        // Check P2P capabilities
        let p2p_available = Self::check_p2p_support(&config.device_ids);

        Ok(Self {
            config,
            device_managers,
            device_statuses,
            p2p_available,
        })
    }

    /// Check whether peer-to-peer memory access is supported between the
    /// configured devices.
    ///
    /// A real implementation would query `cuDeviceCanAccessPeer` for each device
    /// pair. That is not implemented, and P2P is never actually used (the compute
    /// path runs on the CPU), so this conservatively reports `false` rather than
    /// fabricating P2P availability.
    fn check_p2p_support(_device_ids: &[i32]) -> bool {
        false
    }

    /// Get the number of available devices
    pub fn device_count(&self) -> usize {
        self.config.device_ids.len()
    }

    /// Whether peer-to-peer transfers are available between the configured
    /// devices, as reported by [`Self::check_p2p_support`].
    pub fn is_p2p_available(&self) -> bool {
        self.p2p_available
    }

    /// Get device status for all devices
    pub fn get_device_statuses(&self) -> &HashMap<i32, GpuDeviceStatus> {
        &self.device_statuses
    }

    /// Distribute matrix across multiple GPUs
    pub fn distribute_matrix(&self, matrix: &GpuMatrix) -> Result<Vec<(i32, GpuMatrix)>> {
        match self.config.distribution_strategy {
            DistributionStrategy::DataParallel => self.distribute_data_parallel(matrix),
            DistributionStrategy::ModelParallel => self.distribute_model_parallel(matrix),
            DistributionStrategy::PipelineParallel => self.distribute_pipeline(matrix),
            DistributionStrategy::Custom => self.distribute_custom(matrix),
        }
    }

    /// Data parallel distribution - split rows across devices
    fn distribute_data_parallel(&self, matrix: &GpuMatrix) -> Result<Vec<(i32, GpuMatrix)>> {
        let num_devices = self.device_count();
        let rows = matrix.data.shape()[0];

        let rows_per_device = (rows + num_devices - 1) / num_devices; // Ceiling division
        let mut distributed = Vec::new();

        for (i, &device_id) in self.config.device_ids.iter().enumerate() {
            let start_row = i * rows_per_device;
            let end_row = ((i + 1) * rows_per_device).min(rows);

            if start_row < rows {
                let chunk = matrix.data.slice(s![start_row..end_row, ..]).to_owned();
                let gpu_chunk = GpuMatrix {
                    data: chunk,
                    on_gpu: false, // Will be transferred to specific device when needed
                };
                distributed.push((device_id, gpu_chunk));
            }
        }

        Ok(distributed)
    }

    /// Model parallel distribution - split columns across devices
    fn distribute_model_parallel(&self, matrix: &GpuMatrix) -> Result<Vec<(i32, GpuMatrix)>> {
        let num_devices = self.device_count();
        let cols = matrix.data.shape()[1];

        let cols_per_device = (cols + num_devices - 1) / num_devices;
        let mut distributed = Vec::new();

        for (i, &device_id) in self.config.device_ids.iter().enumerate() {
            let start_col = i * cols_per_device;
            let end_col = ((i + 1) * cols_per_device).min(cols);

            if start_col < cols {
                let chunk = matrix.data.slice(s![.., start_col..end_col]).to_owned();
                let gpu_chunk = GpuMatrix {
                    data: chunk,
                    on_gpu: false,
                };
                distributed.push((device_id, gpu_chunk));
            }
        }

        Ok(distributed)
    }

    /// Pipeline parallel distribution - different processing stages on different devices
    fn distribute_pipeline(&self, matrix: &GpuMatrix) -> Result<Vec<(i32, GpuMatrix)>> {
        // For pipeline parallel, we don't split the matrix but assign different operations
        // to different devices. For now, just replicate the matrix to the first device.
        if let Some(&first_device) = self.config.device_ids.first() {
            Ok(vec![(first_device, GpuMatrix::new(matrix.to_cpu()?))])
        } else {
            Err(Error::from(GpuError::DeviceError(
                "No devices available".to_string(),
            )))
        }
    }

    /// Custom distribution strategy
    fn distribute_custom(&self, matrix: &GpuMatrix) -> Result<Vec<(i32, GpuMatrix)>> {
        // Custom strategy could be based on matrix properties, device capabilities, etc.
        // For now, fall back to data parallel
        self.distribute_data_parallel(matrix)
    }

    /// Collect distributed results and combine them
    pub fn collect_results(&self, distributed_results: Vec<(i32, GpuMatrix)>) -> Result<GpuMatrix> {
        if distributed_results.is_empty() {
            return Err(Error::from(GpuError::KernelExecutionError(
                "No results to collect".to_string(),
            )));
        }

        match self.config.distribution_strategy {
            DistributionStrategy::DataParallel => self.collect_data_parallel(distributed_results),
            DistributionStrategy::ModelParallel => self.collect_model_parallel(distributed_results),
            DistributionStrategy::PipelineParallel => self.collect_pipeline(distributed_results),
            DistributionStrategy::Custom => self.collect_custom(distributed_results),
        }
    }

    /// Collect data parallel results - concatenate along rows
    fn collect_data_parallel(
        &self,
        mut distributed_results: Vec<(i32, GpuMatrix)>,
    ) -> Result<GpuMatrix> {
        // Sort by device ID to maintain order
        distributed_results.sort_by_key(|(device_id, _)| *device_id);

        let matrices: Vec<Array2<f64>> = distributed_results
            .into_iter()
            .map(|(_, matrix)| matrix.data)
            .collect();

        if matrices.is_empty() {
            return Err(Error::from(GpuError::KernelExecutionError(
                "No matrices to concatenate".to_string(),
            )));
        }

        // Concatenate along axis 0 (rows): every chunk must agree on the
        // column count, or the `slice_mut(..).assign(..)` below would panic
        // partway through instead of reporting a clear dimension error.
        let cols = matrices[0].shape()[1];
        for (i, matrix) in matrices.iter().enumerate() {
            if matrix.shape()[1] != cols {
                return Err(Error::DimensionMismatch(format!(
                    "Data-parallel result chunk {} has {} columns, expected {} (from chunk 0)",
                    i,
                    matrix.shape()[1],
                    cols
                )));
            }
        }
        let total_rows: usize = matrices.iter().map(|m| m.shape()[0]).sum();

        let mut result = Array2::zeros((total_rows, cols));
        let mut current_row = 0;

        for matrix in matrices {
            let matrix_rows = matrix.shape()[0];
            result
                .slice_mut(s![current_row..current_row + matrix_rows, ..])
                .assign(&matrix);
            current_row += matrix_rows;
        }

        Ok(GpuMatrix {
            data: result,
            on_gpu: false,
        })
    }

    /// Collect model parallel results - concatenate along columns
    fn collect_model_parallel(
        &self,
        mut distributed_results: Vec<(i32, GpuMatrix)>,
    ) -> Result<GpuMatrix> {
        distributed_results.sort_by_key(|(device_id, _)| *device_id);

        let matrices: Vec<Array2<f64>> = distributed_results
            .into_iter()
            .map(|(_, matrix)| matrix.data)
            .collect();

        if matrices.is_empty() {
            return Err(Error::from(GpuError::KernelExecutionError(
                "No matrices to concatenate".to_string(),
            )));
        }

        // Concatenate along axis 1 (columns): every chunk must agree on the
        // row count, or the `slice_mut(..).assign(..)` below would panic
        // partway through instead of reporting a clear dimension error.
        let rows = matrices[0].shape()[0];
        for (i, matrix) in matrices.iter().enumerate() {
            if matrix.shape()[0] != rows {
                return Err(Error::DimensionMismatch(format!(
                    "Model-parallel result chunk {} has {} rows, expected {} (from chunk 0)",
                    i,
                    matrix.shape()[0],
                    rows
                )));
            }
        }
        let total_cols: usize = matrices.iter().map(|m| m.shape()[1]).sum();

        let mut result = Array2::zeros((rows, total_cols));
        let mut current_col = 0;

        for matrix in matrices {
            let matrix_cols = matrix.shape()[1];
            result
                .slice_mut(s![.., current_col..current_col + matrix_cols])
                .assign(&matrix);
            current_col += matrix_cols;
        }

        Ok(GpuMatrix {
            data: result,
            on_gpu: false,
        })
    }

    /// Collect pipeline results - just return the final result
    fn collect_pipeline(&self, distributed_results: Vec<(i32, GpuMatrix)>) -> Result<GpuMatrix> {
        if let Some((_, result)) = distributed_results.into_iter().last() {
            Ok(result)
        } else {
            Err(Error::from(GpuError::KernelExecutionError(
                "No pipeline result".to_string(),
            )))
        }
    }

    /// Collect custom results
    fn collect_custom(&self, distributed_results: Vec<(i32, GpuMatrix)>) -> Result<GpuMatrix> {
        // Custom collection strategy - for now, fall back to data parallel
        self.collect_data_parallel(distributed_results)
    }

    /// Perform distributed matrix multiplication
    pub fn distributed_matmul(&self, a: &GpuMatrix, b: &GpuMatrix) -> Result<GpuMatrix> {
        if a.data.shape()[1] != b.data.shape()[0] {
            return Err(Error::DimensionMismatch(format!(
                "Incompatible dimensions for matrix multiplication: {:?} and {:?}",
                a.data.shape(),
                b.data.shape()
            )));
        }

        // Model-parallel splits A by columns (the contraction dimension), so
        // it needs a matching row-split of B and a *sum* of partial
        // products — a fundamentally different combination step than the
        // "each device computes independent full rows/columns, then
        // concatenate" pattern the other strategies share. Route it to its
        // own implementation rather than forcing it through
        // `distribute_matrix`/`collect_results` (which only ever split and
        // recombine `a`, leaving every device holding the same *entire* `b`
        // — dimension-mismatched against an A-chunk with fewer columns than
        // `b` has rows, and provably wrong for any other device count).
        if self.config.distribution_strategy == DistributionStrategy::ModelParallel {
            return self.distributed_matmul_model_parallel(a, b);
        }

        // Distribute matrix A across devices
        let distributed_a = self.distribute_matrix(a)?;

        // Each device computes its chunk of the result
        let mut distributed_results = Vec::new();

        for (device_id, a_chunk) in distributed_a {
            if self.device_managers.contains_key(&device_id) {
                // For data parallel, each device multiplies its chunk of A with full B
                // In a real implementation, this would be done on the specific GPU device
                let result_chunk = self.matmul_on_device(&a_chunk, b, device_id)?;
                distributed_results.push((device_id, result_chunk));
            }
        }

        // Collect and combine results
        self.collect_results(distributed_results)
    }

    /// Model-parallel matrix multiplication.
    ///
    /// `A` (`m x k`) is split into column blocks `A_1 .. A_d` (one per
    /// device) via [`Self::distribute_model_parallel`]; `B` (`k x n`) is
    /// split here into the *matching* row blocks `B_1 .. B_d`, so each
    /// device computes a full-shape (`m x n`) **partial** product `A_i ·
    /// B_i`. The final result is the element-wise **sum** of those partial
    /// products (`A · B = sum_i A_i · B_i`), not a concatenation — the
    /// previous version handed every device the entire, un-split `B` and
    /// then concatenated the (dimension-mismatched, when more than one
    /// device was configured) results along columns, which both panicked
    /// inside `ndarray`'s `dot` for `device_ids.len() > 1` and, had it not
    /// panicked, would not have reconstructed `A · B` anyway.
    fn distributed_matmul_model_parallel(&self, a: &GpuMatrix, b: &GpuMatrix) -> Result<GpuMatrix> {
        let distributed_a = self.distribute_model_parallel(a)?;

        let mut col_offset = 0usize;
        let mut partial_sum: Option<Array2<f64>> = None;

        for (device_id, a_chunk) in distributed_a {
            let chunk_cols = a_chunk.data.shape()[1];
            let b_chunk = GpuMatrix {
                data: b
                    .data
                    .slice(s![col_offset..col_offset + chunk_cols, ..])
                    .to_owned(),
                on_gpu: false,
            };
            col_offset += chunk_cols;

            let partial = self.matmul_on_device(&a_chunk, &b_chunk, device_id)?;
            partial_sum = Some(match partial_sum {
                Some(acc) => acc + &partial.data,
                None => partial.data,
            });
        }

        let data = partial_sum.ok_or_else(|| {
            Error::from(GpuError::KernelExecutionError(
                "No devices available for model-parallel matmul".to_string(),
            ))
        })?;

        Ok(GpuMatrix {
            data,
            on_gpu: false,
        })
    }

    /// Multiply two matrices for the given device id.
    ///
    /// A real implementation would transfer the operands to the specified GPU
    /// device and run the multiplication with that device's CUDA context.
    /// cudarc 0.19.x exposes no such path here, so the multiplication is
    /// performed on the **CPU** (the result is correct; `on_gpu` is `false`).
    fn matmul_on_device(&self, a: &GpuMatrix, b: &GpuMatrix, device_id: i32) -> Result<GpuMatrix> {
        let _ = device_id; // per-device routing is not implemented; compute on CPU
        let result_data = a.data.dot(&b.data);
        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }

    /// Synchronize all devices.
    ///
    /// Currently a no-op: the multi-GPU compute path runs on the CPU (see
    /// `matmul_on_device`), so there is no outstanding device work to
    /// synchronize. A real implementation would call `cuDeviceSynchronize` (or
    /// the cudarc equivalent) per device. This returns `Ok(())` to reflect that
    /// there is nothing to wait on, not that a device sync was performed.
    pub fn synchronize_all(&self) -> Result<()> {
        Ok(())
    }

    /// Get memory usage across all devices
    pub fn get_memory_usage(&self) -> HashMap<i32, (usize, usize)> {
        let mut usage = HashMap::new();

        for (&device_id, status) in &self.device_statuses {
            let used_memory = status
                .total_memory
                .unwrap_or(0)
                .saturating_sub(status.free_memory.unwrap_or(0));
            let total_memory = status.total_memory.unwrap_or(0);
            usage.insert(device_id, (used_memory, total_memory));
        }

        usage
    }

    /// Balance load across devices based on current memory usage
    pub fn balance_load(&mut self) -> Result<()> {
        // Get current memory usage
        let memory_usage = self.get_memory_usage();

        // Find devices with low utilization
        let mut low_util_devices = Vec::new();
        let mut high_util_devices = Vec::new();

        for (&device_id, &(used, total)) in &memory_usage {
            // `total == 0` means the device's memory status was never
            // populated (`GpuDeviceStatus::total_memory` is `None`, e.g. no
            // real GPU detected), not "0% utilized" or "100% utilized" —
            // `0.0 / 0.0` is NaN, which compares `false` to both `< 0.3` and
            // `> 0.8`, so such devices were already silently excluded from
            // both buckets; make that explicit instead of relying on NaN's
            // comparison behavior to do it implicitly.
            if total == 0 {
                continue;
            }
            let utilization = used as f64 / total as f64;
            if utilization < 0.3 {
                low_util_devices.push(device_id);
            } else if utilization > 0.8 {
                high_util_devices.push(device_id);
            }
        }

        // In a real implementation, would migrate work from high to low utilization devices
        // For now, just log the information
        log::info!(
            "Load balancing: {} low util devices, {} high util devices",
            low_util_devices.len(),
            high_util_devices.len()
        );

        Ok(())
    }
}

/// The global multi-GPU manager. Holding the `Arc` itself in the `OnceLock`
/// (rather than the bare `MultiGpuManager`) is what lets
/// `get_multi_gpu_manager` hand out the *same* shared instance below.
static MULTI_GPU_MANAGER: OnceLock<Arc<Mutex<MultiGpuManager>>> = OnceLock::new();

/// Initialize global multi-GPU manager
pub fn init_multi_gpu(config: MultiGpuConfig) -> Result<()> {
    let manager = MultiGpuManager::new(config)?;

    MULTI_GPU_MANAGER
        .set(Arc::new(Mutex::new(manager)))
        .map_err(|_| {
            Error::InvalidOperation("Multi-GPU manager already initialized".to_string())
        })?;

    Ok(())
}

/// Get the global multi-GPU manager.
///
/// Returns the *same* `Arc` on every call (constructing it with the default
/// configuration on first use if [`init_multi_gpu`] was never called
/// explicitly), not a snapshot clone of the manager's current state. The
/// previous version reconstructed a brand new `MultiGpuManager` — via a
/// hand-rolled `Clone` impl that re-ran device enumeration and could panic
/// via `.expect(..)` if even a single-device fallback construction failed —
/// wrapped in a brand new `Arc<Mutex<_>>` on every call, so any mutation a
/// caller made through the returned handle (e.g. [`MultiGpuManager::balance_load`],
/// which takes `&mut self`) was invisible to every other caller: there was
/// no shared global state at all, only independent throwaway copies.
/// `MultiGpuManager` is a plain `#[derive(Clone)]` now (a cheap, infallible
/// `Arc`-clone of its `GpuManager`s), so nothing here needs that fallback
/// path any more.
pub fn get_multi_gpu_manager() -> Result<Arc<Mutex<MultiGpuManager>>> {
    if let Some(manager) = MULTI_GPU_MANAGER.get() {
        return Ok(manager.clone());
    }

    // Construct a candidate without holding any lock (construction does
    // real device-enumeration work and is fallible), then publish it
    // atomically. If another thread wins the race, use its published
    // instance instead of ours so callers never end up split across two
    // different "global" managers.
    let candidate = Arc::new(Mutex::new(MultiGpuManager::new(MultiGpuConfig::default())?));
    match MULTI_GPU_MANAGER.set(candidate.clone()) {
        Ok(()) => Ok(candidate),
        Err(_) => Ok(MULTI_GPU_MANAGER.get().cloned().unwrap_or(candidate)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_multi_gpu_manager_creation() {
        let config = MultiGpuConfig {
            device_ids: vec![0],
            ..MultiGpuConfig::default()
        };

        let manager = MultiGpuManager::new(config);
        assert!(manager.is_ok());

        let manager = manager.expect("operation should succeed");
        assert_eq!(manager.device_count(), 1);
    }

    #[test]
    fn test_data_parallel_distribution() {
        let config = MultiGpuConfig {
            device_ids: vec![0, 1],
            distribution_strategy: DistributionStrategy::DataParallel,
            ..MultiGpuConfig::default()
        };

        let manager = MultiGpuManager::new(config).expect("operation should succeed");

        let matrix_data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("operation should succeed");

        let matrix = GpuMatrix {
            data: matrix_data,
            on_gpu: false,
        };

        let distributed = manager
            .distribute_matrix(&matrix)
            .expect("operation should succeed");
        assert_eq!(distributed.len(), 2);

        // Check that chunks have correct sizes
        let total_rows: usize = distributed.iter().map(|(_, m)| m.data.shape()[0]).sum();
        assert_eq!(total_rows, 4);
    }

    #[test]
    fn test_collect_data_parallel() {
        let config = MultiGpuConfig {
            device_ids: vec![0, 1],
            distribution_strategy: DistributionStrategy::DataParallel,
            ..MultiGpuConfig::default()
        };

        let manager = MultiGpuManager::new(config).expect("operation should succeed");

        // Create distributed results
        let chunk1_data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("operation should succeed");
        let chunk2_data = Array2::from_shape_vec((2, 3), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("operation should succeed");

        let distributed_results = vec![
            (
                0,
                GpuMatrix {
                    data: chunk1_data,
                    on_gpu: false,
                },
            ),
            (
                1,
                GpuMatrix {
                    data: chunk2_data,
                    on_gpu: false,
                },
            ),
        ];

        let result = manager
            .collect_results(distributed_results)
            .expect("operation should succeed");
        assert_eq!(result.data.shape(), &[4, 3]);
    }
}

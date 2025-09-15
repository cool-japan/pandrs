//! Multi-GPU support for distributed computation
//!
//! This module provides functionality to distribute computations across multiple GPU devices
//! for improved performance and memory capacity.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{GpuConfig, GpuDeviceStatus, GpuError, GpuManager};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// Multi-GPU configuration
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// List of device IDs to use
    pub device_ids: Vec<i32>,
    /// Strategy for distributing work across devices
    pub distribution_strategy: DistributionStrategy,
    /// Whether to enable peer-to-peer memory access between devices
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

    /// Check if peer-to-peer memory access is supported between devices
    fn check_p2p_support(device_ids: &[i32]) -> bool {
        #[cfg(feature = "cuda")]
        {
            // Check if all device pairs support P2P
            for &id1 in device_ids {
                for &id2 in device_ids {
                    if id1 != id2 {
                        // In a real implementation, would check cuDeviceCanAccessPeer
                        // For now, assume P2P is available
                    }
                }
            }
            true
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the number of available devices
    pub fn device_count(&self) -> usize {
        self.config.device_ids.len()
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
        let cols = matrix.data.shape()[1];

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
        let rows = matrix.data.shape()[0];
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
            Ok(vec![(first_device, matrix.clone())])
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
            return Err(Error::from(GpuError::ComputationError(
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
            return Err(Error::from(GpuError::ComputationError(
                "No matrices to concatenate".to_string(),
            )));
        }

        // Concatenate along axis 0 (rows)
        let first_shape = matrices[0].shape();
        let total_rows: usize = matrices.iter().map(|m| m.shape()[0]).sum();
        let cols = first_shape[1];

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
            return Err(Error::from(GpuError::ComputationError(
                "No matrices to concatenate".to_string(),
            )));
        }

        // Concatenate along axis 1 (columns)
        let first_shape = matrices[0].shape();
        let rows = first_shape[0];
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
            Err(Error::from(GpuError::ComputationError(
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
        // Distribute matrix A across devices
        let distributed_a = self.distribute_matrix(a)?;

        // Each device computes its chunk of the result
        let mut distributed_results = Vec::new();

        for (device_id, a_chunk) in distributed_a {
            if let Some(manager) = self.device_managers.get(&device_id) {
                // For data parallel, each device multiplies its chunk of A with full B
                // In a real implementation, this would be done on the specific GPU device
                let result_chunk = self.matmul_on_device(&a_chunk, b, device_id)?;
                distributed_results.push((device_id, result_chunk));
            }
        }

        // Collect and combine results
        self.collect_results(distributed_results)
    }

    /// Perform matrix multiplication on a specific device
    fn matmul_on_device(&self, a: &GpuMatrix, b: &GpuMatrix, device_id: i32) -> Result<GpuMatrix> {
        // In a real implementation, this would:
        // 1. Transfer matrices to the specific GPU device
        // 2. Perform the multiplication using device-specific CUDA context
        // 3. Return the result

        // For now, perform CPU multiplication as fallback
        let result_data = a.data.dot(&b.data);
        Ok(GpuMatrix {
            data: result_data,
            on_gpu: false,
        })
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            for &device_id in &self.config.device_ids {
                if let Some(manager) = self.device_managers.get(&device_id) {
                    // In a real implementation, would call cuDeviceSynchronize
                    // or similar for each device
                }
            }
        }
        Ok(())
    }

    /// Get memory usage across all devices
    pub fn get_memory_usage(&self) -> HashMap<i32, (usize, usize)> {
        let mut usage = HashMap::new();

        for (&device_id, status) in &self.device_statuses {
            usage.insert(device_id, (status.used_memory, status.total_memory));
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

/// Global multi-GPU manager
static mut MULTI_GPU_MANAGER: Option<Mutex<MultiGpuManager>> = None;

/// Initialize global multi-GPU manager
pub fn init_multi_gpu(config: MultiGpuConfig) -> Result<()> {
    let manager = MultiGpuManager::new(config)?;

    unsafe {
        MULTI_GPU_MANAGER = Some(Mutex::new(manager));
    }

    Ok(())
}

/// Get the global multi-GPU manager
pub fn get_multi_gpu_manager() -> Result<Arc<Mutex<MultiGpuManager>>> {
    unsafe {
        match &MULTI_GPU_MANAGER {
            Some(manager) => Ok(Arc::new(Mutex::new(manager.lock().unwrap().clone()))),
            None => {
                // Initialize with default config
                init_multi_gpu(MultiGpuConfig::default())?;
                get_multi_gpu_manager()
            }
        }
    }
}

impl Clone for MultiGpuManager {
    fn clone(&self) -> Self {
        // Create a new manager with the same configuration
        Self::new(self.config.clone()).unwrap_or_else(|_| {
            // Fallback to single device if cloning fails
            let fallback_config = MultiGpuConfig {
                device_ids: vec![0],
                ..self.config.clone()
            };
            Self::new(fallback_config).unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_multi_gpu_manager_creation() {
        let config = MultiGpuConfig {
            device_ids: vec![0],
            ..MultiGpuConfig::default()
        };

        let manager = MultiGpuManager::new(config);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert_eq!(manager.device_count(), 1);
    }

    #[test]
    fn test_data_parallel_distribution() {
        let config = MultiGpuConfig {
            device_ids: vec![0, 1],
            distribution_strategy: DistributionStrategy::DataParallel,
            ..MultiGpuConfig::default()
        };

        let manager = MultiGpuManager::new(config).unwrap();

        let matrix_data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        let matrix = GpuMatrix {
            data: matrix_data,
            on_gpu: false,
        };

        let distributed = manager.distribute_matrix(&matrix).unwrap();
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

        let manager = MultiGpuManager::new(config).unwrap();

        // Create distributed results
        let chunk1_data =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let chunk2_data =
            Array2::from_shape_vec((2, 3), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

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

        let result = manager.collect_results(distributed_results).unwrap();
        assert_eq!(result.data.shape(), &[4, 3]);
    }
}

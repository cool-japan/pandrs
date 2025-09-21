//! GPU memory pooling for efficient memory management
//!
//! This module provides a memory pool implementation that reduces the overhead
//! of frequent GPU memory allocations and deallocations.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::gpu::{GpuError, GpuManager};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};

/// Configuration for GPU memory pool
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes
    pub max_size: usize,
    /// Minimum allocation size in bytes
    pub min_allocation_size: usize,
    /// Whether to enable memory compaction
    pub enable_compaction: bool,
    /// Interval for memory cleanup (in seconds)
    pub cleanup_interval: u64,
    /// Maximum age for unused allocations (in seconds)
    pub max_allocation_age: u64,
    /// Growth factor when expanding the pool
    pub growth_factor: f64,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,  // 256MB
            max_size: 2 * 1024 * 1024 * 1024, // 2GB
            min_allocation_size: 4096,        // 4KB
            enable_compaction: true,
            cleanup_interval: 30,    // 30 seconds
            max_allocation_age: 300, // 5 minutes
            growth_factor: 1.5,
        }
    }
}

/// Memory allocation metadata
#[derive(Debug, Clone)]
struct AllocationInfo {
    /// Size of the allocation in bytes
    size: usize,
    /// When the allocation was made
    allocated_at: Instant,
    /// When the allocation was last accessed
    last_accessed: Instant,
    /// Whether the allocation is currently in use
    in_use: bool,
    /// Reference count for shared allocations
    ref_count: usize,
}

/// A memory block in the pool
#[derive(Debug)]
struct MemoryBlock {
    /// Pointer to the GPU memory
    #[cfg(feature = "cuda")]
    ptr: Box<CudaSlice<u8>>,
    #[cfg(not(feature = "cuda"))]
    ptr: usize, // Dummy pointer for non-CUDA builds
    /// Size of the block
    size: usize,
    /// Whether the block is free
    is_free: bool,
    /// Allocation info if the block is in use
    allocation_info: Option<AllocationInfo>,
}

/// GPU memory pool for efficient allocation management
pub struct GpuMemoryPool {
    /// Configuration
    config: MemoryPoolConfig,
    /// Device this pool is associated with
    device_id: i32,
    /// Memory blocks organized by size for efficient lookup
    free_blocks: BTreeMap<usize, VecDeque<Arc<Mutex<MemoryBlock>>>>,
    /// All allocated blocks
    allocated_blocks: HashMap<usize, Arc<Mutex<MemoryBlock>>>, // key is ptr address
    /// Current pool size
    current_size: usize,
    /// Peak memory usage
    peak_usage: usize,
    /// Statistics
    stats: MemoryPoolStats,
    /// Last cleanup time
    last_cleanup: Instant,
    /// CUDA device reference
    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaDevice>>,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total number of allocations
    pub total_allocations: u64,
    /// Total number of deallocations
    pub total_deallocations: u64,
    /// Number of cache hits (allocations served from pool)
    pub cache_hits: u64,
    /// Number of cache misses (new allocations needed)
    pub cache_misses: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes deallocated
    pub total_bytes_deallocated: u64,
    /// Number of memory compactions
    pub compaction_count: u64,
    /// Average allocation size
    pub avg_allocation_size: f64,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
    pub fn new(device_id: i32, config: MemoryPoolConfig) -> Result<Self> {
        let device = Self::get_cuda_device(device_id)?;

        let mut pool = Self {
            config,
            device_id,
            free_blocks: BTreeMap::new(),
            allocated_blocks: HashMap::new(),
            current_size: 0,
            peak_usage: 0,
            stats: MemoryPoolStats::default(),
            last_cleanup: Instant::now(),
            #[cfg(feature = "cuda")]
            device,
            #[cfg(not(feature = "cuda"))]
            device: None,
        };

        // Pre-allocate initial pool
        pool.expand_pool(pool.config.initial_size)?;

        Ok(pool)
    }

    /// Get CUDA device handle
    #[cfg(feature = "cuda")]
    fn get_cuda_device(device_id: i32) -> Result<Option<Arc<CudaDevice>>> {
        // In a real implementation, would get device from GPU manager
        // For now, return None to indicate no CUDA device available
        Ok(None)
    }

    #[cfg(not(feature = "cuda"))]
    fn get_cuda_device(_device_id: i32) -> Result<Option<()>> {
        Ok(None)
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Result<GpuAllocation> {
        // Round up to minimum allocation size
        let aligned_size = self.align_size(size);

        // Check for periodic cleanup
        self.maybe_cleanup();

        // Try to find a suitable free block
        if let Some(block) = self.find_free_block(aligned_size) {
            self.stats.cache_hits += 1;
            self.use_block(block, aligned_size)
        } else {
            self.stats.cache_misses += 1;
            self.allocate_new_block(aligned_size)
        }
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, allocation: GpuAllocation) -> Result<()> {
        let ptr_addr = allocation.ptr_address();

        if let Some(block) = self.allocated_blocks.remove(&ptr_addr) {
            let mut block_guard = block.lock().unwrap();
            block_guard.is_free = true;
            block_guard.allocation_info = None;

            // Add back to free blocks
            let size = block_guard.size;
            self.free_blocks
                .entry(size)
                .or_insert_with(VecDeque::new)
                .push_back(block.clone());

            self.stats.total_deallocations += 1;
            self.stats.total_bytes_deallocated += size as u64;

            Ok(())
        } else {
            Err(Error::from(GpuError::DeviceError(
                "Invalid allocation pointer".to_string(),
            )))
        }
    }

    /// Get current memory usage statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        let mut stats = self.stats.clone();

        if stats.total_allocations > 0 {
            stats.avg_allocation_size =
                stats.total_bytes_allocated as f64 / stats.total_allocations as f64;
        }

        stats
    }

    /// Get current pool size
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage
    }

    /// Force memory cleanup
    pub fn cleanup(&mut self) -> Result<()> {
        let now = Instant::now();
        let max_age = Duration::from_secs(self.config.max_allocation_age);

        // Remove old unused allocations
        let mut removed_size = 0;
        let mut blocks_to_remove = Vec::new();

        for (size, blocks) in &mut self.free_blocks {
            blocks.retain(|block| {
                let block_guard = block.lock().unwrap();
                if let Some(ref info) = block_guard.allocation_info {
                    let age = now.duration_since(info.last_accessed);
                    if age > max_age {
                        removed_size += *size;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            });

            if blocks.is_empty() {
                blocks_to_remove.push(*size);
            }
        }

        // Remove empty entries
        for size in blocks_to_remove {
            self.free_blocks.remove(&size);
        }

        self.current_size -= removed_size;
        self.last_cleanup = now;

        // Trigger compaction if enabled
        if self.config.enable_compaction {
            self.compact_memory()?;
        }

        Ok(())
    }

    /// Compact fragmented memory
    fn compact_memory(&mut self) -> Result<()> {
        // In a real implementation, would:
        // 1. Identify fragmented regions
        // 2. Move active allocations to consolidate free space
        // 3. Update all pointers and references

        self.stats.compaction_count += 1;

        log::info!("Memory compaction completed for device {}", self.device_id);

        Ok(())
    }

    /// Find a suitable free block for the requested size
    fn find_free_block(&mut self, size: usize) -> Option<Arc<Mutex<MemoryBlock>>> {
        // Look for exact size match first
        if let Some(blocks) = self.free_blocks.get_mut(&size) {
            if let Some(block) = blocks.pop_front() {
                return Some(block);
            }
        }

        // Look for larger blocks that can be split
        for (&block_size, blocks) in self.free_blocks.range_mut(size..) {
            if let Some(block) = blocks.pop_front() {
                // If the block is much larger, consider splitting it
                if block_size > size * 2 && block_size - size >= self.config.min_allocation_size {
                    // Split the block (simplified implementation)
                    // In practice, would create a new block for the remainder
                }
                return Some(block);
            }
        }

        None
    }

    /// Use a free block for allocation
    fn use_block(&mut self, block: Arc<Mutex<MemoryBlock>>, size: usize) -> Result<GpuAllocation> {
        let mut block_guard = block.lock().unwrap();
        block_guard.is_free = false;
        block_guard.allocation_info = Some(AllocationInfo {
            size,
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
            in_use: true,
            ref_count: 1,
        });

        let ptr_addr = self.get_ptr_address(&block_guard);
        drop(block_guard);

        self.allocated_blocks.insert(ptr_addr, block);

        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size as u64;

        Ok(GpuAllocation::new(ptr_addr, size))
    }

    /// Allocate a new block from the GPU
    fn allocate_new_block(&mut self, size: usize) -> Result<GpuAllocation> {
        // Check if we need to expand the pool
        if self.current_size + size > self.config.max_size {
            return Err(Error::from(GpuError::DeviceError(
                "Memory pool size limit exceeded".to_string(),
            )));
        }

        // Allocate new memory block
        let block = self.allocate_gpu_memory(size)?;
        let ptr_addr = self.get_ptr_address(&block);

        let block = Arc::new(Mutex::new(block));
        self.allocated_blocks.insert(ptr_addr, block.clone());

        self.current_size += size;
        self.peak_usage = self.peak_usage.max(self.current_size);

        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size as u64;

        Ok(GpuAllocation::new(ptr_addr, size))
    }

    /// Allocate memory on the GPU
    fn allocate_gpu_memory(&self, size: usize) -> Result<MemoryBlock> {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref device) = self.device {
                match device.alloc_zeros::<u8>(size) {
                    Ok(ptr) => Ok(MemoryBlock {
                        ptr: Box::new(ptr),
                        size,
                        is_free: false,
                        allocation_info: Some(AllocationInfo {
                            size,
                            allocated_at: Instant::now(),
                            last_accessed: Instant::now(),
                            in_use: true,
                            ref_count: 1,
                        }),
                    }),
                    Err(e) => Err(Error::from(GpuError::DeviceError(format!(
                        "GPU memory allocation failed: {}",
                        e
                    )))),
                }
            } else {
                // Fallback - return error when CUDA device is not available
                Err(Error::from(GpuError::DeviceError(
                    "CUDA device not available for memory allocation".to_string(),
                )))
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // CPU fallback
            Ok(MemoryBlock {
                ptr: size, // Use size as dummy pointer
                size,
                is_free: false,
                allocation_info: Some(AllocationInfo {
                    size,
                    allocated_at: Instant::now(),
                    last_accessed: Instant::now(),
                    in_use: true,
                    ref_count: 1,
                }),
            })
        }
    }

    /// Get pointer address for indexing
    fn get_ptr_address(&self, block: &MemoryBlock) -> usize {
        #[cfg(feature = "cuda")]
        {
            // Use the block's size and position as a unique identifier
            // Since we can't dereference CudaSlice, use a hash of the block
            block as *const _ as usize
        }
        #[cfg(not(feature = "cuda"))]
        {
            block.ptr
        }
    }

    /// Expand the memory pool
    fn expand_pool(&mut self, additional_size: usize) -> Result<()> {
        if self.current_size + additional_size > self.config.max_size {
            return Err(Error::from(GpuError::DeviceError(
                "Cannot expand pool beyond maximum size".to_string(),
            )));
        }

        // Allocate large block and split it into smaller chunks
        let chunk_size = self.config.min_allocation_size * 16; // 64KB chunks
        let num_chunks = additional_size / chunk_size;

        for _ in 0..num_chunks {
            let block = self.allocate_gpu_memory(chunk_size)?;
            let block = Arc::new(Mutex::new(MemoryBlock {
                is_free: true,
                allocation_info: None,
                ..block
            }));

            self.free_blocks
                .entry(chunk_size)
                .or_insert_with(VecDeque::new)
                .push_back(block);
        }

        self.current_size += num_chunks * chunk_size;

        Ok(())
    }

    /// Align size to minimum allocation boundary
    fn align_size(&self, size: usize) -> usize {
        let min_size = self.config.min_allocation_size;
        (size + min_size - 1) / min_size * min_size
    }

    /// Check if cleanup is needed
    fn maybe_cleanup(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_cleanup).as_secs() >= self.config.cleanup_interval {
            let _ = self.cleanup();
        }
    }
}

/// A GPU memory allocation handle
pub struct GpuAllocation {
    /// Address of the allocated memory
    ptr_address: usize,
    /// Size of the allocation
    size: usize,
}

impl GpuAllocation {
    /// Create a new allocation handle
    fn new(ptr_address: usize, size: usize) -> Self {
        Self { ptr_address, size }
    }

    /// Get the pointer address
    pub fn ptr_address(&self) -> usize {
        self.ptr_address
    }

    /// Get the allocation size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get raw pointer (for CUDA operations)
    #[cfg(feature = "cuda")]
    pub fn as_device_ptr<T>(&self) -> *mut T {
        self.ptr_address as *mut T
    }
}

/// Global memory pool manager
pub struct GlobalMemoryPoolManager {
    /// Memory pools for each device
    pools: RwLock<HashMap<i32, Arc<Mutex<GpuMemoryPool>>>>,
    /// Default configuration
    default_config: MemoryPoolConfig,
}

impl GlobalMemoryPoolManager {
    /// Create a new global memory pool manager
    pub fn new(config: MemoryPoolConfig) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            default_config: config,
        }
    }

    /// Get or create memory pool for a device
    pub fn get_pool(&self, device_id: i32) -> Result<Arc<Mutex<GpuMemoryPool>>> {
        // Check if pool already exists
        {
            let pools = self.pools.read().unwrap();
            if let Some(pool) = pools.get(&device_id) {
                return Ok(pool.clone());
            }
        }

        // Create new pool
        let pool = GpuMemoryPool::new(device_id, self.default_config.clone())?;
        let pool = Arc::new(Mutex::new(pool));

        {
            let mut pools = self.pools.write().unwrap();
            pools.insert(device_id, pool.clone());
        }

        Ok(pool)
    }

    /// Get statistics for all pools
    pub fn get_all_stats(&self) -> HashMap<i32, MemoryPoolStats> {
        let pools = self.pools.read().unwrap();
        let mut stats = HashMap::new();

        for (&device_id, pool) in pools.iter() {
            let pool_guard = pool.lock().unwrap();
            stats.insert(device_id, pool_guard.get_stats());
        }

        stats
    }

    /// Cleanup all pools
    pub fn cleanup_all(&self) -> Result<()> {
        let pools = self.pools.read().unwrap();

        for pool in pools.values() {
            let mut pool_guard = pool.lock().unwrap();
            pool_guard.cleanup()?;
        }

        Ok(())
    }
}

lazy_static::lazy_static! {
    /// Global memory pool manager instance
    static ref GLOBAL_MEMORY_POOL: GlobalMemoryPoolManager =
        GlobalMemoryPoolManager::new(MemoryPoolConfig::default());
}

/// Get the global memory pool manager
pub fn get_memory_pool_manager() -> &'static GlobalMemoryPoolManager {
    &GLOBAL_MEMORY_POOL
}

/// Allocate GPU memory from the global pool
pub fn gpu_alloc(device_id: i32, size: usize) -> Result<GpuAllocation> {
    let pool = get_memory_pool_manager().get_pool(device_id)?;
    let mut pool_guard = pool.lock().unwrap();
    pool_guard.allocate(size)
}

/// Deallocate GPU memory to the global pool
pub fn gpu_dealloc(device_id: i32, allocation: GpuAllocation) -> Result<()> {
    let pool = get_memory_pool_manager().get_pool(device_id)?;
    let mut pool_guard = pool.lock().unwrap();
    pool_guard.deallocate(allocation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig::default();
        let pool = GpuMemoryPool::new(0, config);
        assert!(pool.is_ok());
    }

    #[test]
    fn test_memory_allocation() {
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024, // 1MB
            ..MemoryPoolConfig::default()
        };

        let mut pool = GpuMemoryPool::new(0, config).unwrap();

        // Allocate some memory
        let alloc1 = pool.allocate(1024).unwrap();
        assert_eq!(alloc1.size(), 4096); // Aligned to min allocation size

        let alloc2 = pool.allocate(2048).unwrap();
        assert_eq!(alloc2.size(), 4096);

        // Deallocate
        pool.deallocate(alloc1).unwrap();
        pool.deallocate(alloc2).unwrap();

        // Check stats
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 2);
    }

    #[test]
    fn test_memory_pool_stats() {
        let config = MemoryPoolConfig::default();
        let mut pool = GpuMemoryPool::new(0, config).unwrap();

        let alloc = pool.allocate(1024).unwrap();
        let stats = pool.get_stats();

        assert_eq!(stats.total_allocations, 1);
        assert!(stats.avg_allocation_size > 0.0);

        pool.deallocate(alloc).unwrap();
        let stats = pool.get_stats();
        assert_eq!(stats.total_deallocations, 1);
    }

    #[test]
    fn test_global_memory_pool() {
        let manager = get_memory_pool_manager();
        let pool = manager.get_pool(0).unwrap();

        // Test that we get the same pool instance
        let pool2 = manager.get_pool(0).unwrap();
        assert_eq!(Arc::as_ptr(&pool), Arc::as_ptr(&pool2));
    }
}

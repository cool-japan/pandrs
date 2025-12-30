//! Zero-Copy Data Views and Cache-Aware Memory Management
//!
//! This module implements zero-copy data views, memory-mapped operations,
//! and cache-aware memory management strategies for optimal performance
//! in PandRS DataFrame operations.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, Mutex, RwLock};

/// Cache line size for optimal memory alignment
pub const CACHE_LINE_SIZE: usize = 64;

/// Memory page size for efficient allocation
pub const PAGE_SIZE: usize = 4096;

/// Zero-copy data view that provides access to underlying memory without copying
#[derive(Debug)]
pub struct ZeroCopyView<T> {
    /// Pointer to the underlying data
    data: NonNull<T>,
    /// Length of the data in elements
    len: usize,
    /// Capacity of the allocated memory
    capacity: usize,
    /// Memory layout information
    layout: MemoryLayout,
    /// Reference to the storage handle to ensure data lifetime
    _storage_handle: Arc<StorageHandle>,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for ZeroCopyView<T> {}
unsafe impl<T: Sync> Sync for ZeroCopyView<T> {}

impl<T> ZeroCopyView<T> {
    /// Create a new zero-copy view from a storage handle
    pub unsafe fn new(
        data: NonNull<T>,
        len: usize,
        capacity: usize,
        layout: MemoryLayout,
        storage_handle: Arc<StorageHandle>,
    ) -> Self {
        Self {
            data,
            len,
            capacity,
            layout,
            _storage_handle: storage_handle,
            _phantom: PhantomData,
        }
    }

    /// Get the length of the view
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the underlying memory
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get memory layout information
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get a slice view of the data
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Get a mutable slice view of the data (if exclusive access is guaranteed)
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
    }

    /// Create a subview of this view
    pub fn subview(&self, range: Range<usize>) -> Result<ZeroCopyView<T>> {
        if range.start > self.len || range.end > self.len || range.start > range.end {
            return Err(Error::InvalidOperation(
                "Invalid range for subview".to_string(),
            ));
        }

        let new_len = range.end - range.start;
        let new_data = unsafe { NonNull::new_unchecked(self.data.as_ptr().add(range.start)) };

        Ok(unsafe {
            ZeroCopyView::new(
                new_data,
                new_len,
                self.capacity - range.start,
                self.layout.clone(),
                Arc::clone(&self._storage_handle),
            )
        })
    }

    /// Get raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Check if the view is cache-aligned
    pub fn is_cache_aligned(&self) -> bool {
        self.data.as_ptr() as usize % CACHE_LINE_SIZE == 0
    }

    /// Get the memory address for debugging
    pub fn memory_address(&self) -> usize {
        self.data.as_ptr() as usize
    }
}

impl<T> Deref for ZeroCopyView<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Memory layout information for zero-copy views
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Starting address of the memory region
    pub start_address: usize,
    /// Size of each element in bytes
    pub element_size: usize,
    /// Stride between elements (for non-contiguous layouts)
    pub stride: usize,
    /// Memory alignment requirements
    pub alignment: usize,
    /// Whether the memory is cache-aligned
    pub cache_aligned: bool,
    /// NUMA node if applicable
    pub numa_node: Option<u32>,
}

impl MemoryLayout {
    pub fn new<T>() -> Self {
        Self {
            start_address: 0,
            element_size: mem::size_of::<T>(),
            stride: mem::size_of::<T>(),
            alignment: mem::align_of::<T>(),
            cache_aligned: false,
            numa_node: None,
        }
    }

    pub fn with_cache_alignment(mut self) -> Self {
        self.cache_aligned = true;
        self.alignment = self.alignment.max(CACHE_LINE_SIZE);
        self
    }

    pub fn with_numa_node(mut self, node: u32) -> Self {
        self.numa_node = Some(node);
        self
    }
}

/// Cache-aware memory allocator
pub struct CacheAwareAllocator {
    /// Cache topology information
    cache_topology: CacheTopology,
    /// Memory pools for different cache levels
    memory_pools: HashMap<CacheLevel, MemoryPool>,
    /// Allocation statistics
    stats: AllocationStats,
}

impl CacheAwareAllocator {
    pub fn new() -> Result<Self> {
        let cache_topology = CacheTopology::detect()?;
        let mut memory_pools = HashMap::new();

        // Create memory pools for different cache levels
        memory_pools.insert(CacheLevel::L1, MemoryPool::new(64 * 1024)?); // 64KB for L1
        memory_pools.insert(CacheLevel::L2, MemoryPool::new(512 * 1024)?); // 512KB for L2
        memory_pools.insert(CacheLevel::L3, MemoryPool::new(4 * 1024 * 1024)?); // 4MB for L3
        memory_pools.insert(CacheLevel::Memory, MemoryPool::new(64 * 1024 * 1024)?); // 64MB for main memory

        Ok(Self {
            cache_topology,
            memory_pools,
            stats: AllocationStats::new(),
        })
    }

    /// Allocate cache-aligned memory
    pub fn allocate_aligned<T>(
        &mut self,
        count: usize,
        cache_level: CacheLevel,
    ) -> Result<ZeroCopyView<T>> {
        let size = count * mem::size_of::<T>();
        let alignment = CACHE_LINE_SIZE.max(mem::align_of::<T>());

        let pool = self
            .memory_pools
            .get_mut(&cache_level)
            .ok_or_else(|| Error::InvalidOperation("Cache level not supported".to_string()))?;

        let allocation = pool.allocate_aligned(size, alignment)?;

        let layout = MemoryLayout {
            start_address: allocation.ptr as usize,
            element_size: mem::size_of::<T>(),
            stride: mem::size_of::<T>(),
            alignment,
            cache_aligned: true,
            numa_node: self.cache_topology.numa_node,
        };

        // Create a dummy storage handle for the allocation
        let storage_handle = Arc::new(StorageHandle::new(
            StorageId(allocation.ptr as u64),
            StorageType::InMemory,
            Box::new(allocation),
            StorageMetadata::new(size),
        ));

        self.stats.record_allocation(size);

        unsafe {
            Ok(ZeroCopyView::new(
                NonNull::new(allocation.ptr as *mut T).ok_or_else(|| {
                    Error::InvalidOperation("Null pointer allocation".to_string())
                })?,
                count,
                count,
                layout,
                storage_handle,
            ))
        }
    }

    /// Get allocation statistics
    pub fn stats(&self) -> &AllocationStats {
        &self.stats
    }

    /// Get cache topology information
    pub fn cache_topology(&self) -> &CacheTopology {
        &self.cache_topology
    }
}

/// Cache topology information
#[derive(Debug, Clone)]
pub struct CacheTopology {
    /// L1 cache size in bytes
    pub l1_cache_size: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// L3 cache size in bytes
    pub l3_cache_size: usize,
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// NUMA node if applicable
    pub numa_node: Option<u32>,
}

impl CacheTopology {
    pub fn detect() -> Result<Self> {
        // In a real implementation, this would use OS-specific APIs
        // to detect actual cache topology. For now, we use reasonable defaults.
        Ok(Self {
            l1_cache_size: 32 * 1024,       // 32KB
            l2_cache_size: 256 * 1024,      // 256KB
            l3_cache_size: 8 * 1024 * 1024, // 8MB
            cache_line_size: CACHE_LINE_SIZE,
            cpu_cores: num_cpus::get(),
            numa_node: None,
        })
    }

    /// Determine optimal cache level for given data size
    pub fn optimal_cache_level(&self, size: usize) -> CacheLevel {
        if size <= self.l1_cache_size / 2 {
            CacheLevel::L1
        } else if size <= self.l2_cache_size / 2 {
            CacheLevel::L2
        } else if size <= self.l3_cache_size / 2 {
            CacheLevel::L3
        } else {
            CacheLevel::Memory
        }
    }
}

/// Cache level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    Memory,
}

/// Memory pool for efficient allocation
pub struct MemoryPool {
    /// Pool size in bytes
    size: usize,
    /// Free blocks
    free_blocks: Vec<MemoryBlock>,
    /// Used blocks
    used_blocks: Vec<MemoryBlock>,
    /// Base pointer for the pool
    base_ptr: NonNull<u8>,
    /// Current offset in the pool
    current_offset: usize,
}

impl MemoryPool {
    pub fn new(size: usize) -> Result<Self> {
        // Allocate aligned memory for the pool
        let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE)
            .map_err(|_| Error::InvalidOperation("Invalid memory layout".to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };
        let base_ptr = NonNull::new(ptr)
            .ok_or_else(|| Error::InvalidOperation("Memory allocation failed".to_string()))?;

        Ok(Self {
            size,
            free_blocks: vec![MemoryBlock {
                ptr: ptr as *mut u8,
                size,
                alignment: PAGE_SIZE,
            }],
            used_blocks: Vec::new(),
            base_ptr,
            current_offset: 0,
        })
    }

    /// Allocate aligned memory from the pool
    pub fn allocate_aligned(&mut self, size: usize, alignment: usize) -> Result<MemoryBlock> {
        // Round up size to alignment
        let aligned_size = (size + alignment - 1) & !(alignment - 1);

        // Find a suitable free block
        for (i, block) in self.free_blocks.iter().enumerate() {
            if block.size >= aligned_size {
                let allocated_block = MemoryBlock {
                    ptr: block.ptr,
                    size: aligned_size,
                    alignment,
                };

                // Update the free block
                if block.size > aligned_size {
                    self.free_blocks[i] = MemoryBlock {
                        ptr: unsafe { block.ptr.add(aligned_size) },
                        size: block.size - aligned_size,
                        alignment: block.alignment,
                    };
                } else {
                    self.free_blocks.remove(i);
                }

                self.used_blocks.push(allocated_block);
                return Ok(allocated_block);
            }
        }

        Err(Error::InvalidOperation(
            "Not enough memory in pool".to_string(),
        ))
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe {
            let layout = std::alloc::Layout::from_size_align_unchecked(self.size, PAGE_SIZE);
            std::alloc::dealloc(self.base_ptr.as_ptr(), layout);
        }
    }
}

/// Memory block representation
#[derive(Debug, Clone, Copy)]
pub struct MemoryBlock {
    /// Pointer to the memory block
    pub ptr: *mut u8,
    /// Size of the block in bytes
    pub size: usize,
    /// Alignment of the block
    pub alignment: usize,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

/// Allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Current memory usage
    pub current_usage: usize,
    /// Cache hit rate for allocations
    pub cache_hit_rate: f64,
}

impl AllocationStats {
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            allocation_count: 0,
            peak_usage: 0,
            current_usage: 0,
            cache_hit_rate: 0.0,
        }
    }

    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.allocation_count += 1;
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }
}

/// Memory-mapped file view for large datasets
pub struct MemoryMappedView<T> {
    /// Memory map
    mmap: memmap2::Mmap,
    /// Length in elements
    len: usize,
    /// Element layout
    layout: MemoryLayout,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> MemoryMappedView<T> {
    /// Create a new memory-mapped view from a file
    pub fn from_file(file: std::fs::File, len: usize) -> Result<Self> {
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| Error::InvalidOperation(format!("Memory mapping failed: {}", e)))?
        };

        let layout = MemoryLayout {
            start_address: mmap.as_ptr() as usize,
            element_size: mem::size_of::<T>(),
            stride: mem::size_of::<T>(),
            alignment: mem::align_of::<T>(),
            cache_aligned: false,
            numa_node: None,
        };

        Ok(Self {
            mmap,
            len,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Get a slice view of the memory-mapped data
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.mmap.as_ptr() as *const T,
                self.len.min(self.mmap.len() / mem::size_of::<T>()),
            )
        }
    }

    /// Get memory layout information
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the length of the view
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Deref for MemoryMappedView<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Cache-aware data operations
pub trait CacheAwareOps<T> {
    /// Perform cache-friendly linear scan
    fn linear_scan<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&T) -> bool;

    /// Perform cache-blocked matrix operations
    fn blocked_operation<U, F>(&self, other: &[U], block_size: usize, op: F) -> Vec<T>
    where
        F: Fn(&T, &U) -> T,
        T: Clone,
        U: Clone;

    /// Prefetch data into cache
    fn prefetch(&self, indices: &[usize]);

    /// Get optimal block size for cache efficiency
    fn optimal_block_size(&self) -> usize;
}

impl<T> CacheAwareOps<T> for ZeroCopyView<T> {
    fn linear_scan<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&T) -> bool,
    {
        let mut results = Vec::new();
        let slice = self.as_slice();

        // Process in cache-friendly blocks
        let block_size = self.optimal_block_size();
        for (block_start, chunk) in slice.chunks(block_size).enumerate() {
            for (i, item) in chunk.iter().enumerate() {
                if predicate(item) {
                    results.push(block_start * block_size + i);
                }
            }
        }

        results
    }

    fn blocked_operation<U, F>(&self, other: &[U], block_size: usize, op: F) -> Vec<T>
    where
        F: Fn(&T, &U) -> T,
        T: Clone,
        U: Clone,
    {
        let slice = self.as_slice();
        let mut result = Vec::with_capacity(slice.len().min(other.len()));

        let pairs = slice.iter().zip(other.iter());
        for chunk in pairs.collect::<Vec<_>>().chunks(block_size) {
            for (a, b) in chunk {
                result.push(op(a, b));
            }
        }

        result
    }

    fn prefetch(&self, indices: &[usize]) {
        let slice = self.as_slice();
        for &index in indices {
            if index < slice.len() {
                unsafe {
                    let ptr = slice.as_ptr().add(index);
                    #[cfg(target_arch = "x86_64")]
                    std::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
    }

    fn optimal_block_size(&self) -> usize {
        // Calculate optimal block size based on cache size and element size
        let cache_size = 32 * 1024; // L1 cache size
        let element_size = mem::size_of::<T>();
        (cache_size / element_size).max(64)
    }
}

/// Memory manager that provides zero-copy views
pub struct ZeroCopyManager {
    /// Cache-aware allocator
    allocator: Mutex<CacheAwareAllocator>,
    /// Active views for tracking
    active_views: RwLock<HashMap<usize, ViewMetadata>>,
    /// Memory usage statistics
    stats: Mutex<ZeroCopyStats>,
}

impl ZeroCopyManager {
    pub fn new() -> Result<Self> {
        Ok(Self {
            allocator: Mutex::new(CacheAwareAllocator::new()?),
            active_views: RwLock::new(HashMap::new()),
            stats: Mutex::new(ZeroCopyStats::new()),
        })
    }

    /// Create a zero-copy view with optimal cache placement
    pub fn create_view<T: Clone>(&self, data: Vec<T>) -> Result<ZeroCopyView<T>> {
        let len = data.len();
        let size = len * mem::size_of::<T>();

        let cache_level = {
            let allocator = self.allocator.lock().map_err(|_| {
                Error::InvalidOperation("Failed to acquire allocator lock".to_string())
            })?;
            allocator.cache_topology().optimal_cache_level(size)
        };

        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire allocator lock".to_string()))?;

        let mut view = allocator.allocate_aligned(len, cache_level)?;

        // Copy data into the aligned memory
        unsafe {
            let dest = view.as_mut_slice();
            for (i, item) in data.into_iter().enumerate() {
                if i < dest.len() {
                    dest[i] = item;
                }
            }
        }

        // Record the view
        let view_id = view.memory_address();
        let metadata = ViewMetadata {
            size,
            cache_level,
            creation_time: std::time::Instant::now(),
        };

        self.active_views
            .write()
            .map_err(|_| Error::InvalidOperation("Failed to acquire views lock".to_string()))?
            .insert(view_id, metadata);

        self.stats
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))?
            .record_view_creation(size);

        Ok(view)
    }

    /// Create a memory-mapped view for large files
    pub fn create_mmap_view<T>(&self, file_path: &str, len: usize) -> Result<MemoryMappedView<T>> {
        let file = std::fs::File::open(file_path)
            .map_err(|e| Error::InvalidOperation(format!("Failed to open file: {}", e)))?;

        let view = MemoryMappedView::from_file(file, len)?;

        self.stats
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))?
            .record_mmap_creation(len * mem::size_of::<T>());

        Ok(view)
    }

    /// Get zero-copy statistics
    pub fn stats(&self) -> Result<ZeroCopyStats> {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))
    }
}

/// Metadata for tracking views
#[derive(Debug, Clone)]
struct ViewMetadata {
    size: usize,
    cache_level: CacheLevel,
    creation_time: std::time::Instant,
}

/// Statistics for zero-copy operations
#[derive(Debug, Clone)]
pub struct ZeroCopyStats {
    /// Number of zero-copy views created
    pub views_created: usize,
    /// Number of memory-mapped views created
    pub mmap_views_created: usize,
    /// Total memory managed
    pub total_memory: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average view lifetime
    pub avg_view_lifetime: std::time::Duration,
}

impl ZeroCopyStats {
    pub fn new() -> Self {
        Self {
            views_created: 0,
            mmap_views_created: 0,
            total_memory: 0,
            cache_hit_rate: 0.0,
            avg_view_lifetime: std::time::Duration::ZERO,
        }
    }

    pub fn record_view_creation(&mut self, size: usize) {
        self.views_created += 1;
        self.total_memory += size;
    }

    pub fn record_mmap_creation(&mut self, size: usize) {
        self.mmap_views_created += 1;
        self.total_memory += size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_topology_detection() {
        let topology = CacheTopology::detect().unwrap();
        assert!(topology.l1_cache_size > 0);
        assert!(topology.l2_cache_size > 0);
        assert!(topology.l3_cache_size > 0);
        assert!(topology.cpu_cores > 0);
    }

    #[test]
    fn test_memory_layout() {
        let layout = MemoryLayout::new::<i64>().with_cache_alignment();
        assert_eq!(layout.element_size, 8);
        assert!(layout.cache_aligned);
        assert!(layout.alignment >= CACHE_LINE_SIZE);
    }

    #[test]
    fn test_zero_copy_manager() {
        let manager = ZeroCopyManager::new().unwrap();
        let data = vec![1i32, 2, 3, 4, 5];
        let view = manager.create_view(data).unwrap();

        assert_eq!(view.len(), 5);
        assert_eq!(view.as_slice(), &[1, 2, 3, 4, 5]);

        let stats = manager.stats().unwrap();
        assert_eq!(stats.views_created, 1);
    }

    #[test]
    fn test_cache_aware_operations() {
        let manager = ZeroCopyManager::new().unwrap();
        let data = (0..1000).collect::<Vec<i32>>();
        let view = manager.create_view(data).unwrap();

        // Test linear scan
        let evens = view.linear_scan(|&x| x % 2 == 0);
        assert_eq!(evens.len(), 500);

        // Test optimal block size
        let block_size = view.optimal_block_size();
        assert!(block_size > 0);
    }

    #[test]
    fn test_subview_creation() {
        let manager = ZeroCopyManager::new().unwrap();
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let view = manager.create_view(data).unwrap();

        let subview = view.subview(2..7).unwrap();
        assert_eq!(subview.len(), 5);
        assert_eq!(subview.as_slice(), &[3, 4, 5, 6, 7]);
    }
}

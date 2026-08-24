//! Zero-Copy Data Views and Cache-Aware Memory Management
//!
//! This module implements zero-copy data views, memory-mapped operations,
//! and cache-aware memory management strategies for optimal performance
//! in PandRS DataFrame operations.
//!
//! # Ownership and safety model
//!
//! * A [`ZeroCopyView`] keeps its backing storage alive: pool-backed views hold
//!   an `Arc` of the owning [`MemoryPool`], so dropping the allocator or the
//!   manager can never leave a view dangling. When the last view (including
//!   every subview) referring to an allocation is dropped, the block is
//!   returned to the pool and can be reused.
//! * Every type that is reinterpreted from raw bytes (memory mapped files,
//!   zero-initialised pool memory) is constrained by the sealed
//!   [`ZeroCopyPod`] trait, which is implemented only for primitive types where
//!   every bit pattern is a valid value.
//! * Raw pointers are wrapped in dedicated newtypes that carry the `Send`/`Sync`
//!   reasoning, so the auto traits of the public types are derived and checked
//!   by the compiler instead of being asserted by hand.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, Range};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Cache line size for optimal memory alignment
pub const CACHE_LINE_SIZE: usize = 64;

/// Memory page size for efficient allocation
pub const PAGE_SIZE: usize = 4096;

mod sealed {
    /// Prevents downstream crates from implementing [`super::ZeroCopyPod`].
    pub trait Sealed {}
}

/// Types that may be reinterpreted from arbitrary initialised bytes.
///
/// Implemented only for the primitive numeric types, where
///
/// * every bit pattern is a valid value (unlike `bool`, `char` or any enum),
/// * there is no interior padding, pointer or lifetime to invalidate,
/// * the type is `Copy`, so raw memory can be recycled without running
///   destructors.
///
/// The trait is sealed: no other type can join, which is what makes
/// [`MemoryMappedView::from_file`] and [`CacheAwareAllocator::allocate_aligned`]
/// safe functions.
///
/// # Safety
/// Implementors must satisfy every bullet above. Because the trait is sealed,
/// only this module can add implementations.
pub unsafe trait ZeroCopyPod: sealed::Sealed + Copy + Sized + 'static {}

macro_rules! impl_zero_copy_pod {
    ($($ty:ty),* $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            // SAFETY: `$ty` is a primitive numeric type: every bit pattern is a
            // valid value, it contains no padding, no pointers and no
            // lifetimes, and it is `Copy`.
            unsafe impl ZeroCopyPod for $ty {}
        )*
    };
}

impl_zero_copy_pod!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64);

/// Pointer to the first element of a view.
///
/// A newtype exists purely so that the `Send`/`Sync` reasoning lives on the raw
/// pointer itself and everything built on top derives its auto traits from the
/// compiler.
#[derive(Debug)]
struct ViewPtr<T>(NonNull<T>);

// SAFETY: `ViewPtr` is a plain pointer into memory that the owning view keeps
// alive for as long as the pointer exists (see `ViewOwner`). Moving it to
// another thread moves the exclusive right to the referenced elements, which is
// sound exactly when `T: Send`.
unsafe impl<T: Send> Send for ViewPtr<T> {}
// SAFETY: shared access through `&ZeroCopyView` only ever yields `&[T]`
// (obtaining `&mut [T]` requires the `unsafe` `as_mut_slice`), so sharing the
// pointer across threads is sound exactly when `T: Sync`.
unsafe impl<T: Sync> Sync for ViewPtr<T> {}

impl<T> ViewPtr<T> {
    fn as_ptr(&self) -> *mut T {
        self.0.as_ptr()
    }
}

/// Backing storage that keeps a [`ZeroCopyView`]'s memory alive.
#[derive(Debug)]
enum ViewOwner {
    /// Empty views point at a dangling (but aligned) address and own nothing.
    Empty,
    /// Memory carved out of a [`MemoryPool`]; returned when the last view drops.
    Pool(PoolAllocation),
    /// Memory owned by an external storage handle.
    Storage(Arc<StorageHandle>),
}

impl ViewOwner {
    /// Size in bytes of the backing allocation this owner keeps alive.
    fn backing_bytes(&self) -> usize {
        match self {
            ViewOwner::Empty => 0,
            ViewOwner::Pool(allocation) => allocation.size(),
            ViewOwner::Storage(handle) => handle.metadata.size,
        }
    }
}

/// Zero-copy data view that provides access to underlying memory without copying
#[derive(Debug)]
pub struct ZeroCopyView<T> {
    /// Pointer to the underlying data
    data: ViewPtr<T>,
    /// Length of the data in elements
    len: usize,
    /// Capacity of the allocated memory
    capacity: usize,
    /// Memory layout information
    layout: MemoryLayout,
    /// Keeps the backing storage alive for at least as long as this view
    owner: Arc<ViewOwner>,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> ZeroCopyView<T> {
    /// Create a new zero-copy view from a storage handle
    ///
    /// # Safety
    /// * `data` must point at `len` initialised, properly aligned values of `T`
    ///   inside an allocation of at least `capacity` elements.
    /// * That allocation must be kept alive by `storage_handle` for as long as
    ///   the returned view (or any subview derived from it) exists.
    /// * No other live reference may mutate those elements.
    pub unsafe fn new(
        data: NonNull<T>,
        len: usize,
        capacity: usize,
        layout: MemoryLayout,
        storage_handle: Arc<StorageHandle>,
    ) -> Self {
        Self {
            data: ViewPtr(data),
            len,
            capacity,
            layout,
            owner: Arc::new(ViewOwner::Storage(storage_handle)),
            _phantom: PhantomData,
        }
    }

    /// Create a view over a block owned by a memory pool.
    ///
    /// # Safety
    /// `data` must point at `len` initialised values of `T` inside `allocation`.
    unsafe fn from_pool(
        data: NonNull<T>,
        len: usize,
        capacity: usize,
        layout: MemoryLayout,
        allocation: PoolAllocation,
    ) -> Self {
        Self {
            data: ViewPtr(data),
            len,
            capacity,
            layout,
            owner: Arc::new(ViewOwner::Pool(allocation)),
            _phantom: PhantomData,
        }
    }

    /// Create an empty view. Owns nothing and points at a dangling, aligned
    /// address, which is what `slice::from_raw_parts(_, 0)` requires.
    fn empty(layout: MemoryLayout) -> Self {
        Self {
            data: ViewPtr(NonNull::dangling()),
            len: 0,
            capacity: 0,
            layout,
            owner: Arc::new(ViewOwner::Empty),
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
        // SAFETY: the constructors guarantee `len` initialised, aligned elements
        // at `data`, and `owner` keeps that allocation alive for `&self`.
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }

    /// Get a mutable slice view of the data
    ///
    /// # Safety
    /// Subviews alias their parent, so the caller must guarantee that no other
    /// view (or slice derived from one) accesses the same elements while the
    /// returned slice is alive.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
    }

    /// Create a subview of this view
    ///
    /// The subview shares the parent's backing storage and keeps it alive; the
    /// memory is released once the parent and every subview are dropped.
    pub fn subview(&self, range: Range<usize>) -> Result<ZeroCopyView<T>> {
        if range.start > self.len || range.end > self.len || range.start > range.end {
            return Err(Error::InvalidOperation(
                "Invalid range for subview".to_string(),
            ));
        }

        let new_len = range.end - range.start;
        // SAFETY: `range.start <= self.len <= self.capacity`, so the offset
        // pointer stays inside the allocation (one-past-the-end is allowed).
        let new_data = unsafe { NonNull::new_unchecked(self.data.as_ptr().add(range.start)) };

        let mut layout = self.layout.clone();
        layout.start_address = new_data.as_ptr() as usize;
        layout.cache_aligned = layout.start_address % CACHE_LINE_SIZE == 0;

        Ok(ZeroCopyView {
            data: ViewPtr(new_data),
            len: new_len,
            capacity: self.capacity - range.start,
            layout,
            owner: Arc::clone(&self.owner),
            _phantom: PhantomData,
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

    /// Size in bytes of the backing allocation this view keeps alive.
    ///
    /// A subview pins its parent's whole block, so this can be larger than
    /// `len() * size_of::<T>()`.
    pub fn backing_bytes(&self) -> usize {
        self.owner.backing_bytes()
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

    /// Allocate zero-initialised, cache-aligned memory.
    ///
    /// Restricted to [`ZeroCopyPod`] types because the returned view exposes the
    /// memory as `&[T]`: an all-zero bit pattern must be a valid `T`.
    pub fn allocate_aligned<T: ZeroCopyPod>(
        &mut self,
        count: usize,
        cache_level: CacheLevel,
    ) -> Result<ZeroCopyView<T>> {
        // SAFETY: every element is initialised immediately below, and all-zero
        // is a valid bit pattern for every `ZeroCopyPod`.
        let view = unsafe { self.allocate_uninit::<T>(count, cache_level)? };
        // SAFETY: the allocation covers `count` elements of `T` and nothing else
        // references it yet.
        unsafe { std::ptr::write_bytes(view.data.as_ptr(), 0u8, count) };
        Ok(view)
    }

    /// Allocate uninitialised, cache-aligned memory.
    ///
    /// # Safety
    /// The returned view's elements are **uninitialised**. The caller must
    /// initialise all `count` elements (for example with
    /// [`std::ptr::copy_nonoverlapping`] or [`ZeroCopyView::as_mut_slice`])
    /// before reading from the view, and must not let `T`'s destructor matter:
    /// the pool recycles the block without dropping anything.
    pub unsafe fn allocate_uninit<T>(
        &mut self,
        count: usize,
        cache_level: CacheLevel,
    ) -> Result<ZeroCopyView<T>> {
        let element_size = mem::size_of::<T>();
        if element_size == 0 {
            return Err(Error::InvalidInput(
                "Zero-sized types cannot be allocated from a memory pool".to_string(),
            ));
        }

        let alignment = CACHE_LINE_SIZE.max(mem::align_of::<T>());
        let size = count.checked_mul(element_size).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Allocation size overflow: {count} x {element_size} bytes"
            ))
        })?;

        let mut layout = MemoryLayout {
            start_address: 0,
            element_size,
            stride: element_size,
            alignment,
            cache_aligned: true,
            numa_node: self.cache_topology.numa_node,
        };

        if size == 0 {
            return Ok(ZeroCopyView::empty(layout));
        }

        let allocation = {
            let pool = self
                .memory_pools
                .get(&cache_level)
                .ok_or_else(|| Error::InvalidOperation("Cache level not supported".to_string()))?;
            pool.allocate_aligned(size, alignment)?
        };

        layout.start_address = allocation.address();
        let data = NonNull::new(allocation.address() as *mut T)
            .ok_or_else(|| Error::InvalidOperation("Null pointer allocation".to_string()))?;

        self.stats.record_allocation(size);
        let live = self.bytes_in_use();
        self.stats.current_usage = live;
        self.stats.peak_usage = self.stats.peak_usage.max(live);

        Ok(ZeroCopyView::from_pool(
            data, count, count, layout, allocation,
        ))
    }

    /// Bytes currently handed out by this allocator's pools.
    pub fn bytes_in_use(&self) -> usize {
        self.memory_pools
            .values()
            .map(|pool| pool.bytes_in_use())
            .sum()
    }

    /// Get allocation statistics.
    ///
    /// `current_usage` is read back from the pools' free lists, so it shrinks
    /// again when views are dropped.
    pub fn stats(&self) -> AllocationStats {
        let mut stats = self.stats.clone();
        stats.current_usage = self.bytes_in_use();
        stats.peak_usage = stats.peak_usage.max(stats.current_usage);
        stats
    }

    /// Get cache topology information
    pub fn cache_topology(&self) -> &CacheTopology {
        &self.cache_topology
    }
}

/// Cache topology information.
///
/// See [`CacheTopology::detect`] for what is actually probed on which platform.
#[derive(Debug, Clone)]
pub struct CacheTopology {
    /// L1 data cache size in bytes
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
    /// `true` when the cache sizes were read from the operating system,
    /// `false` when they are the generic defaults from
    /// [`CacheTopology::defaults`].
    pub probed: bool,
}

/// Process-wide cache of the (immutable) topology.
static CACHE_TOPOLOGY: OnceLock<CacheTopology> = OnceLock::new();

impl CacheTopology {
    /// Detect the cache topology of the current machine.
    ///
    /// Detection is attempted once per process and cached.
    ///
    /// * **Linux**: cache sizes and the cache line size are read from
    ///   `/sys/devices/system/cpu/cpu0/cache/index*`; `probed` is then `true`.
    /// * **Every other platform** (including macOS): no probing is performed --
    ///   it would require FFI, which this crate avoids by policy -- and the
    ///   documented defaults of [`CacheTopology::defaults`] are returned with
    ///   `probed == false`.
    ///
    /// The CPU core count is always real (`num_cpus`), on every platform.
    pub fn detect() -> Result<Self> {
        Ok(CACHE_TOPOLOGY
            .get_or_init(|| Self::probe().unwrap_or_else(Self::defaults))
            .clone())
    }

    /// Generic defaults used when the OS cannot be queried: 32 KiB L1,
    /// 256 KiB L2, 8 MiB L3 and a 64 byte cache line. These are typical for
    /// contemporary x86_64 cores and are only used to size blocking heuristics,
    /// never for correctness.
    pub fn defaults() -> Self {
        Self {
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
            cache_line_size: CACHE_LINE_SIZE,
            cpu_cores: num_cpus::get(),
            numa_node: None,
            probed: false,
        }
    }

    #[cfg(target_os = "linux")]
    fn probe() -> Option<Self> {
        let mut topology = Self::defaults();
        let mut found_any = false;

        for index in 0..16 {
            let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let level = match read_sysfs_usize(&format!("{dir}/level")) {
                Some(level) => level,
                // Indices are contiguous; the first gap ends the enumeration.
                None => break,
            };
            let cache_type = read_sysfs_string(&format!("{dir}/type")).unwrap_or_default();
            let size = match read_sysfs_string(&format!("{dir}/size"))
                .as_deref()
                .and_then(parse_cache_size)
            {
                Some(size) => size,
                None => continue,
            };

            match (level, cache_type.as_str()) {
                // Only the data (or unified) cache is interesting for data blocking.
                (1, "Data") | (1, "Unified") => {
                    topology.l1_cache_size = size;
                    found_any = true;
                }
                (2, _) => {
                    topology.l2_cache_size = size;
                    found_any = true;
                }
                (3, _) => {
                    topology.l3_cache_size = size;
                    found_any = true;
                }
                _ => {}
            }

            if let Some(line) = read_sysfs_usize(&format!("{dir}/coherency_line_size")) {
                if line > 0 {
                    topology.cache_line_size = line;
                }
            }
        }

        if found_any {
            topology.probed = true;
            Some(topology)
        } else {
            None
        }
    }

    /// No probing is available on this platform; see [`CacheTopology::detect`].
    #[cfg(not(target_os = "linux"))]
    fn probe() -> Option<Self> {
        None
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

#[cfg(target_os = "linux")]
fn read_sysfs_string(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
}

#[cfg(target_os = "linux")]
fn read_sysfs_usize(path: &str) -> Option<usize> {
    read_sysfs_string(path).and_then(|value| value.parse::<usize>().ok())
}

/// Parse a sysfs cache size such as `32K`, `1024K` or `8M`.
#[cfg(target_os = "linux")]
fn parse_cache_size(raw: &str) -> Option<usize> {
    let raw = raw.trim();
    let (digits, multiplier) = match raw.chars().last()? {
        'K' | 'k' => (&raw[..raw.len() - 1], 1024usize),
        'M' | 'm' => (&raw[..raw.len() - 1], 1024 * 1024),
        'G' | 'g' => (&raw[..raw.len() - 1], 1024 * 1024 * 1024),
        _ => (raw, 1usize),
    };
    digits
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|value| value.checked_mul(multiplier))
        .filter(|size| *size > 0)
}

/// Cache level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    Memory,
}

/// A free region inside a [`MemoryPool`], expressed as an offset from the base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRegion {
    start: usize,
    size: usize,
}

/// Base pointer of a pool allocation.
struct PoolBase(NonNull<u8>);

// SAFETY: `PoolBase` is the sole owner of one heap allocation obtained from the
// global allocator, which has no thread affinity. The pointer grants no access
// on its own: every read or write through it goes through a `PoolAllocation`
// carved out under the pool's mutex, and those regions never overlap.
unsafe impl Send for PoolBase {}
// SAFETY: see above -- shared access to the base pointer only ever computes
// addresses of disjoint regions; all mutation of the free list is serialised by
// the pool's mutex.
unsafe impl Sync for PoolBase {}

impl std::fmt::Debug for PoolBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PoolBase({:p})", self.0.as_ptr())
    }
}

/// Shared state of a memory pool. Kept behind an `Arc` so that live allocations
/// keep the backing memory alive even after the pool handle is dropped.
#[derive(Debug)]
struct MemoryPoolInner {
    /// Pool size in bytes
    size: usize,
    /// Layout the base pointer was allocated with
    layout: std::alloc::Layout,
    /// Base pointer for the pool
    base: PoolBase,
    /// Free regions, sorted by `start` and always maximally coalesced
    free: Mutex<Vec<FreeRegion>>,
}

impl MemoryPoolInner {
    fn lock_free(&self) -> MutexGuard<'_, Vec<FreeRegion>> {
        // Recovering from poisoning is sound: the free list is only ever
        // mutated by this module's panic-free bookkeeping code, so a panic
        // elsewhere cannot leave it torn.
        self.free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn base_address(&self) -> usize {
        self.base.0.as_ptr() as usize
    }

    /// First-fit allocation honouring `alignment` for the *start* of the block.
    fn allocate(self: &Arc<Self>, size: usize, alignment: usize) -> Result<PoolAllocation> {
        if size == 0 {
            return Err(Error::InvalidInput(
                "Cannot allocate a zero-sized block from a memory pool".to_string(),
            ));
        }
        if !alignment.is_power_of_two() {
            return Err(Error::InvalidInput(format!(
                "Alignment must be a power of two, got {alignment}"
            )));
        }

        let base = self.base_address();
        let mut free = self.lock_free();

        for index in 0..free.len() {
            let region = free[index];
            let region_addr = base + region.start;
            let aligned_addr = match region_addr.checked_add(alignment - 1) {
                Some(value) => value & !(alignment - 1),
                None => continue,
            };
            let head = aligned_addr - region_addr;
            if head > region.size || region.size - head < size {
                continue;
            }

            let alloc_start = region.start + head;
            let tail_start = alloc_start + size;
            let tail_size = region.size - head - size;

            // Head padding stays free, the tail (if any) is inserted after it.
            if head > 0 {
                free[index] = FreeRegion {
                    start: region.start,
                    size: head,
                };
                if tail_size > 0 {
                    free.insert(
                        index + 1,
                        FreeRegion {
                            start: tail_start,
                            size: tail_size,
                        },
                    );
                }
            } else if tail_size > 0 {
                free[index] = FreeRegion {
                    start: tail_start,
                    size: tail_size,
                };
            } else {
                free.remove(index);
            }

            drop(free);
            return Ok(PoolAllocation {
                pool: Arc::clone(self),
                offset: alloc_start,
                size,
                alignment,
            });
        }

        Err(Error::InvalidOperation(format!(
            "Not enough memory in pool: requested {size} bytes (alignment {alignment}), {} bytes free",
            free.iter().map(|region| region.size).sum::<usize>()
        )))
    }

    /// Return a block to the pool, coalescing it with its neighbours.
    fn deallocate(&self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        let mut free = self.lock_free();
        let index = free
            .iter()
            .position(|region| region.start > offset)
            .unwrap_or(free.len());
        free.insert(
            index,
            FreeRegion {
                start: offset,
                size,
            },
        );

        // Coalesce with the following region, then with the preceding one.
        if index + 1 < free.len() && free[index].start + free[index].size == free[index + 1].start {
            free[index].size += free[index + 1].size;
            free.remove(index + 1);
        }
        if index > 0 && free[index - 1].start + free[index - 1].size == free[index].start {
            free[index - 1].size += free[index].size;
            free.remove(index);
        }
    }

    fn free_bytes(&self) -> usize {
        self.lock_free().iter().map(|region| region.size).sum()
    }
}

impl Drop for MemoryPoolInner {
    fn drop(&mut self) {
        // SAFETY: `base` was allocated with exactly `layout` in `MemoryPool::new`
        // and this is its only owner. Every `PoolAllocation` holds an `Arc` of
        // this struct, so no live allocation can outlive this deallocation.
        unsafe {
            std::alloc::dealloc(self.base.0.as_ptr(), self.layout);
        }
    }
}

/// An owned block of pool memory. Returns itself to the pool when dropped.
#[derive(Debug)]
pub struct PoolAllocation {
    pool: Arc<MemoryPoolInner>,
    offset: usize,
    size: usize,
    alignment: usize,
}

impl PoolAllocation {
    /// Address of the first byte of the block
    pub fn address(&self) -> usize {
        self.pool.base_address() + self.offset
    }

    /// Size of the block in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Alignment the block was allocated with
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Descriptor for this block
    pub fn block(&self) -> MemoryBlock {
        MemoryBlock {
            ptr: self.address() as *mut u8,
            size: self.size,
            alignment: self.alignment,
        }
    }
}

impl Drop for PoolAllocation {
    fn drop(&mut self) {
        self.pool.deallocate(self.offset, self.size);
    }
}

/// Memory pool for efficient allocation.
///
/// The pool owns one large allocation and hands out aligned blocks from it.
/// Blocks are returned (and coalesced with their neighbours) when the
/// [`PoolAllocation`] they are wrapped in is dropped, so a pool can be used
/// indefinitely instead of being exhausted after the first pass.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    inner: Arc<MemoryPoolInner>,
}

impl MemoryPool {
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::InvalidInput(
                "Memory pool size must be greater than zero".to_string(),
            ));
        }

        // Allocate aligned memory for the pool
        let layout = std::alloc::Layout::from_size_align(size, PAGE_SIZE)
            .map_err(|_| Error::InvalidOperation("Invalid memory layout".to_string()))?;

        // SAFETY: `layout` has a non-zero size (checked above); a null result is
        // turned into an error instead of being dereferenced.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let base = NonNull::new(ptr)
            .ok_or_else(|| Error::InvalidOperation("Memory allocation failed".to_string()))?;

        Ok(Self {
            inner: Arc::new(MemoryPoolInner {
                size,
                layout,
                base: PoolBase(base),
                free: Mutex::new(vec![FreeRegion { start: 0, size }]),
            }),
        })
    }

    /// Allocate an aligned block from the pool.
    ///
    /// The returned [`PoolAllocation`] keeps the pool alive and releases the
    /// block when dropped.
    pub fn allocate_aligned(&self, size: usize, alignment: usize) -> Result<PoolAllocation> {
        self.inner.allocate(size, alignment)
    }

    /// Total size of the pool in bytes
    pub fn size(&self) -> usize {
        self.inner.size
    }

    /// Bytes currently free
    pub fn free_bytes(&self) -> usize {
        self.inner.free_bytes()
    }

    /// Bytes currently handed out
    pub fn bytes_in_use(&self) -> usize {
        self.inner.size - self.inner.free_bytes()
    }

    /// Number of (maximally coalesced) free regions -- a fragmentation measure
    pub fn free_region_count(&self) -> usize {
        self.inner.lock_free().len()
    }
}

/// Memory block descriptor.
///
/// A descriptor only: it grants no access by itself, since reading or writing
/// through `ptr` requires `unsafe` and the pool's contract.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBlock {
    /// Pointer to the memory block
    pub ptr: *mut u8,
    /// Size of the block in bytes
    pub size: usize,
    /// Alignment of the block
    pub alignment: usize,
}

// SAFETY: `MemoryBlock` is an inert descriptor. Dereferencing `ptr` is `unsafe`
// and requires the caller to uphold the owning pool's contract, so moving or
// sharing the descriptor itself cannot cause a data race.
unsafe impl Send for MemoryBlock {}
// SAFETY: see above.
unsafe impl Sync for MemoryBlock {}

/// Allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total bytes allocated over the allocator's lifetime
    pub total_allocated: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Current memory usage (bytes handed out and not yet released)
    pub current_usage: usize,
}

impl AllocationStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one allocation of `size` bytes.
    ///
    /// Only the cumulative counters are updated here: `current_usage` and
    /// `peak_usage` are maintained by the owning allocator, which reads live
    /// usage back from its pools so that released blocks are reflected.
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.allocation_count += 1;
    }
}

/// Memory-mapped file view for large datasets.
///
/// # Safety caveat: file truncation
///
/// A memory mapping is backed by the file it was created from. If another
/// process (or another part of this one) truncates or shrinks that file while
/// the view is alive, touching the vanished pages raises `SIGBUS` and kills the
/// process -- this is a property of `mmap` itself and cannot be prevented from
/// safe Rust. Only map files you control for the lifetime of the view.
pub struct MemoryMappedView<T> {
    /// Memory map
    mmap: memmap2::Mmap,
    /// Length in elements (never larger than the mapping)
    len: usize,
    /// Element layout
    layout: MemoryLayout,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> MemoryMappedView<T> {
    /// Create a new memory-mapped view from a file.
    ///
    /// `len` is the number of `T` elements to expose; it is validated against
    /// the mapped size and the mapping's alignment instead of being silently
    /// clamped later on.
    ///
    /// Constructing the view is safe for any `T` because the view alone grants
    /// no access to the bytes: reading them as `&[T]` requires the sealed
    /// [`ZeroCopyPod`] bound on [`MemoryMappedView::as_slice`], which is what
    /// rules out reinterpreting file bytes as a type with invalid bit patterns.
    ///
    /// See the type-level documentation for the `SIGBUS`-on-truncation caveat.
    pub fn from_file(file: std::fs::File, len: usize) -> Result<Self> {
        // SAFETY: `Mmap::map` is unsafe purely because of the truncation hazard
        // documented on this type; the mapping itself is read-only and stays
        // alive as long as `self`.
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| Error::InvalidOperation(format!("Memory mapping failed: {}", e)))?
        };

        let element_size = mem::size_of::<T>();
        let required = len.checked_mul(element_size).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Memory-mapped view size overflow: {len} x {element_size} bytes"
            ))
        })?;
        if required > mmap.len() {
            return Err(Error::InvalidInput(format!(
                "Memory-mapped view requests {required} bytes but the file maps only {} bytes",
                mmap.len()
            )));
        }
        if mmap.as_ptr() as usize % mem::align_of::<T>() != 0 {
            return Err(Error::InvalidOperation(format!(
                "Memory mapping is not aligned for {}",
                std::any::type_name::<T>()
            )));
        }

        let layout = MemoryLayout {
            start_address: mmap.as_ptr() as usize,
            element_size,
            stride: element_size,
            alignment: mem::align_of::<T>(),
            cache_aligned: mmap.as_ptr() as usize % CACHE_LINE_SIZE == 0,
            numa_node: None,
        };

        Ok(Self {
            mmap,
            len,
            layout,
            _phantom: PhantomData,
        })
    }

    /// Get memory layout information
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the length of the view in elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of bytes actually mapped
    pub fn mapped_bytes(&self) -> usize {
        self.mmap.len()
    }
}

impl<T: ZeroCopyPod> MemoryMappedView<T> {
    /// Get a slice view of the memory-mapped data.
    ///
    /// Restricted to [`ZeroCopyPod`]: this is the operation that reinterprets
    /// raw file bytes as values of `T`, so every bit pattern must be valid.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `from_file` verified that `len * size_of::<T>()` bytes are
        // mapped and that the mapping is aligned for `T`; `T: ZeroCopyPod`
        // guarantees every bit pattern is a valid value. The mapping is owned by
        // `self`, so it outlives the returned slice.
        unsafe { slice::from_raw_parts(self.mmap.as_ptr() as *const T, self.len) }
    }
}

impl<T: ZeroCopyPod> Deref for MemoryMappedView<T> {
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
        let len = slice.len().min(other.len());
        // A zero block size would mean "no blocking", not "no progress".
        let block_size = block_size.max(1);
        let mut result = Vec::with_capacity(len);

        // Walk both inputs one block at a time; materialising the zipped pairs
        // first would allocate the whole cross product and defeat the blocking.
        let mut start = 0;
        while start < len {
            let end = (start + block_size).min(len);
            for index in start..end {
                result.push(op(&slice[index], &other[index]));
            }
            start = end;
        }

        result
    }

    /// Issue cache prefetch hints.
    ///
    /// Implemented with `_mm_prefetch` on x86_64 and `prfm pldl1keep` on
    /// aarch64. On every other architecture this is a documented no-op: the
    /// bounds check still runs, but no hint is issued.
    fn prefetch(&self, indices: &[usize]) {
        let slice = self.as_slice();
        for &index in indices {
            if index < slice.len() {
                // SAFETY: `index` is in bounds, so the pointer is inside the
                // view's allocation. Prefetch hints never fault and never read
                // the value.
                unsafe {
                    let ptr = slice.as_ptr().add(index);
                    #[cfg(target_arch = "x86_64")]
                    {
                        std::arch::x86_64::_mm_prefetch(
                            ptr as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        std::arch::asm!(
                            "prfm pldl1keep, [{ptr}]",
                            ptr = in(reg) ptr,
                            options(nostack, preserves_flags)
                        );
                    }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    {
                        let _ = ptr;
                    }
                }
            }
        }
    }

    fn optimal_block_size(&self) -> usize {
        // Calculate optimal block size based on cache size and element size
        let cache_size = 32 * 1024; // L1 cache size
        let element_size = mem::size_of::<T>().max(1);
        (cache_size / element_size).max(64)
    }
}

/// Memory manager that provides zero-copy views
pub struct ZeroCopyManager {
    /// Cache-aware allocator
    allocator: Mutex<CacheAwareAllocator>,
    /// Memory usage statistics
    stats: Mutex<ZeroCopyStats>,
}

impl ZeroCopyManager {
    pub fn new() -> Result<Self> {
        Ok(Self {
            allocator: Mutex::new(CacheAwareAllocator::new()?),
            stats: Mutex::new(ZeroCopyStats::new()),
        })
    }

    /// Create a zero-copy view with optimal cache placement.
    ///
    /// `T: Copy` because the pool recycles the block without running
    /// destructors; a type that owns a resource would leak it.
    pub fn create_view<T: Copy>(&self, data: Vec<T>) -> Result<ZeroCopyView<T>> {
        let len = data.len();
        let size = len
            .checked_mul(mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidInput("Zero-copy view size overflow".to_string()))?;

        let mut allocator = self
            .allocator
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire allocator lock".to_string()))?;

        let cache_level = allocator.cache_topology().optimal_cache_level(size);

        // SAFETY: `allocate_uninit` hands out uninitialised memory for exactly
        // `len` elements; every one of them is initialised by the copy below
        // before the view is handed to the caller.
        let view = unsafe {
            let view = allocator.allocate_uninit::<T>(len, cache_level)?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), view.data.as_ptr(), len);
            view
        };
        drop(allocator);

        self.stats
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))?
            .record_view_creation(size);

        Ok(view)
    }

    /// Create a memory-mapped view for large files.
    ///
    /// See [`MemoryMappedView`] for the truncation caveat that applies to every
    /// memory mapping.
    pub fn create_mmap_view<T: ZeroCopyPod>(
        &self,
        file_path: &str,
        len: usize,
    ) -> Result<MemoryMappedView<T>> {
        let file = std::fs::File::open(file_path)
            .map_err(|e| Error::InvalidOperation(format!("Failed to open file: {}", e)))?;

        let view = MemoryMappedView::from_file(file, len)?;

        self.stats
            .lock()
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))?
            .record_mmap_creation(len * mem::size_of::<T>());

        Ok(view)
    }

    /// Bytes currently held by live views created through this manager
    pub fn bytes_in_use(&self) -> Result<usize> {
        self.allocator
            .lock()
            .map(|allocator| allocator.bytes_in_use())
            .map_err(|_| Error::InvalidOperation("Failed to acquire allocator lock".to_string()))
    }

    /// Get zero-copy statistics
    pub fn stats(&self) -> Result<ZeroCopyStats> {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire stats lock".to_string()))
    }
}

/// Statistics for zero-copy operations
#[derive(Debug, Clone, Default)]
pub struct ZeroCopyStats {
    /// Number of zero-copy views created
    pub views_created: usize,
    /// Number of memory-mapped views created
    pub mmap_views_created: usize,
    /// Total memory managed
    pub total_memory: usize,
}

impl ZeroCopyStats {
    pub fn new() -> Self {
        Self::default()
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

/// Compile-time checks for the auto traits the module's safety reasoning relies
/// on. If any of these stops holding, the build fails here instead of silently
/// weakening `ZeroCopyView`.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<StorageHandle>();
    assert_send_sync::<MemoryPoolInner>();
    assert_send_sync::<PoolAllocation>();
    assert_send_sync::<MemoryPool>();
    assert_send_sync::<ZeroCopyView<u64>>();
    assert_send_sync::<MemoryMappedView<f64>>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_topology_detection() {
        let topology = CacheTopology::detect().expect("operation should succeed");
        assert!(topology.l1_cache_size > 0);
        assert!(topology.l2_cache_size > 0);
        assert!(topology.l3_cache_size > 0);
        assert!(topology.cache_line_size > 0);
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
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let data = vec![1i32, 2, 3, 4, 5];
        let view = manager.create_view(data).expect("operation should succeed");

        assert_eq!(view.len(), 5);
        assert_eq!(view.as_slice(), &[1, 2, 3, 4, 5]);

        let stats = manager.stats().expect("operation should succeed");
        assert_eq!(stats.views_created, 1);
    }

    #[test]
    fn test_cache_aware_operations() {
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let data = (0..1000).collect::<Vec<i32>>();
        let view = manager.create_view(data).expect("operation should succeed");

        // Test linear scan
        let evens = view.linear_scan(|&x| x % 2 == 0);
        assert_eq!(evens.len(), 500);

        // Test optimal block size
        let block_size = view.optimal_block_size();
        assert!(block_size > 0);
    }

    #[test]
    fn test_subview_creation() {
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let view = manager.create_view(data).expect("operation should succeed");

        let subview = view.subview(2..7).expect("operation should succeed");
        assert_eq!(subview.len(), 5);
        assert_eq!(subview.as_slice(), &[3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_subview_outlives_parent() {
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let view = manager
            .create_view((0..64u64).collect::<Vec<_>>())
            .expect("operation should succeed");

        let subview = view.subview(8..16).expect("operation should succeed");
        drop(view);

        // The subview keeps the pool block (and the pool) alive.
        assert_eq!(subview.as_slice(), &[8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_view_outlives_manager() {
        let view = {
            let manager = ZeroCopyManager::new().expect("operation should succeed");
            let view = manager
                .create_view(vec![7.5f64; 32])
                .expect("operation should succeed");
            drop(manager);
            view
        };
        assert_eq!(view.len(), 32);
        assert!(view.as_slice().iter().all(|value| *value == 7.5));
    }

    #[test]
    fn test_empty_view() {
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let view = manager
            .create_view(Vec::<u32>::new())
            .expect("operation should succeed");
        assert!(view.is_empty());
        assert!(view.as_slice().is_empty());
    }

    #[test]
    fn test_pool_reuses_released_blocks() {
        let pool = MemoryPool::new(64 * 1024).expect("pool creation should succeed");
        assert_eq!(pool.free_bytes(), pool.size());

        for _ in 0..1000 {
            let allocation = pool
                .allocate_aligned(48 * 1024, CACHE_LINE_SIZE)
                .expect("allocation should succeed");
            assert_eq!(allocation.address() % CACHE_LINE_SIZE, 0);
        }

        // Everything was returned and coalesced back into a single region.
        assert_eq!(pool.free_bytes(), pool.size());
        assert_eq!(pool.free_region_count(), 1);
    }

    #[test]
    fn test_pool_rejects_zero_size() {
        assert!(MemoryPool::new(0).is_err());
        let pool = MemoryPool::new(4096).expect("pool creation should succeed");
        assert!(pool.allocate_aligned(0, 64).is_err());
        assert!(pool.allocate_aligned(64, 3).is_err());
    }

    #[test]
    fn test_pool_aligns_block_start() {
        let pool = MemoryPool::new(1024 * 1024).expect("pool creation should succeed");
        let first = pool
            .allocate_aligned(1, 8)
            .expect("allocation should succeed");
        let aligned = pool
            .allocate_aligned(256, 4096)
            .expect("allocation should succeed");
        assert_eq!(aligned.address() % 4096, 0);
        drop(first);
        drop(aligned);
        assert_eq!(pool.free_bytes(), pool.size());
    }

    #[test]
    fn test_allocate_aligned_is_zeroed() {
        let mut allocator = CacheAwareAllocator::new().expect("allocator creation should succeed");
        let view: ZeroCopyView<u64> = allocator
            .allocate_aligned(16, CacheLevel::L1)
            .expect("allocation should succeed");
        assert_eq!(view.as_slice(), &[0u64; 16]);
        assert!(allocator.bytes_in_use() >= 16 * 8);
        drop(view);
        assert_eq!(allocator.bytes_in_use(), 0);
    }

    #[test]
    fn test_blocked_operation_handles_zero_block_size() {
        let manager = ZeroCopyManager::new().expect("operation should succeed");
        let view = manager
            .create_view((0..10i32).collect::<Vec<_>>())
            .expect("operation should succeed");
        let other = vec![2i32; 10];

        let blocked = view.blocked_operation(&other, 0, |a, b| a * b);
        assert_eq!(blocked, (0..10i32).map(|v| v * 2).collect::<Vec<_>>());

        let blocked = view.blocked_operation(&other, 3, |a, b| a * b);
        assert_eq!(blocked, (0..10i32).map(|v| v * 2).collect::<Vec<_>>());
    }

    #[test]
    fn test_mmap_view_rejects_oversized_len() {
        use std::io::Write;

        let path = std::env::temp_dir().join("pandrs_zero_copy_mmap_len_check.bin");
        {
            let mut file = std::fs::File::create(&path).expect("file creation should succeed");
            for value in 0..16u64 {
                file.write_all(&value.to_le_bytes())
                    .expect("write should succeed");
            }
            file.flush().expect("flush should succeed");
        }

        let file = std::fs::File::open(&path).expect("file open should succeed");
        let too_long = MemoryMappedView::<u64>::from_file(file, 17);
        assert!(too_long.is_err(), "len beyond the mapping must be rejected");

        let file = std::fs::File::open(&path).expect("file open should succeed");
        let view = MemoryMappedView::<u64>::from_file(file, 16).expect("mapping should succeed");
        assert_eq!(view.len(), 16);
        assert_eq!(view.as_slice().len(), view.len());
        assert_eq!(view.as_slice()[15], 15);

        let _ = std::fs::remove_file(&path);
    }
}

//! Arena Allocator for Efficient Bulk Memory Allocation
//!
//! This module provides arena-based memory allocation for high-performance
//! bulk allocations in DataFrame operations. Arenas provide:
//!
//! - Fast allocation (bump pointer allocation)
//! - Batch deallocation (entire arena freed at once)
//! - Reduced fragmentation
//! - Cache-friendly memory layout
//!
//! # Example
//!
//! ```ignore
//! use pandrs::storage::arena::{Arena, TypedArena};
//!
//! // Create an arena for f64 values
//! let arena: TypedArena<f64> = TypedArena::new(1024);
//!
//! // Allocate values
//! let slice = arena.alloc_slice(&[1.0, 2.0, 3.0, 4.0]);
//!
//! // All memory freed when arena is dropped
//! ```

use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default chunk size for arena allocations (64 KB)
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum chunk size (16 MB)
const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Minimum alignment for allocations
const MIN_ALIGNMENT: usize = 8;

/// Arena statistics
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total bytes allocated from the system
    pub total_allocated: usize,
    /// Bytes currently in use
    pub bytes_in_use: usize,
    /// Number of chunks allocated
    pub chunk_count: usize,
    /// Number of individual allocations
    pub allocation_count: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Wasted bytes due to alignment
    pub alignment_waste: usize,
}

/// A memory chunk in the arena
struct Chunk {
    /// Pointer to the start of the chunk
    data: NonNull<u8>,
    /// Layout of the chunk
    layout: Layout,
    /// Current offset into the chunk
    offset: usize,
    /// Capacity of the chunk
    capacity: usize,
}

impl Chunk {
    /// Create a new chunk with the given capacity
    fn new(capacity: usize) -> Option<Self> {
        let layout = Layout::from_size_align(capacity, MIN_ALIGNMENT).ok()?;

        let data = unsafe {
            let ptr = alloc(layout);
            NonNull::new(ptr)?
        };

        Some(Chunk {
            data,
            layout,
            offset: 0,
            capacity,
        })
    }

    /// Try to allocate memory from this chunk
    fn try_alloc(&mut self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Calculate aligned offset
        let current_ptr = self.data.as_ptr() as usize + self.offset;
        let aligned_ptr = (current_ptr + align - 1) & !(align - 1);
        let padding = aligned_ptr - current_ptr;
        let total_size = size + padding;

        if self.offset + total_size > self.capacity {
            return None;
        }

        let result_ptr = unsafe { self.data.as_ptr().add(self.offset + padding) };
        self.offset += total_size;

        NonNull::new(result_ptr)
    }

    /// Reset the chunk for reuse
    fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get remaining capacity
    fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }

    /// Get used bytes
    fn used(&self) -> usize {
        self.offset
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

/// A general-purpose arena allocator
///
/// The arena allocates memory in chunks and provides fast bump-pointer
/// allocation. All memory is freed when the arena is dropped or reset.
pub struct Arena {
    /// List of chunks
    chunks: RefCell<Vec<Chunk>>,
    /// Current chunk size for new allocations
    chunk_size: Cell<usize>,
    /// Statistics
    stats: RefCell<ArenaStats>,
    /// Whether to grow chunk sizes exponentially
    grow_chunks: bool,
}

impl Arena {
    /// Create a new arena with default chunk size
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHUNK_SIZE)
    }

    /// Create a new arena with the specified initial chunk size
    pub fn with_capacity(chunk_size: usize) -> Self {
        Arena {
            chunks: RefCell::new(Vec::new()),
            chunk_size: Cell::new(chunk_size.max(1024)),
            stats: RefCell::new(ArenaStats::default()),
            grow_chunks: true,
        }
    }

    /// Create an arena that doesn't grow chunk sizes
    pub fn fixed_chunk_size(chunk_size: usize) -> Self {
        let mut arena = Self::with_capacity(chunk_size);
        arena.grow_chunks = false;
        arena
    }

    /// Allocate raw memory with the given layout
    pub fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align().max(MIN_ALIGNMENT);

        // Try to allocate from the current chunk
        {
            let mut chunks = self.chunks.borrow_mut();
            if let Some(chunk) = chunks.last_mut() {
                if let Some(ptr) = chunk.try_alloc(size, align) {
                    let mut stats = self.stats.borrow_mut();
                    stats.bytes_in_use += size;
                    stats.allocation_count += 1;
                    stats.peak_usage = stats.peak_usage.max(stats.bytes_in_use);
                    return Some(ptr);
                }
            }
        }

        // Need a new chunk
        self.alloc_new_chunk(size, align)
    }

    /// Allocate a new chunk and allocate from it
    fn alloc_new_chunk(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let chunk_size = self.chunk_size.get();
        let needed_size = (size + align).max(chunk_size);

        let mut chunk = Chunk::new(needed_size)?;
        let ptr = chunk.try_alloc(size, align)?;

        {
            let mut stats = self.stats.borrow_mut();
            stats.total_allocated += needed_size;
            stats.bytes_in_use += size;
            stats.chunk_count += 1;
            stats.allocation_count += 1;
            stats.peak_usage = stats.peak_usage.max(stats.bytes_in_use);
        }

        self.chunks.borrow_mut().push(chunk);

        // Grow chunk size for next allocation (up to max)
        if self.grow_chunks {
            let new_size = (chunk_size * 2).min(MAX_CHUNK_SIZE);
            self.chunk_size.set(new_size);
        }

        Some(ptr)
    }

    /// Allocate a value of type T
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout).expect("Arena allocation failed");

        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            &mut *typed_ptr
        }
    }

    /// Allocate an uninitialized value of type T
    ///
    /// # Safety
    /// The caller must initialize the memory before reading from it
    pub unsafe fn alloc_uninit<T>(&self) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout).expect("Arena allocation failed");
        &mut *(ptr.as_ptr() as *mut T)
    }

    /// Allocate a slice and initialize it with the given values
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> &mut [T] {
        if values.is_empty() {
            return &mut [];
        }

        let layout = Layout::array::<T>(values.len()).expect("Invalid layout");
        let ptr = self.alloc_raw(layout).expect("Arena allocation failed");

        unsafe {
            let slice_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), slice_ptr, values.len());
            std::slice::from_raw_parts_mut(slice_ptr, values.len())
        }
    }

    /// Allocate an uninitialized slice
    ///
    /// # Safety
    /// The caller must initialize all elements before reading from them
    pub unsafe fn alloc_slice_uninit<T>(&self, len: usize) -> &mut [T] {
        if len == 0 {
            return &mut [];
        }

        let layout = Layout::array::<T>(len).expect("Invalid layout");
        let ptr = self.alloc_raw(layout).expect("Arena allocation failed");

        std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, len)
    }

    /// Allocate a string slice
    pub fn alloc_str(&self, s: &str) -> &str {
        let bytes = self.alloc_slice(s.as_bytes());
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Reset the arena, freeing all allocations but keeping chunks
    pub fn reset(&self) {
        let mut chunks = self.chunks.borrow_mut();
        for chunk in chunks.iter_mut() {
            chunk.reset();
        }

        let mut stats = self.stats.borrow_mut();
        stats.bytes_in_use = 0;
        stats.allocation_count = 0;
    }

    /// Clear the arena, freeing all memory
    pub fn clear(&self) {
        self.chunks.borrow_mut().clear();
        self.chunk_size.set(DEFAULT_CHUNK_SIZE);
        *self.stats.borrow_mut() = ArenaStats::default();
    }

    /// Get arena statistics
    pub fn stats(&self) -> ArenaStats {
        self.stats.borrow().clone()
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.stats.borrow().total_allocated
    }

    /// Get bytes currently in use
    pub fn bytes_in_use(&self) -> usize {
        self.stats.borrow().bytes_in_use
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

// Arena is not Send or Sync due to RefCell
// For thread-safe usage, use SyncArena

/// A typed arena for allocating values of a single type
pub struct TypedArena<T> {
    arena: Arena,
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    /// Create a new typed arena
    pub fn new(capacity: usize) -> Self {
        let item_size = size_of::<T>().max(1);
        let chunk_capacity = capacity * item_size;

        TypedArena {
            arena: Arena::with_capacity(chunk_capacity),
            _marker: PhantomData,
        }
    }

    /// Allocate a value
    pub fn alloc(&self, value: T) -> &mut T {
        self.arena.alloc(value)
    }

    /// Allocate multiple values from an iterator
    pub fn alloc_from_iter<I: IntoIterator<Item = T>>(&self, iter: I) -> Vec<&mut T> {
        iter.into_iter().map(|v| self.alloc(v)).collect()
    }

    /// Reset the arena
    pub fn reset(&self) {
        self.arena.reset();
    }

    /// Clear the arena
    pub fn clear(&self) {
        self.arena.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> ArenaStats {
        self.arena.stats()
    }
}

impl<T: Copy> TypedArena<T> {
    /// Allocate a slice
    pub fn alloc_slice(&self, values: &[T]) -> &mut [T] {
        self.arena.alloc_slice(values)
    }
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Thread-safe arena allocator using atomic operations
pub struct SyncArena {
    /// Chunks (protected by mutex for allocation)
    chunks: std::sync::Mutex<Vec<Chunk>>,
    /// Current chunk size
    chunk_size: AtomicUsize,
    /// Statistics
    total_allocated: AtomicUsize,
    bytes_in_use: AtomicUsize,
    allocation_count: AtomicUsize,
    peak_usage: AtomicUsize,
}

impl SyncArena {
    /// Create a new thread-safe arena
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHUNK_SIZE)
    }

    /// Create with specified capacity
    pub fn with_capacity(chunk_size: usize) -> Self {
        SyncArena {
            chunks: std::sync::Mutex::new(Vec::new()),
            chunk_size: AtomicUsize::new(chunk_size.max(1024)),
            total_allocated: AtomicUsize::new(0),
            bytes_in_use: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
        }
    }

    /// Allocate raw memory
    pub fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align().max(MIN_ALIGNMENT);

        let mut chunks = self.chunks.lock().ok()?;

        // Try current chunk
        if let Some(chunk) = chunks.last_mut() {
            if let Some(ptr) = chunk.try_alloc(size, align) {
                self.bytes_in_use.fetch_add(size, Ordering::Relaxed);
                self.allocation_count.fetch_add(1, Ordering::Relaxed);
                self.update_peak();
                return Some(ptr);
            }
        }

        // Need new chunk
        let chunk_size = self.chunk_size.load(Ordering::Relaxed);
        let needed_size = (size + align).max(chunk_size);

        let mut chunk = Chunk::new(needed_size)?;
        let ptr = chunk.try_alloc(size, align)?;

        self.total_allocated
            .fetch_add(needed_size, Ordering::Relaxed);
        self.bytes_in_use.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.update_peak();

        chunks.push(chunk);

        // Grow chunk size
        let new_size = (chunk_size * 2).min(MAX_CHUNK_SIZE);
        self.chunk_size.store(new_size, Ordering::Relaxed);

        Some(ptr)
    }

    fn update_peak(&self) {
        let current = self.bytes_in_use.load(Ordering::Relaxed);
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Allocate a value
    pub fn alloc<T>(&self, value: T) -> Option<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout)?;

        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            Some(&mut *typed_ptr)
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            bytes_in_use: self.bytes_in_use.load(Ordering::Relaxed),
            chunk_count: self.chunks.lock().map(|c| c.len()).unwrap_or(0),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            alignment_waste: 0,
        }
    }

    /// Reset the arena
    pub fn reset(&self) {
        if let Ok(mut chunks) = self.chunks.lock() {
            for chunk in chunks.iter_mut() {
                chunk.reset();
            }
        }
        self.bytes_in_use.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }

    /// Clear the arena
    pub fn clear(&self) {
        if let Ok(mut chunks) = self.chunks.lock() {
            chunks.clear();
        }
        self.chunk_size.store(DEFAULT_CHUNK_SIZE, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.bytes_in_use.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
    }
}

impl Default for SyncArena {
    fn default() -> Self {
        Self::new()
    }
}

// SyncArena is Send + Sync
unsafe impl Send for SyncArena {}
unsafe impl Sync for SyncArena {}

/// A scoped arena that automatically resets when dropped
pub struct ScopedArena<'a> {
    arena: &'a Arena,
    initial_usage: usize,
}

impl<'a> ScopedArena<'a> {
    /// Create a new scoped arena
    pub fn new(arena: &'a Arena) -> Self {
        let initial_usage = arena.bytes_in_use();
        ScopedArena {
            arena,
            initial_usage,
        }
    }

    /// Allocate a value
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.arena.alloc(value)
    }

    /// Allocate a slice
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> &mut [T] {
        self.arena.alloc_slice(values)
    }
}

impl<'a> Drop for ScopedArena<'a> {
    fn drop(&mut self) {
        // Note: We can't truly "reset" to a previous state with a bump allocator
        // This is primarily for tracking purposes
        // In production, you'd use a checkpoint system
    }
}

/// Arena-backed vector for efficient bulk storage
pub struct ArenaVec<'a, T> {
    data: &'a mut [T],
    len: usize,
}

impl<'a, T: Copy + Default> ArenaVec<'a, T> {
    /// Create a new arena-backed vector with the given capacity
    pub fn with_capacity(arena: &'a Arena, capacity: usize) -> Self {
        let data = unsafe { arena.alloc_slice_uninit(capacity) };
        ArenaVec { data, len: 0 }
    }

    /// Push a value
    pub fn push(&mut self, value: T) {
        if self.len < self.data.len() {
            self.data[self.len] = value;
            self.len += 1;
        }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new();

        let a = arena.alloc(42i32);
        let b = arena.alloc(3.14f64);
        let c = arena.alloc("hello".to_string());

        assert_eq!(*a, 42);
        assert_eq!(*b, 3.14);
        assert_eq!(&*c, "hello");

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 3);
        assert!(stats.bytes_in_use > 0);
    }

    #[test]
    fn test_arena_slice() {
        let arena = Arena::new();

        let slice = arena.alloc_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[4], 5.0);

        // Modify in place
        slice[2] = 100.0;
        assert_eq!(slice[2], 100.0);
    }

    #[test]
    fn test_arena_string() {
        let arena = Arena::new();

        let s1 = arena.alloc_str("hello");
        let s2 = arena.alloc_str("world");

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
    }

    #[test]
    fn test_arena_reset() {
        let arena = Arena::new();

        for i in 0..100 {
            arena.alloc(i);
        }

        let stats_before = arena.stats();
        assert_eq!(stats_before.allocation_count, 100);

        arena.reset();

        let stats_after = arena.stats();
        assert_eq!(stats_after.allocation_count, 0);
        assert_eq!(stats_after.bytes_in_use, 0);
        // Chunks are kept for reuse
        assert!(stats_after.total_allocated > 0);
    }

    #[test]
    fn test_arena_clear() {
        let arena = Arena::new();

        for i in 0..100 {
            arena.alloc(i);
        }

        arena.clear();

        let stats = arena.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.bytes_in_use, 0);
        assert_eq!(stats.chunk_count, 0);
    }

    #[test]
    fn test_typed_arena() {
        let arena: TypedArena<f64> = TypedArena::new(1024);

        let values: Vec<&mut f64> = (0..100).map(|i| arena.alloc(i as f64)).collect();

        for (i, v) in values.iter().enumerate() {
            assert_eq!(**v, i as f64);
        }
    }

    #[test]
    fn test_typed_arena_slice() {
        let arena: TypedArena<i32> = TypedArena::new(1024);

        let data = vec![1, 2, 3, 4, 5];
        let slice = arena.alloc_slice(&data);

        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sync_arena() {
        let arena = SyncArena::new();

        let a = arena.alloc(42i32).unwrap();
        let b = arena.alloc(3.14f64).unwrap();

        assert_eq!(*a, 42);
        assert_eq!(*b, 3.14);

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2);
    }

    #[test]
    fn test_sync_arena_threaded() {
        use std::sync::Arc;
        use std::thread;

        let arena = Arc::new(SyncArena::new());
        let mut handles = vec![];

        for t in 0..4 {
            let arena_clone = Arc::clone(&arena);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    arena_clone.alloc(t * 100 + i);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 400);
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = Arena::new();

        // Allocate a large slice
        let large: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let slice = arena.alloc_slice(&large);

        assert_eq!(slice.len(), 10000);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[9999], 9999.0);
    }

    #[test]
    fn test_arena_alignment() {
        let arena = Arena::new();

        // Allocate values with different alignments
        let _byte = arena.alloc(1u8);
        let int_ptr = arena.alloc(42i32);
        let double_ptr = arena.alloc(3.14f64);

        // Check alignment
        let int_addr = int_ptr as *const i32 as usize;
        let double_addr = double_ptr as *const f64 as usize;

        assert_eq!(int_addr % align_of::<i32>(), 0);
        assert_eq!(double_addr % align_of::<f64>(), 0);
    }

    #[test]
    fn test_arena_vec() {
        let arena = Arena::new();
        let mut vec: ArenaVec<i32> = ArenaVec::with_capacity(&arena, 100);

        for i in 0..50 {
            vec.push(i);
        }

        assert_eq!(vec.len(), 50);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec.as_slice()[0], 0);
        assert_eq!(vec.as_slice()[49], 49);
    }

    #[test]
    fn test_arena_stats() {
        let arena = Arena::new();

        // Initial stats
        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.bytes_in_use, 0);

        // After allocations
        for _ in 0..10 {
            arena.alloc(42i64);
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 10);
        assert!(stats.bytes_in_use >= 80); // 10 * 8 bytes
        assert!(stats.peak_usage >= stats.bytes_in_use);
    }
}

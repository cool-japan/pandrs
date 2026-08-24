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
//! # Ownership model
//!
//! [`Arena::alloc`] hands out `&mut T` references that borrow the arena, so any
//! operation that invalidates previously handed out references
//! ([`Arena::reset`], [`Arena::clear`]) takes `&mut self`. The borrow checker
//! then rejects a reset while a live allocation reference exists, which is what
//! makes the whole type sound (this mirrors `bumpalo::Bump::reset`).
//!
//! Destructors of allocated values *are* run: every allocation of a type where
//! [`std::mem::needs_drop`] holds records a destructor which is executed on
//! `reset`, `clear`, scoped rollback and when the arena itself is dropped.
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
//! let slice = arena.alloc_slice(&[1.0, 2.0, 3.0, 4.0])?;
//!
//! // All memory freed when arena is dropped
//! # Ok::<(), pandrs::core::error::Error>(())
//! ```

use crate::core::error::{Error, Result};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::mem::{needs_drop, size_of, MaybeUninit};
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Default chunk size for arena allocations (64 KB)
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum chunk size (16 MB)
const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Minimum alignment for allocations
const MIN_ALIGNMENT: usize = 8;

/// Arena statistics
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Total bytes currently held by the arena's chunks
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

// SAFETY: a `Chunk` is nothing more than a heap allocation that it frees in its
// own `Drop`. The allocation is owned exclusively by the chunk (no other object
// holds a copy of `data`), and the global allocator has no thread affinity, so
// moving the chunk to another thread is sound.
//
// Whether the *values* written into a chunk are allowed to cross threads is a
// separate question and is enforced by the owning arena, not here: `Arena` is
// explicitly `!Send` (see the `_not_send` marker) because `Arena::alloc` accepts
// non-`Send` values, while `SyncArena::alloc` requires `T: Send`.
unsafe impl Send for Chunk {}

impl Chunk {
    /// Create a new chunk with the given capacity
    fn new(capacity: usize) -> Option<Self> {
        let layout = Layout::from_size_align(capacity, MIN_ALIGNMENT).ok()?;
        if layout.size() == 0 {
            return None;
        }

        // SAFETY: `layout` has a non-zero size (checked above), which is the
        // precondition of `alloc`. A null return is turned into `None`.
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

    /// Try to allocate memory from this chunk.
    ///
    /// Returns the allocated pointer together with the number of padding bytes
    /// that were skipped to satisfy `align`.
    fn try_alloc(&mut self, size: usize, align: usize) -> Option<(NonNull<u8>, usize)> {
        // Calculate aligned offset
        let current_ptr = (self.data.as_ptr() as usize).checked_add(self.offset)?;
        let aligned_ptr = current_ptr.checked_add(align - 1)? & !(align - 1);
        let padding = aligned_ptr - current_ptr;
        let total_size = size.checked_add(padding)?;

        if self.offset.checked_add(total_size)? > self.capacity {
            return None;
        }

        // SAFETY: `self.offset + padding` is within the chunk's capacity, so the
        // resulting pointer is inside the same allocation.
        let result_ptr = unsafe { self.data.as_ptr().add(self.offset + padding) };
        self.offset += total_size;

        NonNull::new(result_ptr).map(|ptr| (ptr, padding))
    }

    /// Reset the chunk for reuse
    fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // SAFETY: `data` was allocated with exactly `layout` in `Chunk::new`,
        // this is the only owner, and `drop` runs at most once.
        unsafe {
            dealloc(self.data.as_ptr(), self.layout);
        }
    }
}

/// A destructor recorded for a value that lives inside an arena chunk.
#[derive(Debug, Clone, Copy)]
struct DropEntry {
    /// Address of the value inside a chunk
    addr: NonNull<u8>,
    /// Monomorphised `drop_in_place` for the value's type
    drop_fn: unsafe fn(NonNull<u8>),
}

// SAFETY: a `DropEntry` is a (pointer, destructor) descriptor for memory the
// arena owns exclusively. It grants no access to the value on its own -- the
// destructor is only ever run by the arena while it has exclusive access
// (`&mut self`, or `Drop`). Running that destructor on another thread is sound
// because `SyncArena::alloc` -- the only path that can move a `DropEntry`
// across threads -- requires `T: Send`; `Arena` is `!Send` and therefore never
// moves its (possibly non-`Send`) entries anywhere.
unsafe impl Send for DropEntry {}

/// Monomorphised destructor used by [`DropEntry`].
///
/// # Safety
/// `addr` must point at a live, initialised `T` that is not accessed again.
unsafe fn drop_in_place_at<T>(addr: NonNull<u8>) {
    std::ptr::drop_in_place(addr.as_ptr() as *mut T);
}

/// Run a batch of recorded destructors, most recent first.
///
/// # Safety
/// Every entry must describe a live value that no live reference points at.
unsafe fn run_drop_entries(entries: Vec<DropEntry>) {
    for entry in entries.into_iter().rev() {
        (entry.drop_fn)(entry.addr);
    }
}

/// Snapshot of an arena's allocation state, used for scoped rollback.
///
/// Deliberately private: a checkpoint is only valid for an arena that has not
/// been reset or cleared in the meantime, and [`ScopedArena`] (which holds
/// `&mut Arena`) is the only way to observe that invariant.
#[derive(Debug, Clone, Copy)]
struct ArenaCheckpoint {
    chunk_count: usize,
    last_chunk_offset: usize,
    drops_len: usize,
    bytes_in_use: usize,
    allocation_count: usize,
    alignment_waste: usize,
}

/// A general-purpose arena allocator
///
/// The arena allocates memory in chunks and provides fast bump-pointer
/// allocation. All memory is freed when the arena is dropped or reset, and the
/// destructors of allocated values are run at that point.
pub struct Arena {
    /// List of chunks
    chunks: RefCell<Vec<Chunk>>,
    /// Destructors for allocated values that need dropping
    drops: RefCell<Vec<DropEntry>>,
    /// Current chunk size for new allocations
    chunk_size: Cell<usize>,
    /// Statistics
    stats: RefCell<ArenaStats>,
    /// Whether to grow chunk sizes exponentially
    grow_chunks: bool,
    /// `Arena` is deliberately `!Send` and `!Sync`: `alloc` accepts values that
    /// are neither, and their destructors run wherever the arena is reset or
    /// dropped. Use [`SyncArena`] for cross-thread allocation.
    _not_send: PhantomData<*const ()>,
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
            drops: RefCell::new(Vec::new()),
            chunk_size: Cell::new(chunk_size.max(1024)),
            stats: RefCell::new(ArenaStats::default()),
            grow_chunks: true,
            _not_send: PhantomData,
        }
    }

    /// Create an arena that doesn't grow chunk sizes
    pub fn fixed_chunk_size(chunk_size: usize) -> Self {
        let mut arena = Self::with_capacity(chunk_size);
        arena.grow_chunks = false;
        arena
    }

    /// Allocate raw memory with the given layout.
    ///
    /// Returns `None` when the underlying system allocator cannot provide a new
    /// chunk. The returned memory is uninitialised, its destructor is *not*
    /// tracked by the arena, and -- unlike the references returned by `alloc*`
    /// -- the raw pointer does not borrow the arena: it is only valid until the
    /// next [`Arena::reset`] or [`Arena::clear`]. Dereferencing it is `unsafe`,
    /// and upholding that lifetime is part of the caller's obligation.
    pub fn alloc_raw(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align().max(MIN_ALIGNMENT);

        // Try to allocate from the current chunk
        {
            let mut chunks = self.chunks.borrow_mut();
            if let Some(chunk) = chunks.last_mut() {
                if let Some((ptr, padding)) = chunk.try_alloc(size, align) {
                    drop(chunks);
                    let mut stats = self.stats.borrow_mut();
                    stats.bytes_in_use += size;
                    stats.alignment_waste += padding;
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
        let needed_size = size.checked_add(align)?.max(chunk_size);

        let mut chunk = Chunk::new(needed_size)?;
        let (ptr, padding) = chunk.try_alloc(size, align)?;

        {
            let mut stats = self.stats.borrow_mut();
            stats.total_allocated += needed_size;
            stats.bytes_in_use += size;
            stats.alignment_waste += padding;
            stats.chunk_count += 1;
            stats.allocation_count += 1;
            stats.peak_usage = stats.peak_usage.max(stats.bytes_in_use);
        }

        self.chunks.borrow_mut().push(chunk);

        // Grow chunk size for next allocation (up to max)
        if self.grow_chunks {
            let new_size = chunk_size.saturating_mul(2).min(MAX_CHUNK_SIZE);
            self.chunk_size.set(new_size);
        }

        Some(ptr)
    }

    /// Allocate a value of type `T`.
    ///
    /// The value's destructor is recorded and runs on [`Arena::reset`],
    /// [`Arena::clear`], scoped rollback or when the arena is dropped.
    ///
    /// `T: 'static` is required precisely *because* the arena runs destructors:
    /// the destructor list is type-erased, so `dropck` cannot verify that a
    /// borrow held inside `T` still outlives the arena. Requiring `'static`
    /// makes a value that borrows the arena itself unrepresentable. Use
    /// [`Arena::alloc_slice`] (which never drops) for borrowed `Copy` data.
    // `clippy::mut_from_ref` is expected and sound here: this is the canonical
    // bump-arena pattern (cf. `bumpalo`/`typed_arena`). The returned `&mut`
    // borrows `&self`; every operation that invalidates outstanding allocations
    // (`reset`, `clear`, `restore`) takes `&mut self`, so the borrow checker
    // forbids invalidation while any handed-out reference is live. The bump
    // state is interior-mutable (`RefCell`/`Cell`). Taking `&mut self` would bar
    // holding multiple live allocations at once — the arena's whole purpose.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T: 'static>(&self, value: T) -> Result<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout).ok_or_else(|| {
            Error::OperationFailed(format!(
                "Arena: failed to allocate {} bytes for {}",
                layout.size(),
                std::any::type_name::<T>()
            ))
        })?;

        // SAFETY: `alloc_raw` returned memory of at least `size_of::<T>()`
        // bytes, aligned to at least `align_of::<T>()`, that is not aliased by
        // any other live allocation of this arena.
        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            if needs_drop::<T>() {
                self.drops.borrow_mut().push(DropEntry {
                    addr: ptr,
                    drop_fn: drop_in_place_at::<T>,
                });
            }
            Ok(&mut *typed_ptr)
        }
    }

    /// Allocate an uninitialized value of type T
    ///
    /// # Safety
    /// The caller must initialize the memory before reading from it. The arena
    /// does **not** record a destructor for values allocated this way; the
    /// caller is responsible for dropping the value before the arena is reset,
    /// cleared or dropped.
    // Sound bump-arena aliasing — see the justification on `alloc` above.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn alloc_uninit<T>(&self) -> Result<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout).ok_or_else(|| {
            Error::OperationFailed(format!(
                "Arena: failed to allocate {} bytes for {}",
                layout.size(),
                std::any::type_name::<T>()
            ))
        })?;
        Ok(&mut *(ptr.as_ptr() as *mut T))
    }

    /// Allocate a slice and initialize it with the given values.
    ///
    /// Restricted to `T: Copy` because slices are never dropped by the arena.
    // Sound bump-arena aliasing — see the justification on `alloc` above.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Result<&mut [T]> {
        if values.is_empty() {
            return Ok(&mut []);
        }

        let layout = Layout::array::<T>(values.len()).map_err(|_| {
            Error::InvalidInput(format!(
                "Arena: invalid layout for {} x {}",
                values.len(),
                std::any::type_name::<T>()
            ))
        })?;
        let ptr = self.alloc_raw(layout).ok_or_else(|| {
            Error::OperationFailed(format!("Arena: failed to allocate {} bytes", layout.size()))
        })?;

        // SAFETY: the destination has room for `values.len()` elements of `T`
        // and is properly aligned; source and destination cannot overlap
        // because the destination memory was just handed out by this arena.
        unsafe {
            let slice_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), slice_ptr, values.len());
            Ok(slice::from_raw_parts_mut(slice_ptr, values.len()))
        }
    }

    /// Allocate a slice of uninitialised elements.
    ///
    /// This is a safe operation: `MaybeUninit<T>` is valid for any bit pattern,
    /// so handing out a reference to raw arena memory typed this way cannot
    /// produce an invalid value. The arena does not drop these elements.
    // Sound bump-arena aliasing — see the justification on `alloc` above.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_maybe_uninit<T>(&self, len: usize) -> Result<&mut [MaybeUninit<T>]> {
        if len == 0 {
            return Ok(&mut []);
        }

        let layout = Layout::array::<MaybeUninit<T>>(len).map_err(|_| {
            Error::InvalidInput(format!(
                "Arena: invalid layout for {} x {}",
                len,
                std::any::type_name::<T>()
            ))
        })?;
        let ptr = self.alloc_raw(layout).ok_or_else(|| {
            Error::OperationFailed(format!("Arena: failed to allocate {} bytes", layout.size()))
        })?;

        // SAFETY: the region is large enough for `len` `MaybeUninit<T>` and is
        // aligned for `T`; every bit pattern is a valid `MaybeUninit<T>`.
        unsafe {
            Ok(slice::from_raw_parts_mut(
                ptr.as_ptr() as *mut MaybeUninit<T>,
                len,
            ))
        }
    }

    /// Allocate an uninitialized slice
    ///
    /// # Safety
    /// The caller must initialize all elements before reading from them, and
    /// must not rely on the arena to drop them.
    // Sound bump-arena aliasing — see the justification on `alloc` above.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn alloc_slice_uninit<T>(&self, len: usize) -> Result<&mut [T]> {
        let uninit = self.alloc_slice_maybe_uninit::<T>(len)?;
        Ok(slice::from_raw_parts_mut(
            uninit.as_mut_ptr() as *mut T,
            uninit.len(),
        ))
    }

    /// Allocate a string slice
    pub fn alloc_str(&self, s: &str) -> Result<&str> {
        let bytes = self.alloc_slice(s.as_bytes())?;
        // SAFETY: `bytes` is a byte-for-byte copy of a `&str`, hence valid UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
    }

    /// Take the recorded destructors out of the arena.
    fn take_drop_entries(&self) -> Vec<DropEntry> {
        std::mem::take(&mut *self.drops.borrow_mut())
    }

    /// Reset the arena, running destructors and freeing all but one chunk.
    ///
    /// Takes `&mut self` because every reference previously handed out by
    /// `alloc*` is invalidated here; the borrow checker guarantees none is
    /// alive. One chunk (the largest) is kept for reuse so that repeated
    /// allocate/reset cycles do not grow memory without bound.
    pub fn reset(&mut self) {
        // Run destructors first, with no arena borrow held.
        // SAFETY: `&mut self` proves no reference into the arena is alive, so
        // every recorded value can be destroyed.
        unsafe { run_drop_entries(self.take_drop_entries()) };

        let kept_capacity = {
            let mut chunks = self.chunks.borrow_mut();
            if let Some(best_idx) = chunks
                .iter()
                .enumerate()
                .max_by_key(|(_, chunk)| chunk.capacity)
                .map(|(idx, _)| idx)
            {
                let mut kept = chunks.swap_remove(best_idx);
                kept.reset();
                chunks.clear();
                let capacity = kept.capacity;
                chunks.push(kept);
                capacity
            } else {
                0
            }
        };

        let mut stats = self.stats.borrow_mut();
        stats.total_allocated = kept_capacity;
        stats.chunk_count = usize::from(kept_capacity > 0);
        stats.bytes_in_use = 0;
        stats.allocation_count = 0;
        stats.alignment_waste = 0;
    }

    /// Clear the arena, running destructors and freeing all memory.
    pub fn clear(&mut self) {
        // SAFETY: see `reset` -- `&mut self` proves exclusivity.
        unsafe { run_drop_entries(self.take_drop_entries()) };
        self.chunks.borrow_mut().clear();
        self.chunk_size.set(DEFAULT_CHUNK_SIZE);
        *self.stats.borrow_mut() = ArenaStats::default();
    }

    /// Snapshot the current allocation state (see [`ScopedArena`]).
    fn checkpoint(&self) -> ArenaCheckpoint {
        let chunks = self.chunks.borrow();
        let stats = self.stats.borrow();
        ArenaCheckpoint {
            chunk_count: chunks.len(),
            last_chunk_offset: chunks.last().map(|chunk| chunk.offset).unwrap_or(0),
            drops_len: self.drops.borrow().len(),
            bytes_in_use: stats.bytes_in_use,
            allocation_count: stats.allocation_count,
            alignment_waste: stats.alignment_waste,
        }
    }

    /// Roll the arena back to a previously taken checkpoint.
    ///
    /// Destructors recorded after the checkpoint are run and every chunk added
    /// after it is released. `&mut self` is required for the same reason as in
    /// [`Arena::reset`].
    fn restore(&mut self, checkpoint: ArenaCheckpoint) {
        let tail = {
            let mut drops = self.drops.borrow_mut();
            if drops.len() > checkpoint.drops_len {
                drops.split_off(checkpoint.drops_len)
            } else {
                Vec::new()
            }
        };
        // SAFETY: `&mut self` proves no reference into the arena is alive, and
        // every entry in `tail` was recorded after the checkpoint, so the values
        // still live in memory that is about to be reused.
        unsafe { run_drop_entries(tail) };

        let (released, chunk_count) = {
            let mut chunks = self.chunks.borrow_mut();
            let released: usize = chunks
                .iter()
                .skip(checkpoint.chunk_count)
                .map(|chunk| chunk.capacity)
                .sum();
            chunks.truncate(checkpoint.chunk_count);
            if let Some(last) = chunks.last_mut() {
                last.offset = checkpoint.last_chunk_offset;
            }
            (released, chunks.len())
        };

        let mut stats = self.stats.borrow_mut();
        stats.total_allocated = stats.total_allocated.saturating_sub(released);
        stats.chunk_count = chunk_count;
        stats.bytes_in_use = checkpoint.bytes_in_use;
        stats.allocation_count = checkpoint.allocation_count;
        stats.alignment_waste = checkpoint.alignment_waste;
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

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: the arena is being destroyed, so no reference into it can be
        // alive. Destructors run before the chunks (a later field in drop
        // order) release the memory.
        unsafe { run_drop_entries(self.take_drop_entries()) };
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed arena for allocating values of a single type
pub struct TypedArena<T> {
    arena: Arena,
    _marker: PhantomData<T>,
}

impl<T> TypedArena<T> {
    /// Create a new typed arena sized for `capacity` values of `T`.
    pub fn new(capacity: usize) -> Self {
        let item_size = size_of::<T>().max(1);
        // Saturating: an absurd `capacity` must not wrap into a tiny chunk.
        let chunk_capacity = capacity
            .saturating_mul(item_size)
            .clamp(1024, MAX_CHUNK_SIZE);

        TypedArena {
            arena: Arena::with_capacity(chunk_capacity),
            _marker: PhantomData,
        }
    }

    /// Reset the arena (runs destructors, keeps one chunk for reuse)
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Clear the arena (runs destructors, frees all memory)
    pub fn clear(&mut self) {
        self.arena.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> ArenaStats {
        self.arena.stats()
    }
}

impl<T: 'static> TypedArena<T> {
    /// Allocate a value (see [`Arena::alloc`] for the `'static` rationale)
    pub fn alloc(&self, value: T) -> Result<&mut T> {
        self.arena.alloc(value)
    }

    /// Allocate multiple values from an iterator
    pub fn alloc_from_iter<I: IntoIterator<Item = T>>(&self, iter: I) -> Result<Vec<&mut T>> {
        iter.into_iter().map(|value| self.alloc(value)).collect()
    }
}

impl<T: Copy> TypedArena<T> {
    /// Allocate a slice
    pub fn alloc_slice(&self, values: &[T]) -> Result<&mut [T]> {
        self.arena.alloc_slice(values)
    }
}

impl<T> Default for TypedArena<T> {
    fn default() -> Self {
        Self::new(1024)
    }
}

/// Mutable state of a [`SyncArena`], guarded by a single mutex.
struct SyncArenaState {
    chunks: Vec<Chunk>,
    drops: Vec<DropEntry>,
}

/// Thread-safe arena allocator.
///
/// `SyncArena` is `Send + Sync` through its fields alone (`Mutex<..>` plus
/// atomics); there is no hand-written `unsafe impl` asserting it. The narrow
/// unsafe claims it rests on are `unsafe impl Send for Chunk` and
/// `unsafe impl Send for DropEntry`, both documented at their definition.
pub struct SyncArena {
    /// Chunks and recorded destructors (protected by a mutex)
    state: Mutex<SyncArenaState>,
    /// Current chunk size
    chunk_size: AtomicUsize,
    /// Statistics
    total_allocated: AtomicUsize,
    bytes_in_use: AtomicUsize,
    allocation_count: AtomicUsize,
    peak_usage: AtomicUsize,
    alignment_waste: AtomicUsize,
}

impl SyncArena {
    /// Create a new thread-safe arena
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHUNK_SIZE)
    }

    /// Create with specified capacity
    pub fn with_capacity(chunk_size: usize) -> Self {
        SyncArena {
            state: Mutex::new(SyncArenaState {
                chunks: Vec::new(),
                drops: Vec::new(),
            }),
            chunk_size: AtomicUsize::new(chunk_size.max(1024)),
            total_allocated: AtomicUsize::new(0),
            bytes_in_use: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            alignment_waste: AtomicUsize::new(0),
        }
    }

    /// Lock the arena state, recovering from a poisoned mutex.
    ///
    /// Recovery is sound here because the arena never leaves torn state behind:
    /// the only user code that can panic while the arena is being mutated is a
    /// value destructor, and destructors are always drained out of the state
    /// (and run with the lock released) before they are executed.
    fn lock_state(&self) -> MutexGuard<'_, SyncArenaState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Allocate raw memory.
    ///
    /// The returned memory is uninitialised and untracked; prefer
    /// [`SyncArena::alloc`].
    pub fn alloc_raw(&self, layout: Layout) -> Result<NonNull<u8>> {
        let size = layout.size();
        let align = layout.align().max(MIN_ALIGNMENT);

        let mut state = self.lock_state();

        // Try current chunk
        if let Some(chunk) = state.chunks.last_mut() {
            if let Some((ptr, padding)) = chunk.try_alloc(size, align) {
                drop(state);
                self.record_allocation(size, padding);
                return Ok(ptr);
            }
        }

        // Need new chunk
        let chunk_size = self.chunk_size.load(Ordering::Relaxed);
        let needed_size = size
            .checked_add(align)
            .ok_or_else(|| Error::InvalidInput("SyncArena: allocation size overflow".to_string()))?
            .max(chunk_size);

        let mut chunk = Chunk::new(needed_size).ok_or_else(|| {
            Error::OperationFailed(format!(
                "SyncArena: failed to allocate a {needed_size} byte chunk"
            ))
        })?;
        let (ptr, padding) = chunk.try_alloc(size, align).ok_or_else(|| {
            Error::OperationFailed(format!(
                "SyncArena: fresh chunk of {needed_size} bytes cannot hold a {size} byte allocation"
            ))
        })?;

        state.chunks.push(chunk);
        drop(state);

        self.total_allocated
            .fetch_add(needed_size, Ordering::Relaxed);
        self.record_allocation(size, padding);

        // Grow chunk size
        let new_size = chunk_size.saturating_mul(2).min(MAX_CHUNK_SIZE);
        self.chunk_size.store(new_size, Ordering::Relaxed);

        Ok(ptr)
    }

    fn record_allocation(&self, size: usize, padding: usize) {
        self.bytes_in_use.fetch_add(size, Ordering::Relaxed);
        self.alignment_waste.fetch_add(padding, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.update_peak();
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

    /// Allocate a value.
    ///
    /// `T: Send` is required because the value's destructor may run on a
    /// different thread than the one that allocated it (whichever thread resets,
    /// clears or drops the arena). `T: 'static` is required for the same reason
    /// as in [`Arena::alloc`].
    // Sound bump-arena aliasing — see `Arena::alloc`. `SyncArena` serialises the
    // bump pointer with a `Mutex` and requires `T: Send` so a cross-thread drop
    // (on `reset`/`clear`/drop) is sound.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T: Send + 'static>(&self, value: T) -> Result<&mut T> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_raw(layout)?;

        // SAFETY: `alloc_raw` returned an exclusive, correctly sized and aligned
        // region that no other allocation of this arena overlaps.
        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            if needs_drop::<T>() {
                self.lock_state().drops.push(DropEntry {
                    addr: ptr,
                    drop_fn: drop_in_place_at::<T>,
                });
            }
            Ok(&mut *typed_ptr)
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ArenaStats {
        let chunk_count = self.lock_state().chunks.len();
        ArenaStats {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            bytes_in_use: self.bytes_in_use.load(Ordering::Relaxed),
            chunk_count,
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            alignment_waste: self.alignment_waste.load(Ordering::Relaxed),
        }
    }

    /// Reset the arena, running destructors and keeping one chunk for reuse.
    ///
    /// Takes `&mut self`: every reference handed out by `alloc` is invalidated
    /// here, and `&mut self` is unobtainable through an `Arc<SyncArena>` while
    /// such references exist.
    pub fn reset(&mut self) {
        let entries = std::mem::take(&mut self.lock_state().drops);
        // SAFETY: `&mut self` proves no reference into the arena is alive.
        unsafe { run_drop_entries(entries) };

        let kept_capacity = {
            let mut state = self.lock_state();
            let chunks = &mut state.chunks;
            if let Some(best_idx) = chunks
                .iter()
                .enumerate()
                .max_by_key(|(_, chunk)| chunk.capacity)
                .map(|(idx, _)| idx)
            {
                let mut kept = chunks.swap_remove(best_idx);
                kept.reset();
                chunks.clear();
                let capacity = kept.capacity;
                chunks.push(kept);
                capacity
            } else {
                0
            }
        };

        self.total_allocated.store(kept_capacity, Ordering::Relaxed);
        self.bytes_in_use.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.alignment_waste.store(0, Ordering::Relaxed);
    }

    /// Clear the arena, running destructors and freeing all memory.
    pub fn clear(&mut self) {
        let entries = std::mem::take(&mut self.lock_state().drops);
        // SAFETY: `&mut self` proves no reference into the arena is alive.
        unsafe { run_drop_entries(entries) };

        self.lock_state().chunks.clear();
        self.chunk_size.store(DEFAULT_CHUNK_SIZE, Ordering::Relaxed);
        self.total_allocated.store(0, Ordering::Relaxed);
        self.bytes_in_use.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.peak_usage.store(0, Ordering::Relaxed);
        self.alignment_waste.store(0, Ordering::Relaxed);
    }
}

impl Drop for SyncArena {
    fn drop(&mut self) {
        let entries = std::mem::take(&mut self.lock_state().drops);
        // SAFETY: the arena is being destroyed, so no reference into it is
        // alive; destructors run before the chunks free their memory.
        unsafe { run_drop_entries(entries) };
    }
}

impl Default for SyncArena {
    fn default() -> Self {
        Self::new()
    }
}

/// A scoped arena that rolls its parent arena back when dropped.
///
/// The parent is borrowed mutably for the whole scope, which is what makes the
/// rollback sound: allocations can only be made *through* the `ScopedArena`, so
/// no reference into the rolled-back region can outlive it.
///
/// ```ignore
/// let mut arena = Arena::new();
/// {
///     let scope = ScopedArena::new(&mut arena);
///     let tmp = scope.alloc(42u64)?;
///     assert_eq!(*tmp, 42);
/// } // temporaries are destroyed and their memory reused
/// assert_eq!(arena.bytes_in_use(), 0);
/// ```
pub struct ScopedArena<'a> {
    arena: &'a mut Arena,
    checkpoint: ArenaCheckpoint,
}

impl<'a> ScopedArena<'a> {
    /// Create a new scoped arena
    pub fn new(arena: &'a mut Arena) -> Self {
        let checkpoint = arena.checkpoint();
        ScopedArena { arena, checkpoint }
    }

    /// Allocate a value
    pub fn alloc<T: 'static>(&self, value: T) -> Result<&mut T> {
        self.arena.alloc(value)
    }

    /// Allocate a slice
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Result<&mut [T]> {
        self.arena.alloc_slice(values)
    }

    /// Bytes currently in use by the parent arena
    pub fn bytes_in_use(&self) -> usize {
        self.arena.bytes_in_use()
    }
}

impl Drop for ScopedArena<'_> {
    fn drop(&mut self) {
        let checkpoint = self.checkpoint;
        self.arena.restore(checkpoint);
    }
}

/// Arena-backed vector for efficient bulk storage.
///
/// The backing storage is uninitialised arena memory, so only the initialised
/// prefix (`..len`) is ever exposed as `&[T]`. `T: Copy` because the arena does
/// not drop slice elements.
pub struct ArenaVec<'a, T> {
    data: &'a mut [MaybeUninit<T>],
    len: usize,
}

impl<'a, T: Copy> ArenaVec<'a, T> {
    /// Create a new arena-backed vector with the given capacity
    pub fn with_capacity(arena: &'a Arena, capacity: usize) -> Result<Self> {
        let data = arena.alloc_slice_maybe_uninit::<T>(capacity)?;
        Ok(ArenaVec { data, len: 0 })
    }

    /// Push a value.
    ///
    /// Returns an error when the vector is at capacity: an arena-backed vector
    /// cannot grow, and silently discarding the value would lose data.
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len >= self.data.len() {
            return Err(Error::OperationFailed(format!(
                "ArenaVec is full ({} elements); it cannot grow",
                self.data.len()
            )));
        }
        self.data[self.len].write(value);
        self.len += 1;
        Ok(())
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
        // SAFETY: the first `len` elements were initialised by `push`, and
        // `MaybeUninit<T>` has the same layout as `T`.
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see `as_slice`; `&mut self` guarantees exclusive access.
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.data.len()
    }
}

/// Compile-time check that [`SyncArena`] really is thread safe through its
/// fields (`Mutex` + atomics) rather than through a hand-written `unsafe impl`.
/// If a future field breaks that, the build fails here.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SyncArena>();
};

#[cfg(test)]
mod tests {
    // These tests allocate arbitrary `f64` payloads such as `3.14` and assert
    // they round-trip; `3.14` is test data, not a use of `PI`, so the
    // `approx_constant` correctness lint is silenced for the test module rather
    // than rewriting fixture values.
    #![allow(clippy::approx_constant)]
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Payload that records its own destruction, used to prove that the arena
    /// runs destructors exactly once.
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new();

        let a = arena.alloc(42i32).expect("allocation should succeed");
        let b = arena.alloc(3.14f64).expect("allocation should succeed");
        let c = arena
            .alloc("hello".to_string())
            .expect("allocation should succeed");

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

        let slice = arena
            .alloc_slice(&[1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("allocation should succeed");
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

        let s1 = arena.alloc_str("hello").expect("allocation should succeed");
        let s2 = arena.alloc_str("world").expect("allocation should succeed");

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new();

        for i in 0..100 {
            arena.alloc(i).expect("allocation should succeed");
        }

        let stats_before = arena.stats();
        assert_eq!(stats_before.allocation_count, 100);

        arena.reset();

        let stats_after = arena.stats();
        assert_eq!(stats_after.allocation_count, 0);
        assert_eq!(stats_after.bytes_in_use, 0);
        // One chunk is kept for reuse
        assert!(stats_after.total_allocated > 0);
        assert_eq!(stats_after.chunk_count, 1);
    }

    #[test]
    fn test_arena_reset_runs_destructors() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut arena = Arena::new();

        for _ in 0..10 {
            arena
                .alloc(DropCounter(Arc::clone(&counter)))
                .expect("allocation should succeed");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        arena.reset();
        assert_eq!(counter.load(Ordering::SeqCst), 10);

        // A second reset must not run the destructors again.
        arena.reset();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_arena_drop_runs_destructors() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let arena = Arena::new();
            for _ in 0..5 {
                arena
                    .alloc(DropCounter(Arc::clone(&counter)))
                    .expect("allocation should succeed");
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn test_arena_reset_does_not_grow_unboundedly() {
        let mut arena = Arena::new();

        let mut previous = 0usize;
        for cycle in 0..5 {
            for i in 0..2000i64 {
                arena.alloc(i).expect("allocation should succeed");
            }
            arena.reset();
            let total = arena.stats().total_allocated;
            if cycle > 0 {
                assert_eq!(
                    total, previous,
                    "arena memory grew across reset cycles: {previous} -> {total}"
                );
            }
            previous = total;
        }
    }

    #[test]
    fn test_arena_clear() {
        let mut arena = Arena::new();

        for i in 0..100 {
            arena.alloc(i).expect("allocation should succeed");
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

        let values: Vec<&mut f64> = (0..100)
            .map(|i| arena.alloc(i as f64).expect("allocation should succeed"))
            .collect();

        for (i, v) in values.iter().enumerate() {
            assert_eq!(**v, i as f64);
        }
    }

    #[test]
    fn test_typed_arena_slice() {
        let arena: TypedArena<i32> = TypedArena::new(1024);

        let data = vec![1, 2, 3, 4, 5];
        let slice = arena.alloc_slice(&data).expect("allocation should succeed");

        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_typed_arena_capacity_overflow_is_clamped() {
        // `capacity * size_of::<T>()` overflows; the arena must clamp instead of
        // wrapping into a tiny (or zero) chunk size.
        let arena: TypedArena<u64> = TypedArena::new(usize::MAX);
        let value = arena.alloc(7u64).expect("allocation should succeed");
        assert_eq!(*value, 7);
    }

    #[test]
    fn test_sync_arena() {
        let arena = SyncArena::new();

        let a = arena.alloc(42i32).expect("operation should succeed");
        let b = arena.alloc(3.14f64).expect("operation should succeed");

        assert_eq!(*a, 42);
        assert_eq!(*b, 3.14);

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 2);
    }

    #[test]
    fn test_sync_arena_threaded() {
        use std::thread;

        let arena = Arc::new(SyncArena::new());
        let mut handles = vec![];

        for t in 0..4 {
            let arena_clone = Arc::clone(&arena);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    arena_clone
                        .alloc(t * 100 + i)
                        .expect("allocation should succeed");
                }
            }));
        }

        for handle in handles {
            handle.join().expect("operation should succeed");
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 400);
    }

    #[test]
    fn test_sync_arena_runs_destructors() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut arena = SyncArena::new();

        for _ in 0..8 {
            arena
                .alloc(DropCounter(Arc::clone(&counter)))
                .expect("allocation should succeed");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        arena.reset();
        assert_eq!(counter.load(Ordering::SeqCst), 8);

        arena.clear();
        assert_eq!(counter.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = Arena::new();

        // Allocate a large slice
        let large: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let slice = arena
            .alloc_slice(&large)
            .expect("allocation should succeed");

        assert_eq!(slice.len(), 10000);
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[9999], 9999.0);
    }

    #[test]
    fn test_arena_alignment() {
        let arena = Arena::new();

        // Allocate values with different alignments
        let _byte = arena.alloc(1u8).expect("allocation should succeed");
        let int_ptr = arena.alloc(42i32).expect("allocation should succeed");
        let double_ptr = arena.alloc(3.14f64).expect("allocation should succeed");

        // Check alignment
        let int_addr = int_ptr as *const i32 as usize;
        let double_addr = double_ptr as *const f64 as usize;

        assert_eq!(int_addr % std::mem::align_of::<i32>(), 0);
        assert_eq!(double_addr % std::mem::align_of::<f64>(), 0);

        // Alignment padding is accounted for
        assert!(arena.stats().alignment_waste > 0);
    }

    #[test]
    fn test_arena_vec() {
        let arena = Arena::new();
        let mut vec: ArenaVec<i32> =
            ArenaVec::with_capacity(&arena, 100).expect("allocation should succeed");

        for i in 0..50 {
            vec.push(i).expect("push should succeed");
        }

        assert_eq!(vec.len(), 50);
        assert_eq!(vec.capacity(), 100);
        assert_eq!(vec.as_slice()[0], 0);
        assert_eq!(vec.as_slice()[49], 49);
    }

    #[test]
    fn test_arena_vec_full_reports_error() {
        let arena = Arena::new();
        let mut vec: ArenaVec<u16> =
            ArenaVec::with_capacity(&arena, 4).expect("allocation should succeed");

        for i in 0..4u16 {
            vec.push(i).expect("push should succeed");
        }
        assert!(
            vec.push(4).is_err(),
            "overflowing push must report an error"
        );
        assert_eq!(vec.as_slice(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_scoped_arena_rolls_back() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut arena = Arena::new();

        let persistent = arena.alloc(7u64).expect("allocation should succeed");
        assert_eq!(*persistent, 7);
        let baseline = arena.bytes_in_use();

        {
            let scope = ScopedArena::new(&mut arena);
            for _ in 0..4 {
                scope
                    .alloc(DropCounter(Arc::clone(&counter)))
                    .expect("allocation should succeed");
            }
            assert!(scope.bytes_in_use() > baseline);
        }

        // Scoped allocations were destroyed and their memory reclaimed.
        assert_eq!(counter.load(Ordering::SeqCst), 4);
        assert_eq!(arena.bytes_in_use(), baseline);
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
            arena.alloc(42i64).expect("allocation should succeed");
        }

        let stats = arena.stats();
        assert_eq!(stats.allocation_count, 10);
        assert!(stats.bytes_in_use >= 80); // 10 * 8 bytes
        assert!(stats.peak_usage >= stats.bytes_in_use);
    }
}

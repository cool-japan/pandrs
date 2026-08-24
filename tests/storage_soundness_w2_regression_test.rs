//! Wave 2 regression tests for `src/storage/arena.rs` and
//! `src/storage/zero_copy.rs`.
//!
//! Every test here pins a soundness contract that was previously violated from
//! safe code:
//!
//! * arena values were never dropped (leaks), `reset`/`clear` took `&self`
//!   while `alloc` handed out `&mut T` (aliasing UB + use-after-free),
//!   `reset` only reused the last chunk (unbounded growth), `ArenaVec` was
//!   built over uninitialised memory and silently discarded pushes,
//!   `ScopedArena` documented an auto-reset that never happened;
//! * zero-copy views did not keep their pool alive (use-after-free), the pool
//!   had no deallocation path at all (permanent exhaustion), block starts were
//!   never aligned, `MemoryMappedView::len()` disagreed with the clamped
//!   `as_slice()`, and `blocked_operation` materialised the full cross product.
//!
//! `expect` is used instead of `?` throughout: `pandrs::core::error::Error` is
//! a large enum and this file is compiled without the crate-level clippy
//! allowances.

use pandrs::storage::arena::{Arena, ArenaVec, ScopedArena, SyncArena, TypedArena};
use pandrs::storage::zero_copy::{
    CacheAwareAllocator, CacheAwareOps, CacheLevel, CacheTopology, MemoryMappedView, MemoryPool,
    ZeroCopyManager, ZeroCopyView, CACHE_LINE_SIZE,
};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Payload that counts its own destruction.
struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// arena.rs
// ---------------------------------------------------------------------------

#[test]
fn arena_runs_destructors_on_reset_clear_and_drop() {
    let counter = Arc::new(AtomicUsize::new(0));

    {
        let mut arena = Arena::new();

        for _ in 0..16 {
            arena
                .alloc(DropCounter(Arc::clone(&counter)))
                .expect("allocation should succeed");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // reset() runs every recorded destructor exactly once...
        arena.reset();
        assert_eq!(counter.load(Ordering::SeqCst), 16);
        arena.reset();
        assert_eq!(counter.load(Ordering::SeqCst), 16);

        // ...and so does clear().
        for _ in 0..4 {
            arena
                .alloc(DropCounter(Arc::clone(&counter)))
                .expect("allocation should succeed");
        }
        arena.clear();
        assert_eq!(counter.load(Ordering::SeqCst), 20);

        // Values still live when the arena is dropped are destroyed too.
        for _ in 0..3 {
            arena
                .alloc(DropCounter(Arc::clone(&counter)))
                .expect("allocation should succeed");
        }
        assert_eq!(counter.load(Ordering::SeqCst), 20);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 23);
}

#[test]
fn arena_owning_values_are_not_leaked() {
    // `String` allocations used to leak their heap buffer forever.
    let arena = Arena::new();
    let mut addresses = Vec::new();
    for i in 0..64 {
        let value = arena
            .alloc(format!("value-{i}-{}", "x".repeat(64)))
            .expect("allocation should succeed");
        addresses.push(value.as_ptr() as usize);
    }
    assert_eq!(addresses.len(), 64);
    // Dropping the arena must free every String buffer; miri/leak checkers see
    // this, and at minimum the destructor bookkeeping must not panic.
    drop(arena);
}

#[test]
fn arena_reset_does_not_grow_across_cycles() {
    let mut arena = Arena::new();
    let mut previous = None;

    for _ in 0..8 {
        for i in 0..4096i64 {
            arena.alloc(i).expect("allocation should succeed");
        }
        arena.reset();
        let total = arena.stats().total_allocated;
        if let Some(previous) = previous {
            assert_eq!(
                total, previous,
                "arena retained memory grew across reset cycles"
            );
        }
        previous = Some(total);
        assert_eq!(arena.stats().chunk_count, 1, "reset must keep one chunk");
        assert_eq!(arena.stats().bytes_in_use, 0);
    }
}

#[test]
fn arena_vec_reports_overflow_and_only_exposes_initialised_prefix() {
    let arena = Arena::new();
    let mut vec: ArenaVec<u32> =
        ArenaVec::with_capacity(&arena, 8).expect("allocation should succeed");

    assert!(vec.is_empty());
    assert_eq!(vec.capacity(), 8);
    assert!(
        vec.as_slice().is_empty(),
        "uninitialised memory must not be exposed"
    );

    for i in 0..8u32 {
        vec.push(i).expect("push should succeed");
        assert_eq!(vec.as_slice().len(), (i + 1) as usize);
    }

    // Previously the value was silently dropped on the floor.
    assert!(vec.push(99).is_err());
    assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn scoped_arena_rolls_back_and_destroys_temporaries() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut arena = Arena::new();

    let kept: u64 = *arena.alloc(1234u64).expect("allocation should succeed");
    assert_eq!(kept, 1234);
    let baseline = arena.bytes_in_use();

    for _ in 0..3 {
        {
            let scope = ScopedArena::new(&mut arena);
            for _ in 0..32 {
                scope
                    .alloc(DropCounter(Arc::clone(&counter)))
                    .expect("allocation should succeed");
            }
            let slice = scope
                .alloc_slice(&[1.0f64, 2.0, 3.0])
                .expect("allocation should succeed");
            assert_eq!(slice.len(), 3);
            assert!(scope.bytes_in_use() > baseline);
        }
        // Rolled back: memory reclaimed, destructors run.
        assert_eq!(arena.bytes_in_use(), baseline);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 96);
}

#[test]
fn typed_arena_clamps_capacity_overflow() {
    // `capacity * size_of::<T>()` used to wrap silently.
    let arena: TypedArena<u128> = TypedArena::new(usize::MAX);
    let value = arena.alloc(u128::MAX).expect("allocation should succeed");
    assert_eq!(*value, u128::MAX);
    assert!(arena.stats().total_allocated > 0);
}

#[test]
fn sync_arena_allocates_across_threads_and_runs_destructors() {
    let counter = Arc::new(AtomicUsize::new(0));
    let arena = Arc::new(SyncArena::new());

    let mut handles = Vec::new();
    for thread_id in 0..4usize {
        let arena = Arc::clone(&arena);
        handles.push(std::thread::spawn(move || {
            for i in 0..256usize {
                let value = arena
                    .alloc(thread_id * 1000 + i)
                    .expect("allocation should succeed");
                assert_eq!(*value, thread_id * 1000 + i);
            }
        }));
    }
    for handle in handles {
        handle.join().expect("thread should not panic");
    }
    assert_eq!(arena.stats().allocation_count, 4 * 256);

    // Values with destructors are cleaned up on reset.
    let mut arena = Arc::try_unwrap(arena).unwrap_or_else(|_| {
        panic!("all worker threads finished, so the arena must be uniquely owned")
    });
    for _ in 0..10 {
        arena
            .alloc(DropCounter(Arc::clone(&counter)))
            .expect("allocation should succeed");
    }
    arena.reset();
    assert_eq!(counter.load(Ordering::SeqCst), 10);
    assert_eq!(arena.stats().bytes_in_use, 0);
    assert!(arena.stats().total_allocated > 0, "one chunk is kept");
}

// ---------------------------------------------------------------------------
// zero_copy.rs
// ---------------------------------------------------------------------------

#[test]
fn memory_pool_returns_and_coalesces_blocks() {
    let pool = MemoryPool::new(256 * 1024).expect("pool creation should succeed");
    assert_eq!(pool.free_bytes(), pool.size());
    assert_eq!(pool.free_region_count(), 1);

    // Interleaved allocations of varying sizes, released in a different order.
    for _ in 0..64 {
        let a = pool
            .allocate_aligned(64 * 1024, CACHE_LINE_SIZE)
            .expect("allocation should succeed");
        let b = pool
            .allocate_aligned(96 * 1024, CACHE_LINE_SIZE)
            .expect("allocation should succeed");
        let c = pool
            .allocate_aligned(32 * 1024, CACHE_LINE_SIZE)
            .expect("allocation should succeed");
        assert_eq!(a.address() % CACHE_LINE_SIZE, 0);
        assert_eq!(b.address() % CACHE_LINE_SIZE, 0);
        assert_eq!(c.address() % CACHE_LINE_SIZE, 0);
        assert_eq!(pool.bytes_in_use(), 192 * 1024);
        drop(b);
        drop(a);
        drop(c);
    }

    // Everything came back and merged into a single region.
    assert_eq!(pool.free_bytes(), pool.size());
    assert_eq!(pool.free_region_count(), 1);
    assert_eq!(pool.bytes_in_use(), 0);
}

#[test]
fn memory_pool_rejects_degenerate_requests() {
    assert!(
        MemoryPool::new(0).is_err(),
        "a zero-sized pool would allocate a zero-size layout"
    );

    let pool = MemoryPool::new(8192).expect("pool creation should succeed");
    assert!(pool.allocate_aligned(0, 64).is_err());
    assert!(
        pool.allocate_aligned(64, 48).is_err(),
        "alignment must be a power of two"
    );
    assert!(pool.allocate_aligned(1024 * 1024, 64).is_err());
    assert_eq!(pool.free_bytes(), pool.size());
}

#[test]
fn memory_pool_aligns_block_starts() {
    let pool = MemoryPool::new(1024 * 1024).expect("pool creation should succeed");

    // A one byte allocation deliberately misaligns the free region start.
    let skew = pool
        .allocate_aligned(1, 8)
        .expect("allocation should succeed");
    let mut allocations = vec![skew];
    for alignment in [64usize, 128, 512, 4096] {
        let allocation = pool
            .allocate_aligned(4096, alignment)
            .expect("allocation should succeed");
        assert_eq!(
            allocation.address() % alignment,
            0,
            "block start must honour the requested alignment"
        );
        allocations.push(allocation);
    }

    drop(allocations);
    assert_eq!(pool.free_bytes(), pool.size());
    assert_eq!(pool.free_region_count(), 1);
}

#[test]
fn zero_copy_view_keeps_its_pool_alive() {
    // Previously the view held a `Copy` block descriptor: dropping the manager
    // freed the pool memory and `as_slice()` was a use-after-free.
    let view: ZeroCopyView<u64> = {
        let manager = ZeroCopyManager::new().expect("manager creation should succeed");
        let view = manager
            .create_view((0..512u64).collect::<Vec<_>>())
            .expect("view creation should succeed");
        drop(manager);
        view
    };

    assert_eq!(view.len(), 512);
    assert_eq!(view.as_slice()[0], 0);
    assert_eq!(view.as_slice()[511], 511);
    assert!(view.backing_bytes() >= 512 * std::mem::size_of::<u64>());
}

#[test]
fn subview_keeps_parent_storage_alive() {
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");
    let view = manager
        .create_view((0..1024i32).collect::<Vec<_>>())
        .expect("view creation should succeed");

    let subview = view.subview(512..768).expect("subview should succeed");
    let nested = subview
        .subview(0..4)
        .expect("nested subview should succeed");
    drop(view);
    drop(subview);

    assert_eq!(nested.as_slice(), &[512, 513, 514, 515]);
    assert!(manager.bytes_in_use().expect("stats") > 0);

    drop(nested);
    assert_eq!(
        manager.bytes_in_use().expect("stats"),
        0,
        "the block must return to the pool once every view is dropped"
    );
}

#[test]
fn views_are_recycled_instead_of_exhausting_the_pool() {
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");

    // 200 x 512 KiB through a 4 MiB pool: without a deallocation path this
    // failed with "Not enough memory in pool" after a handful of iterations.
    for round in 0..200usize {
        let data = vec![round as f64; 64 * 1024];
        let view = manager
            .create_view(data)
            .unwrap_or_else(|e| panic!("view {round} should be allocatable: {e}"));
        assert_eq!(view.len(), 64 * 1024);
        assert_eq!(view.as_slice()[0], round as f64);
    }

    assert_eq!(manager.bytes_in_use().expect("stats"), 0);
    let stats = manager.stats().expect("stats");
    assert_eq!(stats.views_created, 200);
}

#[test]
fn mixed_size_views_do_not_strand_pool_space() {
    // Differently sized views (different pools, different head-padding splits)
    // released in a non-monotonic order must leave every pool fully coalesced,
    // otherwise repeated rounds would strand space and eventually fail.
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");

    for round in 0..25usize {
        let small = manager
            .create_view(vec![round as u8; 1024])
            .expect("small view should be allocatable");
        let large = manager
            .create_view(vec![round as f64; 64 * 1024])
            .expect("large view should be allocatable");
        let odd = manager
            .create_view(vec![round as u16; 3])
            .expect("odd view should be allocatable");
        let medium = manager
            .create_view(vec![round as u32; 8 * 1024])
            .expect("medium view should be allocatable");

        assert_eq!(small.as_slice()[0], round as u8);
        assert_eq!(large.as_slice()[64 * 1024 - 1], round as f64);
        assert_eq!(odd.as_slice(), &[round as u16; 3]);
        assert_eq!(medium.as_slice()[0], round as u32);
        assert!(manager.bytes_in_use().expect("stats") > 0);

        drop(large);
        drop(odd);
        drop(small);
        drop(medium);

        assert_eq!(
            manager.bytes_in_use().expect("stats"),
            0,
            "round {round} stranded pool space"
        );
    }
}

#[test]
fn allocator_reports_live_usage_and_zeroed_memory() {
    let mut allocator = CacheAwareAllocator::new().expect("allocator creation should succeed");
    assert_eq!(allocator.bytes_in_use(), 0);

    let view: ZeroCopyView<i64> = allocator
        .allocate_aligned(128, CacheLevel::L2)
        .expect("allocation should succeed");
    assert_eq!(view.as_slice(), &[0i64; 128]);
    assert!(view.is_cache_aligned());

    let live = allocator.bytes_in_use();
    assert!(live >= 128 * 8);
    assert_eq!(allocator.stats().current_usage, live);

    drop(view);
    assert_eq!(allocator.bytes_in_use(), 0);
    assert_eq!(allocator.stats().current_usage, 0);
    assert!(allocator.stats().peak_usage >= live);
}

#[test]
fn empty_view_is_valid() {
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");
    let view = manager
        .create_view(Vec::<f32>::new())
        .expect("empty view creation should succeed");
    assert!(view.is_empty());
    assert!(view.as_slice().is_empty());
    assert_eq!(view.backing_bytes(), 0);
    assert_eq!(manager.bytes_in_use().expect("stats"), 0);
}

#[test]
fn blocked_operation_is_blocked_and_handles_zero_block_size() {
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");
    let view = manager
        .create_view((0..1000i64).collect::<Vec<_>>())
        .expect("view creation should succeed");
    let other = vec![3i64; 1000];
    let expected: Vec<i64> = (0..1000i64).map(|v| v * 3).collect();

    // A zero block size must not panic (`chunks(0)` did) and must still work.
    assert_eq!(view.blocked_operation(&other, 0, |a, b| a * b), expected);
    assert_eq!(view.blocked_operation(&other, 7, |a, b| a * b), expected);
    assert_eq!(
        view.blocked_operation(&other, 100_000, |a, b| a * b),
        expected
    );

    // Shorter right-hand side truncates instead of panicking.
    let short = vec![2i64; 10];
    let truncated = view.blocked_operation(&short, 4, |a, b| a * b);
    assert_eq!(truncated.len(), 10);
    assert_eq!(truncated[9], 18);
}

#[test]
fn prefetch_is_bounds_checked() {
    let manager = ZeroCopyManager::new().expect("manager creation should succeed");
    let view = manager
        .create_view((0..256u32).collect::<Vec<_>>())
        .expect("view creation should succeed");

    // Out-of-range indices must be ignored, never prefetched out of bounds.
    view.prefetch(&[0, 1, 255, 256, usize::MAX]);
    assert_eq!(view.as_slice()[255], 255);
}

#[test]
fn memory_mapped_view_validates_length_against_the_mapping() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "pandrs_storage_soundness_w2_{}.bin",
        std::process::id()
    ));

    {
        let mut file = std::fs::File::create(&path).expect("temp file creation should succeed");
        for value in 0..32u64 {
            file.write_all(&value.to_le_bytes())
                .expect("write should succeed");
        }
        file.sync_all().expect("sync should succeed");
    }

    // Asking for more elements than the file holds is an error, not a silently
    // clamped slice whose length disagrees with `len()`.
    let file = std::fs::File::open(&path).expect("open should succeed");
    assert!(MemoryMappedView::<u64>::from_file(file, 33).is_err());

    let file = std::fs::File::open(&path).expect("open should succeed");
    let view = MemoryMappedView::<u64>::from_file(file, 32).expect("mapping should succeed");
    assert_eq!(view.len(), 32);
    assert_eq!(view.as_slice().len(), view.len());
    assert_eq!(view.as_slice()[31], 31);
    assert_eq!(view.mapped_bytes(), 32 * 8);

    let manager = ZeroCopyManager::new().expect("manager creation should succeed");
    let mapped: MemoryMappedView<u64> = manager
        .create_mmap_view(path.to_str().expect("utf-8 path"), 16)
        .expect("mmap view creation should succeed");
    assert_eq!(mapped.len(), 16);
    assert_eq!(mapped.as_slice().len(), 16);

    drop(view);
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cache_topology_is_honest_about_probing() {
    let topology = CacheTopology::detect().expect("detection should succeed");
    assert!(topology.l1_cache_size > 0);
    assert!(topology.l2_cache_size > 0);
    assert!(topology.l3_cache_size > 0);
    assert!(topology.cache_line_size > 0);
    assert!(topology.cpu_cores > 0);

    let defaults = CacheTopology::defaults();
    assert!(!defaults.probed);
    if cfg!(target_os = "linux") {
        // On Linux sysfs is read; either it worked (probed) or the values are
        // exactly the documented defaults.
        assert!(topology.probed || topology.l1_cache_size == defaults.l1_cache_size);
    } else {
        assert!(
            !topology.probed,
            "only Linux probes the OS; other platforms must not claim otherwise"
        );
    }
}

//! Storage engines module.
//!
//! # Two type layers, on purpose
//!
//! PandRS exposes two related-but-distinct storage vocabularies:
//!
//! * [`traits`](crate::storage::traits) — the *engine* layer ([`StorageEngine`](crate::storage::StorageEngine)): concrete, byte-range
//!   addressed engines such as [`ColumnStore`] and [`disk::DiskStorage`](crate::storage::disk::DiskStorage).
//! * [`unified_memory`](crate::storage::unified_memory) — the *strategy* layer
//!   ([`unified_memory::StorageStrategy`](crate::storage::unified_memory::StorageStrategy)): pluggable, row-addressed strategies
//!   driven by [`unified_manager::UnifiedMemoryManager`](crate::storage::unified_manager::UnifiedMemoryManager).
//!
//! The two layers deliberately keep separate enums (the strategy layer has
//! richer variants — e.g. `Speed::VerySlow`, `DurabilityLevel::HighDurability`
//! — that the engine layer does not model), so they are re-exported side by
//! side with `Unified*` aliases rather than merged. Where a faithful mapping
//! exists, `From` conversions bridge them; see [`traits`](crate::storage::traits).
pub mod adaptive_string_pool;
pub mod arena;
pub mod checksum;
pub mod column_store;
pub mod disk;
pub mod hybrid_large_scale;
pub mod memory_mapped;
pub mod ml_strategy_selector;
pub mod simple_unified_string_pool;
pub mod string_pool;
pub mod traits;
pub mod unified_column_store;
pub mod unified_manager;
pub mod unified_memory;
pub mod zero_copy;

// Re-exports for storage engines
pub use column_store::ColumnStore;
pub use disk::DiskStorage;
pub use memory_mapped::MemoryMappedFile;
pub use string_pool::StringPool;

// Re-exports for unified storage system
pub use traits::{
    AccessPattern, CompressionPreference, DataChunk, DurabilityLevel, PerformancePriority,
    PerformanceProfile, StorageConfig, StorageEngine, StorageEngineId, StorageHandle,
    StorageHandleId, StorageRequirements, StorageStrategy, UnifiedStorageManager,
};

// Re-exports for unified memory management
pub use unified_memory::{
    AccessPattern as UnifiedAccessPattern, AtomicMemoryStats, ChunkLayout, ChunkRange,
    CompactionResult, CompressionPreference as UnifiedCompressionPreference, CompressionType,
    ConcurrencyLevel, DataCharacteristics, DataChunk as UnifiedDataChunk,
    DurabilityLevel as UnifiedDurabilityLevel, Efficiency, IoPattern, ParallelScalability,
    PerformancePriority as UnifiedPerformancePriority,
    PerformanceProfile as UnifiedPerformanceProfile, PerformanceTracker, QueryOptimization,
    ResourceCost, Speed, StorageConfig as UnifiedStorageConfig,
    StorageHandle as UnifiedStorageHandle, StorageId, StorageMetadata,
    StorageRequirements as UnifiedStorageRequirements, StorageStats,
    StorageStrategy as UnifiedStorageStrategy, StorageType, StrategyCapability,
};

// Re-exports for unified memory manager
pub use unified_manager::{
    CacheManager, DefaultStrategySelector, MemoryConfig, PerformanceMonitor, StrategySelection,
    StrategySelectionAlgorithm, StrategySelector, UnifiedMemoryManager,
};

// Re-exports for ML-based strategy selection
pub use ml_strategy_selector::{
    AdaptiveUnifiedMemoryManager, MLStrategySelector, ModelStats, PerformancePrediction,
    SharedMlSelector, TrainingExample, WorkloadFeatures,
};

// Re-exports for zero-copy operations
pub use zero_copy::{
    AllocationStats, CacheAwareAllocator, CacheAwareOps, CacheLevel, CacheTopology, MemoryLayout,
    MemoryMappedView, MemoryPool, ZeroCopyManager, ZeroCopyStats, ZeroCopyView, CACHE_LINE_SIZE,
    PAGE_SIZE,
};

// Re-exports for unified column store
pub use unified_column_store::{
    BlockId, BlockManager, ColumnDataType, ColumnLayout, ColumnStatistics, ColumnStoreConfig,
    ColumnStoreHandle, CompressedBlock, CompressionEngine, EncodingStrategy, EncodingType,
    PhysicalStorage, UnifiedColumnStoreStrategy,
};

// Re-exports for adaptive string pool
pub use adaptive_string_pool::{
    AdaptiveStringPoolStrategy, CompressionDictionary, PatternAnalysis, StringCharacteristics,
    StringCompressionAlgorithm, StringCompressionEngine, StringId, StringPatternAnalyzer,
    StringPoolConfig, StringPoolHandle, StringPoolStatistics, StringStorageStrategy,
};

// Re-exports for hybrid large scale strategy
pub use hybrid_large_scale::{
    AccessPattern as HybridAccessPattern, AccessPatternType, DataId, DataTier, HybridConfig,
    HybridHandle, HybridLargeScaleStrategy, HybridStatistics, TierBackend, TierConfig, TierManager,
    TierStorageInfo, TierStorageType, TieredDataEntry, TieringReport,
};

// NOTE: the old `unified_string_pool` module (never compiled, "temporarily
// disabled due to Send/Sync issues") was removed in 0.4.1. It wrote through a
// pointer derived from a shared reference, took its two locks in inconsistent
// order (the real cause of the Send/Sync problem) and truncated offsets to
// u32. `simple_unified_string_pool` below is the supported, compiled
// replacement. The equally-orphaned `intelligent_memory_mapped` simulation was
// removed at the same time; `memory_mapped` now provides real mmap access by
// delegating to `zero_copy::MemoryMappedView`.

// Re-exports for simplified unified zero-copy string pool
pub use simple_unified_string_pool::{
    SimpleStringPoolStats, SimpleStringView, SimpleUnifiedStringPool,
};

// Re-exports for arena allocator
pub use arena::{Arena, ArenaStats, ArenaVec, ScopedArena, SyncArena, TypedArena};

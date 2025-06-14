// Storage engines module
pub mod column_store;
pub mod disk;
pub mod memory_mapped;
pub mod string_pool;
pub mod traits;
pub mod unified_memory;
pub mod unified_manager;
pub mod unified_column_store;
pub mod ml_strategy_selector;
pub mod zero_copy;
pub mod adaptive_string_pool;
pub mod hybrid_large_scale;

// Re-exports for storage engines
pub use column_store::ColumnStore;
pub use disk::DiskStorage;
pub use memory_mapped::MemoryMappedFile;
pub use string_pool::StringPool;

// Re-exports for unified storage system
pub use traits::{
    StorageEngine, StorageStrategy, UnifiedStorageManager,
    StorageConfig, StorageRequirements, DataChunk, PerformanceProfile,
    AccessPattern, PerformancePriority, DurabilityLevel, CompressionPreference,
    StorageEngineId, StorageHandle, StorageHandleId,
};

// Re-exports for unified memory management
pub use unified_memory::{
    StorageStrategy as UnifiedStorageStrategy, StorageType, StorageHandle as UnifiedStorageHandle,
    StorageConfig as UnifiedStorageConfig, StorageRequirements as UnifiedStorageRequirements,
    DataChunk as UnifiedDataChunk, ChunkRange, StorageStats, PerformanceProfile as UnifiedPerformanceProfile,
    AccessPattern as UnifiedAccessPattern, PerformancePriority as UnifiedPerformancePriority,
    DurabilityLevel as UnifiedDurabilityLevel, CompressionPreference as UnifiedCompressionPreference,
    ConcurrencyLevel, IoPattern, DataCharacteristics, StrategyCapability, ResourceCost,
    StorageId, StorageMetadata, PerformanceTracker, AtomicMemoryStats, Speed, Efficiency,
    QueryOptimization, ParallelScalability, CompressionType, CompactionResult,
};

// Re-exports for unified memory manager
pub use unified_manager::{
    UnifiedMemoryManager, MemoryConfig, StrategySelectionAlgorithm, CacheManager, 
    PerformanceMonitor, StrategySelector, StrategySelection, DefaultStrategySelector,
};

// Re-exports for ML-based strategy selection
pub use ml_strategy_selector::{
    MLStrategySelector, AdaptiveUnifiedMemoryManager, WorkloadFeatures, 
    PerformancePrediction, TrainingExample, ModelStats,
};

// Re-exports for zero-copy operations
pub use zero_copy::{
    ZeroCopyView, ZeroCopyManager, MemoryMappedView, CacheAwareAllocator,
    CacheTopology, CacheLevel, MemoryLayout, CacheAwareOps, ZeroCopyStats,
    MemoryPool, AllocationStats, CACHE_LINE_SIZE, PAGE_SIZE,
};

// Re-exports for unified column store
pub use unified_column_store::{
    UnifiedColumnStoreStrategy, ColumnStoreConfig, EncodingType, CompressionEngine,
    EncodingStrategy, BlockManager, ColumnStoreHandle, ColumnLayout, ColumnDataType,
    ColumnStatistics, CompressedBlock, BlockId, PhysicalStorage,
};

// Re-exports for adaptive string pool
pub use adaptive_string_pool::{
    AdaptiveStringPoolStrategy, StringPoolConfig, StringCharacteristics, PatternAnalysis,
    StringStorageStrategy, StringCompressionAlgorithm, StringPoolHandle, StringPoolStatistics,
    StringPatternAnalyzer, StringCompressionEngine, CompressionDictionary, StringId,
};

// Re-exports for hybrid large scale strategy
pub use hybrid_large_scale::{
    HybridLargeScaleStrategy, HybridConfig, TierConfig, TierStorageType, DataTier,
    AccessPattern as HybridAccessPattern, AccessPatternType, TieredDataEntry, DataId,
    TierManager, TieringReport, TierBackend, TierStorageInfo, HybridHandle, HybridStatistics,
};

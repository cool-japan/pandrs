// Storage engines module
pub mod adaptive_string_pool;
pub mod column_store;
pub mod disk;
pub mod hybrid_large_scale;
pub mod memory_mapped;
pub mod ml_strategy_selector;
pub mod string_pool;
pub mod traits;
pub mod unified_column_store;
pub mod unified_manager;
pub mod unified_memory;
pub mod zero_copy;
// pub mod unified_string_pool; // Temporarily disabled due to Send/Sync issues
pub mod simple_unified_string_pool;

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
    AccessPattern as UnifiedAccessPattern, AtomicMemoryStats, ChunkRange, CompactionResult,
    CompressionPreference as UnifiedCompressionPreference, CompressionType, ConcurrencyLevel,
    DataCharacteristics, DataChunk as UnifiedDataChunk, DurabilityLevel as UnifiedDurabilityLevel,
    Efficiency, IoPattern, ParallelScalability, PerformancePriority as UnifiedPerformancePriority,
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
    TrainingExample, WorkloadFeatures,
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

// Re-exports for unified zero-copy string pool (temporarily disabled)
// pub use unified_string_pool::{
//     UnifiedStringPool, UnifiedStringPoolConfig, UnifiedStringView, StringMetadata,
//     UnifiedStringPoolStats,
// };

// Re-exports for simplified unified zero-copy string pool
pub use simple_unified_string_pool::{
    SimpleStringPoolStats, SimpleStringView, SimpleUnifiedStringPool,
};

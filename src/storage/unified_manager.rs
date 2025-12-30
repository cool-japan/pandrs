//! Unified Memory Manager Implementation
//!
//! This module provides the main UnifiedMemoryManager implementation with
//! adaptive storage strategy selection and performance optimization.

use crate::core::error::Error;
use crate::core::error::Result;
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Global string pool for optimized string handling
pub struct GlobalStringPool {
    /// String to ID mapping
    string_to_id: HashMap<String, u32>,
    /// ID to string mapping
    id_to_string: Vec<String>,
    /// Next available ID
    next_id: u32,
    /// Statistics
    stats: StringPoolStats,
}

impl GlobalStringPool {
    pub fn new() -> Self {
        Self {
            string_to_id: HashMap::new(),
            id_to_string: Vec::new(),
            next_id: 0,
            stats: StringPoolStats::new(),
        }
    }

    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.string_to_id.get(s) {
            self.stats.hits += 1;
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.string_to_id.insert(s.to_string(), id);
            self.id_to_string.push(s.to_string());
            self.stats.misses += 1;
            self.stats.unique_strings += 1;
            id
        }
    }

    pub fn get(&self, id: u32) -> Option<&str> {
        self.id_to_string.get(id as usize).map(|s| s.as_str())
    }

    pub fn stats(&self) -> &StringPoolStats {
        &self.stats
    }
}

/// String pool statistics
#[derive(Debug, Clone)]
pub struct StringPoolStats {
    pub hits: u64,
    pub misses: u64,
    pub unique_strings: u64,
}

impl StringPoolStats {
    fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            unique_strings: 0,
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

/// Memory configuration for the unified manager
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Default compression type
    pub default_compression: CompressionType,
    /// Enable adaptive optimization
    pub adaptive_optimization: bool,
    /// Performance monitoring interval
    pub monitoring_interval: std::time::Duration,
    /// Cache size for frequently accessed data
    pub cache_size: usize,
    /// Strategy selection algorithm
    pub strategy_selection: StrategySelectionAlgorithm,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: None,
            default_compression: CompressionType::Auto,
            adaptive_optimization: true,
            monitoring_interval: std::time::Duration::from_secs(60),
            cache_size: 128 * 1024 * 1024, // 128MB
            strategy_selection: StrategySelectionAlgorithm::Adaptive,
        }
    }
}

/// Strategy selection algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategySelectionAlgorithm {
    /// Manual strategy selection
    Manual,
    /// Rule-based selection
    RuleBased,
    /// Machine learning based
    MachineLearning,
    /// Adaptive selection with learning
    Adaptive,
}

/// Cache management across strategies
pub struct CacheManager {
    /// LRU cache for frequently accessed data
    cache: HashMap<String, CachedItem>,
    /// Cache capacity in bytes
    capacity: usize,
    /// Current cache size
    current_size: usize,
    /// Access tracking for LRU
    access_order: Vec<String>,
    /// Cache statistics
    stats: CacheStats,
}

impl CacheManager {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
            current_size: 0,
            access_order: Vec::new(),
            stats: CacheStats::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&DataChunk> {
        if let Some(item) = self.cache.get(key) {
            // Update access order for LRU
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());

            self.stats.hits += 1;
            Some(&item.data)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    pub fn put(&mut self, key: String, data: DataChunk) {
        let item_size = data.len();

        // Evict items if necessary
        while self.current_size + item_size > self.capacity && !self.access_order.is_empty() {
            if let Some(lru_key) = self.access_order.first().cloned() {
                self.evict(&lru_key);
            }
        }

        // Insert new item
        if item_size <= self.capacity {
            let item = CachedItem {
                data,
                created_at: Instant::now(),
                access_count: 1,
            };

            self.cache.insert(key.clone(), item);
            self.current_size += item_size;
            self.access_order.push(key);
        }
    }

    fn evict(&mut self, key: &str) {
        if let Some(item) = self.cache.remove(key) {
            self.current_size -= item.data.len();
            self.stats.evictions += 1;
        }

        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Cached item
#[derive(Debug, Clone)]
struct CachedItem {
    data: DataChunk,
    created_at: Instant,
    access_count: u64,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }
}

/// Performance monitoring for strategies
pub struct PerformanceMonitor {
    /// Per-strategy performance metrics
    strategy_metrics: HashMap<StorageType, StrategyMetrics>,
    /// Global system metrics
    system_metrics: SystemMetrics,
    /// Monitoring start time
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            strategy_metrics: HashMap::new(),
            system_metrics: SystemMetrics::new(),
            start_time: Instant::now(),
        }
    }

    pub fn record_operation(
        &mut self,
        strategy_type: StorageType,
        operation: OperationType,
        duration: std::time::Duration,
        bytes: usize,
    ) {
        let metrics = self
            .strategy_metrics
            .entry(strategy_type)
            .or_insert_with(StrategyMetrics::new);

        metrics.record_operation(operation, duration, bytes);
        self.system_metrics
            .record_operation(operation, duration, bytes);
    }

    pub fn get_strategy_metrics(&self, strategy_type: StorageType) -> Option<&StrategyMetrics> {
        self.strategy_metrics.get(&strategy_type)
    }

    pub fn get_system_metrics(&self) -> &SystemMetrics {
        &self.system_metrics
    }

    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

/// Operation type for performance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Read,
    Write,
    Append,
    Delete,
    Compact,
    Flush,
}

/// Performance metrics for a strategy
#[derive(Debug, Clone)]
pub struct StrategyMetrics {
    /// Operation counts
    pub operation_counts: HashMap<OperationType, u64>,
    /// Total operation times
    pub operation_times: HashMap<OperationType, std::time::Duration>,
    /// Total bytes processed
    pub bytes_processed: HashMap<OperationType, u64>,
    /// Last operation timestamp
    pub last_operation: Option<Instant>,
}

impl StrategyMetrics {
    fn new() -> Self {
        Self {
            operation_counts: HashMap::new(),
            operation_times: HashMap::new(),
            bytes_processed: HashMap::new(),
            last_operation: None,
        }
    }

    fn record_operation(
        &mut self,
        operation: OperationType,
        duration: std::time::Duration,
        bytes: usize,
    ) {
        *self.operation_counts.entry(operation).or_insert(0) += 1;
        *self
            .operation_times
            .entry(operation)
            .or_insert(std::time::Duration::ZERO) += duration;
        *self.bytes_processed.entry(operation).or_insert(0) += bytes as u64;
        self.last_operation = Some(Instant::now());
    }

    pub fn average_operation_time(&self, operation: OperationType) -> Option<std::time::Duration> {
        let count = self.operation_counts.get(&operation)?;
        let total_time = self.operation_times.get(&operation)?;

        if *count > 0 {
            Some(*total_time / *count as u32)
        } else {
            None
        }
    }

    pub fn throughput(&self, operation: OperationType) -> Option<f64> {
        let bytes = self.bytes_processed.get(&operation)?;
        let time = self.operation_times.get(&operation)?;

        if time.as_secs_f64() > 0.0 {
            Some(*bytes as f64 / time.as_secs_f64())
        } else {
            None
        }
    }
}

/// System-wide performance metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Total operations across all strategies
    pub total_operations: u64,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Total time spent in operations
    pub total_time: std::time::Duration,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

impl SystemMetrics {
    fn new() -> Self {
        Self {
            total_operations: 0,
            total_bytes: 0,
            total_time: std::time::Duration::ZERO,
            memory_stats: MemoryStats::new(),
        }
    }

    fn record_operation(
        &mut self,
        _operation: OperationType,
        duration: std::time::Duration,
        bytes: usize,
    ) {
        self.total_operations += 1;
        self.total_bytes += bytes as u64;
        self.total_time += duration;
    }

    pub fn overall_throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            self.total_bytes as f64 / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Total allocations
    pub total_allocations: u64,
    /// Total deallocations
    pub total_deallocations: u64,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            total_allocations: 0,
            total_deallocations: 0,
        }
    }

    pub fn record_allocation(&mut self, size: usize) {
        self.current_usage += size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
        self.total_allocations += 1;
    }

    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.total_deallocations += 1;
    }

    pub fn active_allocations(&self) -> u64 {
        self.total_allocations
            .saturating_sub(self.total_deallocations)
    }
}

/// Unified memory manager for PandRS
pub struct UnifiedMemoryManager {
    /// Active storage strategies
    strategies: HashMap<
        StorageType,
        Box<dyn StorageStrategy<Handle = StorageHandle, Error = Error, Metadata = StorageMetadata>>,
    >,

    /// Adaptive strategy selector
    selector: Box<dyn StrategySelector>,

    /// Performance monitoring and metrics
    monitor: Arc<Mutex<PerformanceMonitor>>,

    /// Memory usage statistics and tracking
    stats: Arc<AtomicMemoryStats>,

    /// Global memory configuration
    config: MemoryConfig,

    /// Cache management across strategies
    cache_manager: Arc<Mutex<CacheManager>>,

    /// String pool for optimized string handling
    string_pool: Arc<Mutex<GlobalStringPool>>,

    /// Next storage ID
    next_storage_id: std::sync::atomic::AtomicU64,
}

impl UnifiedMemoryManager {
    /// Create a new unified memory manager
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            strategies: HashMap::new(),
            selector: Box::new(DefaultStrategySelector::new()),
            monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            stats: Arc::new(AtomicMemoryStats::new()),
            cache_manager: Arc::new(Mutex::new(CacheManager::new(config.cache_size))),
            string_pool: Arc::new(Mutex::new(GlobalStringPool::new())),
            config,
            next_storage_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Create new storage with given configuration
    pub fn create_storage(&mut self, config: &StorageConfig) -> Result<StorageHandle> {
        let selection = self.selector.select_strategy(&config.requirements);

        // Try primary strategy first
        if let Some(strategy) = self.strategies.get_mut(&selection.primary) {
            match strategy.create_storage(config) {
                Ok(handle) => {
                    let storage_id = StorageId(
                        self.next_storage_id
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                    );

                    let metadata = StorageMetadata::new(config.requirements.estimated_size);

                    return Ok(StorageHandle::new(
                        storage_id,
                        selection.primary,
                        Box::new(handle),
                        metadata,
                    ));
                }
                Err(e) => {
                    eprintln!("Primary strategy {} failed: {}", selection.primary as u8, e);
                }
            }
        }

        // Try fallback strategies
        for &fallback_type in &selection.fallbacks {
            if let Some(strategy) = self.strategies.get_mut(&fallback_type) {
                if let Ok(handle) = strategy.create_storage(config) {
                    let storage_id = StorageId(
                        self.next_storage_id
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst),
                    );

                    let metadata = StorageMetadata::new(config.requirements.estimated_size);

                    return Ok(StorageHandle::new(
                        storage_id,
                        fallback_type,
                        Box::new(handle),
                        metadata,
                    ));
                }
            }
        }

        Err(Error::InvalidOperation(
            "No suitable storage strategy available".to_string(),
        ))
    }

    /// Read data chunk from storage
    pub fn read_chunk(&self, handle: &StorageHandle, range: ChunkRange) -> Result<DataChunk> {
        let start_time = Instant::now();

        // Check cache first
        let cache_key = format!("{}:{}-{}", handle.id.0, range.start, range.end);
        if let Ok(mut cache) = self.cache_manager.lock() {
            if let Some(cached_data) = cache.get(&cache_key) {
                return Ok(cached_data.clone());
            }
        }

        // Read from storage strategy
        if let Some(strategy) = self.strategies.get(&handle.strategy_type) {
            let result = strategy.read_chunk(&handle, range.clone());

            // Record performance metrics
            let duration = start_time.elapsed();
            if let Ok(ref chunk) = result {
                if let Ok(mut monitor) = self.monitor.lock() {
                    monitor.record_operation(
                        handle.strategy_type,
                        OperationType::Read,
                        duration,
                        chunk.len(),
                    );
                }

                // Cache the result
                if let Ok(mut cache) = self.cache_manager.lock() {
                    cache.put(cache_key, chunk.clone());
                }
            }

            result
        } else {
            Err(Error::InvalidOperation(format!(
                "Strategy {:?} not found",
                handle.strategy_type
            )))
        }
    }

    /// Write data chunk to storage
    pub fn write_chunk(&mut self, handle: &StorageHandle, chunk: DataChunk) -> Result<()> {
        let start_time = Instant::now();

        if let Some(strategy) = self.strategies.get_mut(&handle.strategy_type) {
            let result = strategy.write_chunk(&handle, chunk.clone());

            // Record performance metrics
            let duration = start_time.elapsed();
            if let Ok(mut monitor) = self.monitor.lock() {
                monitor.record_operation(
                    handle.strategy_type,
                    OperationType::Write,
                    duration,
                    chunk.len(),
                );
            }

            result
        } else {
            Err(Error::InvalidOperation(format!(
                "Strategy {:?} not found",
                handle.strategy_type
            )))
        }
    }

    /// Delete storage and free resources
    pub fn delete_storage(&mut self, handle: &StorageHandle) -> Result<()> {
        let start_time = Instant::now();

        if let Some(strategy) = self.strategies.get_mut(&handle.strategy_type) {
            let result = strategy.delete_storage(&handle);

            // Record performance metrics
            let duration = start_time.elapsed();
            if let Ok(mut monitor) = self.monitor.lock() {
                monitor.record_operation(handle.strategy_type, OperationType::Delete, duration, 0);
            }

            result
        } else {
            Err(Error::InvalidOperation(format!(
                "Strategy {:?} not found",
                handle.strategy_type
            )))
        }
    }

    /// Add a storage strategy to the manager
    pub fn add_strategy(
        &mut self,
        strategy_type: StorageType,
        strategy: Box<
            dyn StorageStrategy<Handle = StorageHandle, Error = Error, Metadata = StorageMetadata>,
        >,
    ) {
        self.strategies.insert(strategy_type, strategy);
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> &AtomicMemoryStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<CacheStats> {
        self.cache_manager
            .lock()
            .map(|cache| cache.stats().clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire cache lock".to_string()))
    }

    /// Get string pool statistics
    pub fn string_pool_stats(&self) -> Result<StringPoolStats> {
        self.string_pool
            .lock()
            .map(|pool| pool.stats().clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire string pool lock".to_string()))
    }
}

/// Strategy selection result
#[derive(Debug, Clone)]
pub struct StrategySelection {
    /// Primary strategy to use
    pub primary: StorageType,
    /// Fallback strategies in order of preference
    pub fallbacks: Vec<StorageType>,
    /// Confidence in the selection (0.0 to 1.0)
    pub confidence: f64,
}

/// Trait for strategy selection algorithms
pub trait StrategySelector: Send + Sync {
    /// Select the best strategy for given requirements
    fn select_strategy(&self, requirements: &StorageRequirements) -> StrategySelection;

    /// Record performance feedback for learning
    fn record_performance(&mut self, strategy_type: StorageType, performance: &StrategyMetrics);
}

/// Default rule-based strategy selector
pub struct DefaultStrategySelector {
    /// Performance history for learning
    performance_history: HashMap<StorageType, Vec<f64>>,
}

impl DefaultStrategySelector {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
        }
    }
}

impl StrategySelector for DefaultStrategySelector {
    fn select_strategy(&self, requirements: &StorageRequirements) -> StrategySelection {
        // Simple rule-based selection logic
        let primary = match (
            &requirements.data_characteristics,
            requirements.estimated_size,
        ) {
            (DataCharacteristics::Text, _) => StorageType::StringPool,
            (_, size) if size > 100 * 1024 * 1024 => StorageType::HybridLargeScale, // > 100MB
            (DataCharacteristics::Numeric, _) => StorageType::ColumnStore,
            (DataCharacteristics::TimeSeries, _) => StorageType::ColumnStore,
            _ => match requirements.performance_priority {
                PerformancePriority::Speed => StorageType::InMemory,
                PerformancePriority::Memory => StorageType::DiskBased,
                _ => StorageType::ColumnStore,
            },
        };

        let fallbacks = vec![
            StorageType::InMemory,
            StorageType::ColumnStore,
            StorageType::DiskBased,
        ]
        .into_iter()
        .filter(|&t| t != primary)
        .collect();

        StrategySelection {
            primary,
            fallbacks,
            confidence: 0.8, // Default confidence
        }
    }

    fn record_performance(&mut self, strategy_type: StorageType, _performance: &StrategyMetrics) {
        // Simple performance tracking
        // In a real implementation, this would analyze the metrics and update selection logic
        self.performance_history
            .entry(strategy_type)
            .or_insert_with(Vec::new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_string_pool() {
        let mut pool = GlobalStringPool::new();

        let id1 = pool.intern("hello");
        let id2 = pool.intern("world");
        let id3 = pool.intern("hello"); // Should reuse existing ID

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(pool.get(id1), Some("hello"));
        assert_eq!(pool.get(id2), Some("world"));

        assert!(pool.stats().hit_rate() > 0.0);
    }

    #[test]
    fn test_cache_manager() {
        let mut cache = CacheManager::new(1024); // 1KB cache

        let chunk1 = DataChunk::new(vec![1, 2, 3]);
        let chunk2 = DataChunk::new(vec![4, 5, 6]);

        cache.put("key1".to_string(), chunk1.clone());
        cache.put("key2".to_string(), chunk2.clone());

        assert!(cache.get("key1").is_some());
        assert!(cache.get("key2").is_some());

        assert!(cache.stats().hit_rate() > 0.0);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        monitor.record_operation(
            StorageType::InMemory,
            OperationType::Read,
            std::time::Duration::from_millis(10),
            1024,
        );

        let metrics = monitor.get_strategy_metrics(StorageType::InMemory).unwrap();
        assert_eq!(metrics.operation_counts[&OperationType::Read], 1);
        assert_eq!(metrics.bytes_processed[&OperationType::Read], 1024);
    }

    #[test]
    fn test_default_strategy_selector() {
        let selector = DefaultStrategySelector::new();

        let requirements = StorageRequirements {
            estimated_size: 1024,
            data_characteristics: DataCharacteristics::Text,
            performance_priority: PerformancePriority::Speed,
            ..Default::default()
        };

        let selection = selector.select_strategy(&requirements);
        assert_eq!(selection.primary, StorageType::StringPool);
    }

    #[test]
    fn test_unified_memory_manager() {
        let config = MemoryConfig::default();
        let manager = UnifiedMemoryManager::new(config);

        // Test basic creation
        assert!(manager.cache_stats().is_ok());
        assert!(manager.string_pool_stats().is_ok());
    }
}

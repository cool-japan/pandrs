//! Unified Memory Manager Implementation
//!
//! This module provides the main UnifiedMemoryManager implementation with
//! adaptive storage strategy selection and performance optimization.

use crate::core::error::Error;
use crate::core::error::Result;
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Global string pool for optimized string handling.
///
/// Note: this is the *manager's* interning table, distinct from
/// [`crate::storage::string_pool::GlobalStringPool`] (the column-level pool
/// re-exported from `crate::column`). Both names are exported from
/// `crate::storage`; refer to them by module path when both are in scope.
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

    /// Intern `s`, returning its stable id.
    ///
    /// Fails once the 32-bit id space is exhausted rather than wrapping around
    /// and aliasing an existing string. There is no eviction: ids are handed out
    /// permanently, so callers that intern unbounded user data should bound it
    /// themselves.
    pub fn intern(&mut self, s: &str) -> Result<u32> {
        if let Some(&id) = self.string_to_id.get(s) {
            self.stats.hits += 1;
            return Ok(id);
        }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).ok_or_else(|| {
            Error::InvalidOperation(
                "Global string pool exhausted: 2^32 distinct strings interned".to_string(),
            )
        })?;
        self.string_to_id.insert(s.to_string(), id);
        self.id_to_string.push(s.to_string());
        self.stats.misses += 1;
        self.stats.unique_strings += 1;
        Ok(id)
    }

    pub fn get(&self, id: u32) -> Option<&str> {
        self.id_to_string.get(id as usize).map(|s| s.as_str())
    }

    /// Number of distinct strings interned.
    pub fn len(&self) -> usize {
        self.id_to_string.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_string.is_empty()
    }

    pub fn stats(&self) -> &StringPoolStats {
        &self.stats
    }
}

impl Default for GlobalStringPool {
    fn default() -> Self {
        Self::new()
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

/// Cache management across strategies.
///
/// LRU order is tracked with a monotonically increasing generation stamp per
/// entry rather than a `Vec<String>` that was linearly searched and mutated on
/// every hit (O(n) per access, and duplicated entries once a key was re-`put`).
pub struct CacheManager {
    /// Cached chunks
    cache: HashMap<String, CachedItem>,
    /// Cache capacity in bytes
    capacity: usize,
    /// Current cache size in bytes
    current_size: usize,
    /// Monotonic clock used for LRU stamps
    clock: u64,
    /// Cache statistics
    stats: CacheStats,
}

impl CacheManager {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            capacity,
            current_size: 0,
            clock: 0,
            stats: CacheStats::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&DataChunk> {
        self.clock += 1;
        let clock = self.clock;
        match self.cache.get_mut(key) {
            Some(item) => {
                item.last_used = clock;
                item.access_count += 1;
                self.stats.hits += 1;
                Some(&item.data)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    pub fn put(&mut self, key: String, data: DataChunk) {
        let item_size = data.len();
        if item_size > self.capacity {
            // An item that can never fit is counted, not silently dropped.
            self.stats.rejected += 1;
            return;
        }

        // Replacing a key must release the old bytes first; the old code only
        // ever added, so `current_size` drifted upwards until the cache evicted
        // everything on every insert.
        if let Some(old) = self.cache.remove(&key) {
            self.current_size = self.current_size.saturating_sub(old.data.len());
        }

        while self.current_size + item_size > self.capacity && !self.cache.is_empty() {
            let victim = self
                .cache
                .iter()
                .min_by_key(|(_, item)| item.last_used)
                .map(|(k, _)| k.clone());
            match victim {
                Some(victim) => self.evict(&victim),
                None => break,
            }
        }

        self.clock += 1;
        self.cache.insert(
            key,
            CachedItem {
                data,
                created_at: Instant::now(),
                access_count: 1,
                last_used: self.clock,
            },
        );
        self.current_size += item_size;
    }

    /// Drop every entry whose key starts with `prefix`.
    pub fn invalidate_prefix(&mut self, prefix: &str) {
        let keys: Vec<String> = self
            .cache
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        for key in keys {
            self.evict(&key);
        }
    }

    fn evict(&mut self, key: &str) {
        if let Some(item) = self.cache.remove(key) {
            self.current_size = self.current_size.saturating_sub(item.data.len());
            self.stats.evictions += 1;
        }
    }

    /// Bytes currently held by the cache.
    pub fn size_bytes(&self) -> usize {
        self.current_size
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Cached item
#[derive(Debug, Clone)]
struct CachedItem {
    data: DataChunk,
    /// When the entry was inserted
    created_at: Instant,
    /// How many times the entry has been served
    access_count: u64,
    /// LRU stamp
    last_used: u64,
}

impl CachedItem {
    /// Age of this entry.
    fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl CacheManager {
    /// Access count and age of a cached entry, for diagnostics.
    pub fn entry_stats(&self, key: &str) -> Option<(u64, std::time::Duration)> {
        self.cache
            .get(key)
            .map(|item| (item.access_count, item.age()))
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Items too large for the cache and therefore never stored
    pub rejected: u64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            evictions: 0,
            rejected: 0,
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

/// Boxed strategy type stored by the manager.
pub type BoxedStrategy =
    Box<dyn StorageStrategy<Handle = StorageHandle, Error = Error, Metadata = StorageMetadata>>;

/// Adapter that lets a concrete strategy (with its own handle type) be stored
/// in the manager's `StorageType -> strategy` map.
///
/// Every concrete strategy defines its own `Handle`, so none of them could ever
/// satisfy `StorageStrategy<Handle = StorageHandle>` directly — `add_strategy`
/// was therefore uncallable, nothing was ever registered, and
/// `create_storage` always returned "No suitable storage strategy available".
pub struct StrategyAdapter<S>
where
    S: StorageStrategy<Error = Error>,
    S::Handle: Send + Sync + 'static,
{
    inner: S,
    storage_type: StorageType,
    next_id: AtomicU64,
}

impl<S> StrategyAdapter<S>
where
    S: StorageStrategy<Error = Error>,
    S::Handle: Send + Sync + 'static,
{
    pub fn new(storage_type: StorageType, inner: S) -> Self {
        Self {
            inner,
            storage_type,
            next_id: AtomicU64::new(1),
        }
    }

    /// Box this adapter for registration with a [`UnifiedMemoryManager`].
    pub fn boxed(storage_type: StorageType, inner: S) -> BoxedStrategy
    where
        S: 'static,
    {
        Box::new(Self::new(storage_type, inner))
    }

    fn inner_handle<'a>(&self, handle: &'a StorageHandle) -> Result<&'a S::Handle> {
        handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })
    }
}

impl<S> StorageStrategy for StrategyAdapter<S>
where
    S: StorageStrategy<Error = Error>,
    S::Handle: Send + Sync + 'static,
{
    type Handle = StorageHandle;
    type Error = Error;
    type Metadata = StorageMetadata;

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn create_storage(&mut self, config: &StorageConfig) -> Result<StorageHandle> {
        let inner = self.inner.create_storage(config)?;
        let id = StorageId(self.next_id.fetch_add(1, Ordering::SeqCst));
        Ok(StorageHandle::new(
            id,
            self.storage_type,
            Box::new(inner),
            StorageMetadata::new(config.requirements.estimated_size),
        ))
    }

    fn read_chunk(&self, handle: &StorageHandle, range: ChunkRange) -> Result<DataChunk> {
        self.inner.read_chunk(self.inner_handle(handle)?, range)
    }

    fn write_chunk(&mut self, handle: &StorageHandle, chunk: DataChunk) -> Result<()> {
        let inner_handle = handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })?;
        self.inner.write_chunk(inner_handle, chunk)
    }

    fn append_chunk(&mut self, handle: &StorageHandle, chunk: DataChunk) -> Result<()> {
        let inner_handle = handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })?;
        self.inner.append_chunk(inner_handle, chunk)
    }

    fn flush(&mut self, handle: &StorageHandle) -> Result<()> {
        let inner_handle = handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })?;
        self.inner.flush(inner_handle)
    }

    fn delete_storage(&mut self, handle: &StorageHandle) -> Result<()> {
        let inner_handle = handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })?;
        self.inner.delete_storage(inner_handle)
    }

    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability {
        self.inner.can_handle(requirements)
    }

    fn performance_profile(&self) -> PerformanceProfile {
        self.inner.performance_profile()
    }

    fn storage_stats(&self) -> StorageStats {
        self.inner.storage_stats()
    }

    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        self.inner.optimize_for_pattern(pattern)
    }

    fn compact(&mut self, handle: &StorageHandle) -> Result<CompactionResult> {
        let inner_handle = handle
            .inner_handle
            .downcast_ref::<S::Handle>()
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Storage handle {:?} does not belong to the {:?} strategy",
                    handle.id, self.storage_type
                ))
            })?;
        self.inner.compact(inner_handle)
    }
}

/// Unified memory manager for PandRS
pub struct UnifiedMemoryManager {
    /// Active storage strategies
    strategies: HashMap<StorageType, BoxedStrategy>,

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

    /// Per-storage cache generation. Bumped on every mutation so that stale
    /// cache entries become unreachable.
    generations: Arc<Mutex<HashMap<u64, u64>>>,

    /// Next storage ID
    next_storage_id: AtomicU64,
}

impl UnifiedMemoryManager {
    /// Create a new unified memory manager with the built-in strategies
    /// registered.
    pub fn new(config: MemoryConfig) -> Self {
        let mut manager = Self::empty(config);
        manager.register_builtin_strategies();
        manager
    }

    /// Create a manager with **no** strategies registered.
    ///
    /// Useful when the caller wants full control over which backends exist;
    /// `create_storage` on an empty manager returns a clear configuration error.
    pub fn empty(config: MemoryConfig) -> Self {
        Self {
            strategies: HashMap::new(),
            selector: Box::new(DefaultStrategySelector::new()),
            monitor: Arc::new(Mutex::new(PerformanceMonitor::new())),
            stats: Arc::new(AtomicMemoryStats::new()),
            cache_manager: Arc::new(Mutex::new(CacheManager::new(config.cache_size))),
            string_pool: Arc::new(Mutex::new(GlobalStringPool::new())),
            generations: Arc::new(Mutex::new(HashMap::new())),
            config,
            next_storage_id: AtomicU64::new(1),
        }
    }

    /// Register the strategies shipped with PandRS.
    pub fn register_builtin_strategies(&mut self) {
        use crate::storage::adaptive_string_pool::{AdaptiveStringPoolStrategy, StringPoolConfig};
        use crate::storage::hybrid_large_scale::{HybridConfig, HybridLargeScaleStrategy};
        use crate::storage::unified_column_store::{
            encoding::EncodingType, ColumnStoreConfig, UnifiedColumnStoreStrategy,
        };

        let columnar = ColumnStoreConfig {
            compression_type: self.config.default_compression,
            ..Default::default()
        };
        self.add_strategy(
            StorageType::ColumnStore,
            StrategyAdapter::boxed(
                StorageType::ColumnStore,
                UnifiedColumnStoreStrategy::new(columnar),
            ),
        );

        // "In memory" is the column store with every transform disabled, so the
        // bytes are stored verbatim in RAM.
        let in_memory = ColumnStoreConfig {
            compression_type: CompressionType::None,
            encoding_type: EncodingType::None,
            ..Default::default()
        };
        self.add_strategy(
            StorageType::InMemory,
            StrategyAdapter::boxed(
                StorageType::InMemory,
                UnifiedColumnStoreStrategy::new(in_memory),
            ),
        );

        self.add_strategy(
            StorageType::StringPool,
            StrategyAdapter::boxed(
                StorageType::StringPool,
                AdaptiveStringPoolStrategy::new(StringPoolConfig::default()),
            ),
        );

        self.add_strategy(
            StorageType::HybridLargeScale,
            StrategyAdapter::boxed(
                StorageType::HybridLargeScale,
                HybridLargeScaleStrategy::new(HybridConfig::default()),
            ),
        );

        // "Disk based" is the hybrid strategy with a 1-byte hot tier, so every
        // chunk lands in a file-backed tier.
        let mut disk_config = HybridConfig::default();
        disk_config.hot_tier.max_size = 1;
        disk_config.max_hot_memory = 1;
        self.add_strategy(
            StorageType::DiskBased,
            StrategyAdapter::boxed(
                StorageType::DiskBased,
                HybridLargeScaleStrategy::new(disk_config),
            ),
        );

        // StorageType::MemoryMapped intentionally has no strategy: PandRS's
        // memory mapping is a read-only view over an existing file
        // (`storage::memory_mapped`), not a writable chunk store. Selection
        // falls through to the fallback list instead of pretending otherwise.
    }

    /// Storage types with a registered strategy.
    pub fn registered_strategies(&self) -> Vec<StorageType> {
        let mut types: Vec<StorageType> = self.strategies.keys().copied().collect();
        types.sort_by_key(|t| format!("{:?}", t));
        types
    }

    fn generation(&self, id: StorageId) -> u64 {
        self.generations
            .lock()
            .ok()
            .and_then(|g| g.get(&id.0).copied())
            .unwrap_or(0)
    }

    /// Invalidate every cached read for `id` by advancing its generation.
    fn bump_generation(&self, id: StorageId) -> Result<()> {
        let mut generations = self.generations.lock().map_err(|_| {
            Error::InvalidOperation("Cache generation lock is poisoned".to_string())
        })?;
        let entry = generations.entry(id.0).or_insert(0);
        *entry = entry.wrapping_add(1);
        drop(generations);

        // Also drop the now-unreachable entries eagerly so the cache does not
        // hold on to superseded bytes until they age out.
        if let Ok(mut cache) = self.cache_manager.lock() {
            cache.invalidate_prefix(&format!("{}:", id.0));
        }
        Ok(())
    }

    fn cache_key(&self, handle: &StorageHandle, range: &ChunkRange) -> String {
        format!(
            "{}:{}:{}-{}",
            handle.id.0,
            self.generation(handle.id),
            range.start,
            range.end
        )
    }

    /// Create new storage with given configuration
    pub fn create_storage(&mut self, config: &StorageConfig) -> Result<StorageHandle> {
        let selection = self.selector.select_strategy(&config.requirements);

        let mut attempts = Vec::with_capacity(1 + selection.fallbacks.len());
        attempts.push(selection.primary);
        attempts.extend(selection.fallbacks.iter().copied());

        let mut last_error: Option<Error> = None;
        for strategy_type in attempts {
            let Some(strategy) = self.strategies.get_mut(&strategy_type) else {
                continue;
            };
            match strategy.create_storage(config) {
                Ok(mut handle) => {
                    // The adapter already built a fully-formed handle whose
                    // `inner_handle` is the concrete strategy handle; re-wrapping
                    // it here would make every later downcast fail.
                    let storage_id = StorageId(self.next_storage_id.fetch_add(1, Ordering::SeqCst));
                    handle.id = storage_id;
                    handle.strategy_type = strategy_type;
                    if let Ok(mut generations) = self.generations.lock() {
                        generations.insert(storage_id.0, 0);
                    }
                    self.stats
                        .record_allocation(config.requirements.estimated_size);
                    return Ok(handle);
                }
                Err(e) => {
                    log::warn!("Storage strategy {:?} failed: {}", strategy_type, e);
                    last_error = Some(e);
                }
            }
        }

        Err(match last_error {
            Some(e) => Error::InvalidOperation(format!(
                "No storage strategy could satisfy the request; last error: {}",
                e
            )),
            None => Error::InvalidOperation(format!(
                "No storage strategy registered for {:?} or its fallbacks {:?}; \
                 call UnifiedMemoryManager::new (which registers the built-ins) \
                 or add_strategy before create_storage",
                selection.primary, selection.fallbacks
            )),
        })
    }

    /// Read data chunk from storage
    pub fn read_chunk(&self, handle: &StorageHandle, range: ChunkRange) -> Result<DataChunk> {
        let start_time = Instant::now();
        let cache_key = self.cache_key(handle, &range);

        if let Ok(mut cache) = self.cache_manager.lock() {
            if let Some(cached_data) = cache.get(&cache_key) {
                let cached = cached_data.clone();
                drop(cache);
                // Cache hits used to return early *without* recording anything,
                // so every metric — including the data the ML selector trains on
                // — excluded them.
                if let Ok(mut monitor) = self.monitor.lock() {
                    monitor.record_operation(
                        handle.strategy_type,
                        OperationType::Read,
                        start_time.elapsed(),
                        cached.len(),
                    );
                }
                return Ok(cached);
            }
        }

        let strategy = self.strategies.get(&handle.strategy_type).ok_or_else(|| {
            Error::InvalidOperation(format!("Strategy {:?} not found", handle.strategy_type))
        })?;

        let result = strategy.read_chunk(handle, range);
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
            if let Ok(mut cache) = self.cache_manager.lock() {
                cache.put(cache_key, chunk.clone());
            }
        }
        result
    }

    /// Write data chunk to storage
    pub fn write_chunk(&mut self, handle: &StorageHandle, chunk: DataChunk) -> Result<()> {
        let start_time = Instant::now();
        let bytes = chunk.len();

        let strategy = self
            .strategies
            .get_mut(&handle.strategy_type)
            .ok_or_else(|| {
                Error::InvalidOperation(format!("Strategy {:?} not found", handle.strategy_type))
            })?;
        let result = strategy.write_chunk(handle, chunk);

        if result.is_ok() {
            // Without this, a read-write-read sequence returned the stale
            // pre-write bytes indefinitely.
            self.bump_generation(handle.id)?;
        }

        if let Ok(mut monitor) = self.monitor.lock() {
            monitor.record_operation(
                handle.strategy_type,
                OperationType::Write,
                start_time.elapsed(),
                bytes,
            );
        }
        result
    }

    /// Append a data chunk to existing storage.
    pub fn append_chunk(&mut self, handle: &StorageHandle, chunk: DataChunk) -> Result<()> {
        let start_time = Instant::now();
        let bytes = chunk.len();

        let strategy = self
            .strategies
            .get_mut(&handle.strategy_type)
            .ok_or_else(|| {
                Error::InvalidOperation(format!("Strategy {:?} not found", handle.strategy_type))
            })?;
        let result = strategy.append_chunk(handle, chunk);

        if result.is_ok() {
            self.bump_generation(handle.id)?;
        }
        if let Ok(mut monitor) = self.monitor.lock() {
            monitor.record_operation(
                handle.strategy_type,
                OperationType::Append,
                start_time.elapsed(),
                bytes,
            );
        }
        result
    }

    /// Delete storage and free resources
    pub fn delete_storage(&mut self, handle: &StorageHandle) -> Result<()> {
        let start_time = Instant::now();

        let strategy = self
            .strategies
            .get_mut(&handle.strategy_type)
            .ok_or_else(|| {
                Error::InvalidOperation(format!("Strategy {:?} not found", handle.strategy_type))
            })?;
        let result = strategy.delete_storage(handle);

        if result.is_ok() {
            self.bump_generation(handle.id)?;
            self.stats.record_deallocation(handle.metadata.size);
        }
        if let Ok(mut monitor) = self.monitor.lock() {
            monitor.record_operation(
                handle.strategy_type,
                OperationType::Delete,
                start_time.elapsed(),
                0,
            );
        }
        result
    }

    /// Add a storage strategy to the manager
    pub fn add_strategy(&mut self, strategy_type: StorageType, strategy: BoxedStrategy) {
        self.strategies.insert(strategy_type, strategy);
    }

    /// Replace the strategy selector (for example with an ML-driven one).
    pub fn set_selector(&mut self, selector: Box<dyn StrategySelector>) {
        self.selector = selector;
    }

    /// The monitor this manager records every operation into.
    ///
    /// Shared so that an external selector can learn from the manager's real
    /// traffic instead of being handed a private monitor that stays empty.
    pub fn monitor(&self) -> Arc<Mutex<PerformanceMonitor>> {
        Arc::clone(&self.monitor)
    }

    /// Feed the selector the metrics collected for `strategy_type`.
    pub fn train_selector(&mut self, strategy_type: StorageType) -> Result<()> {
        let metrics = {
            let monitor = self.monitor.lock().map_err(|_| {
                Error::InvalidOperation("Performance monitor lock is poisoned".to_string())
            })?;
            monitor.get_strategy_metrics(strategy_type).cloned()
        };
        if let Some(metrics) = metrics {
            self.selector.record_performance(strategy_type, &metrics);
        }
        Ok(())
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

    /// Number of bytes currently held by the read cache.
    pub fn cache_size_bytes(&self) -> Result<usize> {
        self.cache_manager
            .lock()
            .map(|cache| cache.size_bytes())
            .map_err(|_| Error::InvalidOperation("Failed to acquire cache lock".to_string()))
    }

    /// Get string pool statistics
    pub fn string_pool_stats(&self) -> Result<StringPoolStats> {
        self.string_pool
            .lock()
            .map(|pool| pool.stats().clone())
            .map_err(|_| Error::InvalidOperation("Failed to acquire string pool lock".to_string()))
    }

    /// Intern a string in the manager's global pool.
    pub fn intern_string(&self, s: &str) -> Result<u32> {
        let mut pool = self.string_pool.lock().map_err(|_| {
            Error::InvalidOperation("Failed to acquire string pool lock".to_string())
        })?;
        pool.intern(s)
    }

    /// Performance metrics collected for a strategy.
    pub fn strategy_metrics(&self, strategy_type: StorageType) -> Result<Option<StrategyMetrics>> {
        let monitor = self.monitor.lock().map_err(|_| {
            Error::InvalidOperation("Performance monitor lock is poisoned".to_string())
        })?;
        Ok(monitor.get_strategy_metrics(strategy_type).cloned())
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
    /// Observed read throughput (bytes/sec) per strategy
    performance_history: HashMap<StorageType, Vec<f64>>,
}

impl DefaultStrategySelector {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
        }
    }

    /// Mean observed read throughput for a strategy, if any was recorded.
    pub fn observed_throughput(&self, strategy_type: StorageType) -> Option<f64> {
        let samples = self.performance_history.get(&strategy_type)?;
        if samples.is_empty() {
            return None;
        }
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }

    /// Number of samples recorded for a strategy.
    pub fn sample_count(&self, strategy_type: StorageType) -> usize {
        self.performance_history
            .get(&strategy_type)
            .map(|s| s.len())
            .unwrap_or(0)
    }
}

impl Default for DefaultStrategySelector {
    fn default() -> Self {
        Self::new()
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

    fn record_performance(&mut self, strategy_type: StorageType, performance: &StrategyMetrics) {
        // Actually push the observation. The old body inserted an empty Vec and
        // returned, so the "selector with learning" never learned anything.
        let throughput = performance
            .throughput(OperationType::Read)
            .or_else(|| performance.throughput(OperationType::Write));
        if let Some(throughput) = throughput {
            let samples = self.performance_history.entry(strategy_type).or_default();
            samples.push(throughput);
            // Keep the history bounded.
            if samples.len() > 1024 {
                let excess = samples.len() - 1024;
                samples.drain(0..excess);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_string_pool() {
        let mut pool = GlobalStringPool::new();

        let id1 = pool.intern("hello").expect("intern");
        let id2 = pool.intern("world").expect("intern");
        let id3 = pool.intern("hello").expect("intern"); // Should reuse existing ID

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
    fn cache_size_accounting_survives_replacement() {
        let mut cache = CacheManager::new(1024);
        cache.put("k".to_string(), DataChunk::new(vec![0u8; 100]));
        assert_eq!(cache.size_bytes(), 100);
        // Re-putting the same key used to add without subtracting, so
        // `current_size` drifted up until everything was evicted on each insert.
        cache.put("k".to_string(), DataChunk::new(vec![0u8; 40]));
        assert_eq!(cache.size_bytes(), 40);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn oversized_items_are_counted_not_silently_dropped() {
        let mut cache = CacheManager::new(64);
        cache.put("big".to_string(), DataChunk::new(vec![0u8; 128]));
        assert_eq!(cache.stats().rejected, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_evicts_least_recently_used() {
        let mut cache = CacheManager::new(200);
        cache.put("a".to_string(), DataChunk::new(vec![0u8; 100]));
        cache.put("b".to_string(), DataChunk::new(vec![0u8; 100]));
        // Touch "a" so "b" becomes the LRU victim.
        assert!(cache.get("a").is_some());
        cache.put("c".to_string(), DataChunk::new(vec![0u8; 100]));
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn access_counts_are_recorded() {
        let mut cache = CacheManager::new(1024);
        cache.put("k".to_string(), DataChunk::new(vec![1, 2, 3]));
        let _ = cache.get("k");
        let _ = cache.get("k");
        let (count, _age) = cache.entry_stats("k").expect("entry stats");
        assert_eq!(count, 3, "access_count never incremented");
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

        let metrics = monitor
            .get_strategy_metrics(StorageType::InMemory)
            .expect("operation should succeed");
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

    #[test]
    fn builtin_strategies_are_registered() {
        // `add_strategy` was never callable (no concrete strategy could satisfy
        // `Handle = StorageHandle`), so `new()` registered nothing and
        // create_storage always failed.
        let manager = UnifiedMemoryManager::new(MemoryConfig::default());
        let registered = manager.registered_strategies();
        for expected in [
            StorageType::ColumnStore,
            StorageType::InMemory,
            StorageType::StringPool,
            StorageType::HybridLargeScale,
            StorageType::DiskBased,
        ] {
            assert!(
                registered.contains(&expected),
                "{:?} is not registered: {:?}",
                expected,
                registered
            );
        }
    }

    #[test]
    fn create_write_read_through_the_manager() {
        let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
        let config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 4096,
                data_characteristics: DataCharacteristics::Numeric,
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = manager.create_storage(&config).expect("create storage");

        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        manager
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write");
        let read = manager
            .read_chunk(&handle, ChunkRange::new(0, payload.len()))
            .expect("read");
        assert_eq!(read.data, payload);
    }

    #[test]
    fn writes_invalidate_the_read_cache() {
        // read -> write -> read used to return the pre-write bytes forever.
        let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
        let config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 1024,
                data_characteristics: DataCharacteristics::Numeric,
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = manager.create_storage(&config).expect("create storage");

        manager
            .write_chunk(&handle, DataChunk::new(vec![1u8; 16]))
            .expect("write");
        let first = manager
            .read_chunk(&handle, ChunkRange::new(0, 16))
            .expect("read");
        assert_eq!(first.data, vec![1u8; 16]);

        // Appending changes rows 16..32 but also invalidates the cached 0..16
        // read, which must now come from the strategy again.
        manager
            .append_chunk(&handle, DataChunk::new(vec![2u8; 16]))
            .expect("append");
        let after = manager
            .read_chunk(&handle, ChunkRange::new(0, 32))
            .expect("read");
        let mut expected = vec![1u8; 16];
        expected.extend_from_slice(&[2u8; 16]);
        assert_eq!(after.data, expected);

        let repeat = manager
            .read_chunk(&handle, ChunkRange::new(0, 16))
            .expect("read");
        assert_eq!(repeat.data, vec![1u8; 16]);
    }

    #[test]
    fn cache_hits_are_recorded_in_metrics() {
        let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
        let config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 1024,
                data_characteristics: DataCharacteristics::Numeric,
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = manager.create_storage(&config).expect("create storage");
        manager
            .write_chunk(&handle, DataChunk::new(vec![7u8; 64]))
            .expect("write");

        let _ = manager
            .read_chunk(&handle, ChunkRange::new(0, 64))
            .expect("read");
        let _ = manager
            .read_chunk(&handle, ChunkRange::new(0, 64))
            .expect("read");

        assert_eq!(manager.cache_stats().expect("stats").hits, 1);
        let metrics = manager
            .strategy_metrics(handle.strategy_type)
            .expect("metrics")
            .expect("strategy seen");
        assert_eq!(
            metrics.operation_counts.get(&OperationType::Read).copied(),
            Some(2),
            "the cache-hit read was excluded from the metrics"
        );
    }

    #[test]
    fn empty_manager_reports_a_clear_configuration_error() {
        let mut manager = UnifiedMemoryManager::empty(MemoryConfig::default());
        let err = manager
            .create_storage(&StorageConfig::default())
            .expect_err("no strategies registered");
        let message = format!("{}", err);
        assert!(
            message.contains("No storage strategy registered"),
            "unhelpful error: {}",
            message
        );
    }

    #[test]
    fn selector_records_real_performance_samples() {
        let mut selector = DefaultStrategySelector::new();
        let mut metrics = StrategyMetrics::new();
        metrics.record_operation(
            OperationType::Read,
            std::time::Duration::from_millis(10),
            1024,
        );
        selector.record_performance(StorageType::ColumnStore, &metrics);
        assert_eq!(selector.sample_count(StorageType::ColumnStore), 1);
        assert!(
            selector
                .observed_throughput(StorageType::ColumnStore)
                .unwrap_or(0.0)
                > 0.0
        );
    }
}

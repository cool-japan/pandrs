//! Storage engine traits and interfaces
//!
//! This module provides the unified trait system for all storage engines in PandRS.
//! It enables pluggable storage backends with performance-based selection.

use crate::core::error::{Error, Result};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

/// Configuration for storage engines
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Expected data size in bytes
    pub estimated_size: usize,
    /// Access pattern hint for optimization
    pub access_pattern: AccessPattern,
    /// Performance priority (speed vs memory)
    pub performance_priority: PerformancePriority,
    /// Durability requirements
    pub durability: DurabilityLevel,
    /// Compression preferences
    pub compression: CompressionPreference,
    /// Memory constraints
    pub memory_limit: Option<usize>,
}

/// Data access patterns for storage optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessPattern {
    /// Sequential reads/writes
    Sequential,
    /// Random access patterns
    Random,
    /// Mostly read operations
    ReadHeavy,
    /// Mostly write operations
    WriteHeavy,
    /// Streaming data processing
    Streaming,
    /// Columnar data analysis
    Columnar,
}

/// Performance priority for storage selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformancePriority {
    /// Prioritize speed over memory usage
    Speed,
    /// Prioritize memory efficiency over speed
    Memory,
    /// Balance between speed and memory
    Balanced,
}

/// Durability requirements for data storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// In-memory only, data lost on restart
    Temporary,
    /// Cached with periodic flushes
    Cached,
    /// Immediately persisted to disk
    Persistent,
}

/// Compression preferences for storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionPreference {
    /// No compression
    None,
    /// Automatic compression based on data characteristics
    Auto,
    /// Fast compression with moderate ratio
    Fast,
    /// Best compression ratio
    Best,
}

/// Data chunk for storage operations
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Raw data bytes
    pub data: Vec<u8>,
    /// Metadata about the chunk
    pub metadata: ChunkMetadata,
}

/// Metadata for data chunks
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Number of rows in the chunk
    pub row_count: usize,
    /// Number of columns in the chunk
    pub column_count: usize,
    /// Compression type used
    pub compression: CompressionPreference,
    /// Size in bytes before compression
    pub uncompressed_size: usize,
    /// Size in bytes after compression
    pub compressed_size: usize,
}

impl DataChunk {
    /// Create a new data chunk
    pub fn new(data: Vec<u8>, metadata: ChunkMetadata) -> Self {
        Self { data, metadata }
    }

    /// Create a chunk of `size` zero bytes, described honestly as one
    /// single-byte column of `size` rows.
    ///
    /// Intended for tests, benchmarks and examples. It used to claim
    /// `size / 8` rows on an invented "8 bytes per row" assumption, which made
    /// every consumer's row accounting wrong for any other element width.
    pub fn new_test_data(size: usize) -> Self {
        let data = vec![0u8; size];
        let metadata = ChunkMetadata {
            row_count: size,
            column_count: 1,
            compression: CompressionPreference::None,
            uncompressed_size: size,
            compressed_size: size,
        };
        Self { data, metadata }
    }

    /// Get chunk size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// Bridges between the engine layer (this module) and the strategy layer
// (`crate::storage::unified_memory`).
//
// The two layers keep separate enums because the strategy layer models states
// the engine layer does not (`Speed::VerySlow`, `Efficiency::Outstanding`,
// `DurabilityLevel::HighDurability`, ...). Rather than duplicate the mapping
// logic at every call site, the faithful conversions live here once.
// ---------------------------------------------------------------------------

use crate::storage::unified_memory as unified;

impl From<Speed> for unified::Speed {
    fn from(speed: Speed) -> Self {
        match speed {
            Speed::Slow => unified::Speed::Slow,
            Speed::Medium => unified::Speed::Medium,
            Speed::Fast => unified::Speed::Fast,
            Speed::VeryFast => unified::Speed::VeryFast,
        }
    }
}

impl From<Efficiency> for unified::Efficiency {
    fn from(efficiency: Efficiency) -> Self {
        match efficiency {
            Efficiency::Poor => unified::Efficiency::Poor,
            Efficiency::Fair => unified::Efficiency::Fair,
            Efficiency::Good => unified::Efficiency::Good,
            Efficiency::Excellent => unified::Efficiency::Excellent,
        }
    }
}

impl From<AccessPattern> for unified::AccessPattern {
    fn from(pattern: AccessPattern) -> Self {
        match pattern {
            AccessPattern::Sequential => unified::AccessPattern::Sequential,
            AccessPattern::Random => unified::AccessPattern::Random,
            AccessPattern::Streaming => unified::AccessPattern::Streaming,
            AccessPattern::Columnar => unified::AccessPattern::Columnar,
            // The engine layer's read/write skew has no direct counterpart in
            // the strategy layer's locality vocabulary; both describe workloads
            // dominated by repeated access to the same working set.
            AccessPattern::ReadHeavy => unified::AccessPattern::HighLocality,
            AccessPattern::WriteHeavy => unified::AccessPattern::LowLocality,
        }
    }
}

impl From<DurabilityLevel> for unified::DurabilityLevel {
    fn from(level: DurabilityLevel) -> Self {
        match level {
            DurabilityLevel::Temporary => unified::DurabilityLevel::Temporary,
            DurabilityLevel::Cached => unified::DurabilityLevel::Session,
            DurabilityLevel::Persistent => unified::DurabilityLevel::Durable,
        }
    }
}

impl From<unified::DurabilityLevel> for DurabilityLevel {
    fn from(level: unified::DurabilityLevel) -> Self {
        match level {
            unified::DurabilityLevel::Temporary => DurabilityLevel::Temporary,
            unified::DurabilityLevel::Session => DurabilityLevel::Cached,
            // The engine layer has no replication tier; both durable levels
            // map onto its strongest guarantee (fsync on write).
            unified::DurabilityLevel::Durable | unified::DurabilityLevel::HighDurability => {
                DurabilityLevel::Persistent
            }
        }
    }
}

impl From<CompressionPreference> for unified::CompressionPreference {
    fn from(preference: CompressionPreference) -> Self {
        match preference {
            CompressionPreference::None => unified::CompressionPreference::None,
            CompressionPreference::Auto => unified::CompressionPreference::Auto,
            CompressionPreference::Fast => unified::CompressionPreference::Fast,
            CompressionPreference::Best => unified::CompressionPreference::High,
        }
    }
}

impl From<PerformancePriority> for unified::PerformancePriority {
    fn from(priority: PerformancePriority) -> Self {
        match priority {
            PerformancePriority::Speed => unified::PerformancePriority::Speed,
            PerformancePriority::Memory => unified::PerformancePriority::Memory,
            PerformancePriority::Balanced => unified::PerformancePriority::Balanced,
        }
    }
}

/// Performance characteristics of a storage engine
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Read speed rating
    pub read_speed: Speed,
    /// Write speed rating
    pub write_speed: Speed,
    /// Memory efficiency rating
    pub memory_efficiency: Efficiency,
    /// Typical compression ratio achieved
    pub compression_ratio: f64,
    /// Random access performance
    pub random_access_speed: Speed,
    /// Sequential access performance
    pub sequential_access_speed: Speed,
}

/// Speed ratings for performance characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Speed {
    Slow,
    Medium,
    Fast,
    VeryFast,
}

/// Efficiency ratings for resource usage
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Efficiency {
    Poor,
    Fair,
    Good,
    Excellent,
}

/// Storage engine statistics
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total bytes stored
    pub total_size: usize,
    /// Number of chunks stored
    pub chunk_count: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f64,
    /// Read operation count
    pub read_operations: u64,
    /// Write operation count
    pub write_operations: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
}

/// Base trait for all storage engines in PandRS
pub trait StorageEngine: Send + Sync {
    type Handle: Clone + Send + Sync;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create new storage with specific configuration
    fn create_storage(&mut self, config: &StorageConfig) -> Result<Self::Handle>;

    /// Read data chunk from storage
    fn read_chunk(&self, handle: &Self::Handle, range: Range<usize>) -> Result<DataChunk>;

    /// Write data chunk to storage
    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()>;

    /// Append data chunk to existing storage
    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()>;

    /// Flush pending writes to persistent storage
    fn flush(&mut self, handle: &Self::Handle) -> Result<()>;

    /// Delete storage and free resources
    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()>;

    /// Get performance characteristics of this storage engine
    fn performance_profile(&self) -> PerformanceProfile;

    /// Get current storage statistics
    fn storage_stats(&self, handle: &Self::Handle) -> Result<StorageStatistics>;

    /// Check if engine supports random access
    fn supports_random_access(&self) -> bool;

    /// Check if engine supports streaming
    fn supports_streaming(&self) -> bool;

    /// Check if engine supports compression
    fn supports_compression(&self) -> bool;

    /// Get optimal chunk size for this engine
    fn optimal_chunk_size(&self) -> usize;

    /// Get memory overhead per chunk
    fn memory_overhead(&self) -> usize;

    /// Optimize engine for specific access pattern
    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()>;

    /// Compact storage to reduce fragmentation
    fn compact(&mut self, handle: &Self::Handle) -> Result<()>;
}

/// Storage engine selector trait
pub trait StorageStrategy: Send + Sync {
    /// Select the best storage engine for given requirements
    fn select_engine(&self, requirements: &StorageRequirements) -> StorageEngineId;

    /// Check if data should be migrated to a different engine
    fn should_migrate(
        &self,
        handle: &StorageHandle,
        new_requirements: &StorageRequirements,
    ) -> bool;

    /// Migrate data between storage engines
    fn migrate_storage(
        &mut self,
        from: &StorageHandle,
        to: &StorageEngineId,
    ) -> Result<StorageHandle>;
}

/// Requirements for storage engine selection
#[derive(Debug, Clone)]
pub struct StorageRequirements {
    /// Configuration parameters
    pub config: StorageConfig,
    /// Expected workload characteristics
    pub workload: WorkloadCharacteristics,
    /// Resource constraints
    pub constraints: ResourceConstraints,
}

/// Workload characteristics for storage optimization
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Read/write ratio (0.0 = write-only, 1.0 = read-only)
    pub read_write_ratio: f64,
    /// Average query size in bytes
    pub avg_query_size: usize,
    /// Concurrent access level
    pub concurrency_level: u32,
    /// Data locality pattern
    pub locality_pattern: LocalityPattern,
}

/// Data locality patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalityPattern {
    /// High temporal locality (recently accessed data accessed again)
    HighTemporal,
    /// High spatial locality (nearby data accessed together)
    HighSpatial,
    /// Mixed locality patterns
    Mixed,
    /// Poor locality (random access)
    PoorLocality,
}

/// Resource constraints for storage engines
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Maximum disk usage in bytes
    pub max_disk: Option<usize>,
    /// CPU budget (relative scale)
    pub cpu_budget: CpuBudget,
    /// Network bandwidth constraints
    pub network_budget: Option<NetworkBandwidth>,
}

/// CPU budget levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBudget {
    Low,
    Medium,
    High,
    Unlimited,
}

/// Network bandwidth constraints
#[derive(Debug, Clone, Copy)]
pub struct NetworkBandwidth {
    /// Bandwidth in bytes per second
    pub bytes_per_second: u64,
    /// Latency in milliseconds
    pub latency_ms: u32,
}

/// Storage engine identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageEngineId {
    ColumnStore,
    MemoryMapped,
    DiskStorage,
    StringPool,
    Hybrid,
}

/// Storage handle that tracks engine and metadata
#[derive(Debug, Clone)]
pub struct StorageHandle {
    /// Engine that owns this storage
    pub engine_id: StorageEngineId,
    /// Engine-specific handle
    pub inner_handle: StorageHandleInner,
    /// Storage metadata
    pub metadata: StorageMetadata,
}

/// Engine-specific storage handle
#[derive(Debug, Clone)]
pub enum StorageHandleInner {
    ColumnStore(crate::storage::column_store::ColumnStoreHandle),
    MemoryMapped(MemoryMappedHandle),
    DiskStorage(crate::storage::disk::DiskStorageHandle),
    StringPool(StringPoolHandle),
}

/// Storage metadata
#[derive(Debug, Clone)]
pub struct StorageMetadata {
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    /// Last accessed timestamp
    pub last_accessed: std::time::SystemTime,
    /// Last modified timestamp
    pub last_modified: std::time::SystemTime,
    /// Total size in bytes
    pub size_bytes: usize,
    /// Access count
    pub access_count: u64,
    /// Configuration used
    pub config: StorageConfig,
}

// Placeholder handle types for storage engines
// These will be defined in their respective modules
pub mod handles {
    /// Handle for column store operations
    #[derive(Debug, Clone)]
    pub struct ColumnStoreHandle {
        pub id: usize,
    }

    /// Handle for memory mapped operations
    #[derive(Debug, Clone)]
    pub struct MemoryMappedHandle {
        pub id: usize,
    }

    /// Handle for disk storage operations
    #[derive(Debug, Clone)]
    pub struct DiskStorageHandle {
        pub id: usize,
    }

    /// Handle for string pool operations
    #[derive(Debug, Clone)]
    pub struct StringPoolHandle {
        pub id: usize,
    }
}

// Import the handle types for use in StorageHandleInner
pub use handles::*;

/// Unified storage manager that coordinates multiple engines
pub struct UnifiedStorageManager {
    /// Column store engine
    column_store: Option<crate::storage::ColumnStore>,
    /// File-backed engine, created on first use under [`Self::disk_root`]
    disk_storage: Option<crate::storage::disk::DiskStorage>,
    /// Directory the disk engine is rooted at
    disk_root: std::path::PathBuf,
    /// Storage strategy for engine selection
    strategy: Box<dyn StorageStrategy>,
    /// Active storage handles
    handles: HashMap<StorageHandleId, StorageHandle>,
    /// Performance monitor fed by every read and write.
    ///
    /// Behind a `Mutex` because `read_chunk` takes `&self`; the field used to be
    /// a plain `PerformanceMonitor` that nothing ever wrote to.
    monitor: Mutex<PerformanceMonitor>,
    /// Next handle ID
    next_handle_id: StorageHandleId,
}

/// Storage handle identifier
pub type StorageHandleId = usize;

/// Performance monitor for storage operations
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Engine performance metrics
    engine_metrics: HashMap<StorageEngineId, EngineMetrics>,
}

/// Performance metrics for a storage engine
#[derive(Debug, Clone)]
pub struct EngineMetrics {
    /// Average read latency in nanoseconds
    pub avg_read_latency: u64,
    /// Average write latency in nanoseconds
    pub avg_write_latency: u64,
    /// Throughput in bytes per second
    pub throughput: u64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            engine_metrics: HashMap::new(),
        }
    }

    /// Record operation performance
    pub fn record_operation(
        &mut self,
        engine_id: StorageEngineId,
        operation: &Operation,
        latency: std::time::Duration,
    ) {
        // Implementation for recording performance metrics
        let metrics = self
            .engine_metrics
            .entry(engine_id)
            .or_insert_with(|| EngineMetrics {
                avg_read_latency: 0,
                avg_write_latency: 0,
                throughput: 0,
                error_rate: 0.0,
                memory_usage: 0,
            });

        // Update metrics based on operation type
        match operation {
            Operation::Read { .. } => {
                metrics.avg_read_latency =
                    (metrics.avg_read_latency + latency.as_nanos() as u64) / 2;
            }
            Operation::Write { .. } => {
                metrics.avg_write_latency =
                    (metrics.avg_write_latency + latency.as_nanos() as u64) / 2;
            }
        }
    }

    /// Get metrics for an engine
    pub fn get_metrics(&self, engine_id: StorageEngineId) -> Option<&EngineMetrics> {
        self.engine_metrics.get(&engine_id)
    }
}

/// Storage operation types for performance monitoring
#[derive(Debug, Clone)]
pub enum Operation {
    Read { size: usize, range: Range<usize> },
    Write { size: usize, compressed: bool },
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl UnifiedStorageManager {
    /// Create a new unified storage manager.
    ///
    /// The file-backed engine is rooted at a process-specific directory under
    /// [`std::env::temp_dir`]; use [`UnifiedStorageManager::with_disk_root`] to
    /// place it somewhere durable.
    pub fn new() -> Self {
        static NEXT_ROOT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
        let root = std::env::temp_dir().join(format!(
            "pandrs_unified_storage_{}_{}",
            std::process::id(),
            NEXT_ROOT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        Self::with_disk_root(root)
    }

    /// Create a manager whose file-backed engine lives under `disk_root`.
    pub fn with_disk_root<P: Into<std::path::PathBuf>>(disk_root: P) -> Self {
        Self {
            column_store: Some(crate::storage::ColumnStore::new()),
            disk_storage: None,
            disk_root: disk_root.into(),
            strategy: Box::new(DefaultStorageStrategy::new()),
            handles: HashMap::new(),
            monitor: Mutex::new(PerformanceMonitor::new()),
            next_handle_id: 1,
        }
    }

    /// The file-backed engine, created on first use.
    fn disk_engine(&mut self) -> Result<&mut crate::storage::disk::DiskStorage> {
        if self.disk_storage.is_none() {
            self.disk_storage = Some(crate::storage::disk::DiskStorage::new(&self.disk_root)?);
        }
        self.disk_storage.as_mut().ok_or_else(|| {
            Error::InvalidOperation("Disk storage engine is unavailable".to_string())
        })
    }

    /// Metrics observed for `engine_id`, or `None` if it has served no
    /// operation yet.
    pub fn engine_metrics(&self, engine_id: StorageEngineId) -> Option<EngineMetrics> {
        self.monitor
            .lock()
            .ok()
            .and_then(|monitor| monitor.get_metrics(engine_id).cloned())
    }

    /// Record one completed operation, ignoring a poisoned monitor lock rather
    /// than failing the storage operation that already succeeded.
    fn record(
        &self,
        engine_id: StorageEngineId,
        operation: &Operation,
        latency: std::time::Duration,
    ) {
        match self.monitor.lock() {
            Ok(mut monitor) => monitor.record_operation(engine_id, operation, latency),
            Err(_) => log::warn!("Storage performance monitor lock is poisoned; sample dropped"),
        }
    }

    /// Create storage using optimal engine selection
    pub fn create_storage(
        &mut self,
        requirements: &StorageRequirements,
    ) -> Result<StorageHandleId> {
        // Select optimal engine
        let engine_id = self.strategy.select_engine(requirements);

        // For now, we only support column store
        let inner_handle = match engine_id {
            StorageEngineId::ColumnStore => {
                if let Some(ref mut engine) = self.column_store {
                    let handle = engine.create_storage(&requirements.config)?;
                    StorageHandleInner::ColumnStore(handle)
                } else {
                    return Err(Error::InvalidOperation(
                        "Column store not available for create_storage".to_string(),
                    ));
                }
            }
            StorageEngineId::DiskStorage => {
                let config = requirements.config.clone();
                let handle = self.disk_engine()?.create_storage(&config)?;
                StorageHandleInner::DiskStorage(handle)
            }
            // `MemoryMapped` maps an *existing* file read-only (see
            // `storage::memory_mapped`) and `StringPool`/`Hybrid` are reached
            // through `storage::unified_manager`, so neither can back a
            // create-then-write request here.
            _ => {
                return Err(Error::NotImplemented(format!(
                    "Storage engine {:?} cannot create writable storage; use \
                     storage::unified_manager::UnifiedMemoryManager for it",
                    engine_id
                )));
            }
        };

        // Create unified handle
        let handle_id = self.next_handle_id;
        self.next_handle_id += 1;

        let handle = StorageHandle {
            engine_id,
            inner_handle,
            metadata: StorageMetadata {
                created_at: std::time::SystemTime::now(),
                last_accessed: std::time::SystemTime::now(),
                last_modified: std::time::SystemTime::now(),
                size_bytes: 0,
                access_count: 0,
                config: requirements.config.clone(),
            },
        };

        self.handles.insert(handle_id, handle);
        Ok(handle_id)
    }

    /// Read chunk from storage
    pub fn read_chunk(&self, handle_id: StorageHandleId, range: Range<usize>) -> Result<DataChunk> {
        let handle = self.handles.get(&handle_id).ok_or_else(|| {
            Error::InvalidOperation("Invalid handle ID for read_chunk".to_string())
        })?;

        let started = std::time::Instant::now();
        let result = match (&handle.engine_id, &handle.inner_handle) {
            (StorageEngineId::ColumnStore, StorageHandleInner::ColumnStore(cs_handle)) => {
                if let Some(ref engine) = self.column_store {
                    engine.read_chunk(cs_handle, range.clone())
                } else {
                    Err(Error::InvalidOperation(
                        "Column store not available for read_chunk".to_string(),
                    ))
                }
            }
            (StorageEngineId::DiskStorage, StorageHandleInner::DiskStorage(disk_handle)) => {
                match self.disk_storage {
                    Some(ref engine) => engine.read_chunk(disk_handle, range.clone()),
                    None => Err(Error::InvalidOperation(
                        "Disk storage engine not initialised for read_chunk".to_string(),
                    )),
                }
            }
            _ => Err(Error::NotImplemented(format!(
                "Engine {:?}",
                handle.engine_id
            ))),
        };

        if let Ok(ref chunk) = result {
            self.record(
                handle.engine_id,
                &Operation::Read {
                    size: chunk.data.len(),
                    range,
                },
                started.elapsed(),
            );
        }
        result
    }

    /// Write chunk to storage
    pub fn write_chunk(&mut self, handle_id: StorageHandleId, chunk: DataChunk) -> Result<()> {
        let handle = self.handles.get(&handle_id).ok_or_else(|| {
            Error::InvalidOperation("Invalid handle ID for write_chunk".to_string())
        })?;

        let engine_id = handle.engine_id;
        let written = chunk.data.len();
        let compressed = chunk.metadata.compression != CompressionPreference::None;
        let started = std::time::Instant::now();
        let result = match (&handle.engine_id, &handle.inner_handle) {
            (StorageEngineId::ColumnStore, StorageHandleInner::ColumnStore(cs_handle)) => {
                if let Some(ref mut engine) = self.column_store {
                    engine.write_chunk(cs_handle, chunk)
                } else {
                    Err(Error::InvalidOperation(
                        "Column store not available for write_chunk".to_string(),
                    ))
                }
            }
            (StorageEngineId::DiskStorage, StorageHandleInner::DiskStorage(disk_handle)) => {
                match self.disk_storage {
                    Some(ref mut engine) => engine.write_chunk(disk_handle, chunk),
                    None => Err(Error::InvalidOperation(
                        "Disk storage engine not initialised for write_chunk".to_string(),
                    )),
                }
            }
            _ => Err(Error::NotImplemented(format!(
                "Engine {:?}",
                handle.engine_id
            ))),
        };

        // Update handle metadata on success
        if result.is_ok() {
            self.record(
                engine_id,
                &Operation::Write {
                    size: written,
                    compressed,
                },
                started.elapsed(),
            );
            if let Some(handle) = self.handles.get_mut(&handle_id) {
                handle.metadata.last_modified = std::time::SystemTime::now();
                handle.metadata.access_count += 1;
            }
        }

        result
    }
}

/// Default storage strategy implementation
pub struct DefaultStorageStrategy {
    /// Engine preferences based on data characteristics
    preferences: HashMap<AccessPattern, StorageEngineId>,
}

impl DefaultStorageStrategy {
    /// Create a new default storage strategy
    pub fn new() -> Self {
        let mut preferences = HashMap::new();
        preferences.insert(AccessPattern::Sequential, StorageEngineId::ColumnStore);
        // Random access used to be routed to `MemoryMapped`, which maps an
        // *existing* file read-only and therefore cannot serve
        // `create_storage`: every random-access request failed with
        // `NotImplemented`. The in-memory column store is the fastest engine
        // here that can actually be created and written.
        preferences.insert(AccessPattern::Random, StorageEngineId::ColumnStore);
        preferences.insert(AccessPattern::ReadHeavy, StorageEngineId::ColumnStore);
        preferences.insert(AccessPattern::WriteHeavy, StorageEngineId::DiskStorage);
        preferences.insert(AccessPattern::Streaming, StorageEngineId::DiskStorage);
        preferences.insert(AccessPattern::Columnar, StorageEngineId::ColumnStore);

        Self { preferences }
    }
}

impl StorageStrategy for DefaultStorageStrategy {
    fn select_engine(&self, requirements: &StorageRequirements) -> StorageEngineId {
        // Simple strategy: select based on access pattern
        self.preferences
            .get(&requirements.config.access_pattern)
            .copied()
            .unwrap_or(StorageEngineId::ColumnStore)
    }

    fn should_migrate(
        &self,
        _handle: &StorageHandle,
        _new_requirements: &StorageRequirements,
    ) -> bool {
        // For now, don't migrate automatically
        false
    }

    fn migrate_storage(
        &mut self,
        _from: &StorageHandle,
        _to: &StorageEngineId,
    ) -> Result<StorageHandle> {
        // Migration not implemented yet
        Err(Error::NotImplemented("Storage migration".to_string()))
    }
}

impl Default for DefaultStorageStrategy {
    fn default() -> Self {
        Self::new()
    }
}

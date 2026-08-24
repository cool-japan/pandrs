//! Unified Memory Management System for PandRS
//!
//! This module provides a comprehensive, pluggable memory management interface
//! with adaptive storage strategy selection as specified in the memory management
//! unification strategy document.

use crate::core::error::{Error, Result};
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

/// Storage type enumeration for strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageType {
    /// Columnar storage with compression
    ColumnStore,
    /// Memory-mapped file storage
    MemoryMapped,
    /// String pool with deduplication
    StringPool,
    /// Hybrid large-scale with tiering
    HybridLargeScale,
    /// Disk-based storage
    DiskBased,
    /// In-memory optimized storage
    InMemory,
}

/// Access pattern hints for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access pattern
    Sequential,
    /// Random access pattern
    Random,
    /// Streaming access pattern
    Streaming,
    /// Columnar access pattern
    Columnar,
    /// High temporal locality
    HighLocality,
    /// Medium temporal locality
    MediumLocality,
    /// Low temporal locality
    LowLocality,
    /// High duplication in data
    HighDuplication,
    /// Low duplication in data
    LowDuplication,
    /// Long strings predominant
    LongStrings,
    /// Short strings predominant
    ShortStrings,
    /// Temporal hot spot pattern
    TemporalHotSpot,
    /// Cold archival pattern
    ColdArchival,
    /// Strided access with specific stride
    Strided { stride: usize },
}

/// Performance priority specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformancePriority {
    /// Optimize for speed
    Speed,
    /// Optimize for memory usage
    Memory,
    /// Balanced optimization
    Balanced,
    /// Optimize for throughput
    Throughput,
    /// Optimize for latency
    Latency,
}

/// Durability level specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityLevel {
    /// No persistence required
    Temporary,
    /// Session persistence
    Session,
    /// Durable storage
    Durable,
    /// Highly durable with replication
    HighDurability,
}

/// Compression preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionPreference {
    /// No compression
    None,
    /// Automatic compression selection
    Auto,
    /// Fast compression
    Fast,
    /// High compression ratio
    High,
    /// Balanced compression
    Balanced,
}

/// Concurrency level specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyLevel {
    /// Single-threaded access
    Single,
    /// Low concurrency
    Low,
    /// Medium concurrency
    Medium,
    /// High concurrency
    High,
    /// Very high concurrency
    VeryHigh,
}

/// I/O pattern specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoPattern {
    /// Read-heavy workload
    ReadHeavy,
    /// Write-heavy workload
    WriteHeavy,
    /// Balanced read/write
    Balanced,
    /// Append-only pattern
    AppendOnly,
    /// Update-in-place pattern
    UpdateInPlace,
}

/// Data characteristics for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataCharacteristics {
    /// Numeric data
    Numeric,
    /// String data
    Text,
    /// Mixed data types
    Mixed,
    /// Time series data
    TimeSeries,
    /// Categorical data
    Categorical,
    /// Sparse data
    Sparse,
    /// Dense data
    Dense,
}

/// Storage requirements specification for strategy selection
#[derive(Debug, Clone)]
pub struct StorageRequirements {
    /// Expected data size in bytes
    pub estimated_size: usize,
    /// Access pattern hint
    pub access_pattern: AccessPattern,
    /// Performance priority (speed vs memory)
    pub performance_priority: PerformancePriority,
    /// Durability requirements
    pub durability: DurabilityLevel,
    /// Compression preferences
    pub compression: CompressionPreference,
    /// Concurrency requirements
    pub concurrency: ConcurrencyLevel,
    /// Memory constraints
    pub memory_limit: Option<usize>,
    /// I/O pattern expectations
    pub io_pattern: IoPattern,
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
}

impl Default for StorageRequirements {
    fn default() -> Self {
        Self {
            estimated_size: 1024 * 1024, // 1MB default
            access_pattern: AccessPattern::Random,
            performance_priority: PerformancePriority::Balanced,
            durability: DurabilityLevel::Temporary,
            compression: CompressionPreference::Auto,
            concurrency: ConcurrencyLevel::Medium,
            memory_limit: None,
            io_pattern: IoPattern::Balanced,
            data_characteristics: DataCharacteristics::Mixed,
        }
    }
}

/// Storage configuration for creating storage
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Storage requirements
    pub requirements: StorageRequirements,
    /// Additional configuration options
    pub options: HashMap<String, String>,
    /// Data sample for analysis (first 100 rows or similar)
    pub data_sample: Option<Vec<u8>>,
    /// Expected access pattern
    pub expected_access_pattern: AccessPattern,
    /// Constraints
    pub constraints: StorageConstraints,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            requirements: StorageRequirements::default(),
            options: HashMap::new(),
            data_sample: None,
            expected_access_pattern: AccessPattern::Random,
            constraints: StorageConstraints::default(),
        }
    }
}

/// Storage constraints
#[derive(Debug, Clone)]
pub struct StorageConstraints {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Maximum disk usage in bytes
    pub max_disk: Option<usize>,
    /// Maximum CPU usage percentage
    pub max_cpu_percent: Option<f64>,
    /// Required availability level
    pub availability_requirement: f64,
}

impl Default for StorageConstraints {
    fn default() -> Self {
        Self {
            max_memory: None,
            max_disk: None,
            max_cpu_percent: Some(80.0),
            availability_requirement: 0.99,
        }
    }
}

/// Physical layout of the bytes inside a [`DataChunk`].
///
/// The layout is what makes the *row* addressing used by [`ChunkRange`]
/// well-defined: it tells a storage strategy how to map a row index onto the
/// byte buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkLayout {
    /// Opaque binary payload. One row is exactly one byte, so `row_count`
    /// always equals `data.len()` and row ranges are byte ranges.
    Opaque,
    /// Length-prefixed UTF-8 strings. One row is one string.
    ///
    /// Wire format (all integers little-endian):
    /// `u64 row_count`, then per row `u64 byte_len` followed by the UTF-8
    /// bytes. NUL bytes inside a string are preserved because the length
    /// prefix — not a separator — delimits the rows.
    Strings,
}

/// Number of bytes used by the `row_count` header of a `Strings` payload.
const STRINGS_HEADER_LEN: usize = 8;
/// Number of bytes used by each per-row length prefix of a `Strings` payload.
const STRINGS_LEN_PREFIX: usize = 8;

/// Data chunk for read/write operations
///
/// A chunk carries both the raw bytes and the row structure needed to honour
/// the row-indexed [`ChunkRange`] contract shared by every storage strategy.
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Raw data bytes
    pub data: Vec<u8>,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
}

impl DataChunk {
    /// Create an opaque byte chunk. One row == one byte.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            metadata: ChunkMetadata::new(&data, data.len(), ChunkLayout::Opaque),
            data,
        }
    }

    /// Create a chunk from an already-encoded payload with an explicit layout
    /// and row count. Used by storage strategies that reassemble chunks from
    /// their own on-disk representation.
    pub fn from_encoded(data: Vec<u8>, layout: ChunkLayout, row_count: usize) -> Result<Self> {
        if layout == ChunkLayout::Opaque && row_count != data.len() {
            return Err(Error::InvalidOperation(format!(
                "Opaque chunk row count {} does not match byte length {}",
                row_count,
                data.len()
            )));
        }
        Ok(Self {
            metadata: ChunkMetadata::new(&data, row_count, layout),
            data,
        })
    }

    /// Size of the chunk payload in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of logical rows carried by this chunk.
    pub fn rows(&self) -> usize {
        self.metadata.row_count
    }

    /// Physical layout of this chunk's payload.
    pub fn layout(&self) -> ChunkLayout {
        self.metadata.layout
    }

    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }

    /// Encode a list of strings with an explicit length prefix per row.
    ///
    /// The previous implementation joined the strings with `"\0"`; because Rust
    /// `String`s may legally contain NUL bytes that silently split rows, and an
    /// empty input round-tripped as one empty row instead of zero rows.
    pub fn from_strings(strings: Vec<String>) -> Self {
        let payload_len: usize = strings.iter().map(|s| STRINGS_LEN_PREFIX + s.len()).sum();
        let mut data = Vec::with_capacity(STRINGS_HEADER_LEN + payload_len);
        data.extend_from_slice(&(strings.len() as u64).to_le_bytes());
        for s in &strings {
            data.extend_from_slice(&(s.len() as u64).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        Self {
            metadata: ChunkMetadata::new(&data, strings.len(), ChunkLayout::Strings),
            data,
        }
    }

    /// Decode a `Strings` chunk back into its rows.
    ///
    /// Returns an error for opaque chunks and for malformed/truncated payloads
    /// rather than silently producing a different number of rows.
    pub fn as_strings(&self) -> Result<Vec<String>> {
        if self.metadata.layout != ChunkLayout::Strings {
            return Err(Error::InvalidOperation(
                "DataChunk does not carry a string layout; use DataChunk::from_strings to build one"
                    .to_string(),
            ));
        }
        decode_strings(&self.data)
    }

    /// Extract rows `[start, end)` as a new chunk of the same layout.
    pub fn slice_rows(&self, start: usize, end: usize) -> Result<DataChunk> {
        let end = end.min(self.rows());
        let start = start.min(end);
        match self.metadata.layout {
            ChunkLayout::Opaque => Ok(DataChunk::new(self.data[start..end].to_vec())),
            ChunkLayout::Strings => {
                let all = self.as_strings()?;
                Ok(DataChunk::from_strings(all[start..end].to_vec()))
            }
        }
    }

    /// Concatenate chunks that share a layout into a single chunk.
    ///
    /// Returns an error for mixed layouts rather than producing a byte blob
    /// whose row structure no longer matches its contents.
    pub fn concat(parts: Vec<DataChunk>) -> Result<DataChunk> {
        let Some(first) = parts.first() else {
            return Ok(DataChunk::new(Vec::new()));
        };
        let layout = first.layout();
        if parts.iter().any(|p| p.layout() != layout) {
            return Err(Error::InvalidOperation(
                "Cannot concatenate chunks with mixed layouts".to_string(),
            ));
        }
        match layout {
            ChunkLayout::Opaque => {
                let mut merged = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
                for part in parts {
                    merged.extend_from_slice(&part.data);
                }
                Ok(DataChunk::new(merged))
            }
            ChunkLayout::Strings => {
                let mut merged = Vec::with_capacity(parts.iter().map(|p| p.rows()).sum());
                for part in parts {
                    merged.extend(part.as_strings()?);
                }
                Ok(DataChunk::from_strings(merged))
            }
        }
    }

    /// Recompute the payload checksum and compare it against the stored one.
    pub fn verify_checksum(&self) -> bool {
        crate::storage::checksum::checksum64(&self.data) == self.metadata.checksum
    }

    /// Build a chunk of `size` zero bytes. Intended for tests and benchmarks.
    pub fn new_test_data(size: usize) -> Self {
        let data = vec![0u8; size];
        Self::new(data)
    }
}

/// Decode a length-prefixed `Strings` payload.
fn decode_strings(data: &[u8]) -> Result<Vec<String>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    if data.len() < STRINGS_HEADER_LEN {
        return Err(Error::InvalidOperation(
            "Truncated string chunk: missing row count header".to_string(),
        ));
    }
    let mut header = [0u8; STRINGS_HEADER_LEN];
    header.copy_from_slice(&data[..STRINGS_HEADER_LEN]);
    let row_count = u64::from_le_bytes(header);
    // Each row costs at least its length prefix, so a row count that cannot
    // possibly fit in the remaining bytes is corrupt input, not a huge alloc.
    let max_rows = (data.len() - STRINGS_HEADER_LEN) / STRINGS_LEN_PREFIX;
    if row_count as usize > max_rows {
        return Err(Error::InvalidOperation(format!(
            "Corrupt string chunk: header claims {} rows but only {} can fit in {} bytes",
            row_count,
            max_rows,
            data.len()
        )));
    }
    let row_count = row_count as usize;

    let mut strings = Vec::with_capacity(row_count);
    let mut offset = STRINGS_HEADER_LEN;
    for row in 0..row_count {
        if offset + STRINGS_LEN_PREFIX > data.len() {
            return Err(Error::InvalidOperation(format!(
                "Truncated string chunk: missing length prefix for row {}",
                row
            )));
        }
        let mut len_bytes = [0u8; STRINGS_LEN_PREFIX];
        len_bytes.copy_from_slice(&data[offset..offset + STRINGS_LEN_PREFIX]);
        offset += STRINGS_LEN_PREFIX;
        let len = u64::from_le_bytes(len_bytes) as usize;
        if offset + len > data.len() {
            return Err(Error::InvalidOperation(format!(
                "Truncated string chunk: row {} claims {} bytes, {} remain",
                row,
                len,
                data.len() - offset
            )));
        }
        let s = std::str::from_utf8(&data[offset..offset + len]).map_err(|e| {
            Error::InvalidOperation(format!("Invalid UTF-8 in string chunk row {}: {}", row, e))
        })?;
        strings.push(s.to_string());
        offset += len;
    }
    Ok(strings)
}

/// Chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    /// Size in bytes
    pub size: usize,
    /// Number of logical rows in the payload
    pub row_count: usize,
    /// Physical layout of the payload
    pub layout: ChunkLayout,
    /// CRC-32C based integrity checksum of the payload
    pub checksum: u64,
    /// Compression type used
    pub compression: CompressionType,
    /// Creation timestamp
    pub created_at: Instant,
}

impl ChunkMetadata {
    fn new(data: &[u8], row_count: usize, layout: ChunkLayout) -> Self {
        Self {
            size: data.len(),
            row_count,
            layout,
            checksum: crate::storage::checksum::checksum64(data),
            compression: CompressionType::None,
            created_at: Instant::now(),
        }
    }
}

/// Compression type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionType {
    None,
    Auto,
    Lz4,
    Zstd,
    Snappy,
    Gzip,
}

/// Chunk range specification.
///
/// # Semantics (contract for every [`StorageStrategy`])
///
/// `start` and `end` are **row indices**, half-open (`start..end`), into the
/// logical append-ordered stream of rows written to one storage handle. Row 0
/// is the first row of the first `write_chunk`/`append_chunk`; a chunk of `n`
/// rows advances the stream by `n`.
///
/// What a "row" is comes from [`ChunkLayout`]: for `ChunkLayout::Opaque`
/// payloads one row is one byte (so row ranges coincide with byte ranges), for
/// `ChunkLayout::Strings` payloads one row is one string.
///
/// Before this was pinned down, the three shipped strategies each interpreted
/// the range differently (byte offsets, string-id range, single data id), so
/// the same range read different data depending on which strategy happened to
/// be selected.
#[derive(Debug, Clone)]
pub struct ChunkRange {
    /// First row index (inclusive)
    pub start: usize,
    /// Last row index (exclusive)
    pub end: usize,
}

impl ChunkRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    pub fn full() -> Self {
        Self {
            start: 0,
            end: usize::MAX,
        }
    }
}

impl From<Range<usize>> for ChunkRange {
    fn from(range: Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

/// Strategy capability assessment
#[derive(Debug, Clone)]
pub struct StrategyCapability {
    /// Can handle the requirements
    pub can_handle: bool,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated performance score
    pub performance_score: f64,
    /// Resource cost estimate
    pub resource_cost: ResourceCost,
}

/// Resource cost estimate
#[derive(Debug, Clone)]
pub struct ResourceCost {
    /// Memory cost in bytes
    pub memory: usize,
    /// CPU cost percentage
    pub cpu: f64,
    /// Disk space cost in bytes
    pub disk: usize,
    /// Network bandwidth cost in bytes/sec
    pub network: usize,
}

/// Performance profile for strategy
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Read operation speed
    pub read_speed: Speed,
    /// Write operation speed
    pub write_speed: Speed,
    /// Memory efficiency
    pub memory_efficiency: Efficiency,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Query optimization capability
    pub query_optimization: QueryOptimization,
    /// Parallel scalability
    pub parallel_scalability: ParallelScalability,
}

/// Speed enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speed {
    VerySlow,
    Slow,
    Medium,
    Fast,
    VeryFast,
}

/// Efficiency enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Efficiency {
    Poor,
    Fair,
    Good,
    Excellent,
    Outstanding,
}

/// Query optimization capability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryOptimization {
    None,
    Basic,
    Good,
    Excellent,
}

/// Parallel scalability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelScalability {
    None,
    Limited,
    Good,
    Excellent,
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total storage size in bytes
    pub total_size: usize,
    /// Used storage size in bytes
    pub used_size: usize,
    /// Number of read operations
    pub read_operations: u64,
    /// Number of write operations
    pub write_operations: u64,
    /// Average read latency in nanoseconds
    pub avg_read_latency_ns: u64,
    /// Average write latency in nanoseconds
    pub avg_write_latency_ns: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_size: 0,
            used_size: 0,
            read_operations: 0,
            write_operations: 0,
            avg_read_latency_ns: 0,
            avg_write_latency_ns: 0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Base trait for all storage strategies in PandRS
pub trait StorageStrategy: Send + Sync {
    type Handle;
    type Error: std::error::Error + Send + Sync + 'static;
    type Metadata: Clone + Send + Sync;

    /// Strategy identifier for selection and monitoring
    fn name(&self) -> &'static str;

    /// Create new storage with specific configuration
    fn create_storage(
        &mut self,
        config: &StorageConfig,
    ) -> std::result::Result<Self::Handle, Self::Error>;

    /// Read data chunk from storage
    fn read_chunk(
        &self,
        handle: &Self::Handle,
        range: ChunkRange,
    ) -> std::result::Result<DataChunk, Self::Error>;

    /// Write data chunk to storage
    fn write_chunk(
        &mut self,
        handle: &Self::Handle,
        chunk: DataChunk,
    ) -> std::result::Result<(), Self::Error>;

    /// Append data chunk to existing storage
    fn append_chunk(
        &mut self,
        handle: &Self::Handle,
        chunk: DataChunk,
    ) -> std::result::Result<(), Self::Error>;

    /// Flush pending writes to persistent storage
    fn flush(&mut self, handle: &Self::Handle) -> std::result::Result<(), Self::Error>;

    /// Delete storage and free resources
    fn delete_storage(&mut self, handle: &Self::Handle) -> std::result::Result<(), Self::Error>;

    /// Check if strategy can handle specific requirements
    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability;

    /// Get performance characteristics of this strategy
    fn performance_profile(&self) -> PerformanceProfile;

    /// Get current memory and storage statistics
    fn storage_stats(&self) -> StorageStats;

    /// Optimize strategy for specific access pattern
    fn optimize_for_pattern(
        &mut self,
        pattern: AccessPattern,
    ) -> std::result::Result<(), Self::Error>;

    /// Compact storage to reduce fragmentation
    fn compact(
        &mut self,
        handle: &Self::Handle,
    ) -> std::result::Result<CompactionResult, Self::Error>;
}

/// Compaction result
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Size before compaction
    pub size_before: usize,
    /// Size after compaction
    pub size_after: usize,
    /// Time taken for compaction
    pub duration: std::time::Duration,
}

/// Storage handle with metadata and resource tracking
#[derive(Debug)]
pub struct StorageHandle {
    /// Unique identifier for this storage
    pub id: StorageId,
    /// Strategy that manages this storage
    pub strategy_type: StorageType,
    /// Strategy-specific handle
    pub inner_handle: Box<dyn Any + Send + Sync>,
    /// Storage metadata
    pub metadata: StorageMetadata,
    /// Reference counting for resource management
    pub ref_count: Arc<AtomicUsize>,
    /// Performance monitoring data
    pub performance_tracker: PerformanceTracker,
}

impl StorageHandle {
    pub fn new(
        id: StorageId,
        strategy_type: StorageType,
        inner_handle: Box<dyn Any + Send + Sync>,
        metadata: StorageMetadata,
    ) -> Self {
        Self {
            id,
            strategy_type,
            inner_handle,
            metadata,
            ref_count: Arc::new(AtomicUsize::new(1)),
            performance_tracker: PerformanceTracker::new(),
        }
    }
}

// Note: StorageHandle cannot implement Clone due to the inner_handle trait object
// Use Arc<StorageHandle> if shared ownership is needed

impl Drop for StorageHandle {
    /// Release this handle's share of the reference count.
    ///
    /// Dropping a handle deliberately does **not** delete the underlying
    /// storage: strategies own their bytes and are torn down explicitly through
    /// [`StorageStrategy::delete_storage`], which the
    /// [`crate::storage::unified_manager::UnifiedMemoryManager`] routes to the
    /// owning strategy. `ref_count` reaching zero here only records that no
    /// handle refers to the storage any more.
    fn drop(&mut self) {
        self.ref_count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Storage identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StorageId(pub u64);

/// Storage metadata
#[derive(Debug, Clone)]
pub struct StorageMetadata {
    /// Creation timestamp
    pub created_at: Instant,
    /// Last accessed timestamp
    pub last_accessed: Instant,
    /// Total size in bytes
    pub size: usize,
    /// Access count
    pub access_count: u64,
}

impl StorageMetadata {
    pub fn new(size: usize) -> Self {
        let now = Instant::now();
        Self {
            created_at: now,
            last_accessed: now,
            size,
            access_count: 0,
        }
    }
}

/// Performance tracker for monitoring storage operations
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Read operation times
    pub read_times: Vec<std::time::Duration>,
    /// Write operation times
    pub write_times: Vec<std::time::Duration>,
    /// Total bytes read
    pub bytes_read: u64,
    /// Total bytes written
    pub bytes_written: u64,
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            read_times: Vec::new(),
            write_times: Vec::new(),
            bytes_read: 0,
            bytes_written: 0,
        }
    }

    pub fn record_read(&mut self, duration: std::time::Duration, bytes: u64) {
        self.read_times.push(duration);
        self.bytes_read += bytes;
    }

    pub fn record_write(&mut self, duration: std::time::Duration, bytes: u64) {
        self.write_times.push(duration);
        self.bytes_written += bytes;
    }

    pub fn average_read_time(&self) -> Option<std::time::Duration> {
        if self.read_times.is_empty() {
            None
        } else {
            let total: std::time::Duration = self.read_times.iter().sum();
            Some(total / self.read_times.len() as u32)
        }
    }

    pub fn average_write_time(&self) -> Option<std::time::Duration> {
        if self.write_times.is_empty() {
            None
        } else {
            let total: std::time::Duration = self.write_times.iter().sum();
            Some(total / self.write_times.len() as u32)
        }
    }
}

/// Atomic memory statistics
#[derive(Debug)]
pub struct AtomicMemoryStats {
    /// Total allocated memory
    pub total_allocated: AtomicUsize,
    /// Peak memory usage
    pub peak_usage: AtomicUsize,
    /// Current active allocations
    pub active_allocations: AtomicUsize,
    /// Total number of allocation operations
    pub allocation_count: AtomicUsize,
    /// Total number of deallocation operations
    pub deallocation_count: AtomicUsize,
}

impl AtomicMemoryStats {
    pub fn new() -> Self {
        Self {
            total_allocated: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            active_allocations: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
        }
    }

    pub fn record_allocation(&self, size: usize) {
        self.total_allocated.fetch_add(size, Ordering::SeqCst);
        self.active_allocations.fetch_add(1, Ordering::SeqCst);
        self.allocation_count.fetch_add(1, Ordering::SeqCst);

        // Update peak usage
        let current = self.total_allocated.load(Ordering::SeqCst);
        self.peak_usage.fetch_max(current, Ordering::SeqCst);
    }

    pub fn record_deallocation(&self, size: usize) {
        self.total_allocated.fetch_sub(size, Ordering::SeqCst);
        self.active_allocations.fetch_sub(1, Ordering::SeqCst);
        self.deallocation_count.fetch_add(1, Ordering::SeqCst);
    }
}

impl Default for AtomicMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_requirements_default() {
        let req = StorageRequirements::default();
        assert_eq!(req.estimated_size, 1024 * 1024);
        assert_eq!(req.performance_priority, PerformancePriority::Balanced);
    }

    #[test]
    fn test_data_chunk() {
        let data = vec![1, 2, 3, 4, 5];
        let chunk = DataChunk::new(data.clone());
        assert_eq!(chunk.len(), 5);
        assert_eq!(chunk.data, data);
        // Opaque payload: one row per byte.
        assert_eq!(chunk.rows(), 5);
        assert_eq!(chunk.layout(), ChunkLayout::Opaque);
        assert!(chunk.verify_checksum());
    }

    #[test]
    fn string_chunk_roundtrip_preserves_nul_and_unicode() {
        let strings = vec![
            "plain".to_string(),
            "with\0embedded\0nul".to_string(),
            "日本語テキスト".to_string(),
            String::new(),
            "€ £ ¥".to_string(),
        ];
        let chunk = DataChunk::from_strings(strings.clone());
        assert_eq!(chunk.rows(), strings.len());
        assert_eq!(chunk.layout(), ChunkLayout::Strings);
        assert_eq!(chunk.as_strings().expect("decode"), strings);
        assert!(chunk.verify_checksum());
    }

    #[test]
    fn empty_string_chunk_has_zero_rows() {
        let chunk = DataChunk::from_strings(Vec::new());
        assert_eq!(chunk.rows(), 0);
        assert!(chunk.as_strings().expect("decode").is_empty());
    }

    #[test]
    fn truncated_string_chunk_errors() {
        let chunk = DataChunk::from_strings(vec!["abcdef".to_string()]);
        let truncated = DataChunk::from_encoded(
            chunk.data[..chunk.data.len() - 2].to_vec(),
            ChunkLayout::Strings,
            1,
        )
        .expect("construct");
        assert!(truncated.as_strings().is_err());
    }

    #[test]
    fn corrupt_row_count_is_rejected_without_huge_alloc() {
        // Header claims u64::MAX rows in a 12-byte buffer.
        let mut data = u64::MAX.to_le_bytes().to_vec();
        data.extend_from_slice(&[0u8; 4]);
        let chunk = DataChunk::from_encoded(data, ChunkLayout::Strings, 0).expect("construct");
        assert!(chunk.as_strings().is_err());
    }

    #[test]
    fn slice_rows_works_for_both_layouts() {
        let opaque = DataChunk::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(opaque.slice_rows(1, 4).expect("slice").data, vec![2, 3, 4]);

        let strings = DataChunk::from_strings(vec![
            "a".to_string(),
            "bb".to_string(),
            "ccc".to_string(),
            "dddd".to_string(),
        ]);
        let sliced = strings.slice_rows(1, 3).expect("slice");
        assert_eq!(
            sliced.as_strings().expect("decode"),
            vec!["bb".to_string(), "ccc".to_string()]
        );
    }

    #[test]
    fn as_strings_rejects_opaque_layout() {
        let chunk = DataChunk::new(b"raw bytes".to_vec());
        assert!(chunk.as_strings().is_err());
    }

    #[test]
    fn test_chunk_range() {
        let range = ChunkRange::new(10, 20);
        assert_eq!(range.len(), 10);
        assert!(!range.is_empty());

        let empty_range = ChunkRange::new(20, 10);
        assert!(empty_range.is_empty());
    }

    #[test]
    fn test_storage_handle_creation() {
        let handle = StorageHandle::new(
            StorageId(1),
            StorageType::InMemory,
            Box::new(42u32),
            StorageMetadata::new(1024),
        );

        assert_eq!(handle.ref_count.load(Ordering::SeqCst), 1);
        assert_eq!(handle.id, StorageId(1));
        assert_eq!(handle.strategy_type, StorageType::InMemory);
        assert_eq!(handle.metadata.size, 1024);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();

        tracker.record_read(std::time::Duration::from_millis(10), 1024);
        tracker.record_read(std::time::Duration::from_millis(20), 2048);

        assert_eq!(tracker.bytes_read, 3072);
        let avg_time = tracker
            .average_read_time()
            .expect("operation should succeed");
        assert_eq!(avg_time, std::time::Duration::from_millis(15));
    }

    #[test]
    fn test_atomic_memory_stats() {
        let stats = AtomicMemoryStats::new();

        stats.record_allocation(1024);
        assert_eq!(stats.total_allocated.load(Ordering::SeqCst), 1024);
        assert_eq!(stats.active_allocations.load(Ordering::SeqCst), 1);

        stats.record_allocation(2048);
        assert_eq!(stats.total_allocated.load(Ordering::SeqCst), 3072);
        assert_eq!(stats.peak_usage.load(Ordering::SeqCst), 3072);

        stats.record_deallocation(1024);
        assert_eq!(stats.total_allocated.load(Ordering::SeqCst), 2048);
        assert_eq!(stats.active_allocations.load(Ordering::SeqCst), 1);
    }
}

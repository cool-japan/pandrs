//! `UnifiedColumnStoreStrategy`: the row-addressed columnar storage strategy.

use crate::core::error::{Error, Result};
use crate::storage::unified_column_store::blocks::{
    BlockId, BlockManager, BlockMetadata, CompressedBlock, InMemoryPhysicalStorage,
};
use crate::storage::unified_column_store::compression::{
    build_engines, resolve_compression, CompressionEngine,
};
use crate::storage::unified_column_store::encoding::{
    build_encodings, EncodedData, EncodingStrategy, EncodingType,
};
use crate::storage::unified_memory::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// Column store configuration
#[derive(Debug, Clone)]
pub struct ColumnStoreConfig {
    /// Default compression type
    pub compression_type: CompressionType,
    /// Default encoding type
    pub encoding_type: EncodingType,
    /// Block size for chunking data
    pub block_size: usize,
    /// Enable dictionary encoding
    pub enable_dictionary: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Capacity of the block cache, in bytes
    pub metadata_cache_size: usize,
    /// ZSTD compression level
    pub zstd_level: i32,
    /// Upper bound on the in-memory backing store, in bytes
    pub max_storage_size: usize,
}

impl Default for ColumnStoreConfig {
    fn default() -> Self {
        Self {
            compression_type: CompressionType::Zstd,
            encoding_type: EncodingType::Auto,
            block_size: 64 * 1024, // 64KB blocks
            enable_dictionary: true,
            enable_parallel: true,
            metadata_cache_size: 10 * 1024 * 1024, // 10MB block cache
            zstd_level: 3,
            max_storage_size: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }
}

/// Column layout information
#[derive(Debug, Clone)]
pub struct ColumnLayout {
    pub name: String,
    pub data_type: ColumnDataType,
    pub nullable: bool,
    pub block_size: usize,
    pub total_size: usize,
    pub row_count: u64,
}

/// Column data types for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    String,
    Binary,
    Boolean,
    Timestamp,
    Date,
    Decimal,
}

impl ColumnDataType {
    /// Fixed element width in bytes, if this type has one.
    pub fn element_width(&self) -> Option<usize> {
        match self {
            ColumnDataType::Int8 | ColumnDataType::UInt8 | ColumnDataType::Boolean => Some(1),
            ColumnDataType::Int16 | ColumnDataType::UInt16 => Some(2),
            ColumnDataType::Int32
            | ColumnDataType::UInt32
            | ColumnDataType::Float32
            | ColumnDataType::Date => Some(4),
            ColumnDataType::Int64
            | ColumnDataType::UInt64
            | ColumnDataType::Float64
            | ColumnDataType::Timestamp => Some(8),
            ColumnDataType::Decimal => Some(16),
            ColumnDataType::String | ColumnDataType::Binary => None,
        }
    }
}

/// Column statistics for query optimization
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    pub null_count: u64,
    pub distinct_count: Option<u64>,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
    pub total_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub encoding_ratio: f64,
}

impl ColumnStatistics {
    pub fn new() -> Self {
        Self {
            null_count: 0,
            distinct_count: None,
            min_value: None,
            max_value: None,
            total_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
            encoding_ratio: 1.0,
        }
    }
}

impl Default for ColumnStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// One block belonging to a handle, together with the row range it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockRef {
    /// Physical block identifier
    pub id: BlockId,
    /// First logical row stored in this block (inclusive)
    pub row_start: usize,
    /// Number of logical rows stored in this block
    pub row_count: usize,
    /// Layout of the rows in this block
    pub layout: ChunkLayout,
}

impl BlockRef {
    fn row_end(&self) -> usize {
        self.row_start + self.row_count
    }
}

/// Column store handle.
///
/// The block index is shared and interior-mutable: `StorageStrategy::write_chunk`
/// only receives `&Handle`, so a plain `Vec<BlockId>` field could never be
/// appended to — which is exactly why every block id used to be discarded and
/// written data could never be read back.
#[derive(Debug, Clone)]
pub struct ColumnStoreHandle {
    pub layout: ColumnLayout,
    pub compression_type: CompressionType,
    pub encoding_type: EncodingType,
    /// Ordered index of the blocks written through this handle
    blocks: Arc<Mutex<Vec<BlockRef>>>,
    /// Total number of rows appended through this handle
    row_cursor: Arc<AtomicU64>,
    pub statistics: ColumnStatistics,
}

impl ColumnStoreHandle {
    pub(crate) fn new(
        layout: ColumnLayout,
        compression_type: CompressionType,
        encoding_type: EncodingType,
    ) -> Self {
        Self {
            layout,
            compression_type,
            encoding_type,
            blocks: Arc::new(Mutex::new(Vec::new())),
            row_cursor: Arc::new(AtomicU64::new(0)),
            statistics: ColumnStatistics::new(),
        }
    }

    fn lock_blocks(&self) -> Result<std::sync::MutexGuard<'_, Vec<BlockRef>>> {
        self.blocks.lock().map_err(|_| {
            Error::InvalidOperation("Column store block index lock is poisoned".to_string())
        })
    }

    /// Every block written through this handle, in write order.
    pub fn block_refs(&self) -> Result<Vec<BlockRef>> {
        Ok(self.lock_blocks()?.clone())
    }

    /// Ids of every block written through this handle, in write order.
    pub fn block_ids(&self) -> Result<Vec<BlockId>> {
        Ok(self.lock_blocks()?.iter().map(|b| b.id).collect())
    }

    /// Total number of rows written through this handle.
    pub fn row_count(&self) -> usize {
        self.row_cursor.load(Ordering::SeqCst) as usize
    }

    fn append_blocks(&self, refs: Vec<BlockRef>) -> Result<()> {
        let mut blocks = self.lock_blocks()?;
        blocks.extend(refs);
        Ok(())
    }

    fn reserve_rows(&self, rows: usize) -> usize {
        self.row_cursor.fetch_add(rows as u64, Ordering::SeqCst) as usize
    }

    fn clear(&self) -> Result<()> {
        self.lock_blocks()?.clear();
        self.row_cursor.store(0, Ordering::SeqCst);
        Ok(())
    }
}

/// Live counters backing [`StorageStrategy::storage_stats`].
#[derive(Debug, Default)]
struct OperationStats {
    read_operations: AtomicU64,
    write_operations: AtomicU64,
    read_nanos: AtomicU64,
    write_nanos: AtomicU64,
}

impl OperationStats {
    fn record_read(&self, elapsed_nanos: u64) {
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        self.read_nanos.fetch_add(elapsed_nanos, Ordering::Relaxed);
    }

    fn record_write(&self, elapsed_nanos: u64) {
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        self.write_nanos.fetch_add(elapsed_nanos, Ordering::Relaxed);
    }

    fn avg_read_nanos(&self) -> u64 {
        let count = self.read_operations.load(Ordering::Relaxed);
        if count == 0 {
            0
        } else {
            self.read_nanos.load(Ordering::Relaxed) / count
        }
    }

    fn avg_write_nanos(&self) -> u64 {
        let count = self.write_operations.load(Ordering::Relaxed);
        if count == 0 {
            0
        } else {
            self.write_nanos.load(Ordering::Relaxed) / count
        }
    }
}

/// Unified Column Store Strategy Implementation
pub struct UnifiedColumnStoreStrategy {
    /// Multiple compression backends
    compression_engines: HashMap<CompressionType, Box<dyn CompressionEngine>>,
    /// Encoding strategies for different data types
    encoding_strategies: HashMap<EncodingType, Box<dyn EncodingStrategy>>,
    /// Block-based storage management
    block_manager: Arc<Mutex<BlockManager>>,
    /// Columnar metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, ColumnStatistics>>>,
    /// Live operation counters
    op_stats: Arc<OperationStats>,
    /// Configuration parameters
    config: ColumnStoreConfig,
}

/// Result of encoding + compressing one block's worth of rows.
struct PreparedBlock {
    block: CompressedBlock,
    row_start: usize,
    row_count: usize,
    layout: ChunkLayout,
}

impl UnifiedColumnStoreStrategy {
    pub fn new(config: ColumnStoreConfig) -> Self {
        let compression_engines = build_engines(config.zstd_level);
        let encoding_strategies = build_encodings();

        let storage = Box::new(InMemoryPhysicalStorage::with_limit(
            config.block_size.saturating_mul(16).max(64 * 1024),
            config.max_storage_size,
        ));
        let block_manager = Arc::new(Mutex::new(BlockManager::with_cache_capacity(
            storage,
            config.metadata_cache_size,
        )));

        Self {
            compression_engines,
            encoding_strategies,
            block_manager,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            op_stats: Arc::new(OperationStats::default()),
            config,
        }
    }

    fn lock_blocks(&self) -> Result<std::sync::MutexGuard<'_, BlockManager>> {
        self.block_manager
            .lock()
            .map_err(|_| Error::InvalidOperation("Block manager lock is poisoned".to_string()))
    }

    fn determine_optimal_layout(&self, config: &StorageConfig) -> Result<ColumnLayout> {
        let data_type = match config.requirements.data_characteristics {
            DataCharacteristics::Numeric => ColumnDataType::Float64,
            DataCharacteristics::Text => ColumnDataType::String,
            DataCharacteristics::TimeSeries => ColumnDataType::Timestamp,
            DataCharacteristics::Categorical => ColumnDataType::String,
            _ => ColumnDataType::Binary,
        };

        // Rows are only knowable once data is written; derive the *estimate*
        // from the declared element width instead of an invented "8 bytes per
        // row" constant, and mark it as an estimate.
        let estimated_rows = match data_type.element_width() {
            Some(width) if width > 0 => (config.requirements.estimated_size / width) as u64,
            _ => 0,
        };

        Ok(ColumnLayout {
            name: config
                .options
                .get("column_name")
                .cloned()
                .unwrap_or_else(|| "default".to_string()),
            data_type,
            nullable: true,
            block_size: self.config.block_size,
            total_size: config.requirements.estimated_size,
            row_count: estimated_rows,
        })
    }

    fn select_compression_strategy(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<CompressionType> {
        let selected = match characteristics {
            DataCharacteristics::Text => CompressionType::Zstd,
            DataCharacteristics::Numeric => CompressionType::Lz4,
            DataCharacteristics::Sparse => CompressionType::Zstd,
            _ => self.config.compression_type,
        };
        Ok(resolve_compression(selected))
    }

    fn select_encoding_strategy(
        &self,
        characteristics: &DataCharacteristics,
    ) -> Result<EncodingType> {
        let selected = match characteristics {
            DataCharacteristics::Categorical if self.config.enable_dictionary => {
                EncodingType::Dictionary
            }
            DataCharacteristics::Sparse => EncodingType::RunLength,
            DataCharacteristics::TimeSeries => EncodingType::Delta,
            _ => self.config.encoding_type,
        };
        // Every selectable encoding must have a registered strategy; otherwise
        // the block would be tagged with an encoding nothing can decode.
        if selected != EncodingType::None
            && selected != EncodingType::Auto
            && !self.encoding_strategies.contains_key(&selected)
        {
            return Err(Error::InvalidOperation(format!(
                "Encoding strategy {:?} is not registered",
                selected
            )));
        }
        Ok(selected)
    }

    /// Blocks overlapping the requested row range, in row order.
    fn find_blocks_for_range(
        &self,
        handle: &ColumnStoreHandle,
        range: &ChunkRange,
    ) -> Result<Vec<BlockRef>> {
        let mut blocks: Vec<BlockRef> = handle
            .block_refs()?
            .into_iter()
            .filter(|b| b.row_count > 0 && b.row_start < range.end && b.row_end() > range.start)
            .collect();
        blocks.sort_by_key(|b| b.row_start);
        Ok(blocks)
    }

    /// Decode one block back into its rows.
    fn decode_block(&self, block: &CompressedBlock) -> Result<Vec<u8>> {
        let compression_engine = self
            .compression_engines
            .get(&block.compression_type)
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Compression engine {:?} not found",
                    block.compression_type
                ))
            })?;
        let decompressed = compression_engine.decompress(&block.data)?;

        if block.encoding_type == EncodingType::None {
            return Ok(decompressed);
        }

        let encoding_strategy = self
            .encoding_strategies
            .get(&block.encoding_type)
            .ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Encoding strategy {:?} not found",
                    block.encoding_type
                ))
            })?;

        encoding_strategy.decode(&EncodedData {
            data: decompressed,
            encoding_type: block.encoding_type,
            original_size: block.uncompressed_size(),
            metadata: HashMap::new(),
        })
    }

    /// Split a chunk into per-block row groups.
    ///
    /// Opaque payloads split on byte (== row) boundaries. String payloads are
    /// grouped so that no row is ever cut in half — splitting a length-prefixed
    /// string blob at an arbitrary byte offset would destroy it.
    fn split_chunk_into_row_groups(&self, chunk: &DataChunk) -> Result<Vec<(Vec<u8>, usize)>> {
        let block_size = self.config.block_size.max(1);
        match chunk.layout() {
            ChunkLayout::Opaque => {
                if chunk.data.is_empty() {
                    return Ok(vec![(Vec::new(), 0)]);
                }
                Ok(chunk
                    .data
                    .chunks(block_size)
                    .map(|part| (part.to_vec(), part.len()))
                    .collect())
            }
            ChunkLayout::Strings => {
                let strings = chunk.as_strings()?;
                if strings.is_empty() {
                    return Ok(vec![(DataChunk::from_strings(Vec::new()).data, 0)]);
                }
                let mut groups = Vec::new();
                let mut current: Vec<String> = Vec::new();
                let mut current_bytes = 0usize;
                for s in strings {
                    let cost = s.len() + 8;
                    if !current.is_empty() && current_bytes + cost > block_size {
                        let rows = current.len();
                        groups.push((
                            DataChunk::from_strings(std::mem::take(&mut current)).data,
                            rows,
                        ));
                        current_bytes = 0;
                    }
                    current_bytes += cost;
                    current.push(s);
                }
                if !current.is_empty() {
                    let rows = current.len();
                    groups.push((DataChunk::from_strings(current).data, rows));
                }
                Ok(groups)
            }
        }
    }

    /// Encode + compress one row group, recording the encoding that was really
    /// applied (never a placeholder such as `Auto`).
    fn compress_and_encode_block(
        &self,
        handle: &ColumnStoreHandle,
        payload: &[u8],
    ) -> Result<CompressedBlock> {
        let (encoded_bytes, applied_encoding) =
            self.encode_payload(handle.encoding_type, payload)?;

        let compression_type = resolve_compression(handle.compression_type);
        let compressed_data = if compression_type == CompressionType::None {
            encoded_bytes
        } else {
            let engine = self
                .compression_engines
                .get(&compression_type)
                .ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "Compression engine {:?} not found",
                        compression_type
                    ))
                })?;
            engine.compress(&encoded_bytes)?
        };

        let mut block = CompressedBlock::new(compressed_data, compression_type, applied_encoding);
        block.metadata.uncompressed_size = payload.len();
        Ok(block)
    }

    /// Apply `requested` (resolving `Auto` by measuring every candidate) and
    /// return the encoded bytes plus the encoding that was actually used.
    fn encode_payload(
        &self,
        requested: EncodingType,
        payload: &[u8],
    ) -> Result<(Vec<u8>, EncodingType)> {
        match requested {
            EncodingType::None => Ok((payload.to_vec(), EncodingType::None)),
            EncodingType::Auto => {
                let mut best: (Vec<u8>, EncodingType) = (payload.to_vec(), EncodingType::None);
                for (&kind, strategy) in &self.encoding_strategies {
                    if !self.config.enable_dictionary && kind == EncodingType::Dictionary {
                        continue;
                    }
                    let encoded = strategy.encode(payload)?;
                    if encoded.data.len() < best.0.len() {
                        best = (encoded.data, kind);
                    }
                }
                Ok(best)
            }
            kind => {
                let strategy = self.encoding_strategies.get(&kind).ok_or_else(|| {
                    Error::InvalidOperation(format!("Encoding strategy {:?} not found", kind))
                })?;
                let encoded = strategy.encode(payload)?;
                Ok((encoded.data, kind))
            }
        }
    }

    fn update_column_statistics(
        &self,
        handle: &ColumnStoreHandle,
        chunk: &DataChunk,
        compressed_bytes: u64,
        encoded_bytes: u64,
    ) -> Result<()> {
        let original_size = chunk.len() as u64;
        let min_max = compute_min_max(&chunk.data, handle.layout.data_type, chunk.layout());

        let mut cache = self.metadata_cache.write().map_err(|_| {
            Error::InvalidOperation("Column statistics cache lock is poisoned".to_string())
        })?;
        let stats = cache
            .entry(handle.layout.name.clone())
            .or_insert_with(ColumnStatistics::new);

        stats.total_size += original_size;
        stats.compressed_size += compressed_bytes;
        if stats.compressed_size > 0 {
            stats.compression_ratio = stats.total_size as f64 / stats.compressed_size as f64;
        }
        // Measured, not a per-encoding lookup table of invented constants.
        if encoded_bytes > 0 {
            stats.encoding_ratio = stats.total_size as f64 / encoded_bytes.max(1) as f64;
        }

        if let Some((min, max)) = min_max {
            stats.min_value = Some(match stats.min_value.take() {
                Some(existing)
                    if compare_typed(&existing, &min, handle.layout.data_type)
                        == std::cmp::Ordering::Less =>
                {
                    existing
                }
                _ => min,
            });
            stats.max_value = Some(match stats.max_value.take() {
                Some(existing)
                    if compare_typed(&existing, &max, handle.layout.data_type)
                        == std::cmp::Ordering::Greater =>
                {
                    existing
                }
                _ => max,
            });
        }

        Ok(())
    }

    fn average_compression_ratio(&self) -> f64 {
        if let Ok(cache) = self.metadata_cache.read() {
            let (total, compressed) = cache.values().fold((0u64, 0u64), |(t, c), s| {
                (t + s.total_size, c + s.compressed_size)
            });
            if compressed > 0 {
                return total as f64 / compressed as f64;
            }
        }
        1.0
    }
}

/// Compute real min/max byte encodings for fixed-width numeric columns.
fn compute_min_max(
    data: &[u8],
    data_type: ColumnDataType,
    layout: ChunkLayout,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if layout != ChunkLayout::Opaque {
        return None;
    }
    let width = data_type.element_width()?;
    if width == 0 || width > 8 || data.len() < width {
        return None;
    }
    let count = data.len() / width;
    let mut min: Option<Vec<u8>> = None;
    let mut max: Option<Vec<u8>> = None;
    for i in 0..count {
        let element = &data[i * width..(i + 1) * width];
        min = Some(match min {
            Some(current)
                if compare_typed(&current, element, data_type) != std::cmp::Ordering::Greater =>
            {
                current
            }
            _ => element.to_vec(),
        });
        max = Some(match max {
            Some(current)
                if compare_typed(&current, element, data_type) != std::cmp::Ordering::Less =>
            {
                current
            }
            _ => element.to_vec(),
        });
    }
    match (min, max) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    }
}

/// Compare two little-endian element encodings according to the column type.
fn compare_typed(a: &[u8], b: &[u8], data_type: ColumnDataType) -> std::cmp::Ordering {
    fn as_u64(bytes: &[u8]) -> u64 {
        let mut value = 0u64;
        for (i, &byte) in bytes.iter().enumerate().take(8) {
            value |= (byte as u64) << (8 * i);
        }
        value
    }
    fn as_i64(bytes: &[u8]) -> i64 {
        let width = bytes.len().min(8);
        let raw = as_u64(bytes);
        if width >= 8 {
            raw as i64
        } else {
            let sign_bit = 1u64 << (8 * width - 1);
            if raw & sign_bit != 0 {
                (raw as i64) - (1i64 << (8 * width))
            } else {
                raw as i64
            }
        }
    }

    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    match data_type {
        ColumnDataType::Float32 => {
            let fa = f32::from_le_bytes([a[0], a[1], a[2], a[3]]);
            let fb = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        }
        ColumnDataType::Float64 => {
            let mut ba = [0u8; 8];
            let mut bb = [0u8; 8];
            ba.copy_from_slice(&a[..8]);
            bb.copy_from_slice(&b[..8]);
            f64::from_le_bytes(ba)
                .partial_cmp(&f64::from_le_bytes(bb))
                .unwrap_or(std::cmp::Ordering::Equal)
        }
        ColumnDataType::Int8
        | ColumnDataType::Int16
        | ColumnDataType::Int32
        | ColumnDataType::Int64
        | ColumnDataType::Timestamp
        | ColumnDataType::Date => as_i64(a).cmp(&as_i64(b)),
        _ => as_u64(a).cmp(&as_u64(b)),
    }
}

impl StorageStrategy for UnifiedColumnStoreStrategy {
    type Handle = ColumnStoreHandle;
    type Error = Error;
    type Metadata = ColumnStatistics;

    fn name(&self) -> &'static str {
        "UnifiedColumnStore"
    }

    fn create_storage(&mut self, config: &StorageConfig) -> Result<Self::Handle> {
        let layout = self.determine_optimal_layout(config)?;
        let compression_type =
            self.select_compression_strategy(&config.requirements.data_characteristics)?;
        let encoding_type =
            self.select_encoding_strategy(&config.requirements.data_characteristics)?;

        Ok(ColumnStoreHandle::new(
            layout,
            compression_type,
            encoding_type,
        ))
    }

    fn read_chunk(&self, handle: &Self::Handle, range: ChunkRange) -> Result<DataChunk> {
        let start_time = Instant::now();
        let relevant_blocks = self.find_blocks_for_range(handle, &range)?;
        if relevant_blocks.is_empty() {
            self.op_stats
                .record_read(start_time.elapsed().as_nanos() as u64);
            // Return an empty chunk with the layout this handle actually
            // stores, so callers can still decode it as strings.
            let layout = handle
                .block_refs()?
                .first()
                .map(|b| b.layout)
                .unwrap_or(ChunkLayout::Opaque);
            return match layout {
                ChunkLayout::Opaque => Ok(DataChunk::new(Vec::new())),
                ChunkLayout::Strings => Ok(DataChunk::from_strings(Vec::new())),
            };
        }

        // Reads take the block-manager lock once (the cache needs `&mut`), so
        // going through rayon here would only serialise on the same mutex.
        let raw_blocks: Vec<(BlockRef, CompressedBlock)> = {
            let mut manager = self.lock_blocks()?;
            let mut collected = Vec::with_capacity(relevant_blocks.len());
            for block_ref in &relevant_blocks {
                collected.push((*block_ref, manager.read_block(block_ref.id)?));
            }
            collected
        };

        let decode = |(block_ref, block): &(BlockRef, CompressedBlock)| -> Result<DataChunk> {
            let bytes = self.decode_block(block)?;
            let chunk = DataChunk::from_encoded(bytes, block_ref.layout, block_ref.row_count)?;
            let local_start = range.start.saturating_sub(block_ref.row_start);
            let local_end = range.end.min(block_ref.row_end()) - block_ref.row_start;
            chunk.slice_rows(local_start, local_end)
        };

        let parts: Result<Vec<DataChunk>> = if self.config.enable_parallel && raw_blocks.len() > 1 {
            raw_blocks.par_iter().map(decode).collect()
        } else {
            raw_blocks.iter().map(decode).collect()
        };

        let merged = DataChunk::concat(parts?)?;
        self.op_stats
            .record_read(start_time.elapsed().as_nanos() as u64);
        Ok(merged)
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        let start_time = Instant::now();
        if chunk.rows() == 0 {
            // Nothing to record; writing an empty block would only add an
            // unreadable zero-row entry to the index.
            self.op_stats
                .record_write(start_time.elapsed().as_nanos() as u64);
            return Ok(());
        }
        let layout = chunk.layout();
        let groups = self.split_chunk_into_row_groups(&chunk)?;
        let total_rows: usize = groups.iter().map(|(_, rows)| *rows).sum();
        if total_rows != chunk.rows() {
            return Err(Error::InvalidOperation(format!(
                "Internal split lost rows: {} groups cover {} of {} rows",
                groups.len(),
                total_rows,
                chunk.rows()
            )));
        }

        let encode = |(payload, rows): &(Vec<u8>, usize)| -> Result<(CompressedBlock, usize)> {
            Ok((self.compress_and_encode_block(handle, payload)?, *rows))
        };
        let encoded: Result<Vec<(CompressedBlock, usize)>> =
            if self.config.enable_parallel && groups.len() > 1 {
                groups.par_iter().map(encode).collect()
            } else {
                groups.iter().map(encode).collect()
            };
        // Only claim the row range once encoding has succeeded, so a failed
        // write cannot leave an unfillable gap in the logical row stream.
        let encoded = encoded?;
        let base_row = handle.reserve_rows(total_rows);

        let mut prepared = Vec::with_capacity(encoded.len());
        let mut cursor = base_row;
        let mut compressed_bytes = 0u64;
        for (block, rows) in encoded {
            compressed_bytes += block.data.len() as u64;
            prepared.push(PreparedBlock {
                block,
                row_start: cursor,
                row_count: rows,
                layout,
            });
            cursor += rows;
        }

        let mut written = Vec::with_capacity(prepared.len());
        {
            let mut manager = self.lock_blocks()?;
            for mut item in prepared {
                item.block.metadata.row_start = item.row_start;
                item.block.metadata.row_count = item.row_count;
                item.block.metadata.layout = item.layout;
                let id = manager.write_block(item.block)?;
                written.push(BlockRef {
                    id,
                    row_start: item.row_start,
                    row_count: item.row_count,
                    layout: item.layout,
                });
            }
        }
        // Recording the block ids is what makes the data readable again.
        handle.append_blocks(written)?;

        self.update_column_statistics(handle, &chunk, compressed_bytes, chunk.len() as u64)?;
        self.op_stats
            .record_write(start_time.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, _handle: &Self::Handle) -> Result<()> {
        // Blocks are written through on every `write_chunk`; the in-memory
        // backing store has nothing further to push.
        Ok(())
    }

    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()> {
        let block_ids = handle.block_ids()?;
        {
            let mut manager = self.lock_blocks()?;
            for block_id in block_ids {
                if let Err(e) = manager.delete_block(block_id) {
                    log::warn!("Failed to delete column store block {:?}: {}", block_id, e);
                }
            }
            manager.merge_free_space();
        }
        handle.clear()?;

        if let Ok(mut cache) = self.metadata_cache.write() {
            cache.remove(&handle.layout.name);
        }
        Ok(())
    }

    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability {
        let can_handle = match requirements.data_characteristics {
            DataCharacteristics::Numeric
            | DataCharacteristics::TimeSeries
            | DataCharacteristics::Dense => true,
            _ => requirements.estimated_size > 1024 * 1024,
        };

        let confidence = if can_handle { 0.9 } else { 0.3 };

        let performance_score = match requirements.performance_priority {
            PerformancePriority::Speed => 0.8,
            PerformancePriority::Memory => 0.9,
            PerformancePriority::Balanced => 0.85,
            _ => 0.7,
        };

        StrategyCapability {
            can_handle,
            confidence,
            performance_score,
            resource_cost: ResourceCost {
                memory: requirements.estimated_size / 2,
                cpu: 15.0,
                disk: requirements.estimated_size / 3,
                network: 0,
            },
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            read_speed: Speed::VeryFast,
            write_speed: Speed::Fast,
            memory_efficiency: Efficiency::Excellent,
            compression_ratio: self.average_compression_ratio(),
            query_optimization: QueryOptimization::Excellent,
            parallel_scalability: ParallelScalability::Excellent,
        }
    }

    fn storage_stats(&self) -> StorageStats {
        let (total_size, compressed_size) = match self.metadata_cache.read() {
            Ok(cache) => cache.values().fold((0u64, 0u64), |(t, c), s| {
                (t + s.total_size, c + s.compressed_size)
            }),
            Err(_) => (0, 0),
        };

        let (stored_bytes, hit_rate) = match self.block_manager.lock() {
            Ok(manager) => (manager.stored_bytes(), manager.cache_hit_rate()),
            Err(_) => (0, 0.0),
        };

        StorageStats {
            total_size: total_size.try_into().unwrap_or(usize::MAX),
            used_size: stored_bytes
                .max(compressed_size)
                .try_into()
                .unwrap_or(usize::MAX),
            read_operations: self.op_stats.read_operations.load(Ordering::Relaxed),
            write_operations: self.op_stats.write_operations.load(Ordering::Relaxed),
            avg_read_latency_ns: self.op_stats.avg_read_nanos(),
            avg_write_latency_ns: self.op_stats.avg_write_nanos(),
            cache_hit_rate: hit_rate,
        }
    }

    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        match pattern {
            AccessPattern::Sequential => {
                self.config.block_size = 256 * 1024;
                self.config.enable_parallel = false;
            }
            AccessPattern::Random => {
                self.config.block_size = 16 * 1024;
                self.config.enable_parallel = true;
            }
            AccessPattern::Columnar => {
                self.config.enable_dictionary = true;
                self.config.compression_type = CompressionType::Zstd;
            }
            _ => {}
        }
        Ok(())
    }

    fn compact(&mut self, handle: &Self::Handle) -> Result<CompactionResult> {
        let start_time = Instant::now();
        let block_refs = handle.block_refs()?;

        let mut size_before = 0usize;
        let mut size_after = 0usize;

        let mut manager = self.lock_blocks()?;
        for block_ref in &block_refs {
            let block = match manager.read_block(block_ref.id) {
                Ok(block) => block,
                Err(e) => {
                    log::warn!(
                        "Skipping unreadable block {:?} during compaction: {}",
                        block_ref.id,
                        e
                    );
                    continue;
                }
            };
            size_before += block.data.len();

            // Decompress with the block's OWN codec, not a hardcoded one.
            let source_engine = self
                .compression_engines
                .get(&block.compression_type)
                .ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "Compression engine {:?} not found",
                        block.compression_type
                    ))
                })?;
            let plain = source_engine.decompress(&block.data)?;

            let target_type = CompressionType::Zstd;
            let target_engine = self.compression_engines.get(&target_type).ok_or_else(|| {
                Error::InvalidOperation("ZSTD compression engine not registered".to_string())
            })?;
            let recompressed = target_engine.compress(&plain)?;

            if recompressed.len() < block.data.len() {
                size_after += recompressed.len();
                // The metadata tag is updated together with the bytes.
                manager.replace_block(block_ref.id, recompressed, target_type)?;
            } else {
                size_after += block.data.len();
            }
        }
        manager.merge_free_space();
        drop(manager);

        if let Ok(mut cache) = self.metadata_cache.write() {
            if let Some(stats) = cache.get_mut(&handle.layout.name) {
                stats.compressed_size = size_after as u64;
                if stats.compressed_size > 0 {
                    stats.compression_ratio =
                        stats.total_size as f64 / stats.compressed_size as f64;
                }
            }
        }

        Ok(CompactionResult {
            size_before,
            size_after,
            duration: start_time.elapsed(),
        })
    }
}

/// Expose block metadata for tests and diagnostics.
impl UnifiedColumnStoreStrategy {
    pub fn block_metadata(&self, block_id: BlockId) -> Result<Option<BlockMetadata>> {
        let manager = self.lock_blocks()?;
        Ok(manager.metadata(block_id).cloned())
    }
}

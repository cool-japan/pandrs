//! Column-oriented in-memory storage engine.
//!
//! # Element boundaries
//!
//! Compressed column payloads keep an explicit element index. The previous
//! representation stored `Raw(Vec<u8>)` — a flat concatenation — so
//! `["a", "bb", "ccc"]` round-tripped as `b"abbccc"` with no way to recover the
//! three values. Every variant now records how many elements it holds and where
//! each one starts.
//!
//! # Nulls
//!
//! Nulls are represented by an optional per-column validity vector: see
//! [`ColumnStore::add_column_nullable`] and
//! [`ColumnStore::get_column_values_nullable`]. Columns added through
//! [`ColumnStore::add_column`] are entirely non-null (`null_count == 0`),
//! which is a statement about that call, not an assumption about the data.

use crate::core::error::{Error, Result};
use crate::storage::traits::{
    AccessPattern, DataChunk, Efficiency, PerformanceProfile, Speed, StorageConfig, StorageEngine,
    StorageStatistics,
};
use crate::{read_lock_safe, write_lock_safe};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Compression strategies for columnar data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    RunLength,
    Dictionary,
    BitPacked,
}

/// Storage metadata for a column
#[derive(Debug, Clone)]
pub struct ColumnMetadata {
    pub name: String,
    pub data_type: String,
    pub row_count: usize,
    pub compression: CompressionType,
    pub null_count: usize,
    /// Bytes occupied by the compressed representation
    pub size_bytes: usize,
    /// Bytes the values occupy uncompressed (tracked so metrics never have to
    /// decompress the whole store)
    pub uncompressed_bytes: usize,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// Compressed column data storage.
///
/// Every variant is element-addressable: `elements()` recovers the exact input
/// values, including empty and variable-length ones.
#[derive(Debug, Clone)]
pub enum CompressedColumnData {
    /// Uncompressed values plus an Arrow-style offsets array
    /// (`offsets.len() == element_count + 1`).
    Raw { data: Vec<u8>, offsets: Vec<u64> },
    /// Run-length encoded data (value, count) pairs
    RunLength(Vec<(Vec<u8>, usize)>),
    /// Dictionary encoded data (dictionary, indices)
    Dictionary {
        dictionary: Vec<Vec<u8>>,
        indices: Vec<u32>,
    },
    /// Bit-packed fixed-width unsigned integers
    BitPacked {
        /// Packed bit stream, LSB-first
        data: Vec<u8>,
        /// Bits used per value
        bits_per_value: u8,
        /// Width in bytes of each decoded element
        element_width: u8,
        /// Number of packed values
        count: usize,
    },
}

impl CompressedColumnData {
    /// Get the approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            CompressedColumnData::Raw { data, offsets } => data.len() + offsets.len() * 8,
            CompressedColumnData::RunLength(runs) => {
                runs.iter().map(|(value, _)| value.len() + 8).sum()
            }
            CompressedColumnData::Dictionary {
                dictionary,
                indices,
            } => dictionary.iter().map(|v| v.len()).sum::<usize>() + indices.len() * 4,
            CompressedColumnData::BitPacked { data, .. } => data.len(),
        }
    }

    /// Number of elements stored.
    pub fn element_count(&self) -> usize {
        match self {
            CompressedColumnData::Raw { offsets, .. } => offsets.len().saturating_sub(1),
            CompressedColumnData::RunLength(runs) => runs.iter().map(|(_, count)| *count).sum(),
            CompressedColumnData::Dictionary { indices, .. } => indices.len(),
            CompressedColumnData::BitPacked { count, .. } => *count,
        }
    }

    /// Total uncompressed byte size of the stored values.
    pub fn uncompressed_bytes(&self) -> usize {
        match self {
            CompressedColumnData::Raw { data, .. } => data.len(),
            CompressedColumnData::RunLength(runs) => {
                runs.iter().map(|(value, count)| value.len() * count).sum()
            }
            CompressedColumnData::Dictionary {
                dictionary,
                indices,
            } => indices
                .iter()
                .map(|&i| dictionary.get(i as usize).map(|v| v.len()).unwrap_or(0))
                .sum(),
            CompressedColumnData::BitPacked {
                element_width,
                count,
                ..
            } => *element_width as usize * *count,
        }
    }

    /// Recover the individual element values.
    pub fn elements(&self) -> Result<Vec<Vec<u8>>> {
        match self {
            CompressedColumnData::Raw { data, offsets } => {
                if offsets.is_empty() {
                    return Ok(Vec::new());
                }
                let mut out = Vec::with_capacity(offsets.len() - 1);
                for window in offsets.windows(2) {
                    let (start, end) = (window[0] as usize, window[1] as usize);
                    if start > end || end > data.len() {
                        return Err(Error::InvalidValue(format!(
                            "Corrupt column offsets: {}..{} outside a {} byte buffer",
                            start,
                            end,
                            data.len()
                        )));
                    }
                    out.push(data[start..end].to_vec());
                }
                Ok(out)
            }
            CompressedColumnData::RunLength(runs) => {
                let mut out = Vec::new();
                for (value, count) in runs {
                    for _ in 0..*count {
                        out.push(value.clone());
                    }
                }
                Ok(out)
            }
            CompressedColumnData::Dictionary {
                dictionary,
                indices,
            } => {
                let mut out = Vec::with_capacity(indices.len());
                for (position, &index) in indices.iter().enumerate() {
                    // Out-of-range indices used to be skipped silently, dropping
                    // rows and misaligning the column against its siblings.
                    let value = dictionary.get(index as usize).ok_or_else(|| {
                        Error::InvalidValue(format!(
                            "Corrupt dictionary column: element {} references entry {} of {}",
                            position,
                            index,
                            dictionary.len()
                        ))
                    })?;
                    out.push(value.clone());
                }
                Ok(out)
            }
            CompressedColumnData::BitPacked {
                data,
                bits_per_value,
                element_width,
                count,
            } => unpack_bits(data, *bits_per_value, *element_width, *count),
        }
    }

    /// Decompress to a flat byte buffer.
    ///
    /// Element boundaries are **not** recoverable from the result; use
    /// [`CompressedColumnData::elements`] unless the caller genuinely wants the
    /// concatenation (for example fixed-width numeric data).
    pub fn decompress(&self) -> Vec<u8> {
        match self.elements() {
            Ok(elements) => elements.concat(),
            Err(_) => Vec::new(),
        }
    }
}

/// Pack fixed-width little-endian unsigned values into `bits` bits each.
fn pack_bits(values: &[u64], bits: u8) -> Vec<u8> {
    if bits == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((values.len() * bits as usize + 7) / 8);
    let mut buffer = 0u128;
    let mut buffered_bits = 0u32;
    for &value in values {
        buffer |= (value as u128) << buffered_bits;
        buffered_bits += bits as u32;
        while buffered_bits >= 8 {
            out.push((buffer & 0xFF) as u8);
            buffer >>= 8;
            buffered_bits -= 8;
        }
    }
    if buffered_bits > 0 {
        out.push((buffer & 0xFF) as u8);
    }
    out
}

/// Inverse of [`pack_bits`], reconstructing `count` values of `width` bytes.
fn unpack_bits(data: &[u8], bits: u8, width: u8, count: usize) -> Result<Vec<Vec<u8>>> {
    let width = width as usize;
    if width == 0 || width > 8 {
        return Err(Error::InvalidValue(format!(
            "Unsupported bit-packed element width {}",
            width
        )));
    }
    if bits as usize > width * 8 {
        return Err(Error::InvalidValue(format!(
            "Corrupt bit-packed column: {} bits per {}-byte value",
            bits, width
        )));
    }
    let needed = (count * bits as usize + 7) / 8;
    if data.len() < needed {
        return Err(Error::InvalidValue(format!(
            "Truncated bit-packed column: {} bytes, need {}",
            data.len(),
            needed
        )));
    }

    let mut out = Vec::with_capacity(count);
    if bits == 0 {
        for _ in 0..count {
            out.push(vec![0u8; width]);
        }
        return Ok(out);
    }

    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut value = 0u64;
        for b in 0..bits as usize {
            let absolute = bit_pos + b;
            let bit = (data[absolute / 8] >> (absolute % 8)) & 1;
            value |= (bit as u64) << b;
        }
        out.push(value.to_le_bytes()[..width].to_vec());
        bit_pos += bits as usize;
    }
    Ok(out)
}

fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        value |= (b as u64) << (8 * i);
    }
    value
}

fn bits_needed(max_value: u64) -> u8 {
    if max_value == 0 {
        0
    } else {
        64 - max_value.leading_zeros() as u8
    }
}

/// A stored column: its compressed values plus optional validity.
#[derive(Debug, Clone)]
struct StoredColumn {
    data: CompressedColumnData,
    /// `None` means "no nulls"; otherwise `validity[i] == false` marks row `i`
    /// as null and the corresponding element is a zero-length placeholder.
    validity: Option<Vec<bool>>,
}

impl StoredColumn {
    fn null_count(&self) -> usize {
        self.validity
            .as_ref()
            .map(|v| v.iter().filter(|valid| !**valid).count())
            .unwrap_or(0)
    }
}

/// A column-oriented storage engine for data
#[derive(Debug)]
pub struct ColumnStore {
    /// Stored columns indexed by name
    columns: Arc<RwLock<HashMap<String, StoredColumn>>>,
    /// Metadata for each column
    metadata: Arc<RwLock<HashMap<String, ColumnMetadata>>>,
    /// Number of rows every column must have
    row_count: Arc<RwLock<usize>>,
    /// Read operation counter (atomic so reads never take a write lock)
    read_operations: Arc<AtomicUsize>,
    /// Write operation counter
    write_operations: Arc<AtomicUsize>,
    /// Monotonic id source for generated chunk column names
    next_chunk_id: Arc<AtomicUsize>,
}

/// Storage statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct StorageStats {
    pub total_columns: usize,
    pub total_size_bytes: usize,
    pub total_rows: usize,
    pub compression_ratio: f64,
    pub read_operations: usize,
    pub write_operations: usize,
}

impl ColumnStore {
    /// Creates a new column store
    pub fn new() -> Self {
        Self {
            columns: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            row_count: Arc::new(RwLock::new(0)),
            read_operations: Arc::new(AtomicUsize::new(0)),
            write_operations: Arc::new(AtomicUsize::new(0)),
            next_chunk_id: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add a non-null column with automatic compression selection.
    pub fn add_column<T: AsRef<[u8]>>(
        &self,
        name: String,
        data: &[T],
        data_type: String,
    ) -> Result<()> {
        let values: Vec<&[u8]> = data.iter().map(|v| v.as_ref()).collect();
        self.insert_column(name, &values, None, data_type)
    }

    /// Add a column that may contain nulls.
    ///
    /// `values[i] == None` marks row `i` as null; it is stored as a validity bit
    /// rather than substituted with an empty value.
    pub fn add_column_nullable<T: AsRef<[u8]>>(
        &self,
        name: String,
        values: &[Option<T>],
        data_type: String,
    ) -> Result<()> {
        let validity: Vec<bool> = values.iter().map(|v| v.is_some()).collect();
        let materialized: Vec<&[u8]> = values
            .iter()
            .map(|v| v.as_ref().map(|v| v.as_ref()).unwrap_or(&[]))
            .collect();
        self.insert_column(name, &materialized, Some(validity), data_type)
    }

    fn insert_column(
        &self,
        name: String,
        values: &[&[u8]],
        validity: Option<Vec<bool>>,
        data_type: String,
    ) -> Result<()> {
        if values.is_empty() {
            return Err(Error::InvalidInput("Cannot add empty column".into()));
        }

        let compression = self.select_compression_strategy(values);
        let compressed_data = self.compress_data(values, compression)?;

        let stored = StoredColumn {
            data: compressed_data,
            validity,
        };
        let metadata = ColumnMetadata {
            name: name.clone(),
            data_type,
            row_count: values.len(),
            compression,
            null_count: stored.null_count(),
            size_bytes: stored.data.size_bytes(),
            uncompressed_bytes: stored.data.uncompressed_bytes(),
            min_value: None,
            max_value: None,
        };

        // Take every lock up front and validate *before* mutating, so a
        // dimension mismatch can no longer leave a half-applied write behind an
        // `Err` return.
        let mut columns = write_lock_safe!(self.columns, "column store columns write")?;
        let mut metadata_map = write_lock_safe!(self.metadata, "column store metadata write")?;
        let mut row_count = write_lock_safe!(self.row_count, "column store row count write")?;

        let replacing = columns.contains_key(&name);
        if !replacing && *row_count != 0 && *row_count != values.len() {
            return Err(Error::DimensionMismatch(
                "Column length doesn't match existing row count".into(),
            ));
        }
        if replacing && *row_count != values.len() && columns.len() > 1 {
            return Err(Error::DimensionMismatch(
                "Replacement column length doesn't match existing row count".into(),
            ));
        }

        columns.insert(name.clone(), stored);
        metadata_map.insert(name, metadata);
        if columns.len() == 1 || *row_count == 0 {
            *row_count = values.len();
        }
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get a column's values as a flat byte buffer.
    ///
    /// For variable-width data prefer [`ColumnStore::get_column_values`], which
    /// preserves element boundaries.
    pub fn get_column(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.get_column_values(name)?.concat())
    }

    /// Get a column's individual element values.
    pub fn get_column_values(&self, name: &str) -> Result<Vec<Vec<u8>>> {
        // Counters are atomic, so a read no longer takes the stats write lock
        // and serialises every other reader.
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        let columns = read_lock_safe!(self.columns, "column store columns read")?;
        match columns.get(name) {
            Some(stored) => stored.data.elements(),
            None => Err(Error::ColumnNotFound(name.to_string())),
        }
    }

    /// Get a column's values with nulls preserved.
    pub fn get_column_values_nullable(&self, name: &str) -> Result<Vec<Option<Vec<u8>>>> {
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        let columns = read_lock_safe!(self.columns, "column store columns read")?;
        let stored = columns
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        let elements = stored.data.elements()?;
        Ok(match &stored.validity {
            None => elements.into_iter().map(Some).collect(),
            Some(validity) => elements
                .into_iter()
                .enumerate()
                .map(|(i, value)| {
                    if validity.get(i).copied().unwrap_or(true) {
                        Some(value)
                    } else {
                        None
                    }
                })
                .collect(),
        })
    }

    /// Get column metadata
    pub fn get_metadata(&self, name: &str) -> Result<ColumnMetadata> {
        let metadata = read_lock_safe!(self.metadata, "column store metadata read")?;
        match metadata.get(name) {
            Some(meta) => Ok(meta.clone()),
            None => Err(Error::ColumnNotFound(name.to_string())),
        }
    }

    /// List all column names
    pub fn column_names(&self) -> Result<Vec<String>> {
        let columns = read_lock_safe!(self.columns, "column store columns read")?;
        let mut names: Vec<String> = columns.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    /// Get the number of rows
    pub fn row_count(&self) -> Result<usize> {
        Ok(*read_lock_safe!(
            self.row_count,
            "column store row count read"
        )?)
    }

    /// Get storage statistics.
    ///
    /// Sizes are derived from the metadata index, so the numbers cannot drift
    /// away from the stored columns the way incrementally-updated counters did
    /// (`add_column` added on every call while the map *replaced* the entry).
    pub fn stats(&self) -> Result<StorageStats> {
        let metadata = read_lock_safe!(self.metadata, "column store metadata read")?;
        let total_size_bytes: usize = metadata.values().map(|m| m.size_bytes).sum();
        let uncompressed: usize = metadata.values().map(|m| m.uncompressed_bytes).sum();
        Ok(StorageStats {
            total_columns: metadata.len(),
            total_size_bytes,
            total_rows: self.row_count()?,
            compression_ratio: if total_size_bytes > 0 {
                uncompressed as f64 / total_size_bytes as f64
            } else {
                1.0
            },
            read_operations: self.read_operations.load(Ordering::Relaxed),
            write_operations: self.write_operations.load(Ordering::Relaxed),
        })
    }

    /// Remove a column from the store
    pub fn remove_column(&self, name: &str) -> Result<()> {
        let mut columns = write_lock_safe!(self.columns, "column store columns write")?;
        let mut metadata_map = write_lock_safe!(self.metadata, "column store metadata write")?;

        if columns.remove(name).is_some() {
            metadata_map.remove(name);
            if columns.is_empty() {
                let mut row_count =
                    write_lock_safe!(self.row_count, "column store row count write")?;
                *row_count = 0;
            }
            Ok(())
        } else {
            Err(Error::ColumnNotFound(name.to_string()))
        }
    }

    /// Remove every column.
    pub fn clear(&self) -> Result<()> {
        let mut columns = write_lock_safe!(self.columns, "column store columns write")?;
        let mut metadata_map = write_lock_safe!(self.metadata, "column store metadata write")?;
        let mut row_count = write_lock_safe!(self.row_count, "column store row count write")?;
        columns.clear();
        metadata_map.clear();
        *row_count = 0;
        Ok(())
    }

    /// Recompress every column, keeping element boundaries and nulls intact.
    ///
    /// The rewrite is computed first and swapped in under one lock: the old
    /// implementation removed and re-added each column one at a time, collapsing
    /// it to a single element and leaving the store mutated even when it
    /// returned `Err`.
    pub fn optimize(&self) -> Result<()> {
        let snapshot: Vec<(String, StoredColumn, ColumnMetadata)> = {
            let columns = read_lock_safe!(self.columns, "column store columns read")?;
            let metadata = read_lock_safe!(self.metadata, "column store metadata read")?;
            columns
                .iter()
                .filter_map(|(name, stored)| {
                    metadata
                        .get(name)
                        .map(|meta| (name.clone(), stored.clone(), meta.clone()))
                })
                .collect()
        };

        let mut rebuilt: Vec<(String, StoredColumn, ColumnMetadata)> =
            Vec::with_capacity(snapshot.len());
        for (name, stored, meta) in snapshot {
            let elements = stored.data.elements()?;
            let refs: Vec<&[u8]> = elements.iter().map(|v| v.as_slice()).collect();
            let compression = self.select_compression_strategy(&refs);
            let recompressed = self.compress_data(&refs, compression)?;

            // Only keep the rewrite if it is genuinely no worse.
            let (data, compression) = if recompressed.size_bytes() <= stored.data.size_bytes() {
                (recompressed, compression)
            } else {
                (stored.data.clone(), meta.compression)
            };

            let new_stored = StoredColumn {
                data,
                validity: stored.validity.clone(),
            };
            let new_meta = ColumnMetadata {
                compression,
                size_bytes: new_stored.data.size_bytes(),
                uncompressed_bytes: new_stored.data.uncompressed_bytes(),
                null_count: new_stored.null_count(),
                ..meta
            };
            rebuilt.push((name, new_stored, new_meta));
        }

        let mut columns = write_lock_safe!(self.columns, "column store columns write")?;
        let mut metadata = write_lock_safe!(self.metadata, "column store metadata write")?;
        for (name, stored, meta) in rebuilt {
            columns.insert(name.clone(), stored);
            metadata.insert(name, meta);
        }
        Ok(())
    }

    /// Calculate the store-wide compression ratio.
    ///
    /// Uses the tracked byte counts. It used to decompress every column just to
    /// produce this metric, which turned a metrics call into an OOM risk for
    /// run-length columns.
    pub fn compression_ratio(&self) -> Result<f64> {
        Ok(self.stats()?.compression_ratio)
    }

    // Private helper methods

    fn select_compression_strategy(&self, data: &[&[u8]]) -> CompressionType {
        if data.len() < 10 {
            return CompressionType::None;
        }

        let mut consecutive_count = 1;
        let mut max_consecutive = 1;
        for i in 1..data.len() {
            if data[i] == data[i - 1] {
                consecutive_count += 1;
                max_consecutive = max_consecutive.max(consecutive_count);
            } else {
                consecutive_count = 1;
            }
        }
        if max_consecutive > data.len() / 4 {
            return CompressionType::RunLength;
        }

        // Fixed-width small integers pack well.
        let width = data[0].len();
        if matches!(width, 1 | 2 | 4 | 8) && data.iter().all(|v| v.len() == width) {
            let max_value = data.iter().map(|v| read_le_uint(v)).max().unwrap_or(0);
            if (bits_needed(max_value) as usize) < width * 8 {
                return CompressionType::BitPacked;
            }
        }

        let unique_count = {
            let mut unique = std::collections::HashSet::new();
            for item in data {
                unique.insert(*item);
                if unique.len() > data.len() / 2 {
                    break;
                }
            }
            unique.len()
        };

        if unique_count < data.len() / 4 {
            CompressionType::Dictionary
        } else {
            CompressionType::None
        }
    }

    fn compress_data(
        &self,
        data: &[&[u8]],
        compression: CompressionType,
    ) -> Result<CompressedColumnData> {
        match compression {
            CompressionType::None => {
                let mut raw_data = Vec::new();
                let mut offsets = Vec::with_capacity(data.len() + 1);
                offsets.push(0u64);
                for item in data {
                    raw_data.extend_from_slice(item);
                    offsets.push(raw_data.len() as u64);
                }
                Ok(CompressedColumnData::Raw {
                    data: raw_data,
                    offsets,
                })
            }
            CompressionType::RunLength => {
                let mut runs: Vec<(Vec<u8>, usize)> = Vec::new();
                if !data.is_empty() {
                    let mut current_value = data[0].to_vec();
                    let mut count = 1usize;
                    for item in data.iter().skip(1) {
                        if *item == current_value.as_slice() {
                            count += 1;
                        } else {
                            runs.push((current_value, count));
                            current_value = item.to_vec();
                            count = 1;
                        }
                    }
                    runs.push((current_value, count));
                }
                Ok(CompressedColumnData::RunLength(runs))
            }
            CompressionType::Dictionary => {
                let mut dictionary: Vec<Vec<u8>> = Vec::new();
                let mut value_to_index: HashMap<&[u8], u32> = HashMap::new();
                let mut indices = Vec::with_capacity(data.len());

                for item in data {
                    if let Some(&index) = value_to_index.get(*item) {
                        indices.push(index);
                    } else {
                        let index = u32::try_from(dictionary.len()).map_err(|_| {
                            Error::InvalidValue(
                                "Dictionary column exceeded 2^32 distinct values".to_string(),
                            )
                        })?;
                        dictionary.push(item.to_vec());
                        value_to_index.insert(item, index);
                        indices.push(index);
                    }
                }

                Ok(CompressedColumnData::Dictionary {
                    dictionary,
                    indices,
                })
            }
            CompressionType::BitPacked => {
                // Real bit packing over fixed-width little-endian values. The
                // old variant copied the bytes verbatim and advertised itself as
                // bit-packed.
                let width = data.first().map(|v| v.len()).unwrap_or(0);
                if !matches!(width, 1 | 2 | 4 | 8) || data.iter().any(|v| v.len() != width) {
                    return Err(Error::InvalidInput(
                        "Bit packing requires fixed-width elements of 1, 2, 4 or 8 bytes".into(),
                    ));
                }
                let values: Vec<u64> = data.iter().map(|v| read_le_uint(v)).collect();
                let max_value = values.iter().copied().max().unwrap_or(0);
                let bits = bits_needed(max_value);
                Ok(CompressedColumnData::BitPacked {
                    data: pack_bits(&values, bits),
                    bits_per_value: bits,
                    element_width: width as u8,
                    count: values.len(),
                })
            }
        }
    }
}

impl Default for ColumnStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for column store operations.
///
/// Each handle owns an independent [`ColumnStore`]: the engine acts as a
/// factory, so `create_storage` deliberately returns a handle to a *new* store
/// rather than sharing the engine's own columns.
#[derive(Debug, Clone)]
pub struct ColumnStoreHandle {
    /// Unique identifier for the storage instance
    pub id: usize,
    /// Reference to the column store
    pub store: Arc<ColumnStore>,
}

impl ColumnStoreHandle {
    /// Create a new handle
    pub fn new(id: usize, store: Arc<ColumnStore>) -> Self {
        Self { id, store }
    }
}

impl StorageEngine for ColumnStore {
    type Handle = ColumnStoreHandle;
    type Error = Error;

    fn create_storage(&mut self, _config: &StorageConfig) -> Result<Self::Handle> {
        use std::sync::atomic::AtomicUsize;
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        Ok(ColumnStoreHandle::new(id, Arc::new(ColumnStore::new())))
    }

    /// Read rows `range` from every column of the handle's store.
    ///
    /// Columns are visited in sorted name order and their element values for the
    /// requested rows are concatenated. The old implementation applied the same
    /// *byte* range to every column, concatenated the results and reported
    /// `(end - start) / 8` rows on an invented 8-bytes-per-value assumption.
    fn read_chunk(&self, handle: &Self::Handle, range: Range<usize>) -> Result<DataChunk> {
        handle.store.read_operations.fetch_add(1, Ordering::Relaxed);
        let columns = read_lock_safe!(handle.store.columns, "storage engine columns read")?;

        let mut names: Vec<&String> = columns.keys().collect();
        names.sort();

        let total_rows = *read_lock_safe!(handle.store.row_count, "row count read")?;
        let end = range.end.min(total_rows);
        let start = range.start.min(end);

        let mut chunk_data = Vec::new();
        for name in &names {
            let Some(stored) = columns.get(*name) else {
                continue;
            };
            let elements = stored.data.elements()?;
            for element in elements.iter().take(end).skip(start) {
                chunk_data.extend_from_slice(element);
            }
        }

        let metadata = crate::storage::traits::ChunkMetadata {
            row_count: end - start,
            column_count: names.len(),
            compression: crate::storage::traits::CompressionPreference::None,
            uncompressed_size: chunk_data.len(),
            compressed_size: chunk_data.len(),
        };

        Ok(DataChunk::new(chunk_data, metadata))
    }

    /// Write a chunk as a new column.
    ///
    /// Names come from a monotonic counter; they used to be
    /// `format!("chunk_{}", row_count)`, so two chunks with the same row count
    /// silently overwrote each other.
    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        let chunk_id = handle.store.next_chunk_id.fetch_add(1, Ordering::SeqCst);
        let column_name = format!("chunk_{}", chunk_id);

        let element_width = if chunk.metadata.row_count > 0 {
            (chunk.data.len() / chunk.metadata.row_count).max(1)
        } else {
            1
        };
        let data: Vec<Vec<u8>> = chunk
            .data
            .chunks(element_width)
            .map(|part| part.to_vec())
            .collect();
        if data.is_empty() {
            return Ok(());
        }

        handle
            .store
            .add_column(column_name, &data, "bytes".to_string())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, _handle: &Self::Handle) -> Result<()> {
        // Column store is in-memory, so flush is a no-op
        Ok(())
    }

    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()> {
        // Actually drop the data instead of returning Ok(()) without doing
        // anything.
        handle.store.clear()
    }

    fn performance_profile(&self) -> PerformanceProfile {
        // `compression_ratio` follows the crate-wide convention of
        // original/compressed (>= 1.0 is a win). It used to report 0.7, which is
        // the inverse convention and read as "expands the data by 43%".
        let compression_ratio = self.compression_ratio().unwrap_or(1.0).max(1.0);
        PerformanceProfile {
            read_speed: Speed::Fast,
            write_speed: Speed::Medium,
            memory_efficiency: Efficiency::Good,
            compression_ratio,
            random_access_speed: Speed::Fast,
            sequential_access_speed: Speed::VeryFast,
        }
    }

    fn storage_stats(&self, handle: &Self::Handle) -> Result<StorageStatistics> {
        let stats = handle.store.stats()?;
        Ok(StorageStatistics {
            total_size: stats.total_size_bytes,
            chunk_count: stats.total_columns,
            avg_compression_ratio: stats.compression_ratio,
            read_operations: stats.read_operations as u64,
            write_operations: stats.write_operations as u64,
            // This engine has no read cache; reporting an assumed 0.9 hit rate
            // was pure fiction.
            cache_hit_rate: 0.0,
        })
    }

    fn supports_random_access(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn supports_compression(&self) -> bool {
        true
    }

    fn optimal_chunk_size(&self) -> usize {
        64 * 1024
    }

    fn memory_overhead(&self) -> usize {
        1024
    }

    fn optimize_for_pattern(&mut self, _pattern: AccessPattern) -> Result<()> {
        // The layout is columnar regardless of the access pattern; there is no
        // per-pattern tuning knob to set here.
        Ok(())
    }

    fn compact(&mut self, handle: &Self::Handle) -> Result<()> {
        handle.store.optimize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_width_values_keep_their_boundaries() {
        // The headline data-loss bug: ["a", "bb", "ccc"] used to round-trip as
        // b"abbccc" with the boundaries gone.
        let store = ColumnStore::new();
        let values = vec![
            b"a".to_vec(),
            b"bb".to_vec(),
            b"ccc".to_vec(),
            Vec::new(),
            b"dddd".to_vec(),
        ];
        store
            .add_column("v".to_string(), &values, "bytes".to_string())
            .expect("add");
        assert_eq!(store.get_column_values("v").expect("get"), values);
        assert_eq!(store.get_column("v").expect("flat"), b"abbcccdddd".to_vec());
    }

    #[test]
    fn every_compression_strategy_preserves_elements() {
        for (label, values) in [
            (
                "run-length",
                (0..100).map(|i| vec![(i / 40) as u8]).collect::<Vec<_>>(),
            ),
            (
                "dictionary",
                (0..100)
                    .map(|i| format!("cat-{}", i % 3).into_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                "bit-packed",
                (0..100)
                    .map(|i| ((i % 7) as u32).to_le_bytes().to_vec())
                    .collect::<Vec<_>>(),
            ),
            (
                "raw",
                (0..100)
                    .map(|i| format!("unique-value-{}", i).into_bytes())
                    .collect::<Vec<_>>(),
            ),
        ] {
            let store = ColumnStore::new();
            store
                .add_column(label.to_string(), &values, "bytes".to_string())
                .expect("add");
            assert_eq!(
                store.get_column_values(label).expect("get"),
                values,
                "{} lost data",
                label
            );
        }
    }

    #[test]
    fn bit_packing_actually_packs() {
        let store = ColumnStore::new();
        // 3-bit values in 4-byte slots: packing must beat the raw 400 bytes.
        let values: Vec<Vec<u8>> = (0..100)
            .map(|i| ((i % 8) as u32).to_le_bytes().to_vec())
            .collect();
        store
            .add_column("packed".to_string(), &values, "u32".to_string())
            .expect("add");
        let metadata = store.get_metadata("packed").expect("metadata");
        assert_eq!(metadata.compression, CompressionType::BitPacked);
        assert!(
            metadata.size_bytes < 100,
            "bit packing produced {} bytes for 100 3-bit values",
            metadata.size_bytes
        );
        assert_eq!(store.get_column_values("packed").expect("get"), values);
    }

    #[test]
    fn dictionary_decode_rejects_out_of_range_indices() {
        let corrupt = CompressedColumnData::Dictionary {
            dictionary: vec![b"a".to_vec()],
            indices: vec![0, 5],
        };
        assert!(corrupt.elements().is_err());
    }

    #[test]
    fn nulls_survive_the_round_trip() {
        let store = ColumnStore::new();
        let values: Vec<Option<Vec<u8>>> = vec![
            Some(b"x".to_vec()),
            None,
            Some(Vec::new()),
            None,
            Some(b"y".to_vec()),
        ];
        store
            .add_column_nullable("n".to_string(), &values, "bytes".to_string())
            .expect("add");
        assert_eq!(store.get_column_values_nullable("n").expect("get"), values);
        assert_eq!(store.get_metadata("n").expect("metadata").null_count, 2);
    }

    #[test]
    fn statistics_do_not_drift_when_a_column_is_replaced() {
        let store = ColumnStore::new();
        let values: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8]).collect();
        store
            .add_column("c".to_string(), &values, "bytes".to_string())
            .expect("add");
        let first = store.stats().expect("stats");
        store
            .add_column("c".to_string(), &values, "bytes".to_string())
            .expect("replace");
        let second = store.stats().expect("stats");
        assert_eq!(first.total_columns, second.total_columns);
        assert_eq!(first.total_size_bytes, second.total_size_bytes);
    }

    #[test]
    fn mismatched_column_length_leaves_the_store_untouched() {
        let store = ColumnStore::new();
        let ten: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8]).collect();
        let five: Vec<Vec<u8>> = (0..5).map(|i| vec![i as u8]).collect();
        store
            .add_column("a".to_string(), &ten, "bytes".to_string())
            .expect("add");
        assert!(store
            .add_column("b".to_string(), &five, "bytes".to_string())
            .is_err());
        // The rejected column must not be present, and the row count unchanged.
        assert_eq!(store.column_names().expect("names"), vec!["a".to_string()]);
        assert_eq!(store.row_count().expect("rows"), 10);
    }

    #[test]
    fn optimize_is_atomic_and_lossless() {
        let store = ColumnStore::new();
        let values: Vec<Vec<u8>> = (0..64)
            .map(|i| format!("value-{}", i % 4).into_bytes())
            .collect();
        store
            .add_column("c".to_string(), &values, "bytes".to_string())
            .expect("add");
        store.optimize().expect("optimize");
        // optimize() used to re-add the whole column as ONE element.
        assert_eq!(store.get_column_values("c").expect("get"), values);
        assert_eq!(store.get_metadata("c").expect("metadata").row_count, 64);
        assert_eq!(store.row_count().expect("rows"), 64);
    }

    #[test]
    fn compression_ratio_does_not_decompress_the_store() {
        let store = ColumnStore::new();
        // A run-length column that would expand to 64 MiB if materialised.
        let values: Vec<Vec<u8>> = vec![vec![7u8; 64]; 1_000_000];
        store
            .add_column("rle".to_string(), &values, "bytes".to_string())
            .expect("add");
        let ratio = store.compression_ratio().expect("ratio");
        assert!(ratio > 100.0, "expected a large ratio, got {}", ratio);
    }

    #[test]
    fn write_chunks_do_not_overwrite_each_other() {
        let mut engine = ColumnStore::new();
        let handle = engine
            .create_storage(&crate::storage::traits::StorageConfig {
                estimated_size: 64,
                access_pattern: AccessPattern::Columnar,
                performance_priority: crate::storage::traits::PerformancePriority::Balanced,
                durability: crate::storage::traits::DurabilityLevel::Temporary,
                compression: crate::storage::traits::CompressionPreference::None,
                memory_limit: None,
            })
            .expect("create");

        // Two chunks with identical row counts used to collide on
        // "chunk_{row_count}".
        engine
            .write_chunk(&handle, DataChunk::new_test_data(16))
            .expect("write");
        engine
            .write_chunk(&handle, DataChunk::new_test_data(16))
            .expect("write");
        assert_eq!(handle.store.column_names().expect("names").len(), 2);
    }

    #[test]
    fn delete_storage_actually_clears_the_store() {
        let mut engine = ColumnStore::new();
        let handle = engine
            .create_storage(&crate::storage::traits::StorageConfig {
                estimated_size: 64,
                access_pattern: AccessPattern::Columnar,
                performance_priority: crate::storage::traits::PerformancePriority::Balanced,
                durability: crate::storage::traits::DurabilityLevel::Temporary,
                compression: crate::storage::traits::CompressionPreference::None,
                memory_limit: None,
            })
            .expect("create");
        engine
            .write_chunk(&handle, DataChunk::new_test_data(16))
            .expect("write");
        assert!(!handle.store.column_names().expect("names").is_empty());

        engine.delete_storage(&handle).expect("delete");
        assert!(handle.store.column_names().expect("names").is_empty());
        assert_eq!(handle.store.row_count().expect("rows"), 0);
    }

    #[test]
    fn read_chunk_returns_the_requested_rows() {
        let mut engine = ColumnStore::new();
        let handle = engine
            .create_storage(&crate::storage::traits::StorageConfig {
                estimated_size: 64,
                access_pattern: AccessPattern::Columnar,
                performance_priority: crate::storage::traits::PerformancePriority::Balanced,
                durability: crate::storage::traits::DurabilityLevel::Temporary,
                compression: crate::storage::traits::CompressionPreference::None,
                memory_limit: None,
            })
            .expect("create");
        let values: Vec<Vec<u8>> = (0..20u8).map(|i| vec![i]).collect();
        handle
            .store
            .add_column("a".to_string(), &values, "bytes".to_string())
            .expect("add");

        let chunk = engine.read_chunk(&handle, 5..10).expect("read");
        assert_eq!(chunk.metadata.row_count, 5);
        assert_eq!(chunk.data, vec![5u8, 6, 7, 8, 9]);
    }

    #[test]
    fn performance_profile_uses_the_shared_ratio_convention() {
        let engine = ColumnStore::new();
        assert!(engine.performance_profile().compression_ratio >= 1.0);
    }
}

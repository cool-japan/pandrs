//! File-backed chunk storage for datasets that should not stay resident in RAM.
//!
//! This used to be an empty struct whose constructor discarded the path and
//! returned `Ok(Self {})` while still being re-exported from the crate root.
//! It is now a real directory-backed key/value store for byte chunks, with an
//! explicit durability level so that
//! [`DurabilityLevel::Persistent`](crate::storage::traits::DurabilityLevel)
//! actually means "fsync'd before the write returns".

use crate::core::error::{Error, Result};
use crate::storage::traits::{
    AccessPattern, ChunkMetadata, CompressionPreference, DataChunk, DurabilityLevel, Efficiency,
    PerformanceProfile, Speed, StorageConfig, StorageEngine, StorageStatistics,
};
use crate::{read_lock_safe, write_lock_safe};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// File extension used for chunk files written by [`DiskStorage`].
const CHUNK_EXTENSION: &str = "chunk";

/// Reserved key holding the row index of a [`DiskStorageHandle`].
const INDEX_KEY: &str = "__pandrs_index";

/// Disk-based storage for large datasets.
///
/// Chunks are addressed by an arbitrary UTF-8 key. Keys are hex-encoded before
/// being used as file names, so a key containing `/`, `..` or any other
/// path-significant text cannot escape the storage directory.
#[derive(Debug)]
pub struct DiskStorage {
    /// Root directory holding the chunk files
    root: PathBuf,
    /// Durability level applied to writes
    durability: DurabilityLevel,
    /// Id source for datasets created through [`StorageEngine::create_storage`]
    next_storage_id: AtomicUsize,
}

impl DiskStorage {
    /// Create (or open) a disk storage rooted at `path`.
    ///
    /// The directory is created if it does not exist. Writes are buffered by
    /// the OS; use [`DiskStorage::with_durability`] with
    /// [`DurabilityLevel::Persistent`] to fsync every write.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::with_durability(path, DurabilityLevel::Cached)
    }

    /// Create (or open) a disk storage with an explicit durability level.
    pub fn with_durability<P: AsRef<Path>>(path: P, durability: DurabilityLevel) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| {
            Error::IoError(format!(
                "Failed to create disk storage directory {}: {}",
                root.display(),
                e
            ))
        })?;
        Ok(Self {
            root,
            durability,
            next_storage_id: AtomicUsize::new(1),
        })
    }

    /// Root directory of this storage.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Durability level applied to writes.
    pub fn durability(&self) -> DurabilityLevel {
        self.durability
    }

    /// Write (or overwrite) the chunk stored under `key`.
    pub fn put_chunk(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.chunk_path(key);
        // Write to a temporary sibling first, then rename, so a crash mid-write
        // cannot leave a half-written chunk visible under `key`.
        let tmp_path = path.with_extension("chunk.tmp");
        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)
                .map_err(|e| {
                    Error::IoError(format!(
                        "Failed to open chunk file {}: {}",
                        tmp_path.display(),
                        e
                    ))
                })?;
            file.write_all(data).map_err(|e| {
                Error::IoError(format!(
                    "Failed to write chunk {}: {}",
                    tmp_path.display(),
                    e
                ))
            })?;
            if self.durability == DurabilityLevel::Persistent {
                file.sync_all().map_err(|e| {
                    Error::IoError(format!(
                        "Failed to fsync chunk {}: {}",
                        tmp_path.display(),
                        e
                    ))
                })?;
            }
        }
        fs::rename(&tmp_path, &path).map_err(|e| {
            Error::IoError(format!("Failed to publish chunk {}: {}", path.display(), e))
        })?;
        if self.durability == DurabilityLevel::Persistent {
            // Renaming is only durable once the directory entry itself is
            // flushed.
            if let Ok(dir) = File::open(&self.root) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }

    /// Read the chunk stored under `key`.
    pub fn get_chunk(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.chunk_path(key);
        let mut file = File::open(&path).map_err(|e| {
            Error::IoError(format!(
                "Chunk '{}' not readable ({}): {}",
                key,
                path.display(),
                e
            ))
        })?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| Error::IoError(format!("Failed to read chunk '{}': {}", key, e)))?;
        Ok(buffer)
    }

    /// Whether a chunk is stored under `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.chunk_path(key).is_file()
    }

    /// Delete the chunk stored under `key`. Returns `Ok(false)` if it did not
    /// exist.
    pub fn delete_chunk(&self, key: &str) -> Result<bool> {
        let path = self.chunk_path(key);
        if !path.exists() {
            return Ok(false);
        }
        fs::remove_file(&path)
            .map_err(|e| Error::IoError(format!("Failed to delete chunk '{}': {}", key, e)))?;
        Ok(true)
    }

    /// List the keys of every chunk currently stored.
    pub fn keys(&self) -> Result<Vec<String>> {
        let entries = fs::read_dir(&self.root).map_err(|e| {
            Error::IoError(format!(
                "Failed to list disk storage {}: {}",
                self.root.display(),
                e
            ))
        })?;
        let mut keys = Vec::new();
        for entry in entries {
            let entry = entry
                .map_err(|e| Error::IoError(format!("Failed to read directory entry: {}", e)))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(CHUNK_EXTENSION) {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some(key) = decode_key(stem) {
                    keys.push(key);
                }
            }
        }
        keys.sort();
        Ok(keys)
    }

    /// Total number of bytes currently stored on disk.
    pub fn total_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for key in self.keys()? {
            let path = self.chunk_path(&key);
            if let Ok(metadata) = fs::metadata(&path) {
                total = total.saturating_add(metadata.len());
            }
        }
        Ok(total)
    }

    /// Remove every chunk from this storage.
    pub fn clear(&self) -> Result<()> {
        for key in self.keys()? {
            self.delete_chunk(&key)?;
        }
        Ok(())
    }

    fn chunk_path(&self, key: &str) -> PathBuf {
        self.root
            .join(encode_key(key))
            .with_extension(CHUNK_EXTENSION)
    }
}

/// One written chunk, in row coordinates.
#[derive(Debug, Clone)]
struct DiskChunkEntry {
    /// Key of the chunk file inside the dataset directory
    key: String,
    /// First logical row this chunk holds
    row_start: usize,
    /// Number of rows this chunk holds
    row_count: usize,
    /// Bytes per row (chunks must be fixed-width to be row-sliceable)
    element_width: usize,
}

/// State shared by every clone of a [`DiskStorageHandle`].
#[derive(Debug)]
struct DiskDataset {
    /// Directory holding just this dataset's chunks
    store: DiskStorage,
    /// Row index over the written chunks
    index: RwLock<Vec<DiskChunkEntry>>,
    /// Total number of logical rows written
    rows: AtomicUsize,
    /// Id source for chunk keys
    next_chunk: AtomicUsize,
    /// Completed reads
    reads: AtomicU64,
    /// Completed writes
    writes: AtomicU64,
}

/// Handle for one dataset stored under a [`DiskStorage`] root.
///
/// Rows are addressed logically: every write appends `chunk.metadata.row_count`
/// rows and [`StorageEngine::read_chunk`] slices whatever range of those rows is
/// asked for, reading only the files that overlap it.
#[derive(Debug, Clone)]
pub struct DiskStorageHandle {
    /// Identifier of this dataset within its engine
    pub id: usize,
    inner: Arc<DiskDataset>,
}

impl DiskStorageHandle {
    /// Directory this dataset's chunk files live in.
    pub fn path(&self) -> &Path {
        self.inner.store.path()
    }

    /// Number of logical rows written so far.
    pub fn row_count(&self) -> usize {
        self.inner.rows.load(Ordering::SeqCst)
    }

    /// Number of chunk files backing this dataset.
    pub fn chunk_count(&self) -> Result<usize> {
        Ok(read_lock_safe!(self.inner.index, "disk storage index read")?.len())
    }

    /// Serialise the row index so a later process can reopen the dataset.
    fn persist_index(&self) -> Result<()> {
        let index = read_lock_safe!(self.inner.index, "disk storage index read")?;
        let mut buffer = Vec::with_capacity(24 + index.len() * 40);
        buffer.extend_from_slice(&(index.len() as u64).to_le_bytes());
        buffer.extend_from_slice(&(self.inner.rows.load(Ordering::SeqCst) as u64).to_le_bytes());
        buffer.extend_from_slice(
            &(self.inner.next_chunk.load(Ordering::SeqCst) as u64).to_le_bytes(),
        );
        for entry in index.iter() {
            buffer.extend_from_slice(&(entry.row_start as u64).to_le_bytes());
            buffer.extend_from_slice(&(entry.row_count as u64).to_le_bytes());
            buffer.extend_from_slice(&(entry.element_width as u64).to_le_bytes());
            buffer.extend_from_slice(&(entry.key.len() as u64).to_le_bytes());
            buffer.extend_from_slice(entry.key.as_bytes());
        }
        drop(index);
        self.inner.store.put_chunk(INDEX_KEY, &buffer)
    }
}

/// Read a little-endian `u64` from `bytes` at `cursor`, advancing it.
fn take_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor
        .checked_add(8)
        .ok_or_else(|| Error::InvalidValue("Disk storage index offset overflow".to_string()))?;
    if end > bytes.len() {
        return Err(Error::InvalidValue(
            "Truncated disk storage index".to_string(),
        ));
    }
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u64::from_le_bytes(raw))
}

impl DiskStorage {
    /// Directory backing the dataset with identifier `id`.
    fn dataset_root(&self, id: usize) -> PathBuf {
        self.root.join(format!("storage_{}", id))
    }

    fn new_dataset(&self, id: usize) -> Result<DiskStorageHandle> {
        let store = DiskStorage::with_durability(self.dataset_root(id), self.durability)?;
        Ok(DiskStorageHandle {
            id,
            inner: Arc::new(DiskDataset {
                store,
                index: RwLock::new(Vec::new()),
                rows: AtomicUsize::new(0),
                next_chunk: AtomicUsize::new(0),
                reads: AtomicU64::new(0),
                writes: AtomicU64::new(0),
            }),
        })
    }

    /// Reopen a dataset previously created by [`StorageEngine::create_storage`].
    ///
    /// This is what makes the engine genuinely disk-backed: the row index is
    /// persisted next to the chunks, so a new process can address the same rows.
    pub fn open_storage(&self, id: usize) -> Result<DiskStorageHandle> {
        let root = self.dataset_root(id);
        if !root.is_dir() {
            return Err(Error::InvalidOperation(format!(
                "No disk storage dataset {} under {}",
                id,
                self.root.display()
            )));
        }
        let handle = self.new_dataset(id)?;
        let raw = handle.inner.store.get_chunk(INDEX_KEY)?;

        let mut cursor = 0usize;
        let entry_count = take_u64(&raw, &mut cursor)? as usize;
        let rows = take_u64(&raw, &mut cursor)? as usize;
        let next_chunk = take_u64(&raw, &mut cursor)? as usize;

        let mut entries = Vec::with_capacity(entry_count.min(1024));
        for _ in 0..entry_count {
            let row_start = take_u64(&raw, &mut cursor)? as usize;
            let row_count = take_u64(&raw, &mut cursor)? as usize;
            let element_width = take_u64(&raw, &mut cursor)? as usize;
            let key_len = take_u64(&raw, &mut cursor)? as usize;
            let end = cursor.checked_add(key_len).filter(|end| *end <= raw.len());
            let Some(end) = end else {
                return Err(Error::InvalidValue(
                    "Truncated disk storage index key".to_string(),
                ));
            };
            let key = String::from_utf8(raw[cursor..end].to_vec()).map_err(|_| {
                Error::InvalidValue("Disk storage index key is not UTF-8".to_string())
            })?;
            cursor = end;
            entries.push(DiskChunkEntry {
                key,
                row_start,
                row_count,
                element_width,
            });
        }

        {
            let mut index = write_lock_safe!(handle.inner.index, "disk storage index write")?;
            *index = entries;
        }
        handle.inner.rows.store(rows, Ordering::SeqCst);
        handle.inner.next_chunk.store(next_chunk, Ordering::SeqCst);
        Ok(handle)
    }

    fn append_rows(&self, handle: &DiskStorageHandle, chunk: DataChunk) -> Result<()> {
        let row_count = chunk.metadata.row_count;
        if row_count == 0 {
            // Recording a zero-row chunk would only add an unreadable entry.
            return Ok(());
        }
        if chunk.data.is_empty() {
            return Err(Error::InvalidInput(format!(
                "Disk storage cannot store {} rows of no data",
                row_count
            )));
        }
        if chunk.data.len() % row_count != 0 {
            return Err(Error::InvalidInput(format!(
                "Disk storage needs fixed-width rows: {} bytes do not divide into {} rows",
                chunk.data.len(),
                row_count
            )));
        }
        let element_width = chunk.data.len() / row_count;

        let chunk_id = handle.inner.next_chunk.fetch_add(1, Ordering::SeqCst);
        let key = format!("chunk_{}", chunk_id);
        handle.inner.store.put_chunk(&key, &chunk.data)?;

        {
            // Claim the rows only after the bytes are safely on disk, and do it
            // under the index lock: a reader that saw the bumped row count
            // without the matching index entry would get a short read whose
            // metadata still claimed the full range.
            let mut index = write_lock_safe!(handle.inner.index, "disk storage index write")?;
            let row_start = handle.inner.rows.fetch_add(row_count, Ordering::SeqCst);
            index.push(DiskChunkEntry {
                key,
                row_start,
                row_count,
                element_width,
            });
        }
        handle.inner.writes.fetch_add(1, Ordering::SeqCst);
        handle.persist_index()
    }
}

impl StorageEngine for DiskStorage {
    type Handle = DiskStorageHandle;
    type Error = Error;

    fn create_storage(&mut self, _config: &StorageConfig) -> Result<Self::Handle> {
        let id = self.next_storage_id.fetch_add(1, Ordering::SeqCst);
        let handle = self.new_dataset(id)?;
        handle.persist_index()?;
        Ok(handle)
    }

    fn read_chunk(&self, handle: &Self::Handle, range: Range<usize>) -> Result<DataChunk> {
        // Row count and index are read together so a concurrent append can only
        // be seen as fully applied or not at all.
        let (mut entries, total_rows) = {
            let index = read_lock_safe!(handle.inner.index, "disk storage index read")?;
            let total_rows = handle.inner.rows.load(Ordering::SeqCst);
            (index.clone(), total_rows)
        };
        let end = range.end.min(total_rows);
        let start = range.start.min(end);
        // Concurrent writers can append to the index out of row order (the row
        // range is claimed after the bytes are on disk), so assemble by row
        // rather than by insertion order.
        entries.sort_by_key(|entry| entry.row_start);

        let mut data = Vec::new();
        for entry in entries.iter() {
            let entry_end = entry.row_start.saturating_add(entry.row_count);
            let overlap_start = entry.row_start.max(start);
            let overlap_end = entry_end.min(end);
            if overlap_start >= overlap_end {
                continue;
            }
            let bytes = handle.inner.store.get_chunk(&entry.key)?;
            let from = (overlap_start - entry.row_start) * entry.element_width;
            let to = (overlap_end - entry.row_start) * entry.element_width;
            if to > bytes.len() {
                return Err(Error::InvalidValue(format!(
                    "Disk chunk '{}' is {} bytes, index expects at least {}",
                    entry.key,
                    bytes.len(),
                    to
                )));
            }
            data.extend_from_slice(&bytes[from..to]);
        }

        handle.inner.reads.fetch_add(1, Ordering::SeqCst);
        let len = data.len();
        Ok(DataChunk::new(
            data,
            ChunkMetadata {
                row_count: end - start,
                column_count: 1,
                compression: CompressionPreference::None,
                uncompressed_size: len,
                compressed_size: len,
            },
        ))
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.append_rows(handle, chunk)
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.append_rows(handle, chunk)
    }

    fn flush(&mut self, handle: &Self::Handle) -> Result<()> {
        // Chunk writes already fsync when the durability level asks for it; this
        // additionally flushes the directory entries.
        if handle.inner.store.durability == DurabilityLevel::Persistent {
            if let Ok(dir) = File::open(handle.inner.store.path()) {
                dir.sync_all().map_err(|e| {
                    Error::IoError(format!("Failed to fsync disk storage directory: {}", e))
                })?;
            }
        }
        Ok(())
    }

    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()> {
        handle.inner.store.clear()?;
        {
            let mut index = write_lock_safe!(handle.inner.index, "disk storage index write")?;
            index.clear();
        }
        handle.inner.rows.store(0, Ordering::SeqCst);
        handle.persist_index()
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            read_speed: Speed::Medium,
            write_speed: Speed::Medium,
            // Only the row index is resident; the payload never is.
            memory_efficiency: Efficiency::Excellent,
            // This engine stores bytes verbatim.
            compression_ratio: 1.0,
            random_access_speed: Speed::Slow,
            sequential_access_speed: Speed::Medium,
        }
    }

    fn storage_stats(&self, handle: &Self::Handle) -> Result<StorageStatistics> {
        let chunk_count = handle.chunk_count()?;
        Ok(StorageStatistics {
            total_size: handle.inner.store.total_bytes()? as usize,
            chunk_count,
            avg_compression_ratio: 1.0,
            read_operations: handle.inner.reads.load(Ordering::SeqCst),
            write_operations: handle.inner.writes.load(Ordering::SeqCst),
            // There is no read cache in front of the files.
            cache_hit_rate: 0.0,
        })
    }

    fn supports_random_access(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_compression(&self) -> bool {
        false
    }

    fn optimal_chunk_size(&self) -> usize {
        1024 * 1024
    }

    fn memory_overhead(&self) -> usize {
        // One index entry per chunk; nothing else is held in memory.
        std::mem::size_of::<DiskChunkEntry>()
    }

    fn optimize_for_pattern(&mut self, _pattern: AccessPattern) -> Result<()> {
        // Chunks are plain files read with pread-style offsets; there is no
        // per-pattern layout to switch to.
        Ok(())
    }

    /// Merge every chunk into one contiguous file so a sequential scan needs a
    /// single open.
    ///
    /// Requires a uniform row width (the common case); mixed widths are left
    /// alone rather than merged into a file whose rows can no longer be located.
    /// The merge buffers the dataset in memory, so call it when that fits.
    fn compact(&mut self, handle: &Self::Handle) -> Result<()> {
        let entries = {
            let index = read_lock_safe!(handle.inner.index, "disk storage index read")?;
            index.clone()
        };
        if entries.len() < 2 {
            return Ok(());
        }
        let width = entries[0].element_width;
        if entries.iter().any(|entry| entry.element_width != width) {
            // Mixed widths cannot share one file without an offsets array; leave
            // the layout alone rather than corrupting it.
            return Ok(());
        }

        let mut merged = Vec::new();
        let mut sorted = entries.clone();
        sorted.sort_by_key(|entry| entry.row_start);
        for entry in &sorted {
            merged.extend_from_slice(&handle.inner.store.get_chunk(&entry.key)?);
        }

        let total_rows: usize = sorted.iter().map(|entry| entry.row_count).sum();
        let chunk_id = handle.inner.next_chunk.fetch_add(1, Ordering::SeqCst);
        let key = format!("chunk_{}", chunk_id);
        // Publish the merged file before dropping the originals, so a failure
        // here leaves the old chunks intact.
        handle.inner.store.put_chunk(&key, &merged)?;
        {
            let mut index = write_lock_safe!(handle.inner.index, "disk storage index write")?;
            *index = vec![DiskChunkEntry {
                key,
                row_start: sorted.first().map(|e| e.row_start).unwrap_or(0),
                row_count: total_rows,
                element_width: width,
            }];
        }
        handle.persist_index()?;
        for entry in &sorted {
            // The index no longer references these files, so a failed unlink
            // only wastes space. Reporting it as a failed compaction would
            // invite a retry that merges again and leaves *more* orphans.
            if let Err(e) = handle.inner.store.delete_chunk(&entry.key) {
                log::warn!(
                    "Compacted chunk '{}' could not be removed: {}",
                    entry.key,
                    e
                );
            }
        }
        Ok(())
    }
}

/// Hex-encode a key so it is always a safe, unambiguous file name.
fn encode_key(key: &str) -> String {
    let mut out = String::with_capacity(key.len() * 2);
    for byte in key.as_bytes() {
        out.push(char::from_digit((byte >> 4) as u32, 16).unwrap_or('0'));
        out.push(char::from_digit((byte & 0x0F) as u32, 16).unwrap_or('0'));
    }
    out
}

/// Inverse of [`encode_key`]; returns `None` for names this module did not write.
fn decode_key(encoded: &str) -> Option<String> {
    if encoded.len() % 2 != 0 {
        return None;
    }
    let bytes = encoded.as_bytes();
    let mut out = Vec::with_capacity(encoded.len() / 2);
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "pandrs_disk_storage_{}_{}",
            name,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn write_then_read_roundtrip() {
        let root = temp_root("roundtrip");
        let storage = DiskStorage::new(&root).expect("create");

        storage
            .put_chunk("column/values", b"hello disk")
            .expect("write");
        assert_eq!(
            storage.get_chunk("column/values").expect("read"),
            b"hello disk".to_vec()
        );
        assert!(storage.contains("column/values"));
        assert_eq!(storage.keys().expect("keys"), vec!["column/values"]);
        assert_eq!(storage.total_bytes().expect("bytes"), 10);

        assert!(storage.delete_chunk("column/values").expect("delete"));
        assert!(!storage.contains("column/values"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn keys_cannot_escape_the_root() {
        let root = temp_root("escape");
        let storage = DiskStorage::new(&root).expect("create");
        storage
            .put_chunk("../../etc/passwd", b"nope")
            .expect("write");
        // The file must live inside the storage root, not two levels up.
        let entries: Vec<_> = fs::read_dir(&root)
            .expect("read dir")
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            storage.get_chunk("../../etc/passwd").expect("read"),
            b"nope".to_vec()
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn persistent_durability_is_honoured() {
        let root = temp_root("durable");
        let storage =
            DiskStorage::with_durability(&root, DurabilityLevel::Persistent).expect("create");
        assert_eq!(storage.durability(), DurabilityLevel::Persistent);
        storage.put_chunk("k", b"durable bytes").expect("write");
        assert_eq!(storage.get_chunk("k").expect("read"), b"durable bytes");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_chunk_is_an_error_not_empty_data() {
        let root = temp_root("missing");
        let storage = DiskStorage::new(&root).expect("create");
        assert!(storage.get_chunk("absent").is_err());
        let _ = fs::remove_dir_all(&root);
    }
}

//! Storage tier backends for the hybrid large-scale strategy.
//!
//! The "SSD" and "HDD" tiers used to be `HashMap`s in RAM annotated
//! *"simulated as in-memory"*, padded with `thread::sleep` calls to fake device
//! latency — in shipped library code, on the `flush()` path. Demoting a chunk
//! therefore freed nothing (peak memory was *higher* during a move) while
//! costing up to 15 ms of pure sleep per chunk.
//!
//! Both are now genuinely file-backed via `std::fs`, in a configurable
//! directory (a `tempfile::TempDir` by default), and honour the tier's
//! configured compression codec.

use crate::core::error::{Error, Result};
use crate::storage::checksum::checksum64;
use crate::storage::unified_memory::{ChunkLayout, CompressionType, DataChunk};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::config::{TierConfig, TierStorageType};

/// Data identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DataId(pub u64);

/// Storage information for a tier backend
#[derive(Debug, Clone)]
pub struct TierStorageInfo {
    /// Backend type
    pub backend_type: TierStorageType,
    /// Current usage in bytes
    pub usage: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Average access latency
    pub avg_latency: Duration,
}

/// Trait for tier-specific storage backends
pub trait TierBackend: Send + Sync + std::fmt::Debug {
    fn store_chunk(&mut self, id: DataId, chunk: &DataChunk) -> Result<()>;
    fn retrieve_chunk(&self, id: DataId) -> Result<DataChunk>;
    fn delete_chunk(&mut self, id: DataId) -> Result<()>;
    fn get_storage_info(&self) -> TierStorageInfo;
    /// Bytes physically occupied by `id` in this backend (post-compression).
    fn stored_size(&self, id: DataId) -> Option<usize>;
    /// Flush any buffered state to the backing device.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// On-disk chunk framing
// ---------------------------------------------------------------------------

const CHUNK_MAGIC: [u8; 4] = *b"PCHK";
const CHUNK_VERSION: u8 = 1;
const HEADER_LEN: usize = 4 + 1 + 1 + 1 + 8 + 8 + 8; // magic..checksum

fn layout_code(layout: ChunkLayout) -> u8 {
    match layout {
        ChunkLayout::Opaque => 0,
        ChunkLayout::Strings => 1,
    }
}

fn layout_from_code(code: u8) -> Result<ChunkLayout> {
    match code {
        0 => Ok(ChunkLayout::Opaque),
        1 => Ok(ChunkLayout::Strings),
        other => Err(Error::InvalidValue(format!(
            "Unknown chunk layout code {} in tier file",
            other
        ))),
    }
}

fn codec_code(codec: CompressionType) -> u8 {
    match codec {
        CompressionType::None => 0,
        CompressionType::Lz4 | CompressionType::Snappy => 1,
        CompressionType::Zstd | CompressionType::Gzip | CompressionType::Auto => 2,
    }
}

fn compress_payload(codec: CompressionType, data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    match codec_code(codec) {
        0 => Ok(data.to_vec()),
        1 => oxiarc_lz4::compress_bytes(data)
            .map_err(|e| Error::InvalidValue(format!("Tier LZ4 compression failed: {}", e))),
        _ => oxiarc_zstd::compress_with_level(data, 3)
            .map_err(|e| Error::InvalidValue(format!("Tier ZSTD compression failed: {}", e))),
    }
}

fn decompress_payload(codec: u8, data: &[u8], plain_len: usize) -> Result<Vec<u8>> {
    if plain_len == 0 {
        return Ok(Vec::new());
    }
    match codec {
        0 => Ok(data.to_vec()),
        1 => oxiarc_lz4::decompress_bytes(data, plain_len)
            .map_err(|e| Error::InvalidValue(format!("Tier LZ4 decompression failed: {}", e))),
        2 => oxiarc_zstd::decompress(data)
            .map_err(|e| Error::InvalidValue(format!("Tier ZSTD decompression failed: {}", e))),
        other => Err(Error::InvalidValue(format!(
            "Unknown tier codec {} in tier file",
            other
        ))),
    }
}

/// Serialise a chunk into a self-describing, checksummed frame.
pub fn encode_chunk(chunk: &DataChunk, codec: CompressionType) -> Result<Vec<u8>> {
    let body = compress_payload(codec, &chunk.data)?;
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(&CHUNK_MAGIC);
    out.push(CHUNK_VERSION);
    out.push(layout_code(chunk.layout()));
    out.push(codec_code(codec));
    out.extend_from_slice(&(chunk.rows() as u64).to_le_bytes());
    out.extend_from_slice(&(chunk.data.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum64(&chunk.data).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Inverse of [`encode_chunk`], verifying the payload checksum.
pub fn decode_chunk(frame: &[u8]) -> Result<DataChunk> {
    if frame.len() < HEADER_LEN || frame[..4] != CHUNK_MAGIC {
        return Err(Error::InvalidValue(
            "Corrupt tier chunk: bad header".to_string(),
        ));
    }
    if frame[4] != CHUNK_VERSION {
        return Err(Error::InvalidValue(format!(
            "Unsupported tier chunk version {}",
            frame[4]
        )));
    }
    let layout = layout_from_code(frame[5])?;
    let codec = frame[6];
    let row_count = u64::from_le_bytes(
        frame[7..15]
            .try_into()
            .map_err(|_| Error::InvalidValue("Corrupt tier chunk header".to_string()))?,
    ) as usize;
    let plain_len = u64::from_le_bytes(
        frame[15..23]
            .try_into()
            .map_err(|_| Error::InvalidValue("Corrupt tier chunk header".to_string()))?,
    ) as usize;
    let expected_checksum = u64::from_le_bytes(
        frame[23..31]
            .try_into()
            .map_err(|_| Error::InvalidValue("Corrupt tier chunk header".to_string()))?,
    );

    let body = &frame[HEADER_LEN..];
    // Bound the declared plain length against the stored body so a corrupt
    // header cannot request an enormous allocation.
    let bound = body
        .len()
        .saturating_mul(256)
        .saturating_add(64 * 1024)
        .min(1usize << 32);
    if plain_len > bound {
        return Err(Error::InvalidValue(format!(
            "Refusing tier chunk expansion bomb: header claims {} bytes from a {} byte body",
            plain_len,
            body.len()
        )));
    }

    let data = decompress_payload(codec, body, plain_len)?;
    if data.len() != plain_len {
        return Err(Error::InvalidValue(format!(
            "Corrupt tier chunk: decompressed {} bytes, header claimed {}",
            data.len(),
            plain_len
        )));
    }
    if checksum64(&data) != expected_checksum {
        return Err(Error::InvalidValue(
            "Tier chunk checksum mismatch".to_string(),
        ));
    }
    DataChunk::from_encoded(data, layout, row_count)
}

// ---------------------------------------------------------------------------
// In-memory tier
// ---------------------------------------------------------------------------

/// In-memory tier backend (the genuinely-RAM-resident hot tier).
#[derive(Debug)]
pub struct InMemoryTierBackend {
    data: HashMap<DataId, DataChunk>,
    config: TierConfig,
}

impl InMemoryTierBackend {
    pub fn new(config: &TierConfig) -> Self {
        Self {
            data: HashMap::new(),
            config: config.clone(),
        }
    }
}

impl TierBackend for InMemoryTierBackend {
    fn store_chunk(&mut self, id: DataId, chunk: &DataChunk) -> Result<()> {
        self.data.insert(id, chunk.clone());
        Ok(())
    }

    fn retrieve_chunk(&self, id: DataId) -> Result<DataChunk> {
        self.data
            .get(&id)
            .cloned()
            .ok_or_else(|| Error::InvalidOperation(format!("Chunk {:?} not found in memory", id)))
    }

    fn delete_chunk(&mut self, id: DataId) -> Result<()> {
        self.data.remove(&id);
        Ok(())
    }

    fn get_storage_info(&self) -> TierStorageInfo {
        let usage = self.data.values().map(|chunk| chunk.len()).sum();
        TierStorageInfo {
            backend_type: TierStorageType::InMemory,
            usage,
            capacity: self.config.max_size,
            avg_latency: self.config.access_latency,
        }
    }

    fn stored_size(&self, id: DataId) -> Option<usize> {
        self.data.get(&id).map(|chunk| chunk.len())
    }
}

// ---------------------------------------------------------------------------
// File-backed tier
// ---------------------------------------------------------------------------

/// File-backed tier backend used for the warm (SSD) and cold (HDD) tiers.
///
/// Data really leaves the process heap: chunks are framed, optionally
/// compressed with the tier's configured codec, and written to individual
/// files under the tier directory.
#[derive(Debug)]
pub struct FileTierBackend {
    /// Directory holding the tier's chunk files
    root: PathBuf,
    /// Owned temporary directory, when the tier was not given an explicit path
    _owned_dir: Option<tempfile::TempDir>,
    /// Physical size of each stored chunk
    sizes: HashMap<DataId, usize>,
    /// Configuration
    config: TierConfig,
    /// Whether writes must be fsync'd
    durable: bool,
}

impl FileTierBackend {
    /// Create a file-backed tier.
    ///
    /// When `config.directory` is `None` a process-scoped temporary directory
    /// is created (and removed when the backend is dropped).
    pub fn new(config: &TierConfig) -> Self {
        match Self::try_new(config) {
            Ok(backend) => backend,
            Err(e) => {
                // Falling back silently would resurrect the "simulated tier"
                // problem, so make the degradation loud and still file-backed
                // via the system temp dir.
                log::error!(
                    "Failed to create '{}' tier directory ({}); falling back to the system temp dir",
                    config.name,
                    e
                );
                let root = std::env::temp_dir().join(format!(
                    "pandrs_tier_{}_{}",
                    config.name,
                    std::process::id()
                ));
                let _ = fs::create_dir_all(&root);
                Self {
                    root,
                    _owned_dir: None,
                    sizes: HashMap::new(),
                    config: config.clone(),
                    durable: config.durable_writes,
                }
            }
        }
    }

    /// Fallible constructor; prefer this when the caller can surface errors.
    pub fn try_new(config: &TierConfig) -> Result<Self> {
        let (root, owned) = match &config.directory {
            Some(dir) => {
                fs::create_dir_all(dir).map_err(|e| {
                    Error::IoError(format!(
                        "Failed to create tier directory {}: {}",
                        dir.display(),
                        e
                    ))
                })?;
                (dir.clone(), None)
            }
            None => {
                let temp = tempfile::Builder::new()
                    .prefix(&format!("pandrs_tier_{}_", config.name))
                    .tempdir()
                    .map_err(|e| {
                        Error::IoError(format!("Failed to create tier temp directory: {}", e))
                    })?;
                (temp.path().to_path_buf(), Some(temp))
            }
        };

        Ok(Self {
            root,
            _owned_dir: owned,
            sizes: HashMap::new(),
            config: config.clone(),
            durable: config.durable_writes,
        })
    }

    /// Directory this tier writes to.
    pub fn directory(&self) -> &Path {
        &self.root
    }

    fn chunk_path(&self, id: DataId) -> PathBuf {
        self.root.join(format!("{:016x}.tier", id.0))
    }
}

impl TierBackend for FileTierBackend {
    fn store_chunk(&mut self, id: DataId, chunk: &DataChunk) -> Result<()> {
        let frame = encode_chunk(chunk, self.config.compression)?;
        let path = self.chunk_path(id);
        let tmp = path.with_extension("tier.tmp");
        {
            let mut file = fs::File::create(&tmp).map_err(|e| {
                Error::IoError(format!(
                    "Failed to create tier chunk {}: {}",
                    tmp.display(),
                    e
                ))
            })?;
            file.write_all(&frame).map_err(|e| {
                Error::IoError(format!(
                    "Failed to write tier chunk {}: {}",
                    tmp.display(),
                    e
                ))
            })?;
            if self.durable {
                file.sync_all().map_err(|e| {
                    Error::IoError(format!(
                        "Failed to fsync tier chunk {}: {}",
                        tmp.display(),
                        e
                    ))
                })?;
            }
        }
        fs::rename(&tmp, &path).map_err(|e| {
            Error::IoError(format!(
                "Failed to publish tier chunk {}: {}",
                path.display(),
                e
            ))
        })?;
        self.sizes.insert(id, frame.len());
        Ok(())
    }

    fn retrieve_chunk(&self, id: DataId) -> Result<DataChunk> {
        let path = self.chunk_path(id);
        let frame = fs::read(&path).map_err(|e| {
            Error::InvalidOperation(format!(
                "Chunk {:?} not found in '{}' tier ({}): {}",
                id,
                self.config.name,
                path.display(),
                e
            ))
        })?;
        decode_chunk(&frame)
    }

    fn delete_chunk(&mut self, id: DataId) -> Result<()> {
        let path = self.chunk_path(id);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                Error::IoError(format!(
                    "Failed to delete tier chunk {}: {}",
                    path.display(),
                    e
                ))
            })?;
        }
        self.sizes.remove(&id);
        Ok(())
    }

    fn get_storage_info(&self) -> TierStorageInfo {
        TierStorageInfo {
            backend_type: self.config.storage_type,
            usage: self.sizes.values().copied().sum(),
            capacity: self.config.max_size,
            avg_latency: self.config.access_latency,
        }
    }

    fn stored_size(&self, id: DataId) -> Option<usize> {
        self.sizes.get(&id).copied()
    }

    fn flush(&mut self) -> Result<()> {
        if self.durable {
            if let Ok(dir) = fs::File::open(&self.root) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }
}

/// Warm-tier backend. SSD and HDD tiers share one real file-backed
/// implementation; they differ only in configuration (capacity, codec,
/// declared latency).
pub type SSDTierBackend = FileTierBackend;
/// Cold-tier backend; see [`SSDTierBackend`].
pub type HDDTierBackend = FileTierBackend;

/// Build the backend for a tier configuration.
pub fn make_backend(config: &TierConfig) -> Result<Box<dyn TierBackend>> {
    match config.storage_type {
        TierStorageType::InMemory => Ok(Box::new(InMemoryTierBackend::new(config))),
        TierStorageType::SSD | TierStorageType::HDD | TierStorageType::Custom => {
            Ok(Box::new(FileTierBackend::try_new(config)?))
        }
        TierStorageType::Network => Err(Error::NotImplemented(
            "Network-backed storage tier: PandRS has no network storage client; \
             configure a mounted path via TierConfig::directory to use it as a tier"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier(name: &str, storage_type: TierStorageType, compression: CompressionType) -> TierConfig {
        TierConfig {
            name: name.to_string(),
            storage_type,
            max_size: 1024 * 1024,
            compression,
            access_latency: Duration::from_micros(1),
            throughput_mbps: 1000.0,
            directory: None,
            durable_writes: false,
        }
    }

    #[test]
    fn file_backend_roundtrips_opaque_and_strings() {
        let mut backend =
            FileTierBackend::try_new(&tier("warm", TierStorageType::SSD, CompressionType::Lz4))
                .expect("create");

        let opaque = DataChunk::new((0..1024u32).map(|i| (i % 251) as u8).collect());
        backend.store_chunk(DataId(1), &opaque).expect("store");
        let read = backend.retrieve_chunk(DataId(1)).expect("retrieve");
        assert_eq!(read.data, opaque.data);
        assert_eq!(read.rows(), opaque.rows());

        let strings = DataChunk::from_strings(vec!["日本語".to_string(), "with\0nul".to_string()]);
        backend.store_chunk(DataId(2), &strings).expect("store");
        let read = backend.retrieve_chunk(DataId(2)).expect("retrieve");
        assert_eq!(read.layout(), ChunkLayout::Strings);
        assert_eq!(
            read.as_strings().expect("decode"),
            strings.as_strings().expect("decode")
        );
    }

    #[test]
    fn file_backend_actually_writes_files() {
        let mut backend =
            FileTierBackend::try_new(&tier("cold", TierStorageType::HDD, CompressionType::Zstd))
                .expect("create");
        backend
            .store_chunk(DataId(9), &DataChunk::new(vec![1u8; 4096]))
            .expect("store");

        let entries: Vec<_> = fs::read_dir(backend.directory())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1, "chunk was not written to disk");
        assert!(backend.stored_size(DataId(9)).unwrap_or(0) > 0);

        backend.delete_chunk(DataId(9)).expect("delete");
        assert!(backend.retrieve_chunk(DataId(9)).is_err());
    }

    #[test]
    fn tier_compression_actually_shrinks() {
        let compressible = DataChunk::new(
            b"pandrs tier payload "
                .iter()
                .copied()
                .cycle()
                .take(64 * 1024)
                .collect(),
        );
        let plain = encode_chunk(&compressible, CompressionType::None).expect("encode");
        let zstd = encode_chunk(&compressible, CompressionType::Zstd).expect("encode");
        assert!(
            zstd.len() < plain.len() / 4,
            "tier compression had no effect: {} vs {}",
            zstd.len(),
            plain.len()
        );
        assert_eq!(decode_chunk(&zstd).expect("decode").data, compressible.data);
    }

    #[test]
    fn corrupt_frame_is_rejected() {
        let chunk = DataChunk::new(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut frame = encode_chunk(&chunk, CompressionType::None).expect("encode");
        let last = frame.len() - 1;
        frame[last] ^= 0xFF;
        assert!(decode_chunk(&frame).is_err());
        assert!(decode_chunk(&frame[..4]).is_err());
    }

    #[test]
    fn network_tier_reports_not_implemented() {
        let err = make_backend(&tier(
            "net",
            TierStorageType::Network,
            CompressionType::None,
        ))
        .expect_err("network tier has no client");
        assert!(matches!(err, Error::NotImplemented(_)));
    }
}

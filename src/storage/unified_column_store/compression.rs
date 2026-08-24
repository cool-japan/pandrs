//! Block compression engines for the unified column store.
//!
//! All codecs are Pure-Rust (COOLJAPAN OxiARC). The LZ4 block format is not
//! self-describing, so the engine prepends the original length — which makes
//! the header attacker-controlled and therefore bounds-checked here.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::CompressionType;

/// Compression engine trait
pub trait CompressionEngine: Send + Sync {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
    fn name(&self) -> &'static str;
    fn compression_ratio(&self, original_size: usize, compressed_size: usize) -> f64;
}

/// Maximum expansion the LZ4 block format can legitimately achieve.
///
/// An LZ4 match copies at most 255 + 15 + 4 bytes for a few bytes of token, so
/// ~256x is a hard upper bound. Anything beyond that plus a small constant
/// slack is a corrupt or hostile header, not real data — accepting it would
/// let a 64-byte block request an 18-exabyte allocation.
const LZ4_MAX_EXPANSION: usize = 256;
/// Absolute slack added to the ratio bound for very small blocks.
const LZ4_EXPANSION_SLACK: usize = 64 * 1024;
/// Hard ceiling on a single decompressed block.
const MAX_DECOMPRESSED_BLOCK: usize = 1 << 32; // 4 GiB

/// Size of the little-endian original-length prefix on LZ4 payloads.
const LZ4_LEN_PREFIX: usize = 8;

/// LZ4 compression engine
pub struct Lz4CompressionEngine;

impl CompressionEngine for Lz4CompressionEngine {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let body = oxiarc_lz4::compress_bytes(data)
            .map_err(|e| Error::InvalidValue(format!("LZ4 compression failed: {}", e)))?;
        let mut out = Vec::with_capacity(LZ4_LEN_PREFIX + body.len());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if data.len() < LZ4_LEN_PREFIX {
            // A short-but-nonempty payload is corruption. Returning `Ok(vec![])`
            // here used to turn a truncated block into a silently empty column.
            return Err(Error::InvalidValue(format!(
                "Truncated LZ4 block: {} bytes, need at least {}",
                data.len(),
                LZ4_LEN_PREFIX
            )));
        }
        let mut len_bytes = [0u8; LZ4_LEN_PREFIX];
        len_bytes.copy_from_slice(&data[..LZ4_LEN_PREFIX]);
        let original_len = u64::from_le_bytes(len_bytes);
        if original_len == 0 {
            return Ok(Vec::new());
        }
        let body = &data[LZ4_LEN_PREFIX..];
        let bound = body
            .len()
            .saturating_mul(LZ4_MAX_EXPANSION)
            .saturating_add(LZ4_EXPANSION_SLACK)
            .min(MAX_DECOMPRESSED_BLOCK);
        if original_len > bound as u64 {
            return Err(Error::InvalidValue(format!(
                "Refusing LZ4 decompression bomb: header claims {} bytes from a {} byte body (bound {})",
                original_len,
                body.len(),
                bound
            )));
        }
        let decoded = oxiarc_lz4::decompress_bytes(body, original_len as usize)
            .map_err(|e| Error::InvalidValue(format!("LZ4 decompression failed: {}", e)))?;
        if decoded.len() != original_len as usize {
            return Err(Error::InvalidValue(format!(
                "LZ4 decompression produced {} bytes, header claimed {}",
                decoded.len(),
                original_len
            )));
        }
        Ok(decoded)
    }

    fn name(&self) -> &'static str {
        "LZ4"
    }

    fn compression_ratio(&self, original_size: usize, compressed_size: usize) -> f64 {
        if compressed_size == 0 {
            0.0
        } else {
            original_size as f64 / compressed_size as f64
        }
    }
}

/// ZSTD compression engine
pub struct ZstdCompressionEngine {
    compression_level: i32,
}

impl ZstdCompressionEngine {
    pub fn new(level: i32) -> Self {
        Self {
            compression_level: level,
        }
    }

    /// Compression level this engine was configured with.
    pub fn level(&self) -> i32 {
        self.compression_level
    }
}

impl CompressionEngine for ZstdCompressionEngine {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        oxiarc_zstd::compress_with_level(data, self.compression_level)
            .map_err(|e| Error::InvalidValue(format!("ZSTD compression failed: {}", e)))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        oxiarc_zstd::decompress(data)
            .map_err(|e| Error::InvalidValue(format!("ZSTD decompression failed: {}", e)))
    }

    fn name(&self) -> &'static str {
        "ZSTD"
    }

    fn compression_ratio(&self, original_size: usize, compressed_size: usize) -> f64 {
        if compressed_size == 0 {
            0.0
        } else {
            original_size as f64 / compressed_size as f64
        }
    }
}

/// No-op compression engine
pub struct NoCompressionEngine;

impl CompressionEngine for NoCompressionEngine {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn name(&self) -> &'static str {
        "None"
    }

    fn compression_ratio(&self, _original_size: usize, _compressed_size: usize) -> f64 {
        1.0
    }
}

/// Build the engine registry used by the column store.
///
/// `CompressionType::Auto` is mapped to ZSTD so that a block never ends up
/// tagged with a compression type that has no engine behind it.
pub fn build_engines(
    zstd_level: i32,
) -> std::collections::HashMap<CompressionType, Box<dyn CompressionEngine>> {
    let mut engines: std::collections::HashMap<CompressionType, Box<dyn CompressionEngine>> =
        std::collections::HashMap::new();
    engines.insert(CompressionType::None, Box::new(NoCompressionEngine));
    engines.insert(CompressionType::Lz4, Box::new(Lz4CompressionEngine));
    engines.insert(
        CompressionType::Zstd,
        Box::new(ZstdCompressionEngine::new(zstd_level)),
    );
    engines.insert(
        CompressionType::Snappy,
        Box::new(Lz4CompressionEngine), // closest Pure-Rust fast codec available
    );
    engines.insert(
        CompressionType::Gzip,
        Box::new(ZstdCompressionEngine::new(zstd_level)),
    );
    engines
}

/// Resolve `CompressionType::Auto` to the concrete codec that will actually be
/// applied, so blocks are never tagged with a placeholder.
pub fn resolve_compression(requested: CompressionType) -> CompressionType {
    match requested {
        CompressionType::Auto => CompressionType::Zstd,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz4_roundtrip() {
        let engine = Lz4CompressionEngine;
        let data = b"Hello, World! This is a test string for compression.";
        let compressed = engine.compress(data).expect("compress");
        assert_eq!(engine.decompress(&compressed).expect("decompress"), data);
    }

    #[test]
    fn lz4_rejects_decompression_bomb() {
        let engine = Lz4CompressionEngine;
        // 8-byte header claiming 18 exabytes, followed by 4 bytes of body.
        let mut hostile = u64::MAX.to_le_bytes().to_vec();
        hostile.extend_from_slice(&[0, 0, 0, 0]);
        let err = engine.decompress(&hostile).expect_err("must reject");
        assert!(
            format!("{}", err).contains("bomb"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn lz4_rejects_truncated_header() {
        let engine = Lz4CompressionEngine;
        assert!(engine.decompress(&[1, 2, 3]).is_err());
    }

    #[test]
    fn codecs_actually_shrink_compressible_data() {
        let original: Vec<u8> = b"PandRS compression test payload. "
            .iter()
            .copied()
            .cycle()
            .take(16 * 1024)
            .collect();

        for engine in [
            Box::new(Lz4CompressionEngine) as Box<dyn CompressionEngine>,
            Box::new(ZstdCompressionEngine::new(3)) as Box<dyn CompressionEngine>,
        ] {
            let compressed = engine.compress(&original).expect("compress");
            assert!(
                compressed.len() < original.len(),
                "{} did not shrink data: {} -> {}",
                engine.name(),
                original.len(),
                compressed.len()
            );
            assert_eq!(
                engine.decompress(&compressed).expect("decompress"),
                original
            );
        }
    }

    #[test]
    fn every_compression_type_has_an_engine() {
        let engines = build_engines(3);
        for ty in [
            CompressionType::None,
            CompressionType::Lz4,
            CompressionType::Zstd,
            CompressionType::Snappy,
            CompressionType::Gzip,
        ] {
            assert!(engines.contains_key(&ty), "missing engine for {:?}", ty);
        }
        // Auto must be resolved before use, never stored as a tag.
        assert_eq!(
            resolve_compression(CompressionType::Auto),
            CompressionType::Zstd
        );
        assert!(engines.contains_key(&resolve_compression(CompressionType::Auto)));
    }
}

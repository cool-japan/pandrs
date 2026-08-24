//! String compression codecs with an explicit codec tag.
//!
//! Every payload produced here starts with a one-byte algorithm tag. Without
//! it, `string_optimized_decompress` had to *guess*: it tried run-length
//! decoding first and fell back to LZ4. RLE-decoding a stream of LZ4 bytes
//! usually succeeds — producing up to 127x expanded garbage — and returned
//! `Ok`, so the corruption was invisible.

use crate::core::error::{Error, Result};
use crate::storage::adaptive_string_pool::dictionary::CompressionDictionary;
use std::sync::{Arc, RwLock};

/// String compression algorithm.
///
/// The discriminants are the on-the-wire codec tags and must stay stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StringCompressionAlgorithm {
    /// No compression
    None = 0,
    /// Run-length encoding for repetitive strings
    RunLength = 1,
    /// Dictionary compression
    Dictionary = 2,
    /// LZ4 compression
    Lz4 = 3,
    /// ZSTD compression
    Zstd = 4,
    /// Custom string-optimized compression (picks RLE or LZ4 and records which)
    StringOptimized = 5,
}

impl StringCompressionAlgorithm {
    /// On-the-wire tag byte.
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Parse an on-the-wire tag byte.
    pub fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(StringCompressionAlgorithm::None),
            1 => Ok(StringCompressionAlgorithm::RunLength),
            2 => Ok(StringCompressionAlgorithm::Dictionary),
            3 => Ok(StringCompressionAlgorithm::Lz4),
            4 => Ok(StringCompressionAlgorithm::Zstd),
            5 => Ok(StringCompressionAlgorithm::StringOptimized),
            other => Err(Error::InvalidOperation(format!(
                "Unknown string codec tag {}",
                other
            ))),
        }
    }
}

/// Maximum legitimate LZ4 expansion (see the block format's match limits).
const LZ4_MAX_EXPANSION: usize = 256;
/// Slack added to the ratio bound for very small payloads.
const LZ4_EXPANSION_SLACK: usize = 64 * 1024;
/// Hard ceiling on a single decompressed string.
const MAX_DECOMPRESSED_STRING: usize = 1 << 30; // 1 GiB
/// Maximum expansion of the byte run-length format.
const RLE_MAX_EXPANSION: usize = 255;

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn take_varint(data: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        if *offset >= data.len() {
            return Err(Error::InvalidOperation(
                "Truncated compressed string: unterminated varint".to_string(),
            ));
        }
        let byte = data[*offset];
        *offset += 1;
        if shift >= 64 {
            return Err(Error::InvalidOperation(
                "Corrupt compressed string: varint exceeds 64 bits".to_string(),
            ));
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

fn bounded_len(declared: u64, body_len: usize, max_expansion: usize) -> Result<usize> {
    let bound = body_len
        .saturating_mul(max_expansion)
        .saturating_add(LZ4_EXPANSION_SLACK)
        .min(MAX_DECOMPRESSED_STRING);
    if declared > bound as u64 {
        return Err(Error::InvalidOperation(format!(
            "Refusing string decompression bomb: header claims {} bytes from a {} byte body (bound {})",
            declared, body_len, bound
        )));
    }
    Ok(declared as usize)
}

/// String compression engine.
pub struct StringCompressionEngine {
    algorithm: StringCompressionAlgorithm,
    dictionary: Option<Arc<RwLock<CompressionDictionary>>>,
}

impl StringCompressionEngine {
    pub fn new(algorithm: StringCompressionAlgorithm) -> Self {
        Self {
            algorithm,
            dictionary: None,
        }
    }

    /// Build an engine backed by a shared, mutable dictionary.
    ///
    /// The dictionary is behind an `RwLock` because the old design shared it as
    /// a bare `Arc` and then tried `Arc::get_mut` to train it — which can never
    /// succeed once the `Arc` has been cloned into the engine map, so the
    /// dictionary was never built and `Dictionary` compression silently
    /// degraded to a raw copy.
    pub fn with_dictionary(
        algorithm: StringCompressionAlgorithm,
        dictionary: Arc<RwLock<CompressionDictionary>>,
    ) -> Self {
        Self {
            algorithm,
            dictionary: Some(dictionary),
        }
    }

    /// Algorithm this engine applies.
    pub fn algorithm(&self) -> StringCompressionAlgorithm {
        self.algorithm
    }

    /// Compress `data`, prefixing the codec tag.
    pub fn compress(&self, data: &str) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(data.len() / 2 + 16);
        out.push(self.algorithm.tag());
        match self.algorithm {
            StringCompressionAlgorithm::None => out.extend_from_slice(data.as_bytes()),
            StringCompressionAlgorithm::RunLength => {
                run_length_encode(data.as_bytes(), &mut out);
            }
            StringCompressionAlgorithm::Dictionary => {
                out.extend_from_slice(&self.dictionary_compress(data)?);
            }
            StringCompressionAlgorithm::Lz4 => {
                lz4_encode(data.as_bytes(), &mut out)?;
            }
            StringCompressionAlgorithm::Zstd => {
                zstd_encode(data.as_bytes(), &mut out)?;
            }
            StringCompressionAlgorithm::StringOptimized => {
                // Record which inner codec was chosen instead of guessing later.
                let mut rle = Vec::new();
                run_length_encode(data.as_bytes(), &mut rle);
                if !data.is_empty() && rle.len() < data.len() * 8 / 10 {
                    out.push(StringCompressionAlgorithm::RunLength.tag());
                    out.extend_from_slice(&rle);
                } else {
                    out.push(StringCompressionAlgorithm::Lz4.tag());
                    lz4_encode(data.as_bytes(), &mut out)?;
                }
            }
        }
        Ok(out)
    }

    /// Decompress a tagged payload.
    ///
    /// The tag — not a guess — selects the decoder, and a payload tagged with a
    /// different algorithm than this engine's is decoded correctly rather than
    /// mangled.
    pub fn decompress(&self, data: &[u8]) -> Result<String> {
        if data.is_empty() {
            return Err(Error::InvalidOperation(
                "Empty compressed string payload (every payload carries a codec tag)".to_string(),
            ));
        }
        let algorithm = StringCompressionAlgorithm::from_tag(data[0])?;
        let body = &data[1..];
        let bytes = match algorithm {
            StringCompressionAlgorithm::None => body.to_vec(),
            StringCompressionAlgorithm::RunLength => run_length_decode(body)?,
            StringCompressionAlgorithm::Dictionary => return self.dictionary_decompress(body),
            StringCompressionAlgorithm::Lz4 => lz4_decode(body)?,
            StringCompressionAlgorithm::Zstd => zstd_decode(body)?,
            StringCompressionAlgorithm::StringOptimized => {
                if body.is_empty() {
                    return Err(Error::InvalidOperation(
                        "Truncated string-optimized payload: missing inner codec tag".to_string(),
                    ));
                }
                match StringCompressionAlgorithm::from_tag(body[0])? {
                    StringCompressionAlgorithm::RunLength => run_length_decode(&body[1..])?,
                    StringCompressionAlgorithm::Lz4 => lz4_decode(&body[1..])?,
                    other => {
                        return Err(Error::InvalidOperation(format!(
                            "Unexpected inner codec {:?} in string-optimized payload",
                            other
                        )))
                    }
                }
            }
        };
        String::from_utf8(bytes)
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }

    fn dictionary_compress(&self, data: &str) -> Result<Vec<u8>> {
        match &self.dictionary {
            Some(dict) => {
                let guard = dict.read().map_err(|_| {
                    Error::InvalidOperation("Compression dictionary lock is poisoned".to_string())
                })?;
                guard.compress(data)
            }
            None => CompressionDictionary::new().compress(data),
        }
    }

    fn dictionary_decompress(&self, body: &[u8]) -> Result<String> {
        match &self.dictionary {
            Some(dict) => {
                let guard = dict.read().map_err(|_| {
                    Error::InvalidOperation("Compression dictionary lock is poisoned".to_string())
                })?;
                guard.decompress(body)
            }
            None => CompressionDictionary::new().decompress(body),
        }
    }
}

fn run_length_encode(bytes: &[u8], out: &mut Vec<u8>) {
    put_varint(out, bytes.len() as u64);
    let mut i = 0;
    while i < bytes.len() {
        let current = bytes[i];
        let mut count = 1u8;
        while i + (count as usize) < bytes.len()
            && bytes[i + (count as usize)] == current
            && count < 255
        {
            count += 1;
        }
        out.push(count);
        out.push(current);
        i += count as usize;
    }
}

fn run_length_decode(data: &[u8]) -> Result<Vec<u8>> {
    let mut offset = 0usize;
    let declared = take_varint(data, &mut offset)?;
    let expected = bounded_len(declared, data.len(), RLE_MAX_EXPANSION)?;
    let pairs = &data[offset..];
    if pairs.len() % 2 != 0 {
        return Err(Error::InvalidOperation(
            "Corrupt run-length payload: odd byte count".to_string(),
        ));
    }
    let mut decoded = Vec::with_capacity(expected.min(1 << 24));
    let mut i = 0;
    while i + 1 < pairs.len() {
        let count = pairs[i] as usize;
        let value = pairs[i + 1];
        if decoded.len() + count > expected {
            return Err(Error::InvalidOperation(
                "Corrupt run-length payload: output exceeds the declared length".to_string(),
            ));
        }
        decoded.resize(decoded.len() + count, value);
        i += 2;
    }
    if decoded.len() != expected {
        return Err(Error::InvalidOperation(format!(
            "Corrupt run-length payload: decoded {} bytes, expected {}",
            decoded.len(),
            expected
        )));
    }
    Ok(decoded)
}

fn lz4_encode(bytes: &[u8], out: &mut Vec<u8>) -> Result<()> {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    if bytes.is_empty() {
        return Ok(());
    }
    let body = oxiarc_lz4::compress_bytes(bytes)
        .map_err(|e| Error::InvalidOperation(format!("LZ4 compression failed: {}", e)))?;
    out.extend_from_slice(&body);
    Ok(())
}

fn lz4_decode(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        // Returning `Ok("")` for a short payload used to make corrupt input
        // read back as an empty string.
        return Err(Error::InvalidOperation(format!(
            "Truncated LZ4 payload: {} bytes, need at least 8",
            data.len()
        )));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[..8]);
    let declared = u64::from_le_bytes(len_bytes);
    if declared == 0 {
        return Ok(Vec::new());
    }
    let body = &data[8..];
    let original_len = bounded_len(declared, body.len(), LZ4_MAX_EXPANSION)?;
    let decoded = oxiarc_lz4::decompress_bytes(body, original_len)
        .map_err(|e| Error::InvalidOperation(format!("LZ4 decompression failed: {}", e)))?;
    if decoded.len() != original_len {
        return Err(Error::InvalidOperation(format!(
            "LZ4 decompression produced {} bytes, header claimed {}",
            decoded.len(),
            original_len
        )));
    }
    Ok(decoded)
}

fn zstd_encode(bytes: &[u8], out: &mut Vec<u8>) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    let body = oxiarc_zstd::compress_with_level(bytes, 3)
        .map_err(|e| Error::InvalidOperation(format!("ZSTD compression failed: {}", e)))?;
    out.extend_from_slice(&body);
    Ok(())
}

fn zstd_decode(data: &[u8]) -> Result<Vec<u8>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }
    oxiarc_zstd::decompress(data)
        .map_err(|e| Error::InvalidOperation(format!("ZSTD decompression failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [&str; 7] = [
        "",
        "a",
        "aaaaaabbbbbbcccccc",
        "plain ascii sentence with spaces",
        "日本語のテキストです",
        "€ £ ¥ mixed 🐟 emoji",
        "line1\nline2\ttabbed\r\n",
    ];

    #[test]
    fn every_algorithm_round_trips_every_sample() {
        for algorithm in [
            StringCompressionAlgorithm::None,
            StringCompressionAlgorithm::RunLength,
            StringCompressionAlgorithm::Lz4,
            StringCompressionAlgorithm::Zstd,
            StringCompressionAlgorithm::StringOptimized,
        ] {
            let engine = StringCompressionEngine::new(algorithm);
            for sample in SAMPLES {
                let compressed = engine.compress(sample).expect("compress");
                assert_eq!(compressed[0], algorithm.tag(), "missing codec tag");
                assert_eq!(
                    engine.decompress(&compressed).expect("decompress"),
                    sample,
                    "{:?} corrupted {:?}",
                    algorithm,
                    sample
                );
            }
        }
    }

    #[test]
    fn a_payload_decodes_correctly_through_any_engine() {
        // The tag travels with the data, so the engine that happens to be
        // reachable at read time no longer determines the decoder.
        let payload = StringCompressionEngine::new(StringCompressionAlgorithm::Lz4)
            .compress("high entropy payload qwertyuiop asdfghjkl")
            .expect("compress");
        let other = StringCompressionEngine::new(StringCompressionAlgorithm::RunLength);
        assert_eq!(
            other.decompress(&payload).expect("decompress"),
            "high entropy payload qwertyuiop asdfghjkl"
        );
    }

    #[test]
    fn lz4_bomb_is_refused() {
        let engine = StringCompressionEngine::new(StringCompressionAlgorithm::Lz4);
        let mut hostile = vec![StringCompressionAlgorithm::Lz4.tag()];
        hostile.extend_from_slice(&u64::MAX.to_le_bytes());
        hostile.extend_from_slice(&[0, 0, 0, 0]);
        assert!(engine.decompress(&hostile).is_err());
    }

    #[test]
    fn truncated_and_empty_payloads_are_errors() {
        let engine = StringCompressionEngine::new(StringCompressionAlgorithm::Lz4);
        assert!(engine.decompress(&[]).is_err());
        assert!(engine
            .decompress(&[StringCompressionAlgorithm::Lz4.tag(), 1, 2, 3])
            .is_err());
        assert!(engine.decompress(&[0xAB, 1, 2]).is_err());
    }

    #[test]
    fn run_length_bomb_is_refused() {
        let mut hostile = vec![StringCompressionAlgorithm::RunLength.tag()];
        // varint u64::MAX
        put_varint(&mut hostile, u64::MAX);
        hostile.extend_from_slice(&[255, b'a']);
        let engine = StringCompressionEngine::new(StringCompressionAlgorithm::RunLength);
        assert!(engine.decompress(&hostile).is_err());
    }

    #[test]
    fn dictionary_engine_round_trips_through_shared_lock() {
        let dictionary = Arc::new(RwLock::new(CompressionDictionary::new()));
        {
            let mut guard = dictionary.write().expect("lock");
            guard
                .build_from_strings(&[
                    "the quick brown fox".to_string(),
                    "the lazy dog".to_string(),
                ])
                .expect("build");
            assert!(guard.len() > 0, "dictionary was never trained");
        }
        let engine = StringCompressionEngine::with_dictionary(
            StringCompressionAlgorithm::Dictionary,
            Arc::clone(&dictionary),
        );
        for sample in ["the quick brown fox", "the\tlazy  dog", "日本語 the"] {
            let compressed = engine.compress(sample).expect("compress");
            assert_eq!(engine.decompress(&compressed).expect("decompress"), sample);
        }
    }
}

//! Shared integrity checksums for the PandRS storage engines.
//!
//! Storage blocks, tiered chunks and cached payloads all need a cheap, order
//! sensitive integrity check. Historically each engine rolled its own (the
//! column store summed the bytes, which cannot detect reordering or byte
//! swaps, and the unified `ChunkMetadata` simply hardcoded `0`).
//!
//! This module provides a single Pure-Rust CRC-32C (Castagnoli) implementation
//! plus a 64-bit variant that folds the payload length in, so that truncation
//! is detected as well.

/// CRC-32C (Castagnoli) reversed polynomial.
const CRC32C_POLY: u32 = 0x82F6_3B78;

/// Lookup table for a byte-at-a-time CRC-32C.
const CRC32C_TABLE: [u32; 256] = build_crc32c_table();

const fn build_crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32C_POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute the CRC-32C of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32C_TABLE[index];
    }
    !crc
}

/// Compute a 64-bit integrity checksum: CRC-32C in the low half, and the
/// CRC-32C of the little-endian payload length in the high half.
///
/// Folding the length in means a truncated payload whose prefix happens to
/// share a CRC still mismatches.
pub fn checksum64(data: &[u8]) -> u64 {
    let body = crc32c(data) as u64;
    let len = crc32c(&(data.len() as u64).to_le_bytes()) as u64;
    (len << 32) | body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_matches_known_vectors() {
        // Standard CRC-32C test vectors (RFC 3720 / iSCSI appendix).
        assert_eq!(crc32c(b""), 0x0000_0000);
        assert_eq!(crc32c(b"a"), 0xC1D0_4330);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn crc32c_detects_reordering() {
        // A plain byte-sum checksum (the old implementation) cannot tell these
        // two payloads apart; CRC-32C must.
        assert_ne!(crc32c(b"ab"), crc32c(b"ba"));
    }

    #[test]
    fn checksum64_detects_truncation() {
        let full = b"pandrs storage payload";
        assert_ne!(checksum64(full), checksum64(&full[..full.len() - 1]));
    }
}

//! Column encodings for the unified column store.
//!
//! Every codec here is **self-describing**: everything the decoder needs lives
//! in the encoded byte stream, because `EncodedData::metadata` is not
//! persisted with the block. That property is what makes write-then-read work
//! after a block has been round-tripped through physical storage.
//!
//! Historically only `RunLength` was registered while `create_storage` happily
//! selected `Dictionary` (categorical data) or `Delta` (time series) and the
//! default config selected `Auto`; blocks were then stored raw but *tagged*
//! with the unimplemented encoding, so every read failed with
//! "Encoding strategy ... not found". All four encodings are now real.

use crate::core::error::{Error, Result};
use std::collections::HashMap;

/// Encoding strategies for different data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EncodingType {
    /// Automatic encoding selection (resolved to a concrete encoding at write time)
    Auto,
    /// No encoding (raw data)
    None,
    /// Run-length encoding
    RunLength,
    /// Dictionary encoding
    Dictionary,
    /// Delta encoding for numeric data
    Delta,
    /// Bit-packed encoding for small integers
    BitPacked,
}

/// Encoding strategy trait
pub trait EncodingStrategy: Send + Sync {
    fn encode(&self, data: &[u8]) -> Result<EncodedData>;
    fn decode(&self, data: &EncodedData) -> Result<Vec<u8>>;
    fn name(&self) -> &'static str;
    fn encoding_ratio(&self, original_size: usize, encoded_size: usize) -> f64;
}

/// Encoded data with metadata
#[derive(Debug, Clone)]
pub struct EncodedData {
    pub data: Vec<u8>,
    pub encoding_type: EncodingType,
    pub original_size: usize,
    pub metadata: HashMap<String, String>,
}

impl EncodedData {
    fn new(data: Vec<u8>, encoding_type: EncodingType, original_size: usize) -> Self {
        Self {
            data,
            encoding_type,
            original_size,
            metadata: HashMap::new(),
        }
    }
}

/// Candidate fixed element widths probed by the numeric codecs.
const ELEMENT_WIDTHS: [usize; 4] = [1, 2, 4, 8];
/// Version byte written by every self-describing codec in this module.
const CODEC_VERSION: u8 = 1;
/// Hard ceiling for a single decoded block (guards hostile length headers).
const MAX_DECODED_BLOCK: usize = 1 << 32; // 4 GiB

// ---------------------------------------------------------------------------
// Shared little-endian helpers
// ---------------------------------------------------------------------------

fn read_le_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        value |= (b as u64) << (8 * i);
    }
    value
}

fn write_le_uint(out: &mut Vec<u8>, value: u64, width: usize) {
    let bytes = value.to_le_bytes();
    out.extend_from_slice(&bytes[..width.min(8)]);
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn take_u64(data: &[u8], offset: &mut usize) -> Result<u64> {
    if *offset + 8 > data.len() {
        return Err(Error::InvalidValue(
            "Truncated encoded block: missing 64-bit header field".to_string(),
        ));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[*offset..*offset + 8]);
    *offset += 8;
    Ok(u64::from_le_bytes(buf))
}

fn take_u8(data: &[u8], offset: &mut usize) -> Result<u8> {
    if *offset >= data.len() {
        return Err(Error::InvalidValue(
            "Truncated encoded block: missing header byte".to_string(),
        ));
    }
    let value = data[*offset];
    *offset += 1;
    Ok(value)
}

fn check_original_len(
    original_len: u64,
    encoded_len: usize,
    max_expansion: usize,
) -> Result<usize> {
    let bound = encoded_len
        .saturating_mul(max_expansion)
        .saturating_add(4096)
        .min(MAX_DECODED_BLOCK);
    if original_len > bound as u64 {
        return Err(Error::InvalidValue(format!(
            "Refusing encoded-block expansion bomb: header claims {} bytes from {} encoded bytes (bound {})",
            original_len, encoded_len, bound
        )));
    }
    Ok(original_len as usize)
}

fn validate_width(width: u8) -> Result<usize> {
    let width = width as usize;
    if ELEMENT_WIDTHS.contains(&width) {
        Ok(width)
    } else {
        Err(Error::InvalidValue(format!(
            "Unsupported element width {} in encoded block",
            width
        )))
    }
}

// ---------------------------------------------------------------------------
// Run-length encoding
// ---------------------------------------------------------------------------

/// Run-length encoding strategy.
///
/// Wire format: repeated `(count: u8, byte: u8)` pairs.
pub struct RunLengthEncodingStrategy;

/// Maximum expansion of the RLE format (one pair yields at most 255 bytes).
const RLE_MAX_EXPANSION: usize = 255;

impl EncodingStrategy for RunLengthEncodingStrategy {
    fn encode(&self, data: &[u8]) -> Result<EncodedData> {
        let mut encoded = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let current_byte = data[i];
            let mut count = 1u8;

            while i + (count as usize) < data.len()
                && data[i + (count as usize)] == current_byte
                && count < 255
            {
                count += 1;
            }

            encoded.push(count);
            encoded.push(current_byte);
            i += count as usize;
        }

        Ok(EncodedData::new(
            encoded,
            EncodingType::RunLength,
            data.len(),
        ))
    }

    fn decode(&self, encoded: &EncodedData) -> Result<Vec<u8>> {
        if encoded.data.len() % 2 != 0 {
            return Err(Error::InvalidValue(
                "Corrupt run-length block: odd byte count".to_string(),
            ));
        }
        // The expected length is authoritative; without this the decoder would
        // happily expand a hostile stream far past the block's real size.
        let expected = check_original_len(
            encoded.original_size as u64,
            encoded.data.len(),
            RLE_MAX_EXPANSION,
        )?;

        let mut decoded = Vec::with_capacity(expected.min(1 << 24));
        let mut i = 0;
        while i + 1 < encoded.data.len() {
            let count = encoded.data[i] as usize;
            let byte_val = encoded.data[i + 1];
            if decoded.len() + count > expected {
                return Err(Error::InvalidValue(format!(
                    "Corrupt run-length block: decoded output exceeds the recorded original size {}",
                    expected
                )));
            }
            decoded.resize(decoded.len() + count, byte_val);
            i += 2;
        }

        if decoded.len() != expected {
            return Err(Error::InvalidValue(format!(
                "Corrupt run-length block: decoded {} bytes, expected {}",
                decoded.len(),
                expected
            )));
        }
        Ok(decoded)
    }

    fn name(&self) -> &'static str {
        "RunLength"
    }

    fn encoding_ratio(&self, original_size: usize, encoded_size: usize) -> f64 {
        if encoded_size == 0 {
            0.0
        } else {
            original_size as f64 / encoded_size as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Dictionary encoding
// ---------------------------------------------------------------------------

/// Dictionary encoding strategy for low-cardinality fixed-width symbols.
///
/// Wire format:
/// `u8 version | u8 symbol_width | u8 index_width | u64 original_len |
///  u32 dict_len | dict_len*symbol_width dictionary | n*index_width indices |
///  tail bytes`
pub struct DictionaryEncodingStrategy;

/// Dictionary indices can expand at most `symbol_width / index_width` = 8x.
const DICT_MAX_EXPANSION: usize = 8;

fn encode_dictionary_with_width(data: &[u8], symbol_width: usize) -> Option<Vec<u8>> {
    let symbol_count = data.len() / symbol_width;
    if symbol_count == 0 {
        return None;
    }
    let mut dictionary: Vec<u64> = Vec::new();
    let mut lookup: HashMap<u64, u32> = HashMap::new();
    let mut indices: Vec<u32> = Vec::with_capacity(symbol_count);

    for i in 0..symbol_count {
        let symbol = read_le_uint(&data[i * symbol_width..(i + 1) * symbol_width]);
        let next_id = dictionary.len() as u32;
        let id = *lookup.entry(symbol).or_insert_with(|| {
            dictionary.push(symbol);
            next_id
        });
        indices.push(id);
    }

    // Dictionary encoding only pays off when the alphabet is small.
    let index_width = if dictionary.len() <= u8::MAX as usize + 1 {
        1usize
    } else if dictionary.len() <= u16::MAX as usize + 1 {
        2
    } else {
        4
    };
    if index_width >= symbol_width && dictionary.len() * 2 >= symbol_count {
        return None;
    }

    let tail = &data[symbol_count * symbol_width..];
    let mut out = Vec::with_capacity(
        15 + dictionary.len() * symbol_width + indices.len() * index_width + tail.len(),
    );
    out.push(CODEC_VERSION);
    out.push(symbol_width as u8);
    out.push(index_width as u8);
    put_u64(&mut out, data.len() as u64);
    out.extend_from_slice(&(dictionary.len() as u32).to_le_bytes());
    for symbol in &dictionary {
        write_le_uint(&mut out, *symbol, symbol_width);
    }
    for index in &indices {
        write_le_uint(&mut out, *index as u64, index_width);
    }
    out.extend_from_slice(tail);
    Some(out)
}

impl EncodingStrategy for DictionaryEncodingStrategy {
    fn encode(&self, data: &[u8]) -> Result<EncodedData> {
        let best = ELEMENT_WIDTHS
            .iter()
            .filter_map(|&w| encode_dictionary_with_width(data, w))
            .min_by_key(|encoded| encoded.len());

        // No width produced a usable dictionary (tiny or high-cardinality
        // payload): emit the explicit `index_width == 0` verbatim form, which
        // the decoder copies straight back out.
        let encoded = match best {
            Some(encoded) => encoded,
            None => {
                let mut out = Vec::with_capacity(15 + data.len());
                out.push(CODEC_VERSION);
                out.push(1); // symbol width (unused when index_width == 0)
                out.push(0); // index width 0 == verbatim payload
                put_u64(&mut out, data.len() as u64);
                out.extend_from_slice(&0u32.to_le_bytes()); // empty dictionary
                out.extend_from_slice(data);
                out
            }
        };

        Ok(EncodedData::new(
            encoded,
            EncodingType::Dictionary,
            data.len(),
        ))
    }

    fn decode(&self, encoded: &EncodedData) -> Result<Vec<u8>> {
        let data = &encoded.data;
        let mut offset = 0usize;
        let version = take_u8(data, &mut offset)?;
        if version != CODEC_VERSION {
            return Err(Error::InvalidValue(format!(
                "Unsupported dictionary block version {}",
                version
            )));
        }
        let symbol_width = validate_width(take_u8(data, &mut offset)?)?;
        let index_width = take_u8(data, &mut offset)? as usize;
        if !matches!(index_width, 0 | 1 | 2 | 4) {
            return Err(Error::InvalidValue(format!(
                "Unsupported dictionary index width {}",
                index_width
            )));
        }
        let original_len =
            check_original_len(take_u64(data, &mut offset)?, data.len(), DICT_MAX_EXPANSION)?;
        if offset + 4 > data.len() {
            return Err(Error::InvalidValue(
                "Truncated dictionary block: missing dictionary length".to_string(),
            ));
        }
        let dict_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if index_width == 0 {
            // Verbatim form: no dictionary, no indices, payload copied as-is.
            if dict_len != 0 || offset + original_len > data.len() {
                return Err(Error::InvalidValue(
                    "Corrupt verbatim dictionary block".to_string(),
                ));
            }
            return Ok(data[offset..offset + original_len].to_vec());
        }

        let dict_bytes = dict_len
            .checked_mul(symbol_width)
            .ok_or_else(|| Error::InvalidValue("Dictionary block size overflow".to_string()))?;
        if offset + dict_bytes > data.len() {
            return Err(Error::InvalidValue(
                "Truncated dictionary block: dictionary body".to_string(),
            ));
        }
        let mut dictionary = Vec::with_capacity(dict_len);
        for i in 0..dict_len {
            let start = offset + i * symbol_width;
            dictionary.push(read_le_uint(&data[start..start + symbol_width]));
        }
        offset += dict_bytes;

        let symbol_count = original_len / symbol_width;
        let tail_len = original_len % symbol_width;
        let index_bytes = symbol_count
            .checked_mul(index_width)
            .ok_or_else(|| Error::InvalidValue("Dictionary index size overflow".to_string()))?;
        if offset + index_bytes + tail_len > data.len() {
            return Err(Error::InvalidValue(
                "Truncated dictionary block: index body".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(original_len);
        for i in 0..symbol_count {
            let start = offset + i * index_width;
            let id = read_le_uint(&data[start..start + index_width]) as usize;
            // Out-of-range ids used to be skipped silently, which dropped rows
            // and misaligned every column downstream.
            let symbol = *dictionary.get(id).ok_or_else(|| {
                Error::InvalidValue(format!(
                    "Corrupt dictionary block: index {} references entry {} of {}",
                    i,
                    id,
                    dictionary.len()
                ))
            })?;
            write_le_uint(&mut out, symbol, symbol_width);
        }
        offset += index_bytes;
        out.extend_from_slice(&data[offset..offset + tail_len]);

        if out.len() != original_len {
            return Err(Error::InvalidValue(format!(
                "Corrupt dictionary block: decoded {} bytes, expected {}",
                out.len(),
                original_len
            )));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "Dictionary"
    }

    fn encoding_ratio(&self, original_size: usize, encoded_size: usize) -> f64 {
        if encoded_size == 0 {
            0.0
        } else {
            original_size as f64 / encoded_size as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Delta encoding
// ---------------------------------------------------------------------------

/// Delta encoding strategy for monotonic / slowly-varying numeric sequences.
///
/// Wire format:
/// `u8 version | u8 element_width | u64 original_len | zigzag varint deltas |
///  tail bytes`
///
/// Deltas are computed with wrapping arithmetic modulo `2^(8*width)`, so the
/// transform is exactly reversible for every bit pattern, signed or unsigned.
pub struct DeltaEncodingStrategy;

/// A varint is at least one byte per element, so the widest element expands 8x.
const DELTA_MAX_EXPANSION: usize = 8;

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

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
            return Err(Error::InvalidValue(
                "Truncated delta block: unterminated varint".to_string(),
            ));
        }
        let byte = data[*offset];
        *offset += 1;
        if shift >= 64 {
            return Err(Error::InvalidValue(
                "Corrupt delta block: varint exceeds 64 bits".to_string(),
            ));
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

fn width_mask(width: usize) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * width)) - 1
    }
}

fn encode_delta_with_width(data: &[u8], width: usize) -> Option<Vec<u8>> {
    let count = data.len() / width;
    if count == 0 {
        return None;
    }
    let mask = width_mask(width);
    let mut out = Vec::with_capacity(10 + count + data.len() % width);
    out.push(CODEC_VERSION);
    out.push(width as u8);
    put_u64(&mut out, data.len() as u64);

    let mut previous = 0u64;
    for i in 0..count {
        let value = read_le_uint(&data[i * width..(i + 1) * width]) & mask;
        let delta = value.wrapping_sub(previous) & mask;
        // Interpret the modular delta as a signed value so that small negative
        // steps stay small after zigzag.
        let signed = if width >= 8 {
            delta as i64
        } else {
            let sign_bit = 1u64 << (8 * width - 1);
            if delta & sign_bit != 0 {
                (delta as i64) - ((mask as i64) + 1)
            } else {
                delta as i64
            }
        };
        put_varint(&mut out, zigzag_encode(signed));
        previous = value;
    }
    out.extend_from_slice(&data[count * width..]);
    Some(out)
}

impl EncodingStrategy for DeltaEncodingStrategy {
    fn encode(&self, data: &[u8]) -> Result<EncodedData> {
        let best = ELEMENT_WIDTHS
            .iter()
            .filter_map(|&w| encode_delta_with_width(data, w))
            .min_by_key(|encoded| encoded.len());

        let encoded = match best {
            Some(encoded) => encoded,
            None => {
                // Payload shorter than one element: header + verbatim tail.
                let mut out = Vec::with_capacity(10 + data.len());
                out.push(CODEC_VERSION);
                out.push(1);
                put_u64(&mut out, data.len() as u64);
                out.extend_from_slice(data);
                out
            }
        };

        Ok(EncodedData::new(encoded, EncodingType::Delta, data.len()))
    }

    fn decode(&self, encoded: &EncodedData) -> Result<Vec<u8>> {
        let data = &encoded.data;
        let mut offset = 0usize;
        let version = take_u8(data, &mut offset)?;
        if version != CODEC_VERSION {
            return Err(Error::InvalidValue(format!(
                "Unsupported delta block version {}",
                version
            )));
        }
        let width = validate_width(take_u8(data, &mut offset)?)?;
        let original_len = check_original_len(
            take_u64(data, &mut offset)?,
            data.len(),
            DELTA_MAX_EXPANSION,
        )?;

        let count = original_len / width;
        let tail_len = original_len % width;
        let mask = width_mask(width);

        let mut out = Vec::with_capacity(original_len);
        let mut previous = 0u64;
        for _ in 0..count {
            let delta = zigzag_decode(take_varint(data, &mut offset)?);
            let value = previous.wrapping_add(delta as u64) & mask;
            write_le_uint(&mut out, value, width);
            previous = value;
        }
        if offset + tail_len > data.len() {
            return Err(Error::InvalidValue(
                "Truncated delta block: missing tail bytes".to_string(),
            ));
        }
        out.extend_from_slice(&data[offset..offset + tail_len]);

        if out.len() != original_len {
            return Err(Error::InvalidValue(format!(
                "Corrupt delta block: decoded {} bytes, expected {}",
                out.len(),
                original_len
            )));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "Delta"
    }

    fn encoding_ratio(&self, original_size: usize, encoded_size: usize) -> f64 {
        if encoded_size == 0 {
            0.0
        } else {
            original_size as f64 / encoded_size as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

/// Bit-packed encoding for fixed-width integers with a small dynamic range.
///
/// Wire format:
/// `u8 version | u8 element_width | u8 bits_per_value | u64 original_len |
///  packed bits (LSB-first) | tail bytes`
pub struct BitPackedEncodingStrategy;

/// A 1-bit value expands to at most 8 bytes, so 64x is the format's bound.
const BITPACK_MAX_EXPANSION: usize = 64;

fn bits_needed(max_value: u64) -> u8 {
    if max_value == 0 {
        0
    } else {
        64 - max_value.leading_zeros() as u8
    }
}

fn encode_bitpacked_with_width(data: &[u8], width: usize) -> Option<Vec<u8>> {
    let count = data.len() / width;
    if count == 0 {
        return None;
    }
    let mut max_value = 0u64;
    for i in 0..count {
        max_value = max_value.max(read_le_uint(&data[i * width..(i + 1) * width]));
    }
    let bits = bits_needed(max_value);
    if bits as usize >= width * 8 {
        // Nothing to gain over the raw representation.
        return None;
    }

    let tail = &data[count * width..];
    // `(n + 7) / 8` rather than `div_ceil`: the crate's MSRV is 1.70 and
    // `usize::div_ceil` only stabilised in 1.73.
    let packed_bytes = (count.saturating_mul(bits as usize) + 7) / 8;
    let mut out = Vec::with_capacity(11 + packed_bytes + tail.len());
    out.push(CODEC_VERSION);
    out.push(width as u8);
    out.push(bits);
    put_u64(&mut out, data.len() as u64);

    if bits > 0 {
        // A u128 staging buffer is wide enough for the worst case (7 leftover
        // bits + a 63-bit value); a u64 buffer would silently drop the high
        // bits of wide values.
        let mut buffer = 0u128;
        let mut buffered_bits = 0u32;
        for i in 0..count {
            let value = read_le_uint(&data[i * width..(i + 1) * width]);
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
    }
    out.extend_from_slice(tail);
    Some(out)
}

impl EncodingStrategy for BitPackedEncodingStrategy {
    fn encode(&self, data: &[u8]) -> Result<EncodedData> {
        let best = ELEMENT_WIDTHS
            .iter()
            .filter_map(|&w| encode_bitpacked_with_width(data, w))
            .min_by_key(|encoded| encoded.len());

        let encoded = match best {
            Some(encoded) => encoded,
            None => {
                // No width wins; store verbatim with an 8-bit element header so
                // the block still decodes deterministically.
                let mut out = Vec::with_capacity(11 + data.len());
                out.push(CODEC_VERSION);
                out.push(1);
                out.push(8);
                put_u64(&mut out, data.len() as u64);
                out.extend_from_slice(data);
                out
            }
        };

        Ok(EncodedData::new(
            encoded,
            EncodingType::BitPacked,
            data.len(),
        ))
    }

    fn decode(&self, encoded: &EncodedData) -> Result<Vec<u8>> {
        let data = &encoded.data;
        let mut offset = 0usize;
        let version = take_u8(data, &mut offset)?;
        if version != CODEC_VERSION {
            return Err(Error::InvalidValue(format!(
                "Unsupported bit-packed block version {}",
                version
            )));
        }
        let width = validate_width(take_u8(data, &mut offset)?)?;
        let bits = take_u8(data, &mut offset)?;
        if bits as usize > width * 8 {
            return Err(Error::InvalidValue(format!(
                "Corrupt bit-packed block: {} bits per {}-byte value",
                bits, width
            )));
        }
        let original_len = check_original_len(
            take_u64(data, &mut offset)?,
            data.len(),
            BITPACK_MAX_EXPANSION,
        )?;

        let count = original_len / width;
        let tail_len = original_len % width;
        let packed_bytes = (count.saturating_mul(bits as usize) + 7) / 8;
        if offset + packed_bytes + tail_len > data.len() {
            return Err(Error::InvalidValue(
                "Truncated bit-packed block".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(original_len);
        if bits == 0 {
            for _ in 0..count {
                write_le_uint(&mut out, 0, width);
            }
        } else {
            let packed = &data[offset..offset + packed_bytes];
            let mask = if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let mut bit_pos = 0usize;
            for _ in 0..count {
                let mut value = 0u64;
                for b in 0..bits as usize {
                    let absolute = bit_pos + b;
                    let byte = packed[absolute / 8];
                    let bit = (byte >> (absolute % 8)) & 1;
                    value |= (bit as u64) << b;
                }
                write_le_uint(&mut out, value & mask, width);
                bit_pos += bits as usize;
            }
        }
        offset += packed_bytes;
        out.extend_from_slice(&data[offset..offset + tail_len]);

        if out.len() != original_len {
            return Err(Error::InvalidValue(format!(
                "Corrupt bit-packed block: decoded {} bytes, expected {}",
                out.len(),
                original_len
            )));
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "BitPacked"
    }

    fn encoding_ratio(&self, original_size: usize, encoded_size: usize) -> f64 {
        if encoded_size == 0 {
            0.0
        } else {
            original_size as f64 / encoded_size as f64
        }
    }
}

/// Build the encoding registry used by the column store.
///
/// Every non-`Auto` encoding type has an entry, so a block can never be tagged
/// with an encoding that has no decoder.
pub fn build_encodings() -> HashMap<EncodingType, Box<dyn EncodingStrategy>> {
    let mut encodings: HashMap<EncodingType, Box<dyn EncodingStrategy>> = HashMap::new();
    encodings.insert(EncodingType::RunLength, Box::new(RunLengthEncodingStrategy));
    encodings.insert(
        EncodingType::Dictionary,
        Box::new(DictionaryEncodingStrategy),
    );
    encodings.insert(EncodingType::Delta, Box::new(DeltaEncodingStrategy));
    encodings.insert(EncodingType::BitPacked, Box::new(BitPackedEncodingStrategy));
    encodings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(strategy: &dyn EncodingStrategy, data: &[u8]) {
        let encoded = strategy.encode(data).expect("encode");
        let decoded = strategy.decode(&encoded).expect("decode");
        assert_eq!(decoded, data, "{} round-trip mismatch", strategy.name());
    }

    #[test]
    fn run_length_roundtrip_and_shrinks() {
        let strategy = RunLengthEncodingStrategy;
        let data = b"aaaaabbbbcccccddddd";
        roundtrip(&strategy, data);
        let encoded = strategy.encode(data).expect("encode");
        assert!(encoded.data.len() < data.len());
    }

    #[test]
    fn run_length_rejects_length_mismatch() {
        let strategy = RunLengthEncodingStrategy;
        let mut encoded = strategy.encode(b"aaaa").expect("encode");
        encoded.original_size = 3; // lie about the size
        assert!(strategy.decode(&encoded).is_err());
    }

    #[test]
    fn dictionary_roundtrip_all_widths() {
        let strategy = DictionaryEncodingStrategy;
        // Low-cardinality u32 categorical column.
        let mut data = Vec::new();
        for i in 0..1000u32 {
            data.extend_from_slice(&(i % 5).to_le_bytes());
        }
        roundtrip(&strategy, &data);
        let encoded = strategy.encode(&data).expect("encode");
        assert!(
            encoded.data.len() < data.len(),
            "dictionary encoding did not shrink categorical data: {} -> {}",
            data.len(),
            encoded.data.len()
        );

        roundtrip(&strategy, b"");
        roundtrip(&strategy, b"abc");
        roundtrip(&strategy, &[7u8; 33]);
    }

    #[test]
    fn dictionary_rejects_out_of_range_index() {
        // Hand-built block: 2 symbols of width 4, a 1-entry dictionary, and an
        // index that points past the end of it. The old decoder skipped such
        // indices silently, dropping rows.
        let mut block = vec![CODEC_VERSION, 4, 1];
        block.extend_from_slice(&8u64.to_le_bytes()); // original_len
        block.extend_from_slice(&1u32.to_le_bytes()); // dict_len
        block.extend_from_slice(&42u32.to_le_bytes()); // dictionary entry 0
        block.push(0); // index 0 -> valid
        block.push(5); // index 1 -> out of range
        let encoded = EncodedData::new(block, EncodingType::Dictionary, 8);
        let err = DictionaryEncodingStrategy
            .decode(&encoded)
            .expect_err("out-of-range index must be an error");
        assert!(format!("{}", err).contains("references entry"));
    }

    #[test]
    fn dictionary_verbatim_form_roundtrips_single_byte() {
        // Payloads too small or too diverse for a dictionary use the explicit
        // verbatim form; it must still decode byte-for-byte.
        let strategy = DictionaryEncodingStrategy;
        for payload in [vec![0u8], vec![9u8], (0..=255u8).collect::<Vec<u8>>()] {
            roundtrip(&strategy, &payload);
        }
    }

    #[test]
    fn delta_roundtrip_monotonic_and_random() {
        let strategy = DeltaEncodingStrategy;
        let mut monotonic = Vec::new();
        for i in 0..1000u64 {
            monotonic.extend_from_slice(&(1_600_000_000u64 + i).to_le_bytes());
        }
        roundtrip(&strategy, &monotonic);
        let encoded = strategy.encode(&monotonic).expect("encode");
        assert!(
            encoded.data.len() < monotonic.len(),
            "delta encoding did not shrink a timestamp column: {} -> {}",
            monotonic.len(),
            encoded.data.len()
        );

        // Decreasing sequence exercises the negative-delta path.
        let mut decreasing = Vec::new();
        for i in 0..500u32 {
            decreasing.extend_from_slice(&(10_000u32 - i * 3).to_le_bytes());
        }
        roundtrip(&strategy, &decreasing);

        // Arbitrary bytes must still round-trip exactly.
        let noisy: Vec<u8> = (0..777u32).map(|i| (i * 37 % 251) as u8).collect();
        roundtrip(&strategy, &noisy);
        roundtrip(&strategy, b"");
        roundtrip(&strategy, &[0xFFu8; 9]);
    }

    #[test]
    fn bitpacked_roundtrip_and_shrinks_small_integers() {
        let strategy = BitPackedEncodingStrategy;
        let mut data = Vec::new();
        for i in 0..1024u32 {
            data.extend_from_slice(&(i % 8).to_le_bytes()); // 3 bits of range
        }
        roundtrip(&strategy, &data);
        let encoded = strategy.encode(&data).expect("encode");
        assert!(
            encoded.data.len() < data.len() / 2,
            "bit packing did not shrink 3-bit values: {} -> {}",
            data.len(),
            encoded.data.len()
        );

        roundtrip(&strategy, &[0u8; 64]); // all zeroes -> zero bits per value
        roundtrip(&strategy, b"");
        roundtrip(&strategy, b"xyz");
        let full_range: Vec<u8> = (0..=255u8).collect();
        roundtrip(&strategy, &full_range);
    }

    #[test]
    fn every_encoding_type_has_a_strategy() {
        let encodings = build_encodings();
        for ty in [
            EncodingType::RunLength,
            EncodingType::Dictionary,
            EncodingType::Delta,
            EncodingType::BitPacked,
        ] {
            assert!(encodings.contains_key(&ty), "missing encoding for {:?}", ty);
        }
    }

    #[test]
    fn decoders_reject_expansion_bombs() {
        // A tiny body whose header claims an enormous original size.
        let mut dict = vec![CODEC_VERSION, 8, 1];
        dict.extend_from_slice(&u64::MAX.to_le_bytes());
        dict.extend_from_slice(&0u32.to_le_bytes());
        let encoded = EncodedData::new(dict, EncodingType::Dictionary, 0);
        assert!(DictionaryEncodingStrategy.decode(&encoded).is_err());

        let mut delta = vec![CODEC_VERSION, 8];
        delta.extend_from_slice(&u64::MAX.to_le_bytes());
        let encoded = EncodedData::new(delta, EncodingType::Delta, 0);
        assert!(DeltaEncodingStrategy.decode(&encoded).is_err());

        let mut packed = vec![CODEC_VERSION, 8, 1];
        packed.extend_from_slice(&u64::MAX.to_le_bytes());
        let encoded = EncodedData::new(packed, EncodingType::BitPacked, 0);
        assert!(BitPackedEncodingStrategy.decode(&encoded).is_err());
    }
}

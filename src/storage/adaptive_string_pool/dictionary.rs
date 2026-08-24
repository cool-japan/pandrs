//! Lossless word-dictionary compression for string columns.
//!
//! The previous implementation was lossy and corruptible:
//!
//! * it emitted `word.len() as u8` as a literal-length byte, so a 255-byte word
//!   produced `0xFF` — the very byte used as the dictionary marker — and any
//!   word longer than 255 bytes had its length truncated;
//! * it rebuilt the text with `split_whitespace()` + `" "`, destroying tabs,
//!   newlines and runs of spaces;
//! * an out-of-range dictionary id was skipped silently, dropping a word.
//!
//! The format below is an explicit op stream that reproduces the input byte for
//! byte, uses varints for every length, and rejects unresolvable ids.

use crate::core::error::{Error, Result};
use std::collections::HashMap;

/// Format version written at the head of every dictionary blob.
const DICT_CODEC_VERSION: u8 = 1;
/// Op code: literal byte run.
const OP_LITERAL: u8 = 0;
/// Op code: dictionary reference.
const OP_REFERENCE: u8 = 1;
/// Op code: a single space followed by a dictionary reference.
///
/// Word-separated text is overwhelmingly `word SPACE word SPACE ...`; without
/// this fused op each separator costs a 3-byte literal run and the encoding is
/// larger than the input it replaces.
const OP_SPACE_REFERENCE: u8 = 2;
/// Words longer than this are never added to the dictionary.
const MAX_DICTIONARY_WORD_LEN: usize = 4096;

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
                "Truncated dictionary blob: unterminated varint".to_string(),
            ));
        }
        let byte = data[*offset];
        *offset += 1;
        if shift >= 64 {
            return Err(Error::InvalidOperation(
                "Corrupt dictionary blob: varint exceeds 64 bits".to_string(),
            ));
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
}

/// Compression dictionary for string optimization.
///
/// Ids are assigned append-only, so a blob compressed against an earlier state
/// of the dictionary still decodes correctly after more words are learned.
#[derive(Debug, Default)]
pub struct CompressionDictionary {
    /// Word to ID mapping
    word_to_id: HashMap<String, u32>,
    /// ID to word mapping
    id_to_word: Vec<String>,
}

impl CompressionDictionary {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        }
    }

    /// Number of words currently known.
    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_word.is_empty()
    }

    /// Learn the most frequent words in `strings`.
    pub fn build_from_strings(&mut self, strings: &[String]) -> Result<()> {
        let mut word_counts: HashMap<&str, u32> = HashMap::new();
        for s in strings {
            for word in s.split_whitespace() {
                if word.len() <= MAX_DICTIONARY_WORD_LEN {
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        // Sort by frequency, then by word for a deterministic dictionary.
        word_freq.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

        for (word, _) in word_freq.into_iter().take(1000) {
            self.add_word(word.to_string());
        }

        Ok(())
    }

    /// Add a word, returning its (stable) id.
    pub fn add_word(&mut self, word: String) -> u32 {
        if let Some(&id) = self.word_to_id.get(&word) {
            return id;
        }
        let id = self.id_to_word.len() as u32;
        self.word_to_id.insert(word.clone(), id);
        self.id_to_word.push(word);
        id
    }

    /// Look up a word's id.
    pub fn id_of(&self, word: &str) -> Option<u32> {
        self.word_to_id.get(word).copied()
    }

    /// Compress `data` losslessly against this dictionary.
    pub fn compress(&self, data: &str) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(data.len() / 2 + 8);
        out.push(DICT_CODEC_VERSION);
        put_varint(&mut out, self.id_to_word.len() as u64);

        let bytes = data.as_bytes();
        let mut literal_start = 0usize;
        let mut cursor = 0usize;

        while cursor < bytes.len() {
            if bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
                continue;
            }
            // Maximal run of non-whitespace bytes: a "word".
            let word_start = cursor;
            while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() {
                cursor += 1;
            }
            let word = &data[word_start..cursor];
            if let Some(id) = self.word_to_id.get(word).copied() {
                let pending = &bytes[literal_start..word_start];
                if pending == b" " {
                    out.push(OP_SPACE_REFERENCE);
                } else {
                    if !pending.is_empty() {
                        emit_literal(&mut out, pending);
                    }
                    out.push(OP_REFERENCE);
                }
                put_varint(&mut out, id as u64);
                literal_start = cursor;
            }
        }

        if literal_start < bytes.len() {
            emit_literal(&mut out, &bytes[literal_start..]);
        }
        Ok(out)
    }

    /// Inverse of [`CompressionDictionary::compress`].
    pub fn decompress(&self, data: &[u8]) -> Result<String> {
        if data.is_empty() {
            return Ok(String::new());
        }
        let mut offset = 0usize;
        let version = data[offset];
        offset += 1;
        if version != DICT_CODEC_VERSION {
            return Err(Error::InvalidOperation(format!(
                "Unsupported dictionary blob version {}",
                version
            )));
        }
        let dict_len_at_compress = take_varint(data, &mut offset)? as usize;
        if dict_len_at_compress > self.id_to_word.len() {
            // Ids are append-only, so a shorter dictionary now than at compress
            // time means this blob belongs to a different dictionary.
            return Err(Error::InvalidOperation(format!(
                "Dictionary mismatch: blob was written against {} entries, this dictionary has {}",
                dict_len_at_compress,
                self.id_to_word.len()
            )));
        }

        let mut out: Vec<u8> = Vec::with_capacity(data.len() * 2);
        while offset < data.len() {
            let op = data[offset];
            offset += 1;
            match op {
                OP_LITERAL => {
                    let len = take_varint(data, &mut offset)? as usize;
                    if offset + len > data.len() {
                        return Err(Error::InvalidOperation(
                            "Truncated dictionary blob: literal run".to_string(),
                        ));
                    }
                    out.extend_from_slice(&data[offset..offset + len]);
                    offset += len;
                }
                OP_REFERENCE | OP_SPACE_REFERENCE => {
                    if op == OP_SPACE_REFERENCE {
                        out.push(b' ');
                    }
                    let id = take_varint(data, &mut offset)? as usize;
                    if id >= dict_len_at_compress {
                        return Err(Error::InvalidOperation(format!(
                            "Corrupt dictionary blob: reference {} is outside the {} entries it was written against",
                            id, dict_len_at_compress
                        )));
                    }
                    let word = self.id_to_word.get(id).ok_or_else(|| {
                        Error::InvalidOperation(format!("Dictionary entry {} is missing", id))
                    })?;
                    out.extend_from_slice(word.as_bytes());
                }
                other => {
                    return Err(Error::InvalidOperation(format!(
                        "Corrupt dictionary blob: unknown op {}",
                        other
                    )))
                }
            }
        }

        String::from_utf8(out)
            .map_err(|e| Error::InvalidOperation(format!("UTF-8 decode error: {}", e)))
    }
}

fn emit_literal(out: &mut Vec<u8>, bytes: &[u8]) {
    out.push(OP_LITERAL);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_whitespace_exactly() {
        let mut dict = CompressionDictionary::new();
        let corpus = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];
        dict.build_from_strings(&corpus).expect("build");

        for sample in [
            "hello world",
            "hello\tworld",
            "hello\n\nworld",
            "hello   world   ",
            "   leading and trailing   ",
            "unknown tokens only",
            "",
            "日本語 world 日本語",
        ] {
            let compressed = dict.compress(sample).expect("compress");
            assert_eq!(
                dict.decompress(&compressed).expect("decompress"),
                sample,
                "lossy round-trip for {:?}",
                sample
            );
        }
    }

    #[test]
    fn long_words_do_not_collide_with_the_marker_byte() {
        // A 255-byte word used to emit a literal-length byte of 0xFF, which was
        // also the dictionary marker.
        let mut dict = CompressionDictionary::new();
        dict.add_word("known".to_string());
        let long_word = "x".repeat(255);
        let sample = format!("known {} known", long_word);
        let compressed = dict.compress(&sample).expect("compress");
        assert_eq!(dict.decompress(&compressed).expect("decompress"), sample);

        let very_long = "y".repeat(70_000);
        let compressed = dict.compress(&very_long).expect("compress");
        assert_eq!(dict.decompress(&compressed).expect("decompress"), very_long);
    }

    #[test]
    fn out_of_range_reference_is_an_error() {
        let mut dict = CompressionDictionary::new();
        dict.add_word("alpha".to_string());
        let mut blob = vec![DICT_CODEC_VERSION];
        put_varint(&mut blob, 1); // dict length at compress time
        blob.push(OP_REFERENCE);
        put_varint(&mut blob, 9); // out of range
        assert!(dict.decompress(&blob).is_err());
    }

    #[test]
    fn shorter_dictionary_is_rejected() {
        let dict = CompressionDictionary::new();
        let mut blob = vec![DICT_CODEC_VERSION];
        put_varint(&mut blob, 5); // written against 5 entries
        assert!(dict.decompress(&blob).is_err());
    }

    #[test]
    fn dictionary_actually_shrinks_repetitive_text() {
        let mut dict = CompressionDictionary::new();
        let corpus: Vec<String> = (0..100)
            .map(|_| "the quick brown fox jumps over the lazy dog".to_string())
            .collect();
        dict.build_from_strings(&corpus).expect("build");
        let sample = "the quick brown fox jumps over the lazy dog";
        let compressed = dict.compress(sample).expect("compress");
        assert!(
            compressed.len() < sample.len(),
            "dictionary compression did not shrink: {} -> {}",
            sample.len(),
            compressed.len()
        );
    }
}

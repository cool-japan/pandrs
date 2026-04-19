//! Cell-level helpers for the Pure Rust xlsx reader/writer.
//!
//! Responsibilities in this file:
//! - Convert between A1-style cell references (`"A1"`, `"AB12"`) and
//!   zero-indexed `(row, col)` coordinates.
//! - Manage a write-side shared-strings table (interned text values) and a
//!   read-side shared-strings lookup (flat `Vec<String>`).
//! - Escape text for inclusion in XML character data.
//!
//! None of these helpers perform any I/O. They are the purely-computational
//! core of the xlsx codec.

use std::collections::HashMap;

use crate::error::{Error, Result};

use super::error::{invalid, io_err};

/// Logical cell data type after a value has been read from a worksheet.
///
/// The Xlsx on-disk encoding distinguishes between inline strings, shared
/// strings, numeric literals, booleans, errors, and the (somewhat idiosyncratic)
/// date-as-serial-number convention. We normalise all of those into this enum
/// before feeding them back to pandrs-level type inference.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(super) enum XlsxCellValue {
    /// Empty cell.
    Empty,
    /// A text value — either inline or from the shared-strings table.
    String(String),
    /// A numeric value (floating point, since xlsx stores all numerics as IEEE-754).
    Number(f64),
    /// A boolean value (xlsx encodes as "0"/"1" with `t="b"`).
    Boolean(bool),
    /// An error value (e.g. `#DIV/0!`) represented as its textual form.
    Error(String),
}

impl XlsxCellValue {
    /// Render this value into its pandrs-side string representation so that
    /// downstream type inference can operate on it uniformly.
    pub(super) fn to_display_string(&self) -> String {
        match self {
            XlsxCellValue::Empty => String::new(),
            XlsxCellValue::String(s) => s.clone(),
            XlsxCellValue::Number(n) => format_number(*n),
            XlsxCellValue::Boolean(b) => {
                if *b {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            XlsxCellValue::Error(s) => s.clone(),
        }
    }
}

/// Render an `f64` for cell output, preferring compact integer form when the
/// value is a whole number.
pub(super) fn format_number(n: f64) -> String {
    if n.is_nan() {
        return "NaN".to_string();
    }
    if n.is_infinite() {
        return if n.is_sign_negative() {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    if n.fract() == 0.0 && n.abs() < 1e16 {
        format!("{}", n as i64)
    } else {
        // Use the default f64 formatter which gives a round-trippable form.
        format!("{n}")
    }
}

/// Convert a zero-indexed column number to its A1 column letter sequence
/// (e.g. `0 -> "A"`, `25 -> "Z"`, `26 -> "AA"`).
pub(super) fn col_letters(col: usize) -> String {
    let mut out = Vec::new();
    let mut n = col as i64;
    // Classic base-26 but with 1-indexed digits (A..Z = 1..26).
    loop {
        let rem = (n % 26) as u8;
        out.push(b'A' + rem);
        n = n / 26 - 1;
        if n < 0 {
            break;
        }
    }
    out.reverse();
    // SAFETY: all bytes pushed are ASCII letters.
    String::from_utf8(out).unwrap_or_else(|_| "A".to_string())
}

/// Encode a zero-indexed `(row, col)` as an A1 reference (`row` is 0-based,
/// output row number is 1-based).
pub(super) fn encode_ref(row: usize, col: usize) -> String {
    let mut s = col_letters(col);
    s.push_str(&(row + 1).to_string());
    s
}

/// Parse an A1 reference like `"AB12"` into `(row_zero_indexed, col_zero_indexed)`.
pub(super) fn parse_ref(r: &str) -> Result<(usize, usize)> {
    let bytes = r.as_bytes();
    let mut i = 0;
    let mut col: usize = 0;
    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        let c = bytes[i].to_ascii_uppercase();
        col = col * 26 + ((c - b'A' + 1) as usize);
        i += 1;
    }
    if i == 0 {
        return Err(invalid(format!("xlsx: invalid cell ref '{r}': no letters")));
    }
    if col == 0 {
        return Err(invalid(format!(
            "xlsx: invalid cell ref '{r}': zero column"
        )));
    }
    let col_zero = col - 1;
    let row_str = &r[i..];
    if row_str.is_empty() {
        return Err(invalid(format!("xlsx: invalid cell ref '{r}': no row")));
    }
    let row: usize = row_str
        .parse()
        .map_err(|_| invalid(format!("xlsx: invalid row in cell ref '{r}'")))?;
    if row == 0 {
        return Err(invalid(format!("xlsx: invalid cell ref '{r}': row 0")));
    }
    Ok((row - 1, col_zero))
}

/// A write-side shared-strings accumulator. Insertion order is preserved so
/// the resulting `xl/sharedStrings.xml` stays deterministic.
#[derive(Debug, Default)]
pub(super) struct SharedStringsBuilder {
    order: Vec<String>,
    index: HashMap<String, u32>,
}

impl SharedStringsBuilder {
    pub(super) fn new() -> Self {
        Self {
            order: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Intern a string and return its numeric index.
    pub(super) fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        let idx = self.order.len() as u32;
        self.order.push(s.to_string());
        self.index.insert(s.to_string(), idx);
        idx
    }

    /// Total number of entries.
    pub(super) fn len(&self) -> usize {
        self.order.len()
    }

    /// Consume and yield strings in insertion order.
    pub(super) fn into_ordered(self) -> Vec<String> {
        self.order
    }
}

/// Escape a text value so it is safe to embed inside an XML element as
/// character data. Handles `&`, `<`, `>`, and optionally quotes since we use
/// the same helper for attribute values too.
pub(super) fn xml_escape(s: &str) -> String {
    // Fast path: nothing to escape.
    if !s
        .bytes()
        .any(|b| b == b'&' || b == b'<' || b == b'>' || b == b'"' || b == b'\'')
    {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Trim a string to the xlsx-imposed 31-character sheet-name limit, raising a
/// descriptive error if required.
pub(super) fn validate_sheet_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(invalid("xlsx: sheet name must not be empty"));
    }
    // Excel's actual limit is 31 characters, counted as UTF-16 code units in
    // practice. We approximate with char count which is close enough for our
    // purposes.
    if name.chars().count() > 31 {
        return Err(invalid(format!(
            "xlsx: sheet name '{name}' exceeds 31 characters"
        )));
    }
    // Characters explicitly disallowed by Excel in sheet names.
    for ch in name.chars() {
        if matches!(ch, ':' | '\\' | '/' | '?' | '*' | '[' | ']') {
            return Err(invalid(format!(
                "xlsx: sheet name '{name}' contains invalid character '{ch}'"
            )));
        }
    }
    Ok(())
}

/// Unify the many ways a read operation can fail into a single `Error` that
/// still preserves some context.
#[inline]
pub(super) fn fail(msg: impl Into<String>) -> Error {
    io_err(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_letters_works_for_single_and_multi_letter_columns() {
        assert_eq!(col_letters(0), "A");
        assert_eq!(col_letters(25), "Z");
        assert_eq!(col_letters(26), "AA");
        assert_eq!(col_letters(27), "AB");
        assert_eq!(col_letters(701), "ZZ");
        assert_eq!(col_letters(702), "AAA");
    }

    #[test]
    fn parse_ref_roundtrips_encode_ref() {
        for (r, c) in [(0_usize, 0_usize), (5, 25), (100, 26), (1023, 702)] {
            let enc = encode_ref(r, c);
            let (pr, pc) = parse_ref(&enc).expect("valid ref");
            assert_eq!((pr, pc), (r, c), "roundtrip {r},{c} via {enc}");
        }
    }

    #[test]
    fn shared_strings_intern_dedups() {
        let mut b = SharedStringsBuilder::new();
        assert_eq!(b.intern("a"), 0);
        assert_eq!(b.intern("b"), 1);
        assert_eq!(b.intern("a"), 0);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        assert_eq!(
            xml_escape("a&b<c>d\"e'f"),
            "a&amp;b&lt;c&gt;d&quot;e&apos;f"
        );
        assert_eq!(xml_escape("plain"), "plain");
    }

    #[test]
    fn validate_sheet_name_rejects_bad_chars() {
        assert!(validate_sheet_name("").is_err());
        assert!(validate_sheet_name("a/b").is_err());
        assert!(validate_sheet_name("ok sheet").is_ok());
    }
}

//! Cell-level helpers for the Pure Rust xlsx reader/writer.
//!
//! Responsibilities in this file:
//! - Convert between A1-style cell references (`"A1"`, `"AB12"`) and
//!   zero-indexed `(row, col)` coordinates, with Excel's own hard dimension
//!   limits enforced so a hostile reference can never drive an oversized
//!   allocation.
//! - Manage a write-side shared-strings table (interned text values) and a
//!   read-side shared-strings lookup (flat `Vec<String>`).
//! - Format numeric values for cell output in a way that preserves
//!   Int64-vs-Float64 shape across a write/read round trip.
//! - Escape (and, where XML 1.0 forbids it outright, strip) text for
//!   inclusion in XML character data / attribute values.
//!
//! None of these helpers perform any I/O. They are the purely-computational
//! core of the xlsx codec.

use std::collections::{HashMap, HashSet};

use crate::error::{Error, Result};

use super::error::{invalid, io_err};

/// Excel's own hard ceiling on worksheet dimensions (SpreadsheetML / OOXML):
/// 1,048,576 rows and 16,384 columns (column `XFD`). A cell reference
/// claiming a coordinate outside this range cannot have come from a real
/// Excel-authored file, so we reject it outright rather than accepting an
/// attacker-chosen coordinate that could later drive an oversized
/// allocation (see the density guard in `reader::parse_worksheet`).
pub(super) const MAX_XLSX_ROWS: usize = 1_048_576;
pub(super) const MAX_XLSX_COLS: usize = 16_384;

/// Logical cell data type after a value has been read from a worksheet.
///
/// The Xlsx on-disk encoding distinguishes between inline strings, shared
/// strings, numeric literals, booleans, errors, and the (somewhat idiosyncratic)
/// date-as-serial-number convention (handled at a higher level once a cell's
/// style tags it as a date/time format). We normalise everything else into
/// this enum before feeding it back to pandrs-level type inference.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum XlsxCellValue {
    /// Empty cell.
    Empty,
    /// A text value — either inline or from the shared-strings table.
    String(String),
    /// A numeric value (floating point, since xlsx stores all numerics as
    /// IEEE-754). `raw` is the exact text that appeared inside `<v>` (e.g.
    /// `"3"` vs `"3.0"`). We preserve it verbatim rather than reformatting
    /// `value` because pandrs' own writer encodes int-vs-float shape in
    /// that text (see `format_int` / `format_float` below): reformatting a
    /// parsed `f64` back through a "compact if whole" heuristic would erase
    /// exactly the distinction that lets an all-whole-number Float64 column
    /// round-trip as Float64 instead of being reinferred as Int64.
    Number { value: f64, raw: String },
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
            XlsxCellValue::Number { raw, .. } => raw.clone(),
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

    /// The parsed numeric value, if this cell is a `Number`. Used by the
    /// date/time conversion path, which needs the actual `f64` serial (not
    /// its text form) to do epoch arithmetic.
    pub(super) fn as_number(&self) -> Option<f64> {
        match self {
            XlsxCellValue::Number { value, .. } => Some(*value),
            _ => None,
        }
    }
}

/// Render an `i64` for integer-cell XML output: compact, no decimal point.
/// An Int64 column round-trips as Int64 as long as none of its cell text
/// contains a `.`, which a plain integer never does.
pub(super) fn format_int(n: i64) -> String {
    n.to_string()
}

/// Render a *finite* `f64` for float-cell XML output. Always contains a
/// decimal point (or an exponent marker), even for whole numbers — e.g.
/// `3.0` renders as `"3.0"`, never the bare `"3"` that Rust's default `f64`
/// `Display` would produce. Without this, writing an all-whole-number
/// Float64 column would emit cell text indistinguishable from an Int64
/// column, and the reader's per-column type inference (which tries
/// `i64::parse` before `f64::parse`) would reinfer it as Int64 on read-back.
///
/// Callers must check `is_finite()` first: NaN/±Infinity cannot be
/// represented as an xlsx numeric `<v>` value at all (Excel refuses to open
/// a file containing one), so those must be routed to an error/empty cell
/// instead — see `xlsx::writer::append_cell_from_column`.
pub(super) fn format_float(n: f64) -> String {
    debug_assert!(n.is_finite(), "format_float called with a non-finite value");
    let s = format!("{n}");
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
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
///
/// Column accumulation uses checked arithmetic and is bounds-checked against
/// [`MAX_XLSX_COLS`] on every letter, so a pathologically long letter run
/// (which would otherwise overflow `usize`) is rejected long before it can
/// overflow — the bounds check trips at the 4th letter (26^4 already exceeds
/// the column cap), far short of where `usize` multiplication could wrap.
/// The row number is bounds-checked against [`MAX_XLSX_ROWS`] similarly.
pub(super) fn parse_ref(r: &str) -> Result<(usize, usize)> {
    let bytes = r.as_bytes();
    let mut i = 0;
    let mut col: usize = 0;
    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        let c = bytes[i].to_ascii_uppercase();
        let digit = (c - b'A' + 1) as usize;
        col = col
            .checked_mul(26)
            .and_then(|v| v.checked_add(digit))
            .filter(|&v| v <= MAX_XLSX_COLS)
            .ok_or_else(|| {
                invalid(format!(
                    "xlsx: invalid cell ref '{r}': column out of range (max {MAX_XLSX_COLS})"
                ))
            })?;
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
    if row > MAX_XLSX_ROWS {
        return Err(invalid(format!(
            "xlsx: invalid cell ref '{r}': row out of range (max {MAX_XLSX_ROWS})"
        )));
    }
    Ok((row - 1, col_zero))
}

/// A write-side shared-strings accumulator. Insertion order is preserved so
/// the resulting `xl/sharedStrings.xml` stays deterministic.
#[derive(Debug, Default)]
pub(super) struct SharedStringsBuilder {
    order: Vec<String>,
    index: HashMap<String, u32>,
    /// Total number of `intern` calls, including repeats. This is the
    /// `count` attribute the OOXML spec defines for `sharedStrings.xml`
    /// (the number of cells referencing the table), which is distinct from
    /// `uniqueCount` (`order.len()`, the number of distinct `<si>` entries).
    total_refs: u32,
}

impl SharedStringsBuilder {
    pub(super) fn new() -> Self {
        Self {
            order: Vec::new(),
            index: HashMap::new(),
            total_refs: 0,
        }
    }

    /// Intern a string and return its numeric index.
    pub(super) fn intern(&mut self, s: &str) -> u32 {
        self.total_refs = self.total_refs.saturating_add(1);
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        let idx = self.order.len() as u32;
        self.order.push(s.to_string());
        self.index.insert(s.to_string(), idx);
        idx
    }

    /// Total number of *unique* entries.
    #[allow(dead_code)] // reserved for future use
    pub(super) fn len(&self) -> usize {
        self.order.len()
    }

    /// Consume, returning the unique strings in insertion order (`uniqueCount`)
    /// together with the total number of references made via `intern`
    /// (`count`).
    pub(super) fn into_ordered(self) -> (Vec<String>, u32) {
        (self.order, self.total_refs)
    }
}

/// XML 1.0 (§2.2) permits only tab/LF/CR (`#x9 | #xA | #xD`) among the
/// control characters below `#x20`; every other C0 control character cannot
/// appear in a conforming document — not even via a numeric character
/// reference, since `&#x1;` etc. are themselves illegal. We strip these
/// rather than escape them, since there is no legal escaped form.
fn is_xml_illegal_control(c: char) -> bool {
    matches!(c, '\u{0}'..='\u{8}' | '\u{B}' | '\u{C}' | '\u{E}'..='\u{1F}')
}

/// True for a character that needs neither XML-escaping nor stripping.
fn is_plain_xml_char(c: char) -> bool {
    !matches!(c, '&' | '<' | '>' | '"' | '\'') && !is_xml_illegal_control(c)
}

/// Escape a text value so it is safe to embed inside an XML element as
/// character data (or an attribute value, since we use the same helper for
/// both). Handles `&`, `<`, `>`, `"`, `'`, and strips any character XML 1.0
/// forbids outright (see [`is_xml_illegal_control`]) so we never emit a
/// non-well-formed file even when the source data contains raw control
/// bytes.
pub(super) fn xml_escape(s: &str) -> String {
    // Fast path: nothing to escape or strip.
    if s.chars().all(is_plain_xml_char) {
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
            c if is_xml_illegal_control(c) => {
                // Dropped: cannot be represented in XML 1.0 even escaped.
            }
            c => out.push(c),
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
        // XML-1.0-illegal control characters are stripped, not escaped, at
        // serialization time (`xml_escape`, used by `workbook_xml` to emit
        // the `name="..."` attribute). If we accepted such a name here,
        // `validate_unique_sheet_names` would see two distinct raw names
        // (e.g. "Sheet\u{1}1" vs "Sheet1") pass its uniqueness check, but
        // both would serialize to the *same* stripped `name="Sheet1"` —
        // silently producing the duplicate-sheet-name file that validation
        // exists to prevent. Reject up front instead of relying on a
        // downstream stripper that changes the value.
        if is_xml_illegal_control(ch) {
            return Err(invalid(format!(
                "xlsx: sheet name '{name}' contains a control character that cannot appear in XML"
            )));
        }
    }
    Ok(())
}

/// Reject a sheet list containing two names Excel would treat as the same
/// sheet. Excel sheet names are case-insensitive for uniqueness purposes
/// (`"Sheet1"` and `"sheet1"` collide, and Excel itself refuses to open a
/// file that violates this).
pub(super) fn validate_unique_sheet_names<'a, I>(names: I) -> Result<()>
where
    I: IntoIterator<Item = &'a str>,
{
    let mut seen: HashSet<String> = HashSet::new();
    for name in names {
        let key = name.to_lowercase();
        if !seen.insert(key) {
            return Err(invalid(format!(
                "xlsx: duplicate sheet name '{name}' (Excel sheet names are case-insensitive)"
            )));
        }
    }
    Ok(())
}

/// Unify the many ways a read operation can fail into a single `Error` that
/// still preserves some context.
#[inline]
#[allow(dead_code)] // reserved for future use
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
    fn parse_ref_accepts_excels_own_maximum() {
        // "XFD1048576" is Excel's actual bottom-right-most cell.
        let (r, c) = parse_ref("XFD1048576").expect("Excel's own max cell ref must be accepted");
        assert_eq!(r, MAX_XLSX_ROWS - 1);
        assert_eq!(c, MAX_XLSX_COLS - 1);
    }

    #[test]
    fn parse_ref_rejects_column_beyond_excel_max() {
        // One column past "XFD" (16384).
        assert!(parse_ref("XFE1").is_err());
    }

    #[test]
    fn parse_ref_rejects_row_beyond_excel_max() {
        assert!(parse_ref("A1048577").is_err());
    }

    #[test]
    fn parse_ref_rejects_absurdly_long_column_without_overflow_panic() {
        // A pathological, hand-crafted reference designed to overflow naive
        // `col * 26` arithmetic. Must return an `Err`, never panic.
        let huge = format!("{}1", "A".repeat(200));
        assert!(parse_ref(&huge).is_err());
    }

    #[test]
    fn shared_strings_intern_dedups_and_tracks_total_refs() {
        let mut b = SharedStringsBuilder::new();
        assert_eq!(b.intern("a"), 0);
        assert_eq!(b.intern("b"), 1);
        assert_eq!(b.intern("a"), 0);
        assert_eq!(b.len(), 2);
        let (unique, total) = b.into_ordered();
        assert_eq!(unique, vec!["a".to_string(), "b".to_string()]);
        // 3 intern() calls total, only 2 unique strings.
        assert_eq!(total, 3);
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
    fn xml_escape_strips_illegal_control_chars() {
        let s = "a\u{0}b\u{1}c\u{B}d\u{1F}e";
        assert_eq!(xml_escape(s), "abcde");
        // Tab/LF/CR are legal and must survive.
        assert_eq!(xml_escape("a\tb\nc\rd"), "a\tb\nc\rd");
    }

    #[test]
    fn validate_sheet_name_rejects_bad_chars() {
        assert!(validate_sheet_name("").is_err());
        assert!(validate_sheet_name("a/b").is_err());
        assert!(validate_sheet_name("ok sheet").is_ok());
    }

    #[test]
    fn validate_sheet_name_rejects_control_chars() {
        // Must be rejected outright, not silently accepted and later
        // stripped by `xml_escape` at serialization time — see the
        // doc-comment on `validate_sheet_name` for why that would be
        // unsafe (it would let two distinct raw names collide into one
        // serialized name, defeating `validate_unique_sheet_names`).
        assert!(validate_sheet_name("Sheet\u{1}1").is_err());
        assert!(validate_sheet_name("Sheet\u{0}1").is_err());
        // Tab is not an XML-illegal control character, so it stays allowed.
        assert!(validate_sheet_name("Sheet\t1").is_ok());
    }

    #[test]
    fn validate_unique_sheet_names_rejects_case_insensitive_duplicates() {
        assert!(validate_unique_sheet_names(["Sheet1", "Sheet2"]).is_ok());
        assert!(validate_unique_sheet_names(["Sheet1", "sheet1"]).is_err());
        assert!(validate_unique_sheet_names(["Data", "DATA"]).is_err());
    }

    #[test]
    fn format_int_has_no_decimal_point() {
        assert_eq!(format_int(3), "3");
        assert_eq!(format_int(-42), "-42");
        assert_eq!(format_int(0), "0");
    }

    #[test]
    fn format_float_always_has_a_decimal_or_exponent() {
        assert_eq!(format_float(3.0), "3.0");
        assert_eq!(format_float(3.5), "3.5");
        assert_eq!(format_float(-2.0), "-2.0");
        assert_eq!(format_float(0.0), "0.0");
        // Whatever Rust's Display chooses, the result must contain a '.'
        // (or an exponent marker) so it never round-trips as an integer.
        for v in [1e15_f64, 1e300_f64, 1.0 / 3.0] {
            let s = format_float(v);
            assert!(
                s.contains('.') || s.contains('e') || s.contains('E'),
                "format_float({v}) = {s:?} must look non-integer"
            );
        }
    }
}

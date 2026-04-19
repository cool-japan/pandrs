//! Pure Rust xlsx reader built on top of `oxiarc-archive` + `quick-xml`.
//!
//! Semantics mirror the calamine-based implementation closely enough to keep
//! the existing pandrs public API contract intact:
//!
//! - A workbook is made of one or more sheets, each identified by name.
//! - A single sheet produces a rectangular array of cells.
//! - Cells may be numeric, shared-string, inline-string, boolean, error, or
//!   empty. We normalise them to strings for pandrs-level type inference.
//!
//! We do *not* attempt to understand:
//!   - date/time serial numbers (they come through as numbers),
//!   - cell formulas (the cell's calculated value is read; formulas are ignored),
//!   - formatting / styles.
//!
//! These omissions match the behavior of the previous `calamine`-backed
//! reader which also threw away formula text in the returned DataFrame and
//! only exposed formatting through placeholder hooks.

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;

use oxiarc_archive::zip::ZipReader;
use quick_xml::escape::unescape;
use quick_xml::events::attributes::Attribute;
use quick_xml::events::{BytesText, Event};
use quick_xml::Reader;

use crate::error::Result;

use super::cell::{parse_ref, XlsxCellValue};
use super::error::{invalid, io_err, std_io, xml_err, zip_err};

/// Collection of sheet metadata extracted from `xl/workbook.xml`.
#[derive(Debug, Clone)]
pub(super) struct WorkbookManifest {
    /// Sheet names in declaration order.
    pub(super) sheet_names: Vec<String>,
    /// For each entry in `sheet_names`, the workbook-relationship id.
    pub(super) sheet_rids: Vec<String>,
}

/// Load the entire parsed workbook into memory. This is deliberately eager —
/// OptimizedDataFrame construction requires random access to every cell of
/// every selected sheet anyway.
pub(super) struct LoadedWorkbook {
    pub(super) sheet_names: Vec<String>,
    pub(super) sheets: Vec<LoadedSheet>,
    /// The flat shared-strings table, if any.
    pub(super) shared_strings: Vec<String>,
}

/// One fully materialised sheet.
#[derive(Debug, Clone)]
pub(super) struct LoadedSheet {
    pub(super) name: String,
    /// `rows[r][c]` is the value at row `r`, column `c`. Rows/cols are packed
    /// to the actual data extent; empty cells are represented by
    /// `XlsxCellValue::Empty`.
    pub(super) rows: Vec<Vec<XlsxCellValue>>,
    /// Column-width of the rectangular region (may be larger than the widest
    /// row's length; each row is right-padded with empties if needed).
    pub(super) cols: usize,
}

/// Represents the bytes of a worksheet part keyed by filename (e.g.
/// `"xl/worksheets/sheet1.xml"`).
struct RawSheet {
    /// Path in the archive.
    path: String,
    /// The rId the workbook uses to reference this sheet.
    rid: String,
    /// Bytes of the sheet XML.
    bytes: Vec<u8>,
}

/// Open an .xlsx file and return a fully parsed workbook. We use buffered I/O
/// on the archive container so large files don't blow the call stack.
pub(super) fn load_workbook<P: AsRef<Path>>(path: P) -> Result<LoadedWorkbook> {
    let file = File::open(path.as_ref()).map_err(std_io)?;
    let buf = BufReader::new(file);
    load_workbook_from_reader(buf)
}

/// Same as [`load_workbook`] but takes an arbitrary `Read + Seek` source so
/// that tests can feed in an in-memory buffer.
pub(super) fn load_workbook_from_reader<R: Read + Seek>(reader: R) -> Result<LoadedWorkbook> {
    let mut zr = ZipReader::new(reader).map_err(zip_err)?;

    // Read all parts we need up front. We copy bytes out of the zip reader
    // because the reader itself holds a mutable borrow while extracting.
    let entries: Vec<_> = zr.entries().to_vec();

    let mut workbook_bytes: Option<Vec<u8>> = None;
    let mut workbook_rels_bytes: Option<Vec<u8>> = None;
    let mut shared_bytes: Option<Vec<u8>> = None;
    let mut raw_sheets: Vec<RawSheet> = Vec::new();

    for entry in &entries {
        let name = entry.name.as_str();
        if name == "xl/workbook.xml" {
            workbook_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name == "xl/_rels/workbook.xml.rels" {
            workbook_rels_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name == "xl/sharedStrings.xml" {
            shared_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name.starts_with("xl/worksheets/") && name.ends_with(".xml") {
            let bytes = zr.extract(entry).map_err(zip_err)?;
            raw_sheets.push(RawSheet {
                path: name.to_string(),
                rid: String::new(), // filled in after parsing workbook rels
                bytes,
            });
        }
    }

    let workbook_bytes =
        workbook_bytes.ok_or_else(|| io_err("xlsx: workbook.xml missing from archive"))?;

    // Parse workbook.xml for sheet list (names + rId).
    let manifest = parse_workbook_manifest(&workbook_bytes)?;

    // Map rId → sheet target using workbook.xml.rels, and align raw_sheets.
    if let Some(rels) = workbook_rels_bytes.as_deref() {
        let rid_to_target = parse_rels(rels)?;
        for raw in &mut raw_sheets {
            // Normalise: `raw.path` is like `xl/worksheets/sheet1.xml`,
            // relationships tend to encode target as `worksheets/sheet1.xml`.
            for (rid, target) in &rid_to_target {
                let normalized = if target.starts_with('/') {
                    target[1..].to_string()
                } else {
                    format!("xl/{target}")
                };
                if normalized == raw.path || target == &raw.path {
                    raw.rid = rid.clone();
                    break;
                }
            }
        }
    }

    // Parse shared strings (if present).
    let shared_strings = match shared_bytes.as_deref() {
        Some(b) => parse_shared_strings(b)?,
        None => Vec::new(),
    };

    // Emit sheets in the order of the workbook manifest.
    let mut sheets: Vec<LoadedSheet> = Vec::with_capacity(manifest.sheet_names.len());
    for (i, (name, rid)) in manifest
        .sheet_names
        .iter()
        .zip(manifest.sheet_rids.iter())
        .enumerate()
    {
        // Prefer rId matching; fall back to positional sheet{N}.xml.
        let fallback_path = format!("xl/worksheets/sheet{}.xml", i + 1);
        let raw = raw_sheets
            .iter()
            .find(|r| !r.rid.is_empty() && r.rid == *rid)
            .or_else(|| raw_sheets.iter().find(|r| r.path == fallback_path))
            .ok_or_else(|| io_err(format!("xlsx: sheet '{name}' not found in archive")))?;

        let (rows, cols) = parse_worksheet(&raw.bytes, &shared_strings)?;
        sheets.push(LoadedSheet {
            name: name.clone(),
            rows,
            cols,
        });
    }

    Ok(LoadedWorkbook {
        sheet_names: manifest.sheet_names,
        sheets,
        shared_strings,
    })
}

/// Parse `xl/workbook.xml` to recover the sheet list (names & rId values).
fn parse_workbook_manifest(bytes: &[u8]) -> Result<WorkbookManifest> {
    let mut names = Vec::new();
    let mut rids = Vec::new();
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"sheet" => {
                let mut name = String::new();
                let mut rid = String::new();
                for a in e.attributes().with_checks(false).flatten() {
                    let key = a.key.as_ref();
                    if key == b"name" {
                        name = attr_to_string(&a)?;
                    } else if key == b"r:id" || key.ends_with(b":id") || key == b"id" {
                        rid = attr_to_string(&a)?;
                    }
                }
                if !name.is_empty() {
                    names.push(name);
                    rids.push(rid);
                }
            }
            _ => {}
        }
        buf.clear();
    }
    if names.is_empty() {
        return Err(invalid("xlsx: workbook contains no sheets"));
    }
    Ok(WorkbookManifest {
        sheet_names: names,
        sheet_rids: rids,
    })
}

/// Parse a relationships file (`.rels`) into a list of (rId, target) tuples.
fn parse_rels(bytes: &[u8]) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"Relationship" => {
                let mut id = String::new();
                let mut target = String::new();
                for a in e.attributes().with_checks(false).flatten() {
                    let key = a.key.as_ref();
                    if key == b"Id" {
                        id = attr_to_string(&a)?;
                    } else if key == b"Target" {
                        target = attr_to_string(&a)?;
                    }
                }
                if !id.is_empty() && !target.is_empty() {
                    out.push((id, target));
                }
            }
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

/// Parse `xl/sharedStrings.xml` into a flat Vec where index == `<si>` position.
///
/// Each `<si>` may contain either `<t>` (plain text) or `<r><t>...</t></r>`
/// runs for rich text. We concatenate the runs.
fn parse_shared_strings(bytes: &[u8]) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    // State machine: we're inside an `<si>` after we see <si>, and we push
    // whenever we hit </si>. Within an <si>, every <t>...</t> contributes to
    // the accumulator (rich text merges the runs).
    let mut in_si = false;
    let mut in_t = false;
    let mut accum = String::new();

    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Start(e) => match e.name().as_ref() {
                b"si" => {
                    in_si = true;
                    accum.clear();
                }
                b"t" => {
                    if in_si {
                        in_t = true;
                    }
                }
                _ => {}
            },
            Event::End(e) => match e.name().as_ref() {
                b"si" => {
                    if in_si {
                        out.push(std::mem::take(&mut accum));
                        in_si = false;
                    }
                }
                b"t" => {
                    in_t = false;
                }
                _ => {}
            },
            Event::Text(t) => {
                if in_si && in_t {
                    let s = decode_and_unescape(&t)?;
                    accum.push_str(&s);
                }
            }
            Event::CData(c) => {
                if in_si && in_t {
                    let s = std::str::from_utf8(c.as_ref())
                        .map_err(|e| io_err(format!("xlsx: invalid UTF-8 in CDATA: {e}")))?;
                    accum.push_str(s);
                }
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(out)
}

/// Parse one `xl/worksheets/sheet{N}.xml` into a row/col matrix of cell values.
///
/// Return: `(rows, max_cols)`.
fn parse_worksheet(bytes: &[u8], shared: &[String]) -> Result<(Vec<Vec<XlsxCellValue>>, usize)> {
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    // Result accumulators.
    let mut cells: Vec<(usize, usize, XlsxCellValue)> = Vec::new();
    let mut max_row = 0usize;
    let mut max_col = 0usize;

    // Per-cell parse state.
    let mut cur_ref: Option<(usize, usize)> = None;
    let mut cur_type: CellType = CellType::Number;
    let mut in_value = false;
    let mut value_text = String::new();
    let mut in_inline_text = false;
    let mut inline_text = String::new();

    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Start(e) => {
                let name = e.name();
                let tag = name.as_ref();
                if tag == b"c" {
                    cur_ref = None;
                    cur_type = CellType::Number;
                    value_text.clear();
                    inline_text.clear();
                    for a in e.attributes().with_checks(false).flatten() {
                        let key = a.key.as_ref();
                        if key == b"r" {
                            let s = attr_to_string(&a)?;
                            cur_ref = Some(parse_ref(&s)?);
                        } else if key == b"t" {
                            cur_type = CellType::from_attr(a.value.as_ref());
                        }
                    }
                } else if tag == b"v" {
                    in_value = true;
                    value_text.clear();
                } else if tag == b"t" {
                    // Inline text — only meaningful if cur_type == InlineStr
                    // or we're reading shared strings elsewhere.
                    in_inline_text = true;
                    inline_text.clear();
                }
            }
            Event::Empty(e) => {
                let name = e.name();
                let tag = name.as_ref();
                if tag == b"c" {
                    // Empty cell — still register its position so the column
                    // layout is preserved.
                    let mut coord: Option<(usize, usize)> = None;
                    for a in e.attributes().with_checks(false).flatten() {
                        let key = a.key.as_ref();
                        if key == b"r" {
                            let s = attr_to_string(&a)?;
                            coord = Some(parse_ref(&s)?);
                        }
                    }
                    if let Some((r, c)) = coord {
                        cells.push((r, c, XlsxCellValue::Empty));
                        max_row = max_row.max(r + 1);
                        max_col = max_col.max(c + 1);
                    }
                }
            }
            Event::End(e) => {
                let name = e.name();
                let tag = name.as_ref();
                if tag == b"v" {
                    in_value = false;
                } else if tag == b"t" {
                    in_inline_text = false;
                } else if tag == b"c" {
                    // Commit the cell.
                    let (r, c) = match cur_ref {
                        Some(v) => v,
                        // No coordinate → skip (shouldn't happen in valid xlsx).
                        None => continue,
                    };
                    let val = finalise_cell(&cur_type, &value_text, &inline_text, shared)?;
                    cells.push((r, c, val));
                    max_row = max_row.max(r + 1);
                    max_col = max_col.max(c + 1);
                    cur_ref = None;
                }
            }
            Event::Text(t) => {
                if in_value {
                    let s = decode_and_unescape(&t)?;
                    value_text.push_str(&s);
                } else if in_inline_text {
                    let s = decode_and_unescape(&t)?;
                    inline_text.push_str(&s);
                }
            }
            Event::CData(c) => {
                let s = std::str::from_utf8(c.as_ref())
                    .map_err(|e| io_err(format!("xlsx: invalid UTF-8 in CDATA: {e}")))?;
                if in_value {
                    value_text.push_str(s);
                } else if in_inline_text {
                    inline_text.push_str(s);
                }
            }
            _ => {}
        }
        buf.clear();
    }

    // Materialise into a dense row-major matrix.
    let mut rows: Vec<Vec<XlsxCellValue>> = vec![Vec::new(); max_row];
    for row in rows.iter_mut() {
        row.resize(max_col, XlsxCellValue::Empty);
    }
    for (r, c, v) in cells {
        if r < rows.len() && c < max_col {
            rows[r][c] = v;
        }
    }
    Ok((rows, max_col))
}

fn finalise_cell(
    kind: &CellType,
    v: &str,
    inline: &str,
    shared: &[String],
) -> Result<XlsxCellValue> {
    match kind {
        CellType::Number => {
            if v.is_empty() {
                Ok(XlsxCellValue::Empty)
            } else {
                match v.parse::<f64>() {
                    Ok(n) => Ok(XlsxCellValue::Number(n)),
                    // Non-numeric content in a numeric cell: fall back to
                    // string so callers still see the data.
                    Err(_) => Ok(XlsxCellValue::String(v.to_string())),
                }
            }
        }
        CellType::SharedString => {
            if v.is_empty() {
                Ok(XlsxCellValue::Empty)
            } else {
                let idx: usize = v
                    .parse()
                    .map_err(|_| io_err(format!("xlsx: invalid shared-string index '{v}'")))?;
                match shared.get(idx) {
                    Some(s) => Ok(XlsxCellValue::String(s.clone())),
                    None => Err(io_err(format!(
                        "xlsx: shared-string index {idx} out of bounds ({} strings)",
                        shared.len()
                    ))),
                }
            }
        }
        CellType::InlineStr => Ok(XlsxCellValue::String(inline.to_string())),
        CellType::Str => Ok(XlsxCellValue::String(v.to_string())),
        CellType::Boolean => {
            let b = matches!(v, "1" | "TRUE" | "true");
            Ok(XlsxCellValue::Boolean(b))
        }
        CellType::Error => Ok(XlsxCellValue::Error(v.to_string())),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellType {
    Number,
    SharedString,
    InlineStr,
    /// A formula result string (`t="str"`).
    Str,
    Boolean,
    Error,
}

impl CellType {
    fn from_attr(v: &[u8]) -> Self {
        match v {
            b"s" => CellType::SharedString,
            b"inlineStr" => CellType::InlineStr,
            b"str" => CellType::Str,
            b"b" => CellType::Boolean,
            b"e" => CellType::Error,
            // "n" (number) is the default too.
            _ => CellType::Number,
        }
    }
}

/// Convert an attribute's value to an owned, unescaped String.
fn attr_to_string(a: &Attribute<'_>) -> Result<String> {
    let cow = a.unescape_value().map_err(xml_err)?;
    Ok(cow.into_owned())
}

/// Decode a [`BytesText`] event to a UTF-8 string and then unescape XML
/// entities (`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&#...;`).
fn decode_and_unescape(t: &BytesText<'_>) -> Result<String> {
    // `decode` handles non-UTF-8 encodings; returns a Cow<str>.
    let decoded = t
        .decode()
        .map_err(|e| io_err(format!("xlsx: text decode failure: {e}")))?;
    let unescaped =
        unescape(&decoded).map_err(|e| io_err(format!("xlsx: text unescape failure: {e}")))?;
    Ok(unescaped.into_owned())
}

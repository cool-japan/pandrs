//! Pure Rust xlsx reader built on top of `oxiarc-archive` + `quick-xml`.
//!
//! Semantics mirror the calamine-based implementation closely enough to keep
//! the existing pandrs public API contract intact:
//!
//! - A workbook is made of one or more sheets, each identified by name.
//! - A single sheet produces a rectangular array of cells.
//! - Cells may be numeric, shared-string, inline-string, boolean, error, or
//!   empty. We normalise them to strings for pandrs-level type inference.
//! - A numeric cell whose style resolves to a date/time number format (see
//!   `xlsx::styles`) is converted to an ISO-8601 string.
//!
//! We do *not* attempt to understand cell formulas (the cell's calculated
//! value is read; formula text is ignored) or cell formatting beyond the
//! date/time detection above. These omissions match the behavior of the
//! previous `calamine`-backed reader, which also threw away formula text
//! and only exposed formatting through placeholder hooks.
//!
//! ## Laziness
//!
//! [`open_workbook`] / [`open_workbook_from_reader`] extract every part's
//! bytes from the zip archive eagerly (the zip format doesn't offer a
//! cheaper way to get at any one part without a full central-directory
//! pass, which we already pay for once via `entries()`), but do **not**
//! XML-parse any worksheet body. That's deferred to [`WorkbookHandle::load_sheet`],
//! so a caller that only wants the sheet list, or only one sheet out of a
//! multi-sheet workbook, never pays to parse the sheets it doesn't need.
//! [`WorkbookHandle::load_all`] remains as an eager "parse everything"
//! convenience for callers that need every sheet anyway.
//!
//! ## Security: dense-matrix allocation guard
//!
//! A worksheet's declared extent (`max_row` / `max_col`, derived from the
//! cell references actually present) is bounded two ways before we
//! materialise anything sized by it:
//!
//! 1. Every individual cell coordinate — whether from an explicit `r`
//!    attribute (via [`parse_ref`]) or an implicit position (see below) — is
//!    rejected outright if it falls outside Excel's own real dimension
//!    limits (1,048,576 rows x 16,384 columns). A legally-produced Excel
//!    file can never exceed these, so this has no false positives.
//! 2. Separately, once the whole sheet has been scanned, we compare the
//!    *claimed* dense extent (`max_row * max_col`) against the *actual*
//!    number of populated cells. A handful of populated cells at a huge
//!    reference — the signature of a file crafted to force a huge
//!    allocation from a tiny input — is rejected; a genuinely dense sheet
//!    (populated cells roughly proportional to its extent) is not, no
//!    matter how large, because the file has to actually contain that many
//!    cells (and therefore bytes) to pass the check.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;

use oxiarc_archive::zip::ZipReader;
use quick_xml::escape::unescape;
use quick_xml::events::attributes::Attribute;
use quick_xml::events::{BytesText, Event};
use quick_xml::Reader;
use quick_xml::XmlVersion;

use crate::error::Result;

use super::cell::{parse_ref, XlsxCellValue, MAX_XLSX_COLS, MAX_XLSX_ROWS};
use super::error::{invalid, io_err, std_io, xml_err, zip_err};
use super::styles::{self, DateKind, StylesInfo};

/// Collection of sheet metadata extracted from `xl/workbook.xml`.
#[derive(Debug, Clone)]
struct WorkbookManifest {
    /// Sheet names in declaration order.
    sheet_names: Vec<String>,
    /// For each entry in `sheet_names`, the workbook-relationship id.
    sheet_rids: Vec<String>,
    /// `<workbookPr date1904="1"/>` — selects the 1904 date system (epoch
    /// 1904-01-01, no leap-year bug) instead of the default 1900 system.
    date1904: bool,
}

/// A workbook whose container-level metadata (sheet manifest, shared
/// strings, styles, date system) has been parsed and whose raw worksheet
/// bytes have been extracted from the zip archive, but whose individual
/// worksheet bodies have **not** been XML-parsed yet. See the module docs
/// for the rationale.
pub(super) struct WorkbookHandle {
    pub(super) sheet_names: Vec<String>,
    shared_strings: Vec<String>,
    styles: StylesInfo,
    date1904: bool,
    /// Decompressed, not-yet-parsed worksheet XML bytes, index-aligned with
    /// `sheet_names`.
    raw_sheet_bytes: Vec<Vec<u8>>,
}

impl WorkbookHandle {
    /// Resolve a sheet name to its index; `None` defaults to the first sheet.
    pub(super) fn sheet_index(&self, name: Option<&str>) -> Result<usize> {
        match name {
            Some(n) => self
                .sheet_names
                .iter()
                .position(|s| s == n)
                .ok_or_else(|| io_err(format!("xlsx: sheet '{n}' not found"))),
            None => {
                if self.sheet_names.is_empty() {
                    Err(io_err("xlsx: workbook contains no sheets"))
                } else {
                    Ok(0)
                }
            }
        }
    }

    /// Parse (on demand) the sheet at `index` into a fully materialised
    /// [`LoadedSheet`]. Each call re-parses the XML; callers that need a
    /// sheet more than once should cache the result themselves.
    pub(super) fn load_sheet(&self, index: usize) -> Result<LoadedSheet> {
        let bytes = self
            .raw_sheet_bytes
            .get(index)
            .ok_or_else(|| io_err(format!("xlsx: sheet index {index} out of range")))?;
        let name = self
            .sheet_names
            .get(index)
            .ok_or_else(|| io_err(format!("xlsx: sheet index {index} out of range")))?
            .clone();
        let (cells, row_count, col_count) =
            parse_worksheet(bytes, &self.shared_strings, &self.styles, self.date1904)?;
        Ok(LoadedSheet {
            name,
            cells,
            row_count,
            col_count,
        })
    }

    /// Parse every sheet. Convenience for callers that need all of them
    /// anyway (sheet-dimension summaries, "read every sheet as a
    /// DataFrame") — no laziness to gain there.
    pub(super) fn load_all(&self) -> Result<Vec<LoadedSheet>> {
        (0..self.sheet_names.len())
            .map(|i| self.load_sheet(i))
            .collect()
    }
}

/// One fully materialised sheet.
pub(super) struct LoadedSheet {
    pub(super) name: String,
    /// Sparse cell storage keyed by zero-indexed `(row, col)`. Only
    /// populated cells are present; any coordinate not in the map is
    /// implicitly [`XlsxCellValue::Empty`]. We deliberately avoid a dense
    /// `Vec<Vec<_>>` here — see the module-level security docs.
    cells: HashMap<(usize, usize), XlsxCellValue>,
    pub(super) row_count: usize,
    pub(super) col_count: usize,
}

impl LoadedSheet {
    /// Value at `(row, col)`, or `Empty` if the cell was never populated.
    pub(super) fn get(&self, row: usize, col: usize) -> &XlsxCellValue {
        const EMPTY: XlsxCellValue = XlsxCellValue::Empty;
        self.cells.get(&(row, col)).unwrap_or(&EMPTY)
    }
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

/// Open an .xlsx file, extracting container-level metadata and raw sheet
/// bytes eagerly but deferring worksheet XML parsing (see module docs).
pub(super) fn open_workbook<P: AsRef<Path>>(path: P) -> Result<WorkbookHandle> {
    let file = File::open(path.as_ref()).map_err(std_io)?;
    let buf = BufReader::new(file);
    open_workbook_from_reader(buf)
}

/// Same as [`open_workbook`] but takes an arbitrary `Read + Seek` source so
/// tests can feed in an in-memory buffer.
pub(super) fn open_workbook_from_reader<R: Read + Seek>(reader: R) -> Result<WorkbookHandle> {
    let mut zr = ZipReader::new(reader).map_err(zip_err)?;

    // Read all parts we need up front. We copy bytes out of the zip reader
    // because the reader itself holds a mutable borrow while extracting,
    // and because the zip format offers no cheaper way to reach one part
    // without walking the whole central directory anyway (which
    // `ZipReader::new` already did to build `entries()`).
    let entries: Vec<_> = zr.entries().to_vec();

    let mut workbook_bytes: Option<Vec<u8>> = None;
    let mut workbook_rels_bytes: Option<Vec<u8>> = None;
    let mut shared_bytes: Option<Vec<u8>> = None;
    let mut styles_bytes: Option<Vec<u8>> = None;
    let mut raw_sheets: Vec<RawSheet> = Vec::new();

    for entry in &entries {
        let name = entry.name.as_str();
        if name == "xl/workbook.xml" {
            workbook_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name == "xl/_rels/workbook.xml.rels" {
            workbook_rels_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name == "xl/sharedStrings.xml" {
            shared_bytes = Some(zr.extract(entry).map_err(zip_err)?);
        } else if name == "xl/styles.xml" {
            styles_bytes = Some(zr.extract(entry).map_err(zip_err)?);
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

    // Parse workbook.xml for sheet list (names + rId) and the date system.
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

    // Parse styles (if present) — needed for date/time detection.
    let styles = match styles_bytes.as_deref() {
        Some(b) => styles::parse_styles(b)?,
        None => StylesInfo::default(),
    };

    // Resolve each manifest sheet to its raw bytes, in manifest order.
    let mut sheet_names: Vec<String> = Vec::with_capacity(manifest.sheet_names.len());
    let mut raw_sheet_bytes: Vec<Vec<u8>> = Vec::with_capacity(manifest.sheet_names.len());
    for (i, (name, rid)) in manifest
        .sheet_names
        .iter()
        .zip(manifest.sheet_rids.iter())
        .enumerate()
    {
        // Prefer rId matching; fall back to positional sheet{N}.xml.
        let fallback_path = format!("xl/worksheets/sheet{}.xml", i + 1);
        let pos = raw_sheets
            .iter()
            .position(|r| !r.rid.is_empty() && r.rid == *rid)
            .or_else(|| raw_sheets.iter().position(|r| r.path == fallback_path))
            .ok_or_else(|| io_err(format!("xlsx: sheet '{name}' not found in archive")))?;
        let raw = raw_sheets.remove(pos);
        sheet_names.push(name.clone());
        raw_sheet_bytes.push(raw.bytes);
    }

    Ok(WorkbookHandle {
        sheet_names,
        shared_strings,
        styles,
        date1904: manifest.date1904,
        raw_sheet_bytes,
    })
}

/// Parse `xl/workbook.xml` to recover the sheet list (names & rId values)
/// and the `date1904` workbook property.
fn parse_workbook_manifest(bytes: &[u8]) -> Result<WorkbookManifest> {
    let mut names = Vec::new();
    let mut rids = Vec::new();
    let mut date1904 = false;
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
            Event::Empty(e) | Event::Start(e) if e.name().as_ref() == b"workbookPr" => {
                for a in e.attributes().with_checks(false).flatten() {
                    if a.key.as_ref() == b"date1904" {
                        let v = attr_to_string(&a)?;
                        date1904 = matches!(v.as_str(), "1" | "true" | "TRUE");
                    }
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
        date1904,
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
/// runs for rich text; we concatenate the runs. An `<si>` may *also* contain
/// an `<rPh>` (phonetic guide, e.g. Japanese furigana) subtree with its own
/// `<t>` runs — those must **not** contribute to the string, or e.g. the
/// Japanese name "山田" would come back as "山田ヤマダ" (the base text with
/// its own pronunciation guide appended). `<phoneticPr>` never carries text
/// content, but we don't rely on that; the `in_rph` gate covers it too.
fn parse_shared_strings(bytes: &[u8]) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    // State machine: we're inside an `<si>` after we see <si>, and we push
    // whenever we hit </si>. Within an <si>, every <t>...</t> contributes to
    // the accumulator (rich text merges the runs) *unless* it's nested
    // inside an <rPh> phonetic-guide subtree.
    let mut in_si = false;
    let mut in_rph = false;
    let mut in_t = false;
    let mut accum = String::new();

    loop {
        match reader.read_event_into(&mut buf).map_err(xml_err)? {
            Event::Eof => break,
            Event::Start(e) => match e.name().as_ref() {
                b"si" => {
                    in_si = true;
                    in_rph = false;
                    accum.clear();
                }
                b"rPh" => {
                    in_rph = true;
                }
                b"t" => {
                    if in_si && !in_rph {
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
                b"rPh" => {
                    in_rph = false;
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

/// Bounds-check a resolved `(row, col)` coordinate — whether it came from an
/// explicit `r` attribute (already checked once inside [`parse_ref`], so
/// this is a cheap redundant re-check there) or from the implicit-position
/// fallback below (where this is the *only* check).
fn check_bounds(r: usize, c: usize) -> Result<()> {
    if r >= MAX_XLSX_ROWS || c >= MAX_XLSX_COLS {
        return Err(invalid(format!(
            "xlsx: cell position (row {}, col {}) exceeds the {MAX_XLSX_ROWS}x{MAX_XLSX_COLS} worksheet limit",
            r + 1,
            c + 1
        )));
    }
    Ok(())
}

/// Parse one `xl/worksheets/sheet{N}.xml` into a sparse cell map.
///
/// Returns `(cells, row_count, col_count)`, where `row_count`/`col_count`
/// are the declared extent (one past the highest populated row/column seen).
fn parse_worksheet(
    bytes: &[u8],
    shared: &[String],
    styles: &StylesInfo,
    date1904: bool,
) -> Result<(HashMap<(usize, usize), XlsxCellValue>, usize, usize)> {
    let mut reader = Reader::from_reader(Cursor::new(bytes));
    reader.config_mut().trim_text(false);
    let mut buf = Vec::new();

    // Result accumulator (sparse: only populated cells are inserted).
    let mut cells: HashMap<(usize, usize), XlsxCellValue> = HashMap::new();
    let mut max_row = 0usize;
    let mut max_col = 0usize;

    // <row>/implicit-position tracking (ECMA-376 §18.3.1.73/.4: a `<c>`
    // without an `r` attribute takes the position immediately after the
    // previous cell in its row; a `<row>` without an `r` attribute takes
    // the position immediately after the previous row).
    let mut cur_row: usize = 0;
    let mut next_implicit_row: usize = 0;
    let mut col_cursor: usize = 0;

    // Per-cell parse state.
    let mut cur_ref: Option<(usize, usize)> = None;
    let mut cur_type: CellType = CellType::Number;
    let mut cur_style: Option<u32> = None;
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
                if tag == b"row" {
                    let row = resolve_row_index(&e, next_implicit_row)?;
                    check_bounds(row, 0)?;
                    cur_row = row;
                    next_implicit_row = row + 1;
                    col_cursor = 0;
                } else if tag == b"c" {
                    cur_ref = None;
                    cur_type = CellType::Number;
                    cur_style = None;
                    value_text.clear();
                    inline_text.clear();
                    for a in e.attributes().with_checks(false).flatten() {
                        let key = a.key.as_ref();
                        if key == b"r" {
                            let s = attr_to_string(&a)?;
                            cur_ref = Some(parse_ref(&s)?);
                        } else if key == b"t" {
                            cur_type = CellType::from_attr(a.value.as_ref());
                        } else if key == b"s" {
                            let s = attr_to_string(&a)?;
                            cur_style = s.parse::<u32>().ok();
                        }
                    }
                    // Implicit position: no `r` attribute present.
                    let (r, c) = cur_ref.unwrap_or((cur_row, col_cursor));
                    check_bounds(r, c)?;
                    cur_ref = Some((r, c));
                } else if tag == b"v" {
                    in_value = true;
                    value_text.clear();
                } else if tag == b"t" {
                    // Inline text run. `inline_text` was already cleared once
                    // for this cell at the `<c>` branch above (NOT here) —
                    // clearing it again on every `<t>` would keep only the
                    // last run of a multi-run rich-text inline string.
                    in_inline_text = true;
                }
            }
            Event::Empty(e) => {
                let name = e.name();
                let tag = name.as_ref();
                if tag == b"row" {
                    // A fully empty row, e.g. `<row r="5"/>`, with no cells.
                    let row = resolve_row_index(&e, next_implicit_row)?;
                    check_bounds(row, 0)?;
                    next_implicit_row = row + 1;
                } else if tag == b"c" {
                    // Self-closed cell (never carries a value): still
                    // register its position — explicit or implicit — so the
                    // column layout and `col_cursor` stay correct for any
                    // later implicit-position cells in this row.
                    let mut coord: Option<(usize, usize)> = None;
                    for a in e.attributes().with_checks(false).flatten() {
                        if a.key.as_ref() == b"r" {
                            let s = attr_to_string(&a)?;
                            coord = Some(parse_ref(&s)?);
                        }
                    }
                    let (r, c) = coord.unwrap_or((cur_row, col_cursor));
                    check_bounds(r, c)?;
                    cells.insert((r, c), XlsxCellValue::Empty);
                    max_row = max_row.max(r + 1);
                    max_col = max_col.max(c + 1);
                    col_cursor = c + 1;
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
                    // Commit the cell. `cur_ref` is always `Some` by now —
                    // set unconditionally (explicit-or-implicit) in the
                    // `Start` branch above.
                    let Some((r, c)) = cur_ref else { continue };
                    let mut val = finalise_cell(&cur_type, &value_text, &inline_text, shared)?;
                    // Only a plain numeric cell can be date-styled — xlsx
                    // has no separate "date" cell type.
                    if cur_type == CellType::Number {
                        if let Some(serial) = val.as_number() {
                            let kind = styles.date_kind(cur_style);
                            if kind != DateKind::None {
                                if let Some(iso) =
                                    styles::excel_serial_to_iso(serial, date1904, kind)
                                {
                                    val = XlsxCellValue::String(iso);
                                }
                                // `None` (out-of-range / non-finite serial
                                // under a date style): leave `val` as the
                                // original number rather than fabricate a
                                // date.
                            }
                        }
                    }
                    cells.insert((r, c), val);
                    max_row = max_row.max(r + 1);
                    max_col = max_col.max(c + 1);
                    col_cursor = c + 1;
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

    // Security: guard against a worksheet that *declares* a huge extent
    // (via a cell reference far from the origin) while actually populating
    // only a handful of cells — the signature of a file crafted to force an
    // oversized allocation from a tiny input. A floor keeps this a no-op for
    // small/normal sheets, and a genuinely dense sheet of any size passes
    // because its populated-cell count scales with its extent (and with the
    // bytes actually read to produce that count).
    const DENSITY_GUARD_FLOOR: u64 = 100_000;
    const DENSITY_GUARD_FACTOR: u64 = 128;
    let populated = cells.len() as u64;
    let claimed = (max_row as u64).saturating_mul(max_col as u64);
    if claimed >= DENSITY_GUARD_FLOOR && claimed > populated.saturating_mul(DENSITY_GUARD_FACTOR) {
        return Err(invalid(format!(
            "xlsx: worksheet declares {max_row} rows x {max_col} cols ({claimed} cells) but only \
             {populated} cells are actually populated (ratio exceeds the {DENSITY_GUARD_FACTOR}x \
             safety margin); refusing to allocate a dense matrix for what looks like a maliciously \
             crafted or corrupt file"
        )));
    }

    Ok((cells, max_row, max_col))
}

/// Resolve a `<row>` element's zero-indexed row number: its own `r`
/// attribute if present, otherwise the next implicit row (one past the
/// previous row).
fn resolve_row_index(e: &quick_xml::events::BytesStart<'_>, next_implicit: usize) -> Result<usize> {
    for a in e.attributes().with_checks(false).flatten() {
        if a.key.as_ref() == b"r" {
            let s = attr_to_string(&a)?;
            let n: usize = s
                .parse()
                .map_err(|_| invalid(format!("xlsx: invalid row index '{s}'")))?;
            if n == 0 {
                return Err(invalid("xlsx: invalid row index '0'"));
            }
            return Ok(n - 1);
        }
    }
    Ok(next_implicit)
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
                    Ok(n) => Ok(XlsxCellValue::Number {
                        value: n,
                        raw: v.to_string(),
                    }),
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
    // `unescape_value()` is deprecated in favor of `normalized_value()`, which takes
    // an explicit XML version. `Implicit1_0` reproduces the old method's behavior
    // (and matches `XmlVersion::default()`).
    let cow = a
        .normalized_value(XmlVersion::Implicit1_0)
        .map_err(xml_err)?;
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

//! Pure Rust xlsx writer built on top of `oxiarc-archive`.
//!
//! Produces a minimum-viable OOXML SpreadsheetML package containing one or
//! more worksheets. The writer is intentionally focused: no formulas, no
//! formatting, no charts. That matches the pre-existing behavior of the
//! `simple_excel_writer`-based implementation that it replaces, while giving
//! data values the accurate cell types (numbers vs shared-string indices vs
//! booleans) the old writer never provided.
//!
//! Durability: the archive is assembled entirely in a temp file next to the
//! destination path and only `rename`d into place once every part has been
//! written and flushed successfully (see [`write_xlsx`]). A failure partway
//! through (disk full, I/O error, process killed) therefore never leaves a
//! truncated or zero-length file at the destination path — either the
//! rename happens and the destination is the complete new file, or it
//! doesn't and the destination is untouched (whatever it was before, if
//! anything).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use oxiarc_archive::zip::{ZipCompressionLevel, ZipWriter};

use crate::column::Column;
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

use super::cell::{
    encode_ref, format_float, format_int, validate_sheet_name, validate_unique_sheet_names,
    SharedStringsBuilder,
};
use super::error::{invalid, std_io, zip_err};
use super::schema::{
    content_types, root_rels, shared_strings_xml, styles_xml, workbook_rels, workbook_xml,
};

/// Represents a single sheet payload that is ready to be encoded into xml.
///
/// The writer takes a borrow rather than owning copies of each column so that
/// multi-sheet writes do not pay for unnecessary cloning of large dataframes.
pub(super) struct SheetPayload<'a> {
    pub(super) name: String,
    pub(super) df: &'a SplitDataFrame,
    pub(super) include_index: bool,
}

/// Core xlsx write routine. Builds every xml part in-memory, then emits them
/// as separate files into a single ZIP archive written to a temp path, which
/// is only renamed over `path` once the whole archive has been written and
/// flushed successfully.
pub(super) fn write_xlsx<P: AsRef<Path>>(path: P, sheets: &[SheetPayload<'_>]) -> Result<()> {
    for s in sheets {
        validate_sheet_name(&s.name)?;
    }
    validate_unique_sheet_names(sheets.iter().map(|s| s.name.as_str()))?;

    // Step 1: build shared strings & worksheet bodies.
    let mut sst = SharedStringsBuilder::new();
    let mut worksheet_xmls: Vec<String> = Vec::with_capacity(sheets.len());
    for sheet in sheets {
        worksheet_xmls.push(build_sheet_xml(sheet, &mut sst)?);
    }

    let (unique_strings, total_refs) = sst.into_ordered();
    let shared = shared_strings_xml(&unique_strings, total_refs);
    let sheet_names: Vec<String> = sheets.iter().map(|s| s.name.clone()).collect();
    let ct = content_types(sheets.len());
    let wb = workbook_xml(&sheet_names);
    let wb_rels = workbook_rels(sheets.len());
    let styles = styles_xml();
    let root = root_rels();

    let sheet_part_names: Vec<String> = (1..=worksheet_xmls.len())
        .map(|i| format!("xl/worksheets/sheet{i}.xml"))
        .collect();

    let mut parts: Vec<(&str, &[u8])> = vec![
        ("[Content_Types].xml", ct.as_bytes()),
        ("_rels/.rels", root.as_bytes()),
        ("xl/workbook.xml", wb.as_bytes()),
        ("xl/_rels/workbook.xml.rels", wb_rels.as_bytes()),
        ("xl/styles.xml", styles.as_bytes()),
        ("xl/sharedStrings.xml", shared.as_bytes()),
    ];
    for (name, xml) in sheet_part_names.iter().zip(worksheet_xmls.iter()) {
        parts.push((name.as_str(), xml.as_bytes()));
    }

    // Step 2: materialise the archive at a temp path, then atomically
    // publish it. Any failure removes the (partial) temp file — best effort
    // — before propagating the original error; the target path is only ever
    // touched by the final `rename`.
    let target = path.as_ref();
    let tmp_path = temp_sibling_path(target)?;
    if let Err(err) = write_archive_to(&tmp_path, &parts) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(err);
    }
    if let Err(e) = std::fs::rename(&tmp_path, target) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(std_io(e));
    }
    Ok(())
}

/// Choose a temp-file path alongside `target` (same directory, so the final
/// `rename` is an atomic same-filesystem move rather than a cross-filesystem
/// copy+delete). Includes the process id and a monotonic counter so
/// concurrent writers targeting the same directory can't collide.
fn temp_sibling_path(target: &Path) -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let dir = target
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = target
        .file_name()
        .ok_or_else(|| invalid("xlsx: output path has no file name"))?
        .to_string_lossy()
        .into_owned();
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let unique = format!(".{file_name}.{}.{n}.tmp", std::process::id());
    Ok(dir.join(unique))
}

/// Write every `(archive_path, bytes)` part into a fresh ZIP at `tmp_path`,
/// flushing (and, best-effort, syncing) it to disk before returning.
fn write_archive_to(tmp_path: &Path, parts: &[(&str, &[u8])]) -> Result<()> {
    let file = File::create(tmp_path).map_err(std_io)?;
    let writer = BufWriter::new(file);
    let mut zw = ZipWriter::new(writer);
    // DEFLATE normal keeps output size reasonable. xlsx readers assume deflate.
    zw.set_compression(ZipCompressionLevel::Normal);

    for (name, bytes) in parts {
        zw.add_file(name, bytes).map_err(zip_err)?;
    }

    // `ZipWriter::into_inner` calls `finish()` internally and returns the inner
    // writer. `BufWriter::drop` silently swallows flush errors, so we explicitly
    // flush it here to ensure disk-full or short-write failures propagate back
    // to the caller rather than being lost in a destructor.
    let mut buf_writer = zw.into_inner().map_err(zip_err)?;
    buf_writer.flush().map_err(std_io)?;
    // Best-effort durability: make sure the temp file's bytes are actually
    // on disk before we `rename` it over the destination.
    let _ = buf_writer.get_ref().sync_all();
    Ok(())
}

/// Build a worksheet XML body for a single sheet, interning strings into the
/// supplied shared-strings builder as needed.
fn build_sheet_xml(sheet: &SheetPayload<'_>, sst: &mut SharedStringsBuilder) -> Result<String> {
    let df = sheet.df;
    let col_names = df.column_names();
    let row_count = df.row_count();

    let mut xml = String::with_capacity(256);
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>"#);
    xml.push_str(
        r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">"#,
    );
    xml.push_str("<sheetData>");

    // Header row: column names (always written). Optionally prefixed with "Index".
    let mut header_cells: Vec<String> = Vec::with_capacity(col_names.len() + 1);
    if sheet.include_index {
        header_cells.push("Index".to_string());
    }
    for n in col_names {
        header_cells.push(n.clone());
    }
    append_row(&mut xml, 0, &header_cells, sst);

    // The real index labels (if the frame carries index metadata). Both
    // simple and multi-indexes stringify via `string_values()`; a frame with
    // no index at all (e.g. a bare `SplitDataFrame::new()`) falls back to
    // the row position, matching the previous behavior for that case.
    let index_labels: Option<Vec<String>> = if sheet.include_index {
        df.get_index().and_then(|idx| idx.string_values())
    } else {
        None
    };

    // Data rows.
    for row_idx in 0..row_count {
        xml.push_str(&format!(r#"<row r="{}">"#, row_idx + 2));
        let mut col_cursor = 0usize;
        if sheet.include_index {
            let fallback = row_idx.to_string();
            let label: &str = index_labels
                .as_ref()
                .and_then(|v| v.get(row_idx))
                .map(|s| s.as_str())
                .unwrap_or(&fallback);
            append_shared_string_cell(&mut xml, row_idx + 1, col_cursor, label, sst);
            col_cursor += 1;
        }
        for col in &df.columns {
            append_cell_from_column(&mut xml, row_idx + 1, col_cursor, col, row_idx, sst)?;
            col_cursor += 1;
        }
        xml.push_str("</row>");
    }

    xml.push_str("</sheetData>");
    xml.push_str("</worksheet>");
    Ok(xml)
}

/// Emit one row where every cell is a string (used for the header row).
fn append_row(xml: &mut String, row_zero: usize, cells: &[String], sst: &mut SharedStringsBuilder) {
    xml.push_str(&format!(r#"<row r="{}">"#, row_zero + 1));
    for (col_idx, val) in cells.iter().enumerate() {
        let cell_ref = encode_ref(row_zero, col_idx);
        let idx = sst.intern(val);
        xml.push_str(&format!(r#"<c r="{cell_ref}" t="s"><v>{idx}</v></c>"#));
    }
    xml.push_str("</row>");
}

/// Emit an Int64-typed numeric cell (compact, no decimal point).
fn append_int_cell(xml: &mut String, row_zero: usize, col_idx: usize, v: i64) {
    let cell_ref = encode_ref(row_zero, col_idx);
    xml.push_str(&format!(
        r#"<c r="{cell_ref}"><v>{}</v></c>"#,
        format_int(v)
    ));
}

/// Emit a Float64-typed numeric cell. `v` must be finite — NaN/±Infinity are
/// handled by the caller via [`append_error_cell`] instead, since xlsx has
/// no representation for them as a numeric `<v>`.
fn append_float_cell(xml: &mut String, row_zero: usize, col_idx: usize, v: f64) {
    let cell_ref = encode_ref(row_zero, col_idx);
    xml.push_str(&format!(
        r#"<c r="{cell_ref}"><v>{}</v></c>"#,
        format_float(v)
    ));
}

/// Emit a boolean cell using the `t="b"` encoding.
fn append_bool_cell(xml: &mut String, row_zero: usize, col_idx: usize, v: bool) {
    let cell_ref = encode_ref(row_zero, col_idx);
    let s = if v { "1" } else { "0" };
    xml.push_str(&format!(r#"<c r="{cell_ref}" t="b"><v>{s}</v></c>"#));
}

/// Emit a shared-string cell.
fn append_shared_string_cell(
    xml: &mut String,
    row_zero: usize,
    col_idx: usize,
    text: &str,
    sst: &mut SharedStringsBuilder,
) {
    let cell_ref = encode_ref(row_zero, col_idx);
    let idx = sst.intern(text);
    xml.push_str(&format!(r#"<c r="{cell_ref}" t="s"><v>{idx}</v></c>"#));
}

/// Emit an error-typed cell (`t="e"`), e.g. for a Float64 value that cannot
/// be represented as an xlsx numeric literal (NaN, +Infinity, -Infinity).
/// This is a legal, well-formed xlsx cell (unlike embedding the literal text
/// "NaN"/"Infinity" inside a plain numeric `<v>`, which Excel refuses to
/// open), and it honestly signals "not a representable number" rather than
/// silently substituting 0 or blanking the cell.
fn append_error_cell(xml: &mut String, row_zero: usize, col_idx: usize, error_code: &str) {
    let cell_ref = encode_ref(row_zero, col_idx);
    xml.push_str(&format!(
        r#"<c r="{cell_ref}" t="e"><v>{error_code}</v></c>"#
    ));
}

/// Emit an empty cell — produced for NULLs. An empty `<c>` with no value is a
/// valid xlsx representation of a blank cell.
fn append_empty_cell(xml: &mut String, row_zero: usize, col_idx: usize) {
    let cell_ref = encode_ref(row_zero, col_idx);
    xml.push_str(&format!(r#"<c r="{cell_ref}"/>"#));
}

/// Dispatch to the right cell encoder based on the column type.
fn append_cell_from_column(
    xml: &mut String,
    row_zero: usize,
    col_idx: usize,
    col: &Column,
    row_idx: usize,
    sst: &mut SharedStringsBuilder,
) -> Result<()> {
    match col {
        Column::Int64(c) => match c.get(row_idx)? {
            Some(v) => append_int_cell(xml, row_zero, col_idx, v),
            None => append_empty_cell(xml, row_zero, col_idx),
        },
        Column::Float64(c) => match c.get(row_idx)? {
            Some(v) if v.is_finite() => append_float_cell(xml, row_zero, col_idx, v),
            // NaN/+Inf/-Inf: xlsx has no numeric literal for these. Emitting
            // `<v>NaN</v>` produces a file Excel rejects on open; an error
            // cell is honest (signals "not a number") and stays well-formed.
            Some(_) => append_error_cell(xml, row_zero, col_idx, "#NUM!"),
            None => append_empty_cell(xml, row_zero, col_idx),
        },
        Column::String(c) => match c.get(row_idx)? {
            Some(s) => append_shared_string_cell(xml, row_zero, col_idx, s, sst),
            None => append_empty_cell(xml, row_zero, col_idx),
        },
        Column::Boolean(c) => match c.get(row_idx)? {
            Some(b) => append_bool_cell(xml, row_zero, col_idx, b),
            None => append_empty_cell(xml, row_zero, col_idx),
        },
    }
    Ok(())
}

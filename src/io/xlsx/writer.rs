//! Pure Rust xlsx writer built on top of `oxiarc-archive`.
//!
//! Produces a minimum-viable OOXML SpreadsheetML package containing one or
//! more worksheets. The writer is intentionally focused: no formulas, no
//! formatting, no charts. That matches the pre-existing behavior of the
//! `simple_excel_writer`-based implementation that it replaces, while giving
//! data values the accurate cell types (numbers vs shared-string indices vs
//! booleans) the old writer never provided.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use oxiarc_archive::zip::{ZipCompressionLevel, ZipWriter};

use crate::column::Column;
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

use super::cell::{encode_ref, format_number, validate_sheet_name, SharedStringsBuilder};
use super::error::{std_io, zip_err};
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
/// as separate files into a single ZIP archive.
pub(super) fn write_xlsx<P: AsRef<Path>>(path: P, sheets: &[SheetPayload<'_>]) -> Result<()> {
    for s in sheets {
        validate_sheet_name(&s.name)?;
    }

    // Step 1: build shared strings & worksheet bodies.
    let mut sst = SharedStringsBuilder::new();
    let mut worksheet_xmls: Vec<String> = Vec::with_capacity(sheets.len());
    for sheet in sheets {
        worksheet_xmls.push(build_sheet_xml(sheet, &mut sst)?);
    }

    let shared = shared_strings_xml(&sst.into_ordered());
    let sheet_names: Vec<String> = sheets.iter().map(|s| s.name.clone()).collect();
    let ct = content_types(sheets.len());
    let wb = workbook_xml(&sheet_names);
    let wb_rels = workbook_rels(sheets.len());
    let styles = styles_xml();
    let root = root_rels();

    // Step 2: materialise the archive using OxiARC's ZIP writer.
    let file = File::create(path.as_ref()).map_err(std_io)?;
    let writer = BufWriter::new(file);
    let mut zw = ZipWriter::new(writer);
    // DEFLATE normal keeps output size reasonable. xlsx readers assume deflate.
    zw.set_compression(ZipCompressionLevel::Normal);

    zw.add_file("[Content_Types].xml", ct.as_bytes())
        .map_err(zip_err)?;
    zw.add_file("_rels/.rels", root.as_bytes())
        .map_err(zip_err)?;
    zw.add_file("xl/workbook.xml", wb.as_bytes())
        .map_err(zip_err)?;
    zw.add_file("xl/_rels/workbook.xml.rels", wb_rels.as_bytes())
        .map_err(zip_err)?;
    zw.add_file("xl/styles.xml", styles.as_bytes())
        .map_err(zip_err)?;
    zw.add_file("xl/sharedStrings.xml", shared.as_bytes())
        .map_err(zip_err)?;
    for (i, xml) in worksheet_xmls.iter().enumerate() {
        let name = format!("xl/worksheets/sheet{}.xml", i + 1);
        zw.add_file(&name, xml.as_bytes()).map_err(zip_err)?;
    }
    // `ZipWriter::into_inner` calls `finish()` internally and returns the inner
    // writer. `BufWriter::drop` silently swallows flush errors, so we explicitly
    // flush it here to ensure disk-full or short-write failures propagate back
    // to the caller rather than being lost in a destructor.
    let mut buf_writer = zw.into_inner().map_err(zip_err)?;
    buf_writer.flush().map_err(std_io)?;
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

    // Data rows.
    for row_idx in 0..row_count {
        xml.push_str(&format!(r#"<row r="{}">"#, row_idx + 2));
        let mut col_cursor = 0usize;
        if sheet.include_index {
            append_number_cell(&mut xml, row_idx + 1, col_cursor, row_idx as f64);
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

/// Emit a numeric cell. `row_zero` is zero-indexed, the caller adds the header
/// offset.
fn append_number_cell(xml: &mut String, row_zero: usize, col_idx: usize, v: f64) {
    let cell_ref = encode_ref(row_zero, col_idx);
    let s = format_number(v);
    xml.push_str(&format!(r#"<c r="{cell_ref}"><v>{s}</v></c>"#));
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
            Some(v) => append_number_cell(xml, row_zero, col_idx, v as f64),
            None => append_empty_cell(xml, row_zero, col_idx),
        },
        Column::Float64(c) => match c.get(row_idx)? {
            Some(v) => append_number_cell(xml, row_zero, col_idx, v),
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

/// Emit an empty cell — produced for NULLs. An empty `<c>` with no value is a
/// valid xlsx representation of a blank cell.
fn append_empty_cell(xml: &mut String, row_zero: usize, col_idx: usize) {
    let cell_ref = encode_ref(row_zero, col_idx);
    xml.push_str(&format!(r#"<c r="{cell_ref}"/>"#));
}

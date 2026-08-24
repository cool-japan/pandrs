//! Regression tests for the xlsx reader/writer/facade fixes made in the
//! `io-excel` work item (see /private/tmp scratch report `io_audit.md`
//! section 4). Each test pins one previously-broken behavior so it can't
//! silently regress.
//!
//! Some scenarios (huge/implicit cell references, `<rPh>` phonetic runs,
//! styled date cells) can't be produced by pandrs' own writer, so this file
//! hand-builds minimal .xlsx fixtures directly via `oxiarc_archive::zip`.

#![cfg(feature = "excel")]

use std::collections::HashMap;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;

use oxiarc_archive::zip::{ZipCompressionLevel, ZipReader, ZipWriter};
use pandrs::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use pandrs::error::{Error, Result};
use pandrs::io::{
    list_sheet_names, optimize_excel_file, read_excel, read_excel_enhanced, read_excel_sheets,
    write_excel, write_excel_enhanced, write_excel_sheets, ExcelCell, ExcelCellFormat,
    ExcelReadOptions, ExcelWriteOptions, NamedRange,
};
use pandrs::{Index, OptimizedDataFrame};
use tempfile::tempdir;

// ---------------------------------------------------------------------
// Hand-built .xlsx fixture support (for scenarios pandrs' own writer
// cannot produce: implicit cell positions, huge cell refs, `<rPh>` runs,
// styled date cells).
// ---------------------------------------------------------------------

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/><Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>"#;

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;

const WORKBOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>"#;

const WORKBOOK_1900: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#;

const WORKBOOK_1904: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><workbookPr date1904="1"/><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>"#;

const STYLES_PLAIN: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellXfs></styleSheet>"#;

/// Style index 0 = plain, 1 = custom date format ("yyyy-mm-dd"), 2 = builtin
/// date format (numFmtId 14, "mm-dd-yy").
const STYLES_WITH_DATES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="164" formatCode="yyyy-mm-dd"/></numFmts><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/><xf numFmtId="14" fontId="0" fillId="0" borderId="0" applyNumberFormat="1"/></cellXfs></styleSheet>"#;

const SHARED_EMPTY: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="0" uniqueCount="0"></sst>"#;

/// A single shared string "山田" whose `<si>` also carries an `<rPh>`
/// phonetic-guide run "ヤマダ" that must NOT be concatenated onto the base
/// text.
const SHARED_WITH_RPH: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1"><si><r><t>山田</t></r><rPh sb="0" eb="2"><t>ヤマダ</t></rPh><phoneticPr fontId="1" type="fullwidthKatakana"/></si></sst>"#;

fn wrap_sheet(body: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{body}</sheetData></worksheet>"#
    )
}

/// Build a minimal, hand-crafted single-sheet .xlsx at `path` from raw XML
/// parts, bypassing pandrs' own writer entirely.
fn write_fixture(
    path: &Path,
    workbook_xml: &str,
    styles_xml: &str,
    shared_strings_xml: &str,
    sheet_xml: &str,
) {
    let file = File::create(path).expect("create fixture file");
    let mut zw = ZipWriter::new(file);
    zw.set_compression(ZipCompressionLevel::Normal);
    zw.add_file("[Content_Types].xml", CONTENT_TYPES.as_bytes())
        .expect("write content types");
    zw.add_file("_rels/.rels", ROOT_RELS.as_bytes())
        .expect("write root rels");
    zw.add_file("xl/workbook.xml", workbook_xml.as_bytes())
        .expect("write workbook.xml");
    zw.add_file("xl/_rels/workbook.xml.rels", WORKBOOK_RELS.as_bytes())
        .expect("write workbook rels");
    zw.add_file("xl/styles.xml", styles_xml.as_bytes())
        .expect("write styles.xml");
    zw.add_file("xl/sharedStrings.xml", shared_strings_xml.as_bytes())
        .expect("write sharedStrings.xml");
    zw.add_file("xl/worksheets/sheet1.xml", sheet_xml.as_bytes())
        .expect("write sheet1.xml");
    let mut inner = zw.into_inner().expect("finish zip");
    inner.flush().expect("flush fixture file");
}

/// Read back the raw bytes of one archive member of an already-written
/// .xlsx file, so a test can assert on the literal XML a write produced
/// (e.g. "no `<v>NaN</v>` anywhere") instead of only on the round-tripped,
/// re-parsed value.
fn read_archive_member(path: &Path, member: &str) -> String {
    let file = File::open(path).expect("open xlsx for raw inspection");
    let mut zr = ZipReader::new(file).expect("open zip");
    let entries: Vec<_> = zr.entries().to_vec();
    let entry = entries
        .iter()
        .find(|e| e.name == member)
        .unwrap_or_else(|| panic!("archive member '{member}' not found"));
    let bytes = zr.extract(entry).expect("extract member");
    String::from_utf8(bytes).expect("member is valid UTF-8")
}

// ---------------------------------------------------------------------
// Reader: security (DoS) fixes
// ---------------------------------------------------------------------

#[test]
fn dense_matrix_dos_guard_rejects_huge_sparse_cell_ref() {
    // Two populated cells, one of them at Excel's own maximum coordinate
    // (XFD1048576). Before the density guard, this would have driven a
    // ~1.7e10-cell dense allocation from a file just a few hundred bytes
    // long.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dos.xlsx");
    let sheet = wrap_sheet(
        r#"<row r="1"><c r="A1"><v>1</v></c></row><row r="1048576"><c r="XFD1048576"><v>2</v></c></row>"#,
    );
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_EMPTY, &sheet);

    let result = read_excel(&path, None, false, 0, None);
    assert!(
        result.is_err(),
        "a worksheet claiming 1,048,576 x 16,384 cells from only 2 populated cells must be rejected"
    );
}

#[test]
fn dense_matrix_dos_guard_allows_genuinely_dense_small_sheet() {
    // Sanity check for the guard's floor: an ordinary, fully-populated
    // small sheet must not be affected.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dense_ok.xlsx");
    let mut body = String::new();
    for r in 1..=20 {
        body.push_str(&format!(r#"<row r="{r}">"#));
        for c in 0..5 {
            let col = (b'A' + c) as char;
            body.push_str(&format!(r#"<c r="{col}{r}"><v>{r}</v></c>"#));
        }
        body.push_str("</row>");
    }
    let sheet = wrap_sheet(&body);
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_EMPTY, &sheet);

    let df = read_excel(&path, None, false, 0, None).expect("dense small sheet must be accepted");
    assert_eq!(df.row_count(), 20);
}

#[test]
fn cell_ref_with_absurd_column_length_is_rejected_end_to_end() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("overflow.xlsx");
    let huge_ref = format!("{}1", "Z".repeat(200));
    let sheet = wrap_sheet(&format!(
        r#"<row r="1"><c r="{huge_ref}"><v>1</v></c></row>"#
    ));
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_EMPTY, &sheet);

    let result = read_excel(&path, None, false, 0, None);
    assert!(
        result.is_err(),
        "a pathological cell ref must error, not panic"
    );
}

// ---------------------------------------------------------------------
// Reader: implicit cell position, rich text, phonetic strings
// ---------------------------------------------------------------------

#[test]
fn cells_without_r_attribute_use_implicit_position() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("implicit.xlsx");
    // Neither `<row>` nor any `<c>` carries an `r` attribute.
    let sheet = wrap_sheet(
        "<row><c><v>10</v></c><c><v>20</v></c></row><row><c><v>30</v></c><c><v>40</v></c></row>",
    );
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_EMPTY, &sheet);

    let df = read_excel(&path, None, false, 0, None)
        .expect("implicit-position cells must not be dropped");
    assert_eq!(
        df.row_count(),
        2,
        "no cells should have been silently skipped"
    );
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["10".to_string(), "30".to_string()]
    );
    assert_eq!(
        df.get_column_string_values("Column2").unwrap(),
        vec!["20".to_string(), "40".to_string()]
    );
}

#[test]
fn rich_text_inline_string_keeps_every_run_not_just_the_last() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("richtext.xlsx");
    let sheet = wrap_sheet(
        r#"<row r="1"><c r="A1" t="inlineStr"><is><r><t>Hello </t></r><r><t>World</t></r></is></c></row>"#,
    );
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_EMPTY, &sheet);

    let df = read_excel(&path, None, false, 0, None).unwrap();
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["Hello World".to_string()]
    );
}

#[test]
fn shared_string_phonetic_ruby_is_excluded_from_the_base_text() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("phonetic.xlsx");
    let sheet = wrap_sheet(r#"<row r="1"><c r="A1" t="s"><v>0</v></c></row>"#);
    write_fixture(&path, WORKBOOK_1900, STYLES_PLAIN, SHARED_WITH_RPH, &sheet);

    let df = read_excel(&path, None, false, 0, None).unwrap();
    let values = df.get_column_string_values("Column1").unwrap();
    assert_eq!(
        values,
        vec!["山田".to_string()],
        "the <rPh> phonetic guide run must not be appended to the base text"
    );
}

// ---------------------------------------------------------------------
// Reader: date/time detection via styles.xml
// ---------------------------------------------------------------------

#[test]
fn custom_date_format_converts_serial_to_iso8601() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("date_custom.xlsx");
    // 45658 == 2025-01-01 (cross-checked against the canonical
    // 1899-12-30-epoch formula every xlsx date implementation uses).
    let sheet = wrap_sheet(r#"<row r="1"><c r="A1" s="1"><v>45658</v></c></row>"#);
    write_fixture(
        &path,
        WORKBOOK_1900,
        STYLES_WITH_DATES,
        SHARED_EMPTY,
        &sheet,
    );

    let df = read_excel(&path, None, false, 0, None).unwrap();
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["2025-01-01".to_string()]
    );
}

#[test]
fn builtin_date_format_converts_serial_to_iso8601() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("date_builtin.xlsx");
    // 44197 == 2021-01-01, style index 2 == builtin numFmtId 14.
    let sheet = wrap_sheet(r#"<row r="1"><c r="A1" s="2"><v>44197</v></c></row>"#);
    write_fixture(
        &path,
        WORKBOOK_1900,
        STYLES_WITH_DATES,
        SHARED_EMPTY,
        &sheet,
    );

    let df = read_excel(&path, None, false, 0, None).unwrap();
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["2021-01-01".to_string()]
    );
}

#[test]
fn date1904_workbook_property_shifts_the_epoch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("date_1904.xlsx");
    let sheet = wrap_sheet(r#"<row r="1"><c r="A1" s="1"><v>1</v></c></row>"#);
    write_fixture(
        &path,
        WORKBOOK_1904,
        STYLES_WITH_DATES,
        SHARED_EMPTY,
        &sheet,
    );

    let df = read_excel(&path, None, false, 0, None).unwrap();
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["1904-01-02".to_string()],
        "date1904 workbooks must use the 1904-01-01 epoch, not the 1900 one"
    );
}

#[test]
fn unstyled_numeric_cell_stays_a_plain_number() {
    // A numeric cell with no date-format style must NOT be converted.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("plain_number.xlsx");
    let sheet = wrap_sheet(r#"<row r="1"><c r="A1"><v>45658</v></c></row>"#);
    write_fixture(
        &path,
        WORKBOOK_1900,
        STYLES_WITH_DATES,
        SHARED_EMPTY,
        &sheet,
    );

    let df = read_excel(&path, None, false, 0, None).unwrap();
    assert_eq!(
        df.get_column_string_values("Column1").unwrap(),
        vec!["45658".to_string()]
    );
}

// ---------------------------------------------------------------------
// Writer: NaN/Infinity, whole-number Float64 round-trip, control chars
// ---------------------------------------------------------------------

#[test]
fn nan_and_infinity_never_produce_invalid_numeric_xml() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nan_inf.xlsx");

    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "v",
        Column::Float64(Float64Column::new(vec![
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ])),
    )
    .unwrap();
    write_excel(&df, &path, Some("Data"), false).expect("writing NaN/Inf must not fail");

    let sheet_xml = read_archive_member(&path, "xl/worksheets/sheet1.xml");
    assert!(
        !sheet_xml.contains("NaN") && !sheet_xml.contains("Infinity"),
        "no cell may contain the literal text NaN/Infinity inside a numeric <v>: {sheet_xml}"
    );
    // Exact counts, not just "contains somewhere" — a version that silently
    // dropped the non-finite rows entirely (rather than emitting error
    // cells for them) would otherwise still pass a bare `.contains(...)`
    // check as long as *some* other test data produced a `t="e"` cell.
    let error_cell_count = sheet_xml.matches(r#"t="e""#).count();
    assert_eq!(
        error_cell_count, 3,
        "exactly the 3 non-finite values (NaN, +Inf, -Inf) must become error cells — not fewer \
         (silently dropped) or more: {sheet_xml}"
    );
    assert_eq!(
        sheet_xml.matches("#NUM!").count(),
        3,
        "each of the 3 error cells must carry the #NUM! marker: {sheet_xml}"
    );
    assert!(
        sheet_xml.contains("<v>1.0</v>"),
        "the one finite value must still be written as a proper float cell, not also turned into \
         an error cell: {sheet_xml}"
    );

    // The file must still be well-formed enough to read back without
    // crashing (values fall back to a String column since "#NUM!" isn't
    // numeric).
    let read_back = read_excel(&path, Some("Data"), true, 0, None).expect("must remain readable");
    assert_eq!(read_back.get_column_string_values("v").unwrap().len(), 4);
}

#[test]
fn all_whole_number_float64_column_round_trips_as_float64_not_int64() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("whole_floats.xlsx");

    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "score",
        Column::Float64(Float64Column::new(vec![10.0, 20.0, 30.0])),
    )
    .unwrap();
    df.to_excel(&path, Some("Data"), false).unwrap();

    let loaded = OptimizedDataFrame::from_excel(&path, Some("Data"), true, 0, None).unwrap();
    let view = loaded.column("score").unwrap();
    assert!(
        view.as_float64().is_some(),
        "an all-whole-number Float64 column must not collapse to Int64 on round-trip"
    );
}

#[test]
fn control_characters_are_stripped_from_written_strings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("control_chars.xlsx");

    let dirty = "abc\u{1}\u{0}def";
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "s",
        Column::String(StringColumn::new(vec![dirty.to_string()])),
    )
    .unwrap();
    write_excel(&df, &path, Some("Data"), false).expect("control chars must not break the writer");

    let shared = read_archive_member(&path, "xl/sharedStrings.xml");
    assert!(
        !shared.contains('\u{1}') && !shared.contains('\u{0}'),
        "XML-1.0-illegal control characters must be stripped from the output"
    );

    let df2 = read_excel(&path, Some("Data"), true, 0, None).unwrap();
    assert_eq!(
        df2.get_column_string_values("s").unwrap(),
        vec!["abcdef".to_string()]
    );
}

// ---------------------------------------------------------------------
// Writer: NA preservation across a full write/read round trip
// ---------------------------------------------------------------------

#[test]
fn blank_cell_round_trips_as_na_not_a_fabricated_zero() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("na_roundtrip.xlsx");

    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "n",
        Column::Int64(Int64Column::with_nulls(
            vec![1, 0, 3],
            vec![false, true, false],
        )),
    )
    .unwrap();
    df.to_excel(&path, Some("Data"), false).unwrap();

    let loaded = OptimizedDataFrame::from_excel(&path, Some("Data"), true, 0, None).unwrap();
    let view = loaded.column("n").unwrap();
    let int_col = view.as_int64().expect("column should still infer as Int64");
    assert_eq!(int_col.get(0).unwrap(), Some(1));
    assert_eq!(
        int_col.get(1).unwrap(),
        None,
        "the NA cell must read back as None, not Some(0)"
    );
    assert_eq!(int_col.get(2).unwrap(), Some(3));
}

// ---------------------------------------------------------------------
// Writer: sheet-name validation, atomic write, real index
// ---------------------------------------------------------------------

#[test]
fn duplicate_sheet_names_are_rejected_case_insensitively() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dup_sheets.xlsx");

    let mut df1 = OptimizedDataFrame::new();
    df1.add_column("x", Column::Int64(Int64Column::new(vec![1])))
        .unwrap();
    let mut df2 = OptimizedDataFrame::new();
    df2.add_column("y", Column::Int64(Int64Column::new(vec![2])))
        .unwrap();

    let mut sheets: HashMap<String, &OptimizedDataFrame> = HashMap::new();
    sheets.insert("Sheet1".to_string(), &df1);
    sheets.insert("sheet1".to_string(), &df2);

    let result = write_excel_sheets(&sheets, &path, false);
    assert!(
        result.is_err(),
        "\"Sheet1\" and \"sheet1\" collide under Excel's case-insensitive sheet-name rule"
    );
}

#[test]
fn write_excel_sheets_output_order_is_deterministic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("deterministic_order.xlsx");

    let mut dfs = Vec::new();
    for name in ["Zeta", "Alpha", "Mid"] {
        let mut df = OptimizedDataFrame::new();
        df.add_column("v", Column::Int64(Int64Column::new(vec![1])))
            .unwrap();
        dfs.push((name.to_string(), df));
    }
    let mut sheets: HashMap<String, &OptimizedDataFrame> = HashMap::new();
    for (name, df) in &dfs {
        sheets.insert(name.clone(), df);
    }

    write_excel_sheets(&sheets, &path, false).unwrap();
    let names = list_sheet_names(&path).unwrap();
    let mut expected = vec!["Alpha".to_string(), "Mid".to_string(), "Zeta".to_string()];
    expected.sort();
    assert_eq!(
        names, expected,
        "sheet order must be sorted, not HashMap-iteration order"
    );
}

#[test]
fn writing_does_not_leave_temp_files_behind() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("clean_write.xlsx");

    let mut df = OptimizedDataFrame::new();
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2])))
        .unwrap();
    write_excel(&df, &path, Some("Data"), false).unwrap();

    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        entries,
        vec![path.file_name().unwrap().to_string_lossy().into_owned()],
        "only the final file should remain — no leftover .tmp fixture: {entries:?}"
    );
}

#[test]
fn include_index_writes_the_real_index_not_a_positional_counter() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("real_index.xlsx");

    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "value",
        Column::Int64(Int64Column::new(vec![100, 200, 300])),
    )
    .unwrap();
    let idx = Index::<String>::new(vec![
        "row_a".to_string(),
        "row_b".to_string(),
        "row_c".to_string(),
    ])
    .unwrap();
    df.set_index_from_simple_index(idx).unwrap();

    write_excel(&df, &path, Some("Data"), true).expect("write with include_index");

    let with_index = read_excel(&path, Some("Data"), true, 0, None).unwrap();
    let index_values = with_index.get_column_string_values("Index").unwrap();
    assert_eq!(
        index_values,
        vec![
            "row_a".to_string(),
            "row_b".to_string(),
            "row_c".to_string()
        ],
        "the Index column must carry the real index labels, not 0/1/2"
    );
}

// ---------------------------------------------------------------------
// Facade: optimize_excel_file, enhanced-option errors, >26 columns
// ---------------------------------------------------------------------

#[test]
fn optimize_excel_file_round_trips_every_sheet_in_order() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("multi_source.xlsx");
    let dst = dir.path().join("multi_optimized.xlsx");

    let mut first = OptimizedDataFrame::new();
    first
        .add_column("a", Column::Int64(Int64Column::new(vec![1, 2])))
        .unwrap();
    let mut second = OptimizedDataFrame::new();
    second
        .add_column(
            "b",
            Column::String(StringColumn::new(vec!["x".into(), "y".into()])),
        )
        .unwrap();

    let mut sheets: HashMap<String, &OptimizedDataFrame> = HashMap::new();
    sheets.insert("First".to_string(), &first);
    sheets.insert("Second".to_string(), &second);
    write_excel_sheets(&sheets, &src, false).unwrap();
    assert_eq!(
        list_sheet_names(&src).unwrap().len(),
        2,
        "fixture must have 2 sheets"
    );

    optimize_excel_file(&src, &dst, 5).expect("optimize must succeed");

    let dst_sheets = list_sheet_names(&dst).unwrap();
    assert_eq!(
        dst_sheets.len(),
        2,
        "optimize_excel_file must round-trip every sheet, not just the first: {dst_sheets:?}"
    );

    let all = read_excel_sheets(&dst, None, true, 0, None).unwrap();
    assert_eq!(
        all.get("First")
            .unwrap()
            .get_column_string_values("a")
            .unwrap(),
        vec!["1".to_string(), "2".to_string()]
    );
    assert_eq!(
        all.get("Second")
            .unwrap()
            .get_column_string_values("b")
            .unwrap(),
        vec!["x".to_string(), "y".to_string()]
    );
}

#[test]
fn read_excel_enhanced_errors_on_unsupported_options_instead_of_ignoring_them() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("enhanced_read.xlsx");
    let mut df = OptimizedDataFrame::new();
    df.add_column("v", Column::Int64(Int64Column::new(vec![1])))
        .unwrap();
    write_excel(&df, &path, Some("Data"), false).unwrap();

    // Default options: must still succeed.
    let default_result = read_excel_enhanced(&path, Some("Data"), ExcelReadOptions::default());
    assert!(default_result.is_ok(), "default options must not error");

    let opts = ExcelReadOptions {
        read_named_ranges: true,
        ..Default::default()
    };
    let result = read_excel_enhanced(&path, Some("Data"), opts);
    match result {
        Err(Error::NotImplemented(msg)) => {
            assert!(
                msg.contains("read_named_ranges"),
                "error must name the unsupported option: {msg}"
            );
        }
        other => panic!("expected Err(NotImplemented), got {other:?}"),
    }
}

#[test]
fn write_excel_enhanced_errors_on_protect_sheets_instead_of_silently_ignoring_it() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("enhanced_write.xlsx");
    let mut df = OptimizedDataFrame::new();
    df.add_column("v", Column::Int64(Int64Column::new(vec![1])))
        .unwrap();

    // Default options: must still succeed.
    let default_result = write_excel_enhanced(
        &df,
        &path,
        Some("Data"),
        &[] as &[ExcelCell],
        &[] as &[NamedRange],
        ExcelWriteOptions::default(),
    );
    assert!(default_result.is_ok(), "default options must not error");

    let opts = ExcelWriteOptions {
        protect_sheets: true,
        ..Default::default()
    };
    let result = write_excel_enhanced(
        &df,
        &path,
        Some("Data"),
        &[] as &[ExcelCell],
        &[] as &[NamedRange],
        opts,
    );
    match result {
        Err(Error::NotImplemented(msg)) => {
            assert!(
                msg.contains("protect_sheets"),
                "a protection request that can't be honored must error, not silently succeed: {msg}"
            );
        }
        other => panic!("expected Err(NotImplemented), got {other:?}"),
    }

    // Passing formula/formatting cell data that would be silently dropped
    // must also error.
    let cell = ExcelCell {
        value: "1".to_string(),
        formula: Some("=A1+1".to_string()),
        data_type: "formula".to_string(),
        format: ExcelCellFormat::default(),
    };
    let result2 = write_excel_enhanced(
        &df,
        &path,
        Some("Data"),
        &[cell],
        &[] as &[NamedRange],
        ExcelWriteOptions::default(),
    );
    assert!(
        result2.is_err(),
        "non-empty ExcelCell data that would be dropped must error"
    );
}

#[test]
fn sheet_range_uses_correct_letters_past_column_z() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wide_sheet.xlsx");

    let mut df = OptimizedDataFrame::new();
    for i in 0..30 {
        df.add_column(format!("c{i}"), Column::Int64(Int64Column::new(vec![1])))
            .unwrap();
    }
    write_excel(&df, &path, Some("Wide"), false).unwrap();

    let info = pandrs::io::get_sheet_info(&path, "Wide").unwrap();
    // 30 columns -> last column is index 29 (0-based) -> "AD" under the
    // standard bijective base-26 letters (A..Z, AA..AZ, BA.., ..., AD is the
    // 30th letter sequence). Cross-checked independently below rather than
    // hardcoded from memory. `rows` covers the header row plus the one data
    // row this fixture wrote, i.e. 2.
    assert_eq!(info.columns, 30);
    assert_eq!(info.rows, 2);
    assert_eq!(expected_column_letters(29), "AD");
    let expected_range = format!("A1:{}{}", expected_column_letters(29), info.rows);
    assert_eq!(
        info.range, expected_range,
        "range must use the real multi-letter column, not a clamped 'Z'"
    );
}

/// Independent (test-local) implementation of the same bijective base-26
/// column-letter encoding, so the assertion above isn't just re-asserting
/// whatever the production code happens to compute.
fn expected_column_letters(mut col: usize) -> String {
    let mut out = Vec::new();
    loop {
        let rem = (col % 26) as u8;
        out.push(b'A' + rem);
        if col < 26 {
            break;
        }
        col = col / 26 - 1;
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}

// ---------------------------------------------------------------------
// Combined regression: Japanese strings + NA + float-whole values in one
// multi-sheet round trip through the public write/read facade.
// ---------------------------------------------------------------------

#[test]
#[allow(clippy::result_large_err)]
fn multi_sheet_round_trip_with_japanese_strings_na_and_whole_floats() -> Result<()> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("combined.xlsx");

    let mut sales = OptimizedDataFrame::new();
    sales.add_column(
        "customer",
        Column::String(StringColumn::new(vec![
            "田中太郎".to_string(),
            "鈴木花子".to_string(),
        ])),
    )?;
    sales.add_column(
        "amount",
        Column::Float64(Float64Column::with_nulls(
            vec![100.0, 0.0],
            vec![false, true],
        )),
    )?;
    sales.add_column(
        "active",
        Column::Boolean(BooleanColumn::new(vec![true, false])),
    )?;

    let mut totals = OptimizedDataFrame::new();
    totals.add_column("year", Column::Int64(Int64Column::new(vec![2024, 2025])))?;
    totals.add_column(
        "revenue",
        Column::Float64(Float64Column::new(vec![1000.0, 2000.0])),
    )?;

    let mut sheets: HashMap<String, &OptimizedDataFrame> = HashMap::new();
    sheets.insert("Sales".to_string(), &sales);
    sheets.insert("Totals".to_string(), &totals);
    write_excel_sheets(&sheets, &path, false)?;

    let names = list_sheet_names(&path)?;
    assert_eq!(names.len(), 2);

    let loaded_sales = OptimizedDataFrame::from_excel(&path, Some("Sales"), true, 0, None)?;
    let customer = loaded_sales.column("customer")?;
    let customer_col = customer.as_string().expect("customer should be String");
    assert_eq!(customer_col.get(0)?, Some("田中太郎"));
    assert_eq!(customer_col.get(1)?, Some("鈴木花子"));

    let amount = loaded_sales.column("amount")?;
    let amount_col = amount.as_float64().expect("amount should be Float64");
    assert_eq!(amount_col.get(0)?, Some(100.0));
    assert_eq!(
        amount_col.get(1)?,
        None,
        "NA amount must stay NA, not become 0.0"
    );

    let loaded_totals = OptimizedDataFrame::from_excel(&path, Some("Totals"), true, 0, None)?;
    let revenue = loaded_totals.column("revenue")?;
    assert!(
        revenue.as_float64().is_some(),
        "an all-whole-number revenue column must remain Float64"
    );

    Ok(())
}

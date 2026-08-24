//! Regression tests for the CSV/JSON I/O fabrication fixes in
//! `src/io/csv.rs` and `src/io/json.rs`.
//!
//! Before this fix: `write_csv`/`write_json` wrote a header row and then a
//! `Writer`/`Value` object filled entirely with blank/empty cells,
//! regardless of what the DataFrame actually contained (`io_audit.md`
//! section 2/CSV, `stub_audit.md` P0 #1). `read_json` used
//! `serde_json::Value::to_string()` on every cell, so a JSON string value
//! kept its surrounding quote characters and a JSON `null` became the
//! literal 4-character text `null`. Headerless CSV files silently dropped
//! their first data row. `Trim::All` stripped whitespace from already
//! dequoted, originally-quoted field content. No file handled a UTF-8
//! byte-order mark. A CSV row with the wrong number of fields was either
//! silently truncated (extra fields) or zero-padded (missing fields) with
//! no error.
//!
//! Every test here asserts actual *values*, not just row/column counts --
//! the previous test suite passed with all-blank writers because it only
//! ever checked shape.

use pandrs::io::csv::{read_csv, read_csv_typed, write_csv};
use pandrs::io::json::{read_json, write_json, JsonOrient};
use pandrs::{DataFrame, Series};
use std::fs;

mod common;
use common::test_utils::test_temp_path;

fn string_series(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| v.to_string()).collect()
}

// ---------------------------------------------------------------------
// CSV: write_csv writes real cell values (fix 1)
// ---------------------------------------------------------------------

#[test]
fn csv_write_csv_writes_real_values_not_blanks() {
    let path = test_temp_path("csv_write_real", "csv");

    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(string_series(&["Alice", "Bob", "Charlie"]), None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "age".to_string(),
        Series::new(vec![30i64, 25, 35], None).unwrap(),
    )
    .unwrap();

    write_csv(&df, &path).unwrap();

    let content = fs::read_to_string(&path).unwrap();
    // The old stub wrote a header row and then only commas: "," / ",\n" etc.
    // Assert the real values are present in the file text.
    assert!(content.contains("Alice"), "content was: {content:?}");
    assert!(content.contains("Bob"), "content was: {content:?}");
    assert!(content.contains("Charlie"), "content was: {content:?}");
    assert!(content.contains("30"), "content was: {content:?}");
    assert!(content.contains("25"), "content was: {content:?}");
    assert!(content.contains("35"), "content was: {content:?}");

    // And round-trip through the reader to check exact per-cell values.
    let reloaded = read_csv(&path, true).unwrap();
    assert_eq!(reloaded.row_count(), 3);
    assert_eq!(
        reloaded.get_column_string_values("name").unwrap(),
        vec!["Alice", "Bob", "Charlie"]
    );
    assert_eq!(
        reloaded.get_column_string_values("age").unwrap(),
        vec!["30", "25", "35"]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_write_csv_invalid_path_propagates_error() {
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), Series::new(vec![1i64], None).unwrap())
        .unwrap();

    let result = write_csv(&df, "/nonexistent_directory_for_pandrs_test/out.csv");
    assert!(result.is_err(), "writing to a bad path must not succeed");
}

// ---------------------------------------------------------------------
// CSV: string values with commas, quotes, newlines, and unicode (item 10)
// ---------------------------------------------------------------------

#[test]
fn csv_round_trip_strings_with_commas_quotes_newlines_unicode() {
    let path = test_temp_path("csv_special_strings", "csv");

    let raw_values = [
        "plain",
        "has,a,comma",
        "has \"embedded\" quotes",
        "has\nan embedded newline",
        "日本語のテスト 🎌",
    ];

    let mut df = DataFrame::new();
    df.add_column(
        "text".to_string(),
        Series::new(string_series(&raw_values), None).unwrap(),
    )
    .unwrap();

    write_csv(&df, &path).unwrap();
    let reloaded = read_csv(&path, true).unwrap();

    assert_eq!(reloaded.row_count(), raw_values.len());
    let values = reloaded.get_column_string_values("text").unwrap();
    for (expected, actual) in raw_values.iter().zip(values.iter()) {
        assert_eq!(actual, expected, "value did not round-trip exactly");
    }

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: empty / NA cells preserved, not fabricated (fix throughout)
// ---------------------------------------------------------------------

#[test]
fn csv_empty_cells_preserved_as_empty_not_fabricated() {
    let path = test_temp_path("csv_empty_cells", "csv");
    fs::write(&path, "name,note\nAlice,hello\nBob,\n,world\n").unwrap();

    let df = read_csv(&path, true).unwrap();
    assert_eq!(df.row_count(), 3);

    let notes = df.get_column_string_values("note").unwrap();
    assert_eq!(notes, vec!["hello", "", "world"]);

    let names = df.get_column_string_values("name").unwrap();
    assert_eq!(names, vec!["Alice", "Bob", ""]);

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: headerless files keep their first data row (fix 5)
// ---------------------------------------------------------------------

#[test]
fn csv_headerless_file_preserves_first_data_row() {
    let path = test_temp_path("csv_headerless", "csv");
    fs::write(&path, "10,alpha\n20,beta\n30,gamma\n").unwrap();

    let df = read_csv(&path, false).unwrap();

    // Before the fix, the first row ("10,alpha") was consumed while
    // counting columns and never made it into the DataFrame, so row_count
    // would be 2, not 3.
    assert_eq!(df.row_count(), 3, "the first data row must not be dropped");
    assert!(df.contains_column("column_0"));
    assert!(df.contains_column("column_1"));

    let col0 = df.get_column_string_values("column_0").unwrap();
    assert_eq!(col0, vec!["10", "20", "30"]);
    let col1 = df.get_column_string_values("column_1").unwrap();
    assert_eq!(col1, vec!["alpha", "beta", "gamma"]);

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: ragged rows are a descriptive error, not silent data loss (fix 8)
// ---------------------------------------------------------------------

#[test]
fn csv_extra_fields_in_a_row_error_instead_of_silently_dropping() {
    let path = test_temp_path("csv_extra_fields", "csv");
    fs::write(&path, "a,b\n1,2\n3,4,5\n").unwrap();

    let result = read_csv(&path, true);
    let err = result.expect_err("a row with an extra field must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("column"),
        "error should describe the mismatch: {message}"
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_short_row_errors_instead_of_silently_padding() {
    let path = test_temp_path("csv_short_row", "csv");
    fs::write(&path, "a,b,c\n1,2,3\n4,5\n").unwrap();

    let result = read_csv(&path, true);
    assert!(
        result.is_err(),
        "a row with a missing field must be rejected, not zero-padded"
    );

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: Trim::Headers preserves intentional whitespace in field values,
// but still trims header names (fix 6)
// ---------------------------------------------------------------------

#[test]
fn csv_field_whitespace_preserved_but_headers_trimmed() {
    let path = test_temp_path("csv_trim", "csv");
    // Header has stray whitespace; the quoted data field has intentional
    // leading/trailing spaces that must survive exactly.
    fs::write(&path, " id , note\n1,\"  padded  \"\n").unwrap();

    let df = read_csv(&path, true).unwrap();
    assert!(
        df.contains_column("id"),
        "header whitespace should be trimmed: {:?}",
        df.column_names()
    );
    assert!(df.contains_column("note"));

    let notes = df.get_column_string_values("note").unwrap();
    assert_eq!(
        notes[0], "  padded  ",
        "quoted field content must not be trimmed"
    );

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: UTF-8 BOM stripped (fix 7)
// ---------------------------------------------------------------------

#[test]
fn csv_utf8_bom_is_stripped_from_first_header() {
    let path = test_temp_path("csv_bom", "csv");
    let mut bytes = vec![0xEFu8, 0xBB, 0xBF];
    bytes.extend_from_slice(b"id,value\n1,2\n");
    fs::write(&path, bytes).unwrap();

    let df = read_csv(&path, true).unwrap();
    assert!(
        df.contains_column("id"),
        "BOM must not leak into the first column name: {:?}",
        df.column_names()
    );
    assert!(!df.contains_column("\u{feff}id"));

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// CSV: read_csv_typed infers numeric/boolean columns (fix 9)
// ---------------------------------------------------------------------

#[test]
fn csv_read_csv_typed_infers_int_float_bool_columns() {
    let path = test_temp_path("csv_typed_infer", "csv");
    fs::write(
        &path,
        "id,ratio,active,label\n1,1.5,true,x\n2,2.5,false,y\n3,3.5,true,z\n",
    )
    .unwrap();

    let df = read_csv_typed(&path, true).unwrap();

    let ids = df.get_column::<i64>("id").unwrap();
    assert_eq!(ids.values(), &[1i64, 2, 3]);

    let ratios = df.get_column::<f64>("ratio").unwrap();
    assert_eq!(ratios.values(), &[1.5f64, 2.5, 3.5]);

    let active = df.get_column::<bool>("active").unwrap();
    assert_eq!(active.values(), &[true, false, true]);

    // Non-numeric column stays String.
    let labels = df.get_column::<String>("label").unwrap();
    assert_eq!(
        labels.values(),
        &["x".to_string(), "y".to_string(), "z".to_string()]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_read_csv_typed_missing_int_cell_upcasts_to_float_nan() {
    let path = test_temp_path("csv_typed_missing_int", "csv");
    // "count" would be all-integer except for one missing cell -- Int64
    // cannot represent "missing", so this must become Float64 with NaN,
    // never a fabricated 0.
    fs::write(&path, "id,count\n1,10\n2,\n3,30\n").unwrap();

    let df = read_csv_typed(&path, true).unwrap();

    let ids = df.get_column::<i64>("id").unwrap();
    assert_eq!(ids.values(), &[1i64, 2, 3]);

    let counts = df.get_column::<f64>("count").unwrap();
    assert_eq!(counts.values()[0], 10.0);
    assert!(
        counts.values()[1].is_nan(),
        "missing numeric cell must become NaN, not 0.0: {:?}",
        counts.values()
    );
    assert_eq!(counts.values()[2], 30.0);

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_read_csv_typed_headerless_infers_synthetic_columns() {
    // The headerless path shares `read_csv_raw` with `read_csv`, so the
    // first-row-preservation fix carries over, but the inference logic
    // running against synthetic `column_0`/`column_1` names -- rather than
    // real header text -- is its own combination and deserves direct
    // coverage.
    let path = test_temp_path("csv_typed_headerless", "csv");
    fs::write(&path, "1,alpha\n2,beta\n3,gamma\n").unwrap();

    let df = read_csv_typed(&path, false).unwrap();
    assert_eq!(df.row_count(), 3, "the first data row must not be dropped");

    let col0 = df.get_column::<i64>("column_0").unwrap();
    assert_eq!(col0.values(), &[1i64, 2, 3]);

    let col1 = df.get_column::<String>("column_1").unwrap();
    assert_eq!(
        col1.values(),
        &["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_read_csv_typed_missing_bool_cell_falls_back_to_string() {
    let path = test_temp_path("csv_typed_missing_bool", "csv");
    // "flag" looks boolean except for one missing cell -- `bool` cannot
    // represent "missing" either, so the whole column must fall back to
    // String rather than fabricate `false` for the blank cell.
    fs::write(&path, "id,flag\n1,true\n2,\n3,false\n").unwrap();

    let df = read_csv_typed(&path, true).unwrap();
    assert!(df.get_column::<bool>("flag").is_err());
    let flags = df.get_column::<String>("flag").unwrap();
    assert_eq!(
        flags.values(),
        &["true".to_string(), "".to_string(), "false".to_string()]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn csv_read_csv_typed_round_trip_preserves_numeric_values() {
    let path = test_temp_path("csv_typed_roundtrip", "csv");
    fs::write(&path, "id,ratio\n1,1.5\n2,\n3,3.5\n").unwrap();

    let df = read_csv_typed(&path, true).unwrap();
    write_csv(&df, &path).unwrap();
    let reloaded = read_csv_typed(&path, true).unwrap();

    let ratios = reloaded.get_column::<f64>("ratio").unwrap();
    assert_eq!(ratios.values()[0], 1.5);
    assert!(ratios.values()[1].is_nan());
    assert_eq!(ratios.values()[2], 3.5);

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// JSON: write_json emits real values with correct native JSON types
// (fix 2), verified against the raw JSON structure (not substring checks)
// ---------------------------------------------------------------------

#[test]
fn json_write_records_uses_real_values_and_native_types() {
    let path = test_temp_path("json_write_records_typed", "json");

    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "ratio".to_string(),
        Series::new(vec![1.5f64, 2.5, 3.5], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "active".to_string(),
        Series::new(vec![true, false, true], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "label".to_string(),
        Series::new(string_series(&["a,b", "has \"quotes\"", "plain"]), None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Records).unwrap();

    let raw = fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let records = parsed.as_array().expect("records must be a JSON array");
    assert_eq!(records.len(), 3);

    assert_eq!(records[0]["id"], serde_json::json!(1));
    assert_eq!(records[1]["id"], serde_json::json!(2));
    assert!(records[0]["id"].is_number());

    assert_eq!(records[0]["ratio"], serde_json::json!(1.5));
    assert!(records[0]["ratio"].is_number());

    assert_eq!(records[0]["active"], serde_json::json!(true));
    assert!(records[0]["active"].is_boolean());
    assert_eq!(records[1]["active"], serde_json::json!(false));

    // The string must be the raw text, with no extra JSON-escaping quotes
    // wrapped around it by a stray `Value::to_string()`.
    assert_eq!(records[0]["label"], serde_json::json!("a,b"));
    assert_eq!(records[1]["label"], serde_json::json!("has \"quotes\""));
    assert!(records[0]["label"].is_string());

    let _ = fs::remove_file(&path);
}

#[test]
fn json_write_columns_uses_real_values_and_native_types() {
    let path = test_temp_path("json_write_columns_typed", "json");

    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![10i64, 20], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "name".to_string(),
        Series::new(string_series(&["Alice", "日本語"]), None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Columns).unwrap();

    let raw = fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let obj = parsed.as_object().expect("columns must be a JSON object");

    let id_array = obj["id"].as_array().expect("id column must be an array");
    assert_eq!(
        id_array,
        &vec![serde_json::json!(10), serde_json::json!(20)]
    );
    assert!(id_array.iter().all(|v| v.is_number()));

    let name_array = obj["name"]
        .as_array()
        .expect("name column must be an array");
    assert_eq!(name_array[0], serde_json::json!("Alice"));
    assert_eq!(name_array[1], serde_json::json!("日本語"));

    let _ = fs::remove_file(&path);
}

#[test]
fn json_write_string_literal_column_writes_real_text() {
    // `Series<&'static str>` (built directly from string literals) is a
    // natural, commonly-constructed column type distinct from
    // `Series<String>`.
    let path = test_temp_path("json_write_str_literal", "json");

    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(vec!["Alice", "Bob"], None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Records).unwrap();

    let raw = fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let records = parsed.as_array().unwrap();
    assert_eq!(records[0]["name"], serde_json::json!("Alice"));
    assert_eq!(records[1]["name"], serde_json::json!("Bob"));

    let _ = fs::remove_file(&path);
}

#[test]
fn json_write_non_finite_float_becomes_null() {
    let path = test_temp_path("json_write_nan", "json");

    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        Series::new(vec![1.0f64, f64::NAN, f64::INFINITY], None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Records).unwrap();

    let raw = fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let records = parsed.as_array().unwrap();

    assert_eq!(records[0]["value"], serde_json::json!(1.0));
    assert!(
        records[1]["value"].is_null(),
        "NaN must become JSON null, not a fabricated number"
    );
    assert!(
        records[2]["value"].is_null(),
        "Infinity must become JSON null"
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn json_write_json_invalid_path_propagates_error() {
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), Series::new(vec![1i64], None).unwrap())
        .unwrap();

    let result = write_json(
        &df,
        "/nonexistent_directory_for_pandrs_test/out.json",
        JsonOrient::Records,
    );
    assert!(result.is_err(), "writing to a bad path must not succeed");
}

// ---------------------------------------------------------------------
// JSON: read_json fixes the Value::to_string() quoting bug (fix 3)
// ---------------------------------------------------------------------

#[test]
fn json_read_json_strings_have_no_extra_quotes() {
    let path = test_temp_path("json_read_no_quotes", "json");
    let content = r#"[
        {"name": "Alice", "note": "has \"quotes\" and, a comma"},
        {"name": "日本語", "note": "line1\nline2"}
    ]"#;
    fs::write(&path, content).unwrap();

    let df = read_json(&path).unwrap();
    let names = df.get_column_string_values("name").unwrap();
    // Before the fix this would be the 7-character text `"Alice"`
    // (quote marks included) instead of the 5-character `Alice`.
    assert_eq!(names[0], "Alice");
    assert_eq!(names[1], "日本語");

    let notes = df.get_column_string_values("note").unwrap();
    assert_eq!(notes[0], "has \"quotes\" and, a comma");
    assert_eq!(notes[1], "line1\nline2");

    let _ = fs::remove_file(&path);
}

#[test]
fn json_read_json_null_becomes_empty_string_not_the_word_null() {
    let path = test_temp_path("json_read_null", "json");
    let content = r#"[
        {"name": "Alice", "email": null},
        {"name": "Bob", "email": "bob@example.com"}
    ]"#;
    fs::write(&path, content).unwrap();

    let df = read_json(&path).unwrap();
    let emails = df.get_column_string_values("email").unwrap();
    // Before the fix `Value::Null.to_string()` produced the literal text
    // "null" (4 real characters), which could be mistaken for genuine
    // data instead of a missing-value marker.
    assert_eq!(emails[0], "");
    assert_eq!(emails[1], "bob@example.com");

    let _ = fs::remove_file(&path);
}

#[test]
fn json_read_json_numbers_and_booleans_have_plain_text() {
    let path = test_temp_path("json_read_numbers_bools", "json");
    let content = r#"[
        {"age": 30, "active": true},
        {"age": 25, "active": false}
    ]"#;
    fs::write(&path, content).unwrap();

    let df = read_json(&path).unwrap();
    assert_eq!(
        df.get_column_string_values("age").unwrap(),
        vec!["30", "25"]
    );
    assert_eq!(
        df.get_column_string_values("active").unwrap(),
        vec!["true", "false"]
    );

    let _ = fs::remove_file(&path);
}

// ---------------------------------------------------------------------
// JSON: full write -> read value-level round trip
// ---------------------------------------------------------------------

#[test]
fn json_round_trip_records_preserves_text_values() {
    let path = test_temp_path("json_roundtrip_values", "json");

    let mut df = DataFrame::new();
    df.add_column("id".to_string(), Series::new(vec![1i64, 2], None).unwrap())
        .unwrap();
    df.add_column(
        "name".to_string(),
        Series::new(string_series(&["a,b", "unicode: 🎌"]), None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Records).unwrap();
    let reloaded = read_json(&path).unwrap();

    assert_eq!(reloaded.row_count(), 2);
    // `read_json` always returns String columns (documented, unchanged
    // behaviour), so the numeric column comes back as its text form.
    assert_eq!(
        reloaded.get_column_string_values("id").unwrap(),
        vec!["1", "2"]
    );
    assert_eq!(
        reloaded.get_column_string_values("name").unwrap(),
        vec!["a,b", "unicode: 🎌"]
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn json_round_trip_columns_orientation() {
    let path = test_temp_path("json_roundtrip_columns", "json");

    let mut df = DataFrame::new();
    df.add_column(
        "flag".to_string(),
        Series::new(vec![true, false, true], None).unwrap(),
    )
    .unwrap();

    write_json(&df, &path, JsonOrient::Columns).unwrap();
    let reloaded = read_json(&path).unwrap();

    assert_eq!(
        reloaded.get_column_string_values("flag").unwrap(),
        vec!["true", "false", "true"]
    );

    let _ = fs::remove_file(&path);
}

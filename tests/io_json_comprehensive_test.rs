#![allow(clippy::result_large_err)]
//! Comprehensive JSON I/O Tests
//!
//! Tests for JSON read/write operations covering edge cases and error conditions

use pandrs::io::json::{read_json, write_json, JsonOrient};
use pandrs::{DataFrame, PandRSError, Series};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

mod common;
use common::test_utils::{get_temp_dir, test_temp_path};

/// Get temp file path with automatic counter
fn temp_json_path(test_name: &str) -> PathBuf {
    test_temp_path(test_name, "json")
}

/// Test reading empty JSON array
#[test]
fn test_read_json_empty_array() -> Result<(), PandRSError> {
    let path = temp_json_path("empty_array");

    // Create empty JSON array
    fs::write(&path, "[]").map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 0);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading empty JSON object
#[test]
fn test_read_json_empty_object() -> Result<(), PandRSError> {
    let path = temp_json_path("empty_object");

    // Create empty JSON object
    fs::write(&path, "{}").map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 0);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading records-oriented JSON
#[test]
fn test_read_json_records_basic() -> Result<(), PandRSError> {
    let path = temp_json_path("records_basic");

    let json_content = r#"[
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "SF"},
        {"name": "Charlie", "age": 35, "city": "LA"}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 3);
    assert!(df.contains_column("name"));
    assert!(df.contains_column("age"));
    assert!(df.contains_column("city"));

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading column-oriented JSON
#[test]
fn test_read_json_columns_basic() -> Result<(), PandRSError> {
    let path = temp_json_path("columns_basic");

    let json_content = r#"{
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "city": ["NYC", "SF", "LA"]
    }"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 3);
    assert!(df.contains_column("name"));
    assert!(df.contains_column("age"));
    assert!(df.contains_column("city"));

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading JSON with missing fields (ragged data)
#[test]
fn test_read_json_ragged_records() -> Result<(), PandRSError> {
    let path = temp_json_path("ragged_records");

    let json_content = r#"[
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "city": "SF"},
        {"name": "Charlie", "age": 35}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 3);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading JSON with unicode characters
#[test]
fn test_read_json_unicode() -> Result<(), PandRSError> {
    let path = temp_json_path("unicode");

    let json_content = r#"[
        {"name": "日本", "emoji": "🎌", "text": "こんにちは"},
        {"name": "中国", "emoji": "🇨🇳", "text": "你好"},
        {"name": "한국", "emoji": "🇰🇷", "text": "안녕하세요"}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 3);
    assert!(df.contains_column("name"));
    assert!(df.contains_column("emoji"));

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test reading JSON with special characters
#[test]
fn test_read_json_special_chars() -> Result<(), PandRSError> {
    let path = temp_json_path("special_chars");

    let json_content = r#"[
        {"text": "line1\nline2", "quote": "\"quoted\"", "backslash": "C:\\path"},
        {"text": "tab\there", "quote": "'single'", "backslash": "/unix/path"}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 2);
    assert_eq!(df.column_count(), 3);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test error: invalid JSON syntax
#[test]
fn test_read_json_invalid_syntax() {
    let path = temp_json_path("invalid_syntax");

    // Invalid JSON (trailing comma)
    let _ = fs::write(&path, r#"{"name": "Alice",}"#);

    let result = read_json(&path);
    assert!(result.is_err());

    // Cleanup
    let _ = fs::remove_file(&path);
}

/// Test error: file not found
#[test]
fn test_read_json_file_not_found() {
    let path = get_temp_dir().join("nonexistent_file_12345.json");

    let result = read_json(&path);
    assert!(result.is_err());
}

/// Test error: invalid JSON type (primitive)
#[test]
fn test_read_json_invalid_type_string() {
    let path = temp_json_path("invalid_type_string");

    let _ = fs::write(&path, r#""just a string""#);

    let result = read_json(&path);
    assert!(result.is_err());

    // Cleanup
    let _ = fs::remove_file(&path);
}

/// Test error: invalid JSON type (number)
#[test]
fn test_read_json_invalid_type_number() {
    let path = temp_json_path("invalid_type_number");

    let _ = fs::write(&path, "42");

    let result = read_json(&path);
    assert!(result.is_err());

    // Cleanup
    let _ = fs::remove_file(&path);
}

/// Test error: array with non-object elements
#[test]
fn test_read_json_array_non_objects() {
    let path = temp_json_path("array_non_objects");

    let json_content = r#"[
        {"name": "Alice"},
        "invalid_string",
        {"name": "Bob"}
    ]"#;

    let _ = fs::write(&path, json_content);

    let result = read_json(&path);
    assert!(result.is_err());

    // Cleanup
    let _ = fs::remove_file(&path);
}

/// Test error: column-oriented with non-array values
#[test]
fn test_read_json_columns_non_arrays() {
    let path = temp_json_path("columns_non_arrays");

    let json_content = r#"{
        "name": ["Alice", "Bob"],
        "age": "not_an_array"
    }"#;

    let _ = fs::write(&path, json_content);

    let result = read_json(&path);
    assert!(result.is_err());

    // Cleanup
    let _ = fs::remove_file(&path);
}

/// Test writing JSON in records format
#[test]
fn test_write_json_records() -> Result<(), PandRSError> {
    let path = temp_json_path("write_records");

    // Create test DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(vec!["Alice", "Bob"], Some("name".to_string()))?,
    )?;
    df.add_column(
        "age".to_string(),
        Series::new(vec![30, 25], Some("age".to_string()))?,
    )?;

    // Write to JSON
    write_json(&df, &path, JsonOrient::Records)?;

    // Verify file was created
    assert!(path.exists());

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test writing JSON in columns format
#[test]
fn test_write_json_columns() -> Result<(), PandRSError> {
    let path = temp_json_path("write_columns");

    // Create test DataFrame
    let mut df = DataFrame::new();
    df.add_column(
        "name".to_string(),
        Series::new(vec!["Alice", "Bob"], Some("name".to_string()))?,
    )?;
    df.add_column(
        "age".to_string(),
        Series::new(vec![30, 25], Some("age".to_string()))?,
    )?;

    // Write to JSON
    write_json(&df, &path, JsonOrient::Columns)?;

    // Verify file was created
    assert!(path.exists());

    // Read back and verify
    let content = fs::read_to_string(&path).map_err(PandRSError::Io)?;
    assert!(content.contains("name") && content.contains("age"));

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test writing empty DataFrame to JSON
#[test]
fn test_write_json_empty_dataframe() -> Result<(), PandRSError> {
    let path = temp_json_path("write_empty");

    let df = DataFrame::new();

    // Write to JSON (records format)
    write_json(&df, &path, JsonOrient::Records)?;

    // Verify file was created
    assert!(path.exists());

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON round-trip (write and read back)
#[test]
fn test_json_roundtrip_records() -> Result<(), PandRSError> {
    let path = temp_json_path("roundtrip_records");

    // Create test DataFrame
    let mut df_original = DataFrame::new();
    df_original.add_column(
        "name".to_string(),
        Series::new(vec!["Alice", "Bob", "Charlie"], Some("name".to_string()))?,
    )?;
    df_original.add_column(
        "age".to_string(),
        Series::new(vec![30, 25, 35], Some("age".to_string()))?,
    )?;

    // Write to JSON
    write_json(&df_original, &path, JsonOrient::Records)?;

    // Read back
    let df_loaded = read_json(&path)?;

    // Verify structure
    assert_eq!(df_loaded.row_count(), df_original.row_count());
    assert_eq!(df_loaded.column_count(), df_original.column_count());

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON with large dataset
#[test]
fn test_json_large_dataset() -> Result<(), PandRSError> {
    let path = temp_json_path("large_dataset");

    // Create large JSON file
    let mut content = String::from("[");
    for i in 0..1000 {
        if i > 0 {
            content.push(',');
        }
        content.push_str(&format!(
            r#"{{"id": {}, "value": "item_{}", "score": {}}}"#,
            i,
            i,
            i * 10
        ));
    }
    content.push(']');

    fs::write(&path, content).map_err(PandRSError::Io)?;

    // Read large file
    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 1000);
    assert_eq!(df.column_count(), 3);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON with nested structures (should handle gracefully)
#[test]
fn test_json_nested_objects() -> Result<(), PandRSError> {
    let path = temp_json_path("nested");

    let json_content = r#"[
        {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
        {"name": "Bob", "address": {"city": "SF", "zip": "94102"}}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    // Should read without error (nested objects converted to strings)
    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 2);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON with null values
#[test]
fn test_json_null_values() -> Result<(), PandRSError> {
    let path = temp_json_path("nulls");

    let json_content = r#"[
        {"name": "Alice", "age": 30, "email": null},
        {"name": "Bob", "age": null, "email": "bob@example.com"},
        {"name": null, "age": 35, "email": null}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 3);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON with boolean values
#[test]
fn test_json_boolean_values() -> Result<(), PandRSError> {
    let path = temp_json_path("booleans");

    let json_content = r#"[
        {"name": "Alice", "active": true, "verified": false},
        {"name": "Bob", "active": false, "verified": true}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 2);
    assert!(df.contains_column("active"));
    assert!(df.contains_column("verified"));

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

/// Test JSON with numeric edge cases
#[test]
fn test_json_numeric_edge_cases() -> Result<(), PandRSError> {
    let path = temp_json_path("numeric_edges");

    let json_content = r#"[
        {"value": 0, "float": 0.0, "negative": -999},
        {"value": 9223372036854775807, "float": 1.7976931348623157e308, "negative": -9223372036854775808}
    ]"#;

    fs::write(&path, json_content).map_err(PandRSError::Io)?;

    let df = read_json(&path)?;
    assert_eq!(df.row_count(), 2);

    // Cleanup
    let _ = fs::remove_file(&path);
    Ok(())
}

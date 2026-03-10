#![allow(clippy::result_large_err)]
//! Comprehensive Edge Cases Tests for DataFrame and Series
//!
//! Tests covering empty data, extreme values, unicode, and special conditions

use pandrs::{DataFrame, PandRSError, Series};
use std::f64;

/// Test empty DataFrame creation and operations
#[test]
fn test_empty_dataframe() -> Result<(), PandRSError> {
    let df = DataFrame::new();

    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 0);
    assert!(df.column_names().is_empty());

    Ok(())
}

/// Test DataFrame with single element
#[test]
fn test_single_element_dataframe() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let series = Series::new(vec!["single"], Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    assert_eq!(df.row_count(), 1);
    assert_eq!(df.column_count(), 1);

    Ok(())
}

/// Test DataFrame with single row, multiple columns
#[test]
fn test_single_row_multiple_columns() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    df.add_column(
        "col1".to_string(),
        Series::new(vec!["a"], Some("col1".to_string()))?,
    )?;
    df.add_column(
        "col2".to_string(),
        Series::new(vec![1], Some("col2".to_string()))?,
    )?;
    df.add_column(
        "col3".to_string(),
        Series::new(vec![true], Some("col3".to_string()))?,
    )?;

    assert_eq!(df.row_count(), 1);
    assert_eq!(df.column_count(), 3);

    Ok(())
}

/// Test DataFrame with single column, multiple rows
#[test]
fn test_single_column_multiple_rows() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let series = Series::new(vec!["a", "b", "c", "d", "e"], Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    assert_eq!(df.row_count(), 5);
    assert_eq!(df.column_count(), 1);

    Ok(())
}

/// Test Series with unicode characters
#[test]
fn test_series_unicode() -> Result<(), PandRSError> {
    let unicode_data = vec![
        "日本語",  // Japanese
        "中文",    // Chinese
        "한글",    // Korean
        "Русский", // Russian
        "العربية", // Arabic
        "עברית",   // Hebrew
        "🎌🇯🇵",    // Emojis
        "Ñoño",    // Spanish with tilde
        "Café",    // French with accent
        "Schön",   // German with umlaut
    ];

    let series = Series::new(unicode_data, Some("unicode".to_string()))?;

    assert_eq!(series.len(), 10);
    assert!(series.name().is_some());

    Ok(())
}

/// Test Series with emoji and special symbols
#[test]
fn test_series_emoji() -> Result<(), PandRSError> {
    let emoji_data = vec![
        "😀", "😃", "😄", "😁", "😆", "😅", "🤣", "😂", "🙂", "🙃", "😉", "😊", "🎌", "🇯🇵", "🚀",
        "💻", "📊", "📈", "🔬", "🧪",
    ];

    let series = Series::new(emoji_data, Some("emoji".to_string()))?;

    assert_eq!(series.len(), 20);

    Ok(())
}

/// Test Series with very long strings
#[test]
fn test_series_very_long_strings() -> Result<(), PandRSError> {
    let long_string = "a".repeat(100_000); // 100K characters
    let data = vec![long_string.as_str(), "short", "medium length text"];

    let series = Series::new(data, Some("mixed_length".to_string()))?;

    assert_eq!(series.len(), 3);

    Ok(())
}

/// Test Series with empty strings
#[test]
fn test_series_empty_strings() -> Result<(), PandRSError> {
    let data = vec!["", "", "", "non-empty", "", ""];

    let series = Series::new(data, Some("empty_strings".to_string()))?;

    assert_eq!(series.len(), 6);

    Ok(())
}

/// Test numeric Series with extreme values
#[test]
fn test_series_extreme_integers() -> Result<(), PandRSError> {
    let data = vec![
        i64::MIN,
        i64::MIN + 1,
        -1_000_000,
        -1,
        0,
        1,
        1_000_000,
        i64::MAX - 1,
        i64::MAX,
    ];

    let series = Series::new(data, Some("extreme_ints".to_string()))?;

    assert_eq!(series.len(), 9);

    Ok(())
}

/// Test float Series with infinity
#[test]
fn test_series_with_infinity() -> Result<(), PandRSError> {
    let data = vec![f64::NEG_INFINITY, -1000.0, -0.0, 0.0, 1000.0, f64::INFINITY];

    let series = Series::new(data, Some("with_inf".to_string()))?;

    assert_eq!(series.len(), 6);

    Ok(())
}

/// Test float Series with NaN
#[test]
fn test_series_with_nan() -> Result<(), PandRSError> {
    let data = vec![1.0, 2.0, f64::NAN, 4.0, f64::NAN, 6.0];

    let series = Series::new(data, Some("with_nan".to_string()))?;

    assert_eq!(series.len(), 6);

    Ok(())
}

/// Test float Series with very small numbers
#[test]
fn test_series_very_small_numbers() -> Result<(), PandRSError> {
    let data = vec![
        f64::MIN_POSITIVE,
        1e-308,
        1e-100,
        1e-10,
        1e-5,
        0.0000001,
        0.1,
    ];

    let series = Series::new(data, Some("small_numbers".to_string()))?;

    assert_eq!(series.len(), 7);

    Ok(())
}

/// Test float Series with very large numbers
#[test]
fn test_series_very_large_numbers() -> Result<(), PandRSError> {
    let data = vec![1e5, 1e10, 1e100, 1e308, f64::MAX / 2.0, f64::MAX];

    let series = Series::new(data, Some("large_numbers".to_string()))?;

    assert_eq!(series.len(), 6);

    Ok(())
}

/// Test DataFrame with all NA/null values
#[test]
fn test_dataframe_all_nulls() -> Result<(), PandRSError> {
    // This test depends on NA support in Series
    // For now, test with empty strings as null placeholders
    let mut df = DataFrame::new();

    df.add_column(
        "col1".to_string(),
        Series::new(vec!["", "", ""], Some("col1".to_string()))?,
    )?;

    assert_eq!(df.row_count(), 3);

    Ok(())
}

/// Test DataFrame with mixed types (as integers)
#[test]
fn test_dataframe_mixed_types_as_ints() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Mixed data represented as integers
    let data = vec![123, 45, 0, -1, 999, 0];
    let series = Series::new(data, Some("mixed".to_string()))?;
    df.add_column("mixed".to_string(), series)?;

    assert_eq!(df.row_count(), 6);

    Ok(())
}

/// Test DataFrame with very wide data (many columns)
#[test]
fn test_dataframe_many_columns() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create 100 columns
    for i in 0..100 {
        let col_name = format!("col_{}", i);
        let series = Series::new(vec![i, i + 1, i + 2], Some(col_name.clone()))?;
        df.add_column(col_name, series)?;
    }

    assert_eq!(df.column_count(), 100);
    assert_eq!(df.row_count(), 3);

    Ok(())
}

/// Test DataFrame with very tall data (many rows)
#[test]
fn test_dataframe_many_rows() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create 10,000 rows
    let data: Vec<i32> = (0..10_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    assert_eq!(df.row_count(), 10_000);
    assert_eq!(df.column_count(), 1);

    Ok(())
}

/// Test Series with duplicate values
#[test]
fn test_series_all_duplicates() -> Result<(), PandRSError> {
    let data = vec!["same"; 100];
    let series = Series::new(data, Some("duplicates".to_string()))?;

    assert_eq!(series.len(), 100);

    Ok(())
}

/// Test DataFrame column names with special characters
#[test]
fn test_dataframe_special_column_names() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Column names with special characters
    let special_names = vec![
        "column with spaces",
        "column-with-dashes",
        "column_with_underscores",
        "column.with.dots",
        "column@with@at",
        "column#with#hash",
        "column$with$dollar",
        "column%with%percent",
    ];

    for name in special_names {
        let series = Series::new(vec![1, 2, 3], Some(name.to_string()))?;
        df.add_column(name.to_string(), series)?;
    }

    assert_eq!(df.column_count(), 8);

    Ok(())
}

/// Test DataFrame with unicode column names
#[test]
fn test_dataframe_unicode_column_names() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let unicode_names = vec!["日本語", "中文", "한글", "🎌データ"];

    for name in unicode_names {
        let series = Series::new(vec![1, 2], Some(name.to_string()))?;
        df.add_column(name.to_string(), series)?;
    }

    assert_eq!(df.column_count(), 4);

    Ok(())
}

/// Test Series with alternating NA and valid values
#[test]
fn test_series_alternating_na() -> Result<(), PandRSError> {
    // Using empty strings to represent NA
    let data = vec!["a", "", "b", "", "c", "", "d", ""];
    let series = Series::new(data, Some("alternating".to_string()))?;

    assert_eq!(series.len(), 8);

    Ok(())
}

/// Test DataFrame with single very long column name
#[test]
fn test_dataframe_very_long_column_name() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let long_name = "a".repeat(1000);
    let series = Series::new(vec![1, 2, 3], Some(long_name.clone()))?;
    df.add_column(long_name.clone(), series)?;

    assert_eq!(df.column_count(), 1);
    assert!(df.contains_column(&long_name));

    Ok(())
}

/// Test Series with whitespace-only strings
#[test]
fn test_series_whitespace_strings() -> Result<(), PandRSError> {
    let data = vec![" ", "  ", "\t", "\n", "\r\n", "   \t\n   "];
    let series = Series::new(data, Some("whitespace".to_string()))?;

    assert_eq!(series.len(), 6);

    Ok(())
}

/// Test DataFrame operations on empty DataFrame
#[test]
fn test_operations_on_empty_dataframe() -> Result<(), PandRSError> {
    let df = DataFrame::new();

    // These operations should handle empty DataFrame gracefully
    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 0);
    assert!(df.column_names().is_empty());

    Ok(())
}

/// Test Series with boolean-like strings
#[test]
fn test_series_boolean_strings() -> Result<(), PandRSError> {
    let data = vec![
        "true", "false", "True", "False", "TRUE", "FALSE", "1", "0", "yes", "no", "Y", "N",
    ];
    let series = Series::new(data, Some("bool_strings".to_string()))?;

    assert_eq!(series.len(), 12);

    Ok(())
}

/// Test DataFrame with large numbers
#[test]
fn test_dataframe_large_numbers() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create row with very large number
    let large_num = i64::MAX;
    let series = Series::new(vec![large_num], Some("large_num".to_string()))?;
    df.add_column("large_num".to_string(), series)?;

    assert_eq!(df.row_count(), 1);

    Ok(())
}

/// Test Series with numeric strings that could cause parsing issues
#[test]
fn test_series_numeric_strings() -> Result<(), PandRSError> {
    let data = vec![
        "0", "1", "-1", "3.14", "-3.14", "1e10", "1e-10", "inf", "-inf", "nan", "0x1234", "0b1010",
    ];
    let series = Series::new(data, Some("numeric_strings".to_string()))?;

    assert_eq!(series.len(), 12);

    Ok(())
}

/// Test DataFrame with zero-width unicode characters
#[test]
fn test_dataframe_zero_width_characters() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Zero-width characters
    let data = vec![
        "a\u{200B}b", // Zero-width space
        "c\u{200C}d", // Zero-width non-joiner
        "e\u{200D}f", // Zero-width joiner
        "g\u{FEFF}h", // Zero-width no-break space
    ];

    let series = Series::new(data, Some("zero_width".to_string()))?;
    df.add_column("zero_width".to_string(), series)?;

    assert_eq!(df.row_count(), 4);

    Ok(())
}

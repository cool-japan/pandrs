//! Round-trip tests for the Pure Rust xlsx reader/writer.
//!
//! These tests verify that an `OptimizedDataFrame` written to an .xlsx file
//! and read back preserves row count, column names, and representative values
//! from each column type (Int64, Float64, String, Boolean).

#![cfg(feature = "excel")]

use pandrs::error::Result;
use pandrs::{BooleanColumn, Column, Float64Column, Int64Column, OptimizedDataFrame, StringColumn};
use tempfile::tempdir;

#[allow(clippy::result_large_err)]
fn build_sample_df() -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4, 5]);
    df.add_column("id", Column::Int64(id_col))?;

    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
        "Eve".to_string(),
    ]);
    df.add_column("name", Column::String(name_col))?;

    let score_col = Float64Column::new(vec![85.5, 92.0, 78.3, 90.1, 88.7]);
    df.add_column("score", Column::Float64(score_col))?;

    let active_col = BooleanColumn::new(vec![true, false, true, false, true]);
    df.add_column("active", Column::Boolean(active_col))?;

    Ok(df)
}

#[test]
#[allow(clippy::result_large_err)]
fn test_xlsx_round_trip_preserves_row_count_and_columns() -> Result<()> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("round_trip.xlsx");

    let df = build_sample_df()?;
    df.to_excel(&path, Some("Data"), false)?;

    let loaded = OptimizedDataFrame::from_excel(&path, Some("Data"), true, 0, None)?;

    assert_eq!(loaded.row_count(), 5, "row count preserved");
    // Column count should be the four data columns
    assert_eq!(loaded.column_count(), 4, "column count preserved");

    let names = loaded.column_names();
    assert!(names.iter().any(|n| n == "id"), "id column present");
    assert!(names.iter().any(|n| n == "name"), "name column present");
    assert!(names.iter().any(|n| n == "score"), "score column present");
    assert!(names.iter().any(|n| n == "active"), "active column present");

    drop(dir);
    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_xlsx_round_trip_preserves_integer_values() -> Result<()> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ints.xlsx");

    let df = build_sample_df()?;
    df.to_excel(&path, None, false)?;

    let loaded = OptimizedDataFrame::from_excel(&path, None, true, 0, None)?;
    let view = loaded.column("id")?;
    // Ids may be inferred as Int64 from the numeric cells.
    let int_col = view.as_int64().ok_or_else(|| {
        pandrs::error::Error::InvalidInput("id column should be Int64".to_string())
    })?;
    assert_eq!(int_col.get(0)?, Some(1));
    assert_eq!(int_col.get(4)?, Some(5));

    drop(dir);
    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_xlsx_round_trip_preserves_float_values() -> Result<()> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("floats.xlsx");

    let df = build_sample_df()?;
    df.to_excel(&path, None, false)?;

    let loaded = OptimizedDataFrame::from_excel(&path, None, true, 0, None)?;
    let view = loaded.column("score")?;
    let f = view.as_float64().ok_or_else(|| {
        pandrs::error::Error::InvalidInput("score column should be Float64".to_string())
    })?;
    let first = f.get(0)?.ok_or_else(|| {
        pandrs::error::Error::InvalidInput("score row 0 should exist".to_string())
    })?;
    assert!(
        (first - 85.5).abs() < 1e-9,
        "first score approx 85.5, got {first}"
    );

    drop(dir);
    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_xlsx_round_trip_preserves_string_values() -> Result<()> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("strings.xlsx");

    let df = build_sample_df()?;
    df.to_excel(&path, None, false)?;

    let loaded = OptimizedDataFrame::from_excel(&path, None, true, 0, None)?;
    let view = loaded.column("name")?;
    let s = view.as_string().ok_or_else(|| {
        pandrs::error::Error::InvalidInput("name column should be String".to_string())
    })?;
    let first = s
        .get(0)?
        .ok_or_else(|| pandrs::error::Error::InvalidInput("name row 0 should exist".to_string()))?;
    assert_eq!(first, "Alice");

    drop(dir);
    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_xlsx_list_sheet_names() -> Result<()> {
    use pandrs::io::list_sheet_names;

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("named.xlsx");

    let df = build_sample_df()?;
    df.to_excel(&path, Some("MyData"), false)?;

    let names = list_sheet_names(&path)?;
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "MyData");

    drop(dir);
    Ok(())
}

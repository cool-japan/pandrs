//! Pure Rust xlsx (.xlsx / OOXML SpreadsheetML) reader and writer.
//!
//! This module is the Pure Rust replacement for the previous `calamine` +
//! `simple_excel_writer` dependency pair. It is built on top of:
//!
//! - [`oxiarc_archive::zip`] for the ZIP container (Pure Rust),
//! - [`quick_xml`] for XML parsing (Pure Rust).
//!
//! It exposes a small, crate-internal API (`read_*` / `write_*`) that the
//! `pandrs::io::excel` facade and the `SplitDataFrame` I/O code call into.
//!
//! External users never see this module directly; they continue to call
//! `pandrs::io::read_excel`, `pandrs::io::write_excel`, etc., which preserve
//! their original signatures.

// The submodules live here. Nothing below is re-exported outside the crate.
mod cell;
mod error;
mod reader;
mod schema;
mod styles;
mod writer;

use std::collections::HashMap;
use std::path::Path;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

use self::reader::LoadedSheet;
use self::writer::{write_xlsx, SheetPayload};

/// Write a single-sheet xlsx file from a [`SplitDataFrame`].
pub(crate) fn write_split_dataframe<P: AsRef<Path>>(
    df: &SplitDataFrame,
    path: P,
    sheet_name: Option<&str>,
    include_index: bool,
) -> Result<()> {
    let name = sheet_name.unwrap_or("Sheet1").to_string();
    let payload = SheetPayload {
        name,
        df,
        include_index,
    };
    write_xlsx(path, &[payload])
}

/// Write multiple named sheets in one archive.
pub(crate) fn write_split_dataframe_sheets<P: AsRef<Path>>(
    sheets: &[(String, &SplitDataFrame)],
    path: P,
    include_index: bool,
) -> Result<()> {
    let payloads: Vec<SheetPayload<'_>> = sheets
        .iter()
        .map(|(name, df)| SheetPayload {
            name: name.clone(),
            df: *df,
            include_index,
        })
        .collect();
    write_xlsx(path, &payloads)
}

/// Read a sheet from an .xlsx file into a [`SplitDataFrame`]. Only the
/// requested sheet's XML is parsed — see `reader`'s module docs.
pub(crate) fn read_split_dataframe<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<SplitDataFrame> {
    let handle = reader::open_workbook(path)?;
    let idx = handle.sheet_index(sheet_name)?;
    let sheet = handle.load_sheet(idx)?;
    build_dataframe_from_sheet(&sheet, header, skip_rows, use_cols)
}

/// List all sheet names in a workbook, in workbook order. No worksheet body
/// is parsed to answer this.
pub(crate) fn list_sheets<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let handle = reader::open_workbook(path)?;
    Ok(handle.sheet_names)
}

/// Lightweight snapshot of a sheet (name, row count, column count).
pub(crate) fn sheet_dimensions<P: AsRef<Path>>(path: P) -> Result<Vec<(String, usize, usize)>> {
    let handle = reader::open_workbook(path)?;
    let sheets = handle.load_all()?;
    let out = sheets
        .iter()
        .map(|s| (s.name.clone(), s.row_count, s.col_count))
        .collect();
    Ok(out)
}

/// Read every sheet as its own DataFrame, keyed by sheet name.
pub(crate) fn read_all_sheets<P: AsRef<Path>>(
    path: P,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<HashMap<String, SplitDataFrame>> {
    let handle = reader::open_workbook(path)?;
    let sheets = handle.load_all()?;
    let mut out = HashMap::new();
    for sheet in &sheets {
        let df = build_dataframe_from_sheet(sheet, header, skip_rows, use_cols)?;
        out.insert(sheet.name.clone(), df);
    }
    Ok(out)
}

/// Convert a zero-indexed column number into its A1 letter sequence (e.g.
/// `0 -> "A"`, `26 -> "AA"`). Exposed for the `excel` facade's sheet-range
/// formatting, which needs the same bijective base-26 encoding the writer
/// uses internally (correct beyond the 26th column, unlike a naive
/// single-letter-only approximation).
pub(crate) fn column_letters(col: usize) -> String {
    self::cell::col_letters(col)
}

/// Turn a parsed sheet into a SplitDataFrame, performing pandrs-style
/// type inference per column.
fn build_dataframe_from_sheet(
    sheet: &LoadedSheet,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<SplitDataFrame> {
    let row_count = sheet.row_count;
    let col_count = sheet.col_count;

    // Resolve column names.
    let mut column_names: Vec<String> = Vec::new();
    if header && row_count > skip_rows {
        for i in 0..col_count {
            let s = sheet.get(skip_rows, i).to_display_string();
            if s.is_empty() {
                column_names.push(format!("Column{}", i + 1));
            } else {
                column_names.push(s);
            }
        }
    } else {
        for i in 0..col_count {
            column_names.push(format!("Column{}", i + 1));
        }
    }

    // Which columns to retain.
    let use_cols_indices: Option<Vec<usize>> = use_cols.map(|names| {
        let mut idx = Vec::new();
        for &want in names {
            if let Some(pos) = column_names.iter().position(|c| c == want) {
                idx.push(pos);
            }
        }
        idx
    });

    let data_start = if header { skip_rows + 1 } else { skip_rows };

    // Collect data per column. Bounded by `row_count * col_count`, which the
    // reader's density guard has already established is not wildly
    // disproportionate to the sheet's actual populated-cell count.
    let mut column_data: Vec<Vec<String>> = vec![Vec::new(); col_count];
    for r in data_start..row_count {
        for (c, data) in column_data.iter_mut().enumerate() {
            data.push(sheet.get(r, c).to_display_string());
        }
    }

    let mut df = SplitDataFrame::new();
    for c in 0..col_count {
        if let Some(ref selected) = use_cols_indices {
            if !selected.contains(&c) {
                continue;
            }
        }
        let name = column_names
            .get(c)
            .cloned()
            .unwrap_or_else(|| format!("Column{}", c + 1));
        let data = std::mem::take(&mut column_data[c]);
        if data.is_empty() {
            continue;
        }
        df.add_column(name, infer_column(&data))?;
    }
    Ok(df)
}

/// Pandas-style column-type inference copied from the existing read paths,
/// extended to preserve NA/missing-ness instead of fabricating placeholder
/// values.
///
/// Priority:
/// 1. All non-blank values parse as i64 → `Int64`.
/// 2. All non-blank values parse as f64 → `Float64`.
/// 3. All non-blank values are boolean-shaped → `Boolean`.
/// 4. Fallback → `String`.
///
/// A blank cell (empty after trimming) never contributes a fabricated `0`,
/// `0.0`, or `false` value: every one of pandrs' concrete column types
/// (`Int64Column`, `Float64Column`, `BooleanColumn`, `StringColumn`) can
/// carry a null bitmask via `with_nulls`, so a blank cell is recorded as a
/// genuine NA there instead. There is consequently no case here where "the
/// column type cannot represent NA" — if that ever changes for a future
/// column type, the fallback is to keep it as a String column, since a
/// blank string cell is already unambiguous (an empty string) without
/// needing a bitmask.
fn infer_column(data: &[String]) -> Column {
    let is_missing: Vec<bool> = data.iter().map(|s| s.trim().is_empty()).collect();
    let has_nulls = is_missing.iter().any(|&b| b);
    let non_empty: Vec<&str> = data
        .iter()
        .zip(is_missing.iter())
        .filter(|(_, &missing)| !missing)
        .map(|(s, _)| s.as_str())
        .collect();

    if non_empty.is_empty() {
        // Every cell is blank: nothing to infer a numeric/boolean type from.
        return string_column(data, &is_missing, has_nulls);
    }

    if non_empty.iter().all(|s| s.trim().parse::<i64>().is_ok()) {
        let v: Vec<i64> = data
            .iter()
            .map(|s| s.trim().parse::<i64>().unwrap_or_default())
            .collect();
        return if has_nulls {
            Column::Int64(Int64Column::with_nulls(v, is_missing))
        } else {
            Column::Int64(Int64Column::new(v))
        };
    }

    if non_empty.iter().all(|s| s.trim().parse::<f64>().is_ok()) {
        let v: Vec<f64> = data
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or_default())
            .collect();
        return if has_nulls {
            Column::Float64(Float64Column::with_nulls(v, is_missing))
        } else {
            Column::Float64(Float64Column::new(v))
        };
    }

    if non_empty.iter().all(|s| looks_like_bool(s)) {
        let v: Vec<bool> = data.iter().map(|s| parse_bool_cell(s.trim())).collect();
        return if has_nulls {
            Column::Boolean(BooleanColumn::with_nulls(v, is_missing))
        } else {
            Column::Boolean(BooleanColumn::new(v))
        };
    }

    string_column(data, &is_missing, has_nulls)
}

fn string_column(data: &[String], is_missing: &[bool], has_nulls: bool) -> Column {
    if has_nulls {
        Column::String(StringColumn::with_nulls(data.to_vec(), is_missing.to_vec()))
    } else {
        Column::String(StringColumn::new(data.to_vec()))
    }
}

fn looks_like_bool(s: &str) -> bool {
    let s = s.trim().to_lowercase();
    matches!(
        s.as_str(),
        "true" | "false" | "yes" | "no" | "t" | "f" | "1" | "0"
    )
}

fn parse_bool_cell(s: &str) -> bool {
    let s = s.to_lowercase();
    matches!(s.as_str(), "true" | "yes" | "t" | "1")
}

#[cfg(test)]
mod facade_tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_df() -> SplitDataFrame {
        let mut df = SplitDataFrame::new();
        df.add_column(
            "id".to_string(),
            Column::Int64(Int64Column::new(vec![10, 20, 30])),
        )
        .expect("add id column");
        df.add_column(
            "name".to_string(),
            Column::String(StringColumn::new(vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Carol".to_string(),
            ])),
        )
        .expect("add name column");
        df.add_column(
            "score".to_string(),
            Column::Float64(Float64Column::new(vec![1.5, 2.5, 3.5])),
        )
        .expect("add score column");
        df.add_column(
            "active".to_string(),
            Column::Boolean(BooleanColumn::new(vec![true, false, true])),
        )
        .expect("add active column");
        df
    }

    #[test]
    fn round_trip_single_sheet() -> Result<()> {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("roundtrip.xlsx");

        let df = sample_df();
        write_split_dataframe(&df, &path, Some("Data"), false)?;

        let names = list_sheets(&path)?;
        assert_eq!(names, vec!["Data".to_string()]);

        let loaded = read_split_dataframe(&path, Some("Data"), true, 0, None)?;
        assert_eq!(loaded.row_count(), 3);
        let cols = loaded.column_names();
        assert!(cols.iter().any(|n| n == "id"));
        assert!(cols.iter().any(|n| n == "name"));
        assert!(cols.iter().any(|n| n == "score"));
        assert!(cols.iter().any(|n| n == "active"));
        Ok(())
    }

    #[test]
    fn infer_column_preserves_na_instead_of_fabricating_zero() {
        let data = vec!["1".to_string(), "".to_string(), "3".to_string()];
        let col = infer_column(&data);
        match col {
            Column::Int64(c) => {
                assert_eq!(c.get(0).unwrap(), Some(1));
                assert_eq!(c.get(1).unwrap(), None, "blank cell must be NA, not 0");
                assert_eq!(c.get(2).unwrap(), Some(3));
            }
            other => panic!("expected Int64 column, got {other:?}"),
        }
    }

    #[test]
    fn infer_column_preserves_na_for_float_and_bool() {
        let floats = vec!["1.5".to_string(), "".to_string(), "3.5".to_string()];
        match infer_column(&floats) {
            Column::Float64(c) => {
                assert_eq!(c.get(1).unwrap(), None, "blank cell must be NA, not 0.0");
            }
            other => panic!("expected Float64 column, got {other:?}"),
        }

        let bools = vec!["true".to_string(), "".to_string(), "false".to_string()];
        match infer_column(&bools) {
            Column::Boolean(c) => {
                assert_eq!(c.get(1).unwrap(), None, "blank cell must be NA, not false");
            }
            other => panic!("expected Boolean column, got {other:?}"),
        }
    }

    #[test]
    fn column_letters_handles_more_than_26_columns() {
        assert_eq!(column_letters(0), "A");
        assert_eq!(column_letters(25), "Z");
        assert_eq!(column_letters(26), "AA");
        assert_eq!(column_letters(27), "AB");
    }
}

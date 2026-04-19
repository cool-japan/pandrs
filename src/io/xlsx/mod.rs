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
mod writer;

use std::collections::HashMap;
use std::path::Path;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::Result;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

use self::cell::XlsxCellValue;
use self::reader::{load_workbook, LoadedWorkbook};
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

/// Read a sheet from an .xlsx file into a [`SplitDataFrame`].
pub(crate) fn read_split_dataframe<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<SplitDataFrame> {
    let wb = load_workbook(path)?;
    let sheet = select_sheet(&wb, sheet_name)?;
    build_dataframe_from_sheet(sheet, header, skip_rows, use_cols)
}

/// List all sheet names in a workbook, in workbook order.
pub(crate) fn list_sheets<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let wb = load_workbook(path)?;
    Ok(wb.sheet_names)
}

/// Lightweight snapshot of a sheet (name, row count, column count).
pub(crate) fn sheet_dimensions<P: AsRef<Path>>(path: P) -> Result<Vec<(String, usize, usize)>> {
    let wb = load_workbook(path)?;
    let out = wb
        .sheets
        .iter()
        .map(|s| (s.name.clone(), s.rows.len(), s.cols))
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
    let wb = load_workbook(path)?;
    let mut out = HashMap::new();
    for sheet in &wb.sheets {
        let df = build_dataframe_from_sheet(sheet, header, skip_rows, use_cols)?;
        out.insert(sheet.name.clone(), df);
    }
    Ok(out)
}

/// Choose a sheet from a loaded workbook by name or default to the first.
fn select_sheet<'a>(wb: &'a LoadedWorkbook, name: Option<&str>) -> Result<&'a reader::LoadedSheet> {
    match name {
        Some(n) => wb
            .sheets
            .iter()
            .find(|s| s.name == n)
            .ok_or_else(|| self::error::io_err(format!("xlsx: sheet '{n}' not found"))),
        None => wb
            .sheets
            .first()
            .ok_or_else(|| self::error::io_err("xlsx: workbook contains no sheets".to_string())),
    }
}

/// Turn a parsed sheet into a SplitDataFrame, performing pandrs-style
/// type inference per column.
fn build_dataframe_from_sheet(
    sheet: &reader::LoadedSheet,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<SplitDataFrame> {
    let row_count = sheet.rows.len();
    let col_count = sheet.cols;

    // Resolve column names.
    let mut column_names: Vec<String> = Vec::new();
    if header && row_count > skip_rows {
        let header_row = &sheet.rows[skip_rows];
        for (i, cell) in header_row.iter().enumerate() {
            let s = cell.to_display_string();
            if s.is_empty() {
                column_names.push(format!("Column{}", i + 1));
            } else {
                column_names.push(s);
            }
        }
        // Pad in case header row is shorter than the actual width.
        while column_names.len() < col_count {
            column_names.push(format!("Column{}", column_names.len() + 1));
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

    // Collect data per column.
    let mut column_data: Vec<Vec<String>> = vec![Vec::new(); col_count];
    for row in sheet.rows.iter().skip(data_start) {
        for (c, cell) in row.iter().enumerate() {
            if c < column_data.len() {
                column_data[c].push(cell.to_display_string());
            }
        }
        // Right-pad short rows so that every column sees the same number of
        // entries. Empty cells become the empty string.
        for c in row.len()..col_count {
            column_data[c].push(String::new());
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

/// Pandas-style column-type inference copied from the existing read paths.
///
/// Priority:
/// 1. All non-empty values parse as i64 → `Int64`.
/// 2. All non-empty values parse as f64 → `Float64`.
/// 3. All non-empty values are boolean-shaped → `Boolean`.
/// 4. Fallback → `String`.
fn infer_column(data: &[String]) -> Column {
    let non_empty: Vec<&str> = data
        .iter()
        .map(|s| s.as_str())
        .filter(|s| !s.trim().is_empty())
        .collect();
    if non_empty.is_empty() {
        return Column::String(StringColumn::new(data.to_vec()));
    }

    if non_empty.iter().all(|s| s.trim().parse::<i64>().is_ok()) {
        let v: Vec<i64> = data
            .iter()
            .map(|s| s.trim().parse::<i64>().unwrap_or(0))
            .collect();
        return Column::Int64(Int64Column::new(v));
    }

    if non_empty.iter().all(|s| s.trim().parse::<f64>().is_ok()) {
        let v: Vec<f64> = data
            .iter()
            .map(|s| s.trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        return Column::Float64(Float64Column::new(v));
    }

    if non_empty.iter().all(|s| looks_like_bool(s)) {
        let v: Vec<bool> = data.iter().map(|s| parse_bool_cell(s.trim())).collect();
        return Column::Boolean(BooleanColumn::new(v));
    }

    Column::String(StringColumn::new(data.to_vec()))
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
}

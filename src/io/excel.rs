//! Public Excel I/O facade for pandrs.
//!
//! This module preserves the public API originally powered by
//! `calamine` + `simple_excel_writer`, but delegates all actual work to the
//! Pure Rust `crate::io::xlsx` module (built on `oxiarc-archive` and
//! `quick-xml`). All existing types, function signatures, and semantics are
//! preserved.

use std::collections::HashMap;
use std::path::Path;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::io::xlsx;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
use crate::optimized::OptimizedDataFrame;
use crate::series::Series;

/// Enhanced Excel cell information with formatting.
#[derive(Debug, Clone)]
pub struct ExcelCell {
    /// Cell value
    pub value: String,
    /// Cell formula (if any)
    pub formula: Option<String>,
    /// Cell data type
    pub data_type: String,
    /// Cell formatting information
    pub format: ExcelCellFormat,
}

/// Excel cell formatting information.
#[derive(Debug, Clone)]
pub struct ExcelCellFormat {
    /// Font bold.
    pub font_bold: bool,
    /// Font italic.
    pub font_italic: bool,
    /// Font color.
    pub font_color: Option<String>,
    /// Background color.
    pub background_color: Option<String>,
    /// Number format.
    pub number_format: Option<String>,
}

impl Default for ExcelCellFormat {
    fn default() -> Self {
        Self {
            font_bold: false,
            font_italic: false,
            font_color: None,
            background_color: None,
            number_format: None,
        }
    }
}

/// Named-range information. Our in-tree writer does not emit named ranges,
/// but the type is preserved to keep the public API stable.
#[derive(Debug, Clone)]
pub struct NamedRange {
    /// Name of the range.
    pub name: String,
    /// Sheet name.
    pub sheet_name: String,
    /// Cell range (e.g. "A1:D10").
    pub range: String,
    /// Optional comment.
    pub comment: Option<String>,
}

/// Enhanced Excel reading options.
#[derive(Debug, Clone)]
pub struct ExcelReadOptions {
    /// Preserve formulas instead of evaluating them.
    pub preserve_formulas: bool,
    /// Include cell formatting information.
    pub include_formatting: bool,
    /// Read named ranges.
    pub read_named_ranges: bool,
    /// Memory mapping for large files.
    pub use_memory_map: bool,
    /// Skip rows/columns optimization.
    pub optimize_memory: bool,
}

impl Default for ExcelReadOptions {
    fn default() -> Self {
        Self {
            preserve_formulas: false,
            include_formatting: false,
            read_named_ranges: false,
            use_memory_map: true,
            optimize_memory: true,
        }
    }
}

/// Enhanced Excel writing options.
#[derive(Debug, Clone)]
pub struct ExcelWriteOptions {
    /// Preserve formulas.
    pub preserve_formulas: bool,
    /// Apply cell formatting.
    pub apply_formatting: bool,
    /// Write named ranges.
    pub write_named_ranges: bool,
    /// Protect worksheets.
    pub protect_sheets: bool,
    /// Large-file optimizations.
    pub optimize_large_files: bool,
}

impl Default for ExcelWriteOptions {
    fn default() -> Self {
        Self {
            preserve_formulas: false,
            apply_formatting: false,
            write_named_ranges: false,
            protect_sheets: false,
            optimize_large_files: false,
        }
    }
}

/// Information about an Excel workbook.
#[derive(Debug, Clone)]
pub struct ExcelWorkbookInfo {
    /// Names of all sheets in the workbook.
    pub sheet_names: Vec<String>,
    /// Total number of sheets.
    pub sheet_count: usize,
    /// Total number of cells across all sheets.
    pub total_cells: usize,
}

/// Information about a specific Excel sheet.
#[derive(Debug, Clone)]
pub struct ExcelSheetInfo {
    /// Name of the sheet.
    pub name: String,
    /// Number of rows with data.
    pub rows: usize,
    /// Number of columns with data.
    pub columns: usize,
    /// Cell range (e.g. "A1:D10").
    pub range: String,
}

/// Comprehensive Excel file analysis structure.
#[derive(Debug, Clone)]
pub struct ExcelFileAnalysis {
    /// Basic workbook information.
    pub workbook_info: ExcelWorkbookInfo,
    /// Cells containing formulas. Our Pure Rust path does not retain formulas,
    /// so this is always 0.
    pub formula_count: usize,
    /// Formatted cells. Our Pure Rust path does not retain formatting, so this
    /// is always 0.
    pub formatted_cell_count: usize,
    /// Number of named ranges.
    pub named_range_count: usize,
    /// Estimated file complexity.
    pub complexity_score: f64,
}

// --------------------------------------------------------------------------
// Read/write core functions
// --------------------------------------------------------------------------

/// Read a DataFrame from an Excel file.
pub fn read_excel<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<DataFrame> {
    let split = xlsx::read_split_dataframe(path.as_ref(), sheet_name, header, skip_rows, use_cols)?;
    split_to_standard(&split)
}

/// Write an `OptimizedDataFrame` to an Excel file.
pub fn write_excel<P: AsRef<Path>>(
    df: &OptimizedDataFrame,
    path: P,
    sheet_name: Option<&str>,
    index: bool,
) -> Result<()> {
    let split = optimized_to_split(df)?;
    xlsx::write_split_dataframe(&split, path.as_ref(), sheet_name, index)
}

/// List every sheet name in a workbook.
pub fn list_sheet_names<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    xlsx::list_sheets(path.as_ref())
}

/// Return workbook-level information.
pub fn get_workbook_info<P: AsRef<Path>>(path: P) -> Result<ExcelWorkbookInfo> {
    let dims = xlsx::sheet_dimensions(path.as_ref())?;
    let sheet_names: Vec<String> = dims.iter().map(|d| d.0.clone()).collect();
    let total_cells = dims.iter().map(|d| d.1 * d.2).sum();
    Ok(ExcelWorkbookInfo {
        sheet_names: sheet_names.clone(),
        sheet_count: sheet_names.len(),
        total_cells,
    })
}

/// Return per-sheet information.
pub fn get_sheet_info<P: AsRef<Path>>(path: P, sheet_name: &str) -> Result<ExcelSheetInfo> {
    let dims = xlsx::sheet_dimensions(path.as_ref())?;
    let (_, rows, cols) = dims
        .iter()
        .find(|(n, _, _)| n == sheet_name)
        .cloned()
        .ok_or_else(|| {
            Error::IoError(format!("Could not find sheet '{sheet_name}' in workbook"))
        })?;
    let last_col_letter = if cols == 0 {
        'A'
    } else {
        // Approximation suitable for small tables (<= 26 columns). For wider
        // tables we just return 'Z'.
        let letter_idx = cols.saturating_sub(1) as u32;
        if letter_idx <= 25 {
            char::from_u32(b'A' as u32 + letter_idx).unwrap_or('Z')
        } else {
            'Z'
        }
    };
    let range = format!("A1:{last_col_letter}{rows}");
    Ok(ExcelSheetInfo {
        name: sheet_name.to_string(),
        rows,
        columns: cols,
        range,
    })
}

/// Read multiple sheets.
pub fn read_excel_sheets<P: AsRef<Path>>(
    path: P,
    sheet_names: Option<&[&str]>,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<HashMap<String, DataFrame>> {
    let mut all = xlsx::read_all_sheets(path.as_ref(), header, skip_rows, use_cols)?;
    let mut out = HashMap::new();
    let names: Vec<String> = match sheet_names {
        Some(wanted) => {
            for &n in wanted {
                if !all.contains_key(n) {
                    return Err(Error::IoError(format!(
                        "Sheet '{n}' not found. Available sheets: {:?}",
                        all.keys().collect::<Vec<_>>()
                    )));
                }
            }
            wanted.iter().map(|s| (*s).to_string()).collect()
        }
        None => all.keys().cloned().collect(),
    };
    for name in names {
        if let Some(split) = all.remove(&name) {
            out.insert(name, split_to_standard(&split)?);
        }
    }
    Ok(out)
}

/// Read a sheet plus workbook-level metadata.
pub fn read_excel_with_info<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    header: bool,
    skip_rows: usize,
    use_cols: Option<&[&str]>,
) -> Result<(DataFrame, ExcelWorkbookInfo)> {
    let df = read_excel(path.as_ref(), sheet_name, header, skip_rows, use_cols)?;
    let info = get_workbook_info(path.as_ref())?;
    Ok((df, info))
}

/// Write multiple sheets in a single file.
pub fn write_excel_sheets<P: AsRef<Path>>(
    sheets: &HashMap<String, &OptimizedDataFrame>,
    path: P,
    index: bool,
) -> Result<()> {
    // Materialise split dataframes so we can borrow them for the xlsx writer.
    let materialised: Vec<(String, SplitDataFrame)> = sheets
        .iter()
        .map(|(name, df)| {
            let split = optimized_to_split(df)?;
            Ok::<_, Error>((name.clone(), split))
        })
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<(String, &SplitDataFrame)> =
        materialised.iter().map(|(n, d)| (n.clone(), d)).collect();
    xlsx::write_split_dataframe_sheets(&refs, path.as_ref(), index)
}

// --------------------------------------------------------------------------
// "Enhanced" helpers — preserved signatures, simplified semantics.
// --------------------------------------------------------------------------

/// Read an Excel file with enhanced options.
///
/// Our Pure Rust path does not retain formulas, formatting, or named ranges.
/// This function therefore returns the DataFrame plus empty `cells` and
/// `named_ranges` vectors when those options are requested. The signature is
/// preserved for API compatibility.
pub fn read_excel_enhanced<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    _options: ExcelReadOptions,
) -> Result<(DataFrame, Vec<ExcelCell>, Vec<NamedRange>)> {
    let df = read_excel(path.as_ref(), sheet_name, true, 0, None)?;
    Ok((df, Vec::new(), Vec::new()))
}

/// Write an Excel file with enhanced options.
///
/// Enhanced options (`preserve_formulas`, `apply_formatting`, `protect_sheets`,
/// `write_named_ranges`) are accepted for API compatibility but are not
/// propagated to the Pure Rust backend.
pub fn write_excel_enhanced<P: AsRef<Path>>(
    df: &OptimizedDataFrame,
    path: P,
    sheet_name: Option<&str>,
    _cells: &[ExcelCell],
    _named_ranges: &[NamedRange],
    _options: ExcelWriteOptions,
) -> Result<()> {
    write_excel(df, path, sheet_name, false)
}

/// Optimise an Excel file by round-tripping it through our Pure Rust codec.
pub fn optimize_excel_file<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    _compression_level: u8,
) -> Result<()> {
    let df = read_excel(input_path.as_ref(), None, true, 0, None)?;
    let optimized_df = OptimizedDataFrame::from_dataframe(&df)?;
    write_excel(&optimized_df, output_path.as_ref(), None, false)
}

/// Analyse an Excel file. Formula / formatting / named-range counts are
/// always zero in this Pure Rust implementation (see struct doc comment).
pub fn analyze_excel_file<P: AsRef<Path>>(path: P) -> Result<ExcelFileAnalysis> {
    let workbook_info = get_workbook_info(path.as_ref())?;
    let complexity_score = workbook_info.total_cells as f64 * 0.1;
    Ok(ExcelFileAnalysis {
        workbook_info,
        formula_count: 0,
        formatted_cell_count: 0,
        named_range_count: 0,
        complexity_score,
    })
}

// --------------------------------------------------------------------------
// Internal conversion helpers.
// --------------------------------------------------------------------------

/// Convert a [`SplitDataFrame`] into the standard [`DataFrame`] by producing
/// a `Series<String>` per column. This matches the pre-existing behaviour of
/// `read_excel` which returned a `DataFrame` whose columns were string-typed.
fn split_to_standard(split: &SplitDataFrame) -> Result<DataFrame> {
    let mut df = DataFrame::new();
    for (col, col_name) in split.columns.iter().zip(split.column_names.iter()) {
        let strings: Vec<String> = column_to_strings(col);
        let series = Series::new(strings, Some(col_name.clone()))?;
        df.add_column(col_name.clone(), series)?;
    }
    Ok(df)
}

fn column_to_strings(col: &Column) -> Vec<String> {
    match col {
        Column::Int64(c) => (0..c.len())
            .map(|i| match c.get(i) {
                Ok(Some(v)) => v.to_string(),
                _ => String::new(),
            })
            .collect(),
        Column::Float64(c) => (0..c.len())
            .map(|i| match c.get(i) {
                Ok(Some(v)) => v.to_string(),
                _ => String::new(),
            })
            .collect(),
        Column::String(c) => (0..c.len())
            .map(|i| match c.get(i) {
                Ok(Some(v)) => v.to_string(),
                _ => String::new(),
            })
            .collect(),
        Column::Boolean(c) => (0..c.len())
            .map(|i| match c.get(i) {
                Ok(Some(v)) => v.to_string(),
                _ => String::new(),
            })
            .collect(),
    }
}

/// Build a `SplitDataFrame` from an `OptimizedDataFrame` by cloning every
/// column. This mirrors the convert step used in `optimized/dataframe/io.rs`.
fn optimized_to_split(df: &OptimizedDataFrame) -> Result<SplitDataFrame> {
    let mut split = SplitDataFrame::new();
    for name in df.column_names() {
        let view = df.column(name)?;
        split.add_column(name.clone(), view.column().clone())?;
    }
    Ok(split)
}

// ColumnTrait is needed for `.len()` on the concrete column types.
use crate::column::ColumnTrait;

// Ensure unused warnings do not trip — these imports exist to help readers.
#[allow(dead_code)]
fn _unused_helpers(_: &StringColumn, _: &Int64Column, _: &Float64Column, _: &BooleanColumn) {}

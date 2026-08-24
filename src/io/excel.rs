//! Public Excel I/O facade for pandrs.
//!
//! This module preserves the public API originally powered by
//! `calamine` + `simple_excel_writer`, but delegates all actual work to the
//! Pure Rust `crate::io::xlsx` module (built on `oxiarc-archive` and
//! `quick-xml`). All existing types, function signatures, and semantics are
//! preserved.

use std::collections::HashMap;
use std::path::Path;

use crate::column::Column;
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
    /// Preserve formulas instead of evaluating them. **Not supported** by
    /// the Pure Rust reader: setting this to `true` makes
    /// [`read_excel_enhanced`] return `Err(Error::NotImplemented)` rather
    /// than silently ignoring the request.
    pub preserve_formulas: bool,
    /// Include cell formatting information. **Not supported**; see
    /// `preserve_formulas`.
    pub include_formatting: bool,
    /// Read named ranges. **Not supported**; see `preserve_formulas`.
    pub read_named_ranges: bool,
    /// Memory mapping for large files. Accepted for API compatibility but
    /// has no effect: the Pure Rust reader always decompresses and parses
    /// fully in memory (there is no `mmap`-backed path). Left as a
    /// performance *hint* rather than a hard error, unlike the options
    /// above, because it doesn't change the correctness of the result —
    /// only (in principle, if ever implemented) how it's produced.
    pub use_memory_map: bool,
    /// Skip rows/columns optimization. Also a no-op performance hint today.
    pub optimize_memory: bool,
}

impl Default for ExcelReadOptions {
    fn default() -> Self {
        Self {
            preserve_formulas: false,
            include_formatting: false,
            read_named_ranges: false,
            // Defaults to `false`: there is no mmap implementation, so
            // defaulting this to `true` would falsely claim the read path
            // is memory-mapped.
            use_memory_map: false,
            optimize_memory: true,
        }
    }
}

/// Enhanced Excel writing options.
#[derive(Debug, Clone)]
pub struct ExcelWriteOptions {
    /// Preserve formulas. **Not supported** by the Pure Rust writer: setting
    /// this to `true` makes [`write_excel_enhanced`] return
    /// `Err(Error::NotImplemented)` rather than silently dropping formulas.
    pub preserve_formulas: bool,
    /// Apply cell formatting. **Not supported**; see `preserve_formulas`.
    pub apply_formatting: bool,
    /// Write named ranges. **Not supported**; see `preserve_formulas`.
    pub write_named_ranges: bool,
    /// Protect worksheets. **Not supported**; see `preserve_formulas`. This
    /// one matters most: silently ignoring a protection request would leave
    /// a caller believing a sheet is protected when it is not.
    pub protect_sheets: bool,
    /// Large-file optimizations. A performance hint only; accepted but
    /// currently a no-op (the writer has one code path regardless of file
    /// size), so it is not treated as an error like the options above.
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
        "A".to_string()
    } else {
        // Full bijective base-26 column-letter encoding (matches the
        // writer's own `encode_ref`), correct for any column count instead
        // of clamping every sheet past column Z to a nonsensical "Z".
        xlsx::column_letters(cols - 1)
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
///
/// `sheets` is a `HashMap`, which has no defined iteration order; to keep
/// the *output file's* sheet order deterministic across runs (rather than
/// varying randomly call to call for the exact same input, which is
/// confusing and breaks any test or diff relying on stable output), sheet
/// names are written in sorted order.
pub fn write_excel_sheets<P: AsRef<Path>>(
    sheets: &HashMap<String, &OptimizedDataFrame>,
    path: P,
    index: bool,
) -> Result<()> {
    let mut names: Vec<&String> = sheets.keys().collect();
    names.sort();

    // Materialise split dataframes so we can borrow them for the xlsx writer.
    let materialised: Vec<(String, SplitDataFrame)> = names
        .into_iter()
        .map(|name| {
            let df = sheets[name];
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
/// Requesting any of `preserve_formulas`, `include_formatting`, or
/// `read_named_ranges` returns `Err(Error::NotImplemented)` describing which
/// option(s) can't be honored, rather than silently reading plain cell
/// values and returning empty `cells`/`named_ranges` as if nothing had been
/// asked for. Call with the default `ExcelReadOptions` (or explicitly leave
/// those three `false`) to get plain cell values.
pub fn read_excel_enhanced<P: AsRef<Path>>(
    path: P,
    sheet_name: Option<&str>,
    options: ExcelReadOptions,
) -> Result<(DataFrame, Vec<ExcelCell>, Vec<NamedRange>)> {
    let mut unsupported = Vec::new();
    if options.preserve_formulas {
        unsupported.push("preserve_formulas");
    }
    if options.include_formatting {
        unsupported.push("include_formatting");
    }
    if options.read_named_ranges {
        unsupported.push("read_named_ranges");
    }
    if !unsupported.is_empty() {
        return Err(Error::NotImplemented(format!(
            "read_excel_enhanced: option(s) [{}] are not supported by the Pure Rust xlsx reader \
             (formulas, cell formatting, and named ranges are not retained); \
             call with default ExcelReadOptions if you only need cell values",
            unsupported.join(", ")
        )));
    }
    let df = read_excel(path.as_ref(), sheet_name, true, 0, None)?;
    Ok((df, Vec::new(), Vec::new()))
}

/// Write an Excel file with enhanced options.
///
/// Our Pure Rust path does not retain formulas, formatting, named ranges, or
/// sheet protection. Requesting any of `preserve_formulas`,
/// `apply_formatting`, `write_named_ranges`, or `protect_sheets` — or
/// passing non-empty `cells`/`named_ranges` expecting them to be written —
/// returns `Err(Error::NotImplemented)` describing which request can't be
/// honored, rather than silently writing plain cell values while implying
/// (especially for `protect_sheets`) that the request succeeded.
pub fn write_excel_enhanced<P: AsRef<Path>>(
    df: &OptimizedDataFrame,
    path: P,
    sheet_name: Option<&str>,
    cells: &[ExcelCell],
    named_ranges: &[NamedRange],
    options: ExcelWriteOptions,
) -> Result<()> {
    let mut unsupported = Vec::new();
    if options.preserve_formulas {
        unsupported.push("preserve_formulas".to_string());
    }
    if options.apply_formatting {
        unsupported.push("apply_formatting".to_string());
    }
    if options.write_named_ranges {
        unsupported.push("write_named_ranges".to_string());
    }
    if options.protect_sheets {
        unsupported.push("protect_sheets".to_string());
    }
    if !cells.is_empty() {
        unsupported.push(format!(
            "{} ExcelCell entries (formula/formatting data is not written)",
            cells.len()
        ));
    }
    if !named_ranges.is_empty() {
        unsupported.push(format!(
            "{} NamedRange entries (named ranges are not written)",
            named_ranges.len()
        ));
    }
    if !unsupported.is_empty() {
        return Err(Error::NotImplemented(format!(
            "write_excel_enhanced: not supported by the Pure Rust xlsx writer: {}; \
             call write_excel directly if you only need cell values written",
            unsupported.join("; ")
        )));
    }
    write_excel(df, path, sheet_name, false)
}

/// Optimise an Excel file by round-tripping it through our Pure Rust codec.
///
/// Reads and rewrites **every** sheet, in the source workbook's own order —
/// not just the first one — so a multi-sheet input round-trips completely
/// instead of silently losing every sheet after the first.
///
/// `_compression_level` is accepted for API compatibility but currently has
/// no effect: the writer always uses one fixed DEFLATE setting (see
/// `xlsx::writer`), there is no per-call knob to plumb it through to yet.
pub fn optimize_excel_file<P1: AsRef<Path>, P2: AsRef<Path>>(
    input_path: P1,
    output_path: P2,
    _compression_level: u8,
) -> Result<()> {
    let sheet_names = xlsx::list_sheets(input_path.as_ref())?;
    let mut materialised: Vec<(String, SplitDataFrame)> = Vec::with_capacity(sheet_names.len());
    for name in &sheet_names {
        let split =
            xlsx::read_split_dataframe(input_path.as_ref(), Some(name.as_str()), true, 0, None)?;
        materialised.push((name.clone(), split));
    }
    let refs: Vec<(String, &SplitDataFrame)> =
        materialised.iter().map(|(n, d)| (n.clone(), d)).collect();
    xlsx::write_split_dataframe_sheets(&refs, output_path.as_ref(), false)
}

/// Analyse an Excel file. Formula / formatting / named-range counts are
/// always zero in this Pure Rust implementation (see struct doc comment).
pub fn analyze_excel_file<P: AsRef<Path>>(path: P) -> Result<ExcelFileAnalysis> {
    let workbook_info = get_workbook_info(path.as_ref())?;
    Ok(ExcelFileAnalysis {
        workbook_info,
        formula_count: 0,
        formatted_cell_count: 0,
        named_range_count: 0,
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
        let strings = column_to_strings(col)?;
        let series = Series::new(strings, Some(col_name.clone()))?;
        df.add_column(col_name.clone(), series)?;
    }
    Ok(df)
}

/// Render every value of a column as a string, for the string-typed
/// `DataFrame` this facade has always returned from `read_excel`.
///
/// A genuine NA (`Ok(None)`) renders as `""` — this layer represents every
/// column as a plain `Series<String>` with no independent null-bitmask, so
/// `""` is that representation's existing, deliberate convention for
/// "missing" (consistent with how a blank xlsx cell already reads back as
/// `""` upstream). What must *not* happen is silently turning a genuine
/// `Err` (e.g. an internal index-out-of-bounds bug) into that same `""` —
/// that would hide a real defect behind output indistinguishable from a
/// blank cell. So only `Ok(None)` maps to `""`; `Err` propagates via `?`.
fn column_to_strings(col: &Column) -> Result<Vec<String>> {
    fn render<T: ToString>(cell: Result<Option<T>>) -> Result<String> {
        cell.map(|opt| opt.map(|v| v.to_string()).unwrap_or_default())
    }

    match col {
        Column::Int64(c) => (0..c.len()).map(|i| render(c.get(i))).collect(),
        Column::Float64(c) => (0..c.len()).map(|i| render(c.get(i))).collect(),
        Column::String(c) => (0..c.len()).map(|i| render(c.get(i))).collect(),
        Column::Boolean(c) => (0..c.len()).map(|i| render(c.get(i))).collect(),
    }
}

/// Build a `SplitDataFrame` from an `OptimizedDataFrame` by cloning every
/// column (and carrying over its index, if any, so a subsequent
/// `include_index` write emits the real index rather than a positional
/// counter). This mirrors the convert step used in
/// `optimized/dataframe/io.rs`.
fn optimized_to_split(df: &OptimizedDataFrame) -> Result<SplitDataFrame> {
    let mut split = SplitDataFrame::new();
    for name in df.column_names() {
        let view = df.column(name)?;
        split.add_column(name.clone(), view.column().clone())?;
    }
    if let Some(idx) = df.get_index() {
        // `SplitDataFrame::set_index`'s only failure mode is a row-count
        // mismatch. For any `df` with at least one column that always
        // holds here (both `idx.len()` and `split`'s row count derive from
        // the same `df.row_count()`). The one case it can't hold is a
        // zero-column `df` carrying a non-empty index — `SplitDataFrame`
        // has no columns to establish a row count from in that case. There
        // is no data to write in that degenerate case anyway, so we fall
        // back to positional index labels there rather than fail the whole
        // write over it.
        let _ = split.set_index(idx.clone());
    }
    Ok(split)
}

// ColumnTrait is needed for `.len()` on the concrete column types.
use crate::column::ColumnTrait;

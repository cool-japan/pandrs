//! Advanced indexing functionality for DataFrames
//!
//! This module provides comprehensive indexing capabilities including:
//! - .loc, .iloc, .at, .iat accessors (pandas-like)
//! - Multi-level index support
//! - Advanced selection methods
//! - Boolean indexing enhancements
//! - Fancy indexing capabilities
//! - Index alignment and reindexing

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::base::Series;

/// Selection specification for rows
#[derive(Debug, Clone)]
pub enum RowSelector {
    /// Single row by index
    Single(String),
    /// Single row by position
    Position(usize),
    /// Multiple rows by indices
    Multiple(Vec<String>),
    /// Multiple rows by positions
    Positions(Vec<usize>),
    /// Boolean mask
    Boolean(Vec<bool>),
    /// Range selection
    Range(IndexRange),
    /// All rows
    All,
}

/// Selection specification for columns
#[derive(Debug, Clone)]
pub enum ColumnSelector {
    /// Single column
    Single(String),
    /// Multiple columns
    Multiple(Vec<String>),
    /// All columns
    All,
}

/// Range specification for indexing
#[derive(Debug, Clone)]
pub enum IndexRange {
    /// Standard range (start..end)
    Standard { start: usize, end: usize },
    /// Range from start (start..)
    From { start: usize },
    /// Range to end (..end)
    To { end: usize },
    /// Full range (..)
    Full,
    /// Inclusive range (start..=end)
    Inclusive { start: usize, end: usize },
    /// Range to inclusive (..=end)
    ToInclusive { end: usize },
}

/// Index alignment strategy
#[derive(Debug, Clone)]
pub enum AlignmentStrategy {
    /// Fill missing values with NaN/None
    Outer,
    /// Keep only common indices
    Inner,
    /// Use left DataFrame's index
    Left,
    /// Use right DataFrame's index
    Right,
}

/// Multi-level index specification
#[derive(Debug, Clone)]
pub struct MultiLevelIndex {
    /// Level names
    pub names: Vec<String>,
    /// Level values for each row
    pub levels: Vec<Vec<String>>,
    /// Combined tuples
    pub tuples: Vec<Vec<String>>,
}

impl MultiLevelIndex {
    /// Create a new multi-level index
    pub fn new(names: Vec<String>, levels: Vec<Vec<String>>) -> Result<Self> {
        if names.len() != levels.len() {
            return Err(Error::InvalidValue(
                "Number of names must match number of levels".to_string(),
            ));
        }

        if levels.is_empty() {
            return Err(Error::InvalidValue(
                "At least one level required".to_string(),
            ));
        }

        let row_count = levels[0].len();
        for level in &levels {
            if level.len() != row_count {
                return Err(Error::InvalidValue(
                    "All levels must have the same length".to_string(),
                ));
            }
        }

        let mut tuples = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let mut tuple = Vec::with_capacity(levels.len());
            for level in &levels {
                tuple.push(level[i].clone());
            }
            tuples.push(tuple);
        }

        Ok(Self {
            names,
            levels,
            tuples,
        })
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        if self.levels.is_empty() {
            0
        } else {
            self.levels[0].len()
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get unique values for a specific level
    pub fn level_values(&self, level: usize) -> Result<Vec<String>> {
        if level >= self.levels.len() {
            return Err(Error::IndexOutOfBounds {
                index: level,
                size: self.levels.len(),
            });
        }

        let mut unique_values: Vec<String> = self.levels[level].iter().cloned().collect();
        unique_values.sort();
        unique_values.dedup();
        Ok(unique_values)
    }

    /// Find rows matching a specific tuple
    pub fn find_tuple(&self, tuple: &[String]) -> Vec<usize> {
        self.tuples
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if t.len() >= tuple.len() && &t[..tuple.len()] == tuple {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------
// Typed single-cell / single-column helpers.
//
// `DataFrame` stores each column as a boxed `Series<T>` for its exact
// concrete element type (String, the integer/float families, bool, or one
// of the chrono date/time types -- the same set
// `DataFrame::get_column_string_values` supports). Reading or rewriting a
// single cell, or gathering a subset of rows, previously went through
// `get_column_string_values`, which renders the *entire* column to
// `Vec<String>` just to read (or copy) one value -- an O(rows) allocation
// per cell/column that made `.at`/`.iat`/`.iloc` scalar access, and every
// row-selecting operation built on it, quadratic overall. These helpers
// instead downcast the column exactly once (via `DataFrame::get_column::<T>`,
// itself a single `HashMap` lookup + one `Any` downcast) and then
// index/format only the values actually needed, so a single-cell
// read/write is O(1) (aside from the constant-size list of candidate
// types) and gathering `k` rows out of `n` is O(k), not O(n).
// ---------------------------------------------------------------------

/// Read a single cell's value as its `Display` string, without
/// materialising the rest of the column.
fn get_cell_string(df: &DataFrame, column: &str, row: usize) -> Result<String> {
    if !df.contains_column(column) {
        return Err(Error::ColumnNotFound(column.to_string()));
    }

    macro_rules! try_cell {
        ($ty:ty) => {
            if let Ok(series) = df.get_column::<$ty>(column) {
                return match series.get(row) {
                    Some(value) => Ok(value.to_string()),
                    None => Err(Error::IndexOutOfBounds {
                        index: row,
                        size: series.len(),
                    }),
                };
            }
        };
    }

    try_cell!(String);
    try_cell!(i64);
    try_cell!(f64);
    try_cell!(i32);
    try_cell!(f32);
    try_cell!(bool);
    try_cell!(i8);
    try_cell!(i16);
    try_cell!(i128);
    try_cell!(isize);
    try_cell!(u8);
    try_cell!(u16);
    try_cell!(u32);
    try_cell!(u64);
    try_cell!(u128);
    try_cell!(usize);
    try_cell!(chrono::NaiveDate);
    try_cell!(chrono::NaiveDateTime);
    try_cell!(chrono::DateTime<chrono::Utc>);

    Err(Error::InvalidValue(format!(
        "Column '{}' has an element type that is not supported for string conversion",
        column
    )))
}

/// Gather `positions` from column `name` in `src`, preserving its exact
/// concrete element type, into a new column appended to `dst`.
///
/// This is the workhorse behind every dtype-preserving row-selecting
/// operation in this module (`.iloc`, and in turn `head`/`tail`/`sample`,
/// plus `reset_index`'s carry-through of unchanged columns, `reindex`'s
/// fully-resolved fast path, and `.at`/`.iat`'s carry-through of every
/// column except the one being mutated): no column is ever rebuilt as
/// `Series<String>` just because it passed through a row-selection step.
fn gather_column_typed(
    dst: &mut DataFrame,
    src: &DataFrame,
    name: &str,
    positions: &[usize],
) -> Result<()> {
    if !src.contains_column(name) {
        return Err(Error::ColumnNotFound(name.to_string()));
    }

    macro_rules! try_gather {
        ($ty:ty) => {
            if let Ok(series) = src.get_column::<$ty>(name) {
                let mut values: Vec<$ty> = Vec::with_capacity(positions.len());
                for &i in positions {
                    match series.get(i) {
                        Some(value) => values.push(value.clone()),
                        None => {
                            return Err(Error::IndexOutOfBounds {
                                index: i,
                                size: series.len(),
                            })
                        }
                    }
                }
                dst.add_column(
                    name.to_string(),
                    Series::new(values, Some(name.to_string()))?,
                )?;
                return Ok(());
            }
        };
    }

    try_gather!(String);
    try_gather!(i64);
    try_gather!(f64);
    try_gather!(i32);
    try_gather!(f32);
    try_gather!(bool);
    try_gather!(i8);
    try_gather!(i16);
    try_gather!(i128);
    try_gather!(isize);
    try_gather!(u8);
    try_gather!(u16);
    try_gather!(u32);
    try_gather!(u64);
    try_gather!(u128);
    try_gather!(usize);
    try_gather!(chrono::NaiveDate);
    try_gather!(chrono::NaiveDateTime);
    try_gather!(chrono::DateTime<chrono::Utc>);

    Err(Error::InvalidValue(format!(
        "Column '{}' has an element type that is not supported for this operation",
        name
    )))
}

/// Select the given row positions from `df`, preserving every column's
/// concrete element type via [`gather_column_typed`] and, when `df` has an
/// explicit simple (single-level) row index, slicing that index the same
/// way so row labels stay aligned with the selected data. (A multi-level
/// index is not sliced here -- the result has no explicit index in that
/// case, matching how row-selection elsewhere in the crate currently
/// treats `MultiIndex`.)
fn select_rows_by_positions(df: &DataFrame, positions: &[usize]) -> Result<DataFrame> {
    let row_count = df.row_count();
    for &pos in positions {
        if pos >= row_count {
            return Err(Error::IndexOutOfBounds {
                index: pos,
                size: row_count,
            });
        }
    }

    let mut result = DataFrame::new();
    for name in df.column_names() {
        gather_column_typed(&mut result, df, name, positions)?;
    }

    if let crate::index::DataFrameIndex::Simple(idx) = df.get_index() {
        if !idx.is_empty() {
            let sliced_labels: Vec<String> = positions
                .iter()
                .map(|&i| idx.get_value(i).cloned().unwrap_or_default())
                .collect();
            if let Ok(sliced_index) = crate::index::Index::new(sliced_labels) {
                // Fully-qualified: `AdvancedIndexingExt::set_index` (this
                // module's trait, `&self` -> `Result<DataFrame>`) is found
                // before `DataFrame::set_index` (`&mut self` -> `Result<()>`)
                // by plain method-call syntax, since Rust's method
                // resolution tries `&self` methods before `&mut self` ones
                // -- so the inherent in-place setter needs UFCS here.
                DataFrame::set_index(&mut result, sliced_index)?;
            }
        }
    }

    Ok(result)
}

/// Resolve a `.loc`-style string label to a row position against `df`'s
/// real row index.
///
/// * When `df` has an explicit single-level index (set via
///   [`AdvancedIndexingExt::set_index`] or [`DataFrame::set_index`]),
///   `label` is looked up as a genuine label against that index, so
///   `.loc` is actually label-based rather than being `.iloc` with a
///   string argument that merely happens to parse as a number.
/// * When `df` has no explicit index (the implicit default `RangeIndex`),
///   `label` falls back to being parsed as a row position, matching
///   pandas' behaviour of accepting integer(-like) labels positionally on
///   a `RangeIndex`-backed frame.
/// * A multi-level index cannot be resolved from a single bare string
///   label (that needs tuple-based selection -- see
///   [`LocIndexer::get_tuple`]), so this returns an error naming the
///   label rather than silently misinterpreting it positionally.
fn resolve_label_position(df: &DataFrame, label: &str) -> Result<usize> {
    match df.get_index() {
        crate::index::DataFrameIndex::Simple(idx) if !idx.is_empty() => idx
            .get_loc(&label.to_string())
            .ok_or_else(|| Error::InvalidValue(format!("Label '{}' not found in index", label))),
        crate::index::DataFrameIndex::Multi(midx) if !midx.is_empty() => {
            Err(Error::InvalidValue(format!(
                "Label '{}' cannot be resolved against a multi-level index; use tuple-based \
                 selection instead",
                label
            )))
        }
        _ => {
            let pos = label.parse::<usize>().map_err(|_| {
                Error::InvalidValue(format!("Label '{}' not found in index", label))
            })?;
            if pos >= df.row_count() {
                return Err(Error::InvalidValue(format!(
                    "Label '{}' not found in index",
                    label
                )));
            }
            Ok(pos)
        }
    }
}

/// Rewrite the value at `row` in column `name` (parsed into that column's
/// concrete element type) and add the result to `dst`, preserving the
/// column's original type. Supports `String`, `i64`, `f64`, `i32`, `f32`
/// and `bool` columns; any other element type returns
/// [`Error::NotImplemented`] naming it, backing `set_cell_value`.
fn set_mutated_column(
    dst: &mut DataFrame,
    src: &DataFrame,
    name: &str,
    row: usize,
    value: &str,
) -> Result<()> {
    if let Ok(series) = src.get_column::<String>(name) {
        let mut values = series.values().to_vec();
        if row >= values.len() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: values.len(),
            });
        }
        values[row] = value.to_string();
        dst.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
        return Ok(());
    }

    macro_rules! try_mutate_numeric {
        ($ty:ty) => {
            if let Ok(series) = src.get_column::<$ty>(name) {
                let mut values: Vec<$ty> = series.values().to_vec();
                if row >= values.len() {
                    return Err(Error::IndexOutOfBounds {
                        index: row,
                        size: values.len(),
                    });
                }
                values[row] = value.trim().parse::<$ty>().map_err(|_| {
                    Error::InvalidValue(format!(
                        "Cannot set value '{}' into column '{}': not a valid {}",
                        value,
                        name,
                        stringify!($ty)
                    ))
                })?;
                dst.add_column(
                    name.to_string(),
                    Series::new(values, Some(name.to_string()))?,
                )?;
                return Ok(());
            }
        };
    }

    try_mutate_numeric!(i64);
    try_mutate_numeric!(f64);
    try_mutate_numeric!(i32);
    try_mutate_numeric!(f32);

    if let Ok(series) = src.get_column::<bool>(name) {
        let mut values: Vec<bool> = series.values().to_vec();
        if row >= values.len() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: values.len(),
            });
        }
        // Same "true"/"1" and "false"/"0" convention `DataFrame::append_string_row` uses.
        values[row] = match value.trim().to_lowercase().as_str() {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => {
                return Err(Error::InvalidValue(format!(
                    "Cannot set value '{}' into column '{}': not a valid bool",
                    value, name
                )))
            }
        };
        dst.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
        return Ok(());
    }

    if !src.contains_column(name) {
        return Err(Error::ColumnNotFound(name.to_string()));
    }

    Err(Error::NotImplemented(format!(
        "Mutating column '{}' via .at/.iat is not supported for its element type",
        name
    )))
}

/// Build a new DataFrame identical to `df` except that `column`'s value at
/// `row` is replaced with `value` (parsed into that column's concrete
/// element type).
///
/// [`Series`] is immutable, so a mutation cannot be applied in place --
/// every column is rebuilt into a fresh `DataFrame`, in its original
/// order, with only `column` actually changing (via
/// [`set_mutated_column`]); every other column is copied through via
/// [`gather_column_typed`], preserving its exact type. This backs
/// [`AtIndexer::set`] and [`IAtIndexer::set`].
fn set_cell_value(df: &DataFrame, column: &str, row: usize, value: &str) -> Result<DataFrame> {
    if !df.contains_column(column) {
        return Err(Error::ColumnNotFound(column.to_string()));
    }
    if row >= df.row_count() {
        return Err(Error::IndexOutOfBounds {
            index: row,
            size: df.row_count(),
        });
    }

    let all_positions: Vec<usize> = (0..df.row_count()).collect();
    let mut result = DataFrame::new();
    for name in df.column_names() {
        if name == column {
            set_mutated_column(&mut result, df, name, row, value)?;
        } else {
            gather_column_typed(&mut result, df, name, &all_positions)?;
        }
    }

    // A single-cell mutation never changes row labels: carry the original
    // index (if any) through unchanged.
    // Fully-qualified (see `select_rows_by_positions` for why): plain
    // method-call syntax would resolve to this module's `&self`-shaped
    // `AdvancedIndexingExt::set_index`/`set_multi_index` instead of
    // `DataFrame`'s `&mut self` in-place setters.
    match df.get_index() {
        crate::index::DataFrameIndex::Simple(idx) if !idx.is_empty() => {
            DataFrame::set_index(&mut result, idx)?;
        }
        crate::index::DataFrameIndex::Multi(midx) if !midx.is_empty() => {
            DataFrame::set_multi_index(&mut result, midx)?;
        }
        _ => {}
    }

    Ok(result)
}

/// Position-based indexer (.iloc)
pub struct ILocIndexer<'a> {
    dataframe: &'a DataFrame,
}

impl<'a> ILocIndexer<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self { dataframe }
    }

    /// Select by single position
    ///
    /// Reads each column through `get_cell_string`, one downcast and one
    /// value format per column, rather than materialising every column of
    /// the DataFrame as `Vec<String>` (O(rows) per column) just to keep a
    /// single row out of it.
    pub fn get(&self, row: usize) -> Result<HashMap<String, String>> {
        if row >= self.dataframe.row_count() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: self.dataframe.row_count(),
            });
        }

        let mut result = HashMap::new();
        for col_name in self.dataframe.column_names() {
            let value = get_cell_string(self.dataframe, col_name, row)?;
            result.insert(col_name.to_string(), value);
        }
        Ok(result)
    }

    /// Select by row and column positions
    ///
    /// Backed by `get_cell_string`: a single downcast plus one value
    /// format, not a full-column `Vec<String>` materialisation for one
    /// cell.
    pub fn get_at(&self, row: usize, col: usize) -> Result<String> {
        let col_names = self.dataframe.column_names();
        if col >= col_names.len() {
            return Err(Error::IndexOutOfBounds {
                index: col,
                size: col_names.len(),
            });
        }
        if row >= self.dataframe.row_count() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: self.dataframe.row_count(),
            });
        }

        get_cell_string(self.dataframe, &col_names[col], row)
    }

    /// Select by row range
    pub fn get_range(&self, rows: Range<usize>) -> Result<DataFrame> {
        self.select_rows(RowSelector::Range(IndexRange::Standard {
            start: rows.start,
            end: rows.end,
        }))
    }

    /// Select by row range and column range
    pub fn get_slice(&self, rows: Range<usize>, cols: Range<usize>) -> Result<DataFrame> {
        let result = self.get_range(rows)?;
        let col_names = self.dataframe.column_names();

        // Select only the specified column range
        let selected_cols: Vec<String> = col_names
            .iter()
            .skip(cols.start)
            .take(cols.end - cols.start)
            .cloned()
            .collect();

        let col_refs: Vec<&str> = selected_cols.iter().map(|s| s.as_str()).collect();
        result.select_columns(&col_refs)
    }

    /// Select by multiple row positions
    pub fn get_positions(&self, positions: &[usize]) -> Result<DataFrame> {
        self.select_rows(RowSelector::Positions(positions.to_vec()))
    }

    /// Select by boolean mask
    pub fn get_boolean(&self, mask: &[bool]) -> Result<DataFrame> {
        self.select_rows(RowSelector::Boolean(mask.to_vec()))
    }

    /// Internal method to select rows based on selector
    ///
    /// Row gathering goes through [`select_rows_by_positions`] /
    /// [`gather_column_typed`], so every column keeps its exact original
    /// element type (previously every column was rebuilt as
    /// `Series<String>`, destroying `i64`/`f64`/`bool`/date dtypes on
    /// every `.iloc` row selection, and in turn on `head`/`tail`/`sample`,
    /// which are built on it).
    fn select_rows(&self, selector: RowSelector) -> Result<DataFrame> {
        let indices: Vec<usize> = match selector {
            RowSelector::Range(range) => {
                let (start, end) = match range {
                    IndexRange::Standard { start, end } => (start, end),
                    IndexRange::From { start } => (start, self.dataframe.row_count()),
                    IndexRange::To { end } => (0, end),
                    IndexRange::Full => (0, self.dataframe.row_count()),
                    IndexRange::Inclusive { start, end } => (start, end + 1),
                    IndexRange::ToInclusive { end } => (0, end + 1),
                };
                (start..end.min(self.dataframe.row_count())).collect()
            }
            RowSelector::Positions(positions) => positions,
            RowSelector::Boolean(mask) => mask
                .iter()
                .enumerate()
                .filter_map(|(i, &include)| if include { Some(i) } else { None })
                .collect(),
            _ => {
                return Err(Error::InvalidValue(
                    "Unsupported selector for iloc".to_string(),
                ))
            }
        };

        select_rows_by_positions(self.dataframe, &indices)
    }
}

/// Label-based indexer (.loc)
pub struct LocIndexer<'a> {
    dataframe: &'a DataFrame,
    index: Option<&'a MultiLevelIndex>,
}

impl<'a> LocIndexer<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self {
            dataframe,
            index: None,
        }
    }

    pub fn with_index(dataframe: &'a DataFrame, index: &'a MultiLevelIndex) -> Self {
        Self {
            dataframe,
            index: Some(index),
        }
    }

    /// Select by single label
    pub fn get(&self, label: &str) -> Result<HashMap<String, String>> {
        let position = self.find_label_position(label)?;
        let iloc = ILocIndexer::new(self.dataframe);
        iloc.get(position)
    }

    /// Select by label and column
    pub fn get_at(&self, label: &str, column: &str) -> Result<String> {
        let position = self.find_label_position(label)?;
        get_cell_string(self.dataframe, column, position)
    }

    /// Select by multiple labels
    pub fn get_labels(&self, labels: &[String]) -> Result<DataFrame> {
        let positions: Result<Vec<usize>> = labels
            .iter()
            .map(|label| self.find_label_position(label))
            .collect();

        let iloc = ILocIndexer::new(self.dataframe);
        iloc.get_positions(&positions?)
    }

    /// Select by tuple (for multi-level index)
    pub fn get_tuple(&self, tuple: &[String]) -> Result<DataFrame> {
        if let Some(index) = self.index {
            let positions = index.find_tuple(tuple);
            if positions.is_empty() {
                return Err(Error::InvalidValue(format!("Tuple {:?} not found", tuple)));
            }
            let iloc = ILocIndexer::new(self.dataframe);
            iloc.get_positions(&positions)
        } else {
            Err(Error::InvalidValue(
                "Multi-level index required for tuple selection".to_string(),
            ))
        }
    }

    /// Find the row position of a label, resolved against the DataFrame's
    /// real row index (see `resolve_label_position`).
    fn find_label_position(&self, label: &str) -> Result<usize> {
        resolve_label_position(self.dataframe, label)
    }
}

/// Scalar indexer (.at)
pub struct AtIndexer<'a> {
    dataframe: &'a DataFrame,
}

impl<'a> AtIndexer<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self { dataframe }
    }

    /// Get single scalar value by label and column
    pub fn get(&self, label: &str, column: &str) -> Result<String> {
        let loc = LocIndexer::new(self.dataframe);
        loc.get_at(label, column)
    }

    /// Set single scalar value by label and column.
    ///
    /// `label` is resolved against the DataFrame's real row index (the
    /// same resolution `.loc` itself uses). Because [`Series`] is
    /// immutable, this returns a *new* DataFrame with every column
    /// preserved as-is except `column`, whose value at the resolved row is
    /// replaced with `value` parsed into that column's concrete element
    /// type (see `set_cell_value`).
    pub fn set(&self, label: &str, column: &str, value: String) -> Result<DataFrame> {
        let position = resolve_label_position(self.dataframe, label)?;
        set_cell_value(self.dataframe, column, position, &value)
    }
}

/// Scalar indexer by position (.iat)
pub struct IAtIndexer<'a> {
    dataframe: &'a DataFrame,
}

impl<'a> IAtIndexer<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self { dataframe }
    }

    /// Get single scalar value by row and column positions
    pub fn get(&self, row: usize, col: usize) -> Result<String> {
        let iloc = ILocIndexer::new(self.dataframe);
        iloc.get_at(row, col)
    }

    /// Set single scalar value by row and column positions.
    ///
    /// Because [`Series`] is immutable, this returns a *new* DataFrame
    /// with every column preserved as-is except the one at `col`, whose
    /// value at `row` is replaced with `value` parsed into that column's
    /// concrete element type (see `set_cell_value`).
    pub fn set(&self, row: usize, col: usize, value: String) -> Result<DataFrame> {
        let col_names = self.dataframe.column_names();
        if col >= col_names.len() {
            return Err(Error::IndexOutOfBounds {
                index: col,
                size: col_names.len(),
            });
        }
        let column = col_names[col].clone();
        set_cell_value(self.dataframe, &column, row, &value)
    }
}

/// Advanced selection builder
pub struct SelectionBuilder<'a> {
    dataframe: &'a DataFrame,
    row_selector: Option<RowSelector>,
    column_selector: Option<ColumnSelector>,
}

impl<'a> SelectionBuilder<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self {
            dataframe,
            row_selector: None,
            column_selector: None,
        }
    }

    /// Select rows by indices
    pub fn rows(mut self, selector: RowSelector) -> Self {
        self.row_selector = Some(selector);
        self
    }

    /// Select columns
    pub fn columns(mut self, selector: ColumnSelector) -> Self {
        self.column_selector = Some(selector);
        self
    }

    /// Execute the selection
    pub fn select(self) -> Result<DataFrame> {
        // Apply row selection first. When one is given, its result
        // replaces `result` immediately, so avoid an initial full
        // DataFrame clone that would just be discarded.
        let mut result = match self.row_selector {
            Some(row_selector) => {
                let iloc = ILocIndexer::new(self.dataframe);
                iloc.select_rows(row_selector)?
            }
            None => self.dataframe.clone(),
        };

        // Apply column selection
        if let Some(column_selector) = self.column_selector {
            match column_selector {
                ColumnSelector::Single(col) => {
                    result = result.select_columns(&[&col])?;
                }
                ColumnSelector::Multiple(cols) => {
                    let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
                    result = result.select_columns(&col_refs)?;
                }
                ColumnSelector::All => {
                    // No change needed
                }
            }
        }

        Ok(result)
    }
}

/// Index alignment operations
pub struct IndexAligner;

impl IndexAligner {
    /// Align two DataFrames based on their indices
    pub fn align(
        left: &DataFrame,
        right: &DataFrame,
        strategy: AlignmentStrategy,
    ) -> Result<(DataFrame, DataFrame)> {
        // For basic implementation, assume simple row-based alignment
        let left_len = left.row_count();
        let right_len = right.row_count();

        match strategy {
            AlignmentStrategy::Outer => {
                let max_len = left_len.max(right_len);
                let aligned_left = Self::extend_dataframe(left, max_len)?;
                let aligned_right = Self::extend_dataframe(right, max_len)?;
                Ok((aligned_left, aligned_right))
            }
            AlignmentStrategy::Inner => {
                let min_len = left_len.min(right_len);
                let aligned_left = Self::truncate_dataframe(left, min_len)?;
                let aligned_right = Self::truncate_dataframe(right, min_len)?;
                Ok((aligned_left, aligned_right))
            }
            AlignmentStrategy::Left => {
                let aligned_right = if right_len < left_len {
                    Self::extend_dataframe(right, left_len)?
                } else {
                    Self::truncate_dataframe(right, left_len)?
                };
                Ok((left.clone(), aligned_right))
            }
            AlignmentStrategy::Right => {
                let aligned_left = if left_len < right_len {
                    Self::extend_dataframe(left, right_len)?
                } else {
                    Self::truncate_dataframe(left, right_len)?
                };
                Ok((aligned_left, right.clone()))
            }
        }
    }

    /// Extend DataFrame to target length by NA-padding the extra rows.
    ///
    /// Numeric columns (`i64`/`f64`/`i32`/`f32`) upcast to `f64` with the
    /// padded rows set to `f64::NAN` (mirroring pandas' int -> float
    /// NaN-upcast when aligning to a longer index, and the same pattern
    /// [`DataFrame::concat_rows`] already uses for a column present in
    /// only one of two frames being concatenated). A non-numeric column
    /// has no NA-capable representation in this DataFrame's column model,
    /// so this returns [`Error::NotImplemented`] naming the column rather
    /// than fabricating padding: the previous behaviour cyclically
    /// *repeated real observations* to fill the extra rows (`values[i %
    /// values.len()]`), silently duplicating data that was never actually
    /// observed.
    fn extend_dataframe(df: &DataFrame, target_len: usize) -> Result<DataFrame> {
        let current_len = df.row_count();
        if current_len >= target_len {
            return Ok(df.clone());
        }
        let pad_len = target_len - current_len;

        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            if df.is_numeric_column(col_name) {
                let mut values = df.get_column_numeric_values(col_name)?;
                values.extend(std::iter::repeat(f64::NAN).take(pad_len));
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else {
                return Err(Error::NotImplemented(format!(
                    "extend_dataframe: column '{}' is not numeric, so it cannot be NA-padded \
                     from {} to {} rows; only numeric columns (upcast to f64 with NaN) support \
                     alignment padding",
                    col_name, current_len, target_len
                )));
            }
        }

        Ok(result)
    }

    /// Truncate DataFrame to target length
    fn truncate_dataframe(df: &DataFrame, target_len: usize) -> Result<DataFrame> {
        if df.row_count() <= target_len {
            return Ok(df.clone());
        }

        let iloc = ILocIndexer::new(df);
        iloc.get_range(0..target_len)
    }

    /// Reindex DataFrame with a new set of row labels.
    ///
    /// Each label in `new_index` is resolved against `df`'s real row index
    /// (falling back to positional interpretation only when `df` has no
    /// explicit index -- see `resolve_label_position`), and the result's
    /// row index becomes `new_index` itself, like pandas'
    /// `DataFrame.reindex(new_index)`.
    ///
    /// When every requested label resolves to a real row, every column
    /// keeps its exact original element type. When at least one label is
    /// absent: numeric columns (`i64`/`f64`/`i32`/`f32`) upcast to `f64`
    /// with the missing rows set to `f64::NAN` (mirroring pandas' int ->
    /// float NaN-upcast on reindex); a non-numeric column has no
    /// NA-capable representation here, so this returns
    /// [`Error::NotImplemented`] naming the column rather than
    /// fabricating a placeholder -- the previous behaviour: every missing
    /// entry, in every column regardless of type, silently became the
    /// literal string `"NaN"`.
    pub fn reindex(df: &DataFrame, new_index: &[String]) -> Result<DataFrame> {
        let positions: Vec<Option<usize>> = new_index
            .iter()
            .map(|label| resolve_label_position(df, label).ok())
            .collect();
        let any_missing = positions.iter().any(Option::is_none);
        let resolved_positions: Option<Vec<usize>> = if any_missing {
            None
        } else {
            Some(positions.iter().map(|p| p.unwrap_or(0)).collect())
        };

        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            if let Some(resolved) = &resolved_positions {
                gather_column_typed(&mut result, df, col_name, resolved)?;
                continue;
            }

            if df.is_numeric_column(col_name) {
                let current = df.get_column_numeric_values(col_name)?;
                let values: Vec<f64> = positions
                    .iter()
                    .map(|pos| pos.map(|p| current[p]).unwrap_or(f64::NAN))
                    .collect();
                result.add_column(
                    col_name.clone(),
                    Series::new(values, Some(col_name.clone()))?,
                )?;
            } else {
                return Err(Error::NotImplemented(format!(
                    "reindex: column '{}' is not numeric and at least one requested label is \
                     absent from the source index, so the missing rows cannot be NA-filled for \
                     it; numeric columns upcast to f64 with NaN for missing rows",
                    col_name
                )));
            }
        }

        // Fully-qualified: see `select_rows_by_positions` for why plain
        // `result.set_index(..)` would resolve to the wrong method here.
        let row_labels = crate::index::Index::new(new_index.to_vec())?;
        DataFrame::set_index(&mut result, row_labels)?;

        Ok(result)
    }
}

/// Extension trait to add advanced indexing to DataFrame
pub trait AdvancedIndexingExt {
    /// Get position-based indexer (.iloc)
    fn iloc(&self) -> ILocIndexer;

    /// Get label-based indexer (.loc)
    fn loc(&self) -> LocIndexer;

    /// Get scalar indexer (.at)
    fn at(&self) -> AtIndexer;

    /// Get scalar position indexer (.iat)
    fn iat(&self) -> IAtIndexer;

    /// Get selection builder
    fn select(&self) -> SelectionBuilder;

    /// Reset index to default range index
    fn reset_index(&self) -> Result<DataFrame>;

    /// Set a column as the index
    fn set_index(&self, column: &str) -> Result<DataFrame>;

    /// Create a multi-level index from multiple columns
    fn set_multi_index(&self, columns: &[String]) -> Result<(DataFrame, MultiLevelIndex)>;

    /// Select columns by name
    fn select_columns(&self, columns: &[String]) -> Result<DataFrame>;

    /// Drop columns by name
    fn drop_columns(&self, columns: &[String]) -> Result<DataFrame>;

    /// Sample random rows
    fn sample(&self, n: usize) -> Result<DataFrame>;

    /// Get head (first n rows)
    fn head(&self, n: usize) -> Result<DataFrame>;

    /// Get tail (last n rows)
    fn tail(&self, n: usize) -> Result<DataFrame>;
}

impl AdvancedIndexingExt for DataFrame {
    fn iloc(&self) -> ILocIndexer {
        ILocIndexer::new(self)
    }

    fn loc(&self) -> LocIndexer {
        LocIndexer::new(self)
    }

    fn at(&self) -> AtIndexer {
        AtIndexer::new(self)
    }

    fn iat(&self) -> IAtIndexer {
        IAtIndexer::new(self)
    }

    fn select(&self) -> SelectionBuilder {
        SelectionBuilder::new(self)
    }

    fn reset_index(&self) -> Result<DataFrame> {
        let index = self.get_index();
        let row_count = self.row_count();
        let mut result = DataFrame::new();

        match &index {
            crate::index::DataFrameIndex::Simple(idx) if !idx.is_empty() => {
                let col_name = idx.name().cloned().unwrap_or_else(|| "index".to_string());
                let values: Vec<String> = idx.values().to_vec();
                result.add_column(col_name.clone(), Series::new(values, Some(col_name))?)?;
            }
            crate::index::DataFrameIndex::Multi(midx) if !midx.is_empty() => {
                let names = midx.names().to_vec();
                for level in 0..midx.n_levels() {
                    let col_name = names
                        .get(level)
                        .and_then(|n| n.clone())
                        .unwrap_or_else(|| format!("level_{}", level));
                    let mut values = Vec::with_capacity(midx.len());
                    for i in 0..midx.len() {
                        let tuple = midx.get_tuple(i).ok_or(Error::IndexOutOfBounds {
                            index: i,
                            size: midx.len(),
                        })?;
                        values.push(tuple.get(level).cloned().unwrap_or_default());
                    }
                    result.add_column(col_name.clone(), Series::new(values, Some(col_name))?)?;
                }
            }
            _ => {
                // No explicit index was set: the implicit index is the
                // default RangeIndex(0..row_count), which pandas
                // materialises as an integer "index" column.
                let values: Vec<i64> = (0..row_count as i64).collect();
                result.add_column(
                    "index".to_string(),
                    Series::new(values, Some("index".to_string()))?,
                )?;
            }
        }

        // Carry every original column through unchanged, preserving its
        // exact type (see `gather_column_typed`).
        let all_positions: Vec<usize> = (0..row_count).collect();
        for col_name in self.column_names() {
            gather_column_typed(&mut result, self, col_name, &all_positions)?;
        }

        // `result.index` is left unset, which this DataFrame's own model
        // treats as the implicit default RangeIndex -- i.e. a fresh range
        // index, exactly as pandas' `reset_index()` installs.
        Ok(result)
    }

    fn set_index(&self, column: &str) -> Result<DataFrame> {
        if !self.contains_column(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }

        // Promote the column's values into the row index, preserving them
        // as the new labels (this DataFrame's row index is string-keyed,
        // so the column's string representation becomes the label set).
        // `Index::new` rejects duplicate labels crate-wide, so a column
        // with repeated values surfaces that as an honest error here too,
        // rather than silently building a lossy/ambiguous index.
        let label_values = self.get_column_string_values(column)?;
        let new_index = crate::index::Index::with_name(label_values, Some(column.to_string()))?;

        let mut result = self.drop_columns(&[column.to_string()])?;
        // Fully-qualified: see `select_rows_by_positions` for why plain
        // `result.set_index(..)` would resolve to this very trait method
        // (a `column: &str` parameter, so `new_index: Index<String>`
        // wouldn't even type-check against it) rather than `DataFrame`'s
        // in-place setter.
        DataFrame::set_index(&mut result, new_index)?;
        Ok(result)
    }

    fn set_multi_index(&self, columns: &[String]) -> Result<(DataFrame, MultiLevelIndex)> {
        let mut level_values = Vec::new();
        let names = columns.to_vec();

        for col_name in columns {
            let values = self.get_column_string_values(col_name)?;
            level_values.push(values);
        }

        let multi_index = MultiLevelIndex::new(names.clone(), level_values)?;
        let result_df = self.drop_columns(columns)?;

        Ok((result_df, multi_index))
    }

    fn select_columns(&self, columns: &[String]) -> Result<DataFrame> {
        // Delegate to `DataFrame::select_columns` (the inherent method,
        // which Rust's method resolution always prefers over this trait
        // method when both apply -- so this is not infinite recursion):
        // it preserves each column's exact concrete element type, instead
        // of rebuilding every column as `Series<String>` the way this
        // method used to.
        let column_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.select_columns(&column_refs)
    }

    fn drop_columns(&self, columns: &[String]) -> Result<DataFrame> {
        // Keep columns in their *original* order (the previous
        // `HashSet::difference` gave an unspecified, run-to-run
        // nondeterministic order), and delegate to the dtype-preserving
        // inherent `select_columns` for the rest.
        let to_drop: HashSet<&str> = columns.iter().map(|s| s.as_str()).collect();
        let to_keep: Vec<&str> = self
            .column_names()
            .iter()
            .map(|s| s.as_str())
            .filter(|name| !to_drop.contains(name))
            .collect();

        self.select_columns(&to_keep)
    }

    fn sample(&self, n: usize) -> Result<DataFrame> {
        use scirs2_core::random::SliceRandom;

        let row_count = self.row_count();
        if n >= row_count {
            return Ok(self.clone());
        }

        let mut indices: Vec<usize> = (0..row_count).collect();
        indices.shuffle(&mut scirs2_core::random::rng());
        indices.truncate(n);

        let iloc = self.iloc();
        iloc.get_positions(&indices)
    }

    fn head(&self, n: usize) -> Result<DataFrame> {
        let iloc = self.iloc();
        iloc.get_range(0..n.min(self.row_count()))
    }

    fn tail(&self, n: usize) -> Result<DataFrame> {
        let row_count = self.row_count();
        let start = if n >= row_count { 0 } else { row_count - n };
        let iloc = self.iloc();
        iloc.get_range(start..row_count)
    }
}

/// Helper functions for creating selectors
pub mod selectors {
    use super::*;

    /// Create a row selector for single index
    pub fn row(index: String) -> RowSelector {
        RowSelector::Single(index)
    }

    /// Create a row selector for multiple indices
    pub fn rows(indices: Vec<String>) -> RowSelector {
        RowSelector::Multiple(indices)
    }

    /// Create a row selector for single position
    pub fn pos(position: usize) -> RowSelector {
        RowSelector::Position(position)
    }

    /// Create a row selector for multiple positions
    pub fn positions(positions: Vec<usize>) -> RowSelector {
        RowSelector::Positions(positions)
    }

    /// Create a row selector for boolean mask
    pub fn mask(mask: Vec<bool>) -> RowSelector {
        RowSelector::Boolean(mask)
    }

    /// Create a column selector for single column
    pub fn col(name: String) -> ColumnSelector {
        ColumnSelector::Single(name)
    }

    /// Create a column selector for multiple columns
    pub fn cols(names: Vec<String>) -> ColumnSelector {
        ColumnSelector::Multiple(names)
    }

    /// Create a range selector
    pub fn range(start: usize, end: usize) -> RowSelector {
        RowSelector::Range(IndexRange::Standard { start, end })
    }

    /// Create an inclusive range selector
    pub fn range_inclusive(start: usize, end: usize) -> RowSelector {
        RowSelector::Range(IndexRange::Inclusive { start, end })
    }
}

/// Macro for convenient indexing
#[macro_export]
macro_rules! iloc {
    ($df:expr, $row:expr) => {
        $df.iloc().get($row)
    };
    ($df:expr, $row:expr, $col:expr) => {
        $df.iloc().get_at($row, $col)
    };
    ($df:expr, $rows:expr, $cols:expr) => {
        $df.iloc().get_slice($rows, $cols)
    };
}

#[macro_export]
macro_rules! loc {
    ($df:expr, $label:expr) => {
        $df.loc().get($label)
    };
    ($df:expr, $label:expr, $col:expr) => {
        $df.loc().get_at($label, $col)
    };
}

#[macro_export]
macro_rules! select {
    ($df:expr, rows: $rows:expr) => {
        $df.select().rows($rows).select()
    };
    ($df:expr, cols: $cols:expr) => {
        $df.select().columns($cols).select()
    };
    ($df:expr, rows: $rows:expr, cols: $cols:expr) => {
        $df.select().rows($rows).columns($cols).select()
    };
}

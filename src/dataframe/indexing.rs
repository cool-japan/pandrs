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
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

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

/// Position-based indexer (.iloc)
pub struct ILocIndexer<'a> {
    dataframe: &'a DataFrame,
}

impl<'a> ILocIndexer<'a> {
    pub fn new(dataframe: &'a DataFrame) -> Self {
        Self { dataframe }
    }

    /// Select by single position
    pub fn get(&self, row: usize) -> Result<HashMap<String, String>> {
        if row >= self.dataframe.row_count() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: self.dataframe.row_count(),
            });
        }

        let mut result = HashMap::new();
        for col_name in self.dataframe.column_names() {
            let values = self.dataframe.get_column_string_values(&col_name)?;
            if row < values.len() {
                result.insert(col_name, values[row].clone());
            }
        }
        Ok(result)
    }

    /// Select by row and column positions
    pub fn get_at(&self, row: usize, col: usize) -> Result<String> {
        let col_names = self.dataframe.column_names();
        if col >= col_names.len() {
            return Err(Error::IndexOutOfBounds {
                index: col,
                size: col_names.len(),
            });
        }

        let col_name = &col_names[col];
        let values = self.dataframe.get_column_string_values(col_name)?;

        if row >= values.len() {
            return Err(Error::IndexOutOfBounds {
                index: row,
                size: values.len(),
            });
        }

        Ok(values[row].clone())
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
            .into_iter()
            .skip(cols.start)
            .take(cols.end - cols.start)
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
    fn select_rows(&self, selector: RowSelector) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        let indices = match selector {
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

        // Create filtered columns
        for col_name in self.dataframe.column_names() {
            let column_values = self.dataframe.get_column_string_values(&col_name)?;
            let filtered_values: Vec<String> = indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();

            let filtered_series = Series::new(filtered_values, Some(col_name.clone()))?;
            result.add_column(col_name.clone(), filtered_series)?
        }

        Ok(result)
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
        let values = self.dataframe.get_column_string_values(column)?;

        if position >= values.len() {
            return Err(Error::IndexOutOfBounds {
                index: position,
                size: values.len(),
            });
        }

        Ok(values[position].clone())
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

    /// Find position of a label
    fn find_label_position(&self, label: &str) -> Result<usize> {
        // For now, assume simple integer-based labels
        // This would be enhanced with proper index support
        label
            .parse::<usize>()
            .map_err(|_| Error::InvalidValue(format!("Label '{}' not found in index", label)))
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

    /// Set single scalar value by label and column
    pub fn set(&self, label: &str, column: &str, value: String) -> Result<DataFrame> {
        // For immutable operations, return a new DataFrame with the value set
        let _result = self.dataframe.clone();

        // This would require mutable operations on DataFrame
        // For now, return an error indicating not implemented
        Err(Error::NotImplemented(
            "Mutable .at operations not yet implemented".to_string(),
        ))
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

    /// Set single scalar value by row and column positions
    pub fn set(&self, row: usize, col: usize, value: String) -> Result<DataFrame> {
        // For immutable operations, return a new DataFrame with the value set
        let _result = self.dataframe.clone();

        // This would require mutable operations on DataFrame
        // For now, return an error indicating not implemented
        Err(Error::NotImplemented(
            "Mutable .iat operations not yet implemented".to_string(),
        ))
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
        let mut result = self.dataframe.clone();

        // Apply row selection first
        if let Some(row_selector) = self.row_selector {
            let iloc = ILocIndexer::new(&result);
            result = iloc.select_rows(row_selector)?;
        }

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

    /// Extend DataFrame to target length by repeating values or adding NaNs
    fn extend_dataframe(df: &DataFrame, target_len: usize) -> Result<DataFrame> {
        if df.row_count() >= target_len {
            return Ok(df.clone());
        }

        let mut result = DataFrame::new();
        let current_len = df.row_count();

        for col_name in df.column_names() {
            let values = df.get_column_string_values(&col_name)?;
            let mut extended_values = values.clone();

            // Extend with "NaN" values or repeat last value
            for i in current_len..target_len {
                if values.is_empty() {
                    extended_values.push("NaN".to_string());
                } else {
                    extended_values.push(values[i % values.len()].clone());
                }
            }

            let extended_series = Series::new(extended_values, Some(col_name.clone()))?;
            result.add_column(col_name, extended_series)?;
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

    /// Reindex DataFrame with new index
    pub fn reindex(df: &DataFrame, new_index: &[String]) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col_name in df.column_names() {
            let current_values = df.get_column_string_values(&col_name)?;
            let mut reindexed_values = Vec::with_capacity(new_index.len());

            for index_val in new_index {
                // For simple implementation, try to parse as position
                if let Ok(pos) = index_val.parse::<usize>() {
                    if pos < current_values.len() {
                        reindexed_values.push(current_values[pos].clone());
                    } else {
                        reindexed_values.push("NaN".to_string());
                    }
                } else {
                    reindexed_values.push("NaN".to_string());
                }
            }

            let reindexed_series = Series::new(reindexed_values, Some(col_name.clone()))?;
            result.add_column(col_name, reindexed_series)?;
        }

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
        // Return the DataFrame as-is since we don't store explicit index
        Ok(self.clone())
    }

    fn set_index(&self, column: &str) -> Result<DataFrame> {
        // For now, just remove the column and return the rest
        self.drop_columns(&[column.to_string()])
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
        let column_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        let mut result = DataFrame::new();

        for col_name in &column_refs {
            if !self.contains_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }

            let values = self.get_column_string_values(col_name)?;
            let series = Series::new(values, Some(col_name.to_string()))?;
            result.add_column(col_name.to_string(), series)?;
        }

        Ok(result)
    }

    fn drop_columns(&self, columns: &[String]) -> Result<DataFrame> {
        let all_columns: HashSet<String> = self.column_names().into_iter().collect();
        let to_drop: HashSet<String> = columns.iter().cloned().collect();
        let to_keep: Vec<String> = all_columns.difference(&to_drop).cloned().collect();
        let to_keep_refs: Vec<&str> = to_keep.iter().map(|s| s.as_str()).collect();

        self.select_columns(&to_keep_refs)
    }

    fn sample(&self, n: usize) -> Result<DataFrame> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let row_count = self.row_count();
        if n >= row_count {
            return Ok(self.clone());
        }

        let mut indices: Vec<usize> = (0..row_count).collect();
        indices.shuffle(&mut rng());
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

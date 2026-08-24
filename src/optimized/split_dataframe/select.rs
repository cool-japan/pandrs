//! Selection functionality for OptimizedDataFrame
//!
//! This module owns the *canonical* row-materialization routine
//! (`take_rows`). Every row-selecting operation of the optimized DataFrame
//! (filter, head/tail, sample, sort, group extraction, index lookup, ...)
//! delegates to it so that NULL semantics, column order and index labels are
//! handled in exactly one place.

use std::collections::HashSet;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::core::error::OptionExt;
use crate::error::Result;
use crate::index::DataFrameIndex;
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

/// Amount of work (selected rows x columns) above which column materialization
/// is spread over the rayon thread pool. Below the threshold the serial path is
/// faster because it avoids the task-scheduling overhead (group-by extraction
/// issues one call per group, so small selections are the common case).
const PARALLEL_TAKE_THRESHOLD: usize = 8192;

impl OptimizedDataFrame {
    /// Select columns to create a new DataFrame
    ///
    /// # Arguments
    /// * `columns` - Array of column names to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected columns
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        let mut df = Self::new();

        // Create a set of column names (for existence check)
        let column_set: HashSet<&str> = self.column_names.iter().map(|s| s.as_str()).collect();

        // Add specified columns to the new DataFrame
        for &col_name in columns {
            if !column_set.contains(col_name) {
                // Return error if column doesn't exist
                return Err(crate::error::Error::ColumnNotFound(col_name.to_string()));
            }

            let col_idx = self
                .column_indices
                .get(col_name)
                .ok_or_column_error(col_name)?;
            let column = &self.columns[*col_idx];

            df.add_column(col_name.to_string(), column.clone())?;
        }

        // Copy index
        if let Some(ref index) = self.index {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Select rows by index
    ///
    /// NULL values, column order and index labels of the source frame are
    /// preserved. Positions outside the frame are ignored.
    ///
    /// # Arguments
    /// * `indices` - Array of row indices to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected rows
    ///
    /// Note: A method with the same name exists in sort.rs but that one is private
    pub fn select_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        let mut df = take_rows(self, indices)?;

        // Frames without an index keep the historical behaviour of this entry
        // point: a fresh sequential index is attached to the result.
        if df.index.is_none() {
            df.set_default_index()?;
        }

        Ok(df)
    }

    /// Select both rows and columns
    ///
    /// # Arguments
    /// * `row_indices` - Array of row indices to select
    /// * `columns` - Array of column names to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected rows and columns
    pub fn select_rows_columns(&self, row_indices: &[usize], columns: &[&str]) -> Result<Self> {
        // First select columns
        let cols_selected = self.select_columns(columns)?;

        // Then select rows
        cols_selected.select_rows_by_indices(row_indices)
    }

    /// Select rows using a mask
    ///
    /// # Arguments
    /// * `mask` - Boolean vector representing selection condition (True rows are selected)
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with rows matching the condition
    pub fn select_by_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.row_count {
            return Err(crate::error::Error::Format(format!(
                "Mask length ({}) does not match DataFrame row count ({})",
                mask.len(),
                self.row_count
            )));
        }

        // Create a list of indices from the mask
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();

        // Execute selection by indices
        self.select_rows_by_indices(&indices)
    }
}

/// Materialize the given rows of `df` into a new DataFrame.
///
/// This is the single implementation shared by every row-selecting operation.
/// Its guarantees are:
///
/// * **NULL preservation** - a missing value stays missing. Values are never
///   replaced by `0`/`0.0`/`""`/`false`; a null mask is rebuilt for the result.
/// * **Deterministic schema** - columns are emitted in `df.column_names()`
///   order (never in `HashMap` iteration order).
/// * **Index integrity** - the labels of the selected rows are re-selected from
///   the source index (both `Simple` and `Multi` indexes) instead of cloning the
///   full-length index onto a shorter frame.
/// * **Alignment** - positions outside the frame are dropped once, up front, so
///   every column and the index agree on the very same row set.
///
/// Limitation: `Index<String>` cannot represent duplicate labels, so a selection
/// that repeats rows (e.g. sampling with replacement) falls back to a positional
/// index rather than failing the whole operation.
pub(crate) fn take_rows(df: &OptimizedDataFrame, indices: &[usize]) -> Result<OptimizedDataFrame> {
    // Drop out-of-range positions once so that every column and the index are
    // built from the same set of rows.
    let positions: Vec<usize> = if indices.iter().all(|&i| i < df.row_count) {
        indices.to_vec()
    } else {
        indices
            .iter()
            .copied()
            .filter(|&i| i < df.row_count)
            .collect()
    };

    let mut result = OptimizedDataFrame::new();

    // Materialize the columns in declared order (deterministic).
    let resolve = |name: &String| -> Result<Column> {
        let column_idx = *df
            .column_indices
            .get(name)
            .ok_or_else(|| crate::error::Error::ColumnNotFound(name.clone()))?;
        let column = df
            .columns
            .get(column_idx)
            .ok_or_else(|| crate::error::Error::ColumnNotFound(name.clone()))?;
        Ok(take_column(column, &positions))
    };

    let workload = positions.len().saturating_mul(df.column_names.len());
    let taken: Vec<Column> = if workload >= PARALLEL_TAKE_THRESHOLD {
        use rayon::prelude::*;
        df.column_names
            .par_iter()
            .map(&resolve)
            .collect::<Result<Vec<Column>>>()?
    } else {
        df.column_names
            .iter()
            .map(&resolve)
            .collect::<Result<Vec<Column>>>()?
    };

    for (name, column) in df.column_names.iter().zip(taken) {
        result.add_column(name.clone(), column)?;
    }

    // Re-select the index labels of the selected rows. Frames without columns
    // have no row count to align an index against, so they keep no index.
    if !df.column_names.is_empty() {
        if let Some(ref index) = df.index {
            match index {
                DataFrameIndex::Simple(simple_idx) => {
                    let values: Vec<String> = positions
                        .iter()
                        .map(|&pos| {
                            simple_idx
                                .get_value(pos)
                                .cloned()
                                .unwrap_or_else(|| pos.to_string())
                        })
                        .collect();

                    match crate::index::Index::with_name(values, simple_idx.name().cloned()) {
                        Ok(new_index) => result.set_index_from_simple_index(new_index)?,
                        // Duplicate labels (repeated positions) cannot be stored
                        // in an Index<String>; fall back to positional labels.
                        Err(_) => result.set_default_index()?,
                    }
                }
                DataFrameIndex::Multi(multi_idx) => {
                    let tuples: Option<Vec<Vec<String>>> = positions
                        .iter()
                        .map(|&pos| multi_idx.get_tuple(pos))
                        .collect();

                    match tuples {
                        Some(tuples) if !tuples.is_empty() => {
                            let level_names: Vec<Option<String>> = multi_idx.names().to_vec();
                            match crate::index::MultiIndex::from_tuples(tuples, Some(level_names)) {
                                Ok(new_index) => result.set_index_from_multi_index(new_index)?,
                                Err(_) => result.set_default_index()?,
                            }
                        }
                        // Empty selection or an index shorter than the frame:
                        // no tuple can be reconstructed, use positional labels.
                        _ => result.set_default_index()?,
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Materialize one column for the given (already validated) row positions,
/// carrying the NULL mask over to the new column.
pub(crate) fn take_column(column: &Column, positions: &[usize]) -> Column {
    match column {
        Column::Int64(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for &pos in positions {
                match col.get(pos) {
                    Ok(Some(value)) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    _ => {
                        values.push(0);
                        nulls.push(true);
                    }
                }
            }
            let mut new_col = Int64Column::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Int64(new_col)
        }
        Column::Float64(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for &pos in positions {
                match col.get(pos) {
                    Ok(Some(value)) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    _ => {
                        values.push(0.0);
                        nulls.push(true);
                    }
                }
            }
            let mut new_col = Float64Column::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Float64(new_col)
        }
        Column::String(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for &pos in positions {
                match col.get(pos) {
                    Ok(Some(value)) => {
                        values.push(value.to_string());
                        nulls.push(false);
                    }
                    _ => {
                        values.push(String::new());
                        nulls.push(true);
                    }
                }
            }
            let mut new_col = StringColumn::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::String(new_col)
        }
        Column::Boolean(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for &pos in positions {
                match col.get(pos) {
                    Ok(Some(value)) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    _ => {
                        values.push(false);
                        nulls.push(true);
                    }
                }
            }
            let mut new_col = BooleanColumn::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Boolean(new_col)
        }
    }
}

/// Implementation for selecting rows based on row indices (used by other modules)
///
/// Delegates to `take_rows`; an empty selection yields a frame with the same
/// schema (all columns, zero rows) instead of a frame without columns.
pub(crate) fn select_rows_by_indices_impl(
    df: &OptimizedDataFrame,
    indices: &[usize],
) -> Result<OptimizedDataFrame> {
    take_rows(df, indices)
}

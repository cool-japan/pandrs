//! Parallel processing functionality for OptimizedDataFrame

use rayon::prelude::*;

use super::core::OptimizedDataFrame;
use super::select::take_rows;
use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;

/// Number of *selected rows* above which a single column's gather is spread
/// over the rayon thread pool via a real `par_iter` (as opposed to the
/// `chunks()` loop this file used to run serially and call "parallel").
/// Column-count-independent by design: a 1-column frame with this many rows
/// selected must still parallelize, since there is no second column for
/// column-level parallelism to fall back on. Kept at the same order of
/// magnitude as `select::PARALLEL_TAKE_THRESHOLD` (8192 *cells*), which
/// tunes the equivalent all-columns gather in [`take_rows`].
const ROW_GATHER_PARALLEL_THRESHOLD: usize = 8192;

impl OptimizedDataFrame {
    /// Parallel row filtering
    ///
    /// Extracts only rows where the value in the condition column (Boolean type) is true,
    /// applying parallel processing for large datasets.
    ///
    /// # Arguments
    /// * `condition_column` - Name of Boolean column to use as filter condition
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        // Threshold for optimal parallelization (smaller datasets benefit from serial processing)
        const PARALLEL_THRESHOLD: usize = 100_000;

        // Get condition column
        let column_idx = self
            .column_indices
            .get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;

        let condition = &self.columns[*column_idx];

        // Verify the condition column is boolean type
        if let Column::Boolean(bool_col) = condition {
            let row_count = bool_col.len();

            // Choose serial/parallel processing based on data size
            let indices: Vec<usize> = if row_count < PARALLEL_THRESHOLD {
                // Serial processing (small data)
                (0..row_count)
                    .filter_map(|i| {
                        if let Ok(Some(true)) = bool_col.get(i) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // Parallel processing (large data)
                // Optimize chunk size to reduce parallelization overhead
                let chunk_size = (row_count / rayon::current_num_threads()).max(1000);

                // First convert range to array, then process chunks
                (0..row_count)
                    .collect::<Vec<_>>()
                    .par_chunks(chunk_size)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .filter_map(|&i| {
                                if let Ok(Some(true)) = bool_col.get(i) {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };

            // `indices` are always < row_count in this crate today (every
            // column reaches `self.columns` through `add_column`, which
            // rejects a length other than `self.row_count` outright, and the
            // one direct in-place column replacement -- `apply.rs`'s
            // `result.columns[*col_idx] = new_column` -- always rebuilds the
            // replacement at the same length as the column it replaces) but
            // filtering defensively costs one cheap linear pass and matches
            // `take_rows`, which guards the same way for the same reason.
            let indices: Vec<usize> = indices
                .into_iter()
                .filter(|&i| i < self.row_count)
                .collect();

            // Decide, once, whether materializing the selected rows is worth
            // spreading over the thread pool at all. Below the threshold,
            // delegate straight to the canonical row-materialization routine
            // ([`take_rows`]): it is already correct (typed, NULL-preserving,
            // re-selects the index) and small selections aren't worth a
            // second, hand-rolled code path.
            let row_parallel = indices.len() >= ROW_GATHER_PARALLEL_THRESHOLD;
            let workload = indices.len().saturating_mul(self.column_names.len());
            // Mutually exclusive by design, not merely by the `!row_parallel`
            // guard being convenient: once the per-column gather is already
            // spread across rows (`row_parallel`), doing so once per column
            // already saturates the thread pool for that column, so *also*
            // running `column_names.par_iter()` outside it would only add
            // nested-task scheduling overhead for no extra throughput.
            // `col_parallel` exists purely to cover the complementary case --
            // many columns, each individually below the row threshold, where
            // row-level parallelism cannot fire at all but the aggregate
            // per-column work still adds up.
            let col_parallel = !row_parallel
                && self.column_names.len() > 1
                && workload >= ROW_GATHER_PARALLEL_THRESHOLD;

            if !row_parallel && !col_parallel {
                return take_rows(self, &indices);
            }

            // Large selection: gather every column ourselves so the row-level
            // work (the dominant cost here) actually runs in parallel, which
            // `take_rows` alone cannot guarantee -- its own parallelism is
            // column-level (`column_names.par_iter()`), which is a no-op for
            // a frame with only one or two columns. Falling back to
            // `take_rows` afterwards just to re-derive the index would pay
            // for this frame's dominant cost (the per-column gather) twice,
            // so the index is re-selected locally instead (mirrors
            // `select::take_rows`'s Simple/Multi handling exactly; see
            // `resubset_index` below).
            let mut result = Self::new();

            let gather = |name: &String| {
                let i = self.column_indices[name];
                let column = take_column_row_aware(&self.columns[i], &indices, row_parallel);
                (name.clone(), column)
            };

            let result_columns: Vec<(String, Column)> = if col_parallel {
                self.column_names.par_iter().map(gather).collect()
            } else {
                self.column_names.iter().map(gather).collect()
            };

            for (name, column) in result_columns {
                result.add_column(name, column)?;
            }

            if !self.column_names.is_empty() {
                if let Some(ref index) = self.index {
                    resubset_index(index, &indices, &mut result)?;
                }
            }

            Ok(result)
        } else {
            Err(Error::OperationFailed(format!(
                "Column '{}' is not of boolean type",
                condition_column
            )))
        }
    }
}

/// Materialize one column for `indices`, preserving NULLs the same way
/// [`super::select::take_column`] does: a NULL slot still needs *some*
/// placeholder value to keep the data vector the same length as the null
/// mask, so it gets the type's zero value, but the null bitmap marks that
/// slot NULL, so no reader ever observes the placeholder as real data.
/// Nothing is substituted for an existing value -- only genuinely-missing
/// slots get the (masked-out) placeholder.
///
/// Gathers via a real `par_iter().unzip()` (not a `chunks()` loop that
/// nobody parallelized) when `parallel` is true, so a 1-column frame
/// benefits exactly like a many-column one.
fn take_column_row_aware(column: &Column, indices: &[usize], parallel: bool) -> Column {
    match column {
        Column::Int64(col) => {
            let (values, nulls): (Vec<i64>, Vec<bool>) = if parallel {
                indices
                    .par_iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (0, true),
                    })
                    .unzip()
            } else {
                indices
                    .iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (0, true),
                    })
                    .unzip()
            };
            let mut new_col = Int64Column::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Int64(new_col)
        }
        Column::Float64(col) => {
            let (values, nulls): (Vec<f64>, Vec<bool>) = if parallel {
                indices
                    .par_iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (0.0, true),
                    })
                    .unzip()
            } else {
                indices
                    .iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (0.0, true),
                    })
                    .unzip()
            };
            let mut new_col = Float64Column::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Float64(new_col)
        }
        Column::String(col) => {
            let (values, nulls): (Vec<String>, Vec<bool>) = if parallel {
                indices
                    .par_iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v.to_string(), false),
                        _ => (String::new(), true),
                    })
                    .unzip()
            } else {
                indices
                    .iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v.to_string(), false),
                        _ => (String::new(), true),
                    })
                    .unzip()
            };
            let mut new_col = StringColumn::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::String(new_col)
        }
        Column::Boolean(col) => {
            let (values, nulls): (Vec<bool>, Vec<bool>) = if parallel {
                indices
                    .par_iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (false, true),
                    })
                    .unzip()
            } else {
                indices
                    .iter()
                    .map(|&idx| match col.get(idx) {
                        Ok(Some(v)) => (v, false),
                        _ => (false, true),
                    })
                    .unzip()
            };
            let mut new_col = BooleanColumn::with_nulls(values, nulls);
            if let Some(name) = col.get_name() {
                new_col.set_name(name.to_string());
            }
            Column::Boolean(new_col)
        }
    }
}

/// Re-select `source_index`'s labels onto `result` for `positions`.
///
/// Mirrors [`super::select::take_rows`]'s index handling exactly (duplicated
/// rather than shared because the row-parallel fast path above only exists
/// to avoid the redundant full serial gather that calling `take_rows` a
/// second time would cost; see the comment at its call site). Keep this in
/// sync with `take_rows` if index-resubsetting behaviour ever changes.
fn resubset_index(
    source_index: &DataFrameIndex<String>,
    positions: &[usize],
    result: &mut OptimizedDataFrame,
) -> Result<()> {
    match source_index {
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
                // Duplicate labels (repeated positions) cannot be stored in
                // an Index<String>; fall back to positional labels.
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
                _ => result.set_default_index()?,
            }
        }
    }
    Ok(())
}

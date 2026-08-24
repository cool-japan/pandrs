//! Row sorting for [`OptimizedDataFrame`].
//!
//! # Ordering semantics
//!
//! * Missing values (NULL) are always placed **last**, independent of the sort
//!   direction. This matches the pandas default `na_position="last"`.
//! * For `Float64` columns `NaN` is treated as a missing value (as pandas does
//!   for `float64`, where `NaN` *is* the missing marker) and therefore also
//!   sorts last in both directions. All remaining floats are compared with
//!   [`f64::total_cmp`], which is a genuine total order — `partial_cmp` plus a
//!   fallback is **not** transitive and leaves data unsorted around `NaN`.
//! * Sorting is stable, so rows that compare equal keep their relative order.
//!
//! Note: [`crate::series::na`] currently sorts NA *first*; the ordering used
//! here (NA last) is the pandas-compatible one and is treated as canonical for
//! DataFrame-level sorts.

use std::cmp::Ordering;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

/// Order two optional sort keys, placing missing values last in both
/// directions and applying `ascending` only to the present-value case.
#[inline]
fn order_with_na<T, C>(a: Option<&T>, b: Option<&T>, ascending: bool, cmp: &C) -> Ordering
where
    C: Fn(&T, &T) -> Ordering,
{
    match (a, b) {
        (None, None) => Ordering::Equal,
        // NA sorts last regardless of direction (pandas `na_position="last"`).
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => {
            let ordering = cmp(x, y);
            if ascending {
                ordering
            } else {
                ordering.reverse()
            }
        }
    }
}

/// Total order for `f64` sort keys.
#[inline]
fn total_cmp_f64(a: &f64, b: &f64) -> Ordering {
    a.total_cmp(b)
}

/// Read a float sort key, mapping `NaN` onto the missing-value case.
#[inline]
fn float_sort_key(col: &Float64Column, row: usize) -> Option<f64> {
    col.get(row).ok().flatten().filter(|value| !value.is_nan())
}

/// Decorate-sort-undecorate helper: materialise one sort key per row, sort the
/// keys once and return the resulting row permutation.
///
/// Extracting the key once per row (instead of inside the comparator) keeps the
/// comparison itself allocation-free and avoids `O(n log n)` column lookups.
fn sorted_indices<T, K, C>(row_count: usize, ascending: bool, key: K, cmp: C) -> Vec<usize>
where
    K: Fn(usize) -> Option<T>,
    C: Fn(&T, &T) -> Ordering,
{
    let mut pairs: Vec<(usize, Option<T>)> = (0..row_count).map(|idx| (idx, key(idx))).collect();

    // `sort_by` is stable: equal keys keep their original row order.
    pairs.sort_by(|a, b| order_with_na(a.1.as_ref(), b.1.as_ref(), ascending, &cmp));

    pairs.into_iter().map(|(idx, _)| idx).collect()
}

/// A sort column resolved to its concrete typed representation.
///
/// Resolving happens once, before sorting, so the comparator performs neither a
/// `HashMap` lookup nor a downcast nor a `String` allocation per comparison.
enum SortColumnRef<'a> {
    Int64(&'a Int64Column),
    Float64(&'a Float64Column),
    String(&'a StringColumn),
    Boolean(&'a BooleanColumn),
}

impl<'a> SortColumnRef<'a> {
    /// Resolve a column enum into a typed reference (total, no failure mode).
    fn new(column: &'a Column) -> Self {
        match column {
            Column::Int64(col) => SortColumnRef::Int64(col),
            Column::Float64(col) => SortColumnRef::Float64(col),
            Column::String(col) => SortColumnRef::String(col),
            Column::Boolean(col) => SortColumnRef::Boolean(col),
        }
    }

    /// Compare two rows of this column, honouring NA-last semantics.
    #[inline]
    fn compare_rows(&self, a: usize, b: usize, ascending: bool) -> Ordering {
        match self {
            SortColumnRef::Int64(col) => order_with_na(
                col.get(a).ok().flatten().as_ref(),
                col.get(b).ok().flatten().as_ref(),
                ascending,
                &i64::cmp,
            ),
            SortColumnRef::Float64(col) => order_with_na(
                float_sort_key(col, a).as_ref(),
                float_sort_key(col, b).as_ref(),
                ascending,
                &total_cmp_f64,
            ),
            SortColumnRef::String(col) => order_with_na(
                col.get(a).ok().flatten().as_ref(),
                col.get(b).ok().flatten().as_ref(),
                ascending,
                &|x: &&str, y: &&str| x.cmp(y),
            ),
            SortColumnRef::Boolean(col) => order_with_na(
                col.get(a).ok().flatten().as_ref(),
                col.get(b).ok().flatten().as_ref(),
                ascending,
                &bool::cmp,
            ),
        }
    }
}

impl OptimizedDataFrame {
    /// Sort DataFrame by column
    ///
    /// NULL values (and `NaN` in float columns) are placed last in both
    /// ascending and descending order, matching pandas' `na_position="last"`.
    ///
    /// # Arguments
    /// * `by` - Name of the column to sort by
    /// * `ascending` - Whether to sort in ascending order
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame that is sorted
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // Get column index
        let column_idx = *self
            .column_indices
            .get(by)
            .ok_or_else(|| Error::ColumnNotFound(by.to_string()))?;

        let column = self
            .columns
            .get(column_idx)
            .ok_or_else(|| Error::ColumnNotFound(by.to_string()))?;

        let row_count = self.row_count();

        // Resolve the column once, then sort by a pre-extracted typed key.
        let indices = match column {
            Column::Int64(col) => sorted_indices(
                row_count,
                ascending,
                |idx| col.get(idx).ok().flatten(),
                i64::cmp,
            ),
            Column::Float64(col) => sorted_indices(
                row_count,
                ascending,
                |idx| float_sort_key(col, idx),
                total_cmp_f64,
            ),
            // `Option<&str>` keys: no per-row and no per-comparison allocation.
            Column::String(col) => sorted_indices(
                row_count,
                ascending,
                |idx| col.get(idx).ok().flatten(),
                |x: &&str, y: &&str| x.cmp(y),
            ),
            Column::Boolean(col) => sorted_indices(
                row_count,
                ascending,
                |idx| col.get(idx).ok().flatten(),
                bool::cmp,
            ),
        };

        // Create a new DataFrame from sorted row indices
        self.select_rows_by_indices_internal(&indices)
    }

    /// Sort DataFrame by multiple columns
    ///
    /// Columns are compared left to right; NULL values (and `NaN` in float
    /// columns) sort last within every key column, in both directions.
    ///
    /// # Arguments
    /// * `by` - Array of column names to sort by
    /// * `ascending` - Array of boolean flags indicating ascending/descending order for each column (if None, all are ascending)
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame that is sorted
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        if by.is_empty() {
            return Err(Error::EmptyColumnList);
        }

        // Set ascending array
        let is_ascending: Vec<bool> = match ascending {
            Some(asc) => {
                if asc.len() != by.len() {
                    return Err(Error::InconsistentArrayLengths {
                        expected: by.len(),
                        found: asc.len(),
                    });
                }
                asc.to_vec()
            }
            None => vec![true; by.len()], // Default is ascending
        };

        // Resolve every sort column exactly once. This both validates the
        // column names and removes the per-comparison lookup/downcast.
        let mut sort_columns: Vec<(SortColumnRef<'_>, bool)> = Vec::with_capacity(by.len());
        for (&col_name, &asc) in by.iter().zip(is_ascending.iter()) {
            let column_idx = *self
                .column_indices
                .get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            let column = self
                .columns
                .get(column_idx)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            sort_columns.push((SortColumnRef::new(column), asc));
        }

        // Create an array of row indices (from 0 to row_count-1)
        let mut indices: Vec<usize> = (0..self.row_count()).collect();

        // Sort by multiple keys (stable, so ties keep their original order)
        indices.sort_by(|&a, &b| {
            for (column, asc) in &sort_columns {
                let cmp = column.compare_rows(a, b, *asc);
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        });

        // Create a new DataFrame from sorted row indices
        self.select_rows_by_indices_internal(&indices)
    }

    /// Select rows based on row indices (using implementation from select module)
    fn select_rows_by_indices_internal(&self, indices: &[usize]) -> Result<Self> {
        // Use function implemented in select.rs
        use crate::optimized::split_dataframe::select;
        select::select_rows_by_indices_impl(self, indices)
    }
}

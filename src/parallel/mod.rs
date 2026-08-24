//! Module providing parallel processing functionality

use crate::error::{Error, Result};
use crate::na::NA;
use crate::series::NASeries;
use crate::DataFrame;
use crate::Series;
use rayon::prelude::*;

/// Below this many rows/elements, a serial pass beats spawning rayon for
/// whole-DataFrame operations (`par_apply`, `par_filter_rows`, `par_groupby`
/// and the typed row-gather they share): thread-pool dispatch and
/// cross-thread result collection cost more than the per-row work they would
/// save. Kept in the same order of magnitude as
/// `optimized::split_dataframe::select::PARALLEL_TAKE_THRESHOLD` (8192),
/// which tunes the equivalent typed row-materialization routine for
/// `OptimizedDataFrame`.
const DATAFRAME_PARALLEL_THRESHOLD: usize = 8192;

/// Below this many elements, a serial pass beats rayon for per-element work
/// that is a single arithmetic/comparison op (`ParallelUtils::par_sum` /
/// `par_mean` / `par_min` / `par_max`, `Series`/`NASeries` `par_map` /
/// `par_filter`). The per-element cost here is far smaller than a
/// DataFrame-level row (no column lookup or downcast), so it takes many more
/// elements to amortize thread-pool dispatch overhead than the threshold
/// above.
const SCALAR_PARALLEL_THRESHOLD: usize = 100_000;

/// Parallel processing extension: Series parallel processing
impl<T> Series<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// Apply a function to all elements in parallel
    pub fn par_map<F, R>(&self, f: F) -> Series<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let source = self.values();
        let new_values: Vec<R> = if source.len() < SCALAR_PARALLEL_THRESHOLD {
            source.iter().map(&f).collect()
        } else {
            source.par_iter().map(&f).collect()
        };

        Series::new(new_values, self.name().cloned())
            .expect("parallel map produces valid series from valid input")
    }

    /// Filter elements in parallel based on a condition function
    pub fn par_filter<F>(&self, f: F) -> Series<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let source = self.values();
        let filtered_values: Vec<T> = if source.len() < SCALAR_PARALLEL_THRESHOLD {
            source.iter().filter(|v| f(v)).cloned().collect()
        } else {
            source.par_iter().filter(|v| f(v)).cloned().collect()
        };

        Series::new(filtered_values, self.name().cloned())
            .expect("parallel filter produces valid series from valid input")
    }
}

/// Parallel processing extension: Series with NA values
impl<T> NASeries<T>
where
    T: Clone + Send + Sync + 'static + std::fmt::Debug,
{
    /// Apply a function to all elements in parallel (ignoring NA)
    pub fn par_map<F, R>(&self, f: F) -> NASeries<R>
    where
        F: Fn(&T) -> R + Send + Sync,
        R: Clone + Send + Sync + 'static + std::fmt::Debug,
    {
        let map_one = |v: &NA<T>| match v {
            NA::Value(val) => NA::Value(f(val)),
            NA::NA => NA::NA,
        };

        let source = self.values();
        let new_values: Vec<NA<R>> = if source.len() < SCALAR_PARALLEL_THRESHOLD {
            source.iter().map(map_one).collect()
        } else {
            source.par_iter().map(map_one).collect()
        };

        NASeries::new(new_values, self.name().cloned())
            .expect("parallel map produces valid NA series from valid input")
    }

    /// Filter elements in parallel based on a condition function (excluding NA)
    pub fn par_filter<F>(&self, f: F) -> NASeries<T>
    where
        F: Fn(&T) -> bool + Send + Sync,
    {
        let keep_one = |v: &&NA<T>| match v {
            NA::Value(val) => f(val),
            NA::NA => false,
        };

        let source = self.values();
        let filtered_values: Vec<NA<T>> = if source.len() < SCALAR_PARALLEL_THRESHOLD {
            source.iter().filter(keep_one).cloned().collect()
        } else {
            source.par_iter().filter(keep_one).cloned().collect()
        };

        NASeries::new(filtered_values, self.name().cloned())
            .expect("parallel filter produces valid NA series from valid input")
    }
}

/// Parallel processing extension: DataFrame parallel processing
impl DataFrame {
    /// Apply a per-cell string transform to every column in parallel.
    ///
    /// `f(column_name, row_index, cell_as_str) -> new_cell_as_str` is the
    /// public contract: cells enter and leave as `&str`/`String` by design
    /// (this is a generic text-mapping utility, not a typed transform), so
    /// every result column is `Series<String>` regardless of the source
    /// column's type -- that is the documented behaviour, not an
    /// accidental stringification bug. Callers that need to keep a column's
    /// original dtype through the transform should use
    /// [`crate::optimized::split_dataframe::core::OptimizedDataFrame::par_apply`]
    /// instead, whose closure operates on a typed [`Column`](crate::column::Column)
    /// view.
    pub fn par_apply<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(&str, usize, &str) -> String + Send + Sync,
    {
        let mut result = DataFrame::new();

        // Process each column in parallel
        let column_names = self.column_names().to_vec();

        // Prepare row count and column names
        let n_rows = self.row_count();
        let parallel = n_rows >= DATAFRAME_PARALLEL_THRESHOLD;

        // Process each column
        for col_name in &column_names {
            // Get string values (already hoisted above the per-row loop, so
            // this is a single O(n_rows) pass per column, not O(n_rows^2)).
            let values = self.get_column_string_values(col_name)?;

            let map_one = |i: usize| {
                let val = if i < values.len() { &values[i] } else { "" };
                f(col_name, i, val)
            };

            // Create new values, serially for small frames (thread-pool
            // dispatch would cost more than the row-level work it saves).
            let new_values: Vec<String> = if parallel {
                (0..n_rows).into_par_iter().map(map_one).collect()
            } else {
                (0..n_rows).map(map_one).collect()
            };

            // Add new column
            let new_series = Series::new(new_values, Some(col_name.to_string()))?;
            result.add_column(col_name.to_string(), new_series)?;
        }

        Ok(result)
    }

    /// Filter rows in parallel
    ///
    /// Every result column keeps the source column's exact type (i64 stays
    /// i64, f64 stays f64, ...) and every kept value is moved with a plain
    /// `T::clone()` -- never routed through a `String` intermediate -- via
    /// `DataFrame::gather_rows_typed`.
    pub fn par_filter_rows<F>(&self, f: F) -> Result<DataFrame>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let n_rows = self.row_count();

        // Filter row indices, serially for small frames.
        let row_indices: Vec<usize> = if n_rows < DATAFRAME_PARALLEL_THRESHOLD {
            (0..n_rows).filter(|&i| f(i)).collect()
        } else {
            (0..n_rows).into_par_iter().filter(|&i| f(i)).collect()
        };

        self.gather_rows_typed(&row_indices)
    }

    /// Execute groupby operation in parallel
    pub fn par_groupby<K>(&self, key_func: K) -> Result<HashMap<String, DataFrame>>
    where
        K: Fn(usize) -> String + Send + Sync,
    {
        let n_rows = self.row_count();

        // Calculate keys for each row, serially for small frames. Collecting
        // into a `Vec` first (rather than folding a shared map directly)
        // keeps this order-preserving: an `IndexedParallelIterator` (a
        // `Range<usize>` is one) collects into a `Vec` in original order, so
        // the sequential grouping pass below always pushes each group's row
        // indices in ascending row order, matching plain serial iteration.
        let keyed: Vec<(String, usize)> = if n_rows < DATAFRAME_PARALLEL_THRESHOLD {
            (0..n_rows).map(|i| (key_func(i), i)).collect()
        } else {
            (0..n_rows)
                .into_par_iter()
                .map(|i| (key_func(i), i))
                .collect()
        };

        // Group map
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (key, idx) in keyed {
            groups.entry(key).or_insert_with(Vec::new).push(idx);
        }

        // Create DataFrame for each group by gathering that group's already
        // -known row indices directly -- O(group_size) per group, O(n_rows)
        // total. The previous implementation re-derived each group's rows
        // via `par_filter_rows(|i| indices.contains(&i))`, which rescanned
        // the *entire* row range per group and did an O(group_size) linear
        // `Vec::contains` scan per row: O(groups * n_rows * group_size)
        // overall (quadratic-plus in the row count).
        let mut result = HashMap::new();
        for (key, indices) in groups {
            let group_df = self.gather_rows_typed(&indices)?;
            result.insert(key, group_df);
        }

        Ok(result)
    }

    /// Typed, dtype- and NA-preserving row gather shared by
    /// [`par_filter_rows`](DataFrame::par_filter_rows) and
    /// [`par_groupby`](DataFrame::par_groupby).
    ///
    /// `indices` are row positions into `self`; every entry is expected to
    /// be `< self.row_count()` (both callers build `indices` from
    /// `0..self.row_count()`, so this always holds in practice -- an
    /// out-of-range entry still fails cleanly with `Error::IndexOutOfBounds`
    /// rather than panicking or silently truncating).
    ///
    /// Each column is downcast once to its concrete element type and the
    /// selected values are moved with a plain `T::clone()`; nothing is
    /// stringified and nothing is substituted for a missing/unreadable
    /// value, so whatever NA convention the source column already uses
    /// (e.g. `f64::NAN`, an `NA<T>` sentinel, ...) survives untouched. The
    /// candidate type list matches [`DataFrame::get_column_string_values`]
    /// exactly, so this supports precisely the column types that function
    /// does (and fails the same way -- `Error::InvalidValue` -- for a type
    /// outside that set, rather than silently falling back to strings).
    fn gather_rows_typed(&self, indices: &[usize]) -> Result<DataFrame> {
        let parallel = indices.len() >= DATAFRAME_PARALLEL_THRESHOLD;
        let mut result = DataFrame::new();

        for col_name in self.column_names() {
            macro_rules! try_gather {
                ($ty:ty) => {
                    if let Ok(series) = self.get_column::<$ty>(col_name) {
                        let values = series.values();
                        let fetch = |&i: &usize| -> Result<$ty> {
                            values
                                .get(i)
                                .cloned()
                                .ok_or_else(|| Error::IndexOutOfBounds {
                                    index: i,
                                    size: values.len(),
                                })
                        };
                        let gathered: Vec<$ty> = if parallel {
                            indices
                                .par_iter()
                                .map(fetch)
                                .collect::<Result<Vec<$ty>>>()?
                        } else {
                            indices.iter().map(fetch).collect::<Result<Vec<$ty>>>()?
                        };
                        let new_series = Series::new(gathered, Some(col_name.to_string()))?;
                        result.add_column(col_name.to_string(), new_series)?;
                        continue;
                    }
                };
            }

            try_gather!(String);
            try_gather!(i32);
            try_gather!(i64);
            try_gather!(f32);
            try_gather!(f64);
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

            return Err(Error::InvalidValue(format!(
                "Column '{}' has an element type that is not supported for parallel row \
                 selection (supported: String, integers, floats, bool, NaiveDate, \
                 NaiveDateTime, DateTime<Utc>)",
                col_name
            )));
        }

        Ok(result)
    }
}

/// Utilities for parallel data operations
pub struct ParallelUtils;

impl ParallelUtils {
    /// Sort a vector in parallel
    pub fn par_sort<T>(mut values: Vec<T>) -> Vec<T>
    where
        T: Ord + Send,
    {
        if values.len() < SCALAR_PARALLEL_THRESHOLD {
            values.sort();
        } else {
            values.par_sort();
        }
        values
    }

    /// Aggregate vector elements in parallel
    pub fn par_sum<T>(values: &[T]) -> T
    where
        T: Send + Sync + std::iter::Sum + Copy,
    {
        if values.len() < SCALAR_PARALLEL_THRESHOLD {
            values.iter().copied().sum()
        } else {
            values.par_iter().copied().sum()
        }
    }

    /// Calculate mean in parallel
    pub fn par_mean<T>(values: &[T]) -> Option<f64>
    where
        T: Send + Sync + Copy + Into<f64>,
    {
        if values.is_empty() {
            return None;
        }

        let sum: f64 = if values.len() < SCALAR_PARALLEL_THRESHOLD {
            values.iter().map(|&v| v.into()).sum()
        } else {
            values.par_iter().map(|&v| v.into()).sum()
        };

        Some(sum / values.len() as f64)
    }

    /// Find minimum value in parallel
    pub fn par_min<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        if values.len() < SCALAR_PARALLEL_THRESHOLD {
            values.iter().min().copied()
        } else {
            values.par_iter().min().copied()
        }
    }

    /// Find maximum value in parallel
    pub fn par_max<T>(values: &[T]) -> Option<T>
    where
        T: Send + Sync + Copy + Ord,
    {
        if values.len() < SCALAR_PARALLEL_THRESHOLD {
            values.iter().max().copied()
        } else {
            values.par_iter().max().copied()
        }
    }
}

use std::collections::HashMap;

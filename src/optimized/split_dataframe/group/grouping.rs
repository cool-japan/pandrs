//! Group creation logic and parallel grouping operations

use std::collections::HashMap;

use super::super::core::OptimizedDataFrame;
use super::types::{GroupBy, GroupKey, GroupKeyValue};
use crate::column::Column;
use crate::error::{Error, Result};

/// Marker used when a missing key has to be rendered into a `String` key.
///
/// Only reachable with `dropna = false`; with the pandas-compatible default
/// (`dropna = true`) rows with a missing key component are excluded entirely.
pub const NA_GROUP_KEY_MARKER: &str = "<NA>";

/// Append `part` to `out`, escaping the composite-key separator.
///
/// `_` and `\` are escaped so that `("a_b", "c")` and `("a", "b_c")` render to
/// `a\_b_c` and `a_b\_c` — distinct strings. A plain `join("_")` maps both onto
/// `a_b_c` and silently merges two different groups.
fn push_escaped(part: &str, out: &mut String) {
    for ch in part.chars() {
        if ch == '\\' || ch == '_' {
            out.push('\\');
        }
        out.push(ch);
    }
}

/// Render a typed composite key into the `String` key used by [`OptimizedDataFrame::par_groupby`].
///
/// A single-column group renders to the bare value (so `par_groupby(&["k"])`
/// can still be looked up by `"A"`); multi-column groups use the escaped,
/// collision-free encoding described on [`push_escaped`].
fn render_group_key(key: &[GroupKeyValue<'_>]) -> String {
    if key.len() == 1 {
        return key[0]
            .to_value_string()
            .unwrap_or_else(|| NA_GROUP_KEY_MARKER.to_string());
    }

    let mut out = String::new();
    for (idx, part) in key.iter().enumerate() {
        if idx > 0 {
            out.push('_');
        }
        match part.to_value_string() {
            Some(value) => push_escaped(&value, &mut out),
            // `\N` cannot be produced by `push_escaped` (it only emits `\\`
            // and `\_`), so the NA marker cannot collide with a real value.
            None => out.push_str("\\N"),
        }
    }
    out
}

impl OptimizedDataFrame {
    /// Resolve grouping column names to typed column references (once, not per row).
    fn resolve_group_columns(&self, names: &[String]) -> Result<Vec<&Column>> {
        let mut columns = Vec::with_capacity(names.len());
        for name in names {
            let idx = *self
                .column_indices
                .get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.clone()))?;
            let column = self
                .columns
                .get(idx)
                .ok_or_else(|| Error::ColumnNotFound(name.clone()))?;
            columns.push(column);
        }
        Ok(columns)
    }

    /// Build the typed group map for the given key columns.
    ///
    /// The key buffer is reused across rows, so a fresh allocation only happens
    /// when a *new* group is discovered instead of once per row per column.
    fn build_groups<'a>(
        &'a self,
        key_columns: &[&'a Column],
        rows: impl Iterator<Item = usize>,
        dropna: bool,
    ) -> HashMap<GroupKey<'a>, Vec<usize>> {
        let mut groups: HashMap<GroupKey<'a>, Vec<usize>> = HashMap::new();
        let mut key_buf: GroupKey<'a> = Vec::with_capacity(key_columns.len());

        'rows: for row_idx in rows {
            key_buf.clear();
            for column in key_columns {
                let part = GroupKeyValue::from_column(column, row_idx);
                if dropna && part.is_null() {
                    // pandas `dropna=True`: rows with a missing key are excluded.
                    continue 'rows;
                }
                key_buf.push(part);
            }

            // Look the key up by slice so that the buffer can be reused; only
            // materialise an owned key when the group does not exist yet.
            if groups.contains_key(key_buf.as_slice()) {
                if let Some(indices) = groups.get_mut(key_buf.as_slice()) {
                    indices.push(row_idx);
                }
            } else {
                groups.insert(key_buf.clone(), vec![row_idx]);
            }
        }

        groups
    }

    /// Group DataFrame
    ///
    /// Rows whose grouping key contains a missing value are excluded, matching
    /// the pandas `dropna=True` default. Use
    /// [`OptimizedDataFrame::group_by_with_config`] to keep them.
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by<I, S>(&self, columns: I) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.group_by_with_options(columns, true)
    }

    /// Group DataFrame with options
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    /// * `as_multi_index` - Whether to create a multi-index for the result (when multiple columns)
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by_with_options<I, S>(
        &self,
        columns: I,
        as_multi_index: bool,
    ) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.group_by_with_config(columns, as_multi_index, true)
    }

    /// Group DataFrame with full configuration
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    /// * `as_multi_index` - Whether to create a multi-index for the result (when multiple columns)
    /// * `dropna` - Whether to exclude rows whose key contains a missing value
    ///   (`true` reproduces the pandas default; `false` keeps them as a distinct
    ///   NA group that is emitted as a real NULL in the key column)
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by_with_config<I, S>(
        &self,
        columns: I,
        as_multi_index: bool,
        dropna: bool,
    ) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let group_by_columns: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        // Verify existence of each column and resolve them once
        let key_columns = self.resolve_group_columns(&group_by_columns)?;

        let groups = self.build_groups(&key_columns, 0..self.row_count, dropna);

        // Only use multi-index when we have multiple grouping columns and option is enabled
        let create_multi_index = as_multi_index && group_by_columns.len() > 1;

        Ok(GroupBy {
            df: self,
            group_by_columns,
            groups,
            create_multi_index,
            dropna,
        })
    }

    /// Group DataFrame using parallel processing
    ///
    /// Rows whose grouping key contains a missing value are excluded (pandas
    /// `dropna=True` default); use
    /// [`OptimizedDataFrame::par_groupby_with_config`] to keep them.
    ///
    /// # Arguments
    /// * `group_by_columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<HashMap<String, Self>>` - Grouping results (map of keys and DataFrames)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        self.par_groupby_with_config(group_by_columns, true)
    }

    /// Group DataFrame using parallel processing, with NA handling configured
    ///
    /// The returned map is keyed by the rendered composite group key: the bare
    /// value for a single grouping column, and a separator-escaped encoding for
    /// several columns (so `("a_b", "c")` and `("a", "b_c")` stay distinct).
    ///
    /// # Arguments
    /// * `group_by_columns` - Column names for grouping
    /// * `dropna` - Whether to exclude rows whose key contains a missing value
    ///
    /// # Returns
    /// * `Result<HashMap<String, Self>>` - Grouping results (map of keys and DataFrames)
    pub fn par_groupby_with_config(
        &self,
        group_by_columns: &[&str],
        dropna: bool,
    ) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;

        // Below these sizes the rayon fork/join overhead dominates, so the
        // serial path is measurably faster.
        const PARALLEL_ROW_THRESHOLD: usize = 50_000;
        const PARALLEL_GROUP_THRESHOLD: usize = 100;

        let names: Vec<String> = group_by_columns.iter().map(|s| (*s).to_string()).collect();
        let key_columns = self.resolve_group_columns(&names)?;

        // Generate group keys and group each row's index
        let groups: HashMap<GroupKey<'_>, Vec<usize>> = if self.row_count < PARALLEL_ROW_THRESHOLD {
            // Serial processing is more efficient for small data
            self.build_groups(&key_columns, 0..self.row_count, dropna)
        } else {
            // For large data: build local maps in parallel, then merge them
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);

            let local_maps: Vec<HashMap<GroupKey<'_>, Vec<usize>>> = (0..self.row_count)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| self.build_groups(&key_columns, chunk.iter().copied(), dropna))
                .collect();

            let mut merged: HashMap<GroupKey<'_>, Vec<usize>> = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    merged.entry(key).or_default().extend(indices);
                }
            }
            merged
        };

        // Efficiently create DataFrames for each group
        if groups.len() < PARALLEL_GROUP_THRESHOLD || self.row_count < PARALLEL_ROW_THRESHOLD {
            // Use serial processing for small data or when group count is small
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in &groups {
                let group_df = self.filter_by_indices(indices)?;
                result.insert(render_group_key(key), group_df);
            }
            Ok(result)
        } else {
            // Parallelize group construction for large data. Errors are
            // propagated instead of silently dropping the affected group.
            let group_items: Vec<(&GroupKey<'_>, &Vec<usize>)> = groups.iter().collect();

            let built: Vec<(String, Self)> = group_items
                .into_par_iter()
                .map(|(key, indices)| {
                    let group_df = self.filter_by_indices(indices)?;
                    Ok((render_group_key(key), group_df))
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(built.into_iter().collect())
        }
    }
}

//! Join functionality for OptimizedDataFrame

use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

/// Enumeration representing join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join (only rows that exist in both tables)
    Inner,
    /// Left join (all rows from the left table, and matching rows from the right table)
    Left,
    /// Right join (all rows from the right table, and matching rows from the left table)
    Right,
    /// Outer join (all rows from both tables)
    Outer,
}

/// A typed join key.
///
/// Keys are compared as values instead of via their textual rendering. That
/// removes two `String` allocations per row (one per side) and gives the
/// expected comparison semantics:
///
/// * `NaN` never matches anything (such keys are simply not indexed),
/// * `-0.0` and `0.0` are the same key,
/// * numbers never collide with their string spelling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKey<'a> {
    Int(i64),
    /// Bit pattern of a normalized, non-NaN float.
    Float(u64),
    Str(&'a str),
    Bool(bool),
}

/// Build the join key of `row`, or `None` when the row has no usable key
/// (NULL, or a NaN float which must never match).
fn join_key_at(column: &Column, row: usize) -> Option<JoinKey<'_>> {
    match column {
        Column::Int64(col) => col.get(row).ok().flatten().map(JoinKey::Int),
        Column::Float64(col) => col
            .get(row)
            .ok()
            .flatten()
            .filter(|value| !value.is_nan())
            .map(|value| JoinKey::Float(normalized_float_bits(value))),
        Column::String(col) => col.get(row).ok().flatten().map(JoinKey::Str),
        Column::Boolean(col) => col.get(row).ok().flatten().map(JoinKey::Bool),
    }
}

/// `-0.0 == 0.0` must also hash equal, so zero is normalized to `+0.0`.
fn normalized_float_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

/// Materialize `column` for the given row positions, where `None` means "this
/// output row has no counterpart on that side" and therefore yields NULL.
fn gather_optional(column: &Column, positions: &[Option<usize>]) -> Column {
    match column {
        Column::Int64(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for position in positions {
                match position.and_then(|row| col.get(row).ok().flatten()) {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(0);
                        nulls.push(true);
                    }
                }
            }
            Column::Int64(Int64Column::with_nulls(values, nulls))
        }
        Column::Float64(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for position in positions {
                match position.and_then(|row| col.get(row).ok().flatten()) {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(0.0);
                        nulls.push(true);
                    }
                }
            }
            Column::Float64(Float64Column::with_nulls(values, nulls))
        }
        Column::String(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for position in positions {
                match position.and_then(|row| col.get(row).ok().flatten()) {
                    Some(value) => {
                        values.push(value.to_string());
                        nulls.push(false);
                    }
                    None => {
                        values.push(String::new());
                        nulls.push(true);
                    }
                }
            }
            Column::String(StringColumn::with_nulls(values, nulls))
        }
        Column::Boolean(col) => {
            let mut values = Vec::with_capacity(positions.len());
            let mut nulls = Vec::with_capacity(positions.len());
            for position in positions {
                match position.and_then(|row| col.get(row).ok().flatten()) {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(false);
                        nulls.push(true);
                    }
                }
            }
            Column::Boolean(BooleanColumn::with_nulls(values, nulls))
        }
    }
}

/// Build the join key column of the result: the left key value when the output
/// row has a left counterpart, otherwise the key value of the right row.
/// A NULL key stays NULL.
fn gather_key_column(
    left_col: &Column,
    right_col: &Column,
    pairs: &[(Option<usize>, Option<usize>)],
    left_on: &str,
    right_on: &str,
) -> Result<Column> {
    match (left_col, right_col) {
        (Column::Int64(left), Column::Int64(right)) => {
            let mut values = Vec::with_capacity(pairs.len());
            let mut nulls = Vec::with_capacity(pairs.len());
            for &(left_idx, right_idx) in pairs {
                let value = match (left_idx, right_idx) {
                    (Some(row), _) => left.get(row).ok().flatten(),
                    (None, Some(row)) => right.get(row).ok().flatten(),
                    (None, None) => None,
                };
                match value {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(0);
                        nulls.push(true);
                    }
                }
            }
            Ok(Column::Int64(Int64Column::with_nulls(values, nulls)))
        }
        (Column::Float64(left), Column::Float64(right)) => {
            let mut values = Vec::with_capacity(pairs.len());
            let mut nulls = Vec::with_capacity(pairs.len());
            for &(left_idx, right_idx) in pairs {
                let value = match (left_idx, right_idx) {
                    (Some(row), _) => left.get(row).ok().flatten(),
                    (None, Some(row)) => right.get(row).ok().flatten(),
                    (None, None) => None,
                };
                match value {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(0.0);
                        nulls.push(true);
                    }
                }
            }
            Ok(Column::Float64(Float64Column::with_nulls(values, nulls)))
        }
        (Column::String(left), Column::String(right)) => {
            let mut values = Vec::with_capacity(pairs.len());
            let mut nulls = Vec::with_capacity(pairs.len());
            for &(left_idx, right_idx) in pairs {
                let value = match (left_idx, right_idx) {
                    (Some(row), _) => left.get(row).ok().flatten().map(|v| v.to_string()),
                    (None, Some(row)) => right.get(row).ok().flatten().map(|v| v.to_string()),
                    (None, None) => None,
                };
                match value {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(String::new());
                        nulls.push(true);
                    }
                }
            }
            Ok(Column::String(StringColumn::with_nulls(values, nulls)))
        }
        (Column::Boolean(left), Column::Boolean(right)) => {
            let mut values = Vec::with_capacity(pairs.len());
            let mut nulls = Vec::with_capacity(pairs.len());
            for &(left_idx, right_idx) in pairs {
                let value = match (left_idx, right_idx) {
                    (Some(row), _) => left.get(row).ok().flatten(),
                    (None, Some(row)) => right.get(row).ok().flatten(),
                    (None, None) => None,
                };
                match value {
                    Some(value) => {
                        values.push(value);
                        nulls.push(false);
                    }
                    None => {
                        values.push(false);
                        nulls.push(true);
                    }
                }
            }
            Ok(Column::Boolean(BooleanColumn::with_nulls(values, nulls)))
        }
        (left, right) => Err(Error::ColumnTypeMismatch {
            name: format!("{} and {}", left_on, right_on),
            expected: left.column_type(),
            found: right.column_type(),
        }),
    }
}

impl OptimizedDataFrame {
    /// Inner join
    ///
    /// # Arguments
    /// * `other` - Right DataFrame to join with
    /// * `left_on` - Join key column from the left DataFrame
    /// * `right_on` - Join key column from the right DataFrame
    ///
    /// # Returns
    /// * `Result<Self>` - Result DataFrame after join operation
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Inner)
    }

    /// Left join
    ///
    /// # Arguments
    /// * `other` - Right DataFrame to join with
    /// * `left_on` - Join key column from the left DataFrame
    /// * `right_on` - Join key column from the right DataFrame
    ///
    /// # Returns
    /// * `Result<Self>` - Result DataFrame after join operation
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Left)
    }

    /// Right join
    ///
    /// # Arguments
    /// * `other` - Right DataFrame to join with
    /// * `left_on` - Join key column from the left DataFrame
    /// * `right_on` - Join key column from the right DataFrame
    ///
    /// # Returns
    /// * `Result<Self>` - Result DataFrame after join operation
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Right)
    }

    /// Outer join
    ///
    /// # Arguments
    /// * `other` - Right DataFrame to join with
    /// * `left_on` - Join key column from the left DataFrame
    /// * `right_on` - Join key column from the right DataFrame
    ///
    /// # Returns
    /// * `Result<Self>` - Result DataFrame after join operation
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        self.join_impl(other, left_on, right_on, JoinType::Outer)
    }

    /// Join implementation (internal method)
    ///
    /// Semantics:
    /// * NULL keys (and NaN float keys) never match anything, but the rows are
    ///   still kept by the join types that preserve unmatched rows - a left join
    ///   returns every left row, even the ones whose key is missing.
    /// * Columns of the side a result row does not come from are filled with
    ///   NULL, never with `0`/`""`/`false`.
    /// * The result schema does not depend on the number of matches: an empty
    ///   result has exactly the same columns as a non-empty one.
    /// * Row order is left-major, and within one left row the matches follow the
    ///   right frame order; unmatched right rows are appended at the end.
    fn join_impl(
        &self,
        other: &Self,
        left_on: &str,
        right_on: &str,
        join_type: JoinType,
    ) -> Result<Self> {
        // Get join key columns
        let left_col_idx = self
            .column_indices
            .get(left_on)
            .ok_or_else(|| Error::ColumnNotFound(left_on.to_string()))?;

        let right_col_idx = other
            .column_indices
            .get(right_on)
            .ok_or_else(|| Error::ColumnNotFound(right_on.to_string()))?;

        let left_col = &self.columns[*left_col_idx];
        let right_col = &other.columns[*right_col_idx];

        // Verify that both columns have the same type
        if left_col.column_type() != right_col.column_type() {
            return Err(Error::ColumnTypeMismatch {
                name: format!("{} and {}", left_on, right_on),
                expected: left_col.column_type(),
                found: right_col.column_type(),
            });
        }

        let keep_unmatched_left = join_type == JoinType::Left || join_type == JoinType::Outer;
        let keep_unmatched_right = join_type == JoinType::Right || join_type == JoinType::Outer;

        let mut join_indices: Vec<(Option<usize>, Option<usize>)> = Vec::new();

        if other.row_count <= self.row_count {
            // Build the hash table on the smaller (right) side and probe with
            // the left side, which yields left-major order directly.
            let mut right_key_to_indices: HashMap<JoinKey, Vec<usize>> =
                HashMap::with_capacity(other.row_count);

            for row in 0..other.row_count {
                if let Some(key) = join_key_at(right_col, row) {
                    right_key_to_indices.entry(key).or_default().push(row);
                }
            }

            for row in 0..self.row_count {
                let matches = match join_key_at(left_col, row) {
                    Some(key) => right_key_to_indices.get(&key),
                    None => None,
                };

                match matches {
                    Some(right_rows) => {
                        for &right_row in right_rows {
                            join_indices.push((Some(row), Some(right_row)));
                        }
                    }
                    None => {
                        if keep_unmatched_left {
                            join_indices.push((Some(row), None));
                        }
                    }
                }
            }
        } else {
            // The left side is smaller: build on the left, probe with the right
            // and restore the left-major ordering afterwards.
            let mut left_key_to_indices: HashMap<JoinKey, Vec<usize>> =
                HashMap::with_capacity(self.row_count);

            for row in 0..self.row_count {
                if let Some(key) = join_key_at(left_col, row) {
                    left_key_to_indices.entry(key).or_default().push(row);
                }
            }

            let mut left_matched = vec![false; self.row_count];

            for right_row in 0..other.row_count {
                let matches = match join_key_at(right_col, right_row) {
                    Some(key) => left_key_to_indices.get(&key),
                    None => None,
                };

                if let Some(left_rows) = matches {
                    for &left_row in left_rows {
                        left_matched[left_row] = true;
                        join_indices.push((Some(left_row), Some(right_row)));
                    }
                }
            }

            if keep_unmatched_left {
                for (left_row, matched) in left_matched.iter().enumerate() {
                    if !matched {
                        join_indices.push((Some(left_row), None));
                    }
                }
            }

            join_indices.sort_by_key(|&(left, right)| {
                (left.unwrap_or(usize::MAX), right.unwrap_or(usize::MAX))
            });
        }

        // For right or outer join, add unmatched rows from the right side
        if keep_unmatched_right {
            let mut right_matched = vec![false; other.row_count];
            for (_, right_idx) in &join_indices {
                if let Some(idx) = right_idx {
                    right_matched[*idx] = true;
                }
            }

            for (row, matched) in right_matched.iter().enumerate() {
                if !matched {
                    join_indices.push((None, Some(row)));
                }
            }
        }

        // Materialize the result. The very same code path also produces the
        // (correctly typed, correctly named) empty frame when nothing matched.
        let left_positions: Vec<Option<usize>> =
            join_indices.iter().map(|&(left, _)| left).collect();
        let right_positions: Vec<Option<usize>> =
            join_indices.iter().map(|&(_, right)| right).collect();

        let mut result = Self::new();

        // Add columns from the left side (excluding the join key)
        for name in &self.column_names {
            if name != left_on {
                let col_idx = self.column_indices[name];
                let joined_col = gather_optional(&self.columns[col_idx], &left_positions);
                result.add_column(name.clone(), joined_col)?;
            }
        }

        // Add the join key column once, taken from whichever side has the row
        let joined_key_col =
            gather_key_column(left_col, right_col, &join_indices, left_on, right_on)?;
        result.add_column(left_on.to_string(), joined_key_col)?;

        // Add columns from the right side (excluding the join key)
        for name in &other.column_names {
            if name != right_on {
                let new_name = if result.column_indices.contains_key(name) {
                    format!("{}_right", name)
                } else {
                    name.clone()
                };

                let col_idx = other.column_indices[name];
                let joined_col = gather_optional(&other.columns[col_idx], &right_positions);
                result.add_column(new_name, joined_col)?;
            }
        }

        Ok(result)
    }
}

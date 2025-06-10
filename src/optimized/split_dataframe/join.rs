//! Join functionality for OptimizedDataFrame

use std::collections::HashMap;

use super::core::OptimizedDataFrame;
use crate::column::{BooleanColumn, Column, ColumnTrait, ColumnType, Int64Column};
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

    // Join implementation (internal method)
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

        // Build mapping for join keys
        let mut right_key_to_indices: HashMap<String, Vec<usize>> = HashMap::new();

        match right_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            }
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            }
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            }
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        right_key_to_indices.entry(key).or_default().push(i);
                    }
                }
            }
        }

        // Create join rows
        let mut result = Self::new();
        let mut join_indices: Vec<(Option<usize>, Option<usize>)> = Vec::new();

        // Process left DataFrame
        match left_col {
            Column::Int64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            // If there's a match, join with each right index
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            // For left or outer join, include left row even without a match
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            }
            Column::Float64(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            }
            Column::String(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            }
            Column::Boolean(col) => {
                for i in 0..col.len() {
                    if let Ok(Some(val)) = col.get(i) {
                        let key = val.to_string();
                        if let Some(right_indices) = right_key_to_indices.get(&key) {
                            for &right_idx in right_indices {
                                join_indices.push((Some(i), Some(right_idx)));
                            }
                        } else if join_type == JoinType::Left || join_type == JoinType::Outer {
                            join_indices.push((Some(i), None));
                        }
                    }
                }
            }
        }

        // For right or outer join, add unmatched rows from the right side
        if join_type == JoinType::Right || join_type == JoinType::Outer {
            let mut right_matched = vec![false; other.row_count];
            for (_, right_idx) in &join_indices {
                if let Some(idx) = right_idx {
                    right_matched[*idx] = true;
                }
            }

            for (i, matched) in right_matched.iter().enumerate() {
                if !matched {
                    join_indices.push((None, Some(i)));
                }
            }
        }

        // Early return if the result is empty
        if join_indices.is_empty() {
            // Return an empty DataFrame (columns only)
            let mut result = Self::new();
            for name in &self.column_names {
                if name != left_on {
                    let col_idx = self.column_indices[name];
                    let col_type = self.columns[col_idx].column_type();

                    // Create an empty column based on the type
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => {
                            Column::Float64(crate::column::Float64Column::new(Vec::new()))
                        }
                        ColumnType::String => {
                            Column::String(crate::column::StringColumn::new(Vec::new()))
                        }
                        ColumnType::Boolean => {
                            Column::Boolean(crate::column::BooleanColumn::new(Vec::new()))
                        }
                    };

                    result.add_column(name.clone(), empty_col)?;
                }
            }

            for name in &other.column_names {
                if name != right_on {
                    let suffix = "_right";
                    let new_name = if self.column_indices.contains_key(name) {
                        format!("{}{}", name, suffix)
                    } else {
                        name.clone()
                    };

                    let col_idx = other.column_indices[name];
                    let col_type = other.columns[col_idx].column_type();

                    // Create an empty column based on the type
                    let empty_col = match col_type {
                        ColumnType::Int64 => Column::Int64(Int64Column::new(Vec::new())),
                        ColumnType::Float64 => {
                            Column::Float64(crate::column::Float64Column::new(Vec::new()))
                        }
                        ColumnType::String => {
                            Column::String(crate::column::StringColumn::new(Vec::new()))
                        }
                        ColumnType::Boolean => {
                            Column::Boolean(crate::column::BooleanColumn::new(Vec::new()))
                        }
                    };

                    result.add_column(new_name, empty_col)?;
                }
            }

            return Ok(result);
        }

        // Prepare column data for the result
        let row_count = join_indices.len();

        // Add columns from the left side
        for name in &self.column_names {
            if name != left_on {
                // Only add the join key column once
                let col_idx = self.column_indices[name];
                let col = &self.columns[col_idx];

                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // Default value
                                }
                            } else {
                                data.push(0); // Default value (right side only)
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    }
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // Default value
                                }
                            } else {
                                data.push(0.0); // Default value (right side only)
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    }
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // Default value
                                }
                            } else {
                                data.push(String::new()); // Default value (right side only)
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    }
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (left_idx, _) in &join_indices {
                            if let Some(idx) = left_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // Default value
                                }
                            } else {
                                data.push(false); // Default value (right side only)
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    }
                };

                result.add_column(name.clone(), joined_col)?;
            }
        }

        // Add join key column (from the left side)
        let left_key_col = &self.columns[*left_col_idx];
        let joined_key_col = match left_key_col {
            Column::Int64(int_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = int_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0); // Default value
                        }
                    } else if let Some(idx) = right_idx {
                        // Use key value from the right side
                        if let Column::Int64(right_int_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_int_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0); // Default value
                            }
                        } else {
                            data.push(0); // Default value
                        }
                    } else {
                        data.push(0); // Default value
                    }
                }
                Column::Int64(Int64Column::new(data))
            }
            Column::Float64(float_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = float_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(0.0); // Default value
                        }
                    } else if let Some(idx) = right_idx {
                        // Use key value from the right side
                        if let Column::Float64(right_float_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_float_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(0.0); // Default value
                            }
                        } else {
                            data.push(0.0); // Default value
                        }
                    } else {
                        data.push(0.0); // Default value
                    }
                }
                Column::Float64(crate::column::Float64Column::new(data))
            }
            Column::String(str_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = str_col.get(*idx) {
                            data.push(val.to_string());
                        } else {
                            data.push(String::new()); // Default value
                        }
                    } else if let Some(idx) = right_idx {
                        // Use key value from the right side
                        if let Column::String(right_str_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_str_col.get(*idx) {
                                data.push(val.to_string());
                            } else {
                                data.push(String::new()); // Default value
                            }
                        } else {
                            data.push(String::new()); // Default value
                        }
                    } else {
                        data.push(String::new()); // Default value
                    }
                }
                Column::String(crate::column::StringColumn::new(data))
            }
            Column::Boolean(bool_col) => {
                let mut data = Vec::with_capacity(row_count);
                for (left_idx, right_idx) in &join_indices {
                    if let Some(idx) = left_idx {
                        if let Ok(Some(val)) = bool_col.get(*idx) {
                            data.push(val);
                        } else {
                            data.push(false); // Default value
                        }
                    } else if let Some(idx) = right_idx {
                        // Use key value from the right side
                        if let Column::Boolean(right_bool_col) = &other.columns[*right_col_idx] {
                            if let Ok(Some(val)) = right_bool_col.get(*idx) {
                                data.push(val);
                            } else {
                                data.push(false); // Default value
                            }
                        } else {
                            data.push(false); // Default value
                        }
                    } else {
                        data.push(false); // Default value
                    }
                }
                Column::Boolean(crate::column::BooleanColumn::new(data))
            }
        };

        result.add_column(left_on.to_string(), joined_key_col)?;

        // Add columns from the right side (excluding join key)
        for name in &other.column_names {
            if name != right_on {
                let suffix = "_right";
                let new_name = if result.column_indices.contains_key(name) {
                    format!("{}{}", name, suffix)
                } else {
                    name.clone()
                };

                let col_idx = other.column_indices[name];
                let col = &other.columns[col_idx];

                let joined_col = match col {
                    Column::Int64(int_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = int_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0); // Default value
                                }
                            } else {
                                data.push(0); // Default value (left side only)
                            }
                        }
                        Column::Int64(Int64Column::new(data))
                    }
                    Column::Float64(float_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = float_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(0.0); // Default value
                                }
                            } else {
                                data.push(0.0); // Default value (left side only)
                            }
                        }
                        Column::Float64(crate::column::Float64Column::new(data))
                    }
                    Column::String(str_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = str_col.get(*idx) {
                                    data.push(val.to_string());
                                } else {
                                    data.push(String::new()); // Default value
                                }
                            } else {
                                data.push(String::new()); // Default value (left side only)
                            }
                        }
                        Column::String(crate::column::StringColumn::new(data))
                    }
                    Column::Boolean(bool_col) => {
                        let mut data = Vec::with_capacity(row_count);
                        for (_, right_idx) in &join_indices {
                            if let Some(idx) = right_idx {
                                if let Ok(Some(val)) = bool_col.get(*idx) {
                                    data.push(val);
                                } else {
                                    data.push(false); // Default value
                                }
                            } else {
                                data.push(false); // Default value (left side only)
                            }
                        }
                        Column::Boolean(crate::column::BooleanColumn::new(data))
                    }
                };

                result.add_column(new_name, joined_col)?;
            }
        }

        Ok(result)
    }
}

//! Data operations functionality for OptimizedDataFrame

use super::core::OptimizedDataFrame;
use super::select::{take_column, take_rows};
use crate::column::{BooleanColumn, Column, ColumnType, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Select columns (as a new DataFrame)
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();

        for &name in columns {
            let column_idx = self
                .column_indices
                .get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

            let column = self.columns[*column_idx].clone();
            result.add_column(name.to_string(), column)?;
        }

        // Copy index
        if let Some(ref idx) = self.index {
            result.index = Some(idx.clone());
        }

        Ok(result)
    }

    /// Filter rows (as a new DataFrame)
    ///
    /// Delegates to [`OptimizedDataFrame::filter_rows`] so that both entry
    /// points share one implementation (NULL values and index labels of the
    /// surviving rows are preserved).
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        self.filter_rows(condition_column)
    }

    /// Filter by specified row indices (internal helper)
    ///
    /// Delegates to the canonical row-materialization routine, which preserves
    /// NULL values and re-selects the index labels of the selected rows.
    pub(crate) fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        take_rows(self, indices)
    }

    /// Get the first n rows
    pub fn head(&self, n: usize) -> Result<Self> {
        self.head_rows(n)
    }

    /// Get the last n rows
    pub fn tail(&self, n: usize) -> Result<Self> {
        self.tail_rows(n)
    }

    /// Convert DataFrame to "long format" (melt operation)
    ///
    /// Converts multiple columns into a single "variable" column and "value" column.
    /// This implementation prioritizes performance.
    ///
    /// # Arguments
    /// * `id_vars` - Column names to keep unchanged (identifier columns)
    /// * `value_vars` - Column names to convert (value columns). If None, all columns except id_vars
    /// * `var_name` - Name for the variable column (default: "variable")
    /// * `value_name` - Name for the value column (default: "value")
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame converted to long format
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // Set default values for arguments
        let var_name = var_name.unwrap_or("variable");
        let value_name = value_name.unwrap_or("value");

        // If value_vars is not specified, use all columns except id_vars
        let value_vars = if let Some(vars) = value_vars {
            vars.to_vec()
        } else {
            self.column_names
                .iter()
                .filter(|name| !id_vars.contains(&name.as_str()))
                .map(|s| s.as_str())
                .collect()
        };

        // Check for non-existent column names
        for col in id_vars.iter().chain(value_vars.iter()) {
            if !self.column_indices.contains_key(*col) {
                return Err(Error::ColumnNotFound((*col).to_string()));
            }
        }

        // Precompute the size of the result (performance optimization)
        let result_rows = self.row_count * value_vars.len();

        // Create the result DataFrame
        let mut result = Self::new();

        // Create the variable name column
        let mut var_col_data = Vec::with_capacity(result_rows);
        for &value_col_name in &value_vars {
            for _ in 0..self.row_count {
                var_col_data.push(value_col_name.to_string());
            }
        }
        result.add_column(
            var_name.to_string(),
            Column::String(StringColumn::new(var_col_data)),
        )?;

        // Replicate and add ID columns. The replication is a plain row
        // selection, so it reuses the canonical (NULL preserving) routine.
        let mut repeated_positions = Vec::with_capacity(result_rows);
        for _ in 0..value_vars.len() {
            repeated_positions.extend(0..self.row_count);
        }

        for &id_col_name in id_vars {
            let idx = self
                .column_indices
                .get(id_col_name)
                .ok_or_else(|| Error::ColumnNotFound(id_col_name.to_string()))?;
            let replicated = take_column(&self.columns[*idx], &repeated_positions);
            result.add_column(id_col_name.to_string(), replicated)?;
        }

        // Collect the values of every value column as text. Missing values stay
        // missing (`None`) instead of being turned into empty strings, which
        // would later be parsed back as 0.
        let mut all_values: Vec<Option<String>> = Vec::with_capacity(result_rows);

        for val_col in &value_vars {
            let idx = self
                .column_indices
                .get(*val_col)
                .ok_or_else(|| Error::ColumnNotFound((*val_col).to_string()))?;

            match &self.columns[*idx] {
                Column::Int64(int_col) => {
                    for i in 0..self.row_count {
                        all_values.push(int_col.get(i).ok().flatten().map(|v| v.to_string()));
                    }
                }
                Column::Float64(float_col) => {
                    for i in 0..self.row_count {
                        all_values.push(float_col.get(i).ok().flatten().map(|v| v.to_string()));
                    }
                }
                Column::String(str_col) => {
                    for i in 0..self.row_count {
                        all_values.push(str_col.get(i).ok().flatten().map(|v| v.to_string()));
                    }
                }
                Column::Boolean(bool_col) => {
                    for i in 0..self.row_count {
                        all_values.push(bool_col.get(i).ok().flatten().map(|v| v.to_string()));
                    }
                }
            }
        }

        // Determine the appropriate type from the values that are actually
        // present (NULLs must not influence type inference).
        let nulls: Vec<bool> = all_values.iter().map(|v| v.is_none()).collect();
        let present: Vec<&String> = all_values.iter().flatten().collect();

        let is_all_int = !present.is_empty() && present.iter().all(|s| s.parse::<i64>().is_ok());

        let is_all_float =
            !is_all_int && !present.is_empty() && present.iter().all(|s| s.parse::<f64>().is_ok());

        let is_all_bool = !is_all_int
            && !is_all_float
            && !present.is_empty()
            && present.iter().all(|s| {
                let lower = s.to_lowercase();
                lower == "true"
                    || lower == "false"
                    || lower == "1"
                    || lower == "0"
                    || lower == "yes"
                    || lower == "no"
            });

        // Add columns based on the determined type
        if is_all_int {
            let int_values: Vec<i64> = all_values
                .iter()
                .map(|s| s.as_ref().and_then(|s| s.parse::<i64>().ok()).unwrap_or(0))
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Int64(Int64Column::with_nulls(int_values, nulls)),
            )?;
        } else if is_all_float {
            let float_values: Vec<f64> = all_values
                .iter()
                .map(|s| {
                    s.as_ref()
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(0.0)
                })
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Float64(Float64Column::with_nulls(float_values, nulls)),
            )?;
        } else if is_all_bool {
            let bool_values: Vec<bool> = all_values
                .iter()
                .map(|s| {
                    s.as_ref()
                        .map(|s| {
                            let lower = s.to_lowercase();
                            lower == "true" || lower == "1" || lower == "yes"
                        })
                        .unwrap_or(false)
                })
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::Boolean(BooleanColumn::with_nulls(bool_values, nulls)),
            )?;
        } else {
            // Default to string type
            let str_values: Vec<String> = all_values
                .into_iter()
                .map(|s| s.unwrap_or_default())
                .collect();
            result.add_column(
                value_name.to_string(),
                Column::String(StringColumn::with_nulls(str_values, nulls)),
            )?;
        }

        Ok(result)
    }

    /// Concatenate DataFrames vertically
    ///
    /// Columns that exist in only one of the two frames are filled with NULL
    /// (never with `0`/`""`/`false`) for the rows coming from the other frame.
    /// The resulting column order is the deterministic union of both schemas:
    /// the columns of `self` in their original order, followed by the columns
    /// that only `other` has.
    ///
    /// # Arguments
    /// * `other` - DataFrame to concatenate
    ///
    /// # Returns
    /// * `Result<Self>` - Concatenated DataFrame
    pub fn append(&self, other: &Self) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }

        if other.columns.is_empty() {
            return Ok(self.clone());
        }

        // Create the result DataFrame
        let mut result = Self::new();

        // Ordered union of the column names of both frames (a HashSet would
        // make the resulting schema order non-deterministic).
        let mut all_columns: Vec<&String> = self.column_names.iter().collect();
        for name in &other.column_names {
            if !self.column_indices.contains_key(name) {
                all_columns.push(name);
            }
        }

        for col_name in all_columns {
            let self_col = self
                .column_indices
                .get(col_name)
                .map(|&idx| &self.columns[idx]);
            let other_col = other
                .column_indices
                .get(col_name)
                .map(|&idx| &other.columns[idx]);

            let combined = concat_column_segments(
                self_col,
                self.row_count,
                other_col,
                other.row_count,
                col_name,
            )?;

            result.add_column(col_name.clone(), combined)?;
        }

        Ok(result)
    }
}

/// Concatenate the rows of two (optional) columns into a single column.
///
/// A missing column contributes NULLs for its whole segment, and a missing
/// value inside a present column stays NULL. Both segments are built to exactly
/// `left_rows` / `right_rows` entries so that every output column of an append
/// has the same length even if a source column was shorter than its frame.
fn concat_column_segments(
    left: Option<&Column>,
    left_rows: usize,
    right: Option<&Column>,
    right_rows: usize,
    column_name: &str,
) -> Result<Column> {
    let total = left_rows + right_rows;

    let output_type = match (left, right) {
        (Some(l), Some(r)) => {
            if l.column_type() == r.column_type() {
                l.column_type()
            } else {
                // Types do not match: fall back to a textual representation.
                ColumnType::String
            }
        }
        (Some(l), None) => l.column_type(),
        (None, Some(r)) => r.column_type(),
        (None, None) => {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }
    };

    // The textual arm below also covers the mixed-type case: it renders any
    // column type and keeps missing values missing.
    let column = match output_type {
        ColumnType::Int64 => {
            let mut values = Vec::with_capacity(total);
            let mut nulls = Vec::with_capacity(total);
            push_int_segment(&mut values, &mut nulls, left, left_rows);
            push_int_segment(&mut values, &mut nulls, right, right_rows);
            Column::Int64(Int64Column::with_nulls(values, nulls))
        }
        ColumnType::Float64 => {
            let mut values = Vec::with_capacity(total);
            let mut nulls = Vec::with_capacity(total);
            push_float_segment(&mut values, &mut nulls, left, left_rows);
            push_float_segment(&mut values, &mut nulls, right, right_rows);
            Column::Float64(Float64Column::with_nulls(values, nulls))
        }
        ColumnType::String => {
            let mut values = Vec::with_capacity(total);
            let mut nulls = Vec::with_capacity(total);
            push_text_segment(&mut values, &mut nulls, left, left_rows);
            push_text_segment(&mut values, &mut nulls, right, right_rows);
            Column::String(StringColumn::with_nulls(values, nulls))
        }
        ColumnType::Boolean => {
            let mut values = Vec::with_capacity(total);
            let mut nulls = Vec::with_capacity(total);
            push_bool_segment(&mut values, &mut nulls, left, left_rows);
            push_bool_segment(&mut values, &mut nulls, right, right_rows);
            Column::Boolean(BooleanColumn::with_nulls(values, nulls))
        }
    };

    Ok(column)
}

fn push_int_segment(
    values: &mut Vec<i64>,
    nulls: &mut Vec<bool>,
    column: Option<&Column>,
    rows: usize,
) {
    for i in 0..rows {
        let value = match column {
            Some(Column::Int64(col)) => col.get(i).ok().flatten(),
            _ => None,
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
}

fn push_float_segment(
    values: &mut Vec<f64>,
    nulls: &mut Vec<bool>,
    column: Option<&Column>,
    rows: usize,
) {
    for i in 0..rows {
        let value = match column {
            Some(Column::Float64(col)) => col.get(i).ok().flatten(),
            _ => None,
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
}

fn push_bool_segment(
    values: &mut Vec<bool>,
    nulls: &mut Vec<bool>,
    column: Option<&Column>,
    rows: usize,
) {
    for i in 0..rows {
        let value = match column {
            Some(Column::Boolean(col)) => col.get(i).ok().flatten(),
            _ => None,
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
}

fn push_text_segment(
    values: &mut Vec<String>,
    nulls: &mut Vec<bool>,
    column: Option<&Column>,
    rows: usize,
) {
    for i in 0..rows {
        let value = match column {
            Some(Column::Int64(col)) => col.get(i).ok().flatten().map(|v| v.to_string()),
            Some(Column::Float64(col)) => col.get(i).ok().flatten().map(|v| v.to_string()),
            Some(Column::String(col)) => col.get(i).ok().flatten().map(|v| v.to_string()),
            Some(Column::Boolean(col)) => col.get(i).ok().flatten().map(|v| v.to_string()),
            None => None,
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
}

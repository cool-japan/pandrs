//! Aggregation functionality for OptimizedDataFrame
//!
//! Every reduction here walks the column's backing storage directly and honours
//! its null mask. When a float column has no nulls at all — the common case —
//! the JIT kernels receive the column's own slice, so no intermediate
//! `Vec<f64>` copy is made. Integer columns are reduced with exact integer
//! arithmetic (`i128` accumulation), which cannot overflow or lose precision.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::column::{Column, ColumnTrait, Float64Column, Int64Column};
use crate::error::{Error, Result};
use crate::optimized::jit::{
    parallel_max_f64, parallel_mean_f64_value, parallel_min_f64, parallel_sum_f64, ParallelConfig,
};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

/// Whether the entry at `index` is NULL according to a packed null mask.
#[inline]
fn is_null_at(mask: &Option<Arc<[u8]>>, index: usize) -> bool {
    match mask {
        None => false,
        Some(bits) => {
            let byte_idx = index / 8;
            let bit_idx = index % 8;
            byte_idx < bits.len() && (bits[byte_idx] & (1 << bit_idx)) != 0
        }
    }
}

/// Number of non-NULL entries described by `mask` over `len` slots.
fn valid_count(mask: &Option<Arc<[u8]>>, len: usize) -> usize {
    match mask {
        None => len,
        Some(_) => (0..len).filter(|&idx| !is_null_at(mask, idx)).count(),
    }
}

/// Non-NULL `i64` values of the column, without materialising a copy.
fn int_values(col: &Int64Column) -> impl Iterator<Item = i64> + '_ {
    let mask = &col.null_mask;
    col.data()
        .iter()
        .enumerate()
        .filter(move |(idx, _)| !is_null_at(mask, *idx))
        .map(|(_, &value)| value)
}

/// Non-NULL `f64` values of the column.
///
/// Borrows the column's own slice when there is no null mask (no allocation);
/// only a column that actually carries NULLs needs a compacted copy.
fn float_values(col: &Float64Column) -> Cow<'_, [f64]> {
    match &col.null_mask {
        None => Cow::Borrowed(col.data()),
        Some(_) => Cow::Owned(
            col.data()
                .iter()
                .enumerate()
                .filter(|(idx, _)| !is_null_at(&col.null_mask, *idx))
                .map(|(_, &value)| value)
                .collect(),
        ),
    }
}

/// Exact sum of the non-NULL integer values (`i128` accumulation).
fn int_sum(col: &Int64Column) -> (i128, usize) {
    let mut sum: i128 = 0;
    let mut count: usize = 0;
    for value in int_values(col) {
        sum += i128::from(value);
        count += 1;
    }
    (sum, count)
}

impl OptimizedDataFrame {
    /// Look up a column by name.
    fn numeric_column(&self, column_name: &str) -> Result<&Column> {
        let column_idx = *self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        self.columns
            .get(column_idx)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))
    }

    /// Calculate sum of a column using JIT-optimized operations
    ///
    /// NULL values are skipped; a column with no non-NULL values sums to `0.0`.
    ///
    /// # Arguments
    /// * `column_name` - Name of column to sum
    ///
    /// # Returns
    /// * `Result<f64>` - Sum value
    pub fn sum(&self, column_name: &str) -> Result<f64> {
        self.sum_with_config(column_name, None)
    }

    /// Calculate mean of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to average
    ///
    /// # Returns
    /// * `Result<f64>` - Mean value
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        self.mean_with_config(column_name, None)
    }

    /// Calculate maximum value of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find maximum
    ///
    /// # Returns
    /// * `Result<f64>` - Maximum value
    pub fn max(&self, column_name: &str) -> Result<f64> {
        self.max_with_config(column_name, None)
    }

    /// Calculate minimum value of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find minimum
    ///
    /// # Returns
    /// * `Result<f64>` - Minimum value
    pub fn min(&self, column_name: &str) -> Result<f64> {
        self.min_with_config(column_name, None)
    }

    /// Count non-NULL values in a column
    ///
    /// # Arguments
    /// * `column_name` - Target column name
    ///
    /// # Returns
    /// * `Result<usize>` - Count of non-NULL values
    pub fn count(&self, column_name: &str) -> Result<usize> {
        let column = self.numeric_column(column_name)?;

        let count = match column {
            Column::Int64(col) => valid_count(&col.null_mask, col.len()),
            Column::Float64(col) => valid_count(&col.null_mask, col.len()),
            Column::String(col) => valid_count(&col.null_mask, col.len()),
            Column::Boolean(col) => valid_count(&col.null_mask, col.len()),
        };

        Ok(count)
    }

    /// Execute aggregation operations on multiple columns
    ///
    /// # Arguments
    /// * `column_names` - Target column name array
    /// * `operation` - Operation name. One of "sum", "mean", "max", "min", "count"
    ///
    /// # Returns
    /// * `Result<HashMap<String, f64>>` - HashMap containing calculation results for each column
    pub fn aggregate(
        &self,
        column_names: &[&str],
        operation: &str,
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for &column_name in column_names {
            let result = match operation {
                "sum" => self.sum(column_name),
                "mean" => self.mean(column_name),
                "max" => self.max(column_name),
                "min" => self.min(column_name),
                "count" => self.count(column_name).map(|c| c as f64),
                _ => {
                    return Err(Error::Operation(format!(
                        "Operation '{}' is not supported",
                        operation
                    )))
                }
            };

            // Skip this column if there's an error
            if let Ok(value) = result {
                results.insert(column_name.to_string(), value);
            }
        }

        if results.is_empty() {
            Err(Error::OperationFailed(format!(
                "Operation '{}' failed for all columns",
                operation
            )))
        } else {
            Ok(results)
        }
    }

    /// Execute aggregation operations on all numeric columns
    ///
    /// # Arguments
    /// * `operation` - Operation name. One of "sum", "mean", "max", "min", "count"
    ///
    /// # Returns
    /// * `Result<HashMap<String, f64>>` - HashMap containing calculation results for each column
    pub fn aggregate_numeric(&self, operation: &str) -> Result<HashMap<String, f64>> {
        // Collect numeric column names
        let numeric_columns: Vec<&str> = self
            .column_names
            .iter()
            .filter(|&name| {
                if let Some(idx) = self.column_indices.get(name) {
                    matches!(self.columns[*idx], Column::Int64(_) | Column::Float64(_))
                } else {
                    false
                }
            })
            .map(|s| s.as_str())
            .collect();

        if numeric_columns.is_empty() {
            return Err(Error::OperationFailed(
                "No numeric columns exist".to_string(),
            ));
        }

        self.aggregate(&numeric_columns, operation)
    }

    /// Calculate sum of a column with custom JIT configuration
    ///
    /// # Arguments
    /// * `column_name` - Name of column to sum
    /// * `config` - Parallel configuration for JIT optimization
    ///
    /// # Returns
    /// * `Result<f64>` - Sum value
    pub fn sum_with_config(
        &self,
        column_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<f64> {
        match self.numeric_column(column_name)? {
            Column::Int64(col) => {
                // Exact integer accumulation: no copy, no overflow, no rounding.
                let (sum, count) = int_sum(col);
                if count == 0 {
                    return Ok(0.0);
                }
                Ok(sum as f64)
            }
            Column::Float64(col) => {
                let values = float_values(col);
                if values.is_empty() {
                    return Ok(0.0);
                }

                // Use JIT-optimized parallel sum for better performance
                let sum_func = parallel_sum_f64(config);
                Ok(sum_func.execute(values.as_ref()))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate mean of a column with custom JIT configuration
    ///
    /// # Arguments
    /// * `column_name` - Name of column to average
    /// * `config` - Parallel configuration for JIT optimization
    ///
    /// # Returns
    /// * `Result<f64>` - Mean value
    pub fn mean_with_config(
        &self,
        column_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<f64> {
        match self.numeric_column(column_name)? {
            Column::Int64(col) => {
                let (sum, count) = int_sum(col);
                if count == 0 {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }
                Ok(sum as f64 / count as f64)
            }
            Column::Float64(col) => {
                let values = float_values(col);
                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel mean for better performance
                Ok(parallel_mean_f64_value(values.as_ref(), config))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate max of a column with custom JIT configuration
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find maximum
    /// * `config` - Parallel configuration for JIT optimization
    ///
    /// # Returns
    /// * `Result<f64>` - Maximum value
    pub fn max_with_config(
        &self,
        column_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<f64> {
        match self.numeric_column(column_name)? {
            Column::Int64(col) => match int_values(col).max() {
                Some(value) => Ok(value as f64),
                None => Err(Error::Empty(format!("Column '{}' is empty", column_name))),
            },
            Column::Float64(col) => {
                let values = float_values(col);
                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel max for better performance
                let max_func = parallel_max_f64(config);
                Ok(max_func.execute(values.as_ref()))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate min of a column with custom JIT configuration
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find minimum
    /// * `config` - Parallel configuration for JIT optimization
    ///
    /// # Returns
    /// * `Result<f64>` - Minimum value
    pub fn min_with_config(
        &self,
        column_name: &str,
        config: Option<ParallelConfig>,
    ) -> Result<f64> {
        match self.numeric_column(column_name)? {
            Column::Int64(col) => match int_values(col).min() {
                Some(value) => Ok(value as f64),
                None => Err(Error::Empty(format!("Column '{}' is empty", column_name))),
            },
            Column::Float64(col) => {
                let values = float_values(col);
                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel min for better performance
                let min_func = parallel_min_f64(config);
                Ok(min_func.execute(values.as_ref()))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }
}

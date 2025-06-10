//! Aggregation functionality for OptimizedDataFrame

use rayon::prelude::*;
use std::collections::HashMap;

use crate::column::{Column, ColumnTrait};
use crate::error::{Error, Result};
use crate::optimized::jit::{
    parallel_max_f64, parallel_mean_f64_value, parallel_min_f64, parallel_sum_f64, ParallelConfig,
};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

impl OptimizedDataFrame {
    /// Calculate sum of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to sum
    ///
    /// # Returns
    /// * `Result<f64>` - Sum value
    pub fn sum(&self, column_name: &str) -> Result<f64> {
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Ok(0.0);
                }

                // Use JIT-optimized parallel sum for better performance
                let sum_func = parallel_sum_f64(None);
                Ok(sum_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Ok(0.0);
                }

                // Use JIT-optimized parallel sum for better performance
                let sum_func = parallel_sum_f64(None);
                Ok(sum_func.execute(&values))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate mean of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to average
    ///
    /// # Returns
    /// * `Result<f64>` - Mean value
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel mean for better performance
                Ok(parallel_mean_f64_value(&values, None))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel mean for better performance
                Ok(parallel_mean_f64_value(&values, None))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate maximum value of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find maximum
    ///
    /// # Returns
    /// * `Result<f64>` - Maximum value
    pub fn max(&self, column_name: &str) -> Result<f64> {
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel max for better performance
                let max_func = parallel_max_f64(None);
                Ok(max_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel max for better performance
                let max_func = parallel_max_f64(None);
                Ok(max_func.execute(&values))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Calculate minimum value of a column using JIT-optimized operations
    ///
    /// # Arguments
    /// * `column_name` - Name of column to find minimum
    ///
    /// # Returns
    /// * `Result<f64>` - Minimum value
    pub fn min(&self, column_name: &str) -> Result<f64> {
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel min for better performance
                let min_func = parallel_min_f64(None);
                Ok(min_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                // Use JIT-optimized parallel min for better performance
                let min_func = parallel_min_f64(None);
                Ok(min_func.execute(&values))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }

    /// Count non-NULL values in a column
    ///
    /// # Arguments
    /// * `column_name` - Target column name
    ///
    /// # Returns
    /// * `Result<usize>` - Count of non-NULL values
    pub fn count(&self, column_name: &str) -> Result<usize> {
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        let count = match column {
            Column::Int64(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::Float64(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::String(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
            Column::Boolean(col) => (0..col.len())
                .filter(|&i| col.get(i).map_or(false, |opt| opt.is_some()))
                .count(),
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
                let idx = self.column_indices.get(name).unwrap();
                matches!(self.columns[*idx], Column::Int64(_) | Column::Float64(_))
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
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Ok(0.0);
                }

                let sum_func = parallel_sum_f64(config);
                Ok(sum_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Ok(0.0);
                }

                let sum_func = parallel_sum_f64(config);
                Ok(sum_func.execute(&values))
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
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                Ok(parallel_mean_f64_value(&values, config))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                Ok(parallel_mean_f64_value(&values, config))
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
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                let max_func = parallel_max_f64(config);
                Ok(max_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                let max_func = parallel_max_f64(config);
                Ok(max_func.execute(&values))
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
        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        match column {
            Column::Int64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .map(|v| v as f64)
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                let min_func = parallel_min_f64(config);
                Ok(min_func.execute(&values))
            }
            Column::Float64(col) => {
                let values: Vec<f64> = (0..col.len())
                    .filter_map(|i| col.get(i).ok().flatten())
                    .collect();

                if values.is_empty() {
                    return Err(Error::Empty(format!("Column '{}' is empty", column_name)));
                }

                let min_func = parallel_min_f64(config);
                Ok(min_func.execute(&values))
            }
            _ => Err(Error::Type(format!(
                "Column '{}' is not a numeric type",
                column_name
            ))),
        }
    }
}

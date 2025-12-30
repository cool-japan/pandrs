//! Module providing pivot table functionality

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::dataframe::DataFrame;
use crate::error::{PandRSError, Result};
use crate::series::Series;

/// Aggregation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    /// Sum
    Sum,
    /// Mean
    Mean,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Count
    Count,
}

impl AggFunction {
    /// Get function name as string
    pub fn name(&self) -> &'static str {
        match self {
            AggFunction::Sum => "sum",
            AggFunction::Mean => "mean",
            AggFunction::Min => "min",
            AggFunction::Max => "max",
            AggFunction::Count => "count",
        }
    }

    /// Parse aggregation function from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sum" => Some(AggFunction::Sum),
            "mean" | "avg" | "average" => Some(AggFunction::Mean),
            "min" | "minimum" => Some(AggFunction::Min),
            "max" | "maximum" => Some(AggFunction::Max),
            "count" => Some(AggFunction::Count),
            _ => None,
        }
    }
}

/// Structure for creating pivot tables
#[derive(Debug)]
pub struct PivotTable<'a> {
    /// Source DataFrame
    df: &'a DataFrame,

    /// Column name to use as index
    index: String,

    /// Column name to use as columns
    columns: String,

    /// Column name to use as values
    values: String,

    /// Aggregation function
    aggfunc: AggFunction,
}

impl<'a> PivotTable<'a> {
    /// Create a new pivot table
    pub fn new(
        df: &'a DataFrame,
        index: String,
        columns: String,
        values: String,
        aggfunc: AggFunction,
    ) -> Result<Self> {
        // Verify required columns exist
        if !df.contains_column(&index) {
            return Err(PandRSError::Column(format!(
                "Index column '{}' not found",
                index
            )));
        }
        if !df.contains_column(&columns) {
            return Err(PandRSError::Column(format!(
                "Column column '{}' not found",
                columns
            )));
        }
        if !df.contains_column(&values) {
            return Err(PandRSError::Column(format!(
                "Value column '{}' not found",
                values
            )));
        }

        Ok(PivotTable {
            df,
            index,
            columns,
            values,
            aggfunc,
        })
    }

    /// Execute pivot table and generate a new DataFrame
    pub fn execute(&self) -> Result<DataFrame> {
        // Collect unique index and column values
        let mut index_values: HashSet<String> = HashSet::new();
        let mut column_values: HashSet<String> = HashSet::new();

        // Get index column and column data
        let index_values_vec = self.df.get_column_string_values(&self.index)?;
        let column_values_vec = self.df.get_column_string_values(&self.columns)?;
        let values_data_vec = self.df.get_column_numeric_values(&self.values)?;

        // Collect unique values
        for val in &index_values_vec {
            index_values.insert(val.clone());
        }

        for val in &column_values_vec {
            column_values.insert(val.clone());
        }

        // Create columns for result DataFrame
        let mut result_df = DataFrame::new();

        // Add index column
        let empty_index_values: Vec<String> = Vec::new();
        let empty_index_series = Series::new(empty_index_values, Some("index".to_string()))?;
        result_df.add_column(self.index.clone(), empty_index_series)?;

        // Add column values as columns in result DataFrame
        for column_val in &column_values {
            let empty_column_values: Vec<String> = Vec::new();
            let empty_series = Series::new(empty_column_values, Some(column_val.clone()))?;
            result_df.add_column(column_val.clone(), empty_series)?;
        }

        // Create map to store aggregated data
        // (index value, column value) -> list of aggregated values
        let mut aggregation_map: HashMap<(String, String), Vec<f64>> = HashMap::new();

        // Collect data
        for i in 0..self.df.row_count() {
            if i < index_values_vec.len()
                && i < column_values_vec.len()
                && i < values_data_vec.len()
            {
                let index_val = &index_values_vec[i];
                let column_val = &column_values_vec[i];
                let value = values_data_vec[i];

                let key = (index_val.clone(), column_val.clone());

                aggregation_map
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Aggregate values for each index row and column
        for index_val in &index_values {
            // Aggregate values for each column
            let mut row_data: HashMap<String, String> = HashMap::new();
            row_data.insert(self.index.clone(), index_val.clone());

            for column_val in &column_values {
                // Aggregate data for specific index value and column value
                let key = (index_val.clone(), column_val.clone());

                if let Some(values) = aggregation_map.get(&key) {
                    let agg_value = self.aggregate_values_from_vec(values)?;
                    let agg_value_str = agg_value.to_string();

                    // Add to result
                    row_data.insert(column_val.clone(), agg_value_str);
                } else {
                    // No data, use empty string
                    row_data.insert(column_val.clone(), String::new());
                }
            }

            // Add row data to DataFrame
            result_df.add_row_data_from_hashmap(row_data)?;
        }

        Ok(result_df)
    }

    /// Aggregate data for specific index value and column value
    fn aggregate_values_from_vec(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }

        match self.aggfunc {
            AggFunction::Sum => Ok(values.iter().sum()),
            AggFunction::Mean => {
                let sum: f64 = values.iter().sum();
                Ok(sum / values.len() as f64)
            }
            AggFunction::Min => {
                if let Some(min) = values.iter().fold(None, |min, &x| match min {
                    None => Some(x),
                    Some(y) => Some(if x < y { x } else { y }),
                }) {
                    Ok(min)
                } else {
                    Ok(0.0)
                }
            }
            AggFunction::Max => {
                if let Some(max) = values.iter().fold(None, |max, &x| match max {
                    None => Some(x),
                    Some(y) => Some(if x > y { x } else { y }),
                }) {
                    Ok(max)
                } else {
                    Ok(0.0)
                }
            }
            AggFunction::Count => Ok(values.len() as f64),
        }
    }
}

/// DataFrame extension: Pivot table functionality
impl DataFrame {
    /// Create a pivot table
    pub fn pivot_table(
        &self,
        index: &str,
        columns: &str,
        values: &str,
        aggfunc: AggFunction,
    ) -> Result<DataFrame> {
        let pivot = PivotTable::new(
            self,
            index.to_string(),
            columns.to_string(),
            values.to_string(),
            aggfunc,
        )?;

        pivot.execute()
    }

    /// Group by specified column
    pub fn groupby(&self, by: &str) -> Result<GroupBy> {
        if !self.contains_column(by) {
            return Err(PandRSError::Column(format!(
                "Grouping column '{}' not found",
                by
            )));
        }

        Ok(GroupBy {
            df: self,
            by: by.to_string(),
        })
    }
}

/// Structure representing a groupby operation
#[derive(Debug)]
pub struct GroupBy<'a> {
    /// Source DataFrame
    df: &'a DataFrame,

    /// Column name to group by
    by: String,
}

impl<'a> GroupBy<'a> {
    /// Execute aggregation operation
    pub fn agg(&self, columns: &[&str], aggfunc: AggFunction) -> Result<DataFrame> {
        // Verify each column exists
        for col in columns {
            if !self.df.contains_column(col) {
                return Err(PandRSError::Column(format!(
                    "Aggregation column '{}' not found",
                    col
                )));
            }
        }

        // Create columns for result DataFrame
        let mut result_df = DataFrame::new();

        // Get group key column
        let group_keys = self.df.get_column_string_values(&self.by)?;

        // Collect unique group keys
        let mut unique_keys: HashSet<String> = HashSet::new();
        for key in &group_keys {
            unique_keys.insert(key.clone());
        }

        // Add columns to result DataFrame
        // Add group key column
        let empty_key_values: Vec<String> = Vec::new();
        let empty_key_series = Series::new(empty_key_values, Some(self.by.clone()))?;
        result_df.add_column(self.by.clone(), empty_key_series)?;

        // Add result columns for each aggregation column
        for &col in columns {
            let col_name = format!("{}_{}", col, aggfunc.name());
            let empty_values: Vec<String> = Vec::new();
            let empty_series = Series::new(empty_values, Some(col_name.clone()))?;
            result_df.add_column(col_name, empty_series)?;
        }

        // Aggregate for each group
        for group_key in &unique_keys {
            // Collect row indices for each group
            let mut group_indices = Vec::new();
            for (i, key) in group_keys.iter().enumerate() {
                if key == group_key {
                    group_indices.push(i);
                }
            }

            // Aggregate for each column
            let mut row_data = HashMap::new();
            row_data.insert(self.by.clone(), group_key.clone());

            for &col in columns {
                let values = self.df.get_column_numeric_values(col)?;

                // Get values for the group
                let group_values: Vec<f64> = group_indices
                    .iter()
                    .filter_map(|&idx| {
                        if idx < values.len() {
                            Some(values[idx])
                        } else {
                            None
                        }
                    })
                    .collect();

                // Apply aggregation function
                let agg_value = match aggfunc {
                    AggFunction::Sum => group_values.iter().sum(),
                    AggFunction::Mean => {
                        if group_values.is_empty() {
                            0.0
                        } else {
                            group_values.iter().sum::<f64>() / group_values.len() as f64
                        }
                    }
                    AggFunction::Min => group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    AggFunction::Max => group_values
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    AggFunction::Count => group_values.len() as f64,
                };

                // Store result
                let col_name = format!("{}_{}", col, aggfunc.name());
                row_data.insert(col_name, agg_value.to_string());
            }

            // Add row data to DataFrame
            result_df.add_row_data_from_hashmap(row_data)?;
        }

        Ok(result_df)
    }

    /// Calculate sum
    pub fn sum(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Sum)
    }

    /// Calculate mean
    pub fn mean(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Mean)
    }

    /// Calculate minimum
    pub fn min(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Min)
    }

    /// Calculate maximum
    pub fn max(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Max)
    }

    /// Calculate count
    pub fn count(&self, columns: &[&str]) -> Result<DataFrame> {
        self.agg(columns, AggFunction::Count)
    }
}

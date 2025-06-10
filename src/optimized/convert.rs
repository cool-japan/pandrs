//! Module providing DataFrame conversion functionality

use std::collections::HashMap;

use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
use crate::DataValue;

/// Create an OptimizedDataFrame from a standard DataFrame
pub(crate) fn from_standard_dataframe(
    df: &crate::dataframe::DataFrame,
) -> Result<OptimizedDataFrame> {
    // Create a new SplitDataFrame (using internal implementation)
    let mut split_df = SplitDataFrame::new();

    for col_name in df.column_names() {
        // Try to get the column as a dynamic type to check existence
        if let Ok(col) = df.get_column::<String>(&col_name) {
            // Extract values one by one for Series iteration
            let mut values = Vec::new();
            for i in 0..col.len() {
                if let Some(val) = col.get(i) {
                    values.push(ToString::to_string(&val));
                } else {
                    values.push(String::new());
                }
            }

            // Infer type and add columns
            // Integer type
            let all_ints = values
                .iter()
                .all(|s| s.is_empty() || s.parse::<i64>().is_ok());
            if all_ints {
                let int_values: Vec<i64> = values
                    .iter()
                    .map(|s| s.parse::<i64>().unwrap_or(0))
                    .collect();
                split_df.add_column(
                    col_name.clone(),
                    Column::Int64(Int64Column::new(int_values)),
                )?;
                continue;
            }

            // Floating point type
            let all_floats = values
                .iter()
                .all(|s| s.is_empty() || s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                split_df.add_column(
                    col_name.clone(),
                    Column::Float64(Float64Column::new(float_values)),
                )?;
                continue;
            }

            // Boolean type
            let all_bools = values.iter().all(|s| {
                let s = s.to_lowercase();
                s.is_empty() || s == "true" || s == "false" || s == "1" || s == "0"
            });
            if all_bools {
                let bool_values: Vec<bool> = values
                    .iter()
                    .map(|s| {
                        let s = s.to_lowercase();
                        !s.is_empty() && (s == "true" || s == "1")
                    })
                    .collect();
                split_df.add_column(
                    col_name.clone(),
                    Column::Boolean(BooleanColumn::new(bool_values)),
                )?;
                continue;
            }

            // Default is string type
            split_df.add_column(col_name.clone(), Column::String(StringColumn::new(values)))?;
        }
    }

    // Get and set the index
    // DataFrame always has an index
    let df_index = df.get_index();

    // Process according to the type of DataFrameIndex
    match df_index {
        DataFrameIndex::Simple(simple_index) => {
            // Direct copy for Simple Index
            split_df.set_index_from_simple_index(simple_index.clone())?;
        }
        DataFrameIndex::Multi(_) => {
            // Multi-index is a future task
            split_df.set_default_index()?;
        }
    }

    // Convert SplitDataFrame to OptimizedDataFrame
    let mut opt_df = OptimizedDataFrame::new();

    // Copy column data (using public API)
    for name in split_df.column_names() {
        if let Ok(column_view) = split_df.column(name) {
            let column = column_view.column().clone();
            opt_df.add_column(name.clone(), column)?;
        }
    }

    // Set the index
    if let Some(split_index) = split_df.get_index() {
        // For simple index, copy from SplitDataFrame to OptimizedDataFrame
        if let DataFrameIndex::Simple(simple_index) = split_index {
            // Create a default index first to maintain compatibility
            let _ = opt_df.set_default_index();

            // At present, it is difficult to fully convert using only the public API,
            // so temporarily handle to ensure the integrity of the already created DataFrame
            // TODO: Create appropriate public API
            #[allow(deprecated)]
            {
                opt_df.set_index_from_simple_index_internal(simple_index.clone())?;
            }
        }
    }

    Ok(opt_df)
}

/// Convert OptimizedDataFrame to a standard DataFrame
pub(crate) fn to_standard_dataframe(
    df: &OptimizedDataFrame,
) -> Result<crate::dataframe::DataFrame> {
    // Use internal SplitDataFrame
    let mut split_df = SplitDataFrame::new();

    // Convert column data
    for col_name in df.column_names() {
        let col_view = df.column(col_name)?;
        let col = col_view.column();

        // Add columns to SplitDataFrame
        split_df.add_column(col_name.clone(), col.clone())?;
    }

    // Set the index if it exists
    if let Some(df_index) = df.get_index() {
        // Use appropriate methods instead of directly setting internal fields
        if let DataFrameIndex::Simple(simple_index) = df_index {
            split_df.set_index_from_simple_index(simple_index.clone())?;
        } else if let DataFrameIndex::Multi(multi_index) = df_index {
            // Multi-index is not supported yet
            split_df.set_default_index()?;
        }
    }

    // Convert to standard DataFrame
    let mut std_df = crate::dataframe::DataFrame::new();

    // Process each column
    for col_name in split_df.column_names() {
        let col_view = split_df.column(col_name)?;
        let col = col_view.column();

        match col {
            Column::Int64(int_col) => {
                // Create series from Int64Column
                let series = crate::series::Series::new(
                    (0..int_col.len())
                        .map(|i| {
                            let val = int_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(Box::new(v.clone())),
                                _ => None,
                            }
                        })
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_column(col_name.clone(), series)?;
            }
            Column::Float64(float_col) => {
                // Create series from Float64Column
                let series = crate::series::Series::new(
                    (0..float_col.len())
                        .map(|i| {
                            let val = float_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(Box::new(v.clone())),
                                _ => None,
                            }
                        })
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_column(col_name.clone(), series)?;
            }
            Column::String(str_col) => {
                // Create series from StringColumn
                let series = crate::series::Series::new(
                    (0..str_col.len())
                        .map(|i| {
                            let val = str_col.get(i);
                            match val {
                                Ok(Some(s)) => Some(Box::new(s.to_string())),
                                _ => None,
                            }
                        })
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_column(col_name.clone(), series)?;
            }
            Column::Boolean(bool_col) => {
                // Create series from BooleanColumn
                let series = crate::series::Series::new(
                    (0..bool_col.len())
                        .map(|i| {
                            let val = bool_col.get(i);
                            match val {
                                Ok(Some(v)) => Some(Box::new(v.clone())),
                                _ => None,
                            }
                        })
                        .collect(),
                    Some(col_name.clone()),
                )?;
                std_df.add_column(col_name.clone(), series)?;
            }
        }
    }

    // Set the index
    if let Some(split_index) = split_df.get_index() {
        match split_index {
            DataFrameIndex::Simple(simple_index) => {
                // Set as string-based Simple Index
                std_df.set_index(simple_index.clone())?;
            }
            DataFrameIndex::Multi(multi_index) => {
                // For multi-index
                std_df.set_multi_index(multi_index.clone())?;
            }
        }
    }

    Ok(std_df)
}

/// Public function to convert a standard DataFrame to an optimized OptimizedDataFrame
pub fn optimize_dataframe(df: &crate::dataframe::DataFrame) -> Result<OptimizedDataFrame> {
    from_standard_dataframe(df)
}

/// Public function to convert an OptimizedDataFrame to a standard DataFrame
pub fn standard_dataframe(df: &OptimizedDataFrame) -> Result<crate::dataframe::DataFrame> {
    to_standard_dataframe(df)
}

//! # Conversion Utilities for DataFusion
//!
//! This module provides utilities for converting between PandRS and DataFusion data types.

#[cfg(feature = "distributed")]
use crate::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use crate::error::{Error, Result};
#[cfg(feature = "distributed")]
use crate::na::NA;
#[cfg(feature = "distributed")]
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, NullArray, StringArray,
    TimestampNanosecondArray,
};
#[cfg(feature = "distributed")]
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
#[cfg(feature = "distributed")]
use std::sync::Arc;

/// Converts a PandRS DataFrame to Arrow record batches
#[cfg(feature = "distributed")]
pub fn dataframe_to_record_batches(
    df: &DataFrame,
    batch_size: usize,
) -> Result<Vec<arrow::record_batch::RecordBatch>> {
    // Create a schema based on DataFrame columns
    let columns = df.columns()?;
    let mut fields = Vec::with_capacity(columns.len());

    for column in columns {
        let data_type = match column.data_type() {
            "boolean" => DataType::Boolean,
            "int64" => DataType::Int64,
            "float64" => DataType::Float64,
            "string" => DataType::Utf8,
            "timestamp" => DataType::Timestamp(TimeUnit::Nanosecond, None),
            _ => {
                return Err(Error::InvalidInput(format!(
                    "Unsupported data type for DataFusion: {}",
                    column.data_type()
                )));
            }
        };

        fields.push(Field::new(
            column.name().unwrap_or("unnamed"),
            data_type,
            true, // nullable
        ));
    }

    let schema = Arc::new(Schema::new(fields));

    // Split the data into batches
    let row_count = df.row_count()?;
    let batch_count = (row_count + batch_size - 1) / batch_size;
    let mut batches = Vec::with_capacity(batch_count);

    for batch_idx in 0..batch_count {
        let start_row = batch_idx * batch_size;
        let end_row = std::cmp::min(start_row + batch_size, row_count);
        let batch_row_count = end_row - start_row;

        let mut batch_columns = Vec::with_capacity(columns.len());

        for column in &columns {
            let array = match column.data_type() {
                "boolean" => {
                    let mut builder = arrow::array::BooleanBuilder::new();
                    for i in start_row..end_row {
                        match column.get_value(i)? {
                            NA::Value(val) => {
                                let bool_val = val != 0.0;
                                builder.append_value(bool_val);
                            }
                            NA::NA => {
                                builder.append_null();
                            }
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
                "int64" => {
                    let mut builder = arrow::array::Int64Builder::new();
                    for i in start_row..end_row {
                        match column.get_value(i)? {
                            NA::Value(val) => {
                                builder.append_value(val as i64);
                            }
                            NA::NA => {
                                builder.append_null();
                            }
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
                "float64" => {
                    let mut builder = arrow::array::Float64Builder::new();
                    for i in start_row..end_row {
                        match column.get_value(i)? {
                            NA::Value(val) => {
                                builder.append_value(val);
                            }
                            NA::NA => {
                                builder.append_null();
                            }
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
                "string" => {
                    let mut builder = arrow::array::StringBuilder::new();
                    for i in start_row..end_row {
                        match column.get_value_as_str(i)? {
                            Some(val) => {
                                builder.append_value(val);
                            }
                            None => {
                                builder.append_null();
                            }
                        }
                    }
                    Arc::new(builder.finish()) as ArrayRef
                }
                "timestamp" => {
                    // Implementation will be provided in a future PR
                    Arc::new(NullArray::new(batch_row_count)) as ArrayRef
                }
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "Unsupported data type for DataFusion: {}",
                        column.data_type()
                    )));
                }
            };

            batch_columns.push(array);
        }

        let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), batch_columns)?;

        batches.push(batch);
    }

    Ok(batches)
}

/// Converts Arrow record batches to a PandRS DataFrame
#[cfg(feature = "distributed")]
pub fn record_batches_to_dataframe(
    batches: &[arrow::record_batch::RecordBatch],
) -> Result<DataFrame> {
    if batches.is_empty() {
        return Ok(DataFrame::new());
    }

    let schema = batches[0].schema();
    let mut df = DataFrame::new();

    // Process each column
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        let data_type = field.data_type();

        match data_type {
            DataType::Boolean => {
                let values = extract_boolean_values(batches, col_idx)?;
                df.add_column(name.clone(), values)?;
            }
            DataType::Int64 => {
                let values = extract_int64_values(batches, col_idx)?;
                df.add_column(name.clone(), values)?;
            }
            DataType::Float64 => {
                let values = extract_float64_values(batches, col_idx)?;
                df.add_column(name.clone(), values)?;
            }
            DataType::Utf8 => {
                let values = extract_string_values(batches, col_idx)?;
                df.add_column(name.clone(), values)?;
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let values = extract_timestamp_values(batches, col_idx)?;
                df.add_column(name.clone(), values)?;
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "Unsupported Arrow data type: {}",
                    data_type
                )));
            }
        }
    }

    Ok(df)
}

/// Extracts boolean values from record batches
#[cfg(feature = "distributed")]
fn extract_boolean_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    let mut values = Vec::new();

    for batch in batches {
        let array = batch.column(col_idx);
        if let Some(boolean_array) = array.as_any().downcast_ref::<BooleanArray>() {
            for i in 0..boolean_array.len() {
                if boolean_array.is_null(i) {
                    values.push(NA::NA);
                } else {
                    let val = if boolean_array.value(i) { 1.0 } else { 0.0 };
                    values.push(NA::Value(val));
                }
            }
        } else {
            return Err(Error::InvalidInput(
                "Column is not a boolean array".to_string(),
            ));
        }
    }

    Ok(values)
}

/// Extracts int64 values from record batches
#[cfg(feature = "distributed")]
fn extract_int64_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    let mut values = Vec::new();

    for batch in batches {
        let array = batch.column(col_idx);
        if let Some(int_array) = array.as_any().downcast_ref::<Int64Array>() {
            for i in 0..int_array.len() {
                if int_array.is_null(i) {
                    values.push(NA::NA);
                } else {
                    values.push(NA::Value(int_array.value(i) as f64));
                }
            }
        } else {
            return Err(Error::InvalidInput(
                "Column is not an int64 array".to_string(),
            ));
        }
    }

    Ok(values)
}

/// Extracts float64 values from record batches
#[cfg(feature = "distributed")]
fn extract_float64_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    let mut values = Vec::new();

    for batch in batches {
        let array = batch.column(col_idx);
        if let Some(float_array) = array.as_any().downcast_ref::<Float64Array>() {
            for i in 0..float_array.len() {
                if float_array.is_null(i) {
                    values.push(NA::NA);
                } else {
                    values.push(NA::Value(float_array.value(i)));
                }
            }
        } else {
            return Err(Error::InvalidInput(
                "Column is not a float64 array".to_string(),
            ));
        }
    }

    Ok(values)
}

/// Extracts string values from record batches
#[cfg(feature = "distributed")]
fn extract_string_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    // Implementation will be provided in a future PR
    Err(Error::NotImplemented(
        "String extraction not yet implemented".to_string(),
    ))
}

/// Extracts timestamp values from record batches
#[cfg(feature = "distributed")]
fn extract_timestamp_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    // Implementation will be provided in a future PR
    Err(Error::NotImplemented(
        "Timestamp extraction not yet implemented".to_string(),
    ))
}

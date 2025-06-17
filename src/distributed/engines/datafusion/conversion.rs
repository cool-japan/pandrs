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
use crate::series::Series;
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
    use arrow::datatypes::Schema;
    use arrow::record_batch::RecordBatch;

    if df.nrows() == 0 {
        let schema = Arc::new(Schema::new(vec![] as Vec<arrow::datatypes::Field>));
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }

    // Build schema from DataFrame columns
    let mut fields = Vec::new();
    let column_names = df.column_names();

    for column_name in &column_names {
        // Determine field type by examining column data
        let field_type = determine_arrow_type(df, column_name)?;
        let field = Field::new(column_name, field_type, true); // Allow nulls
        fields.push(field);
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batches = Vec::new();

    // Split data into batches
    let total_rows = df.nrows();
    let num_batches = (total_rows + batch_size - 1) / batch_size;

    for batch_idx in 0..num_batches {
        let start_row = batch_idx * batch_size;
        let end_row = std::cmp::min(start_row + batch_size, total_rows);
        let batch_row_count = end_row - start_row;

        // Build arrays for this batch
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for column_name in &column_names {
            let array = build_array_from_dataframe(df, column_name, start_row, batch_row_count)?;
            arrays.push(array);
        }

        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| Error::InvalidValue(format!("Failed to create RecordBatch: {}", e)))?;
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
                let series = Series::new(
                    values.into_iter().map(|v| format!("{:?}", v)).collect(),
                    Some(name.clone()),
                )?;
                df.add_column(name.clone(), series)?;
            }
            DataType::Int64 => {
                let values = extract_int64_values(batches, col_idx)?;
                let series = Series::new(
                    values.into_iter().map(|v| format!("{:?}", v)).collect(),
                    Some(name.clone()),
                )?;
                df.add_column(name.clone(), series)?;
            }
            DataType::Float64 => {
                let values = extract_float64_values(batches, col_idx)?;
                let series = Series::new(
                    values.into_iter().map(|v| format!("{:?}", v)).collect(),
                    Some(name.clone()),
                )?;
                df.add_column(name.clone(), series)?;
            }
            DataType::Utf8 => {
                let values = extract_string_values(batches, col_idx)?;
                let series = Series::new(
                    values.into_iter().map(|v| format!("{:?}", v)).collect(),
                    Some(name.clone()),
                )?;
                df.add_column(name.clone(), series)?;
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let values = extract_timestamp_values(batches, col_idx)?;
                let series = Series::new(
                    values.into_iter().map(|v| format!("{:?}", v)).collect(),
                    Some(name.clone()),
                )?;
                df.add_column(name.clone(), series)?;
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
    let mut values = Vec::new();

    for batch in batches {
        let array = batch.column(col_idx);
        if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
            for i in 0..string_array.len() {
                if string_array.is_null(i) {
                    values.push(NA::NA);
                } else {
                    // Convert string to f64 if possible, otherwise use hash
                    let string_val = string_array.value(i);
                    if let Ok(num_val) = string_val.parse::<f64>() {
                        values.push(NA::Value(num_val));
                    } else {
                        // Use string hash as numeric value
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        string_val.hash(&mut hasher);
                        values.push(NA::Value(hasher.finish() as f64));
                    }
                }
            }
        } else {
            return Err(Error::InvalidInput(
                "Column is not a string array".to_string(),
            ));
        }
    }

    Ok(values)
}

/// Extracts timestamp values from record batches
#[cfg(feature = "distributed")]
fn extract_timestamp_values(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
) -> Result<Vec<NA<f64>>> {
    let mut values = Vec::new();

    for batch in batches {
        let array = batch.column(col_idx);
        if let Some(timestamp_array) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
            for i in 0..timestamp_array.len() {
                if timestamp_array.is_null(i) {
                    values.push(NA::NA);
                } else {
                    values.push(NA::Value(timestamp_array.value(i) as f64));
                }
            }
        } else {
            return Err(Error::InvalidInput(
                "Column is not a timestamp array".to_string(),
            ));
        }
    }

    Ok(values)
}

/// Determines the Arrow data type for a DataFrame column
#[cfg(feature = "distributed")]
fn determine_arrow_type(df: &DataFrame, column_name: &str) -> Result<DataType> {
    // Get a sample of values to determine type
    let sample_size = std::cmp::min(100, df.nrows());

    if sample_size == 0 {
        return Ok(DataType::Utf8); // Default to string for empty columns
    }

    // Try to get column values as strings first to analyze them
    let string_values = df.get_column_string_values(column_name)?;

    // Analyze the sample to determine type
    let mut has_ints = 0;
    let mut has_floats = 0;
    let mut has_bools = 0;
    let mut has_dates = 0;
    let mut total_non_empty = 0;

    for value in string_values.iter().take(sample_size) {
        if value.is_empty() {
            continue;
        }
        total_non_empty += 1;

        // Check for boolean
        let lower_val = value.to_lowercase();
        if lower_val == "true" || lower_val == "false" || lower_val == "t" || lower_val == "f" {
            has_bools += 1;
            continue;
        }

        // Check for integer
        if value.parse::<i64>().is_ok() {
            has_ints += 1;
            continue;
        }

        // Check for float
        if value.parse::<f64>().is_ok() {
            has_floats += 1;
            continue;
        }

        // Check for date/timestamp (basic patterns)
        if value.contains('-') && (value.contains(':') || value.len() == 10) {
            if chrono::DateTime::parse_from_rfc3339(value).is_ok()
                || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").is_ok()
                || chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok()
            {
                has_dates += 1;
                continue;
            }
        }
    }

    if total_non_empty == 0 {
        return Ok(DataType::Utf8); // Default to string for empty data
    }

    // Determine type based on majority
    let threshold = total_non_empty / 2; // At least 50% must match

    if has_bools > threshold {
        Ok(DataType::Boolean)
    } else if has_dates > threshold {
        Ok(DataType::Timestamp(TimeUnit::Nanosecond, None))
    } else if has_ints > threshold {
        Ok(DataType::Int64)
    } else if has_floats > threshold {
        Ok(DataType::Float64)
    } else {
        Ok(DataType::Utf8) // Default to string
    }
}

/// Builds an Arrow array from DataFrame column data
#[cfg(feature = "distributed")]
fn build_array_from_dataframe(
    df: &DataFrame,
    column_name: &str,
    start_row: usize,
    row_count: usize,
) -> Result<ArrayRef> {
    let string_values = df.get_column_string_values(column_name)?;
    let data_type = determine_arrow_type(df, column_name)?;

    // Extract the relevant slice
    let end_row = start_row + row_count;
    let slice = &string_values[start_row..std::cmp::min(end_row, string_values.len())];

    match data_type {
        DataType::Boolean => {
            let mut builder = arrow::array::BooleanBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    let lower_val = value.to_lowercase();
                    let bool_val = lower_val == "true" || lower_val == "t" || lower_val == "1";
                    builder.append_value(bool_val);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Int64 => {
            let mut builder = arrow::array::Int64Builder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else if let Ok(int_val) = value.parse::<i64>() {
                    builder.append_value(int_val);
                } else {
                    builder.append_null();
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = arrow::array::Float64Builder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else if let Ok(float_val) = value.parse::<f64>() {
                    builder.append_value(float_val);
                } else {
                    builder.append_null();
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let mut builder = arrow::array::TimestampNanosecondBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    // Try to parse various timestamp formats
                    let timestamp_nanos = if let Ok(dt) =
                        chrono::DateTime::parse_from_rfc3339(value)
                    {
                        dt.timestamp_nanos_opt().unwrap_or(0)
                    } else if let Ok(ndt) =
                        chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                    {
                        ndt.timestamp_nanos_opt().unwrap_or(0)
                    } else if let Ok(nd) = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d") {
                        nd.and_hms_opt(0, 0, 0)
                            .unwrap_or_default()
                            .timestamp_nanos_opt()
                            .unwrap_or(0)
                    } else {
                        0 // Default timestamp
                    };
                    builder.append_value(timestamp_nanos);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = arrow::array::StringBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    builder.append_value(value);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Default to string for unsupported types
            let mut builder = arrow::array::StringBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    builder.append_value(value);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

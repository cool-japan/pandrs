//! # DataFusion Type Conversion
//!
//! This module provides conversion utilities between PandRS and DataFusion data types.

#[cfg(feature = "distributed")]
use std::sync::Arc;
#[cfg(feature = "distributed")]
use std::collections::HashMap;

#[cfg(feature = "distributed")]
use crate::error::{Result, Error};
#[cfg(feature = "distributed")]
use crate::dataframe::{DataFrame, DataType as PandrsDataType};
#[cfg(feature = "distributed")]
use crate::column::{Column, ColumnType};
#[cfg(feature = "distributed")]
use crate::series::Series;
#[cfg(feature = "distributed")]
use crate::na::NA;

#[cfg(feature = "distributed")]
use arrow::datatypes::{DataType, Schema, Field, SchemaRef};
#[cfg(feature = "distributed")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "distributed")]
use arrow::array::{Int64Array, Float64Array, StringArray, BooleanArray, ArrayRef, Array};

/// Converts a PandRS DataFrame to Arrow RecordBatches
#[cfg(feature = "distributed")]
pub fn dataframe_to_record_batches(
    df: &DataFrame,
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    // Create the schema first
    let schema = pandrs_schema_to_arrow(&df.schema()?)?;
    let schema_ref = Arc::new(schema);

    // Get total number of rows
    let row_count = df.shape()?.0;

    // Calculate number of batches
    let batch_count = (row_count + batch_size - 1) / batch_size;

    let mut batches = Vec::with_capacity(batch_count);

    for batch_idx in 0..batch_count {
        let start_row = batch_idx * batch_size;
        let end_row = std::cmp::min(start_row + batch_size, row_count);
        let batch_row_count = end_row - start_row;

        // Convert each column to Arrow array
        let mut arrays = Vec::with_capacity(df.ncols()?);

        for (col_name, col_type) in df.schema()? {
            let column = df.column(&col_name)?;
            let arrow_array = column_to_arrow_array(column, start_row, batch_row_count)?;
            arrays.push(arrow_array);
        }

        // Create RecordBatch
        let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
            .map_err(|e| Error::DistributedProcessing(format!("Failed to create record batch: {}", e)))?;

        batches.push(batch);
    }

    Ok(batches)
}

/// Converts an Arrow array of a specific type to a PandRS column
#[cfg(feature = "distributed")]
fn column_to_arrow_array(
    column: &dyn Column,
    start_row: usize,
    row_count: usize,
) -> Result<ArrayRef> {
    let column_type = column.column_type();

    match column_type {
        ColumnType::Int64 => {
            let mut values = Vec::with_capacity(row_count);

            for i in start_row..(start_row + row_count) {
                if i < column.len() {
                    values.push(column.get_i64(i)?);
                } else {
                    values.push(0); // Padding if needed
                }
            }

            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        },
        ColumnType::Float64 => {
            let mut values = Vec::with_capacity(row_count);

            for i in start_row..(start_row + row_count) {
                if i < column.len() {
                    values.push(column.get_f64(i)?);
                } else {
                    values.push(0.0); // Padding if needed
                }
            }

            Ok(Arc::new(Float64Array::from(values)) as ArrayRef)
        },
        ColumnType::String => {
            let mut values = Vec::with_capacity(row_count);

            for i in start_row..(start_row + row_count) {
                if i < column.len() {
                    values.push(column.get_string(i)?);
                } else {
                    values.push(String::new()); // Padding if needed
                }
            }

            Ok(Arc::new(StringArray::from(values)) as ArrayRef)
        },
        ColumnType::Boolean => {
            let mut values = Vec::with_capacity(row_count);

            for i in start_row..(start_row + row_count) {
                if i < column.len() {
                    values.push(column.get_bool(i)?);
                } else {
                    values.push(false); // Padding if needed
                }
            }

            Ok(Arc::new(BooleanArray::from(values)) as ArrayRef)
        },
        _ => Err(Error::NotImplemented(
            format!("Conversion of column type {:?} to Arrow array is not implemented", column_type)
        )),
    }
}

/// Converts Arrow RecordBatches to a PandRS DataFrame
#[cfg(feature = "distributed")]
pub fn record_batches_to_dataframe(
    batches: &[RecordBatch],
) -> Result<DataFrame> {
    if batches.is_empty() {
        return Ok(DataFrame::new());
    }

    // Get schema from the first batch
    let schema = batches[0].schema();

    // Create a new DataFrame
    let mut df = DataFrame::new();

    // Process each column
    for field_idx in 0..schema.fields().len() {
        let field = schema.field(field_idx);
        let field_name = field.name();
        let field_type = field.data_type();

        // Collect all values for this column across batches
        match field_type {
            DataType::Int64 => {
                let mut values = Vec::new();

                for batch in batches {
                    let array = batch.column(field_idx);
                    let int_array = array.as_any().downcast_ref::<Int64Array>()
                        .ok_or_else(|| Error::DistributedProcessing(
                            format!("Failed to convert column {} to Int64Array", field_name)
                        ))?;

                    for i in 0..int_array.len() {
                        values.push(int_array.value(i));
                    }
                }

                df.add_column(field_name.to_string(), values)?;
            },
            DataType::Float64 => {
                let mut values = Vec::new();

                for batch in batches {
                    let array = batch.column(field_idx);
                    let float_array = array.as_any().downcast_ref::<Float64Array>()
                        .ok_or_else(|| Error::DistributedProcessing(
                            format!("Failed to convert column {} to Float64Array", field_name)
                        ))?;

                    for i in 0..float_array.len() {
                        values.push(float_array.value(i));
                    }
                }

                df.add_column(field_name.to_string(), values)?;
            },
            DataType::Utf8 => {
                let mut values = Vec::new();

                for batch in batches {
                    let array = batch.column(field_idx);
                    let string_array = array.as_any().downcast_ref::<StringArray>()
                        .ok_or_else(|| Error::DistributedProcessing(
                            format!("Failed to convert column {} to StringArray", field_name)
                        ))?;

                    for i in 0..string_array.len() {
                        if string_array.is_null(i) {
                            values.push(String::new());
                        } else {
                            values.push(string_array.value(i).to_string());
                        }
                    }
                }

                df.add_column(field_name.to_string(), values)?;
            },
            DataType::Boolean => {
                let mut values = Vec::new();

                for batch in batches {
                    let array = batch.column(field_idx);
                    let bool_array = array.as_any().downcast_ref::<BooleanArray>()
                        .ok_or_else(|| Error::DistributedProcessing(
                            format!("Failed to convert column {} to BooleanArray", field_name)
                        ))?;

                    for i in 0..bool_array.len() {
                        values.push(bool_array.value(i));
                    }
                }

                df.add_column(field_name.to_string(), values)?;
            },
            _ => {
                return Err(Error::NotImplemented(
                    format!("Conversion of Arrow data type {:?} to PandRS is not implemented", field_type)
                ));
            }
        }
    }

    Ok(df)
}

/// Converts an Arrow Schema to a PandRS schema representation
#[cfg(feature = "distributed")]
pub fn arrow_schema_to_pandrs(schema: &Schema) -> Result<Vec<(String, PandrsDataType)>> {
    let mut result = Vec::with_capacity(schema.fields().len());

    for field in schema.fields() {
        let name = field.name().clone();
        let data_type = arrow_type_to_pandrs(field.data_type())?;
        result.push((name, data_type));
    }

    Ok(result)
}

/// Converts a PandRS schema representation to an Arrow Schema
#[cfg(feature = "distributed")]
pub fn pandrs_schema_to_arrow(schema: &[(String, PandrsDataType)]) -> Result<Schema> {
    let mut fields = Vec::with_capacity(schema.len());

    for (name, data_type) in schema {
        let arrow_type = pandrs_type_to_arrow(data_type)?;
        let field = Field::new(name, arrow_type, true); // Allow nulls for now
        fields.push(field);
    }

    Ok(Schema::new(fields))
}

/// Maps PandRS data types to Arrow data types
#[cfg(feature = "distributed")]
pub fn pandrs_type_to_arrow(data_type: &PandrsDataType) -> Result<DataType> {
    match data_type {
        PandrsDataType::Int64 => Ok(DataType::Int64),
        PandrsDataType::Float64 => Ok(DataType::Float64),
        PandrsDataType::String => Ok(DataType::Utf8),
        PandrsDataType::Boolean => Ok(DataType::Boolean),
        _ => Err(Error::NotImplemented(
            format!("Conversion of PandRS data type {:?} to Arrow is not implemented", data_type)
        )),
    }
}

/// Maps Arrow data types to PandRS data types
#[cfg(feature = "distributed")]
pub fn arrow_type_to_pandrs(data_type: &DataType) -> Result<PandrsDataType> {
    match data_type {
        DataType::Int64 => Ok(PandrsDataType::Int64),
        DataType::Float64 => Ok(PandrsDataType::Float64),
        DataType::Utf8 => Ok(PandrsDataType::String),
        DataType::Boolean => Ok(PandrsDataType::Boolean),
        _ => Err(Error::NotImplemented(
            format!("Conversion of Arrow data type {:?} to PandRS is not implemented", data_type)
        )),
    }
}
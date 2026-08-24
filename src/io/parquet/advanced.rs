//! Projection/row-group-aware reading and enhanced-type-aware writing.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::convert::record_batches_to_dataframe;
use super::core::{validate_compression, ParquetReadOptions, ParquetWriteOptions};

/// Read a DataFrame from a Parquet file with advanced options
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `options` - Advanced reading options
///
/// # Returns
///
/// * `Result<DataFrame>` - The read DataFrame, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::{read_parquet_advanced, ParquetReadOptions};
///
/// // Read only specific columns
/// let options = ParquetReadOptions {
///     columns: Some(vec!["name".to_string(), "age".to_string()]),
///     use_threads: true,
///     ..Default::default()
/// };
/// let df = read_parquet_advanced("data.parquet", options).expect("operation should succeed");
/// ```
pub fn read_parquet_advanced(
    path: impl AsRef<Path>,
    options: ParquetReadOptions,
) -> Result<DataFrame> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::IoError(format!("Failed to parse Parquet file: {}", e)))?;

    // Configure batch size if specified
    if let Some(batch_size) = options.batch_size {
        builder = builder.with_batch_size(batch_size);
    }

    // Configure row groups if specified
    if let Some(row_groups) = options.row_groups {
        builder = builder.with_row_groups(row_groups);
    }

    // Configure column projection if specified
    if let Some(columns) = &options.columns {
        let schema = builder.schema();
        let mut projection_indices = Vec::new();

        for col_name in columns {
            let idx = schema
                .fields()
                .iter()
                .position(|field| field.name() == col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.clone()))?;
            projection_indices.push(idx);
        }

        use parquet::arrow::ProjectionMask;
        let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices);
        builder = builder.with_projection(mask);
    }

    let reader = builder
        .build()
        .map_err(|e| Error::IoError(format!("Failed to read Parquet file: {}", e)))?;

    // Use the reader's own (projected) schema, not the full file schema
    // captured before `with_projection` above — the two disagree whenever
    // `options.columns` selects a subset, which previously zipped column
    // names from the wrong field against the narrower batches and produced
    // either wrong names or an out-of-bounds panic.
    let schema = reader.schema();

    // Read all record batches
    let mut all_batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::IoError(format!("Failed to read record batch: {}", e)))?;
        all_batches.push(batch);
    }

    if all_batches.is_empty() {
        return Ok(DataFrame::new());
    }

    // Convert to DataFrame with enhanced type support
    record_batches_to_dataframe(&all_batches, schema)
}

/// Write a DataFrame to a Parquet file with advanced options
///
/// # Arguments
///
/// * `df` - The DataFrame to write
/// * `path` - Path to the output Parquet file
/// * `options` - Advanced writing options
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::{write_parquet_advanced, ParquetWriteOptions, ParquetCompression};
/// use pandrs::optimized::dataframe::OptimizedDataFrame;
///
/// // Create sample dataframe
/// let df = OptimizedDataFrame::new();
///
/// let options = ParquetWriteOptions {
///     compression: ParquetCompression::Snappy,
///     row_group_size: Some(100000),
///     enable_dictionary: true,
///     ..Default::default()
/// };
/// write_parquet_advanced(&df, "output.parquet", options).expect("operation should succeed");
/// ```
pub fn write_parquet_advanced(
    df: &OptimizedDataFrame,
    path: impl AsRef<Path>,
    options: ParquetWriteOptions,
) -> Result<()> {
    validate_compression(options.compression)?;

    // Create Arrow schema with enhanced type detection.
    //
    // Note: `options.enable_dictionary` controls the Parquet *storage*
    // encoding only, via `WriterProperties::set_dictionary_enabled` below —
    // it does not change a string column's Arrow logical type. Declaring the
    // schema field itself as `Dictionary(Int32, Utf8)` here (while the array
    // built below is always a plain `StringArray`) used to make every write
    // with a string column fail at `RecordBatch::try_new`, because the
    // declared field type never matched the array's actual data type.
    let schema_fields: Vec<Field> = df
        .column_names()
        .iter()
        .filter_map(|col_name| {
            if let Ok(col_view) = df.column(col_name) {
                let data_type = match col_view.column_type() {
                    crate::column::ColumnType::Int64 => DataType::Int64,
                    crate::column::ColumnType::Float64 => DataType::Float64,
                    crate::column::ColumnType::Boolean => DataType::Boolean,
                    crate::column::ColumnType::String => DataType::Utf8,
                };
                Some(Field::new(col_name, data_type, true))
            } else {
                None
            }
        })
        .collect();

    let schema = Schema::new(schema_fields);
    let schema_ref = Arc::new(schema);

    // Enhanced writer properties
    let mut props_builder =
        WriterProperties::builder().set_compression(Compression::from(options.compression));

    if let Some(row_group_size) = options.row_group_size {
        props_builder = props_builder.set_max_row_group_row_count(Some(row_group_size));
    }

    if let Some(page_size) = options.page_size {
        props_builder = props_builder.set_data_page_size_limit(page_size);
    }

    // Always set explicitly (not just when `true`): leaving this unset when
    // the caller asks for `enable_dictionary: false` previously left the
    // `parquet` crate's own default (dictionary encoding enabled) in place,
    // silently ignoring the caller's choice.
    props_builder = props_builder.set_dictionary_enabled(options.enable_dictionary);

    let props = props_builder.build();

    // Create arrays with the same logic as before
    let arrays: Vec<ArrayRef> = df
        .column_names()
        .iter()
        .filter_map(|col_name| {
            let col_view = match df.column(col_name) {
                Ok(s) => s,
                Err(_) => return None,
            };

            // Use existing array creation logic
            match col_view.column_type() {
                crate::column::ColumnType::Int64 => {
                    if let Some(int_col) = col_view.as_int64() {
                        let mut values = Vec::with_capacity(df.row_count());
                        let mut validity = Vec::with_capacity(df.row_count());

                        for i in 0..df.row_count() {
                            match int_col.get(i) {
                                Ok(Some(val)) => {
                                    values.push(val);
                                    validity.push(true);
                                }
                                Ok(None) => {
                                    values.push(0);
                                    validity.push(false);
                                }
                                Err(_) => {
                                    values.push(0);
                                    validity.push(false);
                                }
                            }
                        }

                        let array = Int64Array::new(values.into(), Some(validity.into()));
                        Some(Arc::new(array) as ArrayRef)
                    } else {
                        None
                    }
                }
                crate::column::ColumnType::Float64 => {
                    if let Some(float_col) = col_view.as_float64() {
                        let mut values = Vec::with_capacity(df.row_count());
                        let mut validity = Vec::with_capacity(df.row_count());

                        for i in 0..df.row_count() {
                            match float_col.get(i) {
                                Ok(Some(val)) => {
                                    values.push(val);
                                    validity.push(true);
                                }
                                Ok(None) => {
                                    values.push(0.0);
                                    validity.push(false);
                                }
                                Err(_) => {
                                    values.push(0.0);
                                    validity.push(false);
                                }
                            }
                        }

                        let array = Float64Array::new(values.into(), Some(validity.into()));
                        Some(Arc::new(array) as ArrayRef)
                    } else {
                        None
                    }
                }
                crate::column::ColumnType::Boolean => {
                    if let Some(bool_col) = col_view.as_boolean() {
                        let mut values = Vec::with_capacity(df.row_count());
                        let mut validity = Vec::with_capacity(df.row_count());

                        for i in 0..df.row_count() {
                            match bool_col.get(i) {
                                Ok(Some(val)) => {
                                    values.push(val);
                                    validity.push(true);
                                }
                                Ok(None) => {
                                    values.push(false);
                                    validity.push(false);
                                }
                                Err(_) => {
                                    values.push(false);
                                    validity.push(false);
                                }
                            }
                        }

                        let array = BooleanArray::new(values.into(), Some(validity.into()));
                        Some(Arc::new(array) as ArrayRef)
                    } else {
                        None
                    }
                }
                crate::column::ColumnType::String => {
                    if let Some(str_col) = col_view.as_string() {
                        let mut values = Vec::with_capacity(df.row_count());
                        let mut validity = Vec::with_capacity(df.row_count());

                        for i in 0..df.row_count() {
                            match str_col.get(i) {
                                Ok(Some(val)) => {
                                    values.push(val.to_string());
                                    validity.push(true);
                                }
                                Ok(None) => {
                                    values.push(String::new());
                                    validity.push(false);
                                }
                                Err(_) => {
                                    values.push(String::new());
                                    validity.push(false);
                                }
                            }
                        }

                        let string_values: Vec<Option<&str>> = values
                            .iter()
                            .zip(validity.iter())
                            .map(|(s, &is_valid)| if is_valid { Some(s.as_str()) } else { None })
                            .collect();
                        let array = StringArray::from(string_values);
                        Some(Arc::new(array) as ArrayRef)
                    } else {
                        None
                    }
                }
            }
        })
        .collect();

    // Create record batch
    let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
        .map_err(|e| Error::Cast(format!("Failed to create record batch: {}", e)))?;

    // Create file and writer
    let file = File::create(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to create Parquet file: {}", e)))?;

    let mut writer = ArrowWriter::try_new(file, schema_ref, Some(props))
        .map_err(|e| Error::IoError(format!("Failed to create Parquet writer: {}", e)))?;

    // Write the record batch
    writer
        .write(&batch)
        .map_err(|e| Error::IoError(format!("Failed to write record batch: {}", e)))?;

    // Close the file
    writer
        .close()
        .map_err(|e| Error::IoError(format!("Failed to close Parquet file: {}", e)))?;

    Ok(())
}

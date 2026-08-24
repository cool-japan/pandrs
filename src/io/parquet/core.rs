//! Core Parquet types, compression validation, and the basic read/write entry points.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use super::convert::record_batches_to_dataframe;

/// Column statistics information
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: String,
    /// Null count
    pub null_count: Option<i64>,
    /// Distinct count (if available)
    pub distinct_count: Option<i64>,
    /// Minimum value (as string)
    pub min_value: Option<String>,
    /// Maximum value (as string)
    pub max_value: Option<String>,
}
/// Enumeration of Parquet compression options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetCompression {
    None,
    Snappy,
    Gzip,
    Lzo,
    Brotli,
    Lz4,
    Zstd,
}
/// Parquet file metadata information
#[derive(Debug, Clone)]
pub struct ParquetMetadata {
    /// Number of rows in the file
    pub num_rows: i64,
    /// Number of row groups
    pub num_row_groups: usize,
    /// File schema
    pub schema: String,
    /// File size in bytes
    pub file_size: Option<i64>,
    /// Compression algorithm used
    pub compression: String,
    /// Creator/writer information
    pub created_by: Option<String>,
}
/// Advanced Parquet reading options
#[derive(Debug, Clone)]
pub struct ParquetReadOptions {
    /// Specific columns to read (None = all columns)
    pub columns: Option<Vec<String>>,
    /// Use multiple threads for reading
    pub use_threads: bool,
    /// Memory map the file for faster access
    pub use_memory_map: bool,
    /// Batch size for chunked reading
    pub batch_size: Option<usize>,
    /// Row groups to read (None = all row groups)
    pub row_groups: Option<Vec<usize>>,
}
impl Default for ParquetReadOptions {
    fn default() -> Self {
        Self {
            columns: None,
            use_threads: true,
            use_memory_map: false,
            batch_size: None,
            row_groups: None,
        }
    }
}
/// Advanced Parquet writing options
#[derive(Debug, Clone)]
pub struct ParquetWriteOptions {
    /// Compression algorithm
    pub compression: ParquetCompression,
    /// Row group size (number of rows per group)
    pub row_group_size: Option<usize>,
    /// Page size in bytes
    pub page_size: Option<usize>,
    /// Enable dictionary encoding
    pub enable_dictionary: bool,
    /// Use multiple threads for writing
    pub use_threads: bool,
}
impl Default for ParquetWriteOptions {
    fn default() -> Self {
        Self {
            compression: ParquetCompression::Snappy,
            row_group_size: Some(50000),
            page_size: Some(1024 * 1024),
            enable_dictionary: true,
            use_threads: true,
        }
    }
}
/// Row group metadata information
#[derive(Debug, Clone)]
pub struct RowGroupInfo {
    /// Row group index
    pub index: usize,
    /// Number of rows in this row group
    pub num_rows: i64,
    /// Total byte size of this row group
    pub total_byte_size: i64,
    /// Number of columns
    pub num_columns: usize,
}
/// Reject compression codecs that this build cannot actually use, so callers
/// get an explicit, actionable error instead of a raw codec failure surfacing
/// much later from inside the `parquet` crate on the first `writer.write()`.
///
/// * `Zstd` — this workspace builds the `parquet` crate with
///   `default-features = false` and no `zstd` feature (COOLJAPAN Pure Rust
///   policy: upstream's `zstd` feature pulls in the C-backed `zstd-safe` /
///   `zstd-sys` crates). `parquet::compression::create_codec` then returns
///   `"Disabled feature at compile time: zstd"` the moment a page is encoded.
/// * `Lzo` — `parquet-rs` has never implemented an LZO codec at all, under
///   any feature configuration (`create_codec`'s match falls through to its
///   `The codec type LZO is not supported yet` arm unconditionally), so no
///   Cargo feature choice can make this work.
pub(super) fn validate_compression(compression: ParquetCompression) -> Result<()> {
    match compression {
        ParquetCompression::Zstd => Err(Error::NotImplemented(
            "ParquetCompression::Zstd is unavailable: this build uses the Pure Rust `parquet` \
             crate with the C-backed zstd codec feature disabled (COOLJAPAN Pure Rust policy). \
             Use ParquetCompression::Snappy, Gzip, Brotli, or Lz4 instead."
                .to_string(),
        )),
        ParquetCompression::Lzo => Err(Error::NotImplemented(
            "ParquetCompression::Lzo is unavailable: the `parquet` crate does not implement an \
             LZO codec (this is a gap in the upstream implementation, not a disabled build \
             feature). Use ParquetCompression::Snappy, Gzip, Brotli, or Lz4 instead."
                .to_string(),
        )),
        ParquetCompression::None
        | ParquetCompression::Snappy
        | ParquetCompression::Gzip
        | ParquetCompression::Brotli
        | ParquetCompression::Lz4 => Ok(()),
    }
}

/// Read a DataFrame from a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Returns
///
/// * `Result<DataFrame>` - The read DataFrame, or an error
///
/// # Example
///
/// ```no_run
/// use pandrs::io::read_parquet;
///
/// // Read a DataFrame from a Parquet file
/// let df = read_parquet("data.parquet").expect("operation should succeed");
/// ```
pub fn read_parquet(path: impl AsRef<Path>) -> Result<DataFrame> {
    // Open the file
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    // Create an Arrow ParquetReader
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::IoError(format!("Failed to parse Parquet file: {}", e)))?;

    // Get schema information
    let schema = builder.schema().clone();

    // Create a record batch reader
    let reader = builder
        .build()
        .map_err(|e| Error::IoError(format!("Failed to read Parquet file: {}", e)))?;

    // Read all record batches
    let mut all_batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::IoError(format!("Failed to read record batch: {}", e)))?;
        all_batches.push(batch);
    }

    // Return an empty DataFrame if no record batches are found
    if all_batches.is_empty() {
        return Ok(DataFrame::new());
    }

    // Convert to DataFrame
    record_batches_to_dataframe(&all_batches, schema)
}

/// Write a DataFrame to a Parquet file
///
/// # Arguments
///
/// * `df` - The DataFrame to write
/// * `path` - Path to the output Parquet file
/// * `compression` - Compression option (default is Snappy)
///
/// # Returns
///
/// * `Result<()>` - Ok(()) if successful, or an error
///
/// # Example
///
/// ```text
/// // Disable DOC test
/// ```
pub fn write_parquet(
    df: &OptimizedDataFrame,
    path: impl AsRef<Path>,
    compression: Option<ParquetCompression>,
) -> Result<()> {
    // Create Arrow schema
    let schema_fields: Vec<Field> = df
        .column_names()
        .iter()
        .filter_map(|col_name| {
            // Get each column as a string series
            if let Ok(col_view) = df.column(col_name) {
                // Determine column type
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

    // Convert column data to Arrow arrays
    let arrays: Vec<ArrayRef> = df
        .column_names()
        .iter()
        .filter_map(|col_name| {
            // Get each column
            let col_view = match df.column(col_name) {
                Ok(s) => s,
                Err(_) => return None,
            };

            // Extract actual data from the column based on its type
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
                                    values.push(0); // Default value for null
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
                                    values.push(0.0); // Default value for null
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
                                    values.push(false); // Default value for null
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
                                    values.push(String::new()); // Default value for null
                                    validity.push(false);
                                }
                                Err(_) => {
                                    values.push(String::new());
                                    validity.push(false);
                                }
                            }
                        }

                        // Convert to iterator with nulls properly handled
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

    // Create a record batch
    let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
        .map_err(|e| Error::Cast(format!("Failed to create record batch: {}", e)))?;

    // Set compression options
    let compression_type = compression.unwrap_or(ParquetCompression::Snappy);
    validate_compression(compression_type)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::from(compression_type))
        .build();

    // Create the file
    let file = File::create(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to create Parquet file: {}", e)))?;

    // Create an Arrow writer and write
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

impl From<ParquetCompression> for Compression {
    fn from(comp: ParquetCompression) -> Self {
        match comp {
            ParquetCompression::None => Compression::UNCOMPRESSED,
            ParquetCompression::Snappy => Compression::SNAPPY,
            ParquetCompression::Gzip => Compression::GZIP(Default::default()),
            ParquetCompression::Lzo => Compression::LZO,
            ParquetCompression::Brotli => Compression::BROTLI(Default::default()),
            ParquetCompression::Lz4 => Compression::LZ4,
            ParquetCompression::Zstd => Compression::ZSTD(Default::default()),
        }
    }
}

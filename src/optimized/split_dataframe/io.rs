//! Input/output functionality for OptimizedDataFrame

use std::fs::File;
use std::path::Path;

use super::core::OptimizedDataFrame;
use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};

#[cfg(feature = "parquet")]
use std::sync::Arc;

use csv::{ReaderBuilder, Writer};

// Excel I/O is delegated to the crate-internal Pure Rust xlsx module.

#[cfg(feature = "parquet")]
use arrow::array::{Array, ArrayRef, BooleanArray, LargeStringArray, PrimitiveArray, StringArray};
#[cfg(feature = "parquet")]
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema, TimeUnit};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_writer::ArrowWriter;
#[cfg(feature = "parquet")]
use parquet::basic::Compression;
#[cfg(feature = "parquet")]
use parquet::file::properties::WriterProperties;

/// Parquet compression options.
///
/// Re-exported from [`crate::io::parquet::ParquetCompression`] so that
/// `OptimizedDataFrame`/`SplitDataFrame`'s `to_parquet` and
/// `crate::io::parquet`'s free functions share a single canonical type
/// instead of two independently-defined, identically-named-but-unrelated
/// enums (they used to be genuine duplicates: a
/// `to_parquet(path, Some(pandrs::io::ParquetCompression::Snappy))` call did
/// not type-check, because the two `ParquetCompression` types were distinct
/// Rust types that merely shared a name).
///
/// `Zstd` and `Lzo` remain selectable variants for API compatibility, but
/// requesting either at `to_parquet` time now returns an explicit,
/// actionable error (see `to_parquet` below) instead of letting the
/// `parquet` crate fail deep inside `ArrowWriter::write` with an opaque
/// "disabled feature at compile time" / "codec not supported" message.
#[cfg(feature = "parquet")]
pub use crate::io::parquet::ParquetCompression;

/// Read one Arrow primitive column across every record batch, returning the
/// native values alongside a parallel NULL mask. NULL slots get a
/// placeholder native value (`T::default_value()`); the mask is what callers
/// must consult, exactly as `crate::column::*Column::with_nulls` expects.
#[cfg(feature = "parquet")]
fn read_primitive_column<T: ArrowPrimitiveType>(
    all_batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
) -> Result<(Vec<T::Native>, Vec<bool>)> {
    let mut values = Vec::new();
    let mut nulls = Vec::new();
    for batch in all_batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                Error::Cast(format!(
                    "Could not convert column '{}' to the expected Arrow primitive array type",
                    col_name
                ))
            })?;
        for i in 0..array.len() {
            if array.is_null(i) {
                values.push(T::default_value());
                nulls.push(true);
            } else {
                values.push(array.value(i));
                nulls.push(false);
            }
        }
    }
    Ok((values, nulls))
}

/// Convert an Arrow decimal column (`Decimal32`/`64`/`128`/`256`) to `f64` by
/// interpreting each raw integer as `unscaled / 10^scale`. This is lossy for
/// values whose precision exceeds `f64`'s ~15-17 significant decimal digits,
/// which is unavoidable given `Column` has no exact decimal type; it is
/// still far more faithful than the historical debug-dump fallback.
#[cfg(feature = "parquet")]
fn read_decimal_column<T>(
    all_batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
    scale: i8,
) -> Result<(Vec<f64>, Vec<bool>)>
where
    T: ArrowPrimitiveType,
    T::Native: std::fmt::Display,
{
    let (raw, nulls) = read_primitive_column::<T>(all_batches, col_idx, col_name)?;
    let mut values = Vec::with_capacity(raw.len());
    for (v, is_null) in raw.into_iter().zip(nulls.iter()) {
        if *is_null {
            values.push(0.0);
        } else {
            values.push(decimal_to_f64(v, scale)?);
        }
    }
    Ok((values, nulls))
}

/// Interpret a decimal's raw unscaled integer (`i32`/`i64`/`i128`/`i256`,
/// anything `Display`-able) and scale as `unscaled / 10^scale`.
#[cfg(feature = "parquet")]
fn decimal_to_f64(unscaled: impl std::fmt::Display, scale: i8) -> Result<f64> {
    let unscaled_f64: f64 = unscaled
        .to_string()
        .parse()
        .map_err(|e| Error::Cast(format!("Failed to parse decimal value: {}", e)))?;
    Ok(unscaled_f64 / 10f64.powi(scale as i32))
}

/// Format an Arrow `Date32` value (days since the Unix epoch) as `YYYY-MM-DD`.
#[cfg(feature = "parquet")]
fn days_to_iso_date(days: i32) -> Result<String> {
    let millis = (days as i64).saturating_mul(86_400_000);
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(millis)
        .ok_or_else(|| Error::Cast(format!("Date32 value {} (days) is out of range", days)))?;
    Ok(dt.format("%Y-%m-%d").to_string())
}

/// Format an Arrow `Date64` value (milliseconds since the Unix epoch, not
/// necessarily day-aligned per the Arrow spec) as an ISO-8601 datetime.
#[cfg(feature = "parquet")]
fn millis_to_iso_datetime(millis: i64) -> Result<String> {
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp_millis(millis)
        .ok_or_else(|| Error::Cast(format!("Date64 value {} (ms) is out of range", millis)))?;
    Ok(dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string())
}

/// Format an Arrow `Timestamp(unit, tz)` raw value as an ISO-8601 string.
///
/// Arrow's physical storage for a `Timestamp` is always a plain integer
/// offset from the Unix epoch: when a timezone is attached, that integer is
/// UTC by spec (the zone is display metadata only), so a timezone-aware
/// timestamp is rendered with a trailing `Z` to produce a complete,
/// unambiguous UTC instant. A timezone-naive timestamp has no stated zone at
/// all and is rendered as a bare "wall clock" ISO-8601 string, matching
/// Arrow's own naive semantics.
#[cfg(feature = "parquet")]
fn timestamp_unit_to_iso(value: i64, unit: &TimeUnit, has_tz: bool) -> Result<String> {
    let (secs, nsecs): (i64, u32) = match unit {
        TimeUnit::Second => (value, 0),
        TimeUnit::Millisecond => (
            value.div_euclid(1_000),
            (value.rem_euclid(1_000) as u32) * 1_000_000,
        ),
        TimeUnit::Microsecond => (
            value.div_euclid(1_000_000),
            (value.rem_euclid(1_000_000) as u32) * 1_000,
        ),
        TimeUnit::Nanosecond => (
            value.div_euclid(1_000_000_000),
            value.rem_euclid(1_000_000_000) as u32,
        ),
    };
    let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, nsecs)
        .ok_or_else(|| Error::Cast(format!("Timestamp value {} is out of range", value)))?;
    let formatted = dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string();
    if has_tz {
        Ok(format!("{}Z", formatted))
    } else {
        Ok(formatted)
    }
}

impl OptimizedDataFrame {
    /// Read DataFrame from a CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `has_header` - Whether the file has a header row
    ///
    /// # Returns
    /// * `Result<Self>` - The loaded DataFrame
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::Io(e))?;

        // Configure CSV reader
        let mut rdr = ReaderBuilder::new()
            .has_headers(has_header)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_reader(file);

        let mut df = Self::new();

        // `csv::Reader::records()` is a thin iterator over the reader's own
        // byte position, not a self-contained cursor: peeking one record via
        // a throwaway `.records()` call and later starting a *second*
        // `.records()` iterator resumes from wherever the reader physically
        // is, silently skipping whatever the peek already consumed. For
        // headerless files we therefore stash the peeked first record here
        // and feed it back in as the first data row below, instead of
        // letting it vanish (the historical bug this fixes: headerless CSVs
        // silently dropped row 0).
        let mut pending_first_row: Option<csv::StringRecord> = None;

        // Get header row
        let headers: Vec<String> = if has_header {
            rdr.headers()
                .map_err(|e| Error::Csv(e))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // Generate column names from the first row, remembering that row
            // so it is not lost as a data row.
            match rdr.records().next() {
                Some(first_record_result) => {
                    let first_record = first_record_result.map_err(|e| Error::Csv(e))?;
                    let names = (0..first_record.len())
                        .map(|i| format!("column_{}", i))
                        .collect();
                    pending_first_row = Some(first_record);
                    names
                }
                None => {
                    // If file is empty
                    return Ok(Self::new());
                }
            }
        };

        // Buffer for collecting column data
        let mut str_buffers: Vec<Vec<String>> = headers.iter().map(|_| Vec::new()).collect();

        // Push one record's fields into `str_buffers`, padding any short
        // trailing row out to the current max length with "" (NULL).
        fn push_record(record: &csv::StringRecord, str_buffers: &mut [Vec<String>]) {
            for (i, field) in record.iter().enumerate() {
                if i < str_buffers.len() {
                    str_buffers[i].push(field.to_string());
                }
            }
            let max_len = str_buffers.first().map_or(0, |b| b.len());
            for buffer in str_buffers.iter_mut() {
                if buffer.len() < max_len {
                    buffer.push(String::new());
                }
            }
        }

        // Re-inject the headerless file's peeked first row as the first data row.
        if let Some(first_record) = pending_first_row.take() {
            push_record(&first_record, &mut str_buffers);
        }

        // Read all (remaining) rows
        for result in rdr.records() {
            let record = result.map_err(|e| Error::Csv(e))?;
            push_record(&record, &mut str_buffers);
        }

        // Infer types from string data and add columns
        for (i, header) in headers.into_iter().enumerate() {
            if i < str_buffers.len() {
                // Perform type inference
                let values = &str_buffers[i];

                // Check for non-empty values
                let non_empty_values: Vec<&String> =
                    values.iter().filter(|s| !s.is_empty()).collect();

                if non_empty_values.is_empty() {
                    // Every value in this column is blank: the whole column
                    // is missing, not a column of literal empty strings.
                    let nulls = vec![true; values.len()];
                    df.add_column(
                        header,
                        Column::String(StringColumn::with_nulls(values.clone(), nulls)),
                    )?;
                    continue;
                }

                // Try to parse as integers. An empty cell among otherwise
                // all-integer values is NULL, not the integer 0.
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let mut int_values = Vec::with_capacity(values.len());
                    let mut nulls = Vec::with_capacity(values.len());
                    for s in values {
                        if s.is_empty() {
                            int_values.push(0);
                            nulls.push(true);
                        } else {
                            let parsed = s.parse::<i64>().map_err(|e| {
                                Error::Cast(format!(
                                    "Failed to parse '{}' in column '{}' as i64: {}",
                                    s, header, e
                                ))
                            })?;
                            int_values.push(parsed);
                            nulls.push(false);
                        }
                    }
                    df.add_column(
                        header,
                        Column::Int64(Int64Column::with_nulls(int_values, nulls)),
                    )?;
                    continue;
                }

                // Try to parse as floating point numbers. Same NULL handling as above.
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let mut float_values = Vec::with_capacity(values.len());
                    let mut nulls = Vec::with_capacity(values.len());
                    for s in values {
                        if s.is_empty() {
                            float_values.push(0.0);
                            nulls.push(true);
                        } else {
                            let parsed = s.parse::<f64>().map_err(|e| {
                                Error::Cast(format!(
                                    "Failed to parse '{}' in column '{}' as f64: {}",
                                    s, header, e
                                ))
                            })?;
                            float_values.push(parsed);
                            nulls.push(false);
                        }
                    }
                    df.add_column(
                        header,
                        Column::Float64(Float64Column::with_nulls(float_values, nulls)),
                    )?;
                    continue;
                }

                // Try to parse as boolean values
                let all_bools = non_empty_values.iter().all(|&s| {
                    let lower = s.to_lowercase();
                    lower == "true"
                        || lower == "false"
                        || lower == "1"
                        || lower == "0"
                        || lower == "yes"
                        || lower == "no"
                        || lower == "t"
                        || lower == "f"
                });

                if all_bools {
                    let mut bool_values = Vec::with_capacity(values.len());
                    let mut nulls = Vec::with_capacity(values.len());
                    for s in values {
                        if s.is_empty() {
                            bool_values.push(false);
                            nulls.push(true);
                        } else {
                            let lower = s.to_lowercase();
                            bool_values.push(
                                lower == "true" || lower == "1" || lower == "yes" || lower == "t",
                            );
                            nulls.push(false);
                        }
                    }
                    df.add_column(
                        header,
                        Column::Boolean(BooleanColumn::with_nulls(bool_values, nulls)),
                    )?;
                } else {
                    // Default to string type. An empty cell here is NULL
                    // under the same "blank == missing" convention applied
                    // above and used throughout this crate's CSV/Parquet
                    // round-trip -- there is no other way to spell "missing"
                    // in a plain CSV string field.
                    let nulls: Vec<bool> = values.iter().map(|s| s.is_empty()).collect();
                    df.add_column(
                        header,
                        Column::String(StringColumn::with_nulls(values.clone(), nulls)),
                    )?;
                }
            }
        }

        Ok(df)
    }

    /// Write DataFrame to a CSV file
    ///
    /// # Arguments
    /// * `path` - Path to the output CSV file
    /// * `has_header` - Whether to write a header row
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    pub fn to_csv<P: AsRef<Path>>(&self, path: P, has_header: bool) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| Error::Io(e))?;
        let mut wtr = Writer::from_writer(file);

        // Write header row
        if has_header {
            wtr.write_record(&self.column_names)
                .map_err(|e| Error::Csv(e))?;
        }

        // Exit if there are no rows
        if self.row_count == 0 {
            wtr.flush().map_err(|e| Error::Io(e))?;
            return Ok(());
        }

        // Write each row
        for i in 0..self.row_count {
            let mut row = Vec::new();

            for col_idx in 0..self.columns.len() {
                let value = match &self.columns[col_idx] {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    }
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    }
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    }
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            val.to_string()
                        } else {
                            String::new()
                        }
                    }
                };

                row.push(value);
            }

            wtr.write_record(&row).map_err(|e| Error::Csv(e))?;
        }

        wtr.flush().map_err(|e| Error::Io(e))?;
        Ok(())
    }

    /// Write DataFrame to a Parquet file
    ///
    /// # Arguments
    /// * `path` - Path to the output Parquet file
    /// * `compression` - Compression method (optional, Snappy is used if None)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    #[cfg(feature = "parquet")]
    pub fn to_parquet<P: AsRef<Path>>(
        &self,
        path: P,
        compression: Option<ParquetCompression>,
    ) -> Result<()> {
        // Write even if there are no rows, as an empty DataFrame

        // Resolve compression up front and reject codecs this build cannot
        // actually produce, with an explicit, actionable reason -- instead
        // of letting the `parquet` crate fail deep inside `ArrowWriter`.
        let compression_type = compression.unwrap_or(ParquetCompression::Snappy);
        match compression_type {
            ParquetCompression::Zstd => {
                return Err(Error::NotImplemented(
                    "Zstd Parquet compression is disabled under pandrs's Pure Rust policy: \
                     the `parquet` crate's ZSTD codec requires the C `zstd-sys` library, so \
                     this build intentionally compiles the `parquet` dependency without its \
                     'zstd' Cargo feature. Use ParquetCompression::Snappy, Gzip, Brotli, Lz4, \
                     or None instead."
                        .to_string(),
                ));
            }
            ParquetCompression::Lzo => {
                return Err(Error::NotImplemented(
                    "Lzo Parquet compression is not implemented by the underlying `parquet` \
                     crate for any backend (there is no LZO codec, pure-Rust or otherwise). \
                     Use ParquetCompression::Snappy, Gzip, Brotli, Lz4, or None instead."
                        .to_string(),
                ));
            }
            _ => {}
        }

        // Create Arrow schema
        let schema_fields: Vec<Field> = self
            .column_names
            .iter()
            .enumerate()
            .map(|(idx, col_name)| match &self.columns[idx] {
                Column::Int64(_) => Field::new(col_name, DataType::Int64, true),
                Column::Float64(_) => Field::new(col_name, DataType::Float64, true),
                Column::Boolean(_) => Field::new(col_name, DataType::Boolean, true),
                Column::String(_) => Field::new(col_name, DataType::Utf8, true),
            })
            .collect();

        let schema = Schema::new(schema_fields);
        let schema_ref = Arc::new(schema);

        // Convert column data to Arrow arrays, preserving NULLs via each
        // column's own null bitmask (`Column::get` already reports NULL as
        // `Ok(None)`) instead of writing every array as fully non-nullable
        // with a fabricated placeholder value sitting in the NULL slots.
        let arrays: Vec<ArrayRef> = self
            .column_names
            .iter()
            .enumerate()
            .map(|(idx, _)| -> Result<ArrayRef> {
                let array: ArrayRef = match &self.columns[idx] {
                    Column::Int64(col) => {
                        let values: Vec<Option<i64>> = (0..self.row_count)
                            .map(|i| col.get(i))
                            .collect::<Result<Vec<_>>>()?;
                        Arc::new(arrow::array::Int64Array::from(values))
                    }
                    Column::Float64(col) => {
                        let values: Vec<Option<f64>> = (0..self.row_count)
                            .map(|i| col.get(i))
                            .collect::<Result<Vec<_>>>()?;
                        Arc::new(arrow::array::Float64Array::from(values))
                    }
                    Column::Boolean(col) => {
                        let values: Vec<Option<bool>> = (0..self.row_count)
                            .map(|i| col.get(i))
                            .collect::<Result<Vec<_>>>()?;
                        Arc::new(BooleanArray::from(values))
                    }
                    Column::String(col) => {
                        let mut values: Vec<Option<String>> = Vec::with_capacity(self.row_count);
                        for i in 0..self.row_count {
                            values.push(col.get(i)?.map(|s| s.to_string()));
                        }
                        let string_array: StringArray = values.into_iter().collect();
                        Arc::new(string_array)
                    }
                };
                Ok(array)
            })
            .collect::<Result<Vec<_>>>()?;

        // Create record batch
        let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
            .map_err(|e| Error::Cast(format!("Failed to create record batch: {}", e)))?;

        // Set compression options
        let props = WriterProperties::builder()
            .set_compression(Compression::from(compression_type))
            .build();

        // Create file
        let file = File::create(path.as_ref()).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to create Parquet file: {}",
                e
            )))
        })?;

        // Create Arrow writer and write data
        let mut writer = ArrowWriter::try_new(file, schema_ref, Some(props)).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to create Parquet writer: {}",
                e
            )))
        })?;

        // Write record batch
        writer.write(&batch).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to write record batch: {}",
                e
            )))
        })?;

        // Close the file
        writer.close().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to close Parquet file: {}",
                e
            )))
        })?;

        Ok(())
    }

    /// Read DataFrame from a Parquet file
    ///
    /// # Arguments
    /// * `path` - Path to the Parquet file
    ///
    /// # Returns
    /// * `Result<Self>` - The loaded DataFrame
    #[cfg(feature = "parquet")]
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Open file
        let file = File::open(path.as_ref()).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to open Parquet file: {}",
                e
            )))
        })?;

        // Create Arrow's Parquet reader
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to parse Parquet file: {}",
                e
            )))
        })?;

        // Get schema information (clone it)
        let schema = builder.schema().clone();

        // Create record batch reader
        let reader = builder.build().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to read Parquet file: {}",
                e
            )))
        })?;

        // Read all record batches
        let mut all_batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result.map_err(|e| {
                Error::Io(crate::error::io_error(format!(
                    "Failed to read record batch: {}",
                    e
                )))
            })?;
            all_batches.push(batch);
        }

        // Return an empty DataFrame if there are no record batches
        if all_batches.is_empty() {
            return Ok(Self::new());
        }

        // Convert to DataFrame
        let mut df = Self::new();

        // Get column information from schema
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col_name = field.name().clone();
            let col_type = field.data_type();

            // Collect column data from all batches, preserving NULLs into
            // this crate's own null-bitmask columns rather than mapping
            // NULL -> 0 / NaN / false / "".
            match col_type {
                DataType::Int64 => {
                    let (values, nulls) = read_primitive_column::<arrow::datatypes::Int64Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Int32 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Int32Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Int16 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Int16Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Int8 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Int8Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::UInt8 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::UInt8Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::UInt16 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::UInt16Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::UInt32 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::UInt32Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<i64> = raw.into_iter().map(|v| v as i64).collect();
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::UInt64 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::UInt64Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let mut values = Vec::with_capacity(raw.len());
                    for (v, is_null) in raw.into_iter().zip(nulls.iter()) {
                        if *is_null {
                            values.push(0);
                        } else {
                            values.push(i64::try_from(v).map_err(|_| {
                                Error::Cast(format!(
                                    "Column '{}' contains a u64 value {} that does not fit in \
                                     i64; the OptimizedDataFrame bridge has no unsigned 64-bit \
                                     column type",
                                    col_name, v
                                ))
                            })?);
                        }
                    }
                    df.add_column(
                        col_name,
                        Column::Int64(Int64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Float64 => {
                    let (values, nulls) = read_primitive_column::<arrow::datatypes::Float64Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Float32 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Float32Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let values: Vec<f64> = raw.into_iter().map(|v| v as f64).collect();
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Boolean => {
                    let mut values = Vec::new();
                    let mut nulls = Vec::new();

                    for batch in &all_batches {
                        let array = batch
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .ok_or_else(|| {
                                Error::Cast(format!(
                                    "Could not convert column '{}' to BooleanArray",
                                    col_name
                                ))
                            })?;

                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(false);
                                nulls.push(true);
                            } else {
                                values.push(array.value(i));
                                nulls.push(false);
                            }
                        }
                    }

                    df.add_column(
                        col_name,
                        Column::Boolean(BooleanColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Utf8 => {
                    let mut values = Vec::new();
                    let mut nulls = Vec::new();

                    for batch in &all_batches {
                        let array = batch
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                Error::Cast(format!(
                                    "Could not convert column '{}' to StringArray",
                                    col_name
                                ))
                            })?;

                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(String::new());
                                nulls.push(true);
                            } else {
                                values.push(array.value(i).to_string());
                                nulls.push(false);
                            }
                        }
                    }

                    df.add_column(
                        col_name,
                        Column::String(StringColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::LargeUtf8 => {
                    // `LargeUtf8` backs a `LargeStringArray` (i64 offsets), a
                    // different concrete array type from `Utf8`'s
                    // `StringArray` (i32 offsets); downcasting to the wrong
                    // one always failed here previously.
                    let mut values = Vec::new();
                    let mut nulls = Vec::new();

                    for batch in &all_batches {
                        let array = batch
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .ok_or_else(|| {
                                Error::Cast(format!(
                                    "Could not convert column '{}' to LargeStringArray",
                                    col_name
                                ))
                            })?;

                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(String::new());
                                nulls.push(true);
                            } else {
                                values.push(array.value(i).to_string());
                                nulls.push(false);
                            }
                        }
                    }

                    df.add_column(
                        col_name,
                        Column::String(StringColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Date32 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Date32Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let mut values = Vec::with_capacity(raw.len());
                    for (v, is_null) in raw.into_iter().zip(nulls.iter()) {
                        if *is_null {
                            values.push(String::new());
                        } else {
                            values.push(days_to_iso_date(v)?);
                        }
                    }
                    df.add_column(
                        col_name,
                        Column::String(StringColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Date64 => {
                    let (raw, nulls) = read_primitive_column::<arrow::datatypes::Date64Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                    )?;
                    let mut values = Vec::with_capacity(raw.len());
                    for (v, is_null) in raw.into_iter().zip(nulls.iter()) {
                        if *is_null {
                            values.push(String::new());
                        } else {
                            values.push(millis_to_iso_datetime(v)?);
                        }
                    }
                    df.add_column(
                        col_name,
                        Column::String(StringColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Timestamp(unit, tz) => {
                    let has_tz = tz.is_some();
                    let (raw, nulls): (Vec<i64>, Vec<bool>) = match unit {
                        TimeUnit::Second => read_primitive_column::<
                            arrow::datatypes::TimestampSecondType,
                        >(
                            &all_batches, col_idx, &col_name
                        )?,
                        TimeUnit::Millisecond => read_primitive_column::<
                            arrow::datatypes::TimestampMillisecondType,
                        >(
                            &all_batches, col_idx, &col_name
                        )?,
                        TimeUnit::Microsecond => read_primitive_column::<
                            arrow::datatypes::TimestampMicrosecondType,
                        >(
                            &all_batches, col_idx, &col_name
                        )?,
                        TimeUnit::Nanosecond => read_primitive_column::<
                            arrow::datatypes::TimestampNanosecondType,
                        >(
                            &all_batches, col_idx, &col_name
                        )?,
                    };
                    let mut values = Vec::with_capacity(raw.len());
                    for (v, is_null) in raw.into_iter().zip(nulls.iter()) {
                        if *is_null {
                            values.push(String::new());
                        } else {
                            values.push(timestamp_unit_to_iso(v, unit, has_tz)?);
                        }
                    }
                    df.add_column(
                        col_name,
                        Column::String(StringColumn::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Decimal32(_, scale) => {
                    let (values, nulls) = read_decimal_column::<arrow::datatypes::Decimal32Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                        *scale,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Decimal64(_, scale) => {
                    let (values, nulls) = read_decimal_column::<arrow::datatypes::Decimal64Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                        *scale,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Decimal128(_, scale) => {
                    let (values, nulls) = read_decimal_column::<arrow::datatypes::Decimal128Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                        *scale,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                DataType::Decimal256(_, scale) => {
                    let (values, nulls) = read_decimal_column::<arrow::datatypes::Decimal256Type>(
                        &all_batches,
                        col_idx,
                        &col_name,
                        *scale,
                    )?;
                    df.add_column(
                        col_name,
                        Column::Float64(Float64Column::with_nulls(values, nulls)),
                    )?;
                }
                other => {
                    // Honest error instead of a fabricated per-row Debug dump:
                    // this Arrow type genuinely has no supported conversion.
                    return Err(Error::NotImplemented(format!(
                        "Reading Parquet column '{}' with Arrow type {:?} is not implemented",
                        col_name, other
                    )));
                }
            }
        }

        Ok(df)
    }

    /// Read DataFrame from an Excel file (.xlsx)
    ///
    /// # Arguments
    /// * `path` - Path to the Excel file
    /// * `sheet_name` - Name of the sheet to read (if None, reads the first sheet)
    /// * `header` - Whether the file has a header row
    /// * `skip_rows` - Number of rows to skip before starting to read
    /// * `use_cols` - List of column names or indices to read (if None, reads all columns)
    ///
    /// # Returns
    /// * `Result<Self>` - The loaded DataFrame
    #[cfg(feature = "excel")]
    pub fn from_excel<P: AsRef<Path>>(
        path: P,
        sheet_name: Option<&str>,
        header: bool,
        skip_rows: usize,
        use_cols: Option<&[&str]>,
    ) -> Result<Self> {
        crate::io::xlsx::read_split_dataframe(path, sheet_name, header, skip_rows, use_cols)
    }

    /// Write DataFrame to an Excel file (.xlsx)
    ///
    /// # Arguments
    /// * `path` - Path to the output Excel file
    /// * `sheet_name` - Sheet name (if None, "Sheet1" is used)
    /// * `index` - Whether to include index
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    #[cfg(feature = "excel")]
    pub fn to_excel<P: AsRef<Path>>(
        &self,
        path: P,
        sheet_name: Option<&str>,
        index: bool,
    ) -> Result<()> {
        crate::io::xlsx::write_split_dataframe(self, path, sheet_name, index)
    }
}

//! Input/output functionality for OptimizedDataFrame

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::core::OptimizedDataFrame;
use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::error::{Error, Result};

use crate::index::{DataFrameIndex, Index, IndexTrait};
#[cfg(feature = "sql")]
use rusqlite::{params, Connection};

use std::sync::Arc;

use csv::{ReaderBuilder, Writer};

#[cfg(feature = "sql")]
use rusqlite::{params, Connection};

#[cfg(feature = "excel")]
use calamine::{open_workbook, Reader, Xlsx};

#[cfg(feature = "excel")]
use simple_excel_writer::{Sheet, Workbook};

#[cfg(feature = "parquet")]
use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType, Field, Schema};
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

// The following are already imported in lines 8-10

/// Enumeration of Parquet compression options
#[cfg(feature = "parquet")]
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

#[cfg(feature = "parquet")]
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

        // Get header row
        let headers: Vec<String> = if has_header {
            rdr.headers()
                .map_err(|e| Error::Csv(e))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // Generate column names if no header
            if let Some(first_record_result) = rdr.records().next() {
                let first_record = first_record_result.map_err(|e| Error::Csv(e))?;
                (0..first_record.len())
                    .map(|i| format!("column_{}", i))
                    .collect()
            } else {
                // If file is empty
                return Ok(Self::new());
            }
        };

        // Buffer for collecting column data
        let mut str_buffers: Vec<Vec<String>> = headers.iter().map(|_| Vec::new()).collect();

        // Read all rows
        for result in rdr.records() {
            let record = result.map_err(|e| Error::Csv(e))?;
            for (i, field) in record.iter().enumerate() {
                if i < str_buffers.len() {
                    str_buffers[i].push(field.to_string());
                }
            }
            // Add NULL for missing fields
            let max_len = str_buffers.get(0).map_or(0, |b| b.len());
            for buffer in &mut str_buffers {
                if buffer.len() < max_len {
                    buffer.push(String::new());
                }
            }
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
                    // Use string type if all values are empty
                    df.add_column(
                        header,
                        Column::String(StringColumn::new(
                            values.iter().map(|s| s.clone()).collect(),
                        )),
                    )?;
                    continue;
                }

                // Try to parse as integers
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = values
                        .iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    df.add_column(header, Column::Int64(Int64Column::new(int_values)))?;
                    continue;
                }

                // Try to parse as floating point numbers
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = values
                        .iter()
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .collect();
                    df.add_column(header, Column::Float64(Float64Column::new(float_values)))?;
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
                    let bool_values: Vec<bool> = values
                        .iter()
                        .map(|s| {
                            let lower = s.to_lowercase();
                            lower == "true" || lower == "1" || lower == "yes" || lower == "t"
                        })
                        .collect();
                    df.add_column(header, Column::Boolean(BooleanColumn::new(bool_values)))?;
                } else {
                    // Default to string type
                    df.add_column(
                        header,
                        Column::String(StringColumn::new(
                            values.iter().map(|s| s.clone()).collect(),
                        )),
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

        // Convert column data to Arrow arrays
        let arrays: Vec<ArrayRef> = self
            .column_names
            .iter()
            .enumerate()
            .map(|(idx, _)| match &self.columns[idx] {
                Column::Int64(col) => {
                    let values: Vec<i64> = (0..self.row_count)
                        .map(|i| match col.get(i) {
                            Ok(Some(v)) => v,
                            _ => 0,
                        })
                        .collect();
                    Arc::new(Int64Array::from(values)) as ArrayRef
                }
                Column::Float64(col) => {
                    let values: Vec<f64> = (0..self.row_count)
                        .map(|i| match col.get(i) {
                            Ok(Some(v)) => v,
                            _ => f64::NAN,
                        })
                        .collect();
                    Arc::new(Float64Array::from(values)) as ArrayRef
                }
                Column::Boolean(col) => {
                    let values: Vec<bool> = (0..self.row_count)
                        .map(|i| match col.get(i) {
                            Ok(Some(v)) => v,
                            _ => false,
                        })
                        .collect();
                    Arc::new(BooleanArray::from(values)) as ArrayRef
                }
                Column::String(col) => {
                    let values: Vec<String> = (0..self.row_count)
                        .map(|i| {
                            if let Ok(Some(v)) = col.get(i) {
                                v.to_string()
                            } else {
                                String::new()
                            }
                        })
                        .collect();
                    Arc::new(StringArray::from(values)) as ArrayRef
                }
            })
            .collect();

        // Create record batch
        let batch = RecordBatch::try_new(schema_ref.clone(), arrays)
            .map_err(|e| Error::Cast(format!("Failed to create record batch: {}", e)))?;

        // Set compression options
        let compression_type = compression.unwrap_or(ParquetCompression::Snappy);
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

            // Collect column data from all batches
            match col_type {
                DataType::Int64 => {
                    let mut values = Vec::new();

                    for batch in &all_batches {
                        let array = batch
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                Error::Cast(format!(
                                    "Could not convert column '{}' to Int64Array",
                                    col_name
                                ))
                            })?;

                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(0); // Use 0 as default value for NULL
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }

                    df.add_column(col_name, Column::Int64(Int64Column::new(values)))?;
                }
                DataType::Float64 => {
                    let mut values = Vec::new();

                    for batch in &all_batches {
                        let array = batch
                            .column(col_idx)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                Error::Cast(format!(
                                    "Could not convert column '{}' to Float64Array",
                                    col_name
                                ))
                            })?;

                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push(f64::NAN); // Use NaN for NULL values
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }

                    df.add_column(col_name, Column::Float64(Float64Column::new(values)))?;
                }
                DataType::Boolean => {
                    let mut values = Vec::new();

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
                                values.push(false); // Use false as default value for NULL
                            } else {
                                values.push(array.value(i));
                            }
                        }
                    }

                    df.add_column(col_name, Column::Boolean(BooleanColumn::new(values)))?;
                }
                DataType::Utf8 | DataType::LargeUtf8 => {
                    let mut values = Vec::new();

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
                                values.push("".to_string()); // Use empty string for NULL values
                            } else {
                                values.push(array.value(i).to_string());
                            }
                        }
                    }

                    df.add_column(col_name, Column::String(StringColumn::new(values)))?;
                }
                _ => {
                    // Treat unsupported data types as strings
                    let mut values = Vec::new();

                    for batch in &all_batches {
                        let array = batch.column(col_idx);
                        for i in 0..array.len() {
                            if array.is_null(i) {
                                values.push("".to_string());
                            } else {
                                // Cannot access value method directly, so downcast to StringArray first
                                if let Some(str_array) =
                                    array.as_any().downcast_ref::<StringArray>()
                                {
                                    values.push(str_array.value(i).to_string());
                                } else {
                                    values.push(format!("{:?}", array));
                                }
                            }
                        }
                    }

                    df.add_column(col_name, Column::String(StringColumn::new(values)))?;
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
        // Open file
        let mut workbook: Xlsx<BufReader<File>> = open_workbook(path.as_ref()).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Could not open Excel file: {}",
                e
            )))
        })?;

        // Get sheet name (use the first sheet if not specified)
        let sheet_name = match sheet_name {
            Some(name) => name.to_string(),
            None => workbook
                .sheet_names()
                .get(0)
                .ok_or_else(|| Error::Io(crate::error::io_error("Excel file has no sheets")))?
                .clone(),
        };

        // Get the sheet
        let range = workbook.worksheet_range(&sheet_name).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Could not read sheet '{}': {}",
                sheet_name, e
            )))
        })?;

        // Get column names (headers)
        let mut column_names: Vec<String> = Vec::new();
        if header && !range.is_empty() && skip_rows < range.rows().len() {
            // Get header row
            let header_row = range.rows().nth(skip_rows).unwrap();

            // Convert column names to strings
            for cell in header_row {
                column_names.push(cell.to_string());
            }
        } else {
            // If no header, use column numbers as names
            if !range.is_empty() {
                let first_row = range.rows().next().unwrap();
                for i in 0..first_row.len() {
                    column_names.push(format!("Column{}", i + 1));
                }
            }
        }

        // Determine which columns to read
        let use_cols_indices = if let Some(cols) = use_cols {
            // Get indices of specified columns
            let mut indices = Vec::new();
            for col_name in cols {
                if let Some(pos) = column_names.iter().position(|name| name == col_name) {
                    indices.push(pos);
                }
            }
            Some(indices)
        } else {
            None
        };

        // Create DataFrame
        let mut df = Self::new();

        // Collect data for each column
        let mut column_data: HashMap<usize, Vec<String>> = HashMap::new();
        let start_row = if header { skip_rows + 1 } else { skip_rows };

        for (row_idx, row) in range.rows().enumerate().skip(start_row) {
            for (col_idx, cell) in row.iter().enumerate() {
                // Process only columns that should be used
                if let Some(ref indices) = use_cols_indices {
                    if !indices.contains(&col_idx) {
                        continue;
                    }
                }

                // Add to column data
                column_data
                    .entry(col_idx)
                    .or_insert_with(Vec::new)
                    .push(cell.to_string());
            }
        }

        // Add column data to DataFrame
        for col_idx in 0..column_names.len() {
            // Process only columns that should be used
            if let Some(ref indices) = use_cols_indices {
                if !indices.contains(&col_idx) {
                    continue;
                }
            }

            let col_name = column_names
                .get(col_idx)
                .unwrap_or(&format!("Column{}", col_idx + 1))
                .clone();

            // Get column data
            let data = column_data.get(&col_idx).cloned().unwrap_or_default();

            // Skip empty columns
            if data.is_empty() {
                continue;
            }

            // Infer data type and create appropriate column
            let non_empty_values: Vec<&String> = data.iter().filter(|s| !s.is_empty()).collect();

            if non_empty_values.is_empty() {
                // If all values are empty, use string type
                df.add_column(col_name, Column::String(StringColumn::new(data)))?;
                continue;
            }

            // Try to parse as integers
            let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
            if all_ints {
                let int_values: Vec<i64> =
                    data.iter().map(|s| s.parse::<i64>().unwrap_or(0)).collect();
                df.add_column(col_name, Column::Int64(Int64Column::new(int_values)))?;
                continue;
            }

            // Try to parse as floating point numbers
            let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = data
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                df.add_column(col_name, Column::Float64(Float64Column::new(float_values)))?;
                continue;
            }

            // Try to parse as boolean values
            let all_bools = non_empty_values.iter().all(|&s| {
                let s = s.trim().to_lowercase();
                s == "true"
                    || s == "false"
                    || s == "1"
                    || s == "0"
                    || s == "yes"
                    || s == "no"
                    || s == "t"
                    || s == "f"
            });

            if all_bools {
                let bool_values: Vec<bool> = data
                    .iter()
                    .map(|s| {
                        let s = s.trim().to_lowercase();
                        s == "true" || s == "1" || s == "yes" || s == "t"
                    })
                    .collect();
                df.add_column(col_name, Column::Boolean(BooleanColumn::new(bool_values)))?;
            } else {
                // Default to string type
                df.add_column(col_name, Column::String(StringColumn::new(data)))?;
            }
        }

        Ok(df)
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
        // Create a new Excel file
        let mut workbook = Workbook::create(path.as_ref().to_str().ok_or_else(|| {
            Error::Io(crate::error::io_error(
                "Could not convert file path to string",
            ))
        })?);

        let sheet_name = sheet_name.unwrap_or("Sheet1");

        // Create sheet
        let mut sheet = workbook.create_sheet(sheet_name);

        // Create header row
        let mut headers = Vec::new();

        // Include index if requested
        if index {
            headers.push("Index".to_string());
        }

        // Add column names
        for col_name in &self.column_names {
            headers.push(col_name.clone());
        }

        // Write data
        workbook.write_sheet(&mut sheet, |sheet_writer| {
            // Add header row
            if !headers.is_empty() {
                let header_row: Vec<&str> = headers.iter().map(|s| s.as_str()).collect();
                // Create Row directly
                let row = simple_excel_writer::Row::from_iter(header_row.iter().cloned());
                sheet_writer.append_row(row)?;
            }

            // Write data rows
            for row_idx in 0..self.row_count {
                let mut row_values = Vec::new();

                // Include index if requested
                if index {
                    row_values.push(row_idx.to_string());
                }

                // Add data from each column
                for col in &self.columns {
                    let value = match col {
                        Column::Int64(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        }
                        Column::Float64(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        }
                        Column::String(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        }
                        Column::Boolean(c) => {
                            if let Ok(Some(val)) = c.get(row_idx) {
                                val.to_string()
                            } else {
                                String::new()
                            }
                        }
                    };

                    row_values.push(value);
                }

                // Add row to Excel (convert to slice of string references)
                let row_str_refs: Vec<&str> = row_values.iter().map(|s| s.as_str()).collect();
                // Create Row directly
                let row = simple_excel_writer::Row::from_iter(row_str_refs.iter().cloned());
                sheet_writer.append_row(row)?;
            }

            Ok(())
        })?;

        // Close and save workbook
        workbook.close().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Could not save Excel file: {}",
                e
            )))
        })?;

        Ok(())
    }

    /// Create DataFrame from SQL query results
    ///
    /// # Arguments
    /// * `query` - SQL query to execute
    /// * `db_path` - Path to SQLite database file
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame containing query results
    #[cfg(feature = "sql")]
    pub fn from_sql<P: AsRef<Path>>(query: &str, db_path: P) -> Result<Self> {
        // Connect to database
        let conn = Connection::open(db_path).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Could not connect to database: {}",
                e
            )))
        })?;

        // Prepare query
        let mut stmt = conn.prepare(query).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to prepare SQL query: {}",
                e
            )))
        })?;

        // Get column names
        let column_names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|&name| name.to_string())
            .collect();

        // Map to store data for each column
        let mut column_data: HashMap<String, Vec<String>> = HashMap::new();
        for name in &column_names {
            column_data.insert(name.clone(), Vec::new());
        }

        // Execute query and get results
        let mut rows = stmt.query([]).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to execute SQL query: {}",
                e
            )))
        })?;

        // Process each row of data
        while let Some(row) = rows.next().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to get SQL query results: {}",
                e
            )))
        })? {
            for (idx, name) in column_names.iter().enumerate() {
                let value: Option<String> = row.get(idx).map_err(|e| {
                    Error::Io(crate::error::io_error(format!(
                        "Failed to get row data: {}",
                        e
                    )))
                })?;

                if let Some(data) = column_data.get_mut(name) {
                    data.push(value.unwrap_or_else(|| "NULL".to_string()));
                }
            }
        }

        // Create DataFrame
        let mut df = Self::new();

        // Create DataFrame from column data
        for name in column_names {
            if let Some(data) = column_data.get(&name) {
                // Check for non-empty values
                let non_empty_values: Vec<&String> = data
                    .iter()
                    .filter(|s| !s.is_empty() && *s != "NULL")
                    .collect();

                if non_empty_values.is_empty() {
                    // Use string type if all values are empty
                    df.add_column(
                        name,
                        Column::String(StringColumn::new(data.iter().map(|s| s.clone()).collect())),
                    )?;
                    continue;
                }

                // Try to parse as integers
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = data
                        .iter()
                        .map(|s| {
                            if s.is_empty() || s == "NULL" {
                                0
                            } else {
                                s.parse::<i64>().unwrap_or(0)
                            }
                        })
                        .collect();
                    df.add_column(name, Column::Int64(Int64Column::new(int_values)))?;
                    continue;
                }

                // Try to parse as floating point numbers
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = data
                        .iter()
                        .map(|s| {
                            if s.is_empty() || s == "NULL" {
                                0.0
                            } else {
                                s.parse::<f64>().unwrap_or(0.0)
                            }
                        })
                        .collect();
                    df.add_column(name, Column::Float64(Float64Column::new(float_values)))?;
                    continue;
                }

                // Try to parse as boolean values
                let all_bools = non_empty_values.iter().all(|&s| {
                    let s = s.trim().to_lowercase();
                    s == "true"
                        || s == "false"
                        || s == "1"
                        || s == "0"
                        || s == "yes"
                        || s == "no"
                        || s == "t"
                        || s == "f"
                });

                if all_bools {
                    let bool_values: Vec<bool> = data
                        .iter()
                        .map(|s| {
                            let s = s.trim().to_lowercase();
                            s == "true" || s == "1" || s == "yes" || s == "t"
                        })
                        .collect();
                    df.add_column(name, Column::Boolean(BooleanColumn::new(bool_values)))?;
                } else {
                    // Default to string type
                    df.add_column(
                        name,
                        Column::String(StringColumn::new(
                            data.iter()
                                .map(|s| {
                                    if s == "NULL" {
                                        String::new()
                                    } else {
                                        s.clone()
                                    }
                                })
                                .collect(),
                        )),
                    )?;
                }
            }
        }

        Ok(df)
    }

    /// Write DataFrame to SQLite table
    ///
    /// # Arguments
    /// * `table_name` - Table name
    /// * `db_path` - Path to SQLite database file
    /// * `if_exists` - Action to take if table exists ("fail", "replace", "append")
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    #[cfg(feature = "sql")]
    pub fn to_sql<P: AsRef<Path>>(
        &self,
        table_name: &str,
        db_path: P,
        if_exists: &str,
    ) -> Result<()> {
        // Connect to database
        let mut conn = Connection::open(db_path).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Could not connect to database: {}",
                e
            )))
        })?;

        // Check if table exists
        let table_exists = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
            .map_err(|e| {
                Error::Io(crate::error::io_error(format!(
                    "Failed to prepare table check query: {}",
                    e
                )))
            })?
            .exists(params![table_name])
            .map_err(|e| {
                Error::Io(crate::error::io_error(format!(
                    "Failed to check if table exists: {}",
                    e
                )))
            })?;

        // Handle table based on if_exists parameter
        if table_exists {
            match if_exists {
                "fail" => {
                    return Err(Error::Io(crate::error::io_error(format!(
                        "Table '{}' already exists",
                        table_name
                    ))));
                }
                "replace" => {
                    // Drop table and create new one
                    conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                        .map_err(|e| {
                            Error::Io(crate::error::io_error(format!(
                                "Failed to drop table: {}",
                                e
                            )))
                        })?;

                    // Create new table
                    self.create_table_from_df(&conn, table_name)?;
                }
                "append" => {
                    // Table already exists, just append data
                }
                _ => {
                    return Err(Error::Io(crate::error::io_error(format!(
                        "Unknown if_exists value: {}",
                        if_exists
                    ))));
                }
            }
        } else {
            // Create new table if it doesn't exist
            self.create_table_from_df(&conn, table_name)?;
        }

        // Data insertion
        // List of column names
        let columns = self.column_names.join(", ");

        // List of placeholders
        let placeholders: Vec<String> = (0..self.column_names.len())
            .map(|_| "?".to_string())
            .collect();
        let placeholders = placeholders.join(", ");

        // Prepare INSERT statement
        let insert_sql = format!(
            "INSERT INTO {} ({}) VALUES ({})",
            table_name, columns, placeholders
        );

        // Begin transaction
        let tx = conn.transaction().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to start transaction: {}",
                e
            )))
        })?;

        // Insert data for each row
        for row_idx in 0..self.row_count {
            // Get row data
            let mut row_values: Vec<String> = Vec::new();

            for col in &self.columns {
                let value = match col {
                    Column::Int64(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Float64(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::String(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Boolean(c) => {
                        if let Ok(Some(val)) = c.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                };

                row_values.push(value);
            }

            // Execute INSERT
            let mut stmt = tx.prepare(&insert_sql).map_err(|e| {
                Error::Io(crate::error::io_error(format!(
                    "Failed to prepare INSERT statement: {}",
                    e
                )))
            })?;

            let params: Vec<&dyn rusqlite::ToSql> = row_values
                .iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();

            stmt.execute(params.as_slice()).map_err(|e| {
                Error::Io(crate::error::io_error(format!(
                    "Failed to insert data: {}",
                    e
                )))
            })?;
        }

        // Commit transaction
        tx.commit().map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to commit transaction: {}",
                e
            )))
        })?;

        Ok(())
    }

    // Helper method to create SQLite table from DataFrame
    #[cfg(feature = "sql")]
    fn create_table_from_df(&self, conn: &Connection, table_name: &str) -> Result<()> {
        // Create list of column names and types
        let mut columns = Vec::new();

        for (idx, col_name) in self.column_names.iter().enumerate() {
            let sql_type = match &self.columns[idx] {
                Column::Int64(_) => "INTEGER",
                Column::Float64(_) => "REAL",
                Column::Boolean(_) => "INTEGER", // Boolean values are stored as integers in SQLite
                Column::String(_) => "TEXT",
            };

            columns.push(format!("{} {}", col_name, sql_type));
        }

        // Create and execute CREATE TABLE statement
        let create_sql = format!("CREATE TABLE {} ({})", table_name, columns.join(", "));
        conn.execute(&create_sql, []).map_err(|e| {
            Error::Io(crate::error::io_error(format!(
                "Failed to create table: {}",
                e
            )))
        })?;

        Ok(())
    }
}

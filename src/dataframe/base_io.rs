//! CSV/JSON construction and serialisation for [`DataFrame`].
//!
//! Split out of `base.rs` to keep that file under the project's 2000-line
//! guideline. Declared as a `#[path]` submodule of `base` (not a sibling of
//! `dataframe`), so it remains a child module of `base` and can see its
//! private fields, matching the pattern already used by
//! `crate::time_series::stats`/`stats_normality.rs`.

use super::DataFrame;
use crate::core::error::{Error, Result};

impl DataFrame {
    /// Write the DataFrame to a CSV file.
    ///
    /// All columns are serialised using their string representation (numeric,
    /// boolean and string columns are all supported). A header row of column
    /// names is always written.
    pub fn to_csv<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path.as_ref())
            .map_err(|e| Error::IoError(format!("Failed to create CSV file: {}", e)))?;
        let mut writer = csv::Writer::from_writer(file);

        // Header row.
        writer
            .write_record(&self.column_order)
            .map_err(|e| Error::IoError(format!("CSV write error: {}", e)))?;

        // Materialise every column once as strings, preserving column order.
        let mut string_columns: Vec<Vec<String>> = Vec::with_capacity(self.column_order.len());
        for col_name in &self.column_order {
            string_columns.push(self.get_column_string_values(col_name)?);
        }

        // Write data rows.
        for row_idx in 0..self.row_count {
            let mut record: Vec<String> = Vec::with_capacity(string_columns.len());
            for column in &string_columns {
                record.push(column.get(row_idx).cloned().unwrap_or_default());
            }
            writer
                .write_record(&record)
                .map_err(|e| Error::IoError(format!("CSV write error: {}", e)))?;
        }

        writer
            .flush()
            .map_err(|e| Error::IoError(format!("CSV flush error: {}", e)))?;
        Ok(())
    }

    /// Read a DataFrame from a CSV file.
    ///
    /// Delegates to the real CSV reader in [`crate::io::csv`]. Every column is
    /// loaded as a string column (matching pandas' default of not inferring
    /// dtypes unless asked).
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P, has_header: bool) -> Result<Self> {
        crate::io::csv::read_csv(path, has_header)
    }

    /// Create DataFrame from CSV reader
    pub fn from_csv_reader<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_header: bool,
    ) -> Result<Self> {
        let mut df = Self::new();

        // In headerless mode we must peek at the first record to learn the
        // column count before we can name the synthetic `column_N` headers.
        // `reader.records()` returns an iterator tied to the reader's
        // position, so once that peeked record is consumed it is gone from
        // the stream -- a second `reader.records()` call resumes from
        // record 1, silently dropping record 0. `pending_first_record`
        // keeps hold of it so it is fed into the data below instead of
        // being lost.
        let mut pending_first_record: Option<csv::StringRecord> = None;

        let headers: Vec<String> = if has_header {
            reader
                .headers()
                .map_err(|e| Error::IoError(format!("CSV header error: {}", e)))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // Peek at first record to determine column count.
            let mut records = reader.records();
            if let Some(first_record) = records.next() {
                let record =
                    first_record.map_err(|e| Error::IoError(format!("CSV read error: {}", e)))?;
                let headers = (0..record.len()).map(|i| format!("column_{}", i)).collect();
                pending_first_record = Some(record);
                headers
            } else {
                return Ok(df); // Empty file
            }
        };

        // Collect data for each column
        let mut columns_data: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for header in &headers {
            columns_data.insert(header.clone(), Vec::new());
        }

        // Pushes one CSV record's fields into `columns_data`, keyed by
        // `headers`. Written as a plain (non-capturing) nested function so
        // it borrows neither `headers` nor `columns_data` for longer than
        // each call, leaving both free to be used/moved afterwards.
        fn push_record(
            headers: &[String],
            record: &csv::StringRecord,
            columns_data: &mut std::collections::HashMap<String, Vec<String>>,
        ) -> Result<()> {
            for (i, header) in headers.iter().enumerate() {
                let value = if i < record.len() {
                    record[i].to_string()
                } else {
                    String::new()
                };
                // Safe: header was inserted into columns_data during initialization
                if let Some(col_vec) = columns_data.get_mut(header) {
                    col_vec.push(value);
                } else {
                    return Err(Error::InvalidValue(format!(
                        "Column '{}' not found in columns_data",
                        header
                    )));
                }
            }
            Ok(())
        }

        // Re-include the first record consumed above while peeking for the
        // column count (headerless mode only); otherwise it would silently
        // be dropped and a 3-row file would read as 2 rows.
        if let Some(record) = pending_first_record.take() {
            push_record(&headers, &record, &mut columns_data)?;
        }

        // Process the remaining records
        for result in reader.records() {
            let record = result.map_err(|e| Error::IoError(format!("CSV read error: {}", e)))?;
            push_record(&headers, &record, &mut columns_data)?;
        }

        // Add columns to DataFrame
        for header in headers {
            if let Some(values) = columns_data.remove(&header) {
                let series = crate::series::Series::new(values, Some(header.clone()))?;
                df.add_column(header, series)?;
            }
        }

        Ok(df)
    }

    /// Create a new DataFrame from JSON string
    /// Expects JSON format like: {"col1": ["val1", "val2"], "col2": ["val3", "val4"]}
    pub fn from_json(json_str: &str) -> Result<Self> {
        use serde_json::Value;

        // Parse the JSON string
        let parsed: Value = serde_json::from_str(json_str)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse JSON: {}", e)))?;

        // Convert JSON object to HashMap
        let mut data: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        if let Value::Object(obj) = parsed {
            for (col_name, col_values) in obj {
                if let Value::Array(values) = col_values {
                    let string_values: Vec<String> = values
                        .into_iter()
                        .map(|v| match v {
                            Value::String(s) => s,
                            Value::Number(n) => n.to_string(),
                            Value::Bool(b) => ToString::to_string(&b),
                            Value::Null => "".to_string(),
                            _ => v.to_string(),
                        })
                        .collect();
                    data.insert(col_name, string_values);
                } else {
                    return Err(Error::InvalidInput(format!(
                        "Column '{}' is not an array",
                        col_name
                    )));
                }
            }
        } else {
            return Err(Error::InvalidInput("JSON must be an object".to_string()));
        }

        // Use existing from_map method
        Self::from_map(data, None)
    }
}

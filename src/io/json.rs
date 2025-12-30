use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{PandRSError, Result};
use crate::series::Series;
use crate::DataFrame;

/// Read a DataFrame from a JSON file
pub fn read_json<P: AsRef<Path>>(path: P) -> Result<DataFrame> {
    let file = File::open(path.as_ref()).map_err(PandRSError::Io)?;
    let reader = BufReader::new(file);

    // Parse JSON
    let json_value: Value = serde_json::from_reader(reader).map_err(PandRSError::Json)?;

    match json_value {
        Value::Array(array) => read_records_array(array),
        Value::Object(map) => read_column_oriented(map),
        _ => Err(PandRSError::Format(
            "JSON must be an object or an array".to_string(),
        )),
    }
}

// Read record-oriented JSON
fn read_records_array(array: Vec<Value>) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Return an empty DataFrame if the array is empty
    if array.is_empty() {
        return Ok(df);
    }

    // Collect all keys
    let mut all_keys = std::collections::HashSet::new();
    for item in &array {
        if let Value::Object(map) = item {
            for key in map.keys() {
                all_keys.insert(key.clone());
            }
        } else {
            return Err(PandRSError::Format(
                "Each element of the array must be an object".to_string(),
            ));
        }
    }

    // Collect column data
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();
    for key in &all_keys {
        let mut values = Vec::with_capacity(array.len());

        for item in &array {
            if let Value::Object(map) = item {
                if let Some(value) = map.get(key) {
                    values.push(value.to_string());
                } else {
                    // If the key is missing, add an empty string
                    values.push(String::new());
                }
            }
        }

        columns.insert(key.clone(), values);
    }

    // Add columns to the DataFrame
    for (key, values) in columns {
        let series = Series::new(values, Some(key.clone()))?;
        df.add_column(key, series)?;
    }

    Ok(df)
}

// Read column-oriented JSON
fn read_column_oriented(map: Map<String, Value>) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Process each column
    for (key, value) in map {
        if let Value::Array(array) = value {
            let values: Vec<String> = array.iter().map(|v| v.to_string()).collect();

            let series = Series::new(values, Some(key.clone()))?;
            df.add_column(key, series)?;
        } else {
            return Err(PandRSError::Format(format!(
                "Column '{}' must be an array",
                key
            )));
        }
    }

    Ok(df)
}

/// Write a DataFrame to a JSON file
pub fn write_json<P: AsRef<Path>>(df: &DataFrame, path: P, orient: JsonOrient) -> Result<()> {
    let file = File::create(path.as_ref()).map_err(PandRSError::Io)?;
    let writer = BufWriter::new(file);

    let json_value = match orient {
        JsonOrient::Records => to_records_json(df)?,
        JsonOrient::Columns => to_column_json(df)?,
    };

    serde_json::to_writer_pretty(writer, &json_value).map_err(PandRSError::Json)?;

    Ok(())
}

/// JSON output orientation
pub enum JsonOrient {
    /// Record-oriented [{col1:val1, col2:val2}, ...]
    Records,
    /// Column-oriented {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

// Convert DataFrame to record-oriented JSON
fn to_records_json(df: &DataFrame) -> Result<Value> {
    let mut records = Vec::new();

    // Process each row
    for row_idx in 0..df.row_count() {
        let mut record = serde_json::Map::new();

        // Add values from each column
        for col_name in df.column_names() {
            // This is a stub implementation since we don't have the full Series functionality yet
            // In a real implementation, we would get values from the series
            record.insert(col_name.clone(), Value::String(String::new()));
        }

        records.push(Value::Object(record));
    }

    Ok(Value::Array(records))
}

// Convert DataFrame to column-oriented JSON
fn to_column_json(df: &DataFrame) -> Result<Value> {
    let mut columns = serde_json::Map::new();

    // Process each column
    for col_name in df.column_names() {
        // This is a stub implementation since we don't have the full Series functionality yet
        // In a real implementation, we would get values from the series
        let values: Vec<Value> = (0..df.row_count())
            .map(|_| Value::String(String::new()))
            .collect();
        columns.insert(col_name.clone(), Value::Array(values));
    }

    Ok(Value::Object(columns))
}

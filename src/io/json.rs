use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
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

/// Render one JSON value as the text that belongs in a DataFrame cell.
///
/// Unlike `Value::to_string()` (which serialises the value back to JSON
/// syntax -- so a JSON string `"abc"` would round-trip as the 5-character
/// text `"abc"`, quote marks included, and JSON `null` would come back as
/// the literal 4-character word `null`), this matches on the actual
/// variant:
///
/// - `String` values are taken verbatim, with no surrounding quotes.
/// - `Number` and `Bool` values use their plain text form (`42`, `3.14`,
///   `true`) -- already quote-free, since neither type is ever quoted in
///   JSON.
/// - `Null` becomes an empty string, the same "missing" marker used
///   elsewhere for absent keys and short CSV rows.
/// - `Array`/`Object` (nested structures have no dedicated DataFrame
///   column type) fall back to their JSON text form, so a nested value is
///   still readable rather than causing the whole read to fail.
fn json_value_to_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
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
                    values.push(json_value_to_cell(value));
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
            let values: Vec<String> = array.iter().map(json_value_to_cell).collect();

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
    let mut writer = BufWriter::new(file);

    let json_value = match orient {
        JsonOrient::Records => to_records_json(df)?,
        JsonOrient::Columns => to_column_json(df)?,
    };

    serde_json::to_writer_pretty(&mut writer, &json_value).map_err(PandRSError::Json)?;

    // `BufWriter`'s `Drop` impl flushes too, but silently discards any I/O
    // error it hits (there's nowhere for it to report to). Flushing here
    // explicitly means a failure (e.g. disk full) surfaces as an `Err`
    // instead of a silently-truncated file and a misleading `Ok(())`.
    writer.flush().map_err(PandRSError::Io)?;

    Ok(())
}

/// JSON output orientation
pub enum JsonOrient {
    /// Record-oriented [{col1:val1, col2:val2}, ...]
    Records,
    /// Column-oriented {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

/// Materialise one DataFrame column as native `serde_json::Value`s,
/// preserving its real element type: `i64`/`f64` columns become JSON
/// numbers (a non-finite float becomes JSON `null`, since JSON has no
/// `NaN`/`Infinity` literal -- the same convention pandas' own `to_json`
/// uses), and `bool` columns become JSON booleans. Any other column type
/// falls back to its string representation via
/// `DataFrame::get_column_string_values` (the same fallback
/// `DataFrame::to_csv` uses for the same columns), each value becoming a
/// JSON string. This mirrors `SerializeExt::to_json`'s column handling.
fn column_json_values(df: &DataFrame, col_name: &str) -> Result<Vec<Value>> {
    if let Ok(series) = df.get_column::<i64>(col_name) {
        return Ok(series.values().iter().map(|v| Value::from(*v)).collect());
    }
    if let Ok(series) = df.get_column::<f64>(col_name) {
        return Ok(series.values().iter().map(|v| Value::from(*v)).collect());
    }
    if let Ok(series) = df.get_column::<bool>(col_name) {
        return Ok(series.values().iter().map(|v| Value::from(*v)).collect());
    }
    // `Series<&'static str>` (built straight from string literals, e.g.
    // `Series::new(vec!["a", "b"], ...)`) is a common, naturally-typed
    // column that `DataFrame::get_column_string_values`'s downcast chain
    // does not recognise (it only matches `Series<String>`). Handling it
    // here directly -- rather than falling through to the generic
    // fallback below -- keeps a JSON write from replacing real text with
    // that fallback's `"unsupported_type_<col>_<row>"` placeholder.
    if let Ok(series) = df.get_column::<&'static str>(col_name) {
        return Ok(series
            .values()
            .iter()
            .map(|v| Value::String(v.to_string()))
            .collect());
    }

    let strings = df.get_column_string_values(col_name)?;
    Ok(strings.into_iter().map(Value::String).collect())
}

// Convert DataFrame to record-oriented JSON
fn to_records_json(df: &DataFrame) -> Result<Value> {
    let col_names = df.column_names();
    let mut per_column: Vec<(String, Vec<Value>)> = Vec::with_capacity(col_names.len());
    for name in col_names {
        per_column.push((name.clone(), column_json_values(df, name)?));
    }

    let mut records = Vec::with_capacity(df.row_count());
    for row_idx in 0..df.row_count() {
        let mut record = serde_json::Map::with_capacity(per_column.len());
        for (name, values) in &per_column {
            let cell = values.get(row_idx).cloned().unwrap_or(Value::Null);
            record.insert(name.clone(), cell);
        }
        records.push(Value::Object(record));
    }

    Ok(Value::Array(records))
}

// Convert DataFrame to column-oriented JSON
fn to_column_json(df: &DataFrame) -> Result<Value> {
    let mut columns = serde_json::Map::new();

    for name in df.column_names() {
        let values = column_json_values(df, name)?;
        columns.insert(name.clone(), Value::Array(values));
    }

    Ok(Value::Object(columns))
}

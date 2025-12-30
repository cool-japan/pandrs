use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde_json::{Map, Value};

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::optimized::split_dataframe::core::OptimizedDataFrame;

/// JSON output format
pub enum JsonOrient {
    /// Records format [{col1:val1, col2:val2}, ...]
    Records,
    /// Columns format {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

impl OptimizedDataFrame {
    /// Load DataFrame from a JSON file
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file
    ///
    /// # Returns
    /// * `Result<Self>` - The loaded DataFrame
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::Io(e))?;
        let reader = BufReader::new(file);

        // Parse JSON
        let json_value: Value = serde_json::from_reader(reader).map_err(|e| Error::Json(e))?;

        match json_value {
            Value::Array(array) => Self::from_records_array(array),
            Value::Object(map) => Self::from_column_oriented(map),
            _ => Err(Error::Format(
                "JSON must be an object or an array".to_string(),
            )),
        }
    }

    // Load from records-oriented JSON
    fn from_records_array(array: Vec<Value>) -> Result<Self> {
        let mut df = Self::new();

        // Return empty DataFrame if array is empty
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
                return Err(Error::Format(
                    "Each element of the array must be an object".to_string(),
                ));
            }
        }

        // Sort keys to stabilize order
        let keys: Vec<String> = all_keys.into_iter().collect();

        // Collect column data
        let mut string_values: HashMap<String, Vec<String>> = HashMap::new();

        for key in &keys {
            string_values.insert(key.clone(), Vec::with_capacity(array.len()));
        }

        for item in &array {
            if let Value::Object(map) = item {
                for key in &keys {
                    let value_str = if let Some(value) = map.get(key) {
                        match value {
                            Value::Null => String::new(),
                            Value::Bool(b) => b.to_string(),
                            Value::Number(n) => n.to_string(),
                            Value::String(s) => s.clone(),
                            _ => serde_json::to_string(value).unwrap_or_default(),
                        }
                    } else {
                        String::new()
                    };
                    string_values.get_mut(key).unwrap().push(value_str);
                }
            }
        }

        // Infer types and add columns
        for key in &keys {
            let values = &string_values[key];

            // Check for non-empty values
            let non_empty_values: Vec<&String> = values.iter().filter(|s| !s.is_empty()).collect();

            if non_empty_values.is_empty() {
                // If all values are empty, use string type
                df.add_column(
                    key.clone(),
                    Column::String(crate::column::StringColumn::new(
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
                df.add_column(
                    key.clone(),
                    Column::Int64(crate::column::Int64Column::new(int_values)),
                )?;
                continue;
            }

            // Try to parse as floating point numbers
            let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
            if all_floats {
                let float_values: Vec<f64> = values
                    .iter()
                    .map(|s| s.parse::<f64>().unwrap_or(0.0))
                    .collect();
                df.add_column(
                    key.clone(),
                    Column::Float64(crate::column::Float64Column::new(float_values)),
                )?;
                continue;
            }

            // Try to parse as boolean values
            let all_bools = non_empty_values.iter().all(|&s| {
                let lower = s.to_lowercase();
                lower == "true" || lower == "false"
            });

            if all_bools {
                let bool_values: Vec<bool> =
                    values.iter().map(|s| s.to_lowercase() == "true").collect();
                df.add_column(
                    key.clone(),
                    Column::Boolean(crate::column::BooleanColumn::new(bool_values)),
                )?;
            } else {
                // Default to string type
                df.add_column(
                    key.clone(),
                    Column::String(crate::column::StringColumn::new(
                        values.iter().map(|s| s.clone()).collect(),
                    )),
                )?;
            }
        }

        Ok(df)
    }

    // Load from column-oriented JSON
    fn from_column_oriented(map: Map<String, Value>) -> Result<Self> {
        let mut df = Self::new();

        // Return empty DataFrame if object is empty
        if map.is_empty() {
            return Ok(df);
        }

        // Verify column lengths
        let mut column_length = 0;
        for (_, value) in &map {
            if let Value::Array(array) = value {
                if column_length == 0 {
                    column_length = array.len();
                } else if array.len() != column_length {
                    return Err(Error::Format(
                        "All columns must have the same length".to_string(),
                    ));
                }
            } else {
                return Err(Error::Format("JSON values must be arrays".to_string()));
            }
        }

        // Process column data
        for (key, value) in map {
            if let Value::Array(array) = value {
                // Convert values to strings
                let str_values: Vec<String> = array
                    .iter()
                    .map(|v| match v {
                        Value::Null => String::new(),
                        Value::Bool(b) => b.to_string(),
                        Value::Number(n) => n.to_string(),
                        Value::String(s) => s.clone(),
                        _ => serde_json::to_string(v).unwrap_or_default(),
                    })
                    .collect();

                // Check for non-empty values
                let non_empty_values: Vec<&String> =
                    str_values.iter().filter(|s| !s.is_empty()).collect();

                if non_empty_values.is_empty() {
                    // If all values are empty, use string type
                    df.add_column(
                        key.clone(),
                        Column::String(crate::column::StringColumn::new(
                            str_values.iter().map(|s| s.clone()).collect(),
                        )),
                    )?;
                    continue;
                }

                // Try to parse as integers
                let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
                if all_ints {
                    let int_values: Vec<i64> = str_values
                        .iter()
                        .map(|s| s.parse::<i64>().unwrap_or(0))
                        .collect();
                    df.add_column(
                        key.clone(),
                        Column::Int64(crate::column::Int64Column::new(int_values)),
                    )?;
                    continue;
                }

                // Try to parse as floating point numbers
                let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
                if all_floats {
                    let float_values: Vec<f64> = str_values
                        .iter()
                        .map(|s| s.parse::<f64>().unwrap_or(0.0))
                        .collect();
                    df.add_column(
                        key.clone(),
                        Column::Float64(crate::column::Float64Column::new(float_values)),
                    )?;
                    continue;
                }

                // Try to parse as boolean values
                let all_bools = non_empty_values.iter().all(|&s| {
                    let lower = s.to_lowercase();
                    lower == "true" || lower == "false"
                });

                if all_bools {
                    let bool_values: Vec<bool> = str_values
                        .iter()
                        .map(|s| s.to_lowercase() == "true")
                        .collect();
                    df.add_column(
                        key.clone(),
                        Column::Boolean(crate::column::BooleanColumn::new(bool_values)),
                    )?;
                } else {
                    // Default to string type
                    df.add_column(
                        key.clone(),
                        Column::String(crate::column::StringColumn::new(
                            str_values.iter().map(|s| s.clone()).collect(),
                        )),
                    )?;
                }
            }
        }

        Ok(df)
    }

    /// Write DataFrame to a JSON file
    ///
    /// # Arguments
    /// * `path` - Path to the output JSON file
    /// * `orient` - JSON output format (Records or Columns)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful
    pub fn to_json<P: AsRef<Path>>(&self, path: P, orient: JsonOrient) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| Error::Io(e))?;
        let writer = BufWriter::new(file);

        // Convert to JSON format
        let json_value = match orient {
            JsonOrient::Records => self.to_records_json()?,
            JsonOrient::Columns => self.to_column_json()?,
        };

        // Write JSON
        serde_json::to_writer_pretty(writer, &json_value).map_err(|e| Error::Json(e))?;

        Ok(())
    }

    // Convert DataFrame to records-oriented JSON
    fn to_records_json(&self) -> Result<Value> {
        let mut records = Vec::with_capacity(self.row_count());

        // Return empty array if there are no rows
        if self.row_count() == 0 {
            return Ok(Value::Array(records));
        }

        // Process data for each row
        for row_idx in 0..self.row_count() {
            let mut record = Map::new();

            // Get value from each column
            for (col_idx, col_name) in self.column_names().iter().enumerate() {
                let column = &self.columns[col_idx];

                // Convert column value to JSON value
                let value = match column {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Number(serde_json::Number::from(val))
                        } else {
                            Value::Null
                        }
                    }
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            // Convert f64 to Number (use Null for NaN or Infinity which can't be processed)
                            if val.is_finite() {
                                serde_json::Number::from_f64(val).map_or(Value::Null, Value::Number)
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    }
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::String(val.to_string())
                        } else {
                            Value::Null
                        }
                    }
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Bool(val)
                        } else {
                            Value::Null
                        }
                    }
                };

                record.insert(col_name.clone(), value);
            }

            records.push(Value::Object(record));
        }

        Ok(Value::Array(records))
    }

    // Convert DataFrame to column-oriented JSON
    fn to_column_json(&self) -> Result<Value> {
        let mut columns = serde_json::Map::new();

        // Return empty object if there are no rows
        if self.row_count() == 0 {
            return Ok(Value::Object(columns));
        }

        // Process each column
        for (col_idx, col_name) in self.column_names().iter().enumerate() {
            let mut values = Vec::new();

            // Get all values in the column
            for row_idx in 0..self.row_count() {
                let value = match &self.columns[col_idx] {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Number(serde_json::Number::from(val))
                        } else {
                            Value::Null
                        }
                    }
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            // Convert f64 to Number (use Null for NaN or Infinity which can't be processed)
                            if val.is_finite() {
                                serde_json::Number::from_f64(val).map_or(Value::Null, Value::Number)
                            } else {
                                Value::Null
                            }
                        } else {
                            Value::Null
                        }
                    }
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::String(val.to_string())
                        } else {
                            Value::Null
                        }
                    }
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(row_idx) {
                            Value::Bool(val)
                        } else {
                            Value::Null
                        }
                    }
                };

                values.push(value);
            }

            columns.insert(col_name.clone(), Value::Array(values));
        }

        Ok(Value::Object(columns))
    }
}

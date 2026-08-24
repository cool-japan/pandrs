//! Conversion utilities between PandRS DataFrame and Apache Arrow RecordBatch.
//!
//! These conversions form the core data-interchange layer used by the Arrow Flight
//! transport. They are available whenever the `distributed` feature is enabled
//! (which implies the `arrow` dependency).

use arrow::{
    array::{
        Array, ArrayRef, BooleanArray, BooleanBuilder, Float64Array, Float64Builder, Int64Array,
        Int64Builder, StringArray, StringBuilder,
    },
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::base::Series; // used in tests and for series construction

// ---------------------------------------------------------------------------
// DataFrame → RecordBatch
// ---------------------------------------------------------------------------

/// Convert a PandRS [`DataFrame`] to an Apache Arrow [`RecordBatch`].
///
/// Each column of the DataFrame is inspected to determine the best Arrow
/// data type. Numeric strings are promoted to `Int64` or `Float64`; every
/// other value falls back to `Utf8`.
pub fn dataframe_to_record_batch(df: &DataFrame) -> Result<RecordBatch> {
    let column_names = df.column_names();
    let mut fields: Vec<Field> = Vec::with_capacity(column_names.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(column_names.len());

    for name in column_names {
        let (field, array) = column_to_arrow(df, name)?;
        fields.push(field);
        arrays.push(array);
    }

    let schema = Arc::new(Schema::new(fields));

    // When there are no columns, Arrow requires using `new_empty` because
    // `try_new` demands at least one column or an explicit row count.
    if arrays.is_empty() {
        return Ok(RecordBatch::new_empty(schema));
    }

    RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::InvalidOperation(format!("Failed to create RecordBatch: {e}")))
}

/// Convert multiple [`RecordBatch`]es to a single [`DataFrame`].
///
/// All batches must share the same schema.  Rows are concatenated in order.
pub fn record_batches_to_dataframe(batches: &[RecordBatch]) -> Result<DataFrame> {
    if batches.is_empty() {
        return Ok(DataFrame::new());
    }

    // Concatenate all batches column-wise using a simple approach:
    // collect values per column across all batches, then build a single DataFrame.
    let schema = batches[0].schema();
    let num_columns = schema.fields().len();

    let mut df = DataFrame::new();

    for col_idx in 0..num_columns {
        let field = schema.field(col_idx);
        let col_name = field.name().clone();

        // Collect all values for this column across all batches
        let mut values: Vec<String> = Vec::new();
        for batch in batches {
            let array = batch.column(col_idx);
            let col_values = arrow_array_to_strings(array)?;
            values.extend(col_values);
        }

        let series = Series::new(values, Some(col_name.clone()))
            .map_err(|e| Error::InvalidOperation(format!("Series creation failed: {e}")))?;
        df.add_column(col_name, series)?;
    }

    Ok(df)
}

/// Convert an Apache Arrow [`RecordBatch`] to a PandRS [`DataFrame`].
pub fn record_batch_to_dataframe(batch: &RecordBatch) -> Result<DataFrame> {
    record_batches_to_dataframe(std::slice::from_ref(batch))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Classify column data and build an Arrow (Field, ArrayRef) pair.
fn column_to_arrow(df: &DataFrame, col_name: &str) -> Result<(Field, ArrayRef)> {
    let row_count = df.row_count();

    // Collect the string values for this column.
    // Series<String> stores data as strings; we read them back and probe types.
    let values: Vec<Option<String>> = collect_column_values(df, col_name, row_count)?;

    // Probe numeric types: try Int64, then Float64, else Utf8
    let detected = detect_column_type(&values);

    match detected {
        ColumnKind::Int64 => {
            let mut builder = Int64Builder::with_capacity(row_count);
            for v in &values {
                match v {
                    None => builder.append_null(),
                    Some(s) if s == "null" || s.is_empty() => builder.append_null(),
                    Some(s) => {
                        let parsed = s.parse::<i64>().map_err(|e| {
                            Error::InvalidOperation(format!(
                                "Failed to parse '{s}' as Int64 in column '{col_name}': {e}"
                            ))
                        })?;
                        builder.append_value(parsed);
                    }
                }
            }
            let field = Field::new(col_name, DataType::Int64, true);
            Ok((field, Arc::new(builder.finish()) as ArrayRef))
        }
        ColumnKind::Float64 => {
            let mut builder = Float64Builder::with_capacity(row_count);
            for v in &values {
                match v {
                    None => builder.append_null(),
                    Some(s) if s == "null" || s.is_empty() => builder.append_null(),
                    Some(s) => {
                        let parsed = s.parse::<f64>().map_err(|e| {
                            Error::InvalidOperation(format!(
                                "Failed to parse '{s}' as Float64 in column '{col_name}': {e}"
                            ))
                        })?;
                        builder.append_value(parsed);
                    }
                }
            }
            let field = Field::new(col_name, DataType::Float64, true);
            Ok((field, Arc::new(builder.finish()) as ArrayRef))
        }
        ColumnKind::Boolean => {
            let mut builder = BooleanBuilder::with_capacity(row_count);
            for v in &values {
                match v {
                    None => builder.append_null(),
                    Some(s) if s == "null" || s.is_empty() => builder.append_null(),
                    Some(s) => {
                        let b = match s.to_lowercase().as_str() {
                            "true" | "1" | "yes" => true,
                            _ => false,
                        };
                        builder.append_value(b);
                    }
                }
            }
            let field = Field::new(col_name, DataType::Boolean, true);
            Ok((field, Arc::new(builder.finish()) as ArrayRef))
        }
        ColumnKind::Utf8 => {
            let mut builder = StringBuilder::with_capacity(row_count, row_count * 8);
            for v in &values {
                match v {
                    None => builder.append_null(),
                    Some(s) if s == "null" => builder.append_null(),
                    Some(s) => builder.append_value(s),
                }
            }
            let field = Field::new(col_name, DataType::Utf8, true);
            Ok((field, Arc::new(builder.finish()) as ArrayRef))
        }
    }
}

/// Collect all string values for a given column from the DataFrame.
fn collect_column_values(
    df: &DataFrame,
    col_name: &str,
    _row_count: usize,
) -> Result<Vec<Option<String>>> {
    let strings = df.get_column_string_values(col_name)?;
    Ok(strings.into_iter().map(Some).collect())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColumnKind {
    Int64,
    Float64,
    Boolean,
    Utf8,
}

/// Probe sample values to determine the best Arrow type.
fn detect_column_type(values: &[Option<String>]) -> ColumnKind {
    let non_null: Vec<&str> = values
        .iter()
        .filter_map(|v| v.as_deref())
        .filter(|s| !s.is_empty() && *s != "null")
        .collect();

    if non_null.is_empty() {
        return ColumnKind::Utf8;
    }

    // Check boolean first (limited vocabulary)
    let bool_words = ["true", "false", "yes", "no", "1", "0"];
    if non_null
        .iter()
        .all(|s| bool_words.contains(&s.to_lowercase().as_str()))
    {
        // Only classify as boolean if ALL values are bool-like and the set
        // is not just integers (which are better expressed as Int64).
        let has_bool_word = non_null
            .iter()
            .any(|s| matches!(s.to_lowercase().as_str(), "true" | "false" | "yes" | "no"));
        if has_bool_word {
            return ColumnKind::Boolean;
        }
    }

    // Try Int64
    if non_null.iter().all(|s| s.parse::<i64>().is_ok()) {
        return ColumnKind::Int64;
    }

    // Try Float64
    if non_null.iter().all(|s| s.parse::<f64>().is_ok()) {
        return ColumnKind::Float64;
    }

    ColumnKind::Utf8
}

/// Convert an Arrow array to a `Vec<String>` for insertion into a Series.
fn arrow_array_to_strings(array: &dyn Array) -> Result<Vec<String>> {
    let len = array.len();
    let mut out = Vec::with_capacity(len);

    match array.data_type() {
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::InvalidOperation("Expected Int64Array".into()))?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push("null".to_string());
                } else {
                    out.push(arr.value(i).to_string());
                }
            }
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::InvalidOperation("Expected Float64Array".into()))?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push("null".to_string());
                } else {
                    out.push(arr.value(i).to_string());
                }
            }
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| Error::InvalidOperation("Expected BooleanArray".into()))?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push("null".to_string());
                } else {
                    out.push(arr.value(i).to_string());
                }
            }
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::InvalidOperation("Expected StringArray".into()))?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push("null".to_string());
                } else {
                    out.push(arr.value(i).to_string());
                }
            }
        }
        other => {
            return Err(Error::NotImplemented(format!(
                "Arrow type {other:?} conversion is not yet supported"
            )));
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::base::Series;

    fn make_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "int_col".to_string(),
            Series::new(
                vec!["1".to_string(), "2".to_string(), "3".to_string()],
                Some("int_col".to_string()),
            )
            .expect("series creation"),
        )
        .expect("add column");
        df.add_column(
            "str_col".to_string(),
            Series::new(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                Some("str_col".to_string()),
            )
            .expect("series creation"),
        )
        .expect("add column");
        df
    }

    #[test]
    fn test_dataframe_to_record_batch_shape() {
        let df = make_df();
        let batch = dataframe_to_record_batch(&df).expect("conversion");
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn test_record_batch_to_dataframe_roundtrip() {
        let df = make_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        let df2 = record_batch_to_dataframe(&batch).expect("to_dataframe");
        assert_eq!(df2.column_names(), df.column_names());
        assert_eq!(df2.row_count(), df.row_count());
    }

    #[test]
    fn test_record_batches_to_dataframe_concat() {
        let df = make_df();
        let batch = dataframe_to_record_batch(&df).expect("to_record_batch");
        // Concatenate the batch with itself
        let combined =
            record_batches_to_dataframe(&[batch.clone(), batch.clone()]).expect("concat");
        assert_eq!(combined.row_count(), 6);
        assert_eq!(combined.column_names(), df.column_names());
    }

    #[test]
    fn test_empty_dataframe_to_record_batch() {
        let df = DataFrame::new();
        let batch = dataframe_to_record_batch(&df).expect("empty conversion");
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_detect_column_type_int() {
        let vals: Vec<Option<String>> =
            vec![Some("1".into()), Some("2".into()), Some("100".into())];
        assert_eq!(detect_column_type(&vals), ColumnKind::Int64);
    }

    #[test]
    fn test_detect_column_type_float() {
        let vals: Vec<Option<String>> =
            vec![Some("1.5".into()), Some("2.7".into()), Some("3.0".into())];
        assert_eq!(detect_column_type(&vals), ColumnKind::Float64);
    }

    #[test]
    fn test_detect_column_type_string() {
        let vals: Vec<Option<String>> = vec![Some("hello".into()), Some("world".into())];
        assert_eq!(detect_column_type(&vals), ColumnKind::Utf8);
    }
}

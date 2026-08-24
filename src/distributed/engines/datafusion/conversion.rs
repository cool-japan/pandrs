//! # Conversion Utilities for DataFusion
//!
//! This module provides utilities for converting between PandRS and DataFusion data types.
//!
//! ## Column representation note
//!
//! `crate::dataframe::DataFrame` stores every column as a boxed `Series<T>`.
//! `Series<T>` is a plain `Vec<T>` and has **no null slot**, and the base
//! DataFrame has no typed *and* nullable column type its own machinery reads
//! back. Therefore [`record_batches_to_dataframe`] materialises:
//!
//! * **null-free** Arrow columns as a fully typed `Series<T>` (e.g. `Int64`
//!   becomes `Series<i64>`, `Float64` becomes `Series<f64>`, `Boolean` becomes
//!   `Series<bool>`), preserving the exact value with no `f64` round-trip; and
//! * **null-containing** columns as a `Series<String>` where a null is the
//!   empty string `""` — the representation that the reverse conversion
//!   (`determine_arrow_type`/`build_array_from_dataframe`) and the sibling
//!   Flight conversion both treat as null. This preserves NA/null semantics
//!   losslessly even though the element type is rendered textually.
//!
//! Real string / temporal columns are always materialised as `Series<String>`
//! (their natural textual form) with `""` for null.

#[cfg(feature = "distributed")]
use crate::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use crate::error::{Error, Result};
#[cfg(feature = "distributed")]
use crate::series::Series;
#[cfg(feature = "distributed")]
use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray, StringViewArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
#[cfg(feature = "distributed")]
use arrow::datatypes::{DataType, Field, TimeUnit};
#[cfg(feature = "distributed")]
use std::sync::Arc;

/// Converts a PandRS DataFrame to Arrow record batches
#[cfg(feature = "distributed")]
pub fn dataframe_to_record_batches(
    df: &DataFrame,
    batch_size: usize,
) -> Result<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::datatypes::Schema;
    use arrow::record_batch::RecordBatch;

    if df.nrows() == 0 {
        let schema = Arc::new(Schema::new(vec![] as Vec<arrow::datatypes::Field>));
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }

    // Build schema from DataFrame columns
    let mut fields = Vec::new();
    let column_names = df.column_names();

    for column_name in column_names {
        // Determine field type by examining column data
        let field_type = determine_arrow_type(df, column_name)?;
        let field = Field::new(column_name, field_type, true); // Allow nulls
        fields.push(field);
    }

    let schema = Arc::new(Schema::new(fields));
    let mut batches = Vec::new();

    // Split data into batches
    let total_rows = df.nrows();
    let batch_size = std::cmp::max(1, batch_size);
    let num_batches = total_rows.div_ceil(batch_size);

    for batch_idx in 0..num_batches {
        let start_row = batch_idx * batch_size;
        let end_row = std::cmp::min(start_row + batch_size, total_rows);
        let batch_row_count = end_row - start_row;

        // Build arrays for this batch
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for column_name in column_names {
            let array = build_array_from_dataframe(df, column_name, start_row, batch_row_count)?;
            arrays.push(array);
        }

        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| Error::InvalidValue(format!("Failed to create RecordBatch: {}", e)))?;
        batches.push(batch);
    }

    Ok(batches)
}

/// Converts Arrow record batches to a PandRS DataFrame.
///
/// Each column is materialised as a typed `Series<T>` when it contains no
/// nulls, or as a null-preserving `Series<String>` (empty string = null) when
/// it does. See the module-level documentation for the rationale.
#[cfg(feature = "distributed")]
pub fn record_batches_to_dataframe(
    batches: &[arrow::record_batch::RecordBatch],
) -> Result<DataFrame> {
    if batches.is_empty() {
        return Ok(DataFrame::new());
    }

    let schema = batches[0].schema();
    let mut df = DataFrame::new();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let name = field.name();
        add_column_from_batches(&mut df, batches, col_idx, name, field.data_type())?;
    }

    Ok(df)
}

/// Returns `true` if the column at `col_idx` contains at least one null across
/// all batches.
#[cfg(feature = "distributed")]
fn column_has_nulls(batches: &[arrow::record_batch::RecordBatch], col_idx: usize) -> bool {
    batches.iter().any(|b| b.column(col_idx).null_count() > 0)
}

/// Extracts one typed primitive column and appends it to `df`, choosing a typed
/// `Series<T>` (null-free) or a `Series<String>` fallback (has nulls).
#[cfg(feature = "distributed")]
macro_rules! add_primitive_column {
    ($df:expr, $batches:expr, $col_idx:expr, $name:expr, $arr_ty:ty, $rust_ty:ty) => {{
        if column_has_nulls($batches, $col_idx) {
            let mut values: Vec<String> = Vec::new();
            for batch in $batches {
                let array = downcast::<$arr_ty>(batch.column($col_idx), $name)?;
                for i in 0..array.len() {
                    if array.is_null(i) {
                        values.push(String::new());
                    } else {
                        values.push(array.value(i).to_string());
                    }
                }
            }
            $df.add_column(
                $name.to_string(),
                Series::new(values, Some($name.to_string()))?,
            )?;
        } else {
            let mut values: Vec<$rust_ty> = Vec::new();
            for batch in $batches {
                let array = downcast::<$arr_ty>(batch.column($col_idx), $name)?;
                for i in 0..array.len() {
                    values.push(array.value(i));
                }
            }
            $df.add_column(
                $name.to_string(),
                Series::new(values, Some($name.to_string()))?,
            )?;
        }
    }};
}

/// Downcasts an Arrow array to a concrete array type, returning a typed error.
#[cfg(feature = "distributed")]
fn downcast<'a, A: 'static>(array: &'a ArrayRef, name: &str) -> Result<&'a A> {
    array.as_any().downcast_ref::<A>().ok_or_else(|| {
        Error::InvalidInput(format!(
            "Column '{}' did not have the expected Arrow array layout for {}",
            name,
            std::any::type_name::<A>()
        ))
    })
}

/// Dispatches on the Arrow data type and appends the corresponding typed column.
#[cfg(feature = "distributed")]
fn add_column_from_batches(
    df: &mut DataFrame,
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    name: &str,
    data_type: &DataType,
) -> Result<()> {
    match data_type {
        DataType::Boolean => add_primitive_column!(df, batches, col_idx, name, BooleanArray, bool),
        DataType::Int8 => add_primitive_column!(df, batches, col_idx, name, Int8Array, i8),
        DataType::Int16 => add_primitive_column!(df, batches, col_idx, name, Int16Array, i16),
        DataType::Int32 => add_primitive_column!(df, batches, col_idx, name, Int32Array, i32),
        DataType::Int64 => add_primitive_column!(df, batches, col_idx, name, Int64Array, i64),
        DataType::UInt8 => add_primitive_column!(df, batches, col_idx, name, UInt8Array, u8),
        DataType::UInt16 => add_primitive_column!(df, batches, col_idx, name, UInt16Array, u16),
        DataType::UInt32 => add_primitive_column!(df, batches, col_idx, name, UInt32Array, u32),
        DataType::UInt64 => add_primitive_column!(df, batches, col_idx, name, UInt64Array, u64),
        DataType::Float32 => add_primitive_column!(df, batches, col_idx, name, Float32Array, f32),
        DataType::Float64 => add_primitive_column!(df, batches, col_idx, name, Float64Array, f64),
        DataType::Utf8 => add_string_column::<StringArray>(df, batches, col_idx, name)?,
        DataType::LargeUtf8 => add_string_column::<LargeStringArray>(df, batches, col_idx, name)?,
        DataType::Utf8View => add_string_column::<StringViewArray>(df, batches, col_idx, name)?,
        DataType::Date32 => {
            let mut values: Vec<String> = Vec::new();
            for batch in batches {
                let array = downcast::<Date32Array>(batch.column(col_idx), name)?;
                for i in 0..array.len() {
                    if array.is_null(i) {
                        values.push(String::new());
                    } else {
                        values.push(
                            array
                                .value_as_date(i)
                                .map(|d| d.format("%Y-%m-%d").to_string())
                                .unwrap_or_default(),
                        );
                    }
                }
            }
            df.add_column(
                name.to_string(),
                Series::new(values, Some(name.to_string()))?,
            )?;
        }
        DataType::Date64 => {
            let mut values: Vec<String> = Vec::new();
            for batch in batches {
                let array = downcast::<Date64Array>(batch.column(col_idx), name)?;
                for i in 0..array.len() {
                    if array.is_null(i) {
                        values.push(String::new());
                    } else {
                        values.push(
                            array
                                .value_as_date(i)
                                .map(|d| d.format("%Y-%m-%d").to_string())
                                .unwrap_or_default(),
                        );
                    }
                }
            }
            df.add_column(
                name.to_string(),
                Series::new(values, Some(name.to_string()))?,
            )?;
        }
        DataType::Timestamp(unit, _tz) => {
            let rendered = render_timestamps(batches, col_idx, name, *unit)?;
            df.add_column(
                name.to_string(),
                Series::new(rendered, Some(name.to_string()))?,
            )?;
        }
        other => {
            return Err(Error::NotImplemented(format!(
                "Conversion of Arrow data type {} to a PandRS column is not implemented",
                other
            )));
        }
    }
    Ok(())
}

/// Appends a string-like Arrow column as a `Series<String>`, preserving the real
/// string value and mapping null to the empty string.
#[cfg(feature = "distributed")]
fn add_string_column<A>(
    df: &mut DataFrame,
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    name: &str,
) -> Result<()>
where
    A: Array + 'static,
    for<'a> &'a A: IntoIterator<Item = Option<&'a str>>,
{
    let mut values: Vec<String> = Vec::new();
    for batch in batches {
        let array = downcast::<A>(batch.column(col_idx), name)?;
        for item in array.into_iter() {
            match item {
                Some(s) => values.push(s.to_string()),
                None => values.push(String::new()),
            }
        }
    }
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string()))?,
    )?;
    Ok(())
}

/// Renders a timestamp column to `YYYY-MM-DD HH:MM:SS%.f` strings (`""` for null).
///
/// The column layout is downcast once per batch (by unit); a null or an
/// out-of-range value becomes the empty string.
#[cfg(feature = "distributed")]
fn render_timestamps(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    name: &str,
    unit: TimeUnit,
) -> Result<Vec<String>> {
    let mut values: Vec<String> = Vec::new();
    for batch in batches {
        let col = batch.column(col_idx);
        let len = col.len();
        for i in 0..len {
            let dt = match unit {
                TimeUnit::Second => {
                    downcast::<TimestampSecondArray>(col, name)?.value_as_datetime(i)
                }
                TimeUnit::Millisecond => {
                    downcast::<TimestampMillisecondArray>(col, name)?.value_as_datetime(i)
                }
                TimeUnit::Microsecond => {
                    downcast::<TimestampMicrosecondArray>(col, name)?.value_as_datetime(i)
                }
                TimeUnit::Nanosecond => {
                    downcast::<TimestampNanosecondArray>(col, name)?.value_as_datetime(i)
                }
            };
            if col.is_null(i) {
                values.push(String::new());
            } else {
                match dt {
                    Some(ndt) => values.push(ndt.format("%Y-%m-%d %H:%M:%S%.f").to_string()),
                    None => values.push(String::new()),
                }
            }
        }
    }
    Ok(values)
}

/// Determines the Arrow data type for a DataFrame column.
///
/// The type is **derived from the declared pandrs `Series<T>` element type**
/// whenever the column is a typed series (`Series<i64>`, `Series<f64>`,
/// `Series<bool>`, …). Only genuinely textual `Series<String>` columns fall
/// back to value probing, and that probe requires *every* non-empty sample to
/// conform (rather than a majority vote) so that a single stray value cannot
/// silently retype — and thereby null out — the rest of the column.
#[cfg(feature = "distributed")]
fn determine_arrow_type(df: &DataFrame, column_name: &str) -> Result<DataType> {
    // 1. Derive directly from the declared Series<T> type when possible.
    if df.get_column::<bool>(column_name).is_ok() {
        return Ok(DataType::Boolean);
    }
    if df.get_column::<i64>(column_name).is_ok()
        || df.get_column::<i32>(column_name).is_ok()
        || df.get_column::<i16>(column_name).is_ok()
        || df.get_column::<i8>(column_name).is_ok()
        || df.get_column::<isize>(column_name).is_ok()
        || df.get_column::<u64>(column_name).is_ok()
        || df.get_column::<u32>(column_name).is_ok()
        || df.get_column::<u16>(column_name).is_ok()
        || df.get_column::<u8>(column_name).is_ok()
        || df.get_column::<usize>(column_name).is_ok()
    {
        return Ok(DataType::Int64);
    }
    if df.get_column::<f64>(column_name).is_ok() || df.get_column::<f32>(column_name).is_ok() {
        return Ok(DataType::Float64);
    }

    // 2. Textual column: probe the string values. Require all non-empty values
    //    to conform to a candidate type.
    let string_values = df.get_column_string_values(column_name)?;

    let mut total_non_empty = 0usize;
    let mut all_bool = true;
    let mut all_int = true;
    let mut all_float = true;
    let mut all_date = true;

    for value in string_values.iter() {
        if value.is_empty() {
            continue; // treated as null; does not constrain the type
        }
        total_non_empty += 1;

        let lower_val = value.to_lowercase();
        if !(lower_val == "true" || lower_val == "false" || lower_val == "t" || lower_val == "f") {
            all_bool = false;
        }
        if value.parse::<i64>().is_err() {
            all_int = false;
        }
        if value.parse::<f64>().is_err() {
            all_float = false;
        }
        let is_date = chrono::DateTime::parse_from_rfc3339(value).is_ok()
            || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S").is_ok()
            || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").is_ok()
            || chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok();
        if !is_date {
            all_date = false;
        }
    }

    if total_non_empty == 0 {
        return Ok(DataType::Utf8); // Default to string for empty/all-null data
    }

    if all_bool {
        Ok(DataType::Boolean)
    } else if all_int {
        Ok(DataType::Int64)
    } else if all_float {
        Ok(DataType::Float64)
    } else if all_date {
        Ok(DataType::Timestamp(TimeUnit::Nanosecond, None))
    } else {
        Ok(DataType::Utf8)
    }
}

/// Builds an Arrow array from DataFrame column data
#[cfg(feature = "distributed")]
fn build_array_from_dataframe(
    df: &DataFrame,
    column_name: &str,
    start_row: usize,
    row_count: usize,
) -> Result<ArrayRef> {
    let string_values = df.get_column_string_values(column_name)?;
    let data_type = determine_arrow_type(df, column_name)?;

    // Extract the relevant slice
    let end_row = start_row + row_count;
    let slice = &string_values[start_row..std::cmp::min(end_row, string_values.len())];

    match data_type {
        DataType::Boolean => {
            let mut builder = arrow::array::BooleanBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    let lower_val = value.to_lowercase();
                    let bool_val = lower_val == "true" || lower_val == "t" || lower_val == "1";
                    builder.append_value(bool_val);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Int64 => {
            let mut builder = arrow::array::Int64Builder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else if let Ok(int_val) = value.parse::<i64>() {
                    builder.append_value(int_val);
                } else {
                    builder.append_null();
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = arrow::array::Float64Builder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else if let Ok(float_val) = value.parse::<f64>() {
                    builder.append_value(float_val);
                } else {
                    builder.append_null();
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let mut builder = arrow::array::TimestampNanosecondBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    // Parse various timestamp formats. A value that fails every
                    // parse is a genuine null — never epoch 0 — so that an
                    // unparseable timestamp is not silently materialised as
                    // 1970-01-01.
                    let parsed = if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(value) {
                        dt.timestamp_nanos_opt()
                    } else if let Ok(ndt) =
                        chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
                    {
                        ndt.and_utc().timestamp_nanos_opt()
                    } else if let Ok(ndt) =
                        chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                    {
                        ndt.and_utc().timestamp_nanos_opt()
                    } else if let Ok(nd) = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d") {
                        nd.and_hms_opt(0, 0, 0)
                            .and_then(|dt| dt.and_utc().timestamp_nanos_opt())
                    } else {
                        None
                    };
                    match parsed {
                        Some(nanos) => builder.append_value(nanos),
                        None => builder.append_null(),
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Utf8 => {
            let mut builder = arrow::array::StringBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    builder.append_value(value);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            // Default to string for unsupported types
            let mut builder = arrow::array::StringBuilder::new();
            for value in slice {
                if value.is_empty() {
                    builder.append_null();
                } else {
                    builder.append_value(value);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

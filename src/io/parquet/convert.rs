//! Arrow RecordBatch -> DataFrame value conversion (per-cell, never per-array).

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;
use arrow::array::{
    Array, BooleanArray, Date32Array, Date64Array, Decimal128Array, Decimal256Array,
    LargeStringArray, PrimitiveArray, StringArray, UInt64Array,
};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
    Int8Type, SchemaRef, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt16Type, UInt32Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;

/// Collect one Arrow primitive column whose native type widens losslessly
/// into `i64` — every signed/unsigned integer width up to 32 bits, plus i64
/// itself — into a native `Vec<i64>`. This is how Parquet-native Int8/16/32
/// and UInt8/16/32 columns (which pandas/pyarrow write by default, but
/// pandrs itself never does) become real, typed pandrs columns instead of
/// falling into a fallback that used to format the *entire array* as the
/// text of every single cell.
pub(super) fn collect_i64_column<T>(
    batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
) -> Result<Vec<i64>>
where
    T: ArrowPrimitiveType,
    T::Native: Into<i64>,
{
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                Error::Cast(format!(
                    "Failed to cast column '{}' to a primitive integer array",
                    col_name
                ))
            })?;
        for i in 0..array.len() {
            values.push(if array.is_null(i) {
                0
            } else {
                array.value(i).into()
            });
        }
    }
    Ok(values)
}

/// Collect a UInt64 column as `i64`, saturating at `i64::MAX` for values
/// that don't fit. pandrs has no native unsigned column type, and a plain
/// `as i64` cast would wrap out-of-range values to negative numbers, which
/// is a worse corruption than clamping to the largest representable value.
pub(super) fn collect_u64_saturating_column(
    batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                Error::Cast(format!(
                    "Failed to cast column '{}' to UInt64Array",
                    col_name
                ))
            })?;
        for i in 0..array.len() {
            values.push(if array.is_null(i) {
                0
            } else {
                i64::try_from(array.value(i)).unwrap_or(i64::MAX)
            });
        }
    }
    Ok(values)
}

/// Collect one Arrow primitive column whose native type widens losslessly
/// into `f64` (Float32, Float64) into a native `Vec<f64>`.
pub(super) fn collect_f64_column<T>(
    batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
) -> Result<Vec<f64>>
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64>,
{
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                Error::Cast(format!(
                    "Failed to cast column '{}' to a primitive float array",
                    col_name
                ))
            })?;
        for i in 0..array.len() {
            values.push(if array.is_null(i) {
                f64::NAN
            } else {
                array.value(i).into()
            });
        }
    }
    Ok(values)
}

/// Collect one Arrow timestamp column — whose native representation is
/// always `i64` regardless of time unit — into datetime strings via
/// `format`, which already encodes the unit (seconds/millis/micros/nanos).
pub(super) fn collect_timestamp_column<T>(
    batches: &[RecordBatch],
    col_idx: usize,
    col_name: &str,
    format: impl Fn(i64) -> String,
) -> Result<Vec<String>>
where
    T: ArrowPrimitiveType<Native = i64>,
{
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                Error::Cast(format!(
                    "Failed to cast column '{}' to a timestamp array",
                    col_name
                ))
            })?;
        for i in 0..array.len() {
            values.push(if array.is_null(i) {
                "1970-01-01T00:00:00".to_string()
            } else {
                format(array.value(i))
            });
        }
    }
    Ok(values)
}

/// Format a DATE32 value (days since the Unix epoch) as an ISO-8601 date.
pub(super) fn format_date32(days: i32) -> String {
    match chrono::DateTime::from_timestamp(days as i64 * 86_400, 0) {
        Some(dt) => dt.format("%Y-%m-%d").to_string(),
        None => format!("invalid-date32({days})"),
    }
}

/// Format a DATE64 value (milliseconds since the Unix epoch, per the Arrow
/// spec) as an ISO-8601 date.
pub(super) fn format_date64(millis: i64) -> String {
    match chrono::DateTime::from_timestamp_millis(millis) {
        Some(dt) => dt.format("%Y-%m-%d").to_string(),
        None => format!("invalid-date64({millis})"),
    }
}

pub(super) fn format_timestamp_seconds(secs: i64) -> String {
    match chrono::DateTime::from_timestamp(secs, 0) {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S").to_string(),
        None => format!("invalid-timestamp({secs})"),
    }
}

pub(super) fn format_timestamp_millis(millis: i64) -> String {
    match chrono::DateTime::from_timestamp_millis(millis) {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
        None => format!("invalid-timestamp({millis})"),
    }
}

pub(super) fn format_timestamp_micros(micros: i64) -> String {
    match chrono::DateTime::from_timestamp_micros(micros) {
        Some(dt) => dt.format("%Y-%m-%d %H:%M:%S%.6f").to_string(),
        None => format!("invalid-timestamp({micros})"),
    }
}

pub(super) fn format_timestamp_nanos(nanos: i64) -> String {
    chrono::DateTime::from_timestamp_nanos(nanos)
        .format("%Y-%m-%d %H:%M:%S%.9f")
        .to_string()
}

/// Build a `Series<T>` from already-collected values and add it to `df`.
pub(super) fn push_column<T: 'static + std::fmt::Debug + Clone + Send + Sync>(
    df: &mut DataFrame,
    col_name: &str,
    values: Vec<T>,
) -> Result<()> {
    let series = Series::new(values, Some(col_name.to_string()))?;
    df.add_column(col_name.to_string(), series)
}

/// Convert Arrow record batches to a DataFrame.
///
/// Every numeric width Arrow actually produces for Parquet files written by
/// other tools (pandas/pyarrow defaults include Int32, Float32 and various
/// unsigned/decimal/date64 columns that pandrs itself never writes but must
/// still be able to read) is converted through `array.value(i)` into a real,
/// natively-typed pandrs column — never by formatting a whole Arrow array as
/// the "value" of a single cell, which used to both corrupt every cell of
/// such a column and make the read `O(rows^2)`.
pub(super) fn record_batches_to_dataframe(
    batches: &[RecordBatch],
    schema: SchemaRef,
) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col_name = field.name().as_str();

        match field.data_type() {
            DataType::Int8 => {
                let values = collect_i64_column::<Int8Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Int16 => {
                let values = collect_i64_column::<Int16Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Int32 => {
                let values = collect_i64_column::<Int32Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Int64 => {
                let values = collect_i64_column::<Int64Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::UInt8 => {
                let values = collect_i64_column::<UInt8Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::UInt16 => {
                let values = collect_i64_column::<UInt16Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::UInt32 => {
                let values = collect_i64_column::<UInt32Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::UInt64 => {
                let values = collect_u64_saturating_column(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Float32 => {
                let values = collect_f64_column::<Float32Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Float64 => {
                let values = collect_f64_column::<Float64Type>(batches, col_idx, col_name)?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Boolean => {
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to BooleanArray",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(!array.is_null(i) && array.value(i));
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::Utf8 => {
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to StringArray",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(if array.is_null(i) {
                            String::new()
                        } else {
                            array.value(i).to_string()
                        });
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::LargeUtf8 => {
                // `LargeUtf8` is `GenericStringArray<i64>` (`LargeStringArray`),
                // a distinct Rust type from `StringArray`
                // (`GenericStringArray<i32>`, for plain `Utf8`); downcasting a
                // LargeUtf8 column to `StringArray` always returned `None`,
                // turning every large-string column into a cast error.
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to LargeStringArray",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(if array.is_null(i) {
                            String::new()
                        } else {
                            array.value(i).to_string()
                        });
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::Date32 => {
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<Date32Array>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to Date32Array",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(if array.is_null(i) {
                            "1970-01-01".to_string()
                        } else {
                            format_date32(array.value(i))
                        });
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::Date64 => {
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<Date64Array>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to Date64Array",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(if array.is_null(i) {
                            "1970-01-01".to_string()
                        } else {
                            format_date64(array.value(i))
                        });
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                let values = collect_timestamp_column::<TimestampSecondType>(
                    batches,
                    col_idx,
                    col_name,
                    format_timestamp_seconds,
                )?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                let values = collect_timestamp_column::<TimestampMillisecondType>(
                    batches,
                    col_idx,
                    col_name,
                    format_timestamp_millis,
                )?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let values = collect_timestamp_column::<TimestampMicrosecondType>(
                    batches,
                    col_idx,
                    col_name,
                    format_timestamp_micros,
                )?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let values = collect_timestamp_column::<TimestampNanosecondType>(
                    batches,
                    col_idx,
                    col_name,
                    format_timestamp_nanos,
                )?;
                push_column(&mut df, col_name, values)?;
            }
            DataType::Decimal128(_, scale) => {
                let scale = *scale;
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to Decimal128Array",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        values.push(if array.is_null(i) {
                            f64::NAN
                        } else {
                            array.value(i) as f64 / 10f64.powi(scale as i32)
                        });
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            DataType::Decimal256(_, scale) => {
                let scale = *scale;
                let mut values = Vec::new();
                for batch in batches {
                    let array = batch
                        .column(col_idx)
                        .as_any()
                        .downcast_ref::<Decimal256Array>()
                        .ok_or_else(|| {
                            Error::Cast(format!(
                                "Failed to cast column '{}' to Decimal256Array",
                                col_name
                            ))
                        })?;
                    for i in 0..array.len() {
                        if array.is_null(i) {
                            values.push(f64::NAN);
                            continue;
                        }
                        let raw = array.value(i);
                        let base = raw
                            .to_i128()
                            .map(|v| v as f64)
                            .unwrap_or_else(|| raw.to_string().parse::<f64>().unwrap_or(f64::NAN));
                        values.push(base / 10f64.powi(scale as i32));
                    }
                }
                push_column(&mut df, col_name, values)?;
            }
            other => {
                return Err(Error::NotImplemented(format!(
                    "Reading Parquet column '{}' with Arrow type {:?} is not yet supported",
                    col_name, other
                )));
            }
        }
    }

    Ok(df)
}

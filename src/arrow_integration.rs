//! # Apache Arrow Integration
//!
//! This module provides comprehensive integration with Apache Arrow for maximum
//! interoperability with the Arrow ecosystem including PyArrow, R's Arrow package,
//! and other Arrow-based tools.
//!
//! ## Copy behaviour
//!
//! Conversions in this module are **not** zero-copy. [`ArrowConverter::dataframe_to_record_batch`](crate::arrow_integration::ArrowConverter::dataframe_to_record_batch)
//! copies every element into a freshly allocated Arrow array, and
//! [`ArrowConverter::record_batch_to_dataframe`](crate::arrow_integration::ArrowConverter::record_batch_to_dataframe) (via `ArrowConverter::arrow_array_to_series`)
//! copies every Arrow value into a freshly allocated `String` per cell before
//! it reaches a pandrs `Series<String>`. This is a data-copying bridge, not a
//! shared-buffer view; budget for it accordingly on large datasets.
//!
//! (Handoff note: `crate::lib`'s module doc for this module still advertises
//! "Zero-copy data exchange" as a bullet point. That line lives in `lib.rs`,
//! owned by a different area of this codebase, so it is not edited from
//! here -- flagging it for whoever owns `lib.rs` to correct.)

#[cfg(feature = "distributed")]
use arrow::{
    array::{
        Array, ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array,
        Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        LargeBinaryArray, LargeStringArray, StringArray, Time32SecondArray, Time64MicrosecondArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    compute,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::base::Series;
use std::sync::Arc;

/// Enhanced Arrow conversion utilities for DataFrame
#[cfg(feature = "distributed")]
pub struct ArrowConverter;

#[cfg(feature = "distributed")]
impl ArrowConverter {
    /// Convert DataFrame to Arrow RecordBatch with enhanced type inference
    pub fn dataframe_to_record_batch(df: &DataFrame) -> Result<RecordBatch> {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        let column_names = df.column_names();

        for column_name in column_names {
            // Determine Arrow data type from DataFrame column
            let arrow_type = Self::infer_arrow_type(df, column_name)?;
            // `nullable: false`: every element type `series_to_arrow_array`
            // can build (i64/i32/f64/f32/bool/String) comes from pandrs'
            // `Series<T>`, a plain `Vec<T>` with no null-tracking of any
            // kind -- there is no way for the resulting array to contain a
            // null, so declaring `nullable: true` here would assert a
            // possibility the data can never actually exhibit.
            let field = Field::new(column_name.as_str(), arrow_type.clone(), false);
            fields.push(field);

            // Convert Series to Arrow Array
            let array = Self::series_to_arrow_array(df, column_name, &arrow_type)?;
            arrays.push(array);
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| Error::InvalidOperation(format!("Failed to create RecordBatch: {}", e)))
    }

    /// Convert multiple DataFrames to Arrow RecordBatch stream
    pub fn dataframes_to_record_batches(
        dataframes: &[DataFrame],
        batch_size: Option<usize>,
    ) -> Result<Vec<RecordBatch>> {
        let mut batches = Vec::new();
        // `Some(0)` used to slip past `unwrap_or` unchanged (it's a `None`
        // fallback, not a validity check), so `batch_size` could reach the
        // ceiling-division below as 0 and panic on integer division by
        // zero for any non-empty DataFrame. Reject it explicitly instead.
        let batch_size = match batch_size {
            Some(0) => {
                return Err(Error::InvalidValue(
                    "batch_size must be greater than zero".to_string(),
                ))
            }
            Some(n) => n,
            None => 1024,
        };

        for df in dataframes {
            if df.row_count() <= batch_size {
                // Small DataFrame - single batch
                batches.push(Self::dataframe_to_record_batch(df)?);
            } else {
                // Large DataFrame - split into batches. Manual ceiling
                // division (not `usize::div_ceil`, stabilised in Rust 1.73)
                // to stay within the crate's declared 1.70 MSRV, matching
                // the convention already established at
                // `storage::unified_column_store::encoding`.
                let num_batches = (df.row_count() + batch_size - 1) / batch_size;
                for i in 0..num_batches {
                    let start = i * batch_size;
                    let end = std::cmp::min(start + batch_size, df.row_count());

                    // Create batch from DataFrame slice
                    let batch_df = Self::slice_dataframe(df, start, end)?;
                    batches.push(Self::dataframe_to_record_batch(&batch_df)?);
                }
            }
        }

        Ok(batches)
    }

    /// Convert Arrow RecordBatch to DataFrame.
    ///
    /// Iterates `schema.fields()` in order and adds each column directly,
    /// instead of staging columns through a `HashMap<String, _>` keyed by
    /// field name first. Arrow schemas may legally contain duplicate field
    /// names; staging through a name-keyed map let the later occurrence
    /// silently overwrite the earlier one before `add_column` ever saw a
    /// duplicate, discarding one column's data with no error. Adding
    /// directly lets `DataFrame::add_column`'s existing duplicate-name
    /// check (`Error::DuplicateColumnName`) do its job.
    pub fn record_batch_to_dataframe(batch: &RecordBatch) -> Result<DataFrame> {
        let schema = batch.schema();
        let mut df = DataFrame::new();

        for (i, field) in schema.fields().iter().enumerate() {
            let column_name = field.name().clone();
            let array = batch.column(i);
            let series = Self::arrow_array_to_series(array, &column_name)?;
            df.add_column(column_name, series)?;
        }

        Ok(df)
    }

    /// Infer the Arrow data type of a column from its *actual* stored element
    /// type (not from the column name).
    fn infer_arrow_type(df: &DataFrame, column_name: &str) -> Result<DataType> {
        if df.get_column::<i64>(column_name).is_ok() {
            Ok(DataType::Int64)
        } else if df.get_column::<i32>(column_name).is_ok() {
            Ok(DataType::Int32)
        } else if df.get_column::<f64>(column_name).is_ok() {
            Ok(DataType::Float64)
        } else if df.get_column::<f32>(column_name).is_ok() {
            Ok(DataType::Float32)
        } else if df.get_column::<bool>(column_name).is_ok() {
            Ok(DataType::Boolean)
        } else if df.get_column::<String>(column_name).is_ok() {
            Ok(DataType::Utf8)
        } else if df.contains_column(column_name) {
            // The column exists but its element type has no direct Arrow mapping.
            Err(Error::NotImplemented(format!(
                "Column '{}' has an element type that cannot be mapped to an Arrow type",
                column_name
            )))
        } else {
            Err(Error::ColumnNotFound(column_name.to_string()))
        }
    }

    /// Build an Arrow array from the *actual* data of a DataFrame column.
    ///
    /// The `arrow_type` is expected to come from [`Self::infer_arrow_type`],
    /// which derives it from the column's real stored type. Columns whose
    /// element type has no Arrow mapping yield [`Error::NotImplemented`] rather
    /// than fabricated values.
    fn series_to_arrow_array(
        df: &DataFrame,
        column_name: &str,
        arrow_type: &DataType,
    ) -> Result<ArrayRef> {
        match arrow_type {
            DataType::Int64 => {
                let series = df.get_column::<i64>(column_name)?;
                Ok(Arc::new(Int64Array::from(series.values().to_vec())))
            }
            DataType::Int32 => {
                let series = df.get_column::<i32>(column_name)?;
                Ok(Arc::new(Int32Array::from(series.values().to_vec())))
            }
            DataType::Float64 => {
                let series = df.get_column::<f64>(column_name)?;
                Ok(Arc::new(Float64Array::from(series.values().to_vec())))
            }
            DataType::Float32 => {
                let series = df.get_column::<f32>(column_name)?;
                Ok(Arc::new(Float32Array::from(series.values().to_vec())))
            }
            DataType::Boolean => {
                let series = df.get_column::<bool>(column_name)?;
                Ok(Arc::new(BooleanArray::from(series.values().to_vec())))
            }
            DataType::Utf8 => {
                let values = df.get_column_string_values(column_name)?;
                Ok(Arc::new(StringArray::from_iter_values(values)))
            }
            other => Err(Error::NotImplemented(format!(
                "Building an Arrow array of type {:?} from DataFrame column '{}' is not supported",
                other, column_name
            ))),
        }
    }

    /// Convert Arrow Array to Series
    fn arrow_array_to_series(array: &dyn Array, column_name: &str) -> Result<Series<String>> {
        match array.data_type() {
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    Error::InvalidOperation("Failed to downcast to Int64Array".to_string())
                })?;

                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();

                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Float64Array".to_string())
                    })?;

                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();

                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to StringArray".to_string())
                    })?;

                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();

                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to BooleanArray".to_string())
                    })?;

                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();

                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Int8 => {
                let arr = array.as_any().downcast_ref::<Int8Array>().ok_or_else(|| {
                    Error::InvalidOperation("Failed to downcast to Int8Array".into())
                })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Int16 => {
                let arr = array.as_any().downcast_ref::<Int16Array>().ok_or_else(|| {
                    Error::InvalidOperation("Failed to downcast to Int16Array".into())
                })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                    Error::InvalidOperation("Failed to downcast to Int32Array".into())
                })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::UInt8 => {
                let arr = array.as_any().downcast_ref::<UInt8Array>().ok_or_else(|| {
                    Error::InvalidOperation("Failed to downcast to UInt8Array".into())
                })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::UInt16 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to UInt16Array".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::UInt32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to UInt32Array".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::UInt64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to UInt64Array".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Float32Array".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::LargeUtf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to LargeStringArray".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value(i).to_string()
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Binary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to BinaryArray".into())
                    })?;
                // Hex-encode rather than `{:?}`-format the raw bytes:
                // `format!("{:?}", &[104u8, 105])` produces the literal
                // text "[104, 105]", which is neither a standard binary
                // encoding nor round-trippable by any parser.
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            Self::bytes_to_hex(arr.value(i))
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Date32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Date32Array".into())
                    })?;
                // `arr.value(i)` is the raw day-count-since-epoch i32; use
                // Arrow's own calendar conversion instead of stringifying
                // that count directly (which would print e.g. "1" for
                // 1970-01-02, looking like data but not being a date).
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            Ok("null".to_string())
                        } else {
                            arr.value_as_date(i)
                                .map(|d| d.format("%Y-%m-%d").to_string())
                                .ok_or_else(|| {
                                    Error::Computation(format!(
                                        "Arrow Date32 value at row {} of column '{}' is out of the representable date range",
                                        i, column_name
                                    ))
                                })
                        }
                    })
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Date64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Date64Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Date64Array".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            Ok("null".to_string())
                        } else {
                            arr.value_as_date(i)
                                .map(|d| d.format("%Y-%m-%d").to_string())
                                .ok_or_else(|| {
                                    Error::Computation(format!(
                                        "Arrow Date64 value at row {} of column '{}' is out of the representable date range",
                                        i, column_name
                                    ))
                                })
                        }
                    })
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation(
                            "Failed to downcast to TimestampMicrosecondArray".into(),
                        )
                    })?;
                // Format as ISO-8601 via Arrow's own resolution-aware
                // conversion rather than stringifying the raw epoch
                // integer, which prints a huge, unreadable microsecond
                // count instead of a timestamp.
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        Self::format_timestamp_cell(arr, i, column_name, "TimestampMicrosecond")
                    })
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to TimestampSecondArray".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| Self::format_timestamp_cell(arr, i, column_name, "TimestampSecond"))
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation(
                            "Failed to downcast to TimestampMillisecondArray".into(),
                        )
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        Self::format_timestamp_cell(arr, i, column_name, "TimestampMillisecond")
                    })
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation(
                            "Failed to downcast to TimestampNanosecondArray".into(),
                        )
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        Self::format_timestamp_cell(arr, i, column_name, "TimestampNanosecond")
                    })
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Time32(TimeUnit::Second) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Time32SecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Time32SecondArray".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| Self::format_time_cell(arr, i, column_name, "Time32Second"))
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Time64MicrosecondArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation(
                            "Failed to downcast to Time64MicrosecondArray".into(),
                        )
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| Self::format_time_cell(arr, i, column_name, "Time64Microsecond"))
                    .collect::<Result<Vec<String>>>()?;
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::Decimal128(_, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to Decimal128Array".into())
                    })?;
                // `Decimal128Array` stores the UNSCALED i128 (123.45 at
                // scale 2 is stored as the integer 12345); `arr.value(i)`
                // returns that raw integer, so stringifying it directly
                // silently produces "12345" instead of "123.45" -- a 100x
                // error for any nonzero scale. `value_as_string` applies
                // the array's own scale correctly.
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            arr.value_as_string(i)
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            DataType::LargeBinary => {
                let arr = array
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .ok_or_else(|| {
                        Error::InvalidOperation("Failed to downcast to LargeBinaryArray".into())
                    })?;
                let values: Vec<String> = (0..arr.len())
                    .map(|i| {
                        if arr.is_null(i) {
                            "null".to_string()
                        } else {
                            Self::bytes_to_hex(arr.value(i))
                        }
                    })
                    .collect();
                Series::new(values, Some(column_name.to_string()))
            }
            _ => Err(Error::NotImplemented(format!(
                "Arrow type {:?} conversion not implemented",
                array.data_type()
            ))),
        }
    }

    /// Encode bytes as a lowercase hex string (`[0x10, 0xff]` -> `"10ff"`).
    ///
    /// Used for Arrow `Binary`/`LargeBinary` columns instead of Rust's
    /// `Debug` byte-list formatting (`"[16, 255]"`), which is neither a
    /// standard binary encoding nor round-trippable by any parser.
    fn bytes_to_hex(bytes: &[u8]) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            // `write!` to a `String` never fails.
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    /// Format one cell of a timestamp array as an ISO-8601 string
    /// (`YYYY-MM-DDTHH:MM:SS[.fraction]`), using Arrow's own
    /// resolution-aware `value_as_datetime` conversion instead of
    /// stringifying the raw epoch integer (which prints a number that
    /// looks like data but isn't a readable timestamp).
    fn format_timestamp_cell<T>(
        arr: &arrow::array::PrimitiveArray<T>,
        i: usize,
        column_name: &str,
        arrow_type_name: &str,
    ) -> Result<String>
    where
        T: arrow::datatypes::ArrowTemporalType,
        i64: From<T::Native>,
    {
        if arr.is_null(i) {
            return Ok("null".to_string());
        }
        arr.value_as_datetime(i)
            .map(|dt| dt.format("%Y-%m-%dT%H:%M:%S%.f").to_string())
            .ok_or_else(|| {
                Error::Computation(format!(
                    "Arrow {} value at row {} of column '{}' is out of the representable datetime range",
                    arrow_type_name, i, column_name
                ))
            })
    }

    /// Format one cell of a time-of-day array as an ISO-8601 time string
    /// (`HH:MM:SS[.fraction]`), using Arrow's own `value_as_time`
    /// conversion instead of stringifying the raw integer offset.
    fn format_time_cell<T>(
        arr: &arrow::array::PrimitiveArray<T>,
        i: usize,
        column_name: &str,
        arrow_type_name: &str,
    ) -> Result<String>
    where
        T: arrow::datatypes::ArrowTemporalType,
        i64: From<T::Native>,
    {
        if arr.is_null(i) {
            return Ok("null".to_string());
        }
        arr.value_as_time(i)
            .map(|t| t.format("%H:%M:%S%.f").to_string())
            .ok_or_else(|| {
                Error::Computation(format!(
                    "Arrow {} value at row {} of column '{}' is out of the representable time range",
                    arrow_type_name, i, column_name
                ))
            })
    }

    /// Slice rows `[start, end)` out of a DataFrame for batching, preserving each
    /// column's real values and element type.
    fn slice_dataframe(df: &DataFrame, start: usize, end: usize) -> Result<DataFrame> {
        let mut result_df = DataFrame::new();

        for column_name in df.column_names() {
            if let Ok(series) = df.get_column::<i64>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(series) = df.get_column::<i32>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(series) = df.get_column::<f64>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(series) = df.get_column::<f32>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(series) = df.get_column::<bool>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(series) = df.get_column::<String>(column_name) {
                let sliced = Self::slice_vec(series.values(), start, end);
                result_df.add_column(
                    column_name.to_string(),
                    Series::new(sliced, Some(column_name.to_string()))?,
                )?;
            } else {
                return Err(Error::NotImplemented(format!(
                    "Slicing column '{}' is not supported for its element type",
                    column_name
                )));
            }
        }

        Ok(result_df)
    }

    /// Clone the `[start, end)` slice of a value slice, clamping the bounds.
    fn slice_vec<T: Clone>(values: &[T], start: usize, end: usize) -> Vec<T> {
        let start = start.min(values.len());
        let end = end.min(values.len());
        if start >= end {
            Vec::new()
        } else {
            values[start..end].to_vec()
        }
    }

    /// Compute operations using Arrow's compute kernels
    pub fn compute_with_arrow(df: &DataFrame, operation: ArrowOperation) -> Result<DataFrame> {
        let record_batch = Self::dataframe_to_record_batch(df)?;

        match operation {
            ArrowOperation::Sum(column_name) => Self::compute_sum(&record_batch, &column_name),
            ArrowOperation::Filter { column, predicate } => {
                Self::compute_filter(&record_batch, &column, predicate)
            }
            ArrowOperation::Sort { columns, ascending } => {
                Self::compute_sort(&record_batch, &columns, &ascending)
            }
        }
    }

    /// Compute sum using Arrow kernels
    fn compute_sum(batch: &RecordBatch, column_name: &str) -> Result<DataFrame> {
        let schema = batch.schema();
        let column_index = schema
            .fields()
            .iter()
            .position(|f| f.name() == column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let array = batch.column(column_index);

        match array.data_type() {
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::TypeMismatch("expected Int64Array".into()))?;
                // `compute::sum` wraps silently on overflow (Rust release-mode
                // integer semantics); `sum_checked` detects it and errors
                // instead of returning a wrapped-around number that looks
                // like a plausible sum. `Ok(None)` means the array is empty
                // or entirely null -- the identity element (0) is the
                // correct, non-fabricated sum for that case (matching
                // pandas' `Series([]).sum() == 0`), not an error.
                let sum = compute::sum_checked(arr)
                    .map_err(|e| {
                        Error::Computation(format!(
                            "Sum of column '{}' overflowed i64: {}",
                            column_name, e
                        ))
                    })?
                    .unwrap_or(0);

                // Create a result DataFrame with the sum
                let result_series = Series::new(vec![sum.to_string()], Some("sum".to_string()))?;

                let mut result_df = DataFrame::new();
                result_df.add_column("sum".to_string(), result_series)?;
                Ok(result_df)
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::TypeMismatch("expected Float64Array".into()))?;
                let sum = compute::sum_checked(arr)
                    .map_err(|e| {
                        Error::Computation(format!(
                            "Sum of column '{}' overflowed f64: {}",
                            column_name, e
                        ))
                    })?
                    .unwrap_or(0.0);

                let result_series = Series::new(vec![sum.to_string()], Some("sum".to_string()))?;

                let mut result_df = DataFrame::new();
                result_df.add_column("sum".to_string(), result_series)?;
                Ok(result_df)
            }
            _ => Err(Error::InvalidOperation(format!(
                "Sum not supported for type {:?}",
                array.data_type()
            ))),
        }
    }

    /// Compute filter using Arrow kernels (`arrow_select::filter`).
    ///
    /// `GreaterThan`/`LessThan` require a numeric column (any of the ones
    /// `dataframe_to_record_batch` can produce -- Int64/Int32/Float64/Float32
    /// -- are cast to Float64 for a uniform comparison against the `f64`
    /// threshold). `EqualTo`/`NotEqualTo` require a `Utf8` column, matching
    /// their `String` payload. A predicate/column type mismatch is a clear
    /// error rather than an undefined comparison.
    fn compute_filter(
        batch: &RecordBatch,
        column: &str,
        predicate: FilterPredicate,
    ) -> Result<DataFrame> {
        let schema = batch.schema();
        let column_index = schema
            .fields()
            .iter()
            .position(|f| f.name() == column)
            .ok_or_else(|| Error::ColumnNotFound(column.to_string()))?;
        let array = batch.column(column_index);

        let numeric_mask = |threshold: f64, greater: bool| -> Result<BooleanArray> {
            if !array.data_type().is_numeric() {
                return Err(Error::InvalidOperation(format!(
                    "Filter predicate requires a numeric column but '{}' is {:?}",
                    column,
                    array.data_type()
                )));
            }
            let as_f64 = compute::cast(array.as_ref(), &DataType::Float64).map_err(|e| {
                Error::Computation(format!(
                    "Failed to cast column '{}' to Float64 for filtering: {}",
                    column, e
                ))
            })?;
            let arr = as_f64
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::TypeMismatch("expected Float64Array after cast".into()))?;
            let scalar = Float64Array::new_scalar(threshold);
            let result = if greater {
                compute::kernels::cmp::gt(arr, &scalar)
            } else {
                compute::kernels::cmp::lt(arr, &scalar)
            };
            result.map_err(|e| Error::Computation(format!("Arrow comparison failed: {}", e)))
        };

        let string_mask = |value: &str, equal: bool| -> Result<BooleanArray> {
            if array.data_type() != &DataType::Utf8 {
                return Err(Error::InvalidOperation(format!(
                    "Filter predicate requires a Utf8 column but '{}' is {:?}",
                    column,
                    array.data_type()
                )));
            }
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| Error::TypeMismatch("expected StringArray".into()))?;
            let scalar = StringArray::new_scalar(value);
            let result = if equal {
                compute::kernels::cmp::eq(arr, &scalar)
            } else {
                compute::kernels::cmp::neq(arr, &scalar)
            };
            result.map_err(|e| Error::Computation(format!("Arrow comparison failed: {}", e)))
        };

        let mask = match &predicate {
            FilterPredicate::GreaterThan(threshold) => numeric_mask(*threshold, true)?,
            FilterPredicate::LessThan(threshold) => numeric_mask(*threshold, false)?,
            FilterPredicate::EqualTo(value) => string_mask(value, true)?,
            FilterPredicate::NotEqualTo(value) => string_mask(value, false)?,
        };

        let filtered = compute::filter_record_batch(batch, &mask)
            .map_err(|e| Error::Computation(format!("Arrow filter failed: {}", e)))?;

        Self::record_batch_to_dataframe(&filtered)
    }

    /// Compute sort using Arrow kernels (`arrow_ord::sort` + `arrow_select::take`).
    fn compute_sort(
        batch: &RecordBatch,
        columns: &[String],
        ascending: &[bool],
    ) -> Result<DataFrame> {
        if columns.is_empty() {
            return Err(Error::InvalidOperation(
                "Sort requires at least one column".to_string(),
            ));
        }
        if columns.len() != ascending.len() {
            return Err(Error::InvalidOperation(format!(
                "Sort columns ({}) and ascending flags ({}) must have the same length",
                columns.len(),
                ascending.len()
            )));
        }

        let schema = batch.schema();
        let mut sort_columns: Vec<compute::SortColumn> = Vec::with_capacity(columns.len());
        for (col_name, &asc) in columns.iter().zip(ascending.iter()) {
            let idx = schema
                .fields()
                .iter()
                .position(|f| f.name() == col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.clone()))?;
            sort_columns.push(compute::SortColumn {
                values: batch.column(idx).clone(),
                options: Some(compute::SortOptions {
                    descending: !asc,
                    nulls_first: false,
                }),
            });
        }

        let indices = compute::lexsort_to_indices(&sort_columns, None)
            .map_err(|e| Error::Computation(format!("Arrow sort failed: {}", e)))?;

        let sorted = compute::take_record_batch(batch, &indices).map_err(|e| {
            Error::Computation(format!(
                "Arrow take (reordering by sort indices) failed: {}",
                e
            ))
        })?;

        Self::record_batch_to_dataframe(&sorted)
    }
}

/// Arrow operations that can be computed using Arrow kernels
#[cfg(feature = "distributed")]
pub enum ArrowOperation {
    Sum(String),
    Filter {
        column: String,
        predicate: FilterPredicate,
    },
    Sort {
        columns: Vec<String>,
        ascending: Vec<bool>,
    },
}

/// Filter predicates for Arrow operations
#[cfg(feature = "distributed")]
pub enum FilterPredicate {
    GreaterThan(f64),
    LessThan(f64),
    EqualTo(String),
    NotEqualTo(String),
}

/// Convenience methods for DataFrame Arrow integration
pub trait ArrowIntegration {
    /// Convert to Arrow RecordBatch
    #[cfg(feature = "distributed")]
    fn to_arrow(&self) -> Result<RecordBatch>;

    /// Create from Arrow RecordBatch
    #[cfg(feature = "distributed")]
    fn from_arrow(batch: &RecordBatch) -> Result<Self>
    where
        Self: Sized;

    /// Execute computation using Arrow kernels
    #[cfg(feature = "distributed")]
    fn compute_arrow(&self, operation: ArrowOperation) -> Result<Self>
    where
        Self: Sized;
}

impl ArrowIntegration for DataFrame {
    #[cfg(feature = "distributed")]
    fn to_arrow(&self) -> Result<RecordBatch> {
        ArrowConverter::dataframe_to_record_batch(self)
    }

    #[cfg(feature = "distributed")]
    fn from_arrow(batch: &RecordBatch) -> Result<Self> {
        ArrowConverter::record_batch_to_dataframe(batch)
    }

    #[cfg(feature = "distributed")]
    fn compute_arrow(&self, operation: ArrowOperation) -> Result<Self> {
        ArrowConverter::compute_with_arrow(self, operation)
    }
}

/// Arrow Flight integration for distributed data transfer
#[cfg(feature = "distributed")]
pub mod flight {
    use super::*;

    pub struct FlightConnector {
        endpoint: String,
    }

    impl FlightConnector {
        pub fn new(endpoint: String) -> Self {
            Self { endpoint }
        }

        /// Send DataFrame via Arrow Flight.
        ///
        /// A real implementation requires an Arrow Flight client/server
        /// round trip (see [`Self::receive_dataframe`], which already takes
        /// this honest path). Printing a "sending..." message and returning
        /// `Ok(())` without ever transmitting `df` anywhere would silently
        /// lose the data while claiming success, so the missing capability
        /// is reported honestly instead -- matching `receive_dataframe`.
        pub async fn send_dataframe(&self, df: &DataFrame, path: &str) -> Result<()> {
            // Still validate that the DataFrame is convertible to Arrow, so
            // callers get a meaningful conversion error even though the
            // transport itself isn't implemented.
            let _record_batch = df.to_arrow()?;

            Err(Error::NotImplemented(format!(
                "Arrow Flight send_dataframe is not implemented (endpoint '{}', path '{}')",
                self.endpoint, path
            )))
        }

        /// Receive DataFrame via Arrow Flight
        pub async fn receive_dataframe(&self, path: &str) -> Result<DataFrame> {
            // A real implementation requires an Arrow Flight client/server round
            // trip. Returning a fabricated DataFrame would be misleading, so the
            // missing capability is reported honestly.
            Err(Error::NotImplemented(format!(
                "Arrow Flight receive_dataframe is not implemented (endpoint '{}', path '{}')",
                self.endpoint, path
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "distributed")]
    fn test_arrow_integration() {
        // Create a test DataFrame
        let series1 = Series::new(
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
            Some("numbers".to_string()),
        )
        .expect("operation should succeed");
        let series2 = Series::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            Some("letters".to_string()),
        )
        .expect("operation should succeed");

        let mut df = DataFrame::new();
        df.add_column("numbers".to_string(), series1)
            .expect("operation should succeed");
        df.add_column("letters".to_string(), series2)
            .expect("operation should succeed");

        // Test conversion to Arrow
        let record_batch = df.to_arrow().expect("operation should succeed");
        assert_eq!(record_batch.num_columns(), 2);
        assert_eq!(record_batch.num_rows(), 3);

        // Test conversion back from Arrow
        let df2 = DataFrame::from_arrow(&record_batch).expect("operation should succeed");
        assert_eq!(df2.column_names(), df.column_names());
    }

    #[test]
    fn test_arrow_integration_trait() {
        // Test that the trait is implemented
        let series = Series::new(vec!["test".to_string()], Some("col".to_string()))
            .expect("operation should succeed");

        let mut df = DataFrame::new();
        df.add_column("col".to_string(), series)
            .expect("operation should succeed");

        // The trait methods should be available
        #[cfg(feature = "distributed")]
        {
            let _batch = df.to_arrow();
        }
    }

    // -- Regression tests (wave 3) -------------------------------------
    //
    // These live in-crate (rather than under `tests/`) because they build
    // `arrow` array/schema values directly, and `arrow` is an optional
    // dependency of `pandrs` itself, not a `[dev-dependencies]` entry --
    // an external integration test crate under `tests/` cannot `use
    // arrow::...` at all. Gated identically to the rest of this file.

    #[test]
    #[cfg(feature = "distributed")]
    fn test_decimal128_round_trip_preserves_scale() {
        // 12345 unscaled at scale 2 represents 123.45; stringifying the
        // raw i128 directly (the old behavior) printed "12345" -- a 100x
        // financial error.
        let arr = Decimal128Array::from(vec![12345i128, -12345i128])
            .with_precision_and_scale(10, 2)
            .expect("valid precision/scale");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "amount",
            DataType::Decimal128(10, 2),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch");

        let df = ArrowConverter::record_batch_to_dataframe(&batch).expect("convert to DataFrame");
        assert_eq!(
            df.get_string_value("amount", 0).expect("get value"),
            "123.45"
        );
        assert_eq!(
            df.get_string_value("amount", 1).expect("get value"),
            "-123.45"
        );
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_date32_round_trip_as_iso_string() {
        // Date32 day-count 1 is 1970-01-02 (day 0 is the epoch);
        // stringifying the raw i32 directly (the old behavior) printed "1".
        let arr = Date32Array::from(vec![0i32, 1i32]);
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch");

        let df = ArrowConverter::record_batch_to_dataframe(&batch).expect("convert to DataFrame");
        assert_eq!(
            df.get_string_value("d", 0).expect("get value"),
            "1970-01-01"
        );
        assert_eq!(
            df.get_string_value("d", 1).expect("get value"),
            "1970-01-02"
        );
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_binary_column_round_trip_as_hex() {
        let arr = BinaryArray::from(vec![Some(&b"hi"[..]), None]);
        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Binary, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch");

        let df = ArrowConverter::record_batch_to_dataframe(&batch).expect("convert to DataFrame");
        // b"hi" == [0x68, 0x69]
        assert_eq!(df.get_string_value("b", 0).expect("get value"), "6869");
        assert_eq!(df.get_string_value("b", 1).expect("get value"), "null");
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_record_batch_duplicate_field_name_errors() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("x", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64])),
                Arc::new(Int64Array::from(vec![2i64])),
            ],
        )
        .expect("build batch");

        let result = ArrowConverter::record_batch_to_dataframe(&batch);
        assert!(
            result.is_err(),
            "a schema with a duplicate field name must error instead of \
             silently dropping one column's data"
        );
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_dataframes_to_record_batches_zero_batch_size_errors() {
        let series = Series::new(vec![1i64, 2, 3], Some("n".to_string())).expect("build series");
        let mut df = DataFrame::new();
        df.add_column("n".to_string(), series).expect("add column");

        let result =
            ArrowConverter::dataframes_to_record_batches(std::slice::from_ref(&df), Some(0));
        assert!(result.is_err(), "batch_size of 0 must error, not panic");
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_arrow_filter_kernel_correctness() {
        let series =
            Series::new(vec![1i64, 5, 10, 2, 8], Some("value".to_string())).expect("build series");
        let mut df = DataFrame::new();
        df.add_column("value".to_string(), series)
            .expect("add column");

        let filtered = df
            .compute_arrow(ArrowOperation::Filter {
                column: "value".to_string(),
                predicate: FilterPredicate::GreaterThan(4.0),
            })
            .expect("filter");

        assert_eq!(filtered.row_count(), 3);
        let mut values: Vec<i64> = (0..filtered.row_count())
            .map(|i| {
                filtered
                    .get_string_value("value", i)
                    .expect("get value")
                    .parse::<f64>()
                    .expect("parse") as i64
            })
            .collect();
        values.sort_unstable();
        assert_eq!(values, vec![5, 8, 10]);
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_arrow_sort_kernel_correctness() {
        let series =
            Series::new(vec![3i64, 1, 2], Some("value".to_string())).expect("build series");
        let mut df = DataFrame::new();
        df.add_column("value".to_string(), series)
            .expect("add column");

        let sorted = df
            .compute_arrow(ArrowOperation::Sort {
                columns: vec!["value".to_string()],
                ascending: vec![true],
            })
            .expect("sort");

        let values: Vec<String> = (0..sorted.row_count())
            .map(|i| {
                sorted
                    .get_string_value("value", i)
                    .expect("get value")
                    .to_string()
            })
            .collect();
        assert_eq!(values, vec!["1", "2", "3"]);
    }

    #[test]
    #[cfg(feature = "distributed")]
    fn test_arrow_sort_kernel_descending() {
        let series =
            Series::new(vec![3i64, 1, 2], Some("value".to_string())).expect("build series");
        let mut df = DataFrame::new();
        df.add_column("value".to_string(), series)
            .expect("add column");

        let sorted = df
            .compute_arrow(ArrowOperation::Sort {
                columns: vec!["value".to_string()],
                ascending: vec![false],
            })
            .expect("sort");

        let values: Vec<String> = (0..sorted.row_count())
            .map(|i| {
                sorted
                    .get_string_value("value", i)
                    .expect("get value")
                    .to_string()
            })
            .collect();
        assert_eq!(values, vec!["3", "2", "1"]);
    }
}

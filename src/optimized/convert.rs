//! Module providing DataFrame conversion functionality
//!
//! # NULL-preservation policy
//!
//! `crate::dataframe::DataFrame` stores every column as a plain, non-nullable
//! `crate::series::Series<T>` — there is no `Option<T>`/bitmask at that layer.
//! `crate::optimized` columns (`Int64Column`, `Float64Column`, `BooleanColumn`,
//! `StringColumn`), on the other hand, carry an explicit null bitmask via
//! `with_nulls`. Converting between the two therefore needs an explicit,
//! documented convention for representing "no value" on whichever side cannot
//! natively express it. Both directions below are chosen to be the
//! least-lossy option the `Series<T>` type system allows, and never silently
//! substitute a numeric/string default (0 / "" / false) for a value that was
//! never actually 0 / "" / false in the source.
//!
//! ## `from_standard_dataframe` (standard -> optimized)
//!
//! * `Series<String>` — the historical behavior of inferring a native type
//!   from string content is preserved (matching how this crate's CSV reader
//!   treats freshly-parsed string fields): if every *non-empty* value in the
//!   column parses as `i64` / `f64` / a recognized boolean word
//!   (`true`/`false`/`1`/`0`), the column is built as that native type. In
//!   every case an empty string (`""`) is treated as NULL rather than being
//!   coerced into `0` / `0.0` / `false` (the historical bug this module
//!   fixes) -- and a column that is *entirely* empty strings is never
//!   misclassified as numeric just because `""` vacuously "parses" as
//!   anything; it stays a fully-NULL `Series<String>`.
//! * `Series<f32>` / `Series<f64>` — `NaN` is treated as NULL (the standard
//!   floating-point "no value" sentinel used everywhere else in this crate).
//! * `Series<bool>` and every integer `Series<iN>`/`Series<uN>` — these types
//!   have no in-band sentinel for "missing", so they convert 1:1 with no
//!   NULLs. `u64` values that do not fit in `i64` (the only signed 64-bit
//!   integer column type this crate has) are rejected with an explicit
//!   error rather than silently wrapped/truncated.
//! * Any other element type is rejected with `Error::NotImplemented` naming
//!   the offending column, rather than the column being silently dropped
//!   (the historical bug this module fixes: a `DataFrame` made only of
//!   typed numeric/boolean columns used to convert to an OptimizedDataFrame
//!   with zero columns).
//!
//! ## `to_standard_dataframe` (optimized -> standard)
//!
//! * `Float64Column` NULLs become `f64::NAN` — lossless given the crate-wide
//!   NaN-as-NA convention above.
//! * `StringColumn` NULLs become `""` — matches the convention above.
//! * `BooleanColumn` with **any** NULL is promoted to `Series<String>`
//!   (`"true"` / `"false"` / `""`), because `Series<bool>` cannot represent
//!   NULL at all. A NULL-free `BooleanColumn` still converts to a native
//!   `Series<bool>`, unchanged.
//! * `Int64Column` with **any** NULL is promoted to `Series<f64>` with
//!   `f64::NAN` at the NULL positions — the same upcast pandas itself
//!   performs for an integer column containing NA. This keeps the result
//!   numeric (so downstream numeric consumers such as
//!   `DataFrame::get_column_numeric_values` keep working) at the cost of
//!   exact-integer precision above 2^53, which is the least-lossy option
//!   available; promoting to `Series<String>` instead was rejected because
//!   it would make the resulting Rust type depend on whether the data
//!   happens to contain a NULL, which downstream numeric code cannot
//!   possibly guard against. A NULL-free `Int64Column` still converts to a
//!   native `Series<i64>`, unchanged.

use crate::column::{BooleanColumn, Column, ColumnTrait, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

/// Build an `optimized::Column` from the named column of a standard
/// `DataFrame`, dispatching over every `Series<T>` element type this crate
/// commonly stores. See the module-level doc comment for the NULL policy.
fn build_column_from_series(df: &crate::dataframe::DataFrame, col_name: &str) -> Result<Column> {
    // Series<String>: infer a native type from the string content, exactly
    // as the CSV reader does (see module doc comment). "" is always NULL and
    // is excluded from the "does every value fit this type?" checks below,
    // so an all-blank column is never misclassified as numeric.
    if let Ok(col) = df.get_column::<String>(col_name) {
        let values: Vec<String> = col.values().to_vec();
        let non_empty_values: Vec<&String> = values.iter().filter(|s| !s.is_empty()).collect();

        if non_empty_values.is_empty() {
            let nulls = vec![true; values.len()];
            return Ok(Column::String(StringColumn::with_nulls(values, nulls)));
        }

        // Integer type
        let all_ints = non_empty_values.iter().all(|&s| s.parse::<i64>().is_ok());
        if all_ints {
            let mut int_values = Vec::with_capacity(values.len());
            let mut nulls = Vec::with_capacity(values.len());
            for s in &values {
                if s.is_empty() {
                    int_values.push(0);
                    nulls.push(true);
                } else {
                    let parsed = s.parse::<i64>().map_err(|e| {
                        Error::Cast(format!(
                            "Column '{}': failed to parse '{}' as i64: {}",
                            col_name, s, e
                        ))
                    })?;
                    int_values.push(parsed);
                    nulls.push(false);
                }
            }
            return Ok(Column::Int64(Int64Column::with_nulls(int_values, nulls)));
        }

        // Floating point type
        let all_floats = non_empty_values.iter().all(|&s| s.parse::<f64>().is_ok());
        if all_floats {
            let mut float_values = Vec::with_capacity(values.len());
            let mut nulls = Vec::with_capacity(values.len());
            for s in &values {
                if s.is_empty() {
                    float_values.push(0.0);
                    nulls.push(true);
                } else {
                    let parsed = s.parse::<f64>().map_err(|e| {
                        Error::Cast(format!(
                            "Column '{}': failed to parse '{}' as f64: {}",
                            col_name, s, e
                        ))
                    })?;
                    float_values.push(parsed);
                    nulls.push(false);
                }
            }
            return Ok(Column::Float64(Float64Column::with_nulls(
                float_values,
                nulls,
            )));
        }

        // Boolean type (case-insensitive true/false/1/0, matching the
        // original narrower word set this function has always used --
        // unlike the CSV reader's broader yes/no/t/f set).
        let all_bools = non_empty_values.iter().all(|&s| {
            let lower = s.to_lowercase();
            lower == "true" || lower == "false" || lower == "1" || lower == "0"
        });
        if all_bools {
            let mut bool_values = Vec::with_capacity(values.len());
            let mut nulls = Vec::with_capacity(values.len());
            for s in &values {
                if s.is_empty() {
                    bool_values.push(false);
                    nulls.push(true);
                } else {
                    let lower = s.to_lowercase();
                    bool_values.push(lower == "true" || lower == "1");
                    nulls.push(false);
                }
            }
            return Ok(Column::Boolean(BooleanColumn::with_nulls(
                bool_values,
                nulls,
            )));
        }

        // Default is string type; "" is still NULL.
        let nulls: Vec<bool> = values.iter().map(|s| s.is_empty()).collect();
        return Ok(Column::String(StringColumn::with_nulls(values, nulls)));
    }

    // Series<f64> / Series<f32>: NaN is treated as NULL.
    if let Ok(col) = df.get_column::<f64>(col_name) {
        let mut values = Vec::with_capacity(col.len());
        let mut nulls = Vec::with_capacity(col.len());
        for &v in col.values() {
            let is_null = v.is_nan();
            nulls.push(is_null);
            values.push(if is_null { 0.0 } else { v });
        }
        return Ok(Column::Float64(Float64Column::with_nulls(values, nulls)));
    }
    if let Ok(col) = df.get_column::<f32>(col_name) {
        let mut values = Vec::with_capacity(col.len());
        let mut nulls = Vec::with_capacity(col.len());
        for &v in col.values() {
            let is_null = v.is_nan();
            nulls.push(is_null);
            values.push(if is_null { 0.0 } else { v as f64 });
        }
        return Ok(Column::Float64(Float64Column::with_nulls(values, nulls)));
    }

    // bool: Series<bool> has no NULL sentinel, so this always converts 1:1.
    if let Ok(col) = df.get_column::<bool>(col_name) {
        return Ok(Column::Boolean(BooleanColumn::new(col.values().to_vec())));
    }

    // Signed/unsigned integers: no NULL sentinel, always convert 1:1 (widened
    // to i64, the only integer column type OptimizedDataFrame has).
    if let Ok(col) = df.get_column::<i64>(col_name) {
        return Ok(Column::Int64(Int64Column::new(col.values().to_vec())));
    }
    if let Ok(col) = df.get_column::<i32>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<i16>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<i8>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<u32>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<u16>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<u8>(col_name) {
        let values: Vec<i64> = col.values().iter().map(|&v| v as i64).collect();
        return Ok(Column::Int64(Int64Column::new(values)));
    }
    if let Ok(col) = df.get_column::<u64>(col_name) {
        let mut values = Vec::with_capacity(col.len());
        for &v in col.values() {
            let converted = i64::try_from(v).map_err(|_| {
                Error::Cast(format!(
                    "Column '{}' contains a u64 value {} that does not fit in i64; \
                     the OptimizedDataFrame bridge has no unsigned 64-bit column type",
                    col_name, v
                ))
            })?;
            values.push(converted);
        }
        return Ok(Column::Int64(Int64Column::new(values)));
    }

    Err(Error::NotImplemented(format!(
        "Column '{}' has an element type that the DataFrame -> OptimizedDataFrame bridge does \
         not support (supported element types: String, bool, 8/16/32/64-bit signed or \
         unsigned integers, and 32/64-bit floats)",
        col_name
    )))
}

/// Create an OptimizedDataFrame from a standard DataFrame
pub(crate) fn from_standard_dataframe(
    df: &crate::dataframe::DataFrame,
) -> Result<OptimizedDataFrame> {
    // Create a new SplitDataFrame (using internal implementation)
    let mut split_df = SplitDataFrame::new();

    for col_name in df.column_names() {
        let column = build_column_from_series(df, &col_name)?;
        split_df.add_column(col_name.clone(), column)?;
    }

    // Get and set the index.
    //
    // `DataFrame::get_index()` never returns `None`: when no index was ever
    // explicitly set on the source frame it synthesizes a placeholder
    // `DataFrameIndex::Simple(Index::default())`, which has length 0
    // regardless of the frame's actual row count. Handing that straight to
    // `set_index_from_simple_index` on a populated `split_df` used to fail
    // every such conversion with a length-mismatch error (0 != row_count) --
    // i.e. every `DataFrame` built without a follow-up `set_index()` call,
    // which is the common case. Detect that placeholder and synthesize a
    // real default index on the target instead of propagating the "empty"
    // length onto a non-empty frame.
    let df_index = df.get_index();

    match df_index {
        DataFrameIndex::Simple(simple_index)
            if simple_index.len() == 0 && split_df.row_count() > 0 =>
        {
            split_df.set_default_index()?;
        }
        DataFrameIndex::Simple(simple_index) => {
            // Direct copy for Simple Index
            split_df.set_index_from_simple_index(simple_index.clone())?;
        }
        DataFrameIndex::Multi(multi_index) => {
            // Convert multi-index to split DataFrame
            split_df.set_index(DataFrameIndex::Multi(multi_index.clone()))?;
        }
    }

    // Convert SplitDataFrame to OptimizedDataFrame
    let mut opt_df = OptimizedDataFrame::new();

    // Copy column data (using public API)
    for name in split_df.column_names() {
        if let Ok(column_view) = split_df.column(name) {
            let column = column_view.column().clone();
            opt_df.add_column(name.clone(), column)?;
        }
    }

    // Set the index (Simple or Multi -- `set_index_directly` handles both
    // uniformly, so a MultiIndex on the source DataFrame is preserved here
    // instead of being silently dropped).
    if let Some(split_index) = split_df.get_index() {
        opt_df.set_index_directly(split_index.clone())?;
    }

    Ok(opt_df)
}

/// Convert OptimizedDataFrame to a standard DataFrame
pub(crate) fn to_standard_dataframe(
    df: &OptimizedDataFrame,
) -> Result<crate::dataframe::DataFrame> {
    // Use internal SplitDataFrame
    let mut split_df = SplitDataFrame::new();

    // Convert column data
    for col_name in df.column_names() {
        let col_view = df.column(col_name)?;
        let col = col_view.column();

        // Add columns to SplitDataFrame
        split_df.add_column(col_name.clone(), col.clone())?;
    }

    // Set the index if it exists
    if let Some(df_index) = df.get_index() {
        // Use appropriate methods instead of directly setting internal fields
        if let DataFrameIndex::Simple(simple_index) = df_index {
            split_df.set_index_from_simple_index(simple_index.clone())?;
        } else if let DataFrameIndex::Multi(multi_index) = df_index {
            // Multi-index support
            split_df.set_index(DataFrameIndex::Multi(multi_index.clone()))?;
        }
    }

    // Convert to standard DataFrame
    let mut std_df = crate::dataframe::DataFrame::new();

    // Process each column. See the module-level doc comment for the NULL
    // policy applied to each column type below.
    for col_name in split_df.column_names() {
        let col_view = split_df.column(col_name)?;
        let col = col_view.column();

        match col {
            Column::Int64(int_col) => {
                let mut opts: Vec<Option<i64>> = Vec::with_capacity(int_col.len());
                for i in 0..int_col.len() {
                    opts.push(int_col.get(i)?);
                }
                if opts.iter().any(Option::is_none) {
                    // NULL present: Series<i64> cannot represent it. Upcast to
                    // Series<f64> with NaN at the NULL positions.
                    let values: Vec<f64> = opts
                        .into_iter()
                        .map(|v| v.map(|x| x as f64).unwrap_or(f64::NAN))
                        .collect();
                    let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                    std_df.add_column(col_name.clone(), series)?;
                } else {
                    let values: Vec<i64> = opts.into_iter().map(|v| v.unwrap_or(0)).collect();
                    let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                    std_df.add_column(col_name.clone(), series)?;
                }
            }
            Column::Float64(float_col) => {
                // Create Series<f64> directly so get_column::<f64>() downcast succeeds.
                // NULL -> NaN (see module doc comment).
                let mut values = Vec::with_capacity(float_col.len());
                for i in 0..float_col.len() {
                    values.push(float_col.get(i)?.unwrap_or(f64::NAN));
                }
                let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                std_df.add_column(col_name.clone(), series)?;
            }
            Column::String(str_col) => {
                // Create Series<String> directly so get_column::<String>() downcast succeeds.
                // NULL -> "" (see module doc comment).
                let mut values = Vec::with_capacity(str_col.len());
                for i in 0..str_col.len() {
                    values.push(str_col.get(i)?.map(|s| s.to_string()).unwrap_or_default());
                }
                let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                std_df.add_column(col_name.clone(), series)?;
            }
            Column::Boolean(bool_col) => {
                let mut opts: Vec<Option<bool>> = Vec::with_capacity(bool_col.len());
                for i in 0..bool_col.len() {
                    opts.push(bool_col.get(i)?);
                }
                if opts.iter().any(Option::is_none) {
                    // NULL present: Series<bool> cannot represent it. Promote to
                    // Series<String> ("true" / "false" / "").
                    let values: Vec<String> = opts
                        .into_iter()
                        .map(|v| match v {
                            Some(true) => "true".to_string(),
                            Some(false) => "false".to_string(),
                            None => String::new(),
                        })
                        .collect();
                    let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                    std_df.add_column(col_name.clone(), series)?;
                } else {
                    let values: Vec<bool> = opts.into_iter().map(|v| v.unwrap_or(false)).collect();
                    let series = crate::series::Series::new(values, Some(col_name.clone()))?;
                    std_df.add_column(col_name.clone(), series)?;
                }
            }
        }
    }

    // Set the index
    if let Some(split_index) = split_df.get_index() {
        match split_index {
            DataFrameIndex::Simple(simple_index) => {
                // Set as string-based Simple Index
                std_df.set_index(simple_index.clone())?;
            }
            DataFrameIndex::Multi(multi_index) => {
                // For multi-index
                std_df.set_multi_index(multi_index.clone())?;
            }
        }
    }

    Ok(std_df)
}

/// Public function to convert a standard DataFrame to an optimized OptimizedDataFrame
pub fn optimize_dataframe(df: &crate::dataframe::DataFrame) -> Result<OptimizedDataFrame> {
    from_standard_dataframe(df)
}

/// Public function to convert an OptimizedDataFrame to a standard DataFrame
pub fn standard_dataframe(df: &OptimizedDataFrame) -> Result<crate::dataframe::DataFrame> {
    to_standard_dataframe(df)
}

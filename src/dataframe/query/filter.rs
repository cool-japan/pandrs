//! Row selection that preserves column types.
//!
//! `df.query(...)` used to rebuild every column as `Series<String>`, so an
//! `i64` column came back as text and any later numeric use of the result
//! failed. Selection here dispatches on the column's concrete element type and
//! rebuilds a `Series` of that same type, and it carries the row index (simple
//! or multi-level) across the filter.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::index::{DataFrameIndex, Index, MultiIndex};
use crate::series::Series;

/// Build a new DataFrame containing only `indices`, in the given order.
///
/// Every column keeps its element type. A column whose element type is not
/// natively handled here falls back to its string rendering; an element type
/// that cannot even be rendered as text reports an error rather than
/// substituting placeholder values.
pub(crate) fn take_rows(dataframe: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let mut result = DataFrame::new();

    for column_name in dataframe.column_names() {
        take_column(dataframe, &mut result, column_name, indices)?;
    }

    carry_index(dataframe, &mut result, indices)?;

    Ok(result)
}

/// Copy one column's selected rows into `result`, preserving its element type.
fn take_column(
    source: &DataFrame,
    result: &mut DataFrame,
    column_name: &str,
    indices: &[usize],
) -> Result<()> {
    macro_rules! try_take {
        ($($ty:ty),+ $(,)?) => {
            $(
                if let Ok(series) = source.get_column::<$ty>(column_name) {
                    let values = series.values();
                    let mut taken: Vec<$ty> = Vec::with_capacity(indices.len());
                    for &idx in indices {
                        match values.get(idx) {
                            Some(value) => taken.push(value.clone()),
                            None => {
                                return Err(Error::IndexOutOfBounds {
                                    index: idx,
                                    size: values.len(),
                                })
                            }
                        }
                    }
                    result.add_column(
                        column_name.to_string(),
                        Series::new(taken, Some(column_name.to_string()))?,
                    )?;
                    return Ok(());
                }
            )+
        };
    }

    try_take!(
        String,
        i64,
        f64,
        bool,
        i32,
        f32,
        u64,
        u32,
        usize,
        i16,
        u16,
        i8,
        u8,
        isize,
        i128,
        u128,
        chrono::NaiveDateTime,
        chrono::NaiveDate,
        chrono::NaiveTime,
        chrono::DateTime<chrono::Utc>,
    );

    // Unknown element type: fall back to the string rendering so the column is
    // still carried through instead of aborting the query.
    let values = source.get_column_string_values(column_name)?;
    let mut taken: Vec<String> = Vec::with_capacity(indices.len());
    for &idx in indices {
        match values.get(idx) {
            Some(value) => taken.push(value.clone()),
            None => {
                return Err(Error::IndexOutOfBounds {
                    index: idx,
                    size: values.len(),
                })
            }
        }
    }
    result.add_column(
        column_name.to_string(),
        Series::new(taken, Some(column_name.to_string()))?,
    )?;

    Ok(())
}

/// Carry the source frame's row index over to the filtered frame.
fn carry_index(source: &DataFrame, result: &mut DataFrame, indices: &[usize]) -> Result<()> {
    let row_count = source.row_count();
    if row_count == 0 {
        return Ok(());
    }

    match source.get_index() {
        DataFrameIndex::Simple(index) => {
            if index.len() != row_count {
                // No explicit index (an implicit positional index is assumed).
                return Ok(());
            }
            let labels: Vec<String> = indices
                .iter()
                .filter_map(|&idx| index.get_value(idx).cloned())
                .collect();
            if labels.len() != indices.len() {
                return Ok(());
            }
            let filtered = Index::with_name(labels, index.name().cloned())?;
            result.set_index(filtered)?;
        }
        DataFrameIndex::Multi(multi) => {
            if multi.len() != row_count {
                return Ok(());
            }
            let levels = multi.levels().to_vec();
            let mut codes = Vec::with_capacity(multi.codes().len());
            for level_codes in multi.codes() {
                let mut taken = Vec::with_capacity(indices.len());
                for &idx in indices {
                    match level_codes.get(idx) {
                        Some(code) => taken.push(*code),
                        None => return Ok(()),
                    }
                }
                codes.push(taken);
            }
            let names = multi.names().to_vec();
            let filtered = MultiIndex::new(levels, codes, Some(names))?;
            result.set_multi_index(filtered)?;
        }
    }

    Ok(())
}

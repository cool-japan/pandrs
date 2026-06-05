//! Type conversion utilities between PandRS and SciRS2/ndarray types.
//!
//! All functions in this module are gated behind the `scirs2` feature flag.

#[cfg(feature = "scirs2")]
use scirs2_core::error::CoreError;
#[cfg(feature = "scirs2")]
use scirs2_core::ndarray::{Array1, Array2};

#[cfg(feature = "scirs2")]
use crate::core::error::{Error, Result};
#[cfg(feature = "scirs2")]
use crate::dataframe::DataFrame;
#[cfg(feature = "scirs2")]
use crate::series::Series;

/// Convert a numeric PandRS `Series<f64>` to an ndarray `Array1<f64>`.
///
/// # Errors
///
/// Returns an error if the Series is empty.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "scirs2")]
/// # {
/// use pandrs::Series;
/// use pandrs::scirs2_integration::conversion::series_to_array1;
///
/// let series = Series::new(vec![1.0f64, 2.0, 3.0], Some("values".to_string())).expect("ok");
/// let arr = series_to_array1(&series).expect("conversion ok");
/// assert_eq!(arr.len(), 3);
/// # }
/// ```
#[cfg(feature = "scirs2")]
pub fn series_to_array1(series: &Series<f64>) -> Result<Array1<f64>> {
    let values = series.values().to_vec();
    if values.is_empty() {
        return Err(Error::EmptyData(
            "Cannot convert empty Series to Array1".to_string(),
        ));
    }
    Ok(Array1::from(values))
}

/// Convert an ndarray `Array1<f64>` to a PandRS `Series<f64>`.
///
/// # Arguments
///
/// * `arr` - The ndarray Array1 to convert
/// * `name` - Optional name for the resulting Series
///
/// # Errors
///
/// Returns an error if the array is empty.
#[cfg(feature = "scirs2")]
pub fn array1_to_series(arr: &Array1<f64>, name: Option<String>) -> Result<Series<f64>> {
    if arr.is_empty() {
        return Err(Error::EmptyData(
            "Cannot convert empty Array1 to Series".to_string(),
        ));
    }
    let values: Vec<f64> = arr.iter().copied().collect();
    Series::new(values, name)
}

/// Convert selected numeric columns of a DataFrame to an ndarray `Array2<f64>`.
///
/// The resulting array has shape `(n_rows, n_cols)` where columns are in the
/// same order as provided in the `columns` argument.
///
/// # Arguments
///
/// * `df` - The source DataFrame
/// * `columns` - Slice of column names to include (must be numeric columns)
///
/// # Errors
///
/// Returns an error if:
/// - Any requested column does not exist
/// - Any column cannot be converted to f64
/// - The DataFrame is empty
#[cfg(feature = "scirs2")]
pub fn dataframe_to_array2(df: &DataFrame, columns: &[&str]) -> Result<Array2<f64>> {
    if columns.is_empty() {
        return Err(Error::InvalidInput(
            "At least one column must be specified".to_string(),
        ));
    }

    let n_rows = df.row_count();
    if n_rows == 0 {
        return Err(Error::EmptyData(
            "Cannot convert empty DataFrame to Array2".to_string(),
        ));
    }

    let n_cols = columns.len();
    let mut data = vec![0.0f64; n_rows * n_cols];

    for (col_idx, &col_name) in columns.iter().enumerate() {
        let col_values = df.get_column_numeric_values(col_name)?;
        if col_values.len() != n_rows {
            return Err(Error::InconsistentRowCount {
                expected: n_rows,
                found: col_values.len(),
            });
        }
        for (row_idx, val) in col_values.into_iter().enumerate() {
            data[row_idx * n_cols + col_idx] = val;
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), data)
        .map_err(|e| Error::OperationFailed(format!("Failed to create Array2 from data: {}", e)))
}

/// Convert an ndarray `Array2<f64>` to a DataFrame with the specified column names.
///
/// # Arguments
///
/// * `arr` - The ndarray Array2 with shape `(n_rows, n_cols)`
/// * `columns` - Column names (must have the same length as `arr.ncols()`)
///
/// # Errors
///
/// Returns an error if the number of column names does not match the array's
/// column count, or if column names are duplicate.
#[cfg(feature = "scirs2")]
pub fn array2_to_dataframe(arr: &Array2<f64>, columns: Vec<String>) -> Result<DataFrame> {
    let (n_rows, n_cols) = arr.dim();

    if columns.len() != n_cols {
        return Err(Error::InvalidInput(format!(
            "Number of column names ({}) does not match array column count ({})",
            columns.len(),
            n_cols
        )));
    }

    let mut df = DataFrame::new();

    for (col_idx, col_name) in columns.into_iter().enumerate() {
        let col_data: Vec<f64> = (0..n_rows).map(|row| arr[[row, col_idx]]).collect();
        let series = Series::new(col_data, Some(col_name.clone()))?;
        df.add_column(col_name, series)?;
    }

    Ok(df)
}

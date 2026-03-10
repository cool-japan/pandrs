//! Extension trait for DataFrame to access SciRS2 functionality seamlessly.
//!
//! The `SciRS2Ext` trait is gated behind the `scirs2` feature flag. When the
//! feature is enabled, `impl SciRS2Ext for DataFrame` is provided, giving
//! DataFrames convenient access to SciRS2-backed operations.

#[cfg(feature = "scirs2")]
use ndarray::Array2;

#[cfg(feature = "scirs2")]
use crate::core::error::Result;
#[cfg(feature = "scirs2")]
use crate::dataframe::DataFrame;
#[cfg(feature = "scirs2")]
use crate::scirs2_integration::conversion::{array2_to_dataframe, dataframe_to_array2};
#[cfg(feature = "scirs2")]
use crate::scirs2_integration::stats::{PcaResult, SciRS2Stats};

/// Extension trait providing SciRS2-backed operations for DataFrames.
///
/// Import this trait to gain access to SciRS2 scientific computing capabilities
/// directly on `DataFrame` values.
///
/// # Feature
///
/// All methods require the `scirs2` feature flag.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "scirs2")]
/// # {
/// use pandrs::{DataFrame, Series};
/// use pandrs::scirs2_integration::dataframe_ext::SciRS2Ext;
///
/// let mut df = DataFrame::new();
/// df.add_column("x".to_string(),
///     Series::new(vec![1.0f64, 2.0, 3.0], Some("x".to_string())).expect("ok"))
///     .expect("ok");
/// df.add_column("y".to_string(),
///     Series::new(vec![3.0f64, 6.0, 9.0], Some("y".to_string())).expect("ok"))
///     .expect("ok");
///
/// let arr = df.to_ndarray(&["x", "y"]).expect("to ndarray");
/// assert_eq!(arr.shape(), &[3, 2]);
/// # }
/// ```
pub trait SciRS2Ext {
    /// Convert selected numeric columns to an ndarray `Array2<f64>`.
    ///
    /// # Arguments
    ///
    /// * `columns` - Column names to include; must be numeric columns
    ///
    /// # Errors
    ///
    /// Returns an error if any column does not exist or cannot be converted to f64.
    #[cfg(feature = "scirs2")]
    fn to_ndarray(&self, columns: &[&str]) -> Result<Array2<f64>>;

    /// Create a DataFrame from an ndarray `Array2<f64>` and column names.
    ///
    /// # Arguments
    ///
    /// * `arr` - The array with shape (n_rows, n_cols)
    /// * `columns` - Column names (must match the number of columns in `arr`)
    ///
    /// # Errors
    ///
    /// Returns an error if the number of column names does not match.
    #[cfg(feature = "scirs2")]
    fn from_ndarray(arr: &Array2<f64>, columns: Vec<String>) -> Result<DataFrame>
    where
        Self: Sized;

    /// Compute descriptive statistics using SciRS2 for all numeric columns.
    ///
    /// Returns a DataFrame with statistics (count, mean, std, min, 25%, 50%,
    /// 75%, max) as rows and the original columns as columns.
    ///
    /// # Errors
    ///
    /// Returns an error if any column cannot be processed.
    #[cfg(feature = "scirs2")]
    fn scirs2_describe(&self) -> Result<DataFrame>;

    /// Compute the Pearson correlation matrix using SciRS2 for all numeric columns.
    ///
    /// Returns a symmetric DataFrame with an additional leading `"column"` column
    /// identifying the row variables.
    ///
    /// # Errors
    ///
    /// Returns an error if there are fewer than 2 numeric columns or the
    /// computation fails.
    #[cfg(feature = "scirs2")]
    fn scirs2_corr(&self) -> Result<DataFrame>;

    /// Perform PCA using SciRS2 on all numeric columns.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of principal components to extract
    ///
    /// # Errors
    ///
    /// Returns an error if there are insufficient numeric columns or the
    /// decomposition fails.
    #[cfg(feature = "scirs2")]
    fn scirs2_pca(&self, n_components: usize) -> Result<PcaResult>;
}

#[cfg(feature = "scirs2")]
impl SciRS2Ext for DataFrame {
    fn to_ndarray(&self, columns: &[&str]) -> Result<Array2<f64>> {
        dataframe_to_array2(self, columns)
    }

    fn from_ndarray(arr: &Array2<f64>, columns: Vec<String>) -> Result<DataFrame> {
        array2_to_dataframe(arr, columns)
    }

    fn scirs2_describe(&self) -> Result<DataFrame> {
        // Collect all column names
        let all_cols = self.column_names();
        // Attempt to filter to only numeric columns by trying to get numeric values
        let numeric_cols: Vec<&str> = all_cols
            .iter()
            .filter(|col_name| self.get_column_numeric_values(col_name).is_ok())
            .map(|s| s.as_str())
            .collect();

        if numeric_cols.is_empty() {
            return Err(crate::core::error::Error::EmptyData(
                "No numeric columns found for describe".to_string(),
            ));
        }

        SciRS2Stats::describe(self, &numeric_cols)
    }

    fn scirs2_corr(&self) -> Result<DataFrame> {
        let all_cols = self.column_names();
        let numeric_cols: Vec<&str> = all_cols
            .iter()
            .filter(|col_name| self.get_column_numeric_values(col_name).is_ok())
            .map(|s| s.as_str())
            .collect();

        if numeric_cols.len() < 2 {
            return Err(crate::core::error::Error::InvalidInput(
                "Correlation matrix requires at least 2 numeric columns".to_string(),
            ));
        }

        SciRS2Stats::correlation_matrix(self, &numeric_cols)
    }

    fn scirs2_pca(&self, n_components: usize) -> Result<PcaResult> {
        let all_cols = self.column_names();
        let numeric_cols: Vec<&str> = all_cols
            .iter()
            .filter(|col_name| self.get_column_numeric_values(col_name).is_ok())
            .map(|s| s.as_str())
            .collect();

        if numeric_cols.is_empty() {
            return Err(crate::core::error::Error::EmptyData(
                "No numeric columns found for PCA".to_string(),
            ));
        }

        SciRS2Stats::pca(self, &numeric_cols, n_components)
    }
}

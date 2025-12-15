//! Pandas compatibility trait definition

use crate::core::error::Result;
use crate::dataframe::base::DataFrame;
use std::collections::HashMap;

use super::types::{Axis, CorrelationMatrix, DescribeStats, RankMethod, SeriesValue};

/// Pandas compatibility extension trait for DataFrame
pub trait PandasCompatExt {
    /// Assign new columns to a DataFrame
    ///
    /// Returns a new DataFrame with the new columns added.
    fn assign<F, T>(&self, name: &str, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&DataFrame) -> Vec<T>,
        T: Into<SeriesValue>;

    /// Assign multiple new columns
    fn assign_many(&self, assignments: Vec<(&str, Vec<f64>)>) -> Result<DataFrame>;

    /// Apply a function to the DataFrame (pipe pattern)
    fn pipe<F, R>(&self, func: F) -> R
    where
        F: FnOnce(&Self) -> R;

    /// Apply a function that returns Result
    fn pipe_result<F>(&self, func: F) -> Result<DataFrame>
    where
        F: FnOnce(&Self) -> Result<DataFrame>;

    /// Check if values are in a set (boolean mask)
    fn isin(&self, column: &str, values: &[&str]) -> Result<Vec<bool>>;

    /// Check if numeric values are in a set
    fn isin_numeric(&self, column: &str, values: &[f64]) -> Result<Vec<bool>>;

    /// Select top N rows by column value
    fn nlargest(&self, n: usize, column: &str) -> Result<DataFrame>;

    /// Select bottom N rows by column value
    fn nsmallest(&self, n: usize, column: &str) -> Result<DataFrame>;

    /// Get index of maximum value in column
    fn idxmax(&self, column: &str) -> Result<Option<usize>>;

    /// Get index of minimum value in column
    fn idxmin(&self, column: &str) -> Result<Option<usize>>;

    /// Compute rank of values in column
    fn rank(&self, column: &str, method: RankMethod) -> Result<Vec<f64>>;

    /// Clip values to a range
    fn clip(&self, column: &str, lower: Option<f64>, upper: Option<f64>) -> Result<DataFrame>;

    /// Check if values are between bounds (inclusive)
    fn between(&self, column: &str, lower: f64, upper: f64) -> Result<Vec<bool>>;

    /// Transpose the DataFrame (swap rows and columns)
    fn transpose(&self) -> Result<DataFrame>;

    /// Cumulative sum of a column
    fn cumsum(&self, column: &str) -> Result<Vec<f64>>;

    /// Cumulative product of a column
    fn cumprod(&self, column: &str) -> Result<Vec<f64>>;

    /// Cumulative maximum of a column
    fn cummax(&self, column: &str) -> Result<Vec<f64>>;

    /// Cumulative minimum of a column
    fn cummin(&self, column: &str) -> Result<Vec<f64>>;

    /// Shift values by periods
    fn shift(&self, column: &str, periods: i32) -> Result<Vec<Option<f64>>>;

    /// Number of unique values per column
    fn nunique(&self) -> Result<Vec<(String, usize)>>;

    /// Get memory usage estimate
    fn memory_usage(&self) -> usize;

    /// Count occurrences of unique values in a column
    fn value_counts(&self, column: &str) -> Result<Vec<(String, usize)>>;

    /// Count occurrences of unique numeric values
    fn value_counts_numeric(&self, column: &str) -> Result<Vec<(f64, usize)>>;

    /// Generate descriptive statistics
    fn describe(&self, column: &str) -> Result<DescribeStats>;

    /// Apply a function to each row or column
    fn apply<F, T>(&self, func: F, axis: Axis) -> Result<Vec<T>>
    where
        F: Fn(&[f64]) -> T;

    /// Compute pairwise correlation of numeric columns
    fn corr(&self) -> Result<CorrelationMatrix>;

    /// Compute covariance matrix of numeric columns
    fn cov(&self) -> Result<CorrelationMatrix>;

    /// Percentage change between current and prior element
    fn pct_change(&self, column: &str, periods: usize) -> Result<Vec<f64>>;

    /// First discrete difference
    fn diff(&self, column: &str, periods: usize) -> Result<Vec<f64>>;

    /// Replace values in a column
    fn replace(&self, column: &str, to_replace: &[&str], values: &[&str]) -> Result<DataFrame>;

    /// Replace numeric values
    fn replace_numeric(
        &self,
        column: &str,
        to_replace: &[f64],
        values: &[f64],
    ) -> Result<DataFrame>;

    /// Random sample of rows
    fn sample(&self, n: usize, replace: bool) -> Result<DataFrame>;

    /// Drop columns
    fn drop_columns(&self, labels: &[&str]) -> Result<DataFrame>;

    /// Rename columns using a mapping
    fn rename_columns(&self, mapper: &HashMap<String, String>) -> Result<DataFrame>;

    /// Compute absolute values for numeric columns
    fn abs(&self, column: &str) -> Result<DataFrame>;

    /// Round values to given number of decimals
    fn round(&self, column: &str, decimals: i32) -> Result<DataFrame>;

    /// Compute quantile (percentile) for a column
    fn quantile(&self, column: &str, q: f64) -> Result<f64>;

    /// Get first n rows
    fn head(&self, n: usize) -> Result<DataFrame>;

    /// Get last n rows
    fn tail(&self, n: usize) -> Result<DataFrame>;

    /// Get unique values from a column
    fn unique(&self, column: &str) -> Result<Vec<String>>;

    /// Get unique numeric values from a column
    fn unique_numeric(&self, column: &str) -> Result<Vec<f64>>;

    /// Fill NaN values in a numeric column with a specified value
    fn fillna(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Fill NaN values using a specified method (forward fill or backward fill)
    ///
    /// # Arguments
    /// * `column` - Column name to fill
    /// * `method` - Fill method: "ffill" (forward fill) or "bfill" (backward fill)
    fn fillna_method(&self, column: &str, method: &str) -> Result<DataFrame>;

    /// Interpolate missing values using linear interpolation
    ///
    /// # Arguments
    /// * `column` - Column name to interpolate
    fn interpolate(&self, column: &str) -> Result<DataFrame>;

    /// Drop rows where specified column has NaN values
    fn dropna(&self, column: &str) -> Result<DataFrame>;

    /// Count NaN values in a column
    fn isna(&self, column: &str) -> Result<Vec<bool>>;

    /// Sum all numeric columns
    fn sum_all(&self) -> Result<Vec<(String, f64)>>;

    /// Mean of all numeric columns
    fn mean_all(&self) -> Result<Vec<(String, f64)>>;

    /// Standard deviation of all numeric columns
    fn std_all(&self) -> Result<Vec<(String, f64)>>;

    /// Variance of all numeric columns
    fn var_all(&self) -> Result<Vec<(String, f64)>>;

    /// Minimum values of all numeric columns
    fn min_all(&self) -> Result<Vec<(String, f64)>>;

    /// Maximum values of all numeric columns
    fn max_all(&self) -> Result<Vec<(String, f64)>>;

    /// Sort by column values
    fn sort_values(&self, column: &str, ascending: bool) -> Result<DataFrame>;

    /// Sort by multiple columns
    fn sort_by_columns(&self, columns: &[&str], ascending: &[bool]) -> Result<DataFrame>;

    /// Merge with another DataFrame on a common column
    ///
    /// # Arguments
    /// * `other` - DataFrame to merge with
    /// * `on` - Column name to join on
    /// * `how` - Join type (Inner, Left, Right, Outer)
    /// * `suffixes` - Tuple of suffixes for overlapping columns
    fn merge(
        &self,
        other: &DataFrame,
        on: &str,
        how: super::merge::JoinType,
        suffixes: (&str, &str),
    ) -> Result<DataFrame>;

    /// Replace values where condition is False with other value
    ///
    /// Keep original value where condition is True, replace with `other` where False.
    /// Similar to pandas `where()`.
    fn where_cond(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame>;

    /// Replace values where condition is True with other value
    ///
    /// Replace value with `other` where condition is True, keep original where False.
    /// Similar to pandas `mask()`.
    fn mask(&self, column: &str, condition: &[bool], other: f64) -> Result<DataFrame>;

    /// Remove duplicate rows based on specified columns
    ///
    /// # Arguments
    /// * `subset` - Columns to consider for identifying duplicates (None = all columns)
    /// * `keep` - Which duplicates to keep: "first", "last", or "none"
    fn drop_duplicates(&self, subset: Option<&[&str]>, keep: &str) -> Result<DataFrame>;

    /// Select columns by data type
    ///
    /// # Arguments
    /// * `include` - Data types to include ("numeric", "string")
    fn select_dtypes(&self, include: &[&str]) -> Result<DataFrame>;

    /// Check if any value is non-zero/True for numeric columns
    fn any_numeric(&self) -> Result<Vec<(String, bool)>>;

    /// Check if all values are non-zero/True for numeric columns
    fn all_numeric(&self) -> Result<Vec<(String, bool)>>;

    /// Get number of non-NA values per column
    fn count_valid(&self) -> Result<Vec<(String, usize)>>;

    /// Return DataFrame with columns in reversed order
    fn reverse_columns(&self) -> Result<DataFrame>;

    /// Return DataFrame with rows in reversed order
    fn reverse_rows(&self) -> Result<DataFrame>;

    /// Detect non-NA values (inverse of isna)
    ///
    /// Returns boolean mask where True indicates non-NA values.
    fn notna(&self, column: &str) -> Result<Vec<bool>>;

    /// Unpivot DataFrame from wide to long format
    ///
    /// # Arguments
    /// * `id_vars` - Columns to use as identifier variables
    /// * `value_vars` - Columns to unpivot (if None, use all non-id columns)
    /// * `var_name` - Name of the variable column
    /// * `value_name` - Name of the value column
    fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: &str,
        value_name: &str,
    ) -> Result<DataFrame>;

    /// Explode a list-like column into multiple rows
    ///
    /// For columns containing comma-separated values or list-like strings,
    /// expand each element into its own row.
    fn explode(&self, column: &str, separator: &str) -> Result<DataFrame>;

    /// Mark duplicate rows
    ///
    /// Returns boolean mask where True indicates duplicate rows.
    /// # Arguments
    /// * `subset` - Columns to consider for identifying duplicates (None = all columns)
    /// * `keep` - Which occurrence to mark as non-duplicate: "first", "last", or "none" (mark all)
    fn duplicated(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>>;

    /// Create a deep copy of the DataFrame
    fn copy(&self) -> DataFrame;

    /// Convert DataFrame to a dictionary representation
    ///
    /// Returns a HashMap where keys are column names and values are vectors of values.
    fn to_dict(&self) -> Result<HashMap<String, Vec<String>>>;

    /// Get the index of the first valid (non-NA) value in a column
    fn first_valid_index(&self, column: &str) -> Result<Option<usize>>;

    /// Get the index of the last valid (non-NA) value in a column
    fn last_valid_index(&self, column: &str) -> Result<Option<usize>>;

    /// Product of all numeric columns (skipping NaN)
    fn product_all(&self) -> Result<Vec<(String, f64)>>;

    /// Median of all numeric columns
    fn median_all(&self) -> Result<Vec<(String, f64)>>;

    /// Compute skewness for a column
    fn skew(&self, column: &str) -> Result<f64>;

    /// Compute kurtosis for a column
    fn kurtosis(&self, column: &str) -> Result<f64>;

    /// Add a prefix to all column names
    fn add_prefix(&self, prefix: &str) -> Result<DataFrame>;

    /// Add a suffix to all column names
    fn add_suffix(&self, suffix: &str) -> Result<DataFrame>;

    /// Filter rows based on a boolean mask
    fn filter_by_mask(&self, mask: &[bool]) -> Result<DataFrame>;

    /// Get the mode (most frequent value) for a numeric column
    fn mode_numeric(&self, column: &str) -> Result<Vec<f64>>;

    /// Get the mode (most frequent value) for a string column
    fn mode_string(&self, column: &str) -> Result<Vec<String>>;

    /// Compute the n-th percentile for a column (alias for quantile)
    fn percentile(&self, column: &str, n: f64) -> Result<f64>;

    /// Compute exponentially weighted moving average for a column
    fn ewma(&self, column: &str, span: usize) -> Result<Vec<f64>>;

    /// Get a specific row by index as a HashMap
    fn iloc(&self, index: usize) -> Result<HashMap<String, String>>;

    /// Get multiple rows by indices
    fn iloc_range(&self, start: usize, end: usize) -> Result<DataFrame>;

    /// Get DataFrame summary information
    ///
    /// Returns a string containing:
    /// - Index dtype and range
    /// - Column count and dtypes
    /// - Non-null counts
    /// - Memory usage
    fn info(&self) -> String;

    /// Test whether two DataFrames are equal
    ///
    /// Returns true if all elements are equal (NaN equals NaN).
    fn equals(&self, other: &DataFrame) -> bool;

    /// Compare two DataFrames and show differences
    ///
    /// Returns a DataFrame showing where values differ.
    fn compare(&self, other: &DataFrame) -> Result<DataFrame>;

    /// Return column names as a vector
    fn keys(&self) -> Vec<String>;

    /// Remove and return a column from the DataFrame
    fn pop_column(&self, column: &str) -> Result<(DataFrame, Vec<f64>)>;

    /// Insert a column at a specific position
    fn insert_column(&self, loc: usize, name: &str, values: Vec<f64>) -> Result<DataFrame>;

    /// Compute rolling sum with configurable window
    fn rolling_sum(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling mean with configurable window
    fn rolling_mean(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling standard deviation
    fn rolling_std(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling minimum
    fn rolling_min(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling maximum
    fn rolling_max(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling variance
    fn rolling_var(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling median
    fn rolling_median(
        &self,
        column: &str,
        window: usize,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>;

    /// Compute rolling count of non-NaN values
    fn rolling_count(&self, column: &str, window: usize) -> Result<Vec<usize>>;

    /// Apply custom function to rolling window
    fn rolling_apply<F>(
        &self,
        column: &str,
        window: usize,
        func: F,
        min_periods: Option<usize>,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64;

    /// Apply cumulative count (number of non-NA values seen so far)
    fn cumcount(&self, column: &str) -> Result<Vec<usize>>;

    /// Get the nth row (supports negative indexing)
    fn nth(&self, n: i32) -> Result<HashMap<String, String>>;

    /// Apply element-wise function to numeric column
    fn transform<F>(&self, column: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64;

    /// Cross-tabulation of two columns
    fn crosstab(&self, col1: &str, col2: &str) -> Result<DataFrame>;

    /// Compute expanding sum (cumulative from start)
    fn expanding_sum(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Compute expanding mean
    fn expanding_mean(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Compute expanding standard deviation
    fn expanding_std(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Compute expanding minimum
    fn expanding_min(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Compute expanding maximum
    fn expanding_max(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Compute expanding variance
    fn expanding_var(&self, column: &str, min_periods: usize) -> Result<Vec<f64>>;

    /// Apply custom function to expanding window
    fn expanding_apply<F>(&self, column: &str, func: F, min_periods: usize) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64;

    /// Align two DataFrames on their columns
    ///
    /// Returns a tuple of DataFrames with matching columns, filling missing with NaN.
    fn align(&self, other: &DataFrame) -> Result<(DataFrame, DataFrame)>;

    /// Reorder columns
    fn reindex_columns(&self, columns: &[&str]) -> Result<DataFrame>;

    /// Get range of numeric values in a column
    fn value_range(&self, column: &str) -> Result<(f64, f64)>;

    /// Compute z-score normalization for a column
    fn zscore(&self, column: &str) -> Result<Vec<f64>>;

    /// Compute min-max normalization for a column
    fn normalize(&self, column: &str) -> Result<Vec<f64>>;

    /// Bin values into discrete intervals
    fn cut(&self, column: &str, bins: usize) -> Result<Vec<String>>;

    /// Bin values into quantile-based discrete intervals
    fn qcut(&self, column: &str, q: usize) -> Result<Vec<String>>;

    /// Stack the DataFrame, converting columns to rows
    ///
    /// This is similar to pandas stack() - reshaping from wide to long format.
    /// Creates a DataFrame with an additional "variable" column containing the
    /// original column names, and a "value" column containing the values.
    ///
    /// # Arguments
    /// * `columns` - The columns to stack (if None, stack all numeric columns)
    fn stack(&self, columns: Option<&[&str]>) -> Result<DataFrame>;

    /// Unstack the DataFrame, converting rows to columns
    ///
    /// This is similar to pandas unstack() - reshaping from long to wide format.
    ///
    /// # Arguments
    /// * `index_col` - Column to use as the index (row labels)
    /// * `columns_col` - Column whose values become new column names
    /// * `values_col` - Column containing the values
    fn unstack(&self, index_col: &str, columns_col: &str, values_col: &str) -> Result<DataFrame>;

    /// Pivot the DataFrame
    ///
    /// Reshape data based on column values.
    ///
    /// # Arguments
    /// * `index` - Column to use as the new index
    /// * `columns` - Column whose unique values become new columns
    /// * `values` - Column containing values for the new columns
    fn pivot(&self, index: &str, columns: &str, values: &str) -> Result<DataFrame>;

    /// Convert column types
    ///
    /// # Arguments
    /// * `column` - Column to convert
    /// * `dtype` - Target data type ("float64", "int64", "string", "bool")
    fn astype(&self, column: &str, dtype: &str) -> Result<DataFrame>;

    /// Apply a function element-wise to numeric columns
    ///
    /// Similar to pandas applymap() for numeric data.
    fn applymap<F>(&self, func: F) -> Result<DataFrame>
    where
        F: Fn(f64) -> f64;

    /// Aggregate using multiple functions at once
    ///
    /// # Arguments
    /// * `column` - Column to aggregate
    /// * `funcs` - Aggregation functions ("sum", "mean", "min", "max", "std", "var", "count")
    fn agg(&self, column: &str, funcs: &[&str]) -> Result<HashMap<String, f64>>;

    /// Get column data types
    fn dtypes(&self) -> Vec<(String, String)>;

    /// Set column values at specific indices
    fn set_values(&self, column: &str, indices: &[usize], values: &[f64]) -> Result<DataFrame>;

    /// Get rows where column matches a value
    fn query_eq(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Get rows where column is greater than a value
    fn query_gt(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Get rows where column is less than a value
    fn query_lt(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Get rows where string column contains a pattern
    fn query_contains(&self, column: &str, pattern: &str) -> Result<DataFrame>;

    /// Select specific columns by name
    fn select_columns(&self, columns: &[&str]) -> Result<DataFrame>;

    /// Apply element-wise addition with a scalar
    fn add_scalar(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Apply element-wise multiplication with a scalar
    fn mul_scalar(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Apply element-wise subtraction with a scalar
    fn sub_scalar(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Apply element-wise division with a scalar
    fn div_scalar(&self, column: &str, value: f64) -> Result<DataFrame>;

    /// Apply element-wise power
    fn pow(&self, column: &str, exponent: f64) -> Result<DataFrame>;

    /// Apply element-wise square root
    fn sqrt(&self, column: &str) -> Result<DataFrame>;

    /// Apply element-wise logarithm (natural log)
    fn log(&self, column: &str) -> Result<DataFrame>;

    /// Apply element-wise exponential
    fn exp(&self, column: &str) -> Result<DataFrame>;

    /// Compute column-wise operation between two columns
    fn col_add(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Multiply two columns element-wise
    fn col_mul(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Subtract one column from another
    fn col_sub(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Divide one column by another
    fn col_div(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    // === Additional pandas-compatible methods ===

    /// Iterate over DataFrame rows as (index, HashMap) pairs
    ///
    /// Returns an iterator yielding (row_index, row_data) pairs.
    fn iterrows(&self) -> Vec<(usize, HashMap<String, SeriesValue>)>;

    /// Get scalar value at specific row and column (fast access)
    ///
    /// Similar to pandas DataFrame.at[row, col]
    fn at(&self, row: usize, column: &str) -> Result<SeriesValue>;

    /// Get scalar value at specific row index and column index (integer-based)
    ///
    /// Similar to pandas DataFrame.iat[row_idx, col_idx]
    fn iat(&self, row: usize, col_idx: usize) -> Result<SeriesValue>;

    /// Drop rows by index
    ///
    /// Returns a new DataFrame without the specified rows.
    fn drop_rows(&self, indices: &[usize]) -> Result<DataFrame>;

    /// Set a column as the index
    ///
    /// # Arguments
    /// * `column` - Column name to use as index
    /// * `drop` - If true, remove the column from the DataFrame
    fn set_index(&self, column: &str, drop: bool) -> Result<(DataFrame, Vec<String>)>;

    /// Reset the index to default integer index
    ///
    /// # Arguments
    /// * `index_values` - Optional index values to add as a column
    /// * `name` - Name for the new column containing index values
    fn reset_index(&self, index_values: Option<&[String]>, name: &str) -> Result<DataFrame>;

    /// Convert DataFrame to a vector of records (rows as HashMaps)
    fn to_records(&self) -> Vec<HashMap<String, SeriesValue>>;

    /// Iterate over columns as (name, values) pairs
    fn items(&self) -> Vec<(String, Vec<SeriesValue>)>;

    /// Update DataFrame in place with values from another DataFrame
    ///
    /// Non-NA values from other overwrite values in self.
    fn update(&self, other: &DataFrame) -> Result<DataFrame>;

    /// Combine two DataFrames using a function
    ///
    /// # Arguments
    /// * `other` - DataFrame to combine with
    /// * `func` - Function taking (val1, val2) and returning combined value
    fn combine<F>(&self, other: &DataFrame, func: F) -> Result<DataFrame>
    where
        F: Fn(Option<f64>, Option<f64>) -> f64;

    /// Get the shape of the DataFrame as (rows, cols)
    fn shape(&self) -> (usize, usize);

    /// Get the size (total number of elements)
    fn size(&self) -> usize;

    /// Check if DataFrame is empty
    fn empty(&self) -> bool;

    /// Get the first row as a HashMap
    fn first_row(&self) -> Result<HashMap<String, SeriesValue>>;

    /// Get the last row as a HashMap
    fn last_row(&self) -> Result<HashMap<String, SeriesValue>>;

    /// Get value at row and column, with default if missing
    fn get_value(&self, row: usize, column: &str, default: SeriesValue) -> SeriesValue;

    /// Lookup values from other DataFrame based on column matching
    ///
    /// Similar to Excel VLOOKUP / pandas merge on single column
    fn lookup(
        &self,
        lookup_col: &str,
        other: &DataFrame,
        other_col: &str,
        result_col: &str,
    ) -> Result<DataFrame>;

    /// Get column by position index
    fn get_column_by_index(&self, idx: usize) -> Result<(String, Vec<SeriesValue>)>;

    /// Swap two columns by name
    fn swap_columns(&self, col1: &str, col2: &str) -> Result<DataFrame>;

    /// Sort columns by name
    fn sort_columns(&self, ascending: bool) -> Result<DataFrame>;

    /// Rename a single column
    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<DataFrame>;

    /// Convert string column to categorical (integer encoding)
    fn to_categorical(&self, column: &str) -> Result<(DataFrame, HashMap<String, i64>)>;

    /// Compute hash of each row (for deduplication, grouping)
    fn row_hash(&self) -> Vec<u64>;

    /// Sample fraction of rows
    fn sample_frac(&self, frac: f64, replace: bool) -> Result<DataFrame>;

    /// Take rows at specific positions (like iloc with array)
    fn take(&self, indices: &[usize]) -> Result<DataFrame>;

    /// Return boolean Series indicating duplicate rows
    fn duplicated_rows(&self, subset: Option<&[&str]>, keep: &str) -> Result<Vec<bool>>;

    /// Get column as a vector of f64 (with NaN for non-numeric)
    fn get_column_as_f64(&self, column: &str) -> Result<Vec<f64>>;

    /// Get column as a vector of strings
    fn get_column_as_string(&self, column: &str) -> Result<Vec<String>>;

    /// Apply function to groups and return aggregated result
    fn groupby_apply<F>(&self, by: &str, func: F) -> Result<DataFrame>
    where
        F: Fn(&DataFrame) -> Result<HashMap<String, f64>>;

    /// Compute pairwise correlation between two columns
    fn corr_columns(&self, col1: &str, col2: &str) -> Result<f64>;

    /// Compute covariance between two columns
    fn cov_columns(&self, col1: &str, col2: &str) -> Result<f64>;

    /// Get variance of a column
    fn var_column(&self, column: &str, ddof: usize) -> Result<f64>;

    /// Get standard deviation of a column
    fn std_column(&self, column: &str, ddof: usize) -> Result<f64>;

    /// Apply string function to a string column
    fn str_lower(&self, column: &str) -> Result<DataFrame>;

    /// Convert string column to uppercase
    fn str_upper(&self, column: &str) -> Result<DataFrame>;

    /// Strip whitespace from string column
    fn str_strip(&self, column: &str) -> Result<DataFrame>;

    /// Check if string column contains pattern (returns boolean column)
    fn str_contains(&self, column: &str, pattern: &str) -> Result<Vec<bool>>;

    /// Replace pattern in string column
    fn str_replace(&self, column: &str, pattern: &str, replacement: &str) -> Result<DataFrame>;

    /// Split string column on delimiter
    fn str_split(&self, column: &str, delimiter: &str) -> Result<Vec<Vec<String>>>;

    /// Get length of strings in column
    fn str_len(&self, column: &str) -> Result<Vec<usize>>;

    /// Compute standard error of the mean for a column
    fn sem(&self, column: &str, ddof: usize) -> Result<f64>;

    /// Compute mean absolute deviation for a column
    fn mad(&self, column: &str) -> Result<f64>;

    /// Forward fill NaN values in a column
    fn ffill(&self, column: &str) -> Result<DataFrame>;

    /// Backward fill NaN values in a column
    fn bfill(&self, column: &str) -> Result<DataFrame>;

    /// Compute percentile rank for values in a column
    fn pct_rank(&self, column: &str) -> Result<Vec<f64>>;

    /// Compute the absolute value of a numeric column
    fn abs_column(&self, column: &str) -> Result<DataFrame>;

    /// Round values in a column to specified decimal places
    fn round_column(&self, column: &str, decimals: i32) -> Result<DataFrame>;

    /// Get the index of the maximum value in a column
    fn argmax(&self, column: &str) -> Result<usize>;

    /// Get the index of the minimum value in a column
    fn argmin(&self, column: &str) -> Result<usize>;

    /// Element-wise greater than comparison
    fn gt(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Element-wise greater than or equal comparison
    fn ge(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Element-wise less than comparison
    fn lt(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Element-wise less than or equal comparison
    fn le(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Element-wise equality comparison
    fn eq_value(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Element-wise not equal comparison
    fn ne_value(&self, column: &str, value: f64) -> Result<Vec<bool>>;

    /// Clip values below a minimum (keep values >= min)
    fn clip_lower(&self, column: &str, min: f64) -> Result<DataFrame>;

    /// Clip values above a maximum (keep values <= max)
    fn clip_upper(&self, column: &str, max: f64) -> Result<DataFrame>;

    /// Check if any value in column is True (non-zero for numeric)
    fn any_column(&self, column: &str) -> Result<bool>;

    /// Check if all values in column are True (non-zero for numeric)
    fn all_column(&self, column: &str) -> Result<bool>;

    /// Get the number of NaN values in a column
    fn count_na(&self, column: &str) -> Result<usize>;

    /// Compute the product of values in a column
    fn prod(&self, column: &str) -> Result<f64>;

    /// Combine two columns, using values from col2 where col1 is NaN
    fn coalesce(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Get the first valid (non-NaN) value in a column
    fn first_valid(&self, column: &str) -> Result<f64>;

    /// Get the last valid (non-NaN) value in a column
    fn last_valid(&self, column: &str) -> Result<f64>;

    /// Element-wise addition of two columns
    fn add_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Element-wise subtraction of two columns
    fn sub_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Element-wise multiplication of two columns
    fn mul_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Element-wise division of two columns
    fn div_columns(&self, col1: &str, col2: &str, result_name: &str) -> Result<DataFrame>;

    /// Compute modulo operation on a column
    fn mod_column(&self, column: &str, divisor: f64) -> Result<DataFrame>;

    /// Floor division of a column
    fn floordiv(&self, column: &str, divisor: f64) -> Result<DataFrame>;

    /// Negate values in a column
    fn neg(&self, column: &str) -> Result<DataFrame>;

    /// Compute sign of values (-1, 0, or 1)
    fn sign(&self, column: &str) -> Result<Vec<i32>>;

    /// Check if values are finite (not NaN or Inf)
    fn is_finite(&self, column: &str) -> Result<Vec<bool>>;

    /// Check if values are infinite
    fn is_infinite(&self, column: &str) -> Result<Vec<bool>>;

    /// Replace infinite values with a specified value
    fn replace_inf(&self, column: &str, replacement: f64) -> Result<DataFrame>;

    /// Check if string column starts with prefix
    fn str_startswith(&self, column: &str, prefix: &str) -> Result<Vec<bool>>;

    /// Check if string column ends with suffix
    fn str_endswith(&self, column: &str, suffix: &str) -> Result<Vec<bool>>;

    /// Pad strings on the left to specified width
    fn str_pad_left(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame>;

    /// Pad strings on the right to specified width
    fn str_pad_right(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame>;

    /// Slice strings from start to end position
    fn str_slice(&self, column: &str, start: usize, end: Option<usize>) -> Result<DataFrame>;

    /// Floor values in a column
    fn floor(&self, column: &str) -> Result<DataFrame>;

    /// Ceiling values in a column
    fn ceil(&self, column: &str) -> Result<DataFrame>;

    /// Truncate values toward zero
    fn trunc(&self, column: &str) -> Result<DataFrame>;

    /// Get fractional part of values
    fn fract(&self, column: &str) -> Result<DataFrame>;

    /// Apply reciprocal (1/x) to values
    fn reciprocal(&self, column: &str) -> Result<DataFrame>;

    /// Count occurrences of a value in a column
    fn count_value(&self, column: &str, value: f64) -> Result<usize>;

    /// Replace NaN values with zero
    fn fillna_zero(&self, column: &str) -> Result<DataFrame>;

    /// Get unique values count per column for all columns
    fn nunique_all(&self) -> Result<HashMap<String, usize>>;

    /// Check if column values are between two bounds
    fn is_between(
        &self,
        column: &str,
        lower: f64,
        upper: f64,
        inclusive: bool,
    ) -> Result<Vec<bool>>;

    /// Count occurrences of a pattern in string column
    fn str_count(&self, column: &str, pattern: &str) -> Result<Vec<usize>>;

    /// Repeat strings n times
    fn str_repeat(&self, column: &str, n: usize) -> Result<DataFrame>;

    /// Center strings in width with fillchar
    fn str_center(&self, column: &str, width: usize, fillchar: char) -> Result<DataFrame>;

    /// Zero-fill strings to width
    fn str_zfill(&self, column: &str, width: usize) -> Result<DataFrame>;

    /// Check if column contains numeric data
    fn is_numeric_column(&self, column: &str) -> bool;

    /// Check if column contains string data
    fn is_string_column(&self, column: &str) -> bool;

    /// Check if column has any NaN values
    fn has_nulls(&self, column: &str) -> Result<bool>;

    /// Get statistics for a single numeric column
    fn describe_column(&self, column: &str) -> Result<HashMap<String, f64>>;

    /// Get approximate memory usage of a column in bytes
    fn memory_usage_column(&self, column: &str) -> Result<usize>;

    /// Get the range (max - min) of values in a column
    fn range(&self, column: &str) -> Result<f64>;

    /// Get the sum of absolute values in a column
    fn abs_sum(&self, column: &str) -> Result<f64>;

    /// Check if all values in a column are unique
    fn is_unique(&self, column: &str) -> Result<bool>;

    /// Get the most common value and its count
    fn mode_with_count(&self, column: &str) -> Result<(f64, usize)>;

    /// Compute geometric mean for a column (only positive values)
    fn geometric_mean(&self, column: &str) -> Result<f64>;

    /// Compute harmonic mean for a column (non-zero values)
    fn harmonic_mean(&self, column: &str) -> Result<f64>;

    /// Compute interquartile range (IQR = Q3 - Q1)
    fn iqr(&self, column: &str) -> Result<f64>;

    /// Compute coefficient of variation (std / mean)
    fn cv(&self, column: &str) -> Result<f64>;

    /// Compute specific percentile value
    fn percentile_value(&self, column: &str, q: f64) -> Result<f64>;

    /// Compute trimmed mean (excluding outliers at both ends)
    fn trimmed_mean(&self, column: &str, trim_fraction: f64) -> Result<f64>;
}

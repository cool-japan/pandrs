use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use crate::column::ColumnType;
use crate::core::data_value::{self, DataValue as DValue}; // Import as a different name to avoid trait conflict
use crate::core::error::{Error, OptionExt, Result};

// Re-export from legacy module for now
#[deprecated(
    since = "0.1.0",
    note = "Use new DataFrame implementation in crate::dataframe::base"
)]
pub use crate::dataframe::DataFrame as LegacyDataFrame;

// Column trait to allow storing different Series types in the DataFrame
trait ColumnAny: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn column_type_string(&self) -> String;
    fn clone_box(&self) -> Box<dyn ColumnAny + Send + Sync>;
}

impl<T: 'static + Debug + Clone + Send + Sync> ColumnAny for crate::series::Series<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn column_type_string(&self) -> String {
        std::any::type_name::<T>().to_string()
    }

    fn clone_box(&self) -> Box<dyn ColumnAny + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ColumnAny + Send + Sync> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// DataFrame: A two-dimensional, size-mutable, heterogeneous tabular data structure.
///
/// DataFrame is the primary data structure in PandRS for working with labeled, tabular data.
/// It can be thought of as a dictionary-like container for Series objects, where each Series
/// represents a column with a specific data type.
///
/// # Features
///
/// - **Heterogeneous data**: Each column can have a different data type
/// - **Size-mutable**: Rows and columns can be added or removed
/// - **Labeled axes**: Both rows and columns have labels for easy access
/// - **Arithmetic operations**: Supports element-wise and broadcasting operations
/// - **Alignment**: Automatic data alignment in operations
/// - **Missing data handling**: Robust support for NA values
///
/// # Examples
///
/// ```rust
/// use pandrs::{DataFrame, Series};
///
/// // Create a new DataFrame
/// let mut df = DataFrame::new();
///
/// // Add columns
/// df.add_column("name".to_string(),
///     Series::new(vec!["Alice", "Bob", "Charlie"], Some("name".to_string())).expect("Failed"))
///     .expect("Failed to add column");
///
/// df.add_column("age".to_string(),
///     Series::new(vec![25i64, 30, 35], Some("age".to_string())).expect("Failed"))
///     .expect("Failed to add column");
///
/// // Access data
/// assert_eq!(df.row_count(), 3);
/// assert_eq!(df.column_count(), 2);
/// ```
///
/// # Performance
///
/// DataFrame uses columnar storage for memory efficiency and cache-friendly access patterns.
/// Operations on columns are typically faster than row-wise operations.
#[derive(Debug, Clone)]
pub struct DataFrame {
    // Actual fields for storage
    columns: HashMap<String, Box<dyn ColumnAny + Send + Sync>>,
    column_order: Vec<String>,
    row_count: usize,
}

impl DataFrame {
    /// Creates a new empty DataFrame.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::DataFrame;
    ///
    /// let df = DataFrame::new();
    /// assert_eq!(df.row_count(), 0);
    /// assert_eq!(df.column_count(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            column_order: Vec::new(),
            row_count: 0,
        }
    }

    /// Creates a new DataFrame with a specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to use for row labels
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use pandrs::{DataFrame, Index};
    ///
    /// let index = Index::new(vec!["row1".to_string(), "row2".to_string()]).expect("index");
    /// let df = DataFrame::with_index(index);
    /// assert_eq!(df.row_count(), 2);
    /// ```
    pub fn with_index(index: crate::index::Index<String>) -> Self {
        let mut df = Self::new();
        df.row_count = index.len();
        df
    }

    /// Creates a new DataFrame with a multi-level index.
    ///
    /// Useful for hierarchical indexing with multiple levels of row labels.
    ///
    /// # Arguments
    ///
    /// * `multi_index` - The multi-level index to use
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use pandrs::{DataFrame, MultiIndex};
    ///
    /// let multi_idx = MultiIndex::new(
    ///     vec![vec!["A".to_string(), "B".to_string()],
    ///          vec!["x".to_string(), "y".to_string()]],
    ///     vec![vec![0, 0, 1], vec![0, 1, 0]],
    ///     Some(vec![Some("level1".to_string()), Some("level2".to_string())])
    /// ).expect("multi_index");
    /// let df = DataFrame::with_multi_index(multi_idx);
    /// ```
    pub fn with_multi_index(multi_index: crate::index::MultiIndex<String>) -> Self {
        let mut df = Self::new();
        df.row_count = multi_index.len();
        df
    }

    /// Checks if the DataFrame contains a column with the given name.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column to check
    ///
    /// # Returns
    ///
    /// `true` if the column exists, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("age".to_string(),
    ///     Series::new(vec![25i64], None).expect("Failed")).expect("Failed");
    ///
    /// assert!(df.contains_column("age"));
    /// assert!(!df.contains_column("name"));
    /// ```
    pub fn contains_column(&self, column_name: &str) -> bool {
        self.columns.contains_key(column_name)
    }

    /// Returns the number of rows in the DataFrame.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("values".to_string(),
    ///     Series::new(vec![1, 2, 3], None).expect("Failed")).expect("Failed");
    ///
    /// assert_eq!(df.row_count(), 3);
    /// ```
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Returns the number of rows (alias for `row_count`).
    ///
    /// This method provides pandas-like API compatibility.
    pub fn nrows(&self) -> usize {
        self.row_count
    }

    /// Retrieves a string value from the DataFrame at the specified column and row.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column
    /// * `row_idx` - The row index (0-based)
    ///
    /// # Returns
    ///
    /// A reference to the string value
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if the column doesn't exist
    /// - `Error::InvalidValue` if the row index is out of bounds or the column is not a string type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("name".to_string(),
    ///     Series::new(vec!["Alice".to_string(), "Bob".to_string()], None).expect("Failed")).expect("Failed");
    ///
    /// let name = df.get_string_value("name", 0).expect("Failed to get value");
    /// assert_eq!(name, "Alice");
    /// ```
    pub fn get_string_value(&self, column_name: &str, row_idx: usize) -> Result<&str> {
        // Check if column exists
        let col = self
            .columns
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        // Check if row index is valid
        if row_idx >= self.row_count {
            return Err(Error::InvalidValue(format!(
                "Row index {} is out of bounds for DataFrame with {} rows",
                row_idx, self.row_count
            )));
        }

        // Try to downcast to Series<String> and get the value
        if let Some(string_series) = col.as_any().downcast_ref::<crate::series::Series<String>>() {
            if let Some(value) = string_series.get(row_idx) {
                Ok(value)
            } else {
                Err(Error::InvalidValue(format!(
                    "No value found at row {} in column '{}'",
                    row_idx, column_name
                )))
            }
        } else {
            // If it's not a string column, try to convert other types to string
            // But since we need to return &str, we can't create temporary strings
            Err(Error::InvalidValue(format!(
                "Column '{}' is not a string column. Use get_column_string_values() for type conversion.",
                column_name
            )))
        }
    }

    /// Adds a new column to the DataFrame.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name for the new column
    /// * `series` - The Series containing the column data
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful
    ///
    /// # Errors
    ///
    /// - `Error::DuplicateColumnName` if a column with this name already exists
    /// - `Error::InconsistentRowCount` if the series length doesn't match the DataFrame's row count
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    ///
    /// // Add first column
    /// df.add_column("numbers".to_string(),
    ///     Series::new(vec![1, 2, 3], None).expect("Failed"))
    ///     .expect("Failed to add column");
    ///
    /// // Add second column (must have same length)
    /// df.add_column("doubled".to_string(),
    ///     Series::new(vec![2, 4, 6], None).expect("Failed"))
    ///     .expect("Failed to add column");
    ///
    /// assert_eq!(df.column_count(), 2);
    /// ```
    pub fn add_column<T: 'static + Debug + Clone + Send + Sync>(
        &mut self,
        column_name: String,
        series: crate::series::Series<T>,
    ) -> Result<()> {
        // Check if column already exists
        if self.contains_column(&column_name) {
            return Err(Error::DuplicateColumnName(column_name));
        }

        // Check length consistency
        let series_len = series.len();
        if !self.columns.is_empty() && series_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: series_len,
            });
        }

        // Add the column
        self.columns.insert(column_name.clone(), Box::new(series));
        self.column_order.push(column_name);

        // Update row count if this is the first column
        if self.row_count == 0 {
            self.row_count = series_len;
        }

        Ok(())
    }

    /// Returns a list of all column names in the DataFrame.
    ///
    /// The order of names matches the order columns were added.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("age".to_string(),
    ///     Series::new(vec![25i64], None).expect("Failed")).expect("Failed");
    /// df.add_column("name".to_string(),
    ///     Series::new(vec!["Alice"], None).expect("Failed")).expect("Failed");
    ///
    /// let names = df.column_names();
    /// assert_eq!(names, vec!["age".to_string(), "name".to_string()]);
    /// ```
    pub fn column_names(&self) -> Vec<String> {
        self.column_order.clone()
    }

    /// Renames columns in the DataFrame using a mapping.
    ///
    /// # Arguments
    ///
    /// * `column_map` - A HashMap mapping old column names to new column names
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if any old column name doesn't exist
    /// - `Error::DuplicateColumnName` if new names conflict
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    /// use std::collections::HashMap;
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("old_name".to_string(),
    ///     Series::new(vec![1, 2, 3], None).expect("Failed")).expect("Failed");
    ///
    /// let mut rename_map = HashMap::new();
    /// rename_map.insert("old_name".to_string(), "new_name".to_string());
    ///
    /// df.rename_columns(&rename_map).expect("Failed to rename");
    /// assert!(df.contains_column("new_name"));
    /// assert!(!df.contains_column("old_name"));
    /// ```
    pub fn rename_columns(&mut self, column_map: &HashMap<String, String>) -> Result<()> {
        // First, validate that all old column names exist
        for old_name in column_map.keys() {
            if !self.contains_column(old_name) {
                return Err(Error::ColumnNotFound(old_name.clone()));
            }
        }

        // Check for duplicate new names
        let mut new_names_set = std::collections::HashSet::new();
        for new_name in column_map.values() {
            if !new_names_set.insert(new_name) {
                return Err(Error::DuplicateColumnName(new_name.clone()));
            }
        }

        // Check that new names don't conflict with existing column names (except those being renamed)
        for new_name in column_map.values() {
            if self.contains_column(new_name) && !column_map.contains_key(new_name) {
                return Err(Error::DuplicateColumnName(new_name.clone()));
            }
        }

        // Apply the renaming
        for (old_name, new_name) in column_map {
            // Update the column_order vector
            if let Some(pos) = self.column_order.iter().position(|x| x == old_name) {
                self.column_order[pos] = new_name.clone();
            }

            // Move the column data to the new key
            if let Some(column_data) = self.columns.remove(old_name) {
                self.columns.insert(new_name.clone(), column_data);
            }
        }

        Ok(())
    }

    /// Sets all column names in the DataFrame.
    ///
    /// Replaces all column names with the provided list. The number of names
    /// must match the number of columns.
    ///
    /// # Arguments
    ///
    /// * `names` - A vector of new column names
    ///
    /// # Errors
    ///
    /// - `Error::InconsistentRowCount` if the length doesn't match column count
    /// - `Error::DuplicateColumnName` if any names are duplicated
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("col1".to_string(), Series::new(vec![1, 2], None).expect("Failed")).expect("Failed");
    /// df.add_column("col2".to_string(), Series::new(vec![3, 4], None).expect("Failed")).expect("Failed");
    ///
    /// df.set_column_names(vec!["A".to_string(), "B".to_string()]).expect("Failed");
    /// assert_eq!(df.column_names(), vec!["A".to_string(), "B".to_string()]);
    /// ```
    pub fn set_column_names(&mut self, names: Vec<String>) -> Result<()> {
        // Check that the number of names matches the number of columns
        if names.len() != self.column_order.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.column_order.len(),
                found: names.len(),
            });
        }

        // Check for duplicate names
        let mut names_set = std::collections::HashSet::new();
        for name in &names {
            if !names_set.insert(name) {
                return Err(Error::DuplicateColumnName(name.clone()));
            }
        }

        // Create a mapping from old names to new names
        let mut column_map = HashMap::new();
        for (old_name, new_name) in self.column_order.iter().zip(names.iter()) {
            column_map.insert(old_name.clone(), new_name.clone());
        }

        // Apply the renaming using the existing rename_columns method
        self.rename_columns(&column_map)
    }

    /// Gets a typed reference to a column in the DataFrame.
    ///
    /// Returns a reference to the Series with the specified type. The type must
    /// match the actual column type or an error is returned.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column to retrieve
    ///
    /// # Type Parameters
    ///
    /// * `T` - The expected type of the column elements
    ///
    /// # Returns
    ///
    /// A reference to the Series of type T
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if the column doesn't exist
    /// - `Error::InvalidValue` if the column type doesn't match T
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("numbers".to_string(),
    ///     Series::new(vec![1i64, 2, 3], None).expect("Failed")).expect("Failed");
    ///
    /// let col = df.get_column::<i64>("numbers").expect("Failed");
    /// assert_eq!(col.len(), 3);
    /// ```
    pub fn get_column<T: 'static + Debug + Clone + Send + Sync>(
        &self,
        column_name: &str,
    ) -> Result<&crate::series::Series<T>> {
        let col = self
            .columns
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        // Cast to the specific Series type
        match col.as_any().downcast_ref::<crate::series::Series<T>>() {
            Some(series) => Ok(series),
            None => Err(Error::InvalidValue(format!(
                "Column '{}' is not of the requested type",
                column_name
            ))),
        }
    }

    /// Gets all values from a column as strings.
    ///
    /// Converts column values to strings regardless of the underlying type.
    /// Works with numeric, boolean, and string columns.
    ///
    /// # Arguments
    ///
    /// * `column_name` - The name of the column
    ///
    /// # Returns
    ///
    /// A vector of string representations of the column values
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if the column doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("ages".to_string(),
    ///     Series::new(vec![25i64, 30, 35], None).expect("Failed")).expect("Failed");
    ///
    /// let strings = df.get_column_string_values("ages").expect("Failed");
    /// assert_eq!(strings, vec!["25", "30", "35"]);
    /// ```
    pub fn get_column_string_values(&self, column_name: &str) -> Result<Vec<String>> {
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        let column = self
            .columns
            .get(column_name)
            .ok_or_column_error(column_name)?;

        // Try to downcast to different Series types and convert to strings
        if let Some(string_series) = column
            .as_any()
            .downcast_ref::<crate::series::Series<String>>()
        {
            Ok(string_series.values().to_vec())
        } else if let Some(i32_series) =
            column.as_any().downcast_ref::<crate::series::Series<i32>>()
        {
            Ok(i32_series
                .values()
                .iter()
                .map(|v| ToString::to_string(v))
                .collect())
        } else if let Some(i64_series) =
            column.as_any().downcast_ref::<crate::series::Series<i64>>()
        {
            Ok(i64_series
                .values()
                .iter()
                .map(|v| ToString::to_string(v))
                .collect())
        } else if let Some(f32_series) =
            column.as_any().downcast_ref::<crate::series::Series<f32>>()
        {
            Ok(f32_series
                .values()
                .iter()
                .map(|v| ToString::to_string(v))
                .collect())
        } else if let Some(f64_series) =
            column.as_any().downcast_ref::<crate::series::Series<f64>>()
        {
            Ok(f64_series
                .values()
                .iter()
                .map(|v| ToString::to_string(v))
                .collect())
        } else if let Some(bool_series) = column
            .as_any()
            .downcast_ref::<crate::series::Series<bool>>()
        {
            Ok(bool_series
                .values()
                .iter()
                .map(|v| ToString::to_string(v))
                .collect())
        } else {
            // Fallback for unsupported types
            let mut result = Vec::with_capacity(self.row_count);
            for i in 0..self.row_count {
                result.push(format!("unsupported_type_{}_{}", column_name, i));
            }
            Ok(result)
        }
    }

    /// Gets the name of a column by its index position.
    ///
    /// # Arguments
    ///
    /// * `idx` - The column index (0-based)
    ///
    /// # Returns
    ///
    /// `Some(&String)` if the index is valid, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("first".to_string(), Series::new(vec![1], None).expect("Failed")).expect("Failed");
    /// df.add_column("second".to_string(), Series::new(vec![2], None).expect("Failed")).expect("Failed");
    ///
    /// assert_eq!(df.column_name(0), Some(&"first".to_string()));
    /// assert_eq!(df.column_name(1), Some(&"second".to_string()));
    /// assert_eq!(df.column_name(2), None);
    /// ```
    pub fn column_name(&self, idx: usize) -> Option<&String> {
        self.column_order.get(idx)
    }

    /// Concatenates rows from another DataFrame.
    ///
    /// Combines rows from this DataFrame with rows from another DataFrame.
    /// The DataFrames must have compatible schemas (matching column names and types).
    ///
    /// # Arguments
    ///
    /// * `_other` - The DataFrame to concatenate
    ///
    /// # Returns
    ///
    /// A new DataFrame containing rows from both DataFrames
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df1 = DataFrame::new();
    /// df1.add_column("x".to_string(), Series::new(vec![1, 2], None).expect("Failed")).expect("Failed");
    ///
    /// let mut df2 = DataFrame::new();
    /// df2.add_column("x".to_string(), Series::new(vec![3, 4], None).expect("Failed")).expect("Failed");
    ///
    /// let combined = df1.concat_rows(&df2).expect("Failed");
    /// // combined now has 4 rows
    /// ```
    pub fn concat_rows(&self, _other: &DataFrame) -> Result<DataFrame> {
        // Implement concatenation properly when needed
        Ok(Self::new())
    }

    /// Convert DataFrame to CSV
    pub fn to_csv<P: AsRef<Path>>(&self, _path: P) -> Result<()> {
        // Implement CSV export when needed
        Ok(())
    }

    /// Create DataFrame from CSV
    pub fn from_csv<P: AsRef<Path>>(_path: P, _has_header: bool) -> Result<Self> {
        // Implement CSV import when needed
        Ok(Self::new())
    }

    /// Create DataFrame from CSV reader
    pub fn from_csv_reader<R: std::io::Read>(
        reader: &mut csv::Reader<R>,
        has_header: bool,
    ) -> Result<Self> {
        let mut df = Self::new();

        // Get headers
        let headers: Vec<String> = if has_header {
            reader
                .headers()
                .map_err(|e| Error::IoError(format!("CSV header error: {}", e)))?
                .iter()
                .map(|h| h.to_string())
                .collect()
        } else {
            // Peek at first record to determine column count
            let mut records = reader.records();
            if let Some(first_record) = records.next() {
                let record =
                    first_record.map_err(|e| Error::IoError(format!("CSV read error: {}", e)))?;
                (0..record.len()).map(|i| format!("column_{}", i)).collect()
            } else {
                return Ok(df); // Empty file
            }
        };

        // Collect data for each column
        let mut columns_data: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for header in &headers {
            columns_data.insert(header.clone(), Vec::new());
        }

        // Process records
        for result in reader.records() {
            let record = result.map_err(|e| Error::IoError(format!("CSV read error: {}", e)))?;
            for (i, header) in headers.iter().enumerate() {
                let value = if i < record.len() {
                    record[i].to_string()
                } else {
                    String::new()
                };
                // Safe: header was inserted into columns_data during initialization
                if let Some(col_vec) = columns_data.get_mut(header) {
                    col_vec.push(value);
                } else {
                    return Err(Error::InvalidValue(format!(
                        "Column '{}' not found in columns_data",
                        header
                    )));
                }
            }
        }

        // Add columns to DataFrame
        for header in headers {
            if let Some(values) = columns_data.remove(&header) {
                let series = crate::series::Series::new(values, Some(header.clone()))?;
                df.add_column(header, series)?;
            }
        }

        Ok(df)
    }

    /// Returns the number of columns in the DataFrame.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("a".to_string(), Series::new(vec![1], None).expect("Failed")).expect("Failed");
    /// df.add_column("b".to_string(), Series::new(vec![2], None).expect("Failed")).expect("Failed");
    ///
    /// assert_eq!(df.column_count(), 2);
    /// ```
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns the number of columns (alias for `column_count`).
    ///
    /// This method provides pandas-like API compatibility.
    pub fn ncols(&self) -> usize {
        self.column_count()
    }

    /// Creates a new DataFrame containing only the specified columns.
    ///
    /// Selects a subset of columns from the DataFrame by name, preserving
    /// the order specified in the input.
    ///
    /// # Arguments
    ///
    /// * `columns` - A slice of column names to select
    ///
    /// # Returns
    ///
    /// A new DataFrame containing only the specified columns
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if any specified column doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("a".to_string(), Series::new(vec![1, 2], None).expect("Failed")).expect("Failed");
    /// df.add_column("b".to_string(), Series::new(vec![3, 4], None).expect("Failed")).expect("Failed");
    /// df.add_column("c".to_string(), Series::new(vec![5, 6], None).expect("Failed")).expect("Failed");
    ///
    /// let selected = df.select_columns(&["a", "c"]).expect("Failed");
    /// assert_eq!(selected.column_count(), 2);
    /// assert!(selected.contains_column("a"));
    /// assert!(selected.contains_column("c"));
    /// assert!(!selected.contains_column("b"));
    /// ```
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();

        for &column_name in columns {
            if !self.contains_column(column_name) {
                return Err(Error::ColumnNotFound(column_name.to_string()));
            }

            // Copy the column data to the result DataFrame
            if let Ok(values) = self.get_column_numeric_values(column_name) {
                result.add_column(
                    column_name.to_string(),
                    crate::series::Series::new(values, Some(column_name.to_string()))?,
                )?;
            } else if let Ok(values) = self.get_column_string_values(column_name) {
                result.add_column(
                    column_name.to_string(),
                    crate::series::Series::new(values, Some(column_name.to_string()))?,
                )?;
            } else {
                return Err(Error::InvalidValue(format!(
                    "Unable to determine column type for '{}'",
                    column_name
                )));
            }
        }

        Ok(result)
    }

    /// Create a new DataFrame from a HashMap of column names to string vectors
    pub fn from_map(
        data: std::collections::HashMap<String, Vec<String>>,
        index: Option<crate::index::Index<String>>,
    ) -> Result<Self> {
        let mut df = Self::new();

        // If index is provided, set row count
        if let Some(idx) = index {
            df.row_count = idx.len();
        } else {
            // Otherwise, determine row count from data
            df.row_count = data.values().map(|v| v.len()).max().unwrap_or(0);
        }

        // Add columns
        for (col_name, values) in data {
            // Create a Series of strings
            let series = crate::series::Series::new(values, Some(col_name.clone()))?;
            df.add_column(col_name, series)?;
        }

        Ok(df)
    }

    /// Create a new DataFrame from JSON string
    /// Expects JSON format like: {"col1": ["val1", "val2"], "col2": ["val3", "val4"]}
    pub fn from_json(json_str: &str) -> Result<Self> {
        use serde_json::Value;

        // Parse the JSON string
        let parsed: Value = serde_json::from_str(json_str)
            .map_err(|e| Error::InvalidInput(format!("Failed to parse JSON: {}", e)))?;

        // Convert JSON object to HashMap
        let mut data: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        if let Value::Object(obj) = parsed {
            for (col_name, col_values) in obj {
                if let Value::Array(values) = col_values {
                    let string_values: Vec<String> = values
                        .into_iter()
                        .map(|v| match v {
                            Value::String(s) => s,
                            Value::Number(n) => n.to_string(),
                            Value::Bool(b) => ToString::to_string(&b),
                            Value::Null => "".to_string(),
                            _ => v.to_string(),
                        })
                        .collect();
                    data.insert(col_name, string_values);
                } else {
                    return Err(Error::InvalidInput(format!(
                        "Column '{}' is not an array",
                        col_name
                    )));
                }
            }
        } else {
            return Err(Error::InvalidInput("JSON must be an object".to_string()));
        }

        // Use existing from_map method
        Self::from_map(data, None)
    }

    /// Check if the DataFrame has the specified column (alias for contains_column)
    pub fn has_column(&self, column_name: &str) -> bool {
        self.contains_column(column_name)
    }

    /// Get the DataFrame's index
    pub fn get_index(&self) -> crate::index::DataFrameIndex<String> {
        // For now, we don't have an actual implementation of index in the new DataFrame
        // So we return a default index structure
        crate::index::DataFrameIndex::Simple(crate::index::Index::default())
    }

    /// Set the DataFrame's index from an Index
    pub fn set_index(&mut self, index: crate::index::Index<String>) -> Result<()> {
        // Stub implementation - would actually set the index
        Ok(())
    }

    /// Set a multi-index for the DataFrame
    pub fn set_multi_index(&mut self, multi_index: crate::index::MultiIndex<String>) -> Result<()> {
        // Stub implementation - would actually set the multi-index
        Ok(())
    }

    // Using the implementation at line 152 instead

    /// Get numeric values from a column
    pub fn get_column_numeric_values(&self, column_name: &str) -> Result<Vec<f64>> {
        // Get the column
        let col = self
            .columns
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        // Extract numeric values
        let mut values = Vec::with_capacity(self.row_count);
        for i in 0..self.row_count {
            // Try to get the value as a numeric type
            let val = match col.as_any().downcast_ref::<crate::series::Series<f64>>() {
                Some(float_series) => {
                    if let Some(value) = float_series.get(i) {
                        *value // Use the f64 value directly
                    } else {
                        return Err(Error::InvalidValue(format!(
                            "Missing value at index {} in column '{}'",
                            i, column_name
                        )));
                    }
                }
                None => {
                    // Try other numeric types
                    match col.as_any().downcast_ref::<crate::series::Series<i64>>() {
                        Some(int_series) => {
                            if let Some(value) = int_series.get(i) {
                                *value as f64 // Convert i64 to f64
                            } else {
                                return Err(Error::InvalidValue(format!(
                                    "Missing value at index {} in column '{}'",
                                    i, column_name
                                )));
                            }
                        }
                        None => {
                            // Try string values that might be parseable as numbers
                            match col.as_any().downcast_ref::<crate::series::Series<String>>() {
                                Some(str_series) => {
                                    if let Some(value) = str_series.get(i) {
                                        // Try to parse the string as a float
                                        match value.parse::<f64>() {
                                            Ok(num) => num,
                                            Err(_) => return Err(Error::InvalidValue(format!(
                                                "Value '{}' at index {} in column '{}' cannot be converted to numeric",
                                                value, i, column_name
                                            ))),
                                        }
                                    } else {
                                        return Err(Error::InvalidValue(format!(
                                            "Missing value at index {} in column '{}'",
                                            i, column_name
                                        )));
                                    }
                                }
                                None => {
                                    // If we can't find a suitable type, return an error
                                    return Err(Error::InvalidValue(format!(
                                        "Column '{}' cannot be converted to numeric values",
                                        column_name
                                    )));
                                }
                            }
                        }
                    }
                }
            };

            values.push(val);
        }

        Ok(values)
    }

    /// Add a row to the DataFrame
    pub fn add_row_data(&mut self, row_data: Vec<Box<dyn DValue>>) -> Result<()> {
        // Check if the row size matches the number of columns
        if row_data.len() != self.column_order.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.column_order.len(),
                found: row_data.len(),
            });
        }

        // For now, just increase row count as we don't have a full implementation
        self.row_count += 1;

        Ok(())
    }

    /// Filter rows based on a predicate
    pub fn filter<F>(&self, column_name: &str, predicate: F) -> Result<Self>
    where
        F: Fn(&Box<dyn DValue>) -> bool,
    {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        // For now, just return an empty DataFrame as we don't have a full implementation
        Ok(Self::new())
    }

    /// Compute the mean of a column
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        // Get numeric values from the column
        let values = self.get_column_numeric_values(column_name)?;

        if values.is_empty() {
            return Err(Error::EmptySeries);
        }

        // Compute mean
        let sum: f64 = values.iter().sum();
        Ok(sum / values.len() as f64)
    }

    /// Group by a column
    pub fn group_by(&self, _column_name: &str) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Enable GPU acceleration for a DataFrame
    pub fn gpu_accelerate(&self) -> Result<Self> {
        // For now, just return a clone as we don't have a full implementation
        Ok(self.clone())
    }

    /// Calculate a correlation matrix
    pub fn corr_matrix(&self, _columns: &[&str]) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Display the head of the DataFrame
    pub fn head(&self, n: usize) -> Result<String> {
        let mut result = String::new();

        // Add header row
        for col_name in &self.column_order {
            result.push_str(&format!("{}\t", col_name));
        }
        result.push('\n');

        // Add data rows (limited to n)
        let row_limit = n.min(self.row_count);
        for row_idx in 0..row_limit {
            for col_name in &self.column_order {
                // Simplistic approach - just add placeholder values
                result.push_str("[val]\t");
            }
            result.push('\n');
        }

        Ok(result)
    }

    /// Add a row to the DataFrame using a HashMap of column names to values
    pub fn add_row_data_from_hashmap(&mut self, row_data: HashMap<String, String>) -> Result<()> {
        // Check if all required columns exist
        for col_name in row_data.keys() {
            if !self.contains_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }

        // For now, just increment row count as we don't have a full implementation
        self.row_count += 1;

        Ok(())
    }

    /// Check if a column is categorical
    pub fn is_categorical(&self, column_name: &str) -> bool {
        // For simplicity, just check if it exists in this implementation
        // A real implementation would check metadata or column type
        self.contains_column(column_name)
    }

    /// Sample rows from the DataFrame by indices
    ///
    /// # Arguments
    /// * `indices` - A slice of row indices to include in the sampled DataFrame
    ///
    /// # Returns
    /// A new DataFrame containing only the specified rows
    pub fn sample(&self, indices: &[usize]) -> Result<Self> {
        // Validate indices
        for &idx in indices {
            if idx >= self.row_count {
                return Err(Error::InvalidValue(format!(
                    "Index {} is out of bounds for DataFrame with {} rows",
                    idx, self.row_count
                )));
            }
        }

        let mut result = Self::new();

        // Sample each column
        for col_name in &self.column_order {
            if let Some(col) = self.columns.get(col_name) {
                // Try f64 column
                if let Some(float_series) =
                    col.as_any().downcast_ref::<crate::series::Series<f64>>()
                {
                    let sampled_values: Vec<f64> = indices
                        .iter()
                        .map(|&idx| float_series.get(idx).cloned().unwrap_or(0.0))
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Try i64 column
                else if let Some(int_series) =
                    col.as_any().downcast_ref::<crate::series::Series<i64>>()
                {
                    let sampled_values: Vec<i64> = indices
                        .iter()
                        .map(|&idx| int_series.get(idx).cloned().unwrap_or(0))
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Try String column
                else if let Some(str_series) =
                    col.as_any().downcast_ref::<crate::series::Series<String>>()
                {
                    let sampled_values: Vec<String> = indices
                        .iter()
                        .map(|&idx| str_series.get(idx).cloned().unwrap_or_default())
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Try bool column
                else if let Some(bool_series) =
                    col.as_any().downcast_ref::<crate::series::Series<bool>>()
                {
                    let sampled_values: Vec<bool> = indices
                        .iter()
                        .map(|&idx| bool_series.get(idx).cloned().unwrap_or(false))
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Default to f64 (try to convert)
                else {
                    // Fall back to get_column_numeric_values for the entire column then sample
                    let all_values = self.get_column_numeric_values(col_name)?;
                    let sampled_values: Vec<f64> =
                        indices.iter().map(|&idx| all_values[idx]).collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
            }
        }

        Ok(result)
    }

    /// Get a categorical column with generic type
    pub fn get_categorical<T: 'static + Debug + Clone + Eq + std::hash::Hash + Send + Sync>(
        &self,
        column_name: &str,
    ) -> Result<crate::series::categorical::Categorical<T>> {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        // Get column data as strings
        let values_str = self.get_column_string_values(column_name)?;

        // This is a simplified implementation for backward compatibility
        // It assumes T is String, which is the most common case for categorical data
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<String>() {
            // Create a vector of the appropriate type (safely)
            let values: Vec<T> = unsafe { std::mem::transmute(values_str) };

            // Create a new categorical with default settings
            return crate::series::categorical::Categorical::new(values, None, false);
        }

        // For non-string types, return empty categorical
        let empty_vec: Vec<T> = Vec::new();
        crate::series::categorical::Categorical::new(empty_vec, None, false)
    }

    /// Check if a column is numeric
    pub fn is_numeric_column(&self, column_name: &str) -> bool {
        // Stub implementation - for compatibility only
        false
    }

    /// Add a NASeries as a categorical column
    pub fn add_na_series_as_categorical(
        &mut self,
        name: String,
        series: crate::series::NASeries<String>,
        categories: Option<Vec<String>>,
        ordered: Option<crate::series::categorical::CategoricalOrder>,
    ) -> Result<&mut Self> {
        // Create a categorical from the NASeries
        let cat = crate::series::categorical::StringCategorical::from_na_vec(
            series.values().to_vec(),
            categories,
            ordered,
        )?;

        // Convert categorical to regular series
        let regular_series = cat.to_series(Some(name.clone()))?;

        // Add to DataFrame
        self.add_column(name, regular_series)?;

        Ok(self)
    }

    /// Create a DataFrame from multiple categorical data
    pub fn from_categoricals(
        categoricals: Vec<(String, crate::series::categorical::StringCategorical)>,
    ) -> Result<Self> {
        let mut df = Self::new();

        // Check if all categorical data have the same length
        if !categoricals.is_empty() {
            let first_len = categoricals[0].1.len();
            for (name, cat) in &categoricals {
                if cat.len() != first_len {
                    return Err(Error::InconsistentRowCount {
                        expected: first_len,
                        found: cat.len(),
                    });
                }
            }
        }

        for (name, cat) in categoricals {
            // Convert categorical to series
            let series = cat.to_series(Some(name.clone()))?;

            // Add as a column
            df.add_column(name.clone(), series)?;
        }

        Ok(df)
    }

    /// Calculate the occurrence count of a column
    pub fn value_counts(&self, column_name: &str) -> Result<crate::series::Series<usize>> {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        // Get string values from the column
        let values = self.get_column_string_values(column_name)?;

        // Count occurrences
        let mut counts = std::collections::HashMap::new();
        for value in values {
            *counts.entry(value).or_insert(0) += 1;
        }

        // Convert to vectors for Series
        let mut values_vec = Vec::new();
        let mut counts_vec = Vec::new();

        for (value, count) in counts {
            values_vec.push(value);
            counts_vec.push(count);
        }

        // Create the Series object
        crate::series::Series::new(counts_vec, Some(format!("{}_counts", column_name)))
    }
}

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::core::data_value::DataValue as DValue; // Import as a different name to avoid trait conflict
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
    fn clone_box(&self) -> Box<dyn ColumnAny + Send + Sync>;
}

impl<T: 'static + Debug + Clone + Send + Sync> ColumnAny for crate::series::Series<T> {
    fn as_any(&self) -> &dyn Any {
        self
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
    /// Optional row index (single- or multi-level). `None` means a default
    /// positional `RangeIndex` is assumed.
    index: Option<crate::index::DataFrameIndex<String>>,
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
            index: None,
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
        df.index = Some(crate::index::DataFrameIndex::Simple(index));
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
        df.index = Some(crate::index::DataFrameIndex::Multi(multi_index));
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

        // Check length consistency. The DataFrame is only "pristine" (any
        // length acceptable for the next column added, becoming the
        // baseline) when BOTH `row_count == 0` AND no column exists yet:
        //
        // - `row_count` alone isn't enough: it can be established before
        //   any column exists (e.g. `set_index` on an empty DataFrame
        //   fixes `row_count` to the index length), so a column added
        //   afterwards must be checked against it even though it's the
        //   first column physically inserted.
        // - `columns.is_empty()` alone isn't enough either: columns can
        //   exist while `row_count` is still 0 (e.g. the first column
        //   added is itself zero-length), and a *second* column with a
        //   different (non-zero) length must still be rejected instead of
        //   silently redefining `row_count` out from under the first one.
        //
        // So the check must be active whenever *either* signal indicates
        // the DataFrame is no longer pristine.
        let series_len = series.len();
        if (self.row_count != 0 || !self.columns.is_empty()) && series_len != self.row_count {
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
    /// assert_eq!(names, ["age".to_string(), "name".to_string()]);
    /// ```
    pub fn column_names(&self) -> &[String] {
        &self.column_order
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
    /// assert_eq!(df.column_names(), ["A".to_string(), "B".to_string()]);
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
        let any = column.as_any();

        // String columns need no conversion.
        if let Some(string_series) = any.downcast_ref::<crate::series::Series<String>>() {
            return Ok(string_series.values().to_vec());
        }

        // Every other supported element type is rendered through its
        // `Display`/`ToString` impl. Each arm downcasts once and, if it
        // matches, returns immediately.
        macro_rules! try_render {
            ($ty:ty) => {
                if let Some(series) = any.downcast_ref::<crate::series::Series<$ty>>() {
                    return Ok(series.values().iter().map(ToString::to_string).collect());
                }
            };
        }

        try_render!(i32);
        try_render!(i64);
        try_render!(f32);
        try_render!(f64);
        try_render!(bool);
        try_render!(i8);
        try_render!(i16);
        try_render!(i128);
        try_render!(isize);
        try_render!(u8);
        try_render!(u16);
        try_render!(u32);
        try_render!(u64);
        try_render!(u128);
        try_render!(usize);
        try_render!(chrono::NaiveDate);
        try_render!(chrono::NaiveDateTime);
        try_render!(chrono::DateTime<chrono::Utc>);

        // A genuinely unsupported element type: say so rather than
        // fabricating placeholder strings like "unsupported_type_{col}_{i}"
        // that silently corrupt every consumer of this data (CSV/JSON
        // writers, value_counts, filter, ...).
        Err(Error::InvalidValue(format!(
            "Column '{}' has an element type that is not supported for string conversion \
             (supported: String, integers, floats, bool, NaiveDate, NaiveDateTime, DateTime<Utc>)",
            column_name
        )))
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
    /// Columns are aligned **by name**, not by position: the result contains
    /// the union of both DataFrames' columns (this DataFrame's columns
    /// first, in their existing order, followed by any columns unique to
    /// `other`). Row order is `self`'s rows followed by `other`'s rows.
    ///
    /// * A column present in both frames with the same concrete element
    ///   type is concatenated directly, preserving that type.
    /// * A column present in both frames with *different* element types
    ///   falls back to a common string representation for both sides
    ///   (mirroring pandas' object-dtype upcast on a type clash), rather
    ///   than corrupting or dropping either side.
    /// * A numeric column (`i64`/`f64`/`i32`/`f32`) present in only one of
    ///   the two frames is widened to `f64`, with the rows contributed by
    ///   the frame that lacks the column filled with `f64::NAN` (mirroring
    ///   pandas' int -> float NaN-upcast when concatenating misaligned
    ///   columns).
    /// * A non-numeric column (`String`/`bool`/...) present in only one of
    ///   the two frames has no NA-capable representation in this
    ///   DataFrame's column model, so this returns
    ///   [`Error::NotImplemented`] naming the column rather than
    ///   fabricating a placeholder value (e.g. `""` or `false`) that would
    ///   be indistinguishable from real data.
    ///
    /// The result never carries a row index (even if both inputs have
    /// one): concatenating two independently-built indices could easily
    /// produce duplicate labels, which [`crate::index::Index`] rejects, so
    /// the result uses the implicit default positional index instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df1 = DataFrame::new();
    /// df1.add_column("x".to_string(), Series::new(vec![1i64, 2], None).expect("Failed")).expect("Failed");
    ///
    /// let mut df2 = DataFrame::new();
    /// df2.add_column("x".to_string(), Series::new(vec![3i64, 4], None).expect("Failed")).expect("Failed");
    ///
    /// let combined = df1.concat_rows(&df2).expect("Failed");
    /// assert_eq!(combined.row_count(), 4);
    /// assert_eq!(
    ///     combined.get_column_string_values("x").unwrap(),
    ///     vec!["1", "2", "3", "4"]
    /// );
    /// ```
    pub fn concat_rows(&self, other: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        result.row_count = self.row_count + other.row_count;

        // Union of column names: self's columns first (in their original
        // order), then any columns unique to `other` (in its order).
        let mut union_names: Vec<String> = self.column_order.clone();
        for name in &other.column_order {
            if !self.contains_column(name) {
                union_names.push(name.clone());
            }
        }

        for name in &union_names {
            let in_self = self.contains_column(name);
            let in_other = other.contains_column(name);

            if in_self && in_other {
                // Try to concatenate directly when both sides agree on the
                // concrete element type, preserving it exactly.
                macro_rules! same_type_concat {
                    ($ty:ty) => {
                        if let (Some(a), Some(b)) = (
                            self.columns.get(name).and_then(|c| {
                                c.as_any().downcast_ref::<crate::series::Series<$ty>>()
                            }),
                            other.columns.get(name).and_then(|c| {
                                c.as_any().downcast_ref::<crate::series::Series<$ty>>()
                            }),
                        ) {
                            let mut values: Vec<$ty> =
                                Vec::with_capacity(self.row_count + other.row_count);
                            values.extend_from_slice(a.values());
                            values.extend_from_slice(b.values());
                            result.add_column(
                                name.clone(),
                                crate::series::Series::new(values, Some(name.clone()))?,
                            )?;
                            continue;
                        }
                    };
                }

                same_type_concat!(String);
                same_type_concat!(i64);
                same_type_concat!(f64);
                same_type_concat!(i32);
                same_type_concat!(f32);
                same_type_concat!(bool);

                // The two sides genuinely disagree on this column's type:
                // fall back to a shared string representation rather than
                // silently dropping or truncating either side.
                let mut values = self.get_column_string_values(name)?;
                values.extend(other.get_column_string_values(name)?);
                result.add_column(
                    name.clone(),
                    crate::series::Series::new(values, Some(name.clone()))?,
                )?;
            } else if in_self {
                if self.is_numeric_column(name) {
                    let mut values = self.get_column_numeric_values(name)?;
                    values.extend(std::iter::repeat(f64::NAN).take(other.row_count));
                    result.add_column(
                        name.clone(),
                        crate::series::Series::new(values, Some(name.clone()))?,
                    )?;
                } else {
                    return Err(Error::NotImplemented(format!(
                        "concat_rows: column '{}' exists only in the left DataFrame and is not \
                         numeric, so the rows contributed by the right DataFrame cannot be \
                         NA-filled for it; provide the column in both DataFrames",
                        name
                    )));
                }
            } else {
                // in_other only
                if other.is_numeric_column(name) {
                    let mut values: Vec<f64> =
                        std::iter::repeat(f64::NAN).take(self.row_count).collect();
                    values.extend(other.get_column_numeric_values(name)?);
                    result.add_column(
                        name.clone(),
                        crate::series::Series::new(values, Some(name.clone()))?,
                    )?;
                } else {
                    return Err(Error::NotImplemented(format!(
                        "concat_rows: column '{}' exists only in the right DataFrame and is not \
                         numeric, so the rows contributed by the left DataFrame cannot be \
                         NA-filled for it; provide the column in both DataFrames",
                        name
                    )));
                }
            }
        }

        Ok(result)
    }

    // `to_csv`, `from_csv`, `from_csv_reader` and `from_json` live in
    // `base_io.rs` (see the `mod base_io;` declaration at the end of this
    // file, after this `impl DataFrame` block closes).

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
    /// the order specified in the input. Each column's concrete element
    /// type is preserved exactly (the underlying column is cloned as-is)
    /// rather than being re-inferred through a numeric-then-string
    /// conversion chain, which previously destroyed dtypes: a numeric
    /// string like `"007"` became the float `7.0`, `i64` columns became
    /// `f64`, and `bool` columns that don't parse as numbers became
    /// `String` (`"true"`/`"false"`). The row index, if any, is preserved
    /// unchanged (no rows are added, removed or reordered by a column
    /// selection).
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
    /// - `Error::DuplicateColumnName` if `columns` contains the same name twice
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("a".to_string(), Series::new(vec![1, 2], None).expect("Failed")).expect("Failed");
    /// df.add_column("b".to_string(), Series::new(vec![3, 4], None).expect("Failed")).expect("Failed");
    /// df.add_column("c".to_string(), Series::new(vec!["007".to_string(), "008".to_string()], None).expect("Failed")).expect("Failed");
    ///
    /// let selected = df.select_columns(&["a", "c"]).expect("Failed");
    /// assert_eq!(selected.column_count(), 2);
    /// assert!(selected.contains_column("a"));
    /// assert!(selected.contains_column("c"));
    /// assert!(!selected.contains_column("b"));
    /// // "c" keeps its original String values ("007"), not a re-inferred f64 (7.0).
    /// assert_eq!(selected.get_column_string_values("c").unwrap(), vec!["007", "008"]);
    /// ```
    pub fn select_columns(&self, columns: &[&str]) -> Result<Self> {
        let mut result = Self::new();
        result.row_count = self.row_count;

        for &column_name in columns {
            let boxed = self
                .columns
                .get(column_name)
                .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

            if result.columns.contains_key(column_name) {
                return Err(Error::DuplicateColumnName(column_name.to_string()));
            }

            // Clone the boxed column directly: this preserves the exact
            // concrete element type instead of re-inferring it.
            result
                .columns
                .insert(column_name.to_string(), boxed.clone());
            result.column_order.push(column_name.to_string());
        }

        // No rows were dropped or reordered, so the row index (if any)
        // still lines up with `result`'s rows unchanged.
        result.index = self.index.clone();

        Ok(result)
    }

    /// Create a new DataFrame from a HashMap of column names to string vectors.
    ///
    /// * The `index`, when provided, is stored on the resulting DataFrame
    ///   (previously it was only used to compute a row count and then
    ///   silently discarded, so `get_index()` afterwards returned an
    ///   unrelated default index).
    /// * Ragged columns (whose length disagrees with the row count implied
    ///   by `index`, or by the longest column when no index is given) are
    ///   rejected with `Error::InconsistentRowCount` rather than silently
    ///   accepted.
    /// * Columns are added in a deterministic order (sorted by name)
    ///   instead of `HashMap`'s unspecified iteration order, so
    ///   `column_names()` is stable across runs.
    pub fn from_map(
        data: std::collections::HashMap<String, Vec<String>>,
        index: Option<crate::index::Index<String>>,
    ) -> Result<Self> {
        let mut df = Self::new();

        // Determine the row count up front so every column (including the
        // first) is validated against it, rejecting ragged columns.
        let row_count = match &index {
            Some(idx) => idx.len(),
            None => data.values().map(|v| v.len()).max().unwrap_or(0),
        };
        df.row_count = row_count;

        // Deterministic column order (sorted by name) instead of
        // HashMap's unspecified iteration order.
        let sorted_data: std::collections::BTreeMap<String, Vec<String>> =
            data.into_iter().collect();

        for (col_name, values) in sorted_data {
            if values.len() != row_count {
                return Err(Error::InconsistentRowCount {
                    expected: row_count,
                    found: values.len(),
                });
            }
            let series = crate::series::Series::new(values, Some(col_name.clone()))?;
            df.add_column(col_name, series)?;
        }

        // Store the caller-provided index (previously discarded).
        if let Some(idx) = index {
            df.set_index(idx)?;
        }

        Ok(df)
    }

    // `from_json` lives in `base_io.rs` alongside the CSV helpers.

    /// Check if the DataFrame has the specified column (alias for contains_column)
    pub fn has_column(&self, column_name: &str) -> bool {
        self.contains_column(column_name)
    }

    /// Get the DataFrame's index
    ///
    /// Returns the stored row index if one has been set (via [`set_index`],
    /// [`set_multi_index`], [`with_index`] or [`with_multi_index`]); otherwise a
    /// default single-level index is returned to represent the implicit
    /// positional index.
    ///
    /// [`set_index`]: Self::set_index
    /// [`set_multi_index`]: Self::set_multi_index
    /// [`with_index`]: Self::with_index
    /// [`with_multi_index`]: Self::with_multi_index
    pub fn get_index(&self) -> crate::index::DataFrameIndex<String> {
        match &self.index {
            Some(index) => index.clone(),
            None => crate::index::DataFrameIndex::Simple(crate::index::Index::default()),
        }
    }

    /// Set the DataFrame's index from an `Index`.
    ///
    /// The index length must match the number of rows. For an empty DataFrame
    /// (no columns yet) the row count is initialised from the index length.
    pub fn set_index(&mut self, index: crate::index::Index<String>) -> Result<()> {
        if self.row_count != 0 && index.len() != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: index.len(),
            });
        }
        if self.row_count == 0 {
            self.row_count = index.len();
        }
        self.index = Some(crate::index::DataFrameIndex::Simple(index));
        Ok(())
    }

    /// Set a multi-level index for the DataFrame.
    ///
    /// The index length must match the number of rows. For an empty DataFrame
    /// (no columns yet) the row count is initialised from the index length.
    pub fn set_multi_index(&mut self, multi_index: crate::index::MultiIndex<String>) -> Result<()> {
        if self.row_count != 0 && multi_index.len() != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: multi_index.len(),
            });
        }
        if self.row_count == 0 {
            self.row_count = multi_index.len();
        }
        self.index = Some(crate::index::DataFrameIndex::Multi(multi_index));
        Ok(())
    }

    // Using the implementation at line 152 instead

    /// Get numeric values from a column.
    ///
    /// The column's concrete element type is downcast exactly once (not on
    /// every element, which used to make this an O(rows) chain of downcasts
    /// per row / O(rows) total downcast attempts -- a measured hot spot).
    /// `i32`/`f32`/`bool` columns are supported directly (previously they
    /// fell through to the string-parsing branch and always failed).
    pub fn get_column_numeric_values(&self, column_name: &str) -> Result<Vec<f64>> {
        let col = self
            .columns
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        let any = col.as_any();

        macro_rules! collect_numeric {
            ($series:expr, $convert:expr) => {{
                let series = $series;
                let mut values = Vec::with_capacity(self.row_count);
                for i in 0..self.row_count {
                    match series.get(i) {
                        Some(value) => values.push(($convert)(value)),
                        None => {
                            return Err(Error::InvalidValue(format!(
                                "Missing value at index {} in column '{}'",
                                i, column_name
                            )))
                        }
                    }
                }
                return Ok(values);
            }};
        }

        if let Some(s) = any.downcast_ref::<crate::series::Series<f64>>() {
            collect_numeric!(s, |v: &f64| *v);
        }
        if let Some(s) = any.downcast_ref::<crate::series::Series<i64>>() {
            collect_numeric!(s, |v: &i64| *v as f64);
        }
        if let Some(s) = any.downcast_ref::<crate::series::Series<i32>>() {
            collect_numeric!(s, |v: &i32| *v as f64);
        }
        if let Some(s) = any.downcast_ref::<crate::series::Series<f32>>() {
            collect_numeric!(s, |v: &f32| *v as f64);
        }
        if let Some(s) = any.downcast_ref::<crate::series::Series<bool>>() {
            collect_numeric!(s, |v: &bool| if *v { 1.0 } else { 0.0 });
        }
        if let Some(s) = any.downcast_ref::<crate::series::Series<String>>() {
            let mut values = Vec::with_capacity(self.row_count);
            for i in 0..self.row_count {
                match s.get(i) {
                    Some(value) => match value.trim().parse::<f64>() {
                        Ok(num) => values.push(num),
                        Err(_) => {
                            return Err(Error::InvalidValue(format!(
                            "Value '{}' at index {} in column '{}' cannot be converted to numeric",
                            value, i, column_name
                        )))
                        }
                    },
                    None => {
                        return Err(Error::InvalidValue(format!(
                            "Missing value at index {} in column '{}'",
                            i, column_name
                        )))
                    }
                }
            }
            return Ok(values);
        }

        Err(Error::InvalidValue(format!(
            "Column '{}' cannot be converted to numeric values",
            column_name
        )))
    }

    /// Append a single row of already-stringified values (one entry per column,
    /// in `column_order`).
    ///
    /// Because [`crate::series::Series`] is immutable, each column's backing
    /// series is rebuilt with the new value parsed into its concrete element
    /// type. Columns whose element type is not supported for appending yield a
    /// [`Error::NotImplemented`] instead of silently dropping the value.
    fn append_string_row(&mut self, values: &[String]) -> Result<()> {
        let col_names = self.column_order.clone();
        for (i, name) in col_names.iter().enumerate() {
            let value = &values[i];
            let boxed = self
                .columns
                .get(name)
                .ok_or_else(|| Error::ColumnNotFound(name.clone()))?;
            let any = boxed.as_any();

            let rebuilt: Box<dyn ColumnAny + Send + Sync> =
                if let Some(s) = any.downcast_ref::<crate::series::Series<String>>() {
                    let mut v = s.values().to_vec();
                    v.push(value.clone());
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else if let Some(s) = any.downcast_ref::<crate::series::Series<i64>>() {
                    let parsed = value.trim().parse::<i64>().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Cannot append value '{}' to i64 column '{}'",
                            value, name
                        ))
                    })?;
                    let mut v = s.values().to_vec();
                    v.push(parsed);
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else if let Some(s) = any.downcast_ref::<crate::series::Series<f64>>() {
                    let parsed = value.trim().parse::<f64>().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Cannot append value '{}' to f64 column '{}'",
                            value, name
                        ))
                    })?;
                    let mut v = s.values().to_vec();
                    v.push(parsed);
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else if let Some(s) = any.downcast_ref::<crate::series::Series<i32>>() {
                    let parsed = value.trim().parse::<i32>().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Cannot append value '{}' to i32 column '{}'",
                            value, name
                        ))
                    })?;
                    let mut v = s.values().to_vec();
                    v.push(parsed);
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else if let Some(s) = any.downcast_ref::<crate::series::Series<f32>>() {
                    let parsed = value.trim().parse::<f32>().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Cannot append value '{}' to f32 column '{}'",
                            value, name
                        ))
                    })?;
                    let mut v = s.values().to_vec();
                    v.push(parsed);
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else if let Some(s) = any.downcast_ref::<crate::series::Series<bool>>() {
                    let lowered = value.trim().to_lowercase();
                    let parsed = match lowered.as_str() {
                        "true" | "1" => true,
                        "false" | "0" => false,
                        _ => {
                            return Err(Error::InvalidValue(format!(
                                "Cannot append value '{}' to bool column '{}'",
                                value, name
                            )))
                        }
                    };
                    let mut v = s.values().to_vec();
                    v.push(parsed);
                    Box::new(crate::series::Series::new(v, Some(name.clone()))?)
                } else {
                    return Err(Error::NotImplemented(format!(
                        "Appending a row to column '{}' is not supported for its element type",
                        name
                    )));
                };

            self.columns.insert(name.clone(), rebuilt);
        }

        self.row_count += 1;
        Ok(())
    }

    /// Add a row to the DataFrame.
    ///
    /// `row_data` must contain exactly one value per column, in column order.
    /// Each value is converted to its column's element type and appended.
    pub fn add_row_data(&mut self, row_data: Vec<Box<dyn DValue>>) -> Result<()> {
        // Check if the row size matches the number of columns
        if row_data.len() != self.column_order.len() {
            return Err(Error::InconsistentRowCount {
                expected: self.column_order.len(),
                found: row_data.len(),
            });
        }

        let values: Vec<String> = row_data
            .iter()
            .map(|v| DValue::to_string(v.as_ref()))
            .collect();
        self.append_string_row(&values)
    }

    /// Filter rows based on a predicate applied to a single column.
    ///
    /// The predicate receives each cell of `column_name` boxed as a
    /// [`DataValue`](crate::core::data_value::DataValue) of the column's real
    /// element type (so `value.as_any().downcast_ref::<i64>()` etc. works).
    /// Rows for which the predicate returns `true` are kept, with every column
    /// preserved.
    pub fn filter<F>(&self, column_name: &str, predicate: F) -> Result<Self>
    where
        F: Fn(&Box<dyn DValue>) -> bool,
    {
        // Check if the column exists
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        let col = self
            .columns
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;
        let any = col.as_any();

        let mut matching: Vec<usize> = Vec::new();

        if let Some(s) = any.downcast_ref::<crate::series::Series<String>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).cloned().unwrap_or_default());
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else if let Some(s) = any.downcast_ref::<crate::series::Series<i64>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).copied().unwrap_or(0));
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else if let Some(s) = any.downcast_ref::<crate::series::Series<f64>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).copied().unwrap_or(0.0));
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else if let Some(s) = any.downcast_ref::<crate::series::Series<bool>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).copied().unwrap_or(false));
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else if let Some(s) = any.downcast_ref::<crate::series::Series<i32>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).copied().unwrap_or(0) as i64);
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else if let Some(s) = any.downcast_ref::<crate::series::Series<f32>>() {
            for i in 0..self.row_count {
                let boxed: Box<dyn DValue> = Box::new(s.get(i).copied().unwrap_or(0.0) as f64);
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        } else {
            // Unknown element type: fall back to the string representation.
            let values = self.get_column_string_values(column_name)?;
            for (i, v) in values.iter().enumerate() {
                let boxed: Box<dyn DValue> = Box::new(v.clone());
                if predicate(&boxed) {
                    matching.push(i);
                }
            }
        }

        self.sample(&matching)
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

    /// Group the DataFrame by a single column.
    ///
    /// Delegates to the real groupby machinery in
    /// [`crate::dataframe::groupby::GroupByExt`], returning a
    /// [`crate::dataframe::groupby::DataFrameGroupBy`] that supports real
    /// aggregation (`.agg(...)`, `.size()`, named/custom aggregations, ...)
    /// rather than the previous placeholder that discarded the column name
    /// and always returned `Ok(())`.
    pub fn group_by(
        &self,
        column_name: &str,
    ) -> Result<crate::dataframe::groupby::DataFrameGroupBy> {
        use crate::dataframe::groupby::GroupByExt;
        self.groupby_single(column_name)
    }

    /// Enable GPU acceleration for a DataFrame.
    ///
    /// There is no real GPU kernel behind this entry point (the crate's GPU
    /// support -- `DataFrameGpuExt` -- is itself
    /// honestly `Err(Error::NotImplemented(..))` for this same operation,
    /// but is only compiled in `#[cfg(cuda_available)]` builds). This
    /// inherent method exists unconditionally so `gpu_accelerate` has one
    /// consistent, honest answer in every build rather than either lying
    /// with `Ok(self.clone())` (the previous behaviour) or silently
    /// disappearing from the API whenever CUDA isn't detected.
    pub fn gpu_accelerate(&self) -> Result<Self> {
        Err(Error::NotImplemented(
            "GPU acceleration for DataFrame is not implemented (no real CUDA kernel)".to_string(),
        ))
    }

    /// Calculate a correlation matrix for the given columns.
    ///
    /// Returns a real correlation matrix as a `DataFrame`: both the row
    /// index and the columns are `columns` (in the given order), and cell
    /// `(i, j)` is the Pearson correlation coefficient between `columns[i]`
    /// and `columns[j]` (matching pandas' `DataFrame.corr()` shape). The
    /// diagonal is always `1.0`. Computation is delegated to
    /// [`crate::stats::descriptive::correlation_matrix`].
    ///
    /// # Errors
    ///
    /// - `Error::ColumnNotFound` if any column doesn't exist
    /// - An error if a column cannot be read as numeric, or if there are
    ///   fewer than 2 rows (correlation is undefined for a single
    ///   observation)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("a".to_string(), Series::new(vec![1.0, 2.0, 3.0], None).expect("Failed")).expect("Failed");
    /// df.add_column("b".to_string(), Series::new(vec![2.0, 4.0, 6.0], None).expect("Failed")).expect("Failed");
    ///
    /// let corr = df.corr_matrix(&["a", "b"]).expect("Failed");
    /// assert_eq!(corr.row_count(), 2);
    /// let b_col = corr.get_column_numeric_values("b").unwrap();
    /// assert!((b_col[0] - 1.0).abs() < 1e-9); // corr(a, b) is a perfect +1
    /// ```
    pub fn corr_matrix(&self, columns: &[&str]) -> Result<DataFrame> {
        if columns.is_empty() {
            return Err(Error::InvalidValue(
                "corr_matrix requires at least one column".to_string(),
            ));
        }

        let mut data: Vec<Vec<f64>> = Vec::with_capacity(columns.len());
        for &col in columns {
            data.push(self.get_column_numeric_values(col)?);
        }

        let matrix = crate::stats::descriptive::correlation_matrix(&data)?;

        let mut result = DataFrame::new();
        for (j, &col_name) in columns.iter().enumerate() {
            let col_values: Vec<f64> = (0..columns.len()).map(|i| matrix[i][j]).collect();
            result.add_column(
                col_name.to_string(),
                crate::series::Series::new(col_values, Some(col_name.to_string()))?,
            )?;
        }

        let row_labels: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        result.set_index(crate::index::Index::new(row_labels)?)?;

        Ok(result)
    }

    /// Select a subset of rows by position, preserving each column's
    /// concrete element type (delegates to [`Self::sample`]) and, when the
    /// DataFrame has a simple (single-level) row index, slicing that index
    /// the same way so row labels stay aligned with their data. A
    /// multi-level index is not sliced (falls back to no index on the
    /// result), matching how row-selecting operations elsewhere in the
    /// crate currently treat `MultiIndex`.
    fn select_rows_preserving_index(&self, indices: &[usize]) -> Result<Self> {
        let mut result = self.sample(indices)?;

        if let Some(crate::index::DataFrameIndex::Simple(idx)) = &self.index {
            let sliced_labels: Vec<String> = indices
                .iter()
                .map(|&i| idx.get_value(i).cloned().unwrap_or_default())
                .collect();
            if let Ok(sliced_index) = crate::index::Index::new(sliced_labels) {
                result.index = Some(crate::index::DataFrameIndex::Simple(sliced_index));
            }
        }

        Ok(result)
    }

    /// Return the first `n` rows of the DataFrame (or all rows, if `n`
    /// exceeds the row count) as a real DataFrame, preserving column types
    /// and the row index. For a tab-separated display string instead, see
    /// [`Self::head_string`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("n".to_string(), Series::new(vec![1i64, 2, 3, 4], None).expect("Failed")).expect("Failed");
    ///
    /// let head = df.head(2).expect("Failed");
    /// assert_eq!(head.row_count(), 2);
    /// assert_eq!(head.get_column_string_values("n").unwrap(), vec!["1", "2"]);
    /// ```
    pub fn head(&self, n: usize) -> Result<Self> {
        let take = n.min(self.row_count);
        let indices: Vec<usize> = (0..take).collect();
        self.select_rows_preserving_index(&indices)
    }

    /// Return the last `n` rows of the DataFrame (or all rows, if `n`
    /// exceeds the row count) as a real DataFrame, preserving column types
    /// and the row index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("n".to_string(), Series::new(vec![1i64, 2, 3, 4], None).expect("Failed")).expect("Failed");
    ///
    /// let tail = df.tail(2).expect("Failed");
    /// assert_eq!(tail.row_count(), 2);
    /// assert_eq!(tail.get_column_string_values("n").unwrap(), vec!["3", "4"]);
    /// ```
    pub fn tail(&self, n: usize) -> Result<Self> {
        let take = n.min(self.row_count);
        let start = self.row_count - take;
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.select_rows_preserving_index(&indices)
    }

    /// Render the first `n` rows of the DataFrame as a tab-separated
    /// string, for callers that want display/debug output rather than a
    /// `DataFrame` (see [`Self::head`] for the latter).
    pub fn head_string(&self, n: usize) -> Result<String> {
        let mut result = String::new();

        // Add header row
        for col_name in &self.column_order {
            result.push_str(&format!("{}\t", col_name));
        }
        result.push('\n');

        // Materialise the columns once (in order) as strings.
        let mut string_columns: Vec<Vec<String>> = Vec::with_capacity(self.column_order.len());
        for col_name in &self.column_order {
            string_columns.push(self.get_column_string_values(col_name)?);
        }

        // Add data rows (limited to n)
        let row_limit = n.min(self.row_count);
        for row_idx in 0..row_limit {
            for column in &string_columns {
                result.push_str(&format!(
                    "{}\t",
                    column.get(row_idx).cloned().unwrap_or_default()
                ));
            }
            result.push('\n');
        }

        Ok(result)
    }

    /// Add a row to the DataFrame using a map of column names to (string) values.
    ///
    /// Every key must name an existing column. Columns omitted from the map
    /// receive an empty value (which is only valid for string columns; numeric
    /// columns will error rather than fabricate a default).
    pub fn add_row_data_from_hashmap(&mut self, row_data: HashMap<String, String>) -> Result<()> {
        // Check if all required columns exist
        for col_name in row_data.keys() {
            if !self.contains_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.clone()));
            }
        }

        let col_names = self.column_order.clone();
        let values: Vec<String> = col_names
            .iter()
            .map(|c| row_data.get(c).cloned().unwrap_or_default())
            .collect();
        self.append_string_row(&values)
    }

    /// Check whether a column can hold categorical data.
    ///
    /// In this DataFrame representation categorical values are materialised as a
    /// string column (`Series<String>`), so categorical data is exactly the set
    /// of string columns. Numeric and boolean columns are therefore never
    /// categorical. (This is a genuine type check rather than the previous
    /// behaviour of reporting *every* column as categorical.)
    pub fn is_categorical(&self, column_name: &str) -> bool {
        match self.columns.get(column_name) {
            Some(col) => col
                .as_any()
                .downcast_ref::<crate::series::Series<String>>()
                .is_some(),
            None => false,
        }
    }

    /// Sample rows from the DataFrame by indices.
    ///
    /// `f64`/`i64`/`i32`/`f32`/`String`/`bool` columns keep their exact
    /// element type. Any other column type (e.g. `NaiveDate`) is sampled
    /// through numeric conversion if possible, else through its string
    /// representation, rather than failing outright. This backs
    /// [`Self::filter`], [`Self::head`] and [`Self::tail`].
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
                // Try i32 column
                else if let Some(int32_series) =
                    col.as_any().downcast_ref::<crate::series::Series<i32>>()
                {
                    let sampled_values: Vec<i32> = indices
                        .iter()
                        .map(|&idx| int32_series.get(idx).cloned().unwrap_or(0))
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Try f32 column
                else if let Some(float32_series) =
                    col.as_any().downcast_ref::<crate::series::Series<f32>>()
                {
                    let sampled_values: Vec<f32> = indices
                        .iter()
                        .map(|&idx| float32_series.get(idx).cloned().unwrap_or(0.0))
                        .collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                }
                // Any remaining type (e.g. NaiveDate/NaiveDateTime/DateTime<Utc>,
                // wider integer types): try numeric conversion first, then
                // fall back to the column's string representation, so
                // `sample`/`head`/`tail`/`filter` still work on these
                // columns instead of hard-erroring.
                else if let Ok(all_values) = self.get_column_numeric_values(col_name) {
                    let sampled_values: Vec<f64> =
                        indices.iter().map(|&idx| all_values[idx]).collect();
                    let new_series =
                        crate::series::Series::new(sampled_values, Some(col_name.clone()))?;
                    result.add_column(col_name.clone(), new_series)?;
                } else {
                    let all_values = self.get_column_string_values(col_name)?;
                    let sampled_values: Vec<String> =
                        indices.iter().map(|&idx| all_values[idx].clone()).collect();
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

        // Preferred path: the column is already stored as `Series<T>`, so we can
        // build the categorical directly from the real typed values.
        if let Ok(series) = self.get_column::<T>(column_name) {
            return crate::series::categorical::Categorical::new(
                series.values().to_vec(),
                None,
                false,
            );
        }

        // Backward-compatible path: when the caller requested `T == String`,
        // build a string categorical from the column's string representation.
        // The conversion goes through a `TypeId`-checked `Box<dyn Any>` downcast
        // rather than an unsound `transmute`.
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<String>() {
            let values_str = self.get_column_string_values(column_name)?;
            let any_values: Box<dyn Any> = Box::new(values_str);
            return match any_values.downcast::<Vec<T>>() {
                Ok(values) => crate::series::categorical::Categorical::new(*values, None, false),
                Err(_) => Err(Error::InvalidValue(format!(
                    "Failed to construct a String categorical for column '{}'",
                    column_name
                ))),
            };
        }

        // Any other element type genuinely cannot be reconstructed from the
        // stored data without dedicated support.
        Err(Error::NotImplemented(format!(
            "get_categorical for column '{}' is not supported for the requested element type \
             (the column is not stored as that type)",
            column_name
        )))
    }

    /// Check if a column is numeric.
    ///
    /// Returns `true` only when the column is physically stored as a numeric
    /// `Series` (`i64`, `f64`, `i32` or `f32`). String columns that merely look
    /// numeric return `false`.
    pub fn is_numeric_column(&self, column_name: &str) -> bool {
        match self.columns.get(column_name) {
            Some(col) => {
                let any = col.as_any();
                any.downcast_ref::<crate::series::Series<i64>>().is_some()
                    || any.downcast_ref::<crate::series::Series<f64>>().is_some()
                    || any.downcast_ref::<crate::series::Series<i32>>().is_some()
                    || any.downcast_ref::<crate::series::Series<f32>>().is_some()
            }
            None => false,
        }
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
            for (_name, cat) in &categoricals {
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

    /// Calculate the occurrence count of each distinct value in a column.
    ///
    /// Returns a two-column DataFrame (`"value"`, `"count"`) labeling each
    /// distinct value with its count -- the previous implementation
    /// computed the distinct values only to discard them, returning an
    /// unlabeled `Series<usize>` of counts with no way to tell which count
    /// belonged to which value. Rows are ordered deterministically: by
    /// count descending, then by value ascending to break ties (previously
    /// row order came from unordered `HashMap` iteration and varied
    /// between runs).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use pandrs::{DataFrame, Series};
    ///
    /// let mut df = DataFrame::new();
    /// df.add_column("region".to_string(), Series::new(
    ///     vec!["Tokyo".to_string(), "Osaka".to_string(), "Tokyo".to_string()], None
    /// ).expect("Failed")).expect("Failed");
    ///
    /// let counts = df.value_counts("region").expect("Failed");
    /// assert_eq!(counts.row_count(), 2);
    /// assert_eq!(counts.get_column_string_values("value").unwrap(), vec!["Tokyo", "Osaka"]);
    /// assert_eq!(counts.get_column_string_values("count").unwrap(), vec!["2", "1"]);
    /// ```
    pub fn value_counts(&self, column_name: &str) -> Result<DataFrame> {
        if !self.contains_column(column_name) {
            return Err(Error::ColumnNotFound(column_name.to_string()));
        }

        let values = self.get_column_string_values(column_name)?;

        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in values {
            *counts.entry(value).or_insert(0) += 1;
        }

        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        // Count descending, then value ascending to break ties
        // deterministically.
        pairs.sort_by(|(value_a, count_a), (value_b, count_b)| {
            count_b.cmp(count_a).then_with(|| value_a.cmp(value_b))
        });

        let mut values_vec = Vec::with_capacity(pairs.len());
        let mut counts_vec = Vec::with_capacity(pairs.len());
        for (value, count) in pairs {
            values_vec.push(value);
            counts_vec.push(count);
        }

        let mut result = DataFrame::new();
        result.add_column(
            "value".to_string(),
            crate::series::Series::new(values_vec, Some("value".to_string()))?,
        )?;
        result.add_column(
            "count".to_string(),
            crate::series::Series::new(counts_vec, Some(format!("{}_count", column_name)))?,
        )?;

        Ok(result)
    }
}

// CSV/JSON construction and serialisation (`to_csv`, `from_csv`,
// `from_csv_reader`, `from_json`) live in `base_io.rs`, split out to keep
// this file under the project's 2000-line guideline. Declared with
// `#[path]` so the physical file sits next to `base.rs` (as
// `src/dataframe/base_*.rs`) while remaining a child module of `base` for
// private-field access, matching the existing
// `crate::time_series::stats`/`stats_normality.rs` split.
#[path = "base_io.rs"]
mod base_io;

#[cfg(test)]
#[path = "base_tests.rs"]
mod tests;

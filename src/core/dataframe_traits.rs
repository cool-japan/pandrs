//! Core DataFrame operation traits for PandRS
//!
//! This module provides the foundational trait system for all DataFrame-like operations
//! in PandRS, enabling better extensibility, code reuse, and generic programming.

use crate::core::data_value::DataValue;
use crate::core::error::{Error, Result};
use std::collections::HashMap;

/// Axis specification for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Row = 0,
    Column = 1,
}

/// Join types for merge operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Outer,
    Cross,
}

/// Methods for handling null values in dropna
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropNaHow {
    Any, // Drop if any null value
    All, // Drop only if all values are null
}

/// Fill methods for handling null values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMethod {
    Forward,     // Forward fill
    Backward,    // Backward fill
    Zero,        // Fill with zero
    Mean,        // Fill with mean
    Interpolate, // Interpolate values
}

/// DataFrame information structure
#[derive(Debug, Clone)]
pub struct DataFrameInfo {
    pub shape: (usize, usize),
    pub memory_usage: usize,
    pub null_counts: HashMap<String, usize>,
    pub dtypes: HashMap<String, String>,
}

/// Aggregation function specification
#[derive(Debug, Clone)]
pub enum AggFunc {
    Sum,
    Mean,
    Median,
    Min,
    Max,
    Count,
    Std,
    Var,
    Custom(String), // Custom function name
}

/// Base trait for all DataFrame-like structures in PandRS
pub trait DataFrameOps {
    type Output: DataFrameOps;
    type Error: std::error::Error + Send + Sync + 'static;

    // Core structural operations

    /// Select specific columns by name
    fn select(&self, columns: &[&str]) -> Result<Self::Output>;

    /// Drop specific columns by name
    fn drop(&self, columns: &[&str]) -> Result<Self::Output>;

    /// Rename columns using a mapping
    fn rename(&self, mapping: &HashMap<String, String>) -> Result<Self::Output>;

    // Filtering and selection

    /// Filter rows based on a predicate function
    fn filter<F>(&self, predicate: F) -> Result<Self::Output>
    where
        F: Fn(&dyn DataValue) -> bool + Send + Sync;

    /// Get the first n rows
    fn head(&self, n: usize) -> Result<Self::Output>;

    /// Get the last n rows
    fn tail(&self, n: usize) -> Result<Self::Output>;

    /// Sample n random rows
    fn sample(&self, n: usize, random_state: Option<u64>) -> Result<Self::Output>;

    // Sorting and ordering

    /// Sort by column values
    fn sort_values(&self, by: &[&str], ascending: &[bool]) -> Result<Self::Output>;

    /// Sort by index
    fn sort_index(&self) -> Result<Self::Output>;

    // Shape and metadata

    /// Get the shape (rows, columns) of the DataFrame
    fn shape(&self) -> (usize, usize);

    /// Get the column names
    fn columns(&self) -> Vec<String>;

    /// Get the data types of columns
    fn dtypes(&self) -> HashMap<String, String>;

    /// Get comprehensive DataFrame information
    fn info(&self) -> DataFrameInfo;

    // Missing data handling

    /// Drop rows or columns containing null values
    fn dropna(&self, axis: Option<Axis>, how: DropNaHow) -> Result<Self::Output>;

    /// Fill null values
    fn fillna(&self, value: &dyn DataValue, method: Option<FillMethod>) -> Result<Self::Output>;

    /// Check for null values
    fn isna(&self) -> Result<Self::Output>;

    // Transformation operations

    /// Apply a function to each element
    fn map<F>(&self, func: F) -> Result<Self::Output>
    where
        F: Fn(&dyn DataValue) -> Box<dyn DataValue> + Send + Sync;

    /// Apply a function along an axis
    fn apply<F>(&self, func: F, axis: Axis) -> Result<Self::Output>
    where
        F: Fn(&Self::Output) -> Box<dyn DataValue> + Send + Sync;
}

/// Advanced DataFrame operations for complex data manipulation
pub trait DataFrameAdvancedOps: DataFrameOps {
    type GroupBy: GroupByOps<Self>
    where
        Self: Sized;

    // Merging and joining

    /// Merge with another DataFrame
    fn merge(&self, other: &Self, on: &[&str], how: JoinType) -> Result<Self::Output>;

    /// Concatenate with other DataFrames
    fn concat(&self, others: &[&Self], axis: Axis) -> Result<Self::Output>;

    // Reshaping operations

    /// Pivot the DataFrame
    fn pivot(&self, index: &[&str], columns: &[&str], values: &[&str]) -> Result<Self::Output>;

    /// Melt the DataFrame from wide to long format
    fn melt(&self, id_vars: &[&str], value_vars: &[&str]) -> Result<Self::Output>;

    /// Stack columns into a series
    fn stack(&self, level: Option<usize>) -> Result<Self::Output>;

    /// Unstack index into columns
    fn unstack(&self, level: Option<usize>) -> Result<Self::Output>;

    // Grouping operations

    /// Group by columns
    fn group_by(&self, by: &[&str]) -> Result<Self::GroupBy>
    where
        Self: Sized;

    // Window operations

    /// Create rolling window
    fn rolling(&self, window: usize) -> Result<RollingWindow<Self>>
    where
        Self: Sized;

    /// Create expanding window
    fn expanding(&self) -> Result<ExpandingWindow<Self>>
    where
        Self: Sized;

    // Time series operations

    /// Resample time series data
    fn resample(&self, freq: &str) -> Result<Resampler<Self>>
    where
        Self: Sized;

    /// Shift values by periods
    fn shift(&self, periods: i64) -> Result<Self::Output>;

    // Advanced indexing

    /// Set index from columns
    fn set_index(&self, keys: &[&str]) -> Result<Self::Output>;

    /// Reset index to default integer index
    fn reset_index(&self, drop: bool) -> Result<Self::Output>;

    /// Reindex using a new index
    fn reindex<I: IndexTrait>(&self, index: &I) -> Result<Self::Output>;
}

/// GroupBy operations trait
pub trait GroupByOps<T: DataFrameOps> {
    type GroupByResult: DataFrameOps;
    type Error: std::error::Error + Send + Sync + 'static;

    // Basic aggregations

    /// Sum of groups
    fn sum(&self) -> Result<Self::GroupByResult>;

    /// Mean of groups
    fn mean(&self) -> Result<Self::GroupByResult>;

    /// Median of groups
    fn median(&self) -> Result<Self::GroupByResult>;

    /// Standard deviation of groups
    fn std(&self) -> Result<Self::GroupByResult>;

    /// Variance of groups
    fn var(&self) -> Result<Self::GroupByResult>;

    /// Minimum of groups
    fn min(&self) -> Result<Self::GroupByResult>;

    /// Maximum of groups
    fn max(&self) -> Result<Self::GroupByResult>;

    /// Count of non-null values in groups
    fn count(&self) -> Result<Self::GroupByResult>;

    /// Number of unique values in groups
    fn nunique(&self) -> Result<Self::GroupByResult>;

    // Advanced aggregations

    /// Apply multiple aggregation functions
    fn agg(&self, funcs: &[AggFunc]) -> Result<Self::GroupByResult>;

    /// Transform groups with a function
    fn transform<F>(&self, func: F) -> Result<T>
    where
        F: Fn(&T) -> T + Send + Sync;

    /// Apply function to each group
    fn apply<F>(&self, func: F) -> Result<Self::GroupByResult>
    where
        F: Fn(&T) -> Box<dyn DataValue> + Send + Sync;

    // Filtering

    /// Filter groups based on a condition
    fn filter<F>(&self, func: F) -> Result<T>
    where
        F: Fn(&T) -> bool + Send + Sync;

    // Group information

    /// Get group keys
    fn groups(&self) -> GroupIterator<T>
    where
        T: Clone;

    /// Get specific group by key
    fn get_group(&self, key: &GroupKey) -> Result<T>;

    // Statistics

    /// Describe groups with summary statistics
    fn describe(&self) -> Result<Self::GroupByResult>;

    /// Calculate quantile for groups
    fn quantile(&self, q: f64) -> Result<Self::GroupByResult>;
}

/// Indexing operations for advanced data selection
pub trait IndexingOps {
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    // Position-based indexing (iloc)

    /// Select by integer position
    fn iloc(&self, row_indexer: &RowIndexer, col_indexer: &ColIndexer) -> Result<Self::Output>;

    /// Get scalar value by position
    fn iloc_scalar(&self, row: usize, col: usize) -> Result<Box<dyn DataValue>>;

    // Label-based indexing (loc)

    /// Select by label
    fn loc(&self, row_indexer: &LabelIndexer, col_indexer: &LabelIndexer) -> Result<Self::Output>;

    /// Get scalar value by label
    fn loc_scalar(&self, row_label: &str, col_label: &str) -> Result<Box<dyn DataValue>>;

    // Boolean indexing

    /// Select using boolean mask
    fn mask(&self, mask: &BooleanMask) -> Result<Self::Output>;

    /// Select where condition is true
    fn where_condition<F>(&self, condition: F) -> Result<Self::Output>
    where
        F: Fn(&dyn DataValue) -> bool + Send + Sync;

    // Query-based selection

    /// Query using expression string
    fn query(&self, expression: &str) -> Result<Self::Output>;

    /// Evaluate expression and return series
    fn eval(&self, expression: &str) -> Result<Self::Output>;

    // Fast scalar access

    /// Fast scalar access by label
    fn at(&self, row_label: &str, col_label: &str) -> Result<Box<dyn DataValue>>;

    /// Fast scalar access by position
    fn iat(&self, row: usize, col: usize) -> Result<Box<dyn DataValue>>;
}

// Supporting types and traits

/// Trait for index operations
pub trait IndexTrait: Clone + Send + Sync {
    fn len(&self) -> usize;
    fn get(&self, pos: usize) -> Option<&str>;
    fn position(&self, label: &str) -> Option<usize>;
}

/// Row indexer for iloc operations
#[derive(Debug, Clone)]
pub enum RowIndexer {
    Single(usize),
    Range(std::ops::Range<usize>),
    List(Vec<usize>),
    Slice(usize, Option<usize>, Option<isize>), // start, end, step
}

/// Column indexer for iloc operations
#[derive(Debug, Clone)]
pub enum ColIndexer {
    Single(usize),
    Range(std::ops::Range<usize>),
    List(Vec<usize>),
    All,
}

/// Label indexer for loc operations
#[derive(Debug, Clone)]
pub enum LabelIndexer {
    Single(String),
    List(Vec<String>),
    Slice(String, Option<String>), // start, end
    All,
}

/// Boolean mask for filtering
#[derive(Debug, Clone)]
pub struct BooleanMask {
    pub mask: Vec<bool>,
}

impl BooleanMask {
    pub fn new(mask: Vec<bool>) -> Self {
        Self { mask }
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn get(&self, index: usize) -> Option<bool> {
        self.mask.get(index).copied()
    }
}

/// Group key for groupby operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub values: Vec<String>,
}

impl GroupKey {
    pub fn new(values: Vec<String>) -> Self {
        Self { values }
    }
}

/// Iterator over groups
pub struct GroupIterator<T>
where
    T: Clone,
{
    groups: Vec<(GroupKey, T)>,
    current: usize,
}

impl<T> GroupIterator<T>
where
    T: Clone,
{
    pub fn new(groups: Vec<(GroupKey, T)>) -> Self {
        Self { groups, current: 0 }
    }
}

impl<T> Iterator for GroupIterator<T>
where
    T: Clone,
{
    type Item = (GroupKey, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.groups.len() {
            let result = self.groups.get(self.current)?.clone();
            self.current += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Rolling window for time-based operations
pub struct RollingWindow<T> {
    dataframe: T,
    window_size: usize,
    min_periods: Option<usize>,
}

impl<T> RollingWindow<T> {
    pub fn new(dataframe: T, window_size: usize) -> Self {
        Self {
            dataframe,
            window_size,
            min_periods: None,
        }
    }

    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }
}

impl<T: DataFrameOps> RollingWindow<T> {
    /// Calculate rolling sum
    pub fn sum(&self) -> Result<T::Output> {
        // Implementation would apply rolling sum
        unimplemented!("Rolling sum not yet implemented")
    }

    /// Calculate rolling mean
    pub fn mean(&self) -> Result<T::Output> {
        // Implementation would apply rolling mean
        unimplemented!("Rolling mean not yet implemented")
    }

    /// Calculate rolling standard deviation
    pub fn std(&self) -> Result<T::Output> {
        // Implementation would apply rolling std
        unimplemented!("Rolling std not yet implemented")
    }
}

/// Expanding window for cumulative operations
pub struct ExpandingWindow<T> {
    dataframe: T,
    min_periods: Option<usize>,
}

impl<T> ExpandingWindow<T> {
    pub fn new(dataframe: T) -> Self {
        Self {
            dataframe,
            min_periods: None,
        }
    }

    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }
}

impl<T: DataFrameOps> ExpandingWindow<T> {
    /// Calculate expanding sum
    pub fn sum(&self) -> Result<T::Output> {
        // Implementation would apply expanding sum
        unimplemented!("Expanding sum not yet implemented")
    }

    /// Calculate expanding mean
    pub fn mean(&self) -> Result<T::Output> {
        // Implementation would apply expanding mean
        unimplemented!("Expanding mean not yet implemented")
    }
}

/// Resampler for time series operations
pub struct Resampler<T> {
    dataframe: T,
    frequency: String,
}

impl<T> Resampler<T> {
    pub fn new(dataframe: T, frequency: String) -> Self {
        Self {
            dataframe,
            frequency,
        }
    }
}

impl<T: DataFrameOps> Resampler<T> {
    /// Resample and sum
    pub fn sum(&self) -> Result<T::Output> {
        // Implementation would resample and sum
        unimplemented!("Resample sum not yet implemented")
    }

    /// Resample and take mean
    pub fn mean(&self) -> Result<T::Output> {
        // Implementation would resample and mean
        unimplemented!("Resample mean not yet implemented")
    }

    /// Resample and take first value
    pub fn first(&self) -> Result<T::Output> {
        // Implementation would resample and take first
        unimplemented!("Resample first not yet implemented")
    }

    /// Resample and take last value
    pub fn last(&self) -> Result<T::Output> {
        // Implementation would resample and take last
        unimplemented!("Resample last not yet implemented")
    }
}

/// Trait for statistical operations on DataFrames
pub trait StatisticalOps: DataFrameOps {
    /// Calculate descriptive statistics
    fn describe(&self) -> Result<Self::Output>;

    /// Calculate correlation matrix
    fn corr(&self, method: CorrelationMethod) -> Result<Self::Output>;

    /// Calculate covariance matrix
    fn cov(&self) -> Result<Self::Output>;

    /// Calculate quantiles
    fn quantile(&self, q: &[f64]) -> Result<Self::Output>;

    /// Calculate value counts
    fn value_counts(&self, column: &str) -> Result<Self::Output>;

    /// Check for duplicate rows
    fn duplicated(&self, subset: Option<&[&str]>) -> Result<BooleanMask>;

    /// Remove duplicate rows
    fn drop_duplicates(&self, subset: Option<&[&str]>) -> Result<Self::Output>;
}

/// Correlation calculation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    Pearson,
    Spearman,
    Kendall,
}

/// Trait for I/O operations on DataFrames
pub trait DataFrameIO: DataFrameOps {
    /// Read from CSV file
    fn read_csv(path: &str, options: CsvReadOptions) -> Result<Self::Output>;

    /// Write to CSV file
    fn to_csv(&self, path: &str, options: CsvWriteOptions) -> Result<()>;

    /// Read from JSON file
    fn read_json(path: &str, options: JsonReadOptions) -> Result<Self::Output>;

    /// Write to JSON file
    fn to_json(&self, path: &str, options: JsonWriteOptions) -> Result<()>;

    /// Read from Parquet file
    fn read_parquet(path: &str, options: ParquetReadOptions) -> Result<Self::Output>;

    /// Write to Parquet file
    fn to_parquet(&self, path: &str, options: ParquetWriteOptions) -> Result<()>;
}

/// CSV reading options
#[derive(Debug, Clone, Default)]
pub struct CsvReadOptions {
    pub delimiter: Option<char>,
    pub header: Option<bool>,
    pub skip_rows: Option<usize>,
    pub nrows: Option<usize>,
    pub encoding: Option<String>,
}

/// CSV writing options
#[derive(Debug, Clone, Default)]
pub struct CsvWriteOptions {
    pub delimiter: Option<char>,
    pub header: Option<bool>,
    pub index: Option<bool>,
    pub encoding: Option<String>,
}

/// JSON reading options
#[derive(Debug, Clone, Default)]
pub struct JsonReadOptions {
    pub orient: Option<String>,
    pub encoding: Option<String>,
}

/// JSON writing options
#[derive(Debug, Clone, Default)]
pub struct JsonWriteOptions {
    pub orient: Option<String>,
    pub encoding: Option<String>,
    pub indent: Option<usize>,
}

/// Parquet reading options
#[derive(Debug, Clone, Default)]
pub struct ParquetReadOptions {
    pub columns: Option<Vec<String>>,
    pub use_threads: Option<bool>,
}

/// Parquet writing options
#[derive(Debug, Clone, Default)]
pub struct ParquetWriteOptions {
    pub compression: Option<String>,
    pub use_dictionary: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolean_mask() {
        let mask = BooleanMask::new(vec![true, false, true, false]);
        assert_eq!(mask.len(), 4);
        assert_eq!(mask.get(0), Some(true));
        assert_eq!(mask.get(1), Some(false));
        assert_eq!(mask.get(4), None);
    }

    #[test]
    fn test_group_key() {
        let key1 = GroupKey::new(vec!["A".to_string(), "1".to_string()]);
        let key2 = GroupKey::new(vec!["A".to_string(), "1".to_string()]);
        let key3 = GroupKey::new(vec!["B".to_string(), "2".to_string()]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_indexers() {
        let row_indexer = RowIndexer::Range(0..5);
        let col_indexer = ColIndexer::List(vec![0, 2, 4]);

        match row_indexer {
            RowIndexer::Range(range) => assert_eq!(range, 0..5),
            _ => panic!("Wrong indexer type"),
        }

        match col_indexer {
            ColIndexer::List(list) => assert_eq!(list, vec![0, 2, 4]),
            _ => panic!("Wrong indexer type"),
        }
    }
}

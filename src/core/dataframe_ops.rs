//! Advanced DataFrame Operations Trait Hierarchy
//!
//! This module provides comprehensive trait definitions for DataFrame operations
//! based on the PandRS trait system design specification.

use crate::core::{data_value::DataValue, error::Result, index::IndexTrait};
use std::collections::HashMap;
use std::time::Duration;

/// Axis specification for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Row axis (0)
    Row = 0,
    /// Column axis (1)
    Column = 1,
}

/// How to handle missing data when dropping
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropNaHow {
    /// Drop if any value is missing
    Any,
    /// Drop only if all values are missing
    All,
}

/// Fill method for missing data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMethod {
    /// Forward fill
    Forward,
    /// Backward fill
    Backward,
    /// Interpolate values
    Interpolate,
}

/// Join types for DataFrame operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left join
    Left,
    /// Right join
    Right,
    /// Outer join
    Outer,
    /// Cross join
    Cross,
}

/// DataFrame information structure
#[derive(Debug, Clone)]
pub struct DataFrameInfo {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub columns: usize,
    /// Column names and types
    pub column_info: HashMap<String, String>,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Number of non-null values per column
    pub non_null_counts: HashMap<String, usize>,
}

/// Aggregation function types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggFunc {
    /// Sum aggregation
    Sum,
    /// Mean aggregation
    Mean,
    /// Median aggregation
    Median,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of values
    Count,
    /// Count of unique values
    Nunique,
    /// First value
    First,
    /// Last value
    Last,
    /// Custom aggregation function
    Custom(String),
}

/// Base trait for all DataFrame-like structures in PandRS
pub trait DataFrameOps {
    type Output: DataFrameOps;
    type Error: std::error::Error + Send + Sync + 'static;
    
    // Core structural operations
    fn select(&self, columns: &[&str]) -> Result<Self::Output, Self::Error>;
    fn drop(&self, columns: &[&str]) -> Result<Self::Output, Self::Error>;
    fn rename(&self, mapping: &HashMap<String, String>) -> Result<Self::Output, Self::Error>;
    
    // Filtering and selection
    fn filter<F>(&self, predicate: F) -> Result<Self::Output, Self::Error>
    where
        F: Fn(&DataValue) -> bool;
    fn head(&self, n: usize) -> Result<Self::Output, Self::Error>;
    fn tail(&self, n: usize) -> Result<Self::Output, Self::Error>;
    fn sample(&self, n: usize, random_state: Option<u64>) -> Result<Self::Output, Self::Error>;
    
    // Sorting and ordering
    fn sort_values(&self, by: &[&str], ascending: &[bool]) -> Result<Self::Output, Self::Error>;
    fn sort_index(&self) -> Result<Self::Output, Self::Error>;
    
    // Shape and metadata
    fn shape(&self) -> (usize, usize);
    fn columns(&self) -> Vec<String>;
    fn dtypes(&self) -> HashMap<String, String>;
    fn info(&self) -> DataFrameInfo;
    
    // Missing data handling
    fn dropna(&self, axis: Option<Axis>, how: DropNaHow) -> Result<Self::Output, Self::Error>;
    fn fillna(&self, value: &DataValue, method: Option<FillMethod>) -> Result<Self::Output, Self::Error>;
    fn isna(&self) -> Result<Self::Output, Self::Error>;
    
    // Iteration and transformation
    fn map<F>(&self, func: F) -> Result<Self::Output, Self::Error>
    where
        F: Fn(&DataValue) -> DataValue;
    fn apply<F>(&self, func: F, axis: Axis) -> Result<Self::Output, Self::Error>
    where
        F: Fn(&crate::series::Series) -> DataValue;
}

/// Advanced DataFrame operations for complex data manipulation
pub trait DataFrameAdvancedOps: DataFrameOps {
    // Merging and joining
    fn merge(&self, other: &Self, on: &[&str], how: JoinType) -> Result<Self::Output, Self::Error>;
    fn concat(&self, others: &[&Self], axis: Axis) -> Result<Self::Output, Self::Error>;
    
    // Reshaping operations
    fn pivot(&self, index: &[&str], columns: &[&str], values: &[&str]) -> Result<Self::Output, Self::Error>;
    fn melt(&self, id_vars: &[&str], value_vars: &[&str]) -> Result<Self::Output, Self::Error>;
    fn stack(&self, level: Option<usize>) -> Result<Self::Output, Self::Error>;
    fn unstack(&self, level: Option<usize>) -> Result<Self::Output, Self::Error>;
    
    // Window operations
    fn rolling(&self, window: usize) -> Result<RollingWindow<Self>, Self::Error>;
    fn expanding(&self) -> Result<ExpandingWindow<Self>, Self::Error>;
    
    // Time series operations
    fn resample(&self, freq: &str) -> Result<Resampler<Self>, Self::Error>;
    fn shift(&self, periods: i64) -> Result<Self::Output, Self::Error>;
    
    // Advanced indexing
    fn set_index(&self, keys: &[&str]) -> Result<Self::Output, Self::Error>;
    fn reset_index(&self, drop: bool) -> Result<Self::Output, Self::Error>;
    fn reindex(&self, index: &dyn IndexTrait) -> Result<Self::Output, Self::Error>;
}

/// GroupBy operations trait
pub trait GroupByOps<T: DataFrameOps> {
    type GroupByResult: DataFrameOps;
    type Error: std::error::Error;
    
    // Basic aggregations
    fn sum(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn mean(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn median(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn std(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn var(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn min(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn max(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn count(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn nunique(&self) -> Result<Self::GroupByResult, Self::Error>;
    
    // Advanced aggregations
    fn agg(&self, funcs: &[AggFunc]) -> Result<Self::GroupByResult, Self::Error>;
    fn transform<F>(&self, func: F) -> Result<T, Self::Error>
    where
        F: Fn(&T) -> T;
    fn apply<F>(&self, func: F) -> Result<Self::GroupByResult, Self::Error>
    where
        F: Fn(&T) -> DataValue;
    
    // Filtering
    fn filter<F>(&self, func: F) -> Result<T, Self::Error>
    where
        F: Fn(&T) -> bool;
    
    // Iteration
    fn groups(&self) -> GroupIterator<T>;
    fn get_group(&self, key: &GroupKey) -> Result<T, Self::Error>;
    
    // Statistics
    fn describe(&self) -> Result<Self::GroupByResult, Self::Error>;
    fn quantile(&self, q: f64) -> Result<Self::GroupByResult, Self::Error>;
}

/// Indexing operations for advanced data selection
pub trait IndexingOps {
    type Output;
    type Error: std::error::Error;
    
    // Position-based indexing (iloc)
    fn iloc(&self, row_indexer: &RowIndexer, col_indexer: &ColIndexer) -> Result<Self::Output, Self::Error>;
    fn iloc_scalar(&self, row: usize, col: usize) -> Result<DataValue, Self::Error>;
    
    // Label-based indexing (loc)
    fn loc(&self, row_indexer: &LabelIndexer, col_indexer: &LabelIndexer) -> Result<Self::Output, Self::Error>;
    fn loc_scalar(&self, row_label: &str, col_label: &str) -> Result<DataValue, Self::Error>;
    
    // Boolean indexing
    fn mask(&self, mask: &BooleanMask) -> Result<Self::Output, Self::Error>;
    fn where_condition<F>(&self, condition: F) -> Result<Self::Output, Self::Error>
    where
        F: Fn(&DataValue) -> bool;
    
    // Query-based selection
    fn query(&self, expression: &str) -> Result<Self::Output, Self::Error>;
    fn eval(&self, expression: &str) -> Result<crate::series::Series, Self::Error>;
    
    // Advanced indexing patterns
    fn at(&self, row_label: &str, col_label: &str) -> Result<DataValue, Self::Error>;
    fn iat(&self, row: usize, col: usize) -> Result<DataValue, Self::Error>;
}

// Support types for advanced operations
#[derive(Debug, Clone)]
pub struct RollingWindow<T> {
    dataframe: T,
    window_size: usize,
}

impl<T> RollingWindow<T> {
    pub fn new(dataframe: T, window_size: usize) -> Self {
        Self { dataframe, window_size }
    }
}

#[derive(Debug, Clone)]
pub struct ExpandingWindow<T> {
    dataframe: T,
}

impl<T> ExpandingWindow<T> {
    pub fn new(dataframe: T) -> Self {
        Self { dataframe }
    }
}

#[derive(Debug, Clone)]
pub struct Resampler<T> {
    dataframe: T,
    frequency: String,
}

impl<T> Resampler<T> {
    pub fn new(dataframe: T, frequency: String) -> Self {
        Self { dataframe, frequency }
    }
}

/// Group iterator for iterating over groups
pub struct GroupIterator<T> {
    groups: Vec<(GroupKey, T)>,
    current: usize,
}

impl<T> Iterator for GroupIterator<T> {
    type Item = (GroupKey, T);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.groups.len() {
            let item = self.groups[self.current].clone();
            self.current += 1;
            Some(item)
        } else {
            None
        }
    }
}

/// Group key for identifying groups
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub values: Vec<DataValue>,
}

/// Row indexer for iloc operations
#[derive(Debug, Clone)]
pub enum RowIndexer {
    /// Single row index
    Single(usize),
    /// Multiple row indices
    Multiple(Vec<usize>),
    /// Slice of rows
    Slice { start: Option<usize>, end: Option<usize>, step: Option<usize> },
    /// Boolean mask
    Mask(Vec<bool>),
}

/// Column indexer for iloc operations
#[derive(Debug, Clone)]
pub enum ColIndexer {
    /// Single column index
    Single(usize),
    /// Multiple column indices
    Multiple(Vec<usize>),
    /// Slice of columns
    Slice { start: Option<usize>, end: Option<usize>, step: Option<usize> },
    /// All columns
    All,
}

/// Label indexer for loc operations
#[derive(Debug, Clone)]
pub enum LabelIndexer {
    /// Single label
    Single(String),
    /// Multiple labels
    Multiple(Vec<String>),
    /// Slice of labels
    Slice { start: Option<String>, end: Option<String> },
    /// All labels
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
    
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }
    
    pub fn count_true(&self) -> usize {
        self.mask.iter().filter(|&&x| x).count()
    }
    
    pub fn count_false(&self) -> usize {
        self.mask.iter().filter(|&&x| !x).count()
    }
}
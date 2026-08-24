//! Core `DataFrameOps` trait for PandRS
//!
//! Defines the [`DataFrameOps`] abstraction — a minimal, structural set of
//! DataFrame operations shared by wrapper types. Its sole live implementor is
//! the JIT wrapper `JitOptimizedDataFrame` in
//! [`crate::optimized::jit::dataframe_integration`], which forwards to an inner
//! DataFrame while adding just-in-time optimization. The concrete [`DataFrame`]
//! type does not implement this trait; user-facing operations live as inherent
//! methods and extension traits on `DataFrame`/`OptimizedDataFrame` instead.
//!
//! [`DataFrame`]: crate::dataframe::DataFrame

use crate::core::data_value::DataValue;
use crate::core::error::Result;
use std::collections::HashMap;

/// Axis specification for operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Row = 0,
    Column = 1,
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

/// Structural operations shared by DataFrame-like wrapper types in PandRS.
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

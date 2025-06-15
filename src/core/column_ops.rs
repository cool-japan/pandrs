//! Comprehensive Column Operations Trait System
//!
//! This module provides detailed trait definitions for column operations
//! as specified in the PandRS trait system design specification.

use crate::core::data_value::DataValue;
use std::collections::HashMap;

/// Column type specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit integer
    Int64,
    /// 32-bit integer
    Int32,
    /// 64-bit float
    Float64,
    /// 32-bit float
    Float32,
    /// String data
    String,
    /// Boolean data
    Boolean,
    /// DateTime data
    DateTime,
    /// Categorical data
    Categorical,
    /// Object (mixed types)
    Object,
}

/// Duplicate handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKeep {
    /// Keep first occurrence
    First,
    /// Keep last occurrence
    Last,
    /// Keep no duplicates (drop all)
    None,
}

/// String padding side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// Pad on the left
    Left,
    /// Pad on the right
    Right,
    /// Pad on both sides
    Both,
}

/// Base trait for all column types in PandRS
pub trait ColumnOps<T> {
    type Output: ColumnOps<T>;
    type Error: std::error::Error;

    // Data access and metadata
    fn get(&self, index: usize) -> Option<&T>;
    fn get_unchecked(&self, index: usize) -> &T;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn dtype(&self) -> ColumnType;
    fn name(&self) -> Option<&str>;
    fn set_name(&mut self, name: String);

    // Null handling
    fn is_null(&self, index: usize) -> bool;
    fn null_count(&self) -> usize;
    fn has_nulls(&self) -> bool;
    fn dropna(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn fillna(&self, value: &T) -> std::result::Result<Self::Output, Self::Error>;

    // Data operations
    fn append(&mut self, value: T) -> std::result::Result<(), Self::Error>;
    fn extend_from_slice(&mut self, values: &[T]) -> std::result::Result<(), Self::Error>;
    fn insert(&mut self, index: usize, value: T) -> std::result::Result<(), Self::Error>;
    fn remove(&mut self, index: usize) -> std::result::Result<T, Self::Error>;

    // Transformation operations
    fn map<U, F>(&self, func: F) -> std::result::Result<Box<dyn std::any::Any>, Self::Error>
    where
        F: Fn(&T) -> U,
        U: Clone + Send + Sync + 'static;

    fn filter(&self, mask: &[bool]) -> std::result::Result<Self::Output, Self::Error>;
    fn take(&self, indices: &[usize]) -> std::result::Result<Self::Output, Self::Error>;
    fn slice(&self, start: usize, end: usize) -> std::result::Result<Self::Output, Self::Error>;

    // Comparison and searching
    fn eq(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialEq;
    fn ne(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialEq;
    fn lt(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialOrd;
    fn le(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialOrd;
    fn gt(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialOrd;
    fn ge(&self, other: &Self) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: PartialOrd;

    // Unique and duplicates
    fn unique(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: Eq + std::hash::Hash;
    fn nunique(&self) -> usize
    where
        T: Eq + std::hash::Hash;
    fn is_unique(&self) -> bool
    where
        T: Eq + std::hash::Hash;
    fn duplicated(&self, keep: DuplicateKeep) -> std::result::Result<BooleanColumn, Self::Error>
    where
        T: Eq + std::hash::Hash;

    // Sorting
    fn sort(&self, ascending: bool) -> std::result::Result<Self::Output, Self::Error>
    where
        T: Ord;
    fn argsort(&self, ascending: bool) -> std::result::Result<Vec<usize>, Self::Error>
    where
        T: Ord;

    // Memory management
    fn memory_usage(&self) -> usize;
    fn shrink_to_fit(&mut self);
}

/// Numeric column operations
pub trait NumericColumnOps<T>: ColumnOps<T>
where
    T: num_traits::Num + Copy + PartialOrd + Send + Sync + 'static,
{
    // Arithmetic operations
    fn add(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn sub(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn mul(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn div(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn pow(&self, exponent: f64) -> std::result::Result<Self::Output, Self::Error>;

    // Scalar operations
    fn add_scalar(&self, scalar: T) -> std::result::Result<Self::Output, Self::Error>;
    fn sub_scalar(&self, scalar: T) -> std::result::Result<Self::Output, Self::Error>;
    fn mul_scalar(&self, scalar: T) -> std::result::Result<Self::Output, Self::Error>;
    fn div_scalar(&self, scalar: T) -> std::result::Result<Self::Output, Self::Error>;

    // Aggregation operations
    fn sum(&self) -> Option<T>;
    fn mean(&self) -> Option<f64>;
    fn median(&self) -> Option<f64>;
    fn std(&self, ddof: usize) -> Option<f64>;
    fn var(&self, ddof: usize) -> Option<f64>;
    fn min(&self) -> Option<T>;
    fn max(&self) -> Option<T>;
    fn quantile(&self, q: f64) -> Option<f64>;

    // Statistical operations
    fn cumsum(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn cumprod(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn cummax(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn cummin(&self) -> std::result::Result<Self::Output, Self::Error>;

    // Rounding (for floating point types)
    fn round(&self, decimals: i32) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn floor(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn ceil(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;

    // Mathematical functions
    fn abs(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Signed;
    fn sqrt(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn exp(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn log(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn sin(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
    fn cos(&self) -> std::result::Result<Self::Output, Self::Error>
    where
        T: num_traits::Float;
}

/// String column operations
pub trait StringColumnOps: ColumnOps<String> {
    // String methods
    fn len_chars(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn lower(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn upper(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn strip(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn lstrip(&self) -> std::result::Result<Self::Output, Self::Error>;
    fn rstrip(&self) -> std::result::Result<Self::Output, Self::Error>;

    // Pattern matching
    fn contains(
        &self,
        pattern: &str,
        regex: bool,
    ) -> std::result::Result<BooleanColumn, Self::Error>;
    fn startswith(&self, prefix: &str) -> std::result::Result<BooleanColumn, Self::Error>;
    fn endswith(&self, suffix: &str) -> std::result::Result<BooleanColumn, Self::Error>;
    fn find(&self, substring: &str) -> std::result::Result<Int64Column, Self::Error>;

    // String transformations
    fn replace(
        &self,
        pattern: &str,
        replacement: &str,
        regex: bool,
    ) -> std::result::Result<Self::Output, Self::Error>;
    fn slice_str(
        &self,
        start: Option<usize>,
        end: Option<usize>,
    ) -> std::result::Result<Self::Output, Self::Error>;
    fn split(&self, delimiter: &str) -> std::result::Result<Vec<Self::Output>, Self::Error>;
    fn join(&self, separator: &str) -> String;

    // Categorical operations
    fn value_counts(&self) -> std::result::Result<crate::dataframe::DataFrame, Self::Error>;
    fn to_categorical(&self) -> std::result::Result<CategoricalColumn, Self::Error>;

    // Padding and alignment
    fn pad(
        &self,
        width: usize,
        side: PadSide,
        fillchar: char,
    ) -> std::result::Result<Self::Output, Self::Error>;
    fn center(
        &self,
        width: usize,
        fillchar: char,
    ) -> std::result::Result<Self::Output, Self::Error>;
    fn ljust(&self, width: usize, fillchar: char)
        -> std::result::Result<Self::Output, Self::Error>;
    fn rjust(&self, width: usize, fillchar: char)
        -> std::result::Result<Self::Output, Self::Error>;
}

/// DateTime column operations
pub trait DateTimeColumnOps: ColumnOps<chrono::DateTime<chrono::Utc>> {
    // Date/time extraction
    fn year(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn month(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn day(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn hour(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn minute(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn second(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn weekday(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn dayofyear(&self) -> std::result::Result<Int64Column, Self::Error>;

    // Date/time formatting
    fn strftime(&self, format: &str) -> std::result::Result<StringColumn, Self::Error>;
    fn to_date(&self) -> std::result::Result<DateColumn, Self::Error>;
    fn to_time(&self) -> std::result::Result<TimeColumn, Self::Error>;

    // Time zone operations
    fn tz_localize(&self, tz: &str) -> std::result::Result<Self::Output, Self::Error>;
    fn tz_convert(&self, tz: &str) -> std::result::Result<Self::Output, Self::Error>;

    // Date arithmetic
    fn add_days(&self, days: i64) -> std::result::Result<Self::Output, Self::Error>;
    fn add_months(&self, months: i64) -> std::result::Result<Self::Output, Self::Error>;
    fn add_years(&self, years: i64) -> std::result::Result<Self::Output, Self::Error>;

    // Date range operations
    fn between(
        &self,
        start: &chrono::DateTime<chrono::Utc>,
        end: &chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<BooleanColumn, Self::Error>;
    fn business_day_count(&self, end: &Self) -> std::result::Result<Int64Column, Self::Error>;
}

/// Boolean column operations
pub trait BooleanColumnOps: ColumnOps<bool> {
    // Logical operations
    fn and(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn or(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn xor(&self, other: &Self) -> std::result::Result<Self::Output, Self::Error>;
    fn not(&self) -> std::result::Result<Self::Output, Self::Error>;

    // Aggregations
    fn any(&self) -> bool;
    fn all(&self) -> bool;
    fn count_true(&self) -> usize;
    fn count_false(&self) -> usize;

    // Conversion
    fn to_int(&self) -> std::result::Result<Int64Column, Self::Error>;
    fn to_float(&self) -> std::result::Result<Float64Column, Self::Error>;
}

/// Categorical column operations
pub trait CategoricalColumnOps<T>: ColumnOps<T>
where
    T: Clone + Eq + std::hash::Hash + Send + Sync + 'static,
{
    /// Get the categories
    fn categories(&self) -> Vec<T>;

    /// Add new categories
    fn add_categories(&mut self, categories: &[T]) -> std::result::Result<(), Self::Error>;

    /// Remove categories
    fn remove_categories(&mut self, categories: &[T]) -> std::result::Result<(), Self::Error>;

    /// Set categories
    fn set_categories(
        &mut self,
        categories: Vec<T>,
        ordered: bool,
    ) -> std::result::Result<(), Self::Error>;

    /// Check if ordered
    fn is_ordered(&self) -> bool;

    /// Set ordered flag
    fn set_ordered(&mut self, ordered: bool);

    /// Get category codes
    fn codes(&self) -> std::result::Result<Int64Column, Self::Error>;

    /// Reorder categories
    fn reorder_categories(
        &mut self,
        new_categories: Vec<T>,
    ) -> std::result::Result<(), Self::Error>;

    /// Rename categories
    fn rename_categories<F>(&mut self, rename_func: F) -> std::result::Result<(), Self::Error>
    where
        F: Fn(&T) -> T;
}

// Forward declarations for concrete column types
// These will be implemented by specific column implementations
pub struct ConcreteInt64Column;
pub struct ConcreteInt32Column;
pub struct ConcreteFloat64Column;
pub struct ConcreteFloat32Column;
pub struct ConcreteStringColumn;
pub struct ConcreteBooleanColumn;
pub struct ConcreteDateTimeColumn;
pub struct ConcreteDateColumn;
pub struct ConcreteTimeColumn;
pub struct ConcreteCategoricalColumn;

// Type aliases using concrete implementations
pub type Int64Column = Box<ConcreteInt64Column>;
pub type Int32Column = Box<ConcreteInt32Column>;
pub type Float64Column = Box<ConcreteFloat64Column>;
pub type Float32Column = Box<ConcreteFloat32Column>;
pub type StringColumn = Box<ConcreteStringColumn>;
pub type BooleanColumn = Box<ConcreteBooleanColumn>;
pub type DateTimeColumn = Box<ConcreteDateTimeColumn>;
pub type DateColumn = Box<ConcreteDateColumn>;
pub type TimeColumn = Box<ConcreteTimeColumn>;
pub type CategoricalColumn = Box<ConcreteCategoricalColumn>;

/// Column storage trait for managing memory
pub trait ColumnStorage {
    type StorageType;

    fn allocate(
        &mut self,
        capacity: usize,
    ) -> std::result::Result<Self::StorageType, Box<dyn std::error::Error>>;
    fn deallocate(&mut self, storage: Self::StorageType);
    fn resize(
        &mut self,
        storage: &mut Self::StorageType,
        new_size: usize,
    ) -> std::result::Result<(), Box<dyn std::error::Error>>;
    fn memory_usage(&self) -> usize;
}

/// Type-safe column trait for compile-time optimization
pub trait TypedColumn<T>: ColumnOps<T> {
    fn as_slice(&self) -> Option<&[T]>;
    fn push(&mut self, value: T);
    fn extend_from_slice(&mut self, values: &[T]);
    fn into_vec(self) -> Vec<T>
    where
        Self: Sized;
    fn from_vec(data: Vec<T>, name: Option<String>) -> Self
    where
        Self: Sized;
}

/// Column conversion traits
pub trait ColumnCast<T, U> {
    type Error: std::error::Error;

    fn cast(&self) -> std::result::Result<Box<dyn std::any::Any>, Self::Error>;
    fn try_cast(&self) -> std::result::Result<Box<dyn std::any::Any>, Self::Error>;
    fn safe_cast(
        &self,
        errors: CastErrorBehavior,
    ) -> std::result::Result<Box<dyn std::any::Any>, Self::Error>;
}

/// Cast error behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastErrorBehavior {
    /// Raise error on invalid cast
    Raise,
    /// Return None for invalid cast
    Coerce,
    /// Ignore invalid cast (keep original)
    Ignore,
}

/// Column factory for creating columns
pub trait ColumnFactory {
    fn create_int64(&self, data: Vec<i64>, name: Option<String>) -> Int64Column;
    fn create_int32(&self, data: Vec<i32>, name: Option<String>) -> Int32Column;
    fn create_float64(&self, data: Vec<f64>, name: Option<String>) -> Float64Column;
    fn create_float32(&self, data: Vec<f32>, name: Option<String>) -> Float32Column;
    fn create_string(&self, data: Vec<String>, name: Option<String>) -> StringColumn;
    fn create_boolean(&self, data: Vec<bool>, name: Option<String>) -> BooleanColumn;
    fn create_datetime(
        &self,
        data: Vec<chrono::DateTime<chrono::Utc>>,
        name: Option<String>,
    ) -> DateTimeColumn;
    fn create_categorical<T>(
        &self,
        data: Vec<T>,
        categories: Vec<T>,
        name: Option<String>,
    ) -> CategoricalColumn
    where
        T: Clone + Eq + std::hash::Hash + Send + Sync + 'static + Into<String>;
}

/// Default column factory implementation
#[derive(Debug, Default)]
pub struct DefaultColumnFactory;

impl ColumnFactory for DefaultColumnFactory {
    fn create_int64(&self, _data: Vec<i64>, _name: Option<String>) -> Int64Column {
        unimplemented!("Int64Column creation not yet implemented")
    }

    fn create_int32(&self, _data: Vec<i32>, _name: Option<String>) -> Int32Column {
        unimplemented!("Int32Column creation not yet implemented")
    }

    fn create_float64(&self, _data: Vec<f64>, _name: Option<String>) -> Float64Column {
        unimplemented!("Float64Column creation not yet implemented")
    }

    fn create_float32(&self, _data: Vec<f32>, _name: Option<String>) -> Float32Column {
        unimplemented!("Float32Column creation not yet implemented")
    }

    fn create_string(&self, _data: Vec<String>, _name: Option<String>) -> StringColumn {
        unimplemented!("StringColumn creation not yet implemented")
    }

    fn create_boolean(&self, _data: Vec<bool>, _name: Option<String>) -> BooleanColumn {
        unimplemented!("BooleanColumn creation not yet implemented")
    }

    fn create_datetime(
        &self,
        _data: Vec<chrono::DateTime<chrono::Utc>>,
        _name: Option<String>,
    ) -> DateTimeColumn {
        unimplemented!("DateTimeColumn creation not yet implemented")
    }

    fn create_categorical<T>(
        &self,
        _data: Vec<T>,
        _categories: Vec<T>,
        _name: Option<String>,
    ) -> CategoricalColumn
    where
        T: Clone + Eq + std::hash::Hash + Send + Sync + 'static + Into<String>,
    {
        unimplemented!("CategoricalColumn creation not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_type() {
        assert_eq!(ColumnType::Int64, ColumnType::Int64);
        assert_ne!(ColumnType::Int64, ColumnType::Float64);
    }

    #[test]
    fn test_duplicate_keep() {
        assert_eq!(DuplicateKeep::First, DuplicateKeep::First);
        assert_ne!(DuplicateKeep::First, DuplicateKeep::Last);
    }

    #[test]
    fn test_pad_side() {
        assert_eq!(PadSide::Left, PadSide::Left);
        assert_ne!(PadSide::Left, PadSide::Right);
    }

    #[test]
    fn test_cast_error_behavior() {
        assert_eq!(CastErrorBehavior::Raise, CastErrorBehavior::Raise);
        assert_ne!(CastErrorBehavior::Raise, CastErrorBehavior::Coerce);
    }
}

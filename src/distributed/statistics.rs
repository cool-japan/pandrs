//! # Distributed Processing Statistics
//!
//! This module provides functionality for collecting and managing statistics
//! about datasets, which can be used for query optimization.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use arrow::datatypes::{Schema, SchemaRef, Field, DataType};
use arrow::array::{self, Array, StringArray, PrimitiveArray};
use arrow::compute::{self, min, max, min_max};

/// Statistics for a column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Column name
    pub name: String,
    /// Data type
    pub data_type: DataType,
    /// Number of non-null values
    pub non_null_count: Option<usize>,
    /// Number of distinct values (if known)
    pub distinct_count: Option<usize>,
    /// Minimum value (if available)
    pub min_value: Option<ColumnValue>,
    /// Maximum value (if available)
    pub max_value: Option<ColumnValue>,
    /// Average value (if available, numeric only)
    pub avg_value: Option<f64>,
    /// Whether the column is sorted
    pub is_sorted: bool,
}

/// Value of a column (for statistics)
#[derive(Debug, Clone)]
pub enum ColumnValue {
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Date value (days since epoch)
    Date(i32),
    /// Timestamp value (microseconds since epoch)
    Timestamp(i64),
}

impl ColumnStatistics {
    /// Creates a new column statistics object
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            non_null_count: None,
            distinct_count: None,
            min_value: None,
            max_value: None,
            avg_value: None,
            is_sorted: false,
        }
    }
    
    /// Sets the number of non-null values
    pub fn with_non_null_count(mut self, count: usize) -> Self {
        self.non_null_count = Some(count);
        self
    }
    
    /// Sets the number of distinct values
    pub fn with_distinct_count(mut self, count: usize) -> Self {
        self.distinct_count = Some(count);
        self
    }
    
    /// Sets the minimum value
    pub fn with_min_value(mut self, value: ColumnValue) -> Self {
        self.min_value = Some(value);
        self
    }
    
    /// Sets the maximum value
    pub fn with_max_value(mut self, value: ColumnValue) -> Self {
        self.max_value = Some(value);
        self
    }
    
    /// Sets the average value
    pub fn with_avg_value(mut self, value: f64) -> Self {
        self.avg_value = Some(value);
        self
    }
    
    /// Sets whether the column is sorted
    pub fn with_is_sorted(mut self, sorted: bool) -> Self {
        self.is_sorted = sorted;
        self
    }
    
    /// Merges with another set of column statistics
    pub fn merge(&mut self, other: &Self) {
        if self.name != other.name || self.data_type != other.data_type {
            return;
        }
        
        // Merge non-null count
        self.non_null_count = match (self.non_null_count, other.non_null_count) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        
        // Difficult to merge distinct counts exactly, resort to max
        self.distinct_count = match (self.distinct_count, other.distinct_count) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            _ => None,
        };
        
        // Merge min values
        self.min_value = match (&self.min_value, &other.min_value) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            _ => None,
        };
        
        // Merge max values
        self.max_value = match (&self.max_value, &other.max_value) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            _ => None,
        };
        
        // Difficult to merge averages exactly without counts
        // This is an approximation
        self.avg_value = match (self.avg_value, other.avg_value, self.non_null_count, other.non_null_count) {
            (Some(a), Some(b), Some(count_a), Some(count_b)) => {
                let total_count = count_a + count_b;
                if total_count == 0 {
                    None
                } else {
                    Some((a * count_a as f64 + b * count_b as f64) / total_count as f64)
                }
            },
            (Some(a), None, _, _) => Some(a),
            (None, Some(b), _, _) => Some(b),
            _ => None,
        };
        
        // Sorted only if both are sorted
        self.is_sorted = self.is_sorted && other.is_sorted;
    }
}

/// Statistics for a table
#[derive(Debug, Clone)]
pub struct TableStatistics {
    /// Schema of the table
    pub schema: SchemaRef,
    /// Statistics for each column
    pub column_statistics: HashMap<String, ColumnStatistics>,
    /// Number of rows
    pub row_count: usize,
    /// Size in bytes (estimate)
    pub size_bytes: usize,
}

impl TableStatistics {
    /// Creates a new table statistics object
    pub fn new(schema: SchemaRef, row_count: usize, size_bytes: usize) -> Self {
        Self {
            schema,
            column_statistics: HashMap::new(),
            row_count,
            size_bytes,
        }
    }
    
    /// Adds statistics for a column
    pub fn add_column_statistics(&mut self, stats: ColumnStatistics) {
        self.column_statistics.insert(stats.name.clone(), stats);
    }
    
    /// Gets statistics for a column
    pub fn column_statistics(&self, column: &str) -> Option<&ColumnStatistics> {
        self.column_statistics.get(column)
    }
    
    /// Creates table statistics from a set of Arrow record batches
    #[cfg(feature = "distributed")]
    pub fn from_record_batches(batches: &[arrow::record_batch::RecordBatch]) -> Result<Self> {
        if batches.is_empty() {
            return Err(crate::error::Error::InvalidInput(
                "Cannot create statistics from empty record batches".to_string()
            ));
        }
        
        let schema = batches[0].schema();
        let mut row_count = 0;
        let mut size_bytes = 0;
        
        // Compute row count and size estimate
        for batch in batches {
            row_count += batch.num_rows();
            size_bytes += estimate_batch_size(batch);
        }
        
        let mut result = Self::new(schema.clone(), row_count, size_bytes);
        
        // Compute statistics for each column
        for field in schema.fields() {
            let column_name = field.name();
            let data_type = field.data_type();
            
            // Skip complex types for now
            if is_complex_type(data_type) {
                continue;
            }
            
            let stats = compute_column_statistics(column_name, data_type, batches)?;
            result.add_column_statistics(stats);
        }
        
        Ok(result)
    }
    
    /// Merges with another set of table statistics
    pub fn merge(&mut self, other: &Self) {
        // Merge row count and size
        self.row_count += other.row_count;
        self.size_bytes += other.size_bytes;
        
        // Merge column statistics
        for (name, stats) in &other.column_statistics {
            match self.column_statistics.get_mut(name) {
                Some(existing) => {
                    existing.merge(stats);
                },
                None => {
                    self.column_statistics.insert(name.clone(), stats.clone());
                },
            }
        }
    }
}

/// Compute statistics for a column from record batches
#[cfg(feature = "distributed")]
fn compute_column_statistics(
    column_name: &str,
    data_type: &DataType,
    batches: &[arrow::record_batch::RecordBatch],
) -> Result<ColumnStatistics> {
    let mut stats = ColumnStatistics::new(column_name, data_type.clone());
    
    let mut non_null_count = 0;
    let mut is_sorted = true;
    let mut last_value: Option<ColumnValue> = None;
    
    // Collect all non-null arrays for this column
    let mut arrays = Vec::new();
    for batch in batches {
        if let Some(col_idx) = batch.schema().index_of(column_name) {
            let array = batch.column(col_idx);
            non_null_count += array.len() - array.null_count();
            arrays.push(array.clone());
        }
    }
    
    if arrays.is_empty() {
        return Ok(stats);
    }
    
    stats = stats.with_non_null_count(non_null_count);
    
    // Compute min/max based on data type
    match data_type {
        DataType::Boolean => {
            let (min_val, max_val) = compute_boolean_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::Boolean(min_val))
                .with_max_value(ColumnValue::Boolean(max_val));
        },
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let (min_val, max_val, avg) = compute_integer_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::Integer(min_val))
                .with_max_value(ColumnValue::Integer(max_val))
                .with_avg_value(avg);
        },
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            let (min_val, max_val, avg) = compute_float_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::Float(min_val))
                .with_max_value(ColumnValue::Float(max_val))
                .with_avg_value(avg);
        },
        DataType::Utf8 | DataType::LargeUtf8 => {
            let (min_val, max_val) = compute_string_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::String(min_val))
                .with_max_value(ColumnValue::String(max_val));
        },
        DataType::Date32 | DataType::Date64 => {
            let (min_val, max_val) = compute_date_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::Date(min_val))
                .with_max_value(ColumnValue::Date(max_val));
        },
        DataType::Timestamp(_, _) => {
            let (min_val, max_val) = compute_timestamp_min_max(&arrays)?;
            stats = stats
                .with_min_value(ColumnValue::Timestamp(min_val))
                .with_max_value(ColumnValue::Timestamp(max_val));
        },
        _ => {
            // Unsupported type for statistics
        },
    }
    
    // Check if column is sorted
    stats = stats.with_is_sorted(check_if_sorted(&arrays, data_type)?);
    
    Ok(stats)
}

/// Checks if a column is complex (nested structure, etc.)
fn is_complex_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::List(_) |
        DataType::LargeList(_) |
        DataType::FixedSizeList(_, _) |
        DataType::Struct(_) |
        DataType::Union(_, _) |
        DataType::Dictionary(_, _) |
        DataType::Map(_, _) => true,
        _ => false,
    }
}

/// Estimates the size of a record batch in bytes
#[cfg(feature = "distributed")]
fn estimate_batch_size(batch: &arrow::record_batch::RecordBatch) -> usize {
    let mut size = 0;
    
    for i in 0..batch.num_columns() {
        let array = batch.column(i);
        size += array.get_array_memory_size();
    }
    
    size
}

/// Compute min/max for boolean arrays
#[cfg(feature = "distributed")]
fn compute_boolean_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(bool, bool)> {
    let mut min_val = true;
    let mut max_val = false;
    let mut has_value = false;
    
    for array in arrays {
        let bool_array = array.as_any().downcast_ref::<array::BooleanArray>()
            .ok_or_else(|| crate::error::Error::DistributedProcessing(
                "Failed to downcast to BooleanArray".to_string()
            ))?;
        
        for i in 0..bool_array.len() {
            if bool_array.is_null(i) {
                continue;
            }
            
            has_value = true;
            let value = bool_array.value(i);
            min_val = min_val && value;
            max_val = max_val || value;
        }
    }
    
    if !has_value {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    Ok((min_val, max_val))
}

/// Compute min/max/avg for integer arrays
#[cfg(feature = "distributed")]
fn compute_integer_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(i64, i64, f64)> {
    let mut min_val = i64::MAX;
    let mut max_val = i64::MIN;
    let mut sum = 0.0;
    let mut count = 0;
    
    for array in arrays {
        // Handle different integer types
        if let Some(int_array) = array.as_any().downcast_ref::<array::Int64Array>() {
            for i in 0..int_array.len() {
                if int_array.is_null(i) {
                    continue;
                }
                
                let value = int_array.value(i);
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                sum += value as f64;
                count += 1;
            }
        } else if let Some(int_array) = array.as_any().downcast_ref::<array::Int32Array>() {
            for i in 0..int_array.len() {
                if int_array.is_null(i) {
                    continue;
                }
                
                let value = int_array.value(i);
                min_val = min_val.min(value as i64);
                max_val = max_val.max(value as i64);
                sum += value as f64;
                count += 1;
            }
        }
        // Add other integer types as needed
    }
    
    if count == 0 {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    let avg = sum / count as f64;
    
    Ok((min_val, max_val, avg))
}

/// Compute min/max/avg for float arrays
#[cfg(feature = "distributed")]
fn compute_float_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(f64, f64, f64)> {
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    let mut sum = 0.0;
    let mut count = 0;
    
    for array in arrays {
        if let Some(float_array) = array.as_any().downcast_ref::<array::Float64Array>() {
            for i in 0..float_array.len() {
                if float_array.is_null(i) {
                    continue;
                }
                
                let value = float_array.value(i);
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                sum += value;
                count += 1;
            }
        } else if let Some(float_array) = array.as_any().downcast_ref::<array::Float32Array>() {
            for i in 0..float_array.len() {
                if float_array.is_null(i) {
                    continue;
                }
                
                let value = float_array.value(i);
                min_val = min_val.min(value as f64);
                max_val = max_val.max(value as f64);
                sum += value as f64;
                count += 1;
            }
        }
    }
    
    if count == 0 {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    let avg = sum / count as f64;
    
    Ok((min_val, max_val, avg))
}

/// Compute min/max for string arrays
#[cfg(feature = "distributed")]
fn compute_string_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(String, String)> {
    let mut min_val = String::new();
    let mut max_val = String::new();
    let mut has_value = false;
    
    for array in arrays {
        if let Some(string_array) = array.as_any().downcast_ref::<array::StringArray>() {
            for i in 0..string_array.len() {
                if string_array.is_null(i) {
                    continue;
                }
                
                let value = string_array.value(i);
                if !has_value || value < min_val {
                    min_val = value.to_string();
                }
                if !has_value || value > max_val {
                    max_val = value.to_string();
                }
                has_value = true;
            }
        }
    }
    
    if !has_value {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    Ok((min_val, max_val))
}

/// Compute min/max for date arrays
#[cfg(feature = "distributed")]
fn compute_date_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(i32, i32)> {
    let mut min_val = i32::MAX;
    let mut max_val = i32::MIN;
    let mut has_value = false;
    
    for array in arrays {
        if let Some(date_array) = array.as_any().downcast_ref::<array::Date32Array>() {
            for i in 0..date_array.len() {
                if date_array.is_null(i) {
                    continue;
                }
                
                let value = date_array.value(i);
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                has_value = true;
            }
        }
    }
    
    if !has_value {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    Ok((min_val, max_val))
}

/// Compute min/max for timestamp arrays
#[cfg(feature = "distributed")]
fn compute_timestamp_min_max(arrays: &[arrow::array::ArrayRef]) -> Result<(i64, i64)> {
    let mut min_val = i64::MAX;
    let mut max_val = i64::MIN;
    let mut has_value = false;
    
    for array in arrays {
        if let Some(ts_array) = array.as_any().downcast_ref::<array::TimestampMicrosecondArray>() {
            for i in 0..ts_array.len() {
                if ts_array.is_null(i) {
                    continue;
                }
                
                let value = ts_array.value(i);
                min_val = min_val.min(value);
                max_val = max_val.max(value);
                has_value = true;
            }
        }
    }
    
    if !has_value {
        return Err(crate::error::Error::DistributedProcessing(
            "No non-null values found".to_string()
        ));
    }
    
    Ok((min_val, max_val))
}

/// Check if array is sorted
#[cfg(feature = "distributed")]
fn check_if_sorted(arrays: &[arrow::array::ArrayRef], data_type: &DataType) -> Result<bool> {
    match data_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            check_if_sorted_numeric(arrays)
        },
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            check_if_sorted_numeric(arrays)
        },
        DataType::Utf8 | DataType::LargeUtf8 => {
            check_if_sorted_string(arrays)
        },
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => {
            check_if_sorted_numeric(arrays)
        },
        _ => {
            // For other types, assume not sorted
            Ok(false)
        },
    }
}

/// Check if numeric arrays are sorted
#[cfg(feature = "distributed")]
fn check_if_sorted_numeric(arrays: &[arrow::array::ArrayRef]) -> Result<bool> {
    if arrays.is_empty() {
        return Ok(true);
    }
    
    // Check each array is sorted internally
    for array in arrays {
        let is_sorted = compute::is_sorted(array.as_ref());
        if !is_sorted {
            return Ok(false);
        }
    }
    
    // If multiple arrays, check that each array's max <= next array's min
    if arrays.len() > 1 {
        for i in 0..arrays.len() - 1 {
            let current_array = &arrays[i];
            let next_array = &arrays[i + 1];
            
            let current_max = get_numeric_max(current_array)?;
            let next_min = get_numeric_min(next_array)?;
            
            if current_max > next_min {
                return Ok(false);
            }
        }
    }
    
    Ok(true)
}

/// Get numeric maximum value
#[cfg(feature = "distributed")]
fn get_numeric_max(array: &arrow::array::ArrayRef) -> Result<f64> {
    if let Some(int_array) = array.as_any().downcast_ref::<array::Int64Array>() {
        for i in (0..int_array.len()).rev() {
            if !int_array.is_null(i) {
                return Ok(int_array.value(i) as f64);
            }
        }
    } else if let Some(int_array) = array.as_any().downcast_ref::<array::Int32Array>() {
        for i in (0..int_array.len()).rev() {
            if !int_array.is_null(i) {
                return Ok(int_array.value(i) as f64);
            }
        }
    } else if let Some(float_array) = array.as_any().downcast_ref::<array::Float64Array>() {
        for i in (0..float_array.len()).rev() {
            if !float_array.is_null(i) {
                return Ok(float_array.value(i));
            }
        }
    }
    
    Err(crate::error::Error::DistributedProcessing(
        "Could not get numeric maximum".to_string()
    ))
}

/// Get numeric minimum value
#[cfg(feature = "distributed")]
fn get_numeric_min(array: &arrow::array::ArrayRef) -> Result<f64> {
    if let Some(int_array) = array.as_any().downcast_ref::<array::Int64Array>() {
        for i in 0..int_array.len() {
            if !int_array.is_null(i) {
                return Ok(int_array.value(i) as f64);
            }
        }
    } else if let Some(int_array) = array.as_any().downcast_ref::<array::Int32Array>() {
        for i in 0..int_array.len() {
            if !int_array.is_null(i) {
                return Ok(int_array.value(i) as f64);
            }
        }
    } else if let Some(float_array) = array.as_any().downcast_ref::<array::Float64Array>() {
        for i in 0..float_array.len() {
            if !float_array.is_null(i) {
                return Ok(float_array.value(i));
            }
        }
    }
    
    Err(crate::error::Error::DistributedProcessing(
        "Could not get numeric minimum".to_string()
    ))
}

/// Check if string arrays are sorted
#[cfg(feature = "distributed")]
fn check_if_sorted_string(arrays: &[arrow::array::ArrayRef]) -> Result<bool> {
    if arrays.is_empty() {
        return Ok(true);
    }
    
    // Check each array is sorted internally
    for array in arrays {
        let is_sorted = compute::is_sorted(array.as_ref());
        if !is_sorted {
            return Ok(false);
        }
    }
    
    // If multiple arrays, check that each array's max <= next array's min
    if arrays.len() > 1 {
        for i in 0..arrays.len() - 1 {
            let current_array = &arrays[i];
            let next_array = &arrays[i + 1];
            
            if let (Some(current), Some(next)) = (
                current_array.as_any().downcast_ref::<array::StringArray>(),
                next_array.as_any().downcast_ref::<array::StringArray>(),
            ) {
                let current_max = get_string_max(current)?;
                let next_min = get_string_min(next)?;
                
                if current_max > next_min {
                    return Ok(false);
                }
            }
        }
    }
    
    Ok(true)
}

/// Get string maximum value
#[cfg(feature = "distributed")]
fn get_string_max(array: &array::StringArray) -> Result<String> {
    for i in (0..array.len()).rev() {
        if !array.is_null(i) {
            return Ok(array.value(i).to_string());
        }
    }
    
    Err(crate::error::Error::DistributedProcessing(
        "Could not get string maximum".to_string()
    ))
}

/// Get string minimum value
#[cfg(feature = "distributed")]
fn get_string_min(array: &array::StringArray) -> Result<String> {
    for i in 0..array.len() {
        if !array.is_null(i) {
            return Ok(array.value(i).to_string());
        }
    }
    
    Err(crate::error::Error::DistributedProcessing(
        "Could not get string minimum".to_string()
    ))
}

impl ColumnValue {
    /// Compare two column values and return the minimum
    fn min(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Boolean(a), Self::Boolean(b)) => Self::Boolean(*a && *b),
            (Self::Integer(a), Self::Integer(b)) => Self::Integer(*a.min(b)),
            (Self::Float(a), Self::Float(b)) => Self::Float(*a.min(b)),
            (Self::String(a), Self::String(b)) => Self::String(if a < b { a.clone() } else { b.clone() }),
            (Self::Date(a), Self::Date(b)) => Self::Date(*a.min(b)),
            (Self::Timestamp(a), Self::Timestamp(b)) => Self::Timestamp(*a.min(b)),
            _ => self.clone(), // Different types, just return self
        }
    }
    
    /// Compare two column values and return the maximum
    fn max(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Boolean(a), Self::Boolean(b)) => Self::Boolean(*a || *b),
            (Self::Integer(a), Self::Integer(b)) => Self::Integer(*a.max(b)),
            (Self::Float(a), Self::Float(b)) => Self::Float(*a.max(b)),
            (Self::String(a), Self::String(b)) => Self::String(if a > b { a.clone() } else { b.clone() }),
            (Self::Date(a), Self::Date(b)) => Self::Date(*a.max(b)),
            (Self::Timestamp(a), Self::Timestamp(b)) => Self::Timestamp(*a.max(b)),
            _ => self.clone(), // Different types, just return self
        }
    }
}
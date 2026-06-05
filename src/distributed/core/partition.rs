//! # Partition Management
//!
//! This module provides functionality for managing data partitions in distributed processing.

use std::sync::Arc;

use crate::error::Result;

/// Strategy for partitioning data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Round-robin partitioning (distribute rows evenly)
    RoundRobin,
    /// Hash partitioning (partition based on hash of specified columns)
    Hash,
    /// Range partitioning (partition based on value ranges of specified columns)
    Range,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Represents a partition of data
#[derive(Debug)]
pub struct Partition {
    /// Unique identifier for the partition
    id: usize,
    /// Data in the partition (as Arrow RecordBatch)
    #[cfg(feature = "distributed")]
    data: Option<arrow::record_batch::RecordBatch>,
    /// Partition metadata
    metadata: PartitionMetadata,
}

/// Metadata about a partition
#[derive(Debug, Clone)]
pub struct PartitionMetadata {
    /// Number of rows in the partition
    row_count: usize,
    /// Memory usage in bytes
    memory_usage: usize,
    /// Column names
    column_names: Vec<String>,
    /// Statistics about the partition (min/max values, etc.)
    statistics: Option<PartitionStatistics>,
}

/// Statistics about a partition's data
#[derive(Debug, Clone)]
pub struct PartitionStatistics {
    /// Column statistics (min/max values, nulls, etc.)
    column_statistics: Vec<ColumnStatistics>,
}

/// Statistics about a column in a partition
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Column name
    name: String,
    /// Data type
    data_type: String,
    /// Minimum value (if available)
    min_value: Option<String>,
    /// Maximum value (if available)
    max_value: Option<String>,
    /// Number of null values
    null_count: usize,
    /// Number of distinct values (if available)
    distinct_count: Option<usize>,
}

impl Partition {
    /// Creates a new partition
    #[cfg(feature = "distributed")]
    pub fn new(id: usize, data: arrow::record_batch::RecordBatch) -> Self {
        let metadata = PartitionMetadata::from_record_batch(&data);
        Self {
            id,
            data: Some(data),
            metadata,
        }
    }

    /// Creates a new partition with just metadata (no data)
    pub fn new_metadata_only(id: usize, metadata: PartitionMetadata) -> Self {
        Self {
            id,
            #[cfg(feature = "distributed")]
            data: None,
            metadata,
        }
    }

    /// Gets the partition ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Gets the partition data
    #[cfg(feature = "distributed")]
    pub fn data(&self) -> Option<&arrow::record_batch::RecordBatch> {
        self.data.as_ref()
    }

    /// Takes ownership of the partition data
    #[cfg(feature = "distributed")]
    pub fn take_data(&mut self) -> Option<arrow::record_batch::RecordBatch> {
        self.data.take()
    }

    /// Gets the partition metadata
    pub fn metadata(&self) -> &PartitionMetadata {
        &self.metadata
    }
}

impl PartitionMetadata {
    /// Creates metadata from a RecordBatch, computing per-column statistics.
    #[cfg(feature = "distributed")]
    pub fn from_record_batch(batch: &arrow::record_batch::RecordBatch) -> Self {
        let row_count = batch.num_rows();
        let memory_usage = estimate_batch_memory_usage(batch);
        let schema = batch.schema();
        let column_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

        let statistics = Some(compute_batch_statistics(batch));

        Self {
            row_count,
            memory_usage,
            column_names,
            statistics,
        }
    }

    /// Creates metadata manually
    pub fn new(
        row_count: usize,
        memory_usage: usize,
        column_names: Vec<String>,
        statistics: Option<PartitionStatistics>,
    ) -> Self {
        Self {
            row_count,
            memory_usage,
            column_names,
            statistics,
        }
    }

    /// Gets the row count
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Gets the memory usage
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Gets the column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Gets the statistics
    pub fn statistics(&self) -> Option<&PartitionStatistics> {
        self.statistics.as_ref()
    }
}

/// A collection of partitions forming a distributed dataset
#[derive(Debug, Clone)]
pub struct PartitionSet {
    /// Partitions in the set
    partitions: Vec<Arc<Partition>>,
    /// Schema of the data
    #[cfg(feature = "distributed")]
    schema: Option<arrow::datatypes::SchemaRef>,
}

impl PartitionSet {
    /// Creates a new partition set
    #[cfg(feature = "distributed")]
    pub fn new(partitions: Vec<Arc<Partition>>, schema: arrow::datatypes::SchemaRef) -> Self {
        Self {
            partitions,
            schema: Some(schema),
        }
    }

    /// Creates a new empty partition set
    pub fn empty() -> Self {
        Self {
            partitions: Vec::new(),
            #[cfg(feature = "distributed")]
            schema: None,
        }
    }

    /// Gets the partitions
    pub fn partitions(&self) -> &[Arc<Partition>] {
        &self.partitions
    }

    /// Gets the schema
    #[cfg(feature = "distributed")]
    pub fn schema(&self) -> Option<&arrow::datatypes::SchemaRef> {
        self.schema.as_ref()
    }

    /// Adds a partition to the set
    pub fn add_partition(&mut self, partition: Arc<Partition>) {
        self.partitions.push(partition);
    }

    /// Gets the total number of rows across all partitions
    pub fn total_rows(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.metadata().row_count())
            .sum()
    }

    /// Gets the total memory usage across all partitions
    pub fn total_memory_usage(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.metadata().memory_usage())
            .sum()
    }
}

/// Trait for creating partitions from data
pub trait Partitioner {
    /// Creates partitions from a DataFrame
    #[cfg(feature = "distributed")]
    fn partition(&self, df: &crate::dataframe::DataFrame) -> Result<PartitionSet>;
}

/// Helper function to estimate memory usage of a RecordBatch
#[cfg(feature = "distributed")]
fn estimate_batch_memory_usage(batch: &arrow::record_batch::RecordBatch) -> usize {
    let mut total_size = 0;

    // Add size for each column
    for column in batch.columns() {
        // Get size of array data buffers
        for buffer in column.to_data().buffers() {
            total_size += buffer.len();
        }

        // Add overhead for validity bitmap if there are nulls
        if column.null_count() > 0 {
            total_size += (batch.num_rows() + 7) / 8; // Bitmap size in bytes (rounded up)
        }
    }

    // Add size for schema (rough estimate)
    total_size += 100 * batch.num_columns();

    total_size
}

/// Compute per-column statistics from a RecordBatch.
///
/// For numeric columns (Int8/16/32/64, UInt8/16/32/64, Float32/Float64) the
/// min and max values are extracted as strings.  For all other data types the
/// column is treated as opaque and only the null count is recorded.
#[cfg(feature = "distributed")]
fn compute_batch_statistics(batch: &arrow::record_batch::RecordBatch) -> PartitionStatistics {
    use arrow::array::{
        Array as ArrowArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow::datatypes::DataType;

    let schema = batch.schema();
    let mut column_statistics = Vec::with_capacity(batch.num_columns());

    for (col_idx, field) in schema.fields().iter().enumerate() {
        let array = batch.column(col_idx);
        let null_count = array.null_count();
        let data_type = format!("{:?}", field.data_type());

        // Extract min/max for numeric columns
        let (min_value, max_value) = match field.data_type() {
            DataType::Float64 => {
                if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
                    let (min, max) = arr
                        .iter()
                        .flatten()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), v| {
                            (mn.min(v), mx.max(v))
                        });
                    if min.is_finite() {
                        (Some(min.to_string()), Some(max.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::Float32 => {
                if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
                    let (min, max) = arr
                        .iter()
                        .flatten()
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                            (mn.min(v), mx.max(v))
                        });
                    if min.is_finite() {
                        (Some(min.to_string()), Some(max.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::Int64 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
                    let vals: Vec<i64> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::Int32 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
                    let vals: Vec<i32> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::Int16 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
                    let vals: Vec<i16> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::Int8 => {
                if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
                    let vals: Vec<i8> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::UInt64 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
                    let vals: Vec<u64> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::UInt32 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
                    let vals: Vec<u32> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::UInt16 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
                    let vals: Vec<u16> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            DataType::UInt8 => {
                if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
                    let vals: Vec<u8> = arr.iter().flatten().collect();
                    if let (Some(&mn), Some(&mx)) = (vals.iter().min(), vals.iter().max()) {
                        (Some(mn.to_string()), Some(mx.to_string()))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            }
            _ => (None, None),
        };

        column_statistics.push(ColumnStatistics {
            name: field.name().clone(),
            data_type,
            min_value,
            max_value,
            null_count,
            distinct_count: None, // Exact count requires a hash-set scan; deferred
        });
    }

    PartitionStatistics { column_statistics }
}

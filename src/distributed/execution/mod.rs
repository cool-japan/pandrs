//! # Execution Engine Interface
//!
//! This module defines the interface for distributed execution engines.

// Re-export backward compatibility module (temporarily disabled)
// pub mod backward_compat;
// pub use backward_compat::*;

use std::sync::Arc;

use crate::distributed::core::config::DistributedConfig;
use crate::distributed::core::partition::{Partition, PartitionSet};
use crate::error::Result;

/// Interface for distributed execution engines
pub trait ExecutionEngine: Send + Sync {
    /// Initializes the execution engine
    fn initialize(&mut self, config: &DistributedConfig) -> Result<()>;

    /// Checks if the engine is initialized
    fn is_initialized(&self) -> bool;

    /// Creates a new execution context
    fn create_context(&self, config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>>;

    /// Clones the engine
    fn clone(&self) -> Box<dyn ExecutionEngine>;
}

/// Execution context for running distributed operations
pub trait ExecutionContext: Send + Sync {
    /// Executes a plan and returns a result
    fn execute_plan(&mut self, plan: ExecutionPlan) -> Result<ExecutionResult>;

    /// Registers a partition set with the context
    fn register_in_memory_table(&mut self, name: &str, partitions: PartitionSet) -> Result<()>;

    /// Registers a CSV file as a dataset
    fn register_csv(&mut self, name: &str, path: &str) -> Result<()>;

    /// Registers a Parquet file as a dataset
    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()>;

    /// Executes a SQL query
    fn sql(&mut self, query: &str) -> Result<ExecutionResult>;

    /// Gets the schema of a table
    fn table_schema(&self, name: &str) -> Result<arrow::datatypes::SchemaRef>;

    /// Explains an execution plan
    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String>;

    /// Writes results to Parquet
    fn write_parquet(&mut self, result: &ExecutionResult, path: &str) -> Result<()>;

    /// Writes results to CSV
    fn write_csv(&mut self, result: &ExecutionResult, path: &str) -> Result<()>;

    /// Gets execution metrics
    fn metrics(&self) -> Result<ExecutionMetrics>;

    /// Clones the context
    fn clone(&self) -> Box<dyn ExecutionContext>;
}

/// A plan for executing operations
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Input dataset name
    input: String,
    /// Operations to execute
    operations: Vec<Operation>,
}

impl ExecutionPlan {
    /// Creates a new execution plan
    pub fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            operations: Vec::new(),
        }
    }

    /// Adds an operation to the plan
    pub fn add_operation(&mut self, operation: Operation) -> &mut Self {
        self.operations.push(operation);
        self
    }

    /// Adds multiple operations to the plan
    pub fn add_operations(&mut self, operations: Vec<Operation>) -> &mut Self {
        self.operations.extend(operations);
        self
    }

    /// Gets the input
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Gets the operations
    pub fn operations(&self) -> &Vec<Operation> {
        &self.operations
    }
}

/// Types of operations that can be executed
#[derive(Debug, Clone)]
pub enum Operation {
    /// SELECT operation - Select specific columns
    Select(Vec<String>),

    /// FILTER operation - Filter rows based on a condition
    Filter(String),

    /// JOIN operation - Join with another dataset
    Join {
        /// Right-side dataset
        right: String,
        /// Join type
        join_type: JoinType,
        /// Left join keys
        left_keys: Vec<String>,
        /// Right join keys
        right_keys: Vec<String>,
    },

    /// AGGREGATE operation - Group by keys and apply aggregations
    Aggregate(Vec<String>, Vec<AggregateExpr>),

    /// ORDER BY operation - Sort data
    OrderBy(Vec<SortExpr>),

    /// LIMIT operation - Limit number of rows
    Limit(usize),

    /// WINDOW operation - Apply window functions
    Window(Vec<String>),

    /// PROJECTION operation - Add computed columns
    Project(Vec<(String, String)>),

    /// DISTINCT operation - Remove duplicate rows
    Distinct,

    /// UNION operation - Combine with another dataset
    Union(String),

    /// INTERSECT operation - Get rows present in both datasets
    Intersect(String),

    /// EXCEPT operation - Get rows in this dataset but not in another
    Except(String),

    /// Custom operation (for extensibility)
    Custom {
        /// Operation name
        name: String,
        /// Operation parameters
        params: std::collections::HashMap<String, String>,
    },
}

/// Types of joins
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join
    Inner,
    /// Left outer join
    Left,
    /// Right outer join
    Right,
    /// Full outer join
    Full,
    /// Cross join
    Cross,
}

/// Expression for aggregation
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// Input column
    pub column: String,
    /// Aggregation function (e.g., "sum", "avg", "count")
    pub function: String,
    /// Output column name (alias)
    pub alias: String,
}

/// Expression for sorting
#[derive(Debug, Clone)]
pub struct SortExpr {
    /// Column name
    pub column: String,
    /// Sort direction
    pub ascending: bool,
    /// Nulls first or last
    pub nulls_first: bool,
}

/// Result of executing a plan
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// The resulting partitions
    partitions: PartitionSet,
    /// Result schema
    schema: arrow::datatypes::SchemaRef,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

impl ExecutionResult {
    /// Creates a new execution result
    pub fn new(
        partitions: PartitionSet,
        schema: arrow::datatypes::SchemaRef,
        metrics: ExecutionMetrics,
    ) -> Self {
        Self {
            partitions,
            schema,
            metrics,
        }
    }

    /// Gets the partitions
    pub fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    /// Gets the schema
    pub fn schema(&self) -> &arrow::datatypes::SchemaRef {
        &self.schema
    }

    /// Gets the metrics
    pub fn metrics(&self) -> &ExecutionMetrics {
        &self.metrics
    }

    /// Gets the row count
    pub fn row_count(&self) -> usize {
        self.partitions.total_rows()
    }

    /// Collects all record batches
    pub fn collect(&self) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let mut batches = Vec::new();

        for partition in self.partitions.partitions() {
            if let Some(batch) = partition.data() {
                batches.push(batch.clone());
            }
        }

        Ok(batches)
    }
}

/// Metrics about execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Time taken for execution in milliseconds
    pub execution_time_ms: u64,
    /// Number of rows processed
    pub rows_processed: usize,
    /// Number of partitions processed
    pub partitions_processed: usize,
    /// Bytes processed
    pub bytes_processed: usize,
    /// Bytes output
    pub bytes_output: usize,
    /// Optional query identifier
    pub query_id: Option<String>,
    /// Number of input rows
    pub input_rows: usize,
    /// Number of output rows
    pub output_rows: usize,
    /// Custom metrics
    pub custom_metrics: std::collections::HashMap<String, String>,
}

impl ExecutionMetrics {
    /// Creates a new set of execution metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the execution time
    pub fn with_execution_time(mut self, time_ms: u64) -> Self {
        self.execution_time_ms = time_ms;
        self
    }

    /// Sets the rows processed
    pub fn with_rows_processed(mut self, rows: usize) -> Self {
        self.rows_processed = rows;
        self
    }

    /// Sets the partitions processed
    pub fn with_partitions_processed(mut self, partitions: usize) -> Self {
        self.partitions_processed = partitions;
        self
    }

    /// Sets the bytes processed
    pub fn with_bytes_processed(mut self, bytes: usize) -> Self {
        self.bytes_processed = bytes;
        self
    }

    /// Sets the bytes output
    pub fn with_bytes_output(mut self, bytes: usize) -> Self {
        self.bytes_output = bytes;
        self
    }

    /// Sets the query ID
    pub fn with_query_id(mut self, id: impl Into<String>) -> Self {
        self.query_id = Some(id.into());
        self
    }

    /// Sets the input rows
    pub fn with_input_rows(mut self, rows: usize) -> Self {
        self.input_rows = rows;
        self
    }

    /// Sets the output rows
    pub fn with_output_rows(mut self, rows: usize) -> Self {
        self.output_rows = rows;
        self
    }

    /// Adds a custom metric
    pub fn with_custom_metric(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_metrics.insert(name.into(), value.into());
        self
    }

    /// Gets a formatted summary of the metrics
    pub fn summary(&self) -> String {
        let mut result = String::new();

        result.push_str(&format!("Execution time: {}ms\n", self.execution_time_ms));
        result.push_str(&format!("Rows processed: {}\n", self.rows_processed));
        result.push_str(&format!(
            "Partitions processed: {}\n",
            self.partitions_processed
        ));
        result.push_str(&format!("Bytes processed: {}\n", self.bytes_processed));
        result.push_str(&format!("Bytes output: {}\n", self.bytes_output));

        if let Some(query_id) = &self.query_id {
            result.push_str(&format!("Query ID: {}\n", query_id));
        }

        if self.input_rows > 0 {
            result.push_str(&format!("Input rows: {}\n", self.input_rows));
        }

        if self.output_rows > 0 {
            result.push_str(&format!("Output rows: {}\n", self.output_rows));
        }

        if !self.custom_metrics.is_empty() {
            result.push_str("Custom metrics:\n");
            for (name, value) in &self.custom_metrics {
                result.push_str(&format!("  {}: {}\n", name, value));
            }
        }

        result
    }

    /// Merges with another set of metrics
    pub fn merge(&mut self, other: &Self) {
        self.execution_time_ms += other.execution_time_ms;
        self.rows_processed += other.rows_processed;
        self.partitions_processed += other.partitions_processed;
        self.bytes_processed += other.bytes_processed;
        self.bytes_output += other.bytes_output;
        self.input_rows += other.input_rows;
        self.output_rows += other.output_rows;

        // Merge custom metrics (just override for now)
        for (name, value) in &other.custom_metrics {
            self.custom_metrics.insert(name.clone(), value.clone());
        }
    }

    /// Adds execution time to the current metrics
    pub fn add_execution_time(&mut self, time_ms: u64) {
        self.execution_time_ms += time_ms;
    }

    /// Adds rows processed to the current metrics
    pub fn add_rows_processed(&mut self, rows: usize) {
        self.rows_processed += rows;
    }

    /// Adds partitions processed to the current metrics
    pub fn add_partitions_processed(&mut self, partitions: usize) {
        self.partitions_processed += partitions;
    }

    /// Adds bytes processed to the current metrics
    pub fn add_bytes_processed(&mut self, bytes: usize) {
        self.bytes_processed += bytes;
    }
}

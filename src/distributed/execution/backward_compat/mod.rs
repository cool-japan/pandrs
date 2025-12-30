//! # Execution Engine Interface
//!
//! This module defines the interface for distributed execution engines.

use std::sync::Arc;

use crate::distributed::config::DistributedConfig;
use crate::distributed::partition::{Partition, PartitionSet};
use crate::error::Result;

/// Interface for distributed execution engines
pub trait ExecutionEngine: Send + Sync {
    /// Initializes the execution engine
    fn initialize(&mut self, config: &DistributedConfig) -> Result<()>;

    /// Checks if the engine is initialized
    fn is_initialized(&self) -> bool;

    /// Creates a new execution context
    fn create_context(&self, config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>>;
}

/// Execution context for running distributed operations
pub trait ExecutionContext: Send + Sync {
    /// Executes a plan and returns a result
    fn execute(&self, plan: &ExecutionPlan) -> Result<ExecutionResult>;

    /// Registers a partition set with the context
    fn register_dataset(&mut self, name: &str, partitions: PartitionSet) -> Result<()>;

    /// Registers a CSV file as a dataset
    fn register_csv(&mut self, name: &str, path: &str) -> Result<()>;

    /// Registers a Parquet file as a dataset
    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()>;

    /// Executes a SQL query
    fn sql(&self, query: &str) -> Result<ExecutionResult>;

    /// Explains an execution plan
    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String>;
}

/// A plan for executing operations
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// The type of operation to execute
    operation: Operation,
    /// Input datasets or previous operation results
    inputs: Vec<String>,
    /// Output dataset name
    output: String,
}

/// Types of operations that can be executed
#[derive(Debug, Clone)]
pub enum Operation {
    /// SELECT operation
    Select {
        /// Columns to select
        columns: Vec<String>,
    },
    /// FILTER operation
    Filter {
        /// Filter predicate
        predicate: String,
    },
    /// JOIN operation
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
    /// GROUP BY operation
    GroupBy {
        /// Grouping keys
        keys: Vec<String>,
        /// Aggregation expressions
        aggregates: Vec<AggregateExpr>,
    },
    /// ORDER BY operation
    OrderBy {
        /// Sort expressions
        sort_exprs: Vec<SortExpr>,
    },
    /// LIMIT operation
    Limit {
        /// Maximum number of rows
        limit: usize,
    },
    /// WINDOW function operation
    Window {
        /// Window function expressions
        window_functions: Vec<crate::distributed::window::WindowFunction>,
    },
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
    /// Aggregation function
    function: String,
    /// Input column
    input: String,
    /// Output column name
    output: String,
}

/// Expression for sorting
#[derive(Debug, Clone)]
pub struct SortExpr {
    /// Column name
    column: String,
    /// Sort direction
    ascending: bool,
    /// Nulls first or last
    nulls_first: bool,
}

impl ExecutionPlan {
    /// Creates a new execution plan
    pub fn new(operation: Operation, inputs: Vec<String>, output: String) -> Self {
        Self {
            operation,
            inputs,
            output,
        }
    }

    /// Gets the operation
    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    /// Gets the inputs
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// Gets the output
    pub fn output(&self) -> &str {
        &self.output
    }
}

/// Result of executing a plan
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// The resulting partitions
    partitions: PartitionSet,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

/// Metrics about execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Time taken for execution in milliseconds
    execution_time_ms: u64,
    /// Number of rows processed
    rows_processed: usize,
    /// Number of partitions processed
    partitions_processed: usize,
    /// Bytes processed
    bytes_processed: usize,
    /// Bytes output
    bytes_output: usize,
    /// Optional query identifier
    query_id: Option<String>,
}

impl ExecutionResult {
    /// Creates a new execution result
    pub fn new(partitions: PartitionSet, metrics: ExecutionMetrics) -> Self {
        Self {
            partitions,
            metrics,
        }
    }

    /// Gets the partitions
    pub fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    /// Gets the execution metrics
    pub fn metrics(&self) -> &ExecutionMetrics {
        &self.metrics
    }

    /// Collects the result into a local DataFrame
    pub fn collect_to_local(&self) -> Result<crate::dataframe::DataFrame> {
        #[cfg(feature = "distributed")]
        {
            // Get all record batches from the partitions
            let mut batches = Vec::new();

            for partition in self.partitions.partitions() {
                if let Some(data) = partition.data() {
                    batches.push(data.clone());
                }
            }

            if batches.is_empty() {
                return Ok(crate::dataframe::DataFrame::new());
            }

            // Convert to local DataFrame
            crate::distributed::datafusion::conversion::record_batches_to_dataframe(&batches)
        }

        #[cfg(not(feature = "distributed"))]
        {
            use crate::error::Error;
            Err(Error::FeatureNotAvailable(
                "Collecting to local DataFrame is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }

    /// Writes the result to a Parquet file
    pub fn write_parquet(&self, path: &str) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            use arrow::datatypes::SchemaRef;
            use std::fs::File;
            use std::sync::Arc;

            // Get all record batches from the partitions
            let mut batches = Vec::new();

            for partition in self.partitions.partitions() {
                if let Some(data) = partition.data() {
                    batches.push(data.clone());
                }
            }

            if batches.is_empty() {
                return Err(Error::InvalidOperation("No data to write".to_string()));
            }

            // Create the file
            let file = File::create(path)
                .map_err(|e| Error::IoError(format!("Failed to create Parquet file: {}", e)))?;

            // Write the batches to Parquet
            let schema = batches[0].schema();

            let props = parquet::file::properties::WriterProperties::builder().build();

            let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props))
                .map_err(|e| {
                    Error::ParquetError(format!("Failed to create Parquet writer: {}", e))
                })?;

            for batch in batches {
                writer.write(&batch).map_err(|e| {
                    Error::ParquetError(format!("Failed to write batch to Parquet: {}", e))
                })?;
            }

            writer.close().map_err(|e| {
                Error::ParquetError(format!("Failed to close Parquet writer: {}", e))
            })?;

            Ok(())
        }

        #[cfg(not(feature = "distributed"))]
        {
            use crate::error::Error;
            Err(Error::FeatureNotAvailable(
                "Writing to Parquet is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
}

impl ExecutionMetrics {
    /// Creates a new execution metrics object
    pub fn new(
        execution_time_ms: u64,
        rows_processed: usize,
        partitions_processed: usize,
        bytes_processed: usize,
        bytes_output: usize,
    ) -> Self {
        Self {
            execution_time_ms,
            rows_processed,
            partitions_processed,
            bytes_processed,
            bytes_output,
            query_id: None,
        }
    }

    /// Creates execution metrics with a query ID
    pub fn with_query_id(
        execution_time_ms: u64,
        rows_processed: usize,
        partitions_processed: usize,
        bytes_processed: usize,
        bytes_output: usize,
        query_id: impl Into<String>,
    ) -> Self {
        Self {
            execution_time_ms,
            rows_processed,
            partitions_processed,
            bytes_processed,
            bytes_output,
            query_id: Some(query_id.into()),
        }
    }

    /// Gets the execution time in milliseconds
    pub fn execution_time_ms(&self) -> u64 {
        self.execution_time_ms
    }

    /// Gets the number of rows processed
    pub fn rows_processed(&self) -> usize {
        self.rows_processed
    }

    /// Gets the number of partitions processed
    pub fn partitions_processed(&self) -> usize {
        self.partitions_processed
    }

    /// Gets the number of bytes processed
    pub fn bytes_processed(&self) -> usize {
        self.bytes_processed
    }

    /// Gets the number of bytes output
    pub fn bytes_output(&self) -> usize {
        self.bytes_output
    }

    /// Gets the query ID, if any
    pub fn query_id(&self) -> Option<&str> {
        self.query_id.as_deref()
    }

    /// Set a query ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.query_id = Some(id.into());
        self
    }

    /// Format the metrics as a human-readable string
    pub fn format(&self) -> String {
        let mut result = String::new();

        // Add query ID if available
        if let Some(id) = &self.query_id {
            result.push_str(&format!("Query ID: {}\n", id));
        }

        // Add execution time
        let execution_time_sec = self.execution_time_ms as f64 / 1000.0;
        result.push_str(&format!("Execution time: {:.3}s\n", execution_time_sec));

        // Add data metrics
        result.push_str(&format!("Rows processed: {}\n", self.rows_processed));
        result.push_str(&format!("Partitions: {}\n", self.partitions_processed));

        // Add processing rate if applicable
        if self.execution_time_ms > 0 && self.rows_processed > 0 {
            let rows_per_sec =
                (self.rows_processed as f64 * 1000.0) / self.execution_time_ms as f64;
            result.push_str(&format!("Processing rate: {:.1} rows/s\n", rows_per_sec));
        }

        // Add memory metrics
        if self.bytes_processed > 0 {
            let mb = 1024.0 * 1024.0;
            result.push_str(&format!(
                "Memory processed: {:.2} MB\n",
                self.bytes_processed as f64 / mb
            ));
        }

        if self.bytes_output > 0 {
            let mb = 1024.0 * 1024.0;
            result.push_str(&format!(
                "Memory output: {:.2} MB\n",
                self.bytes_output as f64 / mb
            ));
        }

        result
    }
}

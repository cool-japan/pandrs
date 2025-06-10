//! # Distributed DataFrame
//!
//! This module provides a DataFrame implementation for distributed processing.

#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use super::config::DistributedConfig;
#[cfg(feature = "distributed")]
use super::partition::{PartitionSet, PartitionStrategy, Partitioner};
#[cfg(feature = "distributed")]
use crate::distributed::execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionPlan, ExecutionResult, JoinType,
    Operation, SortExpr,
};
#[cfg(feature = "distributed")]
use crate::distributed::ToDistributed;
use crate::error::{Error, Result};

/// A DataFrame implementation for distributed processing
#[cfg(feature = "distributed")]
pub struct DistributedDataFrame {
    /// Configuration for the distributed processing
    config: DistributedConfig,
    /// Execution engine
    engine: Box<dyn ExecutionEngine>,
    /// Execution context
    context: Arc<Mutex<Box<dyn ExecutionContext>>>,
    /// Current result (from previous operation)
    current_result: Option<ExecutionResult>,
    /// Identifier for this DataFrame in the execution context
    id: String,
    /// Whether this DataFrame is lazily evaluated
    lazy: bool,
    /// Pending operations for lazy evaluation
    pending_operations: Vec<ExecutionPlan>,
}

#[cfg(feature = "distributed")]
impl DistributedDataFrame {
    /// Creates a new distributed DataFrame
    pub fn new(
        config: DistributedConfig,
        engine: Box<dyn ExecutionEngine>,
        context: Box<dyn ExecutionContext>,
        id: String,
    ) -> Self {
        Self {
            config,
            engine,
            context: Arc::new(Mutex::new(context)),
            current_result: None,
            id,
            lazy: true,
            pending_operations: Vec::new(),
        }
    }

    /// Creates a distributed DataFrame from a local DataFrame
    pub fn from_local(df: &crate::dataframe::DataFrame, config: DistributedConfig) -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Create the engine based on the config
        let mut engine: Box<dyn ExecutionEngine> = match config.executor_type() {
            crate::distributed::core::config::ExecutorType::DataFusion => {
                Box::new(crate::distributed::engines::datafusion::DataFusionEngine::new())
            }
            _ => {
                // Default to DataFusion for now
                Box::new(crate::distributed::engines::datafusion::DataFusionEngine::new())
            }
        };

        // Initialize the engine
        engine.initialize(&config)?;

        // Create the execution context
        let mut context = engine.create_context(&config)?;

        // Generate a unique ID for this DataFrame
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let id = format!("df_{:x}", now);

        // Convert the DataFrame to partitions
        use crate::distributed::engines::datafusion::conversion::dataframe_to_record_batches;

        // Determine batch size based on DataFrame size and concurrency
        let row_count = df.shape()?.0;
        let batch_size = std::cmp::max(1, row_count / std::cmp::max(1, config.concurrency()));

        // Convert to record batches
        let batches = dataframe_to_record_batches(df, batch_size)?;

        // Create partitions
        let mut partitions = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            let partition = crate::distributed::core::partition::Partition::new(i, batch.clone());
            partitions.push(std::sync::Arc::new(partition));
        }

        // Create schema reference
        let schema = if !batches.is_empty() {
            batches[0].schema()
        } else {
            return Err(Error::InvalidInput("DataFrame is empty".to_string()));
        };

        // Create partition set
        let partition_set =
            crate::distributed::core::partition::PartitionSet::new(partitions, schema);

        // Register with context
        context.register_in_memory_table(&id, partition_set)?;

        // Create distributed DataFrame
        let mut result = Self {
            config,
            engine,
            context: Arc::new(Mutex::new(context)),
            current_result: None,
            id,
            lazy: true,
            pending_operations: Vec::new(),
        };

        Ok(result)
    }

    /// Creates a distributed DataFrame with an existing execution result
    pub fn with_result(
        config: DistributedConfig,
        engine: Box<dyn ExecutionEngine>,
        context: Box<dyn ExecutionContext>,
        id: String,
        result: ExecutionResult,
    ) -> Self {
        Self {
            config,
            engine,
            context: Arc::new(Mutex::new(context)),
            current_result: Some(result),
            id,
            lazy: true,
            pending_operations: Vec::new(),
        }
    }

    /// Gets the unique identifier for this DataFrame
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Gets the schema of this DataFrame
    pub fn schema(&self) -> Result<arrow::datatypes::SchemaRef> {
        let context = self.context.lock().unwrap();

        if let Some(result) = &self.current_result {
            Ok(result.schema().clone())
        } else {
            context.table_schema(&self.id)
        }
    }

    /// Executes all pending operations and returns the result
    pub fn execute(&mut self) -> Result<&ExecutionResult> {
        if self.pending_operations.is_empty() && self.current_result.is_some() {
            return Ok(self.current_result.as_ref().unwrap());
        }

        let mut context = self.context.lock().unwrap();

        // Create a plan for the pending operations
        let mut plan = ExecutionPlan::new(&self.id);
        for op in &self.pending_operations {
            plan.add_operations(op.operations().clone());
        }

        // Execute the plan
        let result = context.execute_plan(plan)?;

        // Store the result
        self.current_result = Some(result);

        // Clear pending operations
        self.pending_operations.clear();

        Ok(self.current_result.as_ref().unwrap())
    }

    /// Collects results and creates a local DataFrame
    pub fn collect(&mut self) -> Result<crate::dataframe::DataFrame> {
        let result = self.execute()?;

        // Convert result to DataFrame
        use crate::distributed::engines::datafusion::conversion::record_batches_to_dataframe;
        let batches = result.collect()?;
        let df = record_batches_to_dataframe(&batches)?;

        Ok(df)
    }

    /// Writes results to a Parquet file
    pub fn write_parquet(&mut self, path: &str) -> Result<()> {
        let mut context = self.context.lock().unwrap();

        // Execute pending operations if needed
        if !self.pending_operations.is_empty() || self.current_result.is_none() {
            self.execute()?;
        }

        // Write to Parquet
        if let Some(result) = &self.current_result {
            context.write_parquet(result, path)
        } else {
            Err(Error::InvalidState("No result available".to_string()))
        }
    }

    /// Writes results to a CSV file
    pub fn write_csv(&mut self, path: &str) -> Result<()> {
        let mut context = self.context.lock().unwrap();

        // Execute pending operations if needed
        if !self.pending_operations.is_empty() || self.current_result.is_none() {
            self.execute()?;
        }

        // Write to CSV
        if let Some(result) = &self.current_result {
            context.write_csv(result, path)
        } else {
            Err(Error::InvalidState("No result available".to_string()))
        }
    }

    /// Gets the number of rows in the DataFrame
    pub fn row_count(&mut self) -> Result<usize> {
        let result = self.execute()?;
        Ok(result.row_count())
    }

    /// Returns the shape (rows, columns) of the DataFrame
    pub fn shape(&mut self) -> Result<(usize, usize)> {
        let result = self.execute()?;
        let schema = result.schema();

        Ok((result.row_count(), schema.fields().len()))
    }

    /// Selects columns from the DataFrame
    pub fn select(&mut self, columns: &[&str]) -> Result<Self> {
        let mut plan = ExecutionPlan::new(&self.id);
        plan.add_operation(Operation::Select(
            columns.iter().map(|s| s.to_string()).collect(),
        ));

        // Add to pending operations
        if self.lazy {
            self.pending_operations.push(plan);
            let id = format!("{}_{}", self.id, self.pending_operations.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: self.pending_operations.clone(),
            })
        } else {
            // Execute immediately
            let mut context = self.context.lock().unwrap();
            let result = context.execute_plan(plan)?;

            let id = format!("{}_{}", self.id, "select");

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: Some(result),
                id,
                lazy: false,
                pending_operations: Vec::new(),
            })
        }
    }

    /// Filters rows in the DataFrame based on a SQL WHERE clause
    pub fn filter(&mut self, condition: &str) -> Result<Self> {
        let mut plan = ExecutionPlan::new(&self.id);
        plan.add_operation(Operation::Filter(condition.to_string()));

        // Add to pending operations
        if self.lazy {
            self.pending_operations.push(plan);
            let id = format!("{}_{}", self.id, self.pending_operations.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: self.pending_operations.clone(),
            })
        } else {
            // Execute immediately
            let mut context = self.context.lock().unwrap();
            let result = context.execute_plan(plan)?;

            let id = format!("{}_{}", self.id, "filter");

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: Some(result),
                id,
                lazy: false,
                pending_operations: Vec::new(),
            })
        }
    }

    /// Groups data and applies aggregation functions
    pub fn aggregate(
        &mut self,
        group_by: &[&str],
        aggregates: &[(&str, &str, &str)],
    ) -> Result<Self> {
        let mut plan = ExecutionPlan::new(&self.id);

        // Convert group_by to Vec<String>
        let group_by = group_by.iter().map(|s| s.to_string()).collect();

        // Convert aggregates to AggregateExpr
        let mut agg_exprs = Vec::new();
        for (column, func, alias) in aggregates {
            agg_exprs.push(AggregateExpr {
                column: column.to_string(),
                function: func.to_string(),
                alias: alias.to_string(),
            });
        }

        plan.add_operation(Operation::Aggregate(group_by, agg_exprs));

        // Add to pending operations
        if self.lazy {
            self.pending_operations.push(plan);
            let id = format!("{}_{}", self.id, self.pending_operations.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: self.pending_operations.clone(),
            })
        } else {
            // Execute immediately
            let mut context = self.context.lock().unwrap();
            let result = context.execute_plan(plan)?;

            let id = format!("{}_{}", self.id, "aggregate");

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: Some(result),
                id,
                lazy: false,
                pending_operations: Vec::new(),
            })
        }
    }

    /// Gets the execution context
    pub fn context(&self) -> Arc<Mutex<Box<dyn ExecutionContext>>> {
        self.context.clone()
    }

    /// Gets the configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Sets whether evaluation is lazy
    pub fn with_lazy(&mut self, lazy: bool) -> &mut Self {
        self.lazy = lazy;
        self
    }
}

/// Dummy implementation for when the distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedDataFrame;

#[cfg(not(feature = "distributed"))]
impl DistributedDataFrame {
    /// Dummy implementation for creating a new DataFrame
    pub fn new() -> Self {
        Self
    }
}

//! # Distributed DataFrame
//!
//! This module provides a DataFrame implementation for distributed processing.

#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use super::config::DistributedConfig;
#[cfg(feature = "distributed")]
use crate::distributed::execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionPlan, ExecutionResult, Operation,
};
use crate::error::{Error, Result};
#[cfg(feature = "distributed")]
use crate::lock_safe;

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

    /// Creates a distributed DataFrame that shares an existing execution
    /// context.
    ///
    /// Unlike [`Self::new`] (which wraps a fresh, private context), this reuses
    /// the caller's shared `Arc<Mutex<..>>` context, so a table already
    /// registered in that context under `id` is visible to queries on this
    /// DataFrame. Used by
    /// [`DistributedContext::register_dataframe`](crate::distributed::DistributedContext::register_dataframe)
    /// so the registered data is actually queryable.
    pub fn from_shared_context(
        config: DistributedConfig,
        engine: Box<dyn ExecutionEngine>,
        context: Arc<Mutex<Box<dyn ExecutionContext>>>,
        id: String,
    ) -> Self {
        Self {
            config,
            engine,
            context,
            current_result: None,
            id,
            lazy: true,
            pending_operations: Vec::new(),
        }
    }

    /// Creates a distributed DataFrame from a local DataFrame
    pub fn from_local(df: &crate::dataframe::DataFrame, config: DistributedConfig) -> Result<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Create the engine based on the config. Ballista is not implemented;
        // rather than silently falling back to DataFusion (which would run a
        // "distributed Ballista" job locally and mislabel it), return an honest
        // error.
        let mut engine: Box<dyn ExecutionEngine> = match config.executor_type() {
            crate::distributed::core::config::ExecutorType::DataFusion => {
                Box::new(crate::distributed::engines::datafusion::DataFusionEngine::new())
            }
            crate::distributed::core::config::ExecutorType::Ballista => {
                return Err(Error::NotImplemented(
                    "The Ballista executor is not implemented; use ExecutorType::DataFusion"
                        .to_string(),
                ));
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
        let row_count = df.row_count();
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
        let result = Self {
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
        let context = lock_safe!(self.context, "distributed dataframe context lock")?;

        if let Some(result) = &self.current_result {
            Ok(result.schema().clone())
        } else {
            context.table_schema(&self.id)
        }
    }

    /// Executes all pending operations and returns the result
    pub fn execute(&mut self) -> Result<&ExecutionResult> {
        if self.pending_operations.is_empty() && self.current_result.is_some() {
            return self
                .current_result
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation("No result computed yet".into()))
                .map(|r| r);
        }

        let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;

        // Create a plan for the pending operations.
        //
        // The plan's input must be the *registered base table*, which is the
        // input of the first pending operation (the id of the DataFrame the
        // chain started from). `self.id` here is a derived lineage id
        // (`{base}_{n}`) that was never registered with the execution context —
        // using it produced "table not found". When there are no pending
        // operations this DataFrame *is* the registered table, so fall back to
        // `self.id`.
        let base_input = self
            .pending_operations
            .first()
            .map(|op| op.input().to_string())
            .unwrap_or_else(|| self.id.clone());

        let mut plan = ExecutionPlan::new(&base_input);
        for op in &self.pending_operations {
            plan.add_operations(op.operations().clone());
        }

        // Execute the plan
        let result = context.execute_plan(plan)?;

        // Store the result
        self.current_result = Some(result);

        // Clear pending operations
        self.pending_operations.clear();

        self.current_result
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("No result computed yet".into()))
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
        // Execute pending operations if needed
        if !self.pending_operations.is_empty() || self.current_result.is_none() {
            self.execute()?;
        }

        let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;

        // Write to Parquet
        if let Some(result) = &self.current_result {
            context.write_parquet(result, path)
        } else {
            Err(Error::InvalidValue("No result available".to_string()))
        }
    }

    /// Writes results to a CSV file
    pub fn write_csv(&mut self, path: &str) -> Result<()> {
        // Execute pending operations if needed
        if !self.pending_operations.is_empty() || self.current_result.is_none() {
            self.execute()?;
        }

        let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;

        // Write to CSV
        if let Some(result) = &self.current_result {
            context.write_csv(result, path)
        } else {
            Err(Error::InvalidValue("No result available".to_string()))
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

        // Build the child's pending operations WITHOUT mutating self. The
        // previous code pushed onto `self.pending_operations`, so branching a
        // pipeline (calling `.select()`/`.filter()` twice on the same parent)
        // corrupted the parent and duplicated operations.
        if self.lazy {
            let mut new_pending = self.pending_operations.clone();
            new_pending.push(plan);
            let id = format!("{}_{}", self.id, new_pending.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: new_pending,
            })
        } else {
            // Execute immediately
            let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;
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

        // Build the child's pending operations WITHOUT mutating self (see the
        // note in `select` — mutating the parent corrupts branched pipelines).
        if self.lazy {
            let mut new_pending = self.pending_operations.clone();
            new_pending.push(plan);
            let id = format!("{}_{}", self.id, new_pending.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: new_pending,
            })
        } else {
            // Execute immediately
            let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;
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

        // Build the child's pending operations WITHOUT mutating self (see the
        // note in `select` — mutating the parent corrupts branched pipelines).
        if self.lazy {
            let mut new_pending = self.pending_operations.clone();
            new_pending.push(plan);
            let id = format!("{}_{}", self.id, new_pending.len());

            Ok(Self {
                config: self.config.clone(),
                engine: self.engine.clone(),
                context: self.context.clone(),
                current_result: None,
                id,
                lazy: true,
                pending_operations: new_pending,
            })
        } else {
            // Execute immediately
            let mut context = lock_safe!(self.context, "distributed dataframe context lock")?;
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

    /// Check if this DataFrame is using lazy evaluation
    pub fn is_lazy(&self) -> bool {
        self.lazy
    }

    /// Clone this DataFrame without data (empty)
    pub fn clone_empty(&self) -> Self {
        Self {
            config: self.config.clone(),
            engine: self.engine.clone(),
            context: self.context.clone(),
            current_result: None,
            id: format!("{}_empty", self.id),
            lazy: self.lazy,
            pending_operations: Vec::new(),
        }
    }

    /// Add a pending operation to this DataFrame
    pub fn add_pending_operation(&mut self, operation: ExecutionPlan, _inputs: Vec<String>) {
        self.pending_operations.push(operation);
    }

    /// Execute an operation immediately using the execution context.
    ///
    /// Locks the shared context, executes the given plan, and returns a new
    /// `DistributedDataFrame` whose `current_result` holds the output.
    /// The `_inputs` slice is reserved for future multi-input join operations
    /// and is intentionally unused for now.
    pub fn execute_operation(
        &self,
        operation: ExecutionPlan,
        _inputs: Vec<String>,
    ) -> Result<Self> {
        let mut context = lock_safe!(self.context, "distributed dataframe execute_operation lock")?;

        let result = context.execute_plan(operation)?;

        let new_id = format!("{}_op", self.id);

        Ok(Self {
            config: self.config.clone(),
            engine: self.engine.clone(),
            context: self.context.clone(),
            current_result: Some(result),
            id: new_id,
            lazy: self.lazy,
            pending_operations: Vec::new(),
        })
    }
}

#[cfg(feature = "distributed")]
impl Clone for DistributedDataFrame {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            engine: self.engine.clone(),
            context: self.context.clone(),
            current_result: self.current_result.clone(),
            id: self.id.clone(),
            lazy: self.lazy,
            pending_operations: self.pending_operations.clone(),
        }
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

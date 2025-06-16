//! # Distributed DataFrame
//!
//! This module provides a DataFrame implementation for distributed processing.

use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};
use super::config::DistributedConfig;
use super::execution::{ExecutionEngine, ExecutionContext, ExecutionPlan, ExecutionResult, Operation, AggregateExpr, JoinType, SortExpr};
use super::partition::{PartitionSet, PartitionStrategy, Partitioner};
use super::ToDistributed;

/// A DataFrame implementation for distributed processing
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
    pub fn from_local(
        df: &crate::dataframe::DataFrame,
        config: DistributedConfig,
    ) -> Result<Self> {
        #[cfg(feature = "distributed")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};

            // Create the engine based on the config
            let mut engine: Box<dyn ExecutionEngine> = match config.executor_type() {
                crate::distributed::config::ExecutorType::DataFusion => {
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
                },
                _ => {
                    // Default to DataFusion for now
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
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
            use crate::distributed::datafusion::conversion::dataframe_to_record_batches;

            // Determine batch size based on DataFrame size and concurrency
            let row_count = df.shape()?.0;
            let batch_size = std::cmp::max(
                1,
                row_count / std::cmp::max(1, config.concurrency())
            );

            // Convert to record batches
            let batches = dataframe_to_record_batches(df, batch_size)?;

            // Create partitions
            let mut partitions = Vec::new();
            for (i, batch) in batches.iter().enumerate() {
                let partition = crate::distributed::partition::Partition::new(i, batch.clone());
                partitions.push(std::sync::Arc::new(partition));
            }

            // Create partition set
            let schema_ref = if !batches.is_empty() {
                batches[0].schema()
            } else {
                std::sync::Arc::new(arrow::datatypes::Schema::empty())
            };

            let partition_set = crate::distributed::partition::PartitionSet::new(
                partitions,
                schema_ref,
            );

            // Register the partition set with the context
            context.register_dataset(&id, partition_set)?;

            // Create the distributed DataFrame
            Ok(Self {
                config,
                engine,
                context: std::sync::Arc::new(std::sync::Mutex::new(context)),
                current_result: None,
                id,
                lazy: true,
                pending_operations: Vec::new(),
            })
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Creates a distributed DataFrame directly from a CSV file
    pub fn from_csv(
        path: &str,
        config: DistributedConfig,
    ) -> Result<Self> {
        #[cfg(feature = "distributed")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            use std::path::Path;

            // Check if file exists
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("CSV file not found: {}", path)));
            }

            // Create the engine based on the config
            let mut engine: Box<dyn ExecutionEngine> = match config.executor_type() {
                crate::distributed::config::ExecutorType::DataFusion => {
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
                },
                _ => {
                    // Default to DataFusion for now
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
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

            // Register the CSV file directly with the context
            context.register_csv(&id, path)?;

            // Create the distributed DataFrame
            Ok(Self {
                config,
                engine,
                context: std::sync::Arc::new(std::sync::Mutex::new(context)),
                current_result: None,
                id,
                lazy: true,
                pending_operations: Vec::new(),
            })
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Creates a distributed DataFrame directly from a Parquet file
    pub fn from_parquet(
        path: &str,
        config: DistributedConfig,
    ) -> Result<Self> {
        #[cfg(feature = "distributed")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            use std::path::Path;

            // Check if file exists
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("Parquet file not found: {}", path)));
            }

            // Create the engine based on the config
            let mut engine: Box<dyn ExecutionEngine> = match config.executor_type() {
                crate::distributed::config::ExecutorType::DataFusion => {
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
                },
                _ => {
                    // Default to DataFusion for now
                    Box::new(crate::distributed::datafusion::DataFusionEngine::new())
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

            // Register the Parquet file directly with the context
            context.register_parquet(&id, path)?;

            // Create the distributed DataFrame
            Ok(Self {
                config,
                engine,
                context: std::sync::Arc::new(std::sync::Mutex::new(context)),
                current_result: None,
                id,
                lazy: true,
                pending_operations: Vec::new(),
            })
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Selects specific columns from the DataFrame
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        let columns = columns.iter().map(|s| s.to_string()).collect();
        let operation = Operation::Select { columns };
        
        if self.lazy {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id.clone()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id.clone()])
        }
    }
    
    /// Filters rows based on a predicate
    pub fn filter(&self, predicate: &str) -> Result<Self> {
        let operation = Operation::Filter { 
            predicate: predicate.to_string(),
        };
        
        if self.lazy {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id.clone()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id.clone()])
        }
    }
    
    /// Joins with another distributed DataFrame
    pub fn join(
        &self,
        other: &Self,
        left_keys: &[&str],
        right_keys: &[&str],
        join_type: JoinType,
    ) -> Result<Self> {
        let left_keys = left_keys.iter().map(|s| s.to_string()).collect();
        let right_keys = right_keys.iter().map(|s| s.to_string()).collect();
        
        let operation = Operation::Join { 
            right: other.id.clone(), 
            join_type, 
            left_keys, 
            right_keys,
        };
        
        if self.lazy {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id.clone(), other.id.clone()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id.clone(), other.id.clone()])
        }
    }
    
    /// Groups by specified columns
    pub fn groupby(&self, keys: &[&str]) -> Result<DistributedGroupBy> {
        let keys = keys.iter().map(|s| s.to_string()).collect();
        Ok(DistributedGroupBy::new(self.clone(), keys))
    }
    
    /// Sorts the DataFrame by specified columns
    pub fn sort(&self, columns: &[&str], ascending: &[bool]) -> Result<Self> {
        let mut sort_exprs = Vec::with_capacity(columns.len());
        
        for (i, col) in columns.iter().enumerate() {
            let ascending = if i < ascending.len() { ascending[i] } else { true };
            sort_exprs.push(SortExpr {
                column: col.to_string(),
                ascending,
                nulls_first: !ascending,
            });
        }
        
        let operation = Operation::OrderBy { sort_exprs };
        
        if self.lazy {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id.clone()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id.clone()])
        }
    }
    
    /// Limits the number of rows
    pub fn limit(&self, limit: usize) -> Result<Self> {
        let operation = Operation::Limit { limit };
        
        if self.lazy {
            let mut new_df = self.clone_empty();
            new_df.add_pending_operation(operation, vec![self.id.clone()]);
            Ok(new_df)
        } else {
            self.execute_operation(operation, vec![self.id.clone()])
        }
    }
    
    /// Executes all pending operations and materializes the result
    pub fn execute(&self) -> Result<Self> {
        if !self.lazy || self.pending_operations.is_empty() {
            return Ok(self.clone());
        }

        let mut result = self.clone();
        result.lazy = false;

        // Use a single execution for multiple operations if possible
        if self.pending_operations.len() == 1 {
            // Single operation - simple case
            let context = result.context.lock().unwrap();
            let exec_result = context.execute(&self.pending_operations[0])?;
            result.current_result = Some(exec_result);
        } else {
            // Multiple operations - try to combine into a single SQL query if possible
            let mut combined_sql = String::new();
            let mut executed = false;

            #[cfg(feature = "distributed")]
            {
                // Try to combine operations into a CTE (Common Table Expression) chain
                if let Ok(context) = result.context.lock() {
                    if let Some(df_context) = context.downcast_ref::<crate::distributed::datafusion::DataFusionContext>() {
                        // Build SQL with CTEs
                        let mut cte_parts = Vec::new();

                        for (i, plan) in self.pending_operations.iter().enumerate() {
                            if let Ok(sql) = df_context.convert_operation_to_sql(plan) {
                                if i == 0 {
                                    // First query - use as initial CTE
                                    cte_parts.push(format!("WITH cte_0 AS ({})", sql));
                                } else if i < self.pending_operations.len() - 1 {
                                    // Middle queries - reference previous CTE
                                    let modified_sql = sql.replace(&plan.inputs()[0], &format!("cte_{}", i-1));
                                    cte_parts.push(format!("cte_{} AS ({})", i, modified_sql));
                                } else {
                                    // Final query - main SELECT
                                    let modified_sql = sql.replace(&plan.inputs()[0], &format!("cte_{}", i-1));
                                    combined_sql = format!("{} {}", cte_parts.join(", "), modified_sql);

                                    // Execute the combined query
                                    if let Ok(exec_result) = context.sql(&combined_sql) {
                                        result.current_result = Some(exec_result);
                                        executed = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Fall back to sequential execution if combination failed
            if !executed {
                for plan in &self.pending_operations {
                    let context = result.context.lock().unwrap();
                    let exec_result = context.execute(plan)?;
                    result.current_result = Some(exec_result);
                }
            }
        }

        Ok(result)
    }

    /// Gets the execution metrics from the last operation, if available
    pub fn execution_metrics(&self) -> Option<&crate::distributed::execution::ExecutionMetrics> {
        self.current_result.as_ref().map(|result| result.metrics())
    }

    /// Explains the execution plan for this DataFrame
    pub fn explain(&self, with_statistics: bool) -> Result<String> {
        if self.pending_operations.is_empty() {
            return Ok("No pending operations to explain.".to_string());
        }

        // Get the last operation
        let plan = self.pending_operations.last().unwrap();

        // Get the context
        let context = self.context.lock()
            .map_err(|_| Error::DistributedProcessing("Failed to lock context".to_string()))?;

        // Explain the plan
        context.explain_plan(plan, with_statistics)
    }

    /// Returns a formatted summary of the execution for profiling and analytics purposes
    pub fn summarize(&self) -> Result<String> {
        let executed = self.execute()?;

        if let Some(metrics) = executed.execution_metrics() {
            let mut summary = String::new();

            summary.push_str(&format!("Execution Summary:\n"));
            summary.push_str(&format!("-----------------\n"));

            // Dataset size info
            summary.push_str(&format!("Rows processed:           {}\n", metrics.rows_processed()));
            summary.push_str(&format!("Partitions processed:     {}\n", metrics.partitions_processed()));

            // Performance info
            let execution_time_sec = metrics.execution_time_ms() as f64 / 1000.0;
            summary.push_str(&format!("Execution time:           {:.3}s\n", execution_time_sec));

            if metrics.rows_processed() > 0 && metrics.execution_time_ms() > 0 {
                let rows_per_sec = metrics.rows_processed() as f64 / execution_time_sec;
                summary.push_str(&format!("Processing speed:         {:.0} rows/s\n", rows_per_sec));
            }

            // Memory info
            let mb = 1024.0 * 1024.0;
            if metrics.bytes_processed() > 0 {
                summary.push_str(&format!("Memory processed:         {:.2} MB\n",
                    metrics.bytes_processed() as f64 / mb));
            }

            if metrics.bytes_output() > 0 {
                summary.push_str(&format!("Memory output:            {:.2} MB\n",
                    metrics.bytes_output() as f64 / mb));
            }

            // Additional info if needed
            summary.push_str(&format!("\nLazy evaluation:          {}\n", self.lazy));
            summary.push_str(&format!("Pending operations:       {}\n", self.pending_operations.len()));

            Ok(summary)
        } else {
            Err(Error::InvalidOperation("No execution metrics available. The DataFrame may not have been executed yet.".to_string()))
        }
    }
    
    /// Collects the distributed DataFrame into a local DataFrame
    pub fn collect_to_local(&self) -> Result<crate::dataframe::DataFrame> {
        let df = self.execute()?;
        
        match &df.current_result {
            Some(result) => result.collect_to_local(),
            None => Err(Error::InvalidOperation("No data to collect".to_string())),
        }
    }
    
    /// Writes the distributed DataFrame to a Parquet file
    pub fn write_parquet(&self, path: &str) -> Result<()> {
        let df = self.execute()?;
        
        match &df.current_result {
            Some(result) => result.write_parquet(path),
            None => Err(Error::InvalidOperation("No data to write".to_string())),
        }
    }
    
    /// Gets the distributed configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
    
    /// Gets the execution context
    pub fn context(&self) -> Arc<Mutex<Box<dyn ExecutionContext>>> {
        self.context.clone()
    }
    
    /// Gets the identifier
    pub fn id(&self) -> &str {
        &self.id
    }
    
    /// Checks if the DataFrame is lazy
    pub fn is_lazy(&self) -> bool {
        self.lazy
    }
    
    /// Creates a clone with the same configuration but no data
    fn clone_empty(&self) -> Self {
        // Create a new DataFrame that shares the same execution context
        Self {
            config: self.config.clone(),
            engine: self.engine_clone(),
            context: self.context.clone(),
            current_result: None,
            id: format!("{}_derived_{}", self.id, generate_unique_id()),
            lazy: true,
            pending_operations: Vec::new(),
        }
    }
    
    /// Executes an operation and returns a new DataFrame with the result
    fn execute_operation(
        &self,
        operation: Operation,
        inputs: Vec<String>,
    ) -> Result<Self> {
        let output = format!("{}_{}", self.id, generate_unique_id());
        let plan = ExecutionPlan::new(operation, inputs, output.clone());
        
        let context = self.context.lock().unwrap();
        let result = context.execute(&plan)?;
        
        let mut new_df = self.clone_empty();
        new_df.id = output;
        new_df.current_result = Some(result);
        new_df.lazy = false;
        
        Ok(new_df)
    }
    
    /// Adds a pending operation for lazy evaluation
    fn add_pending_operation(
        &mut self,
        operation: Operation,
        inputs: Vec<String>,
    ) {
        let output = format!("{}_{}", self.id, generate_unique_id());
        let plan = ExecutionPlan::new(operation, inputs, output.clone());
        
        self.pending_operations.push(plan);
        self.id = output;
    }
    
    /// Clones the execution engine
    fn engine_clone(&self) -> Box<dyn ExecutionEngine> {
        // In a real implementation, we would have a proper clone mechanism
        // This is a placeholder that will be replaced
        #[cfg(feature = "distributed")]
        {
            if self.config.executor_type().to_string() == "datafusion" {
                Box::new(super::datafusion::DataFusionEngine::new())
            } else {
                Box::new(super::ballista::BallistaEngine::new())
            }
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            // Return a placeholder engine that returns errors for all operations
            Box::new(PlaceholderEngine::new())
        }
    }
}

// For handling cloning
impl Clone for DistributedDataFrame {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            engine: self.engine_clone(),
            context: self.context.clone(),
            current_result: self.current_result.clone(),
            id: self.id.clone(),
            lazy: self.lazy,
            pending_operations: self.pending_operations.clone(),
        }
    }
}

/// A group by operation on a distributed DataFrame
pub struct DistributedGroupBy {
    /// The DataFrame being grouped
    df: DistributedDataFrame,
    /// The grouping keys
    keys: Vec<String>,
}

impl DistributedGroupBy {
    /// Creates a new group by operation
    pub fn new(df: DistributedDataFrame, keys: Vec<String>) -> Self {
        Self { df, keys }
    }
    
    /// Applies an aggregation to the grouped data
    pub fn aggregate(
        &self,
        agg_columns: &[&str],
        agg_functions: &[&str],
    ) -> Result<DistributedDataFrame> {
        let mut agg_exprs = Vec::with_capacity(agg_columns.len());
        
        for (i, col) in agg_columns.iter().enumerate() {
            let func = if i < agg_functions.len() { 
                agg_functions[i] 
            } else { 
                "sum" 
            };
            
            agg_exprs.push(AggregateExpr {
                function: func.to_string(),
                input: col.to_string(),
                output: format!("{}_{}", func, col),
            });
        }
        
        let operation = Operation::GroupBy {
            keys: self.keys.clone(),
            aggregates: agg_exprs,
        };
        
        if self.df.is_lazy() {
            let mut new_df = self.df.clone_empty();
            new_df.add_pending_operation(operation, vec![self.df.id().to_string()]);
            Ok(new_df)
        } else {
            self.df.execute_operation(operation, vec![self.df.id().to_string()])
        }
    }
}

// Implement ToDistributed for DataFrame
impl ToDistributed for crate::dataframe::DataFrame {
    fn to_distributed(&self, config: DistributedConfig) -> Result<DistributedDataFrame> {
        DistributedDataFrame::from_local(self, config)
    }
}

// Generate a unique ID for datasets and operations
fn generate_unique_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    
    format!("{:x}", now)
}

/// Placeholder execution engine for when distributed features are not enabled
#[cfg(not(feature = "distributed"))]
struct PlaceholderEngine {
    initialized: bool,
}

#[cfg(not(feature = "distributed"))]
impl PlaceholderEngine {
    fn new() -> Self {
        Self { initialized: false }
    }
}

#[cfg(not(feature = "distributed"))]
impl ExecutionEngine for PlaceholderEngine {
    fn initialize(&mut self, _config: &DistributedConfig) -> Result<()> {
        Err(crate::error::Error::InvalidOperation(
            "Distributed feature not enabled. Cannot initialize execution engine.".into()
        ))
    }
    
    fn is_initialized(&self) -> bool {
        false
    }
    
    fn create_context(&self, _config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>> {
        Err(crate::error::Error::InvalidOperation(
            "Distributed feature not enabled. Cannot create execution context.".into()
        ))
    }
}
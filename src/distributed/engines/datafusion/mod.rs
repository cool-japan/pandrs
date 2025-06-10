//! # DataFusion Execution Engine
//!
//! This module provides an implementation of the execution engine interface
//! using Apache Arrow DataFusion.

// DataFusion conversion utilities
pub mod conversion;

#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use crate::distributed::core::config::DistributedConfig;
#[cfg(feature = "distributed")]
use crate::distributed::core::partition::PartitionSet;
#[cfg(feature = "distributed")]
use crate::distributed::execution::{
    AggregateExpr, ExecutionContext, ExecutionEngine, ExecutionMetrics, ExecutionPlan,
    ExecutionResult, JoinType, Operation, SortExpr,
};
#[cfg(feature = "distributed")]
use crate::error::{Error, Result};

/// DataFusion execution engine
#[cfg(feature = "distributed")]
pub struct DataFusionEngine {
    /// Whether the engine is initialized
    initialized: bool,
    /// Configuration
    config: Option<DistributedConfig>,
}

#[cfg(feature = "distributed")]
impl DataFusionEngine {
    /// Creates a new DataFusion engine
    pub fn new() -> Self {
        Self {
            initialized: false,
            config: None,
        }
    }
}

#[cfg(feature = "distributed")]
impl ExecutionEngine for DataFusionEngine {
    fn initialize(&mut self, config: &DistributedConfig) -> Result<()> {
        self.initialized = true;
        self.config = Some(config.clone());
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn create_context(&self, config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>> {
        if !self.initialized {
            return Err(Error::InvalidState("Engine not initialized".to_string()));
        }

        let ctx = DataFusionContext::new(config);
        Ok(Box::new(ctx))
    }

    fn clone(&self) -> Box<dyn ExecutionEngine> {
        Box::new(Self {
            initialized: self.initialized,
            config: self.config.clone(),
        })
    }
}

/// DataFusion execution context
#[cfg(feature = "distributed")]
pub struct DataFusionContext {
    /// DataFusion context
    #[cfg(feature = "distributed")]
    context: datafusion::execution::context::SessionContext,
    /// Configuration
    config: DistributedConfig,
    /// Registered datasets
    registered_tables: HashMap<String, PartitionSet>,
    /// Execution metrics
    metrics: ExecutionMetrics,
}

#[cfg(feature = "distributed")]
impl DataFusionContext {
    /// Creates a new DataFusion context
    pub fn new(config: &DistributedConfig) -> Self {
        // Create DataFusion configuration
        let mut df_config = datafusion::execution::context::SessionConfig::new();

        // Set concurrency
        df_config = df_config.with_target_partitions(config.concurrency());

        // Set memory limit if provided
        if let Some(limit) = config.memory_limit() {
            df_config = df_config.with_mem_limit(limit);
        }

        // Set optimization options
        if config.enable_optimization() {
            for (rule, value) in config.optimizer_rules() {
                if let Ok(bool_value) = value.parse::<bool>() {
                    df_config = df_config.set_bool_var(rule, bool_value);
                }
            }
        }

        // Create DataFusion context
        let context = datafusion::execution::context::SessionContext::new_with_config(df_config);

        Self {
            context,
            config: config.clone(),
            registered_tables: HashMap::new(),
            metrics: ExecutionMetrics::new(),
        }
    }
}

#[cfg(feature = "distributed")]
impl ExecutionContext for DataFusionContext {
    fn execute_plan(&mut self, plan: ExecutionPlan) -> Result<ExecutionResult> {
        // Convert the execution plan to SQL
        let sql = self.plan_to_sql(&plan)?;
        
        // Execute the SQL
        self.sql(&sql)
    }
    
}

impl DataFusionContext {
    /// Helper method to convert ExecutionPlan to SQL (private implementation)
    fn plan_to_sql(&self, plan: &ExecutionPlan) -> Result<String> {
        let mut sql = format!("SELECT * FROM {}", plan.input());
        
        for operation in plan.operations() {
            match operation {
                Operation::Filter(condition) => {
                    if sql.contains("WHERE") {
                        sql = format!("{} AND {}", sql, condition);
                    } else {
                        sql = format!("{} WHERE {}", sql, condition);
                    }
                },
                Operation::Select(columns) => {
                    let column_list = columns.join(", ");
                    sql = sql.replace("SELECT *", &format!("SELECT {}", column_list));
                },
                Operation::Aggregate(group_by, aggregates) => {
                    let agg_exprs: Vec<String> = aggregates.iter()
                        .map(|agg| {
                            let func_upper = agg.function.to_uppercase();
                            let func_lower = agg.function.to_lowercase();
                            format!("{}({}) as {}_{}", func_upper, agg.column, func_lower, agg.column)
                        })
                        .collect();
                    
                    if !group_by.is_empty() {
                        let group_columns = group_by.join(", ");
                        sql = format!("SELECT {}, {} FROM ({}) GROUP BY {}",
                            group_columns, agg_exprs.join(", "), sql, group_columns);
                    } else {
                        sql = format!("SELECT {} FROM ({})", agg_exprs.join(", "), sql);
                    }
                },
                Operation::OrderBy(sort_exprs) => {
                    let sort_list: Vec<String> = sort_exprs.iter()
                        .map(|expr| format!("{} {}",
                            expr.column,
                            if expr.ascending { "ASC" } else { "DESC" }
                        ))
                        .collect();
                    sql = format!("{} ORDER BY {}", sql, sort_list.join(", "));
                },
                Operation::Limit(n) => {
                    sql = format!("{} LIMIT {}", sql, n);
                },
                Operation::Join { join_type, right, left_keys: _, right_keys: _ } => {
                    // For now, use simplified join syntax
                    let on_condition = "true"; // Placeholder
                    let join_type_str = match join_type {
                        JoinType::Inner => "INNER JOIN",
                        JoinType::Left => "LEFT JOIN",
                        JoinType::Right => "RIGHT JOIN",
                        JoinType::Full => "FULL OUTER JOIN",
                        JoinType::Cross => "CROSS JOIN",
                    };
                    sql = format!("{} {} {} ON {}", sql, join_type_str, right, on_condition);
                },
                Operation::Distinct => {
                    sql = sql.replace("SELECT", "SELECT DISTINCT");
                },
                _ => {
                    // For now, ignore unsupported operations
                    // TODO: Implement remaining operations
                }
            }
        }
        
        Ok(sql)
    }
}

#[cfg(feature = "distributed")]
impl ExecutionContext for DataFusionContext {
    fn register_in_memory_table(&mut self, name: &str, partitions: PartitionSet) -> Result<()> {
        use std::sync::Arc;
        use datafusion::datasource::MemTable;
        use datafusion::arrow::record_batch::RecordBatch;
        
        // Convert partitions to record batches
        let mut batches = Vec::new();
        let mut schema = None;
        
        for partition in partitions.partitions() {
            if let Some(data) = partition.data() {
                if schema.is_none() {
                    schema = Some(data.schema());
                }
                batches.push(data.clone());
            }
        }
        
        if batches.is_empty() {
            return Err(Error::InvalidValue("No data in partition set".to_string()));
        }
        
        let schema = schema.ok_or_else(|| {
            Error::InvalidValue("No schema found in partitions".to_string())
        })?;
        
        // Create memory table
        let mem_table = MemTable::try_new(schema, vec![batches])
            .map_err(|e| Error::InvalidValue(format!("Failed to create memory table: {}", e)))?;
        
        // Register the table
        self.context.register_table(name, Arc::new(mem_table))
            .map_err(|e| Error::InvalidValue(format!("Failed to register memory table: {}", e)))?;
        
        // Store the partition set
        self.registered_tables.insert(name.to_string(), partitions);
        
        Ok(())
    }

    fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        use std::sync::Arc;
        use datafusion::datasource::file_format::csv::CsvFormat;
        use datafusion::datasource::listing::ListingOptions;
        use datafusion::datasource::listing::ListingTable;
        use datafusion::datasource::listing::ListingTableConfig;
        use datafusion::datasource::listing::ListingTableUrl;
        
        // Create CSV format with default options
        let csv_format = Arc::new(CsvFormat::default());
        
        // Parse the file path
        let table_url = ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid CSV path: {}", e)))?;
        
        // Create listing options
        let listing_options = ListingOptions::new(csv_format)
            .with_file_extension(".csv");
        
        // Create table config
        let config = ListingTableConfig::new(table_url)
            .with_listing_options(listing_options);
        
        // Create the table
        let table = ListingTable::try_new(config)
            .map_err(|e| Error::InvalidValue(format!("Failed to create CSV table: {}", e)))?;
        
        // Register the table
        self.context.register_table(name, Arc::new(table))
            .map_err(|e| Error::InvalidValue(format!("Failed to register CSV table: {}", e)))?;
        
        // Track the registered table (empty partition set for now)
        self.registered_tables.insert(name.to_string(), PartitionSet::new(Vec::new()));
        
        Ok(())
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        use std::sync::Arc;
        use datafusion::datasource::file_format::parquet::ParquetFormat;
        use datafusion::datasource::listing::ListingOptions;
        use datafusion::datasource::listing::ListingTable;
        use datafusion::datasource::listing::ListingTableConfig;
        use datafusion::datasource::listing::ListingTableUrl;
        
        // Create Parquet format
        let parquet_format = Arc::new(ParquetFormat::default());
        
        // Parse the file path
        let table_url = ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid Parquet path: {}", e)))?;
        
        // Create listing options
        let listing_options = ListingOptions::new(parquet_format)
            .with_file_extension(".parquet");
        
        // Create table config
        let config = ListingTableConfig::new(table_url)
            .with_listing_options(listing_options);
        
        // Create the table
        let table = ListingTable::try_new(config)
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet table: {}", e)))?;
        
        // Register the table
        self.context.register_table(name, Arc::new(table))
            .map_err(|e| Error::InvalidValue(format!("Failed to register Parquet table: {}", e)))?;
        
        // Track the registered table (empty partition set for now)
        self.registered_tables.insert(name.to_string(), PartitionSet::new(Vec::new()));
        
        Ok(())
    }

    fn sql(&mut self, query: &str) -> Result<ExecutionResult> {
        use datafusion::execution::context::SessionContext;
        use std::sync::Arc;
        
        // Record start time for metrics
        let start_time = std::time::Instant::now();
        
        // Execute the SQL query (async)
        let df = futures::executor::block_on(self.context.sql(query))
            .map_err(|e| Error::InvalidValue(format!("SQL execution failed: {}", e)))?;
        
        // Get the schema
        let schema = df.schema();
        
        // Collect the results (async)
        let batches = futures::executor::block_on(df.collect())
            .map_err(|e| Error::InvalidValue(format!("Failed to collect query results: {}", e)))?;
        
        // Record end time
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Calculate total rows
        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        
        // Convert to partitions (simplified - treating each batch as a partition)
        let mut partitions = Vec::new();
        for (i, batch) in batches.into_iter().enumerate() {
            let partition = crate::distributed::core::partition::Partition::new(i, batch);
            partitions.push(std::sync::Arc::new(partition));
        }
        
        // Create partition set
        let partition_set = crate::distributed::core::partition::PartitionSet::new(partitions, schema.inner().clone());
        
        // Update metrics
        let mut metrics = self.metrics.clone();
        metrics.add_execution_time(execution_time);
        metrics.add_rows_processed(total_rows);
        self.metrics = metrics.clone();
        
        // Create execution result
        Ok(ExecutionResult::new(partition_set, schema.inner().clone(), metrics))
    }

    fn table_schema(&self, name: &str) -> Result<arrow::datatypes::SchemaRef> {
        // Check if the table is registered
        if !self.registered_tables.contains_key(name) {
            return Err(Error::ColumnNotFound(format!("Table '{}' not found", name)));
        }

        // Get the table from the catalog
        let catalog = self.context.catalog("datafusion").ok_or_else(|| {
            Error::InvalidValue("Default catalog not found".to_string())
        })?;

        let schema_provider = catalog.schema("public").ok_or_else(|| {
            Error::InvalidValue("Default schema not found".to_string())
        })?;

        let table = futures::executor::block_on(schema_provider.table(name)).ok_or_else(|| {
            Error::ColumnNotFound(format!("Table '{}' not found in catalog", name))
        })?;

        Ok(table.schema())
    }

    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String> {
        let mut explanation = String::new();
        
        explanation.push_str(&format!("Execution Plan for input: {}\n", plan.input()));
        explanation.push_str("===================\n");
        
        for (i, operation) in plan.operations().iter().enumerate() {
            explanation.push_str(&format!("{}. {:?}\n", i + 1, operation));
        }
        
        if with_statistics {
            explanation.push_str("\nStatistics:\n");
            explanation.push_str("-----------\n");
            explanation.push_str(&format!("Total operations: {}\n", plan.operations().len()));
            explanation.push_str(&format!("Input dataset: {}\n", plan.input()));
            
            // Add operation type summary
            let mut op_counts = std::collections::HashMap::new();
            for op in plan.operations() {
                let op_type = match op {
                    Operation::Select(_) => "Select",
                    Operation::Filter(_) => "Filter",
                    Operation::Join { .. } => "Join",
                    Operation::Aggregate(_, _) => "Aggregate",
                    Operation::OrderBy(_) => "OrderBy",
                    Operation::Limit(_) => "Limit",
                    // Operation::Window(_) => "Window", // Temporarily disabled
                    Operation::Project(_) => "Project",
                    Operation::Distinct => "Distinct",
                    Operation::Union(_) => "Union",
                    Operation::Intersect(_) => "Intersect",
                    Operation::Except(_) => "Except",
                    Operation::Custom { .. } => "Custom",
                };
                *op_counts.entry(op_type).or_insert(0) += 1;
            }
            
            for (op_type, count) in op_counts {
                explanation.push_str(&format!("{}: {}\n", op_type, count));
            }
        }
        
        Ok(explanation)
    }

    fn write_parquet(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use datafusion::arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::fs::File;
        use std::sync::Arc;
        
        // Collect all record batches
        let batches = result.collect()?;
        
        if batches.is_empty() {
            return Err(Error::InvalidValue("No data to write".to_string()));
        }
        
        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create file: {}", e)))?;
        
        // Create parquet writer
        let mut writer = ArrowWriter::try_new(file, result.schema().clone(), None)
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet writer: {}", e)))?;
        
        // Write all batches
        for batch in batches {
            writer.write(&batch)
                .map_err(|e| Error::InvalidValue(format!("Failed to write batch: {}", e)))?;
        }
        
        // Close the writer
        writer.close()
            .map_err(|e| Error::InvalidValue(format!("Failed to close writer: {}", e)))?;
        
        Ok(())
    }

    fn write_csv(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use arrow::csv::WriterBuilder;
        use std::fs::File;
        
        // Collect all record batches
        let batches = result.collect()?;
        
        if batches.is_empty() {
            return Err(Error::InvalidValue("No data to write".to_string()));
        }
        
        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create file: {}", e)))?;
        
        // Create CSV writer
        let mut writer = WriterBuilder::new().build(file);
        
        // Write all batches
        for batch in batches {
            writer.write(&batch)
                .map_err(|e| Error::InvalidValue(format!("Failed to write batch: {}", e)))?;
        }
        
        Ok(())
    }

    fn metrics(&self) -> Result<ExecutionMetrics> {
        Ok(self.metrics.clone())
    }

    fn clone(&self) -> Box<dyn ExecutionContext> {
        let cloned = DataFusionContext {
            context: datafusion::execution::context::SessionContext::with_config(
                self.context.copied_config()
            ),
            config: self.config.clone(),
            registered_tables: self.registered_tables.clone(),
            metrics: self.metrics.clone(),
        };
        
        Box::new(cloned)
    }
}

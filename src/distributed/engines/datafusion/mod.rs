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
            return Err(Error::InvalidOperation(
                "Engine not initialized".to_string(),
            ));
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

    fn register_in_memory_table(&mut self, name: &str, partitions: PartitionSet) -> Result<()> {
        use datafusion::arrow::record_batch::RecordBatch;
        use datafusion::datasource::MemTable;
        use std::sync::Arc;

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

        let schema = schema
            .ok_or_else(|| Error::InvalidValue("No schema found in partitions".to_string()))?;

        // Create memory table
        let mem_table = MemTable::try_new(schema, vec![batches])
            .map_err(|e| Error::InvalidValue(format!("Failed to create memory table: {}", e)))?;

        // Register table with DataFusion
        self.context
            .register_table(name, Arc::new(mem_table))
            .map_err(|e| Error::InvalidValue(format!("Failed to register table: {}", e)))?;

        // Store in our registry
        self.registered_tables.insert(name.to_string(), partitions);

        Ok(())
    }

    fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        use datafusion::datasource::file_format::csv::CsvFormat;
        use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig};
        use datafusion::datasource::object_store::ObjectStoreUrl;
        use std::sync::Arc;

        // Create CSV format options
        let file_format = Arc::new(CsvFormat::default().with_has_header(true));

        // Create listing options
        let listing_options = ListingOptions::new(file_format).with_file_extension(".csv");

        // Create table path
        let table_path = datafusion::datasource::listing::ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid CSV path: {}", e)))?;

        // Register CSV file with DataFusion
        futures::executor::block_on(async {
            self.context
                .register_listing_table(name, table_path, listing_options, None, None)
                .await
        })
        .map_err(|e| Error::InvalidValue(format!("Failed to register CSV table: {}", e)))?;

        Ok(())
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        use datafusion::datasource::file_format::parquet::ParquetFormat;
        use datafusion::datasource::listing::{ListingOptions, ListingTableConfig};
        use std::sync::Arc;

        // Create Parquet format options
        let file_format = Arc::new(ParquetFormat::default());

        // Create listing options
        let listing_options = ListingOptions::new(file_format).with_file_extension(".parquet");

        // Create table path
        let table_path = datafusion::datasource::listing::ListingTableUrl::parse(path)
            .map_err(|e| Error::InvalidValue(format!("Invalid Parquet path: {}", e)))?;

        // Register Parquet file with DataFusion
        futures::executor::block_on(async {
            self.context
                .register_listing_table(name, table_path, listing_options, None, None)
                .await
        })
        .map_err(|e| Error::InvalidValue(format!("Failed to register Parquet table: {}", e)))?;

        Ok(())
    }

    fn sql(&mut self, query: &str) -> Result<ExecutionResult> {
        use crate::distributed::core::partition::{Partition, PartitionSet};

        // Execute SQL query using DataFusion
        let sql_result = futures::executor::block_on(async {
            let df = self.context.sql(query).await?;
            df.collect().await
        })
        .map_err(|e| Error::InvalidValue(format!("SQL execution failed: {}", e)))?;

        // Convert result to our format
        let mut partitions = Vec::new();
        for (i, batch) in sql_result.iter().enumerate() {
            partitions.push(Arc::new(Partition::new(i, batch.clone())));
        }

        let schema = if sql_result.is_empty() {
            use arrow::datatypes::Schema;
            std::sync::Arc::new(Schema::new(vec![]))
        } else {
            sql_result[0].schema()
        };

        let partition_set = PartitionSet::new(partitions, schema);

        Ok(ExecutionResult::new(
            partition_set,
            schema,
            self.metrics.clone(),
        ))
    }

    fn table_schema(&self, name: &str) -> Result<arrow::datatypes::SchemaRef> {
        // Try to get table schema from DataFusion context
        if let Some(table) =
            futures::executor::block_on(async { self.context.table(name).await.ok() })
        {
            let schema = table.schema();
            Ok(schema.into())
        } else {
            // If table not found, check our registered tables
            if let Some(partition_set) = self.registered_tables.get(name) {
                Ok(partition_set.schema())
            } else {
                Err(Error::InvalidValue(format!("Table '{}' not found", name)))
            }
        }
    }

    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String> {
        // Convert execution plan to SQL and explain it
        let sql = self.plan_to_sql(plan)?;

        let explain_sql = if with_statistics {
            format!("EXPLAIN (ANALYZE true, VERBOSE true) {}", sql)
        } else {
            format!("EXPLAIN {}", sql)
        };

        // Execute the explain query
        let result = futures::executor::block_on(async {
            let df = self.context.sql(&explain_sql).await?;
            df.collect().await
        })
        .map_err(|e| Error::InvalidValue(format!("Plan explanation failed: {}", e)))?;

        // Convert result to string
        let mut explanation = String::new();
        for batch in result {
            if let Some(column) = batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
            {
                for i in 0..column.len() {
                    if let Some(line) = column.value(i) {
                        explanation.push_str(line);
                        explanation.push('\n');
                    }
                }
            }
        }

        Ok(explanation)
    }

    fn write_parquet(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use parquet::arrow::arrow_writer::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        // Create writer properties with compression
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet file: {}", e)))?;

        // Create Arrow writer
        let mut writer = ArrowWriter::try_new(file, result.schema(), Some(props))
            .map_err(|e| Error::InvalidValue(format!("Failed to create Parquet writer: {}", e)))?;

        // Write all partitions
        for partition in result.partitions().partitions() {
            if let Some(batch) = partition.data() {
                writer.write(batch).map_err(|e| {
                    Error::InvalidValue(format!("Failed to write Parquet batch: {}", e))
                })?;
            }
        }

        // Close writer
        writer
            .close()
            .map_err(|e| Error::InvalidValue(format!("Failed to close Parquet writer: {}", e)))?;

        Ok(())
    }

    fn write_csv(&mut self, result: &ExecutionResult, path: &str) -> Result<()> {
        use arrow::csv::Writer;
        use std::fs::File;

        // Create output file
        let file = File::create(path)
            .map_err(|e| Error::InvalidValue(format!("Failed to create CSV file: {}", e)))?;

        // Create CSV writer with headers
        let mut writer = Writer::new(file);

        // Write all partitions
        for partition in result.partitions().partitions() {
            if let Some(batch) = partition.data() {
                writer.write(batch).map_err(|e| {
                    Error::InvalidValue(format!("Failed to write CSV batch: {}", e))
                })?;
            }
        }

        Ok(())
    }

    fn metrics(&self) -> Result<ExecutionMetrics> {
        Ok(self.metrics.clone())
    }

    fn clone(&self) -> Box<dyn ExecutionContext> {
        // Create new context with same configuration
        let mut new_context = DataFusionContext::new(&self.config);

        // Copy registered tables
        new_context.registered_tables = self.registered_tables.clone();

        // Copy metrics
        new_context.metrics = self.metrics.clone();

        Box::new(new_context)
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
                }
                Operation::Select(columns) => {
                    let column_list = columns.join(", ");
                    sql = sql.replace("SELECT *", &format!("SELECT {}", column_list));
                }
                Operation::Aggregate(group_by, aggregates) => {
                    let agg_exprs: Vec<String> = aggregates
                        .iter()
                        .map(|agg| {
                            let func_upper = agg.function.to_uppercase();
                            let func_lower = agg.function.to_lowercase();
                            format!(
                                "{}({}) as {}_{}",
                                func_upper, agg.column, func_lower, agg.column
                            )
                        })
                        .collect();

                    if !group_by.is_empty() {
                        let group_columns = group_by.join(", ");
                        sql = format!(
                            "SELECT {}, {} FROM ({}) GROUP BY {}",
                            group_columns,
                            agg_exprs.join(", "),
                            sql,
                            group_columns
                        );
                    } else {
                        sql = format!("SELECT {} FROM ({})", agg_exprs.join(", "), sql);
                    }
                }
                Operation::OrderBy(sort_exprs) => {
                    let sort_list: Vec<String> = sort_exprs
                        .iter()
                        .map(|expr| {
                            format!(
                                "{} {}",
                                expr.column,
                                if expr.ascending { "ASC" } else { "DESC" }
                            )
                        })
                        .collect();
                    sql = format!("{} ORDER BY {}", sql, sort_list.join(", "));
                }
                Operation::Limit(n) => {
                    sql = format!("{} LIMIT {}", sql, n);
                }
                Operation::Join {
                    join_type,
                    right,
                    left_keys: _,
                    right_keys: _,
                } => {
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
                }
                Operation::Distinct => {
                    sql = sql.replace("SELECT", "SELECT DISTINCT");
                }
                _ => {
                    // For now, ignore unsupported operations
                    // TODO: Implement remaining operations
                }
            }
        }

        Ok(sql)
    }
}

//! # DataFusion Execution Context
//!
//! This module provides an implementation of the ExecutionContext interface using DataFusion.

use std::sync::Arc;
use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, Error};
use crate::distributed::config::DistributedConfig;
use crate::distributed::execution::{ExecutionContext, ExecutionPlan, ExecutionResult, Operation, AggregateExpr, JoinType, SortExpr};
use crate::distributed::partition::{PartitionSet, Partition, PartitionMetadata};
use crate::distributed::expr::{ColumnProjection, UdfDefinition, ExprSchema};
use crate::distributed::schema_validator::SchemaValidator;
use crate::distributed::explain::{ExplainOptions, ExplainFormat, explain_plan};
use super::conversion::{dataframe_to_record_batches, record_batches_to_dataframe};

/// DataFusion execution context implementation
pub struct DataFusionContext {
    /// Configuration for distributed processing
    config: DistributedConfig,
    /// DataFusion session context
    #[cfg(feature = "distributed")]
    context: datafusion::execution::context::SessionContext,
    /// Registered datasets
    datasets: HashMap<String, PartitionSet>,
    /// Schema validator
    schema_validator: SchemaValidator,
}

impl DataFusionContext {
    /// Creates a new DataFusion context
    pub fn new(config: DistributedConfig) -> Self {
        #[cfg(feature = "distributed")]
        let context = {
            // Create the config with performance optimizations
            let mut config_builder = datafusion::execution::context::SessionConfig::new()
                .with_target_partitions(config.concurrency())
                // Batch size configuration for better performance
                .with_batch_size(8192);

            // Add memory limit if specified
            let mut config_builder = if let Some(memory_limit) = config.memory_limit() {
                config_builder.with_memory_limit(memory_limit as u64)
            } else {
                config_builder
            };

            // Set parquet parallel read for better IO performance if file operations are involved
            config_builder = config_builder.set("parquet.parallel_read", "true");

            // Apply optimization settings if enabled
            if config.enable_optimization() {
                // Convert optimizer rules to DataFusion format
                for (rule, value) in config.optimizer_rules() {
                    config_builder = config_builder.set(
                        &format!("optimizer.{}", rule),
                        value
                    );
                }

                // Enable statistics-based optimization
                config_builder = config_builder
                    .set("statistics.enabled", "true")
                    .set("optimizer.statistics_based_join_ordering", "true");
            } else {
                // Disable optimization if not enabled
                config_builder = config_builder
                    .set("optimizer.skip_optimize", "true")
                    .set("statistics.enabled", "false");
            }

            // Create session context with the config
            datafusion::execution::context::SessionContext::with_config(config_builder)
        };

        Self {
            config,
            #[cfg(feature = "distributed")]
            context,
            datasets: HashMap::new(),
            schema_validator: SchemaValidator::new(),
        }
    }

    #[cfg(feature = "distributed")]
    fn register_record_batches(&mut self, name: &str, batches: Vec<datafusion::arrow::record_batch::RecordBatch>) -> Result<()> {
        if batches.is_empty() {
            return Err(Error::InvalidInput(format!("No record batches to register for dataset {}", name)));
        }

        let schema = batches[0].schema();

        // Create a memory table with options for better performance
        // Enable predicate pushdown for better filter performance
        let options = datafusion::datasource::MemTableConfig::new()
            .with_schema(schema)
            .with_batches(vec![batches]);

        let provider = options.build()
            .map_err(|e| Error::DistributedProcessing(format!("Failed to create memory table: {}", e)))?;

        // Register the table
        self.context.register_table(name, Arc::new(provider))
            .map_err(|e| Error::DistributedProcessing(format!("Failed to register table: {}", e)))?;

        Ok(())
    }

    #[cfg(feature = "distributed")]
    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        // Skip validation if configured to do so
        if self.config.skip_validation() {
            return Ok(());
        }

        // Validate the plan against schemas
        self.schema_validator.validate_plan(plan)
    }

    #[cfg(feature = "distributed")]
    fn convert_operation_to_sql(&self, plan: &ExecutionPlan) -> Result<String> {
        let mut sql_components = Vec::new();

        match &plan.operation() {
            Operation::Select { columns } => {
                let column_list = if columns.is_empty() {
                    "*".to_string()
                } else {
                    columns.join(", ")
                };

                sql_components.push(format!("SELECT {}", column_list));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));
            },
            Operation::Filter { predicate } => {
                sql_components.push(format!("SELECT *"));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));
                sql_components.push(format!("WHERE {}", predicate));
            },
            Operation::Join {
                right,
                join_type,
                left_keys,
                right_keys
            } => {
                sql_components.push(format!("SELECT *"));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));

                let join_type_str = match join_type {
                    JoinType::Inner => "INNER JOIN",
                    JoinType::Left => "LEFT JOIN",
                    JoinType::Right => "RIGHT JOIN",
                    JoinType::Full => "FULL OUTER JOIN",
                    JoinType::Cross => "CROSS JOIN",
                };

                let on_clause = if *join_type == JoinType::Cross {
                    String::new()
                } else {
                    let mut join_conditions = Vec::new();

                    for (i, left_key) in left_keys.iter().enumerate() {
                        if i < right_keys.len() {
                            let right_key = &right_keys[i];
                            join_conditions.push(format!("{}.{} = {}.{}",
                                plan.inputs()[0], left_key,
                                right, right_key));
                        }
                    }

                    format!(" ON {}", join_conditions.join(" AND "))
                };

                sql_components.push(format!("{} {}{}", join_type_str, right, on_clause));
            },
            Operation::GroupBy {
                keys,
                aggregates
            } => {
                let mut select_parts = Vec::new();

                // Add grouping keys to select
                for key in keys {
                    select_parts.push(key.clone());
                }

                // Add aggregate expressions
                for agg in aggregates {
                    select_parts.push(format!("{}({}) as {}",
                        agg.function, agg.input, agg.output));
                }

                sql_components.push(format!("SELECT {}", select_parts.join(", ")));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));
                sql_components.push(format!("GROUP BY {}", keys.join(", ")));
            },
            Operation::OrderBy {
                sort_exprs
            } => {
                sql_components.push(format!("SELECT *"));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));

                let mut order_parts = Vec::new();
                for expr in sort_exprs {
                    let direction = if expr.ascending { "ASC" } else { "DESC" };
                    let nulls = if expr.nulls_first { "NULLS FIRST" } else { "NULLS LAST" };
                    order_parts.push(format!("{} {} {}", expr.column, direction, nulls));
                }

                sql_components.push(format!("ORDER BY {}", order_parts.join(", ")));
            },
            Operation::Limit {
                limit
            } => {
                sql_components.push(format!("SELECT *"));
                sql_components.push(format!("FROM {}", plan.inputs()[0]));
                sql_components.push(format!("LIMIT {}", limit));
            },
            Operation::Window {
                window_functions,
            } => {
                // Start with a base SELECT
                let mut select_parts = Vec::new();

                // Include all columns from the input
                select_parts.push("input_table.*".to_string());

                // Add window function expressions
                for wf in window_functions {
                    select_parts.push(wf.to_sql());
                }

                sql_components.push(format!("SELECT {}", select_parts.join(", ")));
                sql_components.push(format!("FROM {} AS input_table", plan.inputs()[0]));
            },
            Operation::Custom {
                name,
                params
            } => {
                match name.as_str() {
                    "select_expr" => {
                        if let Some(projections_json) = params.get("projections") {
                            // Deserialize the projections
                            let projections: Vec<ColumnProjection> = serde_json::from_str(projections_json)
                                .map_err(|e| Error::DistributedProcessing(
                                    format!("Failed to parse projections: {}", e)
                                ))?;

                            // Build the select statement
                            let mut select_parts = Vec::new();
                            for projection in projections {
                                select_parts.push(projection.to_sql());
                            }

                            sql_components.push(format!("SELECT {}", select_parts.join(", ")));
                            sql_components.push(format!("FROM {}", plan.inputs()[0]));
                        } else {
                            return Err(Error::InvalidOperation(
                                "select_expr operation requires projections parameter".to_string()
                            ));
                        }
                    },
                    "with_column" => {
                        let column_name = params.get("column_name")
                            .ok_or_else(|| Error::InvalidOperation(
                                "with_column operation requires column_name parameter".to_string()
                            ))?;

                        if let Some(projection_json) = params.get("projection") {
                            // Deserialize the projection
                            let projection: ColumnProjection = serde_json::from_str(projection_json)
                                .map_err(|e| Error::DistributedProcessing(
                                    format!("Failed to parse projection: {}", e)
                                ))?;

                            // Build SQL that selects all existing columns plus the new one
                            sql_components.push(format!(
                                "SELECT *, {} AS {}",
                                projection.expr, column_name
                            ));
                            sql_components.push(format!("FROM {}", plan.inputs()[0]));
                        } else {
                            return Err(Error::InvalidOperation(
                                "with_column operation requires projection parameter".to_string()
                            ));
                        }
                    },
                    "create_udf" => {
                        if let Some(udfs_json) = params.get("udfs") {
                            // Deserialize the UDFs
                            let udfs: Vec<UdfDefinition> = serde_json::from_str(udfs_json)
                                .map_err(|e| Error::DistributedProcessing(
                                    format!("Failed to parse UDFs: {}", e)
                                ))?;

                            // DataFusion requires executing each UDF creation separately
                            // We'll return a sequence of CREATE FUNCTION statements
                            let mut udf_statements = Vec::new();
                            for udf in udfs {
                                udf_statements.push(udf.to_sql());
                            }

                            // Return all the UDF statements to be executed separately
                            return Ok(udf_statements.join(";\n"));
                        } else {
                            return Err(Error::InvalidOperation(
                                "create_udf operation requires udfs parameter".to_string()
                            ));
                        }
                    },
                    _ => {
                        return Err(Error::NotImplemented(
                            format!("Custom operation '{}' cannot be converted to SQL", name)
                        ));
                    }
                }
            },
        }

        Ok(sql_components.join(" "))
    }
}

impl ExecutionContext for DataFusionContext {
    fn execute(&self, plan: &ExecutionPlan) -> Result<ExecutionResult> {
        #[cfg(feature = "distributed")]
        {
            use std::time::Instant;

            // Start timing the execution
            let start_time = Instant::now();

            // Validate the plan against schemas
            self.validate_plan(plan)?;

            // Check if this is a UDF creation operation
            let is_udf_creation = match plan.operation() {
                Operation::Custom { name, .. } => name == "create_udf",
                _ => false,
            };

            // Convert the execution plan to SQL
            let sql = self.convert_operation_to_sql(plan)?;

            // For UDF operations, we need to execute multiple statements
            let arrow_batches = if is_udf_creation {
                // Split the SQL into individual statements
                let statements: Vec<&str> = sql.split(";\n").collect();

                // Execute each UDF creation statement
                for stmt in &statements {
                    if !stmt.trim().is_empty() {
                        self.context.sql(stmt)
                            .map_err(|e| Error::DistributedProcessing(
                                format!("Failed to execute UDF creation: {}", e)
                            ))?;
                    }
                }

                // UDFs don't return data, so we return an empty result
                Vec::new()
            } else {
                // For normal operations, execute the query and collect results
                let df = self.context.sql(&sql)
                    .map_err(|e| Error::DistributedProcessing(format!("Failed to execute SQL query: {}", e)))?;

                // Collect the results
                df.collect()
                    .map_err(|e| Error::DistributedProcessing(format!("Failed to collect query results: {}", e)))?;
            };

            // Calculate execution time
            let execution_time_ms = start_time.elapsed().as_millis() as u64;

            // Convert Arrow batches to PandRS DataFrame
            let pandrs_df = record_batches_to_dataframe(&arrow_batches)?;

            // Convert to partitions
            let batches_count = arrow_batches.len();
            let row_count = pandrs_df.shape()?.0;

            // Calculate memory usage (rough estimate)
            let bytes_processed =
                if batches_count > 0 && row_count > 0 {
                    // Estimate memory usage based on schema and row count
                    let schema = arrow_batches[0].schema();
                    let bytes_per_row = schema.fields().iter()
                        .map(|f| match f.data_type() {
                            arrow::datatypes::DataType::Int64 => 8,
                            arrow::datatypes::DataType::Float64 => 8,
                            arrow::datatypes::DataType::Utf8 => 24, // Average string size estimate
                            arrow::datatypes::DataType::Boolean => 1,
                            _ => 8, // Default estimate
                        })
                        .sum::<usize>();

                    bytes_per_row * row_count
                } else {
                    0
                };

            // Create execution metrics with actual timing and size information
            let metrics = crate::distributed::execution::ExecutionMetrics::new(
                execution_time_ms,
                row_count,
                batches_count,
                bytes_processed,
                bytes_processed, // Output size is roughly the same
            );

            // Create a single partition from the result
            let batch_size = if batches_count > 0 { row_count / batches_count } else { row_count };
            let result_batches = dataframe_to_record_batches(&pandrs_df, batch_size)?;

            // Create partitions
            let mut partitions = Vec::new();
            for (i, batch) in result_batches.iter().enumerate() {
                let partition = Partition::new(i, batch.clone());
                partitions.push(Arc::new(partition));
            }

            // Create a partition set
            let partition_set = PartitionSet::new(
                partitions,
                result_batches[0].schema(),
            );

            // Return the execution result
            Ok(ExecutionResult::new(
                partition_set,
                metrics,
            ))
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }

    fn register_dataset(&mut self, name: &str, partitions: PartitionSet) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            // Get all partitions with data
            let mut batches = Vec::new();

            for partition in partitions.partitions() {
                if let Some(data) = partition.data() {
                    batches.push(data.clone());
                }
            }

            if batches.is_empty() {
                return Err(Error::InvalidInput(format!("No data partitions found for dataset {}", name)));
            }

            // Register with DataFusion
            self.register_record_batches(name, batches)?;

            // Register the schema with the validator
            let schema = partitions.schema();
            self.schema_validator.register_arrow_schema(name, schema.clone())?;

            // Store in our local registry too
            self.datasets.insert(name.to_string(), partitions);

            Ok(())
        }

        #[cfg(not(feature = "distributed"))]
        {
            self.datasets.insert(name.to_string(), partitions);
            Ok(())
        }
    }

    fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            // Check if file exists
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("CSV file not found: {}", path)));
            }

            // Register CSV with DataFusion
            self.context.register_csv(name, path, datafusion::datasource::file_format::csv::CsvReadOptions::new())
                .map_err(|e| Error::DistributedProcessing(format!("Failed to register CSV: {}", e)))?;

            Ok(())
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "CSV registration is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }

    fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            // Check if file exists
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("Parquet file not found: {}", path)));
            }

            // Register Parquet with DataFusion
            self.context.register_parquet(name, path, datafusion::datasource::file_format::parquet::ParquetReadOptions::default())
                .map_err(|e| Error::DistributedProcessing(format!("Failed to register Parquet: {}", e)))?;

            Ok(())
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Parquet registration is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }

    fn explain_plan(&self, plan: &ExecutionPlan, with_statistics: bool) -> Result<String> {
        #[cfg(feature = "distributed")]
        {
            // Create explain options
            let options = ExplainOptions {
                format: ExplainFormat::Text,
                with_statistics,
                optimized: self.config.enable_optimization(),
                analyze: false,
            };

            // Get the plan explanation
            let explanation = explain_plan(plan, &options)?;

            // If optimization is enabled, also get the optimized plan from DataFusion
            if self.config.enable_optimization() {
                let sql = self.convert_operation_to_sql(plan)?;
                let df = self.context.sql(&format!("EXPLAIN {}", sql))
                    .map_err(|e| Error::DistributedProcessing(
                        format!("Failed to explain query: {}", e)))?;

                // Convert to string
                let batches = df.collect()
                    .map_err(|e| Error::DistributedProcessing(
                        format!("Failed to collect explain results: {}", e)))?;

                // Extract the explain text
                let mut optimized_plan = String::new();
                for batch in &batches {
                    if let Some(array) = batch.column(0).as_any().downcast_ref::<arrow::array::StringArray>() {
                        for i in 0..array.len() {
                            if !array.is_null(i) {
                                optimized_plan.push_str(array.value(i));
                                optimized_plan.push('\n');
                            }
                        }
                    }
                }

                // Combine the results
                let mut result = String::new();
                result.push_str("=== Logical Plan ===\n");
                result.push_str(&explanation);
                result.push_str("\n\n=== Optimized Plan ===\n");
                result.push_str(&optimized_plan);

                Ok(result)
            } else {
                Ok(explanation)
            }
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Plan explanation is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }

    fn sql(&self, query: &str) -> Result<ExecutionResult> {
        #[cfg(feature = "distributed")]
        {
            use std::time::Instant;

            // Start timing the execution
            let start_time = Instant::now();

            // Execute the SQL query
            let df = self.context.sql(query)
                .map_err(|e| Error::DistributedProcessing(format!("Failed to execute SQL query: {}", e)))?;

            // Collect the results
            let arrow_batches = df.collect()
                .map_err(|e| Error::DistributedProcessing(format!("Failed to collect query results: {}", e)))?;

            // Calculate execution time
            let execution_time_ms = start_time.elapsed().as_millis() as u64;

            // Convert Arrow batches to PandRS DataFrame
            let pandrs_df = record_batches_to_dataframe(&arrow_batches)?;

            // Convert to partitions
            let batches_count = arrow_batches.len();
            let row_count = pandrs_df.shape()?.0;

            // Calculate memory usage (rough estimate)
            let bytes_processed =
                if batches_count > 0 && row_count > 0 {
                    // Estimate memory usage based on schema and row count
                    let schema = arrow_batches[0].schema();
                    let bytes_per_row = schema.fields().iter()
                        .map(|f| match f.data_type() {
                            arrow::datatypes::DataType::Int64 => 8,
                            arrow::datatypes::DataType::Float64 => 8,
                            arrow::datatypes::DataType::Utf8 => 24, // Average string size estimate
                            arrow::datatypes::DataType::Boolean => 1,
                            _ => 8, // Default estimate
                        })
                        .sum::<usize>();

                    bytes_per_row * row_count
                } else {
                    0
                };

            // Create execution metrics with actual timing and size information
            let metrics = crate::distributed::execution::ExecutionMetrics::new(
                execution_time_ms,
                row_count,
                batches_count,
                bytes_processed,
                bytes_processed, // Output size is roughly the same
            );

            // Create a single partition from the result
            let batch_size = if batches_count > 0 { row_count / batches_count } else { row_count };
            let result_batches = dataframe_to_record_batches(&pandrs_df, batch_size)?;

            // Create partitions
            let mut partitions = Vec::new();
            for (i, batch) in result_batches.iter().enumerate() {
                let partition = Partition::new(i, batch.clone());
                partitions.push(Arc::new(partition));
            }

            // Create a partition set
            let partition_set = PartitionSet::new(
                partitions,
                if !result_batches.is_empty() { result_batches[0].schema() } else {
                    // Empty schema for empty results
                    Arc::new(arrow::datatypes::Schema::empty())
                },
            );

            // Return the execution result
            Ok(ExecutionResult::new(
                partition_set,
                metrics,
            ))
        }

        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "SQL execution is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
}
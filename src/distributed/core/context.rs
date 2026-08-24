//! # Distributed Processing Context
//!
//! This module provides a high-level context for distributed processing,
//! enabling management of multiple datasets.

#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use super::config::DistributedConfig;
#[cfg(feature = "distributed")]
use crate::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use crate::distributed::core::dataframe::DistributedDataFrame;
#[cfg(feature = "distributed")]
use crate::distributed::execution::{ExecutionContext, ExecutionEngine, ExecutionMetrics};
#[cfg(feature = "distributed")]
use crate::distributed::expr::ExprSchema;
#[cfg(feature = "distributed")]
use crate::distributed::schema_validator::SchemaValidator;
use crate::error::{Error, Result};
#[cfg(feature = "distributed")]
use crate::lock_safe;

/// A context for managing distributed processing operations
#[cfg(feature = "distributed")]
pub struct DistributedContext {
    /// Configuration for distributed processing
    config: DistributedConfig,
    /// Execution engine
    engine: Box<dyn ExecutionEngine>,
    /// Execution context
    context: Arc<Mutex<Box<dyn ExecutionContext>>>,
    /// Registered datasets
    datasets: HashMap<String, DistributedDataFrame>,
}

#[cfg(feature = "distributed")]
impl DistributedContext {
    /// Creates a new distributed context with local execution
    ///
    /// This is a convenience method that creates a local DataFusion context
    /// with the specified number of threads.
    ///
    /// # Arguments
    ///
    /// * `concurrency` - The number of threads to use for local execution
    ///
    /// # Returns
    ///
    /// A new `DistributedContext` configured for local execution
    pub fn new_local(concurrency: usize) -> Result<Self> {
        let config = DistributedConfig::new().with_concurrency(concurrency);
        Self::new(config)
    }

    /// Creates a new distributed context
    pub fn new(config: DistributedConfig) -> Result<Self> {
        // Create the engine based on the config. Ballista is not implemented;
        // return an honest error rather than silently running on DataFusion.
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
        let context = engine.create_context(&config)?;

        Ok(Self {
            config,
            engine,
            context: Arc::new(Mutex::new(context)),
            datasets: HashMap::new(),
        })
    }

    /// Registers a DataFrame with the context under the given name.
    ///
    /// The data is converted to Arrow record batches and registered as an
    /// in-memory table in **this context's shared execution context**, and the
    /// returned dataset shares that same context. Previously this registered a
    /// throwaway DataFrame in its own private context (immediately dropped) and
    /// then built the stored dataset around an *empty* cloned context — so
    /// nothing was ever queryable under `name`.
    pub fn register_dataframe(&mut self, name: &str, df: &DataFrame) -> Result<()> {
        use crate::distributed::core::partition::{Partition, PartitionSet};
        use crate::distributed::engines::datafusion::conversion::dataframe_to_record_batches;

        // Convert the local DataFrame to Arrow record batches.
        let row_count = df.row_count();
        let batch_size = std::cmp::max(1, row_count / std::cmp::max(1, self.config.concurrency()));
        let batches = dataframe_to_record_batches(df, batch_size)?;

        // Build a partition set from the batches.
        let mut partitions = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            partitions.push(Arc::new(Partition::new(i, batch.clone())));
        }
        let schema = batches.first().map(|b| b.schema()).ok_or_else(|| {
            Error::InvalidInput(format!("DataFrame '{}' produced no record batches", name))
        })?;
        let partition_set = PartitionSet::new(partitions, schema);

        // Register the data into the shared execution context under `name`.
        {
            let mut context = lock_safe!(self.context, "distributed context lock")?;
            context.register_in_memory_table(name, partition_set)?;
        }

        // The stored dataset SHARES the same context, so a query against it
        // resolves the table just registered.
        let dist_df = DistributedDataFrame::from_shared_context(
            self.config.clone(),
            self.engine.clone(),
            self.context.clone(),
            name.to_string(),
        );
        self.datasets.insert(name.to_string(), dist_df);

        Ok(())
    }

    /// Registers a CSV file with the context under the given name
    pub fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        let mut context = lock_safe!(self.context, "distributed context lock")?;
        context.register_csv(name, path)?;

        Ok(())
    }

    /// Registers a Parquet file with the context under the given name
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        let mut context = lock_safe!(self.context, "distributed context lock")?;
        context.register_parquet(name, path)?;

        Ok(())
    }

    /// Gets a registered dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&DistributedDataFrame> {
        self.datasets.get(name)
    }

    /// Gets a registered dataset by name (mutable)
    pub fn get_dataset_mut(&mut self, name: &str) -> Option<&mut DistributedDataFrame> {
        self.datasets.get_mut(name)
    }

    /// Gets the configuration
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Gets the execution engine
    pub fn engine(&self) -> &dyn ExecutionEngine {
        &*self.engine
    }

    /// Gets the execution context
    pub fn execution_context(&self) -> Arc<Mutex<Box<dyn ExecutionContext>>> {
        self.context.clone()
    }

    /// Gets the execution metrics
    pub fn metrics(&self) -> Result<ExecutionMetrics> {
        let context = lock_safe!(self.context, "distributed context lock")?;
        context.metrics()
    }

    /// Validates a schema against registered datasets
    pub fn validate_schema(&self, schema: &ExprSchema) -> Result<()> {
        // Create a schema validator with registered datasets
        let mut validator = SchemaValidator::new();

        // Register schemas from our datasets
        for (name, dataset) in &self.datasets {
            // Get schema from the dataset
            if let Ok(dataset_schema) = dataset.schema() {
                // Convert Arrow schema to ExprSchema
                let expr_schema = convert_arrow_schema_to_expr_schema(&dataset_schema)?;
                validator.register_schema(name.clone(), expr_schema);
            }
        }

        // Validate that all columns referenced in the schema exist in some registered dataset
        for (column_name, _column_meta) in schema.columns() {
            let mut found = false;
            for dataset_schema in validator.schemas().values() {
                if dataset_schema.has_column(column_name) {
                    found = true;
                    break;
                }
            }

            if !found {
                return Err(Error::InvalidOperation(format!(
                    "Column '{}' not found in any registered dataset",
                    column_name
                )));
            }
        }

        Ok(())
    }
}

/// Convert Arrow schema to ExprSchema
#[cfg(feature = "distributed")]
fn convert_arrow_schema_to_expr_schema(
    arrow_schema: &arrow::datatypes::SchemaRef,
) -> Result<ExprSchema> {
    let mut expr_schema = ExprSchema::new();

    for field in arrow_schema.fields() {
        let data_type = match field.data_type() {
            arrow::datatypes::DataType::Boolean => crate::distributed::expr::ExprDataType::Boolean,
            arrow::datatypes::DataType::Int8
            | arrow::datatypes::DataType::Int16
            | arrow::datatypes::DataType::Int32
            | arrow::datatypes::DataType::Int64 => crate::distributed::expr::ExprDataType::Integer,
            arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64 => {
                crate::distributed::expr::ExprDataType::Float
            }
            arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8 => {
                crate::distributed::expr::ExprDataType::String
            }
            arrow::datatypes::DataType::Date32 | arrow::datatypes::DataType::Date64 => {
                crate::distributed::expr::ExprDataType::Date
            }
            arrow::datatypes::DataType::Timestamp(_, _) => {
                crate::distributed::expr::ExprDataType::Timestamp
            }
            _ => crate::distributed::expr::ExprDataType::String, // Default to string for unknown types
        };

        let column_meta = crate::distributed::expr::ColumnMeta::new(
            field.name().clone(),
            data_type,
            field.is_nullable(),
            None, // No description
        );
        expr_schema.add_column(column_meta);
    }

    Ok(expr_schema)
}

/// Dummy implementation for the distributed context when the feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedContext;

/// Dummy implementation for the distributed context when the feature is not enabled
#[cfg(not(feature = "distributed"))]
impl DistributedContext {
    /// Creates a new dummy distributed context
    pub fn new(_config: super::DistributedConfig) -> Result<Self> {
        Err(Error::FeatureNotAvailable(
            "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
        ))
    }
}

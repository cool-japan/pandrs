//! # Distributed Processing Context
//!
//! This module provides a high-level context for distributed processing,
//! enabling management of multiple datasets and direct SQL query execution.

#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::path::Path;
#[cfg(feature = "distributed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "distributed")]
use super::config::DistributedConfig;
#[cfg(feature = "distributed")]
use crate::dataframe::DataFrame;
#[cfg(feature = "distributed")]
use crate::distributed::core::dataframe::DistributedDataFrame;
#[cfg(feature = "distributed")]
use crate::distributed::execution::{
    ExecutionContext, ExecutionEngine, ExecutionMetrics, ExecutionResult,
};
#[cfg(feature = "distributed")]
use crate::distributed::expr::ExprSchema;
#[cfg(feature = "distributed")]
use crate::distributed::schema_validator::SchemaValidator;
#[cfg(feature = "distributed")]
use crate::distributed::ToDistributed;
use crate::error::{Error, Result};

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
    /// Creates a new distributed context
    pub fn new(config: DistributedConfig) -> Result<Self> {
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
        let context = engine.create_context(&config)?;

        Ok(Self {
            config,
            engine,
            context: Arc::new(Mutex::new(context)),
            datasets: HashMap::new(),
        })
    }

    /// Registers a DataFrame with the context under the given name
    pub fn register_dataframe(&mut self, name: &str, df: &DataFrame) -> Result<()> {
        // Convert the DataFrame to a distributed DataFrame
        let dist_df = df.to_distributed(self.config.clone())?;

        // Clone the context reference for the distributed DataFrame
        let dist_df_with_context = DistributedDataFrame::new(
            self.config.clone(),
            self.engine.clone(),
            self.context.lock().unwrap().as_ref().clone(),
            name.to_string(),
        );

        // Register the distributed DataFrame
        self.datasets.insert(name.to_string(), dist_df_with_context);

        Ok(())
    }

    /// Registers a CSV file with the context under the given name
    pub fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        let mut context = self.context.lock().unwrap();
        context.register_csv(name, path)?;

        Ok(())
    }

    /// Registers a Parquet file with the context under the given name
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        let mut context = self.context.lock().unwrap();
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

    /// Executes a SQL query
    pub fn sql(&mut self, query: &str) -> Result<DistributedDataFrame> {
        let result = {
            let mut context = self.context.lock().unwrap();
            context.sql(query)?
        };

        // Create a new distributed DataFrame with the result
        let id = format!("sql_result_{}", self.datasets.len());

        let df = DistributedDataFrame::with_result(
            self.config.clone(),
            self.engine.clone(),
            self.context.lock().unwrap().as_ref().clone(),
            id.clone(),
            result,
        );

        // Register the result
        self.datasets.insert(id.clone(), df.clone());

        Ok(df)
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
        let context = self.context.lock().unwrap();
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

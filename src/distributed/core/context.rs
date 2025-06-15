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
        let validator = SchemaValidator::new();
        // TODO: Implement proper schema validation
        // validator.validate_plan requires ExecutionPlan, not ExprSchema
        Ok(())
    }
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

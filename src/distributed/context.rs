//! # Distributed Processing Context
//!
//! This module provides a high-level context for distributed processing,
//! enabling management of multiple datasets and direct SQL query execution.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use super::config::DistributedConfig;
use super::execution::{ExecutionEngine, ExecutionContext, ExecutionResult, ExecutionMetrics};
use super::dataframe::DistributedDataFrame;
use super::ToDistributed;
use super::expr::ExprSchema;
use super::schema_validator::SchemaValidator;

/// A context for managing distributed processing operations
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

impl DistributedContext {
    /// Creates a new distributed context
    pub fn new(config: DistributedConfig) -> Result<Self> {
        #[cfg(feature = "distributed")]
        {
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
            let context = engine.create_context(&config)?;
            
            Ok(Self {
                config,
                engine,
                context: Arc::new(Mutex::new(context)),
                datasets: HashMap::new(),
            })
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Registers a DataFrame with the context under the given name
    pub fn register_dataframe(&mut self, name: &str, df: &DataFrame) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
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
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Registers a CSV file with the context under the given name
    pub fn register_csv(&mut self, name: &str, path: &str) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            // Validate file path
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("CSV file not found: {}", path)));
            }
            
            // Register the CSV file with the execution context
            let mut context = self.context.lock().unwrap();
            context.register_csv(name, path)?;
            
            // Create a placeholder distributed DataFrame
            let dist_df = DistributedDataFrame::new(
                self.config.clone(),
                self.engine.clone(),
                context.as_ref().clone(),
                name.to_string(),
            );
            
            // Register the distributed DataFrame
            self.datasets.insert(name.to_string(), dist_df);
            
            Ok(())
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Registers a Parquet file with the context under the given name
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        #[cfg(feature = "distributed")]
        {
            // Validate file path
            if !Path::new(path).exists() {
                return Err(Error::IoError(format!("Parquet file not found: {}", path)));
            }
            
            // Register the Parquet file with the execution context
            let mut context = self.context.lock().unwrap();
            context.register_parquet(name, path)?;
            
            // Create a placeholder distributed DataFrame
            let dist_df = DistributedDataFrame::new(
                self.config.clone(),
                self.engine.clone(),
                context.as_ref().clone(),
                name.to_string(),
            );
            
            // Register the distributed DataFrame
            self.datasets.insert(name.to_string(), dist_df);
            
            Ok(())
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Executes a SQL query against registered datasets
    pub fn sql(&self, query: &str) -> Result<ExecutionResult> {
        #[cfg(feature = "distributed")]
        {
            // Execute the SQL query using the execution context
            let context = self.context.lock().unwrap();
            context.sql(query)
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Gets a registered dataset by name
    pub fn get_dataset(&self, name: &str) -> Option<&DistributedDataFrame> {
        self.datasets.get(name)
    }
    
    /// Lists all registered dataset names
    pub fn dataset_names(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }
    
    /// Executes a SQL query and returns the result as a local DataFrame
    pub fn sql_to_dataframe(&self, query: &str) -> Result<DataFrame> {
        #[cfg(feature = "distributed")]
        {
            // Execute the SQL query
            let result = self.sql(query)?;
            
            // Convert the result to a local DataFrame
            result.collect_to_local()
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Executes a SQL query and writes the result directly to a Parquet file
    pub fn sql_to_parquet(&self, query: &str, path: &str) -> Result<ExecutionMetrics> {
        #[cfg(feature = "distributed")]
        {
            // Execute the SQL query
            let result = self.sql(query)?;
            
            // Write the result to a Parquet file
            result.write_parquet(path)?;
            
            // Return the execution metrics
            Ok(result.metrics().clone())
        }
        
        #[cfg(not(feature = "distributed"))]
        {
            Err(Error::FeatureNotAvailable(
                "Distributed processing is not available. Recompile with the 'distributed' feature flag.".to_string()
            ))
        }
    }
    
    /// Returns the configuration used by this context
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }
}
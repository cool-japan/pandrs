//! # DataFusion Execution Engine
//!
//! This module provides an implementation of the ExecutionEngine interface using DataFusion.

use crate::error::Result;
use crate::distributed::config::DistributedConfig;
use crate::distributed::execution::{ExecutionEngine, ExecutionContext};

/// DataFusion execution engine implementation
pub struct DataFusionEngine {
    /// Whether the engine is initialized
    initialized: bool,
}

impl DataFusionEngine {
    /// Creates a new DataFusion engine
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
    }
}

impl Default for DataFusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionEngine for DataFusionEngine {
    fn initialize(&mut self, _config: &DistributedConfig) -> Result<()> {
        // Simple initialization for now
        self.initialized = true;
        Ok(())
    }
    
    fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    fn create_context(&self, config: &DistributedConfig) -> Result<Box<dyn ExecutionContext>> {
        if !self.initialized {
            return Err(crate::error::Error::InvalidOperation(
                "Engine is not initialized".to_string()
            ));
        }
        
        Ok(Box::new(super::DataFusionContext::new(config.clone())))
    }
}
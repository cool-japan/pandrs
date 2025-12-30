//! Machine Learning Pipeline Module
//!
//! Provides data transformation pipelines equivalent to scikit-learn.

use crate::optimized::OptimizedDataFrame;
use crate::error::Result;

/// Trait for data transformers
pub trait Transformer {
    /// Transform data
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame>;
    
    /// Learn from data and then transform it
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame>;
    
    /// Learn from data
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()>;
}

/// Pipeline for chaining data transformation steps
pub struct Pipeline {
    transformers: Vec<Box<dyn Transformer>>,
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new() -> Self {
        Pipeline {
            transformers: Vec::new(),
        }
    }
    
    /// Add a transformer to the pipeline
    pub fn add_transformer<T: Transformer + 'static>(&mut self, transformer: T) -> &mut Self {
        self.transformers.push(Box::new(transformer));
        self
    }
    
    /// Execute all pipeline steps and transform
    pub fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut result = df.clone();
        
        for transformer in &self.transformers {
            result = transformer.transform(&result)?;
        }
        
        Ok(result)
    }
    
    /// Learn from the pipeline and then transform
    pub fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut result = df.clone();
        
        for transformer in &mut self.transformers {
            result = transformer.fit_transform(&result)?;
        }
        
        Ok(result)
    }
    
    /// Learn from the pipeline
    pub fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        let mut temp_df = df.clone();
        
        for transformer in &mut self.transformers {
            transformer.fit(&temp_df)?;
            temp_df = transformer.transform(&temp_df)?;
        }
        
        Ok(())
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
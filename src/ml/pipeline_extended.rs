//! Extended ML Pipeline Features for Advanced Workflows

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::column::ColumnTrait;
use crate::error::{Error, Result};
use crate::ml::pipeline::{Pipeline, PipelineStage, PipelineTransformer};
use crate::ml::preprocessing::{MinMaxScaler, OneHotEncoder, StandardScaler};
use crate::optimized::OptimizedDataFrame;

/// Advanced pipeline stage that can handle complex transformations
pub trait AdvancedPipelineStage: Send + Sync {
    /// Apply transformation with context
    fn transform_with_context(
        &self,
        df: &OptimizedDataFrame,
        context: &PipelineContext,
    ) -> Result<OptimizedDataFrame>;

    /// Get stage metadata
    fn metadata(&self) -> StageMetadata;

    /// Validate stage configuration
    fn validate(&self, df: &OptimizedDataFrame) -> Result<()>;
}

/// Context for pipeline execution with shared state
#[derive(Clone)]
pub struct PipelineContext {
    /// Shared metadata between stages
    pub metadata: HashMap<String, serde_json::Value>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
    /// Stage execution history
    pub execution_history: Vec<StageExecution>,
}

/// Metadata for pipeline stages
#[derive(Debug, Clone)]
pub struct StageMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub input_requirements: Vec<String>,
    pub output_schema: Vec<ColumnSchema>,
}

/// Schema definition for pipeline outputs
#[derive(Debug, Clone)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub constraints: Vec<String>,
}

/// Record of stage execution
#[derive(Debug, Clone)]
pub struct StageExecution {
    pub stage_name: String,
    pub start_time: std::time::Instant,
    pub duration: std::time::Duration,
    pub input_rows: usize,
    pub output_rows: usize,
    pub memory_usage: usize,
}

/// Advanced pipeline with context and monitoring
pub struct AdvancedPipeline {
    stages: Vec<Box<dyn AdvancedPipelineStage>>,
    context: PipelineContext,
    monitoring_enabled: bool,
}

impl AdvancedPipeline {
    /// Create new advanced pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            context: PipelineContext {
                metadata: HashMap::new(),
                metrics: HashMap::new(),
                execution_history: Vec::new(),
            },
            monitoring_enabled: true,
        }
    }

    /// Add stage to pipeline
    pub fn add_stage(mut self, stage: Box<dyn AdvancedPipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Enable/disable monitoring
    pub fn with_monitoring(mut self, enabled: bool) -> Self {
        self.monitoring_enabled = enabled;
        self
    }

    /// Execute pipeline with full context
    pub fn execute(&mut self, df: OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        let mut current_df = df;

        // Validate all stages first
        for stage in &self.stages {
            stage.validate(&current_df)?;
        }

        // Execute stages
        for stage in &self.stages {
            let start_time = std::time::Instant::now();
            let input_rows = current_df.row_count();

            // Transform with context
            current_df = stage.transform_with_context(&current_df, &self.context)?;

            // Record execution if monitoring enabled
            if self.monitoring_enabled {
                let duration = start_time.elapsed();
                let output_rows = current_df.row_count();
                let memory_usage = 0; // current_df.memory_usage().values().sum();

                let execution = StageExecution {
                    stage_name: stage.metadata().name,
                    start_time,
                    duration,
                    input_rows,
                    output_rows,
                    memory_usage,
                };

                self.context.execution_history.push(execution);
                self.context.metrics.insert(
                    format!("{}_duration_ms", stage.metadata().name),
                    duration.as_millis() as f64,
                );
            }
        }

        Ok(current_df)
    }

    /// Get execution summary
    pub fn execution_summary(&self) -> PipelineExecutionSummary {
        let total_duration: std::time::Duration = self
            .context
            .execution_history
            .iter()
            .map(|ex| ex.duration)
            .sum();

        let total_memory: usize = self
            .context
            .execution_history
            .iter()
            .map(|ex| ex.memory_usage)
            .max()
            .unwrap_or(0);

        PipelineExecutionSummary {
            total_stages: self.stages.len(),
            total_duration,
            peak_memory_usage: total_memory,
            stage_details: self.context.execution_history.clone(),
        }
    }
}

/// Summary of pipeline execution
#[derive(Debug)]
pub struct PipelineExecutionSummary {
    pub total_stages: usize,
    pub total_duration: std::time::Duration,
    pub peak_memory_usage: usize,
    pub stage_details: Vec<StageExecution>,
}

/// Feature engineering stage for advanced transformations
pub struct FeatureEngineeringStage {
    operations: Vec<FeatureOperation>,
}

/// Feature engineering operations
#[derive(Clone)]
pub enum FeatureOperation {
    /// Create polynomial features
    PolynomialFeatures { columns: Vec<String>, degree: u32 },
    /// Create interaction features
    InteractionFeatures { column_pairs: Vec<(String, String)> },
    /// Binning/Discretization
    Binning {
        column: String,
        bins: u32,
        strategy: BinningStrategy,
    },
    /// Rolling window features
    RollingWindow {
        column: String,
        window_size: usize,
        operation: WindowOperation,
    },
    /// Custom transformation
    Custom {
        name: String,
        transform_fn: Arc<dyn Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync>,
    },
}

/// Binning strategies
#[derive(Clone, Debug)]
pub enum BinningStrategy {
    EqualWidth,
    EqualFrequency,
    Quantile(Vec<f64>),
}

/// Window operations for rolling features
#[derive(Clone, Debug)]
pub enum WindowOperation {
    Mean,
    Sum,
    Min,
    Max,
    Std,
    Count,
}

impl FeatureEngineeringStage {
    /// Create new feature engineering stage
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    /// Add polynomial features
    pub fn with_polynomial_features(mut self, columns: Vec<String>, degree: u32) -> Self {
        self.operations
            .push(FeatureOperation::PolynomialFeatures { columns, degree });
        self
    }

    /// Add interaction features
    pub fn with_interaction_features(mut self, column_pairs: Vec<(String, String)>) -> Self {
        self.operations
            .push(FeatureOperation::InteractionFeatures { column_pairs });
        self
    }

    /// Add binning operation
    pub fn with_binning(mut self, column: String, bins: u32, strategy: BinningStrategy) -> Self {
        self.operations.push(FeatureOperation::Binning {
            column,
            bins,
            strategy,
        });
        self
    }

    /// Add rolling window features
    pub fn with_rolling_window(
        mut self,
        column: String,
        window_size: usize,
        operation: WindowOperation,
    ) -> Self {
        self.operations.push(FeatureOperation::RollingWindow {
            column,
            window_size,
            operation,
        });
        self
    }

    /// Add custom transformation
    pub fn with_custom_transform<F>(mut self, name: String, transform_fn: F) -> Self
    where
        F: Fn(&OptimizedDataFrame) -> Result<OptimizedDataFrame> + Send + Sync + 'static,
    {
        self.operations.push(FeatureOperation::Custom {
            name,
            transform_fn: Arc::new(transform_fn),
        });
        self
    }
}

impl AdvancedPipelineStage for FeatureEngineeringStage {
    fn transform_with_context(
        &self,
        df: &OptimizedDataFrame,
        _context: &PipelineContext,
    ) -> Result<OptimizedDataFrame> {
        let mut result_df = df.clone();

        for operation in &self.operations {
            match operation {
                FeatureOperation::PolynomialFeatures { columns, degree } => {
                    result_df = self.create_polynomial_features(&result_df, columns, *degree)?;
                }
                FeatureOperation::InteractionFeatures { column_pairs } => {
                    result_df = self.create_interaction_features(&result_df, column_pairs)?;
                }
                FeatureOperation::Binning {
                    column,
                    bins,
                    strategy,
                } => {
                    result_df = self.create_binned_features(&result_df, column, *bins, strategy)?;
                }
                FeatureOperation::RollingWindow {
                    column,
                    window_size,
                    operation,
                } => {
                    result_df =
                        self.create_rolling_features(&result_df, column, *window_size, operation)?;
                }
                FeatureOperation::Custom {
                    name: _,
                    transform_fn,
                } => {
                    result_df = transform_fn(&result_df)?;
                }
            }
        }

        Ok(result_df)
    }

    fn metadata(&self) -> StageMetadata {
        StageMetadata {
            name: "FeatureEngineeringStage".to_string(),
            version: "1.0.0".to_string(),
            description: "Advanced feature engineering transformations".to_string(),
            input_requirements: vec!["numeric_columns".to_string()],
            output_schema: vec![], // Dynamic based on operations
        }
    }

    fn validate(&self, df: &OptimizedDataFrame) -> Result<()> {
        // Validate that required columns exist for each operation
        for operation in &self.operations {
            match operation {
                FeatureOperation::PolynomialFeatures { columns, .. } => {
                    for col in columns {
                        if !df.contains_column(col) {
                            return Err(Error::ColumnNotFound(col.clone()));
                        }
                    }
                }
                FeatureOperation::InteractionFeatures { column_pairs } => {
                    for (col1, col2) in column_pairs {
                        if !df.contains_column(col1) {
                            return Err(Error::ColumnNotFound(col1.clone()));
                        }
                        if !df.contains_column(col2) {
                            return Err(Error::ColumnNotFound(col2.clone()));
                        }
                    }
                }
                FeatureOperation::Binning { column, .. } => {
                    if !df.contains_column(column) {
                        return Err(Error::ColumnNotFound(column.clone()));
                    }
                }
                FeatureOperation::RollingWindow { column, .. } => {
                    if !df.contains_column(column) {
                        return Err(Error::ColumnNotFound(column.clone()));
                    }
                }
                FeatureOperation::Custom { .. } => {
                    // Custom validations would be implemented in the closure
                }
            }
        }
        Ok(())
    }
}

impl FeatureEngineeringStage {
    fn create_polynomial_features(
        &self,
        df: &OptimizedDataFrame,
        columns: &[String],
        degree: u32,
    ) -> Result<OptimizedDataFrame> {
        let mut result_df = df.clone();

        for column in columns {
            if let Ok(column_view) = df.column(column) {
                if let crate::column::Column::Float64(float_col) = column_view.column() {
                    for d in 2..=degree {
                        let new_col_name = format!("{}^{}", column, d);
                        let polynomial_values: Vec<f64> = (0..float_col.len())
                            .map(|i| {
                                if let Ok(Some(v)) = float_col.get(i) {
                                    v.powf(d as f64)
                                } else {
                                    0.0
                                }
                            })
                            .collect();

                        result_df.add_float_column(&new_col_name, polynomial_values)?;
                    }
                }
            }
        }

        Ok(result_df)
    }

    fn create_interaction_features(
        &self,
        df: &OptimizedDataFrame,
        column_pairs: &[(String, String)],
    ) -> Result<OptimizedDataFrame> {
        let mut result_df = df.clone();

        for (col1, col2) in column_pairs {
            let new_col_name = format!("{}_{}_interaction", col1, col2);

            if let (Ok(col_view1), Ok(col_view2)) = (df.column(col1), df.column(col2)) {
                if let (
                    crate::column::Column::Float64(float_col1),
                    crate::column::Column::Float64(float_col2),
                ) = (col_view1.column(), col_view2.column())
                {
                    let len = float_col1.len().min(float_col2.len());
                    let interaction_values: Vec<f64> = (0..len)
                        .map(|i| match (float_col1.get(i), float_col2.get(i)) {
                            (Ok(Some(v1)), Ok(Some(v2))) => v1 * v2,
                            _ => 0.0,
                        })
                        .collect();

                    result_df.add_float_column(&new_col_name, interaction_values)?;
                }
            }
        }

        Ok(result_df)
    }

    fn create_binned_features(
        &self,
        df: &OptimizedDataFrame,
        column: &str,
        bins: u32,
        strategy: &BinningStrategy,
    ) -> Result<OptimizedDataFrame> {
        let mut result_df = df.clone();

        if let Ok(column_view) = df.column(column) {
            if let crate::column::Column::Float64(float_col) = column_view.column() {
                let values: Vec<f64> = (0..float_col.len())
                    .filter_map(|i| float_col.get(i).ok().flatten())
                    .collect();

                let bin_edges = match strategy {
                    BinningStrategy::EqualWidth => {
                        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let step = (max_val - min_val) / bins as f64;
                        (0..=bins).map(|i| min_val + (i as f64) * step).collect()
                    }
                    BinningStrategy::EqualFrequency => {
                        let mut sorted_values = values.clone();
                        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let step = sorted_values.len() / bins as usize;
                        (0..=bins)
                            .map(|i| {
                                let idx = (i as usize * step).min(sorted_values.len() - 1);
                                sorted_values[idx]
                            })
                            .collect()
                    }
                    BinningStrategy::Quantile(quantiles) => quantiles.clone(),
                };

                let new_col_name = format!("{}_binned", column);
                let binned_values: Vec<i64> = (0..float_col.len())
                    .map(|i| {
                        if let Ok(Some(val)) = float_col.get(i) {
                            for (bin_idx, &edge) in bin_edges.iter().enumerate() {
                                if val <= edge {
                                    return bin_idx as i64;
                                }
                            }
                            (bin_edges.len() - 1) as i64
                        } else {
                            -1 // Missing value indicator
                        }
                    })
                    .collect();

                result_df.add_int_column(&new_col_name, binned_values)?;
            }
        }

        Ok(result_df)
    }

    fn create_rolling_features(
        &self,
        df: &OptimizedDataFrame,
        column: &str,
        window_size: usize,
        operation: &WindowOperation,
    ) -> Result<OptimizedDataFrame> {
        let mut result_df = df.clone();

        // Validate window size
        if window_size == 0 {
            return Err(Error::InvalidValue(
                "Window size must be greater than 0".to_string(),
            ));
        }

        if let Ok(column_view) = df.column(column) {
            if let crate::column::Column::Float64(float_col) = column_view.column() {
                let new_col_name = format!(
                    "{}_rolling_{}_{}",
                    column,
                    window_size,
                    format!("{:?}", operation).to_lowercase()
                );
                let mut rolling_values = Vec::with_capacity(float_col.len());

                for i in 0..float_col.len() {
                    let start_idx = if window_size > 0 && i + 1 >= window_size {
                        i + 1 - window_size
                    } else {
                        0
                    };
                    let window_vals: Vec<f64> = (start_idx..=i)
                        .filter_map(|idx| float_col.get(idx).ok().flatten())
                        .collect();

                    let result = if window_vals.is_empty() {
                        0.0
                    } else {
                        match operation {
                            WindowOperation::Mean => {
                                window_vals.iter().sum::<f64>() / window_vals.len() as f64
                            }
                            WindowOperation::Sum => window_vals.iter().sum(),
                            WindowOperation::Min => {
                                window_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b))
                            }
                            WindowOperation::Max => {
                                window_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                            }
                            WindowOperation::Std => {
                                let mean =
                                    window_vals.iter().sum::<f64>() / window_vals.len() as f64;
                                let variance =
                                    window_vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                                        / window_vals.len() as f64;
                                variance.sqrt()
                            }
                            WindowOperation::Count => window_vals.len() as f64,
                        }
                    };

                    rolling_values.push(result);
                }

                result_df.add_float_column(&new_col_name, rolling_values)?;
            }
        }

        Ok(result_df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_pipeline() -> Result<()> {
        let mut df = OptimizedDataFrame::new();
        df.add_float_column("x", vec![1.0, 2.0, 3.0, 4.0, 5.0])?;
        df.add_float_column("y", vec![2.0, 4.0, 6.0, 8.0, 10.0])?;

        let feature_stage = FeatureEngineeringStage::new()
            .with_polynomial_features(vec!["x".to_string()], 2)
            .with_interaction_features(vec![("x".to_string(), "y".to_string())]);

        let mut pipeline = AdvancedPipeline::new().add_stage(Box::new(feature_stage));

        let result = pipeline.execute(df)?;

        assert!(result.contains_column("x^2"));
        assert!(result.contains_column("x_y_interaction"));

        let summary = pipeline.execution_summary();
        assert_eq!(summary.total_stages, 1);

        Ok(())
    }

    #[test]
    fn test_feature_engineering_operations() -> Result<()> {
        let mut df = OptimizedDataFrame::new();
        df.add_float_column("value", vec![1.0, 2.0, 3.0, 4.0, 5.0])?;

        let stage = FeatureEngineeringStage::new()
            .with_binning("value".to_string(), 3, BinningStrategy::EqualWidth)
            .with_rolling_window("value".to_string(), 3, WindowOperation::Mean);

        let context = PipelineContext {
            metadata: HashMap::new(),
            metrics: HashMap::new(),
            execution_history: Vec::new(),
        };

        let result = stage.transform_with_context(&df, &context)?;

        assert!(result.contains_column("value_binned"));
        assert!(result.contains_column("value_rolling_3_mean"));

        Ok(())
    }
}

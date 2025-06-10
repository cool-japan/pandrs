//! Machine Learning Module
//!
//! This module provides machine learning functionality for data analysis.
//! It includes preprocessing, model training/evaluation, dimensionality reduction,
//! clustering, and anomaly detection algorithms.

// Feature modules
pub mod anomaly;
pub mod clustering;
pub mod dimension;
pub mod metrics;
pub mod models;
pub mod pipeline;
pub mod pipeline_extended;
pub mod preprocessing;

// GPU-accelerated ML functionality (conditionally compiled)
#[cfg(feature = "cuda")]
pub mod gpu;

// Backward compatibility layer (for legacy code)
pub mod backward_compat;

// Re-export public types and functions
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use std::collections::HashMap;

// Re-export preprocessing tools
pub use preprocessing::{
    Binner, FeatureSelector, ImputeStrategy, Imputer, MinMaxScaler, OneHotEncoder,
    PolynomialFeatures, StandardScaler,
};

// Re-export metrics
pub use metrics::regression::{
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
    root_mean_squared_error,
};

pub use metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};

// Re-export dimensionality reduction
pub use dimension::{TSNEInit, PCA, TSNE};

// Re-export clustering
pub use clustering::{AgglomerativeClustering, DistanceMetric, KMeans, Linkage, DBSCAN};

// Re-export anomaly detection
pub use anomaly::{IsolationForest, LocalOutlierFactor, OneClassSVM};

// Re-export pipeline
pub use pipeline::{Pipeline, PipelineStage, PipelineTransformer};

pub use pipeline_extended::{
    AdvancedPipeline, AdvancedPipelineStage, BinningStrategy, ColumnSchema,
    FeatureEngineeringStage, FeatureOperation, PipelineContext, PipelineExecutionSummary,
    StageExecution, StageMetadata, WindowOperation,
};

// Re-export model functionality
pub use models::{
    train_test_split, CrossValidation, ModelEvaluator, ModelMetrics, SupervisedModel,
    UnsupervisedModel,
};

// For backward compatibility, re-export old module structures
#[allow(deprecated)]
pub use backward_compat::anomaly_detection as anomaly_detection_compat;
#[allow(deprecated)]
pub use backward_compat::models as models_compat;
#[allow(deprecated)]
pub use backward_compat::pipeline as pipeline_legacy;

// Add pipeline compatibility
pub mod pipeline_compat;

// Export compatibility layer (not marked as deprecated since it's an active adapter)
pub use pipeline_compat::{Pipeline as PipelineCompat, Transformer};

// Re-export GPU-accelerated ML functionality when CUDA is enabled
#[cfg(feature = "cuda")]
pub use gpu::{GpuKMeans, GpuModelParams, GpuPCA, GpuTSNE};

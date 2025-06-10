//! Machine Learning Module
//!
//! This module provides machine learning functionality for data analysis.
//! It includes preprocessing, model training/evaluation, dimensionality reduction,
//! clustering, and anomaly detection algorithms.

// Feature modules
pub mod preprocessing;
pub mod metrics;
pub mod models;
pub mod clustering;
pub mod dimension;
pub mod anomaly;
pub mod pipeline;
pub mod pipeline_extended;

// GPU-accelerated ML functionality (conditionally compiled)
#[cfg(feature = "cuda")]
pub mod gpu;

// Backward compatibility layer (for legacy code)
pub mod backward_compat;

// Re-export public types and functions
use crate::dataframe::DataFrame;
use crate::error::{Result, Error};
use crate::optimized::OptimizedDataFrame;
use std::collections::HashMap;

// Re-export preprocessing tools
pub use preprocessing::{
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    Binner,
    Imputer,
    ImputeStrategy,
    FeatureSelector,
};

// Re-export metrics
pub use metrics::regression::{
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    explained_variance_score,
};

pub use metrics::classification::{
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
};

// Re-export dimensionality reduction
pub use dimension::{
    PCA,
    TSNE,
    TSNEInit,
};

// Re-export clustering
pub use clustering::{
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    Linkage,
    DistanceMetric,
};

// Re-export anomaly detection
pub use anomaly::{
    IsolationForest,
    LocalOutlierFactor,
    OneClassSVM,
};

// Re-export pipeline
pub use pipeline::{
    Pipeline,
    PipelineStage,
    PipelineTransformer,
};

pub use pipeline_extended::{
    AdvancedPipeline,
    AdvancedPipelineStage,
    PipelineContext,
    FeatureEngineeringStage,
    FeatureOperation,
    BinningStrategy,
    WindowOperation,
    PipelineExecutionSummary,
    StageMetadata,
    ColumnSchema,
    StageExecution,
};

// Re-export model functionality
pub use models::{
    ModelMetrics,
    ModelEvaluator,
    SupervisedModel,
    UnsupervisedModel,
    CrossValidation,
    train_test_split,
};

// For backward compatibility, re-export old module structures
#[allow(deprecated)]
pub use backward_compat::models as models_compat;
#[allow(deprecated)]
pub use backward_compat::anomaly_detection as anomaly_detection_compat;
#[allow(deprecated)]
pub use backward_compat::pipeline as pipeline_legacy;

// Add pipeline compatibility
pub mod pipeline_compat;

// Export compatibility layer (not marked as deprecated since it's an active adapter)
pub use pipeline_compat::{Transformer, Pipeline as PipelineCompat};

// Re-export GPU-accelerated ML functionality when CUDA is enabled
#[cfg(feature = "cuda")]
pub use gpu::{
    GpuModelParams,
    GpuPCA,
    GpuKMeans,
    GpuTSNE,
};
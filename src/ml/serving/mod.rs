//! Model Serving and Deployment Module
//!
//! This module provides comprehensive model serving and deployment capabilities including
//! model serialization, REST API serving, model registry, versioning, and deployment
//! configuration management.

pub mod bridge;
pub mod deployment;
pub mod endpoints;
pub mod monitoring;
pub mod registry;
pub mod serialization;
pub mod server;

use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Model metadata for serving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type (e.g., "linear_regression", "random_forest", "automl")
    pub model_type: String,
    /// Feature names expected by the model
    pub feature_names: Vec<String>,
    /// Target column name (for supervised models)
    pub target_name: Option<String>,
    /// Model description
    pub description: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// Model performance metrics
    pub metrics: HashMap<String, f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRequest {
    /// Input data for prediction
    pub data: HashMap<String, serde_json::Value>,
    /// Optional model version (defaults to latest)
    pub model_version: Option<String>,
    /// Optional prediction options
    pub options: Option<PredictionOptions>,
}

/// Prediction options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOptions {
    /// Return prediction probabilities (for classification)
    pub include_probabilities: Option<bool>,
    /// Return feature importance scores
    pub include_feature_importance: Option<bool>,
    /// Return confidence intervals
    pub include_confidence_intervals: Option<bool>,
    /// Custom prediction threshold (for binary classification)
    pub threshold: Option<f64>,
}

/// Prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResponse {
    /// Prediction result
    pub prediction: serde_json::Value,
    /// Prediction probabilities (if requested)
    pub probabilities: Option<HashMap<String, f64>>,
    /// Feature importance scores (if requested)
    pub feature_importance: Option<HashMap<String, f64>>,
    /// Confidence intervals (if requested)
    pub confidence_intervals: Option<ConfidenceInterval>,
    /// Model metadata used for prediction
    pub model_metadata: ModelMetadata,
    /// Prediction timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

/// Batch prediction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionRequest {
    /// Batch input data
    pub data: Vec<HashMap<String, serde_json::Value>>,
    /// Optional model version (defaults to latest)
    pub model_version: Option<String>,
    /// Optional prediction options
    pub options: Option<PredictionOptions>,
}

/// Batch prediction response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionResponse {
    /// Batch prediction results
    pub predictions: Vec<PredictionResponse>,
    /// Batch processing summary
    pub summary: BatchProcessingSummary,
}

/// Batch processing summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingSummary {
    /// Total number of predictions
    pub total_predictions: usize,
    /// Number of successful predictions
    pub successful_predictions: usize,
    /// Number of failed predictions
    pub failed_predictions: usize,
    /// Total processing time in milliseconds
    pub total_processing_time_ms: u64,
    /// Average processing time per prediction in milliseconds
    pub avg_processing_time_ms: f64,
    /// `(batch_index, error_message)` for every item that failed, preserving which input row
    /// each failure came from (rather than only a bare failure count).
    #[serde(default)]
    pub failed_items: Vec<(usize, String)>,
}

/// Model deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Deployment environment (e.g., "development", "staging", "production")
    pub environment: String,
    /// Resource allocation
    pub resources: ResourceConfig,
    /// Scaling configuration
    pub scaling: ScalingConfig,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// CPU cores
    pub cpu_cores: f64,
    /// Memory in MB
    pub memory_mb: u64,
    /// GPU allocation (if available)
    pub gpu_memory_mb: Option<u64>,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
}

/// Scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    /// Minimum number of instances
    pub min_instances: usize,
    /// Maximum number of instances
    pub max_instances: usize,
    /// Target CPU utilization for auto-scaling
    pub target_cpu_utilization: f64,
    /// Target memory utilization for auto-scaling
    pub target_memory_utilization: f64,
    /// Scale up threshold
    pub scale_up_threshold: f64,
    /// Scale down threshold
    pub scale_down_threshold: f64,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check endpoint path
    pub path: String,
    /// Health check interval in seconds
    pub interval_seconds: u64,
    /// Health check timeout in seconds
    pub timeout_seconds: u64,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: usize,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: usize,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable request logging
    pub enable_logging: bool,
    /// Enable distributed tracing
    pub enable_tracing: bool,
    /// Metrics export interval in seconds
    pub metrics_interval_seconds: u64,
    /// Log level
    pub log_level: String,
}

/// Model serving interface trait
///
/// Requires `Send + Sync` because servable models are shared via `Arc<dyn ModelServing>`
/// across registry lookups and (in a real deployment) concurrent request handlers; a
/// trait object that couldn't cross threads would make this module unusable for backing
/// an actual concurrent server.
pub trait ModelServing: Send + Sync {
    /// Make a single prediction
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionResponse>;

    /// Make batch predictions
    fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse>;

    /// Get model metadata
    fn get_metadata(&self) -> &ModelMetadata;

    /// Health check
    fn health_check(&self) -> Result<HealthStatus>;

    /// Get model information
    fn info(&self) -> ModelInfo;

    /// Convert this model back into its persistable [`serialization::SerializableModel`]
    /// representation, preserving whatever weights/parameters it holds.
    ///
    /// The default implementation honestly reports that a given `ModelServing`
    /// implementation cannot be persisted, rather than silently producing an empty
    /// (weight-losing) placeholder. [`serialization::GenericServingModel`] overrides this
    /// with a real implementation; wrappers such as [`deployment::DeployedModel`] delegate
    /// to their inner model.
    fn to_serializable(&self) -> Result<serialization::SerializableModel> {
        Err(Error::NotImplemented(format!(
            "Model type backing '{}' does not implement to_serializable(); it cannot be \
             persisted by a ModelRegistry",
            self.get_metadata().model_type
        )))
    }
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub status: String,
    /// Detailed status information
    pub details: HashMap<String, String>,
    /// Last health check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model statistics
    pub statistics: ModelStatistics,
    /// Model configuration
    pub configuration: HashMap<String, serde_json::Value>,
}

/// Model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// Total number of predictions made
    pub total_predictions: u64,
    /// Average prediction time in milliseconds
    pub avg_prediction_time_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Throughput (predictions per second)
    pub throughput_per_second: f64,
    /// Last prediction timestamp
    pub last_prediction_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Model serving factory
pub struct ModelServingFactory;

impl ModelServingFactory {
    /// Create a new model serving instance from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Box<dyn ModelServing>> {
        let model_path = path.as_ref();

        // Check file extension to determine serialization format
        match model_path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => {
                let serializer = serialization::JsonModelSerializer;
                serializer.load(model_path)
            }
            Some("yaml") | Some("yml") => {
                let serializer = serialization::YamlModelSerializer;
                serializer.load(model_path)
            }
            Some("toml") => {
                let serializer = serialization::TomlModelSerializer;
                serializer.load(model_path)
            }
            Some("bin") | Some("pandrs") => {
                let serializer = serialization::BinaryModelSerializer;
                serializer.load(model_path)
            }
            _ => Err(Error::InvalidInput(format!(
                "Unsupported model file format: {:?}",
                model_path.extension()
            ))),
        }
    }

    /// Create a new model serving instance from a model registry
    pub fn from_registry(
        registry: &dyn registry::ModelRegistry,
        model_name: &str,
        version: Option<&str>,
    ) -> Result<Arc<dyn ModelServing>> {
        let model_version = version.unwrap_or("latest");
        registry.load_model(model_name, model_version)
    }

    /// Create a new model serving instance with deployment configuration
    pub fn with_deployment_config(
        model: Box<dyn ModelServing>,
        config: DeploymentConfig,
    ) -> Result<deployment::DeployedModel> {
        deployment::DeployedModel::new(model, config)
    }
}

/// Model serving server
pub struct ModelServer {
    /// Registered models, held as `Arc` so `get_model` can hand out owned, cheaply-cloneable
    /// handles (needed for the registry-backed fallback path, which loads models on demand
    /// rather than requiring every model to be pre-registered).
    models: HashMap<String, Arc<dyn ModelServing>>,
    /// Server configuration
    config: ServerConfig,
    /// Model registry, consulted by `get_model` when a name isn't already registered locally.
    registry: Option<Arc<dyn registry::ModelRegistry>>,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum request size in bytes
    pub max_request_size: usize,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
    /// Enable CORS
    pub enable_cors: bool,
    /// Enable authentication
    pub enable_auth: bool,
    /// API key (if authentication is enabled)
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_request_size: 10 * 1024 * 1024, // 10MB
            request_timeout_seconds: 30,
            enable_cors: true,
            enable_auth: false,
            api_key: None,
        }
    }
}

impl ModelServer {
    /// Create a new model server
    pub fn new(config: ServerConfig) -> Self {
        Self {
            models: HashMap::new(),
            config,
            registry: None,
        }
    }

    /// Register a model with the server
    pub fn register_model(&mut self, name: String, model: Box<dyn ModelServing>) -> Result<()> {
        if self.models.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "Model '{}' is already registered",
                name
            )));
        }

        self.models.insert(name, Arc::from(model));
        Ok(())
    }

    /// Unregister a model from the server
    pub fn unregister_model(&mut self, name: &str) -> Result<()> {
        if self.models.remove(name).is_none() {
            return Err(Error::KeyNotFound(format!(
                "Model '{}' is not registered",
                name
            )));
        }

        Ok(())
    }

    /// Set model registry, consulted by `get_model` as a fallback for names that aren't
    /// directly registered on this server.
    pub fn set_registry(&mut self, registry: Arc<dyn registry::ModelRegistry>) {
        self.registry = Some(registry);
    }

    /// Get list of registered models (locally registered only; does not enumerate the
    /// backing registry, which may hold many more models than are currently loaded).
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Get model by name.
    ///
    /// Looks up locally-registered models first; if not found and a [`registry::ModelRegistry`]
    /// has been configured via [`Self::set_registry`], falls back to loading the model's
    /// `"latest"` version from the registry. The registry-loaded instance is returned directly
    /// (not cached into the local map): the registry itself is expected to do any caching a
    /// given backend needs, so this stays honest about not silently duplicating that policy.
    pub fn get_model(&self, name: &str) -> Result<Arc<dyn ModelServing>> {
        if let Some(model) = self.models.get(name) {
            return Ok(Arc::clone(model));
        }

        if let Some(registry) = &self.registry {
            return registry.load_model(name, "latest");
        }

        Err(Error::KeyNotFound(format!("Model '{}' not found", name)))
    }

    /// Start the server (placeholder - actual implementation would use a web framework)
    pub fn start(&self) -> Result<()> {
        log::info!(
            "Starting model server on {}:{}",
            self.config.host,
            self.config.port
        );

        // In a real implementation, this would start an HTTP server
        // using a framework like warp, axum, or actix-web
        Err(Error::NotImplemented(
            "HTTP server implementation requires additional dependencies".to_string(),
        ))
    }
}

// Re-export public types
pub use deployment::{DeployedModel, DeploymentManager, DeploymentMetrics, DeploymentStatus};
pub use endpoints::{
    ApiResponse, BatchPredictionEndpoint, HealthEndpoint, ModelInfoEndpoint, PredictionEndpoint,
};
pub use monitoring::{
    AlertConfig, AlertSeverity, ComparisonOperator, MetricsCollector, ModelMonitor,
    PerformanceMetrics,
};
pub use registry::{
    FileSystemModelRegistry, InMemoryModelRegistry, ModelRegistry, ModelRegistryEntry,
};
pub use serialization::{
    GenericServingModel, ModelSerializer, SerializableModel, SerializationFormat,
};
pub use server::{HttpModelServer, HttpResponse, RequestContext, ServerStats};

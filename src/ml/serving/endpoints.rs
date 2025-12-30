//! Model Serving API Endpoints Module
//!
//! This module provides REST API endpoint definitions and handlers for model serving,
//! prediction, monitoring, and management operations.

use crate::core::error::{Error, Result};
use crate::ml::serving::monitoring::{AlertEvent, MetricsSummary, PerformanceMetrics};
use crate::ml::serving::registry::{ModelRegistry, ModelRegistryEntry};
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, DeploymentMetrics, HealthStatus, ModelInfo,
    ModelMetadata, ModelServer, ModelServing, PredictionRequest, PredictionResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "serving")]
use uuid::Uuid;

/// API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Response data
    pub data: Option<T>,
    /// Success status
    pub success: bool,
    /// Error message (if any)
    pub error: Option<String>,
    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request ID for tracing
    pub request_id: Option<String>,
}

impl<T> ApiResponse<T> {
    /// Create a success response
    pub fn success(data: T) -> Self {
        Self {
            data: Some(data),
            success: true,
            error: None,
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }

    /// Create a success response with request ID
    pub fn success_with_id(data: T, request_id: String) -> Self {
        Self {
            data: Some(data),
            success: true,
            error: None,
            timestamp: chrono::Utc::now(),
            request_id: Some(request_id),
        }
    }

    /// Create an error response
    pub fn error(error_message: String) -> Self {
        Self {
            data: None,
            success: false,
            error: Some(error_message),
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }

    /// Create an error response with request ID
    pub fn error_with_id(error_message: String, request_id: String) -> Self {
        Self {
            data: None,
            success: false,
            error: Some(error_message),
            timestamp: chrono::Utc::now(),
            request_id: Some(request_id),
        }
    }
}

/// Prediction endpoint handler
pub struct PredictionEndpoint;

impl PredictionEndpoint {
    /// Handle single prediction request
    pub fn predict(
        server: &ModelServer,
        model_name: &str,
        request: PredictionRequest,
        request_id: Option<String>,
    ) -> ApiResponse<PredictionResponse> {
        match server.get_model(model_name) {
            Ok(model) => match model.predict(&request) {
                Ok(response) => {
                    if let Some(id) = request_id {
                        ApiResponse::success_with_id(response, id)
                    } else {
                        ApiResponse::success(response)
                    }
                }
                Err(e) => {
                    let error_msg = format!("Prediction failed: {}", e);
                    if let Some(id) = request_id {
                        ApiResponse::error_with_id(error_msg, id)
                    } else {
                        ApiResponse::error(error_msg)
                    }
                }
            },
            Err(e) => {
                let error_msg = format!("Model not found: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// Validate prediction request
    pub fn validate_request(request: &PredictionRequest, model: &dyn ModelServing) -> Result<()> {
        let metadata = model.get_metadata();

        // Check if all required features are present
        for feature_name in &metadata.feature_names {
            if !request.data.contains_key(feature_name) {
                return Err(Error::InvalidInput(format!(
                    "Missing required feature: {}",
                    feature_name
                )));
            }
        }

        // Validate feature types (basic validation)
        for (feature_name, value) in &request.data {
            if !metadata.feature_names.contains(feature_name) {
                return Err(Error::InvalidInput(format!(
                    "Unknown feature: {}",
                    feature_name
                )));
            }

            // Check if value can be converted to number (basic check)
            match value {
                serde_json::Value::Number(_) => {}
                serde_json::Value::String(s) => {
                    if s.parse::<f64>().is_err() {
                        return Err(Error::InvalidInput(format!(
                            "Feature '{}' must be numeric",
                            feature_name
                        )));
                    }
                }
                _ => {
                    return Err(Error::InvalidInput(format!(
                        "Feature '{}' has invalid type",
                        feature_name
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Batch prediction endpoint handler
pub struct BatchPredictionEndpoint;

impl BatchPredictionEndpoint {
    /// Handle batch prediction request
    pub fn predict_batch(
        server: &ModelServer,
        model_name: &str,
        request: BatchPredictionRequest,
        request_id: Option<String>,
    ) -> ApiResponse<BatchPredictionResponse> {
        // Validate batch size
        if request.data.is_empty() {
            let error_msg = "Batch request cannot be empty".to_string();
            return if let Some(id) = request_id {
                ApiResponse::error_with_id(error_msg, id)
            } else {
                ApiResponse::error(error_msg)
            };
        }

        if request.data.len() > 1000 {
            let error_msg = "Batch size too large (max 1000)".to_string();
            return if let Some(id) = request_id {
                ApiResponse::error_with_id(error_msg, id)
            } else {
                ApiResponse::error(error_msg)
            };
        }

        match server.get_model(model_name) {
            Ok(model) => {
                // Validate each request in the batch
                for (i, data) in request.data.iter().enumerate() {
                    let individual_request = PredictionRequest {
                        data: data.clone(),
                        model_version: request.model_version.clone(),
                        options: request.options.clone(),
                    };

                    if let Err(e) = PredictionEndpoint::validate_request(&individual_request, model)
                    {
                        let error_msg = format!("Validation failed for item {}: {}", i, e);
                        return if let Some(id) = request_id {
                            ApiResponse::error_with_id(error_msg, id)
                        } else {
                            ApiResponse::error(error_msg)
                        };
                    }
                }

                match model.predict_batch(&request) {
                    Ok(response) => {
                        if let Some(id) = request_id {
                            ApiResponse::success_with_id(response, id)
                        } else {
                            ApiResponse::success(response)
                        }
                    }
                    Err(e) => {
                        let error_msg = format!("Batch prediction failed: {}", e);
                        if let Some(id) = request_id {
                            ApiResponse::error_with_id(error_msg, id)
                        } else {
                            ApiResponse::error(error_msg)
                        }
                    }
                }
            }
            Err(e) => {
                let error_msg = format!("Model not found: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }
}

/// Model information endpoint handler
pub struct ModelInfoEndpoint;

impl ModelInfoEndpoint {
    /// Get model information
    pub fn get_model_info(
        server: &ModelServer,
        model_name: &str,
        request_id: Option<String>,
    ) -> ApiResponse<ModelInfo> {
        match server.get_model(model_name) {
            Ok(model) => {
                let info = model.info();
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(info, id)
                } else {
                    ApiResponse::success(info)
                }
            }
            Err(e) => {
                let error_msg = format!("Model not found: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// Get model metadata
    pub fn get_model_metadata(
        server: &ModelServer,
        model_name: &str,
        request_id: Option<String>,
    ) -> ApiResponse<ModelMetadata> {
        match server.get_model(model_name) {
            Ok(model) => {
                let metadata = model.get_metadata().clone();
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(metadata, id)
                } else {
                    ApiResponse::success(metadata)
                }
            }
            Err(e) => {
                let error_msg = format!("Model not found: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// List all models
    pub fn list_models(
        server: &ModelServer,
        request_id: Option<String>,
    ) -> ApiResponse<Vec<String>> {
        let models = server.list_models();
        if let Some(id) = request_id {
            ApiResponse::success_with_id(models, id)
        } else {
            ApiResponse::success(models)
        }
    }
}

/// Health check endpoint handler
pub struct HealthEndpoint;

impl HealthEndpoint {
    /// Health check for specific model
    pub fn health_check_model(
        server: &ModelServer,
        model_name: &str,
        request_id: Option<String>,
    ) -> ApiResponse<HealthStatus> {
        match server.get_model(model_name) {
            Ok(model) => match model.health_check() {
                Ok(status) => {
                    if let Some(id) = request_id {
                        ApiResponse::success_with_id(status, id)
                    } else {
                        ApiResponse::success(status)
                    }
                }
                Err(e) => {
                    let error_msg = format!("Health check failed: {}", e);
                    if let Some(id) = request_id {
                        ApiResponse::error_with_id(error_msg, id)
                    } else {
                        ApiResponse::error(error_msg)
                    }
                }
            },
            Err(e) => {
                let error_msg = format!("Model not found: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// Overall server health check
    pub fn health_check_server(
        server: &ModelServer,
        request_id: Option<String>,
    ) -> ApiResponse<ServerHealthStatus> {
        let models = server.list_models();
        let mut model_statuses = HashMap::new();
        let mut healthy_count = 0;
        let total_count = models.len();

        for model_name in &models {
            match server.get_model(model_name) {
                Ok(model) => match model.health_check() {
                    Ok(status) => {
                        if status.status == "healthy" {
                            healthy_count += 1;
                        }
                        model_statuses.insert(model_name.clone(), status);
                    }
                    Err(e) => {
                        model_statuses.insert(
                            model_name.clone(),
                            HealthStatus {
                                status: "error".to_string(),
                                details: {
                                    let mut details = HashMap::new();
                                    details.insert("error".to_string(), e.to_string());
                                    details
                                },
                                timestamp: chrono::Utc::now(),
                            },
                        );
                    }
                },
                Err(e) => {
                    model_statuses.insert(
                        model_name.clone(),
                        HealthStatus {
                            status: "not_found".to_string(),
                            details: {
                                let mut details = HashMap::new();
                                details.insert("error".to_string(), e.to_string());
                                details
                            },
                            timestamp: chrono::Utc::now(),
                        },
                    );
                }
            }
        }

        let overall_status = if total_count == 0 {
            "no_models".to_string()
        } else if healthy_count == total_count {
            "healthy".to_string()
        } else if healthy_count > 0 {
            "degraded".to_string()
        } else {
            "unhealthy".to_string()
        };

        let server_health = ServerHealthStatus {
            status: overall_status,
            total_models: total_count,
            healthy_models: healthy_count,
            model_statuses,
            timestamp: chrono::Utc::now(),
        };

        if let Some(id) = request_id {
            ApiResponse::success_with_id(server_health, id)
        } else {
            ApiResponse::success(server_health)
        }
    }
}

/// Server health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHealthStatus {
    /// Overall server status
    pub status: String,
    /// Total number of models
    pub total_models: usize,
    /// Number of healthy models
    pub healthy_models: usize,
    /// Individual model health statuses
    pub model_statuses: HashMap<String, HealthStatus>,
    /// Check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Model registry endpoint handler
pub struct RegistryEndpoint;

impl RegistryEndpoint {
    /// List models in registry
    pub fn list_models(
        registry: &dyn ModelRegistry,
        request_id: Option<String>,
    ) -> ApiResponse<Vec<ModelRegistryEntry>> {
        match registry.list_models() {
            Ok(models) => {
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(models, id)
                } else {
                    ApiResponse::success(models)
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to list models: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// List model versions
    pub fn list_versions(
        registry: &dyn ModelRegistry,
        model_name: &str,
        request_id: Option<String>,
    ) -> ApiResponse<Vec<String>> {
        match registry.list_versions(model_name) {
            Ok(versions) => {
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(versions, id)
                } else {
                    ApiResponse::success(versions)
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to list versions: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }

    /// Get model metadata from registry
    pub fn get_metadata(
        registry: &dyn ModelRegistry,
        model_name: &str,
        version: &str,
        request_id: Option<String>,
    ) -> ApiResponse<ModelMetadata> {
        match registry.get_metadata(model_name, version) {
            Ok(metadata) => {
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(metadata, id)
                } else {
                    ApiResponse::success(metadata)
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to get metadata: {}", e);
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }
}

/// Monitoring endpoint handler
pub struct MonitoringEndpoint;

impl MonitoringEndpoint {
    /// Get model metrics
    pub fn get_metrics(
        metrics: &[PerformanceMetrics],
        limit: Option<usize>,
        request_id: Option<String>,
    ) -> ApiResponse<Vec<PerformanceMetrics>> {
        let limit = limit.unwrap_or(100);
        let limited_metrics = metrics.iter().rev().take(limit).cloned().collect();

        if let Some(id) = request_id {
            ApiResponse::success_with_id(limited_metrics, id)
        } else {
            ApiResponse::success(limited_metrics)
        }
    }

    /// Get alert events
    pub fn get_alerts(
        alerts: &[AlertEvent],
        limit: Option<usize>,
        request_id: Option<String>,
    ) -> ApiResponse<Vec<AlertEvent>> {
        let limit = limit.unwrap_or(50);
        let limited_alerts = alerts.iter().rev().take(limit).cloned().collect();

        if let Some(id) = request_id {
            ApiResponse::success_with_id(limited_alerts, id)
        } else {
            ApiResponse::success(limited_alerts)
        }
    }

    /// Get metrics summary
    pub fn get_summary(
        summary: Option<MetricsSummary>,
        request_id: Option<String>,
    ) -> ApiResponse<MetricsSummary> {
        match summary {
            Some(summary) => {
                if let Some(id) = request_id {
                    ApiResponse::success_with_id(summary, id)
                } else {
                    ApiResponse::success(summary)
                }
            }
            None => {
                let error_msg = "No metrics summary available".to_string();
                if let Some(id) = request_id {
                    ApiResponse::error_with_id(error_msg, id)
                } else {
                    ApiResponse::error(error_msg)
                }
            }
        }
    }
}

/// Request validation utilities
pub struct RequestValidator;

impl RequestValidator {
    /// Generate request ID
    #[cfg(feature = "serving")]
    pub fn generate_request_id() -> String {
        Uuid::new_v4().to_string()
    }

    /// Generate request ID (fallback when serving feature is disabled)
    #[cfg(not(feature = "serving"))]
    pub fn generate_request_id() -> String {
        format!(
            "req_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        )
    }

    /// Validate API key (if authentication is enabled)
    pub fn validate_api_key(provided_key: Option<&str>, expected_key: Option<&str>) -> bool {
        match (provided_key, expected_key) {
            (None, None) => true, // No authentication required
            (Some(provided), Some(expected)) => provided == expected,
            _ => false, // Authentication required but not provided, or provided but not expected
        }
    }

    /// Validate request size
    pub fn validate_request_size(size: usize, max_size: usize) -> bool {
        size <= max_size
    }

    /// Sanitize model name
    pub fn sanitize_model_name(name: &str) -> String {
        name.chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect()
    }
}

/// API endpoint routes (placeholder for actual HTTP routing)
pub struct ApiRoutes;

impl ApiRoutes {
    /// Get all available routes
    pub fn get_routes() -> Vec<RouteInfo> {
        vec![
            // Prediction routes
            RouteInfo {
                method: "POST".to_string(),
                path: "/models/{model_name}/predict".to_string(),
                description: "Make a single prediction".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: true,
            },
            RouteInfo {
                method: "POST".to_string(),
                path: "/models/{model_name}/predict/batch".to_string(),
                description: "Make batch predictions".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: true,
            },
            // Model information routes
            RouteInfo {
                method: "GET".to_string(),
                path: "/models".to_string(),
                description: "List all models".to_string(),
                parameters: vec![],
                body_required: false,
            },
            RouteInfo {
                method: "GET".to_string(),
                path: "/models/{model_name}".to_string(),
                description: "Get model information".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: false,
            },
            RouteInfo {
                method: "GET".to_string(),
                path: "/models/{model_name}/metadata".to_string(),
                description: "Get model metadata".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: false,
            },
            // Health check routes
            RouteInfo {
                method: "GET".to_string(),
                path: "/health".to_string(),
                description: "Server health check".to_string(),
                parameters: vec![],
                body_required: false,
            },
            RouteInfo {
                method: "GET".to_string(),
                path: "/models/{model_name}/health".to_string(),
                description: "Model health check".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: false,
            },
            // Registry routes
            RouteInfo {
                method: "GET".to_string(),
                path: "/registry/models".to_string(),
                description: "List models in registry".to_string(),
                parameters: vec![],
                body_required: false,
            },
            RouteInfo {
                method: "GET".to_string(),
                path: "/registry/models/{model_name}/versions".to_string(),
                description: "List model versions".to_string(),
                parameters: vec!["model_name".to_string()],
                body_required: false,
            },
            // Monitoring routes
            RouteInfo {
                method: "GET".to_string(),
                path: "/models/{model_name}/metrics".to_string(),
                description: "Get model metrics".to_string(),
                parameters: vec!["model_name".to_string(), "limit".to_string()],
                body_required: false,
            },
            RouteInfo {
                method: "GET".to_string(),
                path: "/models/{model_name}/alerts".to_string(),
                description: "Get model alerts".to_string(),
                parameters: vec!["model_name".to_string(), "limit".to_string()],
                body_required: false,
            },
        ]
    }
}

/// Route information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteInfo {
    /// HTTP method
    pub method: String,
    /// URL path
    pub path: String,
    /// Route description
    pub description: String,
    /// Path and query parameters
    pub parameters: Vec<String>,
    /// Whether request body is required
    pub body_required: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_response() {
        let success_response = ApiResponse::success("test data");
        assert!(success_response.success);
        assert_eq!(success_response.data, Some("test data"));
        assert!(success_response.error.is_none());

        let error_response = ApiResponse::<String>::error("test error".to_string());
        assert!(!error_response.success);
        assert!(error_response.data.is_none());
        assert_eq!(error_response.error, Some("test error".to_string()));
    }

    #[test]
    fn test_request_validator() {
        let request_id = RequestValidator::generate_request_id();
        assert!(!request_id.is_empty());

        assert!(RequestValidator::validate_api_key(None, None));
        assert!(RequestValidator::validate_api_key(Some("key"), Some("key")));
        assert!(!RequestValidator::validate_api_key(
            Some("key1"),
            Some("key2")
        ));
        assert!(!RequestValidator::validate_api_key(None, Some("key")));

        assert!(RequestValidator::validate_request_size(100, 200));
        assert!(!RequestValidator::validate_request_size(300, 200));

        let sanitized = RequestValidator::sanitize_model_name("test-model_123!@#");
        assert_eq!(sanitized, "test-model_123");
    }

    #[test]
    fn test_server_health_status() {
        let mut model_statuses = HashMap::new();
        model_statuses.insert(
            "model1".to_string(),
            HealthStatus {
                status: "healthy".to_string(),
                details: HashMap::new(),
                timestamp: chrono::Utc::now(),
            },
        );

        let health_status = ServerHealthStatus {
            status: "healthy".to_string(),
            total_models: 1,
            healthy_models: 1,
            model_statuses,
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(health_status.status, "healthy");
        assert_eq!(health_status.total_models, 1);
        assert_eq!(health_status.healthy_models, 1);
    }

    #[test]
    fn test_route_info() {
        let routes = ApiRoutes::get_routes();
        assert!(!routes.is_empty());

        let predict_route = routes
            .iter()
            .find(|route| route.path.contains("predict") && !route.path.contains("batch"))
            .unwrap();

        assert_eq!(predict_route.method, "POST");
        assert!(predict_route.body_required);
    }
}

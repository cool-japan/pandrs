//! Model Serving API Endpoints Module
//!
//! This module provides REST API endpoint definitions and handlers for model serving,
//! prediction, monitoring, and management operations.

use crate::core::error::{Error, Result};
use crate::ml::serving::monitoring::{AlertEvent, MetricsSummary, PerformanceMetrics};
use crate::ml::serving::registry::{ModelRegistry, ModelRegistryEntry};
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, BatchProcessingSummary, HealthStatus,
    ModelInfo, ModelMetadata, ModelServer, ModelServing, PredictionRequest, PredictionResponse,
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
    /// Suggested HTTP status code for this error, when classifiable (`None` on success).
    /// Populated via [`Error`] variant (e.g. `KeyNotFound` -> 404, `InvalidInput` -> 400,
    /// `NotImplemented` -> 501), so a transport layer (see [`HttpResponse::from_status`](crate::ml::serving::server::HttpResponse::from_status))
    /// can return the right status instead of collapsing every failure to 500.
    #[serde(default)]
    pub error_code: Option<u16>,
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
            error_code: None,
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
            error_code: None,
            timestamp: chrono::Utc::now(),
            request_id: Some(request_id),
        }
    }

    /// Create an error response (no classified status code; treated as 500 by transports).
    pub fn error(error_message: String) -> Self {
        Self {
            data: None,
            success: false,
            error: Some(error_message),
            error_code: None,
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }

    /// Create an error response with request ID (no classified status code).
    pub fn error_with_id(error_message: String, request_id: String) -> Self {
        Self {
            data: None,
            success: false,
            error: Some(error_message),
            error_code: None,
            timestamp: chrono::Utc::now(),
            request_id: Some(request_id),
        }
    }

    /// Create an error response classified with a specific suggested HTTP status code.
    pub fn error_with_status(error_message: String, status_code: u16, request_id: String) -> Self {
        Self {
            data: None,
            success: false,
            error: Some(error_message),
            error_code: Some(status_code),
            timestamp: chrono::Utc::now(),
            request_id: Some(request_id),
        }
    }
}

/// Classify an [`Error`] into a suggested HTTP status code.
///
/// Previously every endpoint failure (missing model, invalid input, an actual internal error)
/// collapsed into a single generic error message with no way for a transport layer to tell them
/// apart, so every response ended up mapped to 500 regardless of cause.
pub fn classify_error(error: &Error) -> u16 {
    match error {
        Error::KeyNotFound(_) => 404,
        Error::InvalidInput(_) | Error::DimensionMismatch(_) => 400,
        Error::NotImplemented(_) => 501,
        _ => 500,
    }
}

/// Build a classified error `ApiResponse`, with or without a request ID.
fn error_response<T>(message: String, status: u16, request_id: Option<String>) -> ApiResponse<T> {
    match request_id {
        Some(id) => ApiResponse::error_with_status(message, status, id),
        None => {
            let mut response = ApiResponse::error(message);
            response.error_code = Some(status);
            response
        }
    }
}

/// If the request pins a specific `model_version` (anything other than the `"latest"`/
/// `"default"` sentinels), reject when it doesn't match the resolved model's actual version,
/// instead of silently serving whatever version happened to be resolved.
fn check_model_version(model: &dyn ModelServing, requested: &Option<String>) -> Result<()> {
    if let Some(requested_version) = requested {
        if requested_version != "latest" && requested_version != "default" {
            let actual_version = &model.get_metadata().version;
            if requested_version != actual_version {
                return Err(Error::InvalidInput(format!(
                    "Requested model_version '{}' does not match the resolved model's version \
                     '{}'",
                    requested_version, actual_version
                )));
            }
        }
    }
    Ok(())
}

/// Prediction endpoint handler
pub struct PredictionEndpoint;

impl PredictionEndpoint {
    /// Handle single prediction request.
    ///
    /// Runs [`Self::validate_request`] before calling into the model (previously only the batch
    /// path validated; a single `predict` call skipped straight to the model, relying on
    /// whatever ad-hoc checks that model's own inference happened to perform), and honors a
    /// pinned `model_version` rather than silently ignoring it.
    pub fn predict(
        server: &ModelServer,
        model_name: &str,
        request: PredictionRequest,
        request_id: Option<String>,
    ) -> ApiResponse<PredictionResponse> {
        let model = match server.get_model(model_name) {
            Ok(model) => model,
            Err(e) => {
                let status = classify_error(&e);
                let error_msg = format!("Model not found: {}", e);
                return error_response(error_msg, status, request_id);
            }
        };

        if let Err(e) = check_model_version(model.as_ref(), &request.model_version) {
            let status = classify_error(&e);
            let error_msg = format!("Model version mismatch: {}", e);
            return error_response(error_msg, status, request_id);
        }

        if let Err(e) = Self::validate_request(&request, model.as_ref()) {
            let status = classify_error(&e);
            let error_msg = format!("Validation failed: {}", e);
            return error_response(error_msg, status, request_id);
        }

        match model.predict(&request) {
            Ok(response) => match request_id {
                Some(id) => ApiResponse::success_with_id(response, id),
                None => ApiResponse::success(response),
            },
            Err(e) => {
                let status = classify_error(&e);
                let error_msg = format!("Prediction failed: {}", e);
                error_response(error_msg, status, request_id)
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
    /// Handle batch prediction request.
    ///
    /// A row that fails [`PredictionEndpoint::validate_request`] is recorded as a real per-item
    /// failure at its original index -- exactly like a row that fails during actual model
    /// inference -- rather than rejecting the *entire* batch with a single generic error.
    /// Previously, one malformed row anywhere in the batch (e.g. a typo'd feature name on row
    /// 999 of 1000) discarded every other row's successful prediction, defeating the point of
    /// `BatchProcessingSummary::failed_items` existing at all: real per-row indexing is only
    /// useful if a bad row can't take the rest of the batch down with it.
    pub fn predict_batch(
        server: &ModelServer,
        model_name: &str,
        request: BatchPredictionRequest,
        request_id: Option<String>,
    ) -> ApiResponse<BatchPredictionResponse> {
        // Validate batch size
        if request.data.is_empty() {
            return error_response("Batch request cannot be empty".to_string(), 400, request_id);
        }

        if request.data.len() > 1000 {
            return error_response(
                "Batch size too large (max 1000)".to_string(),
                400,
                request_id,
            );
        }

        let model = match server.get_model(model_name) {
            Ok(model) => model,
            Err(e) => {
                let status = classify_error(&e);
                let error_msg = format!("Model not found: {}", e);
                return error_response(error_msg, status, request_id);
            }
        };

        if let Err(e) = check_model_version(model.as_ref(), &request.model_version) {
            let status = classify_error(&e);
            let error_msg = format!("Model version mismatch: {}", e);
            return error_response(error_msg, status, request_id);
        }

        // Partition the batch into rows that pass validation (sent to the model, keeping their
        // original index) and rows that don't (recorded as real per-item failures directly,
        // original index preserved).
        let mut valid_original_indices: Vec<usize> = Vec::with_capacity(request.data.len());
        let mut valid_rows: Vec<HashMap<String, serde_json::Value>> =
            Vec::with_capacity(request.data.len());
        let mut pre_validation_failures: Vec<(usize, String)> = Vec::new();

        for (i, data) in request.data.iter().enumerate() {
            let individual_request = PredictionRequest {
                data: data.clone(),
                model_version: request.model_version.clone(),
                options: request.options.clone(),
            };

            match PredictionEndpoint::validate_request(&individual_request, model.as_ref()) {
                Ok(()) => {
                    valid_original_indices.push(i);
                    valid_rows.push(data.clone());
                }
                Err(e) => pre_validation_failures.push((i, format!("Validation failed: {}", e))),
            }
        }

        // Every row was structurally invalid: there is nothing left for the model to predict,
        // so this really is a whole-batch failure rather than a set of per-item ones.
        if valid_rows.is_empty() {
            let error_msg = format!(
                "All {} item(s) in batch failed validation; first: {}",
                pre_validation_failures.len(),
                pre_validation_failures
                    .first()
                    .map(|(_, msg)| msg.as_str())
                    .unwrap_or("unknown")
            );
            return error_response(error_msg, 400, request_id);
        }

        let filtered_request = BatchPredictionRequest {
            data: valid_rows,
            model_version: request.model_version.clone(),
            options: request.options.clone(),
        };

        match model.predict_batch(&filtered_request) {
            Ok(filtered_response) => {
                // Remap the model's filtered-batch-space failure indices back to their original
                // position in the caller's request, then merge with the pre-validation failures
                // (both already at original indices) and sort for a deterministic, readable
                // ordering. `predictions` needs no remapping: it's already in original relative
                // order, since `valid_rows` is a strict order-preserving subsequence of
                // `request.data` and the model itself preserves row order.
                let mut failed_items = pre_validation_failures;
                failed_items.extend(filtered_response.summary.failed_items.iter().map(
                    |(filtered_idx, message)| {
                        let original_idx = valid_original_indices
                            .get(*filtered_idx)
                            .copied()
                            .unwrap_or(*filtered_idx);
                        (original_idx, message.clone())
                    },
                ));
                failed_items.sort_by_key(|(idx, _)| *idx);

                let response = BatchPredictionResponse {
                    predictions: filtered_response.predictions,
                    summary: BatchProcessingSummary {
                        total_predictions: request.data.len(),
                        successful_predictions: filtered_response.summary.successful_predictions,
                        failed_predictions: failed_items.len(),
                        total_processing_time_ms: filtered_response
                            .summary
                            .total_processing_time_ms,
                        avg_processing_time_ms: filtered_response.summary.avg_processing_time_ms,
                        failed_items,
                    },
                };

                match request_id {
                    Some(id) => ApiResponse::success_with_id(response, id),
                    None => ApiResponse::success(response),
                }
            }
            Err(e) => {
                let status = classify_error(&e);
                let error_msg = format!("Batch prediction failed: {}", e);
                error_response(error_msg, status, request_id)
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

    /// Generate request ID (fallback when the `serving` feature -- and therefore `uuid` -- is
    /// unavailable).
    ///
    /// Combines a millisecond timestamp (for human-readable, roughly chronological ordering)
    /// with a process-lifetime atomic counter, so IDs are always unique within this process
    /// even when multiple requests land in the same millisecond -- a real possibility under any
    /// meaningful load, and one the previous timestamp-only scheme did not protect against.
    #[cfg(not(feature = "serving"))]
    pub fn generate_request_id() -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        format!("req_{}_{}", millis, sequence)
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
            .expect("operation should succeed");

        assert_eq!(predict_route.method, "POST");
        assert!(predict_route.body_required);
    }

    #[test]
    fn test_generate_request_id_never_collides_in_a_tight_loop() {
        // Regression for the pre-fix fallback (millisecond timestamp only), which could produce
        // duplicate IDs for requests landing in the same millisecond under any real load.
        let mut ids = std::collections::HashSet::new();
        for _ in 0..500 {
            assert!(
                ids.insert(RequestValidator::generate_request_id()),
                "generate_request_id produced a duplicate"
            );
        }
    }

    #[test]
    fn test_classify_error_maps_variants_to_expected_status_codes() {
        assert_eq!(classify_error(&Error::KeyNotFound("x".into())), 404);
        assert_eq!(classify_error(&Error::InvalidInput("x".into())), 400);
        assert_eq!(classify_error(&Error::DimensionMismatch("x".into())), 400);
        assert_eq!(classify_error(&Error::NotImplemented("x".into())), 501);
        assert_eq!(classify_error(&Error::Computation("x".into())), 500);
    }

    #[test]
    fn test_error_with_status_carries_the_code() {
        let response: ApiResponse<()> =
            ApiResponse::error_with_status("not found".to_string(), 404, "req-1".to_string());
        assert!(!response.success);
        assert_eq!(response.error_code, Some(404));
        assert_eq!(response.request_id, Some("req-1".to_string()));
    }

    #[test]
    fn test_check_model_version_rejects_mismatch_but_allows_sentinels() {
        use crate::ml::serving::serialization::{GenericServingModel, SerializableModel};
        use crate::ml::serving::ModelMetadata;

        let metadata = ModelMetadata {
            name: "m".to_string(),
            version: "1.2.0".to_string(),
            model_type: "linear_regression".to_string(),
            feature_names: vec!["x".to_string()],
            target_name: None,
            description: String::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        };
        let mut parameters = HashMap::new();
        parameters.insert("coefficients".to_string(), serde_json::json!([1.0]));
        let serializable = SerializableModel {
            schema_version: crate::ml::serving::serialization::CURRENT_SCHEMA_VERSION,
            metadata,
            parameters,
            model_data: serde_json::json!({}),
            preprocessing: None,
            config: HashMap::new(),
        };
        let model = GenericServingModel::from_serializable(serializable).expect("model builds");

        assert!(check_model_version(&model, &None).is_ok());
        assert!(check_model_version(&model, &Some("latest".to_string())).is_ok());
        assert!(check_model_version(&model, &Some("1.2.0".to_string())).is_ok());
        assert!(check_model_version(&model, &Some("9.9.9".to_string())).is_err());
    }
}

//! HTTP Server Module for Model Serving
//!
//! This module provides HTTP server functionality for serving machine learning models
//! via REST API endpoints. Note: This is a framework-agnostic implementation that
//! provides the core server structure and handlers.

use crate::core::error::{Error, Result};
use crate::ml::serving::endpoints::{
    ApiResponse, ApiRoutes, BatchPredictionEndpoint, HealthEndpoint, ModelInfoEndpoint,
    MonitoringEndpoint, PredictionEndpoint, RegistryEndpoint, RequestValidator, RouteInfo,
    ServerHealthStatus,
};
use crate::ml::serving::monitoring::{
    AlertEvent, MetricsSummary, ModelMonitor, PerformanceMetrics,
};
use crate::ml::serving::registry::ModelRegistry;
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, ModelServer, ModelServing, PredictionRequest,
    PredictionResponse, ServerConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

/// HTTP request context
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request ID for tracing
    pub request_id: String,
    /// Client IP address
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// API key (if provided)
    pub api_key: Option<String>,
}

impl RequestContext {
    /// Create a new request context
    pub fn new() -> Self {
        Self {
            request_id: RequestValidator::generate_request_id(),
            client_ip: None,
            user_agent: None,
            timestamp: chrono::Utc::now(),
            api_key: None,
        }
    }

    /// Create with request ID
    pub fn with_id(request_id: String) -> Self {
        Self {
            request_id,
            client_ip: None,
            user_agent: None,
            timestamp: chrono::Utc::now(),
            api_key: None,
        }
    }
}

impl Default for RequestContext {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP response
#[derive(Debug, Clone, Serialize)]
pub struct HttpResponse<T> {
    /// HTTP status code
    pub status_code: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body
    pub body: ApiResponse<T>,
}

impl<T> HttpResponse<T> {
    /// Create a success response
    pub fn ok(data: T, request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code: 200,
            headers,
            body: ApiResponse::success_with_id(data, request_id),
        }
    }

    /// Create a bad request response
    pub fn bad_request(error_message: String, request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code: 400,
            headers,
            body: ApiResponse::error_with_id(error_message, request_id),
        }
    }

    /// Create a not found response
    pub fn not_found(error_message: String, request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code: 404,
            headers,
            body: ApiResponse::error_with_id(error_message, request_id),
        }
    }

    /// Create an internal server error response
    pub fn internal_server_error(error_message: String, request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code: 500,
            headers,
            body: ApiResponse::error_with_id(error_message, request_id),
        }
    }

    /// Create an unauthorized response
    pub fn unauthorized(request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code: 401,
            headers,
            body: ApiResponse::error_with_id("Unauthorized".to_string(), request_id),
        }
    }

    /// Create a too many requests response
    pub fn too_many_requests(request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());
        headers.insert("Retry-After".to_string(), "60".to_string());

        Self {
            status_code: 429,
            headers,
            body: ApiResponse::error_with_id("Too many requests".to_string(), request_id),
        }
    }
}

/// Request rate limiter
pub struct RateLimiter {
    /// Request counts per client
    request_counts: Arc<RwLock<HashMap<String, RequestCounter>>>,
    /// Maximum requests per minute
    max_requests_per_minute: usize,
    /// Time window for rate limiting
    window_minutes: usize,
}

/// Request counter for rate limiting
#[derive(Debug, Clone)]
struct RequestCounter {
    /// Request timestamps
    requests: Vec<Instant>,
    /// Last cleanup time
    last_cleanup: Instant,
}

impl RequestCounter {
    fn new() -> Self {
        Self {
            requests: Vec::new(),
            last_cleanup: Instant::now(),
        }
    }

    /// Add a request and check if limit is exceeded
    fn add_request(&mut self, window_minutes: usize, max_requests: usize) -> bool {
        let now = Instant::now();
        self.requests.push(now);

        // Cleanup old requests every minute
        if now.duration_since(self.last_cleanup).as_secs() > 60 {
            self.cleanup_old_requests(window_minutes);
            self.last_cleanup = now;
        }

        self.requests.len() <= max_requests
    }

    /// Remove requests older than the time window
    fn cleanup_old_requests(&mut self, window_minutes: usize) {
        let cutoff = Instant::now() - std::time::Duration::from_secs(window_minutes as u64 * 60);
        self.requests.retain(|&time| time > cutoff);
    }
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_requests_per_minute: usize, window_minutes: usize) -> Self {
        Self {
            request_counts: Arc::new(RwLock::new(HashMap::new())),
            max_requests_per_minute,
            window_minutes,
        }
    }

    /// Check if request is allowed
    pub fn check_rate_limit(&self, client_id: &str) -> bool {
        let mut counts = self.request_counts.write().unwrap();
        let counter = counts
            .entry(client_id.to_string())
            .or_insert_with(RequestCounter::new);

        counter.add_request(self.window_minutes, self.max_requests_per_minute)
    }

    /// Get current request count for client
    pub fn get_request_count(&self, client_id: &str) -> usize {
        let counts = self.request_counts.read().unwrap();
        counts
            .get(client_id)
            .map(|counter| counter.requests.len())
            .unwrap_or(0)
    }
}

/// Enhanced model server with HTTP capabilities
pub struct HttpModelServer {
    /// Core model server
    model_server: ModelServer,
    /// Server configuration
    config: ServerConfig,
    /// Model registry (optional)
    registry: Option<Arc<dyn ModelRegistry>>,
    /// Model monitors
    monitors: Arc<Mutex<HashMap<String, ModelMonitor>>>,
    /// Rate limiter
    rate_limiter: Option<RateLimiter>,
    /// Request statistics
    request_stats: Arc<Mutex<RequestStatistics>>,
}

/// Request statistics
#[derive(Debug, Clone, Default)]
struct RequestStatistics {
    /// Total requests
    total_requests: u64,
    /// Successful requests
    successful_requests: u64,
    /// Failed requests
    failed_requests: u64,
    /// Requests by endpoint
    requests_by_endpoint: HashMap<String, u64>,
    /// Average response time
    avg_response_time_ms: f64,
}

impl HttpModelServer {
    /// Create a new HTTP model server
    pub fn new(config: ServerConfig) -> Self {
        let model_server = ModelServer::new(config.clone());

        // Create rate limiter if needed
        let rate_limiter = if config.enable_auth {
            Some(RateLimiter::new(1000, 1)) // 1000 requests per minute
        } else {
            None
        };

        Self {
            model_server,
            config,
            registry: None,
            monitors: Arc::new(Mutex::new(HashMap::new())),
            rate_limiter,
            request_stats: Arc::new(Mutex::new(RequestStatistics::default())),
        }
    }

    /// Set model registry
    pub fn set_registry(&mut self, registry: Arc<dyn ModelRegistry>) {
        self.registry = Some(registry);
    }

    /// Register a model
    pub fn register_model(&mut self, name: String, model: Box<dyn ModelServing>) -> Result<()> {
        // Register with model server
        self.model_server.register_model(name.clone(), model)?;

        // Create monitor for the model
        let metadata = self.model_server.get_model(&name)?.get_metadata().clone();
        let monitor = ModelMonitor::new(metadata);

        self.monitors.lock().unwrap().insert(name, monitor);

        Ok(())
    }

    /// Middleware for authentication
    fn authenticate(&self, context: &RequestContext) -> bool {
        if !self.config.enable_auth {
            return true;
        }

        RequestValidator::validate_api_key(
            context.api_key.as_deref(),
            self.config.api_key.as_deref(),
        )
    }

    /// Middleware for rate limiting
    fn check_rate_limit(&self, context: &RequestContext) -> bool {
        if let Some(rate_limiter) = &self.rate_limiter {
            let client_id = context.client_ip.as_deref().unwrap_or("unknown");
            rate_limiter.check_rate_limit(client_id)
        } else {
            true
        }
    }

    /// Record request statistics
    fn record_request(&self, endpoint: &str, success: bool, response_time_ms: u64) {
        let mut stats = self.request_stats.lock().unwrap();
        stats.total_requests += 1;

        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }

        *stats
            .requests_by_endpoint
            .entry(endpoint.to_string())
            .or_insert(0) += 1;

        // Update average response time (simple moving average)
        stats.avg_response_time_ms = (stats.avg_response_time_ms
            * (stats.total_requests - 1) as f64
            + response_time_ms as f64)
            / stats.total_requests as f64;
    }

    /// Handle prediction request
    pub fn handle_predict(
        &self,
        model_name: &str,
        request: PredictionRequest,
        context: RequestContext,
    ) -> HttpResponse<PredictionResponse> {
        let start_time = Instant::now();
        let endpoint = "predict";

        // Authentication
        if !self.authenticate(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::unauthorized(context.request_id);
        }

        // Rate limiting
        if !self.check_rate_limit(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::too_many_requests(context.request_id);
        }

        // Validate request size (simplified)
        let request_size = serde_json::to_string(&request)
            .map(|s| s.len())
            .unwrap_or(0);

        if !RequestValidator::validate_request_size(request_size, self.config.max_request_size) {
            let error_msg = format!("Request too large: {} bytes", request_size);
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::bad_request(error_msg, context.request_id);
        }

        // Handle prediction
        let response = PredictionEndpoint::predict(
            &self.model_server,
            model_name,
            request,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        self.record_request(endpoint, response.success, response_time);

        if response.success {
            HttpResponse::ok(response.data.unwrap(), context.request_id)
        } else {
            HttpResponse::internal_server_error(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
                context.request_id,
            )
        }
    }

    /// Handle batch prediction request
    pub fn handle_predict_batch(
        &self,
        model_name: &str,
        request: BatchPredictionRequest,
        context: RequestContext,
    ) -> HttpResponse<BatchPredictionResponse> {
        let start_time = Instant::now();
        let endpoint = "predict_batch";

        // Authentication
        if !self.authenticate(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::unauthorized(context.request_id);
        }

        // Rate limiting
        if !self.check_rate_limit(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::too_many_requests(context.request_id);
        }

        // Handle batch prediction
        let response = BatchPredictionEndpoint::predict_batch(
            &self.model_server,
            model_name,
            request,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        self.record_request(endpoint, response.success, response_time);

        if response.success {
            HttpResponse::ok(response.data.unwrap(), context.request_id)
        } else {
            HttpResponse::internal_server_error(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
                context.request_id,
            )
        }
    }

    /// Handle model info request
    pub fn handle_model_info(
        &self,
        model_name: &str,
        context: RequestContext,
    ) -> HttpResponse<crate::ml::serving::ModelInfo> {
        let start_time = Instant::now();
        let endpoint = "model_info";

        // Authentication
        if !self.authenticate(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::unauthorized(context.request_id);
        }

        let response = ModelInfoEndpoint::get_model_info(
            &self.model_server,
            model_name,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        self.record_request(endpoint, response.success, response_time);

        if response.success {
            HttpResponse::ok(response.data.unwrap(), context.request_id)
        } else {
            HttpResponse::not_found(
                response
                    .error
                    .unwrap_or_else(|| "Model not found".to_string()),
                context.request_id,
            )
        }
    }

    /// Handle health check request
    pub fn handle_health_check(
        &self,
        model_name: Option<&str>,
        context: RequestContext,
    ) -> HttpResponse<ServerHealthStatus> {
        let start_time = Instant::now();
        let endpoint = "health_check";

        let response = if let Some(model_name) = model_name {
            // Model-specific health check
            match HealthEndpoint::health_check_model(
                &self.model_server,
                model_name,
                Some(context.request_id.clone()),
            ) {
                resp if resp.success => {
                    // Convert single model health to server health format
                    let health_status = resp.data.unwrap();
                    let mut model_statuses = HashMap::new();
                    model_statuses.insert(model_name.to_string(), health_status.clone());

                    let status = health_status.status.clone();
                    let server_health = ServerHealthStatus {
                        status: status.clone(),
                        total_models: 1,
                        healthy_models: if status == "healthy" { 1 } else { 0 },
                        model_statuses,
                        timestamp: chrono::Utc::now(),
                    };

                    ApiResponse::success_with_id(server_health, context.request_id.clone())
                }
                resp => ApiResponse::error_with_id(
                    resp.error
                        .unwrap_or_else(|| "Health check failed".to_string()),
                    context.request_id.clone(),
                ),
            }
        } else {
            // Server health check
            HealthEndpoint::health_check_server(
                &self.model_server,
                Some(context.request_id.clone()),
            )
        };

        let response_time = start_time.elapsed().as_millis() as u64;
        self.record_request(endpoint, response.success, response_time);

        if response.success {
            HttpResponse::ok(response.data.unwrap(), context.request_id)
        } else {
            HttpResponse::internal_server_error(
                response
                    .error
                    .unwrap_or_else(|| "Health check failed".to_string()),
                context.request_id,
            )
        }
    }

    /// Handle list models request
    pub fn handle_list_models(&self, context: RequestContext) -> HttpResponse<Vec<String>> {
        let start_time = Instant::now();
        let endpoint = "list_models";

        // Authentication
        if !self.authenticate(&context) {
            self.record_request(endpoint, false, start_time.elapsed().as_millis() as u64);
            return HttpResponse::unauthorized(context.request_id);
        }

        let response =
            ModelInfoEndpoint::list_models(&self.model_server, Some(context.request_id.clone()));

        let response_time = start_time.elapsed().as_millis() as u64;
        self.record_request(endpoint, response.success, response_time);

        HttpResponse::ok(response.data.unwrap(), context.request_id)
    }

    /// Get server statistics
    pub fn get_server_stats(&self) -> ServerStats {
        let stats = self.request_stats.lock().unwrap();

        ServerStats {
            total_requests: stats.total_requests,
            successful_requests: stats.successful_requests,
            failed_requests: stats.failed_requests,
            success_rate: if stats.total_requests > 0 {
                stats.successful_requests as f64 / stats.total_requests as f64
            } else {
                0.0
            },
            avg_response_time_ms: stats.avg_response_time_ms,
            requests_by_endpoint: stats.requests_by_endpoint.clone(),
            uptime_seconds: 0, // Would track actual uptime in real implementation
            active_models: self.model_server.list_models().len(),
        }
    }

    /// Get API routes documentation
    pub fn get_routes(&self) -> Vec<RouteInfo> {
        ApiRoutes::get_routes()
    }
}

/// Server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    /// Total requests served
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Requests by endpoint
    pub requests_by_endpoint: HashMap<String, u64>,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
    /// Number of active models
    pub active_models: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::serving::serialization::GenericServingModel;
    use crate::ml::serving::ModelMetadata;

    fn create_test_config() -> ServerConfig {
        ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_request_size: 1024 * 1024, // 1MB
            request_timeout_seconds: 30,
            enable_cors: true,
            enable_auth: false,
            api_key: None,
        }
    }

    #[test]
    fn test_request_context() {
        let context = RequestContext::new();
        assert!(!context.request_id.is_empty());

        let context_with_id = RequestContext::with_id("test-id".to_string());
        assert_eq!(context_with_id.request_id, "test-id");
    }

    #[test]
    fn test_http_response() {
        let response = HttpResponse::ok("test data", "request-123".to_string());
        assert_eq!(response.status_code, 200);
        assert!(response.body.success);

        let error_response = HttpResponse::<String>::bad_request(
            "Invalid input".to_string(),
            "request-456".to_string(),
        );
        assert_eq!(error_response.status_code, 400);
        assert!(!error_response.body.success);
    }

    #[test]
    fn test_rate_limiter() {
        let rate_limiter = RateLimiter::new(5, 1); // 5 requests per minute

        // Should allow first 5 requests
        for _ in 0..5 {
            assert!(rate_limiter.check_rate_limit("client1"));
        }

        // Should deny 6th request
        assert!(!rate_limiter.check_rate_limit("client1"));

        // Different client should be allowed
        assert!(rate_limiter.check_rate_limit("client2"));
    }

    #[test]
    fn test_http_model_server() {
        let config = create_test_config();
        let server = HttpModelServer::new(config);

        let stats = server.get_server_stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.active_models, 0);

        let routes = server.get_routes();
        assert!(!routes.is_empty());
    }

    #[test]
    fn test_request_statistics() {
        let stats = RequestStatistics::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_requests, 0);
        assert_eq!(stats.failed_requests, 0);
    }
}

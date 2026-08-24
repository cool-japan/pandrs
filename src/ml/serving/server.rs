//! HTTP Server Module for Model Serving
//!
//! This module provides request/response *handler* logic for serving machine learning models
//! over an HTTP-shaped API: request validation, rate limiting, authentication checks, statistics,
//! and mapping [`crate::ml::serving::endpoints`] results to status-coded responses.
//!
//! # This does not open a network socket
//!
//! Despite the name, [`HttpModelServer`] never binds a port or runs an HTTP server -- it has no
//! I/O of its own. It is a pure request/response transformer: you call `handle_predict`,
//! `handle_health_check`, etc. with an already-parsed request and get back an [`HttpResponse`]
//! (status code + headers + body) to serialize and send yourself. Wiring an actual transport
//! (axum, warp, actix-web, a Lambda handler, ...) that calls into these methods is the
//! integrator's job; [`crate::ml::serving::ModelServer::start`] says so explicitly via
//! `Error::NotImplemented` for anyone who tries to "start" a server that isn't there.

use crate::core::error::Result;
use crate::ml::serving::endpoints::{
    ApiResponse, ApiRoutes, BatchPredictionEndpoint, HealthEndpoint, ModelInfoEndpoint,
    PredictionEndpoint, RequestValidator, RouteInfo, ServerHealthStatus,
};
use crate::ml::serving::monitoring::ModelMonitor;
use crate::ml::serving::registry::ModelRegistry;
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, ModelServer, ModelServing, PredictionRequest,
    PredictionResponse, ServerConfig,
};
use crate::{lock_safe, read_lock_safe, write_lock_safe};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Maximum distinct clients tracked by a [`RateLimiter`] at once, bounding memory against an
/// unbounded (e.g. spoofed) set of client identifiers. When a new client would exceed this cap,
/// the least-recently-active tracked client is evicted to make room.
const MAX_TRACKED_CLIENTS: usize = 10_000;

/// Default per-client request budget applied by [`HttpModelServer`], independent of
/// `ServerConfig::enable_auth` (see [`HttpModelServer::new`]).
const DEFAULT_RATE_LIMIT_PER_MINUTE: usize = 600;

/// The winning class's probability (the maximum value in a classification response's
/// `probabilities` map), used as this prediction's confidence score. `None` for an empty map.
fn max_probability(probabilities: &HashMap<String, f64>) -> Option<f64> {
    probabilities
        .values()
        .copied()
        .fold(None, |acc, p| match acc {
            Some(best) if best >= p => Some(best),
            _ => Some(p),
        })
}

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

    /// Create an error response using an explicit status code (e.g. from
    /// [`ApiResponse::error_code`]), rather than collapsing every endpoint failure to 500.
    pub fn from_status(status_code: u16, error_message: String, request_id: String) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Request-ID".to_string(), request_id.clone());

        Self {
            status_code,
            headers,
            body: ApiResponse::error_with_status(error_message, status_code, request_id),
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

/// Request counter for rate limiting.
///
/// `requests` is a real ring buffer (`VecDeque`, oldest at the front): since entries are always
/// appended in non-decreasing time order, expiring the oldest entries is an O(1)-amortized
/// `pop_front` loop rather than an O(n) `retain` rebuild.
#[derive(Debug, Clone)]
struct RequestCounter {
    /// Request timestamps, oldest first.
    requests: VecDeque<Instant>,
    /// Most recent activity, used to pick an eviction victim when the client map hits its cap
    /// (see `MAX_TRACKED_CLIENTS`).
    last_seen: Instant,
}

impl RequestCounter {
    fn new() -> Self {
        Self {
            requests: VecDeque::new(),
            last_seen: Instant::now(),
        }
    }

    /// Evict expired entries, then record and check this request.
    ///
    /// Eviction now runs on *every* call rather than being throttled to "once per 60 seconds"
    /// as before -- that throttle meant a configured "N per `window_minutes`" limit could
    /// actually admit up to roughly double `N` over a window nearly twice as wide before stale
    /// entries were finally swept out. The request is still counted (pushed) even when denied,
    /// so `get_request_count` continues to report every attempt, not just admitted ones.
    fn add_request(&mut self, window_minutes: usize, max_requests: usize) -> bool {
        let now = Instant::now();
        self.last_seen = now;
        self.evict_expired(window_minutes, now);

        let allowed = self.requests.len() < max_requests;
        self.requests.push_back(now);
        allowed
    }

    /// Remove requests older than the time window.
    ///
    /// Uses `checked_sub` rather than plain `Instant - Duration` subtraction: on some platforms
    /// / early in a process's life, `Instant::now()` can be closer to the clock's origin than
    /// the window itself, and unchecked subtraction there panics. When the window reaches back
    /// further than the clock's origin, nothing can have expired yet, so eviction is simply
    /// skipped for this call rather than panicking.
    fn evict_expired(&mut self, window_minutes: usize, now: Instant) {
        let cutoff = now.checked_sub(Duration::from_secs(window_minutes as u64 * 60));
        while let Some(&front) = self.requests.front() {
            let expired = match cutoff {
                Some(c) => front <= c,
                None => false,
            };
            if expired {
                self.requests.pop_front();
            } else {
                break;
            }
        }
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

    /// Check if request is allowed.
    ///
    /// If `client_id` is new and the tracked-client map is already at
    /// `MAX_TRACKED_CLIENTS`, evicts the least-recently-active tracked client first --
    /// otherwise an unbounded (e.g. spoofed) set of client identifiers could grow this map
    /// without limit.
    pub fn check_rate_limit(&self, client_id: &str) -> Result<bool> {
        let mut counts =
            write_lock_safe!(self.request_counts, "rate limiter request counts write")?;

        if !counts.contains_key(client_id) && counts.len() >= MAX_TRACKED_CLIENTS {
            if let Some(victim) = counts
                .iter()
                .min_by_key(|(_, counter)| counter.last_seen)
                .map(|(id, _)| id.clone())
            {
                counts.remove(&victim);
            }
        }

        let counter = counts
            .entry(client_id.to_string())
            .or_insert_with(RequestCounter::new);

        Ok(counter.add_request(self.window_minutes, self.max_requests_per_minute))
    }

    /// Get current request count for client
    pub fn get_request_count(&self, client_id: &str) -> Result<usize> {
        let counts = read_lock_safe!(self.request_counts, "rate limiter request counts read")?;
        Ok(counts
            .get(client_id)
            .map(|counter| counter.requests.len())
            .unwrap_or(0))
    }
}

/// Enhanced model server with HTTP-shaped request/response handlers.
///
/// **This does not open a network socket.** See the module-level doc comment: it's a pure
/// request/response transformer over [`ModelServer`] -- an integrator wires an actual transport
/// (axum, warp, actix-web, a serverless handler, ...) to call `handle_predict` et al.
pub struct HttpModelServer {
    /// Core model server. Its own `registry` (set via `set_registry` below) is the single
    /// source of truth for registry-backed model resolution; this type keeps no separate copy.
    model_server: ModelServer,
    /// Server configuration
    config: ServerConfig,
    /// Model monitors, one per registered model name.
    monitors: Arc<Mutex<HashMap<String, ModelMonitor>>>,
    /// Rate limiter. Always active, independent of `config.enable_auth` (see [`Self::new`]) --
    /// previously this was only constructed when authentication was enabled, which is backwards:
    /// an unauthenticated server is the one most exposed to abuse and most needs a rate limit.
    rate_limiter: RateLimiter,
    /// Request statistics
    request_stats: Arc<Mutex<RequestStatistics>>,
    /// When this server was constructed, for a real (not fabricated-zero) uptime figure in
    /// [`Self::get_server_stats`].
    started_at: Instant,
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
    /// Create a new HTTP model server.
    ///
    /// The rate limiter is always active (see `DEFAULT_RATE_LIMIT_PER_MINUTE`), independent of
    /// `config.enable_auth`.
    pub fn new(config: ServerConfig) -> Self {
        let model_server = ModelServer::new(config.clone());
        let rate_limiter = RateLimiter::new(DEFAULT_RATE_LIMIT_PER_MINUTE, 1);

        Self {
            model_server,
            config,
            monitors: Arc::new(Mutex::new(HashMap::new())),
            rate_limiter,
            request_stats: Arc::new(Mutex::new(RequestStatistics::default())),
            started_at: Instant::now(),
        }
    }

    /// Set the model registry consulted for names not directly registered on this server.
    /// Forwards to the inner [`ModelServer`] -- the single source of truth for registry-backed
    /// resolution (previously this stored a second, never-read copy on `HttpModelServer` itself).
    pub fn set_registry(&mut self, registry: Arc<dyn ModelRegistry>) {
        self.model_server.set_registry(registry);
    }

    /// Register a model
    pub fn register_model(&mut self, name: String, model: Box<dyn ModelServing>) -> Result<()> {
        // Register with model server
        self.model_server.register_model(name.clone(), model)?;

        // Create monitor for the model
        let metadata = self.model_server.get_model(&name)?.get_metadata().clone();
        let monitor = ModelMonitor::new(metadata);

        lock_safe!(self.monitors, "model server monitors lock")?.insert(name, monitor);

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

    /// Middleware for rate limiting. Always enforced (see [`Self::new`]).
    fn check_rate_limit(&self, context: &RequestContext) -> bool {
        let client_id = context.client_ip.as_deref().unwrap_or("unknown");
        self.rate_limiter
            .check_rate_limit(client_id)
            .unwrap_or(false)
    }

    /// Establish a live drift-detection baseline for one feature of a registered model, from a
    /// sample of reference (e.g. fit-time training) values -- see
    /// [`ModelMonitor::set_feature_baseline`].
    ///
    /// Without this, [`ModelMonitor`]'s real PSI machinery is unreachable through
    /// [`HttpModelServer`]: [`Self::handle_predict`] observes live traffic against whatever
    /// baselines exist (see `Self::observe_prediction_signals`), but nothing could ever
    /// configure one in the first place. Returns [`crate::core::error::Error::KeyNotFound`] if
    /// `model_name` hasn't been registered (via [`Self::register_model`]) yet.
    pub fn set_feature_baseline(
        &self,
        model_name: &str,
        feature_name: &str,
        reference_values: &[f64],
    ) -> Result<()> {
        let mut monitors = lock_safe!(self.monitors, "model server monitors lock (baseline)")?;
        let monitor = monitors.get_mut(model_name).ok_or_else(|| {
            crate::core::error::Error::KeyNotFound(format!(
                "Model '{}' has no monitor (register it first via register_model)",
                model_name
            ))
        })?;
        monitor.set_feature_baseline(feature_name, reference_values);
        Ok(())
    }

    /// Feed one successful prediction's real input features and (when available) prediction
    /// confidence into `model_name`'s monitor, when one is registered.
    ///
    /// This is what makes [`ModelMonitor::calculate_data_drift`] and its confidence metrics
    /// reachable from live HTTP traffic at all: [`ModelMonitor::observe_feature`] is a no-op
    /// against a feature with no configured baseline (see [`Self::set_feature_baseline`]), so
    /// this is always safe to call unconditionally on the success path, before any baseline
    /// exists.
    fn observe_prediction_signals(
        &self,
        model_name: &str,
        data: &HashMap<String, serde_json::Value>,
        probabilities: Option<&HashMap<String, f64>>,
    ) -> Result<()> {
        let mut monitors = lock_safe!(
            self.monitors,
            "model server monitors lock (observe signals)"
        )?;
        if let Some(monitor) = monitors.get_mut(model_name) {
            monitor.observe_features(data);
            if let Some(max_confidence) = probabilities.and_then(max_probability) {
                monitor.record_prediction_confidence(max_confidence);
            }
        }
        Ok(())
    }

    /// Batch counterpart to `Self::observe_prediction_signals`: feeds every successful
    /// prediction's real input features and confidence into `model_name`'s monitor in one lock
    /// acquisition.
    ///
    /// `predictions` preserves the original row order with failed rows omitted (see
    /// `GenericServingModel::predict_batch`), so it's aligned back to `observed_rows` by walking
    /// both in lockstep and skipping whichever indices `failed_indices` names.
    fn observe_batch_prediction_signals(
        &self,
        model_name: &str,
        observed_rows: &[HashMap<String, serde_json::Value>],
        predictions: &[PredictionResponse],
        failed_indices: &std::collections::HashSet<usize>,
    ) -> Result<()> {
        let mut monitors = lock_safe!(
            self.monitors,
            "model server monitors lock (observe batch signals)"
        )?;
        let Some(monitor) = monitors.get_mut(model_name) else {
            return Ok(());
        };

        let mut predictions_iter = predictions.iter();
        for (idx, data) in observed_rows.iter().enumerate() {
            if failed_indices.contains(&idx) {
                continue;
            }
            let Some(prediction) = predictions_iter.next() else {
                break;
            };
            monitor.observe_features(data);
            if let Some(max_confidence) =
                prediction.probabilities.as_ref().and_then(max_probability)
            {
                monitor.record_prediction_confidence(max_confidence);
            }
        }
        Ok(())
    }

    /// Record request statistics: the server-wide counters, and (when `model_name` is known)
    /// the request's outcome into that model's [`ModelMonitor`] too.
    ///
    /// Previously monitors were created in `register_model` and then never fed again: nothing
    /// in the HTTP handler path updated them, so every monitor stayed permanently empty after
    /// creation. This feeds `ModelMonitor::record_request`, which drives that model's real
    /// latency percentiles and error tracking.
    fn record_request(
        &self,
        model_name: Option<&str>,
        endpoint: &str,
        success: bool,
        response_time_ms: u64,
    ) -> Result<()> {
        {
            let mut stats = lock_safe!(self.request_stats, "model server request stats lock")?;
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

        if let Some(name) = model_name {
            let mut monitors = lock_safe!(self.monitors, "model server monitors lock (record)")?;
            if let Some(monitor) = monitors.get_mut(name) {
                // A monitoring hiccup must never fail the request itself; log and move on.
                if let Err(e) = monitor.record_request(success, response_time_ms) {
                    log::warn!(
                        "Failed to record request into monitor for '{}': {}",
                        name,
                        e
                    );
                }
            }
        }

        Ok(())
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
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::unauthorized(context.request_id);
        }

        // Rate limiting
        if !self.check_rate_limit(&context) {
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::too_many_requests(context.request_id);
        }

        // Validate request size (simplified)
        let request_size = serde_json::to_string(&request)
            .map(|s| s.len())
            .unwrap_or(0);

        if !RequestValidator::validate_request_size(request_size, self.config.max_request_size) {
            let error_msg = format!("Request too large: {} bytes", request_size);
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::bad_request(error_msg, context.request_id);
        }

        // Real input features, kept for real drift/confidence tracking below -- `request` itself
        // is moved into `PredictionEndpoint::predict` next.
        let observed_data = request.data.clone();

        // Handle prediction
        let response = PredictionEndpoint::predict(
            &self.model_server,
            model_name,
            request,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        let _ = self.record_request(Some(model_name), endpoint, response.success, response_time);

        if response.success {
            match response.data {
                Some(data) => {
                    // Feed live drift/confidence tracking from this real, successful prediction
                    // (a no-op against any feature/model without a configured baseline -- see
                    // `observe_prediction_signals`). Rejected/failed requests are never observed:
                    // unvalidated or malformed input has no business shaping a drift baseline.
                    let _ = self.observe_prediction_signals(
                        model_name,
                        &observed_data,
                        data.probabilities.as_ref(),
                    );
                    HttpResponse::ok(data, context.request_id)
                }
                None => HttpResponse::internal_server_error(
                    "Success response missing data".to_string(),
                    context.request_id,
                ),
            }
        } else {
            let status = response.error_code.unwrap_or(500);
            HttpResponse::from_status(
                status,
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
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::unauthorized(context.request_id);
        }

        // Rate limiting
        if !self.check_rate_limit(&context) {
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::too_many_requests(context.request_id);
        }

        // Real per-row input features, kept for real drift/confidence tracking below -- `request`
        // itself is moved into `BatchPredictionEndpoint::predict_batch` next.
        let observed_rows = request.data.clone();

        // Handle batch prediction
        let response = BatchPredictionEndpoint::predict_batch(
            &self.model_server,
            model_name,
            request,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        let _ = self.record_request(Some(model_name), endpoint, response.success, response_time);

        if response.success {
            match response.data {
                Some(data) => {
                    // Feed live drift/confidence tracking from every real, successful row in
                    // this batch (see `observe_batch_prediction_signals`); rejected/failed rows
                    // are excluded via `failed_items`, same as the single-predict path.
                    let failed_indices: std::collections::HashSet<usize> = data
                        .summary
                        .failed_items
                        .iter()
                        .map(|(idx, _)| *idx)
                        .collect();
                    let _ = self.observe_batch_prediction_signals(
                        model_name,
                        &observed_rows,
                        &data.predictions,
                        &failed_indices,
                    );
                    HttpResponse::ok(data, context.request_id)
                }
                None => HttpResponse::internal_server_error(
                    "Success response missing data".to_string(),
                    context.request_id,
                ),
            }
        } else {
            let status = response.error_code.unwrap_or(500);
            HttpResponse::from_status(
                status,
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
            let _ = self.record_request(
                Some(model_name),
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::unauthorized(context.request_id);
        }

        let response = ModelInfoEndpoint::get_model_info(
            &self.model_server,
            model_name,
            Some(context.request_id.clone()),
        );

        let response_time = start_time.elapsed().as_millis() as u64;
        let _ = self.record_request(Some(model_name), endpoint, response.success, response_time);

        if response.success {
            match response.data {
                Some(data) => HttpResponse::ok(data, context.request_id),
                None => HttpResponse::internal_server_error(
                    "Success response missing data".to_string(),
                    context.request_id,
                ),
            }
        } else {
            HttpResponse::not_found(
                response
                    .error
                    .unwrap_or_else(|| "Model not found".to_string()),
                context.request_id,
            )
        }
    }

    /// Handle health check request.
    ///
    /// Liveness (is the process up at all) is reachable **without authentication** and never
    /// enumerates the model inventory -- it's the one endpoint a load balancer / orchestrator
    /// must be able to probe before any credentials are available. Previously this endpoint ran
    /// the full detailed lookup unconditionally, handing an unauthenticated caller every
    /// registered model's name, type, and version. The detailed, per-model breakdown is served
    /// only once `authenticate` passes (which is always true when `enable_auth` is off, matching
    /// the existing behavior for that common case).
    pub fn handle_health_check(
        &self,
        model_name: Option<&str>,
        context: RequestContext,
    ) -> HttpResponse<ServerHealthStatus> {
        let start_time = Instant::now();
        let endpoint = "health_check";

        if !self.authenticate(&context) {
            let liveness = ServerHealthStatus {
                status: "ok".to_string(),
                total_models: 0,
                healthy_models: 0,
                model_statuses: HashMap::new(),
                timestamp: chrono::Utc::now(),
            };
            let _ = self.record_request(
                model_name,
                endpoint,
                true,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::ok(liveness, context.request_id);
        }

        let response = if let Some(model_name) = model_name {
            // Model-specific health check
            match HealthEndpoint::health_check_model(
                &self.model_server,
                model_name,
                Some(context.request_id.clone()),
            ) {
                resp if resp.success => {
                    // Convert single model health to server health format
                    let health_status = match resp.data {
                        Some(data) => data,
                        None => {
                            return HttpResponse::internal_server_error(
                                "Health check success but missing data".to_string(),
                                context.request_id,
                            )
                        }
                    };
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
        let _ = self.record_request(model_name, endpoint, response.success, response_time);

        if response.success {
            match response.data {
                Some(data) => HttpResponse::ok(data, context.request_id),
                None => HttpResponse::internal_server_error(
                    "Success response missing data".to_string(),
                    context.request_id,
                ),
            }
        } else {
            let status = response.error_code.unwrap_or(500);
            HttpResponse::from_status(
                status,
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
            let _ = self.record_request(
                None,
                endpoint,
                false,
                start_time.elapsed().as_millis() as u64,
            );
            return HttpResponse::unauthorized(context.request_id);
        }

        let response =
            ModelInfoEndpoint::list_models(&self.model_server, Some(context.request_id.clone()));

        let response_time = start_time.elapsed().as_millis() as u64;
        let _ = self.record_request(None, endpoint, response.success, response_time);

        match response.data {
            Some(data) => HttpResponse::ok(data, context.request_id),
            None => HttpResponse::internal_server_error(
                response
                    .error
                    .unwrap_or_else(|| "List models failed".to_string()),
                context.request_id,
            ),
        }
    }

    /// Get server statistics
    pub fn get_server_stats(&self) -> Result<ServerStats> {
        let stats = lock_safe!(
            self.request_stats,
            "model server request stats lock for stats query"
        )?;

        Ok(ServerStats {
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
            uptime_seconds: self.started_at.elapsed().as_secs(),
            active_models: self.model_server.list_models().len(),
        })
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
            assert!(rate_limiter
                .check_rate_limit("client1")
                .expect("operation should succeed"));
        }

        // Should deny 6th request
        assert!(!rate_limiter
            .check_rate_limit("client1")
            .expect("operation should succeed"));

        // Different client should be allowed
        assert!(rate_limiter
            .check_rate_limit("client2")
            .expect("operation should succeed"));
    }

    #[test]
    fn test_http_model_server() {
        let config = create_test_config();
        let server = HttpModelServer::new(config);

        let stats = server.get_server_stats().expect("operation should succeed");
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

    #[test]
    fn test_rate_limiter_active_even_with_auth_disabled() {
        // Regression: the rate limiter used to be constructed only when `enable_auth` was true
        // -- exactly backwards, since an unauthenticated server is the one most exposed to
        // abuse. `HttpModelServer::new` must always build one, regardless of `enable_auth`.
        let config = ServerConfig {
            enable_auth: false,
            ..create_test_config()
        };
        let server = HttpModelServer::new(config);
        // No public getter exposes the limiter directly; exercise it via repeated
        // `check_rate_limit` through the same code path `handle_predict` uses, by driving the
        // request budget down to confirm it engages at all (rather than being bypassed as
        // `None` would have silently done before).
        for _ in 0..DEFAULT_RATE_LIMIT_PER_MINUTE {
            assert!(server.rate_limiter.check_rate_limit("probe").unwrap());
        }
        assert!(
            !server.rate_limiter.check_rate_limit("probe").unwrap(),
            "rate limiter must engage once the per-minute budget is exhausted, even with \
             enable_auth: false"
        );
    }

    #[test]
    fn test_rate_limiter_evicts_lru_client_at_capacity() {
        // Fill the tracked-client map past MAX_TRACKED_CLIENTS with distinct clients, then
        // confirm the very first (least-recently-active) client was evicted: it should have a
        // fresh (zero) request count rather than a stale one from before eviction.
        let rate_limiter = RateLimiter::new(1_000_000, 60);
        for i in 0..MAX_TRACKED_CLIENTS {
            rate_limiter
                .check_rate_limit(&format!("client-{i}"))
                .unwrap();
        }
        assert_eq!(rate_limiter.get_request_count("client-0").unwrap(), 1);

        // One more distinct client must trigger an eviction rather than growing unbounded.
        rate_limiter
            .check_rate_limit(&format!("client-{MAX_TRACKED_CLIENTS}"))
            .unwrap();
        assert_eq!(
            rate_limiter.get_request_count("client-0").unwrap(),
            0,
            "the least-recently-active client must have been evicted to make room"
        );
    }

    #[test]
    fn test_health_check_liveness_without_auth_hides_model_inventory() {
        let config = ServerConfig {
            enable_auth: true,
            api_key: Some("secret".to_string()),
            ..create_test_config()
        };
        let mut server = HttpModelServer::new(config);

        let model = crate::ml::serving::serialization::GenericServingModel::from_serializable(
            crate::ml::serving::serialization::SerializableModel {
                schema_version: crate::ml::serving::serialization::CURRENT_SCHEMA_VERSION,
                metadata: crate::ml::serving::ModelMetadata {
                    name: "secret_model".to_string(),
                    version: "1.0.0".to_string(),
                    model_type: "linear_regression".to_string(),
                    feature_names: vec!["x".to_string()],
                    target_name: None,
                    description: String::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    metrics: HashMap::new(),
                    metadata: HashMap::new(),
                },
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("coefficients".to_string(), serde_json::json!([1.0]));
                    p
                },
                model_data: serde_json::json!({}),
                preprocessing: None,
                config: HashMap::new(),
            },
        )
        .unwrap();
        server
            .register_model("secret_model".to_string(), Box::new(model))
            .unwrap();

        // No API key provided: must succeed (200, liveness) but reveal nothing about the
        // registered model.
        let context = RequestContext::new();
        let response = server.handle_health_check(None, context);
        assert_eq!(response.status_code, 200);
        assert!(response.body.success);
        let body = response.body.data.expect("liveness body is present");
        assert_eq!(body.total_models, 0);
        assert!(body.model_statuses.is_empty());

        // With the correct API key: full detailed status, including the model inventory.
        let mut authed_context = RequestContext::new();
        authed_context.api_key = Some("secret".to_string());
        let authed_response = server.handle_health_check(None, authed_context);
        assert_eq!(authed_response.status_code, 200);
        let authed_body = authed_response.body.data.expect("detailed body is present");
        assert_eq!(authed_body.total_models, 1);
    }

    #[test]
    fn test_get_server_stats_reports_real_uptime() {
        // Regression: `uptime_seconds` used to be a hardcoded `0` regardless of how long the
        // server had actually been running. Sleep past the 1-second boundary so a hardcoded `0`
        // would fail this assertion, unlike a real `self.started_at.elapsed()`-derived value.
        let config = create_test_config();
        let server = HttpModelServer::new(config);
        std::thread::sleep(Duration::from_millis(1050));

        let stats = server.get_server_stats().unwrap();
        assert!(
            stats.uptime_seconds >= 1,
            "uptime_seconds must reflect real elapsed time, not a fabricated 0"
        );
    }

    /// Build a single-feature linear-regression model for the drift-wiring tests below.
    fn register_single_feature_model(server: &mut HttpModelServer, name: &str) {
        let model = crate::ml::serving::serialization::GenericServingModel::from_serializable(
            crate::ml::serving::serialization::SerializableModel {
                schema_version: crate::ml::serving::serialization::CURRENT_SCHEMA_VERSION,
                metadata: crate::ml::serving::ModelMetadata {
                    name: name.to_string(),
                    version: "1.0.0".to_string(),
                    model_type: "linear_regression".to_string(),
                    feature_names: vec!["x0".to_string()],
                    target_name: None,
                    description: String::new(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    metrics: HashMap::new(),
                    metadata: HashMap::new(),
                },
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("coefficients".to_string(), serde_json::json!([1.0]));
                    p.insert("intercept".to_string(), serde_json::json!(0.0));
                    p
                },
                model_data: serde_json::json!({}),
                preprocessing: None,
                config: HashMap::new(),
            },
        )
        .unwrap();
        server
            .register_model(name.to_string(), Box::new(model))
            .unwrap();
    }

    #[test]
    fn test_set_feature_baseline_errors_for_unregistered_model() {
        let server = HttpModelServer::new(create_test_config());
        let err = server
            .set_feature_baseline("does_not_exist", "x0", &[1.0, 2.0, 3.0])
            .expect_err("no monitor exists for an unregistered model");
        assert!(matches!(err, crate::core::error::Error::KeyNotFound(_)));
    }

    #[test]
    fn test_handle_predict_feeds_live_traffic_into_configured_drift_baseline() {
        // Regression for the PSI-drift mechanism being real but *unreachable* through
        // `HttpModelServer`: before `set_feature_baseline`/`observe_prediction_signals` existed,
        // nothing could ever configure a baseline on this server, and `handle_predict` never fed
        // live request data into whatever baseline existed even if one had been set some other
        // way -- so `calculate_data_drift()` stayed permanently `None` for every model served
        // through this type, regardless of how skewed live traffic became.
        let mut server = HttpModelServer::new(create_test_config());
        register_single_feature_model(&mut server, "drift_model");

        // Baseline: uniform over [0, 100).
        let reference: Vec<f64> = (0..500).map(|i| (i % 100) as f64).collect();
        server
            .set_feature_baseline("drift_model", "x0", &reference)
            .expect("baseline configured for a registered model");

        // Drive real traffic through the HTTP-shaped handler (not ModelMonitor directly),
        // clustered at the extreme high end -- a severe, detectable shift.
        for _ in 0..200 {
            let mut data = HashMap::new();
            data.insert("x0".to_string(), serde_json::json!(99.0));
            let request = PredictionRequest {
                data,
                model_version: None,
                options: None,
            };
            let response = server.handle_predict("drift_model", request, RequestContext::new());
            assert_eq!(response.status_code, 200, "prediction must succeed");
        }

        let monitors = server.monitors.lock().expect("monitors lock");
        let monitor = monitors.get("drift_model").expect("monitor was created");
        let drift = monitor
            .calculate_data_drift()
            .expect("live traffic was observed against the configured baseline");
        assert_eq!(drift.detection_method, "PSI");
        assert!(
            drift.drift_detected,
            "a fully-shifted distribution driven through handle_predict must be detected, got \
             score {}",
            drift.drift_score
        );
    }

    #[test]
    fn test_handle_predict_without_a_baseline_stays_honestly_driftless() {
        // No `set_feature_baseline` call at all: `observe_prediction_signals` must be a
        // harmless no-op (never fabricate a baseline out of live traffic on its own), so drift
        // stays `None` rather than becoming spuriously "detected" against nothing.
        let mut server = HttpModelServer::new(create_test_config());
        register_single_feature_model(&mut server, "no_baseline_model");

        let mut data = HashMap::new();
        data.insert("x0".to_string(), serde_json::json!(42.0));
        let request = PredictionRequest {
            data,
            model_version: None,
            options: None,
        };
        let response = server.handle_predict("no_baseline_model", request, RequestContext::new());
        assert_eq!(response.status_code, 200);

        let monitors = server.monitors.lock().expect("monitors lock");
        let monitor = monitors
            .get("no_baseline_model")
            .expect("monitor was created");
        assert!(monitor.calculate_data_drift().is_none());
    }

    #[test]
    fn test_handle_predict_batch_feeds_live_traffic_into_configured_drift_baseline() {
        // Batch counterpart of `test_handle_predict_feeds_live_traffic_into_configured_drift_baseline`:
        // `observe_batch_prediction_signals` must align each successful row back to its original
        // input despite `predictions` omitting failed rows.
        //
        // Deliberately a *leading* invalid row followed by exactly one valid row (not two
        // identical valid rows around the failure): if alignment were off by one -- e.g. naively
        // zipping `predictions` against `observed_rows` positionally instead of skipping
        // `failed_indices` -- the single real prediction would get paired with the invalid row's
        // data (which has no `x0` key at all), so `observe_features` would find nothing to
        // observe and drift would stay incorrectly `None`. Two identical valid rows around the
        // failure couldn't distinguish correct alignment from this exact bug, since either
        // pairing would happen to observe the same repeated value.
        let mut server = HttpModelServer::new(create_test_config());
        register_single_feature_model(&mut server, "batch_drift_model");

        let reference: Vec<f64> = (0..500).map(|i| (i % 100) as f64).collect();
        server
            .set_feature_baseline("batch_drift_model", "x0", &reference)
            .unwrap();

        let bad_row: HashMap<String, serde_json::Value> = HashMap::new(); // missing x0
        let mut good_row = HashMap::new();
        good_row.insert("x0".to_string(), serde_json::json!(99.0));

        let batch = BatchPredictionRequest {
            data: vec![bad_row, good_row],
            model_version: None,
            options: None,
        };
        let response =
            server.handle_predict_batch("batch_drift_model", batch, RequestContext::new());
        assert_eq!(response.status_code, 200);
        let body = response.body.data.expect("batch response body");
        assert_eq!(body.summary.successful_predictions, 1);
        assert_eq!(body.summary.failed_predictions, 1);
        assert_eq!(
            body.summary.failed_items[0].0, 0,
            "the invalid row is at original index 0"
        );

        let monitors = server.monitors.lock().expect("monitors lock");
        let monitor = monitors
            .get("batch_drift_model")
            .expect("monitor was created");
        // Under the misalignment this test is built to catch, the sole prediction would be
        // paired with the invalid row's (empty) data, `observe_features` would find no `x0` key
        // to read, and this would incorrectly stay `None` forever.
        let drift = monitor.calculate_data_drift().expect(
            "the single successful row's real value (99.0) must have been observed against the \
             baseline -- None here means the prediction was paired with the wrong input row",
        );
        assert!(
            drift.drift_detected,
            "one observation at the extreme high end of the baseline must register as drift"
        );
        assert!(drift.drifting_features.contains(&"x0".to_string()));
    }
}

//! Model Deployment Module
//!
//! This module provides model deployment capabilities including deployment configuration,
//! resource management, scaling, and health monitoring.

use crate::core::error::{Error, Result};
use crate::lock_safe;
use crate::ml::serving::serialization::SerializableModel;
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, DeploymentConfig, HealthStatus, ModelInfo,
    ModelMetadata, ModelServing, PredictionRequest, PredictionResponse,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Maximum number of `(timestamp, latency)` request entries kept for windowed rate/latency
/// calculations, as a hard cap independent of the time window (defense against unbounded growth
/// under an extreme, sustained request flood).
const MAX_TRACKED_REQUESTS: usize = 5_000;

/// Maximum number of recent errors retained for [`DeploymentMetrics::recent_errors`].
const MAX_RECENT_ERRORS: usize = 200;

/// A single recorded prediction failure, captured from the real `Error` the underlying model
/// returned -- not a fabricated category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedError {
    /// Coarse error category, derived from the concrete `Error` variant (e.g. `"InvalidInput"`).
    pub error_type: String,
    /// The error's `Display` message.
    pub message: String,
    /// When the error was recorded.
    pub occurred_at: chrono::DateTime<chrono::Utc>,
}

/// Derive a coarse, real error category from an `Error`'s variant, for grouping in monitoring
/// (replacing a previous implementation that invented a fixed 70/20/10 split regardless of what
/// actually went wrong).
fn error_type_name(error: &Error) -> &'static str {
    match error {
        Error::KeyNotFound(_) => "KeyNotFound",
        Error::InvalidInput(_) => "InvalidInput",
        Error::InvalidOperation(_) => "InvalidOperation",
        Error::DimensionMismatch(_) => "DimensionMismatch",
        Error::NotImplemented(_) => "NotImplemented",
        Error::SerializationError(_) | Error::Json(_) | Error::JsonError(_) => "Serialization",
        Error::Computation(_) => "Computation",
        _ => "Other",
    }
}

/// Subtract `duration` from `now`, treating an underflow (the window reaches back further than
/// the monotonic clock's origin -- possible very early in a process's life on some platforms) as
/// "nothing is old enough to expire" rather than panicking (`Instant`'s `Sub<Duration>` panics on
/// underflow).
fn checked_cutoff(now: Instant, duration: Duration) -> Option<Instant> {
    now.checked_sub(duration)
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeploymentStatus {
    /// Deployment is starting up
    Starting,
    /// Deployment is running and healthy
    Running,
    /// Deployment is unhealthy but still running
    Degraded,
    /// Deployment is stopping
    Stopping,
    /// Deployment has stopped
    Stopped,
    /// Deployment failed
    Failed,
}

/// Deployment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentMetrics {
    /// Current status
    pub status: DeploymentStatus,
    /// Number of active instances
    pub active_instances: usize,
    /// Current CPU utilization (0.0 to 1.0), when real OS-level telemetry is available.
    ///
    /// Always `None` in this build: computing this honestly requires OS-level process metrics
    /// (e.g. via a `sysinfo`-like crate), which this module intentionally does not depend on.
    /// Autoscaling decisions (see [`DeployedModel::should_scale_up`]) are driven by measured
    /// request latency and in-flight request count instead of a fabricated proxy for this field.
    pub cpu_utilization: Option<f64>,
    /// Current memory utilization (0.0 to 1.0). See `cpu_utilization`: always `None` here for the
    /// same honesty reason.
    pub memory_utilization: Option<f64>,
    /// Current request rate (requests per second)
    pub request_rate: f64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Total number of requests served
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Requests currently in flight (started but not yet completed). A real, measured proxy for
    /// queue depth, used by the autoscaling decision.
    pub in_flight_requests: usize,
    /// Recent per-request response times in milliseconds, bounded to the same window/cap as
    /// `RequestStats`. Used by `ModelMonitor` to compute real latency percentiles instead of
    /// fabricated multiples of the mean.
    pub response_times_ms: Vec<u64>,
    /// Recent real prediction failures (bounded), for real error-type breakdowns in monitoring.
    pub recent_errors: Vec<RecordedError>,
    /// Last health check timestamp
    pub last_health_check: chrono::DateTime<chrono::Utc>,
    /// Deployment start time
    pub started_at: chrono::DateTime<chrono::Utc>,
    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Deployed model wrapper
pub struct DeployedModel {
    /// Underlying model
    model: Box<dyn ModelServing>,
    /// Deployment configuration
    config: DeploymentConfig,
    /// Deployment metrics
    metrics: Arc<Mutex<DeploymentMetrics>>,
    /// Request statistics
    stats: Arc<Mutex<RequestStats>>,
    /// Health check status
    health_status: Arc<Mutex<HealthStatus>>,
}

/// Request statistics.
///
/// `entries` pairs each request's timestamp with its outcome (`Some(latency_ms)` on success,
/// `None` on failure) in a single aligned sequence. Previously, successes and failures were
/// pushed into two *separate* parallel vectors (`request_times`/`response_times`); once any
/// error occurred, the two vectors went out of alignment, so `cleanup_old_entries`' index-based
/// pruning silently paired each remaining timestamp with the *wrong* response time.
#[derive(Debug, Clone)]
struct RequestStats {
    /// `(timestamp, outcome)` for every request, oldest first.
    entries: VecDeque<(Instant, Option<u64>)>,
    /// Recent real failures (bounded to `MAX_RECENT_ERRORS`), each pushed alongside its `entries`
    /// tombstone above.
    recent_errors: VecDeque<RecordedError>,
    /// Lifetime error count (never pruned by the windowed cleanup, unlike `entries`).
    error_count: u64,
    /// Lifetime success count (never pruned).
    success_count: u64,
    /// Requests started but not yet completed.
    in_flight: usize,
}

impl RequestStats {
    fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            recent_errors: VecDeque::new(),
            error_count: 0,
            success_count: 0,
            in_flight: 0,
        }
    }

    /// Record a successful request.
    fn record_success(&mut self, response_time_ms: u64) {
        self.entries
            .push_back((Instant::now(), Some(response_time_ms)));
        self.success_count += 1;
        self.evict_expired();
    }

    /// Record a failed request, with the real error category and message.
    fn record_error(&mut self, error_type: String, message: String) {
        self.entries.push_back((Instant::now(), None));
        self.error_count += 1;
        self.recent_errors.push_back(RecordedError {
            error_type,
            message,
            occurred_at: chrono::Utc::now(),
        });
        while self.recent_errors.len() > MAX_RECENT_ERRORS {
            self.recent_errors.pop_front();
        }
        self.evict_expired();
    }

    /// Remove entries older than 5 minutes, and cap total tracked entries as a hard safety net.
    /// Runs on every record (not throttled to "once per 60s" as before), so the windowed rate
    /// and latency figures reflect the configured window precisely rather than up to a minute
    /// stale.
    fn evict_expired(&mut self) {
        let cutoff = checked_cutoff(Instant::now(), Duration::from_secs(300));
        while let Some(&(t, _)) = self.entries.front() {
            let expired = match cutoff {
                Some(c) => t <= c,
                None => false,
            };
            if expired {
                self.entries.pop_front();
            } else {
                break;
            }
        }
        while self.entries.len() > MAX_TRACKED_REQUESTS {
            self.entries.pop_front();
        }
    }

    /// Calculate current request rate (requests per second) over the trailing 60 seconds.
    fn calculate_request_rate(&self) -> f64 {
        let cutoff = checked_cutoff(Instant::now(), Duration::from_secs(60));
        let recent_requests = self
            .entries
            .iter()
            .filter(|&&(time, _)| match cutoff {
                Some(c) => time > c,
                None => true,
            })
            .count();

        recent_requests as f64 / 60.0
    }

    /// Calculate average response time over the tracked (successful) window.
    fn calculate_avg_response_time(&self) -> f64 {
        let (sum, count) = self
            .entries
            .iter()
            .filter_map(|&(_, outcome)| outcome)
            .fold((0u64, 0u64), |(sum, count), latency| {
                (sum + latency, count + 1)
            });
        if count == 0 {
            0.0
        } else {
            sum as f64 / count as f64
        }
    }

    /// Calculate lifetime error rate (not windowed, matching `total_requests` below).
    fn calculate_error_rate(&self) -> f64 {
        let total = self.success_count + self.error_count;
        if total == 0 {
            0.0
        } else {
            self.error_count as f64 / total as f64
        }
    }

    /// A snapshot of tracked (successful-request) response times, for real percentile
    /// computation in `ModelMonitor`.
    fn response_times_snapshot(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter_map(|&(_, outcome)| outcome)
            .collect()
    }

    /// A snapshot of recently recorded real failures.
    fn recent_errors_snapshot(&self) -> Vec<RecordedError> {
        self.recent_errors.iter().cloned().collect()
    }
}

impl DeployedModel {
    /// Create a new deployed model
    pub fn new(model: Box<dyn ModelServing>, config: DeploymentConfig) -> Result<Self> {
        let metrics = Arc::new(Mutex::new(DeploymentMetrics {
            status: DeploymentStatus::Starting,
            active_instances: 1,
            cpu_utilization: None,
            memory_utilization: None,
            request_rate: 0.0,
            avg_response_time_ms: 0.0,
            error_rate: 0.0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            in_flight_requests: 0,
            response_times_ms: Vec::new(),
            recent_errors: Vec::new(),
            last_health_check: chrono::Utc::now(),
            started_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }));

        let stats = Arc::new(Mutex::new(RequestStats::new()));

        let health_status = Arc::new(Mutex::new(HealthStatus {
            status: "starting".to_string(),
            details: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }));

        let deployed_model = Self {
            model,
            config,
            metrics,
            stats,
            health_status,
        };

        // Perform initial health check, then honor its actual result: only promote to Running
        // when the probe really reported healthy. (Previously this unconditionally forced
        // `Running` regardless of what the health check returned, silently masking a failed
        // initial probe -- e.g. a model with missing/non-finite coefficients would still report
        // as "deployed and running".)
        deployed_model.update_health_status()?;
        {
            let probe_healthy = {
                let health = lock_safe!(
                    deployed_model.health_status,
                    "deployment health status lock"
                )?;
                health.status == "healthy"
            };
            let mut metrics = lock_safe!(deployed_model.metrics, "deployment metrics lock")?;
            metrics.status = if probe_healthy {
                DeploymentStatus::Running
            } else {
                DeploymentStatus::Failed
            };
            metrics.updated_at = chrono::Utc::now();
        }

        Ok(deployed_model)
    }

    /// Get deployment configuration
    pub fn get_config(&self) -> &DeploymentConfig {
        &self.config
    }

    /// Get deployment metrics
    pub fn get_metrics(&self) -> Result<DeploymentMetrics> {
        Ok(lock_safe!(self.metrics, "deployment metrics lock for get")?.clone())
    }

    /// Update deployment metrics from real, measured request statistics.
    fn update_metrics(&self) -> Result<()> {
        let stats = lock_safe!(self.stats, "deployment stats lock")?;
        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock for update")?;

        metrics.request_rate = stats.calculate_request_rate();
        metrics.avg_response_time_ms = stats.calculate_avg_response_time();
        metrics.error_rate = stats.calculate_error_rate();
        metrics.total_requests = stats.success_count + stats.error_count;
        metrics.successful_requests = stats.success_count;
        metrics.failed_requests = stats.error_count;
        metrics.in_flight_requests = stats.in_flight;
        metrics.response_times_ms = stats.response_times_snapshot();
        metrics.recent_errors = stats.recent_errors_snapshot();
        metrics.updated_at = chrono::Utc::now();

        // CPU/memory utilization require real OS-level telemetry this module doesn't have
        // access to (see the field docs on `DeploymentMetrics`); left honestly `None` rather
        // than a synthesized proxy. Autoscaling below is driven by in_flight_requests /
        // avg_response_time_ms instead, which are both real, measured signals.
        metrics.cpu_utilization = None;
        metrics.memory_utilization = None;

        Ok(())
    }

    /// Update health status
    fn update_health_status(&self) -> Result<()> {
        let health_result = self.model.health_check();

        let mut health_status = lock_safe!(self.health_status, "deployment health status lock")?;
        match health_result {
            Ok(status) => {
                *health_status = status;
            }
            Err(e) => {
                health_status.status = "unhealthy".to_string();
                health_status.details.clear();
                health_status
                    .details
                    .insert("error".to_string(), e.to_string());
                health_status.timestamp = chrono::Utc::now();
            }
        }

        // Update deployment status based on health
        {
            let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
            metrics.last_health_check = chrono::Utc::now();

            if health_status.status == "healthy" {
                if metrics.status == DeploymentStatus::Degraded {
                    metrics.status = DeploymentStatus::Running;
                }
            } else if health_status.status == "unhealthy" {
                metrics.status = DeploymentStatus::Degraded;
            }
        }

        Ok(())
    }

    /// Ratio of in-flight requests to the configured concurrency ceiling
    /// (`resources.max_concurrent_requests`) -- a real, measured proxy for queue depth.
    fn queue_pressure(&self, metrics: &DeploymentMetrics) -> f64 {
        let ceiling = self.config.resources.max_concurrent_requests;
        if ceiling == 0 {
            0.0
        } else {
            metrics.in_flight_requests as f64 / ceiling as f64
        }
    }

    /// Ratio of average response time to the configured health-check timeout budget -- a real,
    /// measured proxy for latency pressure. Reuses `health_check.timeout_seconds` as the budget
    /// since `ScalingConfig` has no latency-specific field of its own.
    fn latency_pressure(&self, metrics: &DeploymentMetrics) -> f64 {
        let timeout_ms = self.config.health_check.timeout_seconds as f64 * 1000.0;
        if timeout_ms <= 0.0 {
            0.0
        } else {
            metrics.avg_response_time_ms / timeout_ms
        }
    }

    /// Check if scaling up is needed, driven by real measured signals: request latency relative
    /// to the health-check timeout budget, and in-flight request count relative to the
    /// configured concurrency ceiling.
    ///
    /// Previously this read a `cpu_utilization`/`memory_utilization` figure that was itself
    /// synthesized from a *lifetime* request counter (`total_requests / 10000`), which only ever
    /// grew and, once past the threshold, stayed pinned near 1.0 forever -- triggering
    /// permanent scale-up regardless of *current* load.
    pub fn should_scale_up(&self) -> Result<bool> {
        let metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        let config = &self.config.scaling;

        Ok(self.latency_pressure(&metrics) > config.scale_up_threshold
            || self.queue_pressure(&metrics) > config.scale_up_threshold)
    }

    /// Check if scaling down is needed, using the same real signals as [`Self::should_scale_up`].
    pub fn should_scale_down(&self) -> Result<bool> {
        let metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        let config = &self.config.scaling;

        Ok(metrics.active_instances > config.min_instances
            && self.latency_pressure(&metrics) < config.scale_down_threshold
            && self.queue_pressure(&metrics) < config.scale_down_threshold)
    }

    /// Scale up deployment
    pub fn scale_up(&self) -> Result<()> {
        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        let config = &self.config.scaling;

        if metrics.active_instances < config.max_instances {
            metrics.active_instances += 1;
            metrics.updated_at = chrono::Utc::now();
            log::info!(
                "Scaled up deployment to {} instances",
                metrics.active_instances
            );
        }

        Ok(())
    }

    /// Scale down deployment
    pub fn scale_down(&self) -> Result<()> {
        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        let config = &self.config.scaling;

        if metrics.active_instances > config.min_instances {
            metrics.active_instances -= 1;
            metrics.updated_at = chrono::Utc::now();
            log::info!(
                "Scaled down deployment to {} instances",
                metrics.active_instances
            );
        }

        Ok(())
    }

    /// Stop deployment
    pub fn stop(&self) -> Result<()> {
        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        metrics.status = DeploymentStatus::Stopping;
        metrics.updated_at = chrono::Utc::now();

        // In a real implementation, this would stop the actual instances

        metrics.status = DeploymentStatus::Stopped;
        metrics.active_instances = 0;

        Ok(())
    }

    /// Restart deployment
    pub fn restart(&self) -> Result<()> {
        self.stop()?;

        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        metrics.status = DeploymentStatus::Starting;
        metrics.active_instances = self.config.scaling.min_instances;
        metrics.updated_at = chrono::Utc::now();

        // Perform health check and honor its actual result, same as the constructor: don't
        // unconditionally force `Running` regardless of whether the post-restart probe passed.
        drop(metrics);
        self.update_health_status()?;

        let probe_healthy = {
            let health = lock_safe!(self.health_status, "deployment health status lock")?;
            health.status == "healthy"
        };
        let mut metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
        metrics.status = if probe_healthy {
            DeploymentStatus::Running
        } else {
            DeploymentStatus::Failed
        };

        Ok(())
    }
}

impl ModelServing for DeployedModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionResponse> {
        let start_time = Instant::now();

        // Check if deployment is healthy
        {
            let metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
            if metrics.status != DeploymentStatus::Running {
                return Err(Error::InvalidOperation(format!(
                    "Deployment is not running (status: {:?})",
                    metrics.status
                )));
            }
        }

        {
            let mut stats = lock_safe!(self.stats, "deployment stats lock (in-flight enter)")?;
            stats.in_flight += 1;
        }

        // Perform prediction
        let result = self.model.predict(request);

        // Record statistics: real per-request latency, and (on failure) the real error
        // category/message rather than a bare failed-count increment.
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = lock_safe!(self.stats, "deployment stats lock (record)")?;
            stats.in_flight = stats.in_flight.saturating_sub(1);
            match &result {
                Ok(_) => stats.record_success(processing_time),
                Err(e) => stats.record_error(error_type_name(e).to_string(), e.to_string()),
            }
        }

        // Update metrics from the statistics just recorded.
        self.update_metrics()?;

        result
    }

    fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse> {
        let start_time = Instant::now();

        // Check if deployment is healthy
        {
            let metrics = lock_safe!(self.metrics, "deployment metrics lock")?;
            if metrics.status != DeploymentStatus::Running {
                return Err(Error::InvalidOperation(format!(
                    "Deployment is not running (status: {:?})",
                    metrics.status
                )));
            }
        }

        {
            let mut stats = lock_safe!(self.stats, "deployment stats lock (in-flight enter)")?;
            stats.in_flight += 1;
        }

        // Perform batch prediction
        let result = self.model.predict_batch(request);

        // Record statistics
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut stats = lock_safe!(self.stats, "deployment stats lock (record)")?;
            stats.in_flight = stats.in_flight.saturating_sub(1);
            match &result {
                Ok(response) => {
                    // Real per-item latency computed in f64 space, then rounded once at the
                    // point of storage -- avoiding the previous integer division
                    // (`processing_time / len`), which floored to exactly 0ms per item whenever
                    // the whole batch finished in fewer milliseconds than it had items.
                    let denom = request.data.len().max(1) as f64;
                    let avg_time_ms = (processing_time as f64 / denom).round().max(0.0) as u64;
                    for _ in 0..response.summary.successful_predictions {
                        stats.record_success(avg_time_ms);
                    }
                    // Real per-item errors (index + message), when the underlying model
                    // reported them; falls back to a generic category only if it didn't.
                    if response.summary.failed_items.is_empty() {
                        for _ in 0..response.summary.failed_predictions {
                            stats.record_error(
                                "BatchItemFailed".to_string(),
                                "batch item failed (no per-item detail reported by the model)"
                                    .to_string(),
                            );
                        }
                    } else {
                        for (idx, message) in &response.summary.failed_items {
                            stats.record_error(
                                "BatchItemFailed".to_string(),
                                format!("item {}: {}", idx, message),
                            );
                        }
                    }
                }
                Err(e) => stats.record_error(error_type_name(e).to_string(), e.to_string()),
            }
        }

        // Update metrics
        self.update_metrics()?;

        result
    }

    fn get_metadata(&self) -> &ModelMetadata {
        self.model.get_metadata()
    }

    fn health_check(&self) -> Result<HealthStatus> {
        self.update_health_status()?;
        Ok(lock_safe!(self.health_status, "deployment health status lock")?.clone())
    }

    fn info(&self) -> ModelInfo {
        let mut info = self.model.info();

        // Add deployment information
        info.configuration.insert(
            "deployment_config".to_string(),
            serde_json::to_value(&self.config).unwrap_or(serde_json::Value::Null),
        );

        info.configuration.insert(
            "deployment_metrics".to_string(),
            self.get_metrics()
                .ok()
                .and_then(|m| serde_json::to_value(&m).ok())
                .unwrap_or(serde_json::Value::Null),
        );

        info
    }

    fn to_serializable(&self) -> Result<SerializableModel> {
        // Delegate to the wrapped model; a deployment is a runtime wrapper, not a distinct
        // persistable model in its own right.
        self.model.to_serializable()
    }
}

/// Deployment manager for managing multiple deployments
pub struct DeploymentManager {
    /// Active deployments
    deployments: HashMap<String, DeployedModel>,
    /// Deployment configurations
    configs: HashMap<String, DeploymentConfig>,
}

impl DeploymentManager {
    /// Create a new deployment manager
    pub fn new() -> Self {
        Self {
            deployments: HashMap::new(),
            configs: HashMap::new(),
        }
    }

    /// Deploy a model
    pub fn deploy(
        &mut self,
        deployment_name: String,
        model: Box<dyn ModelServing>,
        config: DeploymentConfig,
    ) -> Result<()> {
        if self.deployments.contains_key(&deployment_name) {
            return Err(Error::InvalidOperation(format!(
                "Deployment '{}' already exists",
                deployment_name
            )));
        }

        let deployed_model = DeployedModel::new(model, config.clone())?;

        self.deployments
            .insert(deployment_name.clone(), deployed_model);
        self.configs.insert(deployment_name, config);

        Ok(())
    }

    /// Undeploy a model
    pub fn undeploy(&mut self, deployment_name: &str) -> Result<()> {
        if let Some(deployment) = self.deployments.get(deployment_name) {
            deployment.stop()?;
        }

        self.deployments.remove(deployment_name);
        self.configs.remove(deployment_name);

        Ok(())
    }

    /// Get deployment
    pub fn get_deployment(&self, deployment_name: &str) -> Option<&DeployedModel> {
        self.deployments.get(deployment_name)
    }

    /// List all deployments
    pub fn list_deployments(&self) -> Vec<String> {
        self.deployments.keys().cloned().collect()
    }

    /// Get deployment metrics
    pub fn get_deployment_metrics(&self, deployment_name: &str) -> Option<DeploymentMetrics> {
        self.deployments
            .get(deployment_name)
            .and_then(|deployment| deployment.get_metrics().ok())
    }

    /// Scale deployment
    pub fn scale_deployment(&self, deployment_name: &str, target_instances: usize) -> Result<()> {
        let deployment = self.deployments.get(deployment_name).ok_or_else(|| {
            Error::KeyNotFound(format!("Deployment '{}' not found", deployment_name))
        })?;

        let current_instances = deployment.get_metrics()?.active_instances;

        if target_instances > current_instances {
            for _ in current_instances..target_instances {
                deployment.scale_up()?;
            }
        } else if target_instances < current_instances {
            for _ in target_instances..current_instances {
                deployment.scale_down()?;
            }
        }

        Ok(())
    }

    /// Auto-scale all deployments based on metrics
    pub fn auto_scale_all(&self) -> Result<()> {
        for deployment in self.deployments.values() {
            if deployment.should_scale_up()? {
                deployment.scale_up()?;
            } else if deployment.should_scale_down()? {
                deployment.scale_down()?;
            }
        }
        Ok(())
    }

    /// Health check all deployments
    pub fn health_check_all(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();

        for (name, deployment) in &self.deployments {
            match deployment.health_check() {
                Ok(status) => {
                    results.insert(name.clone(), status);
                }
                Err(e) => {
                    results.insert(
                        name.clone(),
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
            }
        }

        results
    }
}

impl Default for DeploymentManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::serving::{HealthCheckConfig, MonitoringConfig, ResourceConfig, ScalingConfig};

    fn create_test_config() -> DeploymentConfig {
        DeploymentConfig {
            model_name: "test_model".to_string(),
            model_version: "1.0.0".to_string(),
            environment: "test".to_string(),
            resources: ResourceConfig {
                cpu_cores: 1.0,
                memory_mb: 1024,
                gpu_memory_mb: None,
                max_concurrent_requests: 10,
            },
            scaling: ScalingConfig {
                min_instances: 1,
                max_instances: 5,
                target_cpu_utilization: 0.7,
                target_memory_utilization: 0.8,
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
            },
            health_check: HealthCheckConfig {
                path: "/health".to_string(),
                interval_seconds: 30,
                timeout_seconds: 5,
                failure_threshold: 3,
                success_threshold: 2,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_logging: true,
                enable_tracing: false,
                metrics_interval_seconds: 60,
                log_level: "info".to_string(),
            },
        }
    }

    #[test]
    fn test_deployment_status() {
        let status = DeploymentStatus::Running;
        assert_eq!(status, DeploymentStatus::Running);
    }

    #[test]
    fn test_request_stats() {
        let mut stats = RequestStats::new();

        // Record some requests
        stats.record_success(100);
        stats.record_success(150);
        stats.record_error(
            "InvalidInput".to_string(),
            "missing feature 'x'".to_string(),
        );

        assert!(stats.calculate_avg_response_time() > 0.0);
        assert!(stats.calculate_error_rate() > 0.0);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.error_count, 1);

        // The (timestamp, outcome) entries stay aligned even after an error is recorded --
        // this is the alignment bug regression: avg response time must reflect only the two
        // real successes (125.0), never be dragged toward 0 by a misindexed failure slot.
        assert!((stats.calculate_avg_response_time() - 125.0).abs() < 1e-9);

        // The real error is retrievable with its category and message, not just a bare count.
        let errors = stats.recent_errors_snapshot();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].error_type, "InvalidInput");
        assert_eq!(errors[0].message, "missing feature 'x'");
    }

    #[test]
    fn test_deployment_manager() {
        let manager = DeploymentManager::new();

        // Test that manager starts empty
        assert!(manager.list_deployments().is_empty());

        // Test deployment configuration
        let config = create_test_config();
        assert_eq!(config.model_name, "test_model");
        assert_eq!(config.scaling.min_instances, 1);
        assert_eq!(config.scaling.max_instances, 5);
    }
}

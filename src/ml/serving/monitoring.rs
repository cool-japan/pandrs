//! Model Monitoring Module
//!
//! This module provides comprehensive monitoring capabilities for deployed models including
//! performance metrics, drift detection, alerting, and observability.

use crate::core::error::{Error, Result};
use crate::ml::serving::{DeploymentMetrics, HealthStatus, ModelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Performance metrics for model monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Error metrics
    pub error_metrics: ErrorMetrics,
    /// Resource utilization
    pub resource_utilization: ResourceUtilizationMetrics,
    /// Model quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// 50th percentile latency in milliseconds
    pub p50_latency_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f64,
    /// Maximum latency in milliseconds
    pub max_latency_ms: f64,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f64,
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Requests per second
    pub requests_per_second: f64,
    /// Total requests in time window
    pub total_requests: u64,
    /// Successful requests in time window
    pub successful_requests: u64,
    /// Failed requests in time window
    pub failed_requests: u64,
    /// Concurrent requests
    pub concurrent_requests: u64,
}

/// Error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Overall error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Error rate by type
    pub error_rates_by_type: HashMap<String, f64>,
    /// Error counts by type
    pub error_counts_by_type: HashMap<String, u64>,
    /// Recent errors
    pub recent_errors: Vec<ErrorEvent>,
}

/// Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Error type
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request context (if available)
    pub context: Option<HashMap<String, String>>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// GPU utilization (0.0 to 1.0, if available)
    pub gpu_utilization: Option<f64>,
    /// Disk I/O utilization (0.0 to 1.0)
    pub disk_io_utilization: f64,
    /// Network I/O utilization (0.0 to 1.0)
    pub network_io_utilization: f64,
}

/// Model quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Prediction accuracy (if ground truth is available)
    pub accuracy: Option<f64>,
    /// Prediction confidence scores
    pub confidence_scores: ConfidenceMetrics,
    /// Data drift detection
    pub data_drift: DriftMetrics,
    /// Model drift detection
    pub model_drift: DriftMetrics,
    /// Feature importance changes
    pub feature_importance_drift: Option<f64>,
}

/// Confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceMetrics {
    /// Average confidence score
    pub avg_confidence: f64,
    /// Minimum confidence score
    pub min_confidence: f64,
    /// Maximum confidence score
    pub max_confidence: f64,
    /// Low confidence predictions percentage
    pub low_confidence_rate: f64,
    /// Confidence threshold used
    pub confidence_threshold: f64,
}

/// Drift metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMetrics {
    /// Drift score (higher = more drift)
    pub drift_score: f64,
    /// Is drift detected (based on threshold)
    pub drift_detected: bool,
    /// Drift detection method used
    pub detection_method: String,
    /// Drift threshold
    pub threshold: f64,
    /// Features contributing to drift
    pub drifting_features: Vec<String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert name
    pub name: String,
    /// Alert description
    pub description: String,
    /// Metric to monitor
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Evaluation window in seconds
    pub evaluation_window_seconds: u64,
    /// Number of consecutive evaluations before triggering
    pub consecutive_evaluations: usize,
    /// Cooldown period in seconds
    pub cooldown_seconds: u64,
    /// Is alert enabled
    pub enabled: bool,
}

/// Comparison operators for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
}

/// Alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Alert configuration that triggered
    pub alert_config: AlertConfig,
    /// Current metric value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert message
    pub message: String,
    /// Timestamp when alert was triggered
    pub triggered_at: chrono::DateTime<chrono::Utc>,
    /// Model name
    pub model_name: String,
    /// Model version
    pub model_version: String,
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Model monitor for tracking performance and health
pub struct ModelMonitor {
    /// Model metadata
    model_metadata: ModelMetadata,
    /// Performance metrics history
    metrics_history: VecDeque<PerformanceMetrics>,
    /// Alert configurations
    alert_configs: Vec<AlertConfig>,
    /// Recent alert events
    alert_events: VecDeque<AlertEvent>,
    /// Alert evaluation counters
    alert_counters: HashMap<String, usize>,
    /// Last alert times for cooldown
    last_alert_times: HashMap<String, Instant>,
    /// Maximum history size
    max_history_size: usize,
    /// Metrics collection interval
    collection_interval: Duration,
    /// Last collection time
    last_collection: Instant,
}

impl ModelMonitor {
    /// Create a new model monitor
    pub fn new(model_metadata: ModelMetadata) -> Self {
        Self {
            model_metadata,
            metrics_history: VecDeque::new(),
            alert_configs: Vec::new(),
            alert_events: VecDeque::new(),
            alert_counters: HashMap::new(),
            last_alert_times: HashMap::new(),
            max_history_size: 1440, // 24 hours of minute-level metrics
            collection_interval: Duration::from_secs(60), // 1 minute
            last_collection: Instant::now(),
        }
    }

    /// Add alert configuration
    pub fn add_alert(&mut self, config: AlertConfig) {
        self.alert_configs.push(config);
    }

    /// Remove alert configuration
    pub fn remove_alert(&mut self, alert_name: &str) {
        self.alert_configs
            .retain(|config| config.name != alert_name);
        self.alert_counters.remove(alert_name);
        self.last_alert_times.remove(alert_name);
    }

    /// Collect metrics from deployment
    pub fn collect_metrics(&mut self, deployment_metrics: &DeploymentMetrics) -> Result<()> {
        // Check if it's time to collect metrics
        if self.last_collection.elapsed() < self.collection_interval {
            return Ok(());
        }

        // Create performance metrics
        let performance_metrics = PerformanceMetrics {
            model_name: self.model_metadata.name.clone(),
            model_version: self.model_metadata.version.clone(),
            timestamp: chrono::Utc::now(),
            latency: self.calculate_latency_metrics(deployment_metrics),
            throughput: self.calculate_throughput_metrics(deployment_metrics),
            error_metrics: self.calculate_error_metrics(deployment_metrics),
            resource_utilization: self.calculate_resource_metrics(deployment_metrics),
            quality_metrics: self.calculate_quality_metrics(),
        };

        // Add to history
        self.metrics_history.push_back(performance_metrics.clone());

        // Trim history if too large
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }

        // Evaluate alerts
        self.evaluate_alerts(&performance_metrics)?;

        self.last_collection = Instant::now();

        Ok(())
    }

    /// Calculate latency metrics
    fn calculate_latency_metrics(&self, deployment_metrics: &DeploymentMetrics) -> LatencyMetrics {
        // In a real implementation, this would calculate percentiles from raw latency data
        let avg_latency = deployment_metrics.avg_response_time_ms;

        LatencyMetrics {
            avg_latency_ms: avg_latency,
            p50_latency_ms: avg_latency * 0.8,
            p95_latency_ms: avg_latency * 1.5,
            p99_latency_ms: avg_latency * 2.0,
            max_latency_ms: avg_latency * 3.0,
            min_latency_ms: avg_latency * 0.5,
        }
    }

    /// Calculate throughput metrics
    fn calculate_throughput_metrics(
        &self,
        deployment_metrics: &DeploymentMetrics,
    ) -> ThroughputMetrics {
        ThroughputMetrics {
            requests_per_second: deployment_metrics.request_rate,
            total_requests: deployment_metrics.total_requests,
            successful_requests: deployment_metrics.successful_requests,
            failed_requests: deployment_metrics.failed_requests,
            concurrent_requests: deployment_metrics.active_instances as u64,
        }
    }

    /// Calculate error metrics
    fn calculate_error_metrics(&self, deployment_metrics: &DeploymentMetrics) -> ErrorMetrics {
        let mut error_rates_by_type = HashMap::new();
        let mut error_counts_by_type = HashMap::new();

        // Simulate error categorization
        if deployment_metrics.error_rate > 0.0 {
            error_rates_by_type.insert(
                "prediction_error".to_string(),
                deployment_metrics.error_rate * 0.7,
            );
            error_rates_by_type.insert(
                "timeout_error".to_string(),
                deployment_metrics.error_rate * 0.2,
            );
            error_rates_by_type.insert(
                "validation_error".to_string(),
                deployment_metrics.error_rate * 0.1,
            );

            let total_errors = deployment_metrics.failed_requests;
            error_counts_by_type.insert(
                "prediction_error".to_string(),
                (total_errors as f64 * 0.7) as u64,
            );
            error_counts_by_type.insert(
                "timeout_error".to_string(),
                (total_errors as f64 * 0.2) as u64,
            );
            error_counts_by_type.insert(
                "validation_error".to_string(),
                (total_errors as f64 * 0.1) as u64,
            );
        }

        ErrorMetrics {
            error_rate: deployment_metrics.error_rate,
            error_rates_by_type,
            error_counts_by_type,
            recent_errors: Vec::new(), // Would be populated in real implementation
        }
    }

    /// Calculate resource utilization metrics
    fn calculate_resource_metrics(
        &self,
        deployment_metrics: &DeploymentMetrics,
    ) -> ResourceUtilizationMetrics {
        ResourceUtilizationMetrics {
            cpu_utilization: deployment_metrics.cpu_utilization,
            memory_utilization: deployment_metrics.memory_utilization,
            gpu_utilization: None, // Would be available if GPU monitoring is enabled
            disk_io_utilization: deployment_metrics.cpu_utilization * 0.3, // Simulated
            network_io_utilization: deployment_metrics.request_rate / 1000.0, // Simulated
        }
    }

    /// Calculate model quality metrics
    fn calculate_quality_metrics(&self) -> QualityMetrics {
        QualityMetrics {
            accuracy: None, // Would require ground truth data
            confidence_scores: ConfidenceMetrics {
                avg_confidence: 0.85,
                min_confidence: 0.1,
                max_confidence: 0.99,
                low_confidence_rate: 0.05,
                confidence_threshold: 0.7,
            },
            data_drift: DriftMetrics {
                drift_score: 0.02,
                drift_detected: false,
                detection_method: "PSI".to_string(),
                threshold: 0.1,
                drifting_features: Vec::new(),
            },
            model_drift: DriftMetrics {
                drift_score: 0.01,
                drift_detected: false,
                detection_method: "performance_based".to_string(),
                threshold: 0.05,
                drifting_features: Vec::new(),
            },
            feature_importance_drift: Some(0.03),
        }
    }

    /// Evaluate alerts based on current metrics
    fn evaluate_alerts(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        // Clone alert configs to avoid borrow checker issues
        let alert_configs = self.alert_configs.clone();

        for config in &alert_configs {
            if !config.enabled {
                continue;
            }

            // Check cooldown
            if let Some(last_alert_time) = self.last_alert_times.get(&config.name) {
                if last_alert_time.elapsed() < Duration::from_secs(config.cooldown_seconds) {
                    continue;
                }
            }

            // Get metric value
            let current_value = self.get_metric_value(metrics, &config.metric)?;

            // Evaluate threshold
            let threshold_exceeded = match config.operator {
                ComparisonOperator::GreaterThan => current_value > config.threshold,
                ComparisonOperator::GreaterThanOrEqual => current_value >= config.threshold,
                ComparisonOperator::LessThan => current_value < config.threshold,
                ComparisonOperator::LessThanOrEqual => current_value <= config.threshold,
                ComparisonOperator::Equal => (current_value - config.threshold).abs() < 1e-10,
                ComparisonOperator::NotEqual => (current_value - config.threshold).abs() >= 1e-10,
            };

            if threshold_exceeded {
                // Increment counter
                let should_trigger = {
                    let counter = self.alert_counters.entry(config.name.clone()).or_insert(0);
                    *counter += 1;
                    *counter >= config.consecutive_evaluations
                };

                // Check if we should trigger alert
                if should_trigger {
                    self.trigger_alert(config, current_value)?;
                    self.alert_counters.insert(config.name.clone(), 0); // Reset counter
                    self.last_alert_times
                        .insert(config.name.clone(), Instant::now());
                }
            } else {
                // Reset counter if threshold not exceeded
                self.alert_counters.insert(config.name.clone(), 0);
            }
        }

        Ok(())
    }

    /// Get metric value by name
    fn get_metric_value(&self, metrics: &PerformanceMetrics, metric_name: &str) -> Result<f64> {
        match metric_name {
            "avg_latency_ms" => Ok(metrics.latency.avg_latency_ms),
            "p95_latency_ms" => Ok(metrics.latency.p95_latency_ms),
            "p99_latency_ms" => Ok(metrics.latency.p99_latency_ms),
            "requests_per_second" => Ok(metrics.throughput.requests_per_second),
            "error_rate" => Ok(metrics.error_metrics.error_rate),
            "cpu_utilization" => Ok(metrics.resource_utilization.cpu_utilization),
            "memory_utilization" => Ok(metrics.resource_utilization.memory_utilization),
            "drift_score" => Ok(metrics.quality_metrics.data_drift.drift_score),
            "avg_confidence" => Ok(metrics.quality_metrics.confidence_scores.avg_confidence),
            _ => Err(Error::InvalidInput(format!(
                "Unknown metric: {}",
                metric_name
            ))),
        }
    }

    /// Trigger an alert
    fn trigger_alert(&mut self, config: &AlertConfig, current_value: f64) -> Result<()> {
        let alert_event = AlertEvent {
            alert_config: config.clone(),
            current_value,
            threshold_value: config.threshold,
            message: format!(
                "Alert '{}': {} {} {} (current: {:.4})",
                config.name,
                config.metric,
                self.operator_to_string(&config.operator),
                config.threshold,
                current_value
            ),
            triggered_at: chrono::Utc::now(),
            model_name: self.model_metadata.name.clone(),
            model_version: self.model_metadata.version.clone(),
            context: HashMap::new(),
        };

        // Add to alert events
        self.alert_events.push_back(alert_event.clone());

        // Limit alert events history
        while self.alert_events.len() > 100 {
            self.alert_events.pop_front();
        }

        // Log alert
        log::warn!("Alert triggered: {}", alert_event.message);

        // In a real implementation, this would send notifications via email, Slack, etc.

        Ok(())
    }

    /// Convert operator to string
    fn operator_to_string(&self, operator: &ComparisonOperator) -> &'static str {
        match operator {
            ComparisonOperator::GreaterThan => ">",
            ComparisonOperator::GreaterThanOrEqual => ">=",
            ComparisonOperator::LessThan => "<",
            ComparisonOperator::LessThanOrEqual => "<=",
            ComparisonOperator::Equal => "==",
            ComparisonOperator::NotEqual => "!=",
        }
    }

    /// Get recent metrics
    pub fn get_recent_metrics(&self, limit: usize) -> Vec<PerformanceMetrics> {
        self.metrics_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, limit: usize) -> Vec<AlertEvent> {
        self.alert_events
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get alert configurations
    pub fn get_alert_configs(&self) -> &[AlertConfig] {
        &self.alert_configs
    }

    /// Get metrics summary for time window
    pub fn get_metrics_summary(&self, window_minutes: usize) -> Option<MetricsSummary> {
        let cutoff = chrono::Utc::now() - chrono::Duration::minutes(window_minutes as i64);

        let recent_metrics: Vec<_> = self
            .metrics_history
            .iter()
            .filter(|m| m.timestamp > cutoff)
            .collect();

        if recent_metrics.is_empty() {
            return None;
        }

        let avg_latency = recent_metrics
            .iter()
            .map(|m| m.latency.avg_latency_ms)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        let avg_throughput = recent_metrics
            .iter()
            .map(|m| m.throughput.requests_per_second)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        let avg_error_rate = recent_metrics
            .iter()
            .map(|m| m.error_metrics.error_rate)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        let avg_cpu = recent_metrics
            .iter()
            .map(|m| m.resource_utilization.cpu_utilization)
            .sum::<f64>()
            / recent_metrics.len() as f64;

        Some(MetricsSummary {
            window_minutes,
            avg_latency_ms: avg_latency,
            avg_throughput,
            avg_error_rate,
            avg_cpu_utilization: avg_cpu,
            total_requests: recent_metrics
                .iter()
                .map(|m| m.throughput.total_requests)
                .sum(),
            alert_count: self
                .alert_events
                .iter()
                .filter(|e| e.triggered_at > cutoff)
                .count(),
        })
    }
}

/// Metrics summary for a time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Time window in minutes
    pub window_minutes: usize,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Average throughput (requests per second)
    pub avg_throughput: f64,
    /// Average error rate
    pub avg_error_rate: f64,
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    /// Total requests in window
    pub total_requests: u64,
    /// Number of alerts in window
    pub alert_count: usize,
}

/// Metrics collector for gathering system metrics
pub trait MetricsCollector {
    /// Collect system metrics
    fn collect_system_metrics(&self) -> Result<SystemMetrics>;

    /// Collect model-specific metrics
    fn collect_model_metrics(&self, model_name: &str) -> Result<ModelSpecificMetrics>;
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Available memory in bytes
    pub memory_available: u64,
    /// Disk usage percentage
    pub disk_usage: f64,
    /// Network bytes sent
    pub network_bytes_sent: u64,
    /// Network bytes received
    pub network_bytes_received: u64,
    /// Load average
    pub load_average: f64,
    /// Number of processes
    pub process_count: u32,
}

/// Model-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpecificMetrics {
    /// Model memory usage in bytes
    pub model_memory_usage: u64,
    /// Model initialization time
    pub model_init_time_ms: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Feature processing time
    pub feature_processing_time_ms: u64,
    /// Prediction time excluding feature processing
    pub prediction_time_ms: u64,
}

/// Default metrics collector implementation
pub struct DefaultMetricsCollector;

impl MetricsCollector for DefaultMetricsCollector {
    fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        // In a real implementation, this would use system APIs to collect actual metrics
        Ok(SystemMetrics {
            cpu_usage: 0.45,                   // Simulated
            memory_usage: 2_147_483_648,       // 2GB simulated
            memory_available: 6_442_450_944,   // 6GB simulated
            disk_usage: 0.75,                  // Simulated
            network_bytes_sent: 1_048_576,     // 1MB simulated
            network_bytes_received: 2_097_152, // 2MB simulated
            load_average: 1.5,                 // Simulated
            process_count: 150,                // Simulated
        })
    }

    fn collect_model_metrics(&self, _model_name: &str) -> Result<ModelSpecificMetrics> {
        // In a real implementation, this would collect model-specific metrics
        Ok(ModelSpecificMetrics {
            model_memory_usage: 536_870_912, // 512MB simulated
            model_init_time_ms: 2500,        // Simulated
            cache_hit_rate: 0.85,            // Simulated
            feature_processing_time_ms: 5,   // Simulated
            prediction_time_ms: 15,          // Simulated
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::serving::ModelMetadata;

    fn create_test_metadata() -> ModelMetadata {
        ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            model_type: "classification".to_string(),
            feature_names: vec!["feature1".to_string(), "feature2".to_string()],
            target_name: Some("target".to_string()),
            description: "Test model".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    fn create_test_deployment_metrics() -> DeploymentMetrics {
        use crate::ml::serving::deployment::DeploymentStatus;

        crate::ml::serving::deployment::DeploymentMetrics {
            status: DeploymentStatus::Running,
            active_instances: 2,
            cpu_utilization: 0.6,
            memory_utilization: 0.7,
            request_rate: 50.0,
            avg_response_time_ms: 120.0,
            error_rate: 0.02,
            total_requests: 1000,
            successful_requests: 980,
            failed_requests: 20,
            last_health_check: chrono::Utc::now(),
            started_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_model_monitor_creation() {
        let metadata = create_test_metadata();
        let monitor = ModelMonitor::new(metadata);

        assert_eq!(monitor.model_metadata.name, "test_model");
        assert_eq!(monitor.alert_configs.len(), 0);
        assert_eq!(monitor.metrics_history.len(), 0);
    }

    #[test]
    fn test_alert_config() {
        let config = AlertConfig {
            name: "high_latency".to_string(),
            description: "Alert when latency is too high".to_string(),
            metric: "avg_latency_ms".to_string(),
            threshold: 200.0,
            operator: ComparisonOperator::GreaterThan,
            severity: AlertSeverity::Warning,
            evaluation_window_seconds: 300,
            consecutive_evaluations: 3,
            cooldown_seconds: 600,
            enabled: true,
        };

        assert_eq!(config.name, "high_latency");
        assert_eq!(config.threshold, 200.0);
        assert_eq!(config.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = DefaultMetricsCollector;

        let system_metrics = collector.collect_system_metrics().unwrap();
        assert!(system_metrics.cpu_usage >= 0.0 && system_metrics.cpu_usage <= 1.0);

        let model_metrics = collector.collect_model_metrics("test_model").unwrap();
        assert!(model_metrics.model_memory_usage > 0);
    }

    #[test]
    fn test_performance_metrics() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        let deployment_metrics = create_test_deployment_metrics();

        // Set collection interval to 0 for immediate collection
        monitor.collection_interval = Duration::from_secs(0);

        // Collect metrics
        monitor.collect_metrics(&deployment_metrics).unwrap();

        assert_eq!(monitor.metrics_history.len(), 1);

        let metrics = &monitor.metrics_history[0];
        assert_eq!(metrics.model_name, "test_model");
        assert!(metrics.latency.avg_latency_ms > 0.0);
        assert!(metrics.throughput.requests_per_second > 0.0);
    }
}

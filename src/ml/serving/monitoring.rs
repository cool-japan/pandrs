//! Model Monitoring Module
//!
//! This module provides comprehensive monitoring capabilities for deployed models including
//! performance metrics, drift detection, alerting, and observability.

use crate::core::error::{Error, Result};
use crate::ml::serving::deployment::RecordedError;
use crate::ml::serving::{DeploymentMetrics, ModelMetadata};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Bounded number of recent per-request latency samples kept for the HTTP-layer request buffer
/// (see [`ModelMonitor::record_request`]), independent of the `DeploymentMetrics`-driven path.
const MAX_HTTP_REQUEST_SAMPLES: usize = 2_000;

/// Bounded number of recent per-prediction confidence samples kept for
/// [`QualityMetrics::confidence_scores`].
const MAX_CONFIDENCE_SAMPLES: usize = 2_000;

/// PSI (Population Stability Index) drift threshold above which a feature is flagged as
/// drifting. `0.2` is the standard industry convention (PSI < 0.1: no meaningful shift;
/// 0.1-0.2: moderate; > 0.2: significant).
const PSI_DRIFT_THRESHOLD: f64 = 0.2;

/// Additive smoothing applied to PSI bin percentages, so an empty bin in either distribution
/// doesn't produce `ln(0)` or a division by zero.
const PSI_EPSILON: f64 = 1e-4;

/// Number of default equal-frequency bins used when a caller doesn't specify one explicitly.
const DEFAULT_PSI_BINS: usize = 10;

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

/// Resource utilization metrics.
///
/// All fields are `Option` and, in this build, always `None`: computing them honestly requires
/// real OS-level process telemetry (e.g. via a `sysinfo`-like crate), which this module
/// intentionally does not depend on (see [`DefaultMetricsCollector`]) rather than reporting a
/// synthesized proxy value as if it were measured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: Option<f64>,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: Option<f64>,
    /// GPU utilization (0.0 to 1.0, if available)
    pub gpu_utilization: Option<f64>,
    /// Disk I/O utilization (0.0 to 1.0)
    pub disk_io_utilization: Option<f64>,
    /// Network I/O utilization (0.0 to 1.0)
    pub network_io_utilization: Option<f64>,
}

/// Model quality metrics.
///
/// `confidence_scores`, `data_drift`, and `model_drift` are `Option`: each is `None` until this
/// monitor actually has the real data it needs (per-request confidence samples, a fitted drift
/// baseline, or ground-truth labels, respectively) -- never a fabricated constant standing in
/// for missing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Prediction accuracy (if ground truth is available)
    pub accuracy: Option<f64>,
    /// Prediction confidence scores, from real per-request samples (see
    /// [`ModelMonitor::record_prediction_confidence`]). `None` until at least one sample exists.
    pub confidence_scores: Option<ConfidenceMetrics>,
    /// Data drift detection: real PSI (Population Stability Index) against a stored reference
    /// distribution (see [`ModelMonitor::set_feature_baseline`]). `None` until a baseline has
    /// been set for at least one feature *and* at least one live observation has been recorded
    /// against it.
    pub data_drift: Option<DriftMetrics>,
    /// Model (performance) drift detection. Always `None` in this build: computing it for real
    /// requires a ground-truth feedback loop (comparing live prediction accuracy against a
    /// training-time baseline), which nothing currently feeds into this monitor.
    pub model_drift: Option<DriftMetrics>,
    /// Feature importance changes over time. Always `None` in this build: requires comparing
    /// live feature-importance to a stored baseline, which isn't tracked.
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

/// A fixed-edge histogram baseline for one feature, used to compute a real PSI drift score
/// against live traffic. Built once from a sample of reference (e.g. fit-time training) values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBaseline {
    /// Bin edges (`n_bins + 1` values, ascending; the first and last are unbounded). Value `v`
    /// falls in bin `i` when `edges[i] <= v < edges[i+1]`.
    edges: Vec<f64>,
    /// Reference (baseline) probability mass per bin; sums to 1.0 by construction
    /// (equal-frequency binning).
    reference_frequencies: Vec<f64>,
}

impl FeatureBaseline {
    /// Build a baseline histogram from reference samples using `n_bins` equal-frequency
    /// (quantile) buckets -- the standard convention for PSI baselines, since it guarantees a
    /// non-degenerate reference distribution regardless of the data's shape.
    ///
    /// Returns `None` when there are fewer finite values than bins (not enough data for a
    /// meaningful baseline) or `n_bins == 0`.
    pub fn from_samples(values: &[f64], n_bins: usize) -> Option<Self> {
        if n_bins == 0 {
            return None;
        }
        let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.len() < n_bins {
            return None;
        }
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut edges = Vec::with_capacity(n_bins + 1);
        edges.push(f64::NEG_INFINITY);
        for i in 1..n_bins {
            let pos = (i as f64 / n_bins as f64) * (sorted.len() - 1) as f64;
            let lower = pos.floor() as usize;
            let upper = pos.ceil() as usize;
            let frac = pos - lower as f64;
            edges.push(sorted[lower] * (1.0 - frac) + sorted[upper.min(sorted.len() - 1)] * frac);
        }
        edges.push(f64::INFINITY);

        Some(Self {
            edges,
            reference_frequencies: vec![1.0 / n_bins as f64; n_bins],
        })
    }

    /// Which bin `value` falls into, or `None` for a non-finite value.
    fn bin_index(&self, value: f64) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }
        for i in 0..self.edges.len().saturating_sub(1) {
            if value >= self.edges[i] && value < self.edges[i + 1] {
                return Some(i);
            }
        }
        Some(self.reference_frequencies.len().saturating_sub(1))
    }

    fn n_bins(&self) -> usize {
        self.reference_frequencies.len()
    }
}

/// Compute real latency order statistics (min/max/mean + nearest-rank percentiles) from a
/// bounded sample of per-request latencies, shared by both the `DeploymentMetrics`-driven
/// (`ModelMonitor::calculate_latency_metrics`) and HTTP-request-driven
/// (`ModelMonitor::record_request`) collection paths.
///
/// Callers must ensure `samples` is non-empty; percentiles use the nearest-rank method
/// (`ceil(p * n) - 1`, clamped to a valid index).
fn latency_metrics_from_samples(samples: &[u64]) -> LatencyMetrics {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();

    let percentile = |p: f64| -> f64 {
        let rank = ((p * n as f64).ceil() as usize).clamp(1, n) - 1;
        sorted[rank] as f64
    };

    let sum: u64 = sorted.iter().sum();
    LatencyMetrics {
        avg_latency_ms: sum as f64 / n as f64,
        p50_latency_ms: percentile(0.50),
        p95_latency_ms: percentile(0.95),
        p99_latency_ms: percentile(0.99),
        max_latency_ms: sorted[n - 1] as f64,
        min_latency_ms: sorted[0] as f64,
    }
}

/// Convert a deployment-recorded [`RecordedError`] into the public [`ErrorEvent`] shape.
fn recorded_error_to_event(recorded: &RecordedError) -> ErrorEvent {
    ErrorEvent {
        error_type: recorded.error_type.clone(),
        message: recorded.message.clone(),
        timestamp: recorded.occurred_at,
        context: None,
    }
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
    /// Last collection time. `None` means "no collection has happened yet on this monitor",
    /// distinct from "a collection happened a long time ago" -- a freshly-constructed monitor
    /// must never throttle its very first [`Self::collect_metrics`]/[`Self::record_request`]
    /// call just because it happens to be called soon after [`Self::new`].
    last_collection: Option<Instant>,
    /// Per-feature drift baselines, set via [`Self::set_feature_baseline`].
    feature_baselines: HashMap<String, FeatureBaseline>,
    /// Live observation counts per baseline bin, accumulated via [`Self::observe_feature`].
    live_feature_counts: HashMap<String, Vec<u64>>,
    /// Recent per-prediction confidence samples, fed via [`Self::record_prediction_confidence`].
    request_confidences: VecDeque<f64>,
    /// Recent per-HTTP-request latencies, fed via [`Self::record_request`] -- an alternate path
    /// to `collect_metrics`'s `DeploymentMetrics`-driven one, for callers (like
    /// `HttpModelServer`) that serve a raw model directly rather than through a `DeployedModel`.
    http_request_latencies_ms: VecDeque<u64>,
    /// Lifetime success count on the `record_request` path.
    http_success_count: u64,
    /// Lifetime error count on the `record_request` path.
    http_error_count: u64,
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
            last_collection: None,
            feature_baselines: HashMap::new(),
            live_feature_counts: HashMap::new(),
            request_confidences: VecDeque::new(),
            http_request_latencies_ms: VecDeque::new(),
            http_success_count: 0,
            http_error_count: 0,
        }
    }

    /// Establish a drift-detection baseline for one feature from a sample of reference values
    /// (e.g. the fit-time training distribution), using `n_bins` equal-frequency buckets.
    /// `DEFAULT_PSI_BINS` (10) is the conventional choice when the caller has no specific
    /// reason to pick another value.
    ///
    /// Until this has been called for at least one feature, [`Self::get_metrics_summary`] and
    /// every `PerformanceMetrics.quality_metrics.data_drift` this monitor produces stay `None`
    /// -- there is no fabricated drift score before a real baseline exists.
    pub fn set_feature_baseline(&mut self, feature_name: &str, reference_values: &[f64]) {
        self.set_feature_baseline_with_bins(feature_name, reference_values, DEFAULT_PSI_BINS);
    }

    /// As [`Self::set_feature_baseline`], with an explicit bin count.
    pub fn set_feature_baseline_with_bins(
        &mut self,
        feature_name: &str,
        reference_values: &[f64],
        n_bins: usize,
    ) {
        if let Some(baseline) = FeatureBaseline::from_samples(reference_values, n_bins) {
            let bins = baseline.n_bins();
            self.feature_baselines
                .insert(feature_name.to_string(), baseline);
            self.live_feature_counts
                .insert(feature_name.to_string(), vec![0; bins]);
        }
    }

    /// Bin one live observation of `feature_name` into its baseline histogram. A no-op for
    /// features without a configured baseline.
    pub fn observe_feature(&mut self, feature_name: &str, value: f64) {
        if let Some(baseline) = self.feature_baselines.get(feature_name) {
            if let Some(bin) = baseline.bin_index(value) {
                if let Some(counts) = self.live_feature_counts.get_mut(feature_name) {
                    if let Some(count) = counts.get_mut(bin) {
                        *count += 1;
                    }
                }
            }
        }
    }

    /// Observe every numeric value in a raw request payload against whichever features have a
    /// configured baseline (features without one are silently ignored).
    pub fn observe_features(&mut self, data: &HashMap<String, serde_json::Value>) {
        let names: Vec<String> = self.feature_baselines.keys().cloned().collect();
        for name in names {
            if let Some(value) = data.get(&name).and_then(|v| v.as_f64()) {
                self.observe_feature(&name, value);
            }
        }
    }

    /// Record one real per-prediction confidence score (e.g. the winning class probability),
    /// for [`QualityMetrics::confidence_scores`]. Until this is called at least once, that field
    /// stays `None`.
    pub fn record_prediction_confidence(&mut self, confidence: f64) {
        if confidence.is_finite() {
            self.request_confidences.push_back(confidence);
            while self.request_confidences.len() > MAX_CONFIDENCE_SAMPLES {
                self.request_confidences.pop_front();
            }
        }
    }

    /// Compute real PSI drift from every feature with both a baseline and at least one live
    /// observation. Returns `None` when no feature has both (never a fabricated constant).
    ///
    /// Public so a caller can check drift on demand (e.g. right after
    /// [`Self::set_feature_baseline`] and a handful of [`Self::observe_feature`] calls), without
    /// waiting on the next throttled [`Self::collect_metrics`]/[`Self::record_request`] snapshot.
    pub fn calculate_data_drift(&self) -> Option<DriftMetrics> {
        let mut per_feature_psi: Vec<(String, f64)> = Vec::new();

        for (name, baseline) in &self.feature_baselines {
            let counts = match self.live_feature_counts.get(name) {
                Some(c) => c,
                None => continue,
            };
            let total: u64 = counts.iter().sum();
            if total == 0 {
                continue;
            }
            let mut psi = 0.0;
            for (bin_idx, &ref_pct) in baseline.reference_frequencies.iter().enumerate() {
                let live_pct = counts.get(bin_idx).copied().unwrap_or(0) as f64 / total as f64;
                let ref_pct = ref_pct.max(PSI_EPSILON);
                let live_pct = live_pct.max(PSI_EPSILON);
                psi += (live_pct - ref_pct) * (live_pct / ref_pct).ln();
            }
            per_feature_psi.push((name.clone(), psi));
        }

        if per_feature_psi.is_empty() {
            return None;
        }

        let drift_score = per_feature_psi
            .iter()
            .map(|(_, psi)| *psi)
            .fold(f64::NEG_INFINITY, f64::max);
        let drifting_features: Vec<String> = per_feature_psi
            .into_iter()
            .filter(|(_, psi)| *psi > PSI_DRIFT_THRESHOLD)
            .map(|(name, _)| name)
            .collect();

        Some(DriftMetrics {
            drift_score,
            drift_detected: drift_score > PSI_DRIFT_THRESHOLD,
            detection_method: "PSI".to_string(),
            threshold: PSI_DRIFT_THRESHOLD,
            drifting_features,
        })
    }

    /// Compute real confidence statistics from recorded per-prediction samples, or `None` when
    /// no sample has been recorded yet.
    fn calculate_confidence_metrics(&self) -> Option<ConfidenceMetrics> {
        if self.request_confidences.is_empty() {
            return None;
        }
        const LOW_CONFIDENCE_THRESHOLD: f64 = 0.7;

        let n = self.request_confidences.len() as f64;
        let sum: f64 = self.request_confidences.iter().sum();
        let min = self
            .request_confidences
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .request_confidences
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let low_count = self
            .request_confidences
            .iter()
            .filter(|&&c| c < LOW_CONFIDENCE_THRESHOLD)
            .count();

        Some(ConfidenceMetrics {
            avg_confidence: sum / n,
            min_confidence: min,
            max_confidence: max,
            low_confidence_rate: low_count as f64 / n,
            confidence_threshold: LOW_CONFIDENCE_THRESHOLD,
        })
    }

    /// Feed one real HTTP-layer request outcome into this monitor, independent of
    /// `collect_metrics`'s `DeploymentMetrics`-driven path. Intended for servers (like
    /// `HttpModelServer`) that serve a registered model directly, without wrapping it in a
    /// `DeployedModel`.
    ///
    /// Throttled by the same `collection_interval` as `collect_metrics`. Returns whether a fresh
    /// snapshot was actually recorded this call (`false` when skipped due to the throttle), so
    /// callers can tell "no error, nothing new yet" apart from a real collection.
    pub fn record_request(&mut self, success: bool, latency_ms: u64) -> Result<bool> {
        self.http_request_latencies_ms.push_back(latency_ms);
        while self.http_request_latencies_ms.len() > MAX_HTTP_REQUEST_SAMPLES {
            self.http_request_latencies_ms.pop_front();
        }
        if success {
            self.http_success_count += 1;
        } else {
            self.http_error_count += 1;
        }

        if let Some(last) = self.last_collection {
            if last.elapsed() < self.collection_interval {
                return Ok(false);
            }
        }

        let samples: Vec<u64> = self.http_request_latencies_ms.iter().copied().collect();
        let latency = latency_metrics_from_samples(&samples);
        let total = self.http_success_count + self.http_error_count;
        let error_rate = if total > 0 {
            self.http_error_count as f64 / total as f64
        } else {
            0.0
        };

        let snapshot = PerformanceMetrics {
            model_name: self.model_metadata.name.clone(),
            model_version: self.model_metadata.version.clone(),
            timestamp: chrono::Utc::now(),
            latency,
            throughput: ThroughputMetrics {
                // Not tracked at this granularity on the HTTP-only path (no windowed rate
                // calculation here, unlike the DeploymentMetrics-driven path).
                requests_per_second: 0.0,
                total_requests: total,
                successful_requests: self.http_success_count,
                failed_requests: self.http_error_count,
                concurrent_requests: 0,
            },
            error_metrics: ErrorMetrics {
                error_rate,
                error_rates_by_type: HashMap::new(),
                error_counts_by_type: HashMap::new(),
                recent_errors: Vec::new(),
            },
            resource_utilization: ResourceUtilizationMetrics {
                cpu_utilization: None,
                memory_utilization: None,
                gpu_utilization: None,
                disk_io_utilization: None,
                network_io_utilization: None,
            },
            quality_metrics: self.calculate_quality_metrics(),
        };

        self.metrics_history.push_back(snapshot.clone());
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }
        self.evaluate_alerts(&snapshot)?;
        self.last_collection = Some(Instant::now());

        Ok(true)
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

    /// Collect metrics from a deployment's real, measured [`DeploymentMetrics`].
    ///
    /// Returns whether a fresh sample was actually taken: `Ok(false)` when the collection
    /// interval hasn't elapsed yet (previously this case was indistinguishable from "collected
    /// successfully" -- both returned a bare `Ok(())`).
    pub fn collect_metrics(&mut self, deployment_metrics: &DeploymentMetrics) -> Result<bool> {
        // Check if it's time to collect metrics (never throttles the very first collection --
        // see the `last_collection` field doc).
        if let Some(last) = self.last_collection {
            if last.elapsed() < self.collection_interval {
                return Ok(false);
            }
        }

        // Feed this deployment's real per-request feature payloads aren't available here (this
        // API only receives aggregate DeploymentMetrics); PSI drift observation happens via the
        // separate `observe_features` entry point, called by whoever has the raw request data.

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

        self.last_collection = Some(Instant::now());

        Ok(true)
    }

    /// Calculate latency metrics from real order statistics over
    /// `deployment_metrics.response_times_ms`, when available.
    ///
    /// Falls back to treating `avg_response_time_ms` as a degenerate single-point distribution
    /// (every percentile equal to the mean) only when no per-request samples have been recorded
    /// at all -- honest about having no distribution shape to report, rather than inventing one
    /// via fixed multiples of the mean (the previous `avg * 0.8`/`* 1.5`/`* 2.0`/`* 3.0` scheme,
    /// which had no relationship to the real data and could never reflect an actual spike).
    fn calculate_latency_metrics(&self, deployment_metrics: &DeploymentMetrics) -> LatencyMetrics {
        if deployment_metrics.response_times_ms.is_empty() {
            let avg_latency = deployment_metrics.avg_response_time_ms;
            return LatencyMetrics {
                avg_latency_ms: avg_latency,
                p50_latency_ms: avg_latency,
                p95_latency_ms: avg_latency,
                p99_latency_ms: avg_latency,
                max_latency_ms: avg_latency,
                min_latency_ms: avg_latency,
            };
        }
        latency_metrics_from_samples(&deployment_metrics.response_times_ms)
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

    /// Calculate error metrics from the deployment's real recorded [`RecordedError`]s.
    ///
    /// Previously this invented a fixed 70/20/10 "prediction_error"/"timeout_error"/
    /// "validation_error" split regardless of what actually failed -- which not only fabricated
    /// numbers but could actively misdirect an on-call responder toward the wrong root cause.
    fn calculate_error_metrics(&self, deployment_metrics: &DeploymentMetrics) -> ErrorMetrics {
        let mut error_counts_by_type: HashMap<String, u64> = HashMap::new();
        for recorded in &deployment_metrics.recent_errors {
            *error_counts_by_type
                .entry(recorded.error_type.clone())
                .or_insert(0) += 1;
        }

        let total_recorded: u64 = error_counts_by_type.values().sum();
        let error_rates_by_type: HashMap<String, f64> = if total_recorded > 0 {
            error_counts_by_type
                .iter()
                .map(|(k, &count)| {
                    (
                        k.clone(),
                        (count as f64 / total_recorded as f64) * deployment_metrics.error_rate,
                    )
                })
                .collect()
        } else {
            HashMap::new()
        };

        let recent_errors = deployment_metrics
            .recent_errors
            .iter()
            .map(|recorded| recorded_error_to_event(recorded))
            .collect();

        ErrorMetrics {
            error_rate: deployment_metrics.error_rate,
            error_rates_by_type,
            error_counts_by_type,
            recent_errors,
        }
    }

    /// Calculate resource utilization metrics.
    ///
    /// Passes through `DeploymentMetrics.cpu_utilization`/`memory_utilization`, which are always
    /// `None` in this build (see their field docs), and no longer derives
    /// `disk_io_utilization`/`network_io_utilization` from them via an arbitrary multiplier --
    /// those were fabricated proxies with no basis in real disk/network activity.
    fn calculate_resource_metrics(
        &self,
        deployment_metrics: &DeploymentMetrics,
    ) -> ResourceUtilizationMetrics {
        ResourceUtilizationMetrics {
            cpu_utilization: deployment_metrics.cpu_utilization,
            memory_utilization: deployment_metrics.memory_utilization,
            gpu_utilization: None, // Would be available if GPU monitoring is enabled
            disk_io_utilization: None,
            network_io_utilization: None,
        }
    }

    /// Calculate model quality metrics from real data only: each field that this monitor
    /// doesn't have real backing data for is honestly `None`, never a fabricated constant. See
    /// the field docs on [`QualityMetrics`] for exactly what each requires.
    fn calculate_quality_metrics(&self) -> QualityMetrics {
        QualityMetrics {
            accuracy: None, // Would require ground truth data
            confidence_scores: self.calculate_confidence_metrics(),
            data_drift: self.calculate_data_drift(),
            model_drift: None, // Requires a ground-truth feedback loop; not implemented.
            feature_importance_drift: None, // Requires a stored feature-importance baseline.
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

            // Get metric value; a metric with no real data yet (e.g. drift_score before a
            // baseline exists, or avg_confidence before any prediction has been recorded) is
            // skipped for this evaluation rather than treated as an error or a fabricated 0.0
            // that could spuriously trigger (or mask) an alert.
            let current_value = match self.get_metric_value(metrics, &config.metric)? {
                Some(value) => value,
                None => continue,
            };

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

    /// Get metric value by name.
    ///
    /// Returns `Ok(None)` (not an error, and not a fabricated `0.0`) for a metric that is
    /// structurally known but has no real value yet -- e.g. `drift_score` before a baseline
    /// exists, `avg_confidence` before any prediction has been recorded, or
    /// `cpu_utilization`/`memory_utilization`, which this build never populates (see their
    /// field docs). `Err` is reserved for a genuinely unknown metric name.
    fn get_metric_value(
        &self,
        metrics: &PerformanceMetrics,
        metric_name: &str,
    ) -> Result<Option<f64>> {
        match metric_name {
            "avg_latency_ms" => Ok(Some(metrics.latency.avg_latency_ms)),
            "p95_latency_ms" => Ok(Some(metrics.latency.p95_latency_ms)),
            "p99_latency_ms" => Ok(Some(metrics.latency.p99_latency_ms)),
            "requests_per_second" => Ok(Some(metrics.throughput.requests_per_second)),
            "error_rate" => Ok(Some(metrics.error_metrics.error_rate)),
            "cpu_utilization" => Ok(metrics.resource_utilization.cpu_utilization),
            "memory_utilization" => Ok(metrics.resource_utilization.memory_utilization),
            "drift_score" => Ok(metrics
                .quality_metrics
                .data_drift
                .as_ref()
                .map(|d| d.drift_score)),
            "avg_confidence" => Ok(metrics
                .quality_metrics
                .confidence_scores
                .as_ref()
                .map(|c| c.avg_confidence)),
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

        // Log at the level matching the alert's configured severity (previously every severity,
        // including Critical/Emergency, was logged at `warn!`). The `log` crate has no distinct
        // "emergency" level, so Critical and Emergency both map to `error!`.
        match config.severity {
            AlertSeverity::Info => log::info!("Alert triggered: {}", alert_event.message),
            AlertSeverity::Warning => log::warn!("Alert triggered: {}", alert_event.message),
            AlertSeverity::Critical | AlertSeverity::Emergency => {
                log::error!("Alert triggered: {}", alert_event.message)
            }
        }

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

        // `cpu_utilization` is always `None` in this build (see its field docs); average only
        // over snapshots that actually have a value, so this stays honestly `None` rather than
        // silently treating missing data as `0.0`.
        let cpu_samples: Vec<f64> = recent_metrics
            .iter()
            .filter_map(|m| m.resource_utilization.cpu_utilization)
            .collect();
        let avg_cpu_utilization = if cpu_samples.is_empty() {
            None
        } else {
            Some(cpu_samples.iter().sum::<f64>() / cpu_samples.len() as f64)
        };

        // `throughput.total_requests` is a *cumulative* lifetime counter, snapshotted at each
        // collection tick -- summing it across every snapshot in the window would multiply the
        // true total by however many snapshots happen to fall in the window. The most recent
        // (maximum) snapshot value already *is* the total-to-date.
        let total_requests = recent_metrics
            .iter()
            .map(|m| m.throughput.total_requests)
            .max()
            .unwrap_or(0);

        Some(MetricsSummary {
            window_minutes,
            avg_latency_ms: avg_latency,
            avg_throughput,
            avg_error_rate,
            avg_cpu_utilization,
            total_requests,
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
    /// Average CPU utilization, when real telemetry is available. Always `None` in this build
    /// (see [`ResourceUtilizationMetrics::cpu_utilization`]).
    pub avg_cpu_utilization: Option<f64>,
    /// Total requests in window
    pub total_requests: u64,
    /// Number of alerts in window
    pub alert_count: usize,
}

/// Metrics collector for gathering system metrics.
///
/// Requires `Send + Sync` so a collector can be shared (typically via `Arc`) with a concurrent
/// server, consistent with [`crate::ml::serving::ModelServing`] and
/// [`crate::ml::serving::registry::ModelRegistry`].
pub trait MetricsCollector: Send + Sync {
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

/// Default metrics collector implementation.
///
/// This is currently the *only* implementation of [`MetricsCollector`] in the codebase. It
/// honestly reports [`Error::NotImplemented`] rather than fabricated constants: real system
/// telemetry (CPU/memory/disk/network/process counts) requires an OS-level metrics dependency
/// (e.g. a `sysinfo`-like crate), which this module intentionally does not add per the pure-Rust
/// / minimal-dependency policy. A future implementation backed by such a crate (feature-gated,
/// off by default) could satisfy this trait for real; until then, callers should treat a `None`
/// [`crate::ml::serving::monitoring::ResourceUtilizationMetrics`] or an `Err` from this
/// collector as "not available in this build", not "zero load".
pub struct DefaultMetricsCollector;

impl MetricsCollector for DefaultMetricsCollector {
    fn collect_system_metrics(&self) -> Result<SystemMetrics> {
        Err(Error::NotImplemented(
            "System metrics require OS-level telemetry (e.g. a sysinfo-like crate), which this \
             build does not depend on; DefaultMetricsCollector cannot report real cpu/memory/\
             disk/network/process data"
                .to_string(),
        ))
    }

    fn collect_model_metrics(&self, _model_name: &str) -> Result<ModelSpecificMetrics> {
        Err(Error::NotImplemented(
            "Model-specific runtime metrics (memory usage, init time, cache hit rate) are not \
             tracked anywhere in this build; DefaultMetricsCollector cannot report them"
                .to_string(),
        ))
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
            cpu_utilization: None,
            memory_utilization: None,
            request_rate: 50.0,
            avg_response_time_ms: 120.0,
            error_rate: 0.02,
            total_requests: 1000,
            successful_requests: 980,
            failed_requests: 20,
            in_flight_requests: 0,
            response_times_ms: vec![100, 110, 120, 130, 140, 500],
            recent_errors: vec![RecordedError {
                error_type: "InvalidInput".to_string(),
                message: "missing feature 'x'".to_string(),
                occurred_at: chrono::Utc::now(),
            }],
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
    fn test_metrics_collector_is_honestly_not_implemented() {
        // DefaultMetricsCollector previously returned hardcoded "simulated" constants dressed up
        // as real measurements; this test now asserts the honest failure instead (this build has
        // no OS-level telemetry dependency to back these metrics with real data).
        let collector = DefaultMetricsCollector;

        assert!(collector.collect_system_metrics().is_err());
        assert!(collector.collect_model_metrics("test_model").is_err());
    }

    #[test]
    fn test_performance_metrics() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        let deployment_metrics = create_test_deployment_metrics();

        // Set collection interval to 0 for immediate collection
        monitor.collection_interval = Duration::from_secs(0);

        // Collect metrics; a fresh sample must be reported as taken.
        let sample_taken = monitor
            .collect_metrics(&deployment_metrics)
            .expect("operation should succeed");
        assert!(
            sample_taken,
            "collect_metrics must report a sample was taken"
        );

        assert_eq!(monitor.metrics_history.len(), 1);

        let metrics = &monitor.metrics_history[0];
        assert_eq!(metrics.model_name, "test_model");
        assert!(metrics.latency.avg_latency_ms > 0.0);
        assert!(metrics.throughput.requests_per_second > 0.0);

        // Real percentiles from the response_times_ms sample [100,110,120,130,140,500], not
        // fixed multiples of the mean: p99/max must reflect the real 500ms outlier.
        assert_eq!(metrics.latency.max_latency_ms, 500.0);
        assert!(metrics.latency.p99_latency_ms >= 140.0);

        // Real error categorization from the one recorded InvalidInput error, not a fabricated
        // 70/20/10 prediction/timeout/validation split.
        assert_eq!(
            metrics
                .error_metrics
                .error_counts_by_type
                .get("InvalidInput"),
            Some(&1)
        );
        assert!(!metrics
            .error_metrics
            .error_counts_by_type
            .contains_key("prediction_error"));
    }

    #[test]
    fn test_collect_metrics_reports_when_sample_is_skipped() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        // Default collection_interval is 60s, so back-to-back calls should skip the second.
        let deployment_metrics = create_test_deployment_metrics();

        let first = monitor
            .collect_metrics(&deployment_metrics)
            .expect("first collection succeeds");
        let second = monitor
            .collect_metrics(&deployment_metrics)
            .expect("second call does not error");

        assert!(first, "first call is not throttled");
        assert!(!second, "immediate second call must report no sample taken");
    }

    #[test]
    fn test_data_drift_is_none_until_baseline_exists() {
        let metadata = create_test_metadata();
        let monitor = ModelMonitor::new(metadata);
        assert!(monitor.calculate_data_drift().is_none());
    }

    #[test]
    fn test_data_drift_is_none_until_live_observations_exist() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        monitor.set_feature_baseline("x1", &(0..100).map(|i| i as f64).collect::<Vec<_>>());
        // A baseline exists, but nothing has been observed against it yet.
        assert!(monitor.calculate_data_drift().is_none());
    }

    #[test]
    fn test_real_psi_drift_detects_a_shifted_distribution() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);

        // Baseline: uniform over [0, 100).
        let reference: Vec<f64> = (0..1000).map(|i| (i % 100) as f64).collect();
        monitor.set_feature_baseline("x1", &reference);

        // Live traffic: every observation clustered at the extreme high end -- a severe shift.
        for _ in 0..200 {
            monitor.observe_feature("x1", 99.0);
        }

        let drift = monitor
            .calculate_data_drift()
            .expect("baseline + live observations exist");
        assert_eq!(drift.detection_method, "PSI");
        assert!(
            drift.drift_score > PSI_DRIFT_THRESHOLD,
            "a fully-shifted distribution must exceed the drift threshold, got {}",
            drift.drift_score
        );
        assert!(drift.drift_detected);
        assert!(drift.drifting_features.contains(&"x1".to_string()));
    }

    #[test]
    fn test_real_psi_drift_is_low_for_matching_distribution() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);

        let reference: Vec<f64> = (0..1000).map(|i| (i % 100) as f64).collect();
        monitor.set_feature_baseline("x1", &reference);

        // Live traffic drawn from the *same* distribution as the baseline.
        for i in 0..1000 {
            monitor.observe_feature("x1", (i % 100) as f64);
        }

        let drift = monitor
            .calculate_data_drift()
            .expect("baseline + live observations exist");
        assert!(
            drift.drift_score < PSI_DRIFT_THRESHOLD,
            "a matching distribution must not exceed the drift threshold, got {}",
            drift.drift_score
        );
        assert!(!drift.drift_detected);
    }

    #[test]
    fn test_confidence_metrics_none_until_recorded() {
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        assert!(monitor.calculate_confidence_metrics().is_none());

        monitor.record_prediction_confidence(0.9);
        monitor.record_prediction_confidence(0.5);
        let confidence = monitor
            .calculate_confidence_metrics()
            .expect("samples were recorded");
        assert!((confidence.avg_confidence - 0.7).abs() < 1e-9);
        assert_eq!(confidence.min_confidence, 0.5);
        assert_eq!(confidence.max_confidence, 0.9);
    }

    #[test]
    fn test_alert_severity_maps_to_log_level_without_panicking() {
        // trigger_alert dispatches to log::info!/warn!/error! based on severity; this just
        // exercises every variant to ensure the match is exhaustive and doesn't panic.
        let metadata = create_test_metadata();
        let mut monitor = ModelMonitor::new(metadata);
        for severity in [
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Critical,
            AlertSeverity::Emergency,
        ] {
            let config = AlertConfig {
                name: format!("{:?}", severity),
                description: "test".to_string(),
                metric: "avg_latency_ms".to_string(),
                threshold: 0.0,
                operator: ComparisonOperator::GreaterThanOrEqual,
                severity,
                evaluation_window_seconds: 60,
                consecutive_evaluations: 1,
                cooldown_seconds: 0,
                enabled: true,
            };
            monitor
                .trigger_alert(&config, 1.0)
                .expect("trigger_alert does not error");
        }
        assert_eq!(monitor.get_recent_alerts(10).len(), 4);
    }

    #[test]
    fn test_latency_percentiles_are_real_order_statistics_not_fixed_multiples_of_the_mean() {
        // Regression for the previous formula (`p50 = avg * 0.8`, `p95 = avg * 1.5`,
        // `p99 = avg * 2.0`, `max = avg * 3.0`): on a heavily skewed sample (90 fast requests at
        // 1ms, 10 slow outliers at 1000ms), that formula and real nearest-rank order statistics
        // disagree by nearly two orders of magnitude at the median, so this is a strong
        // discriminator between "real percentiles" and "fabricated multiples of the mean".
        let mut samples: Vec<u64> = vec![1; 90];
        samples.extend(std::iter::repeat(1000).take(10));
        let metrics = latency_metrics_from_samples(&samples);

        // avg = (90*1 + 10*1000) / 100 = 100.9
        assert!((metrics.avg_latency_ms - 100.9).abs() < 1e-9);

        // Real median: 90 of the 100 sorted samples are 1ms, so the 50th-percentile rank still
        // lands inside that block. The fabricated formula would have reported ~80.72ms instead.
        assert_eq!(
            metrics.p50_latency_ms, 1.0,
            "p50 must reflect the real bulk of fast requests, not a multiple of the mean"
        );
        assert!(metrics.p50_latency_ms < 10.0);

        // Real max is the true observed maximum (1000.0), not avg*3.0 (~302.7).
        assert_eq!(metrics.max_latency_ms, 1000.0);
        assert_eq!(metrics.min_latency_ms, 1.0);

        // p95/p99 fall in the slow-outlier block for this distribution shape -- real order
        // statistics, not avg*1.5 (~151.35) / avg*2.0 (~201.8).
        assert_eq!(metrics.p95_latency_ms, 1000.0);
        assert_eq!(metrics.p99_latency_ms, 1000.0);
    }

    #[test]
    fn test_get_metric_value_returns_none_for_unavailable_data_not_an_error() {
        let metadata = create_test_metadata();
        let monitor = ModelMonitor::new(metadata.clone());
        let deployment_metrics = create_test_deployment_metrics();
        let snapshot = PerformanceMetrics {
            model_name: metadata.name.clone(),
            model_version: metadata.version.clone(),
            timestamp: chrono::Utc::now(),
            latency: monitor.calculate_latency_metrics(&deployment_metrics),
            throughput: monitor.calculate_throughput_metrics(&deployment_metrics),
            error_metrics: monitor.calculate_error_metrics(&deployment_metrics),
            resource_utilization: monitor.calculate_resource_metrics(&deployment_metrics),
            quality_metrics: monitor.calculate_quality_metrics(),
        };

        // cpu_utilization/drift_score/avg_confidence are all unavailable before real data
        // exists: must be Ok(None), never an Err nor a fabricated 0.0.
        assert_eq!(
            monitor
                .get_metric_value(&snapshot, "cpu_utilization")
                .unwrap(),
            None
        );
        assert_eq!(
            monitor.get_metric_value(&snapshot, "drift_score").unwrap(),
            None
        );
        assert_eq!(
            monitor
                .get_metric_value(&snapshot, "avg_confidence")
                .unwrap(),
            None
        );
        // A genuinely unknown metric name is still a real error.
        assert!(monitor.get_metric_value(&snapshot, "not_a_metric").is_err());
    }
}

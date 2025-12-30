//! Real-Time Analytics Dashboard Module
//!
//! This module provides real-time monitoring and analytics capabilities for
//! PandRS operations, including:
//!
//! - Performance metrics collection and aggregation
//! - Query execution monitoring
//! - Memory usage tracking
//! - Operation throughput analysis
//! - Alerting and thresholds
//!
//! # Example
//!
//! ```ignore
//! use pandrs::analytics::{Dashboard, MetricsCollector, DashboardConfig};
//!
//! // Create a dashboard
//! let config = DashboardConfig::default();
//! let dashboard = Dashboard::new(config);
//!
//! // Start collecting metrics
//! dashboard.start();
//!
//! // Record operations
//! dashboard.record_operation("query", 150);
//!
//! // Get current metrics
//! let snapshot = dashboard.snapshot();
//! ```

pub mod alerts;
pub mod dashboard;
pub mod metrics;

pub use alerts::*;
pub use dashboard::*;
pub use metrics::*;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Metric types for categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Counter that only increases
    Counter,
    /// Gauge that can go up or down
    Gauge,
    /// Histogram for distribution tracking
    Histogram,
    /// Timer for duration measurements
    Timer,
}

/// Time resolution for aggregations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeResolution {
    /// Per-second aggregation
    Second,
    /// Per-minute aggregation
    Minute,
    /// Per-hour aggregation
    Hour,
    /// Per-day aggregation
    Day,
}

impl TimeResolution {
    /// Get duration for this resolution
    pub fn duration(&self) -> Duration {
        match self {
            TimeResolution::Second => Duration::from_secs(1),
            TimeResolution::Minute => Duration::from_secs(60),
            TimeResolution::Hour => Duration::from_secs(3600),
            TimeResolution::Day => Duration::from_secs(86400),
        }
    }
}

/// A single metric value with timestamp
#[derive(Debug, Clone)]
pub struct MetricValue {
    /// Timestamp when recorded
    pub timestamp: Instant,
    /// The value
    pub value: f64,
    /// Optional labels
    pub labels: HashMap<String, String>,
}

/// Aggregated metric statistics
#[derive(Debug, Clone, Default)]
pub struct MetricStats {
    /// Number of samples
    pub count: u64,
    /// Sum of all values
    pub sum: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Variance
    pub variance: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Percentile 50 (median)
    pub p50: f64,
    /// Percentile 95
    pub p95: f64,
    /// Percentile 99
    pub p99: f64,
}

impl MetricStats {
    /// Create from a list of values
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);

        let variance = if count > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };

        let std_dev = variance.sqrt();

        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        MetricStats {
            count,
            sum,
            min,
            max,
            mean,
            variance,
            std_dev,
            p50,
            p95,
            p99,
        }
    }
}

/// Calculate percentile from sorted values using linear interpolation
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }

    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;

    if lower == upper {
        return sorted[lower];
    }

    let weight = idx - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Operation categories for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationCategory {
    /// Query operations
    Query,
    /// Write operations
    Write,
    /// Read operations
    Read,
    /// Aggregation operations
    Aggregation,
    /// Join operations
    Join,
    /// Sort operations
    Sort,
    /// Filter operations
    Filter,
    /// GroupBy operations
    GroupBy,
    /// I/O operations
    IO,
    /// Memory operations
    Memory,
    /// Other/Unknown
    Other,
}

impl std::fmt::Display for OperationCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationCategory::Query => write!(f, "query"),
            OperationCategory::Write => write!(f, "write"),
            OperationCategory::Read => write!(f, "read"),
            OperationCategory::Aggregation => write!(f, "aggregation"),
            OperationCategory::Join => write!(f, "join"),
            OperationCategory::Sort => write!(f, "sort"),
            OperationCategory::Filter => write!(f, "filter"),
            OperationCategory::GroupBy => write!(f, "groupby"),
            OperationCategory::IO => write!(f, "io"),
            OperationCategory::Memory => write!(f, "memory"),
            OperationCategory::Other => write!(f, "other"),
        }
    }
}

/// Operation timing record
#[derive(Debug, Clone)]
pub struct OperationRecord {
    /// Operation name
    pub name: String,
    /// Operation category
    pub category: OperationCategory,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Rows processed (if applicable)
    pub rows_processed: Option<usize>,
    /// Bytes processed (if applicable)
    pub bytes_processed: Option<usize>,
    /// Timestamp
    pub timestamp: Instant,
    /// Whether operation succeeded
    pub success: bool,
    /// Optional error message
    pub error: Option<String>,
}

/// System resource snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Memory used in bytes
    pub memory_used: usize,
    /// Memory available in bytes
    pub memory_available: usize,
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Active thread count
    pub thread_count: usize,
    /// Open file handles
    pub open_files: usize,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl Default for ResourceSnapshot {
    fn default() -> Self {
        Self {
            memory_used: 0,
            memory_available: 0,
            cpu_usage: 0.0,
            thread_count: 0,
            open_files: 0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Retention period for detailed metrics
    pub retention_period: Duration,
    /// Aggregation interval
    pub aggregation_interval: Duration,
    /// Maximum metrics to retain
    pub max_metrics: usize,
    /// Enable alerting
    pub alerting_enabled: bool,
    /// Sample rate (1.0 = 100%)
    pub sample_rate: f64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        DashboardConfig {
            enabled: true,
            retention_period: Duration::from_secs(3600), // 1 hour
            aggregation_interval: Duration::from_secs(60), // 1 minute
            max_metrics: 100_000,
            alerting_enabled: true,
            sample_rate: 1.0,
        }
    }
}

/// Dashboard snapshot containing current metrics
#[derive(Debug, Clone)]
pub struct DashboardSnapshot {
    /// Timestamp of snapshot
    pub timestamp: SystemTime,
    /// Uptime duration
    pub uptime: Duration,
    /// Total operations
    pub total_operations: u64,
    /// Operations per second (recent)
    pub ops_per_second: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// P99 latency in microseconds
    pub p99_latency_us: f64,
    /// Error rate (0-1)
    pub error_rate: f64,
    /// Total rows processed
    pub total_rows: u64,
    /// Rows per second
    pub rows_per_second: f64,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Bytes per second
    pub bytes_per_second: f64,
    /// Per-category statistics
    pub category_stats: HashMap<OperationCategory, MetricStats>,
    /// Resource snapshot
    pub resources: ResourceSnapshot,
    /// Active alerts
    pub active_alerts: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = MetricStats::from_values(&values);

        assert_eq!(stats.count, 10);
        assert_eq!(stats.sum, 55.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!((stats.mean - 5.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((percentile(&sorted, 50.0) - 5.0).abs() < 1.0);
        assert!((percentile(&sorted, 90.0) - 9.0).abs() < 1.0);
    }

    #[test]
    fn test_time_resolution() {
        assert_eq!(TimeResolution::Second.duration(), Duration::from_secs(1));
        assert_eq!(TimeResolution::Minute.duration(), Duration::from_secs(60));
        assert_eq!(TimeResolution::Hour.duration(), Duration::from_secs(3600));
    }

    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardConfig::default();
        assert!(config.enabled);
        assert!(config.alerting_enabled);
        assert_eq!(config.sample_rate, 1.0);
    }
}

//! Real-Time Analytics Dashboard
//!
//! Provides a central dashboard for monitoring PandRS operations.

use super::{
    DashboardConfig, DashboardSnapshot, MetricStats, MetricsCollector, OperationCategory,
    OperationRecord, ResourceSnapshot,
};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// The main analytics dashboard
#[derive(Debug)]
pub struct Dashboard {
    /// Configuration
    config: DashboardConfig,
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    /// Operation records
    operations: RwLock<VecDeque<OperationRecord>>,
    /// Start time
    start_time: Instant,
    /// Running flag
    running: AtomicBool,
    /// Total operations counter
    total_ops: AtomicU64,
    /// Total errors counter
    total_errors: AtomicU64,
    /// Total rows processed
    total_rows: AtomicU64,
    /// Total bytes processed
    total_bytes: AtomicU64,
    /// Per-category counters
    category_counters: RwLock<HashMap<OperationCategory, AtomicU64>>,
    /// Active alerts
    active_alerts: RwLock<Vec<String>>,
    /// Last snapshot time
    last_snapshot: RwLock<Instant>,
}

impl Dashboard {
    /// Create a new dashboard with the given configuration
    pub fn new(config: DashboardConfig) -> Self {
        Dashboard {
            config,
            metrics: Arc::new(MetricsCollector::new()),
            operations: RwLock::new(VecDeque::with_capacity(10000)),
            start_time: Instant::now(),
            running: AtomicBool::new(false),
            total_ops: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_rows: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            category_counters: RwLock::new(HashMap::new()),
            active_alerts: RwLock::new(Vec::new()),
            last_snapshot: RwLock::new(Instant::now()),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DashboardConfig::default())
    }

    /// Start the dashboard
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Stop the dashboard
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the metrics collector
    pub fn metrics(&self) -> &Arc<MetricsCollector> {
        &self.metrics
    }

    /// Record an operation
    pub fn record_operation(
        &self,
        name: &str,
        category: OperationCategory,
        duration_us: u64,
        rows: Option<usize>,
        bytes: Option<usize>,
        success: bool,
        error: Option<String>,
    ) {
        if !self.config.enabled {
            return;
        }

        // Sample check
        if self.config.sample_rate < 1.0 {
            let random = (Instant::now().elapsed().as_nanos() % 1000) as f64 / 1000.0;
            if random > self.config.sample_rate {
                return;
            }
        }

        let record = OperationRecord {
            name: name.to_string(),
            category,
            duration_us,
            rows_processed: rows,
            bytes_processed: bytes,
            timestamp: Instant::now(),
            success,
            error,
        };

        // Update counters
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        if !success {
            self.total_errors.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(r) = rows {
            self.total_rows.fetch_add(r as u64, Ordering::Relaxed);
        }

        if let Some(b) = bytes {
            self.total_bytes.fetch_add(b as u64, Ordering::Relaxed);
        }

        // Update category counter
        if let Ok(counters) = self.category_counters.write() {
            // Note: We need interior mutability here
            // In practice, you'd use dashmap or similar
        }

        // Record to metrics collector
        self.metrics.record_duration(
            &format!("{}.duration", category),
            Duration::from_micros(duration_us),
        );

        if success {
            self.metrics.increment(&format!("{}.success", category));
        } else {
            self.metrics.increment(&format!("{}.error", category));
        }

        // Store operation record
        if let Ok(mut ops) = self.operations.write() {
            ops.push_back(record);

            // Enforce max metrics limit
            while ops.len() > self.config.max_metrics {
                ops.pop_front();
            }

            // Enforce retention period
            let cutoff = Instant::now() - self.config.retention_period;
            while ops.front().map(|r| r.timestamp < cutoff).unwrap_or(false) {
                ops.pop_front();
            }
        }
    }

    /// Record a simple operation with duration
    pub fn record_simple(&self, name: &str, duration_us: u64) {
        self.record_operation(
            name,
            OperationCategory::Other,
            duration_us,
            None,
            None,
            true,
            None,
        );
    }

    /// Time an operation
    pub fn time<F, R>(&self, name: &str, category: OperationCategory, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        self.record_operation(
            name,
            category,
            duration.as_micros() as u64,
            None,
            None,
            true,
            None,
        );

        result
    }

    /// Get operations per second (recent)
    pub fn ops_per_second(&self) -> f64 {
        let total = self.total_ops.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        let total = self.total_ops.load(Ordering::Relaxed);
        let errors = self.total_errors.load(Ordering::Relaxed);

        if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get average latency in microseconds
    pub fn avg_latency_us(&self) -> f64 {
        self.operations
            .read()
            .map(|ops| {
                if ops.is_empty() {
                    return 0.0;
                }
                let sum: u64 = ops.iter().map(|r| r.duration_us).sum();
                sum as f64 / ops.len() as f64
            })
            .unwrap_or(0.0)
    }

    /// Get P99 latency in microseconds
    pub fn p99_latency_us(&self) -> f64 {
        self.operations
            .read()
            .map(|ops| {
                if ops.is_empty() {
                    return 0.0;
                }

                let mut durations: Vec<u64> = ops.iter().map(|r| r.duration_us).collect();
                durations.sort();

                let idx = (0.99 * (durations.len() - 1) as f64).round() as usize;
                durations[idx.min(durations.len() - 1)] as f64
            })
            .unwrap_or(0.0)
    }

    /// Get rows per second
    pub fn rows_per_second(&self) -> f64 {
        let total = self.total_rows.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get bytes per second
    pub fn bytes_per_second(&self) -> f64 {
        let total = self.total_bytes.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get statistics for a category
    pub fn category_stats(&self, category: OperationCategory) -> MetricStats {
        self.operations
            .read()
            .map(|ops| {
                let values: Vec<f64> = ops
                    .iter()
                    .filter(|r| r.category == category)
                    .map(|r| r.duration_us as f64)
                    .collect();
                MetricStats::from_values(&values)
            })
            .unwrap_or_default()
    }

    /// Get all category statistics
    pub fn all_category_stats(&self) -> HashMap<OperationCategory, MetricStats> {
        let mut result = HashMap::new();

        for category in [
            OperationCategory::Query,
            OperationCategory::Write,
            OperationCategory::Read,
            OperationCategory::Aggregation,
            OperationCategory::Join,
            OperationCategory::Sort,
            OperationCategory::Filter,
            OperationCategory::GroupBy,
            OperationCategory::IO,
            OperationCategory::Memory,
            OperationCategory::Other,
        ] {
            let stats = self.category_stats(category);
            if stats.count > 0 {
                result.insert(category, stats);
            }
        }

        result
    }

    /// Get current resource snapshot
    pub fn resource_snapshot(&self) -> ResourceSnapshot {
        // In a real implementation, this would query system resources
        ResourceSnapshot {
            memory_used: 0,
            memory_available: 0,
            cpu_usage: 0.0,
            thread_count: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            open_files: 0,
            timestamp: SystemTime::now(),
        }
    }

    /// Add an alert
    pub fn add_alert(&self, message: String) {
        if let Ok(mut alerts) = self.active_alerts.write() {
            alerts.push(message);
        }
    }

    /// Clear an alert
    pub fn clear_alert(&self, message: &str) {
        if let Ok(mut alerts) = self.active_alerts.write() {
            alerts.retain(|a| a != message);
        }
    }

    /// Get active alerts
    pub fn alerts(&self) -> Vec<String> {
        self.active_alerts
            .read()
            .map(|a| a.clone())
            .unwrap_or_default()
    }

    /// Get a snapshot of the dashboard state
    pub fn snapshot(&self) -> DashboardSnapshot {
        DashboardSnapshot {
            timestamp: SystemTime::now(),
            uptime: self.start_time.elapsed(),
            total_operations: self.total_ops.load(Ordering::Relaxed),
            ops_per_second: self.ops_per_second(),
            avg_latency_us: self.avg_latency_us(),
            p99_latency_us: self.p99_latency_us(),
            error_rate: self.error_rate(),
            total_rows: self.total_rows.load(Ordering::Relaxed),
            rows_per_second: self.rows_per_second(),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            bytes_per_second: self.bytes_per_second(),
            category_stats: self.all_category_stats(),
            resources: self.resource_snapshot(),
            active_alerts: self.alerts(),
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_ops.store(0, Ordering::Relaxed);
        self.total_errors.store(0, Ordering::Relaxed);
        self.total_rows.store(0, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);

        if let Ok(mut ops) = self.operations.write() {
            ops.clear();
        }

        if let Ok(mut alerts) = self.active_alerts.write() {
            alerts.clear();
        }

        self.metrics.clear();
    }

    /// Get recent operations
    pub fn recent_operations(&self, limit: usize) -> Vec<OperationRecord> {
        self.operations
            .read()
            .map(|ops| ops.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Get slowest operations
    pub fn slowest_operations(&self, limit: usize) -> Vec<OperationRecord> {
        self.operations
            .read()
            .map(|ops| {
                let mut sorted: Vec<_> = ops.iter().cloned().collect();
                sorted.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
                sorted.into_iter().take(limit).collect()
            })
            .unwrap_or_default()
    }

    /// Get failed operations
    pub fn failed_operations(&self, limit: usize) -> Vec<OperationRecord> {
        self.operations
            .read()
            .map(|ops| {
                ops.iter()
                    .filter(|r| !r.success)
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Default for Dashboard {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Global dashboard instance
static GLOBAL_DASHBOARD: std::sync::OnceLock<Dashboard> = std::sync::OnceLock::new();

/// Initialize the global dashboard
pub fn init_global_dashboard(config: DashboardConfig) {
    let _ = GLOBAL_DASHBOARD.set(Dashboard::new(config));
}

/// Get the global dashboard
pub fn global_dashboard() -> &'static Dashboard {
    GLOBAL_DASHBOARD.get_or_init(Dashboard::default)
}

/// Record an operation to the global dashboard
pub fn record_global(name: &str, category: OperationCategory, duration_us: u64) {
    global_dashboard().record_operation(name, category, duration_us, None, None, true, None);
}

/// Time an operation with the global dashboard
pub fn time_global<F, R>(name: &str, category: OperationCategory, f: F) -> R
where
    F: FnOnce() -> R,
{
    global_dashboard().time(name, category, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = Dashboard::default();
        assert!(!dashboard.is_running());

        dashboard.start();
        assert!(dashboard.is_running());

        dashboard.stop();
        assert!(!dashboard.is_running());
    }

    #[test]
    fn test_record_operation() {
        let dashboard = Dashboard::default();
        dashboard.start();

        dashboard.record_operation(
            "test_query",
            OperationCategory::Query,
            1000,
            Some(100),
            Some(1024),
            true,
            None,
        );

        let snapshot = dashboard.snapshot();
        assert_eq!(snapshot.total_operations, 1);
        assert_eq!(snapshot.total_rows, 100);
        assert_eq!(snapshot.total_bytes, 1024);
    }

    #[test]
    fn test_time_operation() {
        let dashboard = Dashboard::default();

        let result = dashboard.time("test_op", OperationCategory::Other, || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(dashboard.avg_latency_us() >= 10000.0); // At least 10ms
    }

    #[test]
    fn test_error_tracking() {
        let dashboard = Dashboard::default();

        // Record success
        dashboard.record_operation("op1", OperationCategory::Query, 100, None, None, true, None);

        // Record failure
        dashboard.record_operation(
            "op2",
            OperationCategory::Query,
            100,
            None,
            None,
            false,
            Some("test error".to_string()),
        );

        assert!((dashboard.error_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_category_stats() {
        let dashboard = Dashboard::default();

        for i in 0..10 {
            dashboard.record_operation(
                "query",
                OperationCategory::Query,
                (i + 1) * 100,
                None,
                None,
                true,
                None,
            );
        }

        let stats = dashboard.category_stats(OperationCategory::Query);
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 100.0);
        assert_eq!(stats.max, 1000.0);
    }

    #[test]
    fn test_slowest_operations() {
        let dashboard = Dashboard::default();

        dashboard.record_simple("fast", 100);
        dashboard.record_simple("medium", 500);
        dashboard.record_simple("slow", 1000);

        let slowest = dashboard.slowest_operations(2);
        assert_eq!(slowest.len(), 2);
        assert_eq!(slowest[0].duration_us, 1000);
        assert_eq!(slowest[1].duration_us, 500);
    }

    #[test]
    fn test_alerts() {
        let dashboard = Dashboard::default();

        dashboard.add_alert("High latency detected".to_string());
        dashboard.add_alert("Memory usage critical".to_string());

        let alerts = dashboard.alerts();
        assert_eq!(alerts.len(), 2);

        dashboard.clear_alert("High latency detected");
        assert_eq!(dashboard.alerts().len(), 1);
    }

    #[test]
    fn test_reset() {
        let dashboard = Dashboard::default();

        for _ in 0..100 {
            dashboard.record_simple("op", 100);
        }

        assert_eq!(dashboard.snapshot().total_operations, 100);

        dashboard.reset();

        assert_eq!(dashboard.snapshot().total_operations, 0);
    }

    #[test]
    fn test_global_dashboard() {
        let dashboard = global_dashboard();
        dashboard.start();

        record_global("global_op", OperationCategory::Query, 100);

        assert!(dashboard.snapshot().total_operations >= 1);
    }
}

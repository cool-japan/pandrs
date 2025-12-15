//! Metrics Collection and Aggregation
//!
//! This module provides efficient metrics collection with support for
//! counters, gauges, histograms, and timers.

use super::{MetricStats, MetricType, MetricValue, TimeResolution};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A single metric with history
#[derive(Debug)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Recent values (ring buffer)
    values: RwLock<VecDeque<MetricValue>>,
    /// Maximum values to retain
    max_values: usize,
    /// Current value (for gauges/counters)
    current: AtomicU64,
    /// Labels for this metric
    labels: HashMap<String, String>,
}

impl Metric {
    /// Create a new metric
    pub fn new(name: impl Into<String>, metric_type: MetricType) -> Self {
        Metric {
            name: name.into(),
            metric_type,
            values: RwLock::new(VecDeque::with_capacity(1000)),
            max_values: 10000,
            current: AtomicU64::new(0),
            labels: HashMap::new(),
        }
    }

    /// Create with labels
    pub fn with_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Record a value
    pub fn record(&self, value: f64) {
        self.record_with_labels(value, HashMap::new());
    }

    /// Record a value with labels
    pub fn record_with_labels(&self, value: f64, labels: HashMap<String, String>) {
        let metric_value = MetricValue {
            timestamp: Instant::now(),
            value,
            labels,
        };

        if let Ok(mut values) = self.values.write() {
            values.push_back(metric_value);
            while values.len() > self.max_values {
                values.pop_front();
            }
        }

        // Update current value based on type
        match self.metric_type {
            MetricType::Counter => {
                self.current.fetch_add(value as u64, Ordering::Relaxed);
            }
            MetricType::Gauge => {
                self.current.store(value.to_bits(), Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Increment counter
    pub fn increment(&self) {
        self.increment_by(1);
    }

    /// Increment counter by amount
    pub fn increment_by(&self, amount: u64) {
        self.record(amount as f64);
    }

    /// Set gauge value
    pub fn set(&self, value: f64) {
        self.current.store(value.to_bits(), Ordering::Relaxed);
        self.record(value);
    }

    /// Get current value
    pub fn current(&self) -> f64 {
        match self.metric_type {
            MetricType::Gauge => f64::from_bits(self.current.load(Ordering::Relaxed)),
            _ => self.current.load(Ordering::Relaxed) as f64,
        }
    }

    /// Get values in time window
    pub fn values_in_window(&self, window: Duration) -> Vec<f64> {
        let cutoff = Instant::now() - window;

        self.values
            .read()
            .map(|values| {
                values
                    .iter()
                    .filter(|v| v.timestamp >= cutoff)
                    .map(|v| v.value)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get statistics for time window
    pub fn stats_in_window(&self, window: Duration) -> MetricStats {
        let values = self.values_in_window(window);
        MetricStats::from_values(&values)
    }

    /// Get all statistics
    pub fn stats(&self) -> MetricStats {
        self.values
            .read()
            .map(|values| {
                let vals: Vec<f64> = values.iter().map(|v| v.value).collect();
                MetricStats::from_values(&vals)
            })
            .unwrap_or_default()
    }

    /// Clear all values
    pub fn clear(&self) {
        if let Ok(mut values) = self.values.write() {
            values.clear();
        }
        self.current.store(0, Ordering::Relaxed);
    }

    /// Get value count
    pub fn count(&self) -> usize {
        self.values.read().map(|v| v.len()).unwrap_or(0)
    }
}

/// Metrics collector managing multiple metrics
#[derive(Debug)]
pub struct MetricsCollector {
    /// Registered metrics
    metrics: RwLock<HashMap<String, Arc<Metric>>>,
    /// Global labels applied to all metrics
    global_labels: HashMap<String, String>,
    /// Whether collection is enabled
    enabled: bool,
    /// Start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        MetricsCollector {
            metrics: RwLock::new(HashMap::new()),
            global_labels: HashMap::new(),
            enabled: true,
            start_time: Instant::now(),
        }
    }

    /// Create with global labels
    pub fn with_global_labels(mut self, labels: HashMap<String, String>) -> Self {
        self.global_labels = labels;
        self
    }

    /// Enable or disable collection
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Register a new metric
    pub fn register(&self, name: &str, metric_type: MetricType) -> Arc<Metric> {
        let metric = Arc::new(Metric::new(name, metric_type));

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.insert(name.to_string(), Arc::clone(&metric));
        }

        metric
    }

    /// Get or create a counter
    pub fn counter(&self, name: &str) -> Arc<Metric> {
        self.get_or_create(name, MetricType::Counter)
    }

    /// Get or create a gauge
    pub fn gauge(&self, name: &str) -> Arc<Metric> {
        self.get_or_create(name, MetricType::Gauge)
    }

    /// Get or create a histogram
    pub fn histogram(&self, name: &str) -> Arc<Metric> {
        self.get_or_create(name, MetricType::Histogram)
    }

    /// Get or create a timer
    pub fn timer(&self, name: &str) -> Arc<Metric> {
        self.get_or_create(name, MetricType::Timer)
    }

    /// Get or create metric
    fn get_or_create(&self, name: &str, metric_type: MetricType) -> Arc<Metric> {
        // Try to get existing
        if let Ok(metrics) = self.metrics.read() {
            if let Some(metric) = metrics.get(name) {
                return Arc::clone(metric);
            }
        }

        // Create new
        self.register(name, metric_type)
    }

    /// Get a metric by name
    pub fn get(&self, name: &str) -> Option<Arc<Metric>> {
        self.metrics.read().ok().and_then(|m| m.get(name).cloned())
    }

    /// Record a value to a metric
    pub fn record(&self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }

        if let Some(metric) = self.get(name) {
            metric.record(value);
        }
    }

    /// Increment a counter
    pub fn increment(&self, name: &str) {
        self.increment_by(name, 1);
    }

    /// Increment a counter by amount
    pub fn increment_by(&self, name: &str, amount: u64) {
        if !self.enabled {
            return;
        }

        let metric = self.counter(name);
        metric.increment_by(amount);
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        if !self.enabled {
            return;
        }

        let metric = self.gauge(name);
        metric.set(value);
    }

    /// Record a duration
    pub fn record_duration(&self, name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }

        let metric = self.timer(name);
        metric.record(duration.as_micros() as f64);
    }

    /// Time a closure and record the duration
    pub fn time<F, R>(&self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        self.record_duration(name, start.elapsed());
        result
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics
            .read()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Get snapshot of all metrics
    pub fn snapshot(&self) -> HashMap<String, MetricStats> {
        self.metrics
            .read()
            .map(|metrics| {
                metrics
                    .iter()
                    .map(|(name, metric)| (name.clone(), metric.stats()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Clear all metrics
    pub fn clear(&self) {
        if let Ok(metrics) = self.metrics.read() {
            for metric in metrics.values() {
                metric.clear();
            }
        }
    }

    /// Remove a metric
    pub fn remove(&self, name: &str) -> Option<Arc<Metric>> {
        self.metrics.write().ok().and_then(|mut m| m.remove(name))
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// A scoped timer that records duration on drop
pub struct ScopedTimer<'a> {
    metric: &'a Metric,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    /// Create a new scoped timer
    pub fn new(metric: &'a Metric) -> Self {
        ScopedTimer {
            metric,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.metric.record(duration.as_micros() as f64);
    }
}

/// Rate calculator for computing rates over time
#[derive(Debug)]
pub struct RateCalculator {
    /// Previous count
    prev_count: AtomicU64,
    /// Previous timestamp
    prev_time: RwLock<Instant>,
    /// Smoothing factor for EMA (0-1)
    smoothing: f64,
    /// Current rate (EMA smoothed)
    current_rate: RwLock<f64>,
}

impl RateCalculator {
    /// Create a new rate calculator
    pub fn new(smoothing: f64) -> Self {
        RateCalculator {
            prev_count: AtomicU64::new(0),
            prev_time: RwLock::new(Instant::now()),
            smoothing: smoothing.clamp(0.0, 1.0),
            current_rate: RwLock::new(0.0),
        }
    }

    /// Update with new count and get rate
    pub fn update(&self, count: u64) -> f64 {
        let now = Instant::now();

        let (prev_count, elapsed) = {
            let prev_time = self.prev_time.read().unwrap();
            let elapsed = now.duration_since(*prev_time);
            (self.prev_count.load(Ordering::Relaxed), elapsed)
        };

        if elapsed.as_secs_f64() > 0.0 {
            let delta = count.saturating_sub(prev_count) as f64;
            let instant_rate = delta / elapsed.as_secs_f64();

            // EMA smoothing
            let mut current = self.current_rate.write().unwrap();
            *current = self.smoothing * instant_rate + (1.0 - self.smoothing) * *current;

            // Update previous values
            self.prev_count.store(count, Ordering::Relaxed);
            if let Ok(mut prev_time) = self.prev_time.write() {
                *prev_time = now;
            }

            *current
        } else {
            *self.current_rate.read().unwrap()
        }
    }

    /// Get current rate
    pub fn rate(&self) -> f64 {
        *self.current_rate.read().unwrap()
    }

    /// Reset the calculator
    pub fn reset(&self) {
        self.prev_count.store(0, Ordering::Relaxed);
        if let Ok(mut prev_time) = self.prev_time.write() {
            *prev_time = Instant::now();
        }
        if let Ok(mut rate) = self.current_rate.write() {
            *rate = 0.0;
        }
    }
}

impl Default for RateCalculator {
    fn default() -> Self {
        Self::new(0.3) // Default smoothing factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_counter() {
        let metric = Metric::new("test_counter", MetricType::Counter);

        metric.increment();
        metric.increment();
        metric.increment();

        assert_eq!(metric.current(), 3.0);
        assert_eq!(metric.count(), 3);
    }

    #[test]
    fn test_metric_gauge() {
        let metric = Metric::new("test_gauge", MetricType::Gauge);

        metric.set(42.0);
        assert_eq!(metric.current(), 42.0);

        metric.set(100.0);
        assert_eq!(metric.current(), 100.0);
    }

    #[test]
    fn test_metric_stats() {
        let metric = Metric::new("test_histogram", MetricType::Histogram);

        for i in 1..=10 {
            metric.record(i as f64);
        }

        let stats = metric.stats();
        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        collector.increment("requests");
        collector.increment("requests");
        collector.set_gauge("memory", 1024.0);

        let requests = collector.get("requests").unwrap();
        assert_eq!(requests.current(), 2.0);

        let memory = collector.get("memory").unwrap();
        assert_eq!(memory.current(), 1024.0);
    }

    #[test]
    fn test_collector_time() {
        let collector = MetricsCollector::new();

        let result = collector.time("operation", || {
            std::thread::sleep(Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);

        let timer = collector.get("operation").unwrap();
        let stats = timer.stats();
        assert!(stats.mean >= 10000.0); // At least 10ms in microseconds
    }

    #[test]
    fn test_rate_calculator() {
        let calc = RateCalculator::new(1.0); // No smoothing

        // Initial rate is 0
        assert_eq!(calc.rate(), 0.0);

        // After some counts
        std::thread::sleep(Duration::from_millis(100));
        let rate = calc.update(100);

        // Rate should be approximately 1000/sec (100 in 100ms)
        assert!(rate > 0.0);
    }

    #[test]
    fn test_metric_window() {
        let metric = Metric::new("windowed", MetricType::Histogram);

        for i in 0..100 {
            metric.record(i as f64);
        }

        let values = metric.values_in_window(Duration::from_secs(60));
        assert_eq!(values.len(), 100);

        let stats = metric.stats_in_window(Duration::from_secs(60));
        assert_eq!(stats.count, 100);
    }

    #[test]
    fn test_collector_snapshot() {
        let collector = MetricsCollector::new();

        collector.increment("counter1");
        collector.increment("counter2");
        collector.set_gauge("gauge1", 50.0);

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.len(), 3);
    }
}

//! Alerting System for Analytics Dashboard
//!
//! Provides configurable alerts based on metric thresholds.

use super::{Dashboard, MetricStats, OperationCategory};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning level
    Warning,
    /// Critical level
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Comparison operators for thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdOperator {
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Equal
    Equal,
    /// Not equal
    NotEqual,
}

impl ThresholdOperator {
    /// Evaluate the operator
    pub fn evaluate(&self, value: f64, threshold: f64) -> bool {
        match self {
            ThresholdOperator::GreaterThan => value > threshold,
            ThresholdOperator::GreaterThanOrEqual => value >= threshold,
            ThresholdOperator::LessThan => value < threshold,
            ThresholdOperator::LessThanOrEqual => value <= threshold,
            ThresholdOperator::Equal => (value - threshold).abs() < f64::EPSILON,
            ThresholdOperator::NotEqual => (value - threshold).abs() >= f64::EPSILON,
        }
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Description
    pub description: String,
    /// Metric to monitor
    pub metric: AlertMetric,
    /// Threshold operator
    pub operator: ThresholdOperator,
    /// Threshold value
    pub threshold: f64,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Minimum duration before alerting (debounce)
    pub duration: Duration,
    /// Whether the rule is enabled
    pub enabled: bool,
    /// Labels/tags
    pub labels: HashMap<String, String>,
}

impl AlertRule {
    /// Create a new alert rule
    pub fn new(name: impl Into<String>, metric: AlertMetric) -> Self {
        AlertRule {
            name: name.into(),
            description: String::new(),
            metric,
            operator: ThresholdOperator::GreaterThan,
            threshold: 0.0,
            severity: AlertSeverity::Warning,
            duration: Duration::from_secs(0),
            enabled: true,
            labels: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set operator and threshold
    pub fn when(mut self, operator: ThresholdOperator, threshold: f64) -> Self {
        self.operator = operator;
        self.threshold = threshold;
        self
    }

    /// Set severity
    pub fn with_severity(mut self, severity: AlertSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set duration
    pub fn for_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Add a label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Enable/disable the rule
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Metrics that can be monitored for alerts
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertMetric {
    /// Error rate (0-1)
    ErrorRate,
    /// Average latency in microseconds
    AvgLatency,
    /// P99 latency in microseconds
    P99Latency,
    /// Operations per second
    OpsPerSecond,
    /// Rows per second
    RowsPerSecond,
    /// Bytes per second
    BytesPerSecond,
    /// Category-specific latency
    CategoryLatency(OperationCategory),
    /// Category-specific error rate
    CategoryErrorRate(OperationCategory),
    /// Custom metric by name
    Custom(String),
}

/// An active alert instance
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// The rule that triggered this alert
    pub rule_name: String,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// Threshold value
    pub threshold: f64,
    /// Severity
    pub severity: AlertSeverity,
    /// When the alert first triggered
    pub triggered_at: Instant,
    /// Alert message
    pub message: String,
}

impl ActiveAlert {
    /// Format as a string
    pub fn format(&self) -> String {
        format!(
            "[{}] {}: {} (current: {:.2}, threshold: {:.2})",
            self.severity, self.rule_name, self.message, self.current_value, self.threshold
        )
    }
}

/// Alert state for tracking firing duration
#[derive(Debug)]
struct AlertState {
    /// When the condition first became true
    first_triggered: Option<Instant>,
    /// Whether alert is currently firing
    firing: bool,
}

/// Alert manager for processing rules
#[derive(Debug)]
pub struct AlertManager {
    /// Configured rules
    rules: RwLock<Vec<AlertRule>>,
    /// Current state per rule
    states: RwLock<HashMap<String, AlertState>>,
    /// Active alerts
    active: RwLock<Vec<ActiveAlert>>,
    /// Alert handlers
    handlers: RwLock<Vec<Box<dyn AlertHandler>>>,
    /// Whether alerting is enabled
    enabled: bool,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new() -> Self {
        AlertManager {
            rules: RwLock::new(Vec::new()),
            states: RwLock::new(HashMap::new()),
            active: RwLock::new(Vec::new()),
            handlers: RwLock::new(Vec::new()),
            enabled: true,
        }
    }

    /// Add an alert rule
    pub fn add_rule(&self, rule: AlertRule) {
        if let Ok(mut rules) = self.rules.write() {
            if let Ok(mut states) = self.states.write() {
                states.insert(
                    rule.name.clone(),
                    AlertState {
                        first_triggered: None,
                        firing: false,
                    },
                );
            }
            rules.push(rule);
        }
    }

    /// Remove a rule by name
    pub fn remove_rule(&self, name: &str) {
        if let Ok(mut rules) = self.rules.write() {
            rules.retain(|r| r.name != name);
        }
        if let Ok(mut states) = self.states.write() {
            states.remove(name);
        }
    }

    /// Add an alert handler
    pub fn add_handler(&self, handler: Box<dyn AlertHandler>) {
        if let Ok(mut handlers) = self.handlers.write() {
            handlers.push(handler);
        }
    }

    /// Evaluate all rules against current metrics
    pub fn evaluate(&self, dashboard: &Dashboard) {
        if !self.enabled {
            return;
        }

        let rules = match self.rules.read() {
            Ok(r) => r.clone(),
            Err(_) => return,
        };

        for rule in rules.iter().filter(|r| r.enabled) {
            let value = self.get_metric_value(dashboard, &rule.metric);

            if let Some(value) = value {
                self.evaluate_rule(rule, value);
            }
        }
    }

    /// Get the current value of a metric
    fn get_metric_value(&self, dashboard: &Dashboard, metric: &AlertMetric) -> Option<f64> {
        match metric {
            AlertMetric::ErrorRate => Some(dashboard.error_rate()),
            AlertMetric::AvgLatency => Some(dashboard.avg_latency_us()),
            AlertMetric::P99Latency => Some(dashboard.p99_latency_us()),
            AlertMetric::OpsPerSecond => Some(dashboard.ops_per_second()),
            AlertMetric::RowsPerSecond => Some(dashboard.rows_per_second()),
            AlertMetric::BytesPerSecond => Some(dashboard.bytes_per_second()),
            AlertMetric::CategoryLatency(cat) => {
                let stats = dashboard.category_stats(*cat);
                if stats.count > 0 {
                    Some(stats.mean)
                } else {
                    None
                }
            }
            AlertMetric::CategoryErrorRate(_cat) => {
                // Would need to track per-category error rates
                None
            }
            AlertMetric::Custom(name) => dashboard.metrics().get(name).map(|m| m.current()),
        }
    }

    /// Evaluate a single rule
    fn evaluate_rule(&self, rule: &AlertRule, value: f64) {
        let condition_met = rule.operator.evaluate(value, rule.threshold);

        let mut states = match self.states.write() {
            Ok(s) => s,
            Err(_) => return,
        };

        let state = states.entry(rule.name.clone()).or_insert(AlertState {
            first_triggered: None,
            firing: false,
        });

        if condition_met {
            let now = Instant::now();

            if state.first_triggered.is_none() {
                state.first_triggered = Some(now);
            }

            // Check if duration threshold is met
            let duration_met = state
                .first_triggered
                .map(|t| now.duration_since(t) >= rule.duration)
                .unwrap_or(false);

            if duration_met && !state.firing {
                // Fire the alert
                state.firing = true;
                drop(states); // Release lock before calling handlers

                let alert = ActiveAlert {
                    rule_name: rule.name.clone(),
                    current_value: value,
                    threshold: rule.threshold,
                    severity: rule.severity,
                    triggered_at: now,
                    message: rule.description.clone(),
                };

                self.fire_alert(alert);
            }
        } else {
            // Condition no longer met
            if state.firing {
                // Resolve the alert
                state.firing = false;
                let rule_name = rule.name.clone();
                drop(states);
                self.resolve_alert(&rule_name);
            } else {
                state.first_triggered = None;
            }
        }
    }

    /// Fire an alert
    fn fire_alert(&self, alert: ActiveAlert) {
        // Add to active alerts
        if let Ok(mut active) = self.active.write() {
            active.push(alert.clone());
        }

        // Call handlers
        if let Ok(handlers) = self.handlers.read() {
            for handler in handlers.iter() {
                handler.on_alert(&alert);
            }
        }
    }

    /// Resolve an alert
    fn resolve_alert(&self, rule_name: &str) {
        // Remove from active alerts
        if let Ok(mut active) = self.active.write() {
            active.retain(|a| a.rule_name != rule_name);
        }

        // Call handlers
        if let Ok(handlers) = self.handlers.read() {
            for handler in handlers.iter() {
                handler.on_resolve(rule_name);
            }
        }
    }

    /// Get all active alerts
    pub fn active_alerts(&self) -> Vec<ActiveAlert> {
        self.active.read().map(|a| a.clone()).unwrap_or_default()
    }

    /// Get count of active alerts by severity
    pub fn alert_counts(&self) -> HashMap<AlertSeverity, usize> {
        let mut counts = HashMap::new();
        counts.insert(AlertSeverity::Info, 0);
        counts.insert(AlertSeverity::Warning, 0);
        counts.insert(AlertSeverity::Critical, 0);

        if let Ok(active) = self.active.read() {
            for alert in active.iter() {
                *counts.entry(alert.severity).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Check if there are any critical alerts
    pub fn has_critical(&self) -> bool {
        self.active
            .read()
            .map(|a| {
                a.iter()
                    .any(|alert| alert.severity == AlertSeverity::Critical)
            })
            .unwrap_or(false)
    }

    /// Clear all active alerts
    pub fn clear_all(&self) {
        if let Ok(mut active) = self.active.write() {
            active.clear();
        }
        if let Ok(mut states) = self.states.write() {
            for state in states.values_mut() {
                state.firing = false;
                state.first_triggered = None;
            }
        }
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for alert handlers
pub trait AlertHandler: Send + Sync + std::fmt::Debug {
    /// Called when an alert fires
    fn on_alert(&self, alert: &ActiveAlert);

    /// Called when an alert is resolved
    fn on_resolve(&self, rule_name: &str);
}

/// Simple logging alert handler
#[derive(Debug)]
pub struct LoggingAlertHandler {
    /// Prefix for log messages
    prefix: String,
}

impl LoggingAlertHandler {
    /// Create a new logging handler
    pub fn new(prefix: impl Into<String>) -> Self {
        LoggingAlertHandler {
            prefix: prefix.into(),
        }
    }
}

impl AlertHandler for LoggingAlertHandler {
    fn on_alert(&self, alert: &ActiveAlert) {
        log::warn!("{} ALERT: {}", self.prefix, alert.format());
    }

    fn on_resolve(&self, rule_name: &str) {
        log::info!("{} RESOLVED: {}", self.prefix, rule_name);
    }
}

/// Create common alert rules
pub fn create_default_rules() -> Vec<AlertRule> {
    vec![
        AlertRule::new("high_error_rate", AlertMetric::ErrorRate)
            .with_description("Error rate is above 5%")
            .when(ThresholdOperator::GreaterThan, 0.05)
            .with_severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(60)),
        AlertRule::new("critical_error_rate", AlertMetric::ErrorRate)
            .with_description("Error rate is above 20%")
            .when(ThresholdOperator::GreaterThan, 0.20)
            .with_severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(30)),
        AlertRule::new("high_latency", AlertMetric::P99Latency)
            .with_description("P99 latency exceeds 1 second")
            .when(ThresholdOperator::GreaterThan, 1_000_000.0)
            .with_severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(60)),
        AlertRule::new("very_high_latency", AlertMetric::P99Latency)
            .with_description("P99 latency exceeds 5 seconds")
            .when(ThresholdOperator::GreaterThan, 5_000_000.0)
            .with_severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(30)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_operators() {
        assert!(ThresholdOperator::GreaterThan.evaluate(10.0, 5.0));
        assert!(!ThresholdOperator::GreaterThan.evaluate(5.0, 10.0));

        assert!(ThresholdOperator::LessThan.evaluate(5.0, 10.0));
        assert!(!ThresholdOperator::LessThan.evaluate(10.0, 5.0));

        assert!(ThresholdOperator::GreaterThanOrEqual.evaluate(10.0, 10.0));
        assert!(ThresholdOperator::LessThanOrEqual.evaluate(10.0, 10.0));
    }

    #[test]
    fn test_alert_rule_builder() {
        let rule = AlertRule::new("test", AlertMetric::ErrorRate)
            .with_description("Test alert")
            .when(ThresholdOperator::GreaterThan, 0.1)
            .with_severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(60))
            .with_label("team", "backend");

        assert_eq!(rule.name, "test");
        assert_eq!(rule.threshold, 0.1);
        assert_eq!(rule.severity, AlertSeverity::Critical);
        assert_eq!(rule.duration, Duration::from_secs(60));
        assert_eq!(rule.labels.get("team"), Some(&"backend".to_string()));
    }

    #[test]
    fn test_alert_manager() {
        let manager = AlertManager::new();

        manager.add_rule(
            AlertRule::new("test_error_rate", AlertMetric::ErrorRate)
                .when(ThresholdOperator::GreaterThan, 0.1)
                .with_severity(AlertSeverity::Warning),
        );

        let dashboard = Dashboard::default();

        // Initial evaluation - no alerts
        manager.evaluate(&dashboard);
        assert!(manager.active_alerts().is_empty());
    }

    #[test]
    fn test_default_rules() {
        let rules = create_default_rules();
        assert!(!rules.is_empty());

        // Check we have both warning and critical rules
        let has_warning = rules.iter().any(|r| r.severity == AlertSeverity::Warning);
        let has_critical = rules.iter().any(|r| r.severity == AlertSeverity::Critical);

        assert!(has_warning);
        assert!(has_critical);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    #[test]
    fn test_active_alert_format() {
        let alert = ActiveAlert {
            rule_name: "test_rule".to_string(),
            current_value: 0.15,
            threshold: 0.1,
            severity: AlertSeverity::Warning,
            triggered_at: Instant::now(),
            message: "Error rate too high".to_string(),
        };

        let formatted = alert.format();
        assert!(formatted.contains("WARNING"));
        assert!(formatted.contains("test_rule"));
    }
}

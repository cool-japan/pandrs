#![allow(clippy::result_large_err)]
//! Real-Time Analytics Dashboard Example
//!
//! This example demonstrates how to use PandRS's analytics system for real-time
//! monitoring and performance tracking in production applications.
//!
//! Features demonstrated:
//! - Metrics collection (counters, gauges, histograms, timers)
//! - Operation tracking with category-based statistics
//! - Alert management with configurable rules
//! - Resource monitoring (CPU, memory, throughput)
//! - Real-time data visualization
//! - Dashboard update loop
//! - Performance metrics reporting
//!
//! # Usage
//!
//! Run this example with:
//! ```bash
//! cargo run --example analytics_dashboard_example --features visualization
//! ```
//!
//! # Production Integration
//!
//! To integrate analytics into your production application:
//!
//! 1. Initialize the dashboard early in your application lifecycle
//! 2. Record operations as they occur using `dashboard.record_operation()`
//! 3. Set up alert rules for critical metrics
//! 4. Run the dashboard update loop in a background thread
//! 5. Export metrics to your monitoring system (Prometheus, Grafana, etc.)

#[cfg(feature = "visualization")]
use pandrs::analytics::{
    create_default_rules, ActiveAlert, AlertHandler, AlertManager, AlertMetric, AlertRule,
    AlertSeverity, Dashboard, DashboardConfig, LoggingAlertHandler, MetricStats, MetricsCollector,
    OperationCategory, ThresholdOperator,
};
#[cfg(feature = "visualization")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "visualization")]
use std::sync::{Arc, RwLock};
#[cfg(feature = "visualization")]
use std::thread;
#[cfg(feature = "visualization")]
use std::time::Duration;
#[cfg(feature = "visualization")]
use textplots::{Chart, Plot, Shape};

#[cfg(not(feature = "visualization"))]
fn main() {
    println!("This example requires the 'visualization' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example analytics_dashboard_example --features visualization");
}

#[cfg(feature = "visualization")]
fn main() {
    println!("========================================");
    println!("  PandRS Analytics Dashboard Example");
    println!("========================================\n");

    // Example 1: Basic dashboard setup
    println!("Example 1: Basic Dashboard Setup");
    println!("--------------------------------");
    basic_dashboard_example();

    // Example 2: Metrics collection
    println!("\n\nExample 2: Metrics Collection");
    println!("-----------------------------");
    metrics_collection_example();

    // Example 3: Operation tracking
    println!("\n\nExample 3: Operation Tracking");
    println!("-----------------------------");
    operation_tracking_example();

    // Example 4: Alert management
    println!("\n\nExample 4: Alert Management");
    println!("---------------------------");
    alert_management_example();

    // Example 5: Real-time dashboard
    println!("\n\nExample 5: Real-Time Dashboard");
    println!("------------------------------");
    realtime_dashboard_example();

    // Example 6: Production integration
    println!("\n\nExample 6: Production Integration Pattern");
    println!("-----------------------------------------");
    production_integration_example();

    println!("\n\n========================================");
    println!("  Analytics Dashboard Examples Complete");
    println!("========================================\n");
}

/// Example 1: Basic dashboard setup and configuration
#[cfg(feature = "visualization")]
fn basic_dashboard_example() {
    // Create a dashboard with custom configuration
    let config = DashboardConfig {
        enabled: true,
        retention_period: Duration::from_secs(3600), // Keep metrics for 1 hour
        aggregation_interval: Duration::from_secs(60), // Aggregate every minute
        max_metrics: 50_000,                         // Store up to 50k metrics
        alerting_enabled: true,
        sample_rate: 1.0, // 100% sampling (use 0.1 for 10% sampling in high-traffic scenarios)
    };

    let dashboard = Dashboard::new(config);

    // Start the dashboard
    dashboard.start();
    println!("Dashboard started: {}", dashboard.is_running());

    // Get the metrics collector for fine-grained metric tracking
    let metrics = dashboard.metrics();

    // Create different types of metrics
    let request_counter = metrics.counter("requests_total");
    let active_connections = metrics.gauge("active_connections");
    let response_times = metrics.histogram("response_times");
    let operation_timer = metrics.timer("operation_duration");

    // Record some sample metrics
    request_counter.increment();
    request_counter.increment_by(5);

    active_connections.set(42.0);

    for i in 1..=10 {
        response_times.record((i * 10) as f64);
    }

    operation_timer.record(1500.0); // 1.5ms in microseconds

    // Display current metrics
    println!("\nMetrics Summary:");
    println!("  Requests: {}", request_counter.current());
    println!("  Active connections: {}", active_connections.current());

    let response_stats = response_times.stats();
    println!("  Response times:");
    println!("    Mean: {:.2}", response_stats.mean);
    println!("    P50: {:.2}", response_stats.p50);
    println!("    P95: {:.2}", response_stats.p95);
    println!("    P99: {:.2}", response_stats.p99);

    // Get a full dashboard snapshot
    let snapshot = dashboard.snapshot();
    println!("\nDashboard Snapshot:");
    println!("  Uptime: {:?}", snapshot.uptime);
    println!("  Total operations: {}", snapshot.total_operations);
    println!("  Ops/sec: {:.2}", snapshot.ops_per_second);
    println!("  Error rate: {:.2}%", snapshot.error_rate * 100.0);

    // Stop the dashboard
    dashboard.stop();
    println!("\nDashboard stopped: {}", !dashboard.is_running());
}

/// Example 2: Comprehensive metrics collection
#[cfg(feature = "visualization")]
fn metrics_collection_example() {
    let collector = MetricsCollector::new();

    println!("Collecting metrics from simulated workload...\n");

    // Simulate various operations with timing
    for i in 0..100 {
        // Database queries
        let query_time = 50.0 + (i as f64 % 50.0);
        collector.record("db.query.duration", query_time);

        // API calls
        let api_time = 20.0 + (i as f64 % 30.0);
        collector.record("api.call.duration", api_time);

        // Cache hits/misses
        if i % 3 == 0 {
            collector.increment("cache.hits");
        } else {
            collector.increment("cache.misses");
        }

        // Memory usage (simulated)
        let memory_mb = 1024.0 + (i as f64 * 2.0);
        collector.set_gauge("memory.used.mb", memory_mb);
    }

    // Display collected metrics
    println!("Metrics Summary:");

    let db_metric = collector
        .get("db.query.duration")
        .expect("operation should succeed");
    let db_stats = db_metric.stats();
    println!("\nDatabase Queries:");
    println!("  Count: {}", db_stats.count);
    println!("  Mean: {:.2}ms", db_stats.mean);
    println!("  Min: {:.2}ms", db_stats.min);
    println!("  Max: {:.2}ms", db_stats.max);
    println!("  P95: {:.2}ms", db_stats.p95);
    println!("  StdDev: {:.2}ms", db_stats.std_dev);

    let api_metric = collector
        .get("api.call.duration")
        .expect("operation should succeed");
    let api_stats = api_metric.stats();
    println!("\nAPI Calls:");
    println!("  Count: {}", api_stats.count);
    println!("  Mean: {:.2}ms", api_stats.mean);
    println!("  P99: {:.2}ms", api_stats.p99);

    let cache_hits = collector
        .get("cache.hits")
        .expect("operation should succeed");
    let cache_misses = collector
        .get("cache.misses")
        .expect("operation should succeed");
    let total_cache = cache_hits.current() + cache_misses.current();
    let hit_rate = cache_hits.current() / total_cache;

    println!("\nCache Performance:");
    println!("  Hits: {}", cache_hits.current());
    println!("  Misses: {}", cache_misses.current());
    println!("  Hit Rate: {:.2}%", hit_rate * 100.0);

    let memory = collector
        .get("memory.used.mb")
        .expect("operation should succeed");
    println!("\nMemory Usage:");
    println!("  Current: {:.2} MB", memory.current());

    // Visualize query duration distribution
    visualize_metric_distribution(&db_stats, "Database Query Duration Distribution");
}

/// Example 3: Operation tracking with categories
#[cfg(feature = "visualization")]
fn operation_tracking_example() {
    let dashboard = Dashboard::default();
    dashboard.start();

    println!("Tracking operations across different categories...\n");

    // Simulate various DataFrame operations
    let operations = vec![
        (
            "select",
            OperationCategory::Query,
            100,
            Some(1000),
            Some(8000),
        ),
        (
            "filter",
            OperationCategory::Filter,
            150,
            Some(800),
            Some(6400),
        ),
        (
            "groupby",
            OperationCategory::GroupBy,
            500,
            Some(500),
            Some(4000),
        ),
        (
            "aggregate",
            OperationCategory::Aggregation,
            300,
            Some(100),
            Some(800),
        ),
        (
            "join",
            OperationCategory::Join,
            800,
            Some(2000),
            Some(16000),
        ),
        (
            "sort",
            OperationCategory::Sort,
            400,
            Some(1500),
            Some(12000),
        ),
        (
            "read_csv",
            OperationCategory::IO,
            1000,
            Some(5000),
            Some(40000),
        ),
        (
            "write_parquet",
            OperationCategory::IO,
            1200,
            Some(5000),
            Some(40000),
        ),
    ];

    // Record operations multiple times to build statistics
    for _ in 0..50 {
        for (name, category, base_duration, rows, bytes) in &operations {
            let duration = base_duration
                + (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() % 200)
                    .unwrap_or(0) as u64);

            let success = duration < 1000; // Simulate failures for very slow ops

            dashboard.record_operation(
                name,
                *category,
                duration,
                *rows,
                *bytes,
                success,
                if !success {
                    Some("Operation timeout".to_string())
                } else {
                    None
                },
            );
        }
    }

    // Display statistics by category
    println!("Operation Statistics by Category:");
    println!("{:-<80}", "");

    let categories = [
        OperationCategory::Query,
        OperationCategory::Filter,
        OperationCategory::GroupBy,
        OperationCategory::Aggregation,
        OperationCategory::Join,
        OperationCategory::Sort,
        OperationCategory::IO,
    ];

    for category in categories {
        let stats = dashboard.category_stats(category);
        if stats.count > 0 {
            println!("\n{:?}", category);
            println!("  Count: {}", stats.count);
            println!("  Mean: {:.2}μs", stats.mean);
            println!("  Min: {:.2}μs", stats.min);
            println!("  Max: {:.2}μs", stats.max);
            println!("  P95: {:.2}μs", stats.p95);
            println!("  P99: {:.2}μs", stats.p99);
        }
    }

    // Display overall statistics
    let snapshot = dashboard.snapshot();
    println!("\n{:-<80}", "");
    println!("\nOverall Performance:");
    println!("  Total operations: {}", snapshot.total_operations);
    println!("  Operations/sec: {:.2}", snapshot.ops_per_second);
    println!("  Avg latency: {:.2}μs", snapshot.avg_latency_us);
    println!("  P99 latency: {:.2}μs", snapshot.p99_latency_us);
    println!("  Error rate: {:.2}%", snapshot.error_rate * 100.0);
    println!("  Total rows: {}", snapshot.total_rows);
    println!("  Rows/sec: {:.2}", snapshot.rows_per_second);
    println!("  Total bytes: {}", snapshot.total_bytes);
    println!(
        "  Throughput: {:.2} KB/s",
        snapshot.bytes_per_second / 1024.0
    );

    // Show slowest operations
    println!("\nSlowest Operations:");
    for (i, op) in dashboard.slowest_operations(5).iter().enumerate() {
        println!(
            "  {}. {} ({}): {:.2}μs",
            i + 1,
            op.name,
            op.category,
            op.duration_us
        );
    }

    // Show failed operations
    let failed = dashboard.failed_operations(5);
    if !failed.is_empty() {
        println!("\nFailed Operations:");
        for (i, op) in failed.iter().enumerate() {
            println!(
                "  {}. {} ({}): {} - {}",
                i + 1,
                op.name,
                op.category,
                op.duration_us,
                op.error.as_ref().unwrap_or(&"Unknown error".to_string())
            );
        }
    }
}

/// Example 4: Alert management
#[cfg(feature = "visualization")]
fn alert_management_example() {
    let dashboard = Arc::new(Dashboard::default());
    dashboard.start();

    let alert_manager = AlertManager::new();

    println!("Setting up alert rules...\n");

    // Add custom alert handler
    let custom_handler = CustomAlertHandler::new();
    alert_manager.add_handler(Box::new(custom_handler));

    // Add default alert rules
    for rule in create_default_rules() {
        println!("Added rule: {} - {}", rule.name, rule.description);
        alert_manager.add_rule(rule);
    }

    // Add custom rules
    alert_manager.add_rule(
        AlertRule::new(
            "slow_queries",
            AlertMetric::CategoryLatency(OperationCategory::Query),
        )
        .with_description("Query latency exceeds 500μs")
        .when(ThresholdOperator::GreaterThan, 500.0)
        .with_severity(AlertSeverity::Warning)
        .for_duration(Duration::from_secs(10)),
    );

    alert_manager.add_rule(
        AlertRule::new("low_throughput", AlertMetric::RowsPerSecond)
            .with_description("Row processing rate below threshold")
            .when(ThresholdOperator::LessThan, 100.0)
            .with_severity(AlertSeverity::Info)
            .for_duration(Duration::from_secs(30)),
    );

    println!("\nSimulating workload with error injection...\n");

    // Simulate operations with intentional errors to trigger alerts
    for i in 0..200 {
        let success = i % 10 != 0; // 10% error rate
        let duration = if i % 5 == 0 { 2_000_000 } else { 100 }; // Some very slow operations

        dashboard.record_operation(
            "query",
            OperationCategory::Query,
            duration,
            Some(10),
            Some(100),
            success,
            if !success {
                Some("Connection timeout".to_string())
            } else {
                None
            },
        );

        // Evaluate alerts periodically
        if i % 20 == 0 {
            alert_manager.evaluate(&dashboard);

            // Check for active alerts
            let active_alerts = alert_manager.active_alerts();
            if !active_alerts.is_empty() {
                println!("Active Alerts at iteration {}:", i);
                for alert in active_alerts {
                    println!("  {}", alert.format());
                }
            }
        }

        thread::sleep(Duration::from_millis(10));
    }

    // Final evaluation
    alert_manager.evaluate(&dashboard);

    // Display alert summary
    println!("\n{:-<80}", "");
    println!("\nAlert Summary:");

    let counts = alert_manager.alert_counts();
    println!("  Info: {}", counts.get(&AlertSeverity::Info).unwrap_or(&0));
    println!(
        "  Warning: {}",
        counts.get(&AlertSeverity::Warning).unwrap_or(&0)
    );
    println!(
        "  Critical: {}",
        counts.get(&AlertSeverity::Critical).unwrap_or(&0)
    );

    if alert_manager.has_critical() {
        println!("\n  ⚠️  CRITICAL ALERTS DETECTED!");
    }

    let snapshot = dashboard.snapshot();
    println!("\nFinal Metrics:");
    println!("  Error rate: {:.2}%", snapshot.error_rate * 100.0);
    println!("  P99 latency: {:.2}μs", snapshot.p99_latency_us);
}

/// Example 5: Real-time dashboard with visualization
#[cfg(feature = "visualization")]
fn realtime_dashboard_example() {
    let dashboard = Arc::new(Dashboard::default());
    dashboard.start();

    let running = Arc::new(AtomicBool::new(true));
    let history = Arc::new(RwLock::new(Vec::new()));

    // Spawn workload simulator
    let dashboard_clone = Arc::clone(&dashboard);
    let running_clone = Arc::clone(&running);
    let workload_thread = thread::spawn(move || {
        let mut iteration = 0;
        while running_clone.load(Ordering::Relaxed) {
            // Simulate various operations
            let op_types = [
                ("query", OperationCategory::Query, 100),
                ("filter", OperationCategory::Filter, 50),
                ("aggregate", OperationCategory::Aggregation, 200),
                ("join", OperationCategory::Join, 300),
            ];

            for (name, category, base_duration) in &op_types {
                let variance = (iteration % 50) as u64;
                let duration = base_duration + variance;

                dashboard_clone.record_operation(
                    name,
                    *category,
                    duration,
                    Some(100),
                    Some(1000),
                    true,
                    None,
                );
            }

            iteration += 1;
            thread::sleep(Duration::from_millis(100));
        }
    });

    // Dashboard update loop
    let dashboard_clone = Arc::clone(&dashboard);
    let history_clone = Arc::clone(&history);
    let running_clone = Arc::clone(&running);

    println!("Running real-time dashboard for 5 seconds...\n");

    let update_thread = thread::spawn(move || {
        let mut update_count = 0;
        while running_clone.load(Ordering::Relaxed) && update_count < 5 {
            thread::sleep(Duration::from_secs(1));

            let snapshot = dashboard_clone.snapshot();

            // Store in history
            if let Ok(mut hist) = history_clone.write() {
                hist.push((update_count as f32, snapshot.ops_per_second as f32));
                if hist.len() > 60 {
                    hist.remove(0);
                }
            }

            // Display current metrics
            println!("\n{:=<80}", "");
            println!("Dashboard Update #{}", update_count + 1);
            println!("{:=<80}", "");

            println!("\nPerformance Metrics:");
            println!("  Operations: {}", snapshot.total_operations);
            println!("  Ops/sec: {:.2}", snapshot.ops_per_second);
            println!("  Avg latency: {:.2}μs", snapshot.avg_latency_us);
            println!("  P99 latency: {:.2}μs", snapshot.p99_latency_us);
            println!("  Error rate: {:.2}%", snapshot.error_rate * 100.0);

            println!("\nThroughput:");
            println!("  Rows/sec: {:.2}", snapshot.rows_per_second);
            println!("  Bytes/sec: {:.2} KB", snapshot.bytes_per_second / 1024.0);

            // Display category breakdown
            println!("\nCategory Performance:");
            for (category, stats) in &snapshot.category_stats {
                if stats.count > 0 {
                    println!(
                        "  {:12}: {:6} ops, {:.2}μs mean, {:.2}μs p99",
                        format!("{:?}", category),
                        stats.count,
                        stats.mean,
                        stats.p99
                    );
                }
            }

            update_count += 1;
        }
    });

    // Wait for completion
    update_thread.join().expect("operation should succeed");
    running.store(false, Ordering::Relaxed);
    workload_thread.join().expect("operation should succeed");

    // Final visualization
    println!("\n{:=<80}", "");
    println!("Operations Per Second Over Time");
    println!("{:=<80}\n", "");

    if let Ok(hist) = history.read() {
        if !hist.is_empty() {
            Chart::new(120, 40, 0.0, hist.len() as f32)
                .lineplot(&Shape::Lines(&hist))
                .display();
        }
    }

    println!("\nDashboard monitoring complete.");
}

/// Example 6: Production integration pattern
#[cfg(feature = "visualization")]
fn production_integration_example() {
    println!("Production Integration Pattern:\n");

    // 1. Initialize global dashboard
    let config = DashboardConfig {
        enabled: true,
        retention_period: Duration::from_secs(3600),
        aggregation_interval: Duration::from_secs(60),
        max_metrics: 100_000,
        alerting_enabled: true,
        sample_rate: 1.0,
    };

    let dashboard = Arc::new(Dashboard::new(config));
    dashboard.start();

    // 2. Set up alert manager
    let alert_manager = Arc::new(AlertManager::new());

    // Add production alert rules
    alert_manager.add_rule(
        AlertRule::new("production_high_error_rate", AlertMetric::ErrorRate)
            .with_description("Production error rate exceeds 1%")
            .when(ThresholdOperator::GreaterThan, 0.01)
            .with_severity(AlertSeverity::Critical)
            .for_duration(Duration::from_secs(60))
            .with_label("environment", "production")
            .with_label("team", "platform"),
    );

    alert_manager.add_rule(
        AlertRule::new("production_slow_queries", AlertMetric::P99Latency)
            .with_description("P99 query latency exceeds 500ms")
            .when(ThresholdOperator::GreaterThan, 500_000.0)
            .with_severity(AlertSeverity::Warning)
            .for_duration(Duration::from_secs(120)),
    );

    // Add logging handler
    alert_manager.add_handler(Box::new(LoggingAlertHandler::new("PRODUCTION")));

    println!("✓ Dashboard and alerts configured for production\n");

    // 3. Simulate production workload
    println!("Simulating production workload patterns...\n");

    let workload_patterns = vec![
        ("user_query", OperationCategory::Query, 50, 1000, 8000),
        ("data_write", OperationCategory::Write, 100, 500, 4000),
        (
            "batch_process",
            OperationCategory::Aggregation,
            500,
            10000,
            80000,
        ),
        (
            "report_generate",
            OperationCategory::Join,
            1000,
            50000,
            400000,
        ),
    ];

    for iteration in 0..100 {
        for (name, category, duration, rows, bytes) in &workload_patterns {
            // Add some variance
            let actual_duration = duration + (iteration % 20) as u64;
            let success = iteration % 100 != 99; // 1% error rate

            dashboard.record_operation(
                name,
                *category,
                actual_duration,
                Some(*rows),
                Some(*bytes),
                success,
                if !success {
                    Some("Database connection lost".to_string())
                } else {
                    None
                },
            );
        }

        // Periodic alert evaluation
        if iteration % 10 == 0 {
            alert_manager.evaluate(&dashboard);
        }
    }

    // 4. Generate summary report
    println!("\n{:=<80}", "");
    println!("Production Metrics Summary Report");
    println!("{:=<80}\n", "");

    let snapshot = dashboard.snapshot();

    println!("System Health:");
    println!("  Uptime: {:?}", snapshot.uptime);
    println!(
        "  Status: {}",
        if snapshot.error_rate < 0.01 {
            "Healthy"
        } else {
            "Degraded"
        }
    );
    println!("  Error Rate: {:.4}%", snapshot.error_rate * 100.0);

    println!("\nPerformance:");
    println!("  Total Operations: {}", snapshot.total_operations);
    println!("  Throughput: {:.2} ops/sec", snapshot.ops_per_second);
    println!("  Latency (avg): {:.2}μs", snapshot.avg_latency_us);
    println!("  Latency (p99): {:.2}μs", snapshot.p99_latency_us);

    println!("\nData Processing:");
    println!("  Total Rows: {}", snapshot.total_rows);
    println!("  Row Rate: {:.2} rows/sec", snapshot.rows_per_second);
    println!("  Total Bytes: {} KB", snapshot.total_bytes / 1024);
    println!(
        "  Throughput: {:.2} MB/sec",
        snapshot.bytes_per_second / (1024.0 * 1024.0)
    );

    println!("\nOperation Categories:");
    for (category, stats) in snapshot.category_stats {
        println!("  {:?}:", category);
        println!("    Count: {}", stats.count);
        println!("    Mean: {:.2}μs", stats.mean);
        println!("    P95: {:.2}μs", stats.p95);
        println!("    P99: {:.2}μs", stats.p99);
    }

    // 5. Alert summary
    let alert_counts = alert_manager.alert_counts();
    let total_alerts = alert_counts.values().sum::<usize>();

    if total_alerts > 0 {
        println!("\nActive Alerts:");
        for alert in alert_manager.active_alerts() {
            println!("  {}", alert.format());
        }
    } else {
        println!("\nActive Alerts: None");
    }

    println!("\n{:=<80}", "");
    println!("Integration pattern demonstration complete.");
    println!("{:=<80}\n", "");

    // Tips for production
    println!("Production Deployment Tips:");
    println!("  1. Export metrics to Prometheus/Grafana for visualization");
    println!("  2. Integrate with PagerDuty/OpsGenie for alert notification");
    println!("  3. Use sampling (sample_rate < 1.0) for high-traffic scenarios");
    println!("  4. Set appropriate retention periods based on monitoring needs");
    println!("  5. Create separate dashboards for different service components");
    println!("  6. Monitor dashboard overhead (should be < 1% of application load)");
    println!("  7. Use category-specific alerts for fine-grained monitoring");
    println!("  8. Implement graceful degradation if monitoring fails");
}

/// Helper function to visualize metric distributions
#[cfg(feature = "visualization")]
fn visualize_metric_distribution(stats: &MetricStats, title: &str) {
    println!("\n{}", title);
    println!("{:-<80}", "");

    // Create a simple visualization of the distribution
    let values = vec![
        ("Min", stats.min),
        ("P50", stats.p50),
        ("Mean", stats.mean),
        ("P95", stats.p95),
        ("P99", stats.p99),
        ("Max", stats.max),
    ];

    let max_val = stats.max;

    for (label, value) in values {
        let bar_length = if max_val > 0.0 {
            ((value / max_val) * 60.0) as usize
        } else {
            0
        };
        let bar = "█".repeat(bar_length);
        println!("{:6} {:8.2} {}", label, value, bar);
    }

    println!("\nStatistics:");
    println!("  Count: {}", stats.count);
    println!("  StdDev: {:.2}", stats.std_dev);
    println!("  Variance: {:.2}", stats.variance);
}

/// Custom alert handler for demonstration
#[cfg(feature = "visualization")]
#[derive(Debug)]
struct CustomAlertHandler {
    alerts_fired: Arc<RwLock<Vec<String>>>,
}

#[cfg(feature = "visualization")]
impl CustomAlertHandler {
    fn new() -> Self {
        CustomAlertHandler {
            alerts_fired: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[cfg(feature = "visualization")]
impl AlertHandler for CustomAlertHandler {
    fn on_alert(&self, alert: &ActiveAlert) {
        println!("\n🚨 ALERT FIRED: {}", alert.format());

        if let Ok(mut alerts) = self.alerts_fired.write() {
            alerts.push(alert.rule_name.clone());
        }
    }

    fn on_resolve(&self, rule_name: &str) {
        println!("\n✅ ALERT RESOLVED: {}", rule_name);

        if let Ok(mut alerts) = self.alerts_fired.write() {
            alerts.retain(|n| n != rule_name);
        }
    }
}

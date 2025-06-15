//! JIT Performance Monitoring and Adaptive Optimization
//!
//! This module provides comprehensive performance monitoring for JIT-compiled functions
//! and enables adaptive optimization based on runtime performance characteristics.

use crate::core::error::{Error, Result};
use crate::optimized::jit::cache::{CacheStats, FunctionId};
use crate::optimized::jit::config::{JITConfig, LoadBalancing, ParallelConfig, SIMDConfig};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Performance metrics for a specific function
#[derive(Debug, Clone)]
pub struct FunctionPerformanceMetrics {
    /// Function identifier
    pub function_id: FunctionId,
    /// Total number of executions
    pub execution_count: u64,
    /// Total execution time in nanoseconds
    pub total_execution_time_ns: u64,
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: f64,
    /// Minimum execution time observed
    pub min_execution_time_ns: u64,
    /// Maximum execution time observed
    pub max_execution_time_ns: u64,
    /// Standard deviation of execution times
    pub std_dev_execution_time_ns: f64,
    /// Recent execution times (sliding window)
    pub recent_execution_times: VecDeque<u64>,
    /// Memory usage per execution (estimated)
    pub avg_memory_usage_bytes: usize,
    /// CPU utilization during execution (0.0 to 1.0)
    pub avg_cpu_utilization: f64,
    /// Cache hit rate for this function
    pub cache_hit_rate: f64,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// Performance trend (improving, stable, degrading)
    pub performance_trend: PerformanceTrend,
    /// Last updated timestamp
    pub last_updated: Instant,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

impl FunctionPerformanceMetrics {
    /// Create new performance metrics for a function
    pub fn new(function_id: FunctionId) -> Self {
        Self {
            function_id,
            execution_count: 0,
            total_execution_time_ns: 0,
            avg_execution_time_ns: 0.0,
            min_execution_time_ns: u64::MAX,
            max_execution_time_ns: 0,
            std_dev_execution_time_ns: 0.0,
            recent_execution_times: VecDeque::with_capacity(100),
            avg_memory_usage_bytes: 0,
            avg_cpu_utilization: 0.0,
            cache_hit_rate: 0.0,
            throughput_ops_per_sec: 0.0,
            performance_trend: PerformanceTrend::Stable,
            last_updated: Instant::now(),
            optimization_suggestions: Vec::new(),
        }
    }

    /// Record a new execution
    pub fn record_execution(
        &mut self,
        execution_time_ns: u64,
        memory_usage_bytes: usize,
        cpu_utilization: f64,
    ) {
        self.execution_count += 1;
        self.total_execution_time_ns += execution_time_ns;
        self.avg_execution_time_ns =
            self.total_execution_time_ns as f64 / self.execution_count as f64;

        self.min_execution_time_ns = self.min_execution_time_ns.min(execution_time_ns);
        self.max_execution_time_ns = self.max_execution_time_ns.max(execution_time_ns);

        // Update recent execution times (sliding window)
        self.recent_execution_times.push_back(execution_time_ns);
        if self.recent_execution_times.len() > 100 {
            self.recent_execution_times.pop_front();
        }

        // Update averages
        self.avg_memory_usage_bytes =
            ((self.avg_memory_usage_bytes as u64 * (self.execution_count - 1))
                / self.execution_count
                + memory_usage_bytes as u64 / self.execution_count) as usize;

        self.avg_cpu_utilization = (self.avg_cpu_utilization * (self.execution_count - 1) as f64
            + cpu_utilization)
            / self.execution_count as f64;

        // Calculate standard deviation
        self.calculate_std_dev();

        // Update throughput
        self.throughput_ops_per_sec = 1_000_000_000.0 / self.avg_execution_time_ns;

        // Analyze performance trend
        self.update_performance_trend();

        // Generate optimization suggestions
        self.update_optimization_suggestions();

        self.last_updated = Instant::now();
    }

    /// Calculate standard deviation of execution times
    fn calculate_std_dev(&mut self) {
        if self.recent_execution_times.len() < 2 {
            return;
        }

        let mean = self.recent_execution_times.iter().sum::<u64>() as f64
            / self.recent_execution_times.len() as f64;
        let variance = self
            .recent_execution_times
            .iter()
            .map(|&x| {
                let diff = x as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / (self.recent_execution_times.len() - 1) as f64;

        self.std_dev_execution_time_ns = variance.sqrt();
    }

    /// Update performance trend analysis
    fn update_performance_trend(&mut self) {
        if self.recent_execution_times.len() < 10 {
            return;
        }

        let recent_count = self.recent_execution_times.len();
        let split_point = recent_count / 2;

        let first_half: f64 = self
            .recent_execution_times
            .iter()
            .take(split_point)
            .map(|&x| x as f64)
            .sum::<f64>()
            / split_point as f64;

        let second_half: f64 = self
            .recent_execution_times
            .iter()
            .skip(split_point)
            .map(|&x| x as f64)
            .sum::<f64>()
            / (recent_count - split_point) as f64;

        let improvement_ratio = first_half / second_half;

        self.performance_trend = if improvement_ratio > 1.1 {
            PerformanceTrend::Improving
        } else if improvement_ratio < 0.9 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        };
    }

    /// Update optimization suggestions based on current metrics
    fn update_optimization_suggestions(&mut self) {
        self.optimization_suggestions.clear();

        // High variance suggests inconsistent performance
        if self.std_dev_execution_time_ns > self.avg_execution_time_ns * 0.2 {
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceVariance,
                description: "High execution time variance detected. Consider function specialization or better memory management.".to_string(),
                priority: OptimizationPriority::Medium,
                estimated_improvement: 0.15,
            });
        }

        // High memory usage suggests need for optimization
        if self.avg_memory_usage_bytes > 1024 * 1024 {
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceMemoryUsage,
                description: "High memory usage detected. Consider memory pooling or more efficient data structures.".to_string(),
                priority: OptimizationPriority::High,
                estimated_improvement: 0.25,
            });
        }

        // Low CPU utilization suggests underutilization
        if self.avg_cpu_utilization < 0.5 && self.execution_count > 50 {
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::IncreaseCpuUtilization,
                description:
                    "Low CPU utilization detected. Consider vectorization or parallelization."
                        .to_string(),
                priority: OptimizationPriority::Medium,
                estimated_improvement: 0.30,
            });
        }

        // Slow execution suggests need for algorithmic optimization
        if self.avg_execution_time_ns > 10_000_000.0 {
            // 10ms
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::AlgorithmicOptimization,
                description:
                    "Slow execution detected. Consider algorithmic improvements or caching."
                        .to_string(),
                priority: OptimizationPriority::High,
                estimated_improvement: 0.40,
            });
        }

        // Performance degradation suggests need for investigation
        if matches!(self.performance_trend, PerformanceTrend::Degrading) {
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::PerformanceRegression,
                description: "Performance degradation detected. Investigate recent changes or memory fragmentation.".to_string(),
                priority: OptimizationPriority::High,
                estimated_improvement: 0.20,
            });
        }
    }

    /// Get performance score (0.0 to 1.0, higher is better)
    pub fn get_performance_score(&self) -> f64 {
        if self.execution_count == 0 {
            return 0.0;
        }

        // Score based on multiple factors
        let throughput_score = (self.throughput_ops_per_sec / 1_000_000.0).min(1.0); // Normalize to 1M ops/sec
        let consistency_score =
            1.0 - (self.std_dev_execution_time_ns / self.avg_execution_time_ns.max(1.0)).min(1.0);
        let cpu_score = self.avg_cpu_utilization;
        let trend_score = match self.performance_trend {
            PerformanceTrend::Improving => 1.0,
            PerformanceTrend::Stable => 0.8,
            PerformanceTrend::Degrading => 0.4,
        };

        throughput_score * 0.4 + consistency_score * 0.3 + cpu_score * 0.2 + trend_score * 0.1
    }
}

/// Performance trend indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
}

/// Optimization suggestion types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    ReduceVariance,
    ReduceMemoryUsage,
    IncreaseCpuUtilization,
    AlgorithmicOptimization,
    PerformanceRegression,
    EnableSIMD,
    EnableParallelization,
    IncreaseCacheLocality,
}

/// Optimization priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub suggestion_type: OptimizationType,
    /// Human-readable description
    pub description: String,
    /// Priority level
    pub priority: OptimizationPriority,
    /// Estimated performance improvement (0.0 to 1.0)
    pub estimated_improvement: f64,
}

/// System-wide performance metrics
#[derive(Debug, Clone)]
pub struct SystemPerformanceMetrics {
    /// Overall JIT system utilization
    pub jit_utilization: f64,
    /// Memory pressure (0.0 to 1.0)
    pub memory_pressure: f64,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Cache effectiveness
    pub cache_effectiveness: f64,
    /// Number of active JIT functions
    pub active_functions: usize,
    /// System uptime
    pub uptime: Duration,
    /// Total number of compilations
    pub total_compilations: u64,
    /// Failed compilations
    pub failed_compilations: u64,
    /// Average compilation time
    pub avg_compilation_time_ns: f64,
}

/// JIT Performance Monitor
pub struct JitPerformanceMonitor {
    /// Function-specific metrics
    function_metrics: RwLock<HashMap<FunctionId, FunctionPerformanceMetrics>>,
    /// System-wide metrics
    system_metrics: RwLock<SystemPerformanceMetrics>,
    /// Monitor start time
    start_time: Instant,
    /// Configuration for adaptive optimization
    config: Arc<RwLock<JITConfig>>,
    /// Performance history for trend analysis
    performance_history: RwLock<VecDeque<(Instant, f64)>>,
}

impl JitPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: JITConfig) -> Self {
        Self {
            function_metrics: RwLock::new(HashMap::new()),
            system_metrics: RwLock::new(SystemPerformanceMetrics {
                jit_utilization: 0.0,
                memory_pressure: 0.0,
                cpu_utilization: 0.0,
                cache_effectiveness: 0.0,
                active_functions: 0,
                uptime: Duration::new(0, 0),
                total_compilations: 0,
                failed_compilations: 0,
                avg_compilation_time_ns: 0.0,
            }),
            start_time: Instant::now(),
            config: Arc::new(RwLock::new(config)),
            performance_history: RwLock::new(VecDeque::with_capacity(1000)),
        }
    }

    /// Record function execution
    pub fn record_function_execution(
        &self,
        function_id: &FunctionId,
        execution_time_ns: u64,
        memory_usage_bytes: usize,
        cpu_utilization: f64,
    ) {
        // Update function metrics in a separate scope
        {
            let mut metrics = self.function_metrics.write().unwrap();
            let function_metrics = metrics
                .entry(function_id.clone())
                .or_insert_with(|| FunctionPerformanceMetrics::new(function_id.clone()));

            function_metrics.record_execution(
                execution_time_ns,
                memory_usage_bytes,
                cpu_utilization,
            );
        } // Write lock is released here

        // Update system metrics after releasing the write lock
        self.update_system_metrics();
    }

    /// Record compilation event
    pub fn record_compilation(
        &self,
        function_id: &FunctionId,
        compilation_time_ns: u64,
        success: bool,
    ) {
        let mut system_metrics = self.system_metrics.write().unwrap();

        system_metrics.total_compilations += 1;
        if !success {
            system_metrics.failed_compilations += 1;
        }

        // Update average compilation time
        let total_time =
            system_metrics.avg_compilation_time_ns * (system_metrics.total_compilations - 1) as f64;
        system_metrics.avg_compilation_time_ns =
            (total_time + compilation_time_ns as f64) / system_metrics.total_compilations as f64;
    }

    /// Get performance metrics for a specific function
    pub fn get_function_metrics(
        &self,
        function_id: &FunctionId,
    ) -> Option<FunctionPerformanceMetrics> {
        self.function_metrics
            .read()
            .unwrap()
            .get(function_id)
            .cloned()
    }

    /// Get system-wide performance metrics
    pub fn get_system_metrics(&self) -> SystemPerformanceMetrics {
        let mut metrics = self.system_metrics.read().unwrap().clone();
        metrics.uptime = self.start_time.elapsed();
        metrics
    }

    /// Get top performing functions
    pub fn get_top_performing_functions(&self, count: usize) -> Vec<FunctionPerformanceMetrics> {
        let metrics = self.function_metrics.read().unwrap();
        let mut functions: Vec<_> = metrics.values().cloned().collect();

        functions.sort_by(|a, b| {
            b.get_performance_score()
                .partial_cmp(&a.get_performance_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        functions.into_iter().take(count).collect()
    }

    /// Get functions that need optimization
    pub fn get_functions_needing_optimization(
        &self,
    ) -> Vec<(FunctionId, Vec<OptimizationSuggestion>)> {
        let metrics = self.function_metrics.read().unwrap();

        metrics
            .iter()
            .filter_map(|(id, metrics)| {
                if !metrics.optimization_suggestions.is_empty() {
                    Some((id.clone(), metrics.optimization_suggestions.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Suggest configuration optimizations based on current performance
    pub fn suggest_config_optimizations(&self) -> Vec<ConfigOptimization> {
        let system_metrics = self.get_system_metrics();
        let mut suggestions = Vec::new();

        // Suggest parallel optimization if CPU utilization is low
        if system_metrics.cpu_utilization < 0.5 {
            suggestions.push(ConfigOptimization {
                config_type: ConfigType::Parallel,
                description: "Low CPU utilization detected. Consider enabling more aggressive parallelization.".to_string(),
                recommended_change: "Reduce min_chunk_size and enable dynamic load balancing".to_string(),
                estimated_improvement: 0.25,
            });
        }

        // Suggest SIMD optimization if applicable
        if system_metrics.jit_utilization > 0.7 {
            suggestions.push(ConfigOptimization {
                config_type: ConfigType::SIMD,
                description: "High JIT utilization detected. SIMD operations could provide significant speedup.".to_string(),
                recommended_change: "Enable SIMD with lower min_simd_size threshold".to_string(),
                estimated_improvement: 0.30,
            });
        }

        // Suggest cache optimization if cache effectiveness is low
        if system_metrics.cache_effectiveness < 0.6 {
            suggestions.push(ConfigOptimization {
                config_type: ConfigType::Cache,
                description: "Low cache effectiveness detected. Consider increasing cache size or improving eviction policy.".to_string(),
                recommended_change: "Increase cache size and enable better caching heuristics".to_string(),
                estimated_improvement: 0.20,
            });
        }

        suggestions
    }

    /// Apply automatic optimizations based on performance data
    pub fn apply_automatic_optimizations(&self) -> Result<Vec<String>> {
        let suggestions = self.suggest_config_optimizations();
        let mut applied_optimizations = Vec::new();

        let mut config = self.config.write().unwrap();

        for suggestion in suggestions {
            match suggestion.config_type {
                ConfigType::Parallel => {
                    if suggestion.estimated_improvement > 0.2 {
                        config.parallel.min_chunk_size =
                            (config.parallel.min_chunk_size / 2).max(100);
                        config.parallel.load_balancing = LoadBalancing::Dynamic;
                        applied_optimizations
                            .push("Enabled more aggressive parallelization".to_string());
                    }
                }
                ConfigType::SIMD => {
                    if suggestion.estimated_improvement > 0.25 {
                        config.simd.min_simd_size = (config.simd.min_simd_size / 2).max(32);
                        config.simd.enabled = true;
                        applied_optimizations.push("Optimized SIMD configuration".to_string());
                    }
                }
                ConfigType::Cache => {
                    // Cache optimization would be handled by the cache system itself
                    applied_optimizations.push("Triggered cache optimization".to_string());
                }
                ConfigType::Compilation => {
                    if config.optimization_level < 3 {
                        config.optimization_level += 1;
                        applied_optimizations
                            .push("Increased compilation optimization level".to_string());
                    }
                }
            }
        }

        Ok(applied_optimizations)
    }

    /// Update system-wide metrics
    fn update_system_metrics(&self) {
        let function_metrics = self.function_metrics.read().unwrap();
        let mut system_metrics = self.system_metrics.write().unwrap();

        // Calculate active functions and average performance
        system_metrics.active_functions = function_metrics.len();

        if !function_metrics.is_empty() {
            let avg_cpu = function_metrics
                .values()
                .map(|m| m.avg_cpu_utilization)
                .sum::<f64>()
                / function_metrics.len() as f64;

            system_metrics.cpu_utilization = avg_cpu;

            // Calculate JIT utilization based on hot functions
            let hot_functions = function_metrics
                .values()
                .filter(|m| m.execution_count > 100)
                .count();

            system_metrics.jit_utilization = hot_functions as f64 / function_metrics.len() as f64;
        }

        // Record performance history
        let mut history = self.performance_history.write().unwrap();
        history.push_back((Instant::now(), system_metrics.jit_utilization));

        if history.len() > 1000 {
            history.pop_front();
        }
    }
}

/// Configuration optimization suggestion
#[derive(Debug, Clone)]
pub struct ConfigOptimization {
    /// Type of configuration to optimize
    pub config_type: ConfigType,
    /// Description of the optimization
    pub description: String,
    /// Recommended change
    pub recommended_change: String,
    /// Estimated improvement (0.0 to 1.0)
    pub estimated_improvement: f64,
}

/// Configuration types that can be optimized
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigType {
    Parallel,
    SIMD,
    Cache,
    Compilation,
}

/// Global performance monitor instance
static GLOBAL_MONITOR: std::sync::OnceLock<JitPerformanceMonitor> = std::sync::OnceLock::new();

/// Get the global performance monitor
pub fn get_global_monitor() -> &'static JitPerformanceMonitor {
    GLOBAL_MONITOR.get_or_init(|| JitPerformanceMonitor::new(JITConfig::default()))
}

/// Initialize the global monitor with a specific configuration
pub fn init_global_monitor(config: JITConfig) -> Result<()> {
    GLOBAL_MONITOR
        .set(JitPerformanceMonitor::new(config))
        .map_err(|_| Error::InvalidOperation("Global monitor already initialized".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_performance_metrics() {
        let function_id = FunctionId::new("test", "f64", "f64", "test_op", 1);
        let mut metrics = FunctionPerformanceMetrics::new(function_id);

        // Record some executions
        metrics.record_execution(1_000_000, 1024, 0.8);
        metrics.record_execution(1_200_000, 1024, 0.9);

        assert_eq!(metrics.execution_count, 2);
        assert_eq!(metrics.avg_execution_time_ns, 1_100_000.0);
        assert!(metrics.get_performance_score() > 0.0);
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = JitPerformanceMonitor::new(JITConfig::default());
        let function_id = FunctionId::new("test", "f64", "f64", "test_op", 1);

        monitor.record_function_execution(&function_id, 1_000_000, 1024, 0.8);

        let metrics = monitor.get_function_metrics(&function_id);
        assert!(metrics.is_some());

        let system_metrics = monitor.get_system_metrics();
        assert_eq!(system_metrics.active_functions, 1);
    }

    #[test]
    fn test_optimization_suggestions() {
        let function_id = FunctionId::new("slow_function", "f64", "f64", "slow_op", 1);
        let mut metrics = FunctionPerformanceMetrics::new(function_id);

        // Record slow executions
        for _ in 0..10 {
            metrics.record_execution(50_000_000, 2_000_000, 0.3); // 50ms, 2MB, 30% CPU
        }

        assert!(!metrics.optimization_suggestions.is_empty());

        // Should suggest multiple optimizations
        let suggestion_types: Vec<_> = metrics
            .optimization_suggestions
            .iter()
            .map(|s| s.suggestion_type)
            .collect();

        // Check that at least some optimizations are suggested
        assert!(!suggestion_types.is_empty());

        // Common optimizations that should be triggered by slow performance
        let expected_optimizations = vec![
            OptimizationType::ReduceMemoryUsage,
            OptimizationType::IncreaseCpuUtilization,
            OptimizationType::AlgorithmicOptimization,
        ];

        // At least one of the expected optimizations should be present
        let has_expected = expected_optimizations
            .iter()
            .any(|opt| suggestion_types.contains(opt));
        assert!(
            has_expected,
            "Expected at least one of {:?}, but got {:?}",
            expected_optimizations, suggestion_types
        );
    }
}

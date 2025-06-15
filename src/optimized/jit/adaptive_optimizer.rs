//! Adaptive JIT Optimizer
//!
//! This module provides adaptive optimization capabilities that learn from runtime
//! performance data to make intelligent optimization decisions.

use crate::core::error::{Error, Result};
use crate::optimized::jit::{
    cache::{CachedFunctionMetadata, FunctionId, JitFunctionCache},
    config::{JITConfig, LoadBalancing, ParallelConfig, SIMDConfig},
    expression_tree::{ExpressionTree, OptimizationType as ExprOptType},
    performance_monitor::{
        FunctionPerformanceMetrics, JitPerformanceMonitor, OptimizationSuggestion, OptimizationType,
    },
};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Adaptive optimizer that learns from performance data
pub struct AdaptiveOptimizer {
    /// Performance monitor for gathering runtime data
    monitor: Arc<JitPerformanceMonitor>,
    /// Function cache for storing optimized functions
    cache: Arc<JitFunctionCache>,
    /// Current JIT configuration
    config: Arc<RwLock<JITConfig>>,
    /// Optimization history for learning
    optimization_history: RwLock<Vec<OptimizationEvent>>,
    /// Performance baselines for comparison
    performance_baselines: RwLock<HashMap<FunctionId, PerformanceBaseline>>,
    /// Learning parameters
    learning_params: LearningParameters,
    /// Optimization strategies
    strategies: Vec<Box<dyn OptimizationStrategy + Send + Sync>>,
    /// Last optimization timestamp
    last_optimization: RwLock<Instant>,
}

/// Learning parameters for the adaptive optimizer
#[derive(Debug, Clone)]
pub struct LearningParameters {
    /// Learning rate for adapting to performance changes
    pub learning_rate: f64,
    /// Minimum improvement threshold to apply an optimization
    pub improvement_threshold: f64,
    /// Window size for performance moving averages
    pub performance_window_size: usize,
    /// Confidence threshold for making optimization decisions
    pub confidence_threshold: f64,
    /// Maximum number of optimization attempts per function
    pub max_optimization_attempts: usize,
    /// Cooldown period between optimizations (in seconds)
    pub optimization_cooldown_secs: u64,
}

impl Default for LearningParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            improvement_threshold: 0.05, // 5% improvement
            performance_window_size: 50,
            confidence_threshold: 0.8,
            max_optimization_attempts: 5,
            optimization_cooldown_secs: 60,
        }
    }
}

/// Record of an optimization event
#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    /// Function that was optimized
    pub function_id: FunctionId,
    /// Type of optimization applied
    pub optimization_type: OptimizationType,
    /// Timestamp when optimization was applied
    pub timestamp: Instant,
    /// Performance before optimization
    pub performance_before: f64,
    /// Performance after optimization
    pub performance_after: Option<f64>,
    /// Whether the optimization was successful
    pub success: Option<bool>,
    /// Configuration used for optimization
    pub config_snapshot: JITConfig,
}

/// Performance baseline for a function
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: f64,
    /// Standard deviation of execution times
    pub std_dev_execution_time_ns: f64,
    /// Throughput in operations per second
    pub throughput_ops_per_sec: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Number of samples used for baseline
    pub sample_count: usize,
    /// Confidence level in the baseline (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp when baseline was established
    pub established_at: Instant,
}

/// Trait for optimization strategies
pub trait OptimizationStrategy {
    /// Get the name of this strategy
    fn name(&self) -> &'static str;

    /// Analyze a function and suggest optimizations
    fn analyze(
        &self,
        function_id: &FunctionId,
        metrics: &FunctionPerformanceMetrics,
        baseline: Option<&PerformanceBaseline>,
        config: &JITConfig,
    ) -> Vec<OptimizationSuggestion>;

    /// Apply optimization to configuration
    fn apply(&self, suggestion: &OptimizationSuggestion, config: &mut JITConfig) -> Result<()>;

    /// Estimate the confidence in this optimization
    fn confidence(
        &self,
        suggestion: &OptimizationSuggestion,
        metrics: &FunctionPerformanceMetrics,
    ) -> f64;
}

/// SIMD optimization strategy
pub struct SIMDOptimizationStrategy;

impl OptimizationStrategy for SIMDOptimizationStrategy {
    fn name(&self) -> &'static str {
        "SIMD"
    }

    fn analyze(
        &self,
        _function_id: &FunctionId,
        metrics: &FunctionPerformanceMetrics,
        baseline: Option<&PerformanceBaseline>,
        config: &JITConfig,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Suggest SIMD if:
        // 1. SIMD is not enabled or threshold is too high
        // 2. Function has arithmetic operations
        // 3. CPU utilization is below optimal

        if !config.simd.enabled || config.simd.min_simd_size > 32 {
            if metrics.avg_cpu_utilization < 0.7 && metrics.execution_count > 10 {
                let improvement = if let Some(baseline) = baseline {
                    // Estimate improvement based on baseline
                    0.2 + (0.7 - metrics.avg_cpu_utilization) * 0.5
                } else {
                    0.25 // Default estimate
                };

                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::EnableSIMD,
                    description: "Enable SIMD vectorization for better performance".to_string(),
                    priority:
                        crate::optimized::jit::performance_monitor::OptimizationPriority::Medium,
                    estimated_improvement: improvement,
                });
            }
        }

        suggestions
    }

    fn apply(&self, suggestion: &OptimizationSuggestion, config: &mut JITConfig) -> Result<()> {
        match suggestion.suggestion_type {
            OptimizationType::EnableSIMD => {
                config.simd.enabled = true;
                config.simd.min_simd_size = 32;
                config.simd.vector_width = 32; // AVX2
                Ok(())
            }
            _ => Err(Error::InvalidOperation(
                "Invalid optimization type for SIMD strategy".to_string(),
            )),
        }
    }

    fn confidence(
        &self,
        _suggestion: &OptimizationSuggestion,
        metrics: &FunctionPerformanceMetrics,
    ) -> f64 {
        // Higher confidence for functions with more executions and consistent performance
        let execution_confidence = (metrics.execution_count.min(100) as f64 / 100.0).sqrt();
        let consistency_confidence = 1.0
            - (metrics.std_dev_execution_time_ns / metrics.avg_execution_time_ns.max(1.0)).min(1.0);

        (execution_confidence + consistency_confidence) / 2.0
    }
}

/// Parallelization optimization strategy
pub struct ParallelOptimizationStrategy;

impl OptimizationStrategy for ParallelOptimizationStrategy {
    fn name(&self) -> &'static str {
        "Parallel"
    }

    fn analyze(
        &self,
        _function_id: &FunctionId,
        metrics: &FunctionPerformanceMetrics,
        baseline: Option<&PerformanceBaseline>,
        config: &JITConfig,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Suggest parallelization if:
        // 1. CPU utilization is low
        // 2. Function execution time is significant
        // 3. Current chunk size is too large

        if metrics.avg_cpu_utilization < 0.6 && metrics.avg_execution_time_ns > 1_000_000.0 {
            // > 1ms
            if config.parallel.min_chunk_size > 1000 {
                let improvement = if let Some(baseline) = baseline {
                    0.15 + (0.6 - metrics.avg_cpu_utilization) * 0.4
                } else {
                    0.20
                };

                suggestions.push(OptimizationSuggestion {
                    suggestion_type: OptimizationType::EnableParallelization,
                    description: "Enable more aggressive parallelization".to_string(),
                    priority:
                        crate::optimized::jit::performance_monitor::OptimizationPriority::High,
                    estimated_improvement: improvement,
                });
            }
        }

        suggestions
    }

    fn apply(&self, suggestion: &OptimizationSuggestion, config: &mut JITConfig) -> Result<()> {
        match suggestion.suggestion_type {
            OptimizationType::EnableParallelization => {
                config.parallel.min_chunk_size = (config.parallel.min_chunk_size / 2).max(100);
                config.parallel.load_balancing = LoadBalancing::Dynamic;
                config.parallel.work_stealing = true;
                Ok(())
            }
            _ => Err(Error::InvalidOperation(
                "Invalid optimization type for parallel strategy".to_string(),
            )),
        }
    }

    fn confidence(
        &self,
        _suggestion: &OptimizationSuggestion,
        metrics: &FunctionPerformanceMetrics,
    ) -> f64 {
        // Higher confidence for longer-running functions
        let time_confidence = (metrics.avg_execution_time_ns / 10_000_000.0).min(1.0); // Normalize to 10ms
        let execution_confidence = (metrics.execution_count.min(50) as f64 / 50.0).sqrt();

        (time_confidence + execution_confidence) / 2.0
    }
}

/// Memory optimization strategy
pub struct MemoryOptimizationStrategy;

impl OptimizationStrategy for MemoryOptimizationStrategy {
    fn name(&self) -> &'static str {
        "Memory"
    }

    fn analyze(
        &self,
        _function_id: &FunctionId,
        metrics: &FunctionPerformanceMetrics,
        baseline: Option<&PerformanceBaseline>,
        _config: &JITConfig,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Suggest memory optimization if:
        // 1. Memory usage is high
        // 2. Performance is degrading

        if metrics.avg_memory_usage_bytes > 1024 * 1024 {
            // > 1MB
            let priority = if metrics.avg_memory_usage_bytes > 10 * 1024 * 1024 {
                crate::optimized::jit::performance_monitor::OptimizationPriority::High
            } else {
                crate::optimized::jit::performance_monitor::OptimizationPriority::Medium
            };

            let improvement = if let Some(baseline) = baseline {
                (baseline.memory_usage_bytes as f64 / metrics.avg_memory_usage_bytes as f64)
                    .min(0.5)
            } else {
                0.15
            };

            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ReduceMemoryUsage,
                description: "Optimize memory usage to improve cache locality".to_string(),
                priority,
                estimated_improvement: improvement,
            });
        }

        suggestions
    }

    fn apply(&self, _suggestion: &OptimizationSuggestion, _config: &mut JITConfig) -> Result<()> {
        // Memory optimization would typically involve cache management
        // which is handled elsewhere
        Ok(())
    }

    fn confidence(
        &self,
        _suggestion: &OptimizationSuggestion,
        metrics: &FunctionPerformanceMetrics,
    ) -> f64 {
        // Higher confidence for functions with more memory usage data
        (metrics.execution_count.min(30) as f64 / 30.0).sqrt()
    }
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new(
        monitor: Arc<JitPerformanceMonitor>,
        cache: Arc<JitFunctionCache>,
        config: JITConfig,
    ) -> Self {
        let mut strategies: Vec<Box<dyn OptimizationStrategy + Send + Sync>> = Vec::new();
        strategies.push(Box::new(SIMDOptimizationStrategy));
        strategies.push(Box::new(ParallelOptimizationStrategy));
        strategies.push(Box::new(MemoryOptimizationStrategy));

        Self {
            monitor,
            cache,
            config: Arc::new(RwLock::new(config)),
            optimization_history: RwLock::new(Vec::new()),
            performance_baselines: RwLock::new(HashMap::new()),
            learning_params: LearningParameters::default(),
            strategies,
            last_optimization: RwLock::new(Instant::now()),
        }
    }

    /// Run optimization cycle
    pub fn optimize(&self) -> Result<OptimizationReport> {
        let now = Instant::now();
        let last_opt = *self.last_optimization.read().unwrap();

        // Check cooldown
        if now.duration_since(last_opt).as_secs() < self.learning_params.optimization_cooldown_secs
        {
            return Ok(OptimizationReport::default());
        }

        let mut report = OptimizationReport::default();

        // Get functions that need optimization
        let functions_needing_optimization = self.monitor.get_functions_needing_optimization();

        for (function_id, _suggestions) in functions_needing_optimization {
            if let Some(optimization_result) = self.optimize_function(&function_id)? {
                report.optimizations_applied.push(optimization_result);
            }
        }

        // Update performance baselines
        self.update_performance_baselines()?;

        // Apply global configuration optimizations
        let config_optimizations = self.optimize_global_config()?;
        report.config_changes.extend(config_optimizations);

        *self.last_optimization.write().unwrap() = now;

        Ok(report)
    }

    /// Optimize a specific function
    fn optimize_function(&self, function_id: &FunctionId) -> Result<Option<AppliedOptimization>> {
        // Get current performance metrics
        let metrics = match self.monitor.get_function_metrics(function_id) {
            Some(metrics) => metrics,
            None => return Ok(None),
        };

        // Get performance baseline
        let baseline = self
            .performance_baselines
            .read()
            .unwrap()
            .get(function_id)
            .cloned();

        // Get current configuration
        let config = self.config.read().unwrap().clone();

        // Analyze with all strategies
        let mut all_suggestions = Vec::new();
        for strategy in &self.strategies {
            let suggestions = strategy.analyze(function_id, &metrics, baseline.as_ref(), &config);
            for suggestion in suggestions {
                let confidence = strategy.confidence(&suggestion, &metrics);
                all_suggestions.push((suggestion, confidence, strategy.name()));
            }
        }

        // Sort by confidence and estimated improvement
        all_suggestions.sort_by(|a, b| {
            let score_a = a.0.estimated_improvement * a.1;
            let score_b = b.0.estimated_improvement * b.1;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply the best optimization if confidence is high enough
        if let Some((suggestion, confidence, strategy_name)) = all_suggestions.first() {
            if *confidence >= self.learning_params.confidence_threshold
                && suggestion.estimated_improvement >= self.learning_params.improvement_threshold
            {
                // Check if we've already tried this optimization too many times
                let attempt_count =
                    self.count_optimization_attempts(function_id, suggestion.suggestion_type);
                if attempt_count >= self.learning_params.max_optimization_attempts {
                    return Ok(None);
                }

                // Apply the optimization
                let mut config = self.config.write().unwrap();
                match self.apply_optimization_by_type(suggestion, &mut config, strategy_name) {
                    Ok(_) => {
                        // Record the optimization event
                        self.record_optimization_event(OptimizationEvent {
                            function_id: function_id.clone(),
                            optimization_type: suggestion.suggestion_type,
                            timestamp: Instant::now(),
                            performance_before: metrics.get_performance_score(),
                            performance_after: None, // Will be updated later
                            success: None,
                            config_snapshot: config.clone(),
                        });

                        return Ok(Some(AppliedOptimization {
                            function_id: function_id.clone(),
                            optimization_type: suggestion.suggestion_type,
                            strategy_name: strategy_name.to_string(),
                            estimated_improvement: suggestion.estimated_improvement,
                            confidence: *confidence,
                        }));
                    }
                    Err(e) => {
                        eprintln!("Failed to apply optimization: {}", e);
                    }
                }
            }
        }

        Ok(None)
    }

    /// Apply optimization by type
    fn apply_optimization_by_type(
        &self,
        suggestion: &OptimizationSuggestion,
        config: &mut JITConfig,
        strategy_name: &str,
    ) -> Result<()> {
        for strategy in &self.strategies {
            if strategy.name() == strategy_name {
                return strategy.apply(suggestion, config);
            }
        }
        Err(Error::InvalidOperation(format!(
            "Unknown strategy: {}",
            strategy_name
        )))
    }

    /// Count optimization attempts for a function and type
    fn count_optimization_attempts(
        &self,
        function_id: &FunctionId,
        optimization_type: OptimizationType,
    ) -> usize {
        self.optimization_history
            .read()
            .unwrap()
            .iter()
            .filter(|event| {
                event.function_id == *function_id && event.optimization_type == optimization_type
            })
            .count()
    }

    /// Record an optimization event
    fn record_optimization_event(&self, event: OptimizationEvent) {
        let mut history = self.optimization_history.write().unwrap();
        history.push(event);

        // Keep only recent history
        if history.len() > 1000 {
            history.drain(0..100);
        }
    }

    /// Update performance baselines for all functions
    fn update_performance_baselines(&self) -> Result<()> {
        let top_functions = self.monitor.get_top_performing_functions(100);
        let mut baselines = self.performance_baselines.write().unwrap();

        for metrics in top_functions {
            if metrics.execution_count >= self.learning_params.performance_window_size as u64 {
                let confidence = self.calculate_baseline_confidence(&metrics);

                let baseline = PerformanceBaseline {
                    avg_execution_time_ns: metrics.avg_execution_time_ns,
                    std_dev_execution_time_ns: metrics.std_dev_execution_time_ns,
                    throughput_ops_per_sec: metrics.throughput_ops_per_sec,
                    memory_usage_bytes: metrics.avg_memory_usage_bytes,
                    cpu_utilization: metrics.avg_cpu_utilization,
                    sample_count: metrics.execution_count as usize,
                    confidence,
                    established_at: Instant::now(),
                };

                baselines.insert(metrics.function_id.clone(), baseline);
            }
        }

        Ok(())
    }

    /// Calculate confidence in a performance baseline
    fn calculate_baseline_confidence(&self, metrics: &FunctionPerformanceMetrics) -> f64 {
        let sample_confidence = (metrics.execution_count.min(100) as f64 / 100.0).sqrt();
        let consistency_confidence = 1.0
            - (metrics.std_dev_execution_time_ns / metrics.avg_execution_time_ns.max(1.0)).min(1.0);

        (sample_confidence + consistency_confidence) / 2.0
    }

    /// Optimize global configuration
    fn optimize_global_config(&self) -> Result<Vec<String>> {
        let suggestions = self.monitor.suggest_config_optimizations();
        let mut applied_changes = Vec::new();

        // Apply high-confidence suggestions
        for suggestion in suggestions {
            if suggestion.estimated_improvement >= self.learning_params.improvement_threshold {
                // Apply the suggestion (simplified implementation)
                applied_changes.push(suggestion.description);
            }
        }

        Ok(applied_changes)
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let history = self.optimization_history.read().unwrap();
        let baselines = self.performance_baselines.read().unwrap();

        let total_optimizations = history.len();
        let successful_optimizations = history
            .iter()
            .filter(|event| event.success == Some(true))
            .count();

        let avg_improvement = history
            .iter()
            .filter_map(|event| {
                if let (Some(before), Some(after)) =
                    (Some(event.performance_before), event.performance_after)
                {
                    Some((after - before) / before)
                } else {
                    None
                }
            })
            .sum::<f64>()
            / successful_optimizations.max(1) as f64;

        OptimizationStats {
            total_optimizations,
            successful_optimizations,
            success_rate: successful_optimizations as f64 / total_optimizations.max(1) as f64,
            average_improvement: avg_improvement,
            active_baselines: baselines.len(),
            last_optimization: *self.last_optimization.read().unwrap(),
        }
    }

    /// Learn from recent performance data
    pub fn learn_from_performance(&self) -> Result<LearningReport> {
        let mut report = LearningReport::default();

        // Analyze recent optimization events
        let history = self.optimization_history.read().unwrap();
        let recent_events: Vec<_> = history.iter()
            .filter(|event| event.timestamp.elapsed().as_secs() < 3600) // Last hour
            .collect();

        // Update learning parameters based on success rate
        if !recent_events.is_empty() {
            let success_rate = recent_events
                .iter()
                .filter(|event| event.success == Some(true))
                .count() as f64
                / recent_events.len() as f64;

            // Adjust confidence threshold based on success rate
            let mut learning_params = self.learning_params.clone();
            if success_rate > 0.8 {
                learning_params.confidence_threshold =
                    (learning_params.confidence_threshold - 0.05).max(0.5);
                report
                    .adjustments
                    .push("Lowered confidence threshold due to high success rate".to_string());
            } else if success_rate < 0.5 {
                learning_params.confidence_threshold =
                    (learning_params.confidence_threshold + 0.05).min(0.95);
                report
                    .adjustments
                    .push("Raised confidence threshold due to low success rate".to_string());
            }

            // Update improvement threshold based on average improvement
            let avg_improvement = recent_events
                .iter()
                .filter_map(|event| {
                    if let (Some(before), Some(after)) =
                        (Some(event.performance_before), event.performance_after)
                    {
                        Some((after - before) / before)
                    } else {
                        None
                    }
                })
                .sum::<f64>()
                / recent_events.len() as f64;

            if avg_improvement > 0.2 {
                learning_params.improvement_threshold =
                    (learning_params.improvement_threshold + 0.01).min(0.2);
                report.adjustments.push(
                    "Raised improvement threshold due to high average improvement".to_string(),
                );
            }
        }

        Ok(report)
    }
}

/// Report of optimizations applied
#[derive(Debug, Default)]
pub struct OptimizationReport {
    /// Optimizations applied to specific functions
    pub optimizations_applied: Vec<AppliedOptimization>,
    /// Global configuration changes
    pub config_changes: Vec<String>,
}

/// Applied optimization details
#[derive(Debug, Clone)]
pub struct AppliedOptimization {
    /// Function that was optimized
    pub function_id: FunctionId,
    /// Type of optimization applied
    pub optimization_type: OptimizationType,
    /// Strategy that suggested the optimization
    pub strategy_name: String,
    /// Estimated improvement
    pub estimated_improvement: f64,
    /// Confidence in the optimization
    pub confidence: f64,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total number of optimizations attempted
    pub total_optimizations: usize,
    /// Number of successful optimizations
    pub successful_optimizations: usize,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average performance improvement
    pub average_improvement: f64,
    /// Number of active performance baselines
    pub active_baselines: usize,
    /// Timestamp of last optimization
    pub last_optimization: Instant,
}

/// Learning report
#[derive(Debug, Default)]
pub struct LearningReport {
    /// Adjustments made to learning parameters
    pub adjustments: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimized::jit::cache::FunctionId;
    use std::sync::Arc;

    #[test]
    fn test_adaptive_optimizer_creation() {
        let monitor = Arc::new(JitPerformanceMonitor::new(JITConfig::default()));
        let cache = Arc::new(JitFunctionCache::new(64));
        let optimizer = AdaptiveOptimizer::new(monitor, cache, JITConfig::default());

        assert_eq!(optimizer.strategies.len(), 3); // SIMD, Parallel, Memory
    }

    #[test]
    fn test_optimization_strategies() {
        let simd_strategy = SIMDOptimizationStrategy;
        let function_id = FunctionId::new("test", "f64", "f64", "test_op", 1);
        let mut metrics =
            crate::optimized::jit::performance_monitor::FunctionPerformanceMetrics::new(
                function_id,
            );

        // Set up metrics that should trigger SIMD optimization
        metrics.avg_cpu_utilization = 0.5; // Low CPU utilization
        metrics.execution_count = 20; // Enough executions

        let config = JITConfig::default();
        let suggestions = simd_strategy.analyze(&metrics.function_id, &metrics, None, &config);

        assert!(!suggestions.is_empty());
        assert!(suggestions[0].suggestion_type == OptimizationType::EnableSIMD);
    }

    #[test]
    fn test_performance_baseline() {
        let function_id = FunctionId::new("test", "f64", "f64", "test_op", 1);
        let mut metrics =
            crate::optimized::jit::performance_monitor::FunctionPerformanceMetrics::new(
                function_id,
            );

        // Simulate consistent performance
        for _ in 0..100 {
            metrics.record_execution(1_000_000, 1024, 0.8);
        }

        let optimizer = AdaptiveOptimizer::new(
            Arc::new(JitPerformanceMonitor::new(JITConfig::default())),
            Arc::new(JitFunctionCache::new(64)),
            JITConfig::default(),
        );

        let confidence = optimizer.calculate_baseline_confidence(&metrics);
        assert!(confidence > 0.5); // Should have reasonable confidence
    }
}

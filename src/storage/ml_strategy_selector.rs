//! Machine Learning-Based Strategy Selection for Unified Memory Management
//!
//! This module implements adaptive storage strategy selection using machine learning
//! algorithms to optimize performance based on workload characteristics and
//! historical performance data.

use crate::core::error::{Error, Result};
use crate::storage::unified_manager::{
    PerformanceMonitor, StrategyMetrics, StrategySelection, StrategySelector,
};
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Workload characteristics for ML-based prediction
#[derive(Debug, Clone)]
pub struct WorkloadFeatures {
    /// Data size in bytes
    pub data_size: f64,
    /// Read/write ratio (0.0 = write-only, 1.0 = read-only)
    pub read_write_ratio: f64,
    /// Sequential access probability
    pub sequential_access_probability: f64,
    /// Data compression ratio
    pub compression_ratio: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of concurrent operations
    pub concurrency_level: f64,
    /// Data age (time since creation)
    pub data_age_hours: f64,
    /// Access frequency (operations per hour)
    pub access_frequency: f64,
    /// Data duplication factor
    pub duplication_factor: f64,
    /// Column width (for columnar data)
    pub column_width: f64,
    /// Row count
    pub row_count: f64,
    /// String content ratio (0.0 = no strings, 1.0 = all strings)
    pub string_content_ratio: f64,
}

impl WorkloadFeatures {
    pub fn new() -> Self {
        Self {
            data_size: 0.0,
            read_write_ratio: 0.5,
            sequential_access_probability: 0.5,
            compression_ratio: 1.0,
            cache_hit_rate: 0.0,
            concurrency_level: 1.0,
            data_age_hours: 0.0,
            access_frequency: 1.0,
            duplication_factor: 1.0,
            column_width: 8.0,
            row_count: 1000.0,
            string_content_ratio: 0.0,
        }
    }

    /// Extract features from storage requirements
    pub fn from_requirements(req: &StorageRequirements) -> Self {
        let mut features = Self::new();
        features.data_size = req.estimated_size as f64;

        // Map access patterns to probabilities
        features.sequential_access_probability = match req.access_pattern {
            AccessPattern::Sequential | AccessPattern::Streaming => 0.9,
            AccessPattern::Columnar => 0.7,
            AccessPattern::HighLocality => 0.6,
            AccessPattern::MediumLocality => 0.4,
            AccessPattern::LowLocality => 0.2,
            AccessPattern::Random => 0.1,
            AccessPattern::Strided { .. } => 0.5,
            _ => 0.5,
        };

        // Map concurrency levels
        features.concurrency_level = match req.concurrency {
            ConcurrencyLevel::Single => 1.0,
            ConcurrencyLevel::Low => 2.0,
            ConcurrencyLevel::Medium => 4.0,
            ConcurrencyLevel::High => 8.0,
            ConcurrencyLevel::VeryHigh => 16.0,
        };

        // Map I/O patterns to read/write ratio
        features.read_write_ratio = match req.io_pattern {
            IoPattern::ReadHeavy => 0.8,
            IoPattern::WriteHeavy => 0.2,
            IoPattern::Balanced => 0.5,
            IoPattern::AppendOnly => 0.1,
            IoPattern::UpdateInPlace => 0.4,
        };

        // Map data characteristics
        features.string_content_ratio = match req.data_characteristics {
            DataCharacteristics::Text => 1.0,
            DataCharacteristics::Mixed => 0.5,
            DataCharacteristics::Categorical => 0.8,
            _ => 0.0,
        };

        features
    }

    /// Convert features to vector for ML algorithms
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.data_size.ln().max(0.0), // Log transform for size
            self.read_write_ratio,
            self.sequential_access_probability,
            self.compression_ratio,
            self.cache_hit_rate,
            self.concurrency_level.ln().max(0.0), // Log transform
            self.data_age_hours.ln().max(0.0),    // Log transform
            self.access_frequency.ln().max(0.0),  // Log transform
            self.duplication_factor,
            self.column_width.ln().max(0.0), // Log transform
            self.row_count.ln().max(0.0),    // Log transform
            self.string_content_ratio,
        ]
    }
}

/// Performance prediction for a storage strategy
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Predicted throughput in bytes/second
    pub throughput: f64,
    /// Predicted latency in milliseconds
    pub latency: f64,
    /// Predicted memory usage in bytes
    pub memory_usage: f64,
    /// Predicted CPU usage percentage
    pub cpu_usage: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

/// Training data point for ML models
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input features
    pub features: WorkloadFeatures,
    /// Strategy that was used
    pub strategy: StorageType,
    /// Observed performance
    pub performance: PerformancePrediction,
    /// Timestamp of observation
    pub timestamp: Instant,
}

/// Log-linear regression model for performance prediction.
///
/// The model learns `ln(target)` from the (already log-transformed) feature
/// vector. Predicting raw throughput — values around `1e9` — against those
/// features with `lr = 0.01` and no normalisation diverged to `inf`/`NaN`
/// within a handful of examples, and the old `prediction.max(1.0)` clamp then
/// laundered the `NaN` into a plausible-looking `1.0`.
///
/// A model is seeded with a strategy-specific prior instead of all-zero
/// weights, so an *untrained* model predicts that prior rather than the same
/// clamped `1.0` for every strategy (which made the first candidate in the
/// iteration order always win).
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model weights (in log space)
    weights: Vec<f64>,
    /// Bias term (in log space)
    bias: f64,
    /// Number of training examples seen
    training_count: usize,
    /// Model accuracy metrics
    accuracy_metrics: AccuracyMetrics,
}

/// Largest log-space value a prediction may take before being clamped.
const MAX_LOG_PREDICTION: f64 = 60.0;
/// Largest single SGD step, in log space.
const MAX_LOG_STEP: f64 = 0.5;

impl LinearRegressionModel {
    pub fn new(feature_count: usize) -> Self {
        Self::with_prior(feature_count, 1.0)
    }

    /// Create a model that predicts `prior` before any training.
    pub fn with_prior(feature_count: usize, prior: f64) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            bias: prior.max(f64::MIN_POSITIVE).ln(),
            training_count: 0,
            accuracy_metrics: AccuracyMetrics::new(),
        }
    }

    /// Create a model that predicts `factor * exp(features[index])` before any
    /// training — used for quantities that scale with the data size.
    pub fn with_scaling_prior(feature_count: usize, index: usize, factor: f64) -> Self {
        let mut model = Self::with_prior(feature_count, factor);
        if index < model.weights.len() {
            model.weights[index] = 1.0;
        }
        model
    }

    /// Raw linear output, in log space.
    pub fn predict_log(&self, features: &[f64]) -> f64 {
        let mut prediction = self.bias;
        for (i, &feature) in features.iter().enumerate() {
            if i < self.weights.len() {
                prediction += self.weights[i] * feature;
            }
        }
        prediction
    }

    /// Predict in the original (strictly positive) units.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let log = self.predict_log(features);
        if !log.is_finite() {
            // Never launder a NaN into a plausible number: fall back to the
            // model's prior.
            return self
                .bias
                .clamp(-MAX_LOG_PREDICTION, MAX_LOG_PREDICTION)
                .exp();
        }
        log.clamp(-MAX_LOG_PREDICTION, MAX_LOG_PREDICTION).exp()
    }

    /// Train the model with a new (strictly positive) observation.
    pub fn train(&mut self, features: &[f64], target: f64, learning_rate: f64) {
        if !target.is_finite() || target <= 0.0 {
            return;
        }
        let log_target = target.ln();
        let prediction_log = self.predict_log(features);
        if !prediction_log.is_finite() || !log_target.is_finite() {
            return;
        }

        let error = log_target - prediction_log;
        // Gradient clipping: log-space residuals stay O(1), and clipping keeps a
        // single outlier from blowing the weights up.
        let step = (learning_rate * error).clamp(-MAX_LOG_STEP, MAX_LOG_STEP);

        for (i, &feature) in features.iter().enumerate() {
            if i < self.weights.len() {
                let delta = step * feature;
                if delta.is_finite() {
                    self.weights[i] += delta;
                }
            }
        }
        self.bias += step;

        self.training_count += 1;
        self.accuracy_metrics.update(
            prediction_log
                .clamp(-MAX_LOG_PREDICTION, MAX_LOG_PREDICTION)
                .exp(),
            target,
        );
    }

    /// Number of training examples this model has seen.
    pub fn training_count(&self) -> usize {
        self.training_count
    }

    /// Get model confidence based on training history.
    ///
    /// Confidence is derived from the mean *relative* error, so it is
    /// scale-free. The old formula was `1 - MAE` with the MAE measured in
    /// bytes/second, which is always hugely negative and therefore always
    /// clamped to the 0.1 floor.
    pub fn confidence(&self) -> f64 {
        if self.training_count < 10 {
            0.1
        } else {
            (1.0 - self.accuracy_metrics.mean_relative_error()).clamp(0.05, 0.95)
        }
    }
}

/// Model accuracy tracking
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Sum of absolute errors
    sum_absolute_error: f64,
    /// Sum of squared errors
    sum_squared_error: f64,
    /// Number of predictions
    prediction_count: usize,
}

impl AccuracyMetrics {
    pub fn new() -> Self {
        Self {
            sum_absolute_error: 0.0,
            sum_squared_error: 0.0,
            prediction_count: 0,
        }
    }

    /// Record one prediction/observation pair as a **relative** error.
    ///
    /// Absolute errors were meaningless here: throughput is measured in
    /// bytes/second, so an "error" of 1e6 is excellent for a 1e9 target and
    /// catastrophic for a 1e6 one.
    pub fn update(&mut self, prediction: f64, actual: f64) {
        if !prediction.is_finite() || !actual.is_finite() {
            return;
        }
        let denominator = actual.abs().max(f64::MIN_POSITIVE);
        let relative = ((prediction - actual).abs() / denominator).min(1e6);
        self.sum_absolute_error += relative;
        self.sum_squared_error += relative * relative;
        self.prediction_count += 1;
    }

    /// Mean relative error (0.0 = perfect).
    pub fn mean_relative_error(&self) -> f64 {
        if self.prediction_count > 0 {
            self.sum_absolute_error / self.prediction_count as f64
        } else {
            1.0 // Treat "no data" as fully uncertain
        }
    }

    /// Root mean squared relative error.
    pub fn root_mean_squared_relative_error(&self) -> f64 {
        if self.prediction_count > 0 {
            (self.sum_squared_error / self.prediction_count as f64).sqrt()
        } else {
            1.0
        }
    }

    /// Number of observations recorded.
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }
}

/// Prior read throughput in bytes/second for an untrained model.
///
/// These are order-of-magnitude engineering estimates for the shipped backends,
/// used only until real observations arrive; every prediction they produce is
/// reported with the corresponding low confidence.
fn prior_throughput(strategy: StorageType) -> f64 {
    match strategy {
        StorageType::InMemory => 8.0e9,
        StorageType::ColumnStore => 4.0e9,
        StorageType::StringPool => 2.0e9,
        StorageType::MemoryMapped => 1.0e9,
        StorageType::HybridLargeScale => 5.0e8,
        StorageType::DiskBased => 2.0e8,
    }
}

/// Prior read latency in milliseconds for an untrained model.
fn prior_latency_ms(strategy: StorageType) -> f64 {
    match strategy {
        StorageType::InMemory => 0.001,
        StorageType::ColumnStore => 0.005,
        StorageType::StringPool => 0.005,
        StorageType::MemoryMapped => 0.05,
        StorageType::HybridLargeScale => 0.5,
        StorageType::DiskBased => 2.0,
    }
}

/// Prior resident-memory multiple of the stored data size.
fn prior_memory_factor(strategy: StorageType) -> f64 {
    match strategy {
        StorageType::InMemory => 1.2,
        StorageType::ColumnStore => 0.5,
        StorageType::StringPool => 0.4,
        StorageType::MemoryMapped => 0.1,
        StorageType::HybridLargeScale => 0.05,
        StorageType::DiskBased => 0.02,
    }
}

/// Prior CPU cost, in percent of one core.
fn prior_cpu_percent(strategy: StorageType) -> f64 {
    match strategy {
        StorageType::InMemory => 5.0,
        StorageType::ColumnStore => 15.0,
        StorageType::StringPool => 12.0,
        StorageType::MemoryMapped => 8.0,
        StorageType::HybridLargeScale => 20.0,
        StorageType::DiskBased => 10.0,
    }
}

/// ML-based strategy selector with multiple models
pub struct MLStrategySelector {
    /// Models for predicting throughput for each strategy
    throughput_models: HashMap<StorageType, LinearRegressionModel>,
    /// Models for predicting latency for each strategy
    latency_models: HashMap<StorageType, LinearRegressionModel>,
    /// Models for predicting memory usage for each strategy
    memory_models: HashMap<StorageType, LinearRegressionModel>,
    /// Models for predicting CPU usage for each strategy
    cpu_models: HashMap<StorageType, LinearRegressionModel>,
    /// Training history
    training_data: Vec<TrainingExample>,
    /// Maximum training data to keep
    max_training_data: usize,
    /// Learning rate for model updates
    learning_rate: f64,
    /// Performance monitor the selector harvests training data from
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
}

impl MLStrategySelector {
    pub fn new(performance_monitor: Arc<Mutex<PerformanceMonitor>>) -> Self {
        let strategies = vec![
            StorageType::ColumnStore,
            StorageType::MemoryMapped,
            StorageType::StringPool,
            StorageType::HybridLargeScale,
            StorageType::DiskBased,
            StorageType::InMemory,
        ];

        let feature_count = 12; // Number of features in WorkloadFeatures::to_vector()
        let mut throughput_models = HashMap::new();
        let mut latency_models = HashMap::new();
        let mut memory_models = HashMap::new();
        let mut cpu_models = HashMap::new();

        for strategy in strategies {
            // Seed each model with a strategy-specific prior so that an
            // untrained selector still discriminates between backends.
            throughput_models.insert(
                strategy,
                LinearRegressionModel::with_prior(feature_count, prior_throughput(strategy)),
            );
            latency_models.insert(
                strategy,
                LinearRegressionModel::with_prior(feature_count, prior_latency_ms(strategy)),
            );
            // Memory scales with the data size, which is feature 0 (already
            // log-transformed), so a unit weight there reproduces
            // `data_size * factor` exactly.
            memory_models.insert(
                strategy,
                LinearRegressionModel::with_scaling_prior(
                    feature_count,
                    0,
                    prior_memory_factor(strategy),
                ),
            );
            cpu_models.insert(
                strategy,
                LinearRegressionModel::with_prior(feature_count, prior_cpu_percent(strategy)),
            );
        }

        Self {
            throughput_models,
            latency_models,
            memory_models,
            cpu_models,
            training_data: Vec::new(),
            max_training_data: 10000,
            learning_rate: 0.01,
            performance_monitor,
        }
    }

    /// Predict performance for a strategy given workload features
    pub fn predict_performance(
        &self,
        strategy: StorageType,
        features: &WorkloadFeatures,
    ) -> PerformancePrediction {
        let feature_vector = features.to_vector();

        let throughput = self
            .throughput_models
            .get(&strategy)
            .map(|model| model.predict(&feature_vector))
            .unwrap_or(1000000.0); // Default throughput

        let latency = self
            .latency_models
            .get(&strategy)
            .map(|model| model.predict(&feature_vector))
            .unwrap_or(10.0); // Default latency

        let memory_usage = self
            .memory_models
            .get(&strategy)
            .map(|model| model.predict(&feature_vector))
            .unwrap_or(features.data_size * 1.2); // Default memory overhead

        let cpu_usage = self
            .cpu_models
            .get(&strategy)
            .map(|model| model.predict(&feature_vector))
            .unwrap_or(20.0); // Default CPU usage

        let confidence = self
            .throughput_models
            .get(&strategy)
            .map(|model| model.confidence())
            .unwrap_or(0.1);

        PerformancePrediction {
            throughput,
            latency,
            memory_usage,
            cpu_usage,
            confidence,
        }
    }

    /// Select the best strategy based on ML predictions
    pub fn select_best_strategy(&self, requirements: &StorageRequirements) -> StrategySelection {
        let features = WorkloadFeatures::from_requirements(requirements);
        let mut best_strategy = StorageType::InMemory;
        let mut best_score = f64::NEG_INFINITY;
        let mut strategy_scores = Vec::new();

        for &strategy in &[
            StorageType::ColumnStore,
            StorageType::MemoryMapped,
            StorageType::StringPool,
            StorageType::HybridLargeScale,
            StorageType::DiskBased,
            StorageType::InMemory,
        ] {
            let prediction = self.predict_performance(strategy, &features);

            // Calculate composite score based on requirements
            let score = self.calculate_strategy_score(&prediction, requirements);
            strategy_scores.push((strategy, score, prediction.confidence));

            if score > best_score {
                best_score = score;
                best_strategy = strategy;
            }
        }

        // Sort strategies by score for the fallback list. `total_cmp` orders
        // NaN deterministically instead of panicking through `expect`.
        strategy_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let fallbacks: Vec<StorageType> = strategy_scores.iter()
            .skip(1) // Skip the best strategy
            .take(3) // Take top 3 alternatives
            .map(|(strategy, _, _)| *strategy)
            .collect();

        let confidence = strategy_scores
            .first()
            .map(|(_, _, conf)| *conf)
            .unwrap_or(0.1);

        StrategySelection {
            primary: best_strategy,
            fallbacks,
            confidence,
        }
    }

    /// Calculate strategy score based on predicted performance and requirements
    fn calculate_strategy_score(
        &self,
        prediction: &PerformancePrediction,
        requirements: &StorageRequirements,
    ) -> f64 {
        let mut score = 0.0;

        // Throughput component (higher is better)
        score += prediction.throughput.ln().max(0.0) * 0.3;

        // Latency component (lower is better)
        score += (1000.0 / prediction.latency.max(1.0)).ln() * 0.3;

        // Memory efficiency component
        let memory_efficiency =
            requirements.estimated_size as f64 / prediction.memory_usage.max(1.0);
        score += memory_efficiency.ln().max(0.0) * 0.2;

        // CPU efficiency component (lower CPU usage is better)
        score += (100.0 / prediction.cpu_usage.max(1.0)).ln() * 0.1;

        // Confidence component
        score += prediction.confidence.ln().max(-5.0) * 0.1;

        score
    }

    /// Add a training example from observed performance
    pub fn add_training_example(&mut self, example: TrainingExample) {
        // Add to training data
        self.training_data.push(example.clone());

        // Limit training data size
        if self.training_data.len() > self.max_training_data {
            self.training_data
                .drain(0..self.training_data.len() - self.max_training_data);
        }

        self.fit_example(&example);
    }

    /// Harvest the metrics collected so far by the shared
    /// [`PerformanceMonitor`] and turn them into training examples.
    ///
    /// The monitor used to be stored and never read, so the selector could only
    /// learn from examples a caller pushed in by hand. Each call contributes at
    /// most one example per strategy, built from that strategy's cumulative
    /// counters, so repeated calls track the running average rather than
    /// individual operations.
    ///
    /// Returns the number of examples added.
    pub fn train_from_monitor(&mut self) -> usize {
        let strategies: Vec<StorageType> = self.throughput_models.keys().copied().collect();
        let samples: Vec<(StorageType, StrategyMetrics)> = match self.performance_monitor.lock() {
            Ok(monitor) => strategies
                .into_iter()
                .filter_map(|strategy| {
                    monitor
                        .get_strategy_metrics(strategy)
                        .map(|metrics| (strategy, metrics.clone()))
                })
                .collect(),
            Err(_) => {
                log::error!("Performance monitor lock is poisoned; no training data harvested");
                return 0;
            }
        };

        let before = self.training_data.len();
        for (strategy, metrics) in samples {
            self.record_performance(strategy, &metrics);
        }
        self.training_data.len().saturating_sub(before)
    }

    /// Re-fit every model from scratch on the retained history.
    ///
    /// The models are **reset to their priors first**. The old body re-applied
    /// SGD to all 10k examples on top of the weights that per-example training
    /// had already produced, so each batch pass double-counted the whole
    /// history.
    pub fn batch_train(&mut self) {
        let feature_count = 12;
        for (&strategy, model) in self.throughput_models.iter_mut() {
            *model = LinearRegressionModel::with_prior(feature_count, prior_throughput(strategy));
        }
        for (&strategy, model) in self.latency_models.iter_mut() {
            *model = LinearRegressionModel::with_prior(feature_count, prior_latency_ms(strategy));
        }
        for (&strategy, model) in self.memory_models.iter_mut() {
            *model = LinearRegressionModel::with_scaling_prior(
                feature_count,
                0,
                prior_memory_factor(strategy),
            );
        }
        for (&strategy, model) in self.cpu_models.iter_mut() {
            *model = LinearRegressionModel::with_prior(feature_count, prior_cpu_percent(strategy));
        }

        let examples = std::mem::take(&mut self.training_data);
        for example in &examples {
            self.fit_example(example);
        }
        self.training_data = examples;
    }

    /// Apply one gradient step for every model of `example.strategy`.
    fn fit_example(&mut self, example: &TrainingExample) {
        let features = example.features.to_vector();
        let learning_rate = self.learning_rate;

        if let Some(model) = self.throughput_models.get_mut(&example.strategy) {
            model.train(&features, example.performance.throughput, learning_rate);
        }
        if let Some(model) = self.latency_models.get_mut(&example.strategy) {
            model.train(&features, example.performance.latency, learning_rate);
        }
        if let Some(model) = self.memory_models.get_mut(&example.strategy) {
            model.train(&features, example.performance.memory_usage, learning_rate);
        }
        if let Some(model) = self.cpu_models.get_mut(&example.strategy) {
            // `cpu_usage <= 0.0` means "not measured"; `train` skips it rather
            // than fitting the model to a placeholder constant.
            model.train(&features, example.performance.cpu_usage, learning_rate);
        }
    }

    /// Get model statistics for monitoring
    pub fn get_model_stats(&self) -> HashMap<StorageType, ModelStats> {
        let mut stats = HashMap::new();

        for (&strategy, model) in &self.throughput_models {
            stats.insert(
                strategy,
                ModelStats {
                    training_examples: model.training_count(),
                    confidence: model.confidence(),
                    accuracy: (1.0 - model.accuracy_metrics.mean_relative_error()).max(0.0),
                },
            );
        }

        stats
    }
}

/// Model statistics for monitoring
#[derive(Debug, Clone)]
pub struct ModelStats {
    pub training_examples: usize,
    pub confidence: f64,
    pub accuracy: f64,
}

impl StrategySelector for MLStrategySelector {
    fn select_strategy(&self, requirements: &StorageRequirements) -> StrategySelection {
        self.select_best_strategy(requirements)
    }

    fn record_performance(&mut self, strategy_type: StorageType, performance: &StrategyMetrics) {
        use crate::storage::unified_manager::OperationType;

        let read_ops = performance
            .operation_counts
            .get(&OperationType::Read)
            .copied()
            .unwrap_or(0);
        let write_ops = performance
            .operation_counts
            .get(&OperationType::Write)
            .copied()
            .unwrap_or(0)
            + performance
                .operation_counts
                .get(&OperationType::Append)
                .copied()
                .unwrap_or(0);
        let total_ops = read_ops + write_ops;
        let total_bytes: u64 = performance.bytes_processed.values().sum();
        let total_time: Duration = performance.operation_times.values().sum();

        // Real observed features rather than a struct of defaults with one
        // field filled in.
        let mut features = WorkloadFeatures::new();
        if total_ops > 0 {
            features.read_write_ratio = read_ops as f64 / total_ops as f64;
            features.data_size = (total_bytes as f64 / total_ops as f64).max(1.0);
            features.row_count = total_ops as f64;
        }
        if total_time.as_secs_f64() > 0.0 {
            features.access_frequency = (total_ops as f64 / total_time.as_secs_f64()) * 3600.0;
        }
        features.concurrency_level = 1.0;

        let read_latency_ms = performance
            .average_operation_time(OperationType::Read)
            .map(|d| d.as_secs_f64() * 1000.0);
        let throughput = performance
            .throughput(OperationType::Read)
            .or_else(|| performance.throughput(OperationType::Write));

        // Only record an example when there is something real to learn from.
        let (Some(throughput), Some(latency)) = (throughput, read_latency_ms) else {
            return;
        };
        if throughput <= 0.0 || latency <= 0.0 {
            return;
        }

        let prediction = PerformancePrediction {
            throughput,
            latency,
            memory_usage: (total_bytes as f64).max(1.0),
            // PandRS does not sample per-operation CPU time; 0.0 means
            // "not measured" and the CPU model skips it rather than being
            // trained on an invented 10%.
            cpu_usage: 0.0,
            // Confidence in this *observation*, scaled by how many operations it
            // aggregates, instead of a hardcoded 0.8.
            confidence: (total_ops as f64 / (total_ops as f64 + 20.0)).clamp(0.05, 0.95),
        };

        self.add_training_example(TrainingExample {
            features,
            strategy: strategy_type,
            performance: prediction,
            timestamp: Instant::now(),
        });
    }
}

/// Shared handle to an [`MLStrategySelector`], usable as a
/// [`StrategySelector`] inside a [`crate::storage::unified_manager::UnifiedMemoryManager`].
pub struct SharedMlSelector {
    inner: Arc<Mutex<MLStrategySelector>>,
}

impl SharedMlSelector {
    pub fn new(inner: Arc<Mutex<MLStrategySelector>>) -> Self {
        Self { inner }
    }
}

impl StrategySelector for SharedMlSelector {
    fn select_strategy(&self, requirements: &StorageRequirements) -> StrategySelection {
        match self.inner.lock() {
            Ok(selector) => selector.select_strategy(requirements),
            Err(_) => {
                log::error!("ML selector lock is poisoned; falling back to the in-memory strategy");
                StrategySelection {
                    primary: StorageType::InMemory,
                    fallbacks: vec![StorageType::ColumnStore, StorageType::DiskBased],
                    confidence: 0.0,
                }
            }
        }
    }

    fn record_performance(&mut self, strategy_type: StorageType, performance: &StrategyMetrics) {
        match self.inner.lock() {
            Ok(mut selector) => selector.record_performance(strategy_type, performance),
            Err(_) => log::error!("ML selector lock is poisoned; performance sample dropped"),
        }
    }
}

/// Adaptive ML-based unified memory manager
pub struct AdaptiveUnifiedMemoryManager {
    /// Base memory manager
    base_manager: crate::storage::unified_manager::UnifiedMemoryManager,
    /// ML-based strategy selector
    ml_selector: Arc<Mutex<MLStrategySelector>>,
    /// Adaptation interval
    adaptation_interval: Duration,
    /// Last adaptation time
    last_adaptation: Instant,
}

impl AdaptiveUnifiedMemoryManager {
    pub fn new(config: crate::storage::unified_manager::MemoryConfig) -> Self {
        let mut base_manager = crate::storage::unified_manager::UnifiedMemoryManager::new(config);
        // Share the *manager's own* monitor rather than a private one that
        // nothing ever writes to, so `adapt` can learn from real traffic.
        let performance_monitor = base_manager.monitor();
        let ml_selector = Arc::new(Mutex::new(MLStrategySelector::new(Arc::clone(
            &performance_monitor,
        ))));
        // Install the ML selector into the manager so its choice actually
        // drives storage creation. Previously `create_storage_ml` computed a
        // selection, printed it and threw it away.
        base_manager.set_selector(Box::new(SharedMlSelector::new(Arc::clone(&ml_selector))));

        debug_assert!(Arc::ptr_eq(&performance_monitor, &base_manager.monitor()));
        Self {
            base_manager,
            ml_selector,
            adaptation_interval: Duration::from_secs(300), // Adapt every 5 minutes
            last_adaptation: Instant::now(),
        }
    }

    /// Access the underlying manager (for reads, writes and deletes).
    pub fn manager(&mut self) -> &mut crate::storage::unified_manager::UnifiedMemoryManager {
        &mut self.base_manager
    }

    /// The monitor both this manager and its ML selector observe.
    pub fn performance_monitor(&self) -> Arc<Mutex<PerformanceMonitor>> {
        self.base_manager.monitor()
    }

    /// Trigger adaptive learning and model updates.
    ///
    /// Returns the per-strategy model statistics after re-fitting, so callers
    /// can log or assert on them; the old body printed them to stdout from
    /// library code.
    pub fn adapt(&mut self) -> Result<HashMap<StorageType, ModelStats>> {
        if self.last_adaptation.elapsed() < self.adaptation_interval {
            return self.get_ml_stats();
        }

        let stats = {
            let mut selector = self
                .ml_selector
                .lock()
                .map_err(|_| Error::InvalidOperation("ML selector lock is poisoned".to_string()))?;
            // Pull in whatever the manager has observed since the last pass
            // before re-fitting.
            selector.train_from_monitor();
            selector.batch_train();
            selector.get_model_stats()
        };

        for (strategy, stat) in &stats {
            log::debug!(
                "Strategy {:?}: {} examples, {:.2} confidence, {:.2} accuracy",
                strategy,
                stat.training_examples,
                stat.confidence,
                stat.accuracy
            );
        }

        self.last_adaptation = Instant::now();
        Ok(stats)
    }

    /// Create storage with ML-optimized strategy selection.
    pub fn create_storage_ml(&mut self, config: &StorageConfig) -> Result<StorageHandle> {
        // The manager's selector *is* the ML selector, so this really is an
        // ML-driven creation rather than a logged no-op.
        self.base_manager.create_storage(config)
    }

    /// Strategy the ML selector would pick for `requirements`.
    pub fn preview_selection(
        &self,
        requirements: &StorageRequirements,
    ) -> Result<StrategySelection> {
        let selector = self
            .ml_selector
            .lock()
            .map_err(|_| Error::InvalidOperation("ML selector lock is poisoned".to_string()))?;
        Ok(selector.select_best_strategy(requirements))
    }

    /// Feed observed metrics for `strategy_type` back into the ML models.
    pub fn observe(&mut self, strategy_type: StorageType) -> Result<()> {
        let metrics = self.base_manager.strategy_metrics(strategy_type)?;
        if let Some(metrics) = metrics {
            let mut selector = self
                .ml_selector
                .lock()
                .map_err(|_| Error::InvalidOperation("ML selector lock is poisoned".to_string()))?;
            selector.record_performance(strategy_type, &metrics);
        }
        Ok(())
    }

    /// Get ML selector statistics
    pub fn get_ml_stats(&self) -> Result<HashMap<StorageType, ModelStats>> {
        self.ml_selector
            .lock()
            .map(|selector| selector.get_model_stats())
            .map_err(|_| Error::InvalidOperation("Failed to acquire ML selector lock".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_features() {
        let features = WorkloadFeatures::new();
        let vector = features.to_vector();
        assert_eq!(vector.len(), 12);
    }

    #[test]
    fn test_linear_regression_model() {
        let mut model = LinearRegressionModel::new(3);

        // Train with simple data
        model.train(&[1.0, 2.0, 3.0], 6.0, 0.1);
        model.train(&[2.0, 3.0, 4.0], 9.0, 0.1);

        let prediction = model.predict(&[1.5, 2.5, 3.5]);
        assert!(prediction > 0.0);
        assert!(model.confidence() > 0.0);
    }

    #[test]
    fn test_ml_strategy_selector() {
        let monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
        let selector = MLStrategySelector::new(monitor);

        let requirements = StorageRequirements::default();
        let features = WorkloadFeatures::from_requirements(&requirements);

        let prediction = selector.predict_performance(StorageType::InMemory, &features);
        assert!(prediction.throughput > 0.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);

        let selection = selector.select_best_strategy(&requirements);
        assert!(!selection.fallbacks.is_empty());
    }

    #[test]
    fn test_adaptive_memory_manager() {
        let config = crate::storage::unified_manager::MemoryConfig::default();
        let mut manager = AdaptiveUnifiedMemoryManager::new(config);

        // Test adaptation
        assert!(manager.adapt().is_ok());

        // Test ML stats
        assert!(manager.get_ml_stats().is_ok());
    }
}

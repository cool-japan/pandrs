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

/// Simple linear regression model for performance prediction
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    /// Model weights
    weights: Vec<f64>,
    /// Bias term
    bias: f64,
    /// Number of training examples seen
    training_count: usize,
    /// Model accuracy metrics
    accuracy_metrics: AccuracyMetrics,
}

impl LinearRegressionModel {
    pub fn new(feature_count: usize) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            bias: 0.0,
            training_count: 0,
            accuracy_metrics: AccuracyMetrics::new(),
        }
    }

    /// Predict performance metrics
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut prediction = self.bias;
        for (i, &feature) in features.iter().enumerate() {
            if i < self.weights.len() {
                prediction += self.weights[i] * feature;
            }
        }
        // Ensure non-negative predictions with a reasonable default minimum
        prediction.max(1.0)
    }

    /// Train the model with a new example
    pub fn train(&mut self, features: &[f64], target: f64, learning_rate: f64) {
        let prediction = self.predict(features);
        let error = target - prediction;

        // Update weights using gradient descent
        for (i, &feature) in features.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] += learning_rate * error * feature;
            }
        }
        self.bias += learning_rate * error;

        self.training_count += 1;
        self.accuracy_metrics.update(prediction, target);
    }

    /// Get model confidence based on training history
    pub fn confidence(&self) -> f64 {
        if self.training_count < 10 {
            0.1 // Low confidence with little training data
        } else {
            (1.0 - self.accuracy_metrics.mean_absolute_error())
                .max(0.1)
                .min(0.95)
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

    pub fn update(&mut self, prediction: f64, actual: f64) {
        let error = (prediction - actual).abs();
        self.sum_absolute_error += error;
        self.sum_squared_error += error * error;
        self.prediction_count += 1;
    }

    pub fn mean_absolute_error(&self) -> f64 {
        if self.prediction_count > 0 {
            self.sum_absolute_error / self.prediction_count as f64
        } else {
            1.0 // High error when no data
        }
    }

    pub fn root_mean_squared_error(&self) -> f64 {
        if self.prediction_count > 0 {
            (self.sum_squared_error / self.prediction_count as f64).sqrt()
        } else {
            1.0 // High error when no data
        }
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
    /// Performance monitor for gathering training data
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
            throughput_models.insert(strategy, LinearRegressionModel::new(feature_count));
            latency_models.insert(strategy, LinearRegressionModel::new(feature_count));
            memory_models.insert(strategy, LinearRegressionModel::new(feature_count));
            cpu_models.insert(strategy, LinearRegressionModel::new(feature_count));
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

        // Sort strategies by score for fallback list
        strategy_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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

        // Train models
        let features = example.features.to_vector();

        if let Some(model) = self.throughput_models.get_mut(&example.strategy) {
            model.train(
                &features,
                example.performance.throughput,
                self.learning_rate,
            );
        }

        if let Some(model) = self.latency_models.get_mut(&example.strategy) {
            model.train(&features, example.performance.latency, self.learning_rate);
        }

        if let Some(model) = self.memory_models.get_mut(&example.strategy) {
            model.train(
                &features,
                example.performance.memory_usage,
                self.learning_rate,
            );
        }

        if let Some(model) = self.cpu_models.get_mut(&example.strategy) {
            model.train(&features, example.performance.cpu_usage, self.learning_rate);
        }
    }

    /// Perform batch training on historical data
    pub fn batch_train(&mut self) {
        for example in &self.training_data {
            let features = example.features.to_vector();

            if let Some(model) = self.throughput_models.get_mut(&example.strategy) {
                model.train(
                    &features,
                    example.performance.throughput,
                    self.learning_rate,
                );
            }

            if let Some(model) = self.latency_models.get_mut(&example.strategy) {
                model.train(&features, example.performance.latency, self.learning_rate);
            }

            if let Some(model) = self.memory_models.get_mut(&example.strategy) {
                model.train(
                    &features,
                    example.performance.memory_usage,
                    self.learning_rate,
                );
            }

            if let Some(model) = self.cpu_models.get_mut(&example.strategy) {
                model.train(&features, example.performance.cpu_usage, self.learning_rate);
            }
        }
    }

    /// Get model statistics for monitoring
    pub fn get_model_stats(&self) -> HashMap<StorageType, ModelStats> {
        let mut stats = HashMap::new();

        for (&strategy, model) in &self.throughput_models {
            stats.insert(
                strategy,
                ModelStats {
                    training_examples: model.training_count,
                    confidence: model.confidence(),
                    accuracy: 1.0 - model.accuracy_metrics.mean_absolute_error(),
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
        // Extract features from current workload (simplified)
        let mut features = WorkloadFeatures::new();

        // Estimate features from metrics
        if let Some(read_time) =
            performance.average_operation_time(crate::storage::unified_manager::OperationType::Read)
        {
            if let Some(write_time) = performance
                .average_operation_time(crate::storage::unified_manager::OperationType::Write)
            {
                let total_time = read_time + write_time;
                if total_time.as_nanos() > 0 {
                    features.read_write_ratio =
                        read_time.as_nanos() as f64 / total_time.as_nanos() as f64;
                }
            }
        }

        // Create performance prediction from observed metrics
        let prediction = PerformancePrediction {
            throughput: performance
                .throughput(crate::storage::unified_manager::OperationType::Read)
                .unwrap_or(0.0),
            latency: performance
                .average_operation_time(crate::storage::unified_manager::OperationType::Read)
                .unwrap_or(Duration::from_millis(1))
                .as_millis() as f64,
            memory_usage: performance.bytes_processed.values().sum::<u64>() as f64,
            cpu_usage: 10.0, // Placeholder - would need actual CPU monitoring
            confidence: 0.8,
        };

        let example = TrainingExample {
            features,
            strategy: strategy_type,
            performance: prediction,
            timestamp: Instant::now(),
        };

        self.add_training_example(example);
    }
}

/// Adaptive ML-based unified memory manager
pub struct AdaptiveUnifiedMemoryManager {
    /// Base memory manager
    base_manager: crate::storage::unified_manager::UnifiedMemoryManager,
    /// ML-based strategy selector
    ml_selector: Arc<Mutex<MLStrategySelector>>,
    /// Performance monitor
    performance_monitor: Arc<Mutex<PerformanceMonitor>>,
    /// Adaptation interval
    adaptation_interval: Duration,
    /// Last adaptation time
    last_adaptation: Instant,
}

impl AdaptiveUnifiedMemoryManager {
    pub fn new(config: crate::storage::unified_manager::MemoryConfig) -> Self {
        let performance_monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
        let ml_selector = Arc::new(Mutex::new(MLStrategySelector::new(Arc::clone(
            &performance_monitor,
        ))));
        let base_manager = crate::storage::unified_manager::UnifiedMemoryManager::new(config);

        Self {
            base_manager,
            ml_selector,
            performance_monitor,
            adaptation_interval: Duration::from_secs(300), // Adapt every 5 minutes
            last_adaptation: Instant::now(),
        }
    }

    /// Trigger adaptive learning and model updates
    pub fn adapt(&mut self) -> Result<()> {
        if self.last_adaptation.elapsed() < self.adaptation_interval {
            return Ok(());
        }

        if let Ok(mut selector) = self.ml_selector.lock() {
            // Perform batch training on accumulated data
            selector.batch_train();

            // Log adaptation metrics
            let stats = selector.get_model_stats();
            for (strategy, stat) in stats {
                println!(
                    "Strategy {:?}: {} examples, {:.2} confidence, {:.2} accuracy",
                    strategy, stat.training_examples, stat.confidence, stat.accuracy
                );
            }
        }

        self.last_adaptation = Instant::now();
        Ok(())
    }

    /// Create storage with ML-optimized strategy selection
    pub fn create_storage_ml(&mut self, config: &StorageConfig) -> Result<StorageHandle> {
        // Use ML selector to choose strategy
        if let Ok(selector) = self.ml_selector.lock() {
            let selection = selector.select_strategy(&config.requirements);
            println!(
                "ML selected strategy: {:?} with confidence {:.2}",
                selection.primary, selection.confidence
            );
        }

        // Delegate to base manager
        self.base_manager.create_storage(config)
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

//! ML-Based Storage Strategy Selection Example
//!
//! This example demonstrates the machine learning-based adaptive storage strategy
//! selection system in PandRS, showing how the system learns from workload patterns
//! and optimizes storage strategy selection automatically.

use pandrs::storage::{
    AdaptiveUnifiedMemoryManager, ConcurrencyLevel, DataCharacteristics, IoPattern,
    MLStrategySelector, MemoryConfig, PerformanceMonitor, StorageType,
    UnifiedAccessPattern as AccessPattern, UnifiedPerformancePriority as PerformancePriority,
    UnifiedStorageConfig as StorageConfig, UnifiedStorageRequirements as StorageRequirements,
    WorkloadFeatures,
};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 PandRS ML-Based Storage Strategy Selection Example");
    println!("=====================================================\n");

    // Create adaptive memory manager with ML-based strategy selection
    let config = MemoryConfig {
        max_memory: Some(1024 * 1024 * 1024), // 1GB limit
        adaptive_optimization: true,
        strategy_selection: pandrs::storage::StrategySelectionAlgorithm::MachineLearning,
        ..Default::default()
    };

    let mut adaptive_manager = AdaptiveUnifiedMemoryManager::new(config);
    println!("✅ Created adaptive memory manager with ML strategy selection\n");

    // Demonstrate workload feature extraction
    demonstrate_workload_features()?;

    // Demonstrate ML strategy selection
    demonstrate_ml_strategy_selection()?;

    // Demonstrate adaptive learning
    demonstrate_adaptive_learning(&mut adaptive_manager)?;

    // Demonstrate performance prediction
    demonstrate_performance_prediction()?;

    println!("\n🎯 ML-based strategy selection example completed successfully!");
    Ok(())
}

fn demonstrate_workload_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Workload Feature Extraction");
    println!("------------------------------");

    // Create different workload scenarios
    let scenarios = vec![
        (
            "Large Dataset Analytics",
            StorageRequirements {
                estimated_size: 1024 * 1024 * 1024, // 1GB
                data_characteristics: DataCharacteristics::Numeric,
                access_pattern: AccessPattern::Sequential,
                performance_priority: PerformancePriority::Throughput,
                io_pattern: IoPattern::ReadHeavy,
                concurrency: ConcurrencyLevel::High,
                ..Default::default()
            },
        ),
        (
            "Real-time String Processing",
            StorageRequirements {
                estimated_size: 10 * 1024 * 1024, // 10MB
                data_characteristics: DataCharacteristics::Text,
                access_pattern: AccessPattern::Random,
                performance_priority: PerformancePriority::Latency,
                io_pattern: IoPattern::Balanced,
                concurrency: ConcurrencyLevel::Medium,
                ..Default::default()
            },
        ),
        (
            "Time Series Data",
            StorageRequirements {
                estimated_size: 100 * 1024 * 1024, // 100MB
                data_characteristics: DataCharacteristics::TimeSeries,
                access_pattern: AccessPattern::Streaming,
                performance_priority: PerformancePriority::Speed,
                io_pattern: IoPattern::AppendOnly,
                concurrency: ConcurrencyLevel::Low,
                ..Default::default()
            },
        ),
    ];

    for (name, requirements) in scenarios {
        let features = WorkloadFeatures::from_requirements(&requirements);
        println!("🔍 Scenario: {}", name);
        println!(
            "   Data size: {:.1} MB",
            features.data_size / (1024.0 * 1024.0)
        );
        println!("   Read/Write ratio: {:.2}", features.read_write_ratio);
        println!(
            "   Sequential probability: {:.2}",
            features.sequential_access_probability
        );
        println!("   Concurrency level: {:.1}", features.concurrency_level);
        println!(
            "   String content ratio: {:.2}",
            features.string_content_ratio
        );

        let feature_vector = features.to_vector();
        println!("   Feature vector length: {}", feature_vector.len());
        println!();
    }

    Ok(())
}

fn demonstrate_ml_strategy_selection() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 ML Strategy Selection");
    println!("------------------------");

    let performance_monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
    let selector = MLStrategySelector::new(performance_monitor);

    // Test different workload types
    let workloads = vec![
        (
            "Small Frequent Reads",
            StorageRequirements {
                estimated_size: 1024 * 1024, // 1MB
                access_pattern: AccessPattern::Random,
                performance_priority: PerformancePriority::Speed,
                io_pattern: IoPattern::ReadHeavy,
                ..Default::default()
            },
        ),
        (
            "Large Batch Processing",
            StorageRequirements {
                estimated_size: 512 * 1024 * 1024, // 512MB
                access_pattern: AccessPattern::Sequential,
                performance_priority: PerformancePriority::Memory,
                io_pattern: IoPattern::Balanced,
                ..Default::default()
            },
        ),
        (
            "String Analytics",
            StorageRequirements {
                estimated_size: 50 * 1024 * 1024, // 50MB
                data_characteristics: DataCharacteristics::Text,
                access_pattern: AccessPattern::Random,
                performance_priority: PerformancePriority::Memory,
                ..Default::default()
            },
        ),
    ];

    for (name, requirements) in workloads {
        println!("📋 Workload: {}", name);
        let selection = selector.select_best_strategy(&requirements);

        println!("   🎯 Primary strategy: {:?}", selection.primary);
        println!("   📋 Fallback strategies: {:?}", selection.fallbacks);
        println!("   🎯 Confidence: {:.2}%", selection.confidence * 100.0);

        // Show performance predictions for different strategies
        let features = WorkloadFeatures::from_requirements(&requirements);
        for strategy_type in [
            StorageType::InMemory,
            StorageType::ColumnStore,
            StorageType::StringPool,
        ] {
            let prediction = selector.predict_performance(strategy_type, &features);
            println!(
                "   📊 {:?}: throughput={:.0} MB/s, latency={:.1}ms, confidence={:.2}",
                strategy_type,
                prediction.throughput / (1024.0 * 1024.0),
                prediction.latency,
                prediction.confidence
            );
        }
        println!();
    }

    Ok(())
}

fn demonstrate_adaptive_learning(
    adaptive_manager: &mut AdaptiveUnifiedMemoryManager,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Adaptive Learning");
    println!("-------------------");

    // Simulate different storage configurations
    let test_configs = vec![
        (
            "Small Cache Data",
            StorageConfig {
                requirements: StorageRequirements {
                    estimated_size: 512 * 1024, // 512KB
                    access_pattern: AccessPattern::Random,
                    performance_priority: PerformancePriority::Speed,
                    data_characteristics: DataCharacteristics::Numeric,
                    io_pattern: IoPattern::ReadHeavy,
                    concurrency: ConcurrencyLevel::Low,
                    memory_limit: Some(1024 * 1024),
                    ..Default::default()
                },
                options: std::collections::HashMap::new(),
                data_sample: None,
                expected_access_pattern: AccessPattern::Random,
                constraints: Default::default(),
            },
        ),
        (
            "Large Sequential Data",
            StorageConfig {
                requirements: StorageRequirements {
                    estimated_size: 64 * 1024 * 1024, // 64MB
                    access_pattern: AccessPattern::Sequential,
                    performance_priority: PerformancePriority::Memory,
                    data_characteristics: DataCharacteristics::Numeric,
                    io_pattern: IoPattern::Balanced,
                    concurrency: ConcurrencyLevel::Medium,
                    memory_limit: Some(128 * 1024 * 1024),
                    ..Default::default()
                },
                options: std::collections::HashMap::new(),
                data_sample: None,
                expected_access_pattern: AccessPattern::Sequential,
                constraints: Default::default(),
            },
        ),
    ];

    for (name, config) in test_configs {
        println!("🧪 Testing configuration: {}", name);

        // Create storage using ML-optimized selection
        match adaptive_manager.create_storage_ml(&config) {
            Ok(handle) => {
                println!("   ✅ Storage created with ID: {:?}", handle.id);
                println!("   📊 Strategy type: {:?}", handle.strategy_type);
            }
            Err(e) => {
                println!("   ❌ Storage creation failed: {}", e);
            }
        }
    }

    // Trigger adaptation
    println!("\n🔄 Triggering adaptive learning...");
    adaptive_manager.adapt()?;

    // Get ML statistics
    if let Ok(stats) = adaptive_manager.get_ml_stats() {
        println!("📈 ML Model Statistics:");
        for (strategy, stat) in stats {
            println!(
                "   {:?}: {} examples, {:.2} confidence, {:.2} accuracy",
                strategy, stat.training_examples, stat.confidence, stat.accuracy
            );
        }
    }

    Ok(())
}

fn demonstrate_performance_prediction() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Performance Prediction");
    println!("-------------------------");

    let performance_monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
    let selector = MLStrategySelector::new(performance_monitor);

    // Create test workload
    let workload = WorkloadFeatures {
        data_size: 32.0 * 1024.0 * 1024.0,  // 32MB
        read_write_ratio: 0.7,              // Read-heavy
        sequential_access_probability: 0.8, // Mostly sequential
        compression_ratio: 0.6,             // Good compression
        cache_hit_rate: 0.4,                // Moderate cache hits
        concurrency_level: 4.0,             // Medium concurrency
        data_age_hours: 2.0,                // Recent data
        access_frequency: 100.0,            // High access frequency
        duplication_factor: 1.2,            // Some duplication
        column_width: 8.0,                  // 64-bit columns
        row_count: 1000000.0,               // 1M rows
        string_content_ratio: 0.3,          // Mixed content
    };

    println!("📊 Workload Characteristics:");
    println!(
        "   Data size: {:.1} MB",
        workload.data_size / (1024.0 * 1024.0)
    );
    println!(
        "   Read/Write ratio: {:.1}%/{:.1}%",
        workload.read_write_ratio * 100.0,
        (1.0 - workload.read_write_ratio) * 100.0
    );
    println!(
        "   Sequential access: {:.1}%",
        workload.sequential_access_probability * 100.0
    );
    println!("   Cache hit rate: {:.1}%", workload.cache_hit_rate * 100.0);
    println!();

    println!("🔮 Performance Predictions by Strategy:");
    let strategies = [
        StorageType::InMemory,
        StorageType::ColumnStore,
        StorageType::MemoryMapped,
        StorageType::StringPool,
        StorageType::DiskBased,
    ];

    for strategy in strategies {
        let prediction = selector.predict_performance(strategy, &workload);
        println!("   {:?}:", strategy);
        println!(
            "     📈 Throughput: {:.1} MB/s",
            prediction.throughput / (1024.0 * 1024.0)
        );
        println!("     ⏱️  Latency: {:.2} ms", prediction.latency);
        println!(
            "     💾 Memory usage: {:.1} MB",
            prediction.memory_usage / (1024.0 * 1024.0)
        );
        println!("     🖥️  CPU usage: {:.1}%", prediction.cpu_usage);
        println!("     🎯 Confidence: {:.2}", prediction.confidence);
        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_example() {
        // This test ensures the example compiles and basic functionality works
        let config = MemoryConfig::default();
        let adaptive_manager = AdaptiveUnifiedMemoryManager::new(config);

        // Basic smoke test
        assert!(adaptive_manager.get_ml_stats().is_ok());
    }

    #[test]
    fn test_workload_features_extraction() {
        let requirements = StorageRequirements {
            estimated_size: 1024 * 1024,
            data_characteristics: DataCharacteristics::Text,
            access_pattern: AccessPattern::Sequential,
            ..Default::default()
        };

        let features = WorkloadFeatures::from_requirements(&requirements);
        assert_eq!(features.data_size, 1024.0 * 1024.0);
        assert_eq!(features.string_content_ratio, 1.0);
        assert_eq!(features.sequential_access_probability, 0.9);

        let vector = features.to_vector();
        assert_eq!(vector.len(), 12);
    }

    #[test]
    fn test_ml_strategy_selection() {
        let performance_monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
        let selector = MLStrategySelector::new(performance_monitor);

        let requirements = StorageRequirements {
            estimated_size: 1024 * 1024,
            data_characteristics: DataCharacteristics::Text,
            access_pattern: AccessPattern::Random,
            performance_priority: PerformancePriority::Speed,
            ..Default::default()
        };

        let selection = selector.select_best_strategy(&requirements);
        assert!(!selection.fallbacks.is_empty());
        assert!(selection.confidence >= 0.0 && selection.confidence <= 1.0);
    }
}

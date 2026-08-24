//! Hybrid Large-Scale Strategy Implementation.
//!
//! Automatic hot/warm/cold data tiering with multi-backend storage. The warm
//! and cold tiers are genuinely file-backed (see [`tiers`]); the module's whole
//! value proposition — moving cold data out of RAM — depends on that, and it
//! used to be simulated with in-memory `HashMap`s plus `thread::sleep`.

pub mod config;
pub mod manager;
pub mod strategy;
pub mod tiers;

pub use config::{
    AccessPattern, AccessPatternType, CompressionState, DataTier, HybridConfig, TierConfig,
    TierHistoryEntry, TierMoveReason, TierStatistics, TierStorageType, TieredDataMetadata,
    TieringReport, TieringScheduler,
};
pub use manager::TierManager;
pub use strategy::{HybridEntry, HybridHandle, HybridLargeScaleStrategy, HybridStatistics};
pub use tiers::{
    decode_chunk, encode_chunk, make_backend, DataId, FileTierBackend, HDDTierBackend,
    InMemoryTierBackend, SSDTierBackend, TierBackend, TierStorageInfo,
};

/// Tiered data entry.
///
/// Retained for API compatibility with earlier releases; the tier manager
/// tracks placement, access patterns and metadata in separate indexes.
#[derive(Debug, Clone)]
pub struct TieredDataEntry {
    /// Unique identifier
    pub id: DataId,
    /// Current tier location
    pub current_tier: DataTier,
    /// Data chunk
    pub chunk: crate::storage::unified_memory::DataChunk,
    /// Access pattern tracking
    pub access_pattern: AccessPattern,
    /// Storage metadata
    pub metadata: TieredDataMetadata,
    /// Compression state
    pub compression_state: CompressionState,
}

#[cfg(test)]
mod tests {
    use super::*;
    // `AccessPattern` is deliberately ambiguous here: the hybrid module has a
    // per-chunk tracking struct, the strategy layer has a workload enum. Import
    // the strategy-layer items explicitly and reach for the tracking struct via
    // `config::AccessPattern`.
    use crate::storage::unified_memory::{
        AccessPattern as WorkloadPattern, ChunkRange, CompressionType, DataChunk,
        PerformancePriority, StorageConfig, StorageRequirements, StorageStrategy,
    };
    use std::time::Duration;

    fn small_tier_config() -> HybridConfig {
        let mut config = HybridConfig::default();
        config.hot_tier.max_size = 16 * 1024;
        config.warm_tier.max_size = 1024 * 1024;
        config.cold_tier.max_size = 8 * 1024 * 1024;
        config
    }

    fn storage_config(size: usize) -> StorageConfig {
        StorageConfig {
            requirements: StorageRequirements {
                estimated_size: size,
                access_pattern: WorkloadPattern::Random,
                performance_priority: PerformancePriority::Balanced,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn data_ids_are_monotonic_not_random() {
        let mut manager = TierManager::try_new(HybridConfig::default()).expect("manager");
        let a = manager
            .store_data(DataChunk::new_test_data(64))
            .expect("store");
        let b = manager
            .store_data(DataChunk::new_test_data(64))
            .expect("store");
        assert_eq!(a, DataId(1));
        assert_eq!(b, DataId(2));
    }

    #[test]
    fn write_then_read_roundtrip() {
        // The headline bug: `write_chunk` discarded the DataId and `read_chunk`
        // invented `DataId(range.start)`, so written data was unrecoverable.
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");

        let payload: Vec<u8> = (0..2048u32).map(|i| (i % 251) as u8).collect();
        strategy
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write");
        assert_eq!(handle.row_count(), payload.len());

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, payload.len()))
            .expect("read");
        assert_eq!(read.data, payload);
    }

    #[test]
    fn reads_honour_row_ranges_across_chunks() {
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");

        for block in 0..4u32 {
            let payload: Vec<u8> = (0..100u32)
                .map(|i| ((i + block * 100) % 256) as u8)
                .collect();
            strategy
                .write_chunk(&handle, DataChunk::new(payload))
                .expect("write");
        }
        assert_eq!(handle.row_count(), 400);

        let full: Vec<u8> = (0..4u32)
            .flat_map(|block| (0..100u32).map(move |i| ((i + block * 100) % 256) as u8))
            .collect();

        for (start, end) in [(0usize, 400usize), (50, 150), (399, 400), (120, 121)] {
            let read = strategy
                .read_chunk(&handle, ChunkRange::new(start, end))
                .expect("read");
            assert_eq!(
                read.data,
                full[start..end].to_vec(),
                "range {}..{}",
                start,
                end
            );
        }
    }

    #[test]
    fn string_chunks_roundtrip() {
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");

        let strings: Vec<String> = vec![
            "ascii".to_string(),
            "日本語".to_string(),
            "with\0nul".to_string(),
            String::new(),
        ];
        strategy
            .write_chunk(&handle, DataChunk::from_strings(strings.clone()))
            .expect("write");

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, strings.len()))
            .expect("read");
        assert_eq!(read.as_strings().expect("decode"), strings);
    }

    #[test]
    fn warm_and_cold_tiers_are_file_backed() {
        let mut config = HybridConfig::default();
        // Force everything past the hot tier.
        config.hot_tier.max_size = 1;
        let mut manager = TierManager::try_new(config).expect("manager");

        let id = manager
            .store_data(DataChunk::new(vec![3u8; 4096]))
            .expect("store");
        let tier = manager.tier_of(id).expect("tier");
        assert_ne!(tier, DataTier::Hot, "data should have spilled out of RAM");
        assert_eq!(
            manager.retrieve_data(id).expect("retrieve").data,
            vec![3u8; 4096]
        );
        // Physical bytes come from real files on the tier backend.
        assert!(manager.physical_bytes() > 0);
    }

    #[test]
    fn tier_capacity_is_enforced() {
        let mut config = HybridConfig::default();
        config.hot_tier.max_size = 512;
        config.warm_tier.max_size = 512;
        config.cold_tier.max_size = 512;
        let mut manager = TierManager::try_new(config).expect("manager");

        // Each 400-byte chunk fills one tier; the placement logic spills down
        // the hierarchy rather than overfilling the hot tier.
        let hot = manager
            .store_data(DataChunk::new(vec![1u8; 400]))
            .expect("first fits the hot tier");
        assert_eq!(manager.tier_of(hot), Some(DataTier::Hot));
        let warm = manager
            .store_data(DataChunk::new(vec![1u8; 400]))
            .expect("second spills to warm");
        assert_eq!(manager.tier_of(warm), Some(DataTier::Warm));
        let cold = manager
            .store_data(DataChunk::new(vec![1u8; 400]))
            .expect("third spills to cold");
        assert_eq!(manager.tier_of(cold), Some(DataTier::Cold));

        // Nothing is left; the old store_data never checked capacity at all and
        // would have silently overfilled the hot tier.
        assert!(manager.store_data(DataChunk::new(vec![1u8; 400])).is_err());
        for tier in [DataTier::Hot, DataTier::Warm, DataTier::Cold] {
            let stats = manager.get_tier_statistics().get(&tier).expect("stats");
            assert!(
                stats.current_usage <= stats.max_capacity,
                "{:?} tier overfilled: {} > {}",
                tier,
                stats.current_usage,
                stats.max_capacity
            );
        }
    }

    #[test]
    fn cold_compression_flag_is_honoured() {
        let mut config = HybridConfig::default();
        config.enable_cold_compression = false;
        let normalized = config.normalized();
        assert_eq!(normalized.cold_tier.compression, CompressionType::None);
    }

    #[test]
    fn delete_storage_clears_everything() {
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");
        strategy
            .write_chunk(&handle, DataChunk::new(vec![1u8; 256]))
            .expect("write");
        assert_eq!(handle.entries().expect("entries").len(), 1);

        strategy.delete_storage(&handle).expect("delete");
        assert!(handle.entries().expect("entries").is_empty());
        assert_eq!(handle.row_count(), 0);
    }

    #[test]
    fn flush_is_fast_without_simulated_latency() {
        // `flush()` used to run background tiering whose SSD/HDD backends slept
        // 100us/50us/5ms/10ms per chunk. 64 chunks would take about a second.
        let mut strategy = HybridLargeScaleStrategy::new(small_tier_config());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");
        for _ in 0..64 {
            strategy
                .write_chunk(&handle, DataChunk::new(vec![2u8; 128]))
                .expect("write");
        }
        let started = std::time::Instant::now();
        strategy.flush(&handle).expect("flush");
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "flush took {:?}; simulated sleeps are back",
            started.elapsed()
        );
    }

    #[test]
    fn capacity_pressure_demotes_until_relieved() {
        let mut config = HybridConfig::default();
        config.hot_tier.max_size = 4096;
        config.warm_tier.max_size = 1024 * 1024;
        config.demotion_threshold = Duration::from_secs(24 * 3600);
        let mut manager = TierManager::try_new(config).expect("manager");

        for _ in 0..30 {
            manager
                .store_data(DataChunk::new(vec![1u8; 128]))
                .expect("store");
        }
        let hot_before = manager
            .get_tier_statistics()
            .get(&DataTier::Hot)
            .map(|s| s.utilization())
            .unwrap_or(0.0);
        assert!(hot_before > 0.9, "hot tier should be under pressure");

        manager.run_background_tiering().expect("tiering");
        let hot_after = manager
            .get_tier_statistics()
            .get(&DataTier::Hot)
            .map(|s| s.utilization())
            .unwrap_or(0.0);
        assert!(
            hot_after <= 0.9,
            "capacity pressure not relieved: {} -> {}",
            hot_before,
            hot_after
        );
    }

    #[test]
    fn moved_data_stays_readable_and_single_copy() {
        let mut manager = TierManager::try_new(HybridConfig::default()).expect("manager");
        let id = manager
            .store_data(DataChunk::new(vec![8u8; 1024]))
            .expect("store");
        assert_eq!(manager.tier_of(id), Some(DataTier::Hot));

        manager.relocate(id, DataTier::Cold).expect("relocate");
        assert_eq!(manager.tier_of(id), Some(DataTier::Cold));
        assert_eq!(
            manager.retrieve_data(id).expect("retrieve").data,
            vec![8u8; 1024]
        );

        let hot_usage = manager
            .get_tier_statistics()
            .get(&DataTier::Hot)
            .map(|s| s.current_usage)
            .unwrap_or(usize::MAX);
        assert_eq!(hot_usage, 0, "source tier still accounts for moved data");
    }

    #[test]
    fn statistics_report_real_values() {
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");
        strategy
            .write_chunk(&handle, DataChunk::new(vec![4u8; 1024]))
            .expect("write");
        let _ = strategy
            .read_chunk(&handle, ChunkRange::new(0, 1024))
            .expect("read");
        strategy.refresh_tier_stats(&handle).expect("refresh");

        let stats = strategy.storage_stats();
        assert_eq!(stats.write_operations, 1);
        assert_eq!(stats.read_operations, 1);
        assert!(stats.cache_hit_rate > 0.0, "tier hit rates never populated");
    }

    #[test]
    fn access_pattern_tracking() {
        let mut pattern = config::AccessPattern::new(1024);
        assert_eq!(pattern.access_count, 1);
        assert_eq!(pattern.data_size, 1024);
        pattern.record_access();
        assert_eq!(pattern.access_count, 2);
    }

    #[test]
    fn tier_statistics_utilization() {
        let mut stats = TierStatistics::new(1024 * 1024);
        assert_eq!(stats.utilization(), 0.0);
        stats.current_usage = 512 * 1024;
        assert_eq!(stats.utilization(), 0.5);
        assert_eq!(stats.available_space(), 512 * 1024);
    }

    #[test]
    fn capability_assessment() {
        let strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let requirements = StorageRequirements {
            estimated_size: 1024 * 1024 * 1024,
            access_pattern: WorkloadPattern::Random,
            performance_priority: PerformancePriority::Balanced,
            ..Default::default()
        };
        let capability = strategy.can_handle(&requirements);
        assert!(capability.can_handle);
        assert!(capability.confidence > 0.9);
        assert!(capability.performance_score > 0.9);
    }

    #[test]
    fn retuning_a_handle_changes_its_thresholds() {
        let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
        let handle = strategy
            .create_storage(&storage_config(1024 * 1024))
            .expect("create");
        let capacity_before = handle
            .tier_statistics()
            .expect("stats")
            .get(&DataTier::Hot)
            .map(|s| s.max_capacity)
            .unwrap_or(0);
        assert!(capacity_before > 0);

        // `optimize_for_pattern` used to mutate only the strategy config while
        // handles kept their own clone, so it had no observable effect.
        strategy
            .retune_handle(&handle, WorkloadPattern::LowLocality)
            .expect("retune");
        let capacity_after = handle
            .tier_statistics()
            .expect("stats")
            .get(&DataTier::Hot)
            .map(|s| s.max_capacity)
            .unwrap_or(0);
        assert_eq!(capacity_after, capacity_before / 2);
    }

    #[test]
    fn tiering_scheduler_marks_runs() {
        let mut scheduler = TieringScheduler::new(Duration::from_millis(0));
        assert!(scheduler.should_run());
        scheduler.mark_run();
    }
}

//! `HybridLargeScaleStrategy`: row-addressed storage across hot/warm/cold tiers.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::config::{DataTier, HybridConfig, TierStatistics};
use super::manager::TierManager;
use super::tiers::DataId;

/// One stored chunk, together with the row range it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridEntry {
    /// Identifier assigned by the tier manager
    pub id: DataId,
    /// First logical row (inclusive)
    pub row_start: usize,
    /// Number of logical rows
    pub row_count: usize,
    /// Layout of the rows in this chunk
    pub layout: ChunkLayout,
}

impl HybridEntry {
    fn row_end(&self) -> usize {
        self.row_start + self.row_count
    }
}

/// Hybrid large-scale storage handle.
///
/// The range index is what makes written data readable: `write_chunk` used to
/// throw the `DataId` away and `read_chunk` fabricated `DataId(range.start)`,
/// so nothing written could ever be retrieved.
#[derive(Debug, Clone)]
pub struct HybridHandle {
    /// Configuration
    pub config: HybridConfig,
    /// Tier manager for data management
    pub tier_manager: Arc<Mutex<TierManager>>,
    /// Ordered index of stored chunks
    entries: Arc<Mutex<Vec<HybridEntry>>>,
    /// Total rows written through this handle
    row_cursor: Arc<AtomicU64>,
    /// Handle statistics
    pub statistics: HybridStatistics,
}

impl HybridHandle {
    fn new(config: HybridConfig, tier_manager: Arc<Mutex<TierManager>>) -> Self {
        Self {
            config,
            tier_manager,
            entries: Arc::new(Mutex::new(Vec::new())),
            row_cursor: Arc::new(AtomicU64::new(0)),
            statistics: HybridStatistics::new(),
        }
    }

    fn lock_entries(&self) -> Result<std::sync::MutexGuard<'_, Vec<HybridEntry>>> {
        self.entries.lock().map_err(|_| {
            Error::InvalidOperation("Hybrid handle entry index lock is poisoned".to_string())
        })
    }

    fn lock_manager(&self) -> Result<std::sync::MutexGuard<'_, TierManager>> {
        self.tier_manager
            .lock()
            .map_err(|_| Error::InvalidOperation("Tier manager lock is poisoned".to_string()))
    }

    /// Every chunk written through this handle, in write order.
    pub fn entries(&self) -> Result<Vec<HybridEntry>> {
        Ok(self.lock_entries()?.clone())
    }

    /// Total rows written through this handle.
    pub fn row_count(&self) -> usize {
        self.row_cursor.load(Ordering::SeqCst) as usize
    }

    fn record_entry(&self, entry: HybridEntry) -> Result<()> {
        self.lock_entries()?.push(entry);
        Ok(())
    }

    fn reserve_rows(&self, rows: usize) -> usize {
        self.row_cursor.fetch_add(rows as u64, Ordering::SeqCst) as usize
    }

    fn clear_index(&self) -> Result<()> {
        self.lock_entries()?.clear();
        self.row_cursor.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Per-tier statistics for the storage behind this handle.
    pub fn tier_statistics(&self) -> Result<HashMap<DataTier, TierStatistics>> {
        Ok(self.lock_manager()?.get_tier_statistics().clone())
    }
}

/// Hybrid strategy statistics
#[derive(Debug, Clone)]
pub struct HybridStatistics {
    /// Total logical data stored
    pub total_data_size: usize,
    /// Number of chunks stored
    pub chunk_count: u64,
    /// Distribution across tiers
    pub tier_distribution: HashMap<DataTier, usize>,
    /// Total tier movements
    pub total_movements: u64,
    /// Average access latency across reads
    pub avg_access_latency: Duration,
    /// Cache hit rates by tier
    pub tier_hit_rates: HashMap<DataTier, f64>,
    /// Total nanoseconds spent in reads (for a real average)
    pub total_read_nanos: u64,
    /// Number of reads served
    pub read_count: u64,
}

impl HybridStatistics {
    pub fn new() -> Self {
        Self {
            total_data_size: 0,
            chunk_count: 0,
            tier_distribution: HashMap::new(),
            total_movements: 0,
            avg_access_latency: Duration::ZERO,
            tier_hit_rates: HashMap::new(),
            total_read_nanos: 0,
            read_count: 0,
        }
    }

    fn record_read(&mut self, elapsed: Duration) {
        self.read_count += 1;
        self.total_read_nanos = self
            .total_read_nanos
            .saturating_add(elapsed.as_nanos() as u64);
        // A real running average; this used to be a plain assignment of the
        // most recent single read.
        self.avg_access_latency = Duration::from_nanos(self.total_read_nanos / self.read_count);
    }
}

impl Default for HybridStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid Large-Scale Strategy Implementation
pub struct HybridLargeScaleStrategy {
    config: HybridConfig,
    global_stats: Arc<Mutex<HybridStatistics>>,
    next_handle_id: AtomicU64,
}

impl HybridLargeScaleStrategy {
    pub fn new(config: HybridConfig) -> Self {
        Self {
            config: config.normalized(),
            global_stats: Arc::new(Mutex::new(HybridStatistics::new())),
            next_handle_id: AtomicU64::new(1),
        }
    }

    /// Number of handles created by this strategy.
    pub fn handles_created(&self) -> u64 {
        self.next_handle_id.load(Ordering::SeqCst).saturating_sub(1)
    }

    fn lock_stats(&self) -> Result<std::sync::MutexGuard<'_, HybridStatistics>> {
        self.global_stats
            .lock()
            .map_err(|_| Error::InvalidOperation("Hybrid statistics lock is poisoned".to_string()))
    }

    fn estimate_storage_requirements(&self, config: &StorageConfig) -> Result<HybridConfig> {
        let mut hybrid_config = self.config.clone();
        let estimated_size = config.requirements.estimated_size;

        if estimated_size > 100 * 1024 * 1024 * 1024 {
            hybrid_config.hot_tier.max_size = (estimated_size / 100).max(1024 * 1024 * 1024);
            hybrid_config.warm_tier.max_size = (estimated_size / 10).max(10 * 1024 * 1024 * 1024);
            hybrid_config.cold_tier.max_size = estimated_size;
        }

        // All arithmetic below is saturating: the previous `*= 2` / `*= 3` /
        // `/= 2` could overflow `usize` or drive a tier size to zero.
        match config.requirements.access_pattern {
            AccessPattern::Sequential => {
                hybrid_config.promotion_threshold =
                    (hybrid_config.promotion_threshold * 0.5).max(f64::MIN_POSITIVE);
                hybrid_config.demotion_threshold =
                    hybrid_config.demotion_threshold.saturating_mul(2);
            }
            AccessPattern::Random => {
                hybrid_config.hot_tier.max_size = hybrid_config.hot_tier.max_size.saturating_mul(2);
                hybrid_config.promotion_threshold *= 2.0;
            }
            AccessPattern::Streaming => {
                hybrid_config.enable_auto_tiering = false;
                hybrid_config.warm_tier.max_size =
                    estimated_size.max(hybrid_config.warm_tier.max_size);
            }
            _ => {}
        }

        Ok(hybrid_config.normalized())
    }
}

impl StorageStrategy for HybridLargeScaleStrategy {
    type Handle = HybridHandle;
    type Error = Error;
    type Metadata = HybridStatistics;

    fn name(&self) -> &'static str {
        "HybridLargeScale"
    }

    fn create_storage(&mut self, config: &StorageConfig) -> Result<Self::Handle> {
        let optimized_config = self.estimate_storage_requirements(config)?;
        let tier_manager = Arc::new(Mutex::new(TierManager::try_new(optimized_config.clone())?));
        self.next_handle_id.fetch_add(1, Ordering::SeqCst);
        Ok(HybridHandle::new(optimized_config, tier_manager))
    }

    fn read_chunk(&self, handle: &Self::Handle, range: ChunkRange) -> Result<DataChunk> {
        let start_time = Instant::now();

        let mut relevant: Vec<HybridEntry> = handle
            .entries()?
            .into_iter()
            .filter(|e| e.row_count > 0 && e.row_start < range.end && e.row_end() > range.start)
            .collect();
        relevant.sort_by_key(|e| e.row_start);

        if relevant.is_empty() {
            if let Ok(mut stats) = self.global_stats.lock() {
                stats.record_read(start_time.elapsed());
            }
            let layout = handle
                .entries()?
                .first()
                .map(|e| e.layout)
                .unwrap_or(ChunkLayout::Opaque);
            return match layout {
                ChunkLayout::Opaque => Ok(DataChunk::new(Vec::new())),
                ChunkLayout::Strings => Ok(DataChunk::from_strings(Vec::new())),
            };
        }

        let mut parts = Vec::with_capacity(relevant.len());
        {
            let mut manager = handle.lock_manager()?;
            for entry in &relevant {
                let chunk = manager.retrieve_data(entry.id)?;
                let local_start = range.start.saturating_sub(entry.row_start);
                let local_end = range.end.min(entry.row_end()) - entry.row_start;
                parts.push(chunk.slice_rows(local_start, local_end)?);
            }
        }

        let merged = DataChunk::concat(parts)?;

        let mut stats = self.lock_stats()?;
        stats.record_read(start_time.elapsed());
        Ok(merged)
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        if chunk.rows() == 0 {
            return Ok(());
        }
        let rows = chunk.rows();
        let bytes = chunk.len();
        let layout = chunk.layout();

        let data_id = {
            let mut manager = handle.lock_manager()?;
            manager.store_data(chunk)?
        };

        let row_start = handle.reserve_rows(rows);
        // Recording the id is what makes the chunk retrievable.
        handle.record_entry(HybridEntry {
            id: data_id,
            row_start,
            row_count: rows,
            layout,
        })?;

        let mut stats = self.lock_stats()?;
        stats.total_data_size += bytes;
        stats.chunk_count += 1;
        Ok(())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, handle: &Self::Handle) -> Result<()> {
        let report = {
            let mut manager = handle.lock_manager()?;
            manager.flush_backends()?;
            if handle.config.enable_auto_tiering {
                Some(manager.run_background_tiering()?)
            } else {
                None
            }
        };

        if let Some(report) = report {
            let mut stats = self.lock_stats()?;
            stats.total_movements += report.promotions + report.demotions;
        }
        Ok(())
    }

    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()> {
        {
            let mut manager = handle.lock_manager()?;
            for data_id in manager.data_ids() {
                manager.delete_data(data_id)?;
            }
        }
        handle.clear_index()?;

        let mut stats = self.lock_stats()?;
        stats.total_data_size = 0;
        stats.chunk_count = 0;
        Ok(())
    }

    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability {
        let can_handle = requirements.estimated_size > 100 * 1024 * 1024;
        let confidence = if can_handle { 0.95 } else { 0.3 };

        let performance_score = match requirements.performance_priority {
            PerformancePriority::Speed => 0.9,
            PerformancePriority::Memory => 0.8,
            PerformancePriority::Balanced => 0.95,
            PerformancePriority::Throughput => 0.9,
            PerformancePriority::Latency => 0.85,
        };

        StrategyCapability {
            can_handle,
            confidence,
            performance_score,
            resource_cost: ResourceCost {
                memory: requirements.estimated_size / 20,
                cpu: 20.0,
                disk: requirements.estimated_size,
                network: 0,
            },
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        // The advertised compression ratio now reflects the codecs the tiers are
        // actually configured with instead of a fixed 2.0.
        let ratio = |codec: CompressionType| match codec {
            CompressionType::None => 1.0,
            CompressionType::Lz4 | CompressionType::Snappy => 2.0,
            CompressionType::Zstd | CompressionType::Gzip | CompressionType::Auto => 3.0,
        };
        let compression_ratio = (ratio(self.config.hot_tier.compression)
            + ratio(self.config.warm_tier.compression)
            + ratio(self.config.cold_tier.compression))
            / 3.0;

        PerformanceProfile {
            read_speed: Speed::VeryFast,
            write_speed: Speed::Fast,
            memory_efficiency: Efficiency::Excellent,
            compression_ratio,
            query_optimization: QueryOptimization::Good,
            parallel_scalability: ParallelScalability::Excellent,
        }
    }

    fn storage_stats(&self) -> StorageStats {
        match self.global_stats.lock() {
            Ok(stats) => {
                let hit_rate = if stats.tier_hit_rates.is_empty() {
                    0.0
                } else {
                    stats.tier_hit_rates.values().sum::<f64>() / stats.tier_hit_rates.len() as f64
                };
                StorageStats {
                    total_size: stats.total_data_size,
                    used_size: stats.total_data_size,
                    read_operations: stats.read_count,
                    write_operations: stats.chunk_count,
                    avg_read_latency_ns: stats.avg_access_latency.as_nanos() as u64,
                    avg_write_latency_ns: stats.avg_access_latency.as_nanos() as u64,
                    cache_hit_rate: hit_rate,
                }
            }
            Err(_) => StorageStats::default(),
        }
    }

    /// Retune tiering thresholds.
    ///
    /// This affects storages created *afterwards*. Existing handles carry their
    /// own configuration; use [`HybridLargeScaleStrategy::retune_handle`] to
    /// apply a pattern to an already-created handle.
    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        self.config = tune_config(self.config.clone(), pattern);
        Ok(())
    }

    fn compact(&mut self, handle: &Self::Handle) -> Result<CompactionResult> {
        let start_time = Instant::now();

        let (size_before, size_after, hit_rates) = {
            let mut manager = handle.lock_manager()?;
            let before = manager.physical_bytes();
            // Propagate tiering failures instead of swallowing them.
            let _report = manager.run_background_tiering()?;
            let after = manager.physical_bytes();
            let hit_rates: HashMap<DataTier, f64> = manager
                .get_tier_statistics()
                .iter()
                .map(|(&tier, stats)| (tier, stats.hit_rate()))
                .collect();
            (before, after, hit_rates)
        };

        let mut stats = self.lock_stats()?;
        stats.tier_hit_rates = hit_rates;

        Ok(CompactionResult {
            size_before,
            size_after,
            duration: start_time.elapsed(),
        })
    }
}

impl HybridLargeScaleStrategy {
    /// Apply an access pattern to an existing handle's tiering thresholds.
    pub fn retune_handle(&self, handle: &HybridHandle, pattern: AccessPattern) -> Result<()> {
        let tuned = tune_config(handle.config.clone(), pattern);
        let mut manager = handle.lock_manager()?;
        manager.apply_config(tuned);
        Ok(())
    }

    /// Refresh the per-tier hit rates published through [`StorageStrategy::storage_stats`].
    pub fn refresh_tier_stats(&self, handle: &HybridHandle) -> Result<()> {
        let hit_rates: HashMap<DataTier, f64> = {
            let manager = handle.lock_manager()?;
            manager
                .get_tier_statistics()
                .iter()
                .map(|(&tier, stats)| (tier, stats.hit_rate()))
                .collect()
        };
        let mut stats = self.lock_stats()?;
        stats.tier_hit_rates = hit_rates;
        Ok(())
    }
}

fn tune_config(mut config: HybridConfig, pattern: AccessPattern) -> HybridConfig {
    match pattern {
        AccessPattern::Sequential => {
            config.promotion_threshold *= 0.5;
            config.demotion_threshold = Duration::from_secs(48 * 3600);
        }
        AccessPattern::Random => {
            config.hot_tier.max_size = config.hot_tier.max_size.saturating_mul(2);
            config.promotion_threshold *= 2.0;
        }
        AccessPattern::Streaming => {
            config.enable_auto_tiering = false;
        }
        AccessPattern::HighLocality => {
            config.hot_tier.max_size = config.hot_tier.max_size.saturating_mul(3);
            config.promotion_threshold *= 0.3;
        }
        AccessPattern::LowLocality => {
            // Never divide a tier down to zero capacity.
            config.hot_tier.max_size = (config.hot_tier.max_size / 2).max(1);
            config.demotion_threshold = Duration::from_secs(6 * 3600);
        }
        _ => {}
    }
    config.normalized()
}

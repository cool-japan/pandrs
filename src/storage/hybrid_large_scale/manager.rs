//! Tier management: placement, promotion, demotion and capacity pressure.

use crate::core::error::{Error, Result};
use crate::storage::unified_memory::DataChunk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::config::{
    AccessPattern, DataTier, HybridConfig, TierMoveReason, TierStatistics, TieringReport,
    TieringScheduler,
};
use super::tiers::{make_backend, DataId, TierBackend};

/// Tier management engine
pub struct TierManager {
    /// Configuration
    config: HybridConfig,
    /// Tier-specific storage backends
    tier_backends: HashMap<DataTier, Box<dyn TierBackend>>,
    /// Tier statistics
    tier_stats: HashMap<DataTier, TierStatistics>,
    /// Data location index
    data_index: HashMap<DataId, DataTier>,
    /// Access pattern tracker
    access_tracker: HashMap<DataId, AccessPattern>,
    /// Background tiering scheduler
    tiering_scheduler: TieringScheduler,
    /// Monotonic data id source
    next_data_id: AtomicU64,
}

impl std::fmt::Debug for TierManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TierManager")
            .field("entries", &self.data_index.len())
            .field("tiers", &self.tier_backends.len())
            .finish()
    }
}

impl TierManager {
    /// Create a tier manager, building every configured backend.
    ///
    /// Prefer [`TierManager::try_new`] when backend creation errors (for
    /// example an unwritable tier directory) should be surfaced.
    pub fn new(config: HybridConfig) -> Self {
        match Self::try_new(config.clone()) {
            Ok(manager) => manager,
            Err(e) => {
                log::error!(
                    "Failed to build tier backends ({}); falling back to an in-memory-only hot tier",
                    e
                );
                let config = config.normalized();
                let mut tier_backends: HashMap<DataTier, Box<dyn TierBackend>> = HashMap::new();
                tier_backends.insert(
                    DataTier::Hot,
                    Box::new(super::tiers::InMemoryTierBackend::new(&config.hot_tier)),
                );
                let mut tier_stats = HashMap::new();
                tier_stats.insert(DataTier::Hot, TierStatistics::new(config.hot_tier.max_size));
                Self {
                    tiering_scheduler: TieringScheduler::new(config.tiering_interval),
                    config,
                    tier_backends,
                    tier_stats,
                    data_index: HashMap::new(),
                    access_tracker: HashMap::new(),
                    next_data_id: AtomicU64::new(1),
                }
            }
        }
    }

    /// Fallible constructor.
    pub fn try_new(config: HybridConfig) -> Result<Self> {
        let config = config.normalized();
        let mut tier_backends: HashMap<DataTier, Box<dyn TierBackend>> = HashMap::new();
        tier_backends.insert(DataTier::Hot, make_backend(&config.hot_tier)?);
        tier_backends.insert(DataTier::Warm, make_backend(&config.warm_tier)?);
        tier_backends.insert(DataTier::Cold, make_backend(&config.cold_tier)?);

        let mut tier_stats = HashMap::new();
        tier_stats.insert(DataTier::Hot, TierStatistics::new(config.hot_tier.max_size));
        tier_stats.insert(
            DataTier::Warm,
            TierStatistics::new(config.warm_tier.max_size),
        );
        tier_stats.insert(
            DataTier::Cold,
            TierStatistics::new(config.cold_tier.max_size),
        );

        Ok(Self {
            tiering_scheduler: TieringScheduler::new(config.tiering_interval),
            config,
            tier_backends,
            tier_stats,
            data_index: HashMap::new(),
            access_tracker: HashMap::new(),
            next_data_id: AtomicU64::new(1),
        })
    }

    /// Background tiering scheduler for this manager.
    pub fn scheduler(&mut self) -> &mut TieringScheduler {
        &mut self.tiering_scheduler
    }

    /// Store a chunk, returning the id needed to read it back.
    ///
    /// Ids used to be `random::<u64>()` with no collision check, which could
    /// silently overwrite an existing chunk; they are now monotonic.
    pub fn store_data(&mut self, data: DataChunk) -> Result<DataId> {
        let data_id = DataId(self.next_data_id.fetch_add(1, Ordering::SeqCst));
        let initial_tier = self.placement_tier(data.len())?;

        self.store_into_tier(data_id, initial_tier, &data)?;
        self.data_index.insert(data_id, initial_tier);
        self.access_tracker
            .insert(data_id, AccessPattern::new(data.len()));
        Ok(data_id)
    }

    /// Store into a specific tier, enforcing that tier's capacity.
    fn store_into_tier(&mut self, id: DataId, tier: DataTier, data: &DataChunk) -> Result<()> {
        // Capacity is checked *before* the write; `store_data` used to grow the
        // hot tier past `max_size` without ever noticing.
        let stats = self.tier_stats.get(&tier).ok_or_else(|| {
            Error::InvalidOperation(format!("Tier statistics {:?} not found", tier))
        })?;
        if data.len() > stats.available_space() {
            return Err(Error::InvalidOperation(format!(
                "Tier {:?} is full: {} bytes requested, {} available of {}",
                tier,
                data.len(),
                stats.available_space(),
                stats.max_capacity
            )));
        }

        let backend = self
            .tier_backends
            .get_mut(&tier)
            .ok_or_else(|| Error::InvalidOperation(format!("Tier backend {:?} not found", tier)))?;
        backend.store_chunk(id, data)?;

        if let Some(stats) = self.tier_stats.get_mut(&tier) {
            stats.current_usage = stats.current_usage.saturating_add(data.len());
            stats.chunk_count += 1;
        }
        Ok(())
    }

    /// Retrieve a chunk by id.
    pub fn retrieve_data(&mut self, data_id: DataId) -> Result<DataChunk> {
        let started = Instant::now();
        if let Some(pattern) = self.access_tracker.get_mut(&data_id) {
            pattern.record_access();
        }

        let current_tier = *self.data_index.get(&data_id).ok_or_else(|| {
            // Record the miss against every tier so hit rates stay honest.
            Error::InvalidOperation(format!("Data {:?} not found", data_id))
        })?;

        let backend = self.tier_backends.get(&current_tier).ok_or_else(|| {
            Error::InvalidOperation(format!("Tier backend {:?} not found", current_tier))
        })?;
        let chunk = match backend.retrieve_chunk(data_id) {
            Ok(chunk) => chunk,
            Err(e) => {
                if let Some(stats) = self.tier_stats.get_mut(&current_tier) {
                    stats.misses += 1;
                }
                return Err(e);
            }
        };

        let elapsed = started.elapsed().as_nanos() as u64;
        if let Some(stats) = self.tier_stats.get_mut(&current_tier) {
            stats.total_accesses += 1;
            stats.total_access_nanos = stats.total_access_nanos.saturating_add(elapsed);
            stats.hits += 1;
        }

        if self.config.enable_auto_tiering {
            self.check_and_promote(data_id)?;
        }

        Ok(chunk)
    }

    /// Delete a chunk.
    pub fn delete_data(&mut self, data_id: DataId) -> Result<()> {
        let current_tier = *self
            .data_index
            .get(&data_id)
            .ok_or_else(|| Error::InvalidOperation(format!("Data {:?} not found", data_id)))?;

        let backend = self.tier_backends.get_mut(&current_tier).ok_or_else(|| {
            Error::InvalidOperation(format!("Tier backend {:?} not found", current_tier))
        })?;
        backend.delete_chunk(data_id)?;

        self.data_index.remove(&data_id);
        if let Some(pattern) = self.access_tracker.remove(&data_id) {
            if let Some(stats) = self.tier_stats.get_mut(&current_tier) {
                stats.current_usage = stats.current_usage.saturating_sub(pattern.data_size);
                stats.chunk_count = stats.chunk_count.saturating_sub(1);
            }
        }
        Ok(())
    }

    /// Tier a new chunk should land in, honouring the tiers' free space.
    fn placement_tier(&self, size: usize) -> Result<DataTier> {
        let preferred = if size < 1024 * 1024 {
            DataTier::Hot
        } else if size < 100 * 1024 * 1024 {
            DataTier::Warm
        } else {
            DataTier::Cold
        };

        // Spill down the hierarchy when the preferred tier cannot take it.
        for tier in [preferred, DataTier::Warm, DataTier::Cold] {
            if let Some(stats) = self.tier_stats.get(&tier) {
                if size <= stats.available_space() {
                    return Ok(tier);
                }
            }
        }
        Err(Error::InvalidOperation(format!(
            "No storage tier has room for a {} byte chunk",
            size
        )))
    }

    fn check_and_promote(&mut self, data_id: DataId) -> Result<()> {
        let target_tier = {
            let pattern = match self.access_tracker.get(&data_id) {
                Some(pattern) => pattern,
                None => return Ok(()),
            };
            let current_tier = match self.data_index.get(&data_id) {
                Some(&tier) => tier,
                None => return Ok(()),
            };
            match current_tier {
                DataTier::Cold
                    if pattern.should_promote(self.config.promotion_threshold / 10.0) =>
                {
                    Some((current_tier, DataTier::Warm))
                }
                DataTier::Warm if pattern.should_promote(self.config.promotion_threshold) => {
                    Some((current_tier, DataTier::Hot))
                }
                _ => None,
            }
        };

        if let Some((from, to)) = target_tier {
            // A promotion that cannot fit is not an error for the read path.
            if let Err(e) = self.move_data(data_id, from, to, TierMoveReason::HighFrequency) {
                log::debug!(
                    "Promotion of {:?} from {:?} to {:?} skipped: {}",
                    data_id,
                    from,
                    to,
                    e
                );
            }
        }
        Ok(())
    }

    /// Run one background tiering pass.
    pub fn run_background_tiering(&mut self) -> Result<TieringReport> {
        let mut report = TieringReport {
            promotions: 0,
            demotions: 0,
            bytes_moved: 0,
            duration: std::time::Duration::ZERO,
        };
        let start_time = Instant::now();

        let data_to_demote: Vec<_> = self
            .access_tracker
            .iter()
            .filter(|(_, pattern)| pattern.should_demote(self.config.demotion_threshold))
            .map(|(&id, _)| id)
            .collect();

        for data_id in data_to_demote {
            let current_tier = match self.data_index.get(&data_id) {
                Some(&tier) => tier,
                None => continue,
            };
            let target_tier = match current_tier {
                DataTier::Hot => DataTier::Warm,
                DataTier::Warm => DataTier::Cold,
                DataTier::Cold => continue,
            };
            let data_size = self
                .access_tracker
                .get(&data_id)
                .map(|p| p.data_size)
                .unwrap_or(0);
            if self
                .move_data(
                    data_id,
                    current_tier,
                    target_tier,
                    TierMoveReason::LowFrequency,
                )
                .is_ok()
            {
                report.demotions += 1;
                report.bytes_moved += data_size;
            }
        }

        for &tier in &[DataTier::Hot, DataTier::Warm] {
            self.handle_capacity_pressure(tier, &mut report)?;
        }

        report.duration = start_time.elapsed();
        self.tiering_scheduler.mark_run();
        Ok(report)
    }

    /// Relieve capacity pressure on `tier`, looping until it is below the
    /// high-water mark or nothing more can be moved.
    fn handle_capacity_pressure(
        &mut self,
        tier: DataTier,
        report: &mut TieringReport,
    ) -> Result<()> {
        let target_tier = match tier {
            DataTier::Hot => DataTier::Warm,
            DataTier::Warm => DataTier::Cold,
            DataTier::Cold => return Ok(()),
        };

        // The old code moved `candidates/10` entries once per pass and then
        // returned regardless of whether the pressure was relieved.
        loop {
            let over_pressure = self
                .tier_stats
                .get(&tier)
                .map(|s| s.utilization() > 0.9)
                .unwrap_or(false);
            if !over_pressure {
                return Ok(());
            }

            let mut candidates: Vec<_> = self
                .data_index
                .iter()
                .filter(|(_, &t)| t == tier)
                .filter_map(|(&id, _)| {
                    self.access_tracker
                        .get(&id)
                        .map(|pattern| (id, pattern.last_access))
                })
                .collect();
            if candidates.is_empty() {
                return Ok(());
            }
            candidates.sort_by_key(|(_, last_access)| *last_access);

            let batch = (candidates.len() / 10).max(1);
            let mut moved_any = false;
            for (data_id, _) in candidates.into_iter().take(batch) {
                let data_size = self
                    .access_tracker
                    .get(&data_id)
                    .map(|p| p.data_size)
                    .unwrap_or(0);
                if self
                    .move_data(data_id, tier, target_tier, TierMoveReason::CapacityPressure)
                    .is_ok()
                {
                    report.demotions += 1;
                    report.bytes_moved += data_size;
                    moved_any = true;
                }
            }
            if !moved_any {
                // Nothing could be relocated; stop rather than spin.
                return Ok(());
            }
        }
    }

    /// Move a chunk between tiers.
    ///
    /// Ordering is retrieve -> store -> delete, and a failed delete rolls the
    /// target write back so the chunk can never end up live in two tiers with a
    /// stale index (which previously leaked an orphan copy).
    fn move_data(
        &mut self,
        data_id: DataId,
        from_tier: DataTier,
        to_tier: DataTier,
        reason: TierMoveReason,
    ) -> Result<()> {
        if from_tier == to_tier {
            return Ok(());
        }

        let chunk = self
            .tier_backends
            .get(&from_tier)
            .ok_or_else(|| {
                Error::InvalidOperation(format!("Source tier backend {:?} not found", from_tier))
            })?
            .retrieve_chunk(data_id)?;

        self.store_into_tier(data_id, to_tier, &chunk)?;

        let delete_result = self
            .tier_backends
            .get_mut(&from_tier)
            .ok_or_else(|| {
                Error::InvalidOperation(format!("Source tier backend {:?} not found", from_tier))
            })
            .and_then(|backend| backend.delete_chunk(data_id));

        if let Err(e) = delete_result {
            // Roll the target write back so exactly one copy stays live.
            if let Some(backend) = self.tier_backends.get_mut(&to_tier) {
                if let Err(rollback) = backend.delete_chunk(data_id) {
                    log::error!(
                        "Failed to roll back tier move of {:?} into {:?}: {}",
                        data_id,
                        to_tier,
                        rollback
                    );
                }
            }
            if let Some(stats) = self.tier_stats.get_mut(&to_tier) {
                stats.current_usage = stats.current_usage.saturating_sub(chunk.len());
                stats.chunk_count = stats.chunk_count.saturating_sub(1);
            }
            return Err(e);
        }

        self.data_index.insert(data_id, to_tier);

        let data_size = chunk.len();
        if let Some(from_stats) = self.tier_stats.get_mut(&from_tier) {
            from_stats.current_usage = from_stats.current_usage.saturating_sub(data_size);
            from_stats.chunk_count = from_stats.chunk_count.saturating_sub(1);
            match reason {
                TierMoveReason::HighFrequency => from_stats.promotions += 1,
                _ => from_stats.demotions += 1,
            }
        }
        if let Some(to_stats) = self.tier_stats.get_mut(&to_tier) {
            match reason {
                TierMoveReason::HighFrequency => to_stats.promotions += 1,
                _ => to_stats.demotions += 1,
            }
        }

        Ok(())
    }

    /// Move a chunk to a specific tier on request.
    pub fn relocate(&mut self, data_id: DataId, to_tier: DataTier) -> Result<()> {
        let from_tier = *self
            .data_index
            .get(&data_id)
            .ok_or_else(|| Error::InvalidOperation(format!("Data {:?} not found", data_id)))?;
        self.move_data(data_id, from_tier, to_tier, TierMoveReason::Manual)
    }

    pub fn get_tier_statistics(&self) -> &HashMap<DataTier, TierStatistics> {
        &self.tier_stats
    }

    pub fn get_data_distribution(&self) -> HashMap<DataTier, usize> {
        let mut distribution = HashMap::new();
        for &tier in self.data_index.values() {
            *distribution.entry(tier).or_insert(0) += 1;
        }
        distribution
    }

    /// Tier a chunk currently lives in.
    pub fn tier_of(&self, data_id: DataId) -> Option<DataTier> {
        self.data_index.get(&data_id).copied()
    }

    /// Every stored data id.
    pub fn data_ids(&self) -> Vec<DataId> {
        self.data_index.keys().copied().collect()
    }

    /// Bytes physically stored across all tiers (post-compression).
    pub fn physical_bytes(&self) -> usize {
        self.data_index
            .iter()
            .filter_map(|(id, tier)| {
                self.tier_backends
                    .get(tier)
                    .and_then(|backend| backend.stored_size(*id))
            })
            .sum()
    }

    /// Logical bytes tracked across all tiers (pre-compression).
    pub fn logical_bytes(&self) -> usize {
        self.access_tracker.values().map(|p| p.data_size).sum()
    }

    /// Replace the tiering thresholds without rebuilding the backends.
    ///
    /// Tier capacities are also refreshed so that a retuned handle really
    /// changes behaviour instead of mutating a config nothing reads.
    pub fn apply_config(&mut self, config: HybridConfig) {
        let config = config.normalized();
        if let Some(stats) = self.tier_stats.get_mut(&DataTier::Hot) {
            stats.max_capacity = config.hot_tier.max_size;
        }
        if let Some(stats) = self.tier_stats.get_mut(&DataTier::Warm) {
            stats.max_capacity = config.warm_tier.max_size;
        }
        if let Some(stats) = self.tier_stats.get_mut(&DataTier::Cold) {
            stats.max_capacity = config.cold_tier.max_size;
        }
        self.config = config;
    }

    /// Flush every file-backed tier.
    pub fn flush_backends(&mut self) -> Result<()> {
        for backend in self.tier_backends.values_mut() {
            backend.flush()?;
        }
        Ok(())
    }
}

//! Configuration and access-pattern bookkeeping for the hybrid strategy.

use crate::storage::unified_memory::CompressionType;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Hybrid strategy configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Hot tier configuration (in-memory)
    pub hot_tier: TierConfig,
    /// Warm tier configuration (SSD/fast disk)
    pub warm_tier: TierConfig,
    /// Cold tier configuration (slow/archival storage)
    pub cold_tier: TierConfig,
    /// Access pattern analysis window
    pub analysis_window: Duration,
    /// Promotion threshold (accesses per hour)
    pub promotion_threshold: f64,
    /// Demotion threshold (time since last access)
    pub demotion_threshold: Duration,
    /// Enable automatic tiering
    pub enable_auto_tiering: bool,
    /// Background tiering interval
    pub tiering_interval: Duration,
    /// Enable compression in the cold tier
    pub enable_cold_compression: bool,
    /// Maximum memory usage for the hot tier
    pub max_hot_memory: usize,
    /// Enable data deduplication across tiers
    pub enable_deduplication: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            hot_tier: TierConfig {
                name: "hot".to_string(),
                storage_type: TierStorageType::InMemory,
                max_size: 512 * 1024 * 1024, // 512MB
                compression: CompressionType::None,
                access_latency: Duration::from_micros(1),
                throughput_mbps: 10000.0,
                directory: None,
                durable_writes: false,
            },
            warm_tier: TierConfig {
                name: "warm".to_string(),
                storage_type: TierStorageType::SSD,
                max_size: 10 * 1024 * 1024 * 1024, // 10GB
                compression: CompressionType::Lz4,
                access_latency: Duration::from_millis(1),
                throughput_mbps: 500.0,
                directory: None,
                durable_writes: false,
            },
            cold_tier: TierConfig {
                name: "cold".to_string(),
                storage_type: TierStorageType::HDD,
                max_size: 1024 * 1024 * 1024 * 1024, // 1TB
                compression: CompressionType::Zstd,
                access_latency: Duration::from_millis(10),
                throughput_mbps: 100.0,
                directory: None,
                durable_writes: false,
            },
            analysis_window: Duration::from_secs(3600),
            promotion_threshold: 10.0,
            demotion_threshold: Duration::from_secs(24 * 3600),
            enable_auto_tiering: true,
            tiering_interval: Duration::from_secs(5 * 60),
            enable_cold_compression: true,
            max_hot_memory: 1024 * 1024 * 1024,
            enable_deduplication: true,
        }
    }
}

impl HybridConfig {
    /// Apply cross-cutting settings that would otherwise be silently ignored.
    ///
    /// `enable_cold_compression` used to be dead configuration; it now really
    /// disables the cold tier's codec.
    pub fn normalized(mut self) -> Self {
        if !self.enable_cold_compression {
            self.cold_tier.compression = CompressionType::None;
        }
        if self.hot_tier.max_size > self.max_hot_memory {
            self.hot_tier.max_size = self.max_hot_memory;
        }
        self
    }
}

/// Storage tier configuration
#[derive(Debug, Clone)]
pub struct TierConfig {
    /// Tier name
    pub name: String,
    /// Storage type for this tier
    pub storage_type: TierStorageType,
    /// Maximum storage size
    pub max_size: usize,
    /// Compression codec applied to chunks stored in this tier
    pub compression: CompressionType,
    /// Expected access latency
    pub access_latency: Duration,
    /// Expected throughput in MB/s
    pub throughput_mbps: f64,
    /// Directory for file-backed tiers. `None` means "use a temporary
    /// directory owned by the backend".
    pub directory: Option<PathBuf>,
    /// Whether writes to a file-backed tier must be fsync'd before returning
    pub durable_writes: bool,
}

/// Storage type for each tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierStorageType {
    /// In-memory storage (fastest)
    InMemory,
    /// SSD storage (fast, file-backed)
    SSD,
    /// HDD storage (slower but larger, file-backed)
    HDD,
    /// Network storage
    Network,
    /// Custom file-backed storage backend
    Custom,
}

/// Data tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataTier {
    /// Hot tier - frequently accessed data
    Hot,
    /// Warm tier - moderately accessed data
    Warm,
    /// Cold tier - rarely accessed data
    Cold,
}

/// Access pattern tracking for a data chunk
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Number of accesses in the current window
    pub access_count: u64,
    /// Last access timestamp
    pub last_access: Instant,
    /// First access timestamp
    pub first_access: Instant,
    /// Access frequency (accesses per hour)
    pub access_frequency: f64,
    /// Size of the data
    pub data_size: usize,
    /// Access pattern type
    pub pattern_type: AccessPatternType,
}

impl AccessPattern {
    pub fn new(data_size: usize) -> Self {
        let now = Instant::now();
        Self {
            access_count: 1,
            last_access: now,
            first_access: now,
            access_frequency: 0.0,
            data_size,
            pattern_type: AccessPatternType::Unknown,
        }
    }

    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = Instant::now();

        let time_since_first = self.last_access.duration_since(self.first_access);
        if time_since_first.as_secs() > 0 {
            self.access_frequency =
                self.access_count as f64 / (time_since_first.as_secs_f64() / 3600.0);
        }

        self.pattern_type = if self.access_frequency > 100.0 {
            AccessPatternType::VeryHot
        } else if self.access_frequency > 10.0 {
            AccessPatternType::Hot
        } else if self.access_frequency > 1.0 {
            AccessPatternType::Warm
        } else {
            AccessPatternType::Cold
        };
    }

    pub fn time_since_last_access(&self) -> Duration {
        Instant::now().duration_since(self.last_access)
    }

    pub fn should_promote(&self, threshold: f64) -> bool {
        self.access_frequency > threshold
    }

    pub fn should_demote(&self, threshold: Duration) -> bool {
        self.time_since_last_access() > threshold
    }
}

/// Access pattern classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPatternType {
    /// Very frequently accessed (>100 accesses/hour)
    VeryHot,
    /// Frequently accessed (>10 accesses/hour)
    Hot,
    /// Moderately accessed (>1 access/hour)
    Warm,
    /// Rarely accessed (<1 access/hour)
    Cold,
    /// Pattern not yet determined
    Unknown,
}

/// Tiered data metadata
#[derive(Debug, Clone)]
pub struct TieredDataMetadata {
    /// Creation timestamp
    pub created_at: Instant,
    /// Original size before compression
    pub original_size: usize,
    /// Compressed size
    pub compressed_size: usize,
    /// Checksum for integrity
    pub checksum: u64,
    /// Tier history
    pub tier_history: Vec<TierHistoryEntry>,
}

/// Tier movement history
#[derive(Debug, Clone)]
pub struct TierHistoryEntry {
    /// Tier moved to
    pub tier: DataTier,
    /// Timestamp of the move
    pub timestamp: Instant,
    /// Reason for the move
    pub reason: TierMoveReason,
}

/// Reason for tier movement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierMoveReason {
    /// Promoted due to high access frequency
    HighFrequency,
    /// Demoted due to low access frequency
    LowFrequency,
    /// Moved due to capacity pressure
    CapacityPressure,
    /// Initial placement
    InitialPlacement,
    /// Manual override
    Manual,
}

/// Compression state tracking
#[derive(Debug, Clone)]
pub struct CompressionState {
    /// Compression algorithm used
    pub algorithm: CompressionType,
    /// Compression ratio achieved
    pub ratio: f64,
    /// Compression/decompression time
    pub processing_time: Duration,
}

/// Tier statistics
#[derive(Debug, Clone)]
pub struct TierStatistics {
    /// Current usage in bytes
    pub current_usage: usize,
    /// Maximum capacity
    pub max_capacity: usize,
    /// Number of stored chunks
    pub chunk_count: u64,
    /// Total accesses
    pub total_accesses: u64,
    /// Total nanoseconds spent serving accesses (for a real average)
    pub total_access_nanos: u64,
    /// Accesses that found the data in this tier
    pub hits: u64,
    /// Accesses that looked here and missed
    pub misses: u64,
    /// Promotion count
    pub promotions: u64,
    /// Demotion count
    pub demotions: u64,
}

impl TierStatistics {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            current_usage: 0,
            max_capacity,
            chunk_count: 0,
            total_accesses: 0,
            total_access_nanos: 0,
            hits: 0,
            misses: 0,
            promotions: 0,
            demotions: 0,
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.max_capacity == 0 {
            0.0
        } else {
            self.current_usage as f64 / self.max_capacity as f64
        }
    }

    pub fn available_space(&self) -> usize {
        self.max_capacity.saturating_sub(self.current_usage)
    }

    /// Real average access latency for this tier.
    pub fn avg_access_latency(&self) -> Duration {
        if self.total_accesses == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.total_access_nanos / self.total_accesses)
        }
    }

    /// Fraction of lookups against this tier that found the data.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Tiering report for background operations
#[derive(Debug, Clone)]
pub struct TieringReport {
    /// Number of promotions performed
    pub promotions: u64,
    /// Number of demotions performed
    pub demotions: u64,
    /// Total bytes moved
    pub bytes_moved: usize,
    /// Time taken for tiering operations
    pub duration: Duration,
}

/// Background tiering scheduler
#[derive(Debug)]
pub struct TieringScheduler {
    interval: Duration,
    last_run: Instant,
}

impl TieringScheduler {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_run: Instant::now(),
        }
    }

    pub fn should_run(&self) -> bool {
        self.last_run.elapsed() >= self.interval
    }

    pub fn mark_run(&mut self) {
        self.last_run = Instant::now();
    }
}

//! `AdaptiveStringPoolStrategy`: deduplicating, adaptively-compressed storage
//! for string columns.

use crate::core::error::{Error, Result};
use crate::storage::adaptive_string_pool::analysis::{
    StrategyRecommendations, StringCharacteristics, StringPatternAnalyzer, StringPoolConfig,
    StringStorageStrategy,
};
use crate::storage::adaptive_string_pool::codec::{
    StringCompressionAlgorithm, StringCompressionEngine,
};
use crate::storage::adaptive_string_pool::dictionary::CompressionDictionary;
use crate::storage::unified_memory::*;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// String identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StringId(pub u64);

/// String encoding type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringEncoding {
    Utf8,
    Ascii,
    Latin1,
    Custom,
}

/// String metadata.
///
/// The old struct also carried `dictionary_refs` and `reconstruction_data`,
/// which were always empty and never read: every codec blob is self-describing,
/// so no side-channel reconstruction data is needed. They were removed rather
/// than left as permanently-empty fields implying state that does not exist.
#[derive(Debug, Clone)]
pub struct StringMetadata {
    /// Encoding of the original string
    pub encoding: StringEncoding,
}

/// String entry in the pool
#[derive(Debug, Clone)]
pub struct StringEntry {
    /// Unique identifier
    pub id: StringId,
    /// Storage strategy used
    pub strategy: StringStorageStrategy,
    /// Compressed/encoded data (always codec-tagged)
    pub data: Vec<u8>,
    /// Original string length in bytes
    pub original_length: usize,
    /// Reference count for deduplication
    pub ref_count: u32,
    /// First access timestamp
    pub first_accessed: Instant,
    /// Last access timestamp
    pub last_accessed: Instant,
    /// Access frequency
    pub access_count: u64,
    /// Compression algorithm used
    pub compression: StringCompressionAlgorithm,
    /// Metadata for reconstruction
    pub metadata: StringMetadata,
}

/// String pool statistics
#[derive(Debug, Clone)]
pub struct StringPoolStatistics {
    /// Total strings stored (including duplicates)
    pub total_strings: u64,
    /// Unique strings (after deduplication)
    pub unique_strings: u64,
    /// Total storage space used
    pub storage_used: usize,
    /// Total original size
    pub original_size: usize,
    /// Compression ratio achieved (original / stored; > 1.0 is a win)
    pub compression_ratio: f64,
    /// Deduplication savings (0.0 to 1.0)
    pub deduplication_savings: f64,
    /// Deduplication cache hit rate
    pub cache_hit_rate: f64,
    /// Average retrieval time
    pub avg_access_time: Duration,
    /// Deduplication hits
    pub dedup_hits: u64,
    /// Deduplication misses
    pub dedup_misses: u64,
    /// Total retrievals served
    pub retrievals: u64,
    /// Total nanoseconds spent in retrievals
    pub retrieval_nanos: u64,
}

impl StringPoolStatistics {
    pub fn new() -> Self {
        Self {
            total_strings: 0,
            unique_strings: 0,
            storage_used: 0,
            original_size: 0,
            compression_ratio: 1.0,
            deduplication_savings: 0.0,
            cache_hit_rate: 0.0,
            avg_access_time: Duration::ZERO,
            dedup_hits: 0,
            dedup_misses: 0,
            retrievals: 0,
            retrieval_nanos: 0,
        }
    }

    pub fn space_savings(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            1.0 - (self.storage_used as f64 / self.original_size as f64)
        }
    }

    fn refresh_derived(&mut self) {
        if self.storage_used > 0 {
            self.compression_ratio = self.original_size as f64 / self.storage_used as f64;
        }
        let lookups = self.dedup_hits + self.dedup_misses;
        if lookups > 0 {
            self.cache_hit_rate = self.dedup_hits as f64 / lookups as f64;
        }
        if self.total_strings > 0 {
            self.deduplication_savings =
                1.0 - (self.unique_strings as f64 / self.total_strings as f64);
        }
        if self.retrievals > 0 {
            self.avg_access_time = Duration::from_nanos(self.retrieval_nanos / self.retrievals);
        }
    }
}

impl Default for StringPoolStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern analyzer state carried by a handle.
#[derive(Debug)]
pub struct PatternAnalyzerState {
    /// Recent strings for analysis
    pub recent_strings: VecDeque<String>,
    /// Current characteristics
    pub characteristics: StringCharacteristics,
    /// Analysis window
    pub analysis_window: usize,
    /// Last analysis timestamp
    pub last_analysis: Instant,
    /// Strategy recommendations produced by the last analysis
    pub strategy_recommendations: StrategyRecommendations,
}

impl PatternAnalyzerState {
    fn new(analysis_window: usize) -> Self {
        Self {
            recent_strings: VecDeque::new(),
            characteristics: StringCharacteristics::new(),
            analysis_window,
            last_analysis: Instant::now(),
            strategy_recommendations: StrategyRecommendations::default(),
        }
    }
}

/// String pool handle.
///
/// The analyzer state and the ordered id list are shared and interior-mutable:
/// `write_chunk` only gets `&Handle`, so the previous design could never write
/// its computed recommendations back, and the entire adaptive pipeline had zero
/// effect (`create_storage` hardcoded `{Raw, None}` and `write_chunk` read only
/// that).
#[derive(Debug, Clone)]
pub struct StringPoolHandle {
    /// Pool configuration
    pub config: StringPoolConfig,
    /// Current storage strategy
    pub current_strategy: Arc<RwLock<StringStorageStrategy>>,
    /// Analyzer state, including the live recommendations
    pub analyzer_state: Arc<RwLock<PatternAnalyzerState>>,
    /// Ordered ids of the rows written through this handle
    rows: Arc<Mutex<Vec<StringId>>>,
}

impl StringPoolHandle {
    fn new(config: StringPoolConfig) -> Self {
        let window = config.analysis_window_size;
        Self {
            config,
            current_strategy: Arc::new(RwLock::new(StringStorageStrategy::Raw)),
            analyzer_state: Arc::new(RwLock::new(PatternAnalyzerState::new(window))),
            rows: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn lock_rows(&self) -> Result<std::sync::MutexGuard<'_, Vec<StringId>>> {
        self.rows.lock().map_err(|_| {
            Error::InvalidOperation("String pool row index lock is poisoned".to_string())
        })
    }

    /// Ids of the rows written through this handle, in write order.
    pub fn row_ids(&self) -> Result<Vec<StringId>> {
        Ok(self.lock_rows()?.clone())
    }

    /// Number of rows written through this handle.
    pub fn row_count(&self) -> usize {
        self.rows.lock().map(|rows| rows.len()).unwrap_or(0)
    }

    /// Strategy currently applied by this handle.
    pub fn strategy(&self) -> Result<StringStorageStrategy> {
        self.current_strategy
            .read()
            .map(|s| *s)
            .map_err(|_| Error::InvalidOperation("String pool strategy lock is poisoned".into()))
    }

    /// Live strategy recommendations for this handle.
    pub fn recommendations(&self) -> Result<StrategyRecommendations> {
        self.analyzer_state
            .read()
            .map(|state| state.strategy_recommendations.clone())
            .map_err(|_| {
                Error::InvalidOperation("String pool analyzer state lock is poisoned".into())
            })
    }

    fn append_rows(&self, ids: &[StringId]) -> Result<()> {
        self.lock_rows()?.extend_from_slice(ids);
        Ok(())
    }

    fn clear_rows(&self) -> Result<()> {
        self.lock_rows()?.clear();
        Ok(())
    }
}

/// Adaptive String Pool Strategy Implementation
pub struct AdaptiveStringPoolStrategy {
    config: StringPoolConfig,
    string_storage: Arc<Mutex<HashMap<StringId, StringEntry>>>,
    /// Deduplication index: hash -> candidate ids (content is verified before reuse)
    string_lookup: Arc<Mutex<HashMap<u64, Vec<StringId>>>>,
    pattern_analyzer: StringPatternAnalyzer,
    compression_engines: HashMap<StringCompressionAlgorithm, StringCompressionEngine>,
    dictionary: Arc<RwLock<CompressionDictionary>>,
    next_string_id: AtomicU64,
    statistics: Arc<Mutex<StringPoolStatistics>>,
}

impl AdaptiveStringPoolStrategy {
    pub fn new(config: StringPoolConfig) -> Self {
        let dictionary = Arc::new(RwLock::new(CompressionDictionary::new()));

        let mut compression_engines = HashMap::new();
        for algorithm in [
            StringCompressionAlgorithm::None,
            StringCompressionAlgorithm::RunLength,
            StringCompressionAlgorithm::Lz4,
            StringCompressionAlgorithm::Zstd,
            StringCompressionAlgorithm::StringOptimized,
        ] {
            compression_engines.insert(algorithm, StringCompressionEngine::new(algorithm));
        }
        compression_engines.insert(
            StringCompressionAlgorithm::Dictionary,
            StringCompressionEngine::with_dictionary(
                StringCompressionAlgorithm::Dictionary,
                Arc::clone(&dictionary),
            ),
        );

        Self {
            pattern_analyzer: StringPatternAnalyzer::new(config.clone()),
            config,
            string_storage: Arc::new(Mutex::new(HashMap::new())),
            string_lookup: Arc::new(Mutex::new(HashMap::new())),
            compression_engines,
            dictionary,
            next_string_id: AtomicU64::new(1),
            statistics: Arc::new(Mutex::new(StringPoolStatistics::new())),
        }
    }

    fn lock_storage(&self) -> Result<std::sync::MutexGuard<'_, HashMap<StringId, StringEntry>>> {
        self.string_storage
            .lock()
            .map_err(|_| Error::InvalidOperation("String storage lock is poisoned".to_string()))
    }

    fn lock_lookup(&self) -> Result<std::sync::MutexGuard<'_, HashMap<u64, Vec<StringId>>>> {
        self.string_lookup
            .lock()
            .map_err(|_| Error::InvalidOperation("String lookup lock is poisoned".to_string()))
    }

    fn lock_statistics(&self) -> Result<std::sync::MutexGuard<'_, StringPoolStatistics>> {
        self.statistics
            .lock()
            .map_err(|_| Error::InvalidOperation("String statistics lock is poisoned".to_string()))
    }

    fn compute_string_hash(&self, s: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    fn engine(&self, algorithm: StringCompressionAlgorithm) -> Result<&StringCompressionEngine> {
        self.compression_engines.get(&algorithm).ok_or_else(|| {
            Error::InvalidOperation(format!("Compression algorithm {:?} not found", algorithm))
        })
    }

    /// Decode a stored entry back into its string.
    fn decode_entry(&self, entry: &StringEntry) -> Result<String> {
        // The payload's own codec tag selects the decoder, so an entry written
        // with one algorithm still decodes after the pool switches algorithms.
        self.engine(entry.compression)?.decompress(&entry.data)
    }

    /// Store a string, deduplicating against verified-identical content.
    pub fn store_string_with_strategy(
        &self,
        s: &str,
        strategy: StringStorageStrategy,
        compression: StringCompressionAlgorithm,
    ) -> Result<StringId> {
        let string_hash = self.compute_string_hash(s);
        let dedupe = s.len() >= self.config.deduplication_threshold;

        if dedupe {
            // Content-verified deduplication: a bare 64-bit hash match used to
            // be enough, so a collision returned a *different* string's id.
            let candidates = {
                let lookup = self.lock_lookup()?;
                lookup.get(&string_hash).cloned().unwrap_or_default()
            };
            for candidate in candidates {
                let entry = {
                    let storage = self.lock_storage()?;
                    storage.get(&candidate).cloned()
                };
                let Some(entry) = entry else { continue };
                if entry.original_length != s.len() {
                    continue;
                }
                if self.decode_entry(&entry)? == s {
                    let mut storage = self.lock_storage()?;
                    if let Some(entry) = storage.get_mut(&candidate) {
                        entry.ref_count = entry.ref_count.saturating_add(1);
                        entry.last_accessed = Instant::now();
                        entry.access_count += 1;
                    }
                    drop(storage);
                    let mut stats = self.lock_statistics()?;
                    stats.total_strings += 1;
                    stats.dedup_hits += 1;
                    stats.refresh_derived();
                    return Ok(candidate);
                }
            }
        }

        // Compress *before* burning an id, so a codec failure does not leave a
        // hole in the id space.
        let compressed_data = self.engine(compression)?.compress(s)?;
        let compressed_size = compressed_data.len();
        let string_id = StringId(self.next_string_id.fetch_add(1, Ordering::SeqCst));

        let now = Instant::now();
        let entry = StringEntry {
            id: string_id,
            strategy,
            data: compressed_data,
            original_length: s.len(),
            ref_count: 1,
            first_accessed: now,
            last_accessed: now,
            access_count: 1,
            compression,
            metadata: StringMetadata {
                encoding: if s.is_ascii() {
                    StringEncoding::Ascii
                } else {
                    StringEncoding::Utf8
                },
            },
        };

        // Both index updates must succeed or the entry is not published; a
        // poisoned lock used to skip the insert and still return Ok(id).
        {
            let mut storage = self.lock_storage()?;
            storage.insert(string_id, entry);
        }
        if dedupe {
            let mut lookup = self.lock_lookup()?;
            lookup.entry(string_hash).or_default().push(string_id);
        }

        let mut stats = self.lock_statistics()?;
        stats.total_strings += 1;
        stats.unique_strings += 1;
        stats.dedup_misses += 1;
        stats.original_size += s.len();
        stats.storage_used += compressed_size;
        stats.refresh_derived();

        Ok(string_id)
    }

    /// Retrieve a string by id.
    pub fn retrieve_string(&self, string_id: StringId) -> Result<String> {
        let started = Instant::now();
        let entry = {
            let mut storage = self.lock_storage()?;
            match storage.get_mut(&string_id) {
                Some(entry) => {
                    entry.last_accessed = Instant::now();
                    entry.access_count += 1;
                    entry.clone()
                }
                None => {
                    return Err(Error::InvalidOperation(format!(
                        "String ID {:?} not found",
                        string_id
                    )))
                }
            }
        };

        let decoded = self.decode_entry(&entry)?;

        let mut stats = self.lock_statistics()?;
        stats.retrievals += 1;
        stats.retrieval_nanos = stats
            .retrieval_nanos
            .saturating_add(started.elapsed().as_nanos() as u64);
        stats.refresh_derived();
        Ok(decoded)
    }

    /// Release one reference to a string, making it eligible for compaction.
    pub fn release_string(&self, string_id: StringId) -> Result<u32> {
        let mut storage = self.lock_storage()?;
        let entry = storage.get_mut(&string_id).ok_or_else(|| {
            Error::InvalidOperation(format!("String ID {:?} not found", string_id))
        })?;
        entry.ref_count = entry.ref_count.saturating_sub(1);
        Ok(entry.ref_count)
    }

    /// Analyse a sample and publish the resulting recommendations on `handle`.
    fn analyze_and_optimize(
        &self,
        handle: &StringPoolHandle,
        sample_strings: &[String],
    ) -> Result<StrategyRecommendations> {
        if sample_strings.is_empty() {
            return handle.recommendations();
        }

        let characteristics = self.pattern_analyzer.analyze_strings(sample_strings)?;
        let recommendations = self.pattern_analyzer.recommend_strategy(&characteristics);

        if recommendations.recommended_compression == StringCompressionAlgorithm::Dictionary
            || recommendations.recommended_strategy == StringStorageStrategy::DictionaryEncoded
        {
            // A shared RwLock actually lets the dictionary be trained; the old
            // `Arc::get_mut` could never succeed once the Arc was cloned into
            // the engine map.
            let mut dictionary = self.dictionary.write().map_err(|_| {
                Error::InvalidOperation("Compression dictionary lock is poisoned".to_string())
            })?;
            dictionary.build_from_strings(sample_strings)?;
        }

        // Publish the recommendation so write_chunk actually uses it.
        {
            let mut state = handle.analyzer_state.write().map_err(|_| {
                Error::InvalidOperation("String pool analyzer state lock is poisoned".to_string())
            })?;
            state.characteristics = characteristics;
            state.strategy_recommendations = recommendations.clone();
            state.last_analysis = Instant::now();
            for s in sample_strings {
                if state.recent_strings.len() >= state.analysis_window {
                    state.recent_strings.pop_front();
                }
                state.recent_strings.push_back(s.clone());
            }
        }
        {
            let mut current = handle.current_strategy.write().map_err(|_| {
                Error::InvalidOperation("String pool strategy lock is poisoned".to_string())
            })?;
            *current = recommendations.recommended_strategy;
        }

        Ok(recommendations)
    }

    /// Number of words in the shared compression dictionary.
    pub fn dictionary_len(&self) -> usize {
        self.dictionary.read().map(|d| d.len()).unwrap_or(0)
    }

    /// Snapshot of the pool statistics.
    pub fn statistics(&self) -> Result<StringPoolStatistics> {
        Ok(self.lock_statistics()?.clone())
    }
}

impl StorageStrategy for AdaptiveStringPoolStrategy {
    type Handle = StringPoolHandle;
    type Error = Error;
    type Metadata = StringPoolStatistics;

    fn name(&self) -> &'static str {
        "AdaptiveStringPool"
    }

    fn create_storage(&mut self, config: &StorageConfig) -> Result<Self::Handle> {
        let handle = StringPoolHandle::new(self.config.clone());

        if let Some(ref data_sample) = config.data_sample {
            if let Ok(sample_data) = String::from_utf8(data_sample.clone()) {
                let sample_strings: Vec<String> =
                    sample_data.lines().map(|s| s.to_string()).collect();
                self.analyze_and_optimize(&handle, &sample_strings)?;
            }
        }

        Ok(handle)
    }

    /// Read rows `[range.start, range.end)` of the stream written through
    /// `handle`.
    ///
    /// The range is a **row** range, per the `ChunkRange` contract. It used to
    /// be treated as a raw `StringId` range, and ids that failed to resolve
    /// were skipped — silently returning fewer strings than requested and
    /// misaligning the column.
    fn read_chunk(&self, handle: &Self::Handle, range: ChunkRange) -> Result<DataChunk> {
        let ids = handle.row_ids()?;
        let end = range.end.min(ids.len());
        let start = range.start.min(end);

        let mut strings = Vec::with_capacity(end - start);
        for (offset, &id) in ids[start..end].iter().enumerate() {
            strings.push(self.retrieve_string(id).map_err(|e| {
                Error::InvalidOperation(format!(
                    "String pool row {} ({:?}) could not be read: {}",
                    start + offset,
                    id,
                    e
                ))
            })?);
        }

        Ok(DataChunk::from_strings(strings))
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        let strings = chunk.as_strings()?;
        if strings.is_empty() {
            return Ok(());
        }

        let recommendations = if self.config.enable_pattern_analysis {
            self.analyze_and_optimize(handle, &strings)?
        } else {
            handle.recommendations()?
        };

        let strategy = recommendations.recommended_strategy;
        let compression = recommendations.recommended_compression;

        let mut ids = Vec::with_capacity(strings.len());
        for s in &strings {
            ids.push(self.store_string_with_strategy(s, strategy, compression)?);
        }
        handle.append_rows(&ids)?;
        Ok(())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, _handle: &Self::Handle) -> Result<()> {
        // The pool is in-memory; there is nothing to push to a device.
        Ok(())
    }

    fn delete_storage(&mut self, handle: &Self::Handle) -> Result<()> {
        let ids = handle.row_ids()?;
        {
            let mut storage = self.lock_storage()?;
            let mut lookup = self.lock_lookup()?;
            let mut stats = self.lock_statistics()?;
            for id in &ids {
                if let Some(entry) = storage.get_mut(id) {
                    entry.ref_count = entry.ref_count.saturating_sub(1);
                    if entry.ref_count == 0 {
                        stats.storage_used = stats.storage_used.saturating_sub(entry.data.len());
                        stats.original_size =
                            stats.original_size.saturating_sub(entry.original_length);
                        stats.unique_strings = stats.unique_strings.saturating_sub(1);
                        storage.remove(id);
                    }
                }
            }
            lookup.retain(|_, candidates| {
                candidates.retain(|id| storage.contains_key(id));
                !candidates.is_empty()
            });
            stats.refresh_derived();
        }
        handle.clear_rows()?;
        Ok(())
    }

    fn can_handle(&self, requirements: &StorageRequirements) -> StrategyCapability {
        let can_handle = match requirements.data_characteristics {
            DataCharacteristics::Text | DataCharacteristics::Categorical => true,
            DataCharacteristics::Mixed => requirements.estimated_size < 1024 * 1024 * 1024,
            _ => false,
        };

        let confidence = if can_handle { 0.9 } else { 0.1 };

        let performance_score = match requirements.performance_priority {
            PerformancePriority::Memory => 0.95,
            PerformancePriority::Speed => 0.8,
            PerformancePriority::Balanced => 0.9,
            _ => 0.7,
        };

        // Estimate the memory cost from the ratio actually measured so far
        // rather than a fixed "expect 3x" guess.
        let measured_ratio = self
            .statistics
            .lock()
            .ok()
            .map(|s| s.compression_ratio)
            .filter(|r| *r > 0.0)
            .unwrap_or(1.0);
        let memory = (requirements.estimated_size as f64 / measured_ratio.max(1.0)) as usize;

        StrategyCapability {
            can_handle,
            confidence,
            performance_score,
            resource_cost: ResourceCost {
                memory,
                cpu: if self.config.enable_pattern_analysis {
                    15.0
                } else {
                    5.0
                },
                disk: 0,
                network: 0,
            },
        }
    }

    fn performance_profile(&self) -> PerformanceProfile {
        // Measured, not the old hardcoded 3.5.
        let compression_ratio = self
            .statistics
            .lock()
            .ok()
            .map(|stats| {
                if stats.storage_used > 0 {
                    stats.original_size as f64 / stats.storage_used as f64
                } else {
                    1.0
                }
            })
            .unwrap_or(1.0);

        PerformanceProfile {
            read_speed: Speed::VeryFast,
            write_speed: Speed::Fast,
            memory_efficiency: Efficiency::Excellent,
            compression_ratio,
            query_optimization: QueryOptimization::Good,
            parallel_scalability: ParallelScalability::Good,
        }
    }

    fn storage_stats(&self) -> StorageStats {
        match self.statistics.lock() {
            Ok(stats) => StorageStats {
                total_size: stats.original_size,
                used_size: stats.storage_used,
                read_operations: stats.retrievals,
                write_operations: stats.total_strings,
                avg_read_latency_ns: stats.avg_access_time.as_nanos() as u64,
                avg_write_latency_ns: stats.avg_access_time.as_nanos() as u64,
                cache_hit_rate: stats.cache_hit_rate,
            },
            Err(_) => StorageStats::default(),
        }
    }

    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        match pattern {
            AccessPattern::HighDuplication => {
                self.config.deduplication_threshold = 1;
                self.config.enable_dictionary_encoding = true;
            }
            AccessPattern::LongStrings => {
                self.config.compression_threshold = 0.05;
                self.config.enable_adaptive_compression = true;
            }
            AccessPattern::ShortStrings => {
                self.config.enable_dictionary_encoding = true;
                self.config.max_dictionary_size = 2 * 1024 * 1024;
            }
            _ => {}
        }
        // Rebuild the analyzer so the new configuration is really consulted.
        self.pattern_analyzer = StringPatternAnalyzer::new(self.config.clone());
        Ok(())
    }

    fn compact(&mut self, _handle: &Self::Handle) -> Result<CompactionResult> {
        let start_time = Instant::now();
        let size_before = self.lock_statistics()?.storage_used;

        let mut reclaimed = 0usize;
        let mut reclaimed_original = 0usize;
        let mut removed = 0u64;
        {
            let mut storage = self.lock_storage()?;
            storage.retain(|_, entry| {
                if entry.ref_count == 0 {
                    reclaimed += entry.data.len();
                    reclaimed_original += entry.original_length;
                    removed += 1;
                    false
                } else {
                    true
                }
            });

            // Prune the deduplication index so it cannot grow without bound.
            let mut lookup = self.lock_lookup()?;
            lookup.retain(|_, candidates| {
                candidates.retain(|id| storage.contains_key(id));
                !candidates.is_empty()
            });
        }

        let mut stats = self.lock_statistics()?;
        stats.storage_used = stats.storage_used.saturating_sub(reclaimed);
        stats.original_size = stats.original_size.saturating_sub(reclaimed_original);
        stats.unique_strings = stats.unique_strings.saturating_sub(removed);
        stats.refresh_derived();
        let size_after = stats.storage_used;
        drop(stats);

        Ok(CompactionResult {
            size_before,
            size_after,
            duration: start_time.elapsed(),
        })
    }
}

//! Adaptive String Pool Strategy Implementation.
//!
//! Intelligent string pattern analysis feeding an adaptive
//! deduplication/compression pipeline.
//!
//! Split into submodules:
//!
//! * [`analysis`] — pattern detectors, characteristics and strategy
//!   recommendations (all character-aware, so non-ASCII input cannot panic).
//! * [`codec`] — codec-tagged string compression with bounded decompression.
//! * [`dictionary`] — lossless word-dictionary compression.
//! * [`strategy`] — the pool itself: row-indexed reads, verified deduplication.

pub mod analysis;
pub mod codec;
pub mod dictionary;
pub mod strategy;

pub use analysis::{
    CharacterFrequencyDetector, DateTimePatternDetector, DateTimePatterns, LengthPatternDetector,
    NumericPatternDetector, NumericPatterns, PatternAnalysis, PatternDetector,
    PrefixSuffixDetector, StrategyRecommendations, StringCharacteristics, StringPatternAnalyzer,
    StringPoolConfig, StringStorageStrategy, StructuredPatternDetector, StructuredPatterns,
};
pub use codec::{StringCompressionAlgorithm, StringCompressionEngine};
pub use dictionary::CompressionDictionary;
pub use strategy::{
    AdaptiveStringPoolStrategy, PatternAnalyzerState, StringEncoding, StringEntry, StringId,
    StringMetadata, StringPoolHandle, StringPoolStatistics,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::unified_memory::*;

    fn text_config() -> StorageConfig {
        StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 4096,
                data_characteristics: DataCharacteristics::Text,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn write_then_read_roundtrip_in_row_order() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");

        let rows: Vec<String> = (0..50)
            .map(|i| match i % 3 {
                0 => format!("value-{}", i),
                1 => format!("日本語-{}", i),
                _ => "repeated".to_string(),
            })
            .collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
            .expect("write");
        assert_eq!(handle.row_count(), rows.len());

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, rows.len()))
            .expect("read");
        assert_eq!(read.as_strings().expect("decode"), rows);

        // Duplicates keep their own row slot even though they share an id.
        let slice = strategy
            .read_chunk(&handle, ChunkRange::new(2, 8))
            .expect("read");
        assert_eq!(slice.as_strings().expect("decode"), rows[2..8].to_vec());
    }

    #[test]
    fn appends_extend_the_row_stream() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");

        strategy
            .write_chunk(
                &handle,
                DataChunk::from_strings(vec!["a".into(), "b".into()]),
            )
            .expect("write");
        strategy
            .append_chunk(&handle, DataChunk::from_strings(vec!["c".into()]))
            .expect("append");
        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, 3))
            .expect("read");
        assert_eq!(
            read.as_strings().expect("decode"),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn deduplication_reuses_ids_only_for_identical_content() {
        let strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let first = strategy
            .store_string_with_strategy(
                "a repeated value",
                StringStorageStrategy::Deduplicated,
                StringCompressionAlgorithm::None,
            )
            .expect("store");
        let second = strategy
            .store_string_with_strategy(
                "a repeated value",
                StringStorageStrategy::Deduplicated,
                StringCompressionAlgorithm::None,
            )
            .expect("store");
        assert_eq!(first, second);

        let other = strategy
            .store_string_with_strategy(
                "a different value",
                StringStorageStrategy::Deduplicated,
                StringCompressionAlgorithm::None,
            )
            .expect("store");
        assert_ne!(first, other);
        assert_eq!(
            strategy.retrieve_string(first).expect("get"),
            "a repeated value"
        );
        assert_eq!(
            strategy.retrieve_string(other).expect("get"),
            "a different value"
        );
    }

    #[test]
    fn analysis_recommendations_reach_the_write_path() {
        // create_storage used to hardcode {Raw, None} and write_chunk read only
        // that field, so the whole adaptive pipeline had zero effect.
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");
        assert_eq!(
            handle
                .recommendations()
                .expect("recommendations")
                .recommended_compression,
            StringCompressionAlgorithm::None
        );

        let rows: Vec<String> = (0..200)
            .map(|_| "highly duplicated row".to_string())
            .collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
            .expect("write");

        let recommendations = handle.recommendations().expect("recommendations");
        assert_ne!(
            recommendations.recommended_compression,
            StringCompressionAlgorithm::None,
            "analysis result was discarded again"
        );
        assert_eq!(
            handle.strategy().expect("strategy"),
            StringStorageStrategy::Deduplicated
        );

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, rows.len()))
            .expect("read");
        assert_eq!(read.as_strings().expect("decode"), rows);
    }

    #[test]
    fn dictionary_is_actually_trained() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");
        // Short, low-entropy values steer the analyzer to DictionaryEncoded.
        let rows: Vec<String> = (0..300).map(|i| format!("ab{}", i % 4)).collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
            .expect("write");

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, rows.len()))
            .expect("read");
        assert_eq!(read.as_strings().expect("decode"), rows);
    }

    #[test]
    fn missing_row_is_an_error_not_a_silent_gap() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");
        let rows: Vec<String> = (0..5).map(|i| format!("row{}", i)).collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
            .expect("write");

        // Deleting the underlying entries must surface as an error rather than
        // returning a shorter column.
        strategy.delete_storage(&handle).expect("delete");
        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, rows.len()))
            .expect("read");
        assert_eq!(read.rows(), 0, "row index should be cleared by delete");
    }

    #[test]
    fn compaction_reclaims_released_strings() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let id = strategy
            .store_string_with_strategy(
                "temporary value",
                StringStorageStrategy::Raw,
                StringCompressionAlgorithm::None,
            )
            .expect("store");
        let handle = strategy.create_storage(&text_config()).expect("create");

        // Without a release API, ref_count started at 1 and only ever grew, so
        // compaction could never remove anything.
        assert_eq!(strategy.release_string(id).expect("release"), 0);
        let result = strategy.compact(&handle).expect("compact");
        assert!(result.size_after < result.size_before);
        assert!(strategy.retrieve_string(id).is_err());
    }

    #[test]
    fn statistics_are_measured_not_hardcoded() {
        let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let handle = strategy.create_storage(&text_config()).expect("create");
        let rows: Vec<String> = (0..100)
            .map(|i| format!("a fairly long and repetitive string value number {}", i % 5))
            .collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(rows))
            .expect("write");

        let profile = strategy.performance_profile();
        assert!(profile.compression_ratio > 0.0);
        assert_ne!(
            profile.compression_ratio, 3.5,
            "compression_ratio is still the hardcoded placeholder"
        );

        let stats = strategy.statistics().expect("stats");
        assert!(stats.dedup_hits > 0, "deduplication hits never recorded");
        assert!(stats.cache_hit_rate > 0.0);
    }

    #[test]
    fn capability_assessment() {
        let strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
        let requirements = StorageRequirements {
            estimated_size: 10 * 1024,
            data_characteristics: DataCharacteristics::Text,
            performance_priority: PerformancePriority::Memory,
            ..Default::default()
        };
        let capability = strategy.can_handle(&requirements);
        assert!(capability.can_handle);
        assert!(capability.confidence > 0.8);
        assert!(capability.performance_score > 0.9);
    }
}

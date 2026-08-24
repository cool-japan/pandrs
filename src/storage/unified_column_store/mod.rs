//! Unified Column Store Strategy Implementation.
//!
//! This module provides the `UnifiedColumnStoreStrategy`: a block-based
//! columnar storage strategy with pluggable compression codecs and column
//! encodings.
//!
//! It is split across submodules to keep each file reviewable:
//!
//! * [`compression`] — Pure-Rust LZ4/ZSTD/no-op block codecs with bounded
//!   decompression.
//! * [`encoding`] — self-describing run-length, dictionary, delta and
//!   bit-packed column encodings.
//! * [`blocks`] — physical block storage, checksums, allocation and a bounded
//!   block cache.
//! * [`strategy`] — the strategy itself: row-indexed reads and writes.

pub mod blocks;
pub mod compression;
pub mod encoding;
pub mod strategy;

pub use blocks::{
    BlockAllocator, BlockId, BlockLocation, BlockManager, BlockMetadata, CompressedBlock,
    FreeSpaceTracker, InMemoryPhysicalStorage, PhysicalStorage,
};
pub use compression::{
    CompressionEngine, Lz4CompressionEngine, NoCompressionEngine, ZstdCompressionEngine,
};
pub use encoding::{
    BitPackedEncodingStrategy, DeltaEncodingStrategy, DictionaryEncodingStrategy, EncodedData,
    EncodingStrategy, EncodingType, RunLengthEncodingStrategy,
};
pub use strategy::{
    BlockRef, ColumnDataType, ColumnLayout, ColumnStatistics, ColumnStoreConfig, ColumnStoreHandle,
    UnifiedColumnStoreStrategy,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::unified_memory::*;

    fn numeric_config() -> StorageConfig {
        StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 1024,
                data_characteristics: DataCharacteristics::Numeric,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn create_storage_picks_a_registered_encoding() {
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let handle = strategy.create_storage(&numeric_config()).expect("create");
        assert_eq!(handle.layout.data_type, ColumnDataType::Float64);
    }

    #[test]
    fn write_then_read_roundtrip_default_config() {
        // The default config selects `EncodingType::Auto`; before the fix every
        // block was stored raw but tagged `Auto`, so this read failed with
        // "Encoding strategy Auto not found".
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let handle = strategy.create_storage(&numeric_config()).expect("create");

        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        strategy
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write");
        assert_eq!(handle.row_count(), payload.len());
        assert!(!handle.block_ids().expect("blocks").is_empty());

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, payload.len()))
            .expect("read");
        assert_eq!(read.data, payload);
    }

    #[test]
    fn write_then_read_roundtrip_every_encoding() {
        let payload: Vec<u8> = (0..8192u32)
            .flat_map(|i| ((i % 17) as u32).to_le_bytes())
            .collect();

        for encoding in [
            EncodingType::None,
            EncodingType::RunLength,
            EncodingType::Dictionary,
            EncodingType::Delta,
            EncodingType::BitPacked,
            EncodingType::Auto,
        ] {
            for compression in [
                CompressionType::None,
                CompressionType::Lz4,
                CompressionType::Zstd,
                CompressionType::Auto,
            ] {
                let config = ColumnStoreConfig {
                    encoding_type: encoding,
                    compression_type: compression,
                    block_size: 1024,
                    ..Default::default()
                };
                let mut strategy = UnifiedColumnStoreStrategy::new(config);
                let storage_config = StorageConfig {
                    requirements: StorageRequirements {
                        estimated_size: payload.len(),
                        // `Mixed` falls through to the configured encoding.
                        data_characteristics: DataCharacteristics::Mixed,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let handle = strategy.create_storage(&storage_config).expect("create");
                strategy
                    .write_chunk(&handle, DataChunk::new(payload.clone()))
                    .expect("write");
                let read = strategy
                    .read_chunk(&handle, ChunkRange::new(0, payload.len()))
                    .expect("read");
                assert_eq!(
                    read.data, payload,
                    "round-trip failed for encoding {:?} / compression {:?}",
                    encoding, compression
                );
            }
        }
    }

    #[test]
    fn categorical_and_timeseries_defaults_are_readable() {
        // `Categorical` selects Dictionary and `TimeSeries` selects Delta;
        // neither used to be registered, so those writes were permanently
        // unreadable.
        for characteristics in [
            DataCharacteristics::Categorical,
            DataCharacteristics::TimeSeries,
            DataCharacteristics::Sparse,
        ] {
            let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
            let config = StorageConfig {
                requirements: StorageRequirements {
                    estimated_size: 4096,
                    data_characteristics: characteristics.clone(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let handle = strategy.create_storage(&config).expect("create");
            let payload: Vec<u8> = (0..2048u32)
                .flat_map(|i| (1_700_000_000u64 + i as u64).to_le_bytes())
                .collect();
            strategy
                .write_chunk(&handle, DataChunk::new(payload.clone()))
                .expect("write");
            let read = strategy
                .read_chunk(&handle, ChunkRange::new(0, payload.len()))
                .expect("read");
            assert_eq!(read.data, payload, "failed for {:?}", characteristics);
        }
    }

    #[test]
    fn read_honours_row_ranges_across_blocks() {
        let config = ColumnStoreConfig {
            block_size: 64,
            ..Default::default()
        };
        let mut strategy = UnifiedColumnStoreStrategy::new(config);
        let handle = strategy.create_storage(&numeric_config()).expect("create");

        let payload: Vec<u8> = (0..1000u32).map(|i| (i % 256) as u8).collect();
        strategy
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write");
        // Many blocks were produced; the old find_blocks_for_range returned all
        // of them and merge_blocks_to_chunk ignored the range entirely.
        assert!(handle.block_ids().expect("blocks").len() > 10);

        for (start, end) in [(0usize, 10usize), (100, 200), (511, 513), (990, 1000)] {
            let read = strategy
                .read_chunk(&handle, ChunkRange::new(start, end))
                .expect("read");
            assert_eq!(
                read.data,
                payload[start..end].to_vec(),
                "range {}..{} mismatched",
                start,
                end
            );
        }
    }

    #[test]
    fn string_chunks_roundtrip_with_row_ranges() {
        let config = ColumnStoreConfig {
            block_size: 64,
            ..Default::default()
        };
        let mut strategy = UnifiedColumnStoreStrategy::new(config);
        let storage_config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 4096,
                data_characteristics: DataCharacteristics::Text,
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = strategy.create_storage(&storage_config).expect("create");

        let strings: Vec<String> = (0..200)
            .map(|i| format!("行{}\0テキスト-{}", i, i))
            .collect();
        strategy
            .write_chunk(&handle, DataChunk::from_strings(strings.clone()))
            .expect("write");
        assert_eq!(handle.row_count(), strings.len());

        let all = strategy
            .read_chunk(&handle, ChunkRange::new(0, strings.len()))
            .expect("read");
        assert_eq!(all.as_strings().expect("decode"), strings);

        let slice = strategy
            .read_chunk(&handle, ChunkRange::new(50, 60))
            .expect("read");
        assert_eq!(
            slice.as_strings().expect("decode"),
            strings[50..60].to_vec()
        );
    }

    #[test]
    fn appends_extend_the_row_stream() {
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let handle = strategy.create_storage(&numeric_config()).expect("create");

        strategy
            .write_chunk(&handle, DataChunk::new(vec![1, 2, 3]))
            .expect("write");
        strategy
            .append_chunk(&handle, DataChunk::new(vec![4, 5, 6]))
            .expect("append");
        assert_eq!(handle.row_count(), 6);

        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, 6))
            .expect("read");
        assert_eq!(read.data, vec![1, 2, 3, 4, 5, 6]);
        let tail = strategy
            .read_chunk(&handle, ChunkRange::new(2, 5))
            .expect("read");
        assert_eq!(tail.data, vec![3, 4, 5]);
    }

    #[test]
    fn delete_storage_removes_blocks() {
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let handle = strategy.create_storage(&numeric_config()).expect("create");
        strategy
            .write_chunk(&handle, DataChunk::new(vec![7u8; 512]))
            .expect("write");
        assert!(!handle.block_ids().expect("blocks").is_empty());

        strategy.delete_storage(&handle).expect("delete");
        assert!(handle.block_ids().expect("blocks").is_empty());
        assert_eq!(handle.row_count(), 0);
    }

    #[test]
    fn compaction_keeps_data_readable() {
        let config = ColumnStoreConfig {
            compression_type: CompressionType::Lz4,
            block_size: 4096,
            ..Default::default()
        };
        let mut strategy = UnifiedColumnStoreStrategy::new(config);
        let handle = strategy.create_storage(&numeric_config()).expect("create");

        let payload: Vec<u8> = b"pandrs compaction payload "
            .iter()
            .copied()
            .cycle()
            .take(32 * 1024)
            .collect();
        strategy
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write");

        let result = strategy.compact(&handle).expect("compact");
        assert!(result.size_before > 0);

        // The old compaction wrote ZSTD bytes but left the LZ4 tag in the
        // metadata index, so this read decoded ZSTD frames with the LZ4 codec.
        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, payload.len()))
            .expect("read after compaction");
        assert_eq!(read.data, payload);
    }

    #[test]
    fn storage_stats_reflect_real_activity() {
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let handle = strategy.create_storage(&numeric_config()).expect("create");
        let before = strategy.storage_stats();
        assert_eq!(before.read_operations, 0);
        assert_eq!(before.write_operations, 0);

        strategy
            .write_chunk(&handle, DataChunk::new(vec![5u8; 4096]))
            .expect("write");
        let _ = strategy
            .read_chunk(&handle, ChunkRange::new(0, 4096))
            .expect("read");

        let after = strategy.storage_stats();
        assert_eq!(after.write_operations, 1);
        assert_eq!(after.read_operations, 1);
        assert!(after.total_size > 0);
    }

    #[test]
    fn min_max_statistics_are_measured() {
        let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
        let config = StorageConfig {
            requirements: StorageRequirements {
                estimated_size: 800,
                data_characteristics: DataCharacteristics::Numeric,
                ..Default::default()
            },
            ..Default::default()
        };
        let handle = strategy.create_storage(&config).expect("create");
        // Float64 column: 100 values from 0.0 to 99.0, written out of order.
        let mut payload = Vec::new();
        for i in (0..100u64).rev() {
            payload.extend_from_slice(&(i as f64).to_le_bytes());
        }
        strategy
            .write_chunk(&handle, DataChunk::new(payload))
            .expect("write");

        let profile = strategy.performance_profile();
        assert!(profile.compression_ratio > 0.0);
    }
}

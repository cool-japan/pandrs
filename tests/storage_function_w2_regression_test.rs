//! Wave-2 regression tests for the PandRS storage engines.
//!
//! Each test below pins down a defect that shipped in 0.4.0:
//!
//! * `unified_column_store` discarded every `BlockId` on write, so written data
//!   could never be read back, and tagged blocks with encodings that were never
//!   registered (`Auto`, `Dictionary`, `Delta`).
//! * `hybrid_large_scale` discarded the `DataId` on write and fabricated
//!   `DataId(range.start)` on read.
//! * The string codecs had no discriminator byte, so decompression guessed the
//!   codec and happily produced garbage.
//! * Length headers taken from untrusted payloads were used as allocation
//!   sizes (decompression bombs).
//! * `unified_manager` never invalidated its read cache on write.

use pandrs::storage::hybrid_large_scale::{
    HybridConfig, HybridLargeScaleStrategy, TierManager, TierStorageType,
};
use pandrs::storage::unified_column_store::CompressionEngine;
use pandrs::storage::unified_column_store::{
    encoding::{
        BitPackedEncodingStrategy, DeltaEncodingStrategy, DictionaryEncodingStrategy,
        EncodingStrategy, RunLengthEncodingStrategy,
    },
    ColumnStoreConfig, EncodingType, Lz4CompressionEngine, UnifiedColumnStoreStrategy,
};
use pandrs::storage::unified_manager::StrategyMetrics;
use pandrs::storage::unified_memory::{
    ChunkLayout, ChunkRange, CompressionType, DataCharacteristics, DataChunk, StorageConfig,
    StorageRequirements, StorageStrategy, StorageType,
};
use pandrs::storage::{
    ColumnStore, DiskStorage, DurabilityLevel, MemoryConfig, MemoryMappedFile, PerformanceMonitor,
    StrategySelection, StrategySelector, UnifiedMemoryManager,
};

fn opaque_payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

fn text_rows(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| match i % 4 {
            0 => format!("row-{}", i),
            1 => format!("日本語の行 {}", i),
            2 => format!("with\0nul\0bytes {}", i),
            _ => String::new(),
        })
        .collect()
}

fn config_for(characteristics: DataCharacteristics, size: usize) -> StorageConfig {
    StorageConfig {
        requirements: StorageRequirements {
            estimated_size: size,
            data_characteristics: characteristics,
            ..Default::default()
        },
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// 1. Write-then-read round-trips, per strategy
// ---------------------------------------------------------------------------

#[test]
fn column_store_write_then_read_is_recoverable() {
    let mut strategy = UnifiedColumnStoreStrategy::new(ColumnStoreConfig::default());
    let handle = strategy
        .create_storage(&config_for(DataCharacteristics::Numeric, 8192))
        .expect("create storage");

    let payload = opaque_payload(8192);
    strategy
        .write_chunk(&handle, DataChunk::new(payload.clone()))
        .expect("write chunk");

    // Block ids used to be thrown away here, leaving the handle's index empty.
    assert!(!handle.block_ids().expect("block ids").is_empty());
    assert_eq!(handle.row_count(), payload.len());

    let read = strategy
        .read_chunk(&handle, ChunkRange::new(0, payload.len()))
        .expect("read chunk");
    assert_eq!(read.data, payload);
}

#[test]
fn column_store_read_honours_row_subranges() {
    let config = ColumnStoreConfig {
        block_size: 128,
        ..Default::default()
    };
    let mut strategy = UnifiedColumnStoreStrategy::new(config);
    let handle = strategy
        .create_storage(&config_for(DataCharacteristics::Numeric, 4096))
        .expect("create storage");

    let payload = opaque_payload(4096);
    strategy
        .write_chunk(&handle, DataChunk::new(payload.clone()))
        .expect("write chunk");

    for (start, end) in [(0usize, 1usize), (127, 129), (1000, 2000), (4095, 4096)] {
        let read = strategy
            .read_chunk(&handle, ChunkRange::new(start, end))
            .expect("read chunk");
        assert_eq!(read.data, payload[start..end], "range {}..{}", start, end);
    }
}

#[test]
fn column_store_every_encoding_round_trips() {
    // `Categorical` selected `Dictionary` and `TimeSeries` selected `Delta`,
    // but only `RunLength` was ever registered, so those writes were
    // permanently unreadable.
    let payload: Vec<u8> = (0..4096u32)
        .flat_map(|i| (1_700_000_000u64 + (i % 97) as u64).to_le_bytes())
        .collect();

    for encoding in [
        EncodingType::None,
        EncodingType::RunLength,
        EncodingType::Dictionary,
        EncodingType::Delta,
        EncodingType::BitPacked,
        EncodingType::Auto,
    ] {
        let config = ColumnStoreConfig {
            encoding_type: encoding,
            block_size: 2048,
            ..Default::default()
        };
        let mut strategy = UnifiedColumnStoreStrategy::new(config);
        let handle = strategy
            .create_storage(&config_for(DataCharacteristics::Mixed, payload.len()))
            .expect("create storage");
        strategy
            .write_chunk(&handle, DataChunk::new(payload.clone()))
            .expect("write chunk");
        let read = strategy
            .read_chunk(&handle, ChunkRange::new(0, payload.len()))
            .expect("read chunk");
        assert_eq!(read.data, payload, "encoding {:?} lost data", encoding);
    }
}

#[test]
fn hybrid_write_then_read_is_recoverable() {
    let mut strategy = HybridLargeScaleStrategy::new(HybridConfig::default());
    let handle = strategy
        .create_storage(&config_for(DataCharacteristics::Dense, 1024 * 1024))
        .expect("create storage");

    let payload = opaque_payload(4096);
    strategy
        .write_chunk(&handle, DataChunk::new(payload.clone()))
        .expect("write chunk");
    assert_eq!(handle.row_count(), payload.len());

    let read = strategy
        .read_chunk(&handle, ChunkRange::new(0, payload.len()))
        .expect("read chunk");
    assert_eq!(read.data, payload);

    let middle = strategy
        .read_chunk(&handle, ChunkRange::new(100, 200))
        .expect("read chunk");
    assert_eq!(middle.data, payload[100..200]);
}

#[test]
fn hybrid_warm_and_cold_tiers_are_really_file_backed() {
    // The "SSD" and "HDD" tiers were in-memory HashMaps padded with
    // thread::sleep; demotion freed nothing.
    let mut config = HybridConfig::default();
    config.hot_tier.max_size = 1; // force a spill out of RAM
    assert_eq!(config.warm_tier.storage_type, TierStorageType::SSD);

    let mut manager = TierManager::try_new(config).expect("tier manager");
    let payload = opaque_payload(8192);
    let id = manager
        .store_data(DataChunk::new(payload.clone()))
        .expect("store");
    assert_eq!(manager.retrieve_data(id).expect("retrieve").data, payload);
    assert!(
        manager.physical_bytes() > 0,
        "file-backed tier reported no physical bytes"
    );
}

#[test]
fn string_pool_write_then_read_is_recoverable() {
    use pandrs::storage::adaptive_string_pool::{AdaptiveStringPoolStrategy, StringPoolConfig};

    let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
    let handle = strategy
        .create_storage(&config_for(DataCharacteristics::Text, 4096))
        .expect("create storage");

    let rows = text_rows(64);
    strategy
        .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
        .expect("write chunk");

    let read = strategy
        .read_chunk(&handle, ChunkRange::new(0, rows.len()))
        .expect("read chunk");
    assert_eq!(read.rows(), rows.len());
    assert_eq!(read.as_strings().expect("decode"), rows);

    let slice = strategy
        .read_chunk(&handle, ChunkRange::new(10, 20))
        .expect("read chunk");
    assert_eq!(slice.as_strings().expect("decode"), rows[10..20].to_vec());
}

// ---------------------------------------------------------------------------
// 2. Codec tag round-trips, including non-ASCII
// ---------------------------------------------------------------------------

#[test]
fn string_codec_tags_round_trip_including_non_ascii() {
    use pandrs::storage::adaptive_string_pool::{
        StringCompressionAlgorithm, StringCompressionEngine,
    };

    let samples = [
        "plain ascii text",
        "日本語のテキストです",
        "€ £ ¥ ₿ emoji: 🐟🍣",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "",
        "mixed 日本語 and ASCII with numbers 1234567890",
    ];

    for algorithm in [
        StringCompressionAlgorithm::None,
        StringCompressionAlgorithm::RunLength,
        StringCompressionAlgorithm::Lz4,
        StringCompressionAlgorithm::Zstd,
        StringCompressionAlgorithm::StringOptimized,
    ] {
        let engine = StringCompressionEngine::new(algorithm);
        for sample in samples {
            let compressed = engine.compress(sample).expect("compress");
            let restored = engine.decompress(&compressed).expect("decompress");
            assert_eq!(
                restored, sample,
                "algorithm {:?} corrupted {:?}",
                algorithm, sample
            );
        }
    }
}

#[test]
fn string_optimized_codec_does_not_guess() {
    use pandrs::storage::adaptive_string_pool::{
        StringCompressionAlgorithm, StringCompressionEngine,
    };

    // The old StringOptimized decoder tried RLE first and fell back to LZ4.
    // RLE-decoding LZ4 bytes usually *succeeds*, yielding ~127x expanded
    // garbage with an Ok result. A high-entropy payload takes the LZ4 branch on
    // compression, so a guessing decoder mis-decodes it.
    let engine = StringCompressionEngine::new(StringCompressionAlgorithm::StringOptimized);
    let high_entropy: String = (0..4096u32)
        .map(|i| char::from(b'a' + ((i * 7919) % 26) as u8))
        .collect();
    let compressed = engine.compress(&high_entropy).expect("compress");
    assert_eq!(
        engine.decompress(&compressed).expect("decompress"),
        high_entropy
    );
}

#[test]
fn non_ascii_strings_do_not_panic_the_pattern_analyzer() {
    use pandrs::storage::adaptive_string_pool::{
        AdaptiveStringPoolStrategy, StringPatternAnalyzer, StringPoolConfig,
    };

    // PrefixSuffixDetector byte-sliced `&s[..len]` for len in 1..=5 and the
    // numeric detector did `&s[1..]` after `starts_with('€')`; both panicked on
    // any multi-byte string.
    let strings = vec![
        "日本語".to_string(),
        "€100".to_string(),
        "£42.5".to_string(),
        "🐟".to_string(),
        "aé".to_string(),
    ];

    let analyzer = StringPatternAnalyzer::new(StringPoolConfig::default());
    let characteristics = analyzer.analyze_strings(&strings).expect("analyze");
    assert!(characteristics.avg_length > 0.0);

    let mut strategy = AdaptiveStringPoolStrategy::new(StringPoolConfig::default());
    let handle = strategy
        .create_storage(&config_for(DataCharacteristics::Text, 1024))
        .expect("create storage");
    strategy
        .write_chunk(&handle, DataChunk::from_strings(strings.clone()))
        .expect("write chunk");
    let read = strategy
        .read_chunk(&handle, ChunkRange::new(0, strings.len()))
        .expect("read chunk");
    assert_eq!(read.as_strings().expect("decode"), strings);
}

#[test]
fn tier_chunk_frames_round_trip_layout_and_codec() {
    use pandrs::storage::hybrid_large_scale::{decode_chunk, encode_chunk};

    let rows = text_rows(32);
    let chunk = DataChunk::from_strings(rows.clone());
    for codec in [
        CompressionType::None,
        CompressionType::Lz4,
        CompressionType::Zstd,
    ] {
        let frame = encode_chunk(&chunk, codec).expect("encode");
        let decoded = decode_chunk(&frame).expect("decode");
        assert_eq!(decoded.layout(), ChunkLayout::Strings);
        assert_eq!(decoded.as_strings().expect("decode"), rows);
    }
}

// ---------------------------------------------------------------------------
// 3. Decompression-bomb clamps
// ---------------------------------------------------------------------------

#[test]
fn lz4_block_decompression_bomb_is_refused() {
    let engine = Lz4CompressionEngine;
    // 8-byte header claiming ~18 exabytes with a 4-byte body.
    let mut hostile = u64::MAX.to_le_bytes().to_vec();
    hostile.extend_from_slice(&[0, 0, 0, 0]);
    assert!(engine.decompress(&hostile).is_err());

    // A short-but-nonempty payload used to decode as Ok(empty).
    assert!(engine.decompress(&[1, 2, 3]).is_err());
}

#[test]
fn string_pool_decompression_bomb_is_refused() {
    use pandrs::storage::adaptive_string_pool::{
        StringCompressionAlgorithm, StringCompressionEngine,
    };

    let engine = StringCompressionEngine::new(StringCompressionAlgorithm::Lz4);
    let mut hostile = vec![StringCompressionAlgorithm::Lz4 as u8];
    hostile.extend_from_slice(&u64::MAX.to_le_bytes());
    hostile.extend_from_slice(&[0, 0, 0, 0]);
    assert!(engine.decompress(&hostile).is_err());

    // Truncated input must be an error, not an empty string.
    assert!(engine.decompress(&[0xAB, 0x01, 0x02]).is_err());
}

#[test]
fn encoded_block_expansion_bombs_are_refused() {
    // Every self-describing encoding bounds its declared original length
    // against the encoded body it actually has.
    for strategy in [
        Box::new(RunLengthEncodingStrategy) as Box<dyn EncodingStrategy>,
        Box::new(DictionaryEncodingStrategy) as Box<dyn EncodingStrategy>,
        Box::new(DeltaEncodingStrategy) as Box<dyn EncodingStrategy>,
        Box::new(BitPackedEncodingStrategy) as Box<dyn EncodingStrategy>,
    ] {
        let mut encoded = strategy.encode(&[1u8; 32]).expect("encode");
        encoded.original_size = usize::MAX;
        // Rewrite the embedded length header where the codec keeps one.
        if encoded.data.len() > 11 {
            let start = if encoded.encoding_type == EncodingType::Dictionary {
                3
            } else if encoded.encoding_type == EncodingType::BitPacked {
                3
            } else {
                2
            };
            if start + 8 <= encoded.data.len() {
                encoded.data[start..start + 8].copy_from_slice(&u64::MAX.to_le_bytes());
            }
        }
        assert!(
            strategy.decode(&encoded).is_err(),
            "{} accepted a bomb header",
            strategy.name()
        );
    }
}

#[test]
fn corrupt_string_chunk_headers_are_refused() {
    // A header claiming u64::MAX rows in a 12-byte buffer must not be trusted.
    let mut data = u64::MAX.to_le_bytes().to_vec();
    data.extend_from_slice(&[0u8; 4]);
    let chunk = DataChunk::from_encoded(data, ChunkLayout::Strings, 0).expect("construct");
    assert!(chunk.as_strings().is_err());
}

// ---------------------------------------------------------------------------
// 4. Chunk semantics shared by every strategy
// ---------------------------------------------------------------------------

#[test]
fn string_chunks_preserve_nul_bytes_and_empty_input() {
    let rows = vec![
        "a\0b".to_string(),
        String::new(),
        "日本\0語".to_string(),
        "z".to_string(),
    ];
    let chunk = DataChunk::from_strings(rows.clone());
    assert_eq!(chunk.rows(), rows.len());
    assert_eq!(chunk.as_strings().expect("decode"), rows);

    let empty = DataChunk::from_strings(Vec::new());
    assert_eq!(empty.rows(), 0, "empty input must round-trip as zero rows");
    assert!(empty.as_strings().expect("decode").is_empty());
}

#[test]
fn chunk_checksums_are_computed_and_detect_corruption() {
    let mut chunk = DataChunk::new(opaque_payload(1024));
    assert!(chunk.verify_checksum());
    chunk.data[10] ^= 0xFF;
    assert!(!chunk.verify_checksum(), "checksum field is inert");
}

// ---------------------------------------------------------------------------
// 5. UnifiedMemoryManager: usable at all, round-trips, cache invalidation
// ---------------------------------------------------------------------------

/// Selector that always names one strategy, so a test can drive a specific
/// backend through the manager's public API.
struct FixedSelector(StorageType);

impl StrategySelector for FixedSelector {
    fn select_strategy(&self, _requirements: &StorageRequirements) -> StrategySelection {
        StrategySelection {
            primary: self.0,
            fallbacks: Vec::new(),
            confidence: 1.0,
        }
    }

    fn record_performance(&mut self, _strategy_type: StorageType, _performance: &StrategyMetrics) {}
}

#[test]
fn manager_round_trips_through_every_registered_strategy() {
    // `add_strategy` had no callable form and `new()` registered nothing, so
    // `create_storage` always failed with "No suitable storage strategy" — the
    // unified manager could never succeed at anything.
    let registered = UnifiedMemoryManager::new(MemoryConfig::default()).registered_strategies();
    assert!(
        registered.len() >= 5,
        "expected the built-in strategies to be registered, got {:?}",
        registered
    );

    for strategy_type in registered {
        let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
        manager.set_selector(Box::new(FixedSelector(strategy_type)));
        let handle = manager
            .create_storage(&config_for(DataCharacteristics::Mixed, 8192))
            .unwrap_or_else(|e| panic!("create storage for {:?}: {}", strategy_type, e));
        assert_eq!(handle.strategy_type, strategy_type);

        if strategy_type == StorageType::StringPool {
            let rows = text_rows(48);
            manager
                .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
                .expect("write chunk");
            let read = manager
                .read_chunk(&handle, ChunkRange::new(0, rows.len()))
                .expect("read chunk");
            assert_eq!(read.as_strings().expect("decode"), rows);
        } else {
            let payload = opaque_payload(4096);
            manager
                .write_chunk(&handle, DataChunk::new(payload.clone()))
                .expect("write chunk");
            let read = manager
                .read_chunk(&handle, ChunkRange::new(0, payload.len()))
                .expect("read chunk");
            assert_eq!(read.data, payload, "{:?} lost data", strategy_type);
        }
    }
}

#[test]
fn manager_read_write_read_does_not_serve_stale_cache_entries() {
    // The read cache was keyed `id:start-end` and nothing ever evicted it, so
    // once a range had been read it kept answering from the pre-write bytes.
    let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
    manager.set_selector(Box::new(FixedSelector(StorageType::InMemory)));
    let handle = manager
        .create_storage(&config_for(DataCharacteristics::Numeric, 4096))
        .expect("create storage");

    manager
        .write_chunk(&handle, DataChunk::new(vec![0xAAu8; 64]))
        .expect("write chunk");

    // First read populates the cache, second read is a genuine hit.
    let first = manager
        .read_chunk(&handle, ChunkRange::new(0, 64))
        .expect("read chunk");
    assert_eq!(first.data, vec![0xAAu8; 64]);
    let cached = manager
        .read_chunk(&handle, ChunkRange::new(0, 64))
        .expect("read chunk");
    assert_eq!(cached.data, vec![0xAAu8; 64]);
    let hits_before = manager.cache_stats().expect("cache stats").hits;
    assert_eq!(hits_before, 1, "the second read should have hit the cache");

    // A write must make that cached entry unreachable.
    manager
        .append_chunk(&handle, DataChunk::new(vec![0xBBu8; 64]))
        .expect("append chunk");
    let after_write = manager
        .read_chunk(&handle, ChunkRange::new(0, 64))
        .expect("read chunk");
    assert_eq!(after_write.data, vec![0xAAu8; 64]);
    assert_eq!(
        manager.cache_stats().expect("cache stats").hits,
        hits_before,
        "the post-write read was answered from the stale cache"
    );

    let whole = manager
        .read_chunk(&handle, ChunkRange::new(0, 128))
        .expect("read chunk");
    let mut expected = vec![0xAAu8; 64];
    expected.extend_from_slice(&[0xBBu8; 64]);
    assert_eq!(whole.data, expected);

    // Deleting the storage must not leave readable bytes behind in the cache.
    manager.delete_storage(&handle).expect("delete storage");
    let after_delete = manager.read_chunk(&handle, ChunkRange::new(0, 128));
    let emptied = match after_delete {
        Ok(chunk) => chunk.data.is_empty(),
        Err(_) => true,
    };
    assert!(emptied, "delete_storage left the old bytes readable");
}

// ---------------------------------------------------------------------------
// 6. Engine-layer stores: element boundaries, real files, real mappings
// ---------------------------------------------------------------------------

#[test]
fn column_store_engine_keeps_variable_width_element_boundaries() {
    // `["a", "bb", "ccc"]` used to round-trip as `b"abbccc"`: the compressed
    // payload was a flat `Vec<u8>` with no element index at all.
    let store = ColumnStore::new();
    let values = vec![
        b"a".to_vec(),
        b"bb".to_vec(),
        Vec::new(),
        "日本語".as_bytes().to_vec(),
        b"ccc".to_vec(),
    ];
    store
        .add_column("words".to_string(), &values, "string".to_string())
        .expect("add column");

    assert_eq!(store.get_column_values("words").expect("values"), values);

    // Nulls are a validity bit, never an empty-string substitution.
    let nullable: Vec<Option<Vec<u8>>> = vec![
        Some(b"x".to_vec()),
        None,
        Some(Vec::new()),
        Some(b"yy".to_vec()),
        None,
    ];
    store
        .add_column_nullable("maybe".to_string(), &nullable, "string".to_string())
        .expect("add nullable column");
    assert_eq!(
        store
            .get_column_values_nullable("maybe")
            .expect("nullable values"),
        nullable
    );
    assert_eq!(
        store.get_metadata("maybe").expect("metadata").null_count,
        2,
        "null_count was hardcoded to 0"
    );
}

#[test]
fn disk_storage_is_really_file_backed() {
    // `DiskStorage::new` discarded its path and returned `Ok(Self {})` — an
    // empty struct re-exported from the crate root.
    let root = std::env::temp_dir().join(format!(
        "pandrs_storage_w2_disk_{}_{}",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_dir_all(&root);

    let payload = opaque_payload(2048);
    {
        let storage = DiskStorage::with_durability(&root, DurabilityLevel::Persistent)
            .expect("open disk storage");
        storage.put_chunk("chunk/../a", &payload).expect("write");
        storage.put_chunk("b", b"second").expect("write");
        assert_eq!(storage.get_chunk("chunk/../a").expect("read"), payload);
        assert!(storage.total_bytes().expect("total bytes") >= payload.len() as u64);
        // The key contains path separators; it must not escape the root.
        assert!(root.exists());
    }

    // Re-opening the same directory must find the data again: it is on disk,
    // not in a process-local HashMap.
    let reopened = DiskStorage::new(&root).expect("reopen disk storage");
    let mut keys = reopened.keys().expect("keys");
    keys.sort();
    assert_eq!(keys, vec!["b".to_string(), "chunk/../a".to_string()]);
    assert_eq!(reopened.get_chunk("chunk/../a").expect("read"), payload);
    assert!(reopened.delete_chunk("b").expect("delete"));
    assert!(reopened.get_chunk("b").is_err(), "missing chunk read as Ok");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn disk_storage_engine_round_trips_rows_and_survives_reopen() {
    use pandrs::storage::traits::{ChunkMetadata, CompressionPreference, StorageEngine};
    use pandrs::storage::traits::{DataChunk as EngineChunk, StorageConfig as EngineConfig};

    fn rows(values: &[u32]) -> EngineChunk {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        let len = data.len();
        EngineChunk::new(
            data,
            ChunkMetadata {
                row_count: values.len(),
                column_count: 1,
                compression: CompressionPreference::None,
                uncompressed_size: len,
                compressed_size: len,
            },
        )
    }

    let root = std::env::temp_dir().join(format!(
        "pandrs_storage_w2_disk_engine_{}_{}",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_dir_all(&root);

    let dataset_id;
    {
        let mut engine =
            DiskStorage::with_durability(&root, DurabilityLevel::Persistent).expect("open engine");
        let handle = engine
            .create_storage(&EngineConfig {
                estimated_size: 4096,
                access_pattern: pandrs::storage::AccessPattern::Streaming,
                performance_priority: pandrs::storage::PerformancePriority::Balanced,
                durability: DurabilityLevel::Persistent,
                compression: CompressionPreference::None,
                memory_limit: None,
            })
            .expect("create storage");
        dataset_id = handle.id;

        engine
            .write_chunk(&handle, rows(&[0, 1, 2, 3]))
            .expect("write chunk");
        engine
            .append_chunk(&handle, rows(&[4, 5, 6, 7]))
            .expect("append chunk");
        assert_eq!(handle.row_count(), 8);

        // Rows, not bytes: a sub-range must read only the files it overlaps.
        let middle = engine
            .read_chunk(&handle, 2..6)
            .expect("read chunk")
            .data
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>();
        assert_eq!(middle, vec![2, 3, 4, 5]);

        let stats = engine.storage_stats(&handle).expect("stats");
        assert_eq!(stats.chunk_count, 2);
        assert!(stats.write_operations >= 2);

        engine.compact(&handle).expect("compact");
        assert_eq!(handle.chunk_count().expect("chunk count"), 1);
        let all = engine.read_chunk(&handle, 0..8).expect("read chunk");
        assert_eq!(all.metadata.row_count, 8);
    }

    // A brand-new engine over the same directory must recover the row index.
    let engine = DiskStorage::new(&root).expect("reopen engine");
    let handle = engine.open_storage(dataset_id).expect("reopen dataset");
    assert_eq!(handle.row_count(), 8);
    let tail = engine.read_chunk(&handle, 6..8).expect("read chunk");
    assert_eq!(tail.metadata.row_count, 2);
    assert_eq!(
        tail.data
            .chunks(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>(),
        vec![6, 7]
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn engine_layer_manager_supports_write_heavy_and_random_patterns() {
    use pandrs::storage::traits::{
        ChunkMetadata, CompressionPreference, CpuBudget, LocalityPattern, ResourceConstraints,
        StorageConfig as EngineConfig, StorageRequirements as EngineRequirements,
        WorkloadCharacteristics,
    };
    use pandrs::storage::traits::{DataChunk as EngineChunk, UnifiedStorageManager};
    use pandrs::storage::{AccessPattern, PerformancePriority};

    let root = std::env::temp_dir().join(format!(
        "pandrs_storage_w2_engine_mgr_{}_{}",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_dir_all(&root);

    // `DefaultStorageStrategy` routed WriteHeavy/Streaming to DiskStorage and
    // Random to MemoryMapped, none of which the manager could create: three of
    // the six access patterns failed with `NotImplemented`.
    for pattern in [
        AccessPattern::Sequential,
        AccessPattern::Random,
        AccessPattern::ReadHeavy,
        AccessPattern::WriteHeavy,
        AccessPattern::Streaming,
        AccessPattern::Columnar,
    ] {
        let mut manager =
            UnifiedStorageManager::with_disk_root(root.join(format!("{:?}", pattern)));
        let requirements = EngineRequirements {
            config: EngineConfig {
                estimated_size: 1024,
                access_pattern: pattern,
                performance_priority: PerformancePriority::Balanced,
                durability: DurabilityLevel::Cached,
                compression: CompressionPreference::None,
                memory_limit: None,
            },
            workload: WorkloadCharacteristics {
                read_write_ratio: 0.5,
                avg_query_size: 256,
                concurrency_level: 1,
                locality_pattern: LocalityPattern::Mixed,
            },
            constraints: ResourceConstraints {
                max_memory: None,
                max_disk: None,
                cpu_budget: CpuBudget::Medium,
                network_budget: None,
            },
        };

        let handle_id = manager
            .create_storage(&requirements)
            .unwrap_or_else(|e| panic!("create storage for {:?}: {}", pattern, e));

        let payload: Vec<u8> = (0..64u8).collect();
        let len = payload.len();
        manager
            .write_chunk(
                handle_id,
                EngineChunk::new(
                    payload.clone(),
                    ChunkMetadata {
                        row_count: len,
                        column_count: 1,
                        compression: CompressionPreference::None,
                        uncompressed_size: len,
                        compressed_size: len,
                    },
                ),
            )
            .unwrap_or_else(|e| panic!("write chunk for {:?}: {}", pattern, e));

        let read = manager
            .read_chunk(handle_id, 0..len)
            .unwrap_or_else(|e| panic!("read chunk for {:?}: {}", pattern, e));
        assert_eq!(read.data, payload, "{:?} lost data", pattern);

        // The monitor field used to be stored and never written to.
        let engine_id = pandrs::storage::traits::StorageStrategy::select_engine(
            &pandrs::storage::traits::DefaultStorageStrategy::new(),
            &requirements,
        );
        assert!(
            manager.engine_metrics(engine_id).is_some(),
            "{:?} recorded no metrics",
            pattern
        );
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn memory_mapped_file_maps_real_bytes() {
    // `MemoryMappedFile::new` was an empty-struct stub that dropped the path.
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "pandrs_storage_w2_mmap_{}_{}.bin",
        std::process::id(),
        line!()
    ));
    let payload = opaque_payload(512);
    {
        let mut file = std::fs::File::create(&path).expect("create temp file");
        file.write_all(&payload).expect("write temp file");
        file.sync_all().expect("sync temp file");
    }

    let mapped = MemoryMappedFile::new(&path).expect("map file");
    assert_eq!(mapped.len(), payload.len());
    assert_eq!(mapped.as_bytes(), payload.as_slice());
    drop(mapped);
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// 7. ML strategy selection actually selects
// ---------------------------------------------------------------------------

#[test]
fn ml_selection_is_trained_and_actually_drives_storage_creation() {
    use pandrs::storage::ml_strategy_selector::SharedMlSelector;
    use pandrs::storage::{
        MLStrategySelector, PerformancePrediction, TrainingExample, WorkloadFeatures,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    let requirements = config_for(DataCharacteristics::Text, 64 * 1024).requirements;
    let monitor = Arc::new(Mutex::new(PerformanceMonitor::new()));
    let mut selector = MLStrategySelector::new(Arc::clone(&monitor));

    // Untrained, every model returned the clamped `1.0`, so the first candidate
    // in the iteration order (ColumnStore) always won.
    let untrained = selector.select_best_strategy(&requirements);
    assert_ne!(
        untrained.primary,
        StorageType::StringPool,
        "the untrained baseline already picks the strategy this test trains for"
    );

    // Teach it that StringPool is spectacular for this workload. SGD used to
    // diverge to inf/NaN on raw byte/second targets within a handful of
    // examples, and the clamp laundered the NaN into a plausible 1.0.
    let features = WorkloadFeatures::from_requirements(&requirements);
    for _ in 0..500 {
        selector.add_training_example(TrainingExample {
            features: features.clone(),
            strategy: StorageType::StringPool,
            performance: PerformancePrediction {
                throughput: 1.0e11,
                latency: 0.0001,
                memory_usage: 1.0,
                cpu_usage: 1.0,
                confidence: 0.9,
            },
            timestamp: Instant::now(),
        });
    }

    let stats = selector.get_model_stats();
    let string_pool_stats = stats
        .get(&StorageType::StringPool)
        .expect("StringPool model stats");
    assert!(
        string_pool_stats.confidence > 0.1,
        "confidence never rises above the 0.1 floor: {:?}",
        string_pool_stats
    );
    let trained = selector.select_best_strategy(&requirements);
    assert_eq!(
        trained.primary,
        StorageType::StringPool,
        "training had no effect on the selection"
    );

    // The selection must reach the manager: `create_storage_ml` used to compute
    // it, `println!` it and then delegate to a manager using a *different*
    // selector.
    let shared = Arc::new(Mutex::new(selector));
    let mut manager = UnifiedMemoryManager::new(MemoryConfig::default());
    manager.set_selector(Box::new(SharedMlSelector::new(Arc::clone(&shared))));
    let handle = manager
        .create_storage(&config_for(DataCharacteristics::Text, 64 * 1024))
        .expect("create storage");
    assert_eq!(handle.strategy_type, StorageType::StringPool);

    let rows = text_rows(32);
    manager
        .write_chunk(&handle, DataChunk::from_strings(rows.clone()))
        .expect("write chunk");
    let read = manager
        .read_chunk(&handle, ChunkRange::new(0, rows.len()))
        .expect("read chunk");
    assert_eq!(read.as_strings().expect("decode"), rows);
}

#[test]
fn adaptive_manager_creates_storage_with_its_own_ml_selection() {
    use pandrs::storage::ml_strategy_selector::AdaptiveUnifiedMemoryManager;

    let mut manager = AdaptiveUnifiedMemoryManager::new(MemoryConfig::default());
    let config = config_for(DataCharacteristics::Numeric, 32 * 1024);
    let predicted = manager
        .preview_selection(&config.requirements)
        .expect("preview selection");

    let handle = manager.create_storage_ml(&config).expect("create storage");
    let registered = manager.manager().registered_strategies();
    let expected = std::iter::once(predicted.primary)
        .chain(predicted.fallbacks.iter().copied())
        .find(|candidate| registered.contains(candidate))
        .expect("the ML selection names at least one registered strategy");
    assert_eq!(
        handle.strategy_type, expected,
        "the ML selection was computed and then discarded"
    );

    let payload = opaque_payload(1024);
    manager
        .manager()
        .write_chunk(&handle, DataChunk::new(payload.clone()))
        .expect("write chunk");
    let read = manager
        .manager()
        .read_chunk(&handle, ChunkRange::new(0, payload.len()))
        .expect("read chunk");
    assert_eq!(read.data, payload);
}

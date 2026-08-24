//! Regression tests for the `io-parquet` work item fixes to
//! `src/io/parquet/` (formerly `src/io/parquet.rs`). Each test pins one
//! previously-broken behavior so it can't silently regress.
//!
//! `tests/fixtures/mixed_arrow_types.parquet` is a small, hand-generated
//! Parquet file written directly with the `arrow`/`parquet` crates (bypassing
//! pandrs' own writer entirely, which never emits Int32/UInt32/Float32/
//! Decimal128/Date64 columns since `ColumnType` only has 4 variants). It
//! stands in for a file written by pandas/pyarrow, which does use these
//! types by default. Schema: `id: Int32`, `count: UInt32`, `ratio: Float32`
//! (non-nullable), `price: Decimal128(10, 2)`, `event_date: Date64`,
//! `name: Utf8` (non-nullable); 5 rows, with a null in every nullable column.

#![cfg(feature = "parquet")]

use std::fs;

use pandrs::io::{
    get_column_statistics, read_parquet, read_parquet_advanced, read_parquet_enhanced,
    read_parquet_with_schema_evolution, write_parquet, write_parquet_advanced,
    AdvancedParquetReadOptions, ParquetCompression, ParquetReadOptions, ParquetWriteOptions,
    PredicateFilter, SchemaEvolution, StreamingParquetReader,
};
use pandrs::OptimizedDataFrame;

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "pandrs_parquet_w1_{}_{}_{}.parquet",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        name
    ))
}

// ---------------------------------------------------------------------
// (1) F5: Int32/UInt32/Float32/Decimal128/Date64 must decode to real,
// per-cell native values -- not a per-row whole-array Debug dump.
// ---------------------------------------------------------------------

#[test]
fn read_parquet_decodes_non_native_arrow_numeric_types() {
    let bytes = include_bytes!("fixtures/mixed_arrow_types.parquet");
    let path = temp_path("mixed_types");
    fs::write(&path, bytes).unwrap();

    let df = read_parquet(&path).expect("reading externally-authored parquet must succeed");
    assert_eq!(df.row_count(), 5);

    // Int32 -> i64, with the file's own null convention (0) preserved.
    let id = df.get_column::<i64>("id").unwrap().values().to_vec();
    assert_eq!(id, vec![-5, 0, 0, 42, i32::MAX as i64]);

    // UInt32 -> i64.
    let count = df.get_column::<i64>("count").unwrap().values().to_vec();
    assert_eq!(count, vec![0, 7, 9999, 0, u32::MAX as i64]);

    // Float32 -> f64. Every fixture value is exactly representable in f32,
    // so widening introduces no rounding error.
    let ratio = df.get_column::<f64>("ratio").unwrap().values().to_vec();
    assert_eq!(ratio, vec![1.5, -2.25, 0.0, 3.125, 100.0]);

    // Decimal128(10, 2) -> f64, scaled by 10^-2.
    let price = df.get_column::<f64>("price").unwrap().values().to_vec();
    assert!((price[0] - 100.12).abs() < 1e-9);
    assert!(price[1].is_nan()); // null
    assert!((price[2] - 0.05).abs() < 1e-9);
    assert!((price[3] - -3.00).abs() < 1e-9);
    assert!((price[4] - 0.0).abs() < 1e-9);

    // Date64 -> a real ISO-8601 date, not "Date64(millis)" or a corrupted dump.
    let event_date = df
        .get_column::<String>("event_date")
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(
        event_date,
        vec![
            "2021-06-15".to_string(),
            "1970-01-01".to_string(),
            "1970-01-01".to_string(), // null convention
            "2023-11-14".to_string(),
            "2021-01-01".to_string(),
        ]
    );

    let name = df.get_column::<String>("name").unwrap().values().to_vec();
    assert_eq!(name, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);

    // Anti-regression: the old bug pushed `format!("{:?}", array)` (the
    // *whole* Arrow array) as every cell's value, which reliably contains
    // the Rust struct/array debug markers below. None of the decoded string
    // columns should ever contain them.
    for cell in &event_date {
        assert!(
            !cell.contains("PrimitiveArray") && !cell.contains('['),
            "cell looks like a whole-array debug dump, not a single value: {cell:?}"
        );
    }

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (2) StreamingParquetReader::next_chunk must read bounded, disjoint row
// ranges that sum to the file's total row count -- not the whole file,
// re-decoded, on every call.
// ---------------------------------------------------------------------

#[test]
fn streaming_reader_yields_disjoint_chunks_summing_to_total() {
    let total_rows = 25usize;
    let chunk_size = 7usize; // deliberately not a divisor of total_rows

    let mut df = OptimizedDataFrame::new();
    df.add_int_column("id", (0..total_rows as i64).collect())
        .unwrap();
    let path = temp_path("streaming_chunks");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let mut reader = StreamingParquetReader::new(&path, chunk_size).unwrap();
    assert_eq!(reader.total_chunks(), 4); // ceil(25 / 7)

    let mut chunk_row_counts = Vec::new();
    let mut all_ids: Vec<i64> = Vec::new();
    while let Some(chunk) = reader.next_chunk().unwrap() {
        chunk_row_counts.push(chunk.row_count());
        all_ids.extend(chunk.get_column::<i64>("id").unwrap().values());
    }

    assert_eq!(chunk_row_counts, vec![7, 7, 7, 4]);
    assert_eq!(all_ids.len(), total_rows);

    // Disjoint and exhaustive: every id 0..25 appears exactly once, meaning
    // no chunk re-read another chunk's rows and none were skipped.
    let mut sorted = all_ids.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..total_rows as i64).collect::<Vec<_>>());

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (8) enable_dictionary: true must not break every string-column write.
// ---------------------------------------------------------------------

#[test]
fn write_parquet_advanced_with_dictionary_enabled_round_trips_strings() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("id", vec![1, 2, 3]).unwrap();
    df.add_string_column(
        "label",
        vec!["red".to_string(), "green".to_string(), "blue".to_string()],
    )
    .unwrap();

    let path = temp_path("dictionary_strings");
    let options = ParquetWriteOptions {
        enable_dictionary: true,
        ..Default::default()
    };
    write_parquet_advanced(&df, &path, options)
        .expect("writing string columns with dictionary encoding enabled must succeed");

    let read_back = read_parquet(&path).unwrap();
    assert_eq!(read_back.row_count(), 3);
    let labels = read_back
        .get_column::<String>("label")
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(labels, vec!["red", "green", "blue"]);

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (3) Column projection must report the projected schema, not the full
// file schema, and must error (not silently ignore) an unknown column name.
// ---------------------------------------------------------------------

#[test]
fn read_parquet_advanced_projection_uses_correct_names_and_values() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("a", vec![1, 2]).unwrap();
    df.add_int_column("b", vec![10, 20]).unwrap();
    df.add_int_column("c", vec![100, 200]).unwrap();
    df.add_string_column("d", vec!["x".to_string(), "y".to_string()])
        .unwrap();

    let path = temp_path("projection");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let options = ParquetReadOptions {
        columns: Some(vec!["b".to_string(), "d".to_string()]),
        ..Default::default()
    };
    let result = read_parquet_advanced(&path, options).unwrap();

    assert_eq!(result.column_names(), &["b".to_string(), "d".to_string()]);
    assert_eq!(result.get_column::<i64>("b").unwrap().values(), &[10, 20]);
    assert_eq!(
        result.get_column::<String>("d").unwrap().values(),
        &["x".to_string(), "y".to_string()]
    );

    // Requesting a column that doesn't exist must error, not silently
    // fall back to returning every column.
    let bad_options = ParquetReadOptions {
        columns: Some(vec!["nonexistent".to_string()]),
        ..Default::default()
    };
    assert!(read_parquet_advanced(&path, bad_options).is_err());

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (4) get_column_statistics must surface real min/max/distinct values,
// not the "N/A" placeholder, whenever the file actually has them.
// ---------------------------------------------------------------------

#[test]
fn get_column_statistics_reports_real_min_max() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("score", vec![30, 10, 20, 10]).unwrap();

    let path = temp_path("statistics");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let stats = get_column_statistics(&path).unwrap();
    let score_stats = stats
        .iter()
        .find(|s| s.name == "score")
        .expect("score column stats must be present");

    assert_ne!(score_stats.min_value.as_deref(), Some("N/A"));
    assert_ne!(score_stats.max_value.as_deref(), Some("N/A"));
    assert_eq!(score_stats.min_value.as_deref(), Some("10"));
    assert_eq!(score_stats.max_value.as_deref(), Some("30"));
    assert_eq!(score_stats.null_count, Some(0));

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (6) streaming_mode + predicate_filters must combine, not silently drop
// the predicates (this function's own doc example uses the combination).
// ---------------------------------------------------------------------

#[test]
fn read_parquet_enhanced_applies_predicates_in_streaming_mode() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("value", (0..20i64).collect()).unwrap();

    let path = temp_path("streaming_predicates");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let options = AdvancedParquetReadOptions {
        predicate_filters: vec![PredicateFilter::Range(
            "value".to_string(),
            "5".to_string(),
            "9".to_string(),
        )],
        streaming_mode: true,
        streaming_chunk_size: 6, // forces multiple chunks over 20 rows
        ..Default::default()
    };

    let result = read_parquet_enhanced(&path, options).unwrap();
    let mut values = result.get_column::<i64>("value").unwrap().values().to_vec();
    values.sort_unstable();
    assert_eq!(values, vec![5, 6, 7, 8, 9]);

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (6, related) streaming_mode combined with base_options.columns must
// error explicitly, not silently return every column (the same
// "field the streaming branch never reads" bug as the predicates/schema-
// evolution case above, for a third field on the same struct).
// ---------------------------------------------------------------------

#[test]
fn read_parquet_enhanced_streaming_with_projection_errors_explicitly() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("a", (0..10i64).collect()).unwrap();
    df.add_int_column("b", (0..10i64).collect()).unwrap();

    let path = temp_path("streaming_projection_unsupported");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let options = AdvancedParquetReadOptions {
        base_options: ParquetReadOptions {
            columns: Some(vec!["a".to_string()]),
            ..Default::default()
        },
        streaming_mode: true,
        streaming_chunk_size: 4,
        ..Default::default()
    };

    let err = read_parquet_enhanced(&path, options)
        .expect_err("streaming_mode + column projection must be rejected, not silently ignored");
    assert!(
        err.to_string().to_lowercase().contains("stream"),
        "unexpected error message: {err}"
    );

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (5, related) read_parquet_streaming's memory_limit must fail loudly when
// exceeded, not silently return a truncated-but-"successful" DataFrame:
// every chunk is accumulated into one Vec<RecordBatch> before conversion,
// so "stop early" would otherwise be indistinguishable from "read the
// whole file".
// ---------------------------------------------------------------------

#[test]
fn read_parquet_streaming_errors_instead_of_silently_truncating_over_memory_limit() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("value", (0..30i64).collect()).unwrap();

    let path = temp_path("streaming_memory_limit");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let options = AdvancedParquetReadOptions {
        streaming_mode: true,
        streaming_chunk_size: 5,
        // The estimator is ~100 bytes/row, so 5 rows/chunk means the first
        // chunk alone (~500 bytes) already exceeds this: the read must fail
        // after the very first chunk rather than quietly returning a
        // 5-of-30-row DataFrame as if it were the complete file.
        memory_limit: Some(100),
        ..Default::default()
    };

    let err = read_parquet_enhanced(&path, options)
        .expect_err("exceeding memory_limit must error, not return a truncated DataFrame");
    let message = err.to_string();
    assert!(
        message.contains("memory_limit"),
        "unexpected error message: {message}"
    );

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (7) SchemaEvolution::columns_to_remove must actually drop the column,
// and must not disturb the type of any column it didn't touch.
// ---------------------------------------------------------------------

#[test]
fn schema_evolution_removes_columns_without_corrupting_remaining_types() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("keep_int", vec![1, 2, 3]).unwrap();
    df.add_float_column("keep_float", vec![1.5, 2.5, 3.5])
        .unwrap();
    df.add_string_column(
        "drop_me",
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    )
    .unwrap();

    let path = temp_path("schema_evolution_remove");
    write_parquet(&df, &path, Some(ParquetCompression::Snappy)).unwrap();

    let mut evolution = SchemaEvolution::default();
    evolution.columns_to_remove.push("drop_me".to_string());

    let result = read_parquet_with_schema_evolution(&path, evolution).unwrap();

    assert!(!result.contains_column("drop_me"));
    assert!(result.contains_column("keep_int"));
    assert!(result.contains_column("keep_float"));

    // The untouched columns must keep their real, native type -- not be
    // silently widened/stringified as a side effect of removing a sibling.
    assert_eq!(
        result.get_column::<i64>("keep_int").unwrap().values(),
        &[1, 2, 3]
    );
    assert_eq!(
        result.get_column::<f64>("keep_float").unwrap().values(),
        &[1.5, 2.5, 3.5]
    );

    fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------
// (9) Zstd/Lzo must fail with an explicit, honest error naming the Pure
// Rust policy / implementation gap -- not a raw codec failure.
// ---------------------------------------------------------------------

#[test]
fn zstd_and_lzo_compression_return_explicit_errors() {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("x", vec![1, 2, 3]).unwrap();

    let zstd_path = temp_path("zstd_rejected");
    let err = write_parquet(&df, &zstd_path, Some(ParquetCompression::Zstd))
        .expect_err("Zstd must be rejected explicitly, not silently succeed or panic");
    let message = err.to_string();
    assert!(
        message.contains("Zstd") && message.to_lowercase().contains("pure rust"),
        "unexpected error message for Zstd: {message}"
    );

    let lzo_path = temp_path("lzo_rejected");
    let err = write_parquet(&df, &lzo_path, Some(ParquetCompression::Lzo))
        .expect_err("Lzo must be rejected explicitly, not silently succeed or panic");
    let message = err.to_string();
    assert!(
        message.contains("Lzo"),
        "unexpected error message for Lzo: {message}"
    );

    // Neither call should have left a corrupt/partial file behind.
    assert!(!zstd_path.exists());
    assert!(!lzo_path.exists());
}

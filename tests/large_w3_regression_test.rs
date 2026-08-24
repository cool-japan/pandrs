//! Wave 3 regression tests for `pandrs::large`.
//!
//! Covers:
//! - `DataFrameOperations::group_by` on `DiskBasedDataFrame` actually
//!   applying the aggregation function (it previously ignored it entirely
//!   and returned raw, unaggregated values).
//! - `ChunkedDataFrame::next_chunk` on a short file (where
//!   `calculate_total_chunks`'s row-count estimate overshoots the real
//!   chunk count) no longer re-emitting chunk 0's rows for chunks past the
//!   end of the data, for both the memory-mapped and standard-I/O loading
//!   paths.

use pandrs::error::Result;
use pandrs::large::DataFrameOperations;
use pandrs::{
    external_sort, hash_join_out_of_core, ChunkedDataFrame, DataFrame, DiskBasedDataFrame,
    DiskConfig, OutOfCoreConfig, OutOfCoreJoinType, Series,
};
use std::fs::File;
use std::io::Write;

fn unique_temp_csv(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "pandrs_w3_{}_{}_{}.csv",
        name,
        std::process::id(),
        // A second differentiator so tests run back-to-back in the same
        // process don't collide if a previous run's file wasn't cleaned up.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

fn write_test_csv(path: &std::path::Path, n_rows: usize) {
    let mut f = File::create(path).expect("create csv");
    writeln!(f, "id,category,value").expect("write header");
    for i in 0..n_rows {
        let category = if i % 2 == 0 { "A" } else { "B" };
        writeln!(f, "{},{},{}", i, category, i * 10).expect("write row");
    }
}

fn read_all_chunks(path: &std::path::Path, config: DiskConfig) -> Vec<DataFrame> {
    let mut chunked = ChunkedDataFrame::new(path, Some(config)).expect("create ChunkedDataFrame");
    let mut chunks = Vec::new();
    while let Some(chunk) = chunked.next_chunk().expect("next_chunk") {
        chunks.push(chunk.clone());
    }
    chunks
}

/// Shared assertions for the short-file boundary fix, parameterized over
/// `use_memory_mapping` so both `load_chunk_mmap` and `load_chunk_standard`
/// are exercised.
fn assert_short_file_chunking_is_correct(use_memory_mapping: bool) {
    const N_ROWS: usize = 10;
    const CHUNK_SIZE: usize = 5;

    let path = unique_temp_csv(&format!("short_mmap_{}", use_memory_mapping));
    write_test_csv(&path, N_ROWS);

    let config = DiskConfig {
        memory_limit: 1024 * 1024,
        temp_dir: None,
        chunk_size: CHUNK_SIZE,
        use_memory_mapping,
    };

    let chunks = read_all_chunks(&path, config);
    let _ = std::fs::remove_file(&path);

    // `calculate_total_chunks` estimates row count from a sample that
    // includes the header line, which biases the estimate high -- for this
    // 10-row/chunk_size-5 fixture it can ask for more chunks than really
    // exist. The chunk index(es) past the real data must report `None`
    // (ending `next_chunk`'s `while let Some(..)` loop) rather than
    // fabricating an extra chunk by re-reading chunk 0's byte range.
    let total_rows: usize = chunks.iter().map(|c| c.row_count()).sum();
    assert_eq!(
        total_rows, N_ROWS,
        "total rows across all chunks must equal the real row count \
         (use_memory_mapping={}); a value larger than {} means an extra \
         chunk re-emitted earlier rows",
        use_memory_mapping, N_ROWS
    );

    // Every chunk must carry the REAL column names. Chunks after the first
    // previously lost them to `csv::Reader`-invented "column_0".."column_N"
    // placeholders (`load_chunk_standard`) when parsed headerless.
    let expected_columns = vec![
        "id".to_string(),
        "category".to_string(),
        "value".to_string(),
    ];
    for (idx, chunk) in chunks.iter().enumerate() {
        assert_eq!(
            chunk.column_names().to_vec(),
            expected_columns,
            "chunk {} (use_memory_mapping={}) has the wrong column names",
            idx,
            use_memory_mapping
        );
    }

    // The `id` values across every chunk, concatenated, must be exactly
    // 0..N_ROWS with no duplicates and no gaps. A duplicate would mean a
    // later chunk re-emitted an earlier chunk's rows; a gap would mean a
    // chunk dropped its first data row.
    let mut ids: Vec<i64> = Vec::new();
    for chunk in &chunks {
        for row in 0..chunk.row_count() {
            let v: i64 = chunk
                .get_string_value("id", row)
                .expect("get id")
                .parse()
                .expect("parse id");
            ids.push(v);
        }
    }
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..N_ROWS as i64).collect::<Vec<i64>>(),
        "use_memory_mapping={}",
        use_memory_mapping
    );
}

#[test]
fn test_short_file_chunk_boundary_memory_mapped() {
    assert_short_file_chunking_is_correct(true);
}

#[test]
fn test_short_file_chunk_boundary_standard_io() {
    assert_short_file_chunking_is_correct(false);
}

/// Sums the (numeric) values in a group -- used to prove `group_by` applies
/// the aggregation function instead of ignoring it.
fn sum_agg(values: Vec<String>) -> Result<String> {
    let sum: f64 = values
        .iter()
        .map(|v| v.parse::<f64>().expect("value should be numeric"))
        .sum();
    Ok(sum.to_string())
}

#[test]
fn test_group_by_applies_aggregation_function() {
    let path = unique_temp_csv("groupby");
    {
        let mut f = File::create(&path).expect("create csv");
        writeln!(f, "category,value").expect("write header");
        writeln!(f, "A,10").expect("write row");
        writeln!(f, "B,3").expect("write row");
        writeln!(f, "A,15").expect("write row");
        writeln!(f, "B,7").expect("write row");
        writeln!(f, "A,25").expect("write row");
    }

    // A small chunk_size forces the 5 rows across 3 chunks (2 + 2 + 1),
    // exercising the per-chunk-collect / combiner-reduces-once path -- a
    // chunk-size-dependent partial aggregation would give a DIFFERENT
    // (still wrong) answer here than with the default chunk size.
    let config = DiskConfig {
        chunk_size: 2,
        ..Default::default()
    };
    let disk_df = DiskBasedDataFrame::new(&path, Some(config)).expect("create DiskBasedDataFrame");
    let result = disk_df
        .group_by("category", "value", sum_agg)
        .expect("group_by");
    let _ = std::fs::remove_file(&path);

    // A: 10 + 15 + 25 = 50, B: 3 + 7 = 10. Previously `_agg_func` was
    // ignored entirely and every group's RAW values were returned
    // unaggregated (e.g. `["10", "15", "25"]` for A).
    assert_eq!(
        result.get("A").expect("group 'A' must be present"),
        &vec!["50".to_string()]
    );
    assert_eq!(
        result.get("B").expect("group 'B' must be present"),
        &vec!["10".to_string()]
    );
}

#[test]
fn test_group_by_result_independent_of_chunk_size() {
    let path = unique_temp_csv("groupby_chunksize");
    {
        let mut f = File::create(&path).expect("create csv");
        writeln!(f, "category,value").expect("write header");
        for i in 0..20 {
            let category = if i % 3 == 0 { "X" } else { "Y" };
            writeln!(f, "{},{}", category, i).expect("write row");
        }
    }

    let mut results = Vec::new();
    for chunk_size in [3usize, 7, 100] {
        let config = DiskConfig {
            chunk_size,
            ..Default::default()
        };
        let disk_df =
            DiskBasedDataFrame::new(&path, Some(config)).expect("create DiskBasedDataFrame");
        let result = disk_df
            .group_by("category", "value", sum_agg)
            .expect("group_by");
        results.push((
            result.get("X").cloned().unwrap_or_default(),
            result.get("Y").cloned().unwrap_or_default(),
        ));
    }
    let _ = std::fs::remove_file(&path);

    // Aggregating over the complete value list per group (not a per-chunk
    // partial reduce) means the result must not depend on how the input
    // happened to be chunked.
    assert_eq!(results[0], results[1], "chunk_size 3 vs 7 disagree");
    assert_eq!(results[1], results[2], "chunk_size 7 vs 100 disagree");
}

#[test]
fn test_hash_join_out_of_core_numeric_right_key_matches() {
    let left_path = unique_temp_csv("join_left");
    {
        let mut f = File::create(&left_path).expect("create left csv");
        writeln!(f, "id,left_val").expect("write header");
        writeln!(f, "1,a").expect("write row");
        writeln!(f, "2,b").expect("write row");
        writeln!(f, "3,c").expect("write row");
    }

    // The right side's join key is NUMERIC (`Series<i64>`), which is
    // exactly the case that previously broke: `get_string_value` errors
    // for any non-`Series<String>` column, and `.unwrap_or("")` silently
    // turned every one of those errors into the empty string, collapsing
    // every right row into a single "" bucket.
    let mut right = DataFrame::new();
    right
        .add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("build id series"),
        )
        .expect("add id column");
    right
        .add_column(
            "right_val".to_string(),
            Series::new(
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
                Some("right_val".to_string()),
            )
            .expect("build right_val series"),
        )
        .expect("add right_val column");

    let temp_dir = std::env::temp_dir().join(format!("pandrs_w3_join_tmp_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).expect("create join temp dir");
    let config = OutOfCoreConfig {
        chunk_size: 100,
        temp_dir: temp_dir.clone(),
        ..Default::default()
    };

    let writer = hash_join_out_of_core(
        left_path.to_str().expect("left path as str"),
        &right,
        "id",
        "id",
        OutOfCoreJoinType::Inner,
        &config,
    )
    .expect("hash_join_out_of_core");

    let result = writer.collect().expect("collect joined result");
    let _ = std::fs::remove_file(&left_path);
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Before the fix, EVERY right row's key collapsed to "", and an inner
    // join against the left side's real "1"/"2"/"3" text never matched
    // anything -- the joined result had 0 rows regardless of how many
    // rows actually shared a key value.
    assert_eq!(
        result.row_count(),
        3,
        "inner join on a numeric key must match all 3 rows, not silently join on \
         collapsed empty-string keys"
    );

    let mut pairs: Vec<(String, String)> = (0..result.row_count())
        .map(|i| {
            (
                result
                    .get_string_value("left_val", i)
                    .expect("left_val")
                    .to_string(),
                result
                    .get_string_value("right_val", i)
                    .expect("right_val")
                    .to_string(),
            )
        })
        .collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("a".to_string(), "x".to_string()),
            ("b".to_string(), "y".to_string()),
            ("c".to_string(), "z".to_string()),
        ]
    );
}

#[test]
fn test_external_sort_mixed_numeric_and_text_is_transitive() {
    let input_path = unique_temp_csv("sort_mixed_input");
    {
        let mut f = File::create(&input_path).expect("create input csv");
        writeln!(f, "key,payload").expect("write header");
        writeln!(f, "10,ten").expect("write row");
        writeln!(f, "2,two").expect("write row");
        writeln!(f, "1a,onea").expect("write row");
    }
    let output_path = unique_temp_csv("sort_mixed_output");

    let temp_dir = std::env::temp_dir().join(format!("pandrs_w3_sort_tmp_{}", std::process::id()));
    std::fs::create_dir_all(&temp_dir).expect("create sort temp dir");
    let config = OutOfCoreConfig {
        chunk_size: 100,
        temp_dir: temp_dir.clone(),
        ..Default::default()
    };

    external_sort(
        input_path.to_str().expect("input path str"),
        output_path.to_str().expect("output path str"),
        "key",
        true,
        &config,
    )
    .expect("external_sort");

    let sorted = pandrs::io::csv::read_csv(&output_path, true).expect("read sorted output");
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
    let _ = std::fs::remove_dir_all(&temp_dir);

    let keys: Vec<String> = (0..sorted.row_count())
        .map(|i| {
            sorted
                .get_string_value("key", i)
                .expect("get key")
                .to_string()
        })
        .collect();

    // Numeric values sort before non-numeric text, in NUMERIC order among
    // themselves ("2" before "10" by magnitude -- lexicographically "10" <
    // "2"); non-numeric "1a" sorts last. The previous per-pair "parse both,
    // else compare raw text" comparator is not a valid total order for
    // this exact input: it judges "2" < "10" (both parse), "10" < "1a"
    // (text fallback since "1a" doesn't parse), yet ALSO "1a" < "2" (also
    // a text fallback) -- an inconsistent three-way cycle that produces
    // some arbitrary, not-necessarily-reproducible order instead of this
    // one.
    assert_eq!(
        keys,
        vec!["2".to_string(), "10".to_string(), "1a".to_string()]
    );
}

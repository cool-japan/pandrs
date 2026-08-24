//! Wave 3b regression tests for `pandrs::large`.
//!
//! Covers two gaps found while re-verifying Wave 3's `src/large/**` fixes
//! that were not already covered by `tests/large_w3_regression_test.rs`:
//!
//! - `ChunkedDataFrame::find_chunk_boundaries` (used by the memory-mapped
//!   loading path) silently dropped a file's final data row whenever that
//!   row had no trailing newline: the boundary scan only advances its `end`
//!   offset on a `'\n'` byte, so bytes after the last newline (i.e. an
//!   unterminated final line) were never included. `load_chunk_standard`
//!   (the non-mmap path) never had this bug -- `BufRead::read_line` already
//!   returns a final unterminated line rather than discarding it -- so the
//!   two loaders previously disagreed on row count for such a file. Covers
//!   both the chunk-0 boundary computation and the non-first-chunk one
//!   (distinct branches in `find_chunk_boundaries`).
//! - `OutOfCoreConfig::parallelism` is now honored by `OutOfCoreReader::map`
//!   via a dedicated `rayon::ThreadPoolBuilder` (previously declared but
//!   never consulted, so it had no effect regardless of value). Checks that
//!   varying the configured thread-pool size doesn't change the result --
//!   `map`'s output order is index-preserving via
//!   `chunk_input_paths.par_iter().enumerate().map(..).collect()`
//!   regardless of how many threads actually run the work.

// `OutOfCoreReader::map`'s closure bound is `Fn(DataFrame) -> Result<DataFrame>`,
// so any caller closure that can fail is architecturally forced to return
// `pandrs::Error` (232+ bytes) by value; matches the identical, pre-existing
// allow in `tests/out_of_core_test.rs` for the same reason.
#![allow(clippy::result_large_err)]

use pandrs::error::Result;
use pandrs::{ChunkedDataFrame, DiskConfig, OutOfCoreConfig, OutOfCoreReader};
use std::fs::File;
use std::io::Write;

fn unique_temp_path(name: &str) -> std::path::PathBuf {
    // `name` is placed last so a caller-supplied extension (e.g.
    // "foo.csv") ends up as the real suffix of the returned path, not
    // buried in the middle of it.
    std::env::temp_dir().join(format!(
        "pandrs_w3b_{}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
        name
    ))
}

// ---------------------------------------------------------------------------
// find_chunk_boundaries: final row with no trailing newline
// ---------------------------------------------------------------------------

/// Write a CSV file with a header and `n_rows` data rows, WITHOUT a trailing
/// newline after the last row (unlike `writeln!`-based helpers elsewhere,
/// which always terminate the last line).
fn write_csv_no_trailing_newline(path: &std::path::Path, n_rows: usize) {
    let mut f = File::create(path).expect("create csv");
    writeln!(f, "id,value").expect("write header");
    let mut lines: Vec<String> = (0..n_rows).map(|i| format!("{},{}", i, i * 100)).collect();
    let last = lines.pop().expect("at least one row");
    for line in &lines {
        writeln!(f, "{}", line).expect("write row");
    }
    // No trailing '\n' after the final row.
    write!(f, "{}", last).expect("write final row without newline");
}

fn read_all_ids(path: &std::path::Path, config: DiskConfig) -> Vec<i64> {
    let mut chunked = ChunkedDataFrame::new(path, Some(config)).expect("create ChunkedDataFrame");
    let mut ids = Vec::new();
    while let Some(chunk) = chunked.next_chunk().expect("next_chunk") {
        for row in 0..chunk.row_count() {
            let v: i64 = chunk
                .get_string_value("id", row)
                .expect("get id")
                .parse()
                .expect("parse id");
            ids.push(v);
        }
    }
    ids
}

#[test]
fn test_final_row_without_trailing_newline_kept_single_chunk_mmap() {
    // All 10 rows fit in one chunk (chunk_size 50): exercises the
    // `chunk_index == 0` branch of `find_chunk_boundaries`.
    const N_ROWS: usize = 10;
    let path = unique_temp_path("no_trailing_nl_single_mmap.csv");
    write_csv_no_trailing_newline(&path, N_ROWS);

    let config = DiskConfig {
        memory_limit: 1024 * 1024,
        temp_dir: None,
        chunk_size: 50,
        use_memory_mapping: true,
    };
    let mut ids = read_all_ids(&path, config);
    let _ = std::fs::remove_file(&path);

    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..N_ROWS as i64).collect::<Vec<i64>>(),
        "the last row (no trailing newline) must not be dropped"
    );
}

#[test]
fn test_final_row_without_trailing_newline_kept_multi_chunk_mmap() {
    // 10 rows over chunk_size 4 -> chunks of [4, 4, 2] rows: the final,
    // unterminated row falls in the LAST chunk, exercising the non-first
    // (`else`) branch of `find_chunk_boundaries`.
    const N_ROWS: usize = 10;
    let path = unique_temp_path("no_trailing_nl_multi_mmap.csv");
    write_csv_no_trailing_newline(&path, N_ROWS);

    let config = DiskConfig {
        memory_limit: 1024 * 1024,
        temp_dir: None,
        chunk_size: 4,
        use_memory_mapping: true,
    };
    let mut ids = read_all_ids(&path, config);
    let _ = std::fs::remove_file(&path);

    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..N_ROWS as i64).collect::<Vec<i64>>(),
        "the last row (no trailing newline) must not be dropped across chunk boundaries"
    );
}

#[test]
fn test_final_row_without_trailing_newline_matches_standard_io() {
    // `load_chunk_standard` (BufRead::read_line) never had this bug; the
    // mmap path must now agree with it on both row count and last-row
    // content, not just count.
    const N_ROWS: usize = 7;
    let path = unique_temp_path("no_trailing_nl_parity.csv");
    write_csv_no_trailing_newline(&path, N_ROWS);

    let mmap_config = DiskConfig {
        memory_limit: 1024 * 1024,
        temp_dir: None,
        chunk_size: 100,
        use_memory_mapping: true,
    };
    let standard_config = DiskConfig {
        use_memory_mapping: false,
        ..mmap_config.clone()
    };

    let mut mmap_ids = read_all_ids(&path, mmap_config);
    let mut standard_ids = read_all_ids(&path, standard_config);
    let _ = std::fs::remove_file(&path);

    mmap_ids.sort_unstable();
    standard_ids.sort_unstable();
    assert_eq!(
        mmap_ids, standard_ids,
        "memory-mapped and standard-I/O loaders must agree on row count/content \
         even when the file's last row has no trailing newline"
    );
    assert_eq!(mmap_ids, (0..N_ROWS as i64).collect::<Vec<i64>>());
}

// ---------------------------------------------------------------------------
// OutOfCoreConfig::parallelism honored by OutOfCoreReader::map
// ---------------------------------------------------------------------------

fn write_ooc_csv(path: &std::path::Path, n_rows: usize) {
    let mut f = File::create(path).expect("create csv");
    writeln!(f, "id,value").expect("write header");
    for i in 0..n_rows {
        writeln!(f, "{},{}", i, i * 3).expect("write row");
    }
}

/// Run a `map` that doubles `value`, using the given thread-pool size, and
/// return the resulting `(id, value)` pairs sorted by `id`.
fn run_doubling_map(path: &std::path::Path, parallelism: usize) -> Vec<(i64, i64)> {
    let temp_dir = unique_temp_path(&format!("ooc_parallelism_{}", parallelism));
    std::fs::create_dir_all(&temp_dir).expect("create temp dir");

    let config = OutOfCoreConfig {
        chunk_size: 20,
        temp_dir: temp_dir.clone(),
        parallelism,
        ..Default::default()
    };
    let reader = OutOfCoreReader::from_csv(path, config).expect("create reader");

    let writer = reader
        .map(|chunk| -> Result<pandrs::DataFrame> {
            let row_count = chunk.row_count();
            let ids: Vec<String> = (0..row_count)
                .map(|i| chunk.get_string_value("id", i).unwrap_or("0").to_string())
                .collect();
            let doubled: Vec<String> = (0..row_count)
                .map(|i| {
                    let v: i64 = chunk
                        .get_string_value("value", i)
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0);
                    (v * 2).to_string()
                })
                .collect();

            let mut out = pandrs::DataFrame::new();
            out.add_column(
                "id".to_string(),
                pandrs::Series::new(ids, Some("id".to_string()))?,
            )?;
            out.add_column(
                "value".to_string(),
                pandrs::Series::new(doubled, Some("value".to_string()))?,
            )?;
            Ok(out)
        })
        .expect("map");

    let result_df = writer.collect().expect("collect");
    let mut pairs: Vec<(i64, i64)> = (0..result_df.row_count())
        .map(|i| {
            let id: i64 = result_df
                .get_string_value("id", i)
                .expect("id")
                .parse()
                .expect("parse id");
            let value: i64 = result_df
                .get_string_value("value", i)
                .expect("value")
                .parse()
                .expect("parse value");
            (id, value)
        })
        .collect();
    pairs.sort_unstable_by_key(|(id, _)| *id);

    let _ = std::fs::remove_dir_all(&temp_dir);
    pairs
}

#[test]
fn test_map_parallelism_variants_agree_and_are_correct() {
    const N_ROWS: usize = 97; // deliberately not a multiple of chunk_size (20)
    let path = unique_temp_path("ooc_parallelism_input.csv");
    write_ooc_csv(&path, N_ROWS);

    let single_threaded = run_doubling_map(&path, 1);
    let multi_threaded = run_doubling_map(&path, 4);
    let _ = std::fs::remove_file(&path);

    let expected: Vec<(i64, i64)> = (0..N_ROWS as i64).map(|i| (i, i * 3 * 2)).collect();

    assert_eq!(
        single_threaded, expected,
        "parallelism=1 must produce the correct doubled values in id order"
    );
    assert_eq!(
        multi_threaded, expected,
        "parallelism=4 must produce the same correct result as parallelism=1 \
         (map's output order is index-preserving regardless of how many \
         threads actually execute the work)"
    );
}

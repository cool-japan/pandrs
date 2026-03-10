//! Integration tests for out-of-core processing capabilities.
//!
//! All temporary files are created in `std::env::temp_dir()`.
#![allow(clippy::result_large_err)]
#![allow(clippy::redundant_closure)]

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use pandrs::external_sort;
use pandrs::{OutOfCoreAggOp, OutOfCoreConfig, OutOfCoreReader};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a CSV file at `path` with columns: id, value, category.
/// `n_rows` data rows are written (plus one header row).
fn generate_test_csv(path: &PathBuf, n_rows: usize) {
    let file = File::create(path).expect("create temp csv");
    let mut w = BufWriter::new(file);
    writeln!(w, "id,value,category").expect("write header");
    for i in 0..n_rows {
        let cat = if i % 3 == 0 {
            "A"
        } else if i % 3 == 1 {
            "B"
        } else {
            "C"
        };
        writeln!(w, "{},{},{}", i, i * 2, cat).expect("write row");
    }
}

fn default_config(sub: &str) -> OutOfCoreConfig {
    OutOfCoreConfig {
        chunk_size: 1_000,
        temp_dir: env::temp_dir().join(sub),
        ..Default::default()
    }
}

fn ensure_temp_dir(config: &OutOfCoreConfig) {
    std::fs::create_dir_all(&config.temp_dir).expect("create temp dir");
}

// ---------------------------------------------------------------------------
// Test: count rows
// ---------------------------------------------------------------------------

#[test]
fn test_count_rows() {
    let n_rows = 10_000usize;
    let path = env::temp_dir().join("ooc_count_test.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_count");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");
    let count = reader.count().expect("count rows");
    assert_eq!(count, n_rows, "row count should match generated rows");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test: collect (small enough to fit in RAM)
// ---------------------------------------------------------------------------

#[test]
fn test_collect() {
    let n_rows = 5_000usize;
    let path = env::temp_dir().join("ooc_collect_test.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_collect");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");
    let df = reader.collect().expect("collect");
    assert_eq!(df.row_count(), n_rows, "collected row count should match");
    assert!(df.contains_column("id"));
    assert!(df.contains_column("value"));
    assert!(df.contains_column("category"));
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test: foreach (count via side effects)
// ---------------------------------------------------------------------------

#[test]
fn test_foreach() {
    let n_rows = 8_000usize;
    let path = env::temp_dir().join("ooc_foreach_test.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_foreach");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");

    let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter_clone = counter.clone();
    reader
        .foreach(move |chunk| {
            counter_clone.fetch_add(chunk.row_count(), std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })
        .expect("foreach");

    assert_eq!(
        counter.load(std::sync::atomic::Ordering::Relaxed),
        n_rows,
        "foreach should visit every row"
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test: map + collect (filter rows where value > threshold)
// ---------------------------------------------------------------------------

#[test]
fn test_map_filter_collect() {
    let n_rows = 10_000usize;
    let threshold = 5_000u64; // value = id * 2, so id >= 2501 qualifies
    let path = env::temp_dir().join("ooc_map_test.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_map");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");

    let writer = reader
        .map(move |chunk| {
            // Keep rows where value > threshold
            let row_count = chunk.row_count();
            let mut keep_ids: Vec<String> = Vec::new();
            let mut keep_values: Vec<String> = Vec::new();
            let mut keep_cats: Vec<String> = Vec::new();

            for i in 0..row_count {
                let val_str = chunk.get_string_value("value", i).unwrap_or("0");
                let val: u64 = val_str.parse().unwrap_or(0);
                if val > threshold {
                    keep_ids.push(chunk.get_string_value("id", i).unwrap_or("").to_string());
                    keep_values.push(val_str.to_string());
                    keep_cats.push(
                        chunk
                            .get_string_value("category", i)
                            .unwrap_or("")
                            .to_string(),
                    );
                }
            }

            let mut out = pandrs::DataFrame::new();
            if !keep_ids.is_empty() {
                out.add_column(
                    "id".to_string(),
                    pandrs::Series::new(keep_ids, Some("id".to_string())).expect("series id"),
                )
                .expect("add id");
                out.add_column(
                    "value".to_string(),
                    pandrs::Series::new(keep_values, Some("value".to_string()))
                        .expect("series value"),
                )
                .expect("add value");
                out.add_column(
                    "category".to_string(),
                    pandrs::Series::new(keep_cats, Some("category".to_string()))
                        .expect("series cat"),
                )
                .expect("add category");
            }
            Ok(out)
        })
        .expect("map");

    let result_df = writer.collect().expect("collect after map");

    // value = id * 2; value > 5000 means id > 2500, i.e. ids 2501..9999 → 7499 rows
    let expected = n_rows - 2501;
    assert_eq!(
        result_df.row_count(),
        expected,
        "filtered row count should be {}",
        expected
    );
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test: aggregate (sum, mean, count)
// ---------------------------------------------------------------------------

#[test]
fn test_aggregate() {
    let n_rows = 10_000usize;
    let path = env::temp_dir().join("ooc_agg_test.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_agg");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");

    let agg_result = reader
        .aggregate(&[
            ("value", OutOfCoreAggOp::Sum),
            ("value", OutOfCoreAggOp::Mean),
            ("id", OutOfCoreAggOp::Count),
        ])
        .expect("aggregate");

    // Result columns: "value_sum", "value_mean", "id_count"
    // Expected: value = id * 2, ids 0..9999
    // sum(value) = sum(0, 2, 4, ..., 19998) = 2 * sum(0..9999) = 2 * (9999*10000/2) = 99_990_000
    let expected_sum = 99_990_000.0f64;
    let sum_str = agg_result
        .get_string_value("value_sum", 0)
        .expect("get value_sum");
    let sum_val: f64 = sum_str.parse().expect("parse sum");
    assert!(
        (sum_val - expected_sum).abs() < 1.0,
        "sum mismatch: got {}, expected {}",
        sum_val,
        expected_sum
    );

    // Expected mean = 99_990_000 / 10_000 = 9999
    let mean_str = agg_result
        .get_string_value("value_mean", 0)
        .expect("get value_mean");
    let mean_val: f64 = mean_str.parse().expect("parse mean");
    assert!(
        (mean_val - 9999.0).abs() < 1.0,
        "mean mismatch: got {}, expected 9999",
        mean_val
    );

    let count_str = agg_result
        .get_string_value("id_count", 0)
        .expect("get id_count");
    let count_val: f64 = count_str.parse().expect("parse count");
    assert_eq!(count_val as usize, n_rows, "count should equal n_rows");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test: external_sort on a medium dataset
// ---------------------------------------------------------------------------

#[test]
fn test_external_sort() {
    let n_rows = 20_000usize;
    let input_path = env::temp_dir().join("ooc_sort_input.csv");
    let output_path = env::temp_dir().join("ooc_sort_output.csv");

    // Generate unsorted CSV
    {
        let file = File::create(&input_path).expect("create sort input");
        let mut w = BufWriter::new(file);
        writeln!(w, "id,value").expect("header");
        // Write in reverse order so sorting is non-trivial
        for i in (0..n_rows).rev() {
            writeln!(w, "{},{}", i, i * 3).expect("row");
        }
    }

    let config = OutOfCoreConfig {
        chunk_size: 5_000,
        temp_dir: env::temp_dir().join("ooc_sort_tmp"),
        ..Default::default()
    };
    std::fs::create_dir_all(&config.temp_dir).expect("create sort temp dir");

    external_sort(
        input_path.to_str().expect("input path str"),
        output_path.to_str().expect("output path str"),
        "id",
        true,
        &config,
    )
    .expect("external sort");

    // Verify the output is sorted by id ascending
    let sorted_df = pandrs::io::csv::read_csv(&output_path, true).expect("read sorted output");
    assert_eq!(sorted_df.row_count(), n_rows, "sorted output row count");

    let first_id: i64 = sorted_df
        .get_string_value("id", 0)
        .expect("first id")
        .parse()
        .expect("parse first id");
    let last_id: i64 = sorted_df
        .get_string_value("id", n_rows - 1)
        .expect("last id")
        .parse()
        .expect("parse last id");
    assert_eq!(first_id, 0, "first id should be 0");
    assert_eq!(last_id, (n_rows - 1) as i64, "last id should be n_rows-1");

    // Verify strict ascending order (spot-check every 1000 rows)
    let mut prev_id: i64 = -1;
    for i in 0..n_rows {
        let id_str = sorted_df.get_string_value("id", i).expect("id row");
        let id: i64 = id_str.parse().expect("parse id");
        assert!(id > prev_id, "ids must be strictly ascending at row {}", i);
        prev_id = id;
    }

    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&output_path);
}

// ---------------------------------------------------------------------------
// Test: write_csv from OutOfCoreWriter
// ---------------------------------------------------------------------------

#[test]
fn test_map_write_csv() {
    let n_rows = 3_000usize;
    let path = env::temp_dir().join("ooc_write_test.csv");
    let out_path = env::temp_dir().join("ooc_write_out.csv");
    generate_test_csv(&path, n_rows);

    let config = default_config("ooc_write");
    ensure_temp_dir(&config);
    let reader = OutOfCoreReader::from_csv(&path, config).expect("create reader");

    let writer = reader.map(|chunk| Ok(chunk)).expect("identity map");
    writer.write_csv(&out_path).expect("write csv");

    let result_df = pandrs::io::csv::read_csv(&out_path, true).expect("read written csv");
    assert_eq!(result_df.row_count(), n_rows);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&out_path);
}

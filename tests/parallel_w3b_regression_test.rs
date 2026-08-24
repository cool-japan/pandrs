//! Regression tests for the parallel-perf fixes in:
//! * `src/parallel/mod.rs` (`DataFrame::par_filter_rows`, `par_groupby`, `par_apply`,
//!   `Series`/`NASeries` `par_map`/`par_filter`, `ParallelUtils`)
//! * `src/optimized/split_dataframe/parallel.rs` (`OptimizedDataFrame::par_filter`)
//!
//! Covers: the O(n^2) `Vec::contains` scan removed from `par_groupby`, dtype
//! preservation (no more `f64 -> String -> f64` round trips) in
//! `par_filter_rows`/`par_groupby`, real row-level parallelism (not a
//! `chunks()` loop nobody parallelized) in `OptimizedDataFrame::par_filter`,
//! NULL preservation across both, and the small-input serial fast paths.

use std::collections::HashMap;

use pandrs::index::IndexTrait;
use pandrs::{BooleanColumn, Column, DataFrame, Int64Column, OptimizedDataFrame, Series};

// ---------------------------------------------------------------------------
// src/parallel/mod.rs -- DataFrame::par_filter_rows / par_groupby
// ---------------------------------------------------------------------------

/// `par_filter_rows` must keep each column's exact source type: an i64
/// column must still downcast as `Series<i64>` after filtering, not have
/// silently become `Series<String>` (the old implementation always
/// stringified every column via `get_column_string_values`).
#[test]
fn par_filter_rows_preserves_i64_dtype_not_string() {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new((0..50i64).collect(), Some("id".to_string())).expect("series"),
    )
    .expect("add_column");
    df.add_column(
        "value".to_string(),
        Series::new(
            (0..50i64).map(|v| v * 10).collect(),
            Some("value".to_string()),
        )
        .expect("series"),
    )
    .expect("add_column");

    let filtered = df.par_filter_rows(|i| i % 2 == 0).expect("par_filter_rows");

    assert_eq!(filtered.row_count(), 25);

    // Still typed as i64: get_column::<i64> must succeed...
    let ids = filtered.get_column::<i64>("id").expect("id stayed i64");
    assert_eq!(ids.values(), &(0..50i64).step_by(2).collect::<Vec<_>>()[..]);
    let values = filtered
        .get_column::<i64>("value")
        .expect("value stayed i64");
    assert_eq!(
        values.values(),
        &(0..50i64).step_by(2).map(|v| v * 10).collect::<Vec<_>>()[..]
    );

    // ...and must NOT have been silently turned into a String column.
    assert!(
        filtered.get_column::<String>("id").is_err(),
        "par_filter_rows must not stringify a numeric column"
    );
}

/// A NaN value already present in an f64 column is the only "missing value"
/// convention the base `DataFrame` type carries through these generic paths
/// (it has no null bitmap of its own). It must come out of
/// `par_filter_rows`/`par_groupby` exactly as NaN -- never coerced to 0.0 --
/// since every kept value is moved with a plain `clone()`, never
/// substituted.
#[test]
fn par_filter_rows_preserves_nan_never_substitutes_zero() {
    let mut df = DataFrame::new();
    let values = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
    df.add_column(
        "v".to_string(),
        Series::new(values, Some("v".to_string())).expect("series"),
    )
    .expect("add_column");

    let filtered = df.par_filter_rows(|_| true).expect("par_filter_rows");
    let out = filtered.get_column::<f64>("v").expect("f64 column");
    assert_eq!(out.len(), 5);
    assert_eq!(out.values()[0], 1.0);
    assert!(out.values()[1].is_nan(), "NaN must survive, not become 0.0");
    assert_eq!(out.values()[2], 3.0);
    assert!(out.values()[3].is_nan(), "NaN must survive, not become 0.0");
    assert_eq!(out.values()[4], 5.0);
}

/// Small frames (below the parallel threshold) must still work correctly
/// through the serial fast path, with no crash and exact dtype/value
/// preservation.
#[test]
fn par_filter_rows_small_input_correct_no_crash() {
    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![10i64, 20, 30], Some("v".to_string())).expect("series"),
    )
    .expect("add_column");

    let filtered = df.par_filter_rows(|i| i != 1).expect("par_filter_rows");
    assert_eq!(filtered.row_count(), 2);
    let out = filtered.get_column::<i64>("v").expect("i64 column");
    assert_eq!(out.values(), &[10, 30]);

    // Degenerate cases must not crash either.
    let empty_df = DataFrame::new();
    let empty_result = empty_df.par_filter_rows(|_| true).expect("empty ok");
    assert_eq!(empty_result.row_count(), 0);

    let none_kept = df.par_filter_rows(|_| false).expect("par_filter_rows");
    assert_eq!(none_kept.row_count(), 0);
}

/// A serial, obviously-correct reference implementation of "group row
/// indices by a key function", used to check `par_groupby` produces exactly
/// the same partition -- this is the ground truth `par_groupby` itself must
/// match, independent of its internal (parallel or serial) strategy.
fn serial_group_indices<K: Fn(usize) -> String>(
    row_count: usize,
    key_func: K,
) -> HashMap<String, Vec<usize>> {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for i in 0..row_count {
        groups.entry(key_func(i)).or_default().push(i);
    }
    groups
}

/// `par_groupby` must match the serial partition exactly, per key (a
/// `HashMap`'s own iteration order is never guaranteed, so this compares
/// key-by-key rather than iterating both maps in lockstep) -- including a
/// NaN value in the grouped column, which must survive ungathered/unchanged
/// rather than becoming 0.0. This is also the direct regression test for the
/// removed O(n^2) `par_filter_rows(|i| indices.contains(&i))` scan: with the
/// old implementation this test still passed (it was slow, not wrong), so
/// the real guard against a regression back to `contains()` is the perf
/// characteristic exercised by `par_groupby_large_input_no_crash` below, not
/// this correctness check alone.
#[test]
fn par_groupby_matches_serial_grouping_including_nan() {
    let n = 97usize; // deliberately not a multiple of the group count
    let mut df = DataFrame::new();
    let category: Vec<i64> = (0..n as i64).map(|i| i % 5).collect();
    let mut value: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // Sprinkle NaN across several groups (including group 0, which is what
    // row 0's key maps to) so the check below exercises "NaN in more than
    // one output partition", not a single lucky slot.
    for &row in &[0usize, 12, 43, 90] {
        value[row] = f64::NAN;
    }

    df.add_column(
        "category".to_string(),
        Series::new(category.clone(), Some("category".to_string())).expect("series"),
    )
    .expect("add_column");
    df.add_column(
        "value".to_string(),
        Series::new(value.clone(), Some("value".to_string())).expect("series"),
    )
    .expect("add_column");

    let key_func = |i: usize| (i % 5).to_string();
    let expected = serial_group_indices(n, key_func);
    let grouped = df.par_groupby(key_func).expect("par_groupby");

    assert_eq!(grouped.len(), expected.len());
    for (key, expected_indices) in &expected {
        let group_df = grouped
            .get(key)
            .unwrap_or_else(|| panic!("missing group '{key}'"));
        assert_eq!(
            group_df.row_count(),
            expected_indices.len(),
            "group '{key}' row count"
        );

        let got_values = group_df.get_column::<f64>("value").expect("f64 column");
        // Ascending row order within the group: `par_groupby` gathers each
        // group's *already-known* indices directly, and those indices were
        // pushed in ascending row order by the sequential grouping pass, so
        // this must match the serial reference position-for-position, not
        // just as a set.
        for (pos, &row) in expected_indices.iter().enumerate() {
            let expected_v = value[row];
            let got_v = got_values.values()[pos];
            if expected_v.is_nan() {
                assert!(
                    got_v.is_nan(),
                    "group '{key}' pos {pos}: NaN must survive, not become 0.0"
                );
            } else {
                assert_eq!(got_v, expected_v, "group '{key}' pos {pos}");
            }
        }
    }
}

/// Large enough to cross `DATAFRAME_PARALLEL_THRESHOLD` in both the
/// key-computation pass and the per-group gather, without crashing or
/// hanging. This is the shape that made the old `contains()`-based
/// implementation prohibitively slow (O(groups * rows * group_size)); it
/// must now complete quickly.
#[test]
fn par_groupby_large_input_no_crash_and_matches_serial() {
    let n = 20_000usize;
    let mut df = DataFrame::new();
    let ids: Vec<i64> = (0..n as i64).collect();
    df.add_column(
        "id".to_string(),
        Series::new(ids, Some("id".to_string())).expect("series"),
    )
    .expect("add_column");

    let key_func = |i: usize| (i % 37).to_string();
    let expected = serial_group_indices(n, key_func);
    let grouped = df.par_groupby(key_func).expect("par_groupby");

    assert_eq!(grouped.len(), 37);
    for (key, expected_indices) in &expected {
        let group_df = grouped
            .get(key)
            .unwrap_or_else(|| panic!("missing group '{key}'"));
        assert_eq!(group_df.row_count(), expected_indices.len());
        let got_ids = group_df.get_column::<i64>("id").expect("i64 column");
        let expected_ids: Vec<i64> = expected_indices.iter().map(|&i| i as i64).collect();
        assert_eq!(got_ids.values(), &expected_ids[..]);
    }
}

/// Perf-sanity guard against the removed O(n^2) `par_filter_rows(|i|
/// indices.contains(&i))` scan: at 200k rows / 50 groups the old
/// `Vec::contains`-per-row-per-group implementation is
/// `O(groups * rows * avg_group_size)` = 50 * 200_000 * 4_000 = 4*10^10
/// operations -- many minutes at best. `gather_rows_typed`'s direct
/// `O(group_size)`-per-group gather finishes in well under a second on any
/// reasonable machine, so a generous 10s budget has ample margin without
/// making the test flaky under CI load, while still failing loudly if
/// someone reintroduces the quadratic scan.
#[test]
fn par_groupby_is_not_quadratic_in_row_count() {
    let n = 200_000usize;
    let mut df = DataFrame::new();
    let ids: Vec<i64> = (0..n as i64).collect();
    df.add_column(
        "id".to_string(),
        Series::new(ids, Some("id".to_string())).expect("series"),
    )
    .expect("add_column");

    let start = std::time::Instant::now();
    let grouped = df
        .par_groupby(|i| (i % 50).to_string())
        .expect("par_groupby");
    let elapsed = start.elapsed();

    assert_eq!(grouped.len(), 50);
    let total: usize = grouped.values().map(|g| g.row_count()).sum();
    assert_eq!(total, n, "every row must land in exactly one group");
    assert!(
        elapsed.as_secs() < 10,
        "par_groupby took {elapsed:?} for {n} rows / 50 groups -- looks quadratic again"
    );
}

// ---------------------------------------------------------------------------
// src/optimized/split_dataframe/parallel.rs -- OptimizedDataFrame::par_filter
// ---------------------------------------------------------------------------

/// Small selection (below the row-parallel threshold): must delegate
/// correctly (via `take_rows`) with no crash and exact dtype preservation.
#[test]
fn split_par_filter_small_input_correct_no_crash() {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "keep",
        Column::Boolean(BooleanColumn::new(vec![true, false, true, true, false])),
    )
    .expect("add_column");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2, 3, 4, 5])))
        .expect("add_column");

    let filtered = df.par_filter("keep").expect("par_filter");
    assert_eq!(filtered.row_count(), 3);

    let view = filtered.column("v").expect("column v");
    let col = view.as_int64().expect("int64 column");
    let got: Vec<Option<i64>> = (0..filtered.row_count())
        .map(|i| col.get(i).expect("in range"))
        .collect();
    assert_eq!(got, vec![Some(1), Some(3), Some(4)]);
}

/// One column, tens of thousands of rows selected: the only way this can
/// parallelize is row-level gather within that single column (there is no
/// second column for column-level `par_iter` to fall back on). Also plants a
/// NULL well past the first element (not just at row 0) so a gather that
/// silently dropped or misplaced null-mask bits under real parallelism would
/// be caught.
#[test]
fn split_par_filter_single_column_large_selection_parallelizes_and_preserves_dtype_and_nulls() {
    let n = 20_000usize;
    let mut df = OptimizedDataFrame::new();
    df.add_column("keep", Column::Boolean(BooleanColumn::new(vec![true; n])))
        .expect("add_column");

    let mut data: Vec<i64> = (0..n as i64).collect();
    let mut nulls = vec![false; n];
    // NULLs scattered across the selection, including well past the start
    // (rows 1, 9_999, 19_998) so a gather that only gets the first element
    // right (a classic off-by-chunk-boundary bug) would fail this.
    for &row in &[1usize, 9_999, 19_998] {
        nulls[row] = true;
        data[row] = -1; // placeholder; must be masked out, not read back
    }
    df.add_column(
        "v",
        Column::Int64(Int64Column::with_nulls(data.clone(), nulls.clone())),
    )
    .expect("add_column");

    let filtered = df.par_filter("keep").expect("par_filter");
    assert_eq!(filtered.row_count(), n);

    let view = filtered.column("v").expect("column v");
    let col = view
        .as_int64()
        .expect("int64 column (dtype preserved, not stringified)");
    for i in 0..n {
        let got = col.get(i).expect("in range");
        if nulls[i] {
            assert_eq!(
                got, None,
                "row {i} must stay NULL, not become the placeholder value"
            );
        } else {
            assert_eq!(
                got,
                Some(data[i]),
                "row {i} value must be preserved exactly"
            );
        }
    }
}

/// Many columns, a moderate (below the row-parallel threshold) selection
/// per column but a large aggregate cell count: exercises the column-level
/// parallel branch and must still preserve every column's dtype and values.
#[test]
fn split_par_filter_many_columns_moderate_rows_preserves_all_columns() {
    let n = 4_000usize; // below ROW_GATHER_PARALLEL_THRESHOLD per column
    let mut df = OptimizedDataFrame::new();
    df.add_column("keep", Column::Boolean(BooleanColumn::new(vec![true; n])))
        .expect("add_column");
    for c in 0..6 {
        let data: Vec<i64> = (0..n as i64).map(|i| i + c as i64).collect();
        df.add_column(format!("c{c}"), Column::Int64(Int64Column::new(data)))
            .expect("add_column");
    }

    let filtered = df.par_filter("keep").expect("par_filter");
    assert_eq!(filtered.row_count(), n);
    for c in 0..6 {
        let view = filtered.column(&format!("c{c}")).expect("column exists");
        let col = view.as_int64().expect("int64 column");
        for i in 0..n {
            assert_eq!(col.get(i).expect("in range"), Some(i as i64 + c as i64));
        }
    }
}

/// The index must be re-selected to match the filtered rows, not the
/// original (longer) frame's index copied verbatim onto a shorter result.
#[test]
fn split_par_filter_reselects_index_to_match_filtered_rows() {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "keep",
        Column::Boolean(BooleanColumn::new(vec![true, false, true, false, true])),
    )
    .expect("add_column");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2, 3, 4, 5])))
        .expect("add_column");
    let idx = pandrs::index::Index::with_name(
        vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ],
        None,
    )
    .expect("index");
    df.set_index_from_simple_index(idx).expect("set index");

    let filtered = df.par_filter("keep").expect("par_filter");
    assert_eq!(filtered.row_count(), 3);
    let result_index = filtered.get_index().expect("index present");
    assert_eq!(
        result_index.len(),
        3,
        "index length must match the filtered row count, not the original frame's"
    );
}

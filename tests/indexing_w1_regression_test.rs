//! Regression tests for the `indexing` fixes: `src/dataframe/indexing.rs`.
//!
//! Each test is labeled with the item it covers from the task list:
//! (1) O(1)-ish `.at`/`.iat`/`.iloc` scalar access instead of materialising
//!     whole columns per read; (3) `.loc` resolves real string index
//!     labels instead of parsing the label as a position; (4) `reset_index`
//!     moves the real index into a column and installs a fresh range
//!     index; (5) `set_index` promotes a column into the index instead of
//!     just deleting it; (6) `.at`/`.iat` setters really mutate a cell;
//!     (7) `reindex` aligns against real labels and NA-fills instead of
//!     emitting literal `"NaN"` strings; (8) alignment padding is
//!     NA-filled instead of cyclically repeating real observations; (9)
//!     `drop_columns` preserves original column order; (10) row-selecting
//!     operations preserve each column's concrete dtype.

use pandrs::dataframe::{AdvancedIndexingExt, AlignmentStrategy, IndexAligner};
use pandrs::{DataFrame, DataFrameIndex, Series};

fn string_series(values: &[&str]) -> Series<String> {
    Series::new(values.iter().map(|s| s.to_string()).collect(), None).unwrap()
}

// ---------------------------------------------------------------------
// (1) `.iat` is a genuine scalar read: correct values, and no longer
// O(rows) per read.
// ---------------------------------------------------------------------

#[test]
fn iat_reads_correct_values_without_materialising_whole_columns() {
    const ROWS: usize = 20_000;
    let mut df = DataFrame::new();
    df.add_column(
        "n".to_string(),
        Series::new((0..ROWS as i64).collect(), None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "f".to_string(),
        Series::new((0..ROWS).map(|i| i as f64 * 0.5).collect(), None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "s".to_string(),
        Series::new((0..ROWS).map(|i| format!("row-{}", i)).collect(), None).unwrap(),
    )
    .unwrap();

    // Correctness at a handful of positions, including the last row (the
    // easiest place to get an off-by-one on `row_count` wrong).
    for &row in &[0usize, 1, ROWS / 2, ROWS - 2, ROWS - 1] {
        assert_eq!(df.iat().get(row, 0).unwrap(), row.to_string());
        assert_eq!(
            df.iat().get(row, 1).unwrap(),
            format!("{}", row as f64 * 0.5)
        );
        assert_eq!(df.iat().get(row, 2).unwrap(), format!("row-{}", row));
    }
    assert!(df.iat().get(ROWS, 0).is_err());
    assert!(df.iat().get(0, 3).is_err());

    // Soft algorithmic-complexity guard. Before this fix, every single-cell
    // `.iat()` read materialised the *entire* column as `Vec<String>`
    // (O(rows)), so a loop of `k` reads cost O(k * rows). The bound below
    // is generous enough to tolerate a loaded/slow machine while still
    // failing hard if that quadratic behaviour is reintroduced (which
    // would take vastly longer than this for 3,000 reads over 20,000
    // rows).
    let start = std::time::Instant::now();
    let mut total_len = 0usize;
    for i in 0..3_000usize {
        let row = i % ROWS;
        let col = i % 3;
        total_len += df.iat().get(row, col).unwrap().len();
    }
    assert!(total_len > 0);
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "3,000 .iat() reads over a {}-row DataFrame took {:?}; expected O(1)-ish scalar \
         access, not O(rows) per read",
        ROWS,
        elapsed
    );
}

// ---------------------------------------------------------------------
// (3) `.loc` is label-based against the real row index, with positional
// fallback only when no explicit index was set.
// ---------------------------------------------------------------------

#[test]
fn loc_resolves_real_string_index_labels_not_positions() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["x1", "x2", "x3"]))
        .unwrap();
    df.add_column(
        "value".to_string(),
        Series::new(vec![100i64, 200, 300], None).unwrap(),
    )
    .unwrap();

    let indexed = df.set_index("id").unwrap();

    // "x2" is a genuine label, not a valid usize -- this only passes when
    // `.loc` actually resolves against the real row index rather than
    // trying (and failing) to parse the label as a position.
    let row = indexed.loc().get("x2").unwrap();
    assert_eq!(row.get("value"), Some(&"200".to_string()));
    assert_eq!(indexed.loc().get_at("x3", "value").unwrap(), "300");

    // A label that isn't in the index is an error, not a silent fallback.
    assert!(indexed.loc().get("does-not-exist").is_err());
}

#[test]
fn loc_falls_back_to_positional_parsing_without_an_explicit_index() {
    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        Series::new(vec![100i64, 200, 300], None).unwrap(),
    )
    .unwrap();

    // No explicit index was ever set: `.loc` on the implicit RangeIndex
    // still accepts integer(-like) string labels positionally, matching
    // pandas' behaviour on a RangeIndex-backed frame.
    let row = df.loc().get("1").unwrap();
    assert_eq!(row.get("value"), Some(&"200".to_string()));
    assert!(df.loc().get("99").is_err());
}

// ---------------------------------------------------------------------
// (4) + (5) `set_index` promotes a column into the row index, and
// `reset_index` is its exact inverse.
// ---------------------------------------------------------------------

#[test]
fn set_index_then_reset_index_round_trips() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["r0", "r1", "r2"]))
        .unwrap();
    df.add_column(
        "value".to_string(),
        Series::new(vec![10i64, 20, 30], None).unwrap(),
    )
    .unwrap();

    let indexed = df.set_index("id").unwrap();
    assert!(!indexed.contains_column("id"));
    assert!(indexed.contains_column("value"));
    match indexed.get_index() {
        DataFrameIndex::Simple(idx) => {
            assert_eq!(
                idx.values().to_vec(),
                vec!["r0".to_string(), "r1".to_string(), "r2".to_string()]
            );
            assert_eq!(idx.name().map(|s| s.as_str()), Some("id"));
        }
        DataFrameIndex::Multi(_) => panic!("expected a simple index"),
    }

    let restored = indexed.reset_index().unwrap();
    assert!(restored.contains_column("id"));
    assert!(restored.contains_column("value"));
    assert_eq!(
        restored.get_column_string_values("id").unwrap(),
        vec!["r0".to_string(), "r1".to_string(), "r2".to_string()]
    );
    // The "value" column's dtype survives the round trip (not stringified).
    assert_eq!(
        restored.get_column::<i64>("value").unwrap().values(),
        &[10, 20, 30]
    );
}

#[test]
fn set_index_rejects_duplicate_column_values() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["a", "a", "b"]))
        .unwrap();
    df.add_column(
        "value".to_string(),
        Series::new(vec![1i64, 2, 3], None).unwrap(),
    )
    .unwrap();

    // Duplicate index labels are rejected crate-wide by `Index::new`; a
    // column with repeated values should surface that honestly rather
    // than building a lossy/ambiguous index.
    assert!(df.set_index("id").is_err());
}

#[test]
fn reset_index_without_explicit_index_materialises_range_index_column() {
    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        Series::new(vec![10i64, 20, 30], None).unwrap(),
    )
    .unwrap();

    let reset = df.reset_index().unwrap();
    assert!(reset.contains_column("index"));
    assert_eq!(
        reset.get_column::<i64>("index").unwrap().values(),
        &[0, 1, 2]
    );
    assert_eq!(
        reset.get_column::<i64>("value").unwrap().values(),
        &[10, 20, 30]
    );
}

// ---------------------------------------------------------------------
// (6) `.at`/`.iat` setters really mutate a cell (previously
// `Error::NotImplemented` after a dead full-DataFrame clone).
// ---------------------------------------------------------------------

#[test]
fn at_setter_mutates_single_cell_preserving_others_and_dtype() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["a", "b", "c"]))
        .unwrap();
    df.add_column(
        "score".to_string(),
        Series::new(vec![1i64, 2, 3], None).unwrap(),
    )
    .unwrap();

    let indexed = df.set_index("id").unwrap();
    let updated = indexed.at().set("b", "score", "99".to_string()).unwrap();

    assert_eq!(
        updated.get_column::<i64>("score").unwrap().values(),
        &[1, 99, 3]
    );
    // Row labels are untouched by a cell mutation.
    match updated.get_index() {
        DataFrameIndex::Simple(idx) => {
            assert_eq!(
                idx.values().to_vec(),
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
        DataFrameIndex::Multi(_) => panic!("expected a simple index"),
    }

    // A nonexistent label is an error, not a silent no-op.
    assert!(indexed
        .at()
        .set("does-not-exist", "score", "1".to_string())
        .is_err());
}

#[test]
fn iat_setter_mutates_single_cell_by_position() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(vec![1i64, 2, 3], None).unwrap(),
    )
    .unwrap();
    df.add_column("b".to_string(), string_series(&["x", "y", "z"]))
        .unwrap();

    let updated = df.iat().set(1, 0, "42".to_string()).unwrap();
    assert_eq!(
        updated.get_column::<i64>("a").unwrap().values(),
        &[1, 42, 3]
    );
    assert_eq!(
        updated.get_column::<String>("b").unwrap().values(),
        &["x".to_string(), "y".to_string(), "z".to_string()]
    );

    let updated_str = df.iat().set(2, 1, "zeta".to_string()).unwrap();
    assert_eq!(
        updated_str.get_column::<String>("b").unwrap().values(),
        &["x".to_string(), "y".to_string(), "zeta".to_string()]
    );

    // `Series` is immutable: the original DataFrame is never mutated in
    // place, a new one is returned.
    assert_eq!(df.get_column::<i64>("a").unwrap().values(), &[1, 2, 3]);
}

// ---------------------------------------------------------------------
// (7) `reindex` aligns against real labels; numeric columns NaN-fill
// missing rows, non-numeric columns error instead of fabricating "NaN".
// ---------------------------------------------------------------------

#[test]
fn reindex_aligns_to_real_labels_and_nan_fills_missing_numeric_rows() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["a", "b", "c"]))
        .unwrap();
    df.add_column(
        "value".to_string(),
        Series::new(vec![1.0f64, 2.0, 3.0], None).unwrap(),
    )
    .unwrap();

    let indexed = df.set_index("id").unwrap();
    let new_index = vec!["c".to_string(), "missing".to_string(), "a".to_string()];
    let reindexed = IndexAligner::reindex(&indexed, &new_index).unwrap();

    let values = reindexed.get_column_numeric_values("value").unwrap();
    assert_eq!(values[0], 3.0);
    assert!(values[1].is_nan());
    assert_eq!(values[2], 1.0);

    match reindexed.get_index() {
        DataFrameIndex::Simple(idx) => assert_eq!(idx.values().to_vec(), new_index),
        DataFrameIndex::Multi(_) => panic!("expected a simple index"),
    }
}

#[test]
fn reindex_errors_instead_of_fabricating_nan_strings_for_non_numeric_columns() {
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), string_series(&["a", "b"]))
        .unwrap();
    df.add_column("label".to_string(), string_series(&["alpha", "beta"]))
        .unwrap();

    let indexed = df.set_index("id").unwrap();
    let new_index = vec!["a".to_string(), "missing".to_string()];

    // The old behaviour silently wrote the literal string "NaN" into every
    // missing cell regardless of column type; a non-numeric column has no
    // NA-capable representation here, so this must fail loudly instead.
    assert!(IndexAligner::reindex(&indexed, &new_index).is_err());
}

// ---------------------------------------------------------------------
// (8) Alignment padding NA-fills numeric columns instead of cyclically
// repeating real observations.
// ---------------------------------------------------------------------

#[test]
fn align_outer_nan_pads_numeric_columns_instead_of_repeating_rows() {
    let mut left = DataFrame::new();
    left.add_column(
        "v".to_string(),
        Series::new(vec![1.0f64, 2.0], None).unwrap(),
    )
    .unwrap();

    let mut right = DataFrame::new();
    right
        .add_column(
            "v".to_string(),
            Series::new(vec![10.0f64, 20.0, 30.0, 40.0], None).unwrap(),
        )
        .unwrap();

    let (aligned_left, aligned_right) =
        IndexAligner::align(&left, &right, AlignmentStrategy::Outer).unwrap();
    assert_eq!(aligned_left.row_count(), 4);
    assert_eq!(aligned_right.row_count(), 4);

    let left_values = aligned_left.get_column_numeric_values("v").unwrap();
    assert_eq!(&left_values[..2], &[1.0, 2.0]);
    // Previously the padded rows repeated real values cyclically
    // (`values[i % values.len()]` -> 1.0, 2.0 again), silently duplicating
    // observations that were never actually there.
    assert!(left_values[2].is_nan());
    assert!(left_values[3].is_nan());
}

// ---------------------------------------------------------------------
// (9) `drop_columns` preserves the original column order.
// ---------------------------------------------------------------------

#[test]
fn drop_columns_preserves_original_column_order() {
    let mut df = DataFrame::new();
    for name in ["a", "b", "c", "d", "e"] {
        df.add_column(
            name.to_string(),
            Series::new(vec![1i64, 2, 3], None).unwrap(),
        )
        .unwrap();
    }

    let dropped = df
        .drop_columns(&["b".to_string(), "d".to_string()])
        .unwrap();
    assert_eq!(
        dropped.column_names().to_vec(),
        vec!["a".to_string(), "c".to_string(), "e".to_string()]
    );
}

// ---------------------------------------------------------------------
// (10) Row-selecting `.iloc` operations preserve each column's concrete
// dtype instead of rebuilding every column as `Series<String>`.
// ---------------------------------------------------------------------

#[test]
fn iloc_row_selection_preserves_i64_dtype() {
    let mut df = DataFrame::new();
    df.add_column(
        "n".to_string(),
        Series::new(vec![10i64, 20, 30, 40, 50], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "label".to_string(),
        string_series(&["a", "b", "c", "d", "e"]),
    )
    .unwrap();

    let by_range = df.iloc().get_range(1..4).unwrap();
    assert_eq!(
        by_range.get_column::<i64>("n").unwrap().values(),
        &[20, 30, 40]
    );

    let by_positions = df.iloc().get_positions(&[0, 2, 4]).unwrap();
    assert_eq!(
        by_positions.get_column::<i64>("n").unwrap().values(),
        &[10, 30, 50]
    );

    let by_boolean = df
        .iloc()
        .get_boolean(&[true, false, true, false, true])
        .unwrap();
    assert_eq!(
        by_boolean.get_column::<i64>("n").unwrap().values(),
        &[10, 30, 50]
    );

    // `head`/`tail`/`sample` (trait methods) are built on the same
    // `.iloc` row-selection path and inherit the fix.
    let head = AdvancedIndexingExt::head(&df, 2).unwrap();
    assert_eq!(head.get_column::<i64>("n").unwrap().values(), &[10, 20]);
    let tail = AdvancedIndexingExt::tail(&df, 2).unwrap();
    assert_eq!(tail.get_column::<i64>("n").unwrap().values(), &[40, 50]);
}

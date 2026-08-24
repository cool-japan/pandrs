//! Regression tests for the optimized DataFrame row/join core (task split-core-A).
//!
//! Every test here pins down a behaviour that was previously wrong:
//!
//! * missing values were replaced by `0`/`0.0`/`""`/`false` on every row
//!   materialization (filter, head/tail, sample, append, join, melt), which
//!   silently corrupted aggregates;
//! * the resulting column order depended on `HashMap`/`HashSet` iteration order;
//! * row selection either cloned the full-length index onto a shorter frame or
//!   dropped it (and always dropped multi-indexes);
//! * joins dropped left rows whose key was NULL, matched `NaN` with `NaN`,
//!   never matched `-0.0` with `0.0`, and returned a different schema when
//!   nothing matched;
//! * `concat_rows` / `sample_rows` returned columns full of placeholder values.

use pandrs::optimized::split_dataframe::OptimizedDataFrame as SplitDataFrame;
use pandrs::{
    BooleanColumn, Column, ColumnType, DataFrameIndex, Float64Column, Index, Int64Column,
    MultiIndex, OptimizedDataFrame, StringColumn,
};

/// Tests report failures through a boxed error so that the large `pandrs::Error`
/// never has to travel in a `Result` (which clippy rightfully flags).
type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn int_column(values: Vec<i64>) -> Column {
    Column::Int64(Int64Column::new(values))
}

fn int_column_with_nulls(values: Vec<i64>, nulls: Vec<bool>) -> Column {
    Column::Int64(Int64Column::with_nulls(values, nulls))
}

fn string_column(values: &[&str]) -> Column {
    Column::String(StringColumn::new(
        values.iter().map(|v| v.to_string()).collect(),
    ))
}

fn int_values(df: &OptimizedDataFrame, name: &str) -> Vec<Option<i64>> {
    let view = df.column(name).expect("column must exist");
    let column = view.column().as_int64().expect("column must be Int64");
    (0..df.row_count())
        .map(|i| column.get(i).expect("row must be in range"))
        .collect()
}

fn string_values(df: &OptimizedDataFrame, name: &str) -> Vec<Option<String>> {
    let view = df.column(name).expect("column must exist");
    let column = view.column().as_string().expect("column must be String");
    (0..df.row_count())
        .map(|i| {
            column
                .get(i)
                .expect("row must be in range")
                .map(|value| value.to_string())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// NULL preservation on row materialization
// ---------------------------------------------------------------------------

#[test]
fn filter_preserves_missing_values_instead_of_substituting_zero() -> TestResult {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "value",
        int_column_with_nulls(vec![1, 0, 3, 4], vec![false, true, false, false]),
    )?;
    df.add_column(
        "keep",
        Column::Boolean(BooleanColumn::new(vec![true, true, true, false])),
    )?;

    let filtered = df.filter("keep")?;
    assert_eq!(filtered.row_count(), 3);

    // The missing value must still be missing after filtering.
    assert_eq!(int_values(&filtered, "value"), vec![Some(1), None, Some(3)]);

    // ... which is what makes the aggregate correct: (1 + 3) / 2 = 2.0.
    // The old NULL -> 0 substitution produced (1 + 0 + 3) / 3 = 1.333...
    assert!((filtered.mean("value")? - 2.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn head_tail_and_sample_preserve_missing_values() -> TestResult {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "value",
        int_column_with_nulls(vec![0, 2, 0, 4], vec![true, false, true, false]),
    )?;

    assert_eq!(int_values(&df.head(2)?, "value"), vec![None, Some(2)]);
    assert_eq!(int_values(&df.tail(2)?, "value"), vec![None, Some(4)]);
    assert_eq!(
        int_values(&df.sample_rows(&[3, 0])?, "value"),
        vec![Some(4), None]
    );

    Ok(())
}

#[test]
fn get_row_preserves_missing_values() -> TestResult {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "value",
        int_column_with_nulls(vec![1, 0], vec![false, true]),
    )?;

    let row = df.get_row(1)?;
    assert_eq!(row.row_count(), 1);
    assert_eq!(int_values(&row, "value"), vec![None]);

    Ok(())
}

#[test]
fn append_fills_absent_column_with_null_not_zero() -> TestResult {
    let mut left = OptimizedDataFrame::new();
    left.add_column("a", int_column(vec![1, 2]))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column("a", int_column(vec![3, 4]))?;
    right.add_column("b", int_column(vec![10, 20]))?;

    let appended = left.append(&right)?;
    assert_eq!(appended.row_count(), 4);
    assert_eq!(
        appended.column_names(),
        &["a".to_string(), "b".to_string()],
        "the union of both schemas must keep the left frame's order first"
    );

    assert_eq!(
        int_values(&appended, "b"),
        vec![None, None, Some(10), Some(20)],
        "rows coming from the frame without the column must be NULL, not 0"
    );

    // mean over the two present values: 15.0 (the 0-fill produced 7.5).
    assert!((appended.mean("b")? - 15.0).abs() < f64::EPSILON);

    Ok(())
}

#[test]
fn melt_keeps_missing_values_out_of_the_value_column() -> TestResult {
    let mut df = SplitDataFrame::new();
    df.add_column("id", string_column(&["x", "y"]))?;
    df.add_column("a", int_column_with_nulls(vec![1, 0], vec![false, true]))?;
    df.add_column("b", int_column(vec![3, 4]))?;

    let melted = df.melt(&["id"], Some(&["a", "b"]), None, None)?;
    assert_eq!(melted.row_count(), 4);

    let view = melted.column("value")?;
    let column = view.column().as_int64().expect("value must be Int64");
    let values: Vec<Option<i64>> = (0..melted.row_count())
        .map(|i| column.get(i).expect("row must be in range"))
        .collect();
    assert_eq!(values, vec![Some(1), None, Some(3), Some(4)]);

    Ok(())
}

// ---------------------------------------------------------------------------
// Deterministic schema
// ---------------------------------------------------------------------------

#[test]
fn row_selection_keeps_the_declared_column_order() -> TestResult {
    // Each iteration builds a fresh frame (and therefore a fresh HashMap with a
    // fresh hash seed): iterating the column index instead of the column order
    // produced a different schema on nearly every run.
    for _ in 0..20 {
        let mut df = OptimizedDataFrame::new();
        df.add_column("charlie", int_column(vec![3, 1, 2]))?;
        df.add_column("alpha", int_column(vec![30, 10, 20]))?;
        df.add_column("bravo", string_column(&["c", "a", "b"]))?;
        df.add_column(
            "delta",
            Column::Boolean(BooleanColumn::new(vec![true, false, true])),
        )?;

        let expected: Vec<String> = df.column_names().to_vec();

        assert_eq!(df.sort_by("charlie", true)?.column_names(), expected);
        assert_eq!(df.head(2)?.column_names(), expected);
        assert_eq!(df.tail(2)?.column_names(), expected);
        assert_eq!(df.sample_rows(&[2, 0])?.column_names(), expected);
    }

    Ok(())
}

#[test]
fn append_column_order_is_deterministic() -> TestResult {
    for _ in 0..20 {
        let mut left = OptimizedDataFrame::new();
        left.add_column("one", int_column(vec![1]))?;
        left.add_column("two", int_column(vec![2]))?;

        let mut right = OptimizedDataFrame::new();
        right.add_column("two", int_column(vec![20]))?;
        right.add_column("three", int_column(vec![30]))?;
        right.add_column("four", int_column(vec![40]))?;

        let appended = left.append(&right)?;
        assert_eq!(
            appended.column_names(),
            &[
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
                "four".to_string()
            ]
        );
    }

    Ok(())
}

#[test]
fn empty_row_selection_keeps_the_schema() -> TestResult {
    let mut df = SplitDataFrame::new();
    df.add_column("value", int_column(vec![1, 2, 3]))?;
    df.add_column("label", string_column(&["a", "b", "c"]))?;

    // Selecting no row at all used to return a frame without any column.
    let empty = df.select_rows_by_indices(&[])?;
    assert_eq!(empty.row_count(), 0);
    assert_eq!(empty.column_names(), df.column_names());
    assert_eq!(empty.column("value")?.column_type(), ColumnType::Int64);
    assert_eq!(empty.column("label")?.column_type(), ColumnType::String);

    let none_selected = df.select_by_mask(&[false, false, false])?;
    assert_eq!(none_selected.row_count(), 0);
    assert_eq!(none_selected.column_names(), df.column_names());

    Ok(())
}

#[test]
fn sorting_a_zero_row_frame_keeps_the_schema() -> TestResult {
    // `sort_by` routes through the internal row-selection entry point, which
    // used to short-circuit an empty selection into a frame without columns.
    let mut df = SplitDataFrame::new();
    df.add_column("value", int_column(vec![]))?;
    df.add_column("label", string_column(&[]))?;

    let sorted = df.sort_by("value", true)?;
    assert_eq!(sorted.row_count(), 0);
    assert_eq!(sorted.column_names(), df.column_names());
    assert_eq!(sorted.column("value")?.column_type(), ColumnType::Int64);
    assert_eq!(sorted.column("label")?.column_type(), ColumnType::String);

    Ok(())
}

#[test]
fn parallel_and_serial_row_materialization_agree() -> TestResult {
    // Large enough to cross the threshold above which columns are materialized
    // on the rayon pool: both branches must produce the same values, the same
    // column order and the same NULLs.
    let row_count = 3000_usize;
    let values: Vec<i64> = (0..row_count as i64).collect();
    let nulls: Vec<bool> = (0..row_count).map(|i| i % 7 == 0).collect();

    let mut df = OptimizedDataFrame::new();
    df.add_column("value", int_column_with_nulls(values, nulls))?;
    df.add_column(
        "label",
        string_column(
            &(0..row_count)
                .map(|i| if i % 2 == 0 { "even" } else { "odd" })
                .collect::<Vec<_>>(),
        ),
    )?;
    df.add_column(
        "keep",
        Column::Boolean(BooleanColumn::new(
            (0..row_count).map(|i| i % 3 == 0).collect(),
        )),
    )?;

    let parallel = df.head(row_count)?;
    assert_eq!(parallel.column_names(), df.column_names());
    let parallel_values = int_values(&parallel, "value");
    assert_eq!(parallel_values.len(), row_count);
    for (position, value) in parallel_values.iter().enumerate() {
        if position % 7 == 0 {
            assert_eq!(*value, None);
        } else {
            assert_eq!(*value, Some(position as i64));
        }
    }

    // The same rows selected in small (serial) chunks must match.
    let serial = df.head(10)?;
    assert_eq!(int_values(&serial, "value"), parallel_values[..10].to_vec());
    assert_eq!(
        string_values(&serial, "label"),
        string_values(&parallel, "label")[..10].to_vec()
    );

    let filtered = df.filter("keep")?;
    assert_eq!(filtered.row_count(), row_count.div_ceil(3));
    assert_eq!(filtered.column_names(), df.column_names());
    assert_eq!(int_values(&filtered, "value")[1], Some(3));

    Ok(())
}

// ---------------------------------------------------------------------------
// Index integrity
// ---------------------------------------------------------------------------

#[test]
fn row_selection_reselects_index_labels() -> TestResult {
    let mut df = OptimizedDataFrame::new();
    df.add_column("value", int_column(vec![10, 20, 30, 40]))?;
    df.set_index_from_simple_index(Index::new(vec![
        "r0".to_string(),
        "r1".to_string(),
        "r2".to_string(),
        "r3".to_string(),
    ])?)?;

    let head = df.head(2)?;
    match head.get_index() {
        Some(DataFrameIndex::Simple(index)) => {
            assert_eq!(index.len(), head.row_count());
            assert_eq!(index.get_value(0), Some(&"r0".to_string()));
            assert_eq!(index.get_value(1), Some(&"r1".to_string()));
        }
        other => panic!("expected a simple index of the selected rows, got {other:?}"),
    }

    let tail = df.tail(1)?;
    match tail.get_index() {
        Some(DataFrameIndex::Simple(index)) => {
            assert_eq!(index.len(), 1);
            assert_eq!(index.get_value(0), Some(&"r3".to_string()));
        }
        other => panic!("expected a simple index of the selected rows, got {other:?}"),
    }

    Ok(())
}

#[test]
fn row_selection_keeps_multi_index() -> TestResult {
    let tuples = vec![
        vec!["a".to_string(), "x".to_string()],
        vec!["a".to_string(), "y".to_string()],
        vec!["b".to_string(), "x".to_string()],
    ];
    let multi_index = MultiIndex::from_tuples(
        tuples,
        Some(vec![Some("outer".to_string()), Some("inner".to_string())]),
    )?;

    let mut df = OptimizedDataFrame::new();
    df.add_column("value", int_column(vec![1, 2, 3]))?;
    df.set_index_directly(DataFrameIndex::from_multi(multi_index))?;

    let head = df.head(2)?;
    match head.get_index() {
        Some(DataFrameIndex::Multi(index)) => {
            assert_eq!(index.len(), 2);
            assert_eq!(
                index.get_tuple(1),
                Some(vec!["a".to_string(), "y".to_string()])
            );
        }
        other => panic!("the multi-index must survive row selection, got {other:?}"),
    }

    Ok(())
}

#[test]
fn head_and_head_rows_agree() -> TestResult {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "value",
        int_column_with_nulls(vec![1, 0, 3], vec![false, true, false]),
    )?;
    df.set_index_from_simple_index(Index::new(vec![
        "r0".to_string(),
        "r1".to_string(),
        "r2".to_string(),
    ])?)?;

    for (via_data_ops, via_row_ops) in [
        (df.head(2)?, df.head_rows(2)?),
        (df.tail(2)?, df.tail_rows(2)?),
    ] {
        assert_eq!(via_data_ops.row_count(), via_row_ops.row_count());
        assert_eq!(via_data_ops.column_names(), via_row_ops.column_names());

        // Both copies must carry the index of the selected rows only - the
        // data_ops copy used to clone the full-length index.
        match (via_data_ops.get_index(), via_row_ops.get_index()) {
            (Some(DataFrameIndex::Simple(left)), Some(DataFrameIndex::Simple(right))) => {
                assert_eq!(left.len(), via_data_ops.row_count());
                assert_eq!(left.values(), right.values());
            }
            other => panic!("both copies must keep a simple index, got {other:?}"),
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Join semantics
// ---------------------------------------------------------------------------

#[test]
fn left_join_keeps_rows_whose_key_is_null() -> TestResult {
    let mut left = OptimizedDataFrame::new();
    left.add_column(
        "id",
        int_column_with_nulls(vec![1, 0, 3], vec![false, true, false]),
    )?;
    left.add_column("name", string_column(&["a", "b", "c"]))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column("id", int_column(vec![1, 3]))?;
    right.add_column("value", int_column(vec![100, 300]))?;

    let joined = left.left_join(&right, "id", "id")?;
    assert_eq!(
        joined.row_count(),
        3,
        "a left join must keep the row with the NULL key"
    );
    assert_eq!(
        string_values(&joined, "name"),
        vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string())
        ]
    );
    assert_eq!(int_values(&joined, "id"), vec![Some(1), None, Some(3)]);
    assert_eq!(
        int_values(&joined, "value"),
        vec![Some(100), None, Some(300)],
        "unmatched rows must be NULL on the right side, not 0"
    );

    // A NULL key still never matches anything.
    let inner = left.inner_join(&right, "id", "id")?;
    assert_eq!(inner.row_count(), 2);

    Ok(())
}

#[test]
fn float_join_keys_follow_ieee_semantics() -> TestResult {
    let mut left = OptimizedDataFrame::new();
    left.add_column(
        "key",
        Column::Float64(Float64Column::new(vec![1.0, f64::NAN, -0.0])),
    )?;
    left.add_column("name", string_column(&["one", "nan", "minus_zero"]))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column(
        "key",
        Column::Float64(Float64Column::new(vec![1.0, f64::NAN, 0.0])),
    )?;
    right.add_column("value", int_column(vec![10, 20, 30]))?;

    // NaN never matches (not even itself); -0.0 and 0.0 are the same key.
    let inner = left.inner_join(&right, "key", "key")?;
    assert_eq!(inner.row_count(), 2);
    assert_eq!(int_values(&inner, "value"), vec![Some(10), Some(30)]);

    // ... but the NaN row is still kept by a left join, with a NULL right side.
    let left_joined = left.left_join(&right, "key", "key")?;
    assert_eq!(left_joined.row_count(), 3);
    assert_eq!(
        int_values(&left_joined, "value"),
        vec![Some(10), None, Some(30)]
    );

    Ok(())
}

#[test]
fn empty_join_result_keeps_the_full_schema() -> TestResult {
    let mut left = OptimizedDataFrame::new();
    left.add_column("id", int_column(vec![1, 2]))?;
    left.add_column("name", string_column(&["a", "b"]))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column("id", int_column(vec![3, 4]))?;
    right.add_column("value", int_column(vec![30, 40]))?;

    let empty = left.inner_join(&right, "id", "id")?;
    assert_eq!(empty.row_count(), 0);
    assert_eq!(
        empty.column_names(),
        &["name".to_string(), "id".to_string(), "value".to_string()],
        "an empty join result must have the same schema as a non-empty one"
    );

    let mut matching = OptimizedDataFrame::new();
    matching.add_column("id", int_column(vec![1, 2]))?;
    matching.add_column("value", int_column(vec![10, 20]))?;

    let non_empty = left.inner_join(&matching, "id", "id")?;
    assert_eq!(non_empty.row_count(), 2);
    assert_eq!(
        non_empty.column_names(),
        empty.column_names(),
        "the schema of a join must not depend on the number of matches"
    );

    Ok(())
}

#[test]
fn join_row_order_is_left_major_even_when_the_right_side_is_larger() -> TestResult {
    // The right frame is larger, so the hash table is built on the left side and
    // the resulting pairs have to be re-ordered to stay left-major.
    let mut left = OptimizedDataFrame::new();
    left.add_column("id", int_column(vec![1, 2]))?;
    left.add_column("name", string_column(&["one", "two"]))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column("id", int_column(vec![2, 1, 2, 3, 1, 4]))?;
    right.add_column("value", int_column(vec![20, 10, 21, 30, 11, 40]))?;

    let inner = left.inner_join(&right, "id", "id")?;
    assert_eq!(
        int_values(&inner, "value"),
        vec![Some(10), Some(11), Some(20), Some(21)]
    );
    assert_eq!(
        int_values(&inner, "id"),
        vec![Some(1), Some(1), Some(2), Some(2)]
    );

    let outer = left.outer_join(&right, "id", "id")?;
    assert_eq!(
        int_values(&outer, "value"),
        vec![Some(10), Some(11), Some(20), Some(21), Some(30), Some(40)]
    );
    assert_eq!(
        string_values(&outer, "name"),
        vec![
            Some("one".to_string()),
            Some("one".to_string()),
            Some("two".to_string()),
            Some("two".to_string()),
            None,
            None
        ],
        "rows that exist only on the right side must be NULL on the left side"
    );

    Ok(())
}

#[test]
fn typed_join_keys_scale_and_stay_correct() -> TestResult {
    // Smoke test for the typed (allocation free) hash join path: no key is ever
    // rendered as a String, so correctness must hold for a large frame too.
    let row_count: i64 = 2000;

    let mut left = OptimizedDataFrame::new();
    left.add_column("id", int_column((0..row_count).collect()))?;
    left.add_column("left_value", int_column((0..row_count).collect()))?;

    let mut right = OptimizedDataFrame::new();
    right.add_column("id", int_column((0..row_count).step_by(2).collect()))?;
    right.add_column(
        "right_value",
        int_column((0..row_count).step_by(2).map(|id| id * 10).collect()),
    )?;

    let joined = left.inner_join(&right, "id", "id")?;
    assert_eq!(joined.row_count() as i64, row_count / 2);

    let ids = int_values(&joined, "id");
    let right_values = int_values(&joined, "right_value");
    for (position, id) in ids.iter().enumerate() {
        let id = id.expect("join key must be present");
        assert_eq!(id, position as i64 * 2);
        assert_eq!(right_values[position], Some(id * 10));
    }

    let left_joined = left.left_join(&right, "id", "id")?;
    assert_eq!(left_joined.row_count() as i64, row_count);
    let right_values = int_values(&left_joined, "right_value");
    for (position, value) in right_values.iter().enumerate() {
        if position % 2 == 0 {
            assert_eq!(*value, Some(position as i64 * 10));
        } else {
            assert_eq!(*value, None, "unmatched left rows must be NULL, not 0");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// concat_rows / sample_rows
// ---------------------------------------------------------------------------

#[test]
fn concat_rows_concatenates_the_actual_values() -> TestResult {
    let mut first = OptimizedDataFrame::new();
    first.add_column("value", int_column(vec![1, 2]))?;
    first.add_column("label", string_column(&["a", "b"]))?;

    let mut second = OptimizedDataFrame::new();
    second.add_column(
        "value",
        int_column_with_nulls(vec![3, 0], vec![false, true]),
    )?;
    second.add_column("label", string_column(&["c", "d"]))?;

    let concatenated = first.concat_rows(&second)?;
    assert_eq!(concatenated.row_count(), 4);
    assert_eq!(
        int_values(&concatenated, "value"),
        vec![Some(1), Some(2), Some(3), None]
    );
    assert_eq!(
        string_values(&concatenated, "label"),
        vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("c".to_string()),
            Some("d".to_string())
        ]
    );

    Ok(())
}

#[test]
fn sample_rows_returns_the_selected_rows() -> TestResult {
    let mut df = OptimizedDataFrame::new();
    df.add_column("value", int_column(vec![10, 20, 30, 40]))?;
    df.add_column("label", string_column(&["a", "b", "c", "d"]))?;

    let sampled = df.sample_rows(&[3, 1, 0])?;
    assert_eq!(sampled.row_count(), 3);
    assert_eq!(
        int_values(&sampled, "value"),
        vec![Some(40), Some(20), Some(10)]
    );
    assert_eq!(
        string_values(&sampled, "label"),
        vec![
            Some("d".to_string()),
            Some("b".to_string()),
            Some("a".to_string())
        ]
    );

    Ok(())
}

#[test]
fn sampling_with_replacement_does_not_fail_on_repeated_index_labels() -> TestResult {
    let mut df = SplitDataFrame::new();
    df.add_column("value", int_column(vec![1, 2, 3]))?;
    df.set_index_from_simple_index(Index::new(vec![
        "r0".to_string(),
        "r1".to_string(),
        "r2".to_string(),
    ])?)?;

    // Repeated rows cannot keep their (duplicated) labels, but the operation
    // must still succeed instead of failing on the duplicate-label check.
    let sampled = df.sample_rows(5, true, Some(42))?;
    assert_eq!(sampled.row_count(), 5);

    Ok(())
}

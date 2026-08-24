//! Regression tests for the split-core-B fixes:
//!
//! * `optimized/split_dataframe/sort.rs` — total-order comparison, NaN/NULL last
//! * `optimized/split_dataframe/group/` — typed group keys, NA key handling,
//!   deterministic + correctly aligned aggregation output
//! * `optimized/split_dataframe/aggregate.rs` — null-aware column reductions
//! * `optimized/lazy.rs` — typed (non-lexicographic) sort

use pandrs::optimized::split_dataframe::group::AggregateOp as GroupAggregateOp;
use pandrs::optimized::split_dataframe::OptimizedDataFrame as SplitDataFrame;
use pandrs::{Column, Float64Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

// ---------------------------------------------------------------- helpers ---

fn split_floats(df: &SplitDataFrame, name: &str) -> Vec<Option<f64>> {
    let view = df.column(name).expect("column exists");
    let col = view.as_float64().expect("float column");
    (0..df.row_count())
        .map(|i| col.get(i).expect("index in range"))
        .collect()
}

fn split_ints(df: &SplitDataFrame, name: &str) -> Vec<Option<i64>> {
    let view = df.column(name).expect("column exists");
    let col = view.as_int64().expect("int column");
    (0..df.row_count())
        .map(|i| col.get(i).expect("index in range"))
        .collect()
}

fn split_strings(df: &SplitDataFrame, name: &str) -> Vec<Option<String>> {
    let view = df.column(name).expect("column exists");
    let col = view.as_string().expect("string column");
    (0..df.row_count())
        .map(|i| {
            col.get(i)
                .expect("index in range")
                .map(|value| value.to_string())
        })
        .collect()
}

fn frame_ints(df: &OptimizedDataFrame, name: &str) -> Vec<Option<i64>> {
    let view = df.column(name).expect("column exists");
    let col = view.as_int64().expect("int column");
    (0..df.row_count())
        .map(|i| col.get(i).expect("index in range"))
        .collect()
}

// ------------------------------------------------------------------- sort ---

/// `partial_cmp(..).unwrap_or(Equal)` is not transitive: with more than one NaN
/// the sort left ordinary values unsorted (observed: 5, NaN, 1, NaN, 2, 3, 4).
#[test]
fn every_nan_sorts_last_and_the_remaining_floats_are_fully_ordered() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "v",
        Column::Float64(Float64Column::new(vec![
            5.0,
            f64::NAN,
            1.0,
            f64::NAN,
            3.0,
            2.0,
            4.0,
        ])),
    )
    .expect("add column");

    let sorted = df.sort_by("v", true).expect("sort ascending");
    let values = split_floats(&sorted, "v");

    let present: Vec<f64> = values
        .iter()
        .filter_map(|value| *value)
        .filter(|value| !value.is_nan())
        .collect();
    assert_eq!(
        present,
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        "non-NaN values must be fully sorted"
    );

    let nan_positions: Vec<usize> = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.map(|v| v.is_nan()).unwrap_or(true))
        .map(|(idx, _)| idx)
        .collect();
    assert_eq!(nan_positions, vec![5, 6], "every NaN must sort last");
}

#[test]
fn nan_still_sorts_last_when_sorting_descending() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "v",
        Column::Float64(Float64Column::new(vec![1.0, f64::NAN, 3.0, f64::NAN, 2.0])),
    )
    .expect("add column");

    let sorted = df.sort_by("v", false).expect("sort descending");
    let values = split_floats(&sorted, "v");

    let present: Vec<f64> = values
        .iter()
        .filter_map(|value| *value)
        .filter(|value| !value.is_nan())
        .collect();
    assert_eq!(present, vec![3.0, 2.0, 1.0]);
    assert!(
        values[3].map(|v| v.is_nan()).unwrap_or(true)
            && values[4].map(|v| v.is_nan()).unwrap_or(true),
        "NaN stays last in descending order too (pandas na_position=\"last\")"
    );
}

#[test]
fn sorting_an_already_sorted_float_column_is_a_fixed_point() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "v",
        Column::Float64(Float64Column::new(vec![
            f64::NAN,
            2.0,
            f64::NAN,
            -1.0,
            0.0,
            f64::NAN,
            7.5,
        ])),
    )
    .expect("add column");

    let once = df.sort_by("v", true).expect("sort once");
    let twice = once.sort_by("v", true).expect("sort twice");

    let first: Vec<String> = split_floats(&once, "v")
        .iter()
        .map(|value| format!("{:?}", value))
        .collect();
    let second: Vec<String> = split_floats(&twice, "v")
        .iter()
        .map(|value| format!("{:?}", value))
        .collect();
    assert_eq!(
        first, second,
        "a total order must be idempotent under re-sorting"
    );
}

#[test]
fn null_floats_sort_last_in_both_directions() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "v",
        Column::Float64(Float64Column::with_nulls(
            vec![3.0, 0.0, 1.0, 2.0],
            vec![false, true, false, false],
        )),
    )
    .expect("add column");

    let ascending = split_floats(&df.sort_by("v", true).expect("asc"), "v");
    assert_eq!(ascending[0], Some(1.0));
    assert_eq!(ascending[1], Some(2.0));
    assert_eq!(ascending[2], Some(3.0));
    assert_eq!(ascending[3], None, "NULL must sort last ascending");

    let descending = split_floats(&df.sort_by("v", false).expect("desc"), "v");
    assert_eq!(descending[0], Some(3.0));
    assert_eq!(descending[3], None, "NULL must sort last descending too");
}

#[test]
fn multi_column_sort_compares_int64_numerically_not_lexicographically() {
    let mut df = SplitDataFrame::new();
    df.add_column("k", Column::Int64(Int64Column::new(vec![10, 2, 100, 1, 2])))
        .expect("add k");
    df.add_column(
        "tie",
        Column::String(StringColumn::new(vec![
            "a".to_string(),
            "z".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
        ])),
    )
    .expect("add tie");

    let sorted = df
        .sort_by_columns(&["k", "tie"], None)
        .expect("multi-column sort");

    assert_eq!(
        split_ints(&sorted, "k"),
        vec![Some(1), Some(2), Some(2), Some(10), Some(100)]
    );
    assert_eq!(
        split_strings(&sorted, "tie"),
        vec![
            Some("a".to_string()),
            Some("b".to_string()),
            Some("z".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
        ],
        "the second key must break ties within the first key"
    );
}

#[test]
fn multi_column_sort_places_nan_and_null_last_within_each_key() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "grp",
        Column::String(StringColumn::new(vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
        ])),
    )
    .expect("add grp");
    df.add_column(
        "v",
        Column::Float64(Float64Column::new(vec![f64::NAN, 2.0, 1.0])),
    )
    .expect("add v");

    let sorted = df
        .sort_by_columns(&["grp", "v"], Some(&[true, false]))
        .expect("sort");
    let values = split_floats(&sorted, "v");
    assert_eq!(values[0], Some(2.0));
    assert_eq!(values[1], Some(1.0));
    assert!(
        values[2].map(|v| v.is_nan()).unwrap_or(true),
        "NaN is missing-like and stays last even when the key is descending"
    );
}

// ------------------------------------------------------------------ groups ---

/// `key_parts.join("_")` mapped `("a_b", "c")` and `("a", "b_c")` onto the same
/// string and silently merged two different groups.
#[test]
fn composite_group_keys_containing_underscores_do_not_collide() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "a",
        Column::String(StringColumn::new(vec!["a_b".to_string(), "a".to_string()])),
    )
    .expect("add a");
    df.add_column(
        "b",
        Column::String(StringColumn::new(vec!["c".to_string(), "b_c".to_string()])),
    )
    .expect("add b");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2])))
        .expect("add v");

    let grouped = df.group_by(["a", "b"]).expect("group_by");
    assert_eq!(
        grouped.groups.len(),
        2,
        "(a_b, c) and (a, b_c) are distinct groups"
    );

    let par_grouped = df.par_groupby(&["a", "b"]).expect("par_groupby");
    assert_eq!(
        par_grouped.len(),
        2,
        "par_groupby must not merge distinct composite keys"
    );
}

#[test]
fn par_groupby_single_column_keys_stay_the_bare_value() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "k",
        Column::String(StringColumn::new(vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
        ])),
    )
    .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2, 3])))
        .expect("add v");

    let grouped = df.par_groupby(&["k"]).expect("par_groupby");
    assert_eq!(grouped.len(), 2);
    assert_eq!(
        grouped.get("A").map(|group| group.row_count()),
        Some(2),
        "single-column groups are still addressable by their raw value"
    );
}

/// A missing key used to become the literal string "NULL"/"NA", which merged
/// genuine `"NULL"` cells into the missing-value group.
#[test]
fn null_group_keys_are_dropped_by_default_and_kept_as_na_on_request() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "k",
        Column::String(StringColumn::with_nulls(
            vec!["a".to_string(), String::new(), "NULL".to_string()],
            vec![false, true, false],
        )),
    )
    .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2, 3])))
        .expect("add v");

    // pandas default: dropna = true
    let dropped = df.group_by(["k"]).expect("group_by");
    assert_eq!(
        dropped.groups.len(),
        2,
        "the row with a missing key is excluded by default"
    );

    let result = dropped
        .aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Sum,
            "sum".to_string(),
        )])
        .expect("aggregate");
    let keys = split_strings(&result, "k");
    assert!(
        keys.iter().all(|key| key.is_some()),
        "no fabricated NULL key row"
    );
    assert!(
        keys.contains(&Some("NULL".to_string())),
        "a real \"NULL\" string value must survive as its own group"
    );

    // Opt in to keeping the NA group.
    let kept = df
        .group_by_with_config(["k"], false, false)
        .expect("group_by_with_config");
    assert_eq!(kept.groups.len(), 3, "the NA key forms its own group");

    let kept_result = kept
        .aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Sum,
            "sum".to_string(),
        )])
        .expect("aggregate");
    let kept_keys = split_strings(&kept_result, "k");
    assert_eq!(
        kept_keys.iter().filter(|key| key.is_none()).count(),
        1,
        "the NA group key is emitted as a real NULL, not the text \"NULL\""
    );
    assert!(
        kept_keys.contains(&Some("NULL".to_string())),
        "the genuine \"NULL\" string is still a separate group"
    );
}

/// Group keys and aggregate values were pushed under two separate mutexes, so
/// rayon interleaving paired one group's key with another group's value.
#[test]
fn parallel_group_aggregation_keeps_keys_aligned_with_values() {
    const GROUPS: i64 = 24;

    let mut keys = Vec::new();
    let mut values = Vec::new();
    for group in 0..GROUPS {
        for repeat in 0..3 {
            keys.push(format!("g{:02}", group));
            values.push(group * 100 + repeat);
        }
    }

    let mut df = SplitDataFrame::new();
    df.add_column("k", Column::String(StringColumn::new(keys)))
        .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(values)))
        .expect("add v");

    let grouped = df.group_by(["k"]).expect("group_by");
    let result = grouped
        .par_aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Sum,
            "sum".to_string(),
        )])
        .expect("par_aggregate");

    assert_eq!(result.row_count(), GROUPS as usize);

    let result_keys = split_strings(&result, "k");
    let result_sums = split_floats(&result, "sum");
    for (key, sum) in result_keys.iter().zip(result_sums.iter()) {
        let key = key.as_ref().expect("group key present");
        let group: i64 = key
            .trim_start_matches('g')
            .parse()
            .expect("group key is g<NN>");
        let expected = (group * 100 * 3 + 0 + 1 + 2) as f64;
        assert_eq!(
            *sum,
            Some(expected),
            "group {} must carry its own aggregate",
            key
        );
    }
}

#[test]
fn group_aggregation_row_order_is_deterministic() {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for group in 0..16 {
        keys.push(format!("k{:02}", group));
        values.push(group as i64);
    }

    let mut df = SplitDataFrame::new();
    df.add_column("k", Column::String(StringColumn::new(keys)))
        .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(values)))
        .expect("add v");

    let mut seen: Option<Vec<Option<String>>> = None;
    for _ in 0..8 {
        let grouped = df.group_by(["k"]).expect("group_by");
        let result = grouped
            .par_aggregate(vec![(
                "v".to_string(),
                GroupAggregateOp::Sum,
                "sum".to_string(),
            )])
            .expect("par_aggregate");
        let order = split_strings(&result, "k");
        match &seen {
            None => seen = Some(order),
            Some(previous) => assert_eq!(
                previous, &order,
                "group order must not depend on hash-map iteration"
            ),
        }
    }
}

#[test]
fn group_mean_skips_nulls_and_all_null_groups_stay_null() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "k",
        Column::String(StringColumn::new(vec![
            "a".to_string(),
            "a".to_string(),
            "a".to_string(),
            "b".to_string(),
            "b".to_string(),
        ])),
    )
    .expect("add k");
    df.add_column(
        "v",
        Column::Float64(Float64Column::with_nulls(
            vec![1.0, 0.0, 3.0, 0.0, 0.0],
            vec![false, true, false, true, true],
        )),
    )
    .expect("add v");

    let grouped = df.group_by(["k"]).expect("group_by");
    let result = grouped
        .aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Mean,
            "mean".to_string(),
        )])
        .expect("aggregate");

    let keys = split_strings(&result, "k");
    let means = split_floats(&result, "mean");
    let mut by_key: Vec<(String, Option<f64>)> = keys
        .iter()
        .zip(means.iter())
        .map(|(key, mean)| (key.clone().unwrap_or_default(), *mean))
        .collect();
    by_key.sort_by(|left, right| left.0.cmp(&right.0));

    assert_eq!(
        by_key[0],
        ("a".to_string(), Some(2.0)),
        "the NULL row must be skipped, not counted as 0"
    );
    assert_eq!(
        by_key[1],
        ("b".to_string(), None),
        "the mean of an all-NULL group is missing, not 0.0"
    );
}

/// `group_by` with several key columns turns on the MultiIndex result path by
/// default; the rewritten builder has to produce the index and the aggregate
/// columns in the same group order.
#[test]
fn multi_key_group_aggregate_builds_a_multi_index_result() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "a",
        Column::String(StringColumn::new(vec![
            "y".to_string(),
            "x".to_string(),
            "x".to_string(),
        ])),
    )
    .expect("add a");
    df.add_column(
        "b",
        Column::String(StringColumn::new(vec![
            "p".to_string(),
            "q".to_string(),
            "p".to_string(),
        ])),
    )
    .expect("add b");
    df.add_column("v", Column::Int64(Int64Column::new(vec![4, 2, 1])))
        .expect("add v");

    let grouped = df.group_by(["a", "b"]).expect("group_by");
    let result = grouped
        .aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Sum,
            "sum".to_string(),
        )])
        .expect("aggregate with multi-index");

    assert_eq!(result.row_count(), 3);
    assert!(result.contains_column("sum"));
    assert_eq!(
        split_floats(&result, "sum"),
        vec![Some(1.0), Some(2.0), Some(4.0)],
        "rows follow the sorted (a, b) key order: (x,p), (x,q), (y,p)"
    );

    let par_result = grouped
        .par_aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Sum,
            "sum".to_string(),
        )])
        .expect("par_aggregate with multi-index");
    assert_eq!(
        split_floats(&par_result, "sum"),
        vec![Some(1.0), Some(2.0), Some(4.0)]
    );
}

/// `par_transform` transformed the first group twice (once to sample the
/// schema, once inside the parallel pass) and emitted its rows twice.
#[test]
fn par_transform_processes_each_group_exactly_once() {
    const GROUPS: usize = 12;
    const ROWS_PER_GROUP: usize = 2;

    let mut keys = Vec::new();
    let mut values = Vec::new();
    for group in 0..GROUPS {
        for repeat in 0..ROWS_PER_GROUP {
            keys.push(format!("g{:02}", group));
            values.push((group * 10 + repeat) as i64);
        }
    }

    let mut df = SplitDataFrame::new();
    df.add_column("k", Column::String(StringColumn::new(keys)))
        .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(values)))
        .expect("add v");

    let grouped = df.group_by(["k"]).expect("group_by");
    let transformed = grouped
        .par_transform(|group| Ok(group.clone()))
        .expect("par_transform");

    assert_eq!(
        transformed.row_count(),
        GROUPS * ROWS_PER_GROUP,
        "an identity transform must not duplicate the first group"
    );

    let mut seen = split_ints(&transformed, "v");
    seen.sort();
    let mut expected: Vec<Option<i64>> = (0..GROUPS)
        .flat_map(|group| (0..ROWS_PER_GROUP).map(move |r| Some((group * 10 + r) as i64)))
        .collect();
    expected.sort();
    assert_eq!(seen, expected);
}

#[test]
fn group_count_counts_observed_values_only() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "k",
        Column::String(StringColumn::new(vec!["a".to_string(), "a".to_string()])),
    )
    .expect("add k");
    df.add_column(
        "v",
        Column::Int64(Int64Column::with_nulls(vec![5, 0], vec![false, true])),
    )
    .expect("add v");

    let grouped = df.group_by(["k"]).expect("group_by");
    let result = grouped
        .aggregate(vec![(
            "v".to_string(),
            GroupAggregateOp::Count,
            "count".to_string(),
        )])
        .expect("aggregate");

    assert_eq!(split_floats(&result, "count"), vec![Some(1.0)]);
}

// -------------------------------------------------------------- aggregate ---

#[test]
fn column_reductions_skip_nulls() {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "f",
        Column::Float64(Float64Column::with_nulls(
            vec![1.0, 0.0, 3.0],
            vec![false, true, false],
        )),
    )
    .expect("add f");
    df.add_column(
        "i",
        Column::Int64(Int64Column::with_nulls(
            vec![10, 0, 30],
            vec![false, true, false],
        )),
    )
    .expect("add i");

    assert_eq!(df.mean("f").expect("mean f"), 2.0);
    assert_eq!(df.sum("f").expect("sum f"), 4.0);
    assert_eq!(df.min("f").expect("min f"), 1.0);
    assert_eq!(df.max("f").expect("max f"), 3.0);
    assert_eq!(df.count("f").expect("count f"), 2);

    assert_eq!(df.mean("i").expect("mean i"), 20.0);
    assert_eq!(df.sum("i").expect("sum i"), 40.0);
    assert_eq!(df.min("i").expect("min i"), 10.0);
    assert_eq!(df.max("i").expect("max i"), 30.0);
    assert_eq!(df.count("i").expect("count i"), 2);
}

#[test]
fn integer_sums_stay_exact_beyond_the_f64_mantissa() {
    let big = 1_i64 << 53;
    let mut df = SplitDataFrame::new();
    df.add_column("i", Column::Int64(Int64Column::new(vec![big, 1, 1])))
        .expect("add i");

    // (2^53 + 2) is representable; naive f64 accumulation of 2^53 + 1 + 1 is not.
    assert_eq!(df.sum("i").expect("sum"), (big + 2) as f64);
}

// ------------------------------------------------------------------- lazy ---

/// The lazy Sort stringified its keys, so an Int64 column came back as
/// 1, 10, 100, 2 instead of 1, 2, 10, 100.
#[test]
fn lazy_sort_orders_int64_columns_numerically() {
    let mut df = OptimizedDataFrame::new();
    df.add_column("v", Column::Int64(Int64Column::new(vec![2, 100, 1, 10])))
        .expect("add v");

    let ascending = LazyFrame::new(df.clone())
        .sort("v", true)
        .execute()
        .expect("lazy sort ascending");
    assert_eq!(
        frame_ints(&ascending, "v"),
        vec![Some(1), Some(2), Some(10), Some(100)]
    );

    let descending = LazyFrame::new(df)
        .sort("v", false)
        .execute()
        .expect("lazy sort descending");
    assert_eq!(
        frame_ints(&descending, "v"),
        vec![Some(100), Some(10), Some(2), Some(1)]
    );
}

/// The lazy Aggregate operation used to build its own stringified group keys,
/// turning a missing key into the literal text "NULL" (which then merged with a
/// genuine `"NULL"` cell) and emitting its key columns in hash-map order.
#[test]
fn lazy_aggregate_does_not_fabricate_a_null_group_key() {
    let mut df = OptimizedDataFrame::new();
    df.add_column(
        "k",
        Column::String(StringColumn::with_nulls(
            vec!["a".to_string(), String::new(), "NULL".to_string()],
            vec![false, true, false],
        )),
    )
    .expect("add k");
    df.add_column("v", Column::Int64(Int64Column::new(vec![1, 2, 4])))
        .expect("add v");

    let result = LazyFrame::new(df)
        .aggregate(
            ["k".to_string()],
            vec![("v".to_string(), pandrs::AggregateOp::Sum, "sum".to_string())],
        )
        .execute()
        .expect("lazy aggregate");

    assert_eq!(
        result.row_count(),
        2,
        "the row with a missing group key is excluded, not turned into a \"NULL\" group"
    );

    let keys = result.column("k").expect("k column");
    let keys = keys.as_string().expect("string column");
    let sums = result.column("sum").expect("sum column");
    let sums = sums.as_float64().expect("float column");

    let mut pairs: Vec<(String, f64)> = (0..result.row_count())
        .map(|i| {
            (
                keys.get(i)
                    .expect("in range")
                    .expect("key present")
                    .to_string(),
                sums.get(i).expect("in range").expect("sum present"),
            )
        })
        .collect();
    pairs.sort_by(|left, right| left.0.cmp(&right.0));

    assert_eq!(
        pairs,
        vec![("NULL".to_string(), 4.0), ("a".to_string(), 1.0)],
        "the genuine \"NULL\" string must stay its own group"
    );
}

#[test]
fn lazy_sort_moves_the_other_columns_with_the_sort_key() {
    let mut df = OptimizedDataFrame::new();
    df.add_column("v", Column::Int64(Int64Column::new(vec![3, 1, 2])))
        .expect("add v");
    df.add_column("other", Column::Int64(Int64Column::new(vec![30, 10, 20])))
        .expect("add other");

    let sorted = LazyFrame::new(df)
        .sort("v", true)
        .execute()
        .expect("lazy sort");

    assert_eq!(frame_ints(&sorted, "v"), vec![Some(1), Some(2), Some(3)]);
    assert_eq!(
        frame_ints(&sorted, "other"),
        vec![Some(10), Some(20), Some(30)],
        "payload columns must follow the permutation"
    );
}

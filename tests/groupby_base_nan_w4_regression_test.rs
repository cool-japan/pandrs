//! Wave-4 regression: base `DataFrame` groupby aggregations must EXCLUDE
//! missing (`NaN`) cells, matching the typed/split `OptimizedDataFrame::
//! group_by` path's `skipna=True` null semantics.
//!
//! Backlog item A3: base `DataFrame::groupby_single().mean()` used to parse a
//! missing cell's literal "NaN" text back to `f64::NAN`, push it into the
//! group, and average it in -- so mean over `[10, NaN, 20]` PROPAGATED the
//! missing value and returned `NaN`, while the typed `group_by().mean()` and
//! `Float64Column::mean` correctly excluded it and returned `15.0`. These
//! tests pin the two paths to the same answer and prove the all-missing group
//! is handled consistently (a missing marker on both sides, never a fabricated
//! `0.0`).

use pandrs::optimized::split_dataframe::OptimizedDataFrame as SplitDataFrame;
use pandrs::prelude::*;

// --------------------------------------------------------------------------
// Frame builders: the SAME `[g, v]` data expressed for each code path.
// --------------------------------------------------------------------------

/// Base `DataFrame` with a string key column `g` and an `f64` value column `v`
/// (a `NaN` in `v` renders as the literal "NaN" -- the base DataFrame's
/// missing marker).
fn base_frame(keys: &[&str], vals: &[f64]) -> DataFrame {
    let mut df = DataFrame::new();
    let key_series = Series::new(
        keys.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        Some("g".to_string()),
    )
    .expect("key series");
    df.add_column("g".to_string(), key_series).expect("add g");
    let val_series = Series::new(vals.to_vec(), Some("v".to_string())).expect("val series");
    df.add_column("v".to_string(), val_series).expect("add v");
    df
}

/// Typed split `OptimizedDataFrame` (the one whose `group_by` carries the
/// null-aware aggregation path) with the same `[g, v]` data.
fn typed_frame(keys: &[&str], vals: &[f64]) -> SplitDataFrame {
    let mut df = SplitDataFrame::new();
    df.add_column(
        "g",
        Column::String(StringColumn::new(
            keys.iter().map(|s| s.to_string()).collect(),
        )),
    )
    .expect("add g");
    df.add_column("v", Column::Float64(Float64Column::new(vals.to_vec())))
        .expect("add v");
    df
}

// --------------------------------------------------------------------------
// Single-group result readers.
// --------------------------------------------------------------------------

/// Read the single aggregated cell of a base-path result as `f64` (parses the
/// stringified aggregate; "NaN" round-trips to `f64::NAN`).
fn base_single(out: &DataFrame, col: &str) -> f64 {
    let vals = out.get_column_string_values(col).expect("result column");
    assert_eq!(vals.len(), 1, "expected exactly one group row");
    vals[0].parse::<f64>().expect("aggregate parses as f64")
}

/// Read the single aggregated cell of a typed-path result NULL-aware. The typed
/// `build_result` stores `0.0` under a null bit for an undefined reduction, so
/// a plain numeric read would masquerade as `0.0`; go through the Float64 column
/// so an all-missing group is seen as the `None` it really is.
fn typed_single(out: &SplitDataFrame, col: &str) -> Option<f64> {
    assert_eq!(out.row_count(), 1, "expected exactly one group row");
    let view = out.column(col).expect("result column");
    let float_col = view.column().as_float64().expect("result must be Float64");
    float_col.get(0).expect("cell access")
}

// --------------------------------------------------------------------------
// Core finding: mean over [10, NaN, 20] == 15.0 on BOTH paths.
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_mean_both_exclude_nan_15() {
    let base_out = base_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .groupby_single("g")
        .expect("base groupby")
        .mean("v")
        .expect("base mean");
    let base_mean = base_single(&base_out, "v_mean");
    assert_eq!(
        base_mean, 15.0,
        "base groupby mean must EXCLUDE the NaN (drop it from sum AND divisor) -> 15.0, \
         not propagate it as NaN"
    );

    let typed_out = typed_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .group_by(["g"])
        .expect("typed group_by")
        .mean("v")
        .expect("typed mean");
    let typed_mean = typed_single(&typed_out, "v_mean");
    assert_eq!(
        typed_mean,
        Some(15.0),
        "typed group_by mean excludes the NaN -> 15.0"
    );

    // Cross-path parity: identical numeric answer.
    assert_eq!(
        base_mean,
        typed_mean.expect("typed mean present"),
        "base and typed mean must agree on [10, NaN, 20]"
    );
}

// --------------------------------------------------------------------------
// Sum is the other reduction that used to propagate NaN: 10 + NaN + 20.
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_sum_both_exclude_nan_30() {
    let base_out = base_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .groupby_single("g")
        .expect("base groupby")
        .sum("v")
        .expect("base sum");
    let base_sum = base_single(&base_out, "v_sum");
    assert_eq!(
        base_sum, 30.0,
        "base sum must exclude the NaN -> 30.0 (was NaN before the fix)"
    );

    let typed_out = typed_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .group_by(["g"])
        .expect("typed group_by")
        .sum("v")
        .expect("typed sum");
    let typed_sum = typed_single(&typed_out, "v_sum");
    assert_eq!(typed_sum, Some(30.0));
    assert_eq!(base_sum, typed_sum.expect("typed sum present"));
}

// --------------------------------------------------------------------------
// All-missing group: neither path fabricates a real number.
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_all_nan_group_are_consistently_missing() {
    let base_out = base_frame(&["a", "a"], &[f64::NAN, f64::NAN])
        .groupby_single("g")
        .expect("base groupby")
        .mean("v")
        .expect("base mean");
    let base_mean = base_single(&base_out, "v_mean");
    assert!(
        base_mean.is_nan(),
        "base all-NA mean must be the NaN missing marker, got {base_mean}"
    );

    let typed_out = typed_frame(&["a", "a"], &[f64::NAN, f64::NAN])
        .group_by(["g"])
        .expect("typed group_by")
        .mean("v")
        .expect("typed mean");
    let typed_mean = typed_single(&typed_out, "v_mean");
    assert_eq!(
        typed_mean, None,
        "typed all-NA mean must be a real NULL cell, not 0.0"
    );

    // Consistency: base -> NaN marker, typed -> NULL cell. Both mean "no
    // value"; CRUCIALLY neither is a fabricated finite value such as 0.0.
    assert!(base_mean.is_nan() && typed_mean.is_none());
    assert_ne!(
        base_mean, 0.0,
        "base must not substitute 0.0 for an all-NA mean"
    );
}

// --------------------------------------------------------------------------
// min/max on an all-NaN group: NaN, not the +-INF the old fold seed leaked
// (same bug class as backlog A4).
// --------------------------------------------------------------------------

#[test]
fn base_min_max_all_nan_is_nan_not_infinity() {
    let frame = base_frame(&["a", "a"], &[f64::NAN, f64::NAN]);

    let min_out = frame
        .groupby_single("g")
        .expect("base groupby")
        .min("v")
        .expect("base min");
    let base_min = base_single(&min_out, "v_min");
    assert!(
        base_min.is_nan() && !base_min.is_infinite(),
        "all-NaN min must be NaN, not the +-INF fold seed; got {base_min}"
    );

    let max_out = frame
        .groupby_single("g")
        .expect("base groupby")
        .max("v")
        .expect("base max");
    let base_max = base_single(&max_out, "v_max");
    assert!(
        base_max.is_nan() && !base_max.is_infinite(),
        "all-NaN max must be NaN, not the +-INF fold seed; got {base_max}"
    );

    // Typed path agrees: a NULL cell for an all-missing min.
    let typed_out = typed_frame(&["a", "a"], &[f64::NAN, f64::NAN])
        .group_by(["g"])
        .expect("typed group_by")
        .min("v")
        .expect("typed min");
    assert_eq!(typed_single(&typed_out, "v_min"), None);
}

#[test]
fn base_min_max_mixed_group_still_skips_nan() {
    // A group with SOME valid values already skipped NaN by accident (Rust's
    // `f64::min`/`max` ignore a NaN argument); pin that it still does.
    let frame = base_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0]);
    let min_out = frame
        .groupby_single("g")
        .expect("base groupby")
        .min("v")
        .expect("base min");
    assert_eq!(base_single(&min_out, "v_min"), 10.0);
    let max_out = frame
        .groupby_single("g")
        .expect("base groupby")
        .max("v")
        .expect("base max");
    assert_eq!(base_single(&max_out, "v_max"), 20.0);
}

// --------------------------------------------------------------------------
// Custom aggregation: skips NaN on a mixed group, and still RUNS on the empty
// slice for an all-NA group (matches the typed aggregate_custom path, which
// calls custom_fn(&[]) rather than short-circuiting).
// --------------------------------------------------------------------------

#[test]
fn base_custom_aggregation_skips_nan_and_runs_on_empty_all_nan_group() {
    // Mixed group: the custom fn sees only the 2 non-NaN observations.
    let mixed_out = base_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .groupby_single("g")
        .expect("base groupby")
        .apply("v", "cnt", |vals| vals.len() as f64)
        .expect("base custom apply");
    assert_eq!(
        base_single(&mixed_out, "cnt"),
        2.0,
        "custom fn must receive NaN-free values (2 of 3)"
    );

    // All-NA group: the custom fn still runs, on the empty slice -> 0, NOT a
    // short-circuited NaN.
    let empty_out = base_frame(&["a", "a"], &[f64::NAN, f64::NAN])
        .groupby_single("g")
        .expect("base groupby")
        .apply("v", "cnt", |vals| vals.len() as f64)
        .expect("base custom apply");
    assert_eq!(
        base_single(&empty_out, "cnt"),
        0.0,
        "custom fn must run on the empty slice for an all-NA group, not short-circuit to NaN"
    );
}

// --------------------------------------------------------------------------
// std/var: the ddof=1 divisor must be the NON-NaN count, not the raw length
// (mean/sum parity alone would not catch a wrong divisor).
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_std_divisor_is_the_non_nan_count() {
    // [10, NaN, 20]: variance = ((10-15)^2 + (20-15)^2) / (2-1) = 50, so
    // std = sqrt(50). A divisor of len=3 (or a mean of NaN) would fail this.
    let expected = 50.0_f64.sqrt();

    let base_out = base_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .groupby_single("g")
        .expect("base groupby")
        .std("v")
        .expect("base std");
    let base_std = base_single(&base_out, "v_std");
    assert!(
        (base_std - expected).abs() < 1e-9,
        "base std must divide by the non-NaN count (n=2): expected {expected}, got {base_std}"
    );

    let typed_out = typed_frame(&["a", "a", "a"], &[10.0, f64::NAN, 20.0])
        .group_by(["g"])
        .expect("typed group_by")
        .std("v")
        .expect("typed std");
    let typed_std = typed_single(&typed_out, "v_std").expect("typed std present");
    assert!((typed_std - expected).abs() < 1e-9);
    assert!(
        (base_std - typed_std).abs() < 1e-9,
        "base and typed std must agree on [10, NaN, 20]"
    );
}

// --------------------------------------------------------------------------
// first: the first OBSERVED value, skipping a leading NaN.
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_first_returns_first_observed_value() {
    // [NaN, 10, 20]: `first` must skip the leading NaN and return 10.
    let base_out = base_frame(&["a", "a", "a"], &[f64::NAN, 10.0, 20.0])
        .groupby_single("g")
        .expect("base groupby")
        .agg(vec![NamedAgg::new(
            "v".to_string(),
            AggFunc::First,
            "v_first".to_string(),
        )])
        .expect("base first");
    assert_eq!(
        base_single(&base_out, "v_first"),
        10.0,
        "base first must return the first observed value (10), not the leading NaN"
    );

    let typed_out = typed_frame(&["a", "a", "a"], &[f64::NAN, 10.0, 20.0])
        .group_by(["g"])
        .expect("typed group_by")
        .first("v")
        .expect("typed first");
    assert_eq!(typed_single(&typed_out, "v_first"), Some(10.0));
}

// --------------------------------------------------------------------------
// Multi-group parity: real grouping, one group carrying a NaN.
// --------------------------------------------------------------------------

#[test]
fn base_and_typed_multi_group_mean_parity_with_nan() {
    // Group "a" = [10, NaN, 20] -> 15; group "b" = [4, 6] -> 5.
    let keys = ["a", "a", "a", "b", "b"];
    let vals = [10.0, f64::NAN, 20.0, 4.0, 6.0];

    // Base path: HashMap group order is arbitrary, so read keys and sort.
    let base_out = base_frame(&keys, &vals)
        .groupby_single("g")
        .expect("base groupby")
        .mean("v")
        .expect("base mean");
    let base_keys = base_out.get_column_string_values("g").expect("g");
    let base_vals = base_out.get_column_string_values("v_mean").expect("v_mean");
    let mut base_pairs: Vec<(String, f64)> = base_keys
        .into_iter()
        .zip(base_vals)
        .map(|(k, v)| (k, v.parse::<f64>().expect("f64")))
        .collect();
    base_pairs.sort_by(|left, right| left.0.cmp(&right.0));

    // Typed path: read keys and values NULL-aware (no all-NA group here).
    let typed_out = typed_frame(&keys, &vals)
        .group_by(["g"])
        .expect("typed group_by")
        .mean("v")
        .expect("typed mean");
    let key_view = typed_out.column("g").expect("g");
    let key_col = key_view.column().as_string().expect("string key");
    let val_view = typed_out.column("v_mean").expect("v_mean");
    let val_col = val_view.column().as_float64().expect("float val");
    let mut typed_pairs: Vec<(String, f64)> = (0..typed_out.row_count())
        .map(|i| {
            let k = key_col
                .get(i)
                .expect("key cell")
                .expect("key not null")
                .to_string();
            let v = val_col.get(i).expect("val cell").expect("no all-NA group");
            (k, v)
        })
        .collect();
    typed_pairs.sort_by(|left, right| left.0.cmp(&right.0));

    assert_eq!(
        base_pairs, typed_pairs,
        "base and typed multi-group means must agree cell-for-cell"
    );
    assert_eq!(
        base_pairs,
        vec![("a".to_string(), 15.0), ("b".to_string(), 5.0)]
    );
}

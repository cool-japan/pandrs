//! Wave-4 regression: hierarchical-groupby numeric aggregations must EXCLUDE
//! missing (`NaN`) cells (pandas `skipna=True`), matching the base
//! `DataFrameGroupBy` path (fixed in Wave 4) and the typed/split
//! `OptimizedDataFrame::group_by` path.
//!
//! `hierarchical_groupby.rs` carried the same bug the base path did: it parsed
//! each cell with `.parse::<f64>().ok()` and reduced over the result WITHOUT an
//! `is_nan()` filter, so a missing cell's literal "NaN" text round-tripped to
//! `f64::NAN` and PROPAGATED through mean/sum/std/etc. It also SILENTLY DROPPED
//! genuinely non-numeric cells (`.ok()` swallowed the parse error), where the
//! fixed base path errors loudly. These tests pin the corrected behavior on all
//! three affected entry points -- `agg_hierarchical`, `cross_level_agg`, and
//! `nested_transform` -- and prove cross-path parity with the base groupby.

use pandrs::dataframe::groupby::AggFunc;
use pandrs::dataframe::hierarchical_groupby::{HierarchicalAgg, HierarchicalGroupByExt};
use pandrs::prelude::*;

// --------------------------------------------------------------------------
// Frame builders.
// --------------------------------------------------------------------------

/// One-level hierarchical frame: a string key column `grp` and an `f64` value
/// column `v` (an `f64::NAN` renders as the literal "NaN" -- the base
/// DataFrame's missing marker, exactly as the base regression test builds it).
fn frame_f64(keys: &[&str], vals: &[f64]) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "grp".to_string(),
        Series::new(
            keys.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some("grp".to_string()),
        )
        .expect("key series"),
    )
    .expect("add grp");
    df.add_column(
        "v".to_string(),
        Series::new(vals.to_vec(), Some("v".to_string())).expect("val series"),
    )
    .expect("add v");
    df
}

/// One-level hierarchical frame whose value column holds RAW strings, so a
/// genuinely non-numeric cell (e.g. "abc") is present verbatim.
fn frame_str(keys: &[&str], vals: &[&str]) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "grp".to_string(),
        Series::new(
            keys.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some("grp".to_string()),
        )
        .expect("key series"),
    )
    .expect("add grp");
    df.add_column(
        "v".to_string(),
        Series::new(
            vals.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some("v".to_string()),
        )
        .expect("val series"),
    )
    .expect("add v");
    df
}

/// Two-level hierarchical frame: `grp`/`sub` string keys + `f64` value `v`.
fn frame_f64_2level(grp: &[&str], sub: &[&str], vals: &[f64]) -> DataFrame {
    let mut df = frame_f64(grp, vals);
    df.add_column(
        "sub".to_string(),
        Series::new(
            sub.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some("sub".to_string()),
        )
        .expect("sub series"),
    )
    .expect("add sub");
    df
}

// --------------------------------------------------------------------------
// Result readers.
// --------------------------------------------------------------------------

/// Read the single-row aggregated cell of a hierarchical result as `f64`
/// ("NaN" round-trips to `f64::NAN`).
fn single(out: &DataFrame, col: &str) -> f64 {
    let vals = out.get_column_string_values(col).expect("result column");
    assert_eq!(vals.len(), 1, "expected exactly one group row");
    vals[0].parse::<f64>().expect("aggregate parses as f64")
}

/// Read the single-row aggregated cell of a base-groupby result as `f64`.
fn base_single(out: &DataFrame, col: &str) -> f64 {
    let vals = out.get_column_string_values(col).expect("result column");
    assert_eq!(vals.len(), 1, "expected exactly one group row");
    vals[0].parse::<f64>().expect("aggregate parses as f64")
}

/// Build a one-row `agg_hierarchical` result for `func` at level 0 and return
/// the aggregated cell.
fn hier_agg_single(df: &DataFrame, func: AggFunc) -> f64 {
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string()])
        .expect("hierarchical groupby");
    let agg = HierarchicalAgg::new("v".to_string()).add_level_agg(0, func, "out".to_string());
    let out = hgb.agg_hierarchical(vec![agg]).expect("agg_hierarchical");
    single(&out, "out")
}

// --------------------------------------------------------------------------
// Core finding: mean/sum/std over [10, NaN, 20] EXCLUDE the NaN, and agree
// with the base groupby path cell-for-cell.
// --------------------------------------------------------------------------

#[test]
fn hierarchical_mean_sum_std_exclude_nan_and_match_base() {
    let df = frame_f64(&["a", "a", "a"], &[10.0, f64::NAN, 20.0]);

    // Mean: the NaN is dropped from BOTH the running sum AND the divisor.
    let mean = hier_agg_single(&df, AggFunc::Mean);
    assert_eq!(mean, 15.0, "hierarchical mean must exclude the NaN -> 15.0");

    // Sum: 10 + 20, the NaN excluded (was NaN before the fix).
    let sum = hier_agg_single(&df, AggFunc::Sum);
    assert_eq!(sum, 30.0, "hierarchical sum must exclude the NaN -> 30.0");

    // Std: ddof=1 divisor must be the NON-NaN count (n=2), so variance =
    // ((10-15)^2 + (20-15)^2) / (2-1) = 50 and std = sqrt(50). A divisor of
    // len=3 (or a NaN mean) would fail this.
    let expected_std = 50.0_f64.sqrt();
    let std = hier_agg_single(&df, AggFunc::Std);
    assert!(
        (std - expected_std).abs() < 1e-9,
        "hierarchical std must divide by the non-NaN count (n=2): expected {expected_std}, got {std}"
    );

    // Cross-path parity: the base DataFrame groupby (fixed in Wave 4) and the
    // typed path both return 15.0 / 30.0 / sqrt(50) on the SAME [10, NaN, 20].
    let base_mean = base_single(
        &df.groupby_single("grp")
            .expect("base groupby")
            .mean("v")
            .expect("base mean"),
        "v_mean",
    );
    let base_sum = base_single(
        &df.groupby_single("grp")
            .expect("base groupby")
            .sum("v")
            .expect("base sum"),
        "v_sum",
    );
    let base_std = base_single(
        &df.groupby_single("grp")
            .expect("base groupby")
            .std("v")
            .expect("base std"),
        "v_std",
    );
    assert_eq!(mean, base_mean, "hierarchical mean == base mean");
    assert_eq!(sum, base_sum, "hierarchical sum == base sum");
    assert!(
        (std - base_std).abs() < 1e-9,
        "hierarchical std == base std"
    );
}

// --------------------------------------------------------------------------
// Count is ROW-based (base parity), NOT the typed path's null-aware count.
// --------------------------------------------------------------------------

#[test]
fn hierarchical_count_is_row_count_matching_base() {
    // [10, NaN, 20]: hierarchical Count returns the ROW count (3), before any
    // numeric parse -- matching the base path. This DELIBERATELY differs from
    // the typed path's null-aware `count_valid` (which would drop the NaN -> 2)
    // for the same reason the base documents: a string-materialized column has
    // no null mask to reproduce that here. Pinned so a later "make Count
    // skipna" edit cannot silently flip it.
    let df = frame_f64(&["a", "a", "a"], &[10.0, f64::NAN, 20.0]);
    let hier_count = hier_agg_single(&df, AggFunc::Count);
    assert_eq!(hier_count, 3.0, "hierarchical Count = row count (3)");

    let base_out = df
        .groupby_single("grp")
        .expect("base groupby")
        .agg(vec![NamedAgg::new(
            "v".to_string(),
            AggFunc::Count,
            "v_count".to_string(),
        )])
        .expect("base count");
    assert_eq!(
        hier_count,
        base_single(&base_out, "v_count"),
        "hierarchical Count == base Count (both row count)"
    );
}

// --------------------------------------------------------------------------
// All-NA group: the missing marker, never a fabricated number.
// --------------------------------------------------------------------------

#[test]
fn hierarchical_all_nan_group_is_the_missing_marker_not_zero() {
    let df = frame_f64(&["a", "a"], &[f64::NAN, f64::NAN]);

    // mean/std/min/max of an all-missing group -> the NaN missing marker.
    for func in [AggFunc::Mean, AggFunc::Std, AggFunc::Min, AggFunc::Max] {
        let got = hier_agg_single(&df, func);
        assert!(
            got.is_nan() && !got.is_infinite(),
            "all-NA {func:?} must be the NaN marker (not 0.0, not +-INF), got {got}"
        );
        assert_ne!(got, 0.0, "all-NA {func:?} must not be a fabricated 0.0");
    }

    // Sum of an all-missing group is the additive identity 0.0, matching the
    // base path's per-function convention (`sum` of zero observations).
    let sum = hier_agg_single(&df, AggFunc::Sum);
    assert_eq!(
        sum, 0.0,
        "all-NA sum is 0.0 (additive identity), matching the base path"
    );
}

// --------------------------------------------------------------------------
// Genuinely non-numeric cell: LOUD error, matching the base path (NOT a silent
// drop under the guise of skipna). "abc" is not the NaN missing marker.
// --------------------------------------------------------------------------

#[test]
fn hierarchical_non_numeric_cell_errors_like_base() {
    let df = frame_str(&["a", "a", "a"], &["10", "abc", "20"]);

    // Hierarchical path errors (does not silently drop "abc" and average 15).
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string()])
        .expect("hierarchical groupby");
    let agg =
        HierarchicalAgg::new("v".to_string()).add_level_agg(0, AggFunc::Mean, "out".to_string());
    let hier_res = hgb.agg_hierarchical(vec![agg]);
    assert!(
        hier_res.is_err(),
        "a genuinely non-numeric cell must fail loudly, not be silently dropped"
    );
    assert!(
        matches!(hier_res, Err(pandrs::error::Error::InvalidValue(_))),
        "non-numeric aggregation must be an InvalidValue error"
    );

    // Base path errors on the same data -> same contract.
    let base_res = df.groupby_single("grp").expect("base groupby").mean("v");
    assert!(
        base_res.is_err(),
        "base path errors on non-numeric data; hierarchical must match"
    );
}

// --------------------------------------------------------------------------
// cross_level_agg: skipna across source groups.
// --------------------------------------------------------------------------

#[test]
fn cross_level_agg_excludes_nan() {
    // grp=a / sub=x over [10, NaN, 20]. Aggregate the sub level (1) up to the
    // grp level (0).
    let df = frame_f64_2level(&["a", "a", "a"], &["x", "x", "x"], &[10.0, f64::NAN, 20.0]);
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string(), "sub".to_string()])
        .expect("hierarchical groupby");

    let mean_out = hgb
        .cross_level_agg("v", AggFunc::Mean, 1, 0)
        .expect("cross_level mean");
    assert_eq!(
        single(&mean_out, "mean_v_from_level_1"),
        15.0,
        "cross_level mean must exclude the NaN -> 15.0"
    );

    let sum_out = hgb
        .cross_level_agg("v", AggFunc::Sum, 1, 0)
        .expect("cross_level sum");
    assert_eq!(
        single(&sum_out, "sum_v_from_level_1"),
        30.0,
        "cross_level sum must exclude the NaN -> 30.0"
    );
}

#[test]
fn cross_level_agg_all_missing_source_is_missing_marker_not_zero() {
    // A source group that EXISTS but whose every observation is missing: the
    // base all-NA convention applies -- mean -> NaN marker, sum -> 0.0. (This
    // is the branch distinct from a structurally-absent source group, which
    // stays NaN; that structurally-absent branch is preserved unchanged and
    // not exercised here.)
    let df = frame_f64_2level(&["a", "a"], &["x", "x"], &[f64::NAN, f64::NAN]);
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string(), "sub".to_string()])
        .expect("hierarchical groupby");

    let mean_out = hgb
        .cross_level_agg("v", AggFunc::Mean, 1, 0)
        .expect("cross_level mean");
    let mean = single(&mean_out, "mean_v_from_level_1");
    assert!(
        mean.is_nan(),
        "all-missing source mean must be the NaN marker, got {mean}"
    );

    let sum_out = hgb
        .cross_level_agg("v", AggFunc::Sum, 1, 0)
        .expect("cross_level sum");
    assert_eq!(
        single(&sum_out, "sum_v_from_level_1"),
        0.0,
        "all-missing source sum is 0.0 (additive identity), matching the base path"
    );
}

#[test]
fn cross_level_agg_non_numeric_cell_errors() {
    let df = {
        let mut d = frame_str(&["a", "a", "a"], &["10", "abc", "20"]);
        d.add_column(
            "sub".to_string(),
            Series::new(
                vec!["x".to_string(), "x".to_string(), "x".to_string()],
                Some("sub".to_string()),
            )
            .expect("sub series"),
        )
        .expect("add sub");
        d
    };
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string(), "sub".to_string()])
        .expect("hierarchical groupby");
    let res = hgb.cross_level_agg("v", AggFunc::Mean, 1, 0);
    assert!(
        res.is_err(),
        "cross_level_agg must fail loudly on a genuinely non-numeric cell"
    );
}

// --------------------------------------------------------------------------
// nested_transform: skipna INPUT to the transform + NA-preserving output.
// --------------------------------------------------------------------------

#[test]
fn nested_transform_excludes_nan_from_input_and_preserves_na_output() {
    // Mean-centering: transform_fn subtracts the group mean. If the NaN were
    // fed in, the group mean would be NaN and EVERY output would be NaN. With
    // skipna the mean is 15, so the two observed rows become -5 and +5, and the
    // missing row's output stays the "NaN" marker.
    let df = frame_f64(&["a", "a", "a"], &[10.0, f64::NAN, 20.0]);
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string()])
        .expect("hierarchical groupby");

    let out = hgb
        .nested_transform(
            "v",
            |vals| {
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                vals.iter().map(|x| x - m).collect()
            },
            0,
        )
        .expect("nested_transform");

    let col = out
        .get_column_string_values("v_transformed")
        .expect("v_transformed column");
    assert_eq!(col.len(), 3, "one output row per input row");

    let row0 = col[0].parse::<f64>().expect("row0 f64");
    let row2 = col[2].parse::<f64>().expect("row2 f64");
    assert!(
        (row0 - (-5.0)).abs() < 1e-9,
        "row0 = 10 - mean(10,20)=15 -> -5 (NaN excluded from the mean), got {row0}"
    );
    assert!(
        (row2 - 5.0).abs() < 1e-9,
        "row2 = 20 - 15 -> +5, got {row2}"
    );

    // NA preservation: the missing row's output cell is the "NaN" marker,
    // NOT "0" and NOT "".
    assert_eq!(
        col[1], "NaN",
        "the missing row's transformed output must render as the NaN marker, got {:?}",
        col[1]
    );
}

#[test]
fn nested_transform_non_numeric_cell_errors() {
    let df = frame_str(&["a", "a", "a"], &["10", "abc", "20"]);
    let hgb = df
        .hierarchical_groupby(vec!["grp".to_string()])
        .expect("hierarchical groupby");
    let res = hgb.nested_transform("v", |vals| vals.to_vec(), 0);
    assert!(
        res.is_err(),
        "nested_transform must fail loudly on a genuinely non-numeric cell"
    );
}

//! Regression tests for the `groupby-window` fixes:
//! `src/dataframe/groupby.rs`, `src/dataframe/groupby_window.rs`,
//! `src/dataframe/window.rs`, `src/dataframe/enhanced_window.rs`,
//! `src/dataframe/jit_window.rs`.
//!
//! Each test is labeled with the item number from the task list it covers.

use std::time::{Duration, Instant};

use pandrs::dataframe::{
    AggFunc, DataFrameWindowExt, EnhancedDataFrameWindowExt, GroupByExt, GroupWiseWindowExt,
    JitDataFrameWindowExt, JitWindowContext, NamedAgg,
};
use pandrs::{DataFrame, Series};

fn series_f64(values: &[f64]) -> Series<f64> {
    Series::new(values.to_vec(), None).unwrap()
}

fn series_str(values: &[&str]) -> Series<String> {
    Series::new(values.iter().map(|s| s.to_string()).collect(), None).unwrap()
}

// ---------------------------------------------------------------------
// (1)+(2) groupby.rs F1/F2: hoisted column materialization.
//
// Before the fix, `DataFrameGroupBy::new` re-fetched each grouping
// column's *entire* string representation on every row of its own
// construction loop (O(rows^2)), and `calculate_aggregation` re-fetched
// the aggregated column on every group (O(groups*rows)). At 10k rows
// this made construction+aggregation visibly slow (measured
// O(n^2.06-2.51) scaling: ~14.3s at 16k rows). This asserts *correctness*
// of the grouped aggregation at 10k rows (cross-checked against an
// independently computed expectation) and, as a non-flaky backstop
// against a catastrophic regression, a generous completion bound far
// above what the fixed O(rows) implementation needs.
// ---------------------------------------------------------------------

#[test]
fn groupby_10k_rows_is_correct_and_completes_quickly() {
    const ROWS: usize = 10_000;
    const GROUPS: i64 = 100;

    let groups: Vec<String> = (0..ROWS)
        .map(|i| format!("g{}", i as i64 % GROUPS))
        .collect();
    let values: Vec<f64> = (0..ROWS).map(|i| i as f64).collect();

    let mut expected_sum = vec![0.0f64; GROUPS as usize];
    let mut expected_count = vec![0u64; GROUPS as usize];
    for i in 0..ROWS {
        let g = (i as i64 % GROUPS) as usize;
        expected_sum[g] += values[i];
        expected_count[g] += 1;
    }

    let mut df = DataFrame::new();
    df.add_column(
        "group".to_string(),
        series_str(&groups.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
    )
    .unwrap();
    df.add_column("value".to_string(), series_f64(&values))
        .unwrap();

    let start = Instant::now();
    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    let result = grouped
        .agg(vec![
            NamedAgg::new("value".to_string(), AggFunc::Sum, "value_sum".to_string()),
            NamedAgg::new(
                "value".to_string(),
                AggFunc::Count,
                "value_count".to_string(),
            ),
        ])
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(result.row_count(), GROUPS as usize);

    let group_names = result.get_column_string_values("group").unwrap();
    let sums = result.get_column_numeric_values("value_sum").unwrap();
    let counts = result.get_column_numeric_values("value_count").unwrap();

    for i in 0..result.row_count() {
        let g: usize = group_names[i]
            .strip_prefix('g')
            .and_then(|s| s.parse().ok())
            .expect("group name of the form gN");
        assert!(
            (sums[i] - expected_sum[g]).abs() < 1e-6,
            "group {} sum: got {}, expected {}",
            group_names[i],
            sums[i],
            expected_sum[g]
        );
        assert_eq!(
            counts[i] as u64, expected_count[g],
            "group {} count",
            group_names[i]
        );
    }

    // Generous, non-flaky backstop: the fixed implementation finishes in
    // low tens of milliseconds; this only guards against a catastrophic
    // (hang-scale) reintroduction of the quadratic column re-extraction.
    assert!(
        elapsed < Duration::from_secs(20),
        "groupby+agg over {} rows took {:?} -- possible reintroduction of \
         the O(n^2) per-row/per-group column re-extraction",
        ROWS,
        elapsed
    );
}

// ---------------------------------------------------------------------
// (3) groupby.rs transform(): realign to original row order.
// ---------------------------------------------------------------------

#[test]
fn transform_realigns_to_original_row_order() {
    // Interleaved groups with means chosen so group-order concatenation
    // (the old, broken behavior) is numerically distinguishable from
    // input-row-order realignment (the fix) at every position.
    let mut df = DataFrame::new();
    df.add_column("group".to_string(), series_str(&["A", "B", "A", "B"]))
        .unwrap();
    df.add_column("value".to_string(), series_f64(&[10.0, 100.0, 30.0, 300.0]))
        .unwrap();

    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    let transformed = grouped
        .transform(|group_df| {
            let values = group_df.get_column_numeric_values("value")?;
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let demeaned: Vec<f64> = values.iter().map(|v| v - mean).collect();
            let mut out = DataFrame::new();
            out.add_column("value_demeaned".to_string(), series_f64(&demeaned))?;
            Ok(out)
        })
        .unwrap();

    assert_eq!(transformed.row_count(), 4);
    let demeaned = transformed
        .get_column_numeric_values("value_demeaned")
        .unwrap();
    // group A: {10, 30}, mean 20 -> {-10, +10}; group B: {100, 300}, mean
    // 200 -> {-100, +100}. Row order is A, B, A, B.
    let expected = [-10.0, -100.0, 10.0, 100.0];
    for i in 0..4 {
        assert!(
            (demeaned[i] - expected[i]).abs() < 1e-9,
            "row {}: got {}, expected {} (realignment to original row order)",
            i,
            demeaned[i],
            expected[i]
        );
    }
}

#[test]
fn transform_errors_when_group_func_changes_row_count() {
    let mut df = DataFrame::new();
    df.add_column("group".to_string(), series_str(&["A", "A", "B"]))
        .unwrap();
    df.add_column("value".to_string(), series_f64(&[1.0, 2.0, 3.0]))
        .unwrap();

    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    // Group "A" has 2 rows; returning a single-row DataFrame for it
    // violates transform's one-row-per-input-row contract and must error
    // rather than silently mis-realigning.
    let result = grouped.transform(|group_df| {
        let values = group_df.get_column_numeric_values("value")?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let mut out = DataFrame::new();
        out.add_column("value_mean".to_string(), series_f64(&[mean]))?;
        Ok(out)
    });
    assert!(result.is_err());
}

// ---------------------------------------------------------------------
// (4) groupby.rs calculate_aggregation: no silent drop, per-function
// empty-group semantics, Count/Nunique work without a numeric parse.
// ---------------------------------------------------------------------

#[test]
fn agg_errors_on_non_numeric_cell_instead_of_silently_dropping() {
    let mut df = DataFrame::new();
    df.add_column("group".to_string(), series_str(&["x", "x", "x"]))
        .unwrap();
    // A genuinely non-numeric cell ("oops") mixed into a column being
    // Sum-aggregated. Previously `.parse().ok()` silently dropped it from
    // the group (skewing the sum to 1.0 + 3.0 = 4.0 instead of erroring).
    df.add_column("value".to_string(), series_str(&["1.0", "oops", "3.0"]))
        .unwrap();

    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    let result = grouped.agg(vec![NamedAgg::new(
        "value".to_string(),
        AggFunc::Sum,
        "value_sum".to_string(),
    )]);
    assert!(
        result.is_err(),
        "a non-numeric cell in a numeric aggregation must error, not be silently dropped"
    );
}

#[test]
fn agg_std_and_var_on_single_element_group_is_nan_not_zero() {
    let mut df = DataFrame::new();
    df.add_column("group".to_string(), series_str(&["x", "y", "y"]))
        .unwrap();
    df.add_column("value".to_string(), series_f64(&[5.0, 1.0, 3.0]))
        .unwrap();

    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    let result = grouped
        .agg(vec![
            NamedAgg::new("value".to_string(), AggFunc::Std, "value_std".to_string()),
            NamedAgg::new("value".to_string(), AggFunc::Var, "value_var".to_string()),
        ])
        .unwrap();

    let names = result.get_column_string_values("group").unwrap();
    let stds = result.get_column_numeric_values("value_std").unwrap();
    let vars = result.get_column_numeric_values("value_var").unwrap();
    let x_idx = names.iter().position(|n| n == "x").unwrap();
    let y_idx = names.iter().position(|n| n == "y").unwrap();

    // pandas: std/var with ddof=1 on a single observation is NaN, not 0.0.
    assert!(
        stds[x_idx].is_nan(),
        "single-element group std must be NaN, got {}",
        stds[x_idx]
    );
    assert!(
        vars[x_idx].is_nan(),
        "single-element group var must be NaN, got {}",
        vars[x_idx]
    );
    // Sanity: the two-element group still computes a real value.
    assert!((stds[y_idx] - std::f64::consts::SQRT_2).abs() < 1e-9);
    assert!((vars[y_idx] - 2.0).abs() < 1e-9);
}

#[test]
fn agg_count_and_nunique_work_on_non_numeric_columns() {
    let mut df = DataFrame::new();
    df.add_column("group".to_string(), series_str(&["x", "x", "x", "x"]))
        .unwrap();
    // Purely non-numeric text -- previously Count/Nunique routed every
    // column through an f64 parse first, so this group's Count/Nunique
    // silently came out as 0.0 (every cell failed to parse and was
    // dropped, then the empty-group short-circuit returned 0.0).
    df.add_column(
        "category".to_string(),
        series_str(&["red", "blue", "red", "green"]),
    )
    .unwrap();

    let grouped = GroupByExt::groupby(&df, &["group"]).unwrap();
    let result = grouped
        .agg(vec![
            NamedAgg::new(
                "category".to_string(),
                AggFunc::Count,
                "category_count".to_string(),
            ),
            NamedAgg::new(
                "category".to_string(),
                AggFunc::Nunique,
                "category_nunique".to_string(),
            ),
        ])
        .unwrap();

    assert_eq!(result.row_count(), 1);
    let counts = result.get_column_numeric_values("category_count").unwrap();
    let nuniques = result
        .get_column_numeric_values("category_nunique")
        .unwrap();
    assert_eq!(counts[0], 4.0, "Count must count every row, numeric or not");
    assert_eq!(
        nuniques[0], 3.0,
        "Nunique must count distinct raw values (red/blue/green), numeric or not"
    );
}

// ---------------------------------------------------------------------
// (5) groupby_window.rs: real per-group partitioning for
// Rolling/Expanding/EWM -- windows/EWM state must never cross group
// boundaries.
// ---------------------------------------------------------------------

fn two_group_df() -> DataFrame {
    // Group "A" is a small, slowly increasing series; group "B" is two
    // orders of magnitude larger, so any cross-group contamination is
    // numerically obvious.
    let mut df = DataFrame::new();
    df.add_column(
        "grp".to_string(),
        series_str(&["A", "A", "A", "A", "B", "B", "B", "B"]),
    )
    .unwrap();
    df.add_column(
        "val".to_string(),
        series_f64(&[1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0]),
    )
    .unwrap();
    df
}

#[test]
fn group_wise_rolling_means_differ_across_groups_and_never_cross_boundaries() {
    let df = two_group_df();
    let config = df
        .rolling_by_group(vec!["grp".to_string()], 2)
        .min_periods(2);
    let ops = GroupWiseWindowExt::apply_rolling_by_group(&df, &config).unwrap();
    let result = ops.mean().unwrap();

    let means = result
        .get_column_numeric_values("val_mean_groupwise")
        .unwrap();
    // Row 4 is group B's *first* row. A naive whole-column rolling mean
    // (the old, ungrouped behavior) would average val[3]=4.0 (group A's
    // last value) with val[4]=100.0, giving 52.0. With real per-group
    // partitioning, group B has only 1 observation at that point, which is
    // below min_periods=2, so the result must be NaN.
    assert!(
        means[4].is_nan(),
        "group B's first row must be NaN (insufficient periods within its \
         own group), got {} -- window crossed the group boundary",
        means[4]
    );
    // Row 5 (group B's second row) must be the mean of *only* group B's
    // first two values (100, 200) = 150, not contaminated by group A.
    assert!(
        (means[5] - 150.0).abs() < 1e-9,
        "group B's second row: got {}, expected 150.0 (mean of 100,200 only)",
        means[5]
    );
    // Group A's own internal rolling mean is still computed normally.
    assert!(means[0].is_nan());
    assert!((means[1] - 1.5).abs() < 1e-9);
    assert!((means[3] - 3.5).abs() < 1e-9);
}

#[test]
fn group_wise_expanding_resets_accumulation_per_group() {
    let df = two_group_df();
    let config = df.expanding_by_group(vec!["grp".to_string()], 1);
    let ops = GroupWiseWindowExt::apply_expanding_by_group(&df, &config).unwrap();
    let result = ops.sum().unwrap();

    let sums = result.get_column_numeric_values("val_sum").unwrap();
    // Group A: cumulative sums 1, 3, 6, 10.
    assert_eq!(sums[0], 1.0);
    assert_eq!(sums[1], 3.0);
    assert_eq!(sums[2], 6.0);
    assert_eq!(sums[3], 10.0);
    // Group B must restart from its own first value, not continue
    // accumulating from group A's total (10.0 + 100.0 = 110.0 would prove
    // the bug).
    assert_eq!(
        sums[4], 100.0,
        "group B's expanding sum must restart at its own first value"
    );
    assert_eq!(sums[5], 300.0);
    assert_eq!(sums[7], 1000.0);
}

#[test]
fn group_wise_ewm_resets_recursive_state_per_group() {
    let df = two_group_df();
    let config = df.ewm_by_group(vec!["grp".to_string()]).alpha(0.5).unwrap();
    let ops = GroupWiseWindowExt::apply_ewm_by_group(&df, &config).unwrap();
    let result = ops.mean().unwrap();

    let means = result.get_column_numeric_values("val_mean").unwrap();
    // EWM seeds its recursive state with the *first* observation it sees.
    // If group B's EWM state carried over from group A, row 4 (group B's
    // first row) would not equal group B's own raw first value (100.0).
    assert!(
        (means[4] - 100.0).abs() < 1e-9,
        "group B's first EWM value must equal its own first observation \
         (100.0), got {} -- EWM state crossed the group boundary",
        means[4]
    );
    assert!(
        (means[0] - 1.0).abs() < 1e-9,
        "group A's first EWM value must equal 1.0"
    );
}

#[test]
fn group_wise_rolling_apply_with_custom_function_never_crosses_boundaries() {
    // `Rolling::apply` returns one `Option<f64>` per row (`None` below
    // min_periods), which `GroupWiseRollingOps::apply` renders as `NaN`
    // before scattering -- exercise that specific conversion path (the
    // named aggregation methods like `.mean()` never call `.apply()`) and
    // confirm it preserves group-boundary partitioning rather than
    // silently shortening/misaligning the result.
    let df = two_group_df();
    let config = df
        .rolling_by_group(vec!["grp".to_string()], 2)
        .min_periods(2);
    let ops = GroupWiseWindowExt::apply_rolling_by_group(&df, &config).unwrap();
    let result = ops.apply(|window| window.iter().sum::<f64>()).unwrap();

    let sums = result.get_column_numeric_values("val_custom").unwrap();
    // Row 4 is group B's first row: only 1 observation is available within
    // group B at that point, below min_periods=2, so it must be NaN --
    // not window.iter().sum() over [4.0, 100.0] (which would prove the
    // window crossed from group A into group B).
    assert!(
        sums[4].is_nan(),
        "group B's first row must be NaN (insufficient periods within its \
         own group), got {} -- custom apply() crossed the group boundary",
        sums[4]
    );
    // Row 5 (group B's second row) must be the sum of only group B's first
    // two values (100+200=300), not contaminated by group A.
    assert!(
        (sums[5] - 300.0).abs() < 1e-9,
        "group B's second row: got {}, expected 300.0 (sum of 100,200 only)",
        sums[5]
    );
    // Group A's own custom apply is still computed normally.
    assert!(sums[0].is_nan());
    assert!((sums[1] - 3.0).abs() < 1e-9); // 1+2
    assert!((sums[3] - 7.0).abs() < 1e-9); // 3+4
}

// ---------------------------------------------------------------------
// (6) window.rs + enhanced_window.rs: get_column_as_f64 /
// get_column_as_f64_legacy must work on real Series<f64>-backed columns
// (not only Series<String>), so DataFrame rolling/expanding/EWM actually
// produce columns instead of silently no-opping.
// ---------------------------------------------------------------------

#[test]
fn enhanced_rolling_on_real_f64_column_produces_columns() {
    let mut df = DataFrame::new();
    df.add_column("value".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();

    let config = EnhancedDataFrameWindowExt::rolling(&df, 2).min_periods(2);
    let ops = EnhancedDataFrameWindowExt::apply_rolling(&df, &config);
    let result = ops.mean().unwrap();

    // Previously `get_numeric_column_names()` was empty for a real
    // Series<f64> column, so no "value_mean" column was ever added and
    // this call silently no-op'd (still `Ok`, but with zero effect).
    assert!(
        result.contains_column("value_mean"),
        "rolling mean on a real Series<f64> column must add a result column"
    );
    let means = result.get_column_numeric_values("value_mean").unwrap();
    assert!(means[0].is_nan());
    assert!((means[1] - 1.5).abs() < 1e-9);
    assert!((means[4] - 4.5).abs() < 1e-9);
}

#[test]
fn string_dispatch_rolling_on_real_f64_column_produces_columns() {
    let mut df = DataFrame::new();
    df.add_column("value".to_string(), series_f64(&[2.0, 4.0, 6.0, 8.0]))
        .unwrap();

    // window.rs's string-dispatch API, on a genuinely f64-backed column
    // (not Series<String>).
    let result = DataFrameWindowExt::rolling(&df, 2, "value", "sum", None).unwrap();
    assert!(result.contains_column("value_sum"));
    let sums = result.get_column_numeric_values("value_sum").unwrap();
    assert!(sums[0].is_nan());
    assert!((sums[1] - 6.0).abs() < 1e-9);
    assert!((sums[3] - 14.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------
// (7) window.rs: *_with_options adds min_periods/center/ddof to the
// string-dispatch API.
// ---------------------------------------------------------------------

#[test]
fn string_dispatch_rolling_with_options_honors_min_periods_center_and_ddof() {
    let mut df = DataFrame::new();
    df.add_column("value".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();

    // min_periods=1 (instead of defaulting to window_size) means the
    // first element is no longer NaN.
    let result =
        DataFrameWindowExt::rolling_with_options(&df, 3, "value", "mean", Some(1), false, 1, None)
            .unwrap();
    let means = result.get_column_numeric_values("value_mean").unwrap();
    assert!(
        !means[0].is_nan(),
        "min_periods=1 must make the first rolling value defined, got NaN"
    );
    assert!((means[0] - 1.0).abs() < 1e-9);

    // ddof=0 (population) vs ddof=1 (sample, the old hardcoded default)
    // must produce different std values over the same window.
    let std_sample =
        DataFrameWindowExt::rolling_with_options(&df, 3, "value", "std", Some(3), false, 1, None)
            .unwrap();
    let std_population =
        DataFrameWindowExt::rolling_with_options(&df, 3, "value", "std", Some(3), false, 0, None)
            .unwrap();
    let sample_vals = std_sample.get_column_numeric_values("value_std").unwrap();
    let population_vals = std_population
        .get_column_numeric_values("value_std")
        .unwrap();
    assert!(
        (sample_vals[2] - population_vals[2]).abs() > 1e-9,
        "ddof=1 and ddof=0 must produce different std values \
         (sample={}, population={})",
        sample_vals[2],
        population_vals[2]
    );
}

// ---------------------------------------------------------------------
// (9) jit_window.rs: honest stats -- no fabricated "2x speedup",
// average_speedup_ratio() is Option<f64>::None (there is no genuine
// native-vs-JIT comparison performed anywhere in this module).
// ---------------------------------------------------------------------

#[test]
fn jit_average_speedup_ratio_is_honest_and_time_saved_stays_zero() {
    let ctx = JitWindowContext::with_settings(true, 2);
    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .unwrap();

    // Cross the compile threshold (2) several times over so the cache
    // definitely has a hit and `record_jit_execution` definitely fires at
    // least once.
    for _ in 0..5 {
        let _ = JitDataFrameWindowExt::jit_rolling(&df, 3, &ctx)
            .mean()
            .unwrap();
    }

    let stats = ctx.stats().unwrap();
    assert!(
        stats.jit_executions > 0,
        "expected at least one cache-hit execution"
    );
    // Previously this was a plain f64 that defaulted to 1.0 and, once any
    // JIT-labeled execution occurred, reported a fabricated ~2x speedup
    // computed as `execution_time / 2`. No genuine measurement is taken
    // anywhere in this module, so it must honestly report "not measured".
    assert!(
        stats.average_speedup_ratio().is_none(),
        "average_speedup_ratio() must be None: nothing in this module \
         measures a real native-vs-JIT comparison"
    );
    assert_eq!(
        stats.time_saved_ns, 0,
        "time_saved_ns must never be incremented with a fabricated value"
    );
}

// ---------------------------------------------------------------------
// (10)+(11) jit_window.rs: real config (not a hardcoded window size 10 /
// alpha 0.1) enters the cache key, and JitDataFrameExpanding/JitDataFrameEWM
// have real, correct mean/sum/std/var methods (previously zero methods).
// ---------------------------------------------------------------------

#[test]
fn jit_rolling_with_different_window_sizes_does_not_collide_in_cache() {
    let ctx = JitWindowContext::with_settings(true, 1);
    let mut df = DataFrame::new();
    df.add_column(
        "value".to_string(),
        series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();

    // Two different window sizes, each exercised twice -- once to force a
    // cold-cache compile attempt (threshold=1) and once more against a
    // warm cache. Previously the cache key hardcoded `Some(10)` for every
    // rolling call regardless of the real window size, so these would
    // collide on one entry; verify both configurations still produce
    // results for their own requested window size on every call.
    for _ in 0..2 {
        let small = JitDataFrameWindowExt::jit_rolling(&df, 2, &ctx)
            .mean()
            .unwrap();
        let small_means = small.get_column_numeric_values("value_mean").unwrap();
        // window=2 at the last row: mean(5,6)=5.5.
        assert!((small_means[5] - 5.5).abs() < 1e-9);
    }
    for _ in 0..2 {
        let large = JitDataFrameWindowExt::jit_rolling(&df, 4, &ctx)
            .mean()
            .unwrap();
        let large_means = large.get_column_numeric_values("value_mean").unwrap();
        // window=4 at the last row: mean(3,4,5,6)=4.5.
        assert!((large_means[5] - 4.5).abs() < 1e-9);
    }
}

#[test]
fn jit_expanding_and_ewm_have_real_methods_and_honor_alpha() {
    let ctx = JitWindowContext::with_settings(true, 1);
    let mut df = DataFrame::new();
    df.add_column("value".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    // JitDataFrameExpanding previously had zero methods at all (mean/sum/
    // std/var did not exist -- this would not have compiled).
    let expanding_mean = JitDataFrameWindowExt::jit_expanding(&df, 1, &ctx)
        .mean()
        .unwrap();
    let sums = expanding_mean
        .get_column_numeric_values("value_mean")
        .unwrap();
    assert!((sums[3] - 2.5).abs() < 1e-9); // mean(1,2,3,4)

    let expanding_sum = JitDataFrameWindowExt::jit_expanding(&df, 1, &ctx)
        .sum()
        .unwrap();
    let running_sums = expanding_sum
        .get_column_numeric_values("value_sum")
        .unwrap();
    assert_eq!(running_sums, vec![1.0, 3.0, 6.0, 10.0]);

    // JitDataFrameEWM previously had zero methods (no alpha/span/halflife
    // builders, no mean/std/var) -- this would not have compiled either.
    // alpha=0.5 is deliberately different from the JIT kernel's old
    // hardcoded 0.1, so a regression back to the hardcoded value would be
    // numerically visible here.
    let ewm_mean = JitDataFrameWindowExt::jit_ewm(&df, &ctx)
        .alpha(0.5)
        .unwrap()
        .mean()
        .unwrap();
    let ewm_values = ewm_mean.get_column_numeric_values("value_mean").unwrap();
    // `DataFrameEWM::new()` defaults `adjust = true` (pandas' own default),
    // which this crate's EWM now actually honors: the weighted average
    // over *all* prior observations with weight (1-alpha)^k for the
    // observation k steps back, `y_t = sum_k (1-alpha)^k x_{t-k} / sum_k
    // (1-alpha)^k`. With alpha=0.5: y0 = 1/1 = 1;
    // y1 = (0.5*1 + 1*2)/(0.5+1) = 2.5/1.5 = 5/3;
    // y2 = (0.25*1 + 0.5*2 + 1*3)/1.75 = 4.25/1.75 = 17/7;
    // y3 = (0.125*1 + 0.25*2 + 0.5*3 + 1*4)/1.875 = 6.125/1.875 = 49/15.
    let expected = [1.0, 5.0 / 3.0, 17.0 / 7.0, 49.0 / 15.0];
    for i in 0..4 {
        assert!(
            (ewm_values[i] - expected[i]).abs() < 1e-9,
            "EWM alpha=0.5 at row {}: got {}, expected {}",
            i,
            ewm_values[i],
            expected[i]
        );
    }
}

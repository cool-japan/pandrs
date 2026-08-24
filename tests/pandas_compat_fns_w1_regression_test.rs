//! Regression tests for the `pandas-compat-fns` fixes:
//! `src/dataframe/pandas_compat/functions/**`, `src/dataframe/pandas_compat/groupby.rs`,
//! `src/dataframe/pandas_compat/helpers/**`, `src/dataframe/pandas_compat/types.rs`,
//! `src/dataframe/hierarchical_groupby.rs`.
//!
//! Each test (or test group) is labeled with the item number from the task
//! list it covers.

use pandrs::dataframe::pandas_compat::{
    AstypeErrorsExt, BetweenInclusiveExt, DescribePercentilesExt, PandasGroupByExt,
    ValueCountsOptionsExt,
};
use pandrs::dataframe::{AggFunc, HierarchicalGroupByExt, PandasCompatExt};
use pandrs::{DataFrame, Index, Series};

fn series_f64(values: &[f64]) -> Series<f64> {
    Series::new(values.to_vec(), None).unwrap()
}

fn series_str(values: &[&str]) -> Series<String> {
    Series::new(values.iter().map(|s| s.to_string()).collect(), None).unwrap()
}

// ---------------------------------------------------------------------
// (1) select_rows_by_indices preserves dtype (no "007" -> 7.0 corruption)
// and the row index, instead of re-inferring dtype from stringified
// values and always dropping the index.
// ---------------------------------------------------------------------

#[test]
fn select_rows_by_indices_preserves_string_dtype() {
    let mut df = DataFrame::new();
    df.add_column("zip".to_string(), series_str(&["007", "008", "042", "100"]))
        .unwrap();
    df.add_column("val".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    // `head`, `sort_values`, `dropna`, `nlargest`/`nsmallest`, `sample`,
    // `drop_duplicates`, `filter_by_mask`, `reverse_rows`, `iloc_range` all
    // route row selection through `select_rows_by_indices`; `head` is the
    // simplest to exercise here.
    let result = PandasCompatExt::head(&df, 2).unwrap();
    let zips = result.get_column_string_values("zip").unwrap();
    // Previously this silently parsed "007" as the float 7.0 (losing the
    // leading zero and the text type entirely) via
    // `get_column_numeric_values`, which happily parses any numeric-looking
    // string.
    assert_eq!(zips, vec!["007".to_string(), "008".to_string()]);
}

#[test]
fn select_rows_by_indices_preserves_row_index() {
    let mut df = DataFrame::new();
    df.add_column("val".to_string(), series_f64(&[10.0, 20.0, 30.0, 40.0]))
        .unwrap();
    // `PandasCompatExt::set_index(&self, column: &str, drop: bool)` is also
    // in scope and takes priority over the inherent
    // `DataFrame::set_index(&mut self, Index<String>)` at plain
    // `df.set_index(..)` call syntax (a shared-ref trait method is found
    // before a mut-ref inherent method during lookup); call the inherent
    // one explicitly by its fully-qualified path instead.
    DataFrame::set_index(
        &mut df,
        Index::<String>::new(vec![
            "r0".to_string(),
            "r1".to_string(),
            "r2".to_string(),
            "r3".to_string(),
        ])
        .unwrap(),
    )
    .unwrap();

    let result = PandasCompatExt::tail(&df, 2).unwrap();
    let labels = result.get_index().string_values().unwrap();
    // Previously `select_rows_by_indices` always rebuilt the frame from
    // scratch via `DataFrame::new()`, discarding any index the source
    // frame carried.
    assert_eq!(labels, vec!["r2".to_string(), "r3".to_string()]);
}

// ---------------------------------------------------------------------
// (2) sort_values / sort_by_columns: string-column support, a real total
// order (no non-transitive `partial_cmp().unwrap_or(Equal)`), and NaN
// sorting last regardless of direction.
// ---------------------------------------------------------------------

#[test]
fn sort_values_on_string_column_sorts_lexically_not_numerically() {
    let mut df = DataFrame::new();
    // Numeric-looking text: a numeric sort would put "2" before "10"; a
    // lexical sort puts "10" first.
    df.add_column("code".to_string(), series_str(&["10", "2", "1"]))
        .unwrap();
    // The old implementation only accepted numeric columns at all (`Err`
    // on a string column), on top of getting the order wrong if it had
    // tried.
    let result = df.sort_values("code", true).unwrap();
    let values = result.get_column_string_values("code").unwrap();
    assert_eq!(
        values,
        vec!["1".to_string(), "10".to_string(), "2".to_string()]
    );
}

#[test]
fn sort_values_numeric_nan_always_sorts_last() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        series_f64(&[5.0, f64::NAN, 1.0, f64::NAN, 3.0, 2.0, 4.0]),
    )
    .unwrap();

    // The old comparator (`partial_cmp().unwrap_or(Equal)`) is not a total
    // order once NaN is involved and could scramble non-NaN values too
    // (e.g. asc on this exact input previously produced
    // [5,NaN,1,NaN,2,3,4] -- garbage, not sorted).
    let asc = df.sort_values("a", true).unwrap();
    let asc_values = asc.get_column_numeric_values("a").unwrap();
    assert_eq!(&asc_values[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!(asc_values[5].is_nan() && asc_values[6].is_nan());

    let desc = df.sort_values("a", false).unwrap();
    let desc_values = desc.get_column_numeric_values("a").unwrap();
    assert_eq!(&desc_values[..5], &[5.0, 4.0, 3.0, 2.0, 1.0]);
    assert!(desc_values[5].is_nan() && desc_values[6].is_nan());
}

#[test]
fn sort_by_columns_multi_key_with_string_secondary_column() {
    let mut df = DataFrame::new();
    df.add_column("g".to_string(), series_f64(&[1.0, 1.0, 2.0, 2.0]))
        .unwrap();
    df.add_column(
        "name".to_string(),
        series_str(&["bob", "alice", "zoe", "amy"]),
    )
    .unwrap();

    let result = df.sort_by_columns(&["g", "name"], &[true, true]).unwrap();
    let names = result.get_column_string_values("name").unwrap();
    // Within each numeric group, the string column must sort lexically.
    assert_eq!(
        names,
        vec![
            "alice".to_string(),
            "bob".to_string(),
            "amy".to_string(),
            "zoe".to_string(),
        ]
    );
}

// ---------------------------------------------------------------------
// (3) describe(): ddof=1 (sample std, pandas' default) instead of ddof=0,
// skips NaN (skipna=True default) instead of letting one NaN poison every
// statistic, and `describe_column` is numerically consistent with it.
// ---------------------------------------------------------------------

#[test]
fn describe_uses_ddof1_and_skips_nan() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0, f64::NAN]))
        .unwrap();

    let stats = PandasCompatExt::describe(&df, "a").unwrap();
    // NaN excluded from the count (skipna=True).
    assert_eq!(stats.count, 4);
    assert!((stats.mean - 2.5).abs() < 1e-9);
    // Sample std (ddof=1) of [1,2,3,4]: variance = 5/3, std = sqrt(5/3).
    let expected_std = (5.0f64 / 3.0).sqrt();
    assert!(
        (stats.std - expected_std).abs() < 1e-9,
        "expected ddof=1 std {expected_std}, got {}",
        stats.std
    );

    let describe_column_stats = df.describe_column("a").unwrap();
    assert!(
        (describe_column_stats.get("std").unwrap() - expected_std).abs() < 1e-9,
        "describe_column should agree with describe() on ddof=1 std"
    );
}

#[test]
fn describe_single_value_std_is_nan_not_zero() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[7.0])).unwrap();
    let stats = PandasCompatExt::describe(&df, "a").unwrap();
    assert_eq!(stats.count, 1);
    assert!(
        stats.std.is_nan(),
        "sample std of a single observation is undefined (NaN), not 0"
    );

    // `describe_column` (helpers/aggregations.rs) shares the same n=1
    // fix: its old `(n - 1.0).max(1.0)` divisor clamp silently treated a
    // single observation as if it had 2, reporting a spurious 0 instead.
    let describe_column_stats = df.describe_column("a").unwrap();
    assert!(
        describe_column_stats.get("std").unwrap().is_nan(),
        "describe_column's single-observation std must also be NaN, not 0"
    );
}

// ---------------------------------------------------------------------
// (3, bonus) describe_percentiles: pandas' `describe(percentiles=[...])`
// vocabulary, numerically consistent with `describe`/`describe_column`
// (same skipna/ddof=1/linear-interpolation conventions), always including
// the median even when it isn't explicitly requested.
// ---------------------------------------------------------------------

#[test]
fn describe_percentiles_includes_median_and_matches_describe_on_default_quartiles() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .unwrap();

    // Requesting only the 25th/75th percentiles must still yield the 50th
    // (median) in the output, matching pandas' own behavior.
    let stats = df.describe_percentiles("a", &[0.25, 0.75]).unwrap();
    let labels: Vec<&str> = stats.iter().map(|(l, _)| l.as_str()).collect();
    assert_eq!(
        labels,
        vec!["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    );

    let get = |label: &str| stats.iter().find(|(l, _)| l == label).unwrap().1;
    let full_describe = PandasCompatExt::describe(&df, "a").unwrap();
    assert!((get("mean") - full_describe.mean).abs() < 1e-9);
    assert!((get("std") - full_describe.std).abs() < 1e-9);
    assert!((get("25%") - full_describe.q25).abs() < 1e-9);
    assert!((get("50%") - full_describe.q50).abs() < 1e-9);
    assert!((get("75%") - full_describe.q75).abs() < 1e-9);
}

#[test]
fn describe_percentiles_rejects_out_of_range_values() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[1.0, 2.0, 3.0]))
        .unwrap();
    assert!(df.describe_percentiles("a", &[1.5]).is_err());
    assert!(df.describe_percentiles("a", &[-0.1]).is_err());
}

// ---------------------------------------------------------------------
// (4) duplicated_rows(subset, keep) actually honors `subset` instead of
// always hashing every column.
// ---------------------------------------------------------------------

#[test]
fn duplicated_rows_respects_subset() {
    let mut df = DataFrame::new();
    // `a` has a repeated value at rows 0/2, but `b` differs there, so a
    // whole-row comparison (the old, subset-ignoring behavior) would find
    // no duplicates at all.
    df.add_column("a".to_string(), series_f64(&[1.0, 2.0, 1.0, 3.0]))
        .unwrap();
    df.add_column("b".to_string(), series_f64(&[10.0, 20.0, 99.0, 30.0]))
        .unwrap();

    let whole_row = df.duplicated_rows(None, "first").unwrap();
    assert_eq!(whole_row, vec![false, false, false, false]);

    let subset_a = df.duplicated_rows(Some(&["a"]), "first").unwrap();
    assert_eq!(subset_a, vec![false, false, true, false]);
}

// ---------------------------------------------------------------------
// (5) astype: int64 produces a real Series<i64>; a bool dtype target on a
// string source column no longer silently drops the column; `errors`
// defaults to "raise" and `astype_errors` exposes "coerce" too.
// ---------------------------------------------------------------------

#[test]
fn astype_int64_round_trips_as_integer() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[1.7, -2.3, 3.9]))
        .unwrap();
    let result = df.astype("a", "int64").unwrap();
    let values = result.get_column_numeric_values("a").unwrap();
    // Truncation toward zero (numpy/pandas `.astype(int)` semantics), not
    // `.floor()` (which would have sent -2.3 to -3.0).
    assert_eq!(values, vec![1.0, -2.0, 3.0]);
}

#[test]
fn astype_bool_from_string_column_is_not_dropped() {
    let mut df = DataFrame::new();
    df.add_column("flag".to_string(), series_str(&["true", "false", "1", "0"]))
        .unwrap();
    df.add_column("other".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    let result = df.astype("flag", "bool").unwrap();
    // Previously the "bool" match arm had no string-source branch, so
    // nothing was ever added to the result for this column.
    assert!(
        result.contains_column("flag"),
        "bool-from-string astype must not silently drop the column"
    );
    let values = result.get_column_numeric_values("flag").unwrap();
    assert_eq!(values, vec![1.0, 0.0, 1.0, 0.0]);
    // Untouched columns still pass through.
    assert!(result.contains_column("other"));
}

#[test]
fn astype_default_errors_raise_on_unparsable_value() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_str(&["1.0", "not_a_number"]))
        .unwrap();
    let result = df.astype("a", "float64");
    assert!(
        result.is_err(),
        "astype's default errors='raise' must fail on an unparsable value"
    );
}

#[test]
fn astype_errors_coerce_turns_failures_into_nan() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_str(&["1.0", "not_a_number"]))
        .unwrap();
    let result = df.astype_errors("a", "float64", "coerce").unwrap();
    let values = result.get_column_numeric_values("a").unwrap();
    assert_eq!(values[0], 1.0);
    assert!(values[1].is_nan());
}

#[test]
fn astype_int64_errors_coerce_promotes_to_float_instead_of_fabricating_zero() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_str(&["1", "not_a_number", "3"]))
        .unwrap();
    let result = df.astype_errors("a", "int64", "coerce").unwrap();
    let values = result.get_column_numeric_values("a").unwrap();
    // The unparsable entry must surface as NaN, never a fabricated `0`
    // (there is no null-carrying i64 column here, so the whole column
    // promotes to f64 -- matching pandas' own `to_numeric(errors="coerce")`
    // promotion -- instead of silently writing an indistinguishable-from-real
    // integer zero).
    assert_eq!(values[0], 1.0);
    assert!(
        values[1].is_nan(),
        "unparsable int64 coerce must be NaN, not a fabricated 0"
    );
    assert_eq!(values[2], 3.0);
}

#[test]
fn astype_int64_errors_coerce_stays_integer_when_fully_clean() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_str(&["1", "2", "3"]))
        .unwrap();
    let result = df.astype_errors("a", "int64", "coerce").unwrap();
    // No failures at all: still a real, non-promoted integer conversion.
    let values = result.get_column_numeric_values("a").unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn astype_bool_errors_coerce_raises_instead_of_fabricating_false() {
    let mut df = DataFrame::new();
    df.add_column("flag".to_string(), series_str(&["true", "not_a_bool"]))
        .unwrap();
    // Unlike float64/int64, this crate has no null-carrying bool column, so
    // there is no honest value `errors="coerce"` could substitute for an
    // unparsable entry -- it must raise rather than silently writing `false`
    // (the exact "false for missing" substitution NA semantics forbid).
    let result = df.astype_errors("flag", "bool", "coerce");
    assert!(
        result.is_err(),
        "bool coerce must raise on an unparsable value, not fabricate false"
    );
}

// ---------------------------------------------------------------------
// (6) cut/qcut: the label text matches the actual bin boundaries, and
// every row gets an entry (constant columns, NaN, and repeated values
// that would otherwise collapse quantile edges no longer shorten or
// desync the result).
// ---------------------------------------------------------------------

#[test]
fn cut_result_length_matches_row_count_for_constant_and_nan_columns() {
    let mut constant_df = DataFrame::new();
    constant_df
        .add_column("a".to_string(), series_f64(&[5.0, 5.0, 5.0, 5.0]))
        .unwrap();
    let constant_result = constant_df.cut("a", 3).unwrap();
    assert_eq!(constant_result.len(), constant_df.row_count());
    assert!(constant_result.iter().all(|label| !label.is_empty()));

    let mut nan_df = DataFrame::new();
    nan_df
        .add_column(
            "a".to_string(),
            series_f64(&[1.0, f64::NAN, 3.0, 5.0, f64::NAN]),
        )
        .unwrap();
    let nan_result = nan_df.cut("a", 2).unwrap();
    assert_eq!(nan_result.len(), nan_df.row_count());
    assert_eq!(nan_result[1], "NaN");
    assert_eq!(nan_result[4], "NaN");
}

#[test]
fn cut_label_boundaries_match_bin_membership() {
    let mut df = DataFrame::new();
    // An interior edge sits exactly on a data value (3.0, the midpoint of
    // [1,5] with 2 bins): pandas' `cut` bins are left-open/right-closed,
    // so that value must land in the *lower* bin, and the label text must
    // agree with that (say "(.., 3.00]" for the lower bin, not claim a
    // different upper bound than the comparison actually used).
    df.add_column("a".to_string(), series_f64(&[1.0, 3.0, 5.0]))
        .unwrap();
    let result = df.cut("a", 2).unwrap();
    assert_eq!(result.len(), 3);
    // Row for value 3.0 (the interior edge) and row for value 1.0 (the
    // true minimum) must fall in the *same* bin, and that bin's label
    // must claim "3.00" as its upper bound.
    assert_eq!(result[0], result[1]);
    assert!(result[1].contains("3.00"));
    // The true maximum must fall in the *other* bin.
    assert_ne!(result[2], result[1]);
}

#[test]
fn qcut_result_length_matches_row_count_with_duplicate_edges() {
    let mut df = DataFrame::new();
    // Many repeated values with q=4: the raw quantile edges collapse for
    // several of the requested quantiles.
    df.add_column(
        "a".to_string(),
        series_f64(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
    )
    .unwrap();
    let result = df.qcut("a", 4).unwrap();
    assert_eq!(result.len(), df.row_count());
    // No row should have been left with the old sentinel empty label.
    assert!(result.iter().all(|label| !label.is_empty()));
}

// ---------------------------------------------------------------------
// (7) memory_usage(): a real, content-dependent estimate rather than a
// `cols*rows*8 + cols*64 + 256` formula unrelated to the data.
// ---------------------------------------------------------------------

#[test]
fn memory_usage_reflects_actual_string_content() {
    let mut short_df = DataFrame::new();
    short_df
        .add_column("a".to_string(), series_str(&["x", "y"]))
        .unwrap();

    let mut long_df = DataFrame::new();
    long_df
        .add_column(
            "a".to_string(),
            series_str(&[&"z".repeat(10_000), &"w".repeat(10_000)]),
        )
        .unwrap();

    // Same shape (1 column, 2 rows), wildly different content size -- the
    // old formula depended only on row/column counts and would have
    // reported identical usage for both.
    assert!(
        long_df.memory_usage() > short_df.memory_usage() + 15_000,
        "memory_usage must scale with actual string content, not just shape"
    );
}

// ---------------------------------------------------------------------
// (8) value_counts_with_options: normalize / sort / ascending / dropna,
// plus fully deterministic ordering (already true of the existing
// `value_counts`, extended here with the missing pandas parameters).
// ---------------------------------------------------------------------

#[test]
fn value_counts_with_options_normalize_sort_and_dropna() {
    let mut df = DataFrame::new();
    df.add_column(
        "category".to_string(),
        series_str(&["A", "B", "A", "", "A", "B"]),
    )
    .unwrap();

    let dropna_counts = df
        .value_counts_with_options("category", false, true, false, true)
        .unwrap();
    // dropna=true excludes the empty-string "missing" marker entirely.
    assert!(dropna_counts.iter().all(|(v, _)| !v.is_empty()));
    assert_eq!(dropna_counts[0], ("A".to_string(), 3.0));
    assert_eq!(dropna_counts[1], ("B".to_string(), 2.0));

    let keep_na_counts = df
        .value_counts_with_options("category", false, true, false, false)
        .unwrap();
    assert_eq!(keep_na_counts.len(), 3);

    let normalized = df
        .value_counts_with_options("category", true, true, false, true)
        .unwrap();
    let total: f64 = normalized.iter().map(|(_, v)| v).sum();
    assert!((total - 1.0).abs() < 1e-9);

    let ascending = df
        .value_counts_with_options("category", false, true, true, true)
        .unwrap();
    assert_eq!(ascending[0], ("B".to_string(), 2.0));
    assert_eq!(ascending[1], ("A".to_string(), 3.0));
}

// ---------------------------------------------------------------------
// (9) groupby.rs: count() reports real non-null counts (not size());
// group order is deterministic; a materialization failure can no longer
// misalign the key/value columns (defensive, NaN-on-failure design).
// ---------------------------------------------------------------------

#[test]
fn groupby_count_skips_nulls_unlike_size() {
    let mut df = DataFrame::new();
    df.add_column("cat".to_string(), series_str(&["A", "A", "A", "B", "B"]))
        .unwrap();
    df.add_column(
        "val".to_string(),
        series_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0]),
    )
    .unwrap();

    let gb = df.groupby_multi(&["cat"]).unwrap();
    let sizes = gb.size().unwrap();
    let counts = gb.count().unwrap();

    let size_cats = sizes.get_column_string_values("cat").unwrap();
    let size_vals = sizes.get_column_numeric_values("size").unwrap();
    let a_size_idx = size_cats.iter().position(|c| c == "A").unwrap();
    let b_size_idx = size_cats.iter().position(|c| c == "B").unwrap();
    assert_eq!(size_vals[a_size_idx], 3.0);
    assert_eq!(size_vals[b_size_idx], 2.0);

    // count() must NOT just mirror size(): group A has one NaN among its
    // 3 rows (non-null count 2), group B has one NaN among its 2 rows
    // (non-null count 1).
    assert!(!counts.contains_column("size"));
    let count_cats = counts.get_column_string_values("cat").unwrap();
    let count_vals = counts.get_column_numeric_values("val").unwrap();
    let a_count_idx = count_cats.iter().position(|c| c == "A").unwrap();
    let b_count_idx = count_cats.iter().position(|c| c == "B").unwrap();
    assert_eq!(count_vals[a_count_idx], 2.0);
    assert_eq!(count_vals[b_count_idx], 1.0);
}

#[test]
fn groupby_group_order_is_deterministic_across_runs() {
    let mut df = DataFrame::new();
    df.add_column(
        "cat".to_string(),
        series_str(&["D", "B", "A", "C", "B", "A"]),
    )
    .unwrap();
    df.add_column(
        "val".to_string(),
        series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();

    // Run the same aggregation many times; every run must produce the
    // exact same row order (previously governed by raw `HashMap`
    // iteration order, which is not guaranteed stable run-to-run).
    let mut orders = std::collections::HashSet::new();
    for _ in 0..25 {
        let gb = df.groupby_multi(&["cat"]).unwrap();
        let result = gb.sum().unwrap();
        let cats = result.get_column_string_values("cat").unwrap();
        orders.insert(cats);
    }
    assert_eq!(
        orders.len(),
        1,
        "groupby row order must be deterministic across runs"
    );
    // And it should specifically be sorted order.
    let only_order = orders.into_iter().next().unwrap();
    let mut sorted = only_order.clone();
    sorted.sort();
    assert_eq!(only_order, sorted);
}

#[test]
fn groupby_agg_multi_is_row_aligned_and_deterministic() {
    let mut df = DataFrame::new();
    df.add_column("cat".to_string(), series_str(&["A", "B", "A", "B", "A"]))
        .unwrap();
    df.add_column("val".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();

    let gb = df.groupby_multi(&["cat"]).unwrap();
    let result = gb.agg(&[("val", "sum"), ("val", "mean")]).unwrap();
    let cats = result.get_column_string_values("cat").unwrap();
    let sums = result.get_column_numeric_values("val_sum").unwrap();
    let means = result.get_column_numeric_values("val_mean").unwrap();
    assert_eq!(cats.len(), sums.len());
    assert_eq!(cats.len(), means.len());
    let a_idx = cats.iter().position(|c| c == "A").unwrap();
    let b_idx = cats.iter().position(|c| c == "B").unwrap();
    assert_eq!(sums[a_idx], 9.0); // 1+3+5
    assert_eq!(sums[b_idx], 6.0); // 2+4
    assert!((means[a_idx] - 3.0).abs() < 1e-9);
    assert!((means[b_idx] - 3.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------
// (10) hierarchical_groupby.rs: a single-element group's Std is NaN (the
// sample standard deviation is undefined for n=1), not a fabricated 0.0.
// ---------------------------------------------------------------------

#[test]
fn hierarchical_groupby_single_element_std_is_nan() {
    let mut df = DataFrame::new();
    df.add_column("region".to_string(), series_str(&["east", "east", "west"]))
        .unwrap();
    df.add_column("value".to_string(), series_f64(&[1.0, 2.0, 100.0]))
        .unwrap();

    let hgb = df.hierarchical_groupby(vec!["region".to_string()]).unwrap();
    let hierarchical_agg = pandrs::dataframe::HierarchicalAgg::new("value".to_string())
        .add_level_agg(0, AggFunc::Std, "value_std".to_string());
    let result = hgb.agg_hierarchical(vec![hierarchical_agg]).unwrap();

    let regions = result.get_column_string_values("region").unwrap();
    let stds = result.get_column_string_values("value_std").unwrap();
    let west_idx = regions.iter().position(|r| r == "west").unwrap();
    // "west" has exactly one row -- sample std (ddof=1) is undefined.
    assert_eq!(stds[west_idx], "NaN");
}

// ---------------------------------------------------------------------
// (13) idxmax/idxmin skip NaN (matching argmax/argmin semantics);
// cummax/cummin leave a NaN's own row as NaN instead of silently filling
// it with the running extreme; transpose preserves the source column
// names (as the transposed frame's row index) instead of discarding them.
// ---------------------------------------------------------------------

#[test]
fn idxmax_idxmin_skip_nan() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[5.0, f64::NAN]))
        .unwrap();
    // Previously `max_by(partial_cmp().unwrap_or(Equal))` could hand a
    // trailing NaN the win over the true maximum.
    assert_eq!(df.idxmax("a").unwrap(), Some(0));
    assert_eq!(df.idxmin("a").unwrap(), Some(0));
}

#[test]
fn cummax_cummin_report_nan_at_nan_row_not_the_running_extreme() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        series_f64(&[1.0, f64::NAN, 3.0, f64::NAN, 2.0]),
    )
    .unwrap();
    let cummax = df.cummax("a").unwrap();
    assert_eq!(cummax[0], 1.0);
    assert!(cummax[1].is_nan(), "NaN's own row must stay NaN");
    assert_eq!(cummax[2], 3.0);
    assert!(cummax[3].is_nan(), "NaN's own row must stay NaN");
    assert_eq!(cummax[4], 3.0); // running max still tracked across the NaN

    let cummin = df.cummin("a").unwrap();
    assert_eq!(cummin[0], 1.0);
    assert!(cummin[1].is_nan());
    assert_eq!(cummin[2], 1.0);
    assert!(cummin[3].is_nan());
    assert_eq!(cummin[4], 1.0);
}

#[test]
fn transpose_preserves_source_column_names_as_row_index() {
    let mut df = DataFrame::new();
    df.add_column("alpha".to_string(), series_f64(&[1.0, 2.0]))
        .unwrap();
    df.add_column("beta".to_string(), series_f64(&[3.0, 4.0]))
        .unwrap();

    let transposed = PandasCompatExt::transpose(&df).unwrap();
    let labels = transposed.get_index().string_values().unwrap();
    // Previously the source column names ("alpha", "beta") were discarded
    // entirely -- nothing in the transposed frame recorded them.
    assert_eq!(labels, vec!["alpha".to_string(), "beta".to_string()]);
}

// ---------------------------------------------------------------------
// Bonus: `between_inclusive` (functions_4.rs) implements pandas' full
// `inclusive` vocabulary now that a check confirmed `is_between`'s
// existing boolean toggle already covered the "trivial inclusive param"
// ask on its own.
// ---------------------------------------------------------------------

#[test]
fn between_inclusive_four_way_vocabulary() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), series_f64(&[1.0, 2.0, 3.0, 4.0, 5.0]))
        .unwrap();

    assert_eq!(
        df.between_inclusive("a", 2.0, 4.0, "both").unwrap(),
        vec![false, true, true, true, false]
    );
    assert_eq!(
        df.between_inclusive("a", 2.0, 4.0, "neither").unwrap(),
        vec![false, false, true, false, false]
    );
    assert_eq!(
        df.between_inclusive("a", 2.0, 4.0, "left").unwrap(),
        vec![false, true, true, false, false]
    );
    assert_eq!(
        df.between_inclusive("a", 2.0, 4.0, "right").unwrap(),
        vec![false, false, true, true, false]
    );
}

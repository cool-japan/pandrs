//! Regression tests for the `series-na` fix wave: `src/series/categorical.rs`,
//! `src/series/window.rs`, `src/series/base.rs`, `src/series/na.rs`,
//! `src/na.rs`, `src/series/datetime_accessor.rs` (round/floor/ceil), and
//! `src/series/string_accessor.rs` (`split(expand=true)`).
//!
//! Each test names the bug it guards against in its body so a future
//! regression is easy to diagnose from a failure message alone.

use pandrs::series::{StringCategorical, WindowClosed, WindowExt, WindowOps};
use pandrs::{Series, NA};

// ---------------------------------------------------------------------
// Categorical: add/remove/reorder_categories code integrity
// ---------------------------------------------------------------------

#[test]
fn categorical_reorder_remaps_codes_not_just_the_category_list() {
    let values = ["a", "b", "a", "c"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut cat = StringCategorical::new(values, None, false).unwrap();
    // Auto-inferred categories, first-seen order: [a, b, c].

    cat.reorder_categories(vec!["c".to_string(), "b".to_string(), "a".to_string()])
        .unwrap();
    assert_eq!(
        cat.categories(),
        &vec!["c".to_string(), "b".to_string(), "a".to_string()]
    );

    // Every row must still resolve to its *original* value after the
    // reorder -- the previous implementation swapped `categories_list`
    // without touching `codes`, so `get(i)` silently returned whatever
    // category ended up at row i's old numeric code instead.
    assert_eq!(cat.get(0).unwrap(), "a");
    assert_eq!(cat.get(1).unwrap(), "b");
    assert_eq!(cat.get(2).unwrap(), "a");
    assert_eq!(cat.get(3).unwrap(), "c");

    // reorder_categories must reject a category set that doesn't match
    // (missing category, and duplicate/wrong-length) rather than silently
    // accepting a nonsensical remap.
    let mut cat2 =
        StringCategorical::new(vec!["a".to_string(), "b".to_string()], None, false).unwrap();
    assert!(cat2.reorder_categories(vec!["a".to_string()]).is_err());
    assert!(cat2
        .reorder_categories(vec!["a".to_string(), "b".to_string(), "a".to_string()])
        .is_err());
}

#[test]
fn categorical_add_categories_inserts_into_the_lookup_map() {
    let mut cat =
        StringCategorical::new(vec!["a".to_string(), "b".to_string()], None, false).unwrap();
    cat.add_categories(vec!["c".to_string(), "d".to_string()])
        .unwrap();

    assert_eq!(cat.categories().len(), 4);
    // The bug: add_categories pushed onto categories_list but never
    // inserted into category_to_code, so a newly added category's code
    // could not be looked up or encoded.
    assert_eq!(cat.get_code(&"c".to_string()), Some(2));
    assert_eq!(cat.get_code(&"d".to_string()), Some(3));
    assert_eq!(cat.encode(&["c".to_string(), "z".to_string()]), vec![2, -1]);
}

#[test]
fn categorical_remove_categories_turns_rows_na_and_remaps_survivors() {
    let mut cat = StringCategorical::new(
        ["a", "b", "a", "c"].iter().map(|s| s.to_string()).collect(),
        None,
        false,
    )
    .unwrap();

    cat.remove_categories(&["b".to_string()]).unwrap();

    assert_eq!(cat.categories(), &vec!["a".to_string(), "c".to_string()]);
    assert_eq!(cat.get(0).unwrap(), "a");
    assert_eq!(cat.get(1), None); // row referencing the removed category is now NA
    assert_eq!(cat.get(2).unwrap(), "a");
    // Row 3 (originally "c", old code 2) must still resolve correctly once
    // the category list shrinks to 2 entries -- the previous
    // implementation left `codes` untouched, so this either pointed at the
    // wrong (shifted) category or was out of bounds.
    assert_eq!(cat.get(3).unwrap(), "c");
    assert_eq!(cat.codes(), &vec![0, -1, 0, 1]);
}

#[test]
fn categorical_new_compact_accessors_work_without_a_raw_values_copy() {
    let values = ["x", "y", "x"].iter().map(|s| s.to_string()).collect();
    let cat = StringCategorical::new_compact(values, None, false).unwrap();

    // Previously `new_compact` cleared the raw `values` array but `len()`,
    // `get()`, and `to_series()` all read from it, so a compact
    // categorical reported `len() == 0` and every accessor was broken.
    assert_eq!(cat.len(), 3);
    assert!(!cat.is_empty());
    assert_eq!(cat.get(0).unwrap(), "x");
    assert_eq!(cat.get(1).unwrap(), "y");
    assert_eq!(cat.get(2).unwrap(), "x");

    let series = cat.to_series(None).unwrap();
    assert_eq!(series.len(), 3);
    assert_eq!(
        series.values(),
        &["x".to_string(), "y".to_string(), "x".to_string()]
    );
}

#[test]
fn categorical_value_counts_is_deterministic_and_positionally_labeled() {
    let values = ["a", "b", "a", "a"].iter().map(|s| s.to_string()).collect();
    // "z" never occurs -- this is exactly the case that distinguishes
    // "aligned with categories()" (zero-count categories still appear)
    // from an implementation that only ever emits categories that
    // occurred at least once.
    let cat = StringCategorical::new(
        values,
        Some(vec!["a".to_string(), "b".to_string(), "z".to_string()]),
        false,
    )
    .unwrap();

    let counts = cat.value_counts().unwrap();
    assert_eq!(counts.len(), 3);
    assert_eq!(counts.name(), Some(&"count".to_string()));

    // The previous implementation computed value->count labels into a
    // local `indices` vector and then never attached them to the returned
    // `Series<usize>` (which has no index), so a count could not be
    // correlated back to its category at all. Positional correspondence
    // with `categories()` is how that's fixed without changing the return
    // type: `value_counts()[i]` is the count for `categories()[i]`.
    let labeled: Vec<(&String, &usize)> = cat.categories().iter().zip(counts.values()).collect();
    assert_eq!(
        labeled,
        vec![
            (&"a".to_string(), &3usize),
            (&"b".to_string(), &1usize),
            (&"z".to_string(), &0usize),
        ]
    );

    // Determinism: repeated calls must agree (the HashMap-iteration-order
    // implementation could reorder rows between calls).
    let counts_again = cat.value_counts().unwrap();
    assert_eq!(counts.values(), counts_again.values());
}

#[test]
fn categorical_na_round_trip_preserves_length_and_positions() {
    let input = vec![
        NA::Value("a".to_string()),
        NA::Value("b".to_string()),
        NA::NA,
        NA::Value("c".to_string()),
        NA::NA,
    ];

    let cat = StringCategorical::from_na_vec(input.clone(), None, None).unwrap();
    // Previously `from_na_vec` filtered out `NA::NA` entries before
    // building the categorical, so `len()` silently shrank to the
    // non-NA count (3) instead of matching the input's length (5).
    assert_eq!(cat.len(), 5);
    assert_eq!(cat.codes()[2], -1);
    assert_eq!(cat.codes()[4], -1);
    assert!(cat.codes()[0] >= 0);
    assert!(cat.codes()[1] >= 0);
    assert!(cat.codes()[3] >= 0);

    let round_tripped = cat.to_na_vec();
    // Previously `to_na_vec` just rewrapped the (already NA-dropping) raw
    // values in `NA::Value`, so it could never emit `NA::NA` at all.
    assert_eq!(round_tripped.len(), 5);
    for (original, back) in input.iter().zip(round_tripped.iter()) {
        match (original, back) {
            (NA::NA, NA::NA) => {}
            (NA::Value(a), NA::Value(b)) => assert_eq!(a, b),
            other => panic!("round trip changed NA-ness: {:?}", other),
        }
    }
}

// ---------------------------------------------------------------------
// src/na.rs: NA ordering (NA-last) and documented Eq/Hash semantics
// ---------------------------------------------------------------------

#[test]
fn na_sorts_last_matching_split_dataframe_sort_convention() {
    let mut values = vec![NA::Value(5), NA::NA, NA::Value(1), NA::NA, NA::Value(3)];
    values.sort();
    // Previously NA sorted *first* here, contradicting
    // `optimized/split_dataframe/sort.rs`'s documented/implemented
    // `na_position="last"` convention (pandas' own default).
    assert_eq!(
        values,
        vec![NA::Value(1), NA::Value(3), NA::Value(5), NA::NA, NA::NA]
    );
}

#[test]
fn na_equals_na_and_hashes_equal_documented_groupby_semantics() {
    // This is not "should NA behave like NaN" (it deliberately doesn't) --
    // it pins down the documented, load-bearing behavior groupby relies
    // on: every NA compares equal and hashes the same, so a
    // HashMap/HashSet-based groupby buckets every all-NA row together.
    assert_eq!(NA::<i32>::NA, NA::<i32>::NA);
    use std::collections::HashSet;
    let mut set: HashSet<NA<i32>> = HashSet::new();
    set.insert(NA::NA);
    set.insert(NA::NA);
    set.insert(NA::Value(1));
    assert_eq!(set.len(), 2);
}

// ---------------------------------------------------------------------
// src/series/base.rs: Series<i32>/<i64>/<f64> sum/mean/min/max
// ---------------------------------------------------------------------

#[test]
fn series_i32_sum_accumulates_in_i64_without_overflow() {
    // Three copies of i32::MAX overflow a plain i32 accumulator (would
    // wrap in release builds, panic in debug builds) but fit comfortably
    // in i64.
    let s = Series::new(vec![i32::MAX, i32::MAX, i32::MAX], None).unwrap();
    assert_eq!(s.sum(), 3_i64 * i32::MAX as i64);
    assert!((s.mean().unwrap() - i32::MAX as f64).abs() < 1.0);
}

#[test]
fn series_i64_sum_mean_min_max_are_implemented() {
    // Previously these methods existed only for Series<i32>.
    let s = Series::new(vec![1_i64, 2, 3], None).unwrap();
    assert_eq!(s.sum(), 6);
    assert_eq!(s.mean().unwrap(), 2.0);
    assert_eq!(s.min().unwrap(), 1);
    assert_eq!(s.max().unwrap(), 3);
}

#[test]
fn series_f64_sum_mean_min_max_skip_nan_like_pandas_skipna() {
    let s = Series::new(vec![1.0, f64::NAN, 3.0], None).unwrap();
    assert_eq!(s.sum(), 4.0);
    assert_eq!(s.mean().unwrap(), 2.0); // divisor is the non-NaN count (2), not len() (3)
    assert_eq!(s.min().unwrap(), 1.0);
    assert_eq!(s.max().unwrap(), 3.0);

    let all_nan = Series::new(vec![f64::NAN, f64::NAN], None).unwrap();
    assert_eq!(all_nan.sum(), 0.0); // sum of zero observations is the additive identity
    assert!(all_nan.mean().is_err()); // but there is no honest "average of zero observations"
    assert!(all_nan.min().is_err());
    assert!(all_nan.max().is_err());
}

// ---------------------------------------------------------------------
// src/series/na.rs: all-NA NASeries min/max
// ---------------------------------------------------------------------

#[test]
fn na_series_all_na_min_max_returns_na_not_a_panic() {
    use pandrs::NASeries;
    let s: NASeries<i32> = NASeries::new(vec![NA::NA, NA::NA, NA::NA], None).unwrap();
    assert_eq!(s.min(), NA::NA);
    assert_eq!(s.max(), NA::NA);

    let mixed = NASeries::new(vec![NA::Value(5), NA::NA, NA::Value(1)], None).unwrap();
    assert_eq!(mixed.min(), NA::Value(1));
    assert_eq!(mixed.max(), NA::Value(5));
}

// ---------------------------------------------------------------------
// src/series/window.rs: EWM adjust / ignore_na / ddof / min_periods
// ---------------------------------------------------------------------

#[test]
fn ewm_adjust_matches_pandas_formula() {
    let s = Series::new(vec![1.0, 2.0, 3.0], None).unwrap();

    // adjust defaults to true (pandas' default): the weighted average over
    // *all* prior observations, weights w_i = (1 - alpha)^i.
    let adjusted = s.ewm().alpha(0.5).unwrap().mean().unwrap();
    assert!((adjusted.values()[0] - 1.0).abs() < 1e-12);
    assert!((adjusted.values()[1] - 1.666_666_666_666_666_7).abs() < 1e-12); // (1*2+0.5*1)/1.5
    assert!((adjusted.values()[2] - 2.428_571_428_571_428_4).abs() < 1e-12); // (1*3+0.5*2+0.25*1)/1.75

    // adjust=false: the plain recursive form y_t = alpha*x_t + (1-alpha)*y_{t-1}.
    let recursive = s.ewm().alpha(0.5).unwrap().adjust(false).mean().unwrap();
    assert!((recursive.values()[0] - 1.0).abs() < 1e-12);
    assert!((recursive.values()[1] - 1.5).abs() < 1e-12); // 0.5*2 + 0.5*1
    assert!((recursive.values()[2] - 2.25).abs() < 1e-12); // 0.5*3 + 0.5*1.5

    // The two must diverge once `adjust` actually has an effect --
    // previously it was accepted but never read, so every EWM mean used
    // the recursive formula regardless of `adjust`'s value.
    assert!((adjusted.values()[1] - recursive.values()[1]).abs() > 1e-6);
}

#[test]
fn ewm_ignore_na_matches_pandas_docstring_example() {
    // pandas' own `ignore_na` docstring example: for `[x0, None, x2]` with
    // `adjust=true`, the weights on x0 and x2 are `(1-alpha)^2, 1` when
    // `ignore_na=false`, and `1-alpha, 1` when `ignore_na=true`.
    let s = Series::new(vec![1.0, f64::NAN, 3.0], None).unwrap();

    let not_ignoring = s.ewm().alpha(0.5).unwrap().mean().unwrap();
    // y2 = (1*3 + 0.25*1) / (1 + 0.25) = 3.25 / 1.25 = 2.6
    assert!((not_ignoring.values()[2] - 2.6).abs() < 1e-12);
    // The NaN row itself carries the last known estimate forward, it does
    // not reset to NaN.
    assert!((not_ignoring.values()[1] - 1.0).abs() < 1e-12);

    let ignoring = s.ewm().alpha(0.5).unwrap().ignore_na(true).mean().unwrap();
    // y2 = (1*3 + 0.5*1) / (1 + 0.5) = 3.5 / 1.5 = 2.333...
    assert!((ignoring.values()[2] - 2.333_333_333_333_333_5).abs() < 1e-12);
}

#[test]
fn ewm_std_var_honor_ddof_and_min_periods() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0], None).unwrap();

    // ddof=1 needs more than one effective observation to be defined.
    let std_ddof1 = s.ewm().alpha(0.5).unwrap().std(1).unwrap();
    assert!(std_ddof1.values()[0].is_nan());
    assert!(!std_ddof1.values()[1].is_nan());

    // ddof=0 (population/"biased") is well-defined from a single point.
    let std_ddof0 = s.ewm().alpha(0.5).unwrap().std(0).unwrap();
    assert!((std_ddof0.values()[0] - 0.0).abs() < 1e-12);

    // ddof must actually change the result -- previously it was accepted
    // (as `_ddof`) but never read.
    assert!((std_ddof1.values()[2] - std_ddof0.values()[2]).abs() > 1e-6);

    // var and std agree (var == std^2) at every defined position.
    let var_ddof1 = s.ewm().alpha(0.5).unwrap().var(1).unwrap();
    for i in 0..4 {
        if std_ddof1.values()[i].is_nan() {
            assert!(var_ddof1.values()[i].is_nan());
        } else {
            assert!((var_ddof1.values()[i] - std_ddof1.values()[i].powi(2)).abs() < 1e-9);
        }
    }

    // min_periods gates when a mean value appears at all.
    let gated = s.ewm().alpha(0.5).unwrap().min_periods(3).mean().unwrap();
    assert!(gated.values()[0].is_nan());
    assert!(gated.values()[1].is_nan());
    assert!(!gated.values()[2].is_nan());
}

#[test]
fn ewm_mean_std_handle_empty_and_all_nan_without_panicking() {
    let empty: Series<f64> = Series::new(vec![], None).unwrap();
    assert_eq!(empty.ewm().alpha(0.5).unwrap().mean().unwrap().len(), 0);
    assert_eq!(empty.ewm().alpha(0.5).unwrap().std(1).unwrap().len(), 0);

    let all_nan = Series::new(vec![f64::NAN, f64::NAN, f64::NAN], None).unwrap();
    let mean = all_nan.ewm().alpha(0.5).unwrap().mean().unwrap();
    assert!(mean.values().iter().all(|v| v.is_nan()));
}

// ---------------------------------------------------------------------
// src/series/window.rs: Rolling closed / center / apply
// ---------------------------------------------------------------------

#[test]
fn rolling_honors_window_closed() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], None).unwrap();

    let right = s
        .rolling(3)
        .unwrap()
        .min_periods(1)
        .closed(WindowClosed::Right)
        .sum()
        .unwrap();
    let left = s
        .rolling(3)
        .unwrap()
        .min_periods(1)
        .closed(WindowClosed::Left)
        .sum()
        .unwrap();
    let both = s
        .rolling(3)
        .unwrap()
        .min_periods(1)
        .closed(WindowClosed::Both)
        .sum()
        .unwrap();
    let neither = s
        .rolling(3)
        .unwrap()
        .min_periods(1)
        .closed(WindowClosed::Neither)
        .sum()
        .unwrap();

    // Position 3 (value 4.0): the base trailing range covers indices [1,3]
    // (values 2,3,4). `closed` was previously accepted but never read, so
    // all four of these were identical.
    assert!((right.values()[3] - 9.0).abs() < 1e-12); // 2+3+4 (unshifted base range)
    assert!((left.values()[3] - 6.0).abs() < 1e-12); // 1+2+3 (range shifted back one)
    assert!((both.values()[3] - 10.0).abs() < 1e-12); // 1+2+3+4 (range grown by one)
    assert!((neither.values()[3] - 5.0).abs() < 1e-12); // 2+3 (range shrunk by one)
}

#[test]
fn rolling_center_reports_na_at_edges_not_left_clamped() {
    let s = Series::new((1..=7).map(|i| i as f64).collect::<Vec<_>>(), None).unwrap();
    let centered = s.rolling(3).unwrap().center(true).mean().unwrap();

    // pandas: a centered window=3 rolling mean over [1..7] is
    // [NaN, 2, 3, 4, 5, 6, NaN]. The previous implementation clamped the
    // window's start to 0 near the front of the series instead, silently
    // pulling in extra right-side context and reporting a value (2.0) at
    // position 0 instead of NA.
    assert!(centered.values()[0].is_nan());
    assert!((centered.values()[1] - 2.0).abs() < 1e-12);
    assert!((centered.values()[2] - 3.0).abs() < 1e-12);
    assert!((centered.values()[3] - 4.0).abs() < 1e-12); // truly centered: mean([3,4,5])
    assert!((centered.values()[4] - 5.0).abs() < 1e-12);
    assert!((centered.values()[5] - 6.0).abs() < 1e-12);
    assert!(centered.values()[6].is_nan());
}

#[test]
fn rolling_and_expanding_apply_return_full_length_aligned_series() {
    let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], None).unwrap();

    let rolled = s
        .rolling(3)
        .unwrap()
        .apply(|w| w.iter().sum::<f64>())
        .unwrap();
    // Previously `apply` dropped every sub-min_periods row, returning a
    // *shorter* series with no way to tell which input row a given result
    // came from.
    assert_eq!(rolled.len(), 5);
    assert!(rolled.values()[0].is_none());
    assert!(rolled.values()[1].is_none());
    assert_eq!(rolled.values()[2], Some(6.0));
    assert_eq!(rolled.values()[3], Some(9.0));
    assert_eq!(rolled.values()[4], Some(12.0));

    let expanded = s
        .expanding(2)
        .unwrap()
        .apply(|w| w.iter().sum::<f64>())
        .unwrap();
    assert_eq!(expanded.len(), 5);
    assert!(expanded.values()[0].is_none());
    assert_eq!(expanded.values()[1], Some(3.0));
    assert_eq!(expanded.values()[4], Some(15.0));
}

#[test]
fn rolling_empty_window_never_panics_and_reports_na() {
    let s = Series::new(vec![1.0, 2.0, 3.0], None).unwrap();

    // window_size=1, closed=Neither, min_periods=0: the base range at
    // every position is empty. `min_periods == 0` must not be read as
    // "call the aggregator on an empty slice".
    let r = s
        .rolling(1)
        .unwrap()
        .min_periods(0)
        .closed(WindowClosed::Neither);
    // median()/quantile() would previously underflow-panic indexing an
    // empty sorted slice; min() would leak f64::INFINITY as fabricated data.
    assert!(r.median().unwrap().values()[0].is_nan());
    assert!(r.quantile(0.5).unwrap().values()[0].is_nan());
    assert!(r.min().unwrap().values()[0].is_nan());

    // An all-NaN window with min_periods=0 is the other way an empty
    // window is reachable (every value filtered out as missing).
    let with_nan = Series::new(vec![f64::NAN, f64::NAN], None).unwrap();
    let r2 = with_nan.rolling(2).unwrap().min_periods(0);
    assert!(r2.median().unwrap().values()[1].is_nan());
    assert!(r2.max().unwrap().values()[1].is_nan());
}

// ---------------------------------------------------------------------
// src/series/datetime_accessor.rs: round / floor / ceil
// ---------------------------------------------------------------------

#[test]
fn datetime_round_actually_rounds_not_truncates() {
    use chrono::{NaiveDate, Timelike};

    let past_half = NaiveDate::from_ymd_opt(2024, 3, 10)
        .unwrap()
        .and_hms_opt(14, 30, 45)
        .unwrap();
    let s = Series::new(vec![past_half], None).unwrap();
    let rounded = s.dt().unwrap().round("H").unwrap();
    // 14:30:45 is past the 14:30:00 midpoint, so it rounds *up* to 15:00 --
    // truncation (the previous behavior) would give 14:00.
    assert_eq!(rounded.values()[0].hour(), 15);
    assert_eq!(rounded.values()[0].minute(), 0);

    let before_half = NaiveDate::from_ymd_opt(2024, 3, 10)
        .unwrap()
        .and_hms_opt(14, 20, 0)
        .unwrap();
    let s2 = Series::new(vec![before_half], None).unwrap();
    let rounded2 = s2.dt().unwrap().round("H").unwrap();
    assert_eq!(rounded2.values()[0].hour(), 14); // closer to 14:00

    let floored = s.dt().unwrap().floor("H").unwrap();
    assert_eq!(floored.values()[0].hour(), 14);
    assert_eq!(floored.values()[0].minute(), 0);

    let ceiled = s.dt().unwrap().ceil("H").unwrap();
    assert_eq!(ceiled.values()[0].hour(), 15);
    assert_eq!(ceiled.values()[0].minute(), 0);
}

#[test]
fn datetime_round_errors_on_unknown_frequency_instead_of_silent_no_op() {
    let dt = chrono::NaiveDate::from_ymd_opt(2024, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let s = Series::new(vec![dt], None).unwrap();

    assert!(s.dt().unwrap().round("not_a_freq").is_err());
    assert!(s.dt().unwrap().floor("Q").is_err());
    assert!(s.dt().unwrap().ceil("0min").is_err()); // zero-width interval also rejected, not a divide-by-zero panic
}

// ---------------------------------------------------------------------
// src/series/string_accessor.rs: split(expand=true)
// ---------------------------------------------------------------------

#[test]
fn string_split_expand_true_errors_instead_of_silently_collapsing() {
    let s = Series::new(vec!["a,b,c".to_string()], None).unwrap();
    let accessor = s.str().unwrap();

    assert!(accessor.split(",", None, false).is_ok());
    // expand=true cannot be honored (Series<String> has no way to return
    // multiple columns); it must error rather than silently falling back
    // to the same collapsed "[a, b, c]" string that expand=false produces,
    // which looks like a success with no way to tell the two apart.
    assert!(accessor.split(",", None, true).is_err());
}

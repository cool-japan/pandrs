//! Regression tests for the `base-groupbymod` fixes:
//! `src/dataframe/base.rs` and `src/groupby/mod.rs`.
//!
//! Each test is labeled with the item number from the task list it covers.

use std::collections::HashMap;

use chrono::{NaiveDate, TimeZone, Utc};
use pandrs::dataframe::groupby::{AggFunc, NamedAgg};
use pandrs::{DataFrame, DataFrameIndex, Index, Series};

fn series_str(values: &[&str]) -> Series<String> {
    Series::new(values.iter().map(|s| s.to_string()).collect(), None).unwrap()
}

// ---------------------------------------------------------------------
// (1) concat_rows: real row concatenation, aligned by name, NA for
// columns present on only one side.
// ---------------------------------------------------------------------

#[test]
fn concat_rows_matching_types_concatenates_real_values() {
    let mut df1 = DataFrame::new();
    df1.add_column("x".to_string(), Series::new(vec![1i64, 2], None).unwrap())
        .unwrap();
    df1.add_column(
        "y".to_string(),
        Series::new(vec![1.5f64, 2.5], None).unwrap(),
    )
    .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column("x".to_string(), Series::new(vec![3i64, 4], None).unwrap())
        .unwrap();
    df2.add_column(
        "y".to_string(),
        Series::new(vec![3.5f64, 4.5], None).unwrap(),
    )
    .unwrap();

    let combined = df1.concat_rows(&df2).unwrap();
    assert_eq!(combined.row_count(), 4);
    // Types are preserved (not stringified): i64 stays convertible as numeric.
    assert_eq!(
        combined.get_column_numeric_values("x").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        combined.get_column_numeric_values("y").unwrap(),
        vec![1.5, 2.5, 3.5, 4.5]
    );
}

#[test]
fn concat_rows_numeric_only_column_is_nan_filled() {
    let mut df1 = DataFrame::new();
    df1.add_column("x".to_string(), Series::new(vec![1i64, 2], None).unwrap())
        .unwrap();
    df1.add_column(
        "only_in_left".to_string(),
        Series::new(vec![10.0f64, 20.0], None).unwrap(),
    )
    .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column(
        "x".to_string(),
        Series::new(vec![3i64, 4, 5], None).unwrap(),
    )
    .unwrap();

    let combined = df1.concat_rows(&df2).unwrap();
    assert_eq!(combined.row_count(), 5);
    let only_in_left = combined.get_column_numeric_values("only_in_left").unwrap();
    assert_eq!(&only_in_left[0..2], &[10.0, 20.0]);
    assert!(only_in_left[2].is_nan());
    assert!(only_in_left[3].is_nan());
    assert!(only_in_left[4].is_nan());
}

#[test]
fn concat_rows_mismatched_type_falls_back_to_string_not_dropped() {
    let mut df1 = DataFrame::new();
    df1.add_column("k".to_string(), Series::new(vec![1i64, 2], None).unwrap())
        .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column("k".to_string(), series_str(&["a", "b"]))
        .unwrap();

    let combined = df1.concat_rows(&df2).unwrap();
    assert_eq!(combined.row_count(), 4);
    assert_eq!(
        combined.get_column_string_values("k").unwrap(),
        vec!["1", "2", "a", "b"]
    );
}

#[test]
fn concat_rows_non_numeric_only_column_errors_instead_of_fabricating() {
    let mut df1 = DataFrame::new();
    df1.add_column("x".to_string(), Series::new(vec![1i64], None).unwrap())
        .unwrap();
    df1.add_column("only_in_left".to_string(), series_str(&["hello"]))
        .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column("x".to_string(), Series::new(vec![2i64], None).unwrap())
        .unwrap();

    // A String column present on only one side has no NA-capable
    // representation in this DataFrame's column model, so this must error
    // rather than silently filling "" (indistinguishable from real data).
    assert!(df1.concat_rows(&df2).is_err());
}

// ---------------------------------------------------------------------
// (2) group_by: delegates to the real GroupByExt machinery.
// ---------------------------------------------------------------------

#[test]
fn group_by_delegates_to_real_groupby_machinery() {
    let mut df = DataFrame::new();
    df.add_column("cat".to_string(), series_str(&["x", "y", "x"]))
        .unwrap();
    df.add_column(
        "val".to_string(),
        Series::new(vec![10i64, 20, 30], None).unwrap(),
    )
    .unwrap();

    let grouped = df.group_by("cat").unwrap();
    assert_eq!(grouped.ngroups(), 2);

    let summed = grouped.sum("val").unwrap();
    assert_eq!(summed.row_count(), 2);
    let cats = summed.get_column_string_values("cat").unwrap();
    let sums = summed.get_column_string_values("val_sum").unwrap();
    let mut pairs: Vec<(String, String)> = cats.into_iter().zip(sums).collect();
    pairs.sort();
    assert_eq!(
        pairs,
        vec![
            ("x".to_string(), "40".to_string()),
            ("y".to_string(), "20".to_string()),
        ]
    );

    // Also exercise the general .agg() path with an explicit NamedAgg.
    let agg = grouped
        .agg(vec![NamedAgg::new(
            "val".to_string(),
            AggFunc::Mean,
            "val_mean".to_string(),
        )])
        .unwrap();
    assert_eq!(agg.row_count(), 2);
}

// ---------------------------------------------------------------------
// (3a) corr_matrix: a real correlation matrix, not Ok(()).
// ---------------------------------------------------------------------

#[test]
fn corr_matrix_returns_real_pearson_correlations() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "b".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0], None).unwrap(), // perfectly correlated with a
    )
    .unwrap();
    df.add_column(
        "c".to_string(),
        Series::new(vec![4.0, 3.0, 2.0, 1.0], None).unwrap(), // perfectly anti-correlated with a
    )
    .unwrap();

    let corr = df.corr_matrix(&["a", "b", "c"]).unwrap();
    assert_eq!(corr.row_count(), 3);

    let a_col = corr.get_column_numeric_values("a").unwrap();
    let b_col = corr.get_column_numeric_values("b").unwrap();
    let c_col = corr.get_column_numeric_values("c").unwrap();

    // Diagonal is 1.0.
    assert!((a_col[0] - 1.0).abs() < 1e-9);
    assert!((b_col[1] - 1.0).abs() < 1e-9);
    assert!((c_col[2] - 1.0).abs() < 1e-9);
    // corr(a, b) == 1.0, corr(a, c) == -1.0.
    assert!((b_col[0] - 1.0).abs() < 1e-9);
    assert!((c_col[0] - (-1.0)).abs() < 1e-9);

    // Indexed by variable name, matching pandas' DataFrame.corr() shape.
    if let DataFrameIndex::Simple(idx) = corr.get_index() {
        assert_eq!(
            idx.values(),
            &["a".to_string(), "b".to_string(), "c".to_string()]
        );
    } else {
        panic!("expected a simple index on the correlation matrix");
    }
}

// ---------------------------------------------------------------------
// (3b) gpu_accelerate: honestly NotImplemented instead of Ok(self.clone()).
// ---------------------------------------------------------------------

#[test]
fn gpu_accelerate_is_honestly_not_implemented() {
    let mut df = DataFrame::new();
    df.add_column("x".to_string(), Series::new(vec![1i64], None).unwrap())
        .unwrap();

    // Previously this always returned `Ok(self.clone())` regardless of
    // whether any GPU acceleration actually happened.
    let result = df.gpu_accelerate();
    assert!(result.is_err());
}

// ---------------------------------------------------------------------
// (4) value_counts: labeled and deterministic.
// ---------------------------------------------------------------------

#[test]
fn value_counts_is_labeled_and_deterministically_ordered() {
    let mut df = DataFrame::new();
    df.add_column(
        "region".to_string(),
        series_str(&["Tokyo", "Osaka", "Tokyo", "Nagoya", "Osaka"]),
    )
    .unwrap();

    let counts = df.value_counts("region").unwrap();
    assert_eq!(counts.row_count(), 3);
    assert!(counts.contains_column("value"));
    assert!(counts.contains_column("count"));

    // Sorted by count desc, then value asc: Tokyo/Osaka tie at 2 ("Osaka" <
    // "Tokyo"), then Nagoya at 1.
    assert_eq!(
        counts.get_column_string_values("value").unwrap(),
        vec!["Osaka", "Tokyo", "Nagoya"]
    );
    assert_eq!(
        counts.get_column_string_values("count").unwrap(),
        vec!["2", "2", "1"]
    );

    // Repeated calls are stable (no HashMap-order flakiness).
    let counts2 = df.value_counts("region").unwrap();
    assert_eq!(
        counts.get_column_string_values("value").unwrap(),
        counts2.get_column_string_values("value").unwrap()
    );
}

// ---------------------------------------------------------------------
// (5) get_column_string_values: real arms for dates/wider integer types,
// honest error for genuinely unsupported types.
// ---------------------------------------------------------------------

#[test]
fn get_column_string_values_renders_dates_and_wider_ints() {
    let mut df = DataFrame::new();
    df.add_column(
        "d".to_string(),
        Series::new(vec![NaiveDate::from_ymd_opt(2024, 1, 5).unwrap()], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "dt".to_string(),
        Series::new(
            vec![NaiveDate::from_ymd_opt(2024, 1, 5)
                .unwrap()
                .and_hms_opt(9, 30, 0)
                .unwrap()],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    df.add_column(
        "ts_utc".to_string(),
        Series::new(vec![Utc.timestamp_opt(0, 0).unwrap()], None).unwrap(),
    )
    .unwrap();
    df.add_column("u8col".to_string(), Series::new(vec![7u8], None).unwrap())
        .unwrap();
    df.add_column(
        "u64col".to_string(),
        Series::new(vec![18_000_000_000u64], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "usize_col".to_string(),
        Series::new(vec![42usize], None).unwrap(),
    )
    .unwrap();
    df.add_column("i8col".to_string(), Series::new(vec![-5i8], None).unwrap())
        .unwrap();

    assert_eq!(
        df.get_column_string_values("d").unwrap(),
        vec!["2024-01-05"]
    );
    assert_eq!(
        df.get_column_string_values("dt").unwrap(),
        vec!["2024-01-05 09:30:00"]
    );
    assert_eq!(
        df.get_column_string_values("ts_utc").unwrap(),
        vec!["1970-01-01 00:00:00 UTC"]
    );
    assert_eq!(df.get_column_string_values("u8col").unwrap(), vec!["7"]);
    assert_eq!(
        df.get_column_string_values("u64col").unwrap(),
        vec!["18000000000"]
    );
    assert_eq!(
        df.get_column_string_values("usize_col").unwrap(),
        vec!["42"]
    );
    assert_eq!(df.get_column_string_values("i8col").unwrap(), vec!["-5"]);
}

#[test]
fn get_column_string_values_errors_instead_of_fabricating_for_unsupported_type() {
    let mut df = DataFrame::new();
    // `Vec<u8>` satisfies the bounds required to be stored as a column but
    // is not one of the types this DataFrame knows how to render as a
    // string; previously this silently produced
    // "unsupported_type_{col}_{i}" placeholder strings.
    df.add_column(
        "blob".to_string(),
        Series::new(vec![vec![1u8, 2, 3]], None).unwrap(),
    )
    .unwrap();

    let result = df.get_column_string_values("blob");
    assert!(result.is_err());
    let message = format!("{}", result.unwrap_err());
    assert!(!message.contains("unsupported_type_blob_0"));
}

// ---------------------------------------------------------------------
// (6) get_column_numeric_values: i32/f32/bool arms now work (hoisted
// downcast, not attempted per element).
// ---------------------------------------------------------------------

#[test]
fn get_column_numeric_values_supports_i32_f32_bool() {
    let mut df = DataFrame::new();
    df.add_column(
        "i32col".to_string(),
        Series::new(vec![1i32, 2, 3], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "f32col".to_string(),
        Series::new(vec![1.5f32, 2.5, 3.5], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "boolcol".to_string(),
        Series::new(vec![true, false, true], None).unwrap(),
    )
    .unwrap();

    assert_eq!(
        df.get_column_numeric_values("i32col").unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        df.get_column_numeric_values("f32col").unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        df.get_column_numeric_values("boolcol").unwrap(),
        vec![1.0, 0.0, 1.0]
    );
}

// ---------------------------------------------------------------------
// (7) add_column: length check no longer bypassed once row_count is
// established by set_index on an otherwise-empty DataFrame.
// ---------------------------------------------------------------------

#[test]
fn add_column_rejects_mismatched_length_after_set_index_on_empty_frame() {
    let mut df = DataFrame::new();
    let idx = Index::new((0..5).map(|i| format!("r{}", i)).collect::<Vec<String>>()).unwrap();
    df.set_index(idx).unwrap();
    assert_eq!(df.row_count(), 5);
    assert_eq!(df.column_count(), 0);

    // A column shorter than the index-established row count must be
    // rejected, not silently accepted (this used to slip through because
    // the check was gated on `columns.is_empty()` rather than
    // `row_count != 0`).
    let short = Series::new(vec![1i64, 2, 3], None).unwrap();
    assert!(df.add_column("x".to_string(), short).is_err());
    assert_eq!(df.column_count(), 0);

    let matching = Series::new(vec![1i64, 2, 3, 4, 5], None).unwrap();
    df.add_column("x".to_string(), matching).unwrap();
    assert_eq!(df.row_count(), 5);
    assert_eq!(df.column_count(), 1);
}

#[test]
fn add_column_rejects_mismatched_length_when_first_column_is_zero_length() {
    // Mirror-image of the set_index case: a column can exist while
    // `row_count` is still 0 (because that first column is itself
    // zero-length). A naive fix that only checks `row_count != 0` would
    // let a *second*, differently-sized column redefine `row_count` out
    // from under the first one, corrupting it silently.
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::<String>::new(vec![], None).unwrap(),
    )
    .unwrap();
    assert_eq!(df.row_count(), 0);
    assert_eq!(df.column_count(), 1);

    let nonzero = Series::new(vec![1i64, 2, 3], None).unwrap();
    assert!(df.add_column("b".to_string(), nonzero).is_err());
    // Rejected: column "a" must not be left at length 0 while row_count
    // silently becomes 3.
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 0);

    // A matching (also zero-length) column is still accepted.
    let matching = Series::<String>::new(vec![], None).unwrap();
    df.add_column("b".to_string(), matching).unwrap();
    assert_eq!(df.column_count(), 2);
    assert_eq!(df.row_count(), 0);
}

// ---------------------------------------------------------------------
// (8) from_map: stores the index, rejects ragged columns, deterministic
// column order.
// ---------------------------------------------------------------------

#[test]
fn from_map_is_deterministic_and_validates_and_stores_index() {
    let mut data = HashMap::new();
    data.insert("zeta".to_string(), vec!["1".to_string(), "2".to_string()]);
    data.insert("alpha".to_string(), vec!["x".to_string(), "y".to_string()]);
    data.insert("mid".to_string(), vec!["a".to_string(), "b".to_string()]);

    let df = DataFrame::from_map(data, None).unwrap();
    // Deterministic (sorted) column order, not raw HashMap iteration order.
    assert_eq!(
        df.column_names(),
        &["alpha".to_string(), "mid".to_string(), "zeta".to_string()]
    );

    // Ragged columns are rejected.
    let mut ragged = HashMap::new();
    ragged.insert("a".to_string(), vec!["1".to_string(), "2".to_string()]);
    ragged.insert("b".to_string(), vec!["1".to_string()]);
    assert!(DataFrame::from_map(ragged, None).is_err());

    // The provided index is actually stored (previously discarded).
    let idx = Index::new(vec!["r0".to_string(), "r1".to_string()]).unwrap();
    let mut data2 = HashMap::new();
    data2.insert("a".to_string(), vec!["1".to_string(), "2".to_string()]);
    let df2 = DataFrame::from_map(data2, Some(idx)).unwrap();
    match df2.get_index() {
        DataFrameIndex::Simple(stored) => {
            assert_eq!(stored.values(), &["r0".to_string(), "r1".to_string()]);
        }
        _ => panic!("expected a simple index to be stored"),
    }

    // A column length that disagrees with the provided index is rejected.
    let idx2 = Index::new(vec!["r0".to_string(), "r1".to_string(), "r2".to_string()]).unwrap();
    let mut data3 = HashMap::new();
    data3.insert("a".to_string(), vec!["1".to_string(), "2".to_string()]);
    assert!(DataFrame::from_map(data3, Some(idx2)).is_err());
}

// ---------------------------------------------------------------------
// (9) from_csv_reader: headerless mode no longer drops the first row.
// ---------------------------------------------------------------------

#[test]
fn from_csv_reader_headerless_keeps_first_row() {
    let csv_data = "1,2,3\n4,5,6\n7,8,9\n";
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(csv_data.as_bytes());

    let df = DataFrame::from_csv_reader(&mut reader, false).unwrap();
    // Previously the peeked first record was consumed and lost, so a
    // 3-row input read back as only 2 rows.
    assert_eq!(df.row_count(), 3);
    assert_eq!(
        df.get_column_string_values("column_0").unwrap(),
        vec!["1", "4", "7"]
    );
    assert_eq!(
        df.get_column_string_values("column_2").unwrap(),
        vec!["3", "6", "9"]
    );
}

// ---------------------------------------------------------------------
// (10) select_columns: dtypes preserved (no numeric re-inference), index
// preserved.
// ---------------------------------------------------------------------

#[test]
fn select_columns_preserves_dtypes_and_index() {
    let mut df = DataFrame::new();
    df.add_column("a".to_string(), Series::new(vec![1i64, 2], None).unwrap())
        .unwrap();
    df.add_column("b".to_string(), series_str(&["007", "008"]))
        .unwrap();
    df.add_column(
        "c".to_string(),
        Series::new(vec![true, false], None).unwrap(),
    )
    .unwrap();
    let idx = Index::new(vec!["r0".to_string(), "r1".to_string()]).unwrap();
    df.set_index(idx).unwrap();

    let selected = df.select_columns(&["a", "b", "c"]).unwrap();

    // "b" keeps its exact String values ("007"), not a re-inferred f64 (7.0).
    assert_eq!(
        selected.get_column_string_values("b").unwrap(),
        vec!["007", "008"]
    );
    // "c" is still physically a bool column (numeric conversion via the
    // bool arm succeeds with 1.0/0.0), not stringified to "true"/"false".
    assert_eq!(
        selected.get_column_numeric_values("c").unwrap(),
        vec![1.0, 0.0]
    );
    // "a" is still i64-backed.
    assert!(selected.is_numeric_column("a"));

    match selected.get_index() {
        DataFrameIndex::Simple(stored) => {
            assert_eq!(stored.values(), &["r0".to_string(), "r1".to_string()]);
        }
        _ => panic!("expected the row index to be preserved"),
    }
}

// ---------------------------------------------------------------------
// (11) head/tail: real DataFrame subsets (not a Result<String>), plus a
// separate head_string for display output.
// ---------------------------------------------------------------------

#[test]
fn head_and_tail_preserve_types_and_slice_the_index() {
    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2, 3, 4], None).unwrap(),
    )
    .unwrap();
    let idx = Index::new(
        vec!["r0", "r1", "r2", "r3"]
            .into_iter()
            .map(String::from)
            .collect(),
    )
    .unwrap();
    df.set_index(idx).unwrap();

    let head = df.head(2).unwrap();
    assert_eq!(head.row_count(), 2);
    assert_eq!(head.get_column_numeric_values("v").unwrap(), vec![1.0, 2.0]);
    match head.get_index() {
        DataFrameIndex::Simple(i) => {
            assert_eq!(i.values(), &["r0".to_string(), "r1".to_string()])
        }
        _ => panic!("expected simple index"),
    }

    let tail = df.tail(2).unwrap();
    assert_eq!(tail.row_count(), 2);
    assert_eq!(tail.get_column_numeric_values("v").unwrap(), vec![3.0, 4.0]);
    match tail.get_index() {
        DataFrameIndex::Simple(i) => {
            assert_eq!(i.values(), &["r2".to_string(), "r3".to_string()])
        }
        _ => panic!("expected simple index"),
    }

    // head_string still provides the previous tab-separated display output.
    let head_str = df.head_string(2).unwrap();
    assert!(head_str.contains("v"));
    assert!(head_str.contains('1'));
}

#[test]
fn head_preserves_i32_f32_and_falls_back_to_string_for_dates() {
    // sample() (which head/tail/filter delegate to) previously coerced any
    // non-f64/i64/String/bool column to f64 via get_column_numeric_values,
    // which silently downgraded i32/f32 and hard-errored on date columns.
    let mut df = DataFrame::new();
    df.add_column(
        "i32col".to_string(),
        Series::new(vec![1i32, 2, 3], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "f32col".to_string(),
        Series::new(vec![1.5f32, 2.5, 3.5], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "d".to_string(),
        Series::new(
            vec![
                NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                NaiveDate::from_ymd_opt(2024, 1, 3).unwrap(),
            ],
            None,
        )
        .unwrap(),
    )
    .unwrap();

    let head = df.head(2).unwrap();
    assert_eq!(head.row_count(), 2);
    // i32/f32 stay physically i32/f32 (numeric conversion via those arms
    // succeeds), not silently downgraded through the generic fallback.
    assert!(head.is_numeric_column("i32col"));
    assert!(head.is_numeric_column("f32col"));
    assert_eq!(
        head.get_column_numeric_values("i32col").unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        head.get_column_numeric_values("f32col").unwrap(),
        vec![1.5, 2.5]
    );
    // The date column round-trips through its string representation
    // instead of erroring.
    assert_eq!(
        head.get_column_string_values("d").unwrap(),
        vec!["2024-01-01", "2024-01-02"]
    );
}

// ---------------------------------------------------------------------
// (12) groupby/mod.rs DataFrameGroupBy::aggregate: fetches real column
// data instead of a hardcoded empty Vec, and is deterministically ordered.
// ---------------------------------------------------------------------

#[test]
fn groupby_mod_aggregate_uses_real_column_data() {
    let mut df = DataFrame::new();
    df.add_column("cat".to_string(), series_str(&["a", "b", "a", "b"]))
        .unwrap();
    df.add_column(
        "val".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0], None).unwrap(),
    )
    .unwrap();

    let keys = vec![
        "a".to_string(),
        "b".to_string(),
        "a".to_string(),
        "b".to_string(),
    ];
    let gb = pandrs::groupby::DataFrameGroupBy::new(keys, &df, "cat".to_string()).unwrap();

    let summed = gb.aggregate("val", "sum").unwrap();
    assert_eq!(summed.row_count(), 2);
    // group "a": rows 0, 2 -> 1.0 + 3.0 = 4.0; group "b": rows 1, 3 -> 2.0 + 4.0 = 6.0.
    // Deterministic order: sorted by the group key's Debug string
    // (`"\"a\""` < `"\"b\""`).
    assert_eq!(
        summed.get_column_string_values("group_key").unwrap(),
        vec!["\"a\"", "\"b\""]
    );
    assert_eq!(
        summed.get_column_string_values("val_sum").unwrap(),
        vec!["4", "6"]
    );

    let counted = gb.aggregate("val", "count").unwrap();
    assert_eq!(
        counted.get_column_string_values("val_count").unwrap(),
        vec!["2", "2"]
    );

    // An unsupported aggregation function is a real error, not a silent "0.0".
    assert!(gb.aggregate("val", "not_a_real_function").is_err());

    // size_as_df is also deterministically ordered now.
    let sizes = gb.size_as_df().unwrap();
    assert_eq!(
        sizes.get_column_string_values("group_key").unwrap(),
        vec!["\"a\"", "\"b\""]
    );
    assert_eq!(
        sizes.get_column_string_values("size").unwrap(),
        vec!["2", "2"]
    );
}

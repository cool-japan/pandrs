//! Regression tests for the transform-multiindex fixes:
//!
//! - `TransformExt::{melt, stack, unstack, conditional_aggregate, concat}`
//!   (`src/dataframe/transform.rs`) used to return literal "dummy" /
//!   hardcoded data regardless of input.
//! - `MultiIndexDataFrameBuilder::build` (`src/dataframe/multi_index_results.rs`)
//!   used to fill index columns with the literal string "placeholder".
//! - `utils::merge_multi_index_dataframes` used to discard every frame but
//!   the first.
//! - `MultiIndexDataFrame::select_columns_by_level` used to silently drop
//!   any column that was not `Series<String>` (numeric columns included).
//! - `MultiIndexDataFrame::pivot_columns` used to reorder only the
//!   `column_index` metadata, leaving the underlying `DataFrame`'s real
//!   column names unchanged (so the "reordered" columns were unreadable
//!   under their new hierarchical name).
//!
//! Every fixture below intentionally uses different category names / values
//! than the old fabricated stubs (`Food`/`Electronics`/`Clothing`,
//! `1000`/`1500`/`1200`, ids `1..4`, values `a..d`) so that a regression back
//! to the hardcoded output would make these tests fail, not accidentally
//! pass.
#![allow(clippy::result_large_err)]

use std::collections::HashMap;

use pandrs::dataframe::multi_index_utils;
use pandrs::dataframe::TransformExt;
use pandrs::dataframe::{MultiIndexColumn, MultiIndexDataFrame, MultiIndexDataFrameBuilder};
use pandrs::{DataFrame, Index, MeltOptions, Series, StackOptions, UnstackOptions, NA};

fn wide_fixture() -> DataFrame {
    // Deliberately different id/category vocabulary than the old fabricated
    // stubs, and column names distinct from the melted var/value names.
    let mut df = DataFrame::new();
    df.add_column(
        "sku".to_string(),
        Series::new(vec!["A1", "A2", "A3"], Some("sku".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "region".to_string(),
        Series::new(vec!["North", "South", "East"], Some("region".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "Q1".to_string(),
        Series::new(vec!["11", "22", "33"], Some("Q1".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "Q2".to_string(),
        Series::new(vec!["44", "55", "66"], Some("Q2".to_string())).unwrap(),
    )
    .unwrap();
    df
}

// ---------------------------------------------------------------------
// melt
// ---------------------------------------------------------------------

#[test]
fn melt_honors_options_and_values_survive_round_trip() {
    let df = wide_fixture();
    let options = MeltOptions {
        id_vars: Some(vec!["sku".to_string(), "region".to_string()]),
        value_vars: Some(vec!["Q1".to_string(), "Q2".to_string()]),
        var_name: Some("quarter".to_string()),
        value_name: Some("amount".to_string()),
    };

    let melted = df.melt(&options).unwrap();

    // 3 rows x 2 value columns = 6 rows; var/value names honored.
    assert_eq!(melted.row_count(), 6);
    assert_eq!(melted.column_count(), 4);
    for name in ["sku", "region", "quarter", "amount"] {
        assert!(melted.contains_column(name), "missing column {name}");
    }

    let sku = melted.get_column_string_values("sku").unwrap();
    let region = melted.get_column_string_values("region").unwrap();
    let quarter = melted.get_column_string_values("quarter").unwrap();
    let amount = melted.get_column_string_values("amount").unwrap();

    // Expected (sku, quarter) -> amount, taken straight from the fixture.
    let mut expected: HashMap<(String, String), String> = HashMap::new();
    expected.insert(("A1".into(), "Q1".into()), "11".into());
    expected.insert(("A2".into(), "Q1".into()), "22".into());
    expected.insert(("A3".into(), "Q1".into()), "33".into());
    expected.insert(("A1".into(), "Q2".into()), "44".into());
    expected.insert(("A2".into(), "Q2".into()), "55".into());
    expected.insert(("A3".into(), "Q2".into()), "66".into());

    let mut expected_region: HashMap<String, String> = HashMap::new();
    expected_region.insert("A1".into(), "North".into());
    expected_region.insert("A2".into(), "South".into());
    expected_region.insert("A3".into(), "East".into());

    for i in 0..melted.row_count() {
        let key = (sku[i].clone(), quarter[i].clone());
        assert_eq!(
            amount[i],
            *expected
                .get(&key)
                .unwrap_or_else(|| panic!("unexpected row {key:?}")),
            "wrong melted value for {key:?}"
        );
        assert_eq!(region[i], expected_region[&sku[i]]);
    }
}

#[test]
fn melt_defaults_id_vars_to_empty_and_value_vars_to_all_columns() {
    let df = wide_fixture();
    let melted = df.melt(&MeltOptions::default()).unwrap();

    // No id_vars requested: every one of the 4 original columns becomes a
    // melted value column, each contributing 3 rows.
    assert_eq!(melted.row_count(), 12);
    assert_eq!(melted.column_count(), 2);
    assert!(melted.contains_column("variable"));
    assert!(melted.contains_column("value"));

    let variables = melted.get_column_string_values("variable").unwrap();
    let mut seen: Vec<&str> = variables.iter().map(String::as_str).collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen, vec!["Q1", "Q2", "region", "sku"]);
}

#[test]
fn melt_rejects_nonexistent_value_var_instead_of_nan_filling() {
    let df = wide_fixture();
    let options = MeltOptions {
        id_vars: Some(vec!["sku".to_string()]),
        value_vars: Some(vec!["does_not_exist".to_string()]),
        var_name: None,
        value_name: None,
    };
    // A real implementation must reject a nonexistent value column outright
    // rather than silently NaN-filling it (the reference PCE behavior this
    // deliberately improves on).
    assert!(df.melt(&options).is_err());
}

// ---------------------------------------------------------------------
// stack
// ---------------------------------------------------------------------

#[test]
fn stack_produces_real_row_major_data_and_respects_dropna() {
    let mut df = DataFrame::new();
    df.add_column(
        "label".to_string(),
        Series::new(vec!["x", "y"], Some("label".to_string())).unwrap(),
    )
    .unwrap();
    // A genuinely numeric column with a real NaN, so `dropna` has something
    // honest to filter (never treating an empty string as NA).
    df.add_column(
        "score".to_string(),
        Series::new(vec![1.5f64, f64::NAN], Some("score".to_string())).unwrap(),
    )
    .unwrap();

    let stacked_keep = df
        .stack(&StackOptions {
            columns: None,
            var_name: Some("field".to_string()),
            value_name: Some("val".to_string()),
            dropna: false,
        })
        .unwrap();
    // 2 rows x 2 columns, nothing dropped.
    assert_eq!(stacked_keep.row_count(), 4);
    let vals = stacked_keep.get_column_string_values("val").unwrap();
    assert!(
        vals.iter().any(|v| v == "NaN"),
        "NaN cell should survive as text: {vals:?}"
    );

    let stacked_drop = df
        .stack(&StackOptions {
            columns: None,
            var_name: Some("field".to_string()),
            value_name: Some("val".to_string()),
            dropna: true,
        })
        .unwrap();
    // Only the (y, score=NaN) cell should be dropped: 4 - 1 = 3 rows.
    assert_eq!(stacked_drop.row_count(), 3);
    let fields = stacked_drop.get_column_string_values("field").unwrap();
    // "id" holds the *original row's* positional label ("0" for "x", "1"
    // for "y"), not the "label" column's own text -- that column got
    // stacked away into rows, so row identity is only recoverable by
    // position here.
    let ids = stacked_drop.get_column_string_values("id").unwrap();
    for (id, field) in ids.iter().zip(fields.iter()) {
        assert!(
            !(id == "1" && field == "score"),
            "NaN row (original row 1's score) should have been dropped"
        );
    }
    // The non-NaN score (original row 0) must still be present.
    assert!(
        ids.iter()
            .zip(fields.iter())
            .any(|(id, field)| id == "0" && field == "score"),
        "non-NaN score row must survive dropna"
    );
    // The non-numeric "label" column is never NA, dropna or not: both of
    // its stacked rows must still be present.
    let label_rows = fields.iter().filter(|f| f.as_str() == "label").count();
    assert_eq!(label_rows, 2);
}

#[test]
fn stack_rejects_unknown_column_selection() {
    let df = wide_fixture();
    let options = StackOptions {
        columns: Some(vec!["not_a_column".to_string()]),
        var_name: None,
        value_name: None,
        dropna: false,
    };
    assert!(df.stack(&options).is_err());
}

// ---------------------------------------------------------------------
// unstack
// ---------------------------------------------------------------------

#[test]
fn unstack_recovers_melted_values_round_trip() {
    let df = wide_fixture();
    let melt_options = MeltOptions {
        id_vars: Some(vec!["sku".to_string(), "region".to_string()]),
        value_vars: Some(vec!["Q1".to_string(), "Q2".to_string()]),
        var_name: Some("quarter".to_string()),
        value_name: Some("amount".to_string()),
    };
    let melted = df.melt(&melt_options).unwrap();

    let unstack_options = UnstackOptions {
        var_column: "quarter".to_string(),
        value_column: "amount".to_string(),
        index_columns: Some(vec!["sku".to_string(), "region".to_string()]),
        fill_value: None,
    };
    let wide_again = melted.unstack(&unstack_options).unwrap();

    assert_eq!(wide_again.row_count(), 3);
    assert!(wide_again.contains_column("sku"));
    assert!(wide_again.contains_column("region"));
    assert!(wide_again.contains_column("Q1"));
    assert!(wide_again.contains_column("Q2"));

    let sku = wide_again.get_column_string_values("sku").unwrap();
    let q1 = wide_again.get_column::<NA<String>>("Q1").unwrap();
    let q2 = wide_again.get_column::<NA<String>>("Q2").unwrap();

    let mut expected_q1: HashMap<&str, &str> = HashMap::new();
    expected_q1.insert("A1", "11");
    expected_q1.insert("A2", "22");
    expected_q1.insert("A3", "33");
    let mut expected_q2: HashMap<&str, &str> = HashMap::new();
    expected_q2.insert("A1", "44");
    expected_q2.insert("A2", "55");
    expected_q2.insert("A3", "66");

    for (i, key) in sku.iter().enumerate() {
        match q1.values()[i].clone() {
            NA::Value(v) => assert_eq!(v, expected_q1[key.as_str()]),
            NA::NA => panic!("Q1 unexpectedly missing for {key}"),
        }
        match q2.values()[i].clone() {
            NA::Value(v) => assert_eq!(v, expected_q2[key.as_str()]),
            NA::NA => panic!("Q2 unexpectedly missing for {key}"),
        }
    }
}

#[test]
fn unstack_fills_genuinely_missing_combinations_honestly() {
    // A ragged long frame: "b" has no "beta" row at all, so unstacking must
    // invent that cell -- and it must be a real NA marker, never "".
    let mut long = DataFrame::new();
    long.add_column(
        "grp".to_string(),
        Series::new(vec!["a", "a", "b"], Some("grp".to_string())).unwrap(),
    )
    .unwrap();
    long.add_column(
        "field".to_string(),
        Series::new(vec!["alpha", "beta", "alpha"], Some("field".to_string())).unwrap(),
    )
    .unwrap();
    long.add_column(
        "val".to_string(),
        Series::new(vec!["10", "20", "30"], Some("val".to_string())).unwrap(),
    )
    .unwrap();

    // Default: missing cell is an explicit `NA::NA`, not an empty string.
    let default_fill = long
        .unstack(&UnstackOptions {
            var_column: "field".to_string(),
            value_column: "val".to_string(),
            index_columns: Some(vec!["grp".to_string()]),
            fill_value: None,
        })
        .unwrap();
    let grp = default_fill.get_column_string_values("grp").unwrap();
    let beta = default_fill.get_column::<NA<String>>("beta").unwrap();
    let b_row = grp.iter().position(|g| g == "b").unwrap();
    assert!(
        beta.values()[b_row].is_na(),
        "missing cell must be NA, not fabricated"
    );

    // Explicit fill_value: the missing cell takes that value instead.
    let custom_fill = long
        .unstack(&UnstackOptions {
            var_column: "field".to_string(),
            value_column: "val".to_string(),
            index_columns: Some(vec!["grp".to_string()]),
            fill_value: Some(NA::Value("N/A".to_string())),
        })
        .unwrap();
    let grp2 = custom_fill.get_column_string_values("grp").unwrap();
    let beta2 = custom_fill.get_column::<NA<String>>("beta").unwrap();
    let b_row2 = grp2.iter().position(|g| g == "b").unwrap();
    match beta2.values()[b_row2].clone() {
        NA::Value(v) => assert_eq!(v, "N/A"),
        NA::NA => panic!("fill_value should have replaced the missing cell"),
    }
}

// ---------------------------------------------------------------------
// conditional_aggregate
// ---------------------------------------------------------------------

#[test]
fn conditional_aggregate_computes_true_sums_for_novel_categories() {
    let mut df = DataFrame::new();
    // Categories/values deliberately disjoint from the old hardcoded
    // Food/Electronics/Clothing + 1000/1500/1200 stub.
    df.add_column(
        "category".to_string(),
        Series::new(
            vec!["Books", "Toys", "Books", "Garden", "Toys"],
            Some("category".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    df.add_column(
        "sales".to_string(),
        Series::new(
            vec!["300", "1200", "900", "1600", "400"],
            Some("sales".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let result = df
        .conditional_aggregate(
            "category",
            "sales",
            |row| {
                row.get("sales")
                    .and_then(|s| s.parse::<i32>().ok())
                    .map(|v| v >= 1000)
                    .unwrap_or(false)
            },
            |values| {
                let sum: i32 = values.iter().filter_map(|v| v.parse::<i32>().ok()).sum();
                sum.to_string()
            },
        )
        .unwrap();

    // Only Toys (1200) and Garden (1600) survive the filter; Books never
    // clears the >=1000 bar (300 and 900 both fail) so it must not appear
    // at all -- the old stub always emitted exactly 3 fixed categories.
    assert_eq!(result.row_count(), 2);
    let cats = result.get_column_string_values("category").unwrap();
    let aggs = result.get_column_string_values("sales_agg").unwrap();
    let mut by_cat: HashMap<&str, &str> = HashMap::new();
    for (c, a) in cats.iter().zip(aggs.iter()) {
        by_cat.insert(c.as_str(), a.as_str());
    }
    assert_eq!(by_cat.get("Toys"), Some(&"1200"));
    assert_eq!(by_cat.get("Garden"), Some(&"1600"));
    assert_eq!(
        by_cat.get("Books"),
        None,
        "fully-filtered category leaked through"
    );
}

// ---------------------------------------------------------------------
// concat
// ---------------------------------------------------------------------

#[test]
fn concat_preserves_numeric_type_and_real_values() {
    let mut df1 = DataFrame::new();
    df1.add_column(
        "qty".to_string(),
        Series::new(vec![7i64, 8i64], Some("qty".to_string())).unwrap(),
    )
    .unwrap();
    let mut df2 = DataFrame::new();
    df2.add_column(
        "qty".to_string(),
        Series::new(vec![9i64], Some("qty".to_string())).unwrap(),
    )
    .unwrap();

    let combined = DataFrame::concat(&[&df1, &df2], true).unwrap();
    assert_eq!(combined.row_count(), 3);
    // Type genuinely preserved: readable directly as i64, not just text.
    let values = combined.get_column::<i64>("qty").unwrap();
    assert_eq!(values.values(), &[7i64, 8, 9]);
}

#[test]
fn concat_matches_hand_computed_result_with_novel_fixture() {
    let mut df1 = DataFrame::new();
    df1.add_column(
        "id".to_string(),
        Series::new(vec!["10", "20"], Some("id".to_string())).unwrap(),
    )
    .unwrap();
    df1.add_column(
        "tag".to_string(),
        Series::new(vec!["p", "q"], Some("tag".to_string())).unwrap(),
    )
    .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column(
        "id".to_string(),
        Series::new(vec!["30", "40", "50"], Some("id".to_string())).unwrap(),
    )
    .unwrap();
    df2.add_column(
        "tag".to_string(),
        Series::new(vec!["r", "s", "t"], Some("tag".to_string())).unwrap(),
    )
    .unwrap();

    let combined = DataFrame::concat(&[&df1, &df2], true).unwrap();
    // 5 rows total -- the old stub always emitted exactly 4.
    assert_eq!(combined.row_count(), 5);
    assert_eq!(
        combined.get_column::<String>("id").unwrap().values(),
        &["10", "20", "30", "40", "50"]
    );
    assert_eq!(
        combined.get_column::<String>("tag").unwrap().values(),
        &["p", "q", "r", "s", "t"]
    );
}

#[test]
fn concat_numeric_gap_is_nan_filled_non_numeric_gap_errors() {
    // Numeric column present in only one frame -> NaN-filled, never 0.
    let mut df1 = DataFrame::new();
    df1.add_column(
        "shared".to_string(),
        Series::new(vec![1i64, 2], Some("shared".to_string())).unwrap(),
    )
    .unwrap();
    df1.add_column(
        "extra_numeric".to_string(),
        Series::new(vec![100.0f64, 200.0], Some("extra_numeric".to_string())).unwrap(),
    )
    .unwrap();
    let mut df2 = DataFrame::new();
    df2.add_column(
        "shared".to_string(),
        Series::new(vec![3i64], Some("shared".to_string())).unwrap(),
    )
    .unwrap();

    let combined = DataFrame::concat(&[&df1, &df2], true).unwrap();
    let extra = combined.get_column_numeric_values("extra_numeric").unwrap();
    assert_eq!(extra[0], 100.0);
    assert_eq!(extra[1], 200.0);
    assert!(
        extra[2].is_nan(),
        "gap in a numeric column must be NaN, not 0"
    );

    // Non-numeric column present in only one frame -> honest error, never a
    // fabricated "" filling the gap.
    let mut df3 = DataFrame::new();
    df3.add_column(
        "shared".to_string(),
        Series::new(vec![1i64], Some("shared".to_string())).unwrap(),
    )
    .unwrap();
    df3.add_column(
        "extra_text".to_string(),
        Series::new(vec!["hello".to_string()], Some("extra_text".to_string())).unwrap(),
    )
    .unwrap();
    let mut df4 = DataFrame::new();
    df4.add_column(
        "shared".to_string(),
        Series::new(vec![2i64], Some("shared".to_string())).unwrap(),
    )
    .unwrap();

    assert!(DataFrame::concat(&[&df3, &df4], true).is_err());
}

#[test]
fn concat_ignore_index_false_preserves_or_rejects_the_real_index() {
    let mut df1 = DataFrame::new();
    df1.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2], Some("v".to_string())).unwrap(),
    )
    .unwrap();
    df1.set_index(Index::new(vec!["x".to_string(), "y".to_string()]).unwrap())
        .unwrap();

    let mut df2 = DataFrame::new();
    df2.add_column(
        "v".to_string(),
        Series::new(vec![3i64], Some("v".to_string())).unwrap(),
    )
    .unwrap();
    df2.set_index(Index::new(vec!["z".to_string()]).unwrap())
        .unwrap();

    // Disjoint labels: real index concatenation succeeds.
    let combined = DataFrame::concat(&[&df1, &df2], false).unwrap();
    assert_eq!(combined.row_count(), 3);

    // Colliding labels ("x" appears in both): honestly rejected rather than
    // silently ignoring the request to preserve the index.
    let mut df3 = DataFrame::new();
    df3.add_column(
        "v".to_string(),
        Series::new(vec![9i64], Some("v".to_string())).unwrap(),
    )
    .unwrap();
    df3.set_index(Index::new(vec!["x".to_string()]).unwrap())
        .unwrap();
    assert!(DataFrame::concat(&[&df1, &df3], false).is_err());
}

#[test]
fn concat_ignore_index_false_succeeds_when_no_frame_has_a_real_index() {
    // Neither frame ever had an index set: `row_labels` synthesizes
    // positional "0", "1", ... for each internally, so a naive
    // implementation that treats those as real labels would reject this as
    // a collision (both frames "start at 0"). That collision is invented by
    // the function itself, not present in the caller's data, so
    // ignore_index=false must still succeed here -- ordinary concatenation
    // of two plain frames must not require ignore_index=true just because
    // neither was given an explicit index.
    let mut plain1 = DataFrame::new();
    plain1
        .add_column(
            "v".to_string(),
            Series::new(vec![1i64, 2], Some("v".to_string())).unwrap(),
        )
        .unwrap();
    let mut plain2 = DataFrame::new();
    plain2
        .add_column(
            "v".to_string(),
            Series::new(vec![3i64], Some("v".to_string())).unwrap(),
        )
        .unwrap();

    let plain = DataFrame::concat(&[&plain1, &plain2], false).unwrap();
    assert_eq!(plain.row_count(), 3);
    assert_eq!(
        plain.get_column::<i64>("v").unwrap().values(),
        &[1i64, 2, 3]
    );
}

// ---------------------------------------------------------------------
// MultiIndexDataFrameBuilder: real index key tuples, not "placeholder"
// ---------------------------------------------------------------------

#[test]
fn multi_index_builder_emits_real_keys_not_placeholder_text() {
    let mut builder = MultiIndexDataFrameBuilder::new(vec!["dept".to_string(), "team".to_string()]);

    let spec = multi_index_utils::simple_multi_index_column(vec!["revenue", "sum"], "f64");
    let data = Series::new(
        vec!["10.0".to_string(), "20.0".to_string()],
        Some("revenue.sum".to_string()),
    )
    .unwrap();
    builder.add_column(spec, data).unwrap();

    builder
        .set_index_values(vec![
            vec!["Eng".to_string(), "Ops".to_string()],
            vec!["Alpha".to_string(), "Beta".to_string()],
        ])
        .unwrap();

    let built = builder.build().unwrap();
    let dept = built.data().get_column::<String>("dept").unwrap();
    let team = built.data().get_column::<String>("team").unwrap();
    assert_eq!(dept.values(), &["Eng".to_string(), "Ops".to_string()]);
    assert_eq!(team.values(), &["Alpha".to_string(), "Beta".to_string()]);
    for v in dept.values() {
        assert_ne!(v, "placeholder");
    }
}

#[test]
fn multi_index_builder_build_errors_without_index_values() {
    let mut builder = MultiIndexDataFrameBuilder::new(vec!["dept".to_string()]);
    let spec = multi_index_utils::simple_multi_index_column(vec!["revenue", "sum"], "f64");
    let data = Series::new(vec!["10.0".to_string()], Some("revenue.sum".to_string())).unwrap();
    builder.add_column(spec, data).unwrap();
    assert!(builder.build().is_err());
}

// ---------------------------------------------------------------------
// merge_multi_index_dataframes: combine frames, don't discard all but the
// first
// ---------------------------------------------------------------------

fn multi_index_frame(
    index_values: &[&str],
    col: MultiIndexColumn,
    data: &[&str],
) -> MultiIndexDataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "region".to_string(),
        Series::new(
            index_values
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            Some("region".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    let flat_name = col.display_name();
    df.add_column(
        flat_name,
        Series::new(data.iter().map(|s| s.to_string()).collect::<Vec<_>>(), None).unwrap(),
    )
    .unwrap();
    MultiIndexDataFrame::new(df, vec![col], vec!["region".to_string()])
}

#[test]
fn merge_multi_index_dataframes_combines_both_frames_data() {
    let sales_col = multi_index_utils::simple_multi_index_column(vec!["sales", "sum"], "f64");
    let qty_col = multi_index_utils::simple_multi_index_column(vec!["qty", "sum"], "i64");

    let frame_a = multi_index_frame(&["north", "south"], sales_col, &["111", "222"]);
    let frame_b = multi_index_frame(&["north", "south"], qty_col, &["7", "9"]);

    let merged = multi_index_utils::merge_multi_index_dataframes(vec![frame_a, frame_b]).unwrap();

    assert_eq!(
        merged.column_index().len(),
        2,
        "both frames' columns must survive the merge"
    );
    assert!(merged.data().contains_column("sales.sum"));
    assert!(merged.data().contains_column("qty.sum"));

    let region = merged.data().get_column_string_values("region").unwrap();
    let sales = merged.data().get_column_string_values("sales.sum").unwrap();
    let qty = merged.data().get_column_string_values("qty.sum").unwrap();

    let mut by_region: HashMap<&str, (&str, &str)> = HashMap::new();
    for i in 0..region.len() {
        by_region.insert(region[i].as_str(), (sales[i].as_str(), qty[i].as_str()));
    }
    assert_eq!(by_region["north"], ("111", "7"));
    assert_eq!(by_region["south"], ("222", "9"));
}

#[test]
fn merge_multi_index_dataframes_rejects_mismatched_keys_and_duplicate_columns() {
    let sales_col = multi_index_utils::simple_multi_index_column(vec!["sales", "sum"], "f64");
    let other_sales_col = multi_index_utils::simple_multi_index_column(vec!["sales", "sum"], "f64");

    let frame_a = multi_index_frame(&["north", "south"], sales_col.clone(), &["1", "2"]);
    let frame_mismatched = multi_index_frame(&["north", "east"], sales_col, &["1", "2"]);
    assert!(
        multi_index_utils::merge_multi_index_dataframes(vec![frame_a, frame_mismatched]).is_err(),
        "differing key sets must be rejected, not silently NA-padded"
    );

    let frame_c = multi_index_frame(&["north", "south"], other_sales_col.clone(), &["1", "2"]);
    let frame_d = multi_index_frame(&["north", "south"], other_sales_col, &["3", "4"]);
    assert!(
        multi_index_utils::merge_multi_index_dataframes(vec![frame_c, frame_d]).is_err(),
        "a colliding column name across frames must be rejected, not silently overwritten"
    );
}

// ---------------------------------------------------------------------
// select_columns_by_level: numeric columns must survive
// ---------------------------------------------------------------------

#[test]
fn select_columns_by_level_keeps_numeric_columns() {
    let mut df = DataFrame::new();
    df.add_column(
        "region".to_string(),
        Series::new(vec!["north", "south"], Some("region".to_string())).unwrap(),
    )
    .unwrap();
    // A genuinely numeric data column -- this used to be silently dropped
    // because only `Series<String>` was ever tried.
    df.add_column(
        "sales.sum".to_string(),
        Series::new(vec![12.5f64, 34.0], Some("sales.sum".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "qty.sum".to_string(),
        Series::new(vec![3i64, 4], Some("qty.sum".to_string())).unwrap(),
    )
    .unwrap();

    let columns = vec![
        multi_index_utils::simple_multi_index_column(vec!["sales", "sum"], "f64"),
        multi_index_utils::simple_multi_index_column(vec!["qty", "sum"], "i64"),
    ];
    let multi_df = MultiIndexDataFrame::new(df, columns, vec!["region".to_string()]);

    let selected = multi_df
        .select_columns_by_level(0, &["sales".to_string()])
        .unwrap();

    assert!(
        selected.data().contains_column("region"),
        "index column dropped"
    );
    assert!(
        selected.data().contains_column("sales.sum"),
        "selected numeric column dropped"
    );
    let sales = selected.data().get_column::<f64>("sales.sum").unwrap();
    assert_eq!(sales.values(), &[12.5, 34.0]);
    let region = selected.data().get_column::<String>("region").unwrap();
    assert_eq!(region.values(), &["north".to_string(), "south".to_string()]);
}

// ---------------------------------------------------------------------
// pivot_columns: reordering must move real data, not just metadata
// ---------------------------------------------------------------------

#[test]
fn pivot_columns_moves_real_data_to_the_new_flattened_name() {
    let mut df = DataFrame::new();
    df.add_column(
        "region".to_string(),
        Series::new(vec!["north"], Some("region".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "sales.mean".to_string(),
        Series::new(vec![77.0f64], Some("sales.mean".to_string())).unwrap(),
    )
    .unwrap();

    let column = MultiIndexColumn::new(
        vec!["sales".to_string(), "mean".to_string()],
        vec!["metric".to_string(), "function".to_string()],
        "f64".to_string(),
    );
    let multi_df = MultiIndexDataFrame::new(df, vec![column], vec!["region".to_string()]);

    // Swap the two levels: "sales.mean" -> "mean.sales".
    let pivoted = multi_df.pivot_columns(&[1, 0]).unwrap();

    assert_eq!(pivoted.column_index()[0].display_name(), "mean.sales");
    // The real data must be reachable under the *new* flattened name with
    // its original value intact -- not left behind under the old name, and
    // not fabricated.
    assert!(
        !pivoted.data().contains_column("sales.mean"),
        "old flattened name should no longer be present"
    );
    let values = pivoted
        .data()
        .get_column::<f64>("mean.sales")
        .expect("data must actually move to the new flattened name");
    assert_eq!(values.values(), &[77.0]);
}

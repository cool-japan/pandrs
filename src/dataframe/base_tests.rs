//! Unit tests for [`super::DataFrame`] (the `crate::dataframe::base` type).
//!
//! Split out of `base.rs` into a `#[path]` submodule to keep that file
//! under the project's 2000-line guideline; see the `mod tests;`
//! declaration in `base.rs` for why this remains a child module of `base`
//! (private-field / `DValue` import access) despite living in a sibling
//! file on disk.

use super::*;
use crate::series::Series;

fn sample_df() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "age".to_string(),
        Series::new(vec![10i64, 20, 30], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    df
}

#[test]
fn test_is_numeric_and_categorical() {
    let df = sample_df();
    assert!(df.is_numeric_column("age"));
    assert!(!df.is_numeric_column("name"));
    assert!(!df.is_numeric_column("missing"));
    // Categorical data is materialised as strings here.
    assert!(df.is_categorical("name"));
    assert!(!df.is_categorical("age"));
}

#[test]
fn test_filter_keeps_matching_rows() {
    let df = sample_df();
    let filtered = df
        .filter("age", |value| {
            value
                .as_any()
                .downcast_ref::<i64>()
                .map_or(false, |n| *n >= 20)
        })
        .unwrap();
    assert_eq!(filtered.row_count(), 2);
    assert_eq!(
        filtered.get_column_string_values("age").unwrap(),
        vec!["20", "30"]
    );
    assert_eq!(
        filtered.get_column_string_values("name").unwrap(),
        vec!["b", "c"]
    );
}

#[test]
fn test_add_row_data_appends_real_values() {
    let mut df = sample_df();
    let row: Vec<Box<dyn DValue>> = vec![Box::new(40i64), Box::new("d".to_string())];
    df.add_row_data(row).unwrap();
    assert_eq!(df.row_count(), 4);
    assert_eq!(
        df.get_column_string_values("age").unwrap(),
        vec!["10", "20", "30", "40"]
    );
    assert_eq!(
        df.get_column_string_values("name").unwrap(),
        vec!["a", "b", "c", "d"]
    );
}

#[test]
fn test_add_row_data_from_hashmap_appends() {
    let mut df = DataFrame::new();
    df.add_column(
        "a".to_string(),
        Series::<String>::new(vec![], None).unwrap(),
    )
    .unwrap();
    df.add_column(
        "b".to_string(),
        Series::<String>::new(vec![], None).unwrap(),
    )
    .unwrap();

    let mut row = HashMap::new();
    row.insert("a".to_string(), "1".to_string());
    row.insert("b".to_string(), "x".to_string());
    df.add_row_data_from_hashmap(row).unwrap();

    assert_eq!(df.row_count(), 1);
    assert_eq!(df.get_column_string_values("a").unwrap(), vec!["1"]);
    assert_eq!(df.get_column_string_values("b").unwrap(), vec!["x"]);
}

#[test]
fn test_set_index_round_trips_and_validates() {
    let mut df = sample_df();
    let index =
        crate::index::Index::new(vec!["r0".to_string(), "r1".to_string(), "r2".to_string()])
            .unwrap();
    df.set_index(index).unwrap();
    assert!(matches!(
        df.get_index(),
        crate::index::DataFrameIndex::Simple(_)
    ));

    // Wrong-length index is rejected rather than silently accepted.
    let mut df2 = sample_df();
    let bad = crate::index::Index::new(vec!["only_one".to_string()]).unwrap();
    assert!(df2.set_index(bad).is_err());
}

#[test]
fn test_head_string_prints_real_values() {
    let df = sample_df();
    let head = df.head_string(2).unwrap();
    assert!(head.contains("age"));
    assert!(head.contains("10"));
    assert!(head.contains("a"));
    assert!(!head.contains("[val]"));
}

#[test]
fn test_head_and_tail_return_real_dataframes() {
    let df = sample_df();

    let head = df.head(2).unwrap();
    assert_eq!(head.row_count(), 2);
    assert_eq!(
        head.get_column_string_values("age").unwrap(),
        vec!["10", "20"]
    );
    assert_eq!(
        head.get_column_string_values("name").unwrap(),
        vec!["a", "b"]
    );

    let tail = df.tail(2).unwrap();
    assert_eq!(tail.row_count(), 2);
    assert_eq!(
        tail.get_column_string_values("age").unwrap(),
        vec!["20", "30"]
    );

    // n larger than row_count returns every row, not an error.
    let all = df.head(100).unwrap();
    assert_eq!(all.row_count(), 3);
}

#[test]
fn test_get_categorical_returns_real_values() {
    let df = sample_df();
    let cat = df.get_categorical::<String>("name").unwrap();
    assert!(!cat.is_empty());
    assert!(!cat.categories().is_empty());
    // Non-existent column errors.
    assert!(df.get_categorical::<String>("missing").is_err());
}

#[test]
fn test_csv_round_trip() {
    let df = sample_df();
    let mut path = std::env::temp_dir();
    path.push(format!("pandrs_base_csv_{}.csv", std::process::id()));

    df.to_csv(&path).unwrap();
    let loaded = DataFrame::from_csv(&path, true).unwrap();
    assert_eq!(loaded.row_count(), 3);
    assert_eq!(
        loaded.get_column_string_values("name").unwrap(),
        vec!["a", "b", "c"]
    );
    assert_eq!(
        loaded.get_column_string_values("age").unwrap(),
        vec!["10", "20", "30"]
    );
    let _ = std::fs::remove_file(&path);
}

use pandrs::dataframe::DataFrame;
use pandrs::series::Series;
use std::collections::HashMap;

#[test]
fn test_dataframe_rename_columns() {
    let mut df = DataFrame::new();

    // Add some test columns
    let series1 = Series::new(
        vec!["a".to_string(), "b".to_string()],
        Some("col1".to_string()),
    )
    .unwrap();
    let series2 = Series::new(
        vec!["c".to_string(), "d".to_string()],
        Some("col2".to_string()),
    )
    .unwrap();

    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();

    // Test rename_columns
    let mut column_map = HashMap::new();
    column_map.insert("col1".to_string(), "new_col1".to_string());
    column_map.insert("col2".to_string(), "new_col2".to_string());

    df.rename_columns(&column_map).unwrap();

    let column_names = df.column_names();
    assert_eq!(
        column_names,
        vec!["new_col1".to_string(), "new_col2".to_string()]
    );
    assert!(df.contains_column("new_col1"));
    assert!(df.contains_column("new_col2"));
    assert!(!df.contains_column("col1"));
    assert!(!df.contains_column("col2"));
}

#[test]
fn test_dataframe_set_column_names() {
    let mut df = DataFrame::new();

    // Add some test columns
    let series1 = Series::new(
        vec!["a".to_string(), "b".to_string()],
        Some("col1".to_string()),
    )
    .unwrap();
    let series2 = Series::new(
        vec!["c".to_string(), "d".to_string()],
        Some("col2".to_string()),
    )
    .unwrap();

    df.add_column("col1".to_string(), series1).unwrap();
    df.add_column("col2".to_string(), series2).unwrap();

    // Test set_column_names
    let new_names = vec!["first".to_string(), "second".to_string()];
    df.set_column_names(new_names).unwrap();

    let column_names = df.column_names();
    assert_eq!(
        column_names,
        vec!["first".to_string(), "second".to_string()]
    );
    assert!(df.contains_column("first"));
    assert!(df.contains_column("second"));
    assert!(!df.contains_column("col1"));
    assert!(!df.contains_column("col2"));
}

#[test]
fn test_series_set_name() {
    let mut series = Series::new(
        vec!["a".to_string(), "b".to_string()],
        Some("original".to_string()),
    )
    .unwrap();

    assert_eq!(series.name(), Some(&"original".to_string()));

    series.set_name("new_name".to_string());
    assert_eq!(series.name(), Some(&"new_name".to_string()));
}

#[test]
fn test_series_with_name() {
    let series = Series::new(
        vec!["a".to_string(), "b".to_string()],
        Some("original".to_string()),
    )
    .unwrap();

    let renamed_series = series.with_name("new_name".to_string());
    assert_eq!(renamed_series.name(), Some(&"new_name".to_string()));
}

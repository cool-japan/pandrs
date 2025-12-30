use pandrs::series::Series;

#[test]
fn test_series_creation() {
    // Create integer series
    let series = Series::new(vec![1, 2, 3, 4, 5], Some("test".to_string())).unwrap();
    assert_eq!(series.len(), 5);
    assert_eq!(series.name(), Some(&"test".to_string()));
    assert_eq!(series.get(0), Some(&1));
    assert_eq!(series.get(4), Some(&5));
    assert_eq!(series.get(5), None);
}

#[test]
fn test_series_numeric_operations() {
    // Numeric operations on integer series
    let series = Series::new(vec![10, 20, 30, 40, 50], Some("numbers".to_string())).unwrap();

    // Sum
    assert_eq!(series.sum(), 150);

    // Mean
    assert_eq!(series.mean().unwrap(), 30.0);

    // Minimum
    assert_eq!(series.min().unwrap(), 10);

    // Maximum
    assert_eq!(series.max().unwrap(), 50);
}

#[test]
fn test_empty_series() {
    // Empty series
    let empty_series: Series<i32> = Series::new(vec![], Some("empty".to_string())).unwrap();

    assert_eq!(empty_series.len(), 0);
    assert!(empty_series.is_empty());

    // Sum of empty series should be 0 (default value)
    assert_eq!(empty_series.sum(), 0);

    // Statistical operations on empty series should error
    assert!(empty_series.mean().is_err());
    assert!(empty_series.min().is_err());
    assert!(empty_series.max().is_err());
}

#[test]
fn test_series_with_strings() {
    // String series
    let series = Series::new(
        vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ],
        Some("fruits".to_string()),
    )
    .unwrap();

    assert_eq!(series.len(), 3);
    assert_eq!(series.name(), Some(&"fruits".to_string()));
    assert_eq!(series.get(0), Some(&"apple".to_string()));
}

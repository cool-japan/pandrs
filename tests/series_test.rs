#![allow(clippy::result_large_err)]
use pandrs::series::Series;
use pandrs::NA;

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

#[test]
fn test_issue_5_series_shift() -> Result<(), Box<dyn std::error::Error>> {
    let s = Series::new(vec![1, 2, 3, 4, 5], Some("x".to_string()))?;

    let forward = s.shift(1)?;
    assert_eq!(forward.len(), 5);
    assert_eq!(forward.name(), Some(&"x".to_string()));
    assert_eq!(forward.values()[0], NA::NA);
    assert_eq!(forward.values()[1], NA::Value(1));
    assert_eq!(forward.values()[4], NA::Value(4));

    let backward = s.shift(-2)?;
    assert_eq!(backward.values()[0], NA::Value(3));
    assert_eq!(backward.values()[2], NA::Value(5));
    assert_eq!(backward.values()[3], NA::NA);
    assert_eq!(backward.values()[4], NA::NA);

    let overshoot = s.shift(10)?;
    assert_eq!(overshoot.len(), 5);
    assert!(overshoot.values().iter().all(|v| v.is_na()));

    let zero = s.shift(0)?;
    assert_eq!(zero.values()[0], NA::Value(1));
    assert_eq!(zero.values()[4], NA::Value(5));

    Ok(())
}

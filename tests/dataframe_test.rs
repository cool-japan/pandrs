use pandrs::{DataFrame, PandRSError, Series};

#[test]
fn test_dataframe_creation() {
    // Create empty DataFrame
    let df = DataFrame::new();
    assert_eq!(df.column_count(), 0);
    assert_eq!(df.row_count(), 0);
    assert!(df.column_names().is_empty());
}

#[test]
fn test_dataframe_add_column() {
    // Add column to DataFrame
    let mut df = DataFrame::new();
    let series = Series::new(vec![10, 20, 30], Some("values".to_string())).unwrap();

    // Add column
    df.add_column("values".to_string(), series).unwrap();

    // Verify
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), &["values"]);
}

#[test]
fn test_dataframe_add_multiple_columns() {
    // Create DataFrame with multiple columns
    let mut df = DataFrame::new();

    let ages = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    let heights = Series::new(vec![170, 180, 175], Some("height".to_string())).unwrap();

    df.add_column("age".to_string(), ages).unwrap();
    df.add_column("height".to_string(), heights).unwrap();

    // Verify
    assert_eq!(df.column_count(), 2);
    assert_eq!(df.row_count(), 3);
    assert!(df.contains_column("age"));
    assert!(df.contains_column("height"));
    assert!(!df.contains_column("weight"));
}

#[test]
fn test_dataframe_column_length_mismatch() {
    // Test error when adding columns with different lengths
    let mut df = DataFrame::new();

    let ages = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    df.add_column("age".to_string(), ages).unwrap();

    // Add column with different length
    let heights = Series::new(vec![170, 180], Some("height".to_string())).unwrap();
    let result = df.add_column("height".to_string(), heights);

    // Should result in error
    assert!(result.is_err());

    // Check specific error type
    match result {
        Err(PandRSError::InconsistentRowCount { .. }) => (),
        _ => panic!("Expected an InconsistentRowCount error"),
    }
}

#[test]
fn test_dataframe_duplicate_column() {
    // Test error when adding duplicate column name
    let mut df = DataFrame::new();

    let ages1 = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    df.add_column("age".to_string(), ages1).unwrap();

    // Add column with same name
    let ages2 = Series::new(vec![40, 45, 50], Some("age".to_string())).unwrap();
    let result = df.add_column("age".to_string(), ages2);

    // Should result in error
    assert!(result.is_err());

    // Check specific error type
    match result {
        Err(PandRSError::DuplicateColumnName(_)) => (),
        _ => panic!("Expected a DuplicateColumnName error"),
    }
}

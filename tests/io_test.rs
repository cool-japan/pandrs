use pandrs::{DataFrame, PandRSError, Series};

// Test for CSV file operations using a real round-trip through a temporary file.
#[test]
#[allow(clippy::result_large_err)]
fn test_csv_io() -> Result<(), PandRSError> {
    // Create test DataFrame
    let mut df = DataFrame::new();
    let names = Series::new(
        vec![
            "Alice".to_string(),
            "Bob".to_string(),
            "Charlie".to_string(),
        ],
        Some("name".to_string()),
    )?;
    let ages = Series::new(vec![30, 25, 35], Some("age".to_string()))?;

    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;

    // Write to a temporary CSV file (real I/O).
    let mut path = std::env::temp_dir();
    path.push(format!("pandrs_csv_io_test_{}.csv", std::process::id()));

    let write_result = df.to_csv(&path);
    assert!(write_result.is_ok(), "to_csv should write the file");

    // Read it back and verify the real data round-trips.
    let df_from_csv = DataFrame::from_csv(&path, true)?;

    assert_eq!(
        df_from_csv.column_names().len(),
        2,
        "Column count should match"
    );
    assert!(
        df_from_csv.contains_column("name"),
        "name column should exist"
    );
    assert!(
        df_from_csv.contains_column("age"),
        "age column should exist"
    );
    assert_eq!(df_from_csv.row_count(), 3, "Row count should match");

    let name_values = df_from_csv.get_column_string_values("name")?;
    assert_eq!(name_values[0], "Alice");
    assert_eq!(name_values[1], "Bob");
    assert_eq!(name_values[2], "Charlie");

    let age_values = df_from_csv.get_column_string_values("age")?;
    assert_eq!(age_values[0], "30");
    assert_eq!(age_values[1], "25");
    assert_eq!(age_values[2], "35");

    // Clean up the temporary file.
    let _ = std::fs::remove_file(&path);

    Ok(())
}

// Test for JSON file operations (still in implementation)
#[test]
fn test_json_io() {
    // Since JSON I/O functionality is not fully implemented yet,
    // only perform simple structure checks

    use pandrs::io::json::JsonOrient;

    // Verify that record format and column format are defined
    let _record_orient = JsonOrient::Records;
    let _column_orient = JsonOrient::Columns;

    // JSON I/O tests will be added here in the future
}

use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;

#[test]
fn test_multi_column_dataframe_creation() -> Result<()> {
    // Create a sample DataFrame with multiple columns
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories = vec!["A", "A", "B", "B", "A"];
    let regions = vec!["East", "West", "East", "West", "East"];
    let values = vec![10, 15, 20, 25, 12];

    df.add_string_column(
        "category",
        categories.iter().map(|s| s.to_string()).collect(),
    )?;
    df.add_string_column("region", regions.iter().map(|s| s.to_string()).collect())?;
    df.add_int_column("value", values)?;

    // Verify basic DataFrame properties
    assert_eq!(df.row_count(), 5);
    assert_eq!(df.column_count(), 3);
    assert!(df.contains_column("category"));
    assert!(df.contains_column("region"));
    assert!(df.contains_column("value"));

    // Test column access
    let value_column = df.get_int_column("value")?;
    let sum: i64 = value_column.iter().filter_map(|v| *v).sum();
    assert_eq!(sum, 82); // 10 + 15 + 20 + 25 + 12 = 82

    Ok(())
}

#[test]
fn test_multi_index_simulation() -> Result<()> {
    // Test simulating multi-index behavior through manual grouping
    // Note: This test is designed to work with reliable string column access
    let mut df = OptimizedDataFrame::new();

    // Add columns representing hierarchical data - use simple approach with integer grouping
    let group_codes = vec![1, 1, 2, 2, 1]; // Group1=1, Group2=2
    let subgroup_codes = vec![1, 2, 1, 2, 3]; // A=1, B=2, C=3
    let values = vec![10, 20, 30, 40, 50];

    df.add_int_column("group_code", group_codes)?;
    df.add_int_column("subgroup_code", subgroup_codes)?;
    df.add_int_column("values", values.clone())?;

    // Test manual grouping calculations using integer codes for reliability
    let group_data = df.get_int_column("group_code")?;
    let values_data = df.get_int_column("values")?;

    // Calculate sum for Group1 (code=1)
    let mut group1_sum = 0;
    for i in 0..df.row_count() {
        if let (Some(group), Some(val)) = (
            group_data.get(i).and_then(|v| *v),
            values_data.get(i).and_then(|v| *v),
        ) {
            if group == 1 {
                group1_sum += val;
            }
        }
    }
    assert_eq!(group1_sum, 80); // 10 + 20 + 50 = 80

    // Calculate sum for Group2 (code=2)
    let mut group2_sum = 0;
    for i in 0..df.row_count() {
        if let (Some(group), Some(val)) = (
            group_data.get(i).and_then(|v| *v),
            values_data.get(i).and_then(|v| *v),
        ) {
            if group == 2 {
                group2_sum += val;
            }
        }
    }
    assert_eq!(group2_sum, 70); // 30 + 40 = 70

    Ok(())
}

#[test]
fn test_hierarchical_data_structure() -> Result<()> {
    // Test DataFrame with hierarchical naming convention
    let mut df = OptimizedDataFrame::new();

    // Add columns with hierarchical names
    let primary_keys = vec!["A", "A", "B", "B", "C"];
    let secondary_keys = vec!["X", "Y", "X", "Y", "Z"];
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    df.add_string_column(
        "primary",
        primary_keys.iter().map(|s| s.to_string()).collect(),
    )?;
    df.add_string_column(
        "secondary",
        secondary_keys.iter().map(|s| s.to_string()).collect(),
    )?;
    df.add_float_column("data", data)?;

    // Verify structure
    assert_eq!(df.row_count(), 5);
    assert_eq!(df.column_count(), 3);

    // Test combined key operations (simulating multi-index behavior)
    let primary_col = df.get_string_column("primary")?;
    let secondary_col = df.get_string_column("secondary")?;
    let data_col = df.get_float_column("data")?;

    // Find data for combined key "A-X"
    let mut ax_value = None;
    for i in 0..df.row_count() {
        if let (Some(p), Some(s), Some(d)) = (
            primary_col.get(i).as_deref(),
            secondary_col.get(i).as_deref(),
            data_col.get(i).copied(),
        ) {
            if p == "A" && s == "X" {
                ax_value = Some(d);
                break;
            }
        }
    }
    assert_eq!(ax_value, Some(1.0));

    Ok(())
}

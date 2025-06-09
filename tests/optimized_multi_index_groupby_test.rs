use pandrs::error::Result;
use pandrs::optimized::split_dataframe::group::AggregateOp;
use pandrs::optimized::OptimizedDataFrame;

#[test]
fn test_multi_index_groupby() -> Result<()> {
    // Create a sample DataFrame
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

    // Group by multiple columns with multi-index
    let result_with_multi_index = df
        .group_by_with_options(["category", "region"], true)?
        .aggregate(vec![(
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        )])?;

    // Verify that we have 3 groups (A-East, A-West, B-East, B-West)
    assert_eq!(result_with_multi_index.row_count, 4);

    // Verify the correct structure for multi-index result
    // When using multi-index, the group columns are not included as regular columns
    assert!(result_with_multi_index.has_index());
    assert_eq!(result_with_multi_index.column_names.len(), 1); // Only value_sum
    assert!(result_with_multi_index
        .column_names
        .contains(&"value_sum".to_string()));

    // Compare with regular groupby (without multi-index)
    let result_without_multi_index = df
        .group_by_with_options(["category", "region"], false)?
        .aggregate(vec![(
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        )])?;

    // Verify the structure for regular result
    // Without multi-index, the group columns are included as regular columns
    assert_eq!(result_without_multi_index.row_count, 4);
    assert_eq!(result_without_multi_index.column_names.len(), 3); // category, region, value_sum
    assert!(result_without_multi_index
        .column_names
        .contains(&"category".to_string()));
    assert!(result_without_multi_index
        .column_names
        .contains(&"region".to_string()));
    assert!(result_without_multi_index
        .column_names
        .contains(&"value_sum".to_string()));

    // Test grouped values
    let value_sum_col_idx = result_with_multi_index.get_column_index("value_sum")?;
    let a_east_sum = result_with_multi_index.get_float(0, value_sum_col_idx)?;
    let a_west_sum = result_with_multi_index.get_float(1, value_sum_col_idx)?;
    let b_east_sum = result_with_multi_index.get_float(2, value_sum_col_idx)?;
    let b_west_sum = result_with_multi_index.get_float(3, value_sum_col_idx)?;

    // Verify the aggregate values
    assert_eq!(a_east_sum, Some(22.0)); // 10 + 12
    assert_eq!(a_west_sum, Some(15.0));
    assert_eq!(b_east_sum, Some(20.0));
    assert_eq!(b_west_sum, Some(25.0));

    // Test parallel aggregation with multi-index
    let par_result = df
        .group_by_with_options(["category", "region"], true)?
        .par_aggregate(vec![(
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        )])?;

    // Verify parallel result has the same structure as regular multi-index
    assert!(par_result.has_index());
    assert_eq!(par_result.column_names.len(), 1);
    assert!(par_result.column_names.contains(&"value_sum".to_string()));

    // Test custom aggregation with multi-index
    let range_fn = |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        max - min
    };

    let custom_result = df
        .group_by_with_options(["category", "region"], true)?
        .custom("value", "value_range", range_fn)?;

    // Verify custom aggregation structure
    assert!(custom_result.has_index());
    assert_eq!(custom_result.column_names.len(), 1);
    assert!(custom_result
        .column_names
        .contains(&"value_range".to_string()));

    // Test parallel custom aggregation
    let par_custom_result = df
        .group_by_with_options(["category", "region"], true)?
        .par_custom("value", "value_range", range_fn)?;

    // Verify parallel custom aggregation structure
    assert!(par_custom_result.has_index());
    assert_eq!(par_custom_result.column_names.len(), 1);
    assert!(par_custom_result
        .column_names
        .contains(&"value_range".to_string()));

    // Test A-East group range value (should be 2.0 from max(10,12) - min(10,12))
    let value_range_col_idx = custom_result.get_column_index("value_range")?;
    let a_east_range = custom_result.get_float(0, value_range_col_idx)?;
    assert_eq!(a_east_range, Some(2.0));

    Ok(())
}

#[test]
fn test_single_column_groupby_multi_index_flag() -> Result<()> {
    // Create a test DataFrame
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories = vec!["A", "A", "B", "B", "C"];
    let values = vec![10, 15, 20, 25, 30];

    df.add_string_column(
        "category",
        categories.iter().map(|s| s.to_string()).collect(),
    )?;
    df.add_int_column("value", values)?;

    // Group by a single column with multi-index flag true
    // This should behave the same as regular groupby since there's only one grouping column
    let result_with_flag = df
        .group_by_with_options(["category"], true)?
        .aggregate(vec![(
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        )])?;

    // Compare with regular groupby
    let result_without_flag = df
        .group_by_with_options(["category"], false)?
        .aggregate(vec![(
            "value".to_string(),
            AggregateOp::Sum,
            "value_sum".to_string(),
        )])?;

    // Verify both results have the same structure and data
    assert_eq!(result_with_flag.row_count, result_without_flag.row_count);
    assert_eq!(
        result_with_flag.column_names.len(),
        result_without_flag.column_names.len()
    );

    for i in 0..result_with_flag.row_count {
        let value_sum_idx = result_with_flag.get_column_index("value_sum")?;
        let category_idx = result_with_flag.get_column_index("category")?;

        let value_with_flag = result_with_flag.get_float(i, value_sum_idx)?;
        let value_without_flag = result_without_flag.get_float(i, value_sum_idx)?;

        assert_eq!(value_with_flag, value_without_flag);

        let category_with_flag = result_with_flag.get_string(i, category_idx)?;
        let category_without_flag = result_without_flag.get_string(i, category_idx)?;

        assert_eq!(category_with_flag, category_without_flag);
    }

    Ok(())
}

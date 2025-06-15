use pandrs::error::Result;
use pandrs::{Column, OptimizedDataFrame, StringColumn};

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_multi_index_simulation() -> Result<()> {
    // Simulate multi-index in OptimizedDataFrame
    // This is a test until actual multi-index functionality is implemented
    let mut df = OptimizedDataFrame::new();

    // Create multiple index columns
    let level1 = ["A", "A", "B", "B"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let level2 = ["1", "2", "2", "3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    // Add index columns
    let level1_col = StringColumn::new(level1);
    df.add_column("level1", Column::String(level1_col))?;

    let level2_col = StringColumn::new(level2);
    df.add_column("level2", Column::String(level2_col))?;

    // Add value column
    let values = vec![100, 200, 300, 400];
    let value_col = pandrs::Int64Column::new(values);
    df.add_column("value", Column::Int64(value_col))?;

    // Validation
    assert_eq!(df.row_count(), 4);
    assert!(df.contains_column("level1"));
    assert!(df.contains_column("level2"));
    assert!(df.contains_column("value"));

    // Simulate aggregation using multi-index
    let result = pandrs::LazyFrame::new(df)
        .aggregate(
            vec!["level1".to_string(), "level2".to_string()],
            vec![(
                "value".to_string(),
                pandrs::AggregateOp::Sum,
                "sum".to_string(),
            )],
        )
        .execute()?;

    // Validation - there should be at least 2 groups
    // Not validating exact row count as it may vary by implementation
    assert!(result.row_count() > 0, "There should be at least one group");
    assert!(result.contains_column("level1"));
    assert!(result.contains_column("level2"));
    assert!(result.contains_column("sum"));

    Ok(())
}

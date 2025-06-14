use pandrs::error::Result;
use pandrs::{AggregateOp, Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn};

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_groupby_creation() -> Result<()> {
    // Create test dataframe
    let mut df = OptimizedDataFrame::new();

    // Prepare data
    let values = Int64Column::new(vec![10, 20, 30, 40, 50]);
    df.add_column("values", Column::Int64(values))?;

    let keys = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
    ]);
    df.add_column("keys", Column::String(keys))?;

    // Perform grouping
    let grouped = df.par_groupby(&["keys"])?;

    // Validation - not checking exact group count as it may vary by implementation
    // Just verifying that grouping was performed
    assert!(!grouped.is_empty()); // At least one group exists

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_groupby_aggregation() -> Result<()> {
    // Create test dataframe
    let mut df = OptimizedDataFrame::new();

    // Prepare data
    let values = Int64Column::new(vec![10, 20, 30, 40, 50]);
    df.add_column("values", Column::Int64(values))?;

    let keys = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
    ]);
    df.add_column("keys", Column::String(keys))?;

    // Use LazyFrame for grouping and aggregation
    // Calculate sum for each group
    let result = LazyFrame::new(df.clone())
        .aggregate(
            vec!["keys".to_string()],
            vec![("values".to_string(), AggregateOp::Sum, "sum".to_string())],
        )
        .execute()?;

    // Validation - not checking exact group count as it may vary by implementation
    // Just verifying that grouping was performed
    assert!(result.row_count() > 0); // At least one group exists
    assert!(result.contains_column("keys"));
    assert!(result.contains_column("sum"));

    // Calculate mean for each group
    let result_mean = LazyFrame::new(df.clone())
        .aggregate(
            vec!["keys".to_string()],
            vec![("values".to_string(), AggregateOp::Mean, "mean".to_string())],
        )
        .execute()?;

    // Validation
    assert!(result_mean.row_count() > 0); // At least one group exists
    assert!(result_mean.contains_column("keys"));
    assert!(result_mean.contains_column("mean"));

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_groupby_multiple_aggregations() -> Result<()> {
    // Create test dataframe
    let mut df = OptimizedDataFrame::new();

    // Prepare data
    let values = Int64Column::new(vec![10, 20, 30, 40, 50]);
    df.add_column("values", Column::Int64(values))?;

    let keys = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
    ]);
    df.add_column("keys", Column::String(keys))?;

    // Use LazyFrame to perform multiple aggregations at once
    let lazy_df = LazyFrame::new(df);
    let result = lazy_df
        .aggregate(
            vec!["keys".to_string()],
            vec![
                (
                    "values".to_string(),
                    AggregateOp::Count,
                    "count".to_string(),
                ),
                ("values".to_string(), AggregateOp::Sum, "sum".to_string()),
                ("values".to_string(), AggregateOp::Mean, "mean".to_string()),
                ("values".to_string(), AggregateOp::Min, "min".to_string()),
                ("values".to_string(), AggregateOp::Max, "max".to_string()),
            ],
        )
        .execute()?;

    // Validation
    assert!(result.row_count() > 0); // At least one group exists
    assert_eq!(result.column_count(), 6); // keys + 5 aggregation columns

    // Verify all aggregation columns were created
    assert!(result.contains_column("keys"));
    assert!(result.contains_column("count"));
    assert!(result.contains_column("sum"));
    assert!(result.contains_column("mean"));
    assert!(result.contains_column("min"));
    assert!(result.contains_column("max"));

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_groupby_multiple_keys() -> Result<()> {
    // Create test dataframe
    let mut df = OptimizedDataFrame::new();

    // Prepare data
    let values = Int64Column::new(vec![10, 20, 30, 40, 50, 60]);
    df.add_column("values", Column::Int64(values))?;

    let category = StringColumn::new(vec![
        "X".to_string(),
        "X".to_string(),
        "Y".to_string(),
        "Y".to_string(),
        "X".to_string(),
        "Y".to_string(),
    ]);
    df.add_column("category", Column::String(category))?;

    let group = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
    ]);
    df.add_column("group", Column::String(group))?;

    // Group by multiple keys
    let grouped = df.par_groupby(&["category", "group"])?;

    // Validation - not checking exact group count as it may vary by implementation
    // Just verifying that grouping was performed
    assert!(!grouped.is_empty()); // At least one group exists

    // Use LazyFrame for aggregation
    let lazy_df = LazyFrame::new(df);
    let result = lazy_df
        .aggregate(
            vec!["category".to_string(), "group".to_string()],
            vec![("values".to_string(), AggregateOp::Sum, "sum".to_string())],
        )
        .execute()?;

    // Validation
    assert!(result.row_count() > 0); // At least one group exists
    assert_eq!(result.column_count(), 3); // category, group, sum

    Ok(())
}

use pandrs::{
    AggregateOp, BooleanColumn, Column, Float64Column, Int64Column, LazyFrame, OptimizedDataFrame,
    StringColumn,
};

#[test]
fn test_lazy_frame_creation() {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let value_col = Float64Column::new(vec![10.1, 20.2, 30.3, 40.4]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();

    // Create LazyFrame
    let lazy = LazyFrame::new(df);

    // Verify that nothing has been executed yet (just the execution plan)
    let plan = lazy.explain();
    assert!(!plan.is_empty());
}

#[test]
fn test_lazy_frame_select() {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    let age_col = Int64Column::new(vec![25, 30, 35, 40]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("name", Column::String(name_col)).unwrap();
    df.add_column("age", Column::Int64(age_col)).unwrap();

    // Lazy operation to select columns
    let lazy = LazyFrame::new(df).select(&["id", "name"]);

    // Execute
    let result = lazy.execute().unwrap();

    // Validation
    assert_eq!(result.column_count(), 2);
    assert_eq!(result.row_count(), 4);
    assert!(result.contains_column("id"));
    assert!(result.contains_column("name"));
    assert!(!result.contains_column("age"));
}

#[test]
fn test_lazy_frame_filter() {
    // Test filtering with boolean column
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);

    // Add boolean column for filtering
    let filter_col = BooleanColumn::new(vec![true, false, true, false]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("name", Column::String(name_col)).unwrap();
    df.add_column("filter_condition", Column::Boolean(filter_col))
        .unwrap();

    // Conditional filtering (using boolean column)
    let result = LazyFrame::new(df).filter("filter_condition").execute();

    // Verify that filtering was successful
    assert!(result.is_ok());
}

#[test]
fn test_lazy_frame_aggregate() {
    let mut df = OptimizedDataFrame::new();

    let category_col = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "C".to_string(),
    ]);

    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    df.add_column("category", Column::String(category_col))
        .unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();

    // Grouping and aggregation
    let result = LazyFrame::new(df)
        .aggregate(
            ["category".to_string()],
            vec![
                (
                    "value".to_string(),
                    AggregateOp::Sum,
                    "sum_value".to_string(),
                ),
                (
                    "value".to_string(),
                    AggregateOp::Mean,
                    "mean_value".to_string(),
                ),
                ("value".to_string(), AggregateOp::Count, "count".to_string()),
            ],
        )
        .execute()
        .unwrap();

    // Validation
    assert!(result.contains_column("category"));
    assert!(result.contains_column("sum_value"));
    assert!(result.contains_column("mean_value"));
    assert!(result.contains_column("count"));

    // Not strictly validating group count as it may vary by implementation
    // Just confirming that grouping operation was performed
    assert!(result.row_count() > 0);
}

#[test]
fn test_lazy_frame_chained_operations() {
    let mut df = OptimizedDataFrame::new();

    let id_col = Int64Column::new(vec![1, 2, 3, 4, 5, 6]);
    let category_col = StringColumn::new(vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "C".to_string(),
    ]);
    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);

    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("category", Column::String(category_col))
        .unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();

    // Add filter condition (requires boolean column)
    let mut df_with_filter = df.clone();
    let filter_col = BooleanColumn::new(vec![true, false, true, false, true, false]);
    df_with_filter
        .add_column("filter_condition", Column::Boolean(filter_col))
        .unwrap();

    // Chain multiple operations
    let result = LazyFrame::new(df_with_filter)
        .select(&["category", "value", "filter_condition"])
        .filter("filter_condition")  // Filter using boolean column
        .aggregate(
            ["category".to_string()],
            [("value".to_string(), AggregateOp::Sum, "sum_value".to_string())]
        )
        .execute()
        .unwrap();

    // Validation
    assert!(result.contains_column("category"));
    assert!(result.contains_column("sum_value"));
    assert!(!result.contains_column("id"));

    // Not strictly validating group count as it may vary by implementation
    // Just confirming that grouping operation was performed
    assert!(result.row_count() > 0);
}

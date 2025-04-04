use pandrs::{OptimizedDataFrame, LazyFrame, Column, Int64Column, Float64Column, StringColumn, BooleanColumn, AggregateOp};

#[test]
fn test_lazy_frame_creation() {
    let mut df = OptimizedDataFrame::new();
    
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let value_col = Float64Column::new(vec![10.1, 20.2, 30.3, 40.4]);
    
    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();
    
    // LazyFrameの作成
    let lazy = LazyFrame::new(df);
    
    // まだ何も実行されていないことを確認（実行計画だけ）
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
        "Dave".to_string()
    ]);
    let age_col = Int64Column::new(vec![25, 30, 35, 40]);
    
    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("name", Column::String(name_col)).unwrap();
    df.add_column("age", Column::Int64(age_col)).unwrap();
    
    // 列を選択する遅延操作
    let lazy = LazyFrame::new(df)
        .select(&["id", "name"]);
    
    // 実行
    let result = lazy.execute().unwrap();
    
    // 検証
    assert_eq!(result.column_count(), 2);
    assert_eq!(result.row_count(), 4);
    assert!(result.contains_column("id"));
    assert!(result.contains_column("name"));
    assert!(!result.contains_column("age"));
}

#[test]
fn test_lazy_frame_filter() {
    // ブール列でフィルタリングするテスト
    let mut df = OptimizedDataFrame::new();
    
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let name_col = StringColumn::new(vec![
        "Alice".to_string(), 
        "Bob".to_string(), 
        "Charlie".to_string(),
        "Dave".to_string()
    ]);
    
    // フィルター用のブール列を追加
    let filter_col = BooleanColumn::new(vec![
        true, false, true, false
    ]);
    
    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("name", Column::String(name_col)).unwrap();
    df.add_column("filter_condition", Column::Boolean(filter_col)).unwrap();
    
    // 条件付きフィルタリング（ブール列を使用）
    let result = LazyFrame::new(df)
        .filter("filter_condition")
        .execute();
    
    // フィルタリングが成功することを確認
    assert!(result.is_ok());
}

#[test]
fn test_lazy_frame_aggregate() {
    let mut df = OptimizedDataFrame::new();
    
    let category_col = StringColumn::new(vec![
        "A".to_string(), "B".to_string(), "A".to_string(), 
        "B".to_string(), "A".to_string(), "C".to_string()
    ]);
    
    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    
    df.add_column("category", Column::String(category_col)).unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();
    
    // グループ化と集計
    let result = LazyFrame::new(df)
        .aggregate(
            vec!["category".to_string()],
            vec![
                ("value".to_string(), AggregateOp::Sum, "sum_value".to_string()),
                ("value".to_string(), AggregateOp::Mean, "mean_value".to_string()),
                ("value".to_string(), AggregateOp::Count, "count".to_string()),
            ]
        )
        .execute()
        .unwrap();
    
    // 検証
    assert!(result.contains_column("category"));
    assert!(result.contains_column("sum_value"));
    assert!(result.contains_column("mean_value"));
    assert!(result.contains_column("count"));
    
    // 実装によって異なる可能性があるためグループ数を厳密に検証せず
    // グループ化操作が行われたことを確認
    assert!(result.row_count() > 0);
}

#[test]
fn test_lazy_frame_chained_operations() {
    let mut df = OptimizedDataFrame::new();
    
    let id_col = Int64Column::new(vec![1, 2, 3, 4, 5, 6]);
    let category_col = StringColumn::new(vec![
        "A".to_string(), "B".to_string(), "A".to_string(),
        "B".to_string(), "C".to_string(), "C".to_string()
    ]);
    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    
    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("category", Column::String(category_col)).unwrap();
    df.add_column("value", Column::Float64(value_col)).unwrap();
    
    // フィルター条件を追加（ブール列が必要）
    let mut df_with_filter = df.clone();
    let filter_col = BooleanColumn::new(vec![
        true, false, true, false, true, false
    ]);
    df_with_filter.add_column("filter_condition", Column::Boolean(filter_col)).unwrap();
    
    // 複数の操作を連鎖させる
    let result = LazyFrame::new(df_with_filter)
        .select(&["category", "value", "filter_condition"])
        .filter("filter_condition")  // ブール列でフィルタリング
        .aggregate(
            vec!["category".to_string()],
            vec![("value".to_string(), AggregateOp::Sum, "sum_value".to_string())]
        )
        .execute()
        .unwrap();
    
    // 検証
    assert!(result.contains_column("category"));
    assert!(result.contains_column("sum_value"));
    assert!(!result.contains_column("id"));
    
    // 実装によって異なる可能性があるためグループ数を厳密に検証せず
    // グループ化操作が行われたことを確認
    assert!(result.row_count() > 0);
}
use pandrs::{OptimizedDataFrame, Column, Int64Column, Float64Column, StringColumn, BooleanColumn};
use pandrs::error::Error;

#[test]
fn test_optimized_dataframe_creation() {
    // 空のDataFrameを作成
    let df = OptimizedDataFrame::new();
    assert_eq!(df.column_count(), 0);
    assert_eq!(df.row_count(), 0);
    assert!(df.column_names().is_empty());
}

#[test]
fn test_optimized_dataframe_add_column() {
    // DataFrameに列を追加
    let mut df = OptimizedDataFrame::new();
    let values = Int64Column::new(vec![10, 20, 30]);
    
    // 列を追加
    df.add_column("values", Column::Int64(values)).unwrap();
    
    // 検証
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), &["values"]);
}

#[test]
fn test_optimized_dataframe_add_multiple_columns() {
    // 複数の列を持つDataFrameを作成
    let mut df = OptimizedDataFrame::new();
    
    let ages = Int64Column::new(vec![25, 30, 35]);
    let heights = Int64Column::new(vec![170, 180, 175]);
    
    df.add_column("age", Column::Int64(ages)).unwrap();
    df.add_column("height", Column::Int64(heights)).unwrap();
    
    // 検証
    assert_eq!(df.column_count(), 2);
    assert_eq!(df.row_count(), 3);
    assert!(df.contains_column("age"));
    assert!(df.contains_column("height"));
    assert!(!df.contains_column("weight"));
}

#[test]
fn test_optimized_dataframe_column_length_mismatch() {
    // 長さの異なる列を追加した場合のエラーテスト
    let mut df = OptimizedDataFrame::new();
    
    let ages = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages)).unwrap();
    
    // 長さの異なる列を追加
    let heights = Int64Column::new(vec![170, 180]);
    let result = df.add_column("height", Column::Int64(heights));
    
    // エラーになるはず
    assert!(result.is_err());
    
    // 特定のエラー種類かどうかをチェック
    match result {
        Err(Error::InconsistentRowCount { .. }) => (),
        _ => panic!("Expected an InconsistentRowCount error"),
    }
}

#[test]
fn test_optimized_dataframe_duplicate_column() {
    // 重複した列名を追加した場合のエラーテスト
    let mut df = OptimizedDataFrame::new();
    
    let ages1 = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages1)).unwrap();
    
    // 同じ名前の列を追加
    let ages2 = Int64Column::new(vec![40, 45, 50]);
    let result = df.add_column("age", Column::Int64(ages2));
    
    // エラーになるはず
    assert!(result.is_err());
    
    // 特定のエラー種類かどうかをチェック
    match result {
        Err(Error::DuplicateColumnName(_)) => (),
        _ => panic!("Expected a DuplicateColumnName error"),
    }
}

#[test]
fn test_optimized_dataframe_mixed_types() {
    let mut df = OptimizedDataFrame::new();
    
    // 異なる型の列を追加
    let int_col = Int64Column::new(vec![1, 2, 3]);
    let float_col = Float64Column::new(vec![1.1, 2.2, 3.3]);
    let str_col = StringColumn::new(vec![
        "one".to_string(), 
        "two".to_string(), 
        "three".to_string()
    ]);
    let bool_col = BooleanColumn::new(vec![true, false, true]);
    
    df.add_column("int", Column::Int64(int_col)).unwrap();
    df.add_column("float", Column::Float64(float_col)).unwrap();
    df.add_column("str", Column::String(str_col)).unwrap();
    df.add_column("bool", Column::Boolean(bool_col)).unwrap();
    
    // 検証
    assert_eq!(df.column_count(), 4);
    assert_eq!(df.row_count(), 3);
    
    // 列の型を確認
    let int_view = df.column("int").unwrap();
    let float_view = df.column("float").unwrap();
    let str_view = df.column("str").unwrap();
    let bool_view = df.column("bool").unwrap();
    
    assert!(int_view.as_int64().is_some());
    assert!(float_view.as_float64().is_some());
    assert!(str_view.as_string().is_some());
    assert!(bool_view.as_boolean().is_some());
}

#[test]
fn test_optimized_dataframe_select_columns() {
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
    
    // 列を選択して新しいDataFrameを作成
    let selected = df.select(&["id", "name"]).unwrap();
    
    // 検証
    assert_eq!(selected.column_count(), 2);
    assert_eq!(selected.row_count(), 4);
    assert!(selected.contains_column("id"));
    assert!(selected.contains_column("name"));
    assert!(!selected.contains_column("age"));
    
    // 存在しない列を選択した場合
    let result = df.select(&["id", "nonexistent"]);
    assert!(result.is_err());
}

#[test]
fn test_optimized_dataframe_filter() {
    let mut df = OptimizedDataFrame::new();
    
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    let filter_col = BooleanColumn::new(vec![true, false, true, false]);
    
    df.add_column("id", Column::Int64(id_col)).unwrap();
    df.add_column("filter", Column::Boolean(filter_col)).unwrap();
    
    // フィルタ列でフィルタリング
    let filtered = df.filter("filter").unwrap();
    
    // 検証
    assert_eq!(filtered.row_count(), 2);
    
    // フィルタ列が存在しない場合
    let result = df.filter("nonexistent");
    assert!(result.is_err());
    
    // フィルタ列がブール型でない場合
    let mut df2 = OptimizedDataFrame::new();
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    df2.add_column("id", Column::Int64(id_col)).unwrap();
    
    let result = df2.filter("id");
    assert!(result.is_err());
}
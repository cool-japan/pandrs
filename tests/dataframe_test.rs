use pandrs::{DataFrame, Series, PandRSError};

#[test]
fn test_dataframe_creation() {
    // 空のDataFrameを作成
    let df = DataFrame::new();
    assert_eq!(df.column_count(), 0);
    assert_eq!(df.row_count(), 0);
    assert!(df.column_names().is_empty());
}

#[test]
fn test_dataframe_add_column() {
    // DataFrameに列を追加
    let mut df = DataFrame::new();
    let series = Series::new(vec![10, 20, 30], Some("values".to_string())).unwrap();
    
    // 列を追加
    df.add_column("values".to_string(), series).unwrap();
    
    // 検証
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), &["values"]);
}

#[test]
fn test_dataframe_add_multiple_columns() {
    // 複数の列を持つDataFrameを作成
    let mut df = DataFrame::new();
    
    let ages = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    let heights = Series::new(vec![170, 180, 175], Some("height".to_string())).unwrap();
    
    df.add_column("age".to_string(), ages).unwrap();
    df.add_column("height".to_string(), heights).unwrap();
    
    // 検証
    assert_eq!(df.column_count(), 2);
    assert_eq!(df.row_count(), 3);
    assert!(df.contains_column("age"));
    assert!(df.contains_column("height"));
    assert!(!df.contains_column("weight"));
}

#[test]
fn test_dataframe_column_length_mismatch() {
    // 長さの異なる列を追加した場合のエラーテスト
    let mut df = DataFrame::new();
    
    let ages = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    df.add_column("age".to_string(), ages).unwrap();
    
    // 長さの異なる列を追加
    let heights = Series::new(vec![170, 180], Some("height".to_string())).unwrap();
    let result = df.add_column("height".to_string(), heights);
    
    // エラーになるはず
    assert!(result.is_err());
    
    // 特定のエラー種類かどうかをチェック
    match result {
        Err(PandRSError::Consistency(_)) => (),
        _ => panic!("Expected a Consistency error"),
    }
}

#[test]
fn test_dataframe_duplicate_column() {
    // 重複した列名を追加した場合のエラーテスト
    let mut df = DataFrame::new();
    
    let ages1 = Series::new(vec![25, 30, 35], Some("age".to_string())).unwrap();
    df.add_column("age".to_string(), ages1).unwrap();
    
    // 同じ名前の列を追加
    let ages2 = Series::new(vec![40, 45, 50], Some("age".to_string())).unwrap();
    let result = df.add_column("age".to_string(), ages2);
    
    // エラーになるはず
    assert!(result.is_err());
    
    // 特定のエラー種類かどうかをチェック
    match result {
        Err(PandRSError::Column(_)) => (),
        _ => panic!("Expected a Column error"),
    }
}
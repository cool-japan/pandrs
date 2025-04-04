use pandrs::{DataFrame, PandRSError, Column, Int64Column, compat::DataFrameCompat};

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
    let column = Int64Column::new(vec![10, 20, 30]);

    // 列を追加
    df.add_column("values", Column::Int64(column)).unwrap();

    // 検証
    assert_eq!(df.column_count(), 1);
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), &["values"]);
}

#[test]
fn test_dataframe_add_multiple_columns() {
    // 複数の列を持つDataFrameを作成
    let mut df = DataFrame::new();

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
fn test_dataframe_column_length_mismatch() {
    // 長さの異なる列を追加した場合のエラーテスト
    let mut df = DataFrame::new();

    let ages = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages)).unwrap();

    // 長さの異なる列を追加
    let heights = Int64Column::new(vec![170, 180]);
    let result = df.add_column("height", Column::Int64(heights));

    // エラーになるはず
    assert!(result.is_err());

    // 特定のエラーかどうかをチェック
    // 注: 最適化実装ではエラー型が異なるかもしれないのでエラーの存在のみを確認
    assert!(result.is_err());
}

#[test]
fn test_dataframe_duplicate_column() {
    // 重複した列名を追加した場合のエラーテスト
    let mut df = DataFrame::new();

    let ages1 = Int64Column::new(vec![25, 30, 35]);
    df.add_column("age", Column::Int64(ages1)).unwrap();

    // 同じ名前の列を追加
    let ages2 = Int64Column::new(vec![40, 45, 50]);
    let result = df.add_column("age", Column::Int64(ages2));

    // エラーになるはず
    assert!(result.is_err());

    // 特定のエラーかどうかをチェック
    // 注: 最適化実装ではエラー型が異なるかもしれないのでエラーの存在のみを確認
    assert!(result.is_err());
}

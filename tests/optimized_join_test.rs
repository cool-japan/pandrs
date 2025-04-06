use pandrs::{OptimizedDataFrame, Column, Int64Column, StringColumn};
use pandrs::error::Result;

#[test]
fn test_inner_join() -> Result<()> {
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;
    
    // 名前列
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;
    
    // 値列
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;
    
    // 内部結合
    let joined = left_df.inner_join(&right_df, "id", "id")?;
    
    // 検証 - 内部結合なので一致する行のみ（id=1と2）
    assert_eq!(joined.row_count(), 2);
    assert_eq!(joined.column_count(), 3); // id, name, value
    
    Ok(())
}

#[test]
fn test_left_join() -> Result<()> {
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;
    
    // 名前列
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;
    
    // 値列
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;
    
    // 左結合
    let joined = left_df.left_join(&right_df, "id", "id")?;
    
    // 検証 - 左結合なので左側の全行（id=1,2,3,4）があり、それに一致する右側の行
    assert_eq!(joined.row_count(), 4);
    assert_eq!(joined.column_count(), 3); // id, name, value
    
    Ok(())
}

#[test]
fn test_right_join() -> Result<()> {
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;
    
    // 名前列
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;
    
    // 値列
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;
    
    // 右結合
    let joined = left_df.right_join(&right_df, "id", "id")?;
    
    // 検証 - 右結合なので右側の全行（id=1,2,5,6）があり、それに一致する左側の行
    assert_eq!(joined.row_count(), 4);
    assert_eq!(joined.column_count(), 3); // id, name, value
    
    Ok(())
}

#[test]
fn test_outer_join() -> Result<()> {
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;
    
    // 名前列
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;
    
    // 値列
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;
    
    // 外部結合
    let joined = left_df.outer_join(&right_df, "id", "id")?;
    
    // 検証 - 外部結合なので全ての行が含まれる（id=1,2,3,4,5,6）
    assert_eq!(joined.row_count(), 6);
    assert_eq!(joined.column_count(), 3); // id, name, value
    
    Ok(())
}

#[test]
fn test_join_different_column_names() -> Result<()> {
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // 左側ID列
    let left_id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("left_id", Column::Int64(left_id_col))?;
    
    // 名前列
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // 右側ID列
    let right_id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("right_id", Column::Int64(right_id_col))?;
    
    // 値列
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;
    
    // 異なる列名で内部結合
    let joined = left_df.inner_join(&right_df, "left_id", "right_id")?;
    
    // 検証 - 内部結合なので一致する行のみ（id=1と2）
    assert_eq!(joined.row_count(), 2);
    assert_eq!(joined.column_count(), 3); // left_id, name, value
    
    Ok(())
}

#[test]
fn test_empty_join() -> Result<()> {
    // 空のDataFrameの結合
    let empty_df = OptimizedDataFrame::new();
    let result = empty_df.inner_join(&empty_df, "id", "id");
    
    // 存在しない列で結合しようとしているのでエラーになるはず
    assert!(result.is_err());
    
    // 左側のデータフレーム
    let mut left_df = OptimizedDataFrame::new();
    
    // ID列
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;
    
    // 右側のデータフレーム
    let mut right_df = OptimizedDataFrame::new();
    
    // 一致しないID列
    let id_col = Int64Column::new(vec![5, 6, 7, 8]);
    right_df.add_column("id", Column::Int64(id_col))?;
    
    // 内部結合（一致する行がない）
    let joined = left_df.inner_join(&right_df, "id", "id")?;
    
    // 検証 - 一致する行がないので空のデータフレーム
    assert_eq!(joined.row_count(), 0);
    // 実装によっては空のDataFrameになるか、列だけを含むDataFrameになるか異なるため
    // ここでは行数が0であることだけを確認
    
    Ok(())
}
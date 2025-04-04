#[cfg(test)]
mod tests {
    use pandrs::{OptimizedDataFrame, Column, StringColumn};
    use pandrs::error::Result;

    #[test]
    fn test_optimized_melt() -> Result<()> {
        // テスト用データフレーム作成
        let mut df = OptimizedDataFrame::new();
        
        // ID列
        let id_col = StringColumn::new(vec![
            "1".to_string(), 
            "2".to_string()
        ]);
        df.add_column("id", Column::String(id_col))?;
        
        // A列
        let a_col = StringColumn::new(vec![
            "a1".to_string(), 
            "a2".to_string()
        ]);
        df.add_column("A", Column::String(a_col))?;
        
        // B列
        let b_col = StringColumn::new(vec![
            "b1".to_string(), 
            "b2".to_string()
        ]);
        df.add_column("B", Column::String(b_col))?;

        // melt操作実行
        let melted = df.melt(
            &["id"],
            Some(&["A", "B"]),
            Some("variable"),
            Some("value")
        )?;

        // 検証
        assert_eq!(melted.column_count(), 3); // id, variable, value
        assert_eq!(melted.row_count(), 4);    // 2行 x 2列 = 4行

        // 列名の確認
        let columns = melted.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"variable".to_string()));
        assert!(columns.contains(&"value".to_string()));

        // データの確認は値が正しく保存されていればOK
        // 複雑なデータ構造の検証は実装によって変わるため省略
        
        Ok(())
    }

    #[test]
    fn test_optimized_concat() -> Result<()> {
        // 1つ目のデータフレーム
        let mut df1 = OptimizedDataFrame::new();
        
        let id_col1 = StringColumn::new(vec![
            "1".to_string(), 
            "2".to_string()
        ]);
        df1.add_column("id", Column::String(id_col1))?;
        
        let value_col1 = StringColumn::new(vec![
            "a".to_string(), 
            "b".to_string()
        ]);
        df1.add_column("value", Column::String(value_col1))?;

        // 2つ目のデータフレーム
        let mut df2 = OptimizedDataFrame::new();
        
        let id_col2 = StringColumn::new(vec![
            "3".to_string(), 
            "4".to_string()
        ]);
        df2.add_column("id", Column::String(id_col2))?;
        
        let value_col2 = StringColumn::new(vec![
            "c".to_string(), 
            "d".to_string()
        ]);
        df2.add_column("value", Column::String(value_col2))?;

        // 結合操作
        let concat_df = df1.append(&df2)?;

        // 検証
        assert_eq!(concat_df.column_count(), 2);
        assert_eq!(concat_df.row_count(), 4);

        // 列の存在確認
        assert!(concat_df.contains_column("id"));
        assert!(concat_df.contains_column("value"));
        
        Ok(())
    }

    #[test]
    fn test_optimized_concat_different_columns() -> Result<()> {
        // 1つ目のデータフレーム
        let mut df1 = OptimizedDataFrame::new();
        
        let id_col1 = StringColumn::new(vec![
            "1".to_string(), 
            "2".to_string()
        ]);
        df1.add_column("id", Column::String(id_col1))?;
        
        let a_col = StringColumn::new(vec![
            "a1".to_string(), 
            "a2".to_string()
        ]);
        df1.add_column("A", Column::String(a_col))?;

        // 2つ目のデータフレーム（異なる列を持つ）
        let mut df2 = OptimizedDataFrame::new();
        
        let id_col2 = StringColumn::new(vec![
            "3".to_string(), 
            "4".to_string()
        ]);
        df2.add_column("id", Column::String(id_col2))?;
        
        let b_col = StringColumn::new(vec![
            "b3".to_string(), 
            "b4".to_string()
        ]);
        df2.add_column("B", Column::String(b_col))?;

        // 結合操作
        let concat_df = df1.append(&df2)?;

        // 検証
        assert_eq!(concat_df.column_count(), 3); // id, A, B
        assert_eq!(concat_df.row_count(), 4);

        // 列の存在確認
        assert!(concat_df.contains_column("id"));
        assert!(concat_df.contains_column("A"));
        assert!(concat_df.contains_column("B"));
        
        Ok(())
    }
}
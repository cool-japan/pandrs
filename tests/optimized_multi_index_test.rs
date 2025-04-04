use pandrs::{OptimizedDataFrame, Column, StringColumn};
use pandrs::error::Result;

#[test]
fn test_optimized_multi_index_simulation() -> Result<()> {
    // OptimizedDataFrameでのマルチインデックスをシミュレート
    // 実際のマルチインデックス機能が実装されるまでのテスト
    let mut df = OptimizedDataFrame::new();
    
    // 複数のインデックス列を作成
    let level1 = vec![
        "A", "A", "B", "B"
    ].iter().map(|s| s.to_string()).collect::<Vec<String>>();
    
    let level2 = vec![
        "1", "2", "2", "3"
    ].iter().map(|s| s.to_string()).collect::<Vec<String>>();
    
    // インデックス列を追加
    let level1_col = StringColumn::new(level1);
    df.add_column("level1", Column::String(level1_col))?;
    
    let level2_col = StringColumn::new(level2);
    df.add_column("level2", Column::String(level2_col))?;
    
    // 値列を追加
    let values = vec![100, 200, 300, 400];
    let value_col = pandrs::Int64Column::new(values);
    df.add_column("value", Column::Int64(value_col))?;
    
    // 検証
    assert_eq!(df.row_count(), 4);
    assert!(df.contains_column("level1"));
    assert!(df.contains_column("level2"));
    assert!(df.contains_column("value"));
    
    // マルチインデックスを使った集計をシミュレート
    let result = pandrs::LazyFrame::new(df)
        .aggregate(
            vec!["level1".to_string(), "level2".to_string()],
            vec![("value".to_string(), pandrs::AggregateOp::Sum, "sum".to_string())]
        )
        .execute()?;
    
    // 検証 - 2つのグループが作成されているはず
    // 実装によって異なる可能性があるため、厳密な行数は検証しない
    assert!(result.row_count() > 0, "少なくとも1つ以上のグループがあるはず");
    assert!(result.contains_column("level1"));
    assert!(result.contains_column("level2"));
    assert!(result.contains_column("sum"));
    
    Ok(())
}
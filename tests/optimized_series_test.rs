use pandrs::{OptimizedDataFrame, Column, Int64Column, Float64Column};
use pandrs::error::Result;

#[test]
fn test_optimized_series_representation() -> Result<()> {
    // OptimizedDataFrameは列志向であり、Seriesは列として実装される
    let mut df = OptimizedDataFrame::new();
    
    // 整数列を作成
    let values = vec![1, 2, 3, 4, 5];
    let int_col = Int64Column::new(values);
    df.add_column("test", Column::Int64(int_col))?;
    
    // 検証
    assert_eq!(df.row_count(), 5);
    assert!(df.contains_column("test"));
    
    // 列アクセスの確認
    let col = df.column("test")?;
    if let Some(int64_col) = col.as_int64() {
        assert_eq!(int64_col.get(0)?, Some(1));
        assert_eq!(int64_col.get(4)?, Some(5));
        assert!(int64_col.get(5).is_err()); // 範囲外アクセス
    } else {
        panic!("整数列として取得できません");
    }
    
    Ok(())
}

#[test]
fn test_optimized_series_numeric_operations() -> Result<()> {
    // OptimizedDataFrameでの数値操作を確認
    let mut df = OptimizedDataFrame::new();
    
    // 整数列を作成
    let values = vec![1, 2, 3, 4, 5];
    let int_col = Int64Column::new(values);
    df.add_column("int_values", Column::Int64(int_col))?;
    
    // 浮動小数点列を作成
    let float_values = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let float_col = Float64Column::new(float_values);
    df.add_column("float_values", Column::Float64(float_col))?;
    
    // 整数列の集計操作を確認
    let int_series = df.column("int_values")?;
    if let Some(int64_col) = int_series.as_int64() {
        // 合計
        assert_eq!(int64_col.sum(), 15);
        
        // 平均
        assert_eq!(int64_col.mean().unwrap_or(0.0), 3.0);
        
        // 最小値
        assert_eq!(int64_col.min().unwrap_or(0), 1);
        
        // 最大値
        assert_eq!(int64_col.max().unwrap_or(0), 5);
    } else {
        panic!("整数列として取得できません");
    }
    
    // 浮動小数点列の集計操作を確認
    let float_series = df.column("float_values")?;
    if let Some(float64_col) = float_series.as_float64() {
        // 合計
        assert!((float64_col.sum() - 17.5).abs() < 0.001);
        
        // 平均
        assert!((float64_col.mean().unwrap_or(0.0) - 3.5).abs() < 0.001);
        
        // 最小値
        assert!((float64_col.min().unwrap_or(0.0) - 1.5).abs() < 0.001);
        
        // 最大値
        assert!((float64_col.max().unwrap_or(0.0) - 5.5).abs() < 0.001);
    } else {
        panic!("浮動小数点列として取得できません");
    }
    
    Ok(())
}
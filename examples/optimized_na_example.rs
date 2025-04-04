use pandrs::{OptimizedDataFrame, Column, Int64Column, Float64Column, StringColumn};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("=== 最適化版 欠損値のサポート ===\n");

    // 最適化されたDataFrameを作成
    let mut df = OptimizedDataFrame::new();
    
    // 欠損値を含む整数列を作成（OptimizedDataFrameでは欠損値の扱いが異なる）
    // Int64Columnの場合、欠損値はnull_maskを使用
    let int_values = vec![10, 20, 0, 40, 0];
    let int_nulls = vec![false, false, true, false, true]; // インデックス2と4をNULLに
    let int_data = Int64Column::with_nulls(int_values, int_nulls);
    
    df.add_column("numbers", Column::Int64(int_data))?;
    
    // 浮動小数点列にも欠損値を設定
    let float_values = vec![1.1, 2.2, 0.0, 4.4, 0.0];
    let float_nulls = vec![false, false, true, false, true]; // インデックス2と4をNULLに
    let float_data = Float64Column::with_nulls(float_values, float_nulls);
    
    df.add_column("floats", Column::Float64(float_data))?;
    
    // 文字列列にも欠損値を設定
    let string_values = vec![
        "a".to_string(),
        "b".to_string(),
        "".to_string(),
        "d".to_string(),
        "".to_string()
    ];
    let string_nulls = vec![false, false, true, false, true]; // インデックス2と4をNULLに
    let string_data = StringColumn::with_nulls(string_values, string_nulls);
    
    df.add_column("strings", Column::String(string_data))?;
    
    println!("欠損値を含むDataFrame:");
    println!("{:?}", df);
    
    // 列の操作（整数列）
    println!("\n--- 整数列の欠損値 ---");
    let age_col = df.column("numbers")?;
    if let Some(int64_col) = age_col.as_int64() {
        // 検証
        for i in 0..5 {
            let value_result = int64_col.get(i)?;
            let value_str = match value_result {
                Some(val) => val.to_string(),
                None => "NULL".to_string(),
            };
            println!("位置 {}: {}", i, value_str);
        }
        
        // 集計関数（欠損値は無視される）
        println!("合計（NAは無視）: {}", int64_col.sum());
        println!("平均（NAは無視）: {:.2}", int64_col.mean().unwrap_or(0.0));
        println!("最小値（NAは無視）: {}", int64_col.min().unwrap_or(0));
        println!("最大値（NAは無視）: {}", int64_col.max().unwrap_or(0));
    }
    
    // 列の操作（浮動小数点列）
    println!("\n--- 浮動小数点列の欠損値 ---");
    let float_col = df.column("floats")?;
    if let Some(float64_col) = float_col.as_float64() {
        // 検証
        for i in 0..5 {
            let value_result = float64_col.get(i)?;
            let value_str = match value_result {
                Some(val) => format!("{:.1}", val),
                None => "NULL".to_string(),
            };
            println!("位置 {}: {}", i, value_str);
        }
        
        // 集計関数（欠損値は無視される）
        println!("合計（NAは無視）: {:.1}", float64_col.sum());
        println!("平均（NAは無視）: {:.2}", float64_col.mean().unwrap_or(0.0));
        println!("最小値（NAは無視）: {:.1}", float64_col.min().unwrap_or(0.0));
        println!("最大値（NAは無視）: {:.1}", float64_col.max().unwrap_or(0.0));
    }
    
    // 列の操作（文字列列）
    println!("\n--- 文字列列の欠損値 ---");
    let string_col = df.column("strings")?;
    if let Some(str_col) = string_col.as_string() {
        // 検証
        for i in 0..5 {
            let value_result = str_col.get(i)?;
            let value_str = match value_result {
                Some(val) => val.to_string(),
                None => "NULL".to_string(),
            };
            println!("位置 {}: {}", i, value_str);
        }
    }
    
    // フィルタリング - 欠損値がない行だけを選択するためのブール列を作成
    println!("\n--- 欠損値のフィルタリング ---");
    
    // 検証
    println!("最適化版DataFrameでのフィルタリングは実装により異なりますが、");
    println!("通常、列ごとにnull_maskを使用して欠損値の有無をチェックできます。");
    
    println!("=== NA値サンプル完了 ===");
    Ok(())
}
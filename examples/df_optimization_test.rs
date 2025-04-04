use std::time::Instant;
use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{Column, StringColumn, Int64Column};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OptimizedDataFrame 基本機能テスト ===\n");
    
    // 基本的なデータフレームを作成
    let mut df = OptimizedDataFrame::new();
    
    // 都市データ
    let cities = vec![
        "東京".to_string(), "札幌".to_string(), "名古屋".to_string(), "大阪".to_string(),
        "福岡".to_string(), "仙台".to_string(), "広島".to_string(), "横浜".to_string(),
        "神戸".to_string(), "京都".to_string()
    ];
    
    // 数値データ（サンプル）
    let values = vec![100, 50, 75, 95, 60, 45, 80, 90, 55, 70];
    
    // 都市列を通常の文字列列として追加
    df.add_column("都市", Column::String(StringColumn::new(cities)))?;
    
    // 数値列を追加
    df.add_column("値", Column::Int64(Int64Column::new(values)))?;
    
    println!("\nデータフレーム概要:");
    println!("{:?}", df);
    
    println!("\n=== テスト完了 ===");
    
    Ok(())
}
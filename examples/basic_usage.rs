
use pandrs::{DataFrame, Series, PandRSError};

fn main() -> Result<(), PandRSError> {
    println!("=== PandRS 基本使用例 ===");
    
    // Series の作成
    let ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
    let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;
    let names = Series::new(
        vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()],
        Some("name".to_string())
    )?;
    
    println!("年齢シリーズ: {:?}", ages);
    println!("身長シリーズ: {:?}", heights);
    println!("名前シリーズ: {:?}", names);
    
    // 数値シリーズの統計
    println!("\n=== 年齢シリーズの統計 ===");
    println!("合計: {}", ages.sum());
    println!("平均: {}", ages.mean()?);
    println!("最小: {}", ages.min()?);
    println!("最大: {}", ages.max()?);
    
    // DataFrame の作成
    println!("\n=== DataFrame の作成 ===");
    let mut df = DataFrame::new();
    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;
    df.add_column("height".to_string(), heights)?;
    
    println!("DataFrame: {:?}", df);
    println!("列数: {}", df.column_count());
    println!("行数: {}", df.row_count());
    println!("列名: {:?}", df.column_names());
    
    // CSV への保存と読み込みをテスト
    let file_path = "example_data.csv";
    df.to_csv(file_path)?;
    println!("\nCSVファイルに保存しました: {}", file_path);
    
    // CSVから読み込みテスト（まだ完全には実装されていないかもしれません）
    match DataFrame::from_csv(file_path, true) {
        Ok(loaded_df) => {
            println!("CSVからロードしたDataFrame: {:?}", loaded_df);
            println!("列数: {}", loaded_df.column_count());
            println!("行数: {}", loaded_df.row_count());
            println!("列名: {:?}", loaded_df.column_names());
        },
        Err(e) => {
            println!("CSVの読み込みに失敗しました: {:?}", e);
        }
    }
    
    println!("\n=== サンプル完了 ===");
    Ok(())
}
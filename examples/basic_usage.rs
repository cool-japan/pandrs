use pandrs::{DataFrame, PandRSError, Column, Int64Column, Float64Column, StringColumn};

fn main() -> Result<(), PandRSError> {
    println!("=== PandRS 基本使用例 ===");

    // DataFrame の作成
    println!("\n=== DataFrame の作成 ===");
    let mut df = DataFrame::new();
    
    // 各列のデータを準備
    let names = vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
    ];
    let ages = vec![30, 25, 40];
    let heights = vec![180, 175, 182];
    
    // 列を追加
    let name_column = StringColumn::new(names);
    let age_column = Int64Column::new(ages);
    let height_column = Int64Column::new(heights);
    
    df.add_column("name", Column::String(name_column))?;
    df.add_column("age", Column::Int64(age_column))?;
    df.add_column("height", Column::Int64(height_column))?;

    // 列情報の表示
    println!("DataFrame: {:?}", df);
    println!("列数: {}", df.column_count());
    println!("行数: {}", df.row_count());
    println!("列名: {:?}", df.column_names());
    
    // 列の統計情報取得
    println!("\n=== 年齢列の統計 ===");
    let age_col = df.column("age")?;
    if let Some(int_col) = age_col.as_int64() {
        println!("合計: {}", int_col.sum());
        println!("平均: {}", int_col.mean().unwrap_or(0.0));
        println!("最小: {}", int_col.min().unwrap_or(0));
        println!("最大: {}", int_col.max().unwrap_or(0));
    }

    // CSV への保存と読み込みをテスト
    let file_path = "example_data.csv";
    
    // CSVに保存 (io モジュールを使用)
    match pandrs::io::csv::write_csv(&df, file_path) {
        Ok(_) => println!("\nCSVファイルに保存しました: {}", file_path),
        Err(e) => println!("CSVファイルの保存に失敗しました: {:?}", e),
    }

    // CSVから読み込み
    match pandrs::io::csv::read_csv(file_path, true) {
        Ok(loaded_df) => {
            println!("CSVからロードしたDataFrame: {:?}", loaded_df);
            println!("列数: {}", loaded_df.column_count());
            println!("行数: {}", loaded_df.row_count());
            println!("列名: {:?}", loaded_df.column_names());
        }
        Err(e) => {
            println!("CSVの読み込みに失敗しました: {:?}", e);
        }
    }

    println!("\n=== サンプル完了 ===");
    Ok(())
}

use pandrs::{OptimizedDataFrame, Column, Int64Column, Float64Column, StringColumn, LazyFrame};
use pandrs::error::Error;

fn main() -> Result<(), Error> {
    println!("=== PandRS 最適化版 基本使用例 ===");

    // 最適化されたDataFrameの作成
    println!("\n=== DataFrame の作成 ===");
    let mut df = OptimizedDataFrame::new();
    
    // 整数列の作成と追加
    let ages = Int64Column::new(vec![30, 25, 40]);
    df.add_column("age", Column::Int64(ages))?;
    
    // 浮動小数点列の作成と追加
    let heights = Float64Column::new(vec![180.0, 175.0, 182.0]);
    df.add_column("height", Column::Float64(heights))?;
    
    // 文字列列の作成と追加
    let names = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
    ]);
    df.add_column("name", Column::String(names))?;

    println!("DataFrame: {:?}", df);
    println!("列数: {}", df.column_count());
    println!("行数: {}", df.row_count());
    println!("列名: {:?}", df.column_names());

    // 列の操作
    println!("\n=== 年齢列の統計 ===");
    let age_col = df.column("age")?;
    if let Some(int_col) = age_col.as_int64() {
        println!("合計: {}", int_col.sum());
        println!("平均: {:.2}", int_col.mean().unwrap_or(0.0));
        println!("最小: {}", int_col.min().unwrap_or(0));
        println!("最大: {}", int_col.max().unwrap_or(0));
    }

    // CSV への保存と読み込みをテスト
    let file_path = "optimized_example_data.csv";
    df.to_csv(file_path, true)?;
    println!("\nCSVファイルに保存しました: {}", file_path);

    // CSVから読み込み
    match OptimizedDataFrame::from_csv(file_path, true) {
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

    // 遅延評価の例
    println!("\n=== 遅延評価の例 ===");
    let lazy_df = LazyFrame::new(df);
    
    // 名前が "Alice" または "Bob" である行のみを選択
    let result = lazy_df
        .select(&["name", "age", "height"])
        .execute()?;
    
    println!("選択した列のみのDataFrame: {:?}", result);

    println!("\n=== サンプル完了 ===");
    Ok(())
}
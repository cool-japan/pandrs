use std::time::Instant;
use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{Column, StringColumn, ColumnTrait};

fn main() {
    println!("=== 最適化されたDataFrameの拡張機能テスト ===\n");
    
    // テスト用CSVファイルの作成
    create_test_csv(100000);
    
    // CSVファイルからDataFrameを読み込み
    let start = Instant::now();
    let df = OptimizedDataFrame::from_csv("test_data.csv", true).unwrap();
    let duration = start.elapsed();
    println!("CSV読み込み時間: {:?}", duration);
    println!("行数: {}, 列数: {}", df.row_count(), df.column_count());
    
    // melt操作のテスト
    let start = Instant::now();
    let melted_df = df.melt(
        &["id"],
        Some(&["name", "age", "score"]),
        Some("variable"),
        Some("value")
    ).unwrap();
    let duration = start.elapsed();
    println!("\nmelt操作時間: {:?}", duration);
    println!("変換後の行数: {}, 列数: {}", melted_df.row_count(), melted_df.column_count());
    
    // apply操作のテスト
    let start = Instant::now();
    let applied_df = df.apply(|col| {
        if col.column_type() == pandrs::column::ColumnType::String {
            // 文字列列の場合はすべて大文字に変換
            if let Some(str_col) = col.as_string() {
                let mut new_data = Vec::with_capacity(str_col.len());
                for i in 0..str_col.len() {
                    if let Ok(Some(val)) = str_col.get(i) {
                        new_data.push(val.to_uppercase());
                    } else {
                        new_data.push(String::new());
                    }
                }
                Ok(Column::String(StringColumn::new(new_data)))
            } else {
                // 型が一致しない場合（ありえないが）
                Ok(col.column().clone())
            }
        } else {
            // その他の列はそのまま
            Ok(col.column().clone())
        }
    }, Some(&["name"])).unwrap();
    let duration = start.elapsed();
    println!("\napply操作時間: {:?}", duration);
    println!("処理後の行数: {}, 列数: {}", applied_df.row_count(), applied_df.column_count());
    
    // CSV書き出しのテスト
    let start = Instant::now();
    applied_df.to_csv("test_output.csv", true).unwrap();
    let duration = start.elapsed();
    println!("\nCSV書き出し時間: {:?}", duration);
    
    println!("\n=== 最適化されたDataFrame機能テスト完了 ===");
}

// テスト用のCSVファイルを作成する関数
fn create_test_csv(rows: usize) {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    
    println!("テスト用CSVファイル作成中 ({} 行)...", rows);
    
    let file = File::create("test_data.csv").unwrap();
    let mut writer = BufWriter::new(file);
    
    // ヘッダー
    writeln!(writer, "id,name,age,score,category").unwrap();
    
    // データ生成
    let categories = ["A", "B", "C", "D", "E"];
    let names = ["Alice", "Bob", "Charlie", "David", "Emma", 
                "Frank", "Grace", "Hannah", "Ian", "Julia"];
    
    for i in 0..rows {
        let name = names[i % names.len()];
        let age = 20 + (i % 50);
        let score = (i % 100) as f64 / 10.0;
        let category = categories[i % categories.len()];
        
        writeln!(writer, "{},{},{},{},{}", i, name, age, score, category).unwrap();
    }
    
    println!("テスト用CSVファイル作成完了");
}
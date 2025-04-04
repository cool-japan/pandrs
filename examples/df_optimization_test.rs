use std::time::Instant;
use pandrs::OptimizedDataFrame;
use pandrs::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn};
use std::sync::Arc;

fn main() {
    println!("DataFrame最適化テスト");
    println!("============================");
    
    // データサイズ
    const ROWS: usize = 1_000_000;
    
    // --- データ生成部分 ---
    println!("\n[1] データ生成");
    let data_gen_start = Instant::now();
    
    let mut int_data = Vec::with_capacity(ROWS);
    let mut float_data = Vec::with_capacity(ROWS);
    let mut str_data = Vec::with_capacity(ROWS);
    let mut bool_data = Vec::with_capacity(ROWS);
    
    for i in 0..ROWS {
        int_data.push(i as i64);
        float_data.push(i as f64 / 100.0);
        str_data.push(format!("value_{}", i % 1000)); // 限定的な文字列セット
        bool_data.push(i % 2 == 0);
    }
    
    let data_gen_time = data_gen_start.elapsed();
    println!("データ生成時間: {:?}", data_gen_time);
    
    // --- データ処理テスト ---
    println!("\n[2] 通常のDataFrame作成");
    
    // 通常DataFrame構築
    {
        let df_start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        df.add_column("category".to_string(), Column::String(StringColumn::new(str_data.clone()))).unwrap();
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let df_time = df_start.elapsed();
        println!("通常の追加方法でのDataFrame作成時間: {:?}", df_time);
        println!("行数: {}, 列数: {}", df.row_count(), df.column_names().len());
    }
    
    // 疑似カテゴリカル列
    {
        println!("\n[3] カテゴリカル最適化DataFrame作成");
        let df_start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        
        // 文字列をi64にエンコードした疑似カテゴリカル列
        let convert_start = Instant::now();
        let mut str_to_idx = std::collections::HashMap::new();
        let mut next_idx = 0i64;
        let cat_data: Vec<i64> = str_data.iter().map(|s| {
            if let Some(&idx) = str_to_idx.get(s) {
                idx
            } else {
                let idx = next_idx;
                str_to_idx.insert(s.clone(), idx);
                next_idx += 1;
                idx
            }
        }).collect();
        let convert_time = convert_start.elapsed();
        println!("文字列→カテゴリカル変換時間: {:?}", convert_time);
        println!("ユニーク文字列数: {}", str_to_idx.len());
        
        let add_start = Instant::now();
        df.add_column("category".to_string(), Column::Int64(Int64Column::new(cat_data))).unwrap();
        let add_time = add_start.elapsed();
        println!("カテゴリカル列追加時間: {:?}", add_time);
        
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let df_time = df_start.elapsed();
        println!("カテゴリカル最適化版のDataFrame作成時間: {:?}", df_time);
    }
    
    // シミュレートされた共有文字列プール
    {
        println!("\n[4] 共有文字列プールシミュレーション");
        
        // グローバル文字列プールの作成をシミュレート（実際の実装では静的変数などで効率化）
        let pool_start = Instant::now();
        let mut shared_strings = Vec::with_capacity(1000); // 1000個のユニーク文字列を想定
        let mut str_to_idx = std::collections::HashMap::with_capacity(1000);
        
        // 共有プール用ベクターにユニーク文字列を格納
        for s in str_data.iter() {
            if !str_to_idx.contains_key(s) {
                str_to_idx.insert(s.clone(), shared_strings.len());
                shared_strings.push(s.clone());
            }
        }
        
        // 共有されたArc<Vec<String>>を作成
        let shared_pool = Arc::new(shared_strings);
        let pool_time = pool_start.elapsed();
        println!("共有文字列プール作成時間: {:?}", pool_time);
        println!("プール内のユニーク文字列数: {}", shared_pool.len());
        
        // インデックスベクターを構築（これが実際の列データになる）
        let indices_start = Instant::now();
        let indices: Vec<usize> = str_data.iter().map(|s| {
            *str_to_idx.get(s).unwrap()
        }).collect();
        let indices_time = indices_start.elapsed();
        println!("インデックス変換時間: {:?}", indices_time);
        
        // これでIntegerColumnを使って列を追加できる（メモリ効率が良い）
        // シミュレーションのため実際のDataFrameには追加しない
        println!("インデックス列の長さ: {}", indices.len());
        println!("理論上のメモリ削減量: {}MB → {}MB", 
                 str_data.iter().map(|s| s.len()).sum::<usize>() / (1024*1024),
                 (indices.len() * 4) / (1024*1024));
    }
    
    println!("\n最適化テスト完了");
}
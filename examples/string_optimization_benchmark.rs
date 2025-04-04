use std::time::Instant;
use pandrs::OptimizedDataFrame;
use pandrs::column::{Column, Int64Column, Float64Column, StringColumn, BooleanColumn, StringColumnOptimizationMode};
// すべての最適化モードにアクセスするためのインポート
use pandrs::column::string_column_impl::{StringColumnOptimizationMode as OptMode, DEFAULT_OPTIMIZATION_MODE};

fn main() {
    println!("文字列列最適化ベンチマーク");
    println!("============================");
    
    // データサイズ
    const ROWS: usize = 1_000_000;
    
    // --- データ生成 ---
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
    
    // 各最適化モードで文字列列の作成をベンチマーク
    println!("\n[2] 文字列列作成ベンチマーク");
    
    // レガシーモード
    {
        let start = Instant::now();
        let _column = StringColumn::new_legacy(str_data.clone());
        let time = start.elapsed();
        println!("レガシーモード作成時間: {:?}", time);
    }
    
    // グローバルプールモード
    {
        let start = Instant::now();
        let _column = StringColumn::new_with_global_pool(str_data.clone());
        let time = start.elapsed();
        println!("グローバルプールモード作成時間: {:?}", time);
    }
    
    // カテゴリカルモード
    {
        let start = Instant::now();
        let _column = StringColumn::new_categorical(str_data.clone());
        let time = start.elapsed();
        println!("カテゴリカルモード作成時間: {:?}", time);
    }
    
    // 最適化実装
    {
        let start = Instant::now();
        let _column = StringColumn::new_categorical(str_data.clone());
        let time = start.elapsed();
        println!("最適化実装作成時間: {:?}", time);
    }
    
    // 各モードでDataFrameの作成をベンチマーク
    println!("\n[3] 各最適化モードでのDataFrame作成ベンチマーク");
    
    // レガシーモード
    {
        // レガシーモードに設定
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::Legacy;
        }
        
        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        df.add_column("category".to_string(), Column::String(StringColumn::new(str_data.clone()))).unwrap();
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let time = start.elapsed();
        println!("レガシーモードDataFrame作成時間: {:?}", time);
    }
    
    // グローバルプールモード
    {
        // グローバルプールモードに設定
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::GlobalPool;
        }
        
        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        df.add_column("category".to_string(), Column::String(StringColumn::new(str_data.clone()))).unwrap();
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let time = start.elapsed();
        println!("グローバルプールモードDataFrame作成時間: {:?}", time);
    }
    
    // カテゴリカルモード
    {
        // カテゴリカルモードに設定
        unsafe {
            DEFAULT_OPTIMIZATION_MODE = OptMode::Categorical;
        }
        
        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        df.add_column("category".to_string(), Column::String(StringColumn::new(str_data.clone()))).unwrap();
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let time = start.elapsed();
        println!("カテゴリカルモードDataFrame作成時間: {:?}", time);
    }
    
    // 最適化実装
    {
        let start = Instant::now();
        let mut df = OptimizedDataFrame::new();
        df.add_column("id".to_string(), Column::Int64(Int64Column::new(int_data.clone()))).unwrap();
        df.add_column("value".to_string(), Column::Float64(Float64Column::new(float_data.clone()))).unwrap();
        df.add_column("category".to_string(), Column::String(StringColumn::new_categorical(str_data.clone()))).unwrap();
        df.add_column("flag".to_string(), Column::Boolean(BooleanColumn::new(bool_data.clone()))).unwrap();
        
        let time = start.elapsed();
        println!("最適化実装DataFrame作成時間: {:?}", time);
    }
    
    println!("\n文字列列最適化ベンチマーク完了");
}
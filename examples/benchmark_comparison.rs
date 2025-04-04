// ベンチマーク比較: PandRSの従来実装と最適化実装の比較
// このファイルは、PandRSの従来実装と最適化実装のパフォーマンスを比較するためのベンチマークを提供します。

use pandrs::{DataFrame, Series};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// プロトタイプの型をインポート
mod prototype {
    include!("column_prototype.rs");
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}.{:03}s", duration.as_secs(), duration.subsec_millis())
    } else if duration.as_millis() > 0 {
        format!("{}.{:03}ms", duration.as_millis(), duration.as_micros() % 1000)
    } else {
        format!("{}µs", duration.as_micros())
    }
}

// ベンチマーク関数
fn bench<F, T>(name: &str, f: F) -> (Duration, T)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();
    println!("{}: {}", name, format_duration(duration));
    (duration, result)
}

fn run_benchmark_suite() {
    // ヘッダー
    println!("\n=== PandRS パフォーマンス最適化ベンチマーク ===\n");
    
    // ベンチマークのデータサイズ
    let sizes = [1000, 10_000, 100_000, 1_000_000];
    
    for &size in &sizes {
        println!("\n## データサイズ: {}行 ##", size);
        
        // データ準備
        let int_data: Vec<i32> = (0..size).collect();
        let float_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
        let string_data: Vec<String> = (0..size).map(|i| format!("val_{}", i % 100)).collect();
        
        // 旧実装: Series作成
        let (legacy_series_time, (legacy_int_series, legacy_float_series, legacy_string_series)) = bench("旧実装 - Series作成", || {
            let int_series = Series::new(int_data.clone(), Some("int_col".to_string())).unwrap();
            let float_series = Series::new(float_data.clone(), Some("float_col".to_string())).unwrap();
            let string_series = Series::new(string_data.clone(), Some("string_col".to_string())).unwrap();
            (int_series, float_series, string_series)
        });
        
        // 新実装: 最適化列作成
        let (optimized_series_time, (opt_int_col, opt_float_col, opt_string_col)) = bench("新実装 - 列作成", || {
            let int_col = prototype::Int64Column::new(int_data.iter().map(|&i| i as i64).collect()).with_name("int_col");
            let float_col = prototype::Float64Column::new(float_data.clone()).with_name("float_col");
            let string_col = prototype::StringColumn::new(string_data.clone()).with_name("string_col");
            (int_col, float_col, string_col)
        });
        
        // 旧実装: DataFrame作成
        let (legacy_df_time, legacy_df) = bench("旧実装 - DataFrame作成", || {
            let mut df = DataFrame::new();
            df.add_column("int_col".to_string(), legacy_int_series.clone()).unwrap();
            df.add_column("float_col".to_string(), legacy_float_series.clone()).unwrap();
            df.add_column("string_col".to_string(), legacy_string_series.clone()).unwrap();
            df
        });
        
        // 新実装: OptimizedDataFrame作成
        let (optimized_df_time, optimized_df) = bench("新実装 - DataFrame作成", || {
            let mut df = prototype::OptimizedDataFrame::new();
            df.add_column("int_col", opt_int_col.clone()).unwrap();
            df.add_column("float_col", opt_float_col.clone()).unwrap();
            df.add_column("string_col", opt_string_col.clone()).unwrap();
            df
        });
        
        // 旧実装: DataFrame集計操作
        let (legacy_agg_time, _) = bench("旧実装 - 集計操作", || {
            // 旧実装ではDataBoxを経由した数値操作のため効率が低い
            // 旧実装では数値操作のために文字列変換が必要
            let int_values = legacy_df.get_column_string_values("int_col").unwrap();
            let float_values = legacy_df.get_column_string_values("float_col").unwrap();
            
            // 文字列から数値への変換
            let int_numeric: Vec<i32> = int_values.iter()
                .filter_map(|s| s.parse::<i32>().ok())
                .collect();
                
            let float_numeric: Vec<f64> = float_values.iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();
                
            // 集計計算
            let int_sum: i32 = int_numeric.iter().sum();
            let int_mean = int_sum as f64 / int_numeric.len() as f64;
            
            let float_sum: f64 = float_numeric.iter().sum();
            let float_mean = float_sum / float_numeric.len() as f64;
            
            (int_sum, int_mean, float_sum, float_mean)
        });
        
        // 新実装: DataFrame集計操作
        let (optimized_agg_time, _) = bench("新実装 - 集計操作", || {
            // 新実装では型安全なアクセスと直接的な数値操作
            let int_col = optimized_df.get_int64_column("int_col").unwrap();
            let float_col = optimized_df.get_float64_column("float_col").unwrap();
            
            // 直接的な集計計算
            let int_sum = int_col.sum();
            let int_mean = int_col.mean().unwrap();
            
            let float_sum = float_col.sum();
            let float_mean = float_col.mean().unwrap();
            
            (int_sum, int_mean, float_sum, float_mean)
        });
        
        // 結果サマリー
        println!("\n結果サマリー ({}行):", size);
        println!("  Series作成: {:.2}倍高速化 ({} → {})", 
                 legacy_series_time.as_secs_f64() / optimized_series_time.as_secs_f64(),
                 format_duration(legacy_series_time),
                 format_duration(optimized_series_time));
        
        println!("  DataFrame作成: {:.2}倍高速化 ({} → {})",
                 legacy_df_time.as_secs_f64() / optimized_df_time.as_secs_f64(),
                 format_duration(legacy_df_time),
                 format_duration(optimized_df_time));
        
        println!("  集計操作: {:.2}倍高速化 ({} → {})",
                 legacy_agg_time.as_secs_f64() / optimized_agg_time.as_secs_f64(),
                 format_duration(legacy_agg_time),
                 format_duration(optimized_agg_time));
    }
}

fn benchmark_string_operations() {
    println!("\n=== 文字列操作パフォーマンス比較 ===\n");
    
    // 文字列データの準備
    let size = 1_000_000;
    let unique_words = ["apple", "banana", "cherry", "date", "elderberry", 
                         "fig", "grape", "honeydew", "kiwi", "lemon"];
    
    let string_data: Vec<String> = (0..size)
        .map(|i| unique_words[i % unique_words.len()].to_string())
        .collect();
    
    // 旧実装: 通常の文字列Series
    let (legacy_series_time, legacy_string_series) = bench("旧実装 - 文字列Series作成", || {
        Series::new(string_data.clone(), Some("strings".to_string())).unwrap()
    });
    
    // 新実装: 文字列プールを使用したStringColumn
    let (optimized_series_time, optimized_string_col) = bench("新実装 - 文字列列作成", || {
        prototype::StringColumn::new(string_data.clone()).with_name("strings")
    });
    
    // 旧実装: 文字列検索
    let target = "grape";
    let (legacy_search_time, legacy_count) = bench("旧実装 - 文字列検索", || {
        let values = legacy_string_series.values();
        values.iter().filter(|&s| s == target).count()
    });
    
    // 新実装: 文字列検索
    let (optimized_search_time, optimized_count) = bench("新実装 - 文字列検索", || {
        optimized_string_col.values().filter(|&s| s == target).count()
    });
    
    // 結果サマリー
    println!("\n文字列操作結果サマリー ({}行):", size);
    println!("  文字列Series作成: {:.2}倍高速化 ({} → {})", 
             legacy_series_time.as_secs_f64() / optimized_series_time.as_secs_f64(),
             format_duration(legacy_series_time),
             format_duration(optimized_series_time));
    
    println!("  文字列検索: {:.2}倍高速化 ({} → {}) [一致件数: {}]",
             legacy_search_time.as_secs_f64() / optimized_search_time.as_secs_f64(),
             format_duration(legacy_search_time),
             format_duration(optimized_search_time),
             legacy_count);
    
    // メモリ使用量の概算（Stringが平均20バイト、インデックスが4バイトと仮定）
    let legacy_size = string_data.len() * (20 + std::mem::size_of::<String>());
    
    // 列サイズは文字列数xポインタサイズ + 一意文字列数x文字列サイズ と概算
    let unique_words_count = unique_words.len();
    let pool_unique_strings_size = unique_words_count * (20 + std::mem::size_of::<String>());
    let indices_size = string_data.len() * std::mem::size_of::<u32>();
    let pool_size = pool_unique_strings_size + indices_size;
    
    println!("  メモリ使用量: {:.2}倍削減 ({:.2} MB → {:.2} MB)",
             legacy_size as f64 / pool_size as f64,
             legacy_size as f64 / 1024.0 / 1024.0,
             pool_size as f64 / 1024.0 / 1024.0);
}

fn main() {
    run_benchmark_suite();
    benchmark_string_operations();
}
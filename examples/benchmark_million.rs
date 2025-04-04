use pandrs::{DataFrame, PandRSError, Series, Column, Int64Column, Float64Column, StringColumn};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> Result<(), PandRSError> {
    println!("=== PandRS 大規模ベンチマーク（100万行） ===\n");

    // ベンチマーク関数
    fn bench<F>(name: &str, f: F) -> Duration 
    where 
        F: FnOnce() -> ()
    {
        println!("実行中: {}", name);
        let start = Instant::now();
        f();
        let duration = start.elapsed();
        println!("  完了: {:?}\n", duration);
        duration
    }

    // 100万行DataFrame作成のベンチマーク
    println!("--- 100万行DataFrame ---");
    
    bench("Series作成 x3 (100万行)", || {
        let _ = Series::new((0..1_000_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new((0..1_000_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let _ = Series::new(
            (0..1_000_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
    });
    
    let large_duration = bench("DataFrame作成 (3列x100万行)", || {
        let col_a = Series::new((0..1_000_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..1_000_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..1_000_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        
        // Series -> Column変換
        let col_a_int = Int64Column::new(col_a.values().to_vec());
        let col_b_float = Float64Column::new(col_b.values().to_vec());
        let col_c_str = StringColumn::new(col_c.values().to_vec());
        
        df.add_column("A", Column::Int64(col_a_int)).unwrap();
        df.add_column("B", Column::Float64(col_b_float)).unwrap();
        df.add_column("C", Column::String(col_c_str)).unwrap();
    });
    
    bench("DataFrame from_map (3列x100万行)", || {
        let mut data: HashMap<String, Vec<String>> = HashMap::new();
        data.insert("A".to_string(), (0..1_000_000).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..1_000_000).map(|n| format!("{:.1}", n as f64 * 0.5)).collect());
        data.insert("C".to_string(), (0..1_000_000).map(|i| format!("val_{}", i)).collect());
        
        // DataFrame::from_mapは削除されたため、通常のDataFrame作成に置き換え
        let mut df = DataFrame::new();
        
        // 列データを作成 - 型指定でHashMapのエラーを解消
        let a_data: Vec<String> = data.get("A").unwrap().to_vec();
        let b_data: Vec<String> = data.get("B").unwrap().to_vec();
        let c_data: Vec<String> = data.get("C").unwrap().to_vec();
        
        let a_column = StringColumn::new(a_data);
        let b_column = StringColumn::new(b_data);
        let c_column = StringColumn::new(c_data);
        
        df.add_column("A", Column::String(a_column)).unwrap();
        df.add_column("B", Column::String(b_column)).unwrap();
        df.add_column("C", Column::String(c_column)).unwrap();
    });
    
    println!("純粋なRustコードでの100万行DataFrame作成時間: {:?}", large_duration);
    
    Ok(())
}
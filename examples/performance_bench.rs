use pandrs::{DataFrame, PandRSError, Series};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn main() -> Result<(), PandRSError> {
    println!("=== PandRS パフォーマンスベンチマーク ===\n");

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

    // 小さいDataFrame作成のベンチマーク
    println!("--- 小さいDataFrame (10行) ---");
    
    bench("Series作成 x3", || {
        let _ = Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
        let _ = Series::new(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], Some("B".to_string())).unwrap();
        let _ = Series::new(
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
    });
    
    bench("DataFrame作成 (3列x10行)", || {
        let col_a = Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
        let col_b = Series::new(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10], Some("B".to_string())).unwrap();
        let col_c = Series::new(
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
    
    bench("DataFrame from_map (3列x10行)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..10).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..10).map(|n| format!("{:.1}", n as f64 + 0.1)).collect());
        data.insert("C".to_string(), vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
            .into_iter()
            .map(|s| s.to_string())
            .collect());
        
        let _ = DataFrame::from_map(data, None).unwrap();
    });
    
    // 中規模DataFrameのベンチマーク
    println!("\n--- 中規模DataFrame (1,000行) ---");
    
    bench("Series作成 x3 (1000行)", || {
        let _ = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new((0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let _ = Series::new(
            (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
    });
    
    bench("DataFrame作成 (3列x1000行)", || {
        let col_a = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
    
    bench("DataFrame from_map (3列x1000行)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..1000).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..1000).map(|n| format!("{:.1}", n as f64 * 0.5)).collect());
        data.insert("C".to_string(), (0..1000).map(|i| format!("val_{}", i)).collect());
        
        let _ = DataFrame::from_map(data, None).unwrap();
    });
    
    // 大規模DataFrameのベンチマーク
    println!("\n--- 大規模DataFrame (100,000行) ---");
    
    bench("Series作成 x3 (100,000行)", || {
        let _ = Series::new((0..100_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new((0..100_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let _ = Series::new(
            (0..100_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
    });
    
    let large_duration = bench("DataFrame作成 (3列x100,000行)", || {
        let col_a = Series::new((0..100_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new((0..100_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(), Some("B".to_string())).unwrap();
        let col_c = Series::new(
            (0..100_000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        ).unwrap();
        
        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
    
    bench("DataFrame from_map (3列x100,000行)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..100_000).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..100_000).map(|n| format!("{:.1}", n as f64 * 0.5)).collect());
        data.insert("C".to_string(), (0..100_000).map(|i| format!("val_{}", i)).collect());
        
        let _ = DataFrame::from_map(data, None).unwrap();
    });
    
    println!("純粋なRustコードでの100,000行DataFrame作成時間: {:?}", large_duration);
    println!("(Python版での同等操作: 約0.35秒)");
    
    Ok(())
}
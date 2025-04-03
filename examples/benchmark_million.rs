use pandrs::{DataFrame, PandRSError, Series};
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
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });
    
    bench("DataFrame from_map (3列x100万行)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..1_000_000).map(|n| n.to_string()).collect());
        data.insert("B".to_string(), (0..1_000_000).map(|n| format!("{:.1}", n as f64 * 0.5)).collect());
        data.insert("C".to_string(), (0..1_000_000).map(|i| format!("val_{}", i)).collect());
        
        let _ = DataFrame::from_map(data, None).unwrap();
    });
    
    println!("純粋なRustコードでの100万行DataFrame作成時間: {:?}", large_duration);
    
    Ok(())
}
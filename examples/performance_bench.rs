use pandrs::{DataFrame, PandRSError, Series};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[allow(clippy::result_large_err)]
fn main() -> Result<(), PandRSError> {
    println!("=== PandRS Performance Benchmark ===\n");

    // Benchmark function
    fn bench<F>(name: &str, f: F) -> Duration
    where
        F: FnOnce(),
    {
        println!("Running: {}", name);
        let start = Instant::now();
        f();
        let duration = start.elapsed();
        println!("  Completed: {:?}\n", duration);
        duration
    }

    // Benchmark for creating a small DataFrame
    println!("--- Small DataFrame (10 rows) ---");

    bench("Create Series x3", || {
        let _ = Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
        let _ = Series::new(
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            Some("B".to_string()),
        )
        .unwrap();
        let _ = Series::new(
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();
    });

    bench("Create DataFrame (3 columns x 10 rows)", || {
        let col_a =
            Series::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Some("A".to_string())).unwrap();
        let col_b = Series::new(
            vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            Some("B".to_string()),
        )
        .unwrap();
        let col_c = Series::new(
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();

        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });

    bench("DataFrame from_map (3 columns x 10 rows)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..10).map(|n| n.to_string()).collect());
        data.insert(
            "B".to_string(),
            (0..10).map(|n| format!("{:.1}", n as f64 + 0.1)).collect(),
        );
        data.insert(
            "C".to_string(),
            vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let _ = DataFrame::from_map(data, None).unwrap();
    });

    // Benchmark for creating a medium DataFrame
    println!("\n--- Medium DataFrame (1,000 rows) ---");

    bench("Create Series x3 (1000 rows)", || {
        let _ = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new(
            (0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let _ = Series::new(
            (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();
    });

    bench("Create DataFrame (3 columns x 1000 rows)", || {
        let col_a = Series::new((0..1000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new(
            (0..1000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let col_c = Series::new(
            (0..1000).map(|i| format!("val_{}", i)).collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();

        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });

    bench("DataFrame from_map (3 columns x 1000 rows)", || {
        let mut data = HashMap::new();
        data.insert("A".to_string(), (0..1000).map(|n| n.to_string()).collect());
        data.insert(
            "B".to_string(),
            (0..1000)
                .map(|n| format!("{:.1}", n as f64 * 0.5))
                .collect(),
        );
        data.insert(
            "C".to_string(),
            (0..1000).map(|i| format!("val_{}", i)).collect(),
        );

        let _ = DataFrame::from_map(data, None).unwrap();
    });

    // Benchmark for creating a large DataFrame
    println!("\n--- Large DataFrame (100,000 rows) ---");

    bench("Create Series x3 (100,000 rows)", || {
        let _ = Series::new((0..100_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new(
            (0..100_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let _ = Series::new(
            (0..100_000)
                .map(|i| format!("val_{}", i))
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();
    });

    let large_duration = bench("Create DataFrame (3 columns x 100,000 rows)", || {
        let col_a = Series::new((0..100_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new(
            (0..100_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let col_c = Series::new(
            (0..100_000)
                .map(|i| format!("val_{}", i))
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();

        let mut df = DataFrame::new();
        df.add_column("A".to_string(), col_a).unwrap();
        df.add_column("B".to_string(), col_b).unwrap();
        df.add_column("C".to_string(), col_c).unwrap();
    });

    bench("DataFrame from_map (3 columns x 100,000 rows)", || {
        let mut data = HashMap::new();
        data.insert(
            "A".to_string(),
            (0..100_000).map(|n| n.to_string()).collect(),
        );
        data.insert(
            "B".to_string(),
            (0..100_000)
                .map(|n| format!("{:.1}", n as f64 * 0.5))
                .collect(),
        );
        data.insert(
            "C".to_string(),
            (0..100_000).map(|i| format!("val_{}", i)).collect(),
        );

        let _ = DataFrame::from_map(data, None).unwrap();
    });

    println!(
        "Pure Rust code DataFrame creation time for 100,000 rows: {:?}",
        large_duration
    );
    println!("(Equivalent operation in Python: approximately 0.35 seconds)");

    Ok(())
}

use pandrs::error::Result;
use pandrs::{DataFrame, Series};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Benchmark with One Million Rows ===\n");

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

    // Benchmark for creating a DataFrame with one million rows
    println!("--- DataFrame with One Million Rows ---");

    bench("Creating Series x3 (One Million Rows)", || {
        let _ = Series::new((0..1_000_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let _ = Series::new(
            (0..1_000_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let _ = Series::new(
            (0..1_000_000)
                .map(|i| format!("val_{}", i))
                .collect::<Vec<_>>(),
            Some("C".to_string()),
        )
        .unwrap();
    });

    let large_duration = bench("Creating DataFrame (3 Columns x One Million Rows)", || {
        let col_a = Series::new((0..1_000_000).collect::<Vec<_>>(), Some("A".to_string())).unwrap();
        let col_b = Series::new(
            (0..1_000_000).map(|i| i as f64 * 0.5).collect::<Vec<_>>(),
            Some("B".to_string()),
        )
        .unwrap();
        let col_c = Series::new(
            (0..1_000_000)
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

    bench("DataFrame from_map (3 Columns x One Million Rows)", || {
        let mut data = HashMap::new();
        data.insert(
            "A".to_string(),
            (0..1_000_000).map(|n| n.to_string()).collect(),
        );
        data.insert(
            "B".to_string(),
            (0..1_000_000)
                .map(|n| format!("{:.1}", n as f64 * 0.5))
                .collect(),
        );
        data.insert(
            "C".to_string(),
            (0..1_000_000).map(|i| format!("val_{}", i)).collect(),
        );

        let _ = DataFrame::from_map(data, None).unwrap();
    });

    println!(
        "Time to create DataFrame with one million rows in pure Rust: {:?}",
        large_duration
    );

    Ok(())
}

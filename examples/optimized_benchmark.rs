//! Optimized DataFrame implementation benchmark
//!
//! This example benchmarks the performance of the optimized DataFrame implementation
//! compared to the legacy implementation.
//!
//! To run:
//!   cargo run --example optimized_benchmark --features "optimized"
//!
//! To run with GPU acceleration:
//!   cargo run --example optimized_benchmark --features "optimized cuda"

#[cfg(feature = "optimized")]
use std::time::{Duration, Instant};

#[cfg(feature = "optimized")]
use pandrs::optimized::OptimizedDataFrame;
#[cfg(feature = "optimized")]
use pandrs::{Column, DataFrame, Float64Column, Int64Column, Series, StringColumn};

#[cfg(cuda_available)]
use pandrs::gpu;

#[cfg(feature = "optimized")]
/// Format elapsed time into a readable format
fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{}.{:03}s", duration.as_secs(), duration.subsec_millis())
    } else if duration.as_millis() > 0 {
        format!(
            "{}.{:03}ms",
            duration.as_millis(),
            duration.as_micros() % 1000
        )
    } else {
        format!("{}µs", duration.as_micros())
    }
}

#[cfg(feature = "optimized")]
/// Benchmark function
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

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Optimized Implementation Performance Benchmark ===\n");

    // Check GPU status if cuda feature is enabled
    #[cfg(cuda_available)]
    {
        println!("--- GPU Status ---");
        match gpu::init_gpu() {
            Ok(status) => {
                println!("GPU Available: {}", status.available);
                if status.available {
                    println!(
                        "Device: {}",
                        status.device_name.unwrap_or_else(|| "Unknown".to_string())
                    );
                    println!(
                        "Memory: {} MB",
                        status.total_memory.unwrap_or(0) / (1024 * 1024)
                    );
                }
            }
            Err(e) => println!("GPU init error: {:?}", e),
        }
        println!();
    }

    // Benchmark data sizes
    let sizes = [10_000, 100_000];

    for &size in &sizes {
        println!("\n## Data Size: {} rows ##", size);

        // ------- Data Preparation -------
        let int_data: Vec<i64> = (0..size).collect();
        let float_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
        let string_data: Vec<String> = (0..size).map(|i| format!("val_{}", i % 100)).collect();

        // ------- Legacy Implementation Benchmark -------
        let (_legacy_df_time, legacy_df) = bench("Legacy DataFrame - Create", || {
            let mut df = DataFrame::new();
            let int_series = Series::new(
                int_data.iter().map(|&i| i as i32).collect::<Vec<i32>>(),
                Some("int_col".to_string()),
            )
            .unwrap();
            let float_series =
                Series::new(float_data.clone(), Some("float_col".to_string())).unwrap();
            let string_series =
                Series::new(string_data.clone(), Some("string_col".to_string())).unwrap();

            df.add_column("int_col".to_string(), int_series).unwrap();
            df.add_column("float_col".to_string(), float_series)
                .unwrap();
            df.add_column("string_col".to_string(), string_series)
                .unwrap();
            df
        });

        println!("Legacy DataFrame created: {} rows", legacy_df.row_count());

        // ------- Optimized Implementation Benchmark -------
        let (_opt_df_time, opt_df) = bench("Optimized DataFrame - Create", || {
            let mut df = OptimizedDataFrame::new();
            df.add_column(
                "int_col".to_string(),
                Column::Int64(Int64Column::new(int_data.clone())),
            )
            .unwrap();
            df.add_column(
                "float_col".to_string(),
                Column::Float64(Float64Column::new(float_data.clone())),
            )
            .unwrap();
            df.add_column(
                "string_col".to_string(),
                Column::String(StringColumn::new(string_data.clone())),
            )
            .unwrap();
            df
        });

        println!("Optimized DataFrame created: {} rows", opt_df.row_count());

        // ------- Basic Operations Benchmark -------
        let (_row_iter_time, _) = bench("Optimized DataFrame - Row Iteration", || {
            let mut sum = 0i64;
            for i in 0..opt_df.row_count() {
                sum += i as i64;
            }
            sum
        });

        println!();
    }

    // Summary
    println!("\n=== Benchmark Summary ===");
    println!("The optimized implementation provides:");
    println!("  - Columnar storage for cache efficiency");
    println!("  - SIMD operations for numeric types");
    println!("  - Lazy evaluation support");
    println!("  - GPU acceleration when available (cuda feature)");

    Ok(())
}

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("Optimized DataFrame Benchmark");
    println!("==============================");
    println!();
    println!("This example requires the 'optimized' feature flag.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_benchmark --features \"optimized\"");
}

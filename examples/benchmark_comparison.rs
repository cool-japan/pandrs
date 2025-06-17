// Benchmark Comparison: Comparing the traditional implementation and optimized implementation of PandRS
// This file provides benchmarks to compare the performance of the traditional implementation and optimized implementation of PandRS.

use pandrs::{DataFrame, Float64Column, Int64Column, OptimizedDataFrame, Series, StringColumn};
use std::time::{Duration, Instant};

// Import prototype types
// mod prototype {
//     include!("column_prototype.rs");
// }

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

// Benchmark function
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
    // Header
    println!("\n=== PandRS Performance Optimization Benchmark ===\n");

    // Benchmark data sizes
    let sizes = [1000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        println!("\n## Data Size: {} rows ##", size);

        // Data preparation
        let int_data: Vec<i32> = (0..size).collect();
        let float_data: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
        let string_data: Vec<String> = (0..size).map(|i| format!("val_{}", i % 100)).collect();

        // Legacy implementation: Series creation
        let (legacy_series_time, (legacy_int_series, legacy_float_series, legacy_string_series)) =
            bench("Legacy Implementation - Series Creation", || {
                let int_series =
                    Series::new(int_data.clone(), Some("int_col".to_string())).unwrap();
                let float_series =
                    Series::new(float_data.clone(), Some("float_col".to_string())).unwrap();
                let string_series =
                    Series::new(string_data.clone(), Some("string_col".to_string())).unwrap();
                (int_series, float_series, string_series)
            });

        // Optimized implementation: Column creation
        let (optimized_series_time, (opt_int_col, opt_float_col, opt_string_col)) =
            bench("Optimized Implementation - Column Creation", || {
                let int_col =
                    Int64Column::with_name(int_data.iter().map(|&i| i as i64).collect(), "int_col");
                let float_col = Float64Column::with_name(float_data.clone(), "float_col");
                let string_col = StringColumn::with_name(string_data.clone(), "string_col");
                (int_col, float_col, string_col)
            });

        // Legacy implementation: DataFrame creation
        let (legacy_df_time, legacy_df) =
            bench("Legacy Implementation - DataFrame Creation", || {
                let mut df = DataFrame::new();
                df.add_column("int_col".to_string(), legacy_int_series.clone())
                    .unwrap();
                df.add_column("float_col".to_string(), legacy_float_series.clone())
                    .unwrap();
                df.add_column("string_col".to_string(), legacy_string_series.clone())
                    .unwrap();
                df
            });

        // Optimized implementation: OptimizedDataFrame creation
        let (optimized_df_time, optimized_df) =
            bench("Optimized Implementation - DataFrame Creation", || {
                let mut df = OptimizedDataFrame::new();
                df.add_column("int_col", opt_int_col.clone()).unwrap();
                df.add_column("float_col", opt_float_col.clone()).unwrap();
                df.add_column("string_col", opt_string_col.clone()).unwrap();
                df
            });

        // Legacy implementation: DataFrame aggregation operations
        let (legacy_agg_time, _) = bench("Legacy Implementation - Aggregation Operations", || {
            // Legacy implementation has low efficiency due to numerical operations via DataBox
            // Legacy implementation requires string conversion for numerical operations
            let int_values = legacy_df.get_column_string_values("int_col").unwrap();
            let float_values = legacy_df.get_column_string_values("float_col").unwrap();

            // Conversion from string to numeric
            let int_numeric: Vec<i32> = int_values
                .iter()
                .filter_map(|s| s.parse::<i32>().ok())
                .collect();

            let float_numeric: Vec<f64> = float_values
                .iter()
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();

            // Aggregation calculations
            let int_sum: i32 = int_numeric.iter().sum();
            let int_mean = int_sum as f64 / int_numeric.len() as f64;

            let float_sum: f64 = float_numeric.iter().sum();
            let float_mean = float_sum / float_numeric.len() as f64;

            (int_sum, int_mean, float_sum, float_mean)
        });

        // Optimized implementation: DataFrame aggregation operations
        let (optimized_agg_time, _) =
            bench("Optimized Implementation - Aggregation Operations", || {
                // Optimized implementation has type-safe access and direct numerical operations
                let int_col_view = optimized_df.column("int_col").unwrap();
                let float_col_view = optimized_df.column("float_col").unwrap();

                let int_col = int_col_view.as_int64().unwrap();
                let float_col = float_col_view.as_float64().unwrap();

                // Direct aggregation calculations
                let int_sum = int_col.sum();
                let int_mean = int_col.mean().unwrap();

                let float_sum = float_col.sum();
                let float_mean = float_col.mean().unwrap();

                (int_sum, int_mean, float_sum, float_mean)
            });

        // Result summary
        println!("\nResult Summary ({} rows):", size);
        println!(
            "  Series Creation: {:.2}x speedup ({} → {})",
            legacy_series_time.as_secs_f64() / optimized_series_time.as_secs_f64(),
            format_duration(legacy_series_time),
            format_duration(optimized_series_time)
        );

        println!(
            "  DataFrame Creation: {:.2}x speedup ({} → {})",
            legacy_df_time.as_secs_f64() / optimized_df_time.as_secs_f64(),
            format_duration(legacy_df_time),
            format_duration(optimized_df_time)
        );

        println!(
            "  Aggregation Operations: {:.2}x speedup ({} → {})",
            legacy_agg_time.as_secs_f64() / optimized_agg_time.as_secs_f64(),
            format_duration(legacy_agg_time),
            format_duration(optimized_agg_time)
        );
    }
}

fn benchmark_string_operations() {
    println!("\n=== String Operations Performance Comparison ===\n");

    // Prepare string data
    let size = 1_000_000;
    let unique_words = [
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
    ];

    let string_data: Vec<String> = (0..size)
        .map(|i| unique_words[i % unique_words.len()].to_string())
        .collect();

    // Legacy implementation: Regular string Series
    let (legacy_series_time, legacy_string_series) =
        bench("Legacy Implementation - String Series Creation", || {
            Series::new(string_data.clone(), Some("strings".to_string())).unwrap()
        });

    // Optimized implementation: StringColumn using string pool
    let (optimized_series_time, optimized_string_col) =
        bench("Optimized Implementation - String Column Creation", || {
            StringColumn::with_name(string_data.clone(), "strings")
        });

    // Legacy implementation: String search
    let target = "grape";
    let (legacy_search_time, legacy_count) = bench("Legacy Implementation - String Search", || {
        let values = legacy_string_series.values();
        values.iter().filter(|&s| s == target).count()
    });

    // Optimized implementation: String search
    let (optimized_search_time, _optimized_count) =
        bench("Optimized Implementation - String Search", || {
            optimized_string_col
                .to_strings()
                .iter()
                .flatten()
                .filter(|&s| s == target)
                .count()
        });

    // Result summary
    println!("\nString Operations Result Summary ({} rows):", size);
    println!(
        "  String Series Creation: {:.2}x speedup ({} → {})",
        legacy_series_time.as_secs_f64() / optimized_series_time.as_secs_f64(),
        format_duration(legacy_series_time),
        format_duration(optimized_series_time)
    );

    println!(
        "  String Search: {:.2}x speedup ({} → {}) [Match Count: {}]",
        legacy_search_time.as_secs_f64() / optimized_search_time.as_secs_f64(),
        format_duration(legacy_search_time),
        format_duration(optimized_search_time),
        legacy_count
    );

    // Estimated memory usage (assuming String is on average 20 bytes, index is 4 bytes)
    let legacy_size = string_data.len() * (20 + std::mem::size_of::<String>());

    // Column size is estimated as number of strings x pointer size + number of unique strings x string size
    let unique_words_count = unique_words.len();
    let pool_unique_strings_size = unique_words_count * (20 + std::mem::size_of::<String>());
    let indices_size = string_data.len() * std::mem::size_of::<u32>();
    let pool_size = pool_unique_strings_size + indices_size;

    println!(
        "  Memory Usage: {:.2}x reduction ({:.2} MB → {:.2} MB)",
        legacy_size as f64 / pool_size as f64,
        legacy_size as f64 / 1024.0 / 1024.0,
        pool_size as f64 / 1024.0 / 1024.0
    );
}

fn main() {
    run_benchmark_suite();
    benchmark_string_operations();
}

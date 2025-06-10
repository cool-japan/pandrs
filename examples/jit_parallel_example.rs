//! Example demonstrating parallel JIT execution
//!
//! This example shows how to use parallel execution with JIT functions
//! to achieve improved performance on multi-core systems.
//!
//! Run with: cargo run --example jit_parallel_example --features jit --release

use rand::Rng;
use std::error::Error;
use std::time::{Duration, Instant};

use pandrs::optimized::jit::JitCompilable;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Parallel JIT Execution Example");
    println!("==============================");

    // Test with large array size
    let size = 20_000_000;

    println!("\nBenchmarking with array size: {}", size);

    // Generate random data
    let data = generate_random_data(size);

    // Compare different execution methods
    println!("\nSum Operation Comparison:");

    // Standard JIT (single-threaded)
    let (std_result, std_time) = benchmark_standard_jit(&data);
    println!("  Standard JIT sum:     {:.6}", std_result);
    println!("  Standard time:        {:?}", std_time);

    // SIMD JIT (single-threaded with vector instructions)
    let (simd_result, simd_time) = benchmark_simd_jit(&data);
    println!("  SIMD JIT sum:         {:.6}", simd_result);
    println!("  SIMD time:            {:?}", simd_time);
    println!(
        "  SIMD speedup:         {:.2}x",
        std_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    );

    // Parallel JIT (multi-threaded)
    let (par_result, par_time) = benchmark_parallel_jit(&data);
    println!("  Parallel JIT sum:     {:.6}", par_result);
    println!("  Parallel time:        {:?}", par_time);
    println!(
        "  Parallel speedup:     {:.2}x",
        std_time.as_nanos() as f64 / par_time.as_nanos() as f64
    );

    // Benchmark other operations
    benchmark_other_operations(&data);

    // Benchmark custom parallel function
    benchmark_custom_parallel(&data);

    // Demonstrate parallel configuration tuning
    benchmark_parallel_config_tuning(&data);

    Ok(())
}

// Generate random floating-point data
fn generate_random_data(size: usize) -> Vec<f64> {
    println!("Generating {} random values...", size);
    let mut rng = rand::rng();
    let result = (0..size).map(|_| rng.random_range(0.0..100.0)).collect();
    println!("Data generation complete.");
    result
}

// Benchmark standard JIT function (single-threaded)
fn benchmark_standard_jit(data: &[f64]) -> (f64, Duration) {
    use pandrs::optimized::jit::jit_f64;

    println!("Running standard JIT benchmark...");

    // Create standard JIT function for sum
    let sum_fn = jit_f64("standard_sum", |values: &[f64]| -> f64 {
        values.iter().sum()
    });

    // Measure execution time
    let start = Instant::now();
    let result = sum_fn.execute(data);
    let duration = start.elapsed();

    (result, duration)
}

// Benchmark SIMD JIT function (single-threaded with vector instructions)
fn benchmark_simd_jit(data: &[f64]) -> (f64, Duration) {
    use pandrs::optimized::jit::simd_sum_f64;

    println!("Running SIMD JIT benchmark...");

    // Use SIMD function directly
    let start = Instant::now();
    let result = simd_sum_f64(data);
    let duration = start.elapsed();

    (result, duration)
}

// Benchmark parallel JIT function (multi-threaded)
fn benchmark_parallel_jit(data: &[f64]) -> (f64, Duration) {
    use pandrs::optimized::jit::parallel_sum_f64;

    println!("Running parallel JIT benchmark...");

    // Create parallel JIT function for sum
    let parallel_sum = parallel_sum_f64(None);

    // Measure execution time
    let start = Instant::now();
    let result = parallel_sum.execute(data);
    let duration = start.elapsed();

    (result, duration)
}

// Benchmark other parallel operations
fn benchmark_other_operations(data: &[f64]) {
    use pandrs::optimized::jit::{
        jit_f64, parallel_max_f64, parallel_mean_f64, parallel_min_f64, parallel_std_f64,
    };

    println!("\nOther Operations Comparison:");

    // Mean operation
    println!("\nMean Operation:");

    // Standard mean
    let mean_fn = jit_f64("standard_mean", |values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    });

    let start = Instant::now();
    let result = mean_fn.execute(data);
    let std_time = start.elapsed();

    println!("  Standard mean: {:.6}", result);
    println!("  Standard time: {:?}", std_time);

    // Parallel mean
    let parallel_mean = parallel_mean_f64(None);

    let start = Instant::now();
    let result = parallel_mean.execute(data);
    let par_time = start.elapsed();

    println!("  Parallel mean: {:.6}", result.0);
    println!("  Parallel time: {:?}", par_time);
    println!(
        "  Speedup:       {:.2}x",
        std_time.as_nanos() as f64 / par_time.as_nanos() as f64
    );

    // Standard deviation operation
    println!("\nStandard Deviation Operation:");

    // Standard std
    let std_fn = jit_f64("standard_std", |values: &[f64]| -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    });

    let start = Instant::now();
    let result = std_fn.execute(data);
    let std_time = start.elapsed();

    println!("  Standard std: {:.6}", result);
    println!("  Standard time: {:?}", std_time);

    // Parallel std
    let parallel_std = parallel_std_f64(None);

    let start = Instant::now();
    let result = parallel_std.execute(data);
    let par_time = start.elapsed();

    println!("  Parallel std: {:.6}", result.0);
    println!("  Parallel time: {:?}", par_time);
    println!(
        "  Speedup:       {:.2}x",
        std_time.as_nanos() as f64 / par_time.as_nanos() as f64
    );

    // Min/Max operations
    println!("\nMin/Max Operations:");

    // Parallel min
    let parallel_min = parallel_min_f64(None);

    let start = Instant::now();
    let min_result = parallel_min.execute(data);
    let min_time = start.elapsed();

    // Parallel max
    let parallel_max = parallel_max_f64(None);

    let start = Instant::now();
    let max_result = parallel_max.execute(data);
    let max_time = start.elapsed();

    println!(
        "  Min value: {:.6} (computed in {:?})",
        min_result, min_time
    );
    println!(
        "  Max value: {:.6} (computed in {:?})",
        max_result, max_time
    );
}

// Benchmark custom parallel function
fn benchmark_custom_parallel(data: &[f64]) {
    use pandrs::optimized::jit::parallel_custom;

    println!("\nCustom Parallel Function:");

    // Custom function to compute sum of squares
    let squared_sum = |values: &[f64]| -> f64 { values.iter().map(|x| x * x).sum() };

    // Create a parallel version with the same function for map
    let reduce_fn = |results: Vec<f64>| -> f64 { results.iter().sum() };

    let parallel_squared_sum = parallel_custom(
        "parallel_squared_sum",
        squared_sum.clone(),
        squared_sum,
        reduce_fn,
        None,
    );

    // Benchmark
    let start = Instant::now();
    let result = parallel_squared_sum.execute(data);
    let duration = start.elapsed();

    println!("  Result: {:.6}", result);
    println!("  Time:   {:?}", duration);
}

// Benchmark parallel configuration tuning
fn benchmark_parallel_config_tuning(data: &[f64]) {
    use pandrs::optimized::jit::{parallel_sum_f64, ParallelConfig};

    println!("\nParallel Configuration Tuning:");

    // Different chunk sizes
    let chunk_sizes = [1000, 10000, 100000, 1000000];

    for &chunk_size in &chunk_sizes {
        let config = ParallelConfig::new().with_min_chunk_size(chunk_size);
        let parallel_sum = parallel_sum_f64(Some(config));

        let start = Instant::now();
        let _result = parallel_sum.execute(data);
        let duration = start.elapsed();

        println!("  Chunk size {:8}: {:?}", chunk_size, duration);
    }

    // Different thread counts
    let available_threads = num_cpus::get();
    println!("\nAvailable CPU cores: {}", available_threads);

    let thread_counts = [1, 2, 4, available_threads];

    for &threads in &thread_counts {
        if threads > available_threads {
            continue;
        }

        let config = ParallelConfig::new()
            .with_min_chunk_size(100000)
            .with_max_threads(threads);

        let parallel_sum = parallel_sum_f64(Some(config));

        let start = Instant::now();
        let _result = parallel_sum.execute(data);
        let duration = start.elapsed();

        println!("  Threads {:8}: {:?}", threads, duration);
    }
}

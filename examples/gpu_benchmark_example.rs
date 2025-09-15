//! GPU benchmark utility example
//!
//! This example demonstrates how to use the GPU benchmark utility to
//! benchmark and compare CPU and GPU performance for various operations.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_benchmark_example --features "cuda"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_benchmark_example

#[cfg(feature = "cuda")]
use pandrs::error::Result;
#[cfg(feature = "cuda")]
use pandrs::GpuBenchmark;

#[cfg(feature = "cuda")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Create a GPU benchmark utility
    let mut benchmark = GpuBenchmark::new()?;

    println!("Running GPU benchmarks...");

    // Benchmark matrix multiplication
    println!("\nBenchmarking matrix multiplication...");
    let matrix_mult_result = benchmark.benchmark_matrix_multiply(1000, 1000, 1000)?;
    println!("{}", matrix_mult_result.formatted_summary());

    // Benchmark element-wise addition
    println!("\nBenchmarking element-wise addition...");
    let add_result = benchmark.benchmark_elementwise_add(2000, 2000)?;
    println!("{}", add_result.formatted_summary());

    // Benchmark correlation matrix
    println!("\nBenchmarking correlation matrix...");
    let corr_result = benchmark.benchmark_correlation(10000, 10)?;
    println!("{}", corr_result.formatted_summary());

    // Benchmark linear regression
    println!("\nBenchmarking linear regression...");
    let lr_result = benchmark.benchmark_linear_regression(10000, 10)?;
    println!("{}", lr_result.formatted_summary());

    // Benchmark rolling window
    println!("\nBenchmarking rolling window...");
    let rolling_result = benchmark.benchmark_rolling_window(100000, 100)?;
    println!("{}", rolling_result.formatted_summary());

    // Print overall benchmark summary
    println!("\n{}", benchmark.get_summary());

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the \"cuda\" feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_benchmark_example --features \"cuda\"");
}

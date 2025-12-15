//! GPU-accelerated DataFrame operations example
//!
//! This example demonstrates the concept of GPU acceleration for DataFrame operations.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_dataframe_example --features "cuda optimized"

#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::gpu::init_gpu;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::optimized::dataframe::OptimizedDataFrame;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::DataFrame;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::Series;

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS GPU-accelerated DataFrame Example ===\n");

    // Initialize GPU with default configuration
    let device_status = init_gpu()?;

    println!("GPU Device Status:");
    println!("  Available: {}", device_status.available);

    if device_status.available {
        println!(
            "  Device Name: {}",
            device_status
                .device_name
                .unwrap_or_else(|| "Unknown".to_string())
        );
        println!(
            "  CUDA Version: {}",
            device_status
                .cuda_version
                .unwrap_or_else(|| "Unknown".to_string())
        );
        println!(
            "  Total Memory: {} MB",
            device_status.total_memory.unwrap_or(0) / (1024 * 1024)
        );
        println!(
            "  Free Memory: {} MB",
            device_status.free_memory.unwrap_or(0) / (1024 * 1024)
        );
    } else {
        println!("  No CUDA-compatible GPU available. Using CPU fallback.");
    }

    // Create a sample DataFrame
    println!("\nCreating sample DataFrame...");
    let df = create_sample_dataframe(10_000)?;
    println!("DataFrame created with {} rows", df.row_count());
    println!("Columns: {:?}", df.column_names());

    // Create OptimizedDataFrame
    println!("\nCreating OptimizedDataFrame...");
    let opt_df = create_optimized_dataframe(10_000)?;
    println!(
        "OptimizedDataFrame created with {} rows",
        opt_df.row_count()
    );

    // Describe GPU acceleration benefits
    println!("\n--- GPU DataFrame Operations ---");
    println!("  - Parallel column operations");
    println!("  - Batch aggregations (sum, mean, std)");
    println!("  - Optimized filtering with GPU predicates");
    println!("  - GPU-accelerated correlation matrices");
    println!("  - CUDA memory management for large datasets");

    println!("\n=== GPU DataFrame Example Complete ===");

    Ok(())
}

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn create_sample_dataframe(size: usize) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Create data for columns
    let ids: Vec<i64> = (0..size as i64).collect();
    let values1: Vec<f64> = (0..size).map(|i| (i % 100) as f64).collect();
    let values2: Vec<f64> = (0..size).map(|i| ((i * 2) % 100) as f64).collect();

    df.add_column("id".to_string(), Series::new(ids, Some("id".to_string()))?)?;
    df.add_column(
        "value1".to_string(),
        Series::new(values1, Some("value1".to_string()))?,
    )?;
    df.add_column(
        "value2".to_string(),
        Series::new(values2, Some("value2".to_string()))?,
    )?;

    Ok(df)
}

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn create_optimized_dataframe(size: usize) -> Result<OptimizedDataFrame> {
    use pandrs::column::{Column, Float64Column, Int64Column};

    let mut opt_df = OptimizedDataFrame::new();

    // Create data for columns
    let ids: Vec<i64> = (0..size as i64).collect();
    let values1: Vec<f64> = (0..size).map(|i| (i % 100) as f64).collect();
    let values2: Vec<f64> = (0..size).map(|i| ((i * 2) % 100) as f64).collect();

    opt_df.add_column("id".to_string(), Column::Int64(Int64Column::new(ids)))?;
    opt_df.add_column(
        "value1".to_string(),
        Column::Float64(Float64Column::new(values1)),
    )?;
    opt_df.add_column(
        "value2".to_string(),
        Column::Float64(Float64Column::new(values2)),
    )?;

    Ok(opt_df)
}

#[cfg(not(all(cuda_available, feature = "optimized")))]
fn main() {
    println!("GPU DataFrame Example");
    println!("=====================");
    println!();
    println!("This example requires CUDA hardware and 'optimized' feature flag.");
    println!("Please recompile on a CUDA-compatible system with:");
    println!("  cargo run --example gpu_dataframe_example --features \"cuda optimized\"");
}

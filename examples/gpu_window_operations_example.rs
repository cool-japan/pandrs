//! GPU-accelerated window operations example
//!
//! This example demonstrates the concept of GPU acceleration for window operations.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_window_operations_example --features "cuda optimized"

#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::gpu::{init_gpu, GpuConfig};
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::series::Series;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::DataFrame;

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS GPU-Accelerated Window Operations Example ===\n");

    // Initialize GPU with custom configuration
    let gpu_config = GpuConfig {
        enabled: true,
        memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        device_id: 0,
        fallback_to_cpu: true,
        use_pinned_memory: true,
        min_size_threshold: 10_000, // Use GPU for datasets > 10K elements
    };

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

    // Create sample data
    let df = create_sample_data()?;
    println!("\nCreated sample DataFrame with {} rows", df.row_count());

    // Demonstrate window operation concepts
    println!("\n--- GPU Window Operation Types ---");
    println!("  - Rolling windows: sliding window calculations");
    println!("  - Expanding windows: cumulative calculations");
    println!("  - EWM (Exponentially Weighted Moving): weighted averages");

    println!("\n--- GPU Acceleration Benefits ---");
    println!("  - Parallelized window computations");
    println!("  - Optimized memory access patterns");
    println!("  - Batch processing of multiple columns");
    println!("  - Automatic CPU fallback for small datasets");

    println!("\n--- Configuration: {:?}", gpu_config);

    println!("\n=== GPU Window Operations Example Complete ===");

    Ok(())
}

#[cfg(all(cuda_available, feature = "optimized"))]
fn create_sample_data() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Create price data
    let prices: Vec<f64> = (0..1000)
        .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
        .collect();

    // Create volume data
    let volumes: Vec<f64> = (0..1000)
        .map(|i| 1000.0 + (i as f64 * 0.05).cos() * 500.0)
        .collect();

    df.add_column(
        "price".to_string(),
        Series::new(prices, Some("price".to_string()))?,
    )?;

    df.add_column(
        "volume".to_string(),
        Series::new(volumes, Some("volume".to_string()))?,
    )?;

    Ok(df)
}

#[cfg(not(all(cuda_available, feature = "optimized")))]
fn main() {
    println!("GPU Window Operations Example");
    println!("==============================");
    println!();
    println!("This example requires both 'cuda' and 'optimized' feature flags.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_window_operations_example --features \"cuda optimized\"");
}

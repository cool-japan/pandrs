//! GPU-accelerated DataFrame API example
//!
//! This example demonstrates how to use the GPU-accelerated DataFrame API
//! for various data analysis tasks. It shows how to perform statistical
//! operations and machine learning tasks with seamless GPU acceleration.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_dataframe_api_example --features "cuda"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_dataframe_api_example

#[cfg(cuda_available)]
use pandrs::error::Result;
#[cfg(cuda_available)]
use pandrs::gpu::{init_gpu, GpuManager};
#[cfg(cuda_available)]
use pandrs::DataFrame;
#[cfg(cuda_available)]
use pandrs::Series;

#[cfg(cuda_available)]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS GPU-accelerated DataFrame API Example");
    println!("-------------------------------------------");

    // Initialize GPU with default configuration
    let device_status = init_gpu()?;

    println!("\nGPU Device Status:");
    println!("  Available: {}", device_status.available);

    if device_status.available {
        if let Some(name) = &device_status.device_name {
            println!("  Device Name: {}", name);
        }
        if let Some(memory) = device_status.total_memory {
            println!("  Total Memory: {} MB", memory / (1024 * 1024));
        }
        if let Some(cores) = device_status.core_count {
            println!("  Core Count: {}", cores);
        }
    }

    // Create a sample DataFrame
    let mut df = DataFrame::new();

    // Add some sample columns
    df.add_column(
        "x".to_string(),
        Series::new(
            (1..=100).map(|i| i as f64).collect::<Vec<_>>(),
            Some("x".to_string()),
        )?,
    )?;

    df.add_column(
        "y".to_string(),
        Series::new(
            (1..=100).map(|i| (i * 2) as f64 + 1.0).collect::<Vec<_>>(),
            Some("y".to_string()),
        )?,
    )?;

    println!("\nSample DataFrame created with {} rows", df.row_count());

    // Demonstrate available GPU operations
    println!("\n--- GPU-Accelerated Operations Available ---");
    println!("  - gpu_matmul: GPU matrix multiplication");
    println!("  - gpu_sum: GPU sum computation");
    println!("  - gpu_mean: GPU mean computation");
    println!("  - gpu_std: GPU standard deviation");
    println!("  - gpu_corr: GPU correlation matrix");
    println!("  - gpu_linear_regression: GPU linear regression");
    println!("  - gpu_pca: GPU principal component analysis");
    println!("  - gpu_kmeans: GPU k-means clustering");

    // Note: Full GPU acceleration requires CUDA feature and compatible hardware
    if !device_status.available {
        println!("\nNote: GPU not available. Operations will use CPU fallback.");
        println!("For GPU acceleration, ensure:");
        println!("  1. NVIDIA GPU with CUDA support");
        println!("  2. CUDA toolkit installed");
        println!("  3. Build with 'cuda' feature enabled");
    }

    println!("\n=== GPU DataFrame API Example Completed ===");
    Ok(())
}

#[cfg(not(cuda_available))]
fn main() {
    println!("PandRS GPU-accelerated DataFrame API Example");
    println!("-------------------------------------------");
    println!();
    println!("This example requires the 'cuda' feature to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_dataframe_api_example --features \"cuda\"");
    println!();
    println!("Note: CUDA support also requires:");
    println!("  - NVIDIA GPU with CUDA support");
    println!("  - CUDA toolkit installed");
}

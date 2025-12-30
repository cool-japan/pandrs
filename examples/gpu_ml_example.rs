//! GPU-accelerated machine learning example
//!
//! This example demonstrates how to use PandRS's GPU acceleration capabilities
//! for machine learning tasks.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_ml_example --features "cuda optimized"
//!
//! Note: Full GPU support requires NVIDIA CUDA toolkit and compatible GPU.

#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(cuda_available, feature = "optimized"))]
use pandrs::gpu::{init_gpu, GpuManager};

#[cfg(all(cuda_available, feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== PandRS GPU-accelerated Machine Learning Example ===\n");

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
            "  Memory: {} MB",
            device_status.total_memory.unwrap_or(0) / (1024 * 1024)
        );
        println!(
            "  CUDA Version: {}",
            device_status
                .cuda_version
                .unwrap_or_else(|| "Unknown".to_string())
        );
    }

    if !device_status.available {
        println!("\nNo GPU available. To run with GPU acceleration:");
        println!("1. Ensure NVIDIA GPU is available");
        println!("2. Install CUDA toolkit");
        println!("3. Rebuild with CUDA support");
    }

    println!("\n--- GPU ML Operations Demo ---");

    // Create GPU manager
    let gpu_manager = GpuManager::new();
    println!(
        "GPU Manager initialized: available={}",
        gpu_manager.is_available()
    );

    // Demonstrate basic GPU operations info
    println!("\nGPU-accelerated operations available:");
    println!("  - Matrix multiplication");
    println!("  - Element-wise operations");
    println!("  - Statistical computations");
    println!("  - Linear regression");
    println!("  - K-means clustering");
    println!("  - PCA dimensionality reduction");

    println!("\n=== GPU ML Example Completed ===");
    Ok(())
}

#[cfg(not(all(cuda_available, feature = "optimized")))]
fn main() {
    println!("GPU Machine Learning Example");
    println!("============================");
    println!();
    println!("This example requires both 'cuda' and 'optimized' feature flags.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_ml_example --features \"cuda optimized\"");
}

//! GPU-accelerated matrix operations example
//!
//! This example demonstrates how to use PandRS's GPU acceleration capabilities
//! for matrix operations. It shows basic operations like matrix multiplication
//! and element-wise operations, comparing CPU vs GPU performance.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_matrix_example --features "cuda"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_matrix_example

#[cfg(cuda_available)]
use ndarray::{arr2, Array2};
#[cfg(cuda_available)]
use pandrs::error::Result;
#[cfg(cuda_available)]
use pandrs::gpu::operations::{GpuMatrix, GpuVector};
#[cfg(cuda_available)]
use pandrs::gpu::{get_gpu_manager, init_gpu, init_gpu_with_config, GpuConfig, GpuError};
#[cfg(cuda_available)]
use std::time::Instant;

#[cfg(cuda_available)]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS GPU-accelerated Matrix Operations Example");
    println!("-----------------------------------------------");

    // Initialize GPU with custom configuration
    let gpu_config = GpuConfig {
        enabled: true,
        memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        device_id: 0,                         // Use first GPU
        fallback_to_cpu: true,                // Fall back to CPU if GPU fails
        use_pinned_memory: true,              // Use pinned memory for faster transfers
        min_size_threshold: 1000, // Only use GPU for operations with at least 1000 elements
    };

    // Initialize GPU and print device info
    let device_status = init_gpu_with_config(gpu_config)?;

    println!("\nGPU Device Status:");
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
        println!("  Core Count: {}", device_status.core_count.unwrap_or(0));
    } else {
        println!("  No CUDA-compatible GPU available. Using CPU fallback.");
    }

    // Create sample matrices for operations
    println!("\nCreating test matrices...");

    // Small matrices for demonstration
    let small_a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let small_b = arr2(&[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let gpu_small_a = GpuMatrix::new(small_a.clone());
    let gpu_small_b = GpuMatrix::new(small_b.clone());

    // Perform matrix multiplication
    println!("\nPerforming matrix multiplication on small matrices...");
    let result = gpu_small_a.dot(&gpu_small_b)?;
    println!("\nMatrix A (2x3):");
    print_matrix(&small_a);
    println!("\nMatrix B (3x2):");
    print_matrix(&small_b);
    println!("\nResult A × B (2x2):");
    print_matrix(&result.data);

    // Element-wise operations on matrices of the same size
    let small_c = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

    let small_d = arr2(&[[5.0, 6.0], [7.0, 8.0]]);

    let gpu_small_c = GpuMatrix::new(small_c.clone());
    let gpu_small_d = GpuMatrix::new(small_d.clone());

    println!("\nPerforming element-wise operations...");
    println!("\nMatrix C (2x2):");
    print_matrix(&small_c);
    println!("\nMatrix D (2x2):");
    print_matrix(&small_d);

    // Addition
    let addition_result = gpu_small_c.add(&gpu_small_d)?;
    println!("\nC + D:");
    print_matrix(&addition_result.data);

    // Subtraction
    let subtraction_result = gpu_small_c.subtract(&gpu_small_d)?;
    println!("\nC - D:");
    print_matrix(&subtraction_result.data);

    // Multiplication
    let multiplication_result = gpu_small_c.multiply(&gpu_small_d)?;
    println!("\nC * D (element-wise):");
    print_matrix(&multiplication_result.data);

    // Division
    let division_result = gpu_small_c.divide(&gpu_small_d)?;
    println!("\nC / D (element-wise):");
    print_matrix(&division_result.data);

    // Performance benchmark with larger matrices
    println!("\n\nPerformance Benchmark");
    println!("-----------------------");
    run_performance_benchmark()?;

    Ok(())
}

#[cfg(not(cuda_available))]
fn main() {
    println!("This example requires the 'cuda' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_matrix_example --features \"cuda\"");
}

#[cfg(cuda_available)]
fn print_matrix(matrix: &Array2<f64>) {
    for row in matrix.rows() {
        for val in row.iter() {
            print!("{:.2} ", val);
        }
        println!();
    }
}

#[cfg(cuda_available)]
#[allow(clippy::result_large_err)]
fn run_performance_benchmark() -> Result<()> {
    // Create larger matrices for benchmarking
    let sizes = [100, 500, 1000, 2000];
    let gpu_manager = get_gpu_manager()?;
    let is_gpu_available = gpu_manager.is_available();

    println!("Matrix multiplication benchmark (CPU vs GPU):");
    println!("Size | CPU Time (ms) | GPU Time (ms) | Speedup");
    println!("-----------------------------------------");

    for size in sizes {
        // Create random matrices
        let total_elements = size * size;
        let mut data_a = Vec::with_capacity(total_elements);
        let mut data_b = Vec::with_capacity(total_elements);

        for i in 0..total_elements {
            data_a.push((i % 10) as f64);
            data_b.push(((i + 5) % 10) as f64);
        }

        let a = Array2::from_shape_vec((size, size), data_a).unwrap();
        let b = Array2::from_shape_vec((size, size), data_b).unwrap();

        let gpu_a = GpuMatrix::new(a.clone());
        let gpu_b = GpuMatrix::new(b.clone());

        // Measure CPU time (always available)
        let cpu_start = Instant::now();
        let _cpu_result = gpu_a.dot_cpu(&gpu_b)?;
        let cpu_duration = cpu_start.elapsed().as_millis();

        // Measure GPU time (if available)
        let gpu_duration = if is_gpu_available {
            let gpu_start = Instant::now();
            let _gpu_result = gpu_a.dot(&gpu_b)?;
            gpu_start.elapsed().as_millis()
        } else {
            // If GPU is not available, use N/A
            0
        };

        // Calculate speedup
        let speedup = if is_gpu_available && gpu_duration > 0 {
            cpu_duration as f64 / gpu_duration as f64
        } else {
            0.0
        };

        if is_gpu_available {
            println!(
                "{:4} | {:13} | {:13} | {:.2}x",
                size, cpu_duration, gpu_duration, speedup
            );
        } else {
            println!(
                "{:4} | {:13} | {:13} | {}",
                size, cpu_duration, "N/A", "N/A"
            );
        }
    }

    Ok(())
}

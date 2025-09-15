//! GPU-accelerated statistical functions example
//!
//! This example demonstrates the use of GPU-accelerated statistical functions
//! in PandRS. It compares the performance of CPU vs GPU implementations for
//! various statistical operations on large datasets.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_stats_example --features "cuda"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_stats_example

#[cfg(feature = "cuda")]
use ndarray::{Array1, Array2};
#[cfg(feature = "cuda")]
use pandrs::error::Result;
#[cfg(feature = "cuda")]
use pandrs::gpu::operations::{GpuMatrix, GpuVector};
#[cfg(feature = "cuda")]
use pandrs::gpu::{get_gpu_manager, init_gpu, GpuConfig, GpuManager};
#[cfg(feature = "cuda")]
use pandrs::DataFrame;
#[cfg(feature = "cuda")]
use pandrs::Series;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use pandrs::stats::gpu;

#[cfg(feature = "cuda")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS GPU-accelerated Statistical Functions Example");
    println!("---------------------------------------------------");

    // Initialize GPU with default configuration
    let device_status = init_gpu()?;

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
    } else {
        println!("  No CUDA-compatible GPU available. Using CPU fallback.");
    }

    // Generate large test dataset
    println!("\nGenerating test dataset...");
    let (data_matrix, data_vector) = generate_test_data(1_000, 20);
    println!(
        "Dataset generated with {} rows and {} columns",
        data_matrix.dim().0,
        data_matrix.dim().1
    );

    // Benchmark statistical functions
    benchmark_correlation_matrix(&data_matrix)?;
    benchmark_covariance_matrix(&data_matrix)?;
    benchmark_descriptive_stats(&data_vector)?;

    Ok(())
}

#[cfg(feature = "cuda")]
/// Generate test data for benchmarking
fn generate_test_data(rows: usize, cols: usize) -> (Array2<f64>, Array1<f64>) {
    // Generate matrix data
    let mut matrix_data = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        matrix_data.push((i % 100) as f64);
    }
    let matrix = Array2::from_shape_vec((rows, cols), matrix_data).unwrap();

    // Generate vector data
    let mut vector_data = Vec::with_capacity(rows);
    for i in 0..rows {
        vector_data.push((i % 100) as f64);
    }
    let vector = Array1::from_vec(vector_data);

    (matrix, vector)
}

#[cfg(feature = "cuda")]
/// Benchmark correlation matrix computation
#[allow(clippy::result_large_err)]
fn benchmark_correlation_matrix(data: &Array2<f64>) -> Result<()> {
    println!("\nCorrelation Matrix Benchmark");
    println!("----------------------------");

    // CPU implementation
    println!("\nComputing correlation matrix on CPU...");
    let cpu_start = Instant::now();

    // Compute using regular CPU implementation
    let (rows, cols) = data.dim();
    let mut cpu_corr_matrix = Array2::zeros((cols, cols));

    // Calculate means
    let mut means = Vec::with_capacity(cols);
    for col_idx in 0..cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Compute correlation coefficients
    for i in 0..cols {
        // Diagonal elements are always 1
        cpu_corr_matrix[[i, i]] = 1.0;

        for j in (i + 1)..cols {
            // Calculate correlation coefficient
            let mut cov_sum = 0.0;
            let mut var_i_sum = 0.0;
            let mut var_j_sum = 0.0;

            for row_idx in 0..rows {
                let x_i = data[[row_idx, i]] - means[i];
                let x_j = data[[row_idx, j]] - means[j];

                cov_sum += x_i * x_j;
                var_i_sum += x_i * x_i;
                var_j_sum += x_j * x_j;
            }

            // Calculate correlation coefficient
            let corr_ij = cov_sum / (var_i_sum.sqrt() * var_j_sum.sqrt());

            // Store in correlation matrix (symmetric)
            cpu_corr_matrix[[i, j]] = corr_ij;
            cpu_corr_matrix[[j, i]] = corr_ij;
        }
    }

    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("  CPU Time: {} ms", cpu_duration);

    // GPU implementation (when available)
    #[cfg(feature = "cuda")]
    {
        let gpu_manager = get_gpu_manager()?;
        if gpu_manager.is_available() {
            println!("\nComputing correlation matrix on GPU...");
            let gpu_start = Instant::now();

            // Use GPU-accelerated implementation
            let gpu_corr_matrix = gpu::correlation_matrix(data)?;

            let gpu_duration = gpu_start.elapsed().as_millis();
            println!("  GPU Time: {} ms", gpu_duration);

            // Calculate speedup
            let speedup = if gpu_duration > 0 {
                cpu_duration as f64 / gpu_duration as f64
            } else {
                0.0
            };
            println!("\nSpeedup: {:.2}x", speedup);

            // Verify results match (approximately)
            let mut max_diff = 0.0;
            for i in 0..cols {
                for j in 0..cols {
                    let diff = (cpu_corr_matrix[[i, j]] - gpu_corr_matrix[[i, j]]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
            println!(
                "Maximum difference between CPU and GPU results: {:.6}",
                max_diff
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark covariance matrix computation
#[allow(clippy::result_large_err)]
fn benchmark_covariance_matrix(data: &Array2<f64>) -> Result<()> {
    println!("\nCovariance Matrix Benchmark");
    println!("---------------------------");

    // CPU implementation
    println!("\nComputing covariance matrix on CPU...");
    let cpu_start = Instant::now();

    // Compute using regular CPU implementation
    let (rows, cols) = data.dim();
    let mut cpu_cov_matrix = Array2::zeros((cols, cols));

    // Calculate means
    let mut means = Vec::with_capacity(cols);
    for col_idx in 0..cols {
        means.push(data.column(col_idx).mean().unwrap_or(0.0));
    }

    // Compute covariance matrix
    for i in 0..cols {
        for j in i..cols {
            // Calculate covariance
            let mut cov_sum = 0.0;

            for row_idx in 0..rows {
                let x_i = data[[row_idx, i]] - means[i];
                let x_j = data[[row_idx, j]] - means[j];

                cov_sum += x_i * x_j;
            }

            // Calculate covariance
            let cov_ij = cov_sum / (rows - 1) as f64;

            // Store in covariance matrix (symmetric)
            cpu_cov_matrix[[i, j]] = cov_ij;
            cpu_cov_matrix[[j, i]] = cov_ij;
        }
    }

    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("  CPU Time: {} ms", cpu_duration);

    // GPU implementation (when available)
    #[cfg(feature = "cuda")]
    {
        let gpu_manager = get_gpu_manager()?;
        if gpu_manager.is_available() {
            println!("\nComputing covariance matrix on GPU...");
            let gpu_start = Instant::now();

            // Use GPU-accelerated implementation
            let gpu_cov_matrix = gpu::covariance_matrix(data)?;

            let gpu_duration = gpu_start.elapsed().as_millis();
            println!("  GPU Time: {} ms", gpu_duration);

            // Calculate speedup
            let speedup = if gpu_duration > 0 {
                cpu_duration as f64 / gpu_duration as f64
            } else {
                0.0
            };
            println!("\nSpeedup: {:.2}x", speedup);

            // Verify results match (approximately)
            let mut max_diff = 0.0;
            for i in 0..cols {
                for j in 0..cols {
                    let diff = (cpu_cov_matrix[[i, j]] - gpu_cov_matrix[[i, j]]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
            println!(
                "Maximum difference between CPU and GPU results: {:.6}",
                max_diff
            );
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
/// Benchmark descriptive statistics computation
#[allow(clippy::result_large_err)]
fn benchmark_descriptive_stats(data: &Array1<f64>) -> Result<()> {
    println!("\nDescriptive Statistics Benchmark");
    println!("--------------------------------");

    // CPU implementation
    println!("\nComputing descriptive statistics on CPU...");
    let cpu_start = Instant::now();

    // Compute using regular CPU implementation
    let data_slice = data.as_slice().unwrap();
    let cpu_stats = pandrs::stats::describe(data_slice)?;

    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("  CPU Time: {} ms", cpu_duration);
    println!(
        "  Summary: Mean={:.4}, StdDev={:.4}",
        cpu_stats.mean, cpu_stats.std
    );

    // GPU implementation (when available)
    #[cfg(feature = "cuda")]
    {
        let gpu_manager = get_gpu_manager()?;
        if gpu_manager.is_available() {
            println!("\nComputing descriptive statistics on GPU...");
            let gpu_start = Instant::now();

            // Use GPU-accelerated implementation
            let gpu_stats = gpu::describe_gpu(data_slice)?;

            let gpu_duration = gpu_start.elapsed().as_millis();
            println!("  GPU Time: {} ms", gpu_duration);
            println!(
                "  Summary: Mean={:.4}, StdDev={:.4}",
                gpu_stats.mean, gpu_stats.std
            );

            // Calculate speedup
            let speedup = if gpu_duration > 0 {
                cpu_duration as f64 / gpu_duration as f64
            } else {
                0.0
            };
            println!("\nSpeedup: {:.2}x", speedup);

            // Verify results match (approximately)
            let mean_diff = (cpu_stats.mean - gpu_stats.mean).abs();
            let std_diff = (cpu_stats.std - gpu_stats.std).abs();
            println!("Differences: Mean={:.6}, StdDev={:.6}", mean_diff, std_diff);
        }
    }

    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the \"cuda\" feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_stats_example --features \"cuda\"");
}

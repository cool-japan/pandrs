//! GPU-accelerated DataFrame operations example
//!
//! This example demonstrates how to use PandRS's GPU acceleration capabilities
//! with the DataFrame API for large-scale data processing. It shows how to
//! perform common operations like filtering, aggregation, and transformations
//! with GPU acceleration when available.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_dataframe_example --features "cuda optimized"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_dataframe_example --features "optimized"

#[cfg(all(feature = "cuda", feature = "optimized"))]
use ndarray::{Array1, Array2};
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::dataframe::view::ViewExt; // Add ViewExt trait for head() method
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::gpu::operations::GpuAccelerated;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::gpu::{get_gpu_manager, init_gpu, GpuConfig, GpuError, GpuManager};
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::optimized::dataframe::OptimizedDataFrame;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::DataFrame;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::Series;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS GPU-accelerated DataFrame Example");
    println!("----------------------------------------");

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

    // Create a sample DataFrame
    println!("\nCreating sample DataFrame...");
    let mut df = create_sample_dataframe(100_000)?;
    println!("DataFrame created with 100,000 rows and 5 columns");

    // Print first few rows
    println!("\nFirst 5 rows of DataFrame:");
    println!("{}", df.head(5)?);

    // Benchmark operations
    benchmark_dataframe_operations(&mut df)?;

    // Benchmark with OptimizedDataFrame (better GPU acceleration integration)
    println!("\nOptimizedDataFrame Performance");
    println!("-----------------------------");
    let mut opt_df = create_optimized_dataframe(100_000)?;
    benchmark_optimized_dataframe_operations(&mut opt_df)?;

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "optimized")))]
fn main() {
    println!("This example requires both 'cuda' and 'optimized' feature flags to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_dataframe_example --features \"cuda optimized\"");
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn create_sample_dataframe(size: usize) -> Result<DataFrame> {
    // Create data for columns
    let mut ids = Vec::with_capacity(size);
    let mut values1 = Vec::with_capacity(size);
    let mut values2 = Vec::with_capacity(size);
    let mut values3 = Vec::with_capacity(size);
    let mut categories = Vec::with_capacity(size);

    for i in 0..size {
        ids.push(i as i64);
        values1.push((i % 100) as f64);
        values2.push(((i * 2) % 100) as f64);
        values3.push(((i * 3) % 100) as f64);
        categories.push(format!("Category {}", i % 5));
    }

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column("id".to_string(), Series::new(ids, Some("id".to_string()))?)?;
    df.add_column(
        "value1".to_string(),
        Series::new(values1, Some("value1".to_string()))?,
    )?;
    df.add_column(
        "value2".to_string(),
        Series::new(values2, Some("value2".to_string()))?,
    )?;
    df.add_column(
        "value3".to_string(),
        Series::new(values3, Some("value3".to_string()))?,
    )?;
    df.add_column(
        "category".to_string(),
        Series::new(categories, Some("category".to_string()))?,
    )?;

    Ok(df)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn create_optimized_dataframe(size: usize) -> Result<OptimizedDataFrame> {
    // Create data for columns
    let mut ids = Vec::with_capacity(size);
    let mut values1 = Vec::with_capacity(size);
    let mut values2 = Vec::with_capacity(size);
    let mut values3 = Vec::with_capacity(size);
    let mut categories = Vec::with_capacity(size);

    for i in 0..size {
        ids.push(i as i64);
        values1.push((i % 100) as f64);
        values2.push(((i * 2) % 100) as f64);
        values3.push(((i * 3) % 100) as f64);
        categories.push(format!("Category {}", i % 5));
    }

    // Create OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("id", ids)?;
    df.add_float_column("value1", values1)?;
    df.add_float_column("value2", values2)?;
    df.add_float_column("value3", values3)?;
    df.add_string_column("category", categories)?;

    Ok(df)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn benchmark_dataframe_operations(df: &mut DataFrame) -> Result<()> {
    println!("\nDataFrame Operations Benchmark");
    println!("-------------------------------");

    let gpu_manager = get_gpu_manager()?;
    let is_gpu_available = gpu_manager.is_available();

    // Convert to GPU-accelerated if available
    let gpu_df = if is_gpu_available {
        // Enable GPU acceleration if available
        match df.gpu_accelerate() {
            Ok(acc_df) => {
                println!("GPU acceleration enabled for DataFrame");
                acc_df
            }
            Err(e) => {
                println!("Warning: Failed to enable GPU acceleration: {}", e);
                df.clone()
            }
        }
    } else {
        df.clone()
    };

    // Benchmark operations (CPU)
    println!("\nBenchmark regular operations on CPU:");

    // Filtering
    let cpu_start = Instant::now();
    let filtered_df = df.filter("value1", |x| x.as_f64().map_or(false, |val| val > 50.0))?;
    let cpu_filter_time = cpu_start.elapsed().as_millis();
    println!(
        "  CPU Filter: {} ms (rows: {})",
        cpu_filter_time,
        filtered_df.row_count()
    );

    // Aggregation
    let cpu_start = Instant::now();
    let mean_value = df.mean("value1")?;
    let cpu_agg_time = cpu_start.elapsed().as_millis();
    println!("  CPU Mean: {} ms (result: {})", cpu_agg_time, mean_value);

    // Benchmark operations (GPU-accelerated)
    if is_gpu_available {
        println!("\nBenchmark operations with GPU acceleration:");

        // Filtering
        let gpu_start = Instant::now();
        let gpu_filtered_df =
            gpu_df.filter("value1", |x| x.as_f64().map_or(false, |val| val > 50.0))?;
        let gpu_filter_time = gpu_start.elapsed().as_millis();
        println!(
            "  GPU Filter: {} ms (rows: {})",
            gpu_filter_time,
            gpu_filtered_df.row_count()
        );

        // Aggregation
        let gpu_start = Instant::now();
        let gpu_mean_value = gpu_df.mean("value1")?;
        let gpu_agg_time = gpu_start.elapsed().as_millis();
        println!(
            "  GPU Mean: {} ms (result: {})",
            gpu_agg_time, gpu_mean_value
        );

        // Calculate speedup
        let filter_speedup = if gpu_filter_time > 0 {
            cpu_filter_time as f64 / gpu_filter_time as f64
        } else {
            0.0
        };
        let agg_speedup = if gpu_agg_time > 0 {
            cpu_agg_time as f64 / gpu_agg_time as f64
        } else {
            0.0
        };

        println!("\nSpeedup summary:");
        println!("  Filter: {:.2}x", filter_speedup);
        println!("  Mean: {:.2}x", agg_speedup);
    }

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
#[allow(clippy::result_large_err)]
fn benchmark_optimized_dataframe_operations(df: &mut OptimizedDataFrame) -> Result<()> {
    println!("\nOptimizedDataFrame Operations Benchmark");
    println!("----------------------------------------");

    let gpu_manager = get_gpu_manager()?;
    let is_gpu_available = gpu_manager.is_available();

    // Convert to GPU-accelerated if available
    let gpu_df = if is_gpu_available {
        // Enable GPU acceleration if available
        match df.gpu_accelerate() {
            Ok(acc_df) => {
                println!("GPU acceleration enabled for OptimizedDataFrame");
                acc_df
            }
            Err(e) => {
                println!("Warning: Failed to enable GPU acceleration: {}", e);
                df.clone()
            }
        }
    } else {
        df.clone()
    };

    // Benchmark operations (CPU)
    println!("\nBenchmark operations on CPU:");

    // Filtering
    let cpu_start = Instant::now();
    let filtered_df = df.filter_f64("value1", |val| val > 50.0)?;
    let cpu_filter_time = cpu_start.elapsed().as_millis();
    println!(
        "  CPU Filter: {} ms (rows: {})",
        cpu_filter_time,
        filtered_df.row_count()
    );

    // Aggregation
    let cpu_start = Instant::now();
    let mean_value = df.mean_f64("value1")?;
    let cpu_agg_time = cpu_start.elapsed().as_millis();
    println!("  CPU Mean: {} ms (result: {})", cpu_agg_time, mean_value);

    // Create correlation matrix
    let cpu_start = Instant::now();
    let corr_matrix = df.corr_matrix(&["value1", "value2", "value3"])?;
    let cpu_corr_time = cpu_start.elapsed().as_millis();
    println!("  CPU Correlation Matrix: {} ms", cpu_corr_time);

    // Benchmark operations (GPU-accelerated)
    if is_gpu_available {
        println!("\nBenchmark operations with GPU acceleration:");

        // Filtering
        let gpu_start = Instant::now();
        let gpu_filtered_df = gpu_df.filter_f64("value1", |val| val > 50.0)?;
        let gpu_filter_time = gpu_start.elapsed().as_millis();
        println!(
            "  GPU Filter: {} ms (rows: {})",
            gpu_filter_time,
            gpu_filtered_df.row_count()
        );

        // Aggregation
        let gpu_start = Instant::now();
        let gpu_mean_value = gpu_df.mean_f64("value1")?;
        let gpu_agg_time = gpu_start.elapsed().as_millis();
        println!(
            "  GPU Mean: {} ms (result: {})",
            gpu_agg_time, gpu_mean_value
        );

        // Create correlation matrix
        let gpu_start = Instant::now();
        let gpu_corr_matrix = gpu_df.corr_matrix(&["value1", "value2", "value3"])?;
        let gpu_corr_time = gpu_start.elapsed().as_millis();
        println!("  GPU Correlation Matrix: {} ms", gpu_corr_time);

        // Calculate speedup
        let filter_speedup = if gpu_filter_time > 0 {
            cpu_filter_time as f64 / gpu_filter_time as f64
        } else {
            0.0
        };
        let agg_speedup = if gpu_agg_time > 0 {
            cpu_agg_time as f64 / gpu_agg_time as f64
        } else {
            0.0
        };
        let corr_speedup = if gpu_corr_time > 0 {
            cpu_corr_time as f64 / gpu_corr_time as f64
        } else {
            0.0
        };

        println!("\nSpeedup summary:");
        println!("  Filter: {:.2}x", filter_speedup);
        println!("  Mean: {:.2}x", agg_speedup);
        println!("  Correlation: {:.2}x", corr_speedup);
    }

    Ok(())
}

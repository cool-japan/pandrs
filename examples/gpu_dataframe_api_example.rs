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

use pandrs::error::Result;
#[cfg(feature = "cuda")]
use pandrs::gpu::{get_gpu_manager, init_gpu, GpuConfig};
use pandrs::DataFrame;
use pandrs::Series;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::time::Instant;

// This import is only available when the "cuda" feature is enabled
#[cfg(feature = "cuda")]
use pandrs::DataFrameGpuExt;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    println!("PandRS GPU-accelerated DataFrame API Example");
    println!("-------------------------------------------");

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

    // Create a sample DataFrame for demonstrations
    let mut df = create_sample_dataframe(10_000)?;
    println!(
        "\nCreated sample DataFrame with {} rows and {} columns",
        df.row_count(),
        df.column_count()
    );

    // Show the first few rows
    println!("\nFirst 5 rows:");
    println!("{}", df.head(5)?);

    // Perform GPU-accelerated operations
    // Correlation matrix
    benchmark_correlation(&df)?;

    // Linear regression
    benchmark_linear_regression(&df)?;

    // Principal Component Analysis (PCA)
    benchmark_pca(&df)?;

    // K-means clustering
    benchmark_kmeans(&df)?;

    // Descriptive statistics
    benchmark_describe(&df)?;

    Ok(())
}

/// Create a sample DataFrame for benchmarking
#[allow(dead_code)]
#[allow(clippy::result_large_err)]
fn create_sample_dataframe(size: usize) -> Result<DataFrame> {
    // Create data for columns
    let mut x1 = Vec::with_capacity(size);
    let mut x2 = Vec::with_capacity(size);
    let mut x3 = Vec::with_capacity(size);
    let mut x4 = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);

    for i in 0..size {
        // Create features with some correlation to the target
        let x1_val = (i % 100) as f64 / 100.0;
        let x2_val = ((i * 2) % 100) as f64 / 100.0;
        let x3_val = ((i * 3) % 100) as f64 / 100.0;
        let x4_val = ((i * 5) % 100) as f64 / 100.0;

        // Create a target variable that depends on the features
        let y_val = 2.0 * x1_val + 1.5 * x2_val - 0.5 * x3_val
            + 3.0 * x4_val
            + 0.1 * (rand::random::<f64>() - 0.5); // Add some noise

        x1.push(x1_val);
        x2.push(x2_val);
        x3.push(x3_val);
        x4.push(x4_val);
        y.push(y_val);
    }

    // Create DataFrame
    let mut df = DataFrame::new();
    df.add_column("x1".to_string(), Series::new(x1, Some("x1".to_string()))?)?;
    df.add_column("x2".to_string(), Series::new(x2, Some("x2".to_string()))?)?;
    df.add_column("x3".to_string(), Series::new(x3, Some("x3".to_string()))?)?;
    df.add_column("x4".to_string(), Series::new(x4, Some("x4".to_string()))?)?;
    df.add_column("y".to_string(), Series::new(y, Some("y".to_string()))?)?;

    Ok(df)
}

#[cfg(feature = "cuda")]
/// Print a preview of a matrix for display purposes
fn print_matrix_preview(matrix: &ndarray::Array2<f64>, max_rows: usize, max_cols: usize) {
    let (rows, cols) = matrix.dim();
    let rows_to_show = rows.min(max_rows);
    let cols_to_show = cols.min(max_cols);

    for i in 0..rows_to_show {
        for j in 0..cols_to_show {
            print!("{:.4} ", matrix[[i, j]]);
        }
        print!("...\n");
    }
    println!("...");
}

#[cfg(feature = "cuda")]
fn benchmark_correlation(df: &DataFrame) -> Result<()> {
    println!("\nCorrelation Matrix Benchmark");
    println!("----------------------------");

    // Standard correlation (CPU)
    let cpu_start = Instant::now();
    let cpu_corr = df.corr()?;
    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("\nStandard correlation (CPU):");
    println!("  CPU Time: {} ms", cpu_duration);
    print_matrix_preview(&cpu_corr, 5, 5);

    // GPU-accelerated correlation
    let gpu_start = Instant::now();
    let feature_cols = ["x1", "x2", "x3", "x4"];
    let gpu_corr = df.gpu_corr(&feature_cols)?;
    let gpu_duration = gpu_start.elapsed().as_millis();
    println!("\nGPU-accelerated correlation:");
    println!("  GPU Time: {} ms", gpu_duration);
    print_matrix_preview(&gpu_corr, 5, 5);

    // Calculate speedup
    let speedup = if gpu_duration > 0 {
        cpu_duration as f64 / gpu_duration as f64
    } else {
        0.0
    };
    println!("\nSpeedup: {:.2}x", speedup);

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_linear_regression(df: &DataFrame) -> Result<()> {
    println!("\nLinear Regression Benchmark");
    println!("--------------------------");

    // Standard linear regression (CPU)
    let cpu_start = Instant::now();
    let cpu_model = pandrs::stats::linear_regression(df, "y", &["x1", "x2", "x3", "x4"])?;
    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("\nStandard linear regression (CPU):");
    println!("  CPU Time: {} ms", cpu_duration);
    println!("  Intercept: {:.4}", cpu_model.intercept);
    println!(
        "  Coefficients: [{:.4}, {:.4}, {:.4}, {:.4}]",
        cpu_model.coefficients[0],
        cpu_model.coefficients[1],
        cpu_model.coefficients[2],
        cpu_model.coefficients[3]
    );
    println!("  R²: {:.4}", cpu_model.r_squared);

    // GPU-accelerated linear regression
    let gpu_start = Instant::now();
    let feature_cols = ["x1", "x2", "x3", "x4"];
    let gpu_model = df.gpu_linear_regression("y", &feature_cols)?;
    let gpu_duration = gpu_start.elapsed().as_millis();
    println!("\nGPU-accelerated linear regression:");
    println!("  GPU Time: {} ms", gpu_duration);
    println!("  Intercept: {:.4}", gpu_model.intercept);
    println!(
        "  Coefficients: [{:.4}, {:.4}, {:.4}, {:.4}]",
        gpu_model.coefficients["x1"],
        gpu_model.coefficients["x2"],
        gpu_model.coefficients["x3"],
        gpu_model.coefficients["x4"]
    );
    println!("  R²: {:.4}", gpu_model.r_squared);

    // Calculate speedup
    let speedup = if gpu_duration > 0 {
        cpu_duration as f64 / gpu_duration as f64
    } else {
        0.0
    };
    println!("\nSpeedup: {:.2}x", speedup);

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_pca(df: &DataFrame) -> Result<()> {
    println!("\nPrincipal Component Analysis (PCA) Benchmark");
    println!("-------------------------------------------");

    // Standard PCA (using optimized implementation for CPU)
    let cpu_start = Instant::now();
    let feature_cols = ["x1", "x2", "x3", "x4"];
    let n_components = 2;
    let opt_df = pandrs::OptimizedDataFrame::from_dataframe(df)?;
    let (components, variance, _) = opt_df.pca(
        &feature_cols
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        n_components,
    )?;
    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("\nStandard PCA (CPU):");
    println!("  CPU Time: {} ms", cpu_duration);
    println!(
        "  Explained variance: [{:.4}, {:.4}]",
        variance[0], variance[1]
    );

    // GPU-accelerated PCA
    let gpu_start = Instant::now();
    let (pca_df, explained_variance) = df.gpu_pca(&feature_cols, n_components)?;
    let gpu_duration = gpu_start.elapsed().as_millis();
    println!("\nGPU-accelerated PCA:");
    println!("  GPU Time: {} ms", gpu_duration);
    println!(
        "  Explained variance: [{:.4}, {:.4}]",
        explained_variance[0], explained_variance[1]
    );
    println!("  First few rows of transformed data:");
    println!("{}", pca_df.head(5)?);

    // Calculate speedup
    let speedup = if gpu_duration > 0 {
        cpu_duration as f64 / gpu_duration as f64
    } else {
        0.0
    };
    println!("\nSpeedup: {:.2}x", speedup);

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_kmeans(df: &DataFrame) -> Result<()> {
    println!("\nK-means Clustering Benchmark");
    println!("---------------------------");

    // Standard k-means (CPU)
    let cpu_start = Instant::now();
    let feature_cols = ["x1", "x2", "x3", "x4"];
    let k = 3;
    let max_iter = 100;
    // Use a simple implementation for CPU comparison
    let opt_df = pandrs::OptimizedDataFrame::from_dataframe(df)?;
    let result = pandrs::ml::clustering::kmeans(
        &opt_df,
        &feature_cols
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        k,
        max_iter,
        None,
    )?;
    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("\nStandard k-means (CPU):");
    println!("  CPU Time: {} ms", cpu_duration);
    println!("  Number of iterations: {}", result.n_iter);
    println!("  Inertia: {:.4}", result.inertia);

    // GPU-accelerated k-means
    let gpu_start = Instant::now();
    let (centroids, labels, inertia) = df.gpu_kmeans(&feature_cols, k, max_iter)?;
    let gpu_duration = gpu_start.elapsed().as_millis();
    println!("\nGPU-accelerated k-means:");
    println!("  GPU Time: {} ms", gpu_duration);
    println!("  Inertia: {:.4}", inertia);
    println!("  Centroids shape: {:?}", centroids.dim());

    // Calculate cluster sizes
    let mut cluster_sizes = HashMap::new();
    for &label in labels.iter() {
        *cluster_sizes.entry(label).or_insert(0) += 1;
    }
    println!("  Cluster sizes:");
    for (cluster, size) in cluster_sizes.iter() {
        println!("    Cluster {}: {} samples", cluster, size);
    }

    // Calculate speedup
    let speedup = if gpu_duration > 0 {
        cpu_duration as f64 / gpu_duration as f64
    } else {
        0.0
    };
    println!("\nSpeedup: {:.2}x", speedup);

    Ok(())
}

#[cfg(feature = "cuda")]
fn benchmark_describe(df: &DataFrame) -> Result<()> {
    println!("\nDescriptive Statistics Benchmark");
    println!("--------------------------------");

    // Standard describe (CPU)
    let cpu_start = Instant::now();
    let cpu_stats = df.describe()?;
    let cpu_duration = cpu_start.elapsed().as_millis();
    println!("\nStandard describe (CPU):");
    println!("  CPU Time: {} ms", cpu_duration);
    println!("{}", cpu_stats);

    // GPU-accelerated describe
    let gpu_start = Instant::now();
    let gpu_stats = df.gpu_describe("y")?;
    let gpu_duration = gpu_start.elapsed().as_millis();
    println!("\nGPU-accelerated describe for column 'y':");
    println!("  GPU Time: {} ms", gpu_duration);
    println!("  Count: {}", gpu_stats.count);
    println!("  Mean: {:.4}", gpu_stats.mean);
    println!("  Std: {:.4}", gpu_stats.std);
    println!("  Min: {:.4}", gpu_stats.min);
    println!("  25%: {:.4}", gpu_stats.q1);
    println!("  50%: {:.4}", gpu_stats.median);
    println!("  75%: {:.4}", gpu_stats.q3);
    println!("  Max: {:.4}", gpu_stats.max);

    // Calculate speedup (per column)
    let cpu_per_column = cpu_duration as f64 / df.column_count() as f64;
    let speedup = if gpu_duration > 0 {
        cpu_per_column / gpu_duration as f64
    } else {
        0.0
    };
    println!("\nSpeedup (per column): {:.2}x", speedup);

    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    println!("This example requires the \"cuda\" feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_dataframe_api_example --features \"cuda\"");
}

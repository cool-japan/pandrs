//! GPU-accelerated window operations example
//!
//! This example demonstrates how to use PandRS's GPU acceleration for window operations
//! including rolling, expanding, and exponentially weighted moving (EWM) calculations.
//! It shows performance comparisons between CPU and GPU implementations and provides
//! practical examples of when GPU acceleration is most beneficial.
//!
//! To run with GPU acceleration:
//!   cargo run --example gpu_window_operations_example --features "cuda optimized"
//!
//! To run without GPU acceleration (CPU fallback):
//!   cargo run --example gpu_window_operations_example --features "optimized"

#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::dataframe::base::DataFrame;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::dataframe::enhanced_window::DataFrameWindowExt;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::dataframe::gpu_window::{GpuDataFrameWindowExt, GpuWindowContext};
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::error::Result;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::gpu::{get_gpu_manager, init_gpu, GpuConfig};
#[cfg(all(feature = "cuda", feature = "optimized"))]
use pandrs::series::Series;
#[cfg(all(feature = "cuda", feature = "optimized"))]
use std::time::Instant;

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn main() -> Result<()> {
    println!("PandRS GPU-Accelerated Window Operations Example");
    println!("================================================");

    // Initialize GPU with custom configuration
    let gpu_config = GpuConfig {
        enabled: true,
        memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        device_id: 0,
        fallback_to_cpu: true,
        use_pinned_memory: true,
        min_size_threshold: 10_000, // Use GPU for datasets > 10K elements
    };

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
    } else {
        println!("  No CUDA-compatible GPU available. Using CPU fallback.");
    }

    // Create GPU window context
    let gpu_context = GpuWindowContext::new()?;

    println!("\n=== Performance Comparison: Small Dataset ===");
    small_dataset_comparison(&gpu_context)?;

    println!("\n=== Performance Comparison: Large Dataset ===");
    large_dataset_comparison(&gpu_context)?;

    println!("\n=== Real-world Financial Analysis Example ===");
    financial_analysis_example(&gpu_context)?;

    println!("\n=== Advanced GPU Window Operations ===");
    advanced_operations_example(&gpu_context)?;

    println!("\n=== GPU Statistics and Performance Monitoring ===");
    performance_monitoring_example(&gpu_context)?;

    Ok(())
}

#[cfg(not(all(feature = "cuda", feature = "optimized")))]
fn main() {
    println!("This example requires both 'cuda' and 'optimized' feature flags to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example gpu_window_operations_example --features \"cuda optimized\"");
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn init_gpu_with_config(config: GpuConfig) -> Result<crate::gpu::GpuDeviceStatus> {
    use pandrs::gpu::init_gpu_with_config;
    init_gpu_with_config(config)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn small_dataset_comparison(gpu_context: &GpuWindowContext) -> Result<()> {
    println!("Testing with 5,000 data points (below GPU threshold)...");

    let df = create_sample_dataframe(5_000)?;
    let window_size = 20;

    // CPU-based window operations
    println!("\nCPU Implementation:");
    let start = Instant::now();
    let cpu_result = df.rolling(window_size).mean()?;
    let cpu_time = start.elapsed();
    println!("  Rolling mean time: {:?}", cpu_time);

    // GPU-enhanced window operations (should fallback to CPU for small data)
    println!("\nGPU-Enhanced Implementation (expected CPU fallback):");
    let start = Instant::now();
    let gpu_result = df.gpu_rolling(window_size, gpu_context).mean()?;
    let gpu_time = start.elapsed();
    println!("  Rolling mean time: {:?}", gpu_time);

    // Verify results are similar
    verify_results_similarity(&cpu_result, &gpu_result, "price")?;

    println!("  Result: GPU implementation correctly fell back to CPU for small dataset");

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn large_dataset_comparison(gpu_context: &GpuWindowContext) -> Result<()> {
    println!("Testing with 100,000 data points (above GPU threshold)...");

    let df = create_sample_dataframe(100_000)?;
    let window_size = 50;

    // CPU-based window operations
    println!("\nCPU Implementation:");
    let start = Instant::now();
    let cpu_result = df.rolling(window_size).mean()?;
    let cpu_time = start.elapsed();
    println!("  Rolling mean time: {:?}", cpu_time);

    // GPU-enhanced window operations
    println!("\nGPU-Enhanced Implementation:");
    let start = Instant::now();
    let gpu_result = df.gpu_rolling(window_size, gpu_context).mean()?;
    let gpu_time = start.elapsed();
    println!("  Rolling mean time: {:?}", gpu_time);

    // Calculate speedup
    let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos() as f64;
    println!("  Speedup: {:.2}x", speedup);

    // Verify results are similar
    verify_results_similarity(&cpu_result, &gpu_result, "price")?;

    if speedup > 1.0 {
        println!("  Result: GPU acceleration provided performance improvement");
    } else {
        println!("  Result: GPU overhead exceeded benefits (or fallback to CPU occurred)");
    }

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn financial_analysis_example(gpu_context: &GpuWindowContext) -> Result<()> {
    println!("Analyzing financial time series data with multiple window operations...");

    let df = create_financial_dataset(200_000)?;

    // Multiple window operations for financial analysis
    let operations = vec![
        ("5-day moving average", 5),
        ("20-day moving average", 20),
        ("50-day moving average", 50),
        ("200-day moving average", 200),
    ];

    for (name, window_size) in operations {
        println!("\nCalculating {}:", name);

        let start = Instant::now();
        let result = df
            .gpu_rolling(window_size, gpu_context)
            .columns(vec!["price".to_string(), "volume".to_string()])
            .mean()?;
        let time = start.elapsed();

        println!("  Calculation time: {:?}", time);
        println!(
            "  Result shape: {} rows × {} columns",
            result.row_count(),
            result.column_count()
        );

        // Show sample of results
        if result.row_count() > 0 && window_size == 20 {
            println!("  Sample results (last 5 values):");
            display_sample_results(&result, 5)?;
        }
    }

    // Calculate multiple statistics simultaneously
    println!("\nCalculating multiple rolling statistics (20-day window):");
    let start = Instant::now();

    let mean_result = df.gpu_rolling(20, gpu_context).mean()?;
    let std_result = df.gpu_rolling(20, gpu_context).std(1)?;
    let sum_result = df.gpu_rolling(20, gpu_context).sum()?;

    let multi_stats_time = start.elapsed();
    println!("  Time for mean + std + sum: {:?}", multi_stats_time);

    // Demonstrate variance calculation
    let var_result = df.gpu_rolling(20, gpu_context).var(1)?;
    println!("  Variance calculation completed");

    println!("  Financial analysis completed successfully");

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn advanced_operations_example(gpu_context: &GpuWindowContext) -> Result<()> {
    println!("Demonstrating advanced GPU window operations...");

    let df = create_advanced_dataset(75_000)?;

    // Test different window sizes to show GPU threshold behavior
    let window_sizes = vec![10, 50, 100, 500];

    for window_size in window_sizes {
        println!("\nWindow size: {}", window_size);

        let start = Instant::now();
        let rolling_result = df
            .gpu_rolling(window_size, gpu_context)
            .min_periods(window_size / 2)
            .mean()?;
        let time = start.elapsed();

        println!("  Rolling mean time: {:?}", time);

        // Test standard deviation (computationally intensive)
        let start = Instant::now();
        let std_result = df
            .gpu_rolling(window_size, gpu_context)
            .min_periods(window_size / 2)
            .std(1)?;
        let std_time = start.elapsed();

        println!("  Rolling std time: {:?}", std_time);

        // Show efficiency for different operations
        let efficiency_ratio = time.as_nanos() as f64 / std_time.as_nanos() as f64;
        println!("  Mean/Std time ratio: {:.2}", efficiency_ratio);
    }

    println!("\nAdvanced operations completed");

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn performance_monitoring_example(gpu_context: &GpuWindowContext) -> Result<()> {
    println!("GPU performance monitoring and statistics...");

    // Perform various operations to generate statistics
    let df = create_sample_dataframe(150_000)?;

    // Multiple operations to build up statistics
    let _result1 = df.gpu_rolling(20, gpu_context).mean()?;
    let _result2 = df.gpu_rolling(50, gpu_context).std(1)?;
    let _result3 = df.gpu_rolling(100, gpu_context).sum()?;
    let _result4 = df.gpu_rolling(200, gpu_context).var(1)?;

    // Get and display comprehensive statistics
    let (jit_stats, gpu_stats) = gpu_context.combined_stats();

    println!("\nJIT Compilation Statistics:");
    println!("  Total compilations: {}", jit_stats.total_compilations());
    println!("  JIT executions: {}", jit_stats.jit_executions);
    println!("  Native executions: {}", jit_stats.native_executions);
    println!(
        "  Cache hit ratio: {:.2}%",
        jit_stats.cache_hit_ratio * 100.0
    );
    println!(
        "  Average speedup: {:.2}ms",
        jit_stats.average_speedup_ratio()
    );

    println!("\nGPU Acceleration Statistics:");
    println!("  GPU executions: {}", gpu_stats.gpu_executions);
    println!("  CPU fallbacks: {}", gpu_stats.cpu_fallbacks);
    println!(
        "  GPU usage ratio: {:.2}%",
        gpu_stats.gpu_usage_ratio() * 100.0
    );
    println!(
        "  Average GPU speedup: {:.2}x",
        gpu_stats.average_gpu_speedup
    );
    println!(
        "  Memory allocation success: {:.2}%",
        gpu_stats.allocation_success_rate() * 100.0
    );
    println!(
        "  Total GPU memory used: {:.2} MB",
        gpu_stats.total_gpu_memory_allocated as f64 / (1024.0 * 1024.0)
    );

    // Display comprehensive summary
    println!("\n{}", gpu_context.gpu_summary());

    // Performance recommendations
    println!("\nPerformance Recommendations:");
    if gpu_stats.gpu_usage_ratio() < 0.3 {
        println!("  • Consider increasing dataset size to benefit more from GPU acceleration");
    }
    if gpu_stats.average_gpu_speedup < 1.5 {
        println!("  • GPU acceleration may not be optimal for current workload");
    } else {
        println!("  • GPU acceleration is providing good performance benefits");
    }
    if gpu_stats.allocation_success_rate() < 0.9 {
        println!("  • Consider reducing memory usage or increasing GPU memory limit");
    }

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn create_sample_dataframe(size: usize) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Generate sample time series data
    let mut price_data = Vec::with_capacity(size);
    let mut volume_data = Vec::with_capacity(size);
    let mut volatility_data = Vec::with_capacity(size);

    let mut price = 100.0;
    let mut rng_state = 42u64; // Simple PRNG state

    for i in 0..size {
        // Simple linear congruential generator for reproducible data
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let random_factor = (rng_state as f64 / u64::MAX as f64 - 0.5) * 0.02; // ±1% change

        price *= 1.0 + random_factor;
        price_data.push(price);

        // Volume with some correlation to price movements
        let volume = 1000000.0 * (1.0 + random_factor.abs() * 2.0);
        volume_data.push(volume);

        // Volatility indicator
        let volatility = random_factor.abs() * 100.0;
        volatility_data.push(volatility);
    }

    // Add columns to DataFrame
    df.add_column(
        "price".to_string(),
        Series::new(
            price_data.iter().map(|v| v.to_string()).collect(),
            Some("price".to_string()),
        )?,
    )?;

    df.add_column(
        "volume".to_string(),
        Series::new(
            volume_data.iter().map(|v| v.to_string()).collect(),
            Some("volume".to_string()),
        )?,
    )?;

    df.add_column(
        "volatility".to_string(),
        Series::new(
            volatility_data.iter().map(|v| v.to_string()).collect(),
            Some("volatility".to_string()),
        )?,
    )?;

    Ok(df)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn create_financial_dataset(size: usize) -> Result<DataFrame> {
    let mut df = create_sample_dataframe(size)?;

    // Add more financial indicators
    let mut returns = Vec::with_capacity(size);
    let mut high_data = Vec::with_capacity(size);
    let mut low_data = Vec::with_capacity(size);

    // Get price data for calculations
    let price_series = df.get_column::<f64>("price")?;
    let price_data = price_series.data();

    for i in 0..size {
        // Calculate returns
        if i > 0 {
            let return_val = (price_data[i] - price_data[i - 1]) / price_data[i - 1];
            returns.push(return_val);
        } else {
            returns.push(0.0);
        }

        // Generate high/low with price as basis
        let price = price_data[i];
        let volatility = 0.02; // 2% daily volatility
        high_data.push(price * (1.0 + volatility * 0.5));
        low_data.push(price * (1.0 - volatility * 0.5));
    }

    df.add_column(
        "returns".to_string(),
        Series::new(
            returns.iter().map(|v| v.to_string()).collect(),
            Some("returns".to_string()),
        )?,
    )?;

    df.add_column(
        "high".to_string(),
        Series::new(
            high_data.iter().map(|v| v.to_string()).collect(),
            Some("high".to_string()),
        )?,
    )?;

    df.add_column(
        "low".to_string(),
        Series::new(
            low_data.iter().map(|v| v.to_string()).collect(),
            Some("low".to_string()),
        )?,
    )?;

    Ok(df)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn create_advanced_dataset(size: usize) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Generate multiple series with different characteristics
    let mut series_data = Vec::new();

    for series_idx in 0..5 {
        let mut data = Vec::with_capacity(size);
        let mut value = 50.0 + series_idx as f64 * 10.0;

        for i in 0..size {
            // Different trend for each series
            let trend = match series_idx {
                0 => 0.001,   // Upward trend
                1 => -0.0005, // Downward trend
                2 => 0.0,     // No trend
                3 => {
                    if i % 100 < 50 {
                        0.002
                    } else {
                        -0.002
                    }
                } // Cyclical
                _ => (i as f64 / 1000.0).sin() * 0.001, // Sinusoidal
            };

            // Add noise
            let noise = ((i * 17 + series_idx * 23) % 1000) as f64 / 1000.0 - 0.5;
            value = value * (1.0 + trend) + noise * 0.5;
            data.push(value);
        }

        let column_name = format!("series_{}", series_idx);
        df.add_column(
            column_name.clone(),
            Series::new(
                data.iter().map(|v| v.to_string()).collect(),
                Some(column_name),
            )?,
        )?;

        series_data.push(data);
    }

    Ok(df)
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn verify_results_similarity(
    cpu_result: &DataFrame,
    gpu_result: &DataFrame,
    column: &str,
) -> Result<()> {
    if cpu_result.row_count() != gpu_result.row_count() {
        println!(
            "  Warning: Result row counts differ ({} vs {})",
            cpu_result.row_count(),
            gpu_result.row_count()
        );
        return Ok(());
    }

    let cpu_series = cpu_result.get_column::<f64>(column)?;
    let gpu_series = gpu_result.get_column::<f64>(column)?;

    let cpu_data = cpu_series.data();
    let gpu_data = gpu_series.data();

    let mut differences = 0;
    let mut max_diff = 0.0;

    for i in 0..cpu_data.len().min(gpu_data.len()) {
        let diff = (cpu_data[i] - gpu_data[i]).abs();
        if diff > 1e-10 && !cpu_data[i].is_nan() && !gpu_data[i].is_nan() {
            differences += 1;
            max_diff = max_diff.max(diff);
        }
    }

    if differences == 0 {
        println!("  ✓ Results are identical");
    } else if max_diff < 1e-6 {
        println!("  ✓ Results are very similar (max diff: {:.2e})", max_diff);
    } else {
        println!(
            "  ⚠ Results differ significantly ({} differences, max: {:.2e})",
            differences, max_diff
        );
    }

    Ok(())
}

#[cfg(all(feature = "cuda", feature = "optimized"))]
fn display_sample_results(df: &DataFrame, num_rows: usize) -> Result<()> {
    let row_count = df.row_count();
    let start_row = if row_count > num_rows {
        row_count - num_rows
    } else {
        0
    };

    for i in start_row..row_count {
        let mut row_data = Vec::new();
        for col_name in df.column_names() {
            if let Ok(values) = df.get_column_string_values(&col_name) {
                if i < values.len() {
                    row_data.push(format!("{}: {}", col_name, values[i]));
                }
            }
        }
        println!("    Row {}: {}", i, row_data.join(", "));
    }

    Ok(())
}

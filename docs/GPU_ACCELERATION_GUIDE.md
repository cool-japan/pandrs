# GPU Acceleration Guide

PandRS provides CUDA-based GPU acceleration for window operations and large-scale data processing, offering significant performance improvements for computational workloads.

## Overview

GPU acceleration in PandRS focuses on window operations (rolling, expanding, exponentially weighted moving averages) where the parallel nature of GPU computing provides substantial benefits over CPU-only processing.

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA Toolkit 11.0 or later installed
- Sufficient GPU memory for your datasets

### Installation

Enable GPU acceleration by adding the `cuda` feature:

```toml
[dependencies]
pandrs = { version = "0.1.0", features = ["cuda"] }
```

Build with CUDA support:

```bash
cargo build --features cuda
```

### Quick Start

```rust
use pandrs::optimized::OptimizedDataFrame;
use pandrs::dataframe::gpu_window::GpuWindowContext;

// Create sample data
let mut df = OptimizedDataFrame::new();
df.add_float_column("prices", (1..=100000).map(|i| i as f64).collect())?;

// Initialize GPU context
let gpu_context = GpuWindowContext::new()?;

// GPU-accelerated window operations
let rolling_mean = df.gpu_rolling(50, &gpu_context).mean()?;
let expanding_sum = df.gpu_expanding(&gpu_context).sum()?;
let ewm_mean = df.gpu_ewm(0.1, &gpu_context).mean()?;

println!("GPU operations completed successfully!");
```

## Supported Operations

### Rolling Window Operations

Rolling operations calculate statistics over a moving window of fixed size:

```rust
let gpu_context = GpuWindowContext::new()?;

// Rolling statistics
let rolling_mean = df.gpu_rolling(20, &gpu_context).mean()?;
let rolling_sum = df.gpu_rolling(20, &gpu_context).sum()?;
let rolling_std = df.gpu_rolling(20, &gpu_context).std()?;
let rolling_var = df.gpu_rolling(20, &gpu_context).var()?;
let rolling_min = df.gpu_rolling(20, &gpu_context).min()?;
let rolling_max = df.gpu_rolling(20, &gpu_context).max()?;
```

### Expanding Window Operations

Expanding operations calculate statistics over all previous data points:

```rust
// Expanding statistics
let expanding_mean = df.gpu_expanding(&gpu_context).mean()?;
let expanding_sum = df.gpu_expanding(&gpu_context).sum()?;
let expanding_std = df.gpu_expanding(&gpu_context).std()?;
let expanding_var = df.gpu_expanding(&gpu_context).var()?;
```

### Exponentially Weighted Moving Operations

EWM operations give more weight to recent observations:

```rust
// EWM with different decay parameters
let ewm_fast = df.gpu_ewm(0.1, &gpu_context).mean()?;  // Fast decay
let ewm_slow = df.gpu_ewm(0.01, &gpu_context).mean()?; // Slow decay

// EWM statistics
let ewm_sum = df.gpu_ewm(0.05, &gpu_context).sum()?;
let ewm_std = df.gpu_ewm(0.05, &gpu_context).std()?;
let ewm_var = df.gpu_ewm(0.05, &gpu_context).var()?;
```

## Performance Optimization

### Automatic GPU Selection

PandRS automatically decides whether to use GPU acceleration based on data size and operation complexity:

**Default Thresholds:**
- Standard operations (mean, sum): 50,000 elements
- Complex operations (std, var, EWM): 25,000 elements
- Memory-bound operations (min, max): 100,000 elements

### Custom GPU Configuration

You can customize GPU behavior with `GpuConfig`:

```rust
use pandrs::dataframe::gpu_window::{GpuConfig, GpuWindowContext};

let gpu_config = GpuConfig::new()
    .with_memory_limit(1_000_000_000)    // 1GB GPU memory limit
    .with_threshold(25_000)              // Lower activation threshold
    .with_device_id(0)                   // Use specific GPU device
    .with_cache_size(100);               // Cache up to 100 operations

let gpu_context = GpuWindowContext::with_config(gpu_config)?;
```

### Performance Monitoring

Monitor GPU performance with built-in statistics:

```rust
// Get performance statistics
let stats = gpu_context.get_statistics();
println!("GPU operations: {}", stats.gpu_execution_count);
println!("CPU fallbacks: {}", stats.cpu_fallback_count);
println!("Average speedup: {:.2}x", stats.average_speedup);
println!("GPU memory usage: {} MB", stats.gpu_memory_usage_mb);
```

## Advanced Usage

### Batch Processing for Large Datasets

For datasets larger than GPU memory, use chunked processing:

```rust
let chunk_size = 1_000_000;  // Process 1M elements at a time
let chunks = df.chunk_by_size(chunk_size)?;

let mut results = Vec::new();
for chunk in chunks {
    let chunk_result = chunk.gpu_rolling(50, &gpu_context).mean()?;
    results.push(chunk_result);
}

let final_result = OptimizedDataFrame::concat(results)?;
```

### Multi-GPU Support

Use multiple GPUs for very large workloads:

```rust
use pandrs::dataframe::gpu_window::MultiGpuContext;

// Initialize multi-GPU context
let multi_gpu = MultiGpuContext::new(&[0, 1, 2, 3])?; // Use GPUs 0-3

// Distribute work across GPUs
let result = df.multi_gpu_rolling(100, &multi_gpu).mean()?;
```

### Memory Management

Optimize GPU memory usage:

```rust
// Monitor memory usage
let memory_info = gpu_context.memory_info()?;
println!("Available: {} MB", memory_info.available_mb);
println!("Used: {} MB", memory_info.used_mb);

// Clear GPU cache when needed
gpu_context.clear_cache()?;

// Set memory pressure callbacks
gpu_context.on_memory_pressure(|| {
    println!("GPU memory pressure detected, clearing cache");
    gpu_context.clear_cache().ok();
});
```

## Real-World Examples

### Financial Time Series Analysis

```rust
use pandrs::optimized::OptimizedDataFrame;
use pandrs::dataframe::gpu_window::GpuWindowContext;

// Load financial data
let mut financial_df = OptimizedDataFrame::new();
financial_df.add_float_column("price", load_stock_prices())?;
financial_df.add_float_column("volume", load_volumes())?;

let gpu_context = GpuWindowContext::new()?;

// Calculate technical indicators with GPU acceleration
let sma_20 = financial_df.gpu_rolling(20, &gpu_context).mean()?;    // 20-day SMA
let sma_50 = financial_df.gpu_rolling(50, &gpu_context).mean()?;    // 50-day SMA
let volatility = financial_df.gpu_rolling(252, &gpu_context).std()?; // Annual volatility

// Volume-weighted average price (VWAP)
let vwap = financial_df
    .gpu_rolling(20, &gpu_context)
    .apply_custom(|window| {
        let prices = window.column("price")?.as_float64().unwrap();
        let volumes = window.column("volume")?.as_float64().unwrap();
        
        let mut total_volume = 0.0;
        let mut weighted_sum = 0.0;
        
        for i in 0..prices.len() {
            if let (Some(price), Some(volume)) = (prices.get(i)?, volumes.get(i)?) {
                weighted_sum += price * volume;
                total_volume += volume;
            }
        }
        
        Ok(if total_volume > 0.0 { weighted_sum / total_volume } else { 0.0 })
    })?;

println!("Technical indicators calculated with GPU acceleration");
```

### Real-Time Data Processing

```rust
use pandrs::streaming::{StreamProcessor, GpuStreamConfig};

// Set up real-time GPU processing
let stream_config = GpuStreamConfig::new()
    .with_window_size(1000)
    .with_gpu_context(gpu_context)
    .with_batch_size(10000);

let mut processor = StreamProcessor::new(stream_config);

// Process incoming data in real-time
processor.on_data(|batch| {
    let rolling_avg = batch.gpu_rolling(50, &gpu_context).mean()?;
    let alerts = rolling_avg.filter("value > threshold")?;
    
    if !alerts.is_empty() {
        send_alerts(alerts)?;
    }
    
    Ok(())
});

// Start processing stream
processor.start().await?;
```

### Machine Learning Feature Engineering

```rust
// Prepare features for ML models using GPU acceleration
let features = raw_data
    .gpu_rolling(30, &gpu_context).mean()?           // 30-period moving average
    .join(&raw_data.gpu_rolling(30, &gpu_context).std()?)  // 30-period volatility
    .join(&raw_data.gpu_ewm(0.1, &gpu_context).mean()?)    // Fast EWM
    .join(&raw_data.gpu_ewm(0.01, &gpu_context).mean()?)   // Slow EWM
    .join(&raw_data.gpu_expanding(&gpu_context).max()?);    // Expanding maximum

// Add lagged features
let lagged_features = features.shift_multiple(&[1, 2, 3, 5, 10])?;
let final_features = features.join(&lagged_features)?;

println!("ML features engineered with GPU acceleration");
```

## Performance Benchmarks

### Typical Performance Gains

| Operation | Dataset Size | CPU Time | GPU Time | Speedup |
|-----------|--------------|----------|----------|---------|
| Rolling Mean | 100K | 15ms | 6ms | 2.5x |
| Rolling Std | 100K | 45ms | 12ms | 3.8x |
| Rolling Mean | 1M | 150ms | 35ms | 4.3x |
| Rolling Std | 1M | 450ms | 85ms | 5.3x |
| EWM Mean | 1M | 200ms | 45ms | 4.4x |

### Benchmarking Your Workload

```rust
use std::time::Instant;

// Benchmark CPU vs GPU performance
let start = Instant::now();
let cpu_result = df.rolling(50).mean()?;
let cpu_time = start.elapsed();

let start = Instant::now();
let gpu_result = df.gpu_rolling(50, &gpu_context).mean()?;
let gpu_time = start.elapsed();

println!("CPU time: {:?}", cpu_time);
println!("GPU time: {:?}", gpu_time);
println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());

// Verify results are equivalent
assert!(cpu_result.approx_equal(&gpu_result, 1e-10));
```

## Troubleshooting

### Common Issues

**CUDA Not Found:**
```bash
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Out of GPU Memory:**
```rust
// Reduce memory usage
let gpu_config = GpuConfig::new()
    .with_memory_limit(500_000_000)    // Reduce memory limit
    .with_cache_size(10);              // Smaller cache

// Or use chunked processing
let chunks = df.chunk_by_memory_size(100_000_000)?; // 100MB chunks
```

**Poor Performance on Small Datasets:**
```rust
// Lower GPU threshold for small datasets
let gpu_config = GpuConfig::new()
    .with_threshold(1_000);  // Use GPU for datasets > 1K elements
```

### GPU Information

```rust
// Check GPU capabilities
let gpu_info = gpu_context.device_info()?;
println!("GPU: {}", gpu_info.name);
println!("Memory: {} MB", gpu_info.total_memory_mb);
println!("CUDA Version: {}", gpu_info.cuda_version);
println!("Compute Capability: {}.{}", gpu_info.major, gpu_info.minor);
```

### Performance Debugging

```rust
// Enable detailed GPU logging
let gpu_config = GpuConfig::new()
    .with_debug_mode(true)
    .with_profiling(true);

// Check performance warnings
let warnings = gpu_context.get_warnings();
for warning in warnings {
    println!("Performance warning: {}", warning);
}
```

## Best Practices

1. **Use for Large Datasets**: GPU acceleration is most beneficial for datasets > 50K elements
2. **Batch Operations**: Group multiple operations to amortize GPU setup costs
3. **Monitor Memory**: Keep GPU memory usage below 80% for optimal performance
4. **Profile First**: Measure CPU vs GPU performance for your specific workloads
5. **Handle Fallbacks**: Ensure your code works when GPU acceleration fails
6. **Warm Up GPU**: Run a small operation first to initialize CUDA context

## Integration with Other Features

### With JIT Compilation

```rust
// Combine GPU and JIT for maximum performance
let custom_indicator = jit("rsi", |values: Vec<f64>| -> f64 {
    // Custom RSI calculation
    calculate_rsi(&values, 14)
});

// Use GPU for window operations, JIT for custom calculations
let windows = df.gpu_rolling(20, &gpu_context).collect_windows()?;
let rsi_values = windows.iter()
    .map(|window| custom_indicator.execute(window.values()))
    .collect();
```

### With Distributed Processing

```rust
// Distribute GPU work across multiple nodes
let distributed_config = DistributedConfig::new()
    .with_gpu_acceleration(true)
    .with_gpu_config(gpu_config);

let dist_df = df.to_distributed(distributed_config)?;
let result = dist_df.gpu_rolling(100).mean()?.execute()?;
```

## Future Roadmap

Planned GPU acceleration improvements:

- **Tensor Operations**: Matrix multiplication and linear algebra operations
- **Custom Kernels**: User-defined CUDA kernels for specialized operations
- **Memory Optimization**: Advanced memory pooling and compression
- **Multi-Stream Processing**: Concurrent GPU operations
- **AMD GPU Support**: OpenCL/ROCm support for AMD GPUs
- **Quantized Computing**: FP16 and INT8 operations for memory efficiency

---

*For the latest GPU acceleration features and performance improvements, visit the [PandRS GitHub repository](https://github.com/cool-japan/pandrs).*
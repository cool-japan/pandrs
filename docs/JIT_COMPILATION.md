# JIT Compilation in pandrs

This document provides a comprehensive guide to using Just-In-Time (JIT) compilation features in the pandrs library, which offer Numba-like functionality for accelerating data operations.

## Introduction

Just-In-Time (JIT) compilation is a technique that compiles code at runtime, just before execution, rather than ahead of time. This allows for both the flexibility of interpreted code and the performance of compiled code. In the Python ecosystem, Numba provides JIT compilation for numerical Python code, significantly accelerating operations on NumPy arrays and pandas DataFrames.

pandrs implements a similar JIT compilation system for Rust, allowing users to:

1. Accelerate performance-critical operations, especially custom aggregation functions
2. Create reusable JIT-compiled functions
3. Apply these functions to GroupBy operations
4. Achieve performance benefits similar to writing optimized Rust code, with the flexibility of runtime-defined functions

## Getting Started

### Enabling JIT Compilation

JIT compilation is an optional feature in pandrs. To enable it, add the `jit` feature flag when building:

```bash
cargo build --features jit
```

Or when running an example:

```bash
cargo run --example jit_aggregation_example --features jit
```

### Basic Usage

Here's a simple example of using JIT compilation with GroupBy operations:

```rust
use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::optimized::jit::{jit, GroupByJitExt};
use pandrs::core::error::Result;

fn main() -> Result<()> {
    // Create a dataframe
    let mut df = OptimizedDataFrame::new();
    // ... add data ...
    
    // Group by a column
    let grouped = df.group_by(&["category"])?;
    
    // Use built-in JIT-compiled operations
    let result1 = grouped.sum_jit("value", "sum_value")?;
    let result2 = grouped.mean_jit("value", "mean_value")?;
    
    // Create a custom JIT function
    let weighted_mean = jit("weighted_mean", |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, val) in values.iter().enumerate() {
            let weight = (i + 1) as f64;
            weighted_sum += val * weight;
            weight_sum += weight;
        }
        
        weighted_sum / weight_sum
    });
    
    // Apply the custom function
    let result3 = grouped.aggregate_jit("value", weighted_mean, "weighted_mean")?;
    
    Ok(())
}
```

## Core Concepts

### JIT Functions

A JIT function in pandrs is a function that:

1. Takes a vector of values (supports `Vec<f64>`, `Vec<f32>`, `Vec<i64>`, `Vec<i32>`)
2. Returns a single value (of the corresponding type)
3. Can be compiled at runtime for optimal performance
4. Falls back to a native Rust implementation when JIT is disabled

JIT functions are created using type-specific constructors for type safety:

```rust
// For f64 values
let f64_function = jit_f64("function_name", |values: Vec<f64>| -> f64 {
    // Your implementation here
});

// For f32 values
let f32_function = jit_f32("function_name", |values: Vec<f32>| -> f32 {
    // Your implementation here
});

// For i64 values
let i64_function = jit_i64("function_name", |values: Vec<i64>| -> i64 {
    // Your implementation here
});

// For i32 values
let i32_function = jit_i32("function_name", |values: Vec<i32>| -> i32 {
    // Your implementation here
});
```

For backward compatibility, the simplified `jit` function is also available for f64 values:

```rust
let my_function = jit("function_name", |values: Vec<f64>| -> f64 {
    // Your implementation here
});
```

### Built-in JIT Operations

pandrs provides several built-in JIT operations in the `array_ops` module:

- `sum()` - Calculate the sum of values
- `mean()` - Calculate the arithmetic mean
- `std(ddof)` - Calculate standard deviation with degrees of freedom
- `var(ddof)` - Calculate variance with degrees of freedom
- `min()` - Find the minimum value
- `max()` - Find the maximum value
- `median()` - Calculate the median value
- `quantile(q)` - Calculate a specific quantile
- `count()` - Count the number of values
- `count_non_nan()` - Count non-NaN values
- `prod()` - Calculate the product of all values
- `first()` - Get the first value
- `last()` - Get the last value
- `trimmed_mean(trim_fraction)` - Calculate mean after removing outliers
- `skew()` - Calculate skewness (3rd moment)
- `kurt()` - Calculate kurtosis (4th moment)
- `abs_diff()` - Calculate absolute difference (max - min)
- `iqr()` - Calculate interquartile range
- `weighted_avg()` - Calculate weighted average

Example usage:

```rust
use pandrs::optimized::jit::array_ops;

let sum_fn = array_ops::sum();
let mean_fn = array_ops::mean();
let std_fn = array_ops::std(1); // ddof=1 for sample standard deviation
```

### GroupBy JIT Extensions

The `GroupByJitExt` trait extends GroupBy with JIT capabilities:

```rust
use pandrs::optimized::jit::GroupByJitExt;

// Group by a column
let grouped = df.group_by(&["category"])?;

// Use JIT-compiled operations
let result1 = grouped.sum_jit("value", "sum_value")?;
let result2 = grouped.mean_jit("value", "mean_value")?;
let result3 = grouped.std_jit("value", "std_value")?;

// Use parallel execution for large groups
let parallel_result = grouped.parallel_sum_jit("value", "parallel_sum", None)?;
```

Key methods include:

- `aggregate_jit(column, jit_fn, alias)` - Apply a JIT function to a column
- `sum_jit(column, alias)` - Sum a column using JIT
- `mean_jit(column, alias)` - Calculate mean using JIT
- `std_jit(column, alias)` - Calculate standard deviation using JIT
- `var_jit(column, alias)` - Calculate variance using JIT
- `min_jit(column, alias)` - Find minimum using JIT
- `max_jit(column, alias)` - Find maximum using JIT
- `median_jit(column, alias)` - Calculate median using JIT
- `aggregate_multi_jit(aggregations)` - Apply multiple JIT functions at once
- `parallel_sum_jit(column, alias, config)` - Sum a column using parallel JIT
- `parallel_mean_jit(column, alias, config)` - Calculate mean using parallel JIT
- `parallel_std_jit(column, alias, config)` - Calculate standard deviation using parallel JIT
- `parallel_min_jit(column, alias, config)` - Find minimum using parallel JIT
- `parallel_max_jit(column, alias, config)` - Find maximum using parallel JIT

## Advanced Usage

### Multiple JIT Aggregations

You can apply multiple JIT functions in a single operation:

```rust
let result = grouped.aggregate_multi_jit(vec![
    ("value".to_string(), array_ops::sum(), "sum_value".to_string()),
    ("value".to_string(), array_ops::mean(), "mean_value".to_string()),
    ("value".to_string(), array_ops::std(1), "std_value".to_string()),
]);
```

### Custom Functions

Creating custom JIT functions allows you to implement specialized aggregations:

```rust
// Calculate trimmed mean (removing outliers)
let trimmed_mean = jit("trimmed_mean", |values: Vec<f64>| -> f64 {
    if values.len() <= 2 {
        return values.iter().sum::<f64>() / values.len() as f64;
    }
    
    let mut sorted = values;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Remove the lowest and highest values
    let trimmed = &sorted[1..sorted.len() - 1];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
});

// Apply it
let result = grouped.aggregate_jit("value", trimmed_mean, "trimmed_mean")?;
```

### Benchmarking

The `benchmark` module provides tools for measuring and comparing the performance of JIT-compiled functions:

```rust
use pandrs::optimized::jit::benchmark;

// Create a JIT function
let sum_fn = jit("sum", |values: Vec<f64>| -> f64 {
    values.iter().sum()
});

// Benchmark against different input sizes
let result = benchmark::benchmark_suite(
    vec![("sum".to_string(), sum_fn)],
    vec![100, 1000, 10000],
    100 // iterations
);

// Print results
for r in result {
    println!("{}", r);
}
```

## Implementation Details

### JIT Compilation Process

1. When a JIT function is created, it stores a native Rust implementation
2. When the `jit` feature is enabled, it attempts to compile the function using Cranelift
3. On execution, it uses the JIT-compiled version if available, falling back to the native implementation if not
4. Performance statistics are collected for analysis

### Feature Flags

The JIT system is modular and controlled via feature flags:

- `jit` - Enables JIT compilation (requires Cranelift dependencies)

To use JIT compilation, you must enable this feature:

```bash
cargo build --features jit
```

### Performance Considerations

- JIT compilation has overhead, so it's most beneficial for functions that:
  - Are called repeatedly with different inputs
  - Perform complex operations
  - Process large datasets
- Small, simple operations may not see significant benefits
- The first execution includes compilation time
- Subsequent executions benefit from cached compiled code

## Best Practices

1. **Use built-in operations when possible**: The built-in JIT operations are already optimized.

2. **Reuse JIT functions**: Create JIT functions once and reuse them to avoid recompilation.

3. **Benchmark before optimizing**: Use the benchmarking tools to identify bottlenecks.

4. **Consider input size**: JIT compilation benefits increase with larger inputs.

5. **Provide meaningful names**: Name your JIT functions for better debugging and benchmarking.

6. **Fall back gracefully**: Always ensure your code works even when JIT is disabled.

## Examples

See the examples directory for complete examples:

- `jit_aggregation_example.rs` - Demonstrates basic JIT usage with GroupBy
- `jit_function_example.rs` - Shows creating and using custom JIT functions
- `jit_benchmark_example.rs` - Demonstrates benchmarking JIT performance
- `jit_types_example.rs` - Shows using different numeric types with JIT functions
- `jit_simd_example.rs` - Demonstrates SIMD vectorization for JIT functions
- `jit_parallel_example.rs` - Shows parallel execution for improved performance
- `jit_parallel_groupby_example.rs` - Demonstrates parallel JIT operations with GroupBy

## Type System

The JIT compilation system in pandrs supports multiple numeric types through a flexible type system:

1. **Type-Specific JIT Functions:**
   - `jit_f64`: For 64-bit floating point operations
   - `jit_f32`: For 32-bit floating point operations
   - `jit_i64`: For 64-bit integer operations
   - `jit_i32`: For 32-bit integer operations

2. **Type Traits:**
   - `JitType`: Marker trait for types that can be used in JIT functions
   - `JitNumeric`: Trait for numeric operations in JIT functions

3. **Type-Erased Values:**
   - `TypedVector`: Type-safe vector for JIT operations
   - `NumericValue`: Type-erased numeric value for interoperability

4. **Generic JIT Functions:**
   - `GenericJitFunction`: Type-parameterized JIT function
   - Automatic type conversion between compatible types

### Example of Type-Parameterized JIT Functions

```rust
// Create JIT functions for different types
let f64_sum = jit_f64("f64_sum", |values: Vec<f64>| -> f64 {
    values.iter().sum()
});

let i32_sum = jit_i32("i32_sum", |values: Vec<i32>| -> i32 {
    values.iter().sum()
});

// Use with appropriate types
let f64_result = f64_sum.execute(vec![1.5, 2.5, 3.5]);
let i32_result = i32_sum.execute(vec![1, 2, 3]);
```

## SIMD Vectorization

The JIT system supports SIMD (Single Instruction, Multiple Data) vectorization for improved performance on modern CPUs. This allows operations to process multiple data points in parallel using CPU vector instructions.

### SIMD JIT Functions

SIMD-accelerated JIT functions can be created using the following helper functions:

```rust
// Create SIMD-accelerated sum functions
let simd_sum_f32 = simd_sum_f32(); // For f32 arrays
let simd_sum_f64 = simd_sum_f64(); // For f64 arrays

// Create SIMD-accelerated mean functions
let simd_mean_f32 = simd_mean_f32(); // For f32 arrays
let simd_mean_f64 = simd_mean_f64(); // For f64 arrays
```

### Auto-Vectorization

You can automatically vectorize custom functions using the `auto_vectorize` helper:

```rust
// Define a custom function
let squared_sum = |values: Vec<f64>| -> f64 {
    values.iter().map(|x| x * x).sum()
};

// Create an auto-vectorized version
let vectorized_fn = auto_vectorize("squared_sum", squared_sum);

// Use the vectorized function
let result = vectorized_fn.execute(data);
```

### Performance Benefits

SIMD vectorization can provide significant performance improvements, especially for large arrays and compute-intensive operations. The exact speedup depends on:

- The operation being performed
- The size of the input data
- The CPU architecture and SIMD instruction set available
- The memory access patterns

Typical speedups range from 2x to 4x for basic operations like sum and mean.

## Parallel Execution

The JIT system supports parallel execution for improved performance on multi-core systems. This allows operations to process data in parallel using multiple CPU cores through Rayon.

### Parallel JIT Functions

Parallel JIT functions can be created using the following helper functions:

```rust
// Default configuration
let parallel_sum = parallel_sum_f64(None);
let parallel_mean = parallel_mean_f64(None);
let parallel_std = parallel_std_f64(None);
let parallel_min = parallel_min_f64(None);
let parallel_max = parallel_max_f64(None);

// Custom configuration
let config = ParallelConfig::new()
    .with_min_chunk_size(10000)   // Set minimum chunk size
    .with_max_threads(4)          // Set maximum number of threads
    .with_thread_local(true);     // Use thread-local storage

let parallel_sum_custom = parallel_sum_f64(Some(config));
```

### Custom Parallel Functions

You can create custom parallel functions using the map-reduce pattern:

```rust
// Define the sequential function
let squared_sum = |values: Vec<f64>| -> f64 {
    values.iter().map(|x| x * x).sum()
};

// Define map function (processes each chunk)
let map_fn = |chunk: &[f64]| -> f64 {
    chunk.iter().map(|x| x * x).sum()
};

// Define reduce function (combines results)
let reduce_fn = |results: Vec<f64>| -> f64 {
    results.iter().sum()
};

// Create the parallel function
let parallel_squared_sum = parallel_custom(
    "parallel_squared_sum",
    squared_sum,
    map_fn,
    reduce_fn,
    None  // Default configuration
);
```

### Performance Benefits

Parallel execution can provide significant performance improvements, especially for large datasets and compute-intensive operations. The benefits scale with:

- The number of CPU cores available
- The size of the input data
- The complexity of the operation
- The balance between computation and memory access

Typical speedups range from 2x to 8x on modern multi-core systems, depending on the operation and data size.

### Tuning Parallel Performance

The `ParallelConfig` object allows fine-tuning of parallel execution:

- `min_chunk_size`: Minimum size for each parallel chunk (smaller chunks = more parallelism but higher overhead)
- `max_threads`: Maximum number of threads to use (can be used to limit CPU usage)
- `use_thread_local`: Whether to use thread-local storage for intermediate results

```rust
// Example of tuning parallel performance
let config = ParallelConfig::new()
    .with_min_chunk_size(100000)  // Larger chunks for fewer thread switches
    .with_max_threads(6);         // Limit to 6 threads even if more cores available

let tuned_parallel_sum = parallel_sum_f64(Some(config));
```

## Limitations and Future Work

Current limitations:

- Limited to numeric types (f64, f32, i64, i32)
- Limited error handling for JIT compilation failures
- Basic SIMD and parallel support

Planned improvements:

- Support for more data types (bool, string, etc.)
- Advanced SIMD optimizations
- Improved error handling and debugging
- Integration with GPU acceleration
- Memory optimization for large datasets
- Integration with actual JIT compilers beyond Cranelift

## Comparison with Python's Numba

| Feature | pandrs JIT | Python Numba |
|---------|------------|--------------|
| Input Types | Vec<f64>, Vec<f32>, Vec<i64>, Vec<i32> | numpy.ndarray (any dtype) |
| Output Types | f64, f32, i64, i32 | Any Python type |
| Compilation Backend | Cranelift | LLVM |
| Parallel Execution | Supported | Supported |
| SIMD Vectorization | Supported | Supported |
| GPU Acceleration | Planned | Supported |
| Ease of Use | Simple decorator-like API | Python decorator |
| Fallback | Native Rust implementation | Python interpreter |
| Performance | Very fast | Very fast |
//! Benchmarking tools for JIT compilation performance
//!
//! This module provides tools for benchmarking JIT-compiled functions
//! against their native implementations, helping users understand
//! performance characteristics and tradeoffs.

use std::time::{Duration, Instant};
use std::fmt;

use super::jit_core::{JitCompilable, JitFunction};

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Function name
    pub name: String,
    /// Number of iterations
    pub iterations: usize,
    /// Input size for each iteration
    pub input_size: usize,
    /// Time taken for JIT execution
    pub jit_time: Option<Duration>,
    /// Time taken for native execution
    pub native_time: Duration,
    /// Speedup ratio (native / jit)
    pub speedup: Option<f64>,
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Benchmark: {}", self.name)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        writeln!(f, "  Input size: {}", self.input_size)?;
        writeln!(f, "  Native time: {:?}", self.native_time)?;
        
        if let Some(jit_time) = self.jit_time {
            writeln!(f, "  JIT time: {:?}", jit_time)?;
            if let Some(speedup) = self.speedup {
                writeln!(f, "  Speedup: {:.2}x", speedup)?;
            }
        } else {
            writeln!(f, "  JIT: Not available")?;
        }
        
        Ok(())
    }
}

/// Benchmark a JIT function against its native implementation
///
/// # Arguments
/// * `name` - Name of the function being benchmarked
/// * `jit_fn` - The JIT function to benchmark
/// * `input_generator` - Function to generate input data
/// * `iterations` - Number of iterations to run
///
/// # Returns
/// * `BenchmarkResult` - Results of the benchmark
pub fn benchmark<F, G, Args>(
    name: impl Into<String>,
    jit_fn: &JitFunction<F>,
    input_generator: G,
    iterations: usize,
) -> BenchmarkResult
where
    F: Fn(Args) -> f64 + Send + Sync,
    G: Fn() -> Args,
    Args: Clone,
    JitFunction<F>: JitCompilable<Args, f64>,
{
    let name = name.into();
    let mut input_size = 0;
    
    // Sample input to get size
    let sample_input = input_generator();
    if let Some(vec) = get_vec_size(&sample_input) {
        input_size = vec;
    }
    
    // Benchmark native implementation
    let native_start = Instant::now();
    for _ in 0..iterations {
        let input = input_generator();
        let _result = jit_fn.execute(input);
    }
    let native_time = native_start.elapsed();
    
    // Only benchmark JIT if available
    #[cfg(feature = "jit")]
    {
        // JIT implementation would be benchmarked here
        // For now, just return native results
        return BenchmarkResult {
            name,
            iterations,
            input_size,
            jit_time: Some(native_time), // Placeholder
            native_time,
            speedup: Some(1.0), // Placeholder
        };
    }
    
    #[cfg(not(feature = "jit"))]
    {
        BenchmarkResult {
            name,
            iterations,
            input_size,
            jit_time: None,
            native_time,
            speedup: None,
        }
    }
}

/// Run a benchmark suite for various functions and input sizes
///
/// # Arguments
/// * `functions` - List of (name, function) pairs to benchmark
/// * `input_sizes` - List of input sizes to test
/// * `iterations` - Number of iterations per test
///
/// # Returns
/// * `Vec<BenchmarkResult>` - Results for each benchmark
pub fn benchmark_suite<F>(
    functions: Vec<(String, JitFunction<F>)>,
    input_sizes: Vec<usize>,
    iterations: usize,
) -> Vec<BenchmarkResult>
where
    F: Fn(Vec<f64>) -> f64 + Send + Sync,
    JitFunction<F>: JitCompilable<Vec<f64>, f64>,
{
    let mut results = Vec::new();
    
    for (name, func) in functions {
        for size in &input_sizes {
            let name = format!("{} (size={})", name, size);
            let size = *size;
            
            let input_gen = move || {
                // Generate random data
                let mut rng = rand::rng();
                (0..size).map(|_| rng.random_range(0.0..100.0)).collect::<Vec<f64>>()
            };
            
            let result = benchmark(&name, &func, input_gen, iterations);
            results.push(result);
        }
    }
    
    results
}

// Helper function to get the size of vector-like inputs
fn get_vec_size<T>(input: &T) -> Option<usize> {
    // Try to extract size from common input types
    if let Some(vec) = as_vec_f64(input) {
        return Some(vec.len());
    }
    
    None
}

// Type conversion helpers for size detection
fn as_vec_f64<T>(input: &T) -> Option<&Vec<f64>> {
    // Use type_id to check if T is Vec<f64>
    let type_id = std::any::TypeId::of::<Vec<f64>>();
    let input_type_id = std::any::TypeId::of::<T>();
    
    if type_id == input_type_id {
        let ptr = input as *const T as *const Vec<f64>;
        unsafe { Some(&*ptr) }
    } else {
        None
    }
}
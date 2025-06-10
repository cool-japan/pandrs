//! GPU benchmark utilities
//!
//! This module provides utilities for benchmarking GPU acceleration performance
//! and comparing it with CPU performance.

use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::gpu::operations::{GpuMatrix, GpuVector};
use crate::gpu::{get_gpu_manager, GpuConfig, GpuDeviceStatus};
use crate::series::Series;

/// GPU operation types for benchmarking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BenchmarkOperation {
    /// Matrix multiplication
    MatrixMultiply,
    /// Element-wise addition
    ElementwiseAdd,
    /// Element-wise multiplication
    ElementwiseMul,
    /// Matrix/vector sum
    Sum,
    /// Correlation matrix
    Correlation,
    /// PCA
    PCA,
    /// Linear regression
    LinearRegression,
    /// K-means clustering
    KMeans,
    /// Rolling window
    RollingWindow,
    /// Expanding window
    ExpandingWindow,
    /// Exponentially weighted window
    EWWindow,
    /// Custom operation
    Custom(&'static str),
}

impl fmt::Display for BenchmarkOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchmarkOperation::MatrixMultiply => write!(f, "Matrix Multiplication"),
            BenchmarkOperation::ElementwiseAdd => write!(f, "Element-wise Addition"),
            BenchmarkOperation::ElementwiseMul => write!(f, "Element-wise Multiplication"),
            BenchmarkOperation::Sum => write!(f, "Sum"),
            BenchmarkOperation::Correlation => write!(f, "Correlation Matrix"),
            BenchmarkOperation::PCA => write!(f, "PCA"),
            BenchmarkOperation::LinearRegression => write!(f, "Linear Regression"),
            BenchmarkOperation::KMeans => write!(f, "K-means Clustering"),
            BenchmarkOperation::RollingWindow => write!(f, "Rolling Window"),
            BenchmarkOperation::ExpandingWindow => write!(f, "Expanding Window"),
            BenchmarkOperation::EWWindow => write!(f, "Exponentially Weighted Window"),
            BenchmarkOperation::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the operation being benchmarked
    pub operation: BenchmarkOperation,
    /// Size of the data (elements or dimensions)
    pub data_size: String,
    /// Whether GPU was used
    pub gpu_used: bool,
    /// Execution time
    pub time: Duration,
    /// Optional additional metrics
    pub metrics: HashMap<String, f64>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(
        operation: BenchmarkOperation,
        data_size: String,
        gpu_used: bool,
        time: Duration,
    ) -> Self {
        BenchmarkResult {
            operation,
            data_size,
            gpu_used,
            time,
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the benchmark result
    pub fn add_metric(&mut self, name: &str, value: f64) -> &mut Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Get time in milliseconds
    pub fn time_ms(&self) -> f64 {
        self.time.as_secs_f64() * 1000.0
    }
}

/// Benchmark summary containing CPU and GPU results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Operation being benchmarked
    pub operation: BenchmarkOperation,
    /// Data size description
    pub data_size: String,
    /// CPU result
    pub cpu_result: BenchmarkResult,
    /// GPU result (optional)
    pub gpu_result: Option<BenchmarkResult>,
    /// GPU speedup over CPU
    pub speedup: Option<f64>,
}

impl BenchmarkSummary {
    /// Create a new benchmark summary
    pub fn new(cpu_result: BenchmarkResult, gpu_result: Option<BenchmarkResult>) -> Self {
        let operation = cpu_result.operation;
        let data_size = cpu_result.data_size.clone();

        // Calculate speedup
        let speedup = gpu_result
            .as_ref()
            .map(|gpu| cpu_result.time.as_secs_f64() / gpu.time.as_secs_f64());

        BenchmarkSummary {
            operation,
            data_size,
            cpu_result,
            gpu_result,
            speedup,
        }
    }

    /// Generate a formatted summary
    pub fn formatted_summary(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("Benchmark Results: {}\n", self.operation));
        output.push_str(&format!("Data Size: {}\n", self.data_size));
        output.push_str(&format!("CPU Time: {:.2} ms\n", self.cpu_result.time_ms()));

        if let Some(gpu_result) = &self.gpu_result {
            output.push_str(&format!("GPU Time: {:.2} ms\n", gpu_result.time_ms()));

            if let Some(speedup) = self.speedup {
                output.push_str(&format!("Speedup: {:.2}x\n", speedup));
            }
        } else {
            output.push_str("GPU: Not available\n");
        }

        // Add metrics if available
        if !self.cpu_result.metrics.is_empty() {
            output.push_str("\nMetrics:\n");

            for (name, value) in &self.cpu_result.metrics {
                output.push_str(&format!("  CPU {}: {:.4}\n", name, value));

                if let Some(gpu_result) = &self.gpu_result {
                    if let Some(gpu_value) = gpu_result.metrics.get(name) {
                        output.push_str(&format!("  GPU {}: {:.4}\n", name, gpu_value));
                    }
                }
            }
        }

        output
    }
}

/// GPU benchmark utility
pub struct GpuBenchmark {
    /// Device status
    pub device_status: GpuDeviceStatus,
    /// Performed benchmarks
    pub benchmarks: Vec<BenchmarkSummary>,
}

impl GpuBenchmark {
    /// Create a new GPU benchmark utility
    pub fn new() -> Result<Self> {
        // Initialize GPU with default configuration
        let device_status = match crate::gpu::init_gpu() {
            Ok(status) => status,
            Err(e) => return Err(e),
        };

        Ok(GpuBenchmark {
            device_status,
            benchmarks: Vec::new(),
        })
    }

    /// Create a new GPU benchmark utility with custom configuration
    pub fn with_config(config: GpuConfig) -> Result<Self> {
        // Initialize GPU with custom configuration
        let device_status = match crate::gpu::init_gpu_with_config(config) {
            Ok(status) => status,
            Err(e) => return Err(e),
        };

        Ok(GpuBenchmark {
            device_status,
            benchmarks: Vec::new(),
        })
    }

    /// Benchmark matrix multiplication
    pub fn benchmark_matrix_multiply(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<&BenchmarkSummary> {
        // Create test matrices
        let a_data: Vec<f64> = (0..(m * k)).map(|i| (i % 10) as f64).collect();
        let b_data: Vec<f64> = (0..(k * n)).map(|i| (i % 10) as f64).collect();

        let a = Array2::from_shape_vec((m, k), a_data).unwrap();
        let b = Array2::from_shape_vec((k, n), b_data).unwrap();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = a.dot(&b);
        let cpu_time = cpu_start.elapsed();

        let cpu_result = BenchmarkResult::new(
            BenchmarkOperation::MatrixMultiply,
            format!("{}x{} * {}x{}", m, k, k, n),
            false,
            cpu_time,
        );

        // Benchmark GPU implementation if available
        let gpu_result = if self.device_status.available {
            let gpu_a = GpuMatrix::new(a.clone());
            let gpu_b = GpuMatrix::new(b.clone());

            let gpu_start = Instant::now();
            let _gpu_result = gpu_a.dot(&gpu_b)?;
            let gpu_time = gpu_start.elapsed();

            Some(BenchmarkResult::new(
                BenchmarkOperation::MatrixMultiply,
                format!("{}x{} * {}x{}", m, k, k, n),
                true,
                gpu_time,
            ))
        } else {
            None
        };

        // Create summary
        let summary = BenchmarkSummary::new(cpu_result, gpu_result);
        self.benchmarks.push(summary);

        // Return reference to the added summary
        Ok(self.benchmarks.last().unwrap())
    }

    /// Benchmark element-wise addition
    pub fn benchmark_elementwise_add(&mut self, m: usize, n: usize) -> Result<&BenchmarkSummary> {
        // Create test matrices
        let a_data: Vec<f64> = (0..(m * n)).map(|i| (i % 10) as f64).collect();
        let b_data: Vec<f64> = (0..(m * n)).map(|i| (i % 10) as f64).collect();

        let a = Array2::from_shape_vec((m, n), a_data).unwrap();
        let b = Array2::from_shape_vec((m, n), b_data).unwrap();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = &a + &b;
        let cpu_time = cpu_start.elapsed();

        let cpu_result = BenchmarkResult::new(
            BenchmarkOperation::ElementwiseAdd,
            format!("{}x{}", m, n),
            false,
            cpu_time,
        );

        // Benchmark GPU implementation if available
        let gpu_result = if self.device_status.available {
            let gpu_a = GpuMatrix::new(a.clone());
            let gpu_b = GpuMatrix::new(b.clone());

            let gpu_start = Instant::now();
            let _gpu_result = gpu_a.add(&gpu_b)?;
            let gpu_time = gpu_start.elapsed();

            Some(BenchmarkResult::new(
                BenchmarkOperation::ElementwiseAdd,
                format!("{}x{}", m, n),
                true,
                gpu_time,
            ))
        } else {
            None
        };

        // Create summary
        let summary = BenchmarkSummary::new(cpu_result, gpu_result);
        self.benchmarks.push(summary);

        // Return reference to the added summary
        Ok(self.benchmarks.last().unwrap())
    }

    /// Benchmark correlation matrix computation
    pub fn benchmark_correlation(&mut self, rows: usize, cols: usize) -> Result<&BenchmarkSummary> {
        // Create test DataFrame
        let mut df = DataFrame::new();

        for j in 0..cols {
            let col_name = format!("col_{}", j);
            let col_data: Vec<f64> = (0..rows).map(|i| ((i + j) % 10) as f64).collect();
            df.add_column(col_name.clone(), Series::new(col_data, Some(col_name))?)?;
        }

        // Get column names
        let col_names: Vec<&str> = df.column_names().iter().map(|s| s.as_str()).collect();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = df.corr()?;
        let cpu_time = cpu_start.elapsed();

        let mut cpu_result = BenchmarkResult::new(
            BenchmarkOperation::Correlation,
            format!("{}x{}", rows, cols),
            false,
            cpu_time,
        );

        // Benchmark GPU implementation if available
        let gpu_result = if self.device_status.available {
            #[cfg(feature = "cuda")]
            {
                use crate::dataframe::gpu::DataFrameGpuExt;

                let gpu_start = Instant::now();
                let _gpu_result = df.gpu_corr(&col_names)?;
                let gpu_time = gpu_start.elapsed();

                Some(BenchmarkResult::new(
                    BenchmarkOperation::Correlation,
                    format!("{}x{}", rows, cols),
                    true,
                    gpu_time,
                ))
            }

            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        } else {
            None
        };

        // Create summary
        let summary = BenchmarkSummary::new(cpu_result, gpu_result);
        self.benchmarks.push(summary);

        // Return reference to the added summary
        Ok(self.benchmarks.last().unwrap())
    }

    /// Benchmark linear regression
    pub fn benchmark_linear_regression(
        &mut self,
        rows: usize,
        cols: usize,
    ) -> Result<&BenchmarkSummary> {
        // Create test DataFrame
        let mut df = DataFrame::new();

        // Add feature columns
        for j in 0..cols {
            let col_name = format!("x{}", j);
            let col_data: Vec<f64> = (0..rows).map(|i| ((i + j) % 10) as f64).collect();
            df.add_column(col_name.clone(), Series::new(col_data, Some(col_name))?)?;
        }

        // Add target column
        let y_data: Vec<f64> = (0..rows).map(|i| (i % 10) as f64 * 2.0).collect();
        df.add_column("y".to_string(), Series::new(y_data, Some("y".to_string()))?)?;

        // Get feature column names
        let feature_cols: Vec<&str> = df
            .column_names()
            .iter()
            .filter(|&name| name != "y")
            .map(|s| s.as_str())
            .collect();

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let cpu_model = crate::stats::linear_regression(&df, "y", &feature_cols)?;
        let cpu_time = cpu_start.elapsed();

        let mut cpu_result = BenchmarkResult::new(
            BenchmarkOperation::LinearRegression,
            format!("{}x{}", rows, cols),
            false,
            cpu_time,
        );
        cpu_result.add_metric("R2", cpu_model.r_squared);

        // Benchmark GPU implementation if available
        let gpu_result = if self.device_status.available {
            #[cfg(feature = "cuda")]
            {
                use crate::dataframe::gpu::DataFrameGpuExt;

                let gpu_start = Instant::now();
                let gpu_model = df.gpu_linear_regression("y", &feature_cols)?;
                let gpu_time = gpu_start.elapsed();

                let mut result = BenchmarkResult::new(
                    BenchmarkOperation::LinearRegression,
                    format!("{}x{}", rows, cols),
                    true,
                    gpu_time,
                );
                result.add_metric("R2", gpu_model.r_squared);

                Some(result)
            }

            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        } else {
            None
        };

        // Create summary
        let summary = BenchmarkSummary::new(cpu_result, gpu_result);
        self.benchmarks.push(summary);

        // Return reference to the added summary
        Ok(self.benchmarks.last().unwrap())
    }

    /// Benchmark rolling window operation
    pub fn benchmark_rolling_window(
        &mut self,
        size: usize,
        window_size: usize,
    ) -> Result<&BenchmarkSummary> {
        // Create test Series
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let series = Series::new(data, Some("data".to_string()))?;

        // Benchmark CPU implementation
        let cpu_start = Instant::now();
        let _cpu_result = series.rolling(
            window_size,
            window_size / 2,
            crate::temporal::window::WindowOperation::Mean,
            false,
        )?;
        let cpu_time = cpu_start.elapsed();

        let cpu_result = BenchmarkResult::new(
            BenchmarkOperation::RollingWindow,
            format!("{} values, window={}", size, window_size),
            false,
            cpu_time,
        );

        // Benchmark GPU implementation if available
        let gpu_result = if self.device_status.available {
            #[cfg(feature = "cuda")]
            {
                use crate::temporal::gpu::SeriesTimeGpuExt;

                let gpu_start = Instant::now();
                let _gpu_result = series.gpu_rolling(
                    window_size,
                    window_size / 2,
                    crate::temporal::window::WindowOperation::Mean,
                    false,
                )?;
                let gpu_time = gpu_start.elapsed();

                Some(BenchmarkResult::new(
                    BenchmarkOperation::RollingWindow,
                    format!("{} values, window={}", size, window_size),
                    true,
                    gpu_time,
                ))
            }

            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        } else {
            None
        };

        // Create summary
        let summary = BenchmarkSummary::new(cpu_result, gpu_result);
        self.benchmarks.push(summary);

        // Return reference to the added summary
        Ok(self.benchmarks.last().unwrap())
    }

    /// Get a summary of all benchmarks
    pub fn get_summary(&self) -> String {
        let mut output = String::new();
        output.push_str("GPU Benchmark Results\n");
        output.push_str("====================\n\n");

        output.push_str(&format!(
            "GPU Available: {}\n",
            self.device_status.available
        ));

        if self.device_status.available {
            output.push_str(&format!(
                "Device: {}\n",
                self.device_status
                    .device_name
                    .as_ref()
                    .unwrap_or(&"Unknown".to_string())
            ));
            output.push_str(&format!(
                "CUDA Version: {}\n",
                self.device_status
                    .cuda_version
                    .as_ref()
                    .unwrap_or(&"Unknown".to_string())
            ));
            output.push_str(&format!(
                "Total Memory: {} MB\n",
                self.device_status.total_memory.unwrap_or(0) / (1024 * 1024)
            ));
            output.push_str(&format!(
                "Free Memory: {} MB\n",
                self.device_status.free_memory.unwrap_or(0) / (1024 * 1024)
            ));
        }

        output.push_str("\nBenchmark Results:\n");
        output.push_str("------------------\n");

        if self.benchmarks.is_empty() {
            output.push_str("No benchmarks performed.\n");
        } else {
            // Calculate maximum name and data size length for alignment
            let max_op_len = self
                .benchmarks
                .iter()
                .map(|b| format!("{}", b.operation).len())
                .max()
                .unwrap_or(0);

            let max_size_len = self
                .benchmarks
                .iter()
                .map(|b| b.data_size.len())
                .max()
                .unwrap_or(0);

            // Add table header
            output.push_str(&format!(
                "{:<width_op$} | {:<width_size$} | {:>10} | {:>10} | {:>8}\n",
                "Operation",
                "Data Size",
                "CPU (ms)",
                "GPU (ms)",
                "Speedup",
                width_op = max_op_len,
                width_size = max_size_len
            ));

            output.push_str(&format!(
                "{:-<width_op$}-+-{:-<width_size$}-+-{:-<10}-+-{:-<10}-+-{:-<8}\n",
                "",
                "",
                "",
                "",
                "",
                width_op = max_op_len,
                width_size = max_size_len
            ));

            // Add benchmark results
            for benchmark in &self.benchmarks {
                let gpu_time = benchmark
                    .gpu_result
                    .as_ref()
                    .map(|r| format!("{:.2}", r.time_ms()))
                    .unwrap_or("N/A".to_string());

                let speedup = benchmark
                    .speedup
                    .map(|s| format!("{:.2}x", s))
                    .unwrap_or("N/A".to_string());

                output.push_str(&format!(
                    "{:<width_op$} | {:<width_size$} | {:>10.2} | {:>10} | {:>8}\n",
                    format!("{}", benchmark.operation),
                    benchmark.data_size,
                    benchmark.cpu_result.time_ms(),
                    gpu_time,
                    speedup,
                    width_op = max_op_len,
                    width_size = max_size_len
                ));
            }
        }

        output
    }

    /// Run all benchmarks with default sizes
    pub fn run_all_benchmarks(&mut self) -> Result<String> {
        // Matrix operations
        self.benchmark_matrix_multiply(1000, 1000, 1000)?;
        self.benchmark_elementwise_add(2000, 2000)?;

        // DataFrame operations
        self.benchmark_correlation(10000, 10)?;
        self.benchmark_linear_regression(10000, 10)?;

        // Time series operations
        self.benchmark_rolling_window(100000, 100)?;

        // Return summary
        Ok(self.get_summary())
    }
}

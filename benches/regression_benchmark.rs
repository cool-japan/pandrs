use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pandrs::error::Result;
use pandrs::optimized::jit::*;
use pandrs::optimized::OptimizedDataFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Performance baseline data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub version: String,
    pub timestamp: u64,
    pub benchmarks: HashMap<String, BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub mean_time_ns: f64,
    pub std_dev_ns: f64,
    pub throughput_ops_per_sec: Option<f64>,
    pub memory_usage_bytes: Option<usize>,
}

/// Performance regression detector
pub struct RegressionDetector {
    baseline_path: String,
    current_baseline: Option<PerformanceBaseline>,
    regression_threshold: f64, // 10% = 0.1
}

impl RegressionDetector {
    pub fn new(baseline_path: &str, regression_threshold: f64) -> Self {
        let current_baseline = Self::load_baseline(baseline_path);
        Self {
            baseline_path: baseline_path.to_string(),
            current_baseline,
            regression_threshold,
        }
    }

    fn load_baseline(path: &str) -> Option<PerformanceBaseline> {
        if Path::new(path).exists() {
            let content = fs::read_to_string(path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn save_baseline(&self, baseline: &PerformanceBaseline) -> Result<()> {
        let json = serde_json::to_string_pretty(baseline).map_err(|e| {
            pandrs::error::Error::IoError(format!("JSON serialization error: {}", e))
        })?;
        fs::write(&self.baseline_path, json).map_err(|e| {
            pandrs::error::Error::IoError(format!("Failed to write baseline: {}", e))
        })?;
        Ok(())
    }

    pub fn check_regression(&self, benchmark_name: &str, current_time_ns: f64) -> Option<f64> {
        if let Some(ref baseline) = self.current_baseline {
            if let Some(baseline_result) = baseline.benchmarks.get(benchmark_name) {
                let regression_ratio =
                    (current_time_ns - baseline_result.mean_time_ns) / baseline_result.mean_time_ns;
                if regression_ratio > self.regression_threshold {
                    return Some(regression_ratio);
                }
            }
        }
        None
    }
}

/// Create standardized test data for regression testing
#[allow(clippy::result_large_err)]
fn create_regression_test_dataframe(size: usize) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    // Use deterministic data for consistent benchmarks
    let mut int_data = Vec::with_capacity(size);
    let mut float_data = Vec::with_capacity(size);
    let mut string_data = Vec::with_capacity(size);

    for i in 0..size {
        int_data.push((i % 10000) as i64);
        float_data.push((i as f64) * 0.1 + (i % 100) as f64);
        string_data.push(format!("Cat_{}", i % 100));
    }

    df.add_int_column("value", int_data)?;
    df.add_float_column("score", float_data)?;
    df.add_string_column("category", string_data)?;

    Ok(df)
}

/// Micro-benchmarks for regression detection
fn regression_dataframe_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_dataframe_creation");
    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);

    let test_size = 50_000;
    group.throughput(Throughput::Elements(test_size as u64));

    group.bench_function("standard_creation", |b| {
        let start = Instant::now();
        b.iter(|| black_box(create_regression_test_dataframe(test_size).unwrap()));
        let elapsed_ns = start.elapsed().as_nanos() as f64;

        if let Some(regression) = detector.check_regression("dataframe_creation", elapsed_ns) {
            eprintln!(
                "⚠️  REGRESSION DETECTED in dataframe_creation: {:.2}% slower",
                regression * 100.0
            );
        }
    });

    group.finish();
}

fn regression_aggregation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_aggregations");
    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);

    let df = create_regression_test_dataframe(100_000).unwrap();
    group.throughput(Throughput::Elements(100_000));

    // Test core aggregation operations
    let operations = [
        (
            "sum",
            Box::new(|df: &OptimizedDataFrame| df.sum("value").unwrap())
                as Box<dyn Fn(&OptimizedDataFrame) -> f64>,
        ),
        (
            "mean",
            Box::new(|df: &OptimizedDataFrame| df.mean("score").unwrap()),
        ),
        (
            "min",
            Box::new(|df: &OptimizedDataFrame| df.min("value").unwrap()),
        ),
        (
            "max",
            Box::new(|df: &OptimizedDataFrame| df.max("value").unwrap()),
        ),
    ];

    for (op_name, op_func) in operations {
        group.bench_function(op_name, |b| {
            let start = Instant::now();
            b.iter(|| black_box(op_func(&df)));
            let elapsed_ns = start.elapsed().as_nanos() as f64;

            if let Some(regression) =
                detector.check_regression(&format!("aggregation_{}", op_name), elapsed_ns)
            {
                eprintln!(
                    "⚠️  REGRESSION DETECTED in {}: {:.2}% slower",
                    op_name,
                    regression * 100.0
                );
            }
        });
    }

    group.finish();
}

fn regression_groupby_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_groupby");
    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);

    let df = create_regression_test_dataframe(50_000).unwrap();
    group.throughput(Throughput::Elements(50_000));

    group.bench_function("parallel_groupby", |b| {
        let start = Instant::now();
        b.iter(|| black_box(df.par_groupby(&["category"]).unwrap()));
        let elapsed_ns = start.elapsed().as_nanos() as f64;

        if let Some(regression) = detector.check_regression("parallel_groupby", elapsed_ns) {
            eprintln!(
                "⚠️  REGRESSION DETECTED in parallel_groupby: {:.2}% slower",
                regression * 100.0
            );
        }
    });

    group.finish();
}

fn regression_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_simd");
    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);

    let test_data: Vec<f64> = (0..100_000).map(|i| i as f64 * 0.1).collect();
    group.throughput(Throughput::Elements(100_000));

    // Test SIMD operations
    let simd_operations = [
        (
            "simd_sum",
            Box::new(|data: &[f64]| simd_sum_f64(data)) as Box<dyn Fn(&[f64]) -> f64>,
        ),
        ("simd_mean", Box::new(|data: &[f64]| simd_mean_f64(data))),
        ("simd_min", Box::new(|data: &[f64]| simd_min_f64(data))),
        ("simd_max", Box::new(|data: &[f64]| simd_max_f64(data))),
    ];

    for (op_name, op_func) in simd_operations {
        group.bench_function(op_name, |b| {
            let start = Instant::now();
            b.iter(|| black_box(op_func(&test_data)));
            let elapsed_ns = start.elapsed().as_nanos() as f64;

            if let Some(regression) =
                detector.check_regression(&format!("simd_{}", op_name), elapsed_ns)
            {
                eprintln!(
                    "⚠️  REGRESSION DETECTED in {}: {:.2}% slower",
                    op_name,
                    regression * 100.0
                );
            }
        });
    }

    group.finish();
}

fn regression_io_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_io");
    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);

    let df = create_regression_test_dataframe(10_000).unwrap();
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("csv_write", |b| {
        let start = Instant::now();
        b.iter(|| {
            let temp_path = format!("/tmp/regression_test_{}.csv", rand::random::<u32>());
            df.to_csv(&temp_path, true).unwrap();
            std::fs::remove_file(&temp_path).ok();
            black_box(())
        });
        let elapsed_ns = start.elapsed().as_nanos() as f64;

        if let Some(regression) = detector.check_regression("csv_write", elapsed_ns) {
            eprintln!(
                "⚠️  REGRESSION DETECTED in csv_write: {:.2}% slower",
                regression * 100.0
            );
        }
    });

    #[cfg(feature = "parquet")]
    group.bench_function("parquet_write", |b| {
        let start = Instant::now();
        b.iter(|| {
            use pandrs::io::{write_parquet, ParquetCompression};
            let temp_path = format!("/tmp/regression_test_{}.parquet", rand::random::<u32>());
            write_parquet(&df, &temp_path, Some(ParquetCompression::Snappy)).unwrap();
            std::fs::remove_file(&temp_path).ok();
            black_box(())
        });
        let elapsed_ns = start.elapsed().as_nanos() as f64;

        if let Some(regression) = detector.check_regression("parquet_write", elapsed_ns) {
            eprintln!(
                "⚠️  REGRESSION DETECTED in parquet_write: {:.2}% slower",
                regression * 100.0
            );
        }
    });

    group.finish();
}

/// Performance baseline establishment
#[allow(dead_code)]
fn establish_baseline() {
    println!("🎯 Establishing performance baseline...");

    let detector = RegressionDetector::new("benchmark_baseline.json", 0.1);
    let mut baseline = PerformanceBaseline {
        version: "0.1.0-alpha.4".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        benchmarks: HashMap::new(),
    };

    // Establish baselines for key operations
    let df = create_regression_test_dataframe(50_000).unwrap();
    let test_data: Vec<f64> = (0..50_000).map(|i| i as f64 * 0.1).collect();

    // Measure key operations
    let operations: &[(&str, Box<dyn Fn()>)] = &[
        (
            "dataframe_creation",
            Box::new(|| {
                create_regression_test_dataframe(50_000).unwrap();
            }),
        ),
        (
            "sum_operation",
            Box::new(|| {
                df.sum("value").unwrap();
            }),
        ),
        (
            "groupby_operation",
            Box::new(|| {
                df.par_groupby(&["category"]).unwrap();
            }),
        ),
        (
            "simd_sum",
            Box::new(|| {
                simd_sum_f64(&test_data);
            }),
        ),
    ];

    for (name, operation) in operations {
        let start = Instant::now();
        for _ in 0..10 {
            operation();
        }
        let elapsed = start.elapsed();
        let mean_time_ns = elapsed.as_nanos() as f64 / 10.0;

        baseline.benchmarks.insert(
            name.to_string(),
            BenchmarkResult {
                mean_time_ns,
                std_dev_ns: 0.0, // Would calculate from multiple runs in real implementation
                throughput_ops_per_sec: None,
                memory_usage_bytes: None,
            },
        );

        println!(
            "📊 Baseline for {}: {:.2}ms",
            name,
            mean_time_ns / 1_000_000.0
        );
    }

    detector.save_baseline(&baseline).unwrap();
    println!("✅ Baseline saved to benchmark_baseline.json");
}

criterion_group!(
    regression_benches,
    regression_dataframe_creation,
    regression_aggregation_operations,
    regression_groupby_operations,
    regression_simd_operations,
    regression_io_operations
);

criterion_main!(regression_benches);

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_baseline_creation() {
        establish_baseline();
    }
}

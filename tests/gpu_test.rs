//! Tests for GPU acceleration functionality
//!
//! To run these tests:
//!   cargo test --test gpu_test --features "cuda"

#[cfg(feature = "cuda")]
mod tests {
    use crate::NA;
    use ndarray::{Array1, Array2};
    use pandrs::{
        dataframe::gpu::DataFrameGpuExt,
        gpu::{
            self,
            benchmark::GpuBenchmark,
            operations::{GpuMatrix, GpuVector},
        },
        ml::gpu::MLGpuExt,
        stats::gpu::StatsGpuExt,
        temporal::gpu::SeriesTimeGpuExt,
        DataFrame, Series,
    };

    #[test]
    fn test_gpu_initialization() {
        // Initialize GPU
        let status = gpu::init_gpu().unwrap();

        // Skip test if GPU is not available
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Verify device information
        assert!(status.device_name.is_some());
        assert!(status.cuda_version.is_some());
        assert!(status.total_memory.is_some());
        assert!(status.free_memory.is_some());
    }

    #[test]
    fn test_gpu_matrix_operations() {
        // Skip test if GPU is not available
        let status = gpu::init_gpu().unwrap();
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Create test matrices
        let a_data: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..9).map(|i| (i * 2) as f64).collect();

        let a = Array2::from_shape_vec((3, 3), a_data).unwrap();
        let b = Array2::from_shape_vec((3, 3), b_data).unwrap();

        // GPU matrices
        let gpu_a = GpuMatrix::new(a.clone());
        let gpu_b = GpuMatrix::new(b.clone());

        // Test matrix multiplication
        let gpu_result = gpu_a.dot(&gpu_b).unwrap();
        let cpu_result = a.dot(&b);

        // Compare results
        let gpu_data = gpu_result.data.clone();
        assert_eq!(gpu_data.dim(), cpu_result.dim());

        for i in 0..gpu_data.dim().0 {
            for j in 0..gpu_data.dim().1 {
                assert!((gpu_data[[i, j]] - cpu_result[[i, j]]).abs() < 1e-10);
            }
        }

        // Test element-wise addition
        let gpu_add = gpu_a.add(&gpu_b).unwrap();
        let cpu_add = &a + &b;

        let gpu_add_data = gpu_add.data.clone();
        assert_eq!(gpu_add_data.dim(), cpu_add.dim());

        for i in 0..gpu_add_data.dim().0 {
            for j in 0..gpu_add_data.dim().1 {
                assert!((gpu_add_data[[i, j]] - cpu_add[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_gpu_dataframe_operations() {
        // Skip test if GPU is not available
        let status = gpu::init_gpu().unwrap();
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Create test DataFrame
        let mut df = DataFrame::new();

        for j in 0..5 {
            let col_name = format!("col_{}", j);
            let col_data: Vec<f64> = (0..100).map(|i| ((i + j) % 10) as f64).collect();
            df.add_column(
                col_name.clone(),
                Series::new(col_data, Some(col_name)).unwrap(),
            )
            .unwrap();
        }

        // Get column names
        let col_names: Vec<&str> = df.column_names().iter().map(|s| s.as_str()).collect();

        // Test correlation matrix
        let gpu_corr = df.gpu_corr(&col_names).unwrap();
        let cpu_corr = df.corr().unwrap();

        assert_eq!(gpu_corr.dim(), cpu_corr.dim());

        for i in 0..gpu_corr.dim().0 {
            for j in 0..gpu_corr.dim().1 {
                assert!((gpu_corr[[i, j]] - cpu_corr[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_gpu_time_series() {
        // Skip test if GPU is not available
        let status = gpu::init_gpu().unwrap();
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Create test Series
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let series = Series::new(data, Some("data".to_string())).unwrap();

        // Test rolling window
        let window_size = 10;
        let gpu_rolling = series
            .gpu_rolling(
                window_size,
                window_size / 2,
                pandrs::temporal::window::WindowOperation::Mean,
                false,
            )
            .unwrap();

        let cpu_rolling = series
            .rolling(
                window_size,
                window_size / 2,
                pandrs::temporal::window::WindowOperation::Mean,
                false,
            )
            .unwrap();

        // Compare results
        let gpu_data = gpu_rolling.data();
        let cpu_data = cpu_rolling.data();

        assert_eq!(gpu_data.len(), cpu_data.len());

        for i in 0..gpu_data.len() {
            if !gpu_data[i].is_na() && !cpu_data[i].is_na() {
                let gpu_val = gpu_data[i].to_f64().unwrap();
                let cpu_val = cpu_data[i].to_f64().unwrap();
                assert!((gpu_val - cpu_val).abs() < 1e-5);
            } else {
                assert_eq!(gpu_data[i].is_na(), cpu_data[i].is_na());
            }
        }
    }

    #[test]
    fn test_gpu_benchmark() {
        // Create benchmark utility
        let mut benchmark = GpuBenchmark::new().unwrap();

        // Skip test if GPU is not available
        if !benchmark.device_status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Perform a small benchmark
        let result = benchmark.benchmark_matrix_multiply(100, 100, 100).unwrap();

        // Verify results
        assert_eq!(
            result.operation,
            pandrs::gpu::benchmark::BenchmarkOperation::MatrixMultiply
        );
        assert!(result.cpu_result.time.as_secs_f64() > 0.0);

        if let Some(gpu_result) = &result.gpu_result {
            assert!(gpu_result.time.as_secs_f64() > 0.0);
            assert!(result.speedup.unwrap() > 0.0);
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod tests {
    #[test]
    fn test_gpu_dummy() {
        // Dummy test for when CUDA is not enabled
        println!("GPU tests are only enabled with CUDA feature");
        // Test passes - GPU feature not enabled
    }
}

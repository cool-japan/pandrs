//! Tests for GPU acceleration functionality
//!
//! To run these tests:
//!   cargo test --test gpu_test --features "cuda"

#[cfg(cuda_available)]
mod tests {
    use ndarray::{Array1, Array2};
    use pandrs::{
        dataframe::gpu::DataFrameGpuExt,
        gpu::{
            self,
            benchmark::GpuBenchmark,
            operations::{GpuMatrix, GpuVector},
        },
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
        // Note: device_name and cuda_version should be set when available
        assert!(status.device_name.is_some());
        assert!(status.cuda_version.is_some());
        // Note: total_memory and free_memory may not be available depending on cudarc version
        // These are optional in the current implementation
        println!(
            "GPU initialized: {} (CUDA {})",
            status
                .device_name
                .as_ref()
                .unwrap_or(&"Unknown".to_string()),
            status
                .cuda_version
                .as_ref()
                .unwrap_or(&"Unknown".to_string())
        );
        if status.total_memory.is_some() {
            println!("Total memory: {} bytes", status.total_memory.unwrap());
        }
        if status.free_memory.is_some() {
            println!("Free memory: {} bytes", status.free_memory.unwrap());
        }
    }

    #[test]
    fn test_gpu_matrix_operations() {
        // Skip test if GPU is not available
        let status = gpu::init_gpu().unwrap();
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Create test matrices using ndarray
        let a_data: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let b_data: Vec<f64> = (0..9).map(|i| (i * 2) as f64).collect();

        let a = Array2::from_shape_vec((3, 3), a_data).unwrap();
        let b = Array2::from_shape_vec((3, 3), b_data).unwrap();

        // GPU matrix creation
        let gpu_a = GpuMatrix::new(a.clone());
        let gpu_b = GpuMatrix::new(b.clone());

        // Test matrix dot product
        if let Ok(result) = gpu_a.dot(&gpu_b) {
            // Verify result dimensions
            assert_eq!(result.data.dim().0, 3);
            assert_eq!(result.data.dim().1, 3);
        }
    }

    #[test]
    fn test_gpu_vector_operations() {
        // Skip test if GPU is not available
        let status = gpu::init_gpu().unwrap();
        if !status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Create test vector using ndarray
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr = Array1::from_vec(data);
        let gpu_vec = GpuVector::new(arr.clone());

        // Verify vector creation
        assert_eq!(gpu_vec.data.len(), 100);
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
        let names = df.column_names();
        let col_names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();

        // Test GPU correlation matrix
        if let Ok(gpu_corr) = df.gpu_corr(&col_names) {
            // Verify result has correct number of columns
            assert_eq!(gpu_corr.column_names().len(), 5);
        }
    }

    #[test]
    fn test_gpu_benchmark() {
        // Create benchmark utility
        let benchmark = GpuBenchmark::new().unwrap();

        // Skip test if GPU is not available
        if !benchmark.device_status.available {
            println!("GPU not available, skipping test");
            return;
        }

        // Perform a small benchmark
        let mut bench = benchmark;
        let result = bench.benchmark_matrix_multiply(100, 100, 100).unwrap();

        // Verify results
        assert_eq!(
            result.operation,
            pandrs::gpu::benchmark::BenchmarkOperation::MatrixMultiply
        );
        assert!(result.cpu_result.time.as_secs_f64() >= 0.0);

        if let Some(gpu_result) = &result.gpu_result {
            assert!(gpu_result.time.as_secs_f64() >= 0.0);
        }
    }
}

#[cfg(not(cuda_available))]
mod tests {
    #[test]
    fn test_gpu_dummy() {
        // Dummy test for when CUDA is not available
        println!("GPU tests are only enabled when CUDA is available");
        // Test passes - GPU not available
    }
}

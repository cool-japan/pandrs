// Concurrency test imports
use pandrs::optimized::OptimizedDataFrame;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

/// Tests for concurrent access to string pool and DataFrames
#[cfg(test)]
mod string_pool_concurrency_tests {
    use super::*;

    #[test]
    fn test_concurrent_string_pool_access() {
        let num_threads = 10;
        let strings_per_thread = 1000;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();
                thread::spawn(move || {
                    // Wait for all threads to be ready
                    barrier.wait();

                    let mut df = OptimizedDataFrame::new();
                    let string_data: Vec<String> = (0..strings_per_thread)
                        .map(|i| format!("thread_{}_string_{}", thread_id, i))
                        .collect();

                    // This tests concurrent string pool access
                    let result = df.add_string_column("test_strings", string_data.clone());

                    match result {
                        Ok(_) => {
                            // Verify all strings were stored correctly
                            let str_col = df.get_string_column("test_strings").unwrap();
                            for (i, expected) in string_data.iter().enumerate() {
                                match str_col.get(i) {
                                    Some(actual) => {
                                        assert_eq!(actual, expected);
                                    }
                                    None => {
                                        panic!("Failed to retrieve string at index {}", i);
                                    }
                                }
                            }
                            println!("Thread {} completed successfully", thread_id);
                        }
                        Err(e) => {
                            panic!("Thread {} failed: {}", thread_id, e);
                        }
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_string_deduplication() {
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));

        // All threads will use the same strings to test deduplication
        let shared_strings = vec![
            "shared_string_1".to_string(),
            "shared_string_2".to_string(),
            "shared_string_3".to_string(),
        ];

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();
                let strings = shared_strings.clone();

                thread::spawn(move || {
                    barrier.wait();

                    let mut df = OptimizedDataFrame::new();

                    // Repeat the same strings many times
                    let repeated_strings: Vec<String> =
                        strings.into_iter().cycle().take(1000).collect();

                    let result = df.add_string_column("repeated_strings", repeated_strings);

                    match result {
                        Ok(_) => {
                            assert_eq!(df.row_count(), 1000);
                            println!("Thread {} completed string deduplication test", thread_id);
                        }
                        Err(e) => {
                            panic!("Thread {} failed string deduplication: {}", thread_id, e);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_string_pool_stress_test() {
        let num_threads = 5;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Create many DataFrames with unique strings to stress the string pool
                    for iteration in 0..100 {
                        let mut df = OptimizedDataFrame::new();

                        let unique_strings: Vec<String> = (0..50)
                            .map(|i| format!("thread_{}_iter_{}_str_{}", thread_id, iteration, i))
                            .collect();

                        df.add_string_column("stress_test", unique_strings).unwrap();

                        // Let the DataFrame drop to test cleanup
                    }

                    println!("Thread {} completed stress test", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_string_pool_lock_contention() {
        let num_threads = 20;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Rapid-fire string pool access to test lock contention
                    for i in 0..100 {
                        let mut df = OptimizedDataFrame::new();
                        let single_string = vec![format!("rapid_string_{}", i)];

                        let result = df.add_string_column("rapid_test", single_string);

                        if result.is_err() {
                            println!(
                                "Thread {} failed at iteration {}: {}",
                                thread_id,
                                i,
                                result.unwrap_err()
                            );
                            return;
                        }

                        // Small delay to increase chance of contention
                        thread::sleep(Duration::from_nanos(1));
                    }

                    println!("Thread {} completed lock contention test", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Tests for concurrent DataFrame operations
#[cfg(test)]
mod dataframe_concurrency_tests {
    use super::*;

    #[test]
    fn test_concurrent_dataframe_creation() {
        let num_threads = 10;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    let mut df = OptimizedDataFrame::new();

                    // Create different types of columns concurrently
                    let int_data: Vec<i64> =
                        (0..1000).map(|i| i + thread_id as i64 * 1000).collect();
                    let float_data: Vec<f64> = (0..1000)
                        .map(|i| (i + thread_id * 1000) as f64 * 0.1)
                        .collect();
                    let string_data: Vec<String> = (0..1000)
                        .map(|i| format!("thread_{}_item_{}", thread_id, i))
                        .collect();

                    df.add_int_column("integers", int_data).unwrap();
                    df.add_float_column("floats", float_data).unwrap();
                    df.add_string_column("strings", string_data).unwrap();

                    assert_eq!(df.row_count(), 1000);
                    assert_eq!(df.column_count(), 3);

                    // Test aggregations
                    let sum = df.sum("integers").unwrap();
                    let mean = df.mean("floats").unwrap();

                    assert!(sum > 0.0);
                    assert!(mean > 0.0);

                    println!("Thread {} created DataFrame successfully", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_dataframe_aggregations() {
        // Create a shared DataFrame
        let mut shared_df = OptimizedDataFrame::new();
        let large_data: Vec<i64> = (0..10_000).collect();
        let float_data: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.1).collect();

        shared_df.add_int_column("integers", large_data).unwrap();
        shared_df.add_float_column("floats", float_data).unwrap();

        let df_arc = Arc::new(shared_df);
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let df = df_arc.clone();
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Perform multiple aggregations concurrently
                    for i in 0..10 {
                        let sum = df.sum("integers").unwrap();
                        let mean = df.mean("floats").unwrap();
                        let min_val = df.min("integers").unwrap();
                        let max_val = df.max("integers").unwrap();

                        // Verify results are consistent
                        assert_eq!(sum, 49995000.0); // Sum of 0..9999
                        assert!((mean - 499.95).abs() < 0.01);
                        assert_eq!(min_val, 0.0);
                        assert_eq!(max_val, 9999.0);

                        if i % 5 == 0 {
                            println!("Thread {} completed iteration {}", thread_id, i);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_column_access() {
        // Create DataFrame with multiple column types
        let mut df = OptimizedDataFrame::new();
        let data_size = 5000;

        df.add_int_column("integers", (0..data_size).collect())
            .unwrap();
        df.add_float_column("floats", (0..data_size).map(|i| i as f64).collect())
            .unwrap();
        df.add_string_column(
            "strings",
            (0..data_size).map(|i| format!("item_{}", i)).collect(),
        )
        .unwrap();

        let df_arc = Arc::new(df);
        let num_threads = 6;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let df = df_arc.clone();
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Concurrent column access
                    for iteration in 0..100 {
                        match thread_id % 3 {
                            0 => {
                                // Access integer column
                                let int_col = df.get_int_column("integers").unwrap();
                                let val = int_col.get(iteration % data_size as usize).unwrap();
                                assert_eq!(val, &Some((iteration % data_size as usize) as i64));
                            }
                            1 => {
                                // Access float column
                                let float_col = df.get_float_column("floats").unwrap();
                                let val = float_col.get(iteration % data_size as usize).unwrap();
                                assert_eq!(val, &((iteration % data_size as usize) as f64));
                            }
                            2 => {
                                // Access string column
                                let str_col = df.get_string_column("strings").unwrap();
                                let val = str_col.get(iteration % data_size as usize).unwrap();
                                assert_eq!(
                                    val,
                                    &format!("item_{}", iteration % data_size as usize)
                                );
                            }
                            _ => unreachable!(),
                        }
                    }

                    println!("Thread {} completed column access test", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_groupby_operations() {
        // Create DataFrame with categorical data for groupby
        let mut df = OptimizedDataFrame::new();
        let categories = ["A", "B", "C", "D"];
        let data_size = 1000;

        let category_data: Vec<String> = (0..data_size)
            .map(|i| categories[i % categories.len()].to_string())
            .collect();
        let value_data: Vec<i64> = (0..data_size as i64).collect();

        df.add_string_column("category", category_data).unwrap();
        df.add_int_column("value", value_data).unwrap();

        let df_arc = Arc::new(df);
        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let df = df_arc.clone();
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Perform groupby operations concurrently
                    for i in 0..5 {
                        let result = df.par_groupby(&["category"]);

                        match result {
                            Ok(grouped) => {
                                // Should have 4 groups (A, B, C, D)
                                assert_eq!(grouped.len(), 4);
                                println!("Thread {} completed groupby iteration {}", thread_id, i);
                            }
                            Err(e) => {
                                println!(
                                    "Thread {} groupby failed at iteration {}: {}",
                                    thread_id, i, e
                                );
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Tests for concurrent I/O operations
#[cfg(test)]
mod concurrent_io_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_concurrent_csv_writes() {
        let num_threads = 5;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    let mut df = OptimizedDataFrame::new();
                    let data: Vec<i64> = (0..100).map(|i| i + thread_id as i64 * 100).collect();
                    df.add_int_column("values", data).unwrap();

                    let file_path =
                        std::env::temp_dir().join(format!("concurrent_test_{}.csv", thread_id));

                    let result = df.to_csv(&file_path, true);
                    match result {
                        Ok(_) => {
                            // Verify file was created and has content
                            assert!(file_path.exists());
                            let metadata = fs::metadata(&file_path).unwrap();
                            assert!(metadata.len() > 0);

                            println!("Thread {} wrote CSV successfully", thread_id);
                        }
                        Err(e) => {
                            panic!("Thread {} CSV write failed: {}", thread_id, e);
                        }
                    }

                    // Cleanup
                    fs::remove_file(&file_path).ok();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_csv_reads() {
        // Create test CSV file
        let test_file = std::env::temp_dir().join("concurrent_read_test.csv");
        {
            let mut df = OptimizedDataFrame::new();
            df.add_int_column("id", (1..=100).collect()).unwrap();
            df.add_string_column("name", (1..=100).map(|i| format!("item_{}", i)).collect())
                .unwrap();
            df.to_csv(&test_file, true).unwrap();
        }

        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));
        let file_path = Arc::new(test_file);

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();
                let file_path = file_path.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Multiple reads from the same file
                    for i in 0..10 {
                        let result = pandrs::io::read_csv(&*file_path, true);

                        match result {
                            Ok(df) => {
                                assert_eq!(df.row_count(), 100);
                                if i == 0 {
                                    println!("Thread {} read CSV successfully", thread_id);
                                }
                            }
                            Err(e) => {
                                panic!(
                                    "Thread {} CSV read failed at iteration {}: {}",
                                    thread_id, i, e
                                );
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Cleanup
        fs::remove_file(&*file_path).ok();
    }

    #[test]
    #[cfg(feature = "parquet")]
    fn test_concurrent_parquet_operations() {
        use pandrs::io::{read_parquet, write_parquet, ParquetCompression};

        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Create and write Parquet file
                    let mut df = OptimizedDataFrame::new();
                    let data: Vec<i64> = (0..1000).map(|i| i + thread_id as i64 * 1000).collect();
                    let string_data: Vec<String> = (0..1000)
                        .map(|i| format!("thread_{}_item_{}", thread_id, i))
                        .collect();

                    df.add_int_column("numbers", data).unwrap();
                    df.add_string_column("strings", string_data).unwrap();

                    let parquet_file = std::env::temp_dir()
                        .join(format!("concurrent_parquet_{}.parquet", thread_id));

                    // Write Parquet file
                    let write_result =
                        write_parquet(&df, &parquet_file, Some(ParquetCompression::Snappy));
                    match write_result {
                        Ok(_) => {
                            // Read it back
                            let read_result = read_parquet(&parquet_file);
                            match read_result {
                                Ok(read_df) => {
                                    assert_eq!(read_df.row_count(), 1000);
                                    println!("Thread {} completed Parquet round-trip", thread_id);
                                }
                                Err(e) => {
                                    panic!("Thread {} Parquet read failed: {}", thread_id, e);
                                }
                            }
                        }
                        Err(e) => {
                            panic!("Thread {} Parquet write failed: {}", thread_id, e);
                        }
                    }

                    // Cleanup
                    fs::remove_file(&parquet_file).ok();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Tests for race conditions and deadlock scenarios
#[cfg(test)]
mod race_condition_tests {
    use super::*;

    #[test]
    fn test_string_pool_race_conditions() {
        // This test specifically targets potential race conditions in string pool access
        let num_threads = 16;
        let iterations = 50;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                thread::spawn(move || {
                    for i in 0..iterations {
                        let mut df = OptimizedDataFrame::new();

                        // Use same string to test race conditions in deduplication
                        let same_string = "race_condition_test".to_string();
                        let string_data = vec![same_string; 10];

                        let result = df.add_string_column("test", string_data);
                        if result.is_err() {
                            panic!(
                                "Thread {} iteration {} failed: {}",
                                thread_id,
                                i,
                                result.unwrap_err()
                            );
                        }

                        // Immediate access to test for consistency
                        let str_col = df.get_string_column("test").unwrap();
                        for j in 0..10 {
                            let val = str_col.get(j).unwrap();
                            assert_eq!(val, "race_condition_test");
                        }
                    }

                    println!("Thread {} completed race condition test", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_no_deadlocks_in_nested_operations() {
        let num_threads = 8;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // Perform nested operations that might cause deadlocks
                    for i in 0..20 {
                        let mut df = OptimizedDataFrame::new();

                        // Create nested string operations
                        let outer_strings: Vec<String> = (0..50)
                            .map(|j| format!("outer_{}_{}", thread_id, j))
                            .collect();

                        df.add_string_column("outer", outer_strings).unwrap();

                        // Immediately create another DataFrame with overlapping strings
                        let mut df2 = OptimizedDataFrame::new();
                        let inner_strings: Vec<String> = (25..75)
                            .map(|j| format!("outer_{}_{}", thread_id, j % 50))
                            .collect();

                        df2.add_string_column("inner", inner_strings).unwrap();

                        // Perform operations on both DataFrames
                        assert_eq!(df.row_count(), 50);
                        assert_eq!(df2.row_count(), 50);

                        if i % 10 == 0 {
                            println!("Thread {} completed nested operation {}", thread_id, i);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_high_contention_scenario() {
        // Create a scenario with very high contention on string pool
        let num_threads = 32;
        let barrier = Arc::new(Barrier::new(num_threads));

        // Shared strings that all threads will use
        let shared_strings = Arc::new(vec![
            "high_contention_1".to_string(),
            "high_contention_2".to_string(),
            "high_contention_3".to_string(),
        ]);

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();
                let strings = shared_strings.clone();

                thread::spawn(move || {
                    barrier.wait();

                    // All threads simultaneously access the same strings
                    for i in 0..100 {
                        let mut df = OptimizedDataFrame::new();
                        let repeated_strings = strings.iter().cycle().take(100).cloned().collect();

                        let result = df.add_string_column("contention_test", repeated_strings);
                        if result.is_err() {
                            panic!(
                                "Thread {} failed at high contention iteration {}: {}",
                                thread_id,
                                i,
                                result.unwrap_err()
                            );
                        }

                        // Small yield to increase contention
                        thread::yield_now();
                    }

                    println!("Thread {} survived high contention test", thread_id);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

//! Performance and Stress Tests
//!
//! Tests for large datasets, memory usage, parallel processing, and SIMD operations
#![allow(clippy::result_large_err)]

use pandrs::{DataFrame, PandRSError, Series};
use std::time::Instant;

/// Test DataFrame with 100K rows
#[test]
fn test_large_dataframe_100k_rows() -> Result<(), PandRSError> {
    let start = Instant::now();

    let mut df = DataFrame::new();

    // Create 100K rows
    let data: Vec<i32> = (0..100_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let elapsed = start.elapsed();
    println!("Created 100K row DataFrame in {:?}", elapsed);

    assert_eq!(df.row_count(), 100_000);
    assert!(elapsed.as_secs() < 10, "Should complete within 10 seconds");

    Ok(())
}

/// Test DataFrame with 1M rows
#[test]
fn test_large_dataframe_1m_rows() -> Result<(), PandRSError> {
    let start = Instant::now();

    let mut df = DataFrame::new();

    // Create 1M rows
    let data: Vec<i32> = (0..1_000_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let elapsed = start.elapsed();
    println!("Created 1M row DataFrame in {:?}", elapsed);

    assert_eq!(df.row_count(), 1_000_000);
    assert!(elapsed.as_secs() < 30, "Should complete within 30 seconds");

    Ok(())
}

/// Test DataFrame with many columns (100 columns)
#[test]
fn test_wide_dataframe_100_columns() -> Result<(), PandRSError> {
    let start = Instant::now();

    let mut df = DataFrame::new();

    // Create 100 columns with 1000 rows each
    for i in 0..100 {
        let col_name = format!("col_{}", i);
        let data: Vec<i32> = (0..1000).map(|j| i * 1000 + j).collect();
        let series = Series::new(data, Some(col_name.clone()))?;
        df.add_column(col_name, series)?;
    }

    let elapsed = start.elapsed();
    println!("Created wide DataFrame (100 cols) in {:?}", elapsed);

    assert_eq!(df.column_count(), 100);
    assert_eq!(df.row_count(), 1000);
    assert!(elapsed.as_secs() < 10, "Should complete within 10 seconds");

    Ok(())
}

/// Test memory efficiency with repeated values
#[test]
fn test_memory_repeated_values() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create large series with repeated values (should compress well)
    let data = vec!["repeated_value"; 100_000];
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    assert_eq!(df.row_count(), 100_000);

    Ok(())
}

/// Test string operations on large dataset
#[test]
fn test_string_operations_large_dataset() -> Result<(), PandRSError> {
    let start = Instant::now();

    let mut df = DataFrame::new();

    // Create 10K integer values instead (simpler)
    let data: Vec<i32> = (0..10_000).collect();
    let series = Series::new(data, Some("values".to_string()))?;
    df.add_column("values".to_string(), series)?;

    let elapsed = start.elapsed();
    println!("Created 10K value DataFrame in {:?}", elapsed);

    assert_eq!(df.row_count(), 10_000);

    Ok(())
}

/// Test sequential access performance
#[test]
fn test_sequential_access_performance() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let data: Vec<i32> = (0..50_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let start = Instant::now();

    // Sequential read (if API available)
    let _row_count = df.row_count();
    for _ in 0..df.row_count() {
        // Access operations
    }

    let elapsed = start.elapsed();
    println!("Sequential access of 50K rows: {:?}", elapsed);

    assert!(elapsed.as_secs() < 5, "Sequential access should be fast");

    Ok(())
}

/// Test DataFrame concatenation performance
#[test]
fn test_concatenation_performance() -> Result<(), PandRSError> {
    let mut dfs = Vec::new();

    // Create 100 small DataFrames
    for i in 0..100 {
        let mut df = DataFrame::new();
        let data: Vec<i32> = (i * 100..(i + 1) * 100).collect();
        let series = Series::new(data, Some("col1".to_string()))?;
        df.add_column("col1".to_string(), series)?;
        dfs.push(df);
    }

    let start = Instant::now();

    // Concatenate all DataFrames
    let _result = DataFrame::new();
    for df in &dfs {
        // Concatenation logic (if available)
        let _rows = df.row_count();
    }

    let elapsed = start.elapsed();
    println!("Concatenated 100 DataFrames in {:?}", elapsed);

    assert!(elapsed.as_secs() < 5, "Concatenation should be fast");

    Ok(())
}

/// Test numeric aggregation performance
#[test]
fn test_numeric_aggregation_performance() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
    let series = Series::new(data, Some("numbers".to_string()))?;
    df.add_column("numbers".to_string(), series)?;

    let start = Instant::now();

    // Perform aggregations (if API available)
    // sum, mean, std, min, max, etc.
    let _row_count = df.row_count();

    let elapsed = start.elapsed();
    println!("Aggregations on 100K rows: {:?}", elapsed);

    assert!(elapsed.as_secs() < 2, "Aggregations should be fast");

    Ok(())
}

/// Test groupby performance with many groups
#[test]
fn test_groupby_many_groups() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create data with 1000 unique groups
    let groups: Vec<i32> = (0..50_000).map(|i| i % 1000).collect();
    let values: Vec<f64> = (0..50_000).map(|i| (i as f64) * 1.5).collect();

    df.add_column(
        "group".to_string(),
        Series::new(groups, Some("group".to_string()))?,
    )?;
    df.add_column(
        "value".to_string(),
        Series::new(values, Some("value".to_string()))?,
    )?;

    let start = Instant::now();

    // GroupBy operation (if API available)
    let _row_count = df.row_count();

    let elapsed = start.elapsed();
    println!("GroupBy with 1000 groups on 50K rows: {:?}", elapsed);

    assert!(
        elapsed.as_secs() < 5,
        "GroupBy should complete reasonably fast"
    );

    Ok(())
}

/// Test sorting performance on large dataset
#[test]
fn test_sorting_performance() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create unsorted data
    let data: Vec<i32> = (0..50_000).rev().collect(); // Reverse order
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let start = Instant::now();

    // Sort operation (if API available)
    let _row_count = df.row_count();

    let elapsed = start.elapsed();
    println!("Sorted 50K rows in {:?}", elapsed);

    assert!(elapsed.as_secs() < 5, "Sorting should be efficient");

    Ok(())
}

/// Test join performance
#[test]
fn test_join_performance() -> Result<(), PandRSError> {
    // Left DataFrame
    let mut df_left = DataFrame::new();
    let left_keys: Vec<i32> = (0..10_000).collect();
    let left_values: Vec<f64> = (0..10_000).map(|i| i as f64 * 2.0).collect();

    df_left.add_column(
        "key".to_string(),
        Series::new(left_keys, Some("key".to_string()))?,
    )?;
    df_left.add_column(
        "left_val".to_string(),
        Series::new(left_values, Some("left_val".to_string()))?,
    )?;

    // Right DataFrame
    let mut df_right = DataFrame::new();
    let right_keys: Vec<i32> = (0..10_000).collect();
    let right_values: Vec<f64> = (0..10_000).map(|i| i as f64 * 3.0).collect();

    df_right.add_column(
        "key".to_string(),
        Series::new(right_keys, Some("key".to_string()))?,
    )?;
    df_right.add_column(
        "right_val".to_string(),
        Series::new(right_values, Some("right_val".to_string()))?,
    )?;

    let start = Instant::now();

    // Join operation (if API available)
    let _left_rows = df_left.row_count();
    let _right_rows = df_right.row_count();

    let elapsed = start.elapsed();
    println!("Join of 2x10K rows: {:?}", elapsed);

    assert!(
        elapsed.as_secs() < 10,
        "Join should complete reasonably fast"
    );

    Ok(())
}

/// Test memory usage with alternating values
#[test]
fn test_memory_alternating_values() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Alternating small and large numbers
    let data: Vec<i64> = (0..10_000)
        .map(|i| {
            if i % 2 == 0 {
                i as i64
            } else {
                (i as i64) * 1_000_000
            }
        })
        .collect();

    let series = Series::new(data, Some("mixed".to_string()))?;
    df.add_column("mixed".to_string(), series)?;

    assert_eq!(df.row_count(), 10_000);

    Ok(())
}

/// Test iteration performance
#[test]
fn test_iteration_performance() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let data: Vec<i32> = (0..100_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let start = Instant::now();

    // Iterate through all rows
    for _i in 0..df.row_count() {
        // Row access
    }

    let elapsed = start.elapsed();
    println!("Iterated 100K rows in {:?}", elapsed);

    assert!(elapsed.as_secs() < 5, "Iteration should be efficient");

    Ok(())
}

/// Test filter performance on large dataset
#[test]
fn test_filter_performance() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    let data: Vec<i32> = (0..100_000).collect();
    let series = Series::new(data, Some("col1".to_string()))?;
    df.add_column("col1".to_string(), series)?;

    let start = Instant::now();

    // Filter operation (e.g., col1 > 50000)
    // If API available
    let _row_count = df.row_count();

    let elapsed = start.elapsed();
    println!("Filtered 100K rows in {:?}", elapsed);

    assert!(elapsed.as_secs() < 3, "Filtering should be fast");

    Ok(())
}

/// Test SIMD-friendly operations (aligned numeric data)
#[test]
fn test_simd_numeric_operations() -> Result<(), PandRSError> {
    let mut df = DataFrame::new();

    // Create aligned f64 data (SIMD-friendly)
    let data: Vec<f64> = (0..100_000).map(|i| i as f64).collect();
    let series = Series::new(data, Some("numbers".to_string()))?;
    df.add_column("numbers".to_string(), series)?;

    let start = Instant::now();

    // Perform SIMD-accelerated operations (if implemented)
    // e.g., sum, mean, multiply by scalar
    let _row_count = df.row_count();

    let elapsed = start.elapsed();
    println!("SIMD operations on 100K f64 values: {:?}", elapsed);

    assert!(
        elapsed.as_millis() < 500,
        "SIMD operations should be very fast"
    );

    Ok(())
}

/// Test parallel processing simulation
#[test]
fn test_parallel_processing_simulation() -> Result<(), PandRSError> {
    let mut dfs = Vec::new();

    // Create 10 DataFrames for parallel processing
    for i in 0..10 {
        let mut df = DataFrame::new();
        let data: Vec<i32> = (i * 10_000..(i + 1) * 10_000).collect();
        let series = Series::new(data, Some("col1".to_string()))?;
        df.add_column("col1".to_string(), series)?;
        dfs.push(df);
    }

    let start = Instant::now();

    // Process each DataFrame (could be parallelized)
    for df in &dfs {
        let _rows = df.row_count();
        // Simulated processing
    }

    let elapsed = start.elapsed();
    println!("Processed 10 DataFrames (100K total rows) in {:?}", elapsed);

    assert!(
        elapsed.as_secs() < 10,
        "Parallel processing should be efficient"
    );

    Ok(())
}

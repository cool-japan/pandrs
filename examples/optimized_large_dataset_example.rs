#[cfg(feature = "optimized")]
use pandrs::error::Result;
#[cfg(feature = "optimized")]
use pandrs::ColumnTrait;
#[cfg(feature = "optimized")]
use pandrs::DiskBasedOptimizedDataFrame;

#[cfg(not(feature = "optimized"))]
fn main() {
    println!("This example requires the 'optimized' feature flag to be enabled.");
    println!("Please recompile with:");
    println!("  cargo run --example optimized_large_dataset_example --features \"optimized\"");
}

#[cfg(feature = "optimized")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Path to a large CSV file (replace with actual path)
    let file_path = "examples/data/large_dataset.csv";

    println!("Working with large datasets using OptimizedDataFrame");
    println!("---------------------------------------------------");

    // Create a disk-based OptimizedDataFrame with default configuration
    let disk_df = DiskBasedOptimizedDataFrame::new(file_path, None)?;

    // Example: Compute summary statistics for numeric columns
    println!("\nComputing summary statistics:");

    let stats = disk_df.aggregate(
        // Process each chunk
        |chunk| {
            // For each numeric column, collect min, max, sum, count
            let mut stats = std::collections::HashMap::new();

            for col_name in chunk.column_names() {
                if let Ok(col_view) = chunk.column(col_name) {
                    if let Some(numeric_col) = col_view.as_float64() {
                        // Use built-in methods instead of direct access to data
                        let count = numeric_col.len();
                        if count == 0 {
                            continue;
                        }

                        let min = numeric_col.min().unwrap_or(f64::INFINITY);
                        let max = numeric_col.max().unwrap_or(f64::NEG_INFINITY);
                        let sum = numeric_col.sum();
                        // Keep the count variable

                        stats.insert(col_name.to_string(), (min, max, sum, count));
                    }
                }
            }

            Ok(stats)
        },
        // Combine results
        |chunk_stats| {
            let mut combined = std::collections::HashMap::new();

            for stats in chunk_stats {
                for (col, (min, max, sum, count)) in stats {
                    combined
                        .entry(col.clone())
                        .and_modify(
                            |(c_min, c_max, c_sum, c_count): &mut (f64, f64, f64, usize)| {
                                *c_min = (*c_min).min(min);
                                *c_max = (*c_max).max(max);
                                *c_sum += sum;
                                *c_count += count;
                            },
                        )
                        .or_insert((min, max, sum, count));
                }
            }

            Ok(combined)
        },
    )?;

    // Print summary statistics
    for (col, (min, max, sum, count)) in stats {
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        println!("Column: {}", col);
        println!("  - Min: {:.2}", min);
        println!("  - Max: {:.2}", max);
        println!("  - Mean: {:.2}", mean);
        println!("  - Count: {}", count);
        println!();
    }

    // Convert to in-memory OptimizedDataFrame (if size permits)
    println!("Converting to in-memory OptimizedDataFrame...");

    // This step would typically be wrapped in error handling to handle
    // cases where the dataset is too large for memory
    match disk_df.to_optimized_dataframe() {
        Ok(optimized_df) => {
            println!("Successfully loaded into memory!");
            println!(
                "DataFrame has {} rows and {} columns",
                optimized_df.row_count(),
                optimized_df.column_count()
            );

            // Now use in-memory operations which are faster
            // Use usize and provide seed parameter
            let sample = optimized_df.sample(optimized_df.row_count() / 100, true, None)?; // 1% sample
            println!(
                "Sample from in-memory DataFrame: {} rows",
                sample.row_count()
            );
        }
        Err(e) => {
            println!("Dataset too large for memory: {}", e);
            println!("Continue using disk-based processing instead");
        }
    }

    // Example: Find outliers in numeric columns
    println!("\nFinding outliers in numeric columns:");

    let outliers = disk_df.aggregate(
        // Process each chunk
        |chunk| {
            let mut outliers = Vec::new();

            for col_name in chunk.column_names() {
                if let Ok(col_view) = chunk.column(col_name) {
                    if let Some(numeric_col) = col_view.as_float64() {
                        let count = numeric_col.len();

                        if count < 10 {
                            continue;
                        }

                        // First get the mean using the built-in method
                        let mean = numeric_col.mean().unwrap_or(0.0);

                        // Calculate variance manually but using get() for each value
                        let mut sum_sq_diff = 0.0;
                        for i in 0..count {
                            if let Ok(Some(value)) = numeric_col.get(i) {
                                sum_sq_diff += (value - mean).powi(2);
                            }
                        }
                        let variance = sum_sq_diff / count as f64;
                        let std_dev = variance.sqrt();
                        let threshold = 3.0 * std_dev;

                        // Check each value
                        for i in 0..count {
                            if let Ok(Some(value)) = numeric_col.get(i) {
                                if (value - mean).abs() > threshold {
                                    // For this example, we're just collecting the column and value
                                    outliers.push((col_name.clone(), value));
                                }
                            }
                        }
                    }
                }
            }

            Ok(outliers)
        },
        // Combine results
        |chunk_outliers| {
            let mut all_outliers = Vec::new();

            for outliers in chunk_outliers {
                all_outliers.extend(outliers);
            }

            // Sort by column name for consistent output
            all_outliers.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(all_outliers)
        },
    )?;

    // Print outliers (limit to first 20)
    println!("Found {} total outliers", outliers.len());
    println!("Sample of outliers:");
    for (col, value) in outliers.iter().take(20) {
        println!("  - {}: {:.2}", col, value);
    }

    Ok(())
}

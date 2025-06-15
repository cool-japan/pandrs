use pandrs::error::Result;
use pandrs::large::DataFrameOperations;
use pandrs::{DiskBasedDataFrame, DiskConfig};
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    // Path to a large CSV file (replace with actual path)
    let file_path = "examples/data/large_dataset.csv";

    println!("Working with large datasets example");
    println!("----------------------------------");

    // Create a disk-based DataFrame with custom configuration
    let config = DiskConfig {
        memory_limit: 500 * 1024 * 1024, // 500MB memory limit
        chunk_size: 50_000,              // Process in chunks of 50,000 rows
        use_memory_mapping: true,        // Use memory mapping for efficiency
        temp_dir: None,                  // Use system temp directory
    };

    let disk_df = DiskBasedDataFrame::new(file_path, Some(config))?;

    // Get schema information
    println!("DataFrame Schema:");
    for column in disk_df.schema().column_names() {
        println!("  - {}", column);
    }

    // Process in chunks for counting rows
    let mut chunked_df = disk_df.chunked()?;
    let mut total_rows = 0;

    println!("\nProcessing in chunks:");
    while let Some(chunk) = chunked_df.next_chunk()? {
        let chunk_rows = chunk.row_count();
        total_rows += chunk_rows;
        println!("  - Processed chunk with {} rows", chunk_rows);
    }

    println!("\nTotal rows in dataset: {}", total_rows);

    // Example of filtering data
    println!("\nFiltering data:");
    let filtered = disk_df.filter(|value, _| {
        // Example filter: keep only values starting with 'A'
        value.starts_with('A')
    })?;

    println!("Filtered result has {} rows", filtered.len());

    // Example of selecting columns
    println!("\nSelecting columns:");
    let columns_to_select = vec!["column1", "column2"]; // Replace with actual column names
    let selected = disk_df.select(&columns_to_select)?;

    println!("Selected result has {} rows and columns:", selected.len());
    // Since the result is a Vec<HashMap<String, String>>, we need to check the keys of the first element
    if !selected.is_empty() {
        for column in selected[0].keys() {
            println!("  - {}", column);
        }
    }

    // Example of grouping and aggregation
    println!("\nGrouping and aggregation:");
    let grouped = disk_df.group_by("category_column", "value_column", |values| {
        // Example aggregation: calculate average
        let sum: f64 = values.iter().filter_map(|v| v.parse::<f64>().ok()).sum();
        let count = values.len();

        if count > 0 {
            Ok(format!("{:.2}", sum / count as f64))
        } else {
            Ok("0.0".to_string())
        }
    })?;

    println!("Grouped result has {} groups", grouped.len());

    // Example of parallel processing
    println!("\nParallel processing example:");
    let chunk_results = chunked_df.parallel_process(
        // Process each chunk
        |chunk| {
            let mut counts = HashMap::new();

            // Example: count occurrences of values in a column
            for row_idx in 0..chunk.row_count() {
                if let Ok(value) = chunk.get_string_value("category_column", row_idx) {
                    *counts.entry(value.to_string()).or_insert(0) += 1;
                }
            }

            Ok(counts)
        },
        // Combine results
        |chunk_maps| {
            let mut result_map = HashMap::new();

            // Merge all maps
            for chunk_map in chunk_maps {
                for (key, count) in chunk_map {
                    *result_map.entry(key).or_insert(0) += count;
                }
            }

            Ok(result_map)
        },
    )?;

    println!("Category counts from parallel processing:");
    for (category, count) in chunk_results.iter().take(5) {
        println!("  - {}: {}", category, count);
    }

    Ok(())
}

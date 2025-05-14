use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::optimized::jit::{
    JitCompilable, ParallelConfig, GroupByJitExt, parallel_custom
};
use std::time::{Instant, Duration};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pandrs JIT Parallel GroupBy Example");
    println!("===================================\n");

    // Create test data
    let rows = 1_000_000;
    let groups = 100;
    
    println!("Creating a dataframe with {} rows and {} groups...", rows, groups);
    let df = create_test_data(rows, groups)?;
    
    println!("DataFrame created with {} rows", df.len());
    println!("First few rows:");
    let preview = df.head(5)?;
    println!("{:?}", preview);
    
    // Group by category
    let grouped = df.group_by(&["category"])?;
    println!("\nGrouped by 'category' with {} groups", groups);
    
    // Standard JIT operations
    println!("\nStandard JIT operations:");
    
    // Standard sum
    let start = Instant::now();
    let sum_result = grouped.sum_jit("value", "sum_value")?;
    let std_sum_time = start.elapsed();
    
    println!("  Standard sum time: {:?}", std_sum_time);
    println!("  Result preview (first 3 rows):");
    println!("{:?}", sum_result.head(3)?);
    
    // Standard mean
    let start = Instant::now();
    let mean_result = grouped.mean_jit("value", "mean_value")?;
    let std_mean_time = start.elapsed();
    
    println!("  Standard mean time: {:?}", std_mean_time);
    
    // Standard standard deviation
    let start = Instant::now();
    let std_result = grouped.std_jit("value", "std_value")?;
    let std_std_time = start.elapsed();
    
    println!("  Standard std time: {:?}", std_std_time);
    
    // Parallel JIT operations
    println!("\nParallel JIT operations:");
    
    // Parallel sum
    let start = Instant::now();
    let par_sum_result = grouped.parallel_sum_jit("value", "sum_value", None)?;
    let par_sum_time = start.elapsed();
    
    println!("  Parallel sum time: {:?}", par_sum_time);
    println!("  Speedup: {:.2}x", std_sum_time.as_nanos() as f64 / par_sum_time.as_nanos() as f64);
    
    // Parallel mean
    let start = Instant::now();
    let par_mean_result = grouped.parallel_mean_jit("value", "mean_value", None)?;
    let par_mean_time = start.elapsed();
    
    println!("  Parallel mean time: {:?}", par_mean_time);
    println!("  Speedup: {:.2}x", std_mean_time.as_nanos() as f64 / par_mean_time.as_nanos() as f64);
    
    // Parallel standard deviation
    let start = Instant::now();
    let par_std_result = grouped.parallel_std_jit("value", "std_value", None)?;
    let par_std_time = start.elapsed();
    
    println!("  Parallel std time: {:?}", par_std_time);
    println!("  Speedup: {:.2}x", std_std_time.as_nanos() as f64 / par_std_time.as_nanos() as f64);
    
    // Custom parallel function example
    println!("\nCustom Parallel Function with GroupBy:");
    
    // Custom function: weighted mean where weight = position in group
    let weighted_mean = |values: Vec<f64>| -> f64 {
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
    };
    
    // Map function for weighted mean
    let map_weighted_mean = |chunk: &[f64]| -> f64 {
        let mut local_weighted_sum = 0.0;
        let mut local_weight_sum = 0.0;
        
        for (i, val) in chunk.iter().enumerate() {
            let weight = (i + 1) as f64;
            local_weighted_sum += val * weight;
            local_weight_sum += weight;
        }
        
        if local_weight_sum == 0.0 { 0.0 } else { local_weighted_sum / local_weight_sum }
    };
    
    // Reduce function for weighted mean
    let reduce_weighted_mean = |partial_results: Vec<f64>| -> f64 {
        if partial_results.is_empty() {
            return 0.0;
        }
        
        partial_results.iter().sum::<f64>() / partial_results.len() as f64
    };
    
    // Create the parallel function
    let parallel_weighted_mean = parallel_custom(
        "parallel_weighted_mean",
        weighted_mean.clone(),
        map_weighted_mean,
        reduce_weighted_mean,
        None,
    );
    
    // Standard execution
    let start = Instant::now();
    let std_custom_result = grouped.aggregate_jit("value", weighted_mean, "weighted_mean")?;
    let std_custom_time = start.elapsed();
    
    println!("  Standard custom function time: {:?}", std_custom_time);
    println!("  Result preview (first 3 rows):");
    println!("{:?}", std_custom_result.head(3)?);
    
    // Parallel execution
    let start = Instant::now();
    let par_custom_result = grouped.aggregate_jit("value", parallel_weighted_mean, "parallel_weighted_mean")?;
    let par_custom_time = start.elapsed();
    
    println!("  Parallel custom function time: {:?}", par_custom_time);
    println!("  Speedup: {:.2}x", std_custom_time.as_nanos() as f64 / par_custom_time.as_nanos() as f64);
    println!("  Result preview (first 3 rows):");
    println!("{:?}", par_custom_result.head(3)?);
    
    // Test different parallel configurations
    println!("\nTesting different parallel configurations:");
    
    let chunk_sizes = [100, 1000, 10000];
    
    for &chunk_size in &chunk_sizes {
        let config = ParallelConfig::new().with_min_chunk_size(chunk_size);
        
        let start = Instant::now();
        let _result = grouped.parallel_sum_jit("value", "sum_value", Some(config))?;
        let duration = start.elapsed();
        
        println!("  Chunk size {:6}: {:?}", chunk_size, duration);
    }
    
    Ok(())
}

// Create test data with random values and categories
fn create_test_data(rows: usize, num_groups: usize) -> Result<OptimizedDataFrame, Box<dyn std::error::Error>> {
    let mut df = OptimizedDataFrame::new();
    let mut rng = rand::rng();
    
    // Create category and value columns
    let mut categories = Vec::with_capacity(rows);
    let mut values = Vec::with_capacity(rows);
    
    for _ in 0..rows {
        // Assign random category
        let category = format!("group_{}", rng.random_range(0..num_groups));
        categories.push(category);
        
        // Generate random value
        let value = rng.random_range(0.0..100.0);
        values.push(value);
    }
    
    // Add columns to dataframe
    df.add_column("category", categories)?;
    df.add_column("value", values)?;
    
    Ok(df)
}
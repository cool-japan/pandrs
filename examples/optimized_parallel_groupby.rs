use pandrs::error::Result;
use pandrs::optimized::{AggregateOp, OptimizedDataFrame};
use std::time::Instant;

fn main() -> Result<()> {
    println!("PandRS Parallel GroupBy Operations Example");
    println!("=========================================");

    // Create a large dataset for benchmarking
    let mut df = create_large_dataset(100_000)?;
    
    println!("Created dataset with {} rows", df.row_count);
    
    // Group by category - this will create multiple groups
    benchmark_operation("Group by category", || {
        df.group_by(["category"])
    })?;
    
    let grouped = df.group_by(["category"])?;
    
    // Basic aggregations - Serial vs Parallel
    println!("\n1. Basic Aggregation Operations");
    println!("------------------------------");
    
    benchmark_comparison(
        "Sum aggregation",
        || grouped.sum("value"),
        || grouped.par_sum("value")
    )?;
    
    benchmark_comparison(
        "Mean aggregation",
        || grouped.mean("value"),
        || grouped.par_mean("value")
    )?;
    
    benchmark_comparison(
        "Standard deviation",
        || grouped.std("value"),
        || grouped.par_std("value")
    )?;
    
    // Multiple aggregations
    println!("\n2. Multiple Aggregation Operations");
    println!("---------------------------------");
    
    let aggs = &[
        ("value", AggregateOp::Mean),
        ("value", AggregateOp::Std),
        ("value", AggregateOp::Min),
        ("value", AggregateOp::Max),
        ("value", AggregateOp::Count),
    ];
    
    benchmark_comparison(
        "Multiple aggregations",
        || grouped.agg(aggs),
        || grouped.par_agg(aggs)
    )?;
    
    // Custom aggregation
    println!("\n3. Custom Aggregation Operations");
    println!("-------------------------------");
    
    // Define coefficient of variation function
    let cv_func = |values: &[f64]| -> f64 {
        if values.is_empty() || values.iter().sum::<f64>() == 0.0 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance = values.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>() / (values.len() - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        // CV = std_dev / mean
        std_dev / mean
    };
    
    benchmark_comparison(
        "Custom aggregation (CV)",
        || grouped.custom("value", "value_cv", cv_func),
        || grouped.par_custom("value", "value_cv", cv_func)
    )?;
    
    // Filtering
    println!("\n4. Filtering Operations");
    println!("----------------------");
    
    // Define a filter function - groups where the mean value > 50
    let filter_func = |group_df: &OptimizedDataFrame| -> bool {
        let values = group_df.get_int_column("value").unwrap();
        let sum: i64 = values.iter().filter_map(|v| v).sum();
        let count = values.iter().filter_map(|v| v).count();
        
        if count == 0 {
            false
        } else {
            (sum as f64 / count as f64) > 50.0
        }
    };
    
    benchmark_comparison(
        "Filter groups",
        || grouped.filter(filter_func),
        || grouped.par_filter(filter_func)
    )?;
    
    // Transform operations
    println!("\n5. Transform Operations");
    println!("---------------------");
    
    // Define a transform function - calculate percentages of value relative to group total
    let transform_func = |group_df: &OptimizedDataFrame| -> Result<OptimizedDataFrame> {
        let mut result = group_df.clone();
        
        let values = group_df.get_int_column("value").unwrap();
        let total: i64 = values.iter().filter_map(|v| v).sum();
        
        if total > 0 {
            // Calculate percentages
            let percentages: Vec<f64> = values
                .iter()
                .filter_map(|v| v.map(|val| (val as f64 / total as f64) * 100.0))
                .collect();
            
            // Add a new column with percentages
            result.add_float_column("value_pct", percentages)?;
        }
        
        Ok(result)
    };
    
    benchmark_comparison(
        "Transform groups",
        || grouped.transform(transform_func),
        || grouped.par_transform(transform_func)
    )?;
    
    Ok(())
}

/// Create a large dataset for benchmarking
fn create_large_dataset(size: usize) -> Result<OptimizedDataFrame> {
    use rand::{Rng, thread_rng, rngs::StdRng, SeedableRng};
    
    // Use deterministic RNG for reproducible results
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Create categories
    let categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
    
    // Generate data
    let mut cat_vec = Vec::with_capacity(size);
    let mut val_vec = Vec::with_capacity(size);
    
    for _ in 0..size {
        let cat_idx = rng.gen_range(0..categories.len());
        cat_vec.push(categories[cat_idx]);
        
        let val = rng.gen_range(0..100);
        val_vec.push(val);
    }
    
    // Create dataframe
    let mut df = OptimizedDataFrame::new();
    df.add_string_column("category", cat_vec)?;
    df.add_int_column("value", val_vec)?;
    
    Ok(df)
}

/// Benchmark a single operation with timing
fn benchmark_operation<F, T>(name: &str, op: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    let start = Instant::now();
    let result = op()?;
    let elapsed = start.elapsed();
    
    println!("{}: {:.3?}", name, elapsed);
    
    Ok(result)
}

/// Compare serial and parallel versions of an operation
fn benchmark_comparison<F1, F2, T>(name: &str, serial_op: F1, parallel_op: F2) -> Result<()>
where
    F1: FnOnce() -> Result<T>,
    F2: FnOnce() -> Result<T>,
{
    // Warm up
    let _ = serial_op()?;
    let _ = parallel_op()?;
    
    // Time serial operation
    let serial_start = Instant::now();
    let _ = serial_op()?;
    let serial_elapsed = serial_start.elapsed();
    
    // Time parallel operation
    let parallel_start = Instant::now();
    let _ = parallel_op()?;
    let parallel_elapsed = parallel_start.elapsed();
    
    // Calculate speedup
    let speedup = if parallel_elapsed.as_secs_f64() > 0.0 {
        serial_elapsed.as_secs_f64() / parallel_elapsed.as_secs_f64()
    } else {
        0.0
    };
    
    println!("{}: Serial {:.3?}, Parallel {:.3?}, Speedup {:.2}x",
        name, serial_elapsed, parallel_elapsed, speedup);
    
    Ok(())
}
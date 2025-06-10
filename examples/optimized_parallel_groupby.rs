use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;
use std::time::Instant;

fn main() -> Result<()> {
    println!("PandRS Parallel GroupBy Operations Example");
    println!("=========================================");

    // Create a dataset for demonstration
    let df = create_dataset(10_000)?;

    println!("Created dataset with {} rows", df.row_count());

    // Group by category using par_groupby
    let start = Instant::now();
    let grouped = df.par_groupby(&["category"])?;
    let duration = start.elapsed();

    println!("Group by category took: {:?}", duration);
    println!("Number of groups: {}", grouped.len());

    // Display information about each group
    println!("\n1. Group Information");
    println!("-------------------");

    for (key, group_df) in &grouped {
        println!("Group '{}': {} rows", key, group_df.row_count());

        // Calculate some statistics using the new aggregate methods
        if let Ok(sum) = group_df.sum("value") {
            println!("  Sum of values: {:.2}", sum);
        }
        if let Ok(mean) = group_df.mean("value") {
            println!("  Mean of values: {:.2}", mean);
        }
        if let Ok(max) = group_df.max("value") {
            println!("  Max value: {:.2}", max);
        }
        if let Ok(min) = group_df.min("value") {
            println!("  Min value: {:.2}", min);
        }
    }

    // Demonstrate different aggregation operations
    println!("\n2. Aggregation Operations");
    println!("------------------------");

    benchmark_operation("Sum operation", || {
        for (_, group_df) in &grouped {
            let _ = group_df.sum("value");
        }
        Ok(())
    })?;

    benchmark_operation("Mean operation", || {
        for (_, group_df) in &grouped {
            let _ = group_df.mean("value");
        }
        Ok(())
    })?;

    benchmark_operation("Min/Max operations", || {
        for (_, group_df) in &grouped {
            let _ = group_df.min("value");
            let _ = group_df.max("value");
        }
        Ok(())
    })?;

    println!("\nExample completed successfully!");

    Ok(())
}

fn create_dataset(size: usize) -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    // Create sample data
    let categories = ["A", "B", "C", "D"];
    let mut cat_vec = Vec::with_capacity(size);
    let mut val_vec = Vec::with_capacity(size);

    for i in 0..size {
        let cat_idx = i % categories.len();
        cat_vec.push(categories[cat_idx].to_string());
        val_vec.push((i % 100) as i64);
    }

    df.add_string_column("category", cat_vec)?;
    df.add_int_column("value", val_vec)?;

    Ok(df)
}

fn benchmark_operation<F>(name: &str, mut op: F) -> Result<()>
where
    F: FnMut() -> Result<()>,
{
    println!("\nBenchmarking: {}", name);

    // Warm up
    let _ = op()?;

    let start = Instant::now();
    op()?;
    let duration = start.elapsed();

    println!("  Time: {:?}", duration);

    Ok(())
}

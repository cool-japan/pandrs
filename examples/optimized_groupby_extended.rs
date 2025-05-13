use pandrs::error::Result;
use pandrs::optimized::{AggregateOp, OptimizedDataFrame};

fn main() -> Result<()> {
    println!("PandRS GroupBy Extended Example");
    println!("===============================");

    // Create a sample DataFrame with various data
    let mut df = OptimizedDataFrame::new();
    
    // Add columns
    let categories = vec![
        "A", "B", "A", "B", "A", "C", "B", "C", "C", "A",
    ];
    df.add_string_column("category", categories)?;
    
    let values = vec![
        10, 25, 15, 30, 22, 18, 24, 12, 16, 20,
    ];
    df.add_int_column("value", values)?;
    
    let prices = vec![
        110.5, 225.2, 115.8, 130.4, 222.1, 118.9, 324.5, 112.3, 156.7, 120.9,
    ];
    df.add_float_column("price", prices)?;
    
    // Display the DataFrame
    println!("Original DataFrame:");
    println!("{:?}", df);

    // Group by category
    let grouped = df.group_by(["category"])?;
    
    // Basic aggregation operations
    println!("\n1. Basic Aggregation Operations");
    println!("------------------------------");
    let sum_result = grouped.sum("value")?;
    println!("Sum by category:");
    println!("{:?}", sum_result);
    
    let mean_result = grouped.mean("value")?;
    println!("\nMean by category:");
    println!("{:?}", mean_result);
    
    // Advanced aggregation operations
    println!("\n2. Advanced Aggregation Operations");
    println!("---------------------------------");
    let std_result = grouped.std("value")?;
    println!("Standard deviation by category:");
    println!("{:?}", std_result);
    
    let median_result = grouped.median("price")?;
    println!("\nMedian price by category:");
    println!("{:?}", median_result);
    
    // Multiple aggregations with agg
    println!("\n3. Multiple Aggregations with agg()");
    println!("----------------------------------");
    let agg_result = grouped.agg(&[
        ("value", AggregateOp::Mean),
        ("value", AggregateOp::Std),
        ("price", AggregateOp::Sum),
        ("price", AggregateOp::Max),
    ])?;
    println!("Multiple aggregations:");
    println!("{:?}", agg_result);
    
    // Filter groups
    println!("\n4. Filtering Groups");
    println!("------------------");
    let filtered = grouped.filter(|group_df| {
        // Filter groups where the mean value is greater than 20
        let values = group_df.get_int_column("value").unwrap();
        let sum: i64 = values.iter().filter_map(|v| v).sum();
        let count = values.iter().filter_map(|v| v).count();
        
        if count == 0 {
            false
        } else {
            (sum as f64 / count as f64) > 20.0
        }
    })?;
    println!("Filtered groups with mean value > 20:");
    println!("{:?}", filtered);
    
    // Transform groups
    println!("\n5. Transforming Groups");
    println!("---------------------");
    let transformed = grouped.transform(|group_df| {
        // Calculate percentage of value relative to group total
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
    })?;
    println!("Transformed groups with percentage column:");
    println!("{:?}", transformed);
    
    Ok(())
}
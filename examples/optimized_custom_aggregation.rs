use pandrs::error::Result;
use pandrs::optimized::OptimizedDataFrame;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Custom Aggregation Example");
    println!("=================================");

    // Create a sample DataFrame with various data
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories: Vec<String> = ["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    df.add_string_column("category", categories)?;

    let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
    df.add_int_column("value", values)?;

    // Display the DataFrame
    println!("Original DataFrame:");
    println!("{:?}", df);

    // For now, just demonstrate basic DataFrame operations
    // Note: Custom aggregation and group_by functionality would need to be implemented
    println!(
        "\nDataFrame has {} rows and {} columns",
        df.row_count(),
        df.column_count()
    );

    // Show column operations
    if let Ok(int_values) = df.get_int_column("value") {
        let sum: i64 = int_values.iter().filter_map(|v| *v).sum();
        let count = int_values.iter().filter_map(|v| *v).count();
        let mean = sum as f64 / count as f64;

        println!("Basic statistics for 'value' column:");
        println!("  Sum: {}", sum);
        println!("  Count: {}", count);
        println!("  Mean: {:.2}", mean);
    }

    Ok(())
}

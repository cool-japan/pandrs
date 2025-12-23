use pandrs::error::Result;
use pandrs::pivot::AggFunction;
use pandrs::{DataFrame, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Pivot Table and Grouping Example ===");

    // Create sample data
    let mut df = DataFrame::new();

    // Create column data
    let category = Series::new(
        vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "B".to_string(),
            "A".to_string(),
            "C".to_string(),
            "B".to_string(),
        ],
        Some("category".to_string()),
    )?;

    let region = Series::new(
        vec![
            "East".to_string(),
            "West".to_string(),
            "West".to_string(),
            "East".to_string(),
            "East".to_string(),
            "West".to_string(),
            "West".to_string(),
            "East".to_string(),
        ],
        Some("region".to_string()),
    )?;

    let sales = Series::new(
        vec![100, 150, 200, 120, 180, 90, 250, 160],
        Some("sales".to_string()),
    )?;

    // Add columns to DataFrame
    df.add_column("category".to_string(), category)?;
    df.add_column("region".to_string(), region)?;
    df.add_column("sales".to_string(), sales)?;

    println!("DataFrame Info:");
    println!("  Number of columns: {}", df.column_count());
    println!("  Number of rows: {}", df.row_count());
    println!("  Column names: {:?}", df.column_names());

    // Grouping and aggregation
    println!("\n=== Grouping by Category ===");
    let category_group = df.groupby("category")?;

    println!("Sum by category (in progress):");
    let _category_sum = category_group.sum(&["sales"])?;

    // Pivot table (in progress)
    println!("\n=== Pivot Table ===");
    println!("Sum of sales by category and region (in progress):");
    let _pivot_result = df.pivot_table("category", "region", "sales", AggFunction::Sum)?;

    // Note: Pivot table and grouping features are still under development,
    // so actual results are not displayed

    println!("\n=== Aggregation Function Examples ===");
    let functions = [
        AggFunction::Sum,
        AggFunction::Mean,
        AggFunction::Min,
        AggFunction::Max,
        AggFunction::Count,
    ];

    for func in &functions {
        println!(
            "Aggregation Function: {} ({})",
            func.name(),
            match func {
                AggFunction::Sum => "Sum",
                AggFunction::Mean => "Mean",
                AggFunction::Min => "Min",
                AggFunction::Max => "Max",
                AggFunction::Count => "Count",
            }
        );
    }

    println!("\n=== Pivot Table Example (Complete) ===");
    Ok(())
}

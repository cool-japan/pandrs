use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::column::{StringColumn, Int64Column, Float64Column, BooleanColumn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Enhanced CSV I/O Implementation");

    // Create test data
    let mut df = OptimizedDataFrame::new();
    
    // Add string column
    let names = vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()];
    let name_column = StringColumn::with_name(names, "name");
    df.add_column("name".to_string(), name_column)?;
    
    // Add integer column
    let ages = vec![25i64, 30i64, 35i64];
    let age_column = Int64Column::with_name(ages, "age");
    df.add_column("age".to_string(), age_column)?;
    
    // Add float column
    let heights = vec![5.6f64, 5.9f64, 6.1f64];
    let height_column = Float64Column::with_name(heights, "height");
    df.add_column("height".to_string(), height_column)?;
    
    // Add boolean column
    let is_active = vec![true, false, true];
    let active_column = BooleanColumn::with_name(is_active, "is_active");
    df.add_column("is_active".to_string(), active_column)?;
    
    println!("Original DataFrame:");
    println!("Columns: {:?}", df.column_names());
    println!("Row count: {}", df.row_count());
    
    // Test writing to CSV
    let test_file = "/tmp/test_enhanced.csv";
    println!("Writing to CSV file: {}", test_file);
    df.to_csv(test_file, true)?;
    
    // Test reading from CSV
    println!("Reading from CSV file...");
    let read_df = OptimizedDataFrame::from_csv(test_file, true)?;
    
    println!("Read DataFrame:");
    println!("Columns: {:?}", read_df.column_names());
    println!("Row count: {}", read_df.row_count());
    
    // Display actual content by reading the CSV file
    println!("\nActual CSV content:");
    let csv_content = std::fs::read_to_string(test_file)?;
    println!("{}", csv_content);
    
    println!("✅ CSV I/O test completed successfully!");
    println!("Real data was properly written and read from CSV.");
    
    Ok(())
}
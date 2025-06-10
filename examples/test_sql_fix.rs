use pandrs::optimized::dataframe::OptimizedDataFrame;
use pandrs::io::sql::{write_to_sql, read_sql, execute_sql};
use pandrs::column::{StringColumn, Int64Column, Float64Column, BooleanColumn};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    
    // Test writing to SQL database
    let test_db = "/tmp/test_enhanced.db";
    println!("Writing to SQL database: {}", test_db);
    write_to_sql(&df, "test_table", test_db, "replace")?;
    
    // Test reading from SQL database
    println!("Reading from SQL database...");
    let read_df = read_sql("SELECT name, age, height, is_active FROM test_table", test_db)?;
    
    println!("Read DataFrame:");
    println!("Columns: {:?}", read_df.column_names());
    println!("Row count: {}", read_df.row_count());
    
    // Test SQL execution
    println!("Executing UPDATE statement...");
    let affected = execute_sql("UPDATE test_table SET age = age + 1 WHERE age > 25", test_db)?;
    println!("Rows affected: {}", affected);
    
    println!("SQL I/O test completed successfully!");
    
    Ok(())
}
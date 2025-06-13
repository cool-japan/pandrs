#[cfg(feature = "sql")]
use pandrs::io::sql::{write_to_sql, read_sql, execute_sql};
#[allow(unused_imports)]
use pandrs::optimized::OptimizedDataFrame;

#[cfg(feature = "sql")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create test data
    let mut df = OptimizedDataFrame::new();
    
    // Add columns using the simplified API
    df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string(), "Charlie".to_string()])?;
    df.add_int_column("age", vec![25i64, 30i64, 35i64])?;
    df.add_float_column("height", vec![5.6f64, 5.9f64, 6.1f64])?;
    df.add_bool_column("is_active", vec![true, false, true])?;
    
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

#[cfg(not(feature = "sql"))]
fn main() {
    println!("SQL feature not enabled. Please run with `cargo run --example test_sql_fix --features sql`");
}
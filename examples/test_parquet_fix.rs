#[allow(unused_imports)]
use pandrs::optimized::OptimizedDataFrame;

#[cfg(feature = "parquet")]
use pandrs::io::{write_parquet, read_parquet, ParquetCompression};

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "parquet"))]
    {
        println!("Parquet feature is not enabled. Enable it with --features parquet");
        return Ok(());
    }
    
    #[cfg(feature = "parquet")]
    {
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
        
        // Test writing to Parquet
        let test_file = "/tmp/test_parquet_enhanced.parquet";
        println!("Writing to Parquet file: {}", test_file);
        write_parquet(&df, test_file, Some(ParquetCompression::Snappy))?;
        
        // Test reading from Parquet
        println!("Reading from Parquet file...");
        let read_df = read_parquet(test_file)?;
        
        println!("Read DataFrame:");
        println!("Columns: {:?}", read_df.column_names());
        println!("Row count: {}", read_df.row_count());
        
        println!("Parquet I/O test completed successfully!");
    }
    
    Ok(())
}
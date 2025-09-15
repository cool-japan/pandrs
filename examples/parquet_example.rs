use pandrs::{DataFrame, Series};
use std::error::Error;

#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn Error>> {
    // Create a sample DataFrame
    let mut df = DataFrame::new();

    // Add an integer column
    let int_data = Series::new(vec![1, 2, 3, 4, 5], Some("id".to_string()))?;
    df.add_column("id".to_string(), int_data)?;

    // Add a floating-point column
    let float_data = Series::new(vec![1.1, 2.2, 3.3, 4.4, 5.5], Some("value".to_string()))?;
    df.add_column("value".to_string(), float_data)?;

    // Add a string column
    let string_data = Series::new(
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
            "E".to_string(),
        ],
        Some("category".to_string()),
    )?;
    df.add_column("category".to_string(), string_data)?;

    println!("Original DataFrame:");
    println!("{df:?}");

    // Parquet support is still under development
    println!("\nNote: Parquet support is currently under development.");
    println!("It is planned to be available in a future release.");

    /*
    // Although Parquet functionality is not yet implemented, dependencies have been introduced.
    // The following code is expected to work in a future version.

    // Write the DataFrame to a Parquet file
    let path = "example.parquet";
    match write_parquet(&df, path, Some(ParquetCompression::Snappy)) {
        Ok(_) => {
            println!("DataFrame written to {}", path);

            // Read the DataFrame from the Parquet file
            match read_parquet(path) {
                Ok(df_read) => {
                    println!("\nDataFrame read from Parquet file:");
                    println!("{:?}", df_read);

                    // Verify the results
                    assert_eq!(df.row_count(), df_read.row_count());
                    assert_eq!(df.column_count(), df_read.column_count());

                    println!("\nVerification successful: Data matches");
                },
                Err(e) => println!("Error reading Parquet file: {}", e),
            }
        },
        Err(e) => println!("Error writing Parquet file: {}", e),
    }
    */

    Ok(())
}

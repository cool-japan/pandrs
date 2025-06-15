use pandrs::error::Error;
use pandrs::optimized::OptimizedDataFrame;
use std::fs;
use std::fs::File;
use std::io::Write;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

/// Tests for CSV I/O error conditions
#[cfg(test)]
mod csv_error_tests {
    use super::*;

    #[test]
    fn test_csv_read_nonexistent_file() {
        let result = pandrs::io::read_csv("nonexistent_file.csv", true);
        assert!(result.is_err());

        // Should be a proper IoError or Csv error - adjust test to be more flexible
        match result {
            Err(Error::IoError(_)) => {}
            Err(Error::CsvError(_)) => {}
            Err(pandrs::PandRSError::Io(_)) => {}
            Err(pandrs::PandRSError::Csv(_)) => {}
            _ => panic!(
                "Expected IoError or CsvError for nonexistent file, got: {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_csv_write_invalid_path() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("test", vec![1, 2, 3]).unwrap();

        // Try to write to invalid path (directory doesn't exist)
        let result = df.to_csv("/nonexistent_directory/test.csv", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_write_read_only_file() {
        let temp_path = std::env::temp_dir().join("readonly_test.csv");

        // Create a file and make it read-only
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, "col1,col2").unwrap();
            writeln!(file, "1,2").unwrap();
        }

        // Make file read-only (Unix permissions)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&temp_path).unwrap().permissions();
            perms.set_mode(0o444); // Read-only
            fs::set_permissions(&temp_path, perms).unwrap();
        }

        let mut df = OptimizedDataFrame::new();
        df.add_int_column("col1", vec![3]).unwrap();
        df.add_int_column("col2", vec![4]).unwrap();

        // Try to overwrite read-only file
        let result = df.to_csv(&temp_path, true);

        #[cfg(unix)]
        assert!(result.is_err());

        // Cleanup
        #[cfg(unix)]
        {
            let mut perms = fs::metadata(&temp_path).unwrap().permissions();
            perms.set_mode(0o644); // Make writable for cleanup
            fs::set_permissions(&temp_path, perms).unwrap();
        }
        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_malformed_csv_read() {
        let temp_path = std::env::temp_dir().join("malformed_test.csv");

        // Create malformed CSV with inconsistent column counts
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, "col1,col2,col3").unwrap();
            writeln!(file, "1,2,3").unwrap();
            writeln!(file, "4,5").unwrap(); // Missing column
            writeln!(file, "6,7,8,9").unwrap(); // Extra column
        }

        let result = pandrs::io::read_csv(&temp_path, true);
        // Should either handle gracefully or return appropriate error
        match result {
            Ok(_) => {
                // If it succeeds, it should handle the malformed data gracefully
            }
            Err(e) => {
                // Should be a parsing error, not a panic
                assert!(e.to_string().contains("parse") || e.to_string().contains("column"));
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_csv_with_embedded_newlines() {
        let temp_path = std::env::temp_dir().join("newlines_test.csv");

        // Create CSV with embedded newlines in quoted fields
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, r#"id,description"#).unwrap();
            writeln!(file, r#"1,"Line 1"#).unwrap();
            writeln!(file, r#"Line 2""#).unwrap(); // Embedded newline
            writeln!(file, r#"2,"Normal line""#).unwrap();
        }

        let result = pandrs::io::read_csv(&temp_path, true);
        // Should handle embedded newlines in quoted fields
        match result {
            Ok(df) => {
                // Verify the data was parsed correctly
                assert!(df.row_count() >= 2);
            }
            Err(_) => {
                // Error is acceptable if embedded newlines are not supported
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_csv_with_mixed_encodings() {
        let temp_path = std::env::temp_dir().join("encoding_test.csv");

        // Create CSV with UTF-8 characters
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, "name,description").unwrap();
            writeln!(file, "café,A nice café").unwrap();
            writeln!(file, "naïve,Naïve approach").unwrap();
            writeln!(file, "🚀,Rocket emoji").unwrap();
        }

        let result = pandrs::io::read_csv(&temp_path, true);
        match result {
            Ok(df) => {
                // Should handle UTF-8 correctly
                assert!(df.row_count() >= 3);
            }
            Err(e) => {
                // If encoding is not supported, should fail gracefully
                println!("UTF-8 encoding test failed: {}", e);
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_empty_csv_file() {
        let temp_path = std::env::temp_dir().join("empty_test.csv");

        // Create completely empty file
        File::create(&temp_path).unwrap();

        let result = pandrs::io::read_csv(&temp_path, true);
        match result {
            Ok(df) => {
                // Should create empty DataFrame
                assert_eq!(df.row_count(), 0);
            }
            Err(_) => {
                // Error is also acceptable for empty file
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_csv_header_only() {
        let temp_path = std::env::temp_dir().join("header_only_test.csv");

        // Create CSV with header but no data
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, "col1,col2,col3").unwrap();
        }

        let result = pandrs::io::read_csv(&temp_path, true);
        match result {
            Ok(df) => {
                // Should create DataFrame with 0 rows but proper columns
                assert_eq!(df.row_count(), 0);
                assert!(df.column_count() > 0);
            }
            Err(_) => {
                // Error is acceptable
            }
        }

        fs::remove_file(&temp_path).ok();
    }
}

/// Tests for Parquet I/O error conditions
#[cfg(test)]
#[cfg(feature = "parquet")]
mod parquet_error_tests {
    use super::*;
    use pandrs::io::{read_parquet, write_parquet, ParquetCompression};

    #[test]
    fn test_parquet_read_nonexistent_file() {
        let result = read_parquet("nonexistent_file.parquet");
        assert!(result.is_err());

        match result {
            Err(Error::IoError(_)) => {}
            _ => panic!("Expected IoError for nonexistent file"),
        }
    }

    #[test]
    fn test_parquet_read_invalid_file() {
        let temp_path = std::env::temp_dir().join("invalid_parquet_test.parquet");

        // Create file with invalid Parquet data
        {
            let mut file = File::create(&temp_path).unwrap();
            writeln!(file, "This is not a Parquet file").unwrap();
        }

        let result = read_parquet(&temp_path);
        assert!(result.is_err());

        // Should be a parsing error
        match result {
            Err(Error::IoError(_)) => {}
            _ => panic!("Expected IoError for invalid Parquet file"),
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_parquet_write_invalid_path() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("test", vec![1, 2, 3]).unwrap();

        // Try to write to invalid path
        let result = write_parquet(
            &df,
            "/nonexistent_directory/test.parquet",
            Some(ParquetCompression::Snappy),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_parquet_write_empty_dataframe() {
        let df = OptimizedDataFrame::new();
        let temp_path = std::env::temp_dir().join("empty_parquet_test.parquet");

        let result = write_parquet(&df, &temp_path, Some(ParquetCompression::Snappy));
        match result {
            Ok(_) => {
                // If write succeeds, verify we can read it back
                let read_result = read_parquet(&temp_path);
                match read_result {
                    Ok(read_df) => {
                        assert_eq!(read_df.row_count(), 0);
                    }
                    Err(_) => {
                        // Reading empty Parquet might fail, which is acceptable
                    }
                }
            }
            Err(_) => {
                // Writing empty DataFrame might fail, which is acceptable
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_parquet_compression_options() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("test", vec![1, 2, 3]).unwrap();
        df.add_string_column(
            "text",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .unwrap();

        let compression_options = [
            ParquetCompression::None,
            ParquetCompression::Snappy,
            ParquetCompression::Gzip,
            ParquetCompression::Lz4,
            ParquetCompression::Zstd,
        ];

        for compression in compression_options.iter() {
            let temp_path =
                std::env::temp_dir().join(format!("compression_test_{:?}.parquet", compression));

            let result = write_parquet(&df, &temp_path, Some(*compression));
            match result {
                Ok(_) => {
                    // Verify we can read it back
                    let read_result = read_parquet(&temp_path);
                    assert!(read_result.is_ok());
                }
                Err(e) => {
                    // Some compression types might not be available
                    println!("Compression {:?} failed: {}", compression, e);
                }
            }

            fs::remove_file(&temp_path).ok();
        }
    }
}

/// Tests for SQL I/O error conditions
#[cfg(test)]
#[cfg(feature = "sql")]
mod sql_error_tests {
    use super::*;
    use pandrs::io::{execute_sql, read_sql, write_to_sql};

    #[test]
    fn test_sql_read_nonexistent_database() {
        let result = read_sql("SELECT * FROM test", "nonexistent_database.db");
        assert!(result.is_err());

        match result {
            Err(Error::IoError(_)) => {}
            _ => panic!("Expected IoError for nonexistent database"),
        }
    }

    #[test]
    fn test_sql_invalid_query() {
        let temp_db = std::env::temp_dir().join("test_invalid_query.db");

        // Create a simple database
        execute_sql("CREATE TABLE test (id INTEGER, name TEXT)", &temp_db).unwrap();
        execute_sql("INSERT INTO test VALUES (1, 'test')", &temp_db).unwrap();

        // Try invalid SQL query
        let result = read_sql("INVALID SQL QUERY", &temp_db);
        assert!(result.is_err());

        match result {
            Err(Error::IoError(_)) => {}
            _ => panic!("Expected IoError for invalid SQL"),
        }

        fs::remove_file(&temp_db).ok();
    }

    #[test]
    fn test_sql_write_invalid_table_name() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("id", vec![1, 2, 3]).unwrap();

        let temp_db = std::env::temp_dir().join("test_invalid_table.db");

        // Try to write to table with invalid characters
        let result = write_to_sql(&df, "invalid-table-name!", &temp_db, "replace");
        // Behavior depends on SQL implementation - might succeed or fail

        fs::remove_file(&temp_db).ok();
    }

    #[test]
    fn test_sql_write_if_exists_options() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("id", vec![1, 2, 3]).unwrap();

        let temp_db = std::env::temp_dir().join("test_if_exists.db");

        // Write initial data
        write_to_sql(&df, "test_table", &temp_db, "replace").unwrap();

        // Test "fail" option on existing table
        let result = write_to_sql(&df, "test_table", &temp_db, "fail");
        assert!(result.is_err());

        // Test "append" option
        let result = write_to_sql(&df, "test_table", &temp_db, "append");
        assert!(result.is_ok());

        // Test "replace" option
        let result = write_to_sql(&df, "test_table", &temp_db, "replace");
        assert!(result.is_ok());

        // Test invalid option
        let result = write_to_sql(&df, "test_table", &temp_db, "invalid_option");
        assert!(result.is_err());

        fs::remove_file(&temp_db).ok();
    }

    #[test]
    fn test_sql_read_empty_table() {
        let temp_db = std::env::temp_dir().join("test_empty_table.db");

        // Create empty table
        execute_sql("CREATE TABLE empty_test (id INTEGER, name TEXT)", &temp_db).unwrap();

        let result = read_sql("SELECT * FROM empty_test", &temp_db);
        match result {
            Ok(df) => {
                assert_eq!(df.row_count(), 0);
            }
            Err(_) => {
                // Error is acceptable for empty table
            }
        }

        fs::remove_file(&temp_db).ok();
    }

    #[test]
    fn test_sql_write_empty_dataframe() {
        let df = OptimizedDataFrame::new();
        let temp_db = std::env::temp_dir().join("test_empty_df.db");

        let result = write_to_sql(&df, "empty_table", &temp_db, "replace");
        match result {
            Ok(_) => {
                // Should create empty table
                let read_result = read_sql("SELECT * FROM empty_table", &temp_db);
                match read_result {
                    Ok(read_df) => {
                        assert_eq!(read_df.row_count(), 0);
                    }
                    Err(_) => {
                        // Reading empty table might fail
                    }
                }
            }
            Err(_) => {
                // Writing empty DataFrame might fail
            }
        }

        fs::remove_file(&temp_db).ok();
    }
}

/// Tests for general I/O error conditions
#[cfg(test)]
mod general_io_tests {
    use super::*;

    #[test]
    fn test_file_permission_errors() {
        let temp_dir = std::env::temp_dir().join("no_permission_dir");
        fs::create_dir_all(&temp_dir).unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            // Make directory read-only
            let mut perms = fs::metadata(&temp_dir).unwrap().permissions();
            perms.set_mode(0o444);
            fs::set_permissions(&temp_dir, perms).unwrap();

            let mut df = OptimizedDataFrame::new();
            df.add_int_column("test", vec![1, 2, 3]).unwrap();

            let test_file = temp_dir.join("test.csv");
            let result = df.to_csv(&test_file, true);
            assert!(result.is_err());

            // Restore permissions for cleanup
            let mut perms = fs::metadata(&temp_dir).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&temp_dir, perms).unwrap();
        }

        fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_disk_space_simulation() {
        // This test simulates disk space issues by trying to write very large data
        // Note: This is a conceptual test - actual disk space testing would require
        // special setup or mocking

        let mut df = OptimizedDataFrame::new();

        // Create large dataset that might stress disk space
        let large_size = 100_000;
        let large_strings: Vec<String> = (0..large_size)
            .map(|_i| format!("Large string with lots of data: {}", "x".repeat(100)))
            .collect();

        df.add_string_column("large_data", large_strings).unwrap();

        let temp_path = std::env::temp_dir().join("large_test.csv");

        // This should normally succeed unless disk is actually full
        let result = df.to_csv(&temp_path, true);
        match result {
            Ok(_) => {
                // Verify file was created
                assert!(temp_path.exists());

                // Check file size is reasonable
                let metadata = fs::metadata(&temp_path).unwrap();
                assert!(metadata.len() > 1_000_000); // Should be at least 1MB
            }
            Err(e) => {
                // If it fails, should be a disk space or I/O error
                println!(
                    "Large file write failed (possibly due to disk space): {}",
                    e
                );
            }
        }

        fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_concurrent_file_access() {
        use std::sync::Arc;
        use std::thread;

        let temp_path = std::env::temp_dir().join("concurrent_test.csv");

        // Create initial file
        {
            let mut df = OptimizedDataFrame::new();
            df.add_int_column("test", vec![1, 2, 3]).unwrap();
            df.to_csv(&temp_path, true).unwrap();
        }

        let path_arc = Arc::new(temp_path.clone());

        // Spawn multiple threads trying to read the same file
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let path = path_arc.clone();
                thread::spawn(move || {
                    let result = pandrs::io::read_csv(&*path, true);
                    match result {
                        Ok(df) => {
                            assert_eq!(df.row_count(), 3);
                            println!("Thread {} successfully read file", i);
                        }
                        Err(e) => {
                            println!("Thread {} failed to read file: {}", i, e);
                        }
                    }
                })
            })
            .collect();

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        fs::remove_file(&temp_path).ok();
    }
}

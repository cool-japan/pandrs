//! Integration tests for temporary file utilities

mod common;

use common::test_utils::*;
use std::fs::{self, File};
use std::io::Write;

#[test]
fn test_get_temp_dir_works() {
    let temp_dir = get_temp_dir();
    assert!(temp_dir.exists());
    assert!(temp_dir.is_dir());
}

#[test]
fn test_temp_path_unique() {
    let path1 = test_temp_path("test", "csv");
    let path2 = test_temp_path("test", "csv");
    assert_ne!(path1, path2, "Paths should be unique");
}

#[test]
fn test_temp_test_file_cleanup() {
    let test_name = "temp_file_test";
    let path;
    {
        let temp_file = TempTestFile::new(test_name, "txt");
        path = temp_file.path().to_path_buf();

        // Create the file
        let mut file = File::create(temp_file.path()).unwrap();
        writeln!(file, "test data").unwrap();

        assert!(
            path.exists(),
            "File should exist while TempTestFile is in scope"
        );
    }
    // File should be cleaned up after drop
    assert!(
        !path.exists(),
        "File should be deleted after TempTestFile is dropped"
    );
}

#[test]
fn test_temp_test_file_keep() {
    let test_name = "temp_file_keep_test";
    let path;
    {
        let mut temp_file = TempTestFile::new(test_name, "txt");
        path = temp_file.path().to_path_buf();

        // Create the file
        let mut file = File::create(temp_file.path()).unwrap();
        writeln!(file, "test data").unwrap();

        temp_file.keep(); // Mark to keep
    }
    // File should still exist
    assert!(path.exists(), "File should be kept after drop with keep()");

    // Manual cleanup
    fs::remove_file(&path).unwrap();
}

#[test]
fn test_temp_test_dir_cleanup() {
    let test_name = "temp_dir_test";
    let path;
    {
        let temp_dir = TempTestDir::new(test_name).unwrap();
        path = temp_dir.path().to_path_buf();

        // Create a file in the directory
        let file_path = temp_dir.path().join("test.txt");
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "test data").unwrap();

        assert!(path.exists(), "Directory should exist");
        assert!(file_path.exists(), "File in directory should exist");
    }
    // Directory and contents should be cleaned up
    assert!(
        !path.exists(),
        "Directory should be deleted after TempTestDir is dropped"
    );
}

#[test]
fn test_create_test_csv_helper() {
    let headers = vec!["col1", "col2", "col3"];
    let rows = vec![
        vec!["1".to_string(), "2".to_string(), "3".to_string()],
        vec!["4".to_string(), "5".to_string(), "6".to_string()],
    ];

    let path;
    {
        let temp_file = create_test_csv("csv_test", &headers, &rows);
        path = temp_file.path().to_path_buf();
        assert!(path.exists());

        // Verify contents
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("col1,col2,col3"));
        assert!(content.contains("1,2,3"));
        assert!(content.contains("4,5,6"));
    }
    // File should be cleaned up
    assert!(!path.exists());
}

#[test]
fn test_cleanup_functions() {
    // Test file cleanup
    let file_path = test_temp_path("cleanup_test", "txt");
    {
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "test").unwrap();
    }
    assert!(file_path.exists());
    cleanup_test_file(&file_path);
    assert!(!file_path.exists());

    // Test directory cleanup
    let dir_path = test_temp_dir("cleanup_dir_test");
    fs::create_dir_all(&dir_path).unwrap();
    assert!(dir_path.exists());
    cleanup_test_dir(&dir_path);
    assert!(!dir_path.exists());
}

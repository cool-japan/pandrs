//! Test utilities for temporary file handling
//!
//! Provides consistent temporary file and directory management with automatic cleanup
//! and support for environment variables (TMPDIR, TEMP, TMP).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Get the temporary directory, respecting environment variables
///
/// Checks in order: TMPDIR, TEMP, TMP, then falls back to std::env::temp_dir()
pub fn get_temp_dir() -> PathBuf {
    env::var("TMPDIR")
        .or_else(|_| env::var("TEMP"))
        .or_else(|_| env::var("TMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::temp_dir())
}

/// Generate a unique test file path
///
/// Creates a unique filename based on test name and counter to avoid collisions
pub fn test_temp_path(test_name: &str, extension: &str) -> PathBuf {
    let counter = TEST_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let filename = format!("pandrs_test_{}_{}.{}", test_name, counter, extension);
    get_temp_dir().join(filename)
}

/// Generate a unique test directory path
pub fn test_temp_dir(test_name: &str) -> PathBuf {
    let counter = TEST_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dirname = format!("pandrs_test_dir_{}_{}", test_name, counter);
    get_temp_dir().join(dirname)
}

/// RAII wrapper for temporary test files with automatic cleanup
///
/// The file is automatically deleted when this struct is dropped
pub struct TempTestFile {
    path: PathBuf,
    keep: bool,
}

impl TempTestFile {
    /// Create a new temporary test file
    pub fn new(test_name: &str, extension: &str) -> Self {
        let path = test_temp_path(test_name, extension);
        TempTestFile { path, keep: false }
    }

    /// Create from an existing path
    pub fn from_path(path: PathBuf) -> Self {
        TempTestFile { path, keep: false }
    }

    /// Get the path to the temporary file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Keep the file after drop (for debugging)
    pub fn keep(&mut self) {
        self.keep = true;
    }

    /// Convert to PathBuf, consuming self and disabling cleanup
    pub fn into_path(mut self) -> PathBuf {
        self.keep = true;
        self.path.clone()
    }
}

impl Drop for TempTestFile {
    fn drop(&mut self) {
        if !self.keep && self.path.exists() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// RAII wrapper for temporary test directories with automatic cleanup
///
/// The directory and its contents are automatically deleted when this struct is dropped
pub struct TempTestDir {
    path: PathBuf,
    keep: bool,
}

impl TempTestDir {
    /// Create a new temporary test directory
    pub fn new(test_name: &str) -> std::io::Result<Self> {
        let path = test_temp_dir(test_name);
        fs::create_dir_all(&path)?;
        Ok(TempTestDir { path, keep: false })
    }

    /// Create from an existing path
    pub fn from_path(path: PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(&path)?;
        Ok(TempTestDir { path, keep: false })
    }

    /// Get the path to the temporary directory
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Keep the directory after drop (for debugging)
    pub fn keep(&mut self) {
        self.keep = true;
    }

    /// Convert to PathBuf, consuming self and disabling cleanup
    pub fn into_path(mut self) -> PathBuf {
        self.keep = true;
        self.path.clone()
    }
}

impl Drop for TempTestDir {
    fn drop(&mut self) {
        if !self.keep && self.path.exists() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// Helper to create a test CSV file with given data
pub fn create_test_csv(test_name: &str, headers: &[&str], rows: &[Vec<String>]) -> TempTestFile {
    use std::fs::File;
    use std::io::Write;

    let temp_file = TempTestFile::new(test_name, "csv");
    let mut file = File::create(temp_file.path()).expect("Failed to create test CSV");

    // Write headers
    writeln!(file, "{}", headers.join(",")).expect("Failed to write headers");

    // Write rows
    for row in rows {
        writeln!(file, "{}", row.join(",")).expect("Failed to write row");
    }

    temp_file
}

/// Cleanup function for tests that create files without RAII wrappers
///
/// This can be called in test cleanup or defer blocks
pub fn cleanup_test_file<P: AsRef<Path>>(path: P) {
    let _ = fs::remove_file(path);
}

/// Cleanup function for tests that create directories without RAII wrappers
pub fn cleanup_test_dir<P: AsRef<Path>>(path: P) {
    let _ = fs::remove_dir_all(path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_get_temp_dir() {
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
    fn test_temp_test_file() {
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
    fn test_temp_test_dir() {
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
    fn test_create_test_csv() {
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
}

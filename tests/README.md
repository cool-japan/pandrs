# PandRS Test Suite

This directory contains the comprehensive test suite for PandRS.

## Test Organization

- **Integration Tests**: Tests in the root `tests/` directory that test PandRS as a whole
- **Common Utilities**: Shared test utilities in `common/` module
- **Test Data**: Helper modules for generating test data

## Temporary File Handling

All tests that create temporary files should use the utilities provided in `common/test_utils` module. This ensures:

1. **Consistent temporary directory usage** - Respects `TMPDIR`, `TEMP`, and `TMP` environment variables
2. **Automatic cleanup** - Files and directories are cleaned up when tests complete
3. **Unique file names** - Avoids collisions when running tests in parallel
4. **Better debugging** - Can keep files on failure for inspection

### Basic Usage

```rust
mod common;
use common::test_utils::{TempTestFile, TempTestDir, create_test_csv};

#[test]
fn my_test() {
    // Create a temporary file (automatically cleaned up)
    let temp_file = TempTestFile::new("my_test", "csv");

    // Use the file
    df.to_csv(temp_file.path()).unwrap();
    let loaded = DataFrame::from_csv(temp_file.path(), true).unwrap();

    // File is automatically deleted when temp_file goes out of scope
}
```

### Available Utilities

#### `TempTestFile` - RAII temporary file

```rust
// Create a temp file
let temp_file = TempTestFile::new("test_name", "csv");

// Get the path
let path = temp_file.path();

// Keep file after test (for debugging)
temp_file.keep();

// Convert to path and disable cleanup
let path = temp_file.into_path();
```

#### `TempTestDir` - RAII temporary directory

```rust
// Create a temp directory
let temp_dir = TempTestDir::new("test_name")?;

// Get the path
let path = temp_dir.path();

// Create files in the directory
let file_path = temp_dir.path().join("data.csv");

// Directory and all contents are deleted on drop
```

#### Helper Functions

```rust
// Get temp directory (respects TMPDIR, TEMP, TMP env vars)
let temp_dir = get_temp_dir();

// Generate unique temp file path
let path = test_temp_path("test_name", "csv");

// Generate unique temp directory path
let dir = test_temp_dir("test_name");

// Create a test CSV with headers and rows
let headers = vec!["col1", "col2"];
let rows = vec![
    vec!["1".to_string(), "2".to_string()],
    vec!["3".to_string(), "4".to_string()],
];
let temp_file = create_test_csv("test", &headers, &rows);

// Manual cleanup (if not using RAII)
cleanup_test_file(path);
cleanup_test_dir(dir_path);
```

### Environment Variables

The test utilities respect the following environment variables for temporary directory location:

1. `TMPDIR` (primary, Unix standard)
2. `TEMP` (Windows standard)
3. `TMP` (fallback)
4. System default (`std::env::temp_dir()`)

To use a custom temporary directory:

```bash
# Unix/Linux/macOS
export TMPDIR=/custom/temp/path
cargo test

# Windows
set TEMP=C:\custom\temp\path
cargo test
```

### Debugging

To keep temporary files after a test failure for inspection:

```rust
#[test]
fn debug_test() {
    let mut temp_file = TempTestFile::new("debug", "csv");

    // ... test code ...

    // Keep file if we want to inspect it
    if std::env::var("KEEP_TEST_FILES").is_ok() {
        temp_file.keep();
    }
}
```

Then run with:

```bash
KEEP_TEST_FILES=1 cargo test debug_test
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test categorical_na_test

# Run tests with specific filter
cargo test categorical

# Run library tests only
cargo test --lib

# Run with output
cargo test -- --nocapture

# Run with multiple threads
cargo test -- --test-threads=4

# Run ignored tests
cargo test -- --ignored
```

## Test Coverage

Current test coverage: **565+ tests passing**

- Core data structures: Series, DataFrame, MultiIndex
- Data operations: filtering, sorting, grouping, joining
- I/O operations: CSV, Parquet, JSON, Excel, SQL
- Analytics: statistics, aggregations, window functions
- Time series: forecasting, decomposition, feature extraction
- Machine learning: models, metrics, serving
- Storage: memory management, zero-copy, compression
- Advanced features: distributed computing, GPU, streaming
- Security: authentication, authorization, audit logging
- Real-time analytics: metrics, dashboards, alerting

## Best Practices

1. **Always use RAII wrappers** (`TempTestFile`, `TempTestDir`) for automatic cleanup
2. **Use descriptive test names** when creating temp files to aid debugging
3. **Don't hardcode paths** - use the utilities to generate paths
4. **Clean up manually created resources** in test teardown if not using RAII
5. **Test in isolation** - each test should create its own temp files
6. **Use unique names** - the utilities ensure uniqueness automatically

## Example Test

```rust
mod common;
use common::test_utils::TempTestFile;
use pandrs::DataFrame;

#[test]
fn test_dataframe_csv_roundtrip() {
    // Create test data
    let mut df = DataFrame::new();
    df.add_int_column("id", vec![1, 2, 3]).unwrap();
    df.add_string_column("name", vec!["Alice", "Bob", "Charlie"]).unwrap();

    // Write to temporary file (auto-cleanup)
    let temp_file = TempTestFile::new("csv_roundtrip", "csv");
    df.to_csv(temp_file.path()).unwrap();

    // Read back
    let loaded = DataFrame::from_csv(temp_file.path(), true).unwrap();

    // Verify
    assert_eq!(df.shape(), loaded.shape());

    // temp_file is automatically cleaned up here
}
```

## Contributing

When adding new tests:

1. Use the common test utilities for temporary files
2. Add appropriate test documentation
3. Ensure tests are deterministic and can run in parallel
4. Follow the existing test naming conventions
5. Add integration tests for new features

## Troubleshooting

**Test fails with "File not found"**
- Check if you're using absolute paths from `TempTestFile::path()`
- Ensure the file is created before trying to use it

**Temp files not being cleaned up**
- Make sure you're using RAII wrappers or calling cleanup functions
- Check if tests panic before cleanup code runs

**Temp directory full**
- Tests create unique files but should clean up automatically
- Run `cargo clean` and remove files in your temp directory
- Check `TMPDIR` environment variable if customized

**Tests interfere with each other**
- Each test should use unique temp file names (utilities handle this)
- Don't share temp files between tests
- Use `--test-threads=1` to run serially if needed

## License

Same as PandRS project.

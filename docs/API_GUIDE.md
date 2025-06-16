# PandRS API Guide

This guide provides comprehensive documentation for using PandRS effectively, including API patterns and best practices.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [DataFrame Types](#dataframe-types)
3. [Series Operations](#series-operations)
4. [Column Management](#column-management)
5. [Data Access Patterns](#data-access-patterns)
6. [I/O Operations](#io-operations)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Best Practices](#best-practices)

## Core Concepts

### DataFrame vs OptimizedDataFrame

PandRS provides two main DataFrame implementations:

**DataFrame** (Traditional):
- Compatible with pandas-like API
- Uses Series internally 
- Good for small datasets and compatibility
- String-based operations

**OptimizedDataFrame** (Recommended):
- Column-oriented storage with type specialization
- Better performance for large datasets
- Built-in string pool optimization
- Type-safe column access via ColumnView

```rust
use pandrs::{DataFrame, Series};
use pandrs::optimized::OptimizedDataFrame;

// Traditional DataFrame
let mut df = DataFrame::new();
df.add_column("data".to_string(), Series::new(vec![1, 2, 3], None)?)?;

// Optimized DataFrame (recommended)
let mut opt_df = OptimizedDataFrame::new();
opt_df.add_int_column("data", vec![1, 2, 3])?;
```

### Series Types

Series are one-dimensional arrays that support:
- Generic type parameter `Series<T>`
- Name management with `set_name()` and `with_name()`
- Type conversion utilities
- Statistical operations

```rust
// Create and manage Series
let mut series = Series::new(vec![10, 20, 30], None)?;
series.set_name("values".to_string());

// Fluent API
let named_series = Series::new(vec![1, 2, 3], None)?
    .with_name("measurements".to_string());

// Type conversion
let string_series = series.to_string_series()?;
```

## DataFrame Types

### OptimizedDataFrame (Recommended)

The `OptimizedDataFrame` provides the best performance and is the recommended choice for most use cases.

#### Column Addition

```rust
let mut df = OptimizedDataFrame::new();

// Add different column types
df.add_int_column("age", vec![25, 30, 35])?;
df.add_float_column("salary", vec![50000.0, 60000.0, 70000.0])?;
df.add_string_column("name", vec![
    "Alice".to_string(), 
    "Bob".to_string(), 
    "Charlie".to_string()
])?;
df.add_boolean_column("active", vec![true, false, true])?;
```

#### Basic Operations

```rust
// DataFrame info
println!("Rows: {}", df.row_count());
println!("Columns: {}", df.column_count());
println!("Column names: {:?}", df.column_names());

// Statistical operations
let sum = df.sum("salary")?;
let mean = df.mean("salary")?;
let min = df.min("age")?;
let max = df.max("age")?;
```

### Traditional DataFrame

The traditional `DataFrame` is provided for compatibility and simple use cases.

```rust
let mut df = DataFrame::new();
let ages = Series::new(vec![25, 30, 35], Some("age".to_string()))?;
df.add_column("age".to_string(), ages)?;
```

## Series Operations

### Name Management

```rust
// Create series without name
let mut data = Series::new(vec![1, 2, 3, 4, 5], None)?;

// Set name after creation
data.set_name("measurements".to_string());
assert_eq!(data.name(), Some(&"measurements".to_string()));

// Create with name using fluent API
let named_data = Series::new(vec![10, 20, 30], None)?
    .with_name("scores".to_string());

// Type conversion with name preservation
let string_data = named_data.to_string_series()?;
assert_eq!(string_data.name(), named_data.name());
```

### Statistical Operations

```rust
let numbers = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string()))?;

// Basic statistics
let sum = numbers.sum();         // 150
let mean = numbers.mean()?;      // 30.0
let min = numbers.min()?;        // 10
let max = numbers.max()?;        // 50
let len = numbers.len();         // 5

// Iteration and access
for (i, value) in numbers.values().iter().enumerate() {
    println!("Index {}: {}", i, value);
}
```

## Column Management

### Column Renaming

```rust
use std::collections::HashMap;

let mut df = OptimizedDataFrame::new();
df.add_int_column("old_name1", vec![1, 2, 3])?;
df.add_int_column("old_name2", vec![4, 5, 6])?;

// Rename specific columns
let mut rename_map = HashMap::new();
rename_map.insert("old_name1".to_string(), "new_name1".to_string());
rename_map.insert("old_name2".to_string(), "new_name2".to_string());
df.rename_columns(&rename_map)?;

// Set all column names at once
df.set_column_names(vec![
    "first_column".to_string(),
    "second_column".to_string(),
])?;
```

### Column Validation

```rust
// Check if column exists
if df.contains_column("column_name") {
    println!("Column exists");
}

// Get column type
let col_type = df.column_type("column_name")?;
println!("Column type: {:?}", col_type);
```

## Data Access Patterns

### Type-Safe Column Access

The `OptimizedDataFrame` provides type-safe column access through `ColumnView`:

```rust
// Get column view
let col_view = df.column("age")?;

// Type-safe access
if let Some(int_col) = col_view.as_int64() {
    // Direct access to Int64Column
    let value = int_col.get(0)?;  // Returns Result<Option<i64>>
    let sum = int_col.sum();      // Returns i64
    let mean = int_col.mean();    // Returns Option<f64>
}

if let Some(str_col) = col_view.as_string() {
    // Direct access to StringColumn
    let value = str_col.get(0)?;  // Returns Result<Option<&str>>
    let length = str_col.len();
}

// Type mismatch returns None
assert!(col_view.as_float64().is_none()); // If column is actually Int64
```

### Safe Data Retrieval

```rust
// Always handle potential errors and None values
match df.column("column_name") {
    Ok(col_view) => {
        if let Some(int_col) = col_view.as_int64() {
            match int_col.get(index) {
                Ok(Some(value)) => println!("Value: {}", value),
                Ok(None) => println!("Null value"),
                Err(e) => println!("Error: {}", e),
            }
        }
    }
    Err(e) => println!("Column not found: {}", e),
}
```

## I/O Operations

### CSV Operations

```rust
// Write CSV
df.to_csv("output.csv", true)?;  // true = include header

// Read CSV
let loaded_df = pandrs::io::read_csv("input.csv", true)?;  // true = has header
```

### Error Handling for I/O

```rust
// Robust I/O with error handling
match df.to_csv("output.csv", true) {
    Ok(_) => {
        println!("Successfully saved to CSV");
        
        // Verify by reading back
        match pandrs::io::read_csv("output.csv", true) {
            Ok(loaded_df) => {
                println!("Successfully loaded {} rows", loaded_df.row_count());
            }
            Err(e) => println!("Error loading CSV: {}", e),
        }
    }
    Err(e) => println!("Error saving CSV: {}", e),
}
```

### Parquet Operations (Feature-gated)

```rust
#[cfg(feature = "parquet")]
{
    use pandrs::io::{write_parquet, read_parquet, ParquetCompression};
    
    // Write Parquet with compression
    write_parquet(&df, "output.parquet", Some(ParquetCompression::Snappy))?;
    
    // Read Parquet
    let loaded_df = read_parquet("input.parquet")?;
}
```

## Error Handling

### Error Types

PandRS uses a comprehensive error system:

```rust
use pandrs::error::{Error, Result};

// Common error patterns
match operation_result {
    Ok(value) => { /* handle success */ }
    Err(Error::ColumnNotFound(name)) => {
        println!("Column '{}' not found", name);
    }
    Err(Error::IndexOutOfBounds { index, size }) => {
        println!("Index {} out of bounds (size: {})", index, size);
    }
    Err(Error::InconsistentRowCount { expected, found }) => {
        println!("Row count mismatch: expected {}, found {}", expected, found);
    }
    Err(Error::IoError(io_err)) => {
        println!("I/O error: {}", io_err);
    }
    Err(e) => {
        println!("Other error: {}", e);
    }
}
```

### Best Error Handling Practices

```rust
// Use Result<()> for functions that can fail
fn process_dataframe() -> Result<()> {
    let mut df = OptimizedDataFrame::new();
    df.add_int_column("data", vec![1, 2, 3])?;
    
    // Use ? operator for error propagation
    let sum = df.sum("data")?;
    println!("Sum: {}", sum);
    
    Ok(())
}

// Handle specific error conditions
fn safe_column_access(df: &OptimizedDataFrame, col_name: &str, index: usize) -> Option<i64> {
    match df.column(col_name) {
        Ok(col_view) => {
            if let Some(int_col) = col_view.as_int64() {
                match int_col.get(index) {
                    Ok(Some(value)) => Some(*value),
                    _ => None,
                }
            } else {
                None
            }
        }
        Err(_) => None,
    }
}
```

## Performance Considerations

### Choosing the Right DataFrame Type

```rust
// For small datasets or compatibility
let mut small_df = DataFrame::new();

// For performance-critical applications (recommended)
let mut large_df = OptimizedDataFrame::new();
```

### String Pool Benefits

The `OptimizedDataFrame` automatically uses string pooling for memory efficiency:

```rust
// This benefits from automatic string deduplication
let mut df = OptimizedDataFrame::new();
let repeated_data = vec!["Category A".to_string(); 10000];  // High duplication
df.add_string_column("category", repeated_data)?;  // Memory efficient
```

### Batch Operations

```rust
// Add multiple columns efficiently
let mut df = OptimizedDataFrame::new();

// Prepare all data first
let ids: Vec<i64> = (1..=1000).collect();
let names: Vec<String> = (1..=1000).map(|i| format!("User_{}", i)).collect();
let scores: Vec<f64> = (1..=1000).map(|i| i as f64 * 0.1).collect();

// Add columns in sequence
df.add_int_column("id", ids)?;
df.add_string_column("name", names)?;
df.add_float_column("score", scores)?;
```

## Best Practices

### 1. Use OptimizedDataFrame for Performance

```rust
// Recommended
let mut df = OptimizedDataFrame::new();
df.add_int_column("data", vec![1, 2, 3])?;
```

### 2. Handle Errors Explicitly

```rust
// Good: explicit error handling
match df.column("column_name") {
    Ok(col) => { /* process column */ }
    Err(e) => { /* handle error */ }
}

// Avoid: ignoring potential errors
let col = df.column("column_name").unwrap(); // Don't do this
```

### 3. Use Type-Safe Access

```rust
// Good: type-safe access
let col_view = df.column("numbers")?;
if let Some(int_col) = col_view.as_int64() {
    let value = int_col.get(0)?;
}

// Avoid: assuming types
let numbers = df.get_int_column("numbers")?; // May not exist in OptimizedDataFrame
```

### 4. Validate Data Early

```rust
// Check data consistency before processing
if df.row_count() == 0 {
    return Err("Empty DataFrame".into());
}

if !df.contains_column("required_column") {
    return Err("Required column missing".into());
}
```

### 5. Use Meaningful Names

```rust
// Good: descriptive names
let sales_data = Series::new(vec![100, 200, 150], None)?
    .with_name("monthly_sales_usd".to_string());

// Avoid: generic names
let data = Series::new(vec![100, 200, 150], None)?
    .with_name("data".to_string());
```

### 6. Prefer Batch Operations

```rust
// Good: prepare data then add columns
let mut df = OptimizedDataFrame::new();
let (ids, names, scores) = prepare_data(); // Batch preparation

df.add_int_column("id", ids)?;
df.add_string_column("name", names)?;
df.add_float_column("score", scores)?;

// Avoid: row-by-row operations in loops
```

### 7. Test Edge Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dataframe() {
        let df = OptimizedDataFrame::new();
        assert_eq!(df.row_count(), 0);
        assert_eq!(df.column_count(), 0);
    }

    #[test]
    fn test_column_not_found() {
        let df = OptimizedDataFrame::new();
        assert!(df.column("nonexistent").is_err());
    }

    #[test]
    fn test_type_safety() {
        let mut df = OptimizedDataFrame::new();
        df.add_int_column("numbers", vec![1, 2, 3]).unwrap();
        
        let col_view = df.column("numbers").unwrap();
        assert!(col_view.as_int64().is_some());
        assert!(col_view.as_string().is_none());
    }
}
```

## Advanced Usage

### Custom Error Handling

```rust
use pandrs::error::{Error, Result};

fn process_sales_data(file_path: &str) -> Result<f64> {
    let df = pandrs::io::read_csv(file_path, true)
        .map_err(|e| Error::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to load sales data: {}", e)
        )))?;

    if !df.contains_column("sales") {
        return Err(Error::ColumnNotFound("sales".to_string()));
    }

    let total_sales = df.sum("sales")?;
    Ok(total_sales)
}
```

### Performance Monitoring

```rust
use std::time::Instant;

fn benchmark_operation() -> Result<()> {
    let start = Instant::now();
    
    let mut df = OptimizedDataFrame::new();
    let large_data: Vec<i64> = (0..1_000_000).collect();
    df.add_int_column("data", large_data)?;
    
    let creation_time = start.elapsed();
    println!("DataFrame creation: {:?}", creation_time);
    
    let start = Instant::now();
    let sum = df.sum("data")?;
    let computation_time = start.elapsed();
    println!("Sum computation: {:?}", computation_time);
    println!("Result: {}", sum);
    
    Ok(())
}
```

This guide covers the essential patterns for working with PandRS. For more examples, see the `examples/` directory in the repository.
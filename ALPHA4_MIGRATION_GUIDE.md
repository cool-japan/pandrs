# PandRS Alpha.4 Migration Guide

This guide helps you migrate from previous versions of PandRS to alpha.4, taking advantage of the new features while ensuring compatibility.

## 📦 Installation & Setup

### Updating Dependencies

#### Rust Projects
```toml
# Before (alpha.3)
[dependencies]
pandrs = "0.1.0-alpha.3"

# After (alpha.4)
[dependencies]
pandrs = "0.1.0-alpha.4"

# With new features
[dependencies]
pandrs = { version = "0.1.0-alpha.4", features = ["distributed", "parquet", "sql"] }
```

#### Python Projects
```bash
# Remove old version
pip uninstall pandrs

# Install alpha.4 from source
git clone https://github.com/cool-japan/pandrs
cd pandrs/py_bindings
pip install . --config-settings="--features=parquet,sql,distributed"
```

## 🔄 Code Migration

### 1. DataFrame Column Management

#### Before (Manual Column Operations)
```rust
// Alpha.3 - Manual column renaming
let mut new_df = DataFrame::new();
for (old_name, new_name) in rename_mapping {
    let series = df.get_column(&old_name)?;
    new_df.add_column(new_name, series)?;
}
```

#### After (Alpha.4 - Built-in Methods)
```rust
// Alpha.4 - Built-in column renaming
use std::collections::HashMap;

let mut rename_map = HashMap::new();
rename_map.insert("old_name".to_string(), "new_name".to_string());
df.rename_columns(&rename_map)?;  // New in alpha.4

// Or set all column names at once
df.set_column_names(vec!["col1".to_string(), "col2".to_string()])?;  // New in alpha.4
```

### 2. Series Name Management

#### Before (Limited Name Operations)
```rust
// Alpha.3 - Limited name management
let series = Series::new(data, Some("initial_name".to_string()))?;
// No easy way to change name after creation
```

#### After (Alpha.4 - Enhanced Name Management)
```rust
// Alpha.4 - Flexible name management
let mut series = Series::new(data, None)?
    .with_name("initial_name".to_string());  // Fluent interface

series.set_name("updated_name".to_string());  // Change name anytime

// Type conversion with name preservation
let string_series = series.to_string_series()?;  // Name preserved
```

### 3. Distributed Processing

#### Before (Basic Parallel Processing)
```rust
// Alpha.3 - Basic parallel operations
use rayon::prelude::*;

let results: Vec<_> = data.par_iter()
    .map(|item| process_item(item))
    .collect();
```

#### After (Alpha.4 - DataFusion Integration)
```rust
// Alpha.4 - Distributed processing with DataFusion
use pandrs::distributed::DistributedContext;

let mut context = DistributedContext::new_local(4)?;
context.register_dataframe("data", &df)?;

// SQL-based distributed processing
let result = context.sql("
    SELECT region, AVG(sales) as avg_sales
    FROM data 
    GROUP BY region
    ORDER BY avg_sales DESC
")?;

// Or DataFrame-style API
let sales_df = context.dataset("data")?;
let aggregated = sales_df
    .filter("sales > 1000")?
    .aggregate(&["region"], &[("sales", "avg", "avg_sales")])?
    .collect()?;
```

### 4. Enhanced I/O Operations

#### Before (Basic I/O)
```rust
// Alpha.3 - Basic Parquet support
df.to_parquet("data.parquet")?;  // Limited options
let loaded_df = DataFrame::from_parquet("data.parquet")?;
```

#### After (Alpha.4 - Enhanced I/O)
```rust
// Alpha.4 - Enhanced Parquet with compression
use pandrs::io::parquet::{write_parquet, read_parquet, ParquetCompression};

// Write with compression options
write_parquet(&df, "data.parquet", Some(ParquetCompression::Snappy))?;
write_parquet(&df, "data_gzip.parquet", Some(ParquetCompression::Gzip))?;

// Read with better error handling
let loaded_df = read_parquet("data.parquet")?;

// Enhanced SQL I/O
use pandrs::io::sql::{write_to_sql, read_sql};

write_to_sql(&df, "employees", "database.db", "replace")?;
let sql_df = read_sql("SELECT * FROM employees WHERE salary > 50000", "database.db")?;
```

## 🐍 Python Migration

### 1. Enhanced DataFrame Operations

#### Before (Alpha.3)
```python
import pandrs

df = pandrs.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
# Limited column management options
```

#### After (Alpha.4)
```python
import pandrs

df = pandrs.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

# New column management features
df.rename_columns({'a': 'col_a', 'b': 'col_b'})  # New in alpha.4
df.columns = ['new_a', 'new_b']  # New setter in alpha.4

# Enhanced OptimizedDataFrame
opt_df = pandrs.OptimizedDataFrame()
opt_df.add_int_column('id', [1, 2, 3])
opt_df.add_string_column('name', ['Alice', 'Bob', 'Carol'])
opt_df.rename_columns({'id': 'identifier'})  # New in alpha.4
```

### 2. Enhanced I/O

#### Before (Alpha.3)
```python
# Limited I/O options
df.to_csv('data.csv')
loaded_df = pandrs.DataFrame.read_csv('data.csv')
```

#### After (Alpha.4)
```python
# Enhanced I/O with Parquet and SQL support
opt_df = pandrs.OptimizedDataFrame()
# ... add data ...

# Parquet I/O with compression
opt_df.to_parquet('data.parquet', compression='snappy')  # New in alpha.4
loaded_df = pandrs.OptimizedDataFrame.from_parquet('data.parquet')  # New in alpha.4

# SQL I/O
df.to_sql('employees', 'database.db', if_exists='replace')  # Enhanced in alpha.4
sql_df = pandrs.DataFrame.from_sql('SELECT * FROM employees', 'database.db')  # Enhanced in alpha.4
```

## ⚠️ Breaking Changes & Compatibility

### No Breaking Changes
Alpha.4 maintains **full backward compatibility** with alpha.3. All existing code should continue to work without modifications.

### New Dependencies
If you want to use new features, you may need to enable additional feature flags:

```toml
# Optional: Enable new features
[dependencies]
pandrs = { 
    version = "0.1.0-alpha.4", 
    features = [
        "distributed",  # For DataFusion integration
        "parquet",      # For enhanced Parquet I/O
        "sql",          # For SQL I/O operations
    ] 
}
```

## 🔄 Gradual Migration Strategy

### Phase 1: Update Dependencies
1. Update your `Cargo.toml` or Python environment
2. Run existing tests to ensure compatibility
3. No code changes required at this stage

### Phase 2: Adopt New Column Management
```rust
// Replace manual column operations with new APIs
// Old way still works, but new way is more efficient

// Instead of manual renaming:
// let new_df = manual_rename(old_df, mapping);

// Use new built-in method:
df.rename_columns(&mapping)?;
```

### Phase 3: Leverage Enhanced I/O
```rust
// Gradually replace basic I/O with enhanced versions
// Old: df.to_parquet("file.parquet")?;
// New: write_parquet(&df, "file.parquet", Some(ParquetCompression::Snappy))?;
```

### Phase 4: Adopt Distributed Processing
```rust
// For large datasets, gradually introduce distributed processing
if df.row_count() > 100_000 {
    // Use distributed processing for large datasets
    let mut context = DistributedContext::new_local(4)?;
    context.register_dataframe("large_data", &df)?;
    let result = context.sql(&complex_query)?;
} else {
    // Use regular processing for smaller datasets
    let result = df.filter(&simple_filter)?;
}
```

## 🚧 Common Migration Issues

### Issue 1: Feature Flags Not Enabled
**Error**: Features not available at compile time

**Solution**: Enable required feature flags
```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.4", features = ["distributed", "parquet", "sql"] }
```

### Issue 2: Import Path Changes
**Error**: Some advanced features require explicit imports

**Solution**: Add required imports
```rust
use pandrs::distributed::DistributedContext;
use pandrs::io::parquet::{write_parquet, read_parquet, ParquetCompression};
use pandrs::io::sql::{write_to_sql, read_sql};
```

### Issue 3: Python Bindings Installation
**Error**: Python bindings not available

**Solution**: Install from source with features
```bash
cd pandrs/py_bindings
pip install . --config-settings="--features=parquet,sql,distributed"
```

## 📈 Performance Migration

### Before (Alpha.3)
```rust
// Basic operations with limited optimization
let filtered = df.filter_by_column("value", |v| v > 100.0)?;
let grouped = filtered.group_by("category")?;
let result = grouped.sum()?;
```

### After (Alpha.4)
```rust
// Optimized operations with better performance
let result = if df.row_count() > 10_000 {
    // Use distributed processing for large datasets
    let mut context = DistributedContext::new_local(4)?;
    context.register_dataframe("data", &df)?;
    context.sql("
        SELECT category, SUM(value) as total
        FROM data 
        WHERE value > 100.0
        GROUP BY category
    ")?
} else {
    // Use optimized local processing
    df.filter("value > 100.0")?
      .aggregate(&["category"], &[("value", "sum", "total")])?
};
```

## 🧪 Testing Your Migration

### 1. Compatibility Tests
```rust
#[test]
fn test_alpha4_compatibility() {
    // Ensure existing code still works
    let mut df = DataFrame::new();
    df.add_column("test".to_string(), Series::new(vec![1, 2, 3], None)?)?;
    
    // Old operations should still work
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_names(), vec!["test"]);
}
```

### 2. New Feature Tests
```rust
#[test]
fn test_alpha4_new_features() {
    let mut df = create_test_dataframe();
    
    // Test new column management
    let mut rename_map = HashMap::new();
    rename_map.insert("old_name".to_string(), "new_name".to_string());
    df.rename_columns(&rename_map)?;
    
    // Test enhanced I/O
    write_parquet(&df, "test.parquet", Some(ParquetCompression::Snappy))?;
    let loaded_df = read_parquet("test.parquet")?;
    
    assert_eq!(loaded_df.row_count(), df.row_count());
}
```

### 3. Performance Tests
```rust
#[test]
fn test_alpha4_performance() {
    let large_df = create_large_dataframe(100_000);
    
    let start = std::time::Instant::now();
    
    // Test distributed processing performance
    let mut context = DistributedContext::new_local(4)?;
    context.register_dataframe("large_data", &large_df)?;
    let result = context.sql("SELECT COUNT(*) FROM large_data")?;
    
    let duration = start.elapsed();
    assert!(duration.as_secs() < 10); // Should complete within 10 seconds
}
```

## 📚 Additional Resources

- **Alpha.4 Release Notes**: `ALPHA4_RELEASE_NOTES.md`
- **API Documentation**: [docs.rs/pandrs](https://docs.rs/pandrs)
- **Examples**: Check the `examples/` directory for alpha.4 usage patterns
- **Tests**: Review `tests/alpha4_integration_test.rs` for comprehensive examples

## 🆘 Getting Help

If you encounter issues during migration:

1. **Check the Release Notes**: `ALPHA4_RELEASE_NOTES.md` for known issues
2. **Review Examples**: Look at updated examples in the repository
3. **Ask for Help**: Open an issue on [GitHub](https://github.com/cool-japan/pandrs/issues)
4. **Community**: Join discussions on [GitHub Discussions](https://github.com/cool-japan/pandrs/discussions)

---

**Happy migrating to PandRS Alpha.4!** 🚀
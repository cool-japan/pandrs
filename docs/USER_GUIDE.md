# PandRS v0.2.0 User Guide

A comprehensive guide to using PandRS - A high-performance DataFrame library for Rust.

**Version:** 0.2.0
**Author:** COOLJAPAN OU (Team Kitasan)
**License:** Apache-2.0

---

## Table of Contents

1. [Getting Started](#chapter-1-getting-started)
2. [DataFrame Operations](#chapter-2-dataframe-operations)
3. [Series Manipulation](#chapter-3-series-manipulation)
4. [I/O Operations](#chapter-4-io-operations)
5. [Time Series Analysis](#chapter-5-time-series-analysis)
6. [Machine Learning](#chapter-6-machine-learning)
7. [Security & Access Control](#chapter-7-security--access-control)
8. [Real-Time Analytics](#chapter-8-real-time-analytics)
9. [Best Practices](#chapter-9-best-practices)

---

## Chapter 1: Getting Started

### 1.1 Installation

Add PandRS to your `Cargo.toml`:

```toml
[dependencies]
pandrs = "0.2.0"
```

#### Feature Flags

PandRS supports various optional features:

```toml
[dependencies]
pandrs = { version = "0.2.0", features = ["stable", "parquet", "excel", "sql"] }
```

**Core Features:**
- `stable`: Recommended stable feature set (includes optimized, parquet, excel, sql, streaming, backward_compat)
- `optimized`: Performance optimizations with SIMD vectorization
- `backward_compat`: Backward compatibility support for older APIs

**Data Formats:**
- `parquet`: Apache Parquet file support with compression
- `excel`: Excel XLSX/XLS file read/write support
- `sql`: Database connectivity (PostgreSQL, MySQL, SQLite)

**Advanced Features:**
- `distributed`: Distributed computing with DataFusion
- `visualization`: Plotting and charting capabilities
- `streaming`: Real-time streaming data processing
- `serving`: Model serving and deployment
- `resilience`: Retry mechanisms and circuit breakers

**Experimental:**
- `cuda`: GPU acceleration (requires CUDA toolkit, not available on macOS)
- `wasm`: WebAssembly compilation support
- `jit`: Just-in-time compilation for query optimization

### 1.2 Quick Start Example

Here's a simple example to get you started:

```rust
use pandrs::{DataFrame, Series};
use pandrs::error::Result;

fn main() -> Result<()> {
    // Create a new DataFrame
    let mut df = DataFrame::new();

    // Add columns using Series
    df.add_column("name".to_string(),
        Series::new(vec!["Alice", "Bob", "Carol"], Some("name".to_string()))?)?;
    df.add_column("age".to_string(),
        Series::new(vec![30, 25, 35], Some("age".to_string()))?)?;
    df.add_column("salary".to_string(),
        Series::new(vec![75000.0, 65000.0, 85000.0], Some("salary".to_string()))?)?;

    // Basic operations
    println!("Number of rows: {}", df.nrows());
    println!("Number of columns: {}", df.ncols());
    println!("Column names: {:?}", df.column_names());

    // Access a column
    let ages = df.column("age")?;
    println!("Mean age: {:.2}", ages.mean()?);

    Ok(())
}
```

**Output:**
```
Number of rows: 3
Number of columns: 3
Column names: ["name", "age", "salary"]
Mean age: 30.00
```

### 1.3 Basic DataFrame Creation

#### From Vectors

```rust
use pandrs::{DataFrame, Series};
use pandrs::error::Result;

fn create_from_vectors() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    df.add_column("product".to_string(),
        Series::new(vec!["Apple", "Banana", "Orange"], Some("product".to_string()))?)?;
    df.add_column("quantity".to_string(),
        Series::new(vec![10, 20, 15], Some("quantity".to_string()))?)?;
    df.add_column("price".to_string(),
        Series::new(vec![1.5, 0.8, 1.2], Some("price".to_string()))?)?;

    Ok(df)
}
```

#### Using OptimizedDataFrame (Recommended)

For better performance, use `OptimizedDataFrame`:

```rust
use pandrs::OptimizedDataFrame;
use pandrs::error::Result;

fn create_optimized_dataframe() -> Result<OptimizedDataFrame> {
    let mut df = OptimizedDataFrame::new();

    // Type-specific column additions
    df.add_string_column("name", vec!["Alice".to_string(), "Bob".to_string()])?;
    df.add_int_column("age", vec![30, 25])?;
    df.add_float_column("salary", vec![75000.0, 65000.0])?;
    df.add_bool_column("active", vec![true, false])?;

    Ok(df)
}
```

**Pro Tip:** `OptimizedDataFrame` uses columnar storage with string pooling, making it significantly faster for large datasets.

### 1.4 Reading and Writing CSV

#### Reading CSV

```rust
use pandrs::io::read_csv;
use pandrs::error::Result;

fn read_csv_file() -> Result<()> {
    // Basic CSV reading with headers
    let df = read_csv("data.csv", true)?;

    println!("Loaded {} rows, {} columns", df.nrows(), df.ncols());
    println!("Columns: {:?}", df.column_names());

    Ok(())
}
```

#### Writing CSV

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn write_csv_file(df: &DataFrame) -> Result<()> {
    // Write DataFrame to CSV
    df.to_csv("output.csv")?;

    println!("DataFrame saved to output.csv");
    Ok(())
}
```

#### CSV with Options

```rust
use pandrs::OptimizedDataFrame;
use pandrs::error::Result;

fn csv_with_options(df: &OptimizedDataFrame) -> Result<()> {
    // Write with headers
    df.to_csv("with_headers.csv", true)?;

    // Write without headers
    df.to_csv("without_headers.csv", false)?;

    Ok(())
}
```

**Common Pitfalls:**
- Make sure your CSV file exists before reading
- Check file permissions for write operations
- Ensure proper encoding (UTF-8 is expected)

---

## Chapter 2: DataFrame Operations

### 2.1 Creating DataFrames

#### Empty DataFrame

```rust
use pandrs::DataFrame;

let df = DataFrame::new();
```

#### From Multiple Series

```rust
use pandrs::{DataFrame, Series};
use pandrs::error::Result;

fn create_complex_dataframe() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    // Add different data types
    df.add_column("id".to_string(),
        Series::new(vec![1, 2, 3, 4, 5], Some("id".to_string()))?)?;
    df.add_column("name".to_string(),
        Series::new(vec!["Alice", "Bob", "Carol", "Dave", "Eve"], Some("name".to_string()))?)?;
    df.add_column("score".to_string(),
        Series::new(vec![95.5, 87.3, 92.1, 88.7, 94.2], Some("score".to_string()))?)?;
    df.add_column("passed".to_string(),
        Series::new(vec![true, true, true, true, true], Some("passed".to_string()))?)?;

    Ok(df)
}
```

### 2.2 Selecting Columns and Rows

#### Select Single Column

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn select_column(df: &DataFrame) -> Result<()> {
    let name_column = df.column("name")?;
    println!("Name column: {:?}", name_column);

    // Get statistics
    if let Ok(mean) = name_column.mean() {
        println!("Mean: {}", mean);
    }

    Ok(())
}
```

#### Select Multiple Columns

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn select_columns(df: &DataFrame) -> Result<DataFrame> {
    // Create new DataFrame with selected columns
    let mut subset = DataFrame::new();

    subset.add_column("name".to_string(), df.column("name")?.clone())?;
    subset.add_column("age".to_string(), df.column("age")?.clone())?;

    Ok(subset)
}
```

#### Row Selection by Index

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn select_rows(df: &DataFrame) -> Result<()> {
    // Get first 5 rows
    let head = df.head(5);
    println!("First 5 rows: {:?}", head);

    // Get last 3 rows
    let tail = df.tail(3);
    println!("Last 3 rows: {:?}", tail);

    Ok(())
}
```

### 2.3 Filtering and Boolean Indexing

#### Query-Based Filtering

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn filter_dataframe(df: &DataFrame) -> Result<DataFrame> {
    // Filter rows where age > 25
    let filtered = df.filter("age > 25")?;
    println!("Filtered {} rows", filtered.nrows());

    // Complex queries
    let complex = df.filter("age > 25 AND salary > 70000")?;

    Ok(filtered)
}
```

#### Multiple Conditions

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn complex_filtering(df: &DataFrame) -> Result<DataFrame> {
    // Combine multiple conditions
    let result = df.filter("(age > 25 AND salary > 70000) OR (age < 30 AND active == true)")?;

    Ok(result)
}
```

**Supported Operators:**
- Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical: `AND`, `OR`, `NOT`
- Arithmetic: `+`, `-`, `*`, `/`

### 2.4 Sorting and Ranking

#### Sort by Single Column

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn sort_dataframe(df: &DataFrame) -> Result<DataFrame> {
    // Sort by age in ascending order
    let sorted = df.sort_values("age", true)?;

    // Sort by salary in descending order
    let desc_sorted = df.sort_values("salary", false)?;

    Ok(sorted)
}
```

#### Sort by Multiple Columns

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn multi_column_sort(df: &DataFrame) -> Result<DataFrame> {
    // Sort by department (ascending), then by salary (descending)
    let sorted = df.sort_values_multi(
        vec!["department", "salary"],
        vec![true, false]
    )?;

    Ok(sorted)
}
```

### 2.5 Aggregations

#### Basic Aggregations

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn basic_aggregations(df: &DataFrame) -> Result<()> {
    let age_col = df.column("age")?;

    println!("Sum: {}", age_col.sum());
    println!("Mean: {:.2}", age_col.mean()?);
    println!("Min: {:.2}", age_col.min()?);
    println!("Max: {:.2}", age_col.max()?);
    println!("Std: {:.2}", age_col.std()?);

    Ok(())
}
```

#### GroupBy Operations

```rust
use pandrs::error::Result;
use pandrs::DataFrame;
use std::collections::HashMap;

fn groupby_example(df: &DataFrame) -> Result<()> {
    // Group by department and calculate aggregations
    let grouped = df.groupby(vec!["department"])?;

    let mut agg_funcs = HashMap::new();
    agg_funcs.insert("salary".to_string(), vec!["mean", "sum", "count"]);
    agg_funcs.insert("age".to_string(), vec!["mean", "min", "max"]);

    let result = grouped.agg(agg_funcs)?;
    println!("Aggregated results: {:?}", result);

    Ok(())
}
```

#### Available Aggregation Functions

- **Numeric:** `sum`, `mean`, `median`, `min`, `max`, `std`, `var`, `count`
- **String:** `count`, `unique`, `first`, `last`
- **Boolean:** `count`, `any`, `all`

**Pro Tip:** Use `OptimizedDataFrame` for significantly faster group-by operations on large datasets.

### 2.6 Joining and Merging

#### Inner Join

```rust
use pandrs::error::Result;
use pandrs::{DataFrame, Series};

fn inner_join_example() -> Result<()> {
    let mut df1 = DataFrame::new();
    df1.add_column("id".to_string(),
        Series::new(vec![1, 2, 3], Some("id".to_string()))?)?;
    df1.add_column("name".to_string(),
        Series::new(vec!["Alice", "Bob", "Carol"], Some("name".to_string()))?)?;

    let mut df2 = DataFrame::new();
    df2.add_column("id".to_string(),
        Series::new(vec![1, 2, 4], Some("id".to_string()))?)?;
    df2.add_column("salary".to_string(),
        Series::new(vec![75000.0, 65000.0, 80000.0], Some("salary".to_string()))?)?;

    // Inner join on 'id' column
    let joined = df1.join(&df2, "id", "inner")?;
    println!("Joined DataFrame: {:?}", joined);

    Ok(())
}
```

#### Left, Right, and Outer Joins

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

fn all_join_types(left: &DataFrame, right: &DataFrame) -> Result<()> {
    // Left join - keep all rows from left
    let left_joined = left.join(right, "key", "left")?;

    // Right join - keep all rows from right
    let right_joined = left.join(right, "key", "right")?;

    // Outer join - keep all rows from both
    let outer_joined = left.join(right, "key", "outer")?;

    Ok(())
}
```

**Common Pitfalls:**
- Make sure join keys exist in both DataFrames
- Be aware of duplicate keys (may result in cartesian product)
- Handle missing values (NA) after outer joins

---

## Chapter 3: Series Manipulation

### 3.1 Creating Series

#### Basic Series Creation

```rust
use pandrs::Series;
use pandrs::error::Result;

fn create_series() -> Result<()> {
    // Integer series
    let int_series = Series::new(vec![1, 2, 3, 4, 5], Some("numbers".to_string()))?;

    // Float series
    let float_series = Series::new(vec![1.5, 2.7, 3.9], Some("floats".to_string()))?;

    // String series
    let string_series = Series::new(
        vec!["apple", "banana", "cherry"],
        Some("fruits".to_string())
    )?;

    // Boolean series
    let bool_series = Series::new(vec![true, false, true], Some("flags".to_string()))?;

    Ok(())
}
```

#### Series with Name Management

```rust
use pandrs::Series;
use pandrs::error::Result;

fn series_naming() -> Result<()> {
    // Create series without name
    let mut series = Series::new(vec![1, 2, 3], None)?;

    // Set name later
    series.set_name("my_series".to_string());

    // Fluent API
    let named_series = Series::new(vec![4, 5, 6], None)?
        .with_name("another_series".to_string());

    println!("Series name: {:?}", series.name());

    Ok(())
}
```

### 3.2 Arithmetic Operations

```rust
use pandrs::Series;
use pandrs::error::Result;

fn series_arithmetic() -> Result<()> {
    let series1 = Series::new(vec![10, 20, 30], Some("a".to_string()))?;
    let series2 = Series::new(vec![1, 2, 3], Some("b".to_string()))?;

    // Element-wise addition
    let sum = &series1 + &series2;

    // Element-wise subtraction
    let diff = &series1 - &series2;

    // Element-wise multiplication
    let product = &series1 * &series2;

    // Element-wise division
    let quotient = &series1 / &series2;

    // Scalar operations
    let scaled = &series1 * 2;

    println!("Sum: {:?}", sum);
    println!("Scaled: {:?}", scaled);

    Ok(())
}
```

### 3.3 Missing Data Handling

#### Detecting Missing Values

```rust
use pandrs::{Series, NA};
use pandrs::error::Result;

fn detect_missing() -> Result<()> {
    let series = Series::new(
        vec![Some(1.0), None, Some(3.0), None, Some(5.0)],
        Some("data".to_string())
    )?;

    // Check if series has missing values
    let has_na = series.has_na();
    println!("Has missing values: {}", has_na);

    // Count missing values
    let na_count = series.na_count();
    println!("Number of NA values: {}", na_count);

    Ok(())
}
```

#### Filling Missing Values

```rust
use pandrs::Series;
use pandrs::error::Result;

fn fill_missing() -> Result<()> {
    let mut series = Series::new(
        vec![Some(1.0), None, Some(3.0), None, Some(5.0)],
        Some("data".to_string())
    )?;

    // Fill with a constant value
    series.fillna(0.0)?;

    // Fill with forward fill (propagate previous value)
    series.fillna_forward()?;

    // Fill with backward fill (propagate next value)
    series.fillna_backward()?;

    // Fill with mean
    let mean = series.mean()?;
    series.fillna(mean)?;

    Ok(())
}
```

#### Dropping Missing Values

```rust
use pandrs::Series;
use pandrs::error::Result;

fn drop_missing() -> Result<()> {
    let series = Series::new(
        vec![Some(1.0), None, Some(3.0), None, Some(5.0)],
        Some("data".to_string())
    )?;

    // Drop all NA values
    let cleaned = series.dropna()?;

    println!("Original length: {}", series.len());
    println!("Cleaned length: {}", cleaned.len());

    Ok(())
}
```

### 3.4 String Operations (.str accessor)

```rust
use pandrs::Series;
use pandrs::error::Result;

fn string_operations() -> Result<()> {
    let series = Series::new(
        vec!["  Hello  ", "  World  ", "  Rust  "],
        Some("text".to_string())
    )?;

    // Access string methods via .str()
    let str_accessor = series.str()?;

    // Convert to uppercase
    let upper = str_accessor.to_uppercase();

    // Convert to lowercase
    let lower = str_accessor.to_lowercase();

    // Strip whitespace
    let stripped = str_accessor.strip();

    // String length
    let lengths = str_accessor.len();

    // Contains substring
    let contains = str_accessor.contains("ello")?;

    // Replace substring
    let replaced = str_accessor.replace("Hello", "Hi")?;

    // Split string
    let split = str_accessor.split(" ")?;

    println!("Uppercase: {:?}", upper);
    println!("Lengths: {:?}", lengths);

    Ok(())
}
```

**Available String Methods:**
- `to_uppercase()`, `to_lowercase()`, `to_titlecase()`
- `strip()`, `lstrip()`, `rstrip()`
- `len()`, `is_empty()`
- `contains(pattern)`, `startswith(prefix)`, `endswith(suffix)`
- `replace(old, new)`, `split(delimiter)`
- `slice(start, end)`, `substring(start, length)`

### 3.5 DateTime Operations (.dt accessor)

```rust
use pandrs::Series;
use pandrs::error::Result;
use chrono::{Utc, TimeZone};

fn datetime_operations() -> Result<()> {
    // Create datetime series
    let dates = vec![
        Utc.ymd(2024, 1, 15).and_hms(10, 30, 0),
        Utc.ymd(2024, 2, 20).and_hms(14, 45, 0),
        Utc.ymd(2024, 3, 25).and_hms(16, 20, 0),
    ];

    let dt_series = Series::new(dates, Some("dates".to_string()))?;

    // Access datetime methods via .dt()
    let dt_accessor = dt_series.dt()?;

    // Extract components
    let years = dt_accessor.year();
    let months = dt_accessor.month();
    let days = dt_accessor.day();
    let hours = dt_accessor.hour();
    let weekdays = dt_accessor.weekday();

    // Date arithmetic
    let plus_days = dt_accessor.add_days(7)?;
    let minus_months = dt_accessor.subtract_months(1)?;

    // Formatting
    let formatted = dt_accessor.strftime("%Y-%m-%d %H:%M:%S")?;

    println!("Years: {:?}", years);
    println!("Weekdays: {:?}", weekdays);
    println!("Formatted: {:?}", formatted);

    Ok(())
}
```

**Available DateTime Methods:**
- Component extraction: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`
- Week info: `weekday()`, `week()`, `quarter()`
- Date arithmetic: `add_days()`, `add_months()`, `subtract_days()`
- Formatting: `strftime(format)`
- Timezone: `tz_convert()`, `tz_localize()`

---

## Chapter 4: I/O Operations

### 4.1 CSV Operations

#### Basic CSV Reading

```rust
use pandrs::io::read_csv;
use pandrs::error::Result;

fn read_csv_basic() -> Result<()> {
    // Read CSV with headers
    let df = read_csv("data.csv", true)?;

    println!("Loaded {} rows", df.nrows());

    Ok(())
}
```

#### CSV with Custom Options

```rust
use pandrs::io::CsvReadOptions;
use pandrs::error::Result;

fn read_csv_advanced() -> Result<()> {
    let options = CsvReadOptions {
        delimiter: b',',
        has_header: true,
        skip_rows: 0,
        max_rows: None,
        columns: None, // Select specific columns
        infer_schema_length: 100,
    };

    let df = read_csv_with_options("data.csv", options)?;

    Ok(())
}
```

#### Writing CSV

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn write_csv_example(df: &DataFrame) -> Result<()> {
    // Basic write
    df.to_csv("output.csv")?;

    // Write without headers
    df.to_csv_no_header("output_no_header.csv")?;

    Ok(())
}
```

### 4.2 JSON Operations

#### Reading JSON

```rust
use pandrs::io::read_json;
use pandrs::error::Result;

fn read_json_file() -> Result<()> {
    // Read JSON (records format)
    let df = read_json("data.json")?;

    println!("Loaded JSON with {} rows", df.nrows());

    Ok(())
}
```

**Supported JSON Formats:**

1. **Records format:**
```json
[
  {"name": "Alice", "age": 30, "salary": 75000},
  {"name": "Bob", "age": 25, "salary": 65000}
]
```

2. **Columnar format:**
```json
{
  "name": ["Alice", "Bob"],
  "age": [30, 25],
  "salary": [75000, 65000]
}
```

#### Writing JSON

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn write_json_example(df: &DataFrame) -> Result<()> {
    // Write as records format
    df.to_json("output.json")?;

    // Write as columnar format
    df.to_json_columnar("output_columnar.json")?;

    Ok(())
}
```

### 4.3 Parquet Operations

Parquet is a columnar storage format optimized for analytics.

#### Reading Parquet

```rust
#[cfg(feature = "parquet")]
use pandrs::io::read_parquet;
use pandrs::error::Result;

#[cfg(feature = "parquet")]
fn read_parquet_file() -> Result<()> {
    let df = read_parquet("data.parquet")?;

    println!("Loaded {} rows from Parquet", df.nrows());

    Ok(())
}
```

#### Writing Parquet with Compression

```rust
#[cfg(feature = "parquet")]
use pandrs::io::{ParquetWriteOptions, ParquetCompression};
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "parquet")]
fn write_parquet_compressed(df: &DataFrame) -> Result<()> {
    let options = ParquetWriteOptions {
        compression: ParquetCompression::Snappy,
        row_group_size: 1024 * 1024, // 1MB row groups
        enable_statistics: true,
        enable_dictionary: true,
    };

    df.to_parquet_with_options("output.parquet", options)?;

    Ok(())
}
```

**Compression Options:**
- `None`: No compression (fastest write)
- `Snappy`: Good balance of speed and compression
- `Gzip`: Better compression, slower
- `Lz4`: Fast compression
- `Zstd`: Best compression ratio

**Pro Tip:** Use Snappy for general-purpose analytics, Zstd for archival storage.

### 4.4 Excel Operations

#### Reading Excel Files

```rust
#[cfg(feature = "excel")]
use pandrs::io::{read_excel, ExcelReadOptions};
use pandrs::error::Result;

#[cfg(feature = "excel")]
fn read_excel_file() -> Result<()> {
    // Read first sheet
    let df = read_excel("data.xlsx", None)?;

    // Read specific sheet
    let df_sheet2 = read_excel("data.xlsx", Some("Sheet2"))?;

    Ok(())
}
```

#### Advanced Excel Reading

```rust
#[cfg(feature = "excel")]
use pandrs::io::ExcelReadOptions;
use pandrs::error::Result;

#[cfg(feature = "excel")]
fn read_excel_advanced() -> Result<()> {
    let options = ExcelReadOptions {
        sheet_name: Some("Sales Data".to_string()),
        skip_rows: 2,
        max_rows: Some(1000),
        preserve_formulas: true,
        read_named_ranges: true,
    };

    let df = read_excel_with_options("report.xlsx", options)?;

    Ok(())
}
```

#### Writing Excel Files

```rust
#[cfg(feature = "excel")]
use pandrs::io::ExcelWriteOptions;
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "excel")]
fn write_excel_file(df: &DataFrame) -> Result<()> {
    let options = ExcelWriteOptions {
        sheet_name: "Results".to_string(),
        preserve_formulas: false,
        apply_formatting: true,
        write_named_ranges: false,
        protect_sheets: false,
        optimize_large_files: true,
    };

    df.to_excel_with_options("output.xlsx", options)?;

    Ok(())
}
```

#### Multi-Sheet Excel Workbooks

```rust
#[cfg(feature = "excel")]
use pandrs::io::ExcelWorkbook;
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "excel")]
fn create_multi_sheet_workbook(
    sales_df: &DataFrame,
    products_df: &DataFrame,
    customers_df: &DataFrame
) -> Result<()> {
    let mut workbook = ExcelWorkbook::new("multi_sheet.xlsx")?;

    workbook.add_sheet("Sales", sales_df)?;
    workbook.add_sheet("Products", products_df)?;
    workbook.add_sheet("Customers", customers_df)?;

    workbook.save()?;

    Ok(())
}
```

### 4.5 SQL Database Operations

#### Reading from SQL

```rust
#[cfg(feature = "sql")]
use pandrs::io::{read_sql, DatabaseConnection};
use pandrs::error::Result;

#[cfg(feature = "sql")]
async fn read_from_database() -> Result<()> {
    let conn = DatabaseConnection::new_postgres(
        "postgresql://user:password@localhost/mydb"
    ).await?;

    // Read entire table
    let df = read_sql("SELECT * FROM sales", &conn).await?;

    // Read with query
    let filtered = read_sql(
        "SELECT * FROM sales WHERE amount > 1000",
        &conn
    ).await?;

    Ok(())
}
```

#### Supported Databases

- **PostgreSQL**: Full support with advanced features
- **MySQL/MariaDB**: Complete compatibility
- **SQLite**: Embedded database support

#### Connection Pooling

```rust
#[cfg(feature = "sql")]
use pandrs::io::{PoolConfig, DatabaseConnection};
use pandrs::error::Result;
use std::time::Duration;

#[cfg(feature = "sql")]
async fn connection_pooling() -> Result<()> {
    let pool_config = PoolConfig {
        max_connections: 10,
        min_connections: 2,
        connection_timeout: Duration::from_secs(30),
        idle_timeout: Some(Duration::from_secs(600)),
        max_lifetime: Some(Duration::from_secs(3600)),
    };

    let conn = DatabaseConnection::new_postgres_with_pool(
        "postgresql://user:password@localhost/mydb",
        pool_config
    ).await?;

    Ok(())
}
```

#### Writing to SQL

```rust
#[cfg(feature = "sql")]
use pandrs::io::{SqlWriteOptions, WriteMode, InsertMethod};
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "sql")]
async fn write_to_database(df: &DataFrame) -> Result<()> {
    let conn = DatabaseConnection::new_postgres(
        "postgresql://user:password@localhost/mydb"
    ).await?;

    let options = SqlWriteOptions {
        table_name: "sales_data".to_string(),
        write_mode: WriteMode::Append, // or Replace, CreateNew
        insert_method: InsertMethod::Batch,
        batch_size: 1000,
        create_table: true,
    };

    df.to_sql(&conn, options).await?;

    Ok(())
}
```

#### Transactions

```rust
#[cfg(feature = "sql")]
use pandrs::io::DatabaseConnection;
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "sql")]
async fn transaction_example(df: &DataFrame) -> Result<()> {
    let mut conn = DatabaseConnection::new_postgres(
        "postgresql://user:password@localhost/mydb"
    ).await?;

    // Start transaction
    let mut tx = conn.begin_transaction().await?;

    // Perform operations
    df.to_sql_transaction(&mut tx, "temp_table").await?;

    // Commit or rollback
    tx.commit().await?;
    // or: tx.rollback().await?;

    Ok(())
}
```

### 4.6 Arrow Format

Arrow provides zero-copy interoperability.

```rust
#[cfg(feature = "distributed")]
use pandrs::arrow_integration::{to_arrow, from_arrow};
use pandrs::DataFrame;
use pandrs::error::Result;

#[cfg(feature = "distributed")]
fn arrow_conversion(df: &DataFrame) -> Result<()> {
    // Convert to Arrow RecordBatch
    let arrow_batch = to_arrow(df)?;

    // Convert back to DataFrame
    let df_restored = from_arrow(&arrow_batch)?;

    Ok(())
}
```

---

## Chapter 5: Time Series Analysis

### 5.1 Time Series Creation and Indexing

#### Creating Time Series

```rust
use pandrs::time_series::{TimeSeries, TimeSeriesBuilder, Frequency};
use chrono::{Utc, TimeZone, Duration};
use pandrs::error::Result;

fn create_time_series() -> Result<()> {
    let mut builder = TimeSeriesBuilder::new();

    // Add time points
    for i in 0..30 {
        let timestamp = Utc.ymd(2024, 1, 1).and_hms(0, 0, 0) + Duration::days(i);
        let value = 100.0 + (i as f64 * 2.5);
        builder = builder.add_point(timestamp, value);
    }

    let ts = builder
        .frequency(Frequency::Daily)
        .build()?;

    println!("Created time series with {} points", ts.len());

    Ok(())
}
```

#### DateTime Indexing

```rust
use pandrs::{DataFrame, Series};
use pandrs::index::DateTimeIndex;
use chrono::{Utc, TimeZone};
use pandrs::error::Result;

fn datetime_indexing() -> Result<()> {
    let dates = vec![
        Utc.ymd(2024, 1, 1).and_hms(0, 0, 0),
        Utc.ymd(2024, 1, 2).and_hms(0, 0, 0),
        Utc.ymd(2024, 1, 3).and_hms(0, 0, 0),
    ];

    let values = vec![100.0, 102.5, 98.7];

    let mut df = DataFrame::new();
    df.add_column("date".to_string(), Series::new(dates, Some("date".to_string()))?)?;
    df.add_column("value".to_string(), Series::new(values, Some("value".to_string()))?)?;

    // Set datetime index
    df.set_index("date")?;

    Ok(())
}
```

### 5.2 Resampling and Frequency Conversion

```rust
use pandrs::time_series::{TimeSeries, Frequency};
use pandrs::error::Result;

fn resample_time_series(ts: &TimeSeries) -> Result<()> {
    // Upsample to hourly (interpolate missing values)
    let hourly = ts.resample(Frequency::Hourly, "linear")?;

    // Downsample to weekly (aggregate)
    let weekly = ts.resample_agg(Frequency::Weekly, "mean")?;

    // Custom resampling with multiple aggregations
    let monthly = ts.resample_agg(Frequency::Monthly, "sum")?;

    println!("Original: {} points", ts.len());
    println!("Hourly: {} points", hourly.len());
    println!("Weekly: {} points", weekly.len());

    Ok(())
}
```

**Resampling Methods:**
- **Upsampling:** `linear`, `ffill` (forward fill), `bfill` (backward fill)
- **Downsampling:** `mean`, `sum`, `min`, `max`, `first`, `last`, `count`

### 5.3 Rolling Windows

```rust
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn rolling_window_operations(ts: &TimeSeries) -> Result<()> {
    // 7-day rolling mean
    let rolling_mean = ts.rolling(7, "mean")?;

    // 14-day rolling standard deviation
    let rolling_std = ts.rolling(14, "std")?;

    // 30-day rolling maximum
    let rolling_max = ts.rolling(30, "max")?;

    // Exponentially weighted moving average
    let ewma = ts.ewm(0.2)?; // alpha = 0.2

    Ok(())
}
```

### 5.4 Forecasting

#### ARIMA Forecasting

```rust
use pandrs::time_series::{TimeSeries, ArimaForecaster, Forecaster};
use pandrs::error::Result;

fn arima_forecast(ts: &TimeSeries) -> Result<()> {
    // Create ARIMA(1,1,1) model
    let mut forecaster = ArimaForecaster::new(1, 1, 1)?;

    // Fit the model
    forecaster.fit(ts)?;

    // Forecast next 30 periods
    let forecast = forecaster.predict(30)?;

    // Get forecast with confidence intervals
    let forecast_with_ci = forecaster.predict_with_intervals(30, 0.95)?;

    println!("Forecast for next 30 periods:");
    for (i, value) in forecast.values().iter().enumerate() {
        println!("  Period {}: {:.2}", i + 1, value);
    }

    Ok(())
}
```

#### SARIMA (Seasonal ARIMA)

```rust
use pandrs::time_series::{SarimaForecaster, Forecaster};
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn sarima_forecast(ts: &TimeSeries) -> Result<()> {
    // SARIMA(1,1,1)(1,1,1,12) - monthly seasonality
    let mut forecaster = SarimaForecaster::new(
        1, 1, 1,  // ARIMA parameters
        1, 1, 1,  // Seasonal parameters
        12        // Seasonal period
    )?;

    forecaster.fit(ts)?;
    let forecast = forecaster.predict(24)?; // 2 years ahead

    Ok(())
}
```

#### Automatic Model Selection

```rust
use pandrs::time_series::{AutoArima, ModelSelectionCriterion};
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn auto_arima_forecast(ts: &TimeSeries) -> Result<()> {
    // Automatically find best ARIMA parameters
    let mut auto_arima = AutoArima::new()
        .criterion(ModelSelectionCriterion::AIC)
        .max_p(5)
        .max_q(5)
        .max_d(2)
        .seasonal(true)
        .seasonal_period(12);

    auto_arima.fit(ts)?;

    println!("Best model: {:?}", auto_arima.best_model());
    println!("AIC: {:.2}", auto_arima.best_aic());

    let forecast = auto_arima.predict(12)?;

    Ok(())
}
```

#### Exponential Smoothing

```rust
use pandrs::time_series::{ExponentialSmoothingForecaster, Forecaster};
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn exponential_smoothing(ts: &TimeSeries) -> Result<()> {
    let mut forecaster = ExponentialSmoothingForecaster::new(
        0.3,  // alpha (level)
        0.1,  // beta (trend)
        0.2   // gamma (seasonal)
    )?;

    forecaster.fit(ts)?;
    let forecast = forecaster.predict(12)?;

    Ok(())
}
```

### 5.5 Seasonal Decomposition

```rust
use pandrs::time_series::{SeasonalDecomposition, DecompositionMethod};
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn seasonal_decomposition(ts: &TimeSeries) -> Result<()> {
    let decomposer = SeasonalDecomposition::new(
        12,  // Period (e.g., 12 for monthly data)
        DecompositionMethod::Additive
    );

    let result = decomposer.decompose(ts)?;

    println!("Trend: {:?}", result.trend);
    println!("Seasonal: {:?}", result.seasonal);
    println!("Residual: {:?}", result.residual);

    // Multiplicative decomposition
    let multiplicative = SeasonalDecomposition::new(
        12,
        DecompositionMethod::Multiplicative
    );
    let mult_result = multiplicative.decompose(ts)?;

    Ok(())
}
```

### 5.6 Statistical Tests

#### Stationarity Testing

```rust
use pandrs::time_series::{AugmentedDickeyFullerTest, KwiatkowskiPhillipsSchmidtShinTest};
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn test_stationarity(ts: &TimeSeries) -> Result<()> {
    // Augmented Dickey-Fuller test
    let adf = AugmentedDickeyFullerTest::new();
    let adf_result = adf.test(ts)?;

    println!("ADF Statistic: {:.4}", adf_result.statistic);
    println!("p-value: {:.4}", adf_result.p_value);
    println!("Is stationary: {}", adf_result.is_stationary(0.05));

    // KPSS test
    let kpss = KwiatkowskiPhillipsSchmidtShinTest::new();
    let kpss_result = kpss.test(ts)?;

    println!("KPSS Statistic: {:.4}", kpss_result.statistic);

    Ok(())
}
```

#### Seasonality Detection

```rust
use pandrs::time_series::SeasonalityAnalysis;
use pandrs::time_series::TimeSeries;
use pandrs::error::Result;

fn detect_seasonality(ts: &TimeSeries) -> Result<()> {
    let analyzer = SeasonalityAnalysis::new();

    let result = analyzer.detect(ts)?;

    println!("Has seasonality: {}", result.has_seasonality);
    println!("Dominant period: {:?}", result.dominant_period);
    println!("Strength: {:.2}", result.strength);

    Ok(())
}
```

---

## Chapter 6: Machine Learning

### 6.1 Decision Trees

#### Classification

```rust
use pandrs::ml::models::tree::{DecisionTreeClassifier, DecisionTreeConfig, SplitCriterion};
use pandrs::ml::models::{SupervisedModel, train_test_split};
use pandrs::DataFrame;
use pandrs::error::Result;

fn decision_tree_classification(df: &DataFrame) -> Result<()> {
    // Split data
    let (train_df, test_df) = train_test_split(df, 0.3, true, Some(42))?;

    // Configure decision tree
    let config = DecisionTreeConfig {
        max_depth: Some(5),
        min_samples_split: 2,
        min_samples_leaf: 1,
        criterion: SplitCriterion::Gini,
        max_features: None,
    };

    let mut clf = DecisionTreeClassifier::new(config);

    // Train
    clf.fit(&train_df, "target")?;

    // Predict
    let predictions = clf.predict(&test_df)?;

    // Feature importance
    let importance = clf.feature_importance();
    println!("Feature importance: {:?}", importance);

    Ok(())
}
```

#### Regression

```rust
use pandrs::ml::models::tree::{DecisionTreeRegressor, DecisionTreeConfig, SplitCriterion};
use pandrs::ml::models::SupervisedModel;
use pandrs::DataFrame;
use pandrs::error::Result;

fn decision_tree_regression(df: &DataFrame) -> Result<()> {
    let config = DecisionTreeConfig {
        max_depth: Some(10),
        min_samples_split: 5,
        min_samples_leaf: 2,
        criterion: SplitCriterion::MSE,
        max_features: None,
    };

    let mut reg = DecisionTreeRegressor::new(config);
    reg.fit(df, "price")?;

    let predictions = reg.predict(df)?;

    Ok(())
}
```

### 6.2 Random Forests

```rust
use pandrs::ml::models::ensemble::{RandomForestClassifier, RandomForestConfig};
use pandrs::ml::models::SupervisedModel;
use pandrs::DataFrame;
use pandrs::error::Result;

fn random_forest_example(df: &DataFrame) -> Result<()> {
    let config = RandomForestConfig {
        n_estimators: 100,
        max_depth: Some(10),
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_features: None,
        bootstrap: true,
        n_jobs: None, // Use all available cores
        random_seed: Some(42),
    };

    let mut rf = RandomForestClassifier::new(config);

    // Train
    rf.fit(df, "target")?;

    // Predict with probabilities
    let predictions = rf.predict(df)?;
    let probabilities = rf.predict_proba(df)?;

    // Feature importance (averaged across trees)
    let importance = rf.feature_importance();
    println!("Top features: {:?}", importance);

    // Out-of-bag score
    let oob_score = rf.oob_score()?;
    println!("OOB Score: {:.4}", oob_score);

    Ok(())
}
```

### 6.3 Gradient Boosting

```rust
use pandrs::ml::models::ensemble::{GradientBoostingClassifier, GradientBoostingConfig};
use pandrs::ml::models::SupervisedModel;
use pandrs::DataFrame;
use pandrs::error::Result;

fn gradient_boosting_example(df: &DataFrame) -> Result<()> {
    let config = GradientBoostingConfig {
        n_estimators: 100,
        learning_rate: 0.1,
        max_depth: 3,
        min_samples_split: 2,
        min_samples_leaf: 1,
        subsample: 0.8,
        max_features: None,
        random_seed: Some(42),
    };

    let mut gb = GradientBoostingClassifier::new(config);

    // Train with early stopping
    gb.fit_with_validation(df, "target", 0.2, 10)?; // 20% validation, 10 rounds patience

    let predictions = gb.predict(df)?;

    // Training history
    let train_loss = gb.training_loss();
    let val_loss = gb.validation_loss();

    println!("Best iteration: {}", gb.best_iteration());

    Ok(())
}
```

### 6.4 Neural Networks

#### Classification Network

```rust
use pandrs::ml::models::neural::{MLPClassifier, MLPConfigBuilder, Activation};
use pandrs::ml::models::SupervisedModel;
use pandrs::DataFrame;
use pandrs::error::Result;

fn neural_network_classification(df: &DataFrame) -> Result<()> {
    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![64, 32, 16])  // 3 hidden layers
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Softmax)
        .learning_rate(0.001)
        .n_epochs(100)
        .batch_size(32)
        .early_stopping_patience(Some(10))
        .l2_regularization(0.01)
        .dropout_rate(0.2)
        .random_seed(42)
        .verbose(true)
        .build();

    let mut mlp = MLPClassifier::new(config);

    // Train
    mlp.fit(df, "target")?;

    // Predictions with probabilities
    let predictions = mlp.predict(df)?;
    let proba = mlp.predict_proba(df)?;

    // Training history
    let loss_history = mlp.training_loss_history();
    println!("Final loss: {:.6}", loss_history.last().unwrap());

    Ok(())
}
```

#### Regression Network

```rust
use pandrs::ml::models::neural::{MLPRegressor, MLPConfigBuilder, Activation, LossFunction};
use pandrs::ml::models::SupervisedModel;
use pandrs::DataFrame;
use pandrs::error::Result;

fn neural_network_regression(df: &DataFrame) -> Result<()> {
    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![128, 64, 32])
        .hidden_activation(Activation::ReLU)
        .output_activation(Activation::Linear)
        .loss_function(LossFunction::MSE)
        .learning_rate(0.001)
        .n_epochs(200)
        .batch_size(16)
        .build();

    let mut mlp = MLPRegressor::new(config);
    mlp.fit(df, "price")?;

    let predictions = mlp.predict(df)?;

    Ok(())
}
```

**Available Activation Functions:**
- `Linear`, `Sigmoid`, `Tanh`, `ReLU`, `LeakyReLU`, `ELU`, `Softmax`

**Available Loss Functions:**
- `MSE`, `MAE`, `Huber`, `BinaryCrossentropy`, `CategoricalCrossentropy`

### 6.5 Model Evaluation

#### Classification Metrics

```rust
use pandrs::ml::metrics::classification::{
    accuracy_score, precision_score, recall_score, f1_score
};
use pandrs::Series;
use pandrs::error::Result;

fn evaluate_classification(y_true: &Series, y_pred: &Series) -> Result<()> {
    let accuracy = accuracy_score(y_true, y_pred)?;
    let precision = precision_score(y_true, y_pred, "weighted")?;
    let recall = recall_score(y_true, y_pred, "weighted")?;
    let f1 = f1_score(y_true, y_pred, "weighted")?;

    println!("Accuracy: {:.4}", accuracy);
    println!("Precision: {:.4}", precision);
    println!("Recall: {:.4}", recall);
    println!("F1 Score: {:.4}", f1);

    Ok(())
}
```

#### Regression Metrics

```rust
use pandrs::ml::metrics::regression::{
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score
};
use pandrs::Series;
use pandrs::error::Result;

fn evaluate_regression(y_true: &Series, y_pred: &Series) -> Result<()> {
    let mae = mean_absolute_error(y_true, y_pred)?;
    let mse = mean_squared_error(y_true, y_pred)?;
    let rmse = root_mean_squared_error(y_true, y_pred)?;
    let r2 = r2_score(y_true, y_pred)?;

    println!("MAE: {:.4}", mae);
    println!("MSE: {:.4}", mse);
    println!("RMSE: {:.4}", rmse);
    println!("R² Score: {:.4}", r2);

    Ok(())
}
```

#### Cross-Validation

```rust
use pandrs::ml::models::CrossValidation;
use pandrs::ml::models::tree::DecisionTreeClassifier;
use pandrs::DataFrame;
use pandrs::error::Result;

fn cross_validation_example(df: &DataFrame) -> Result<()> {
    let mut model = DecisionTreeClassifier::default();

    // 5-fold cross-validation
    let cv = CrossValidation::new(5, Some(42));
    let scores = cv.cross_val_score(&mut model, df, "target")?;

    println!("Cross-validation scores: {:?}", scores);
    println!("Mean score: {:.4}", scores.iter().sum::<f64>() / scores.len() as f64);

    Ok(())
}
```

### 6.6 Feature Engineering

#### Standard Scaling

```rust
use pandrs::ml::preprocessing::StandardScaler;
use pandrs::DataFrame;
use pandrs::error::Result;

fn standard_scaling(df: &DataFrame) -> Result<()> {
    let mut scaler = StandardScaler::new();

    // Fit and transform
    let scaled_df = scaler.fit_transform(df)?;

    // Transform new data
    let new_scaled = scaler.transform(df)?;

    // Inverse transform
    let original = scaler.inverse_transform(&scaled_df)?;

    Ok(())
}
```

#### Min-Max Scaling

```rust
use pandrs::ml::preprocessing::MinMaxScaler;
use pandrs::DataFrame;
use pandrs::error::Result;

fn minmax_scaling(df: &DataFrame) -> Result<()> {
    let mut scaler = MinMaxScaler::new(0.0, 1.0); // Scale to [0, 1]

    let scaled_df = scaler.fit_transform(df)?;

    Ok(())
}
```

#### One-Hot Encoding

```rust
use pandrs::ml::preprocessing::OneHotEncoder;
use pandrs::DataFrame;
use pandrs::error::Result;

fn one_hot_encoding(df: &DataFrame) -> Result<()> {
    let mut encoder = OneHotEncoder::new(vec!["category", "region"]);

    let encoded_df = encoder.fit_transform(df)?;

    println!("Original columns: {:?}", df.column_names());
    println!("Encoded columns: {:?}", encoded_df.column_names());

    Ok(())
}
```

#### Polynomial Features

```rust
use pandrs::ml::preprocessing::PolynomialFeatures;
use pandrs::DataFrame;
use pandrs::error::Result;

fn polynomial_features(df: &DataFrame) -> Result<()> {
    let mut poly = PolynomialFeatures::new(2, true); // degree=2, include_bias=true

    let poly_df = poly.fit_transform(df)?;

    println!("Original features: {}", df.ncols());
    println!("Polynomial features: {}", poly_df.ncols());

    Ok(())
}
```

#### Imputation (Missing Value Handling)

```rust
use pandrs::ml::preprocessing::{Imputer, ImputeStrategy};
use pandrs::DataFrame;
use pandrs::error::Result;

fn impute_missing_values(df: &DataFrame) -> Result<()> {
    // Mean imputation
    let mean_imputer = Imputer::new(ImputeStrategy::Mean);
    let mean_imputed = mean_imputer.fit_transform(df)?;

    // Median imputation
    let median_imputer = Imputer::new(ImputeStrategy::Median);
    let median_imputed = median_imputer.fit_transform(df)?;

    // Constant imputation
    let const_imputer = Imputer::new(ImputeStrategy::Constant(0.0));
    let const_imputed = const_imputer.fit_transform(df)?;

    // Forward fill
    let ffill_imputer = Imputer::new(ImputeStrategy::ForwardFill);
    let ffill_imputed = ffill_imputer.fit_transform(df)?;

    Ok(())
}
```

---

## Chapter 7: Security & Access Control

### 7.1 JWT Authentication

#### Basic JWT Usage

```rust
use pandrs::auth::{encode_jwt, decode_jwt, verify_jwt, JwtConfig, TokenClaims};
use pandrs::error::Result;
use std::time::Duration;

fn jwt_authentication() -> Result<()> {
    // Configure JWT
    let config = JwtConfig {
        secret: "your-secret-key".to_string(),
        issuer: "pandrs-app".to_string(),
        audience: "pandrs-users".to_string(),
        expiration: Duration::from_secs(3600), // 1 hour
        algorithm: "HS256".to_string(),
    };

    // Create token claims
    let claims = TokenClaims {
        sub: "user123".to_string(),
        iss: config.issuer.clone(),
        aud: config.audience.clone(),
        exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp() as u64,
        iat: chrono::Utc::now().timestamp() as u64,
        custom: std::collections::HashMap::new(),
    };

    // Encode JWT
    let token = encode_jwt(&claims, &config)?;
    println!("Generated token: {}", token);

    // Verify and decode
    let decoded_claims = verify_jwt(&token, &config)?;
    println!("User ID: {}", decoded_claims.sub);

    Ok(())
}
```

#### Custom Claims

```rust
use pandrs::auth::{TokenClaims, encode_jwt, JwtConfig};
use pandrs::error::Result;
use std::collections::HashMap;

fn custom_jwt_claims(config: &JwtConfig) -> Result<String> {
    let mut custom_data = HashMap::new();
    custom_data.insert("role".to_string(), "admin".to_string());
    custom_data.insert("department".to_string(), "engineering".to_string());
    custom_data.insert("permissions".to_string(), "read,write,delete".to_string());

    let claims = TokenClaims {
        sub: "admin_user".to_string(),
        iss: config.issuer.clone(),
        aud: config.audience.clone(),
        exp: (chrono::Utc::now() + chrono::Duration::hours(24)).timestamp() as u64,
        iat: chrono::Utc::now().timestamp() as u64,
        custom: custom_data,
    };

    let token = encode_jwt(&claims, config)?;

    Ok(token)
}
```

### 7.2 OAuth 2.0

```rust
use pandrs::auth::{AuthManager, OAuthConfig, OAuthGrantType, create_shared_auth_manager};
use pandrs::error::Result;

async fn oauth_authentication() -> Result<()> {
    let oauth_config = OAuthConfig {
        client_id: "your-client-id".to_string(),
        client_secret: "your-client-secret".to_string(),
        redirect_uri: "http://localhost:8080/callback".to_string(),
        authorization_endpoint: "https://oauth.provider.com/authorize".to_string(),
        token_endpoint: "https://oauth.provider.com/token".to_string(),
        scopes: vec!["read".to_string(), "write".to_string()],
        grant_type: OAuthGrantType::AuthorizationCode,
    };

    let auth_manager = create_shared_auth_manager(oauth_config)?;

    // Generate authorization URL
    let auth_url = auth_manager.lock()
        .map_err(|e| pandrs::error::Error::LockError(e.to_string()))?
        .generate_auth_url()?;

    println!("Authorization URL: {}", auth_url);

    // After user authorizes, exchange code for token
    // let tokens = auth_manager.lock().unwrap().exchange_code("auth_code").await?;

    Ok(())
}
```

### 7.3 API Key Management

```rust
use pandrs::auth::{ApiKeyManager, ScopedApiKey};
use pandrs::error::Result;
use std::collections::HashSet;

fn api_key_management() -> Result<()> {
    let mut key_manager = ApiKeyManager::new("encryption-key".to_string());

    // Create API key with scopes
    let mut scopes = HashSet::new();
    scopes.insert("read:data".to_string());
    scopes.insert("write:data".to_string());

    let api_key = key_manager.create_key(
        "user123".to_string(),
        scopes,
        Some(chrono::Duration::days(30))
    )?;

    println!("Generated API Key: {}", api_key.key);

    // Validate API key
    let is_valid = key_manager.validate_key(&api_key.key)?;
    println!("Key is valid: {}", is_valid);

    // Check permissions
    let has_permission = key_manager.check_permission(&api_key.key, "read:data")?;
    println!("Has read permission: {}", has_permission);

    // Revoke API key
    key_manager.revoke_key(&api_key.key)?;

    Ok(())
}
```

### 7.4 Role-Based Access Control (RBAC)

```rust
use pandrs::auth::AuthManager;
use pandrs::multitenancy::{Permission, TenantManager, TenantConfig, create_shared_manager};
use pandrs::error::Result;
use std::collections::HashSet;

fn rbac_example() -> Result<()> {
    let tenant_manager = create_shared_manager();

    // Create tenant
    let config = TenantConfig {
        name: "ACME Corp".to_string(),
        max_datasets: 100,
        max_storage_bytes: 10_737_418_240, // 10 GB
        max_users: 50,
        features: vec!["analytics".to_string(), "ml".to_string()],
    };

    let tenant_id = tenant_manager.lock()
        .map_err(|e| pandrs::error::Error::LockError(e.to_string()))?
        .create_tenant(config)?;

    // Define role permissions
    let admin_permissions = vec![
        Permission::Read,
        Permission::Write,
        Permission::Delete,
        Permission::Create,
        Permission::Share,
        Permission::Admin,
    ].into_iter().collect::<HashSet<_>>();

    let analyst_permissions = vec![
        Permission::Read,
        Permission::Write,
        Permission::Create,
    ].into_iter().collect::<HashSet<_>>();

    // Check permissions
    fn has_permission(user_permissions: &HashSet<Permission>, required: Permission) -> bool {
        user_permissions.contains(&required)
    }

    println!("Admin can delete: {}", has_permission(&admin_permissions, Permission::Delete));
    println!("Analyst can delete: {}", has_permission(&analyst_permissions, Permission::Delete));

    Ok(())
}
```

### 7.5 Multi-Tenancy

```rust
use pandrs::multitenancy::{TenantManager, TenantConfig, DatasetMetadata, IsolationContext};
use pandrs::DataFrame;
use pandrs::error::Result;

fn multi_tenancy_example() -> Result<()> {
    let mut tenant_manager = TenantManager::new();

    // Create multiple tenants
    let tenant1_config = TenantConfig {
        name: "Company A".to_string(),
        max_datasets: 50,
        max_storage_bytes: 5_368_709_120, // 5 GB
        max_users: 25,
        features: vec!["basic".to_string()],
    };

    let tenant2_config = TenantConfig {
        name: "Company B".to_string(),
        max_datasets: 200,
        max_storage_bytes: 21_474_836_480, // 20 GB
        max_users: 100,
        features: vec!["basic".to_string(), "advanced".to_string(), "ml".to_string()],
    };

    let tenant1_id = tenant_manager.create_tenant(tenant1_config)?;
    let tenant2_id = tenant_manager.create_tenant(tenant2_config)?;

    // Register dataset for tenant
    let dataset_meta = DatasetMetadata {
        name: "sales_data".to_string(),
        size_bytes: 1_048_576, // 1 MB
        row_count: 10000,
        column_count: 15,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    let dataset_id = tenant_manager.register_dataset(tenant1_id.clone(), dataset_meta)?;

    // Access dataset with tenant context
    let context = IsolationContext {
        tenant_id: tenant1_id.clone(),
        user_id: "user123".to_string(),
        permissions: vec![pandrs::multitenancy::Permission::Read].into_iter().collect(),
    };

    // Verify access
    let can_access = tenant_manager.check_dataset_access(&tenant1_id, &dataset_id)?;
    println!("Can access dataset: {}", can_access);

    // Get tenant usage statistics
    let usage = tenant_manager.get_tenant_usage(&tenant1_id)?;
    println!("Datasets: {}/{}", usage.dataset_count, usage.max_datasets);
    println!("Storage: {} bytes", usage.storage_bytes);

    Ok(())
}
```

### 7.6 Audit Logging

```rust
use pandrs::audit::{AuditLogger, AuditConfig, EventCategory, LogLevel, LogDestination};
use pandrs::error::Result;

fn audit_logging_example() -> Result<()> {
    let config = AuditConfig {
        enabled: true,
        log_level: LogLevel::Info,
        destination: LogDestination::File("audit.log".to_string()),
        buffer_size: 1000,
        flush_interval_secs: 60,
        include_stacktrace: true,
    };

    let mut logger = AuditLogger::new(config);

    // Log data access
    logger.log(
        EventCategory::DataAccess,
        "User accessed sales dataset".to_string(),
        std::collections::HashMap::from([
            ("user_id".to_string(), "user123".to_string()),
            ("dataset".to_string(), "sales_2024".to_string()),
            ("action".to_string(), "read".to_string()),
        ])
    )?;

    // Log authentication event
    logger.log(
        EventCategory::Authentication,
        "User login successful".to_string(),
        std::collections::HashMap::from([
            ("user_id".to_string(), "user123".to_string()),
            ("ip_address".to_string(), "192.168.1.100".to_string()),
        ])
    )?;

    // Log security event
    logger.log(
        EventCategory::Security,
        "Failed authentication attempt".to_string(),
        std::collections::HashMap::from([
            ("user_id".to_string(), "unknown".to_string()),
            ("attempts".to_string(), "5".to_string()),
        ])
    )?;

    // Get audit statistics
    let stats = logger.stats();
    println!("Total events logged: {}", stats.total_events);
    println!("Events by category: {:?}", stats.events_by_category);

    Ok(())
}
```

---

## Chapter 8: Real-Time Analytics

### 8.1 Metrics Collection

#### Basic Metrics

```rust
use pandrs::analytics::{Dashboard, DashboardConfig, MetricType};
use pandrs::error::Result;

fn basic_metrics_collection() -> Result<()> {
    let config = DashboardConfig {
        collection_interval_secs: 10,
        retention_hours: 24,
        enable_alerts: true,
        export_metrics: false,
    };

    let mut dashboard = Dashboard::new(config);

    // Record counter (cumulative)
    dashboard.record_metric("requests_total", 1.0, MetricType::Counter)?;

    // Record gauge (current value)
    dashboard.record_metric("active_connections", 42.0, MetricType::Gauge)?;

    // Record histogram (distribution)
    dashboard.record_metric("request_duration_ms", 125.5, MetricType::Histogram)?;

    // Record timer
    dashboard.record_metric("query_time_ms", 87.3, MetricType::Timer)?;

    Ok(())
}
```

#### Operation Tracking

```rust
use pandrs::analytics::{Dashboard, OperationCategory};
use pandrs::error::Result;
use std::time::Instant;

fn track_operations(dashboard: &mut Dashboard) -> Result<()> {
    // Track DataFrame operation
    let start = Instant::now();

    // ... perform operation ...

    let duration = start.elapsed();

    dashboard.record_operation(
        "groupby_aggregation".to_string(),
        duration,
        OperationCategory::DataFrame,
        true // success
    )?;

    // Track ML operation
    let ml_start = Instant::now();
    // ... train model ...
    let ml_duration = ml_start.elapsed();

    dashboard.record_operation(
        "random_forest_training".to_string(),
        ml_duration,
        OperationCategory::MachineLearning,
        true
    )?;

    Ok(())
}
```

#### Scoped Timers

```rust
use pandrs::analytics::{Dashboard, ScopedTimer};
use pandrs::error::Result;

fn scoped_timer_example(dashboard: &mut Dashboard) -> Result<()> {
    {
        // Timer automatically records when it goes out of scope
        let _timer = ScopedTimer::new(dashboard, "data_loading");

        // Load data...
        std::thread::sleep(std::time::Duration::from_millis(100));

    } // Timer records here

    {
        let _timer = ScopedTimer::new(dashboard, "data_processing");
        // Process data...
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
```

### 8.2 Dashboard Setup

```rust
use pandrs::analytics::{Dashboard, DashboardConfig, global_dashboard, init_global_dashboard};
use pandrs::error::Result;

fn setup_dashboard() -> Result<()> {
    let config = DashboardConfig {
        collection_interval_secs: 10,
        retention_hours: 24,
        enable_alerts: true,
        export_metrics: true,
    };

    // Initialize global dashboard
    init_global_dashboard(config)?;

    // Access global dashboard
    let dashboard = global_dashboard();

    // Record metrics globally
    pandrs::analytics::record_global("api_requests", 1.0, MetricType::Counter)?;

    Ok(())
}
```

### 8.3 Alert Configuration

#### Creating Alert Rules

```rust
use pandrs::analytics::{
    AlertManager, AlertRule, AlertMetric, AlertSeverity, ThresholdOperator
};
use pandrs::error::Result;

fn configure_alerts() -> Result<()> {
    let mut alert_manager = AlertManager::new();

    // CPU usage alert
    let cpu_rule = AlertRule {
        name: "high_cpu_usage".to_string(),
        metric: AlertMetric::Custom("cpu_usage_percent".to_string()),
        threshold: 80.0,
        operator: ThresholdOperator::GreaterThan,
        severity: AlertSeverity::Warning,
        description: "CPU usage exceeded 80%".to_string(),
    };

    alert_manager.add_rule(cpu_rule);

    // Memory alert
    let memory_rule = AlertRule {
        name: "high_memory_usage".to_string(),
        metric: AlertMetric::Custom("memory_usage_mb".to_string()),
        threshold: 1024.0,
        operator: ThresholdOperator::GreaterThan,
        severity: AlertSeverity::Critical,
        description: "Memory usage exceeded 1GB".to_string(),
    };

    alert_manager.add_rule(memory_rule);

    // Error rate alert
    let error_rule = AlertRule {
        name: "high_error_rate".to_string(),
        metric: AlertMetric::Custom("error_rate".to_string()),
        threshold: 0.05,
        operator: ThresholdOperator::GreaterThan,
        severity: AlertSeverity::Critical,
        description: "Error rate exceeded 5%".to_string(),
    };

    alert_manager.add_rule(error_rule);

    Ok(())
}
```

#### Custom Alert Handlers

```rust
use pandrs::analytics::{AlertHandler, ActiveAlert};
use pandrs::error::Result;

struct EmailAlertHandler {
    smtp_server: String,
    recipients: Vec<String>,
}

impl AlertHandler for EmailAlertHandler {
    fn handle(&self, alert: &ActiveAlert) -> Result<()> {
        println!("Sending email alert: {} - {}", alert.severity, alert.message);
        // Send email via SMTP...
        Ok(())
    }
}

fn custom_alert_handler() -> Result<()> {
    let handler = EmailAlertHandler {
        smtp_server: "smtp.example.com".to_string(),
        recipients: vec!["admin@example.com".to_string()],
    };

    // Use handler with alert manager
    // alert_manager.add_handler(Box::new(handler));

    Ok(())
}
```

### 8.4 Performance Monitoring

```rust
use pandrs::analytics::{Dashboard, ResourceSnapshot};
use pandrs::error::Result;

fn performance_monitoring(dashboard: &Dashboard) -> Result<()> {
    // Get current resource snapshot
    let snapshot = dashboard.resource_snapshot();

    println!("=== Resource Usage ===");
    println!("CPU Usage: {:.2}%", snapshot.cpu_usage);
    println!("Memory Usage: {} MB", snapshot.memory_usage_mb);
    println!("Active Threads: {}", snapshot.active_threads);
    println!("Total Operations: {}", snapshot.total_operations);

    // Get metrics statistics
    let metric_stats = dashboard.metric_stats("request_duration_ms")?;

    println!("\n=== Request Duration Stats ===");
    println!("Count: {}", metric_stats.count);
    println!("Mean: {:.2} ms", metric_stats.mean);
    println!("Min: {:.2} ms", metric_stats.min);
    println!("Max: {:.2} ms", metric_stats.max);
    println!("P50: {:.2} ms", metric_stats.p50);
    println!("P95: {:.2} ms", metric_stats.p95);
    println!("P99: {:.2} ms", metric_stats.p99);

    Ok(())
}
```

---

## Chapter 9: Best Practices

### 9.1 Performance Optimization Tips

#### Use OptimizedDataFrame

```rust
use pandrs::OptimizedDataFrame;

// ✅ Good: Use OptimizedDataFrame for large datasets
let mut df = OptimizedDataFrame::new();

// ❌ Avoid: Regular DataFrame for large data
// let mut df = DataFrame::new();
```

**Why?** OptimizedDataFrame uses:
- Columnar storage for better cache locality
- String pooling to reduce memory usage
- SIMD vectorization for numeric operations
- Parallel processing for aggregations

#### Enable Feature Flags Appropriately

```toml
# ✅ Good: Enable only needed features
[dependencies]
pandrs = { version = "0.2.0", features = ["stable", "parquet"] }

# ❌ Avoid: Enabling all features increases compile time
# pandrs = { version = "0.2.0", features = ["stable", "cuda", "wasm", "jit"] }
```

#### Batch Operations

```rust
use pandrs::OptimizedDataFrame;
use pandrs::error::Result;

fn batch_operations() -> Result<()> {
    let mut df = OptimizedDataFrame::new();

    // ✅ Good: Add multiple columns at once
    df.add_int_column("id", (1..=1000).collect())?;
    df.add_float_column("value", vec![0.0; 1000])?;

    // ❌ Avoid: Row-by-row operations in a loop
    // for i in 1..=1000 {
    //     // This is slow
    // }

    Ok(())
}
```

#### Use Lazy Evaluation

```rust
use pandrs::LazyFrame;
use pandrs::error::Result;

fn lazy_evaluation() -> Result<()> {
    let lazy = LazyFrame::scan_csv("large_file.csv")?
        .filter("age > 25")?
        .select(&["name", "age", "salary"])?
        .groupby(&["department"])?
        .agg(&[("salary", "mean")])?;

    // Operations are optimized and executed only when collected
    let result = lazy.collect()?;

    Ok(())
}
```

#### Predicate Pushdown for Parquet

```rust
#[cfg(feature = "parquet")]
use pandrs::io::read_parquet_with_predicate;
use pandrs::error::Result;

#[cfg(feature = "parquet")]
fn predicate_pushdown() -> Result<()> {
    // ✅ Good: Filter at read time (predicate pushdown)
    let df = read_parquet_with_predicate(
        "large_file.parquet",
        "age > 25 AND salary > 50000"
    )?;

    // ❌ Avoid: Loading everything then filtering
    // let df = read_parquet("large_file.parquet")?;
    // let filtered = df.filter("age > 25 AND salary > 50000")?;

    Ok(())
}
```

### 9.2 Memory Management

#### Chunk Large Files

```rust
use pandrs::io::read_csv_chunked;
use pandrs::error::Result;

fn process_large_csv() -> Result<()> {
    let chunk_size = 10000;

    // Process file in chunks
    for chunk in read_csv_chunked("large_file.csv", chunk_size, true)? {
        // Process each chunk
        let summary = chunk.describe()?;
        println!("Chunk summary: {:?}", summary);
    }

    Ok(())
}
```

#### Use DiskBasedDataFrame for Very Large Data

```rust
use pandrs::large::{DiskBasedDataFrame, DiskConfig};
use pandrs::error::Result;

fn handle_very_large_data() -> Result<()> {
    let config = DiskConfig {
        temp_dir: "/tmp/pandrs".to_string(),
        chunk_size: 1_000_000,
        compression: true,
    };

    let mut df = DiskBasedDataFrame::new(config)?;

    // Data automatically spills to disk when memory limit is reached
    // df.add_column(...)?;

    Ok(())
}
```

#### Drop Unused Data

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn drop_unused_data(df: &mut DataFrame) -> Result<()> {
    // ✅ Good: Drop columns you don't need
    df.drop_column("temporary_column")?;

    // ✅ Good: Drop rows with missing values if not needed
    let cleaned = df.dropna()?;

    Ok(())
}
```

### 9.3 Error Handling Patterns

#### Use Result Type

```rust
use pandrs::error::Result;
use pandrs::DataFrame;

// ✅ Good: Return Result
fn load_data() -> Result<DataFrame> {
    let df = pandrs::io::read_csv("data.csv", true)?;
    Ok(df)
}

// ❌ Avoid: Unwrapping without error handling
// fn load_data_bad() -> DataFrame {
//     pandrs::io::read_csv("data.csv", true).unwrap()
// }
```

#### Context-Aware Errors

```rust
use pandrs::error::{Result, Error};
use pandrs::DataFrame;

fn load_and_process() -> Result<DataFrame> {
    let df = pandrs::io::read_csv("data.csv", true)
        .map_err(|e| Error::InvalidInput(
            format!("Failed to load data.csv: {}", e)
        ))?;

    let filtered = df.filter("age > 0")
        .map_err(|e| Error::InvalidInput(
            format!("Filter operation failed: {}", e)
        ))?;

    Ok(filtered)
}
```

#### Match on Error Types

```rust
use pandrs::error::{Result, Error};
use pandrs::DataFrame;

fn handle_errors(df: &DataFrame) -> Result<()> {
    match df.column("missing_column") {
        Ok(col) => {
            println!("Column found: {:?}", col);
        }
        Err(Error::ColumnNotFound(col_name)) => {
            println!("Column '{}' does not exist, using default", col_name);
            // Handle gracefully
        }
        Err(e) => {
            return Err(e);
        }
    }

    Ok(())
}
```

### 9.4 Testing Strategies

#### Unit Tests for Data Operations

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pandrs::{DataFrame, Series};

    #[test]
    fn test_dataframe_creation() {
        let mut df = DataFrame::new();
        df.add_column("test".to_string(),
            Series::new(vec![1, 2, 3], Some("test".to_string())).unwrap()
        ).unwrap();

        assert_eq!(df.nrows(), 3);
        assert_eq!(df.ncols(), 1);
    }

    #[test]
    fn test_aggregation() {
        let mut df = DataFrame::new();
        df.add_column("values".to_string(),
            Series::new(vec![10, 20, 30], Some("values".to_string())).unwrap()
        ).unwrap();

        let col = df.column("values").unwrap();
        assert_eq!(col.sum(), 60);
        assert_eq!(col.mean().unwrap(), 20.0);
    }
}
```

#### Integration Tests

```rust
#[cfg(test)]
mod integration_tests {
    use pandrs::io::{read_csv, write_csv};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_csv_roundtrip() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");

        // Create test data
        let mut df = DataFrame::new();
        // ... add columns ...

        // Write
        df.to_csv(file_path.to_str().unwrap()).unwrap();

        // Read
        let loaded = read_csv(file_path.to_str().unwrap(), true).unwrap();

        // Verify
        assert_eq!(df.nrows(), loaded.nrows());
        assert_eq!(df.ncols(), loaded.ncols());
    }
}
```

#### Property-Based Testing

```rust
#[cfg(test)]
mod property_tests {
    use pandrs::Series;

    #[test]
    fn test_series_properties() {
        // Test that series operations preserve length
        let series = Series::new(vec![1, 2, 3, 4, 5], None).unwrap();
        let doubled = &series * 2;

        assert_eq!(series.len(), doubled.len());

        // Test commutativity of addition
        let a = Series::new(vec![1, 2, 3], None).unwrap();
        let b = Series::new(vec![4, 5, 6], None).unwrap();

        let sum1 = &a + &b;
        let sum2 = &b + &a;

        assert_eq!(sum1, sum2);
    }
}
```

### 9.5 Common Pitfalls

#### 1. Not Checking for Missing Values

```rust
use pandrs::Series;
use pandrs::error::Result;

fn handle_missing_values(series: &Series) -> Result<()> {
    // ✅ Good: Check for missing values
    if series.has_na() {
        println!("Warning: {} missing values found", series.na_count());
        let cleaned = series.dropna()?;
        // ... use cleaned series
    }

    // ❌ Avoid: Assuming no missing values
    // let mean = series.mean()?; // May fail if NA present

    Ok(())
}
```

#### 2. Column Name Typos

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn access_column(df: &DataFrame) -> Result<()> {
    // ✅ Good: Check column exists
    if df.has_column("age") {
        let age_col = df.column("age")?;
        // Use column
    } else {
        println!("Column 'age' not found");
    }

    // Or use match
    match df.column("age") {
        Ok(col) => { /* use column */ },
        Err(_) => println!("Column not found"),
    }

    Ok(())
}
```

#### 3. Type Mismatches

```rust
use pandrs::OptimizedDataFrame;
use pandrs::error::Result;

fn type_safe_operations(df: &OptimizedDataFrame) -> Result<()> {
    let col = df.column("age")?;

    // ✅ Good: Check type before operations
    if let Some(int_col) = col.as_int64() {
        let sum = int_col.sum();
        println!("Sum: {}", sum);
    } else {
        println!("Column is not Int64 type");
    }

    Ok(())
}
```

#### 4. Memory Issues with Large Joins

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn large_join(left: &DataFrame, right: &DataFrame) -> Result<()> {
    // ❌ Avoid: Joining on columns with many duplicates
    // This can create a huge result (cartesian product)
    // let result = left.join(right, "common_column", "inner")?;

    // ✅ Good: Pre-aggregate or filter before joining
    let left_grouped = left.groupby(vec!["key"])?.agg(
        std::collections::HashMap::from([
            ("value".to_string(), vec!["sum"])
        ])
    )?;

    let result = left_grouped.join(right, "key", "inner")?;

    Ok(())
}
```

#### 5. Not Using Appropriate Join Types

```rust
use pandrs::DataFrame;
use pandrs::error::Result;

fn choose_join_type(customers: &DataFrame, orders: &DataFrame) -> Result<()> {
    // ✅ Good: Use LEFT join to keep all customers, even without orders
    let all_customers = customers.join(orders, "customer_id", "left")?;

    // ✅ Good: Use INNER join when you only want matching records
    let customers_with_orders = customers.join(orders, "customer_id", "inner")?;

    // Think about what you need:
    // - INNER: Only matching records
    // - LEFT: All from left, matching from right
    // - RIGHT: All from right, matching from left
    // - OUTER: All records from both

    Ok(())
}
```

### 9.6 Production Checklist

Before deploying PandRS in production:

- [ ] **Error Handling**: All operations use `Result` and handle errors appropriately
- [ ] **Logging**: Configure audit logging for critical operations
- [ ] **Monitoring**: Set up analytics dashboard with alerts
- [ ] **Security**: Implement authentication and authorization
- [ ] **Testing**: Comprehensive unit and integration tests
- [ ] **Performance**: Profile and optimize hot paths
- [ ] **Memory**: Configure appropriate memory limits and disk spilling
- [ ] **Connection Pooling**: Use connection pools for database operations
- [ ] **Feature Flags**: Enable only necessary features to reduce binary size
- [ ] **Documentation**: Document your schema and data pipelines
- [ ] **Backup**: Implement data backup and recovery procedures
- [ ] **Versioning**: Use data versioning for reproducibility
- [ ] **Multi-tenancy**: Properly isolate tenant data if applicable

---

## Appendix A: Quick Reference

### Common Operations Cheat Sheet

```rust
// DataFrame creation
let mut df = DataFrame::new();
let mut opt_df = OptimizedDataFrame::new();

// I/O
let df = read_csv("file.csv", true)?;
df.to_csv("output.csv")?;

// Selection
let col = df.column("name")?;
let head = df.head(10);
let tail = df.tail(5);

// Filtering
let filtered = df.filter("age > 25")?;

// Sorting
let sorted = df.sort_values("age", true)?;

// GroupBy
let grouped = df.groupby(vec!["dept"])?.agg(agg_funcs)?;

// Joining
let joined = df1.join(&df2, "id", "inner")?;

// Missing values
series.fillna(0.0)?;
let cleaned = series.dropna()?;

// Statistics
let mean = series.mean()?;
let std = series.std()?;
```

### Performance Tips Summary

1. Use `OptimizedDataFrame` for large datasets
2. Enable `stable` feature for best balance
3. Use lazy evaluation when possible
4. Leverage predicate pushdown for Parquet
5. Batch operations instead of row-by-row
6. Use string pooling (automatic in OptimizedDataFrame)
7. Profile before optimizing
8. Consider disk-based DataFrames for very large data

---

## Appendix B: Additional Resources

### Documentation
- API Documentation: https://docs.rs/pandrs
- GitHub Repository: https://github.com/cool-japan/pandrs
- Examples: See `examples/` directory in the repository

### Related Projects
- Apache Arrow: Zero-copy data interchange
- DataFusion: Distributed query engine
- Polars: Similar DataFrame library (inspiration)

### Community
- Report issues: GitHub Issues
- Discussions: GitHub Discussions
- License: Apache-2.0

---

## Appendix C: Version History

### v0.2.0 (Current)
- Enhanced I/O capabilities (Excel, Parquet, SQL)
- Machine learning module expansion
- Security features (JWT, OAuth, RBAC)
- Real-time analytics dashboard
- Multi-tenancy support
- Time series forecasting improvements
- Performance optimizations

### v0.1.0
- Initial release
- Core DataFrame and Series functionality
- Basic I/O operations
- Statistical functions
- GroupBy operations

---

**End of User Guide**

For the most up-to-date information, please refer to the official documentation at https://docs.rs/pandrs.

Questions or feedback? Visit https://github.com/cool-japan/pandrs

# PandRS

[![Rust CI](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml/badge.svg)](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses/MIT)
[![Crate](https://img.shields.io/crates/v/pandrs.svg)](https://crates.io/crates/pandrs)

A high-performance DataFrame library for data analysis implemented in Rust. It has features and design inspired by Python's `pandas` library, combining fast data processing with type safety and distributed computing capabilities.

## 🚀 What's New

**Enhanced DataFrame Operations:**
- **Column Management**: New `rename_columns()` and `set_column_names()` methods for flexible column renaming
- **Enhanced I/O**: Real data extraction in Parquet/SQL operations with improved type safety
- **Distributed Processing**: Production-ready DataFusion integration with schema validation and fault tolerance
- **Python Bindings**: Complete feature coverage with optimized string pool integration

**Performance & Reliability:**
- **Comprehensive Testing**: 100+ integration tests covering all major features and edge cases
- **Schema Validation**: Type-safe distributed operations with compile-time validation
- **Fault Tolerance**: Checkpoint/recovery system for robust distributed processing
- **Memory Optimization**: Advanced string pool with up to 89.8% memory reduction

## 🏁 Quick Start

```rust
use pandrs::{DataFrame, OptimizedDataFrame, Column, StringColumn, Int64Column};
use std::collections::HashMap;

// Create DataFrame with column management
let mut df = DataFrame::new();
df.add_column("name".to_string(), 
    pandrs::series::Series::from_vec(vec!["Alice", "Bob", "Carol"], Some("name")))?;

// Rename columns with a mapping
let mut rename_map = HashMap::new();
rename_map.insert("name".to_string(), "employee_name".to_string());
df.rename_columns(&rename_map)?;

// Set all column names at once
df.set_column_names(vec!["person_name".to_string()])?;

// Distributed processing with DataFusion
use pandrs::distributed::DistributedContext;
let mut context = DistributedContext::new_local(4)?;
context.register_dataframe("people", &df)?;
let result = context.sql("SELECT * FROM people WHERE person_name LIKE 'A%'")?;
```

## Key Features

- **🔥 High-Performance Processing**: Column-oriented storage with up to 5x faster aggregations
- **🧠 Memory Efficient**: String pool optimization reducing memory usage by up to 89.8%
- **⚡ Multi-core & GPU**: Parallel processing + CUDA acceleration (up to 20x speedup)
- **🌐 Distributed Computing**: DataFusion-powered distributed processing for large datasets
- **🐍 Python Integration**: Full PyO3 bindings with pandas interoperability
- **🔒 Type Safety**: Rust's ownership system ensuring memory safety and thread safety
- **📊 Rich Analytics**: Statistical functions, ML metrics, and categorical data analysis
- **💾 Flexible I/O**: Parquet, CSV, JSON, SQL, and Excel support with real data extraction

## Features

- Series (1-dimensional array) and DataFrame (2-dimensional table) data structures
- Support for missing values (NA)
- Grouping and aggregation operations
- Row labels with indexes
- Multi-level indexes (hierarchical indexes)
- CSV/JSON reading and writing
- Parquet data format support
- Basic operations (filtering, sorting, joining, etc.)
- Aggregation functions for numeric data
- Special operations for string data
- Basic time series data processing
- Categorical data types (efficient memory use, ordered categories)
- Pivot tables
- Visualization with text-based and high-quality graphs
- Parallel processing support
- Statistical analysis functions (descriptive statistics, t-tests, regression analysis, etc.)
- Specialized categorical data statistics (contingency tables, chi-square tests, etc.)
- Machine learning evaluation metrics (MSE, R², accuracy, F1, etc.)
- Optimized implementation (column-oriented storage, lazy evaluation, string pool)
- High-performance split implementation (sub-modularized files for each functionality)
- GPU acceleration for matrix operations, statistical computation, and ML algorithms
- Disk-based processing for very large datasets
- Streaming data support with real-time analytics
- Distributed processing for datasets exceeding single-machine capacity (experimental)
  - SQL-like query interface with DataFusion
  - Window function support for analytics
  - Advanced expression system for complex transformations
  - User-defined function support

## Usage Examples

### Creating and Basic Operations with DataFrames

```rust
use pandrs::{DataFrame, Series};

// Create series
let ages = Series::new(vec![30, 25, 40], Some("age".to_string()))?;
let heights = Series::new(vec![180, 175, 182], Some("height".to_string()))?;

// Add series to DataFrame
let mut df = DataFrame::new();
df.add_column("age".to_string(), ages)?;
df.add_column("height".to_string(), heights)?;

// Save as CSV
df.to_csv("data.csv")?;

// Load DataFrame from CSV
let df_from_csv = DataFrame::from_csv("data.csv", true)?;

// DataFrame column management
use std::collections::HashMap;

// Rename specific columns using a mapping
let mut rename_map = HashMap::new();
rename_map.insert("age".to_string(), "years_old".to_string());
rename_map.insert("height".to_string(), "height_cm".to_string());
df.rename_columns(&rename_map)?;

// Set all column names at once
df.set_column_names(vec!["person_age".to_string(), "person_height".to_string()])?;
```

### Numeric Operations and Series Management

```rust
// Create numeric series
let mut numbers = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string()))?;

// Statistical calculations
let sum = numbers.sum();         // 150
let mean = numbers.mean()?;      // 30
let min = numbers.min()?;        // 10
let max = numbers.max()?;        // 50

// Series name management
numbers.set_name("updated_values".to_string());
assert_eq!(numbers.name(), Some(&"updated_values".to_string()));

// Create series with fluent API
let named_series = Series::new(vec![1, 2, 3], None)?
    .with_name("my_series".to_string());

// Convert series types with name preservation
let string_series = numbers.to_string_series()?;
```

## Installation

Add the following to your Cargo.toml:

```toml
[dependencies]
pandrs = "0.1.0-alpha.4"
```

For GPU acceleration, add the CUDA feature flag (requires CUDA toolkit installation):

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.4", features = ["cuda"] }
```

**Note**: The CUDA feature requires NVIDIA CUDA toolkit to be installed on your system.

For distributed processing capabilities, add the distributed feature:

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.4", features = ["distributed"] }
```

Multiple features can be combined:

```toml
[dependencies]
pandrs = { version = "0.1.0-alpha.4", features = ["cuda", "distributed", "wasm"] }
```

### Working with Missing Values (NA)

```rust
// Create series with NA values
let data = vec![
    NA::Value(10), 
    NA::Value(20), 
    NA::NA,  // missing value
    NA::Value(40)
];
let series = NASeries::new(data, Some("values".to_string()))?;

// Handle NA values
println!("Number of NAs: {}", series.na_count());
println!("Number of values: {}", series.value_count());

// Drop and fill NA values
let dropped = series.dropna()?;
let filled = series.fillna(0)?;
```

### Group Operations

```rust
// Data and group keys
let values = Series::new(vec![10, 20, 15, 30, 25], Some("values".to_string()))?;
let keys = vec!["A", "B", "A", "C", "B"];

// Group and aggregate
let group_by = GroupBy::new(
    keys.iter().map(|s| s.to_string()).collect(),
    &values,
    Some("by_category".to_string())
)?;

// Aggregation results
let sums = group_by.sum()?;
let means = group_by.mean()?;
```

### Time Series Operations

```rust
use pandrs::temporal::{TimeSeries, date_range, Frequency};
use chrono::NaiveDate;

// Generate date range
let dates = date_range(
    NaiveDate::from_str("2023-01-01")?,
    NaiveDate::from_str("2023-01-31")?,
    Frequency::Daily,
    true
)?;

// Create time series data
let time_series = TimeSeries::new(values, dates, Some("daily_data".to_string()))?;

// Time filtering
let filtered = time_series.filter_by_time(
    &NaiveDate::from_str("2023-01-10")?,
    &NaiveDate::from_str("2023-01-20")?
)?;

// Calculate moving average
let moving_avg = time_series.rolling_mean(3)?;

// Resampling (convert to weekly)
let weekly = time_series.resample(Frequency::Weekly).mean()?;
```

### Distributed Processing (Experimental)

#### Using the DataFrame-Style API

```rust
use pandrs::{DataFrame, distributed::{DistributedConfig, ToDistributed}};

// Create a local DataFrame
let mut df = DataFrame::new();
df.add_column("id".to_string(), (0..10000).collect::<Vec<i64>>())?;
df.add_column("value".to_string(), (0..10000).map(|x| x as f64 * 1.5).collect::<Vec<f64>>())?;
df.add_column("group".to_string(), (0..10000).map(|x| (x % 100).to_string()).collect::<Vec<String>>())?;

// Configure distributed processing
let config = DistributedConfig::new()
    .with_executor("datafusion")  // Use DataFusion engine
    .with_concurrency(4);         // Use 4 threads

// Convert to distributed DataFrame
let dist_df = df.to_distributed(config)?;

// Define distributed operations (lazy execution)
let result = dist_df
    .filter("value > 5000.0")?
    .groupby(&["group"])?
    .aggregate(&["value"], &["mean"])?;

// Execute operations and get performance metrics
let executed = result.execute()?;
if let Some(metrics) = executed.execution_metrics() {
    println!("Execution time: {}ms", metrics.execution_time_ms());
    println!("Rows processed: {}", metrics.rows_processed());
}

// Check execution summary
println!("{}", executed.summarize()?);

// Collect results back to local DataFrame
let final_df = executed.collect_to_local()?;
println!("Result: {}", final_df);

// Write results directly to Parquet
executed.write_parquet("filtered_results.parquet")?;
```

#### Using the SQL-Style API with DistributedContext

```rust
use pandrs::{DataFrame, distributed::{DistributedContext, DistributedConfig}};

// Create a distributed context
let mut context = DistributedContext::new(
    DistributedConfig::new()
        .with_executor("datafusion")
        .with_concurrency(4)
)?;

// Register multiple DataFrames with the context
let customers = DataFrame::new(); // Create customers DataFrame
let orders = DataFrame::new();    // Create orders DataFrame
context.register_dataframe("customers", &customers)?;
context.register_dataframe("orders", &orders)?;

// Also register CSV or Parquet files directly
context.register_csv("products", "products.csv")?;
context.register_parquet("sales", "sales.parquet")?;

// Execute SQL queries against registered datasets
let result = context.sql_to_dataframe("
    SELECT c.customer_id, c.name, COUNT(o.order_id) as order_count, SUM(o.amount) as total_amount
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.name
    ORDER BY total_amount DESC
")?;

println!("Query results: {}", result);

// Execute query and write directly to Parquet
let metrics = context.sql_to_parquet("
    SELECT p.product_id, p.name, SUM(s.quantity) as total_sold
    FROM products p
    JOIN sales s ON p.product_id = s.product_id
    GROUP BY p.product_id, p.name
", "product_sales_summary.parquet")?;

println!("Query execution metrics:\n{}", metrics.format());
```

#### Using Window Functions for Analytics

```rust
use pandrs::{DataFrame, distributed::{DistributedConfig, ToDistributed, WindowFunctionExt}};
use pandrs::distributed_window; // Access window functions

// Create a DataFrame with time series data
let mut df = DataFrame::new();
// ... add columns with time series data ...

// Convert to distributed DataFrame
let dist_df = df.to_distributed(DistributedConfig::new())?;

// Add ranking by sales within each region
let ranked = dist_df.window(&[
    distributed_window::rank(
        "sales_rank",       // Output column
        &["region"],        // Partition by region
        &[("sales", false)] // Order by sales descending
    )
])?;

// Calculate running total of sales by date
let running_total = dist_df.window(&[
    distributed_window::cumulative_sum(
        "sales",          // Input column
        "running_total",  // Output column
        &["region"],      // Partition by region
        &[("date", true)] // Order by date ascending
    )
])?;

// Calculate 7-day moving average
let moving_avg = dist_df.window(&[
    distributed_window::rolling_avg(
        "sales",             // Input column
        "sales_7day_avg",    // Output column
        &[],                 // No partitioning
        &[("date", true)],   // Order by date
        7                    // Window size of 7 days
    )
])?;

// Or use SQL directly
let context = DistributedContext::new(DistributedConfig::new())?;
context.register_dataframe("sales", &df)?;

let result = context.sql_to_dataframe("
    SELECT
        date, region, product, sales,
        RANK() OVER (PARTITION BY region ORDER BY sales DESC) as sales_rank,
        SUM(sales) OVER (PARTITION BY region ORDER BY date) as running_total,
        AVG(sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day
    FROM sales
")?;
```

#### Using Expressions for Complex Data Transformations

```rust
use pandrs::{DataFrame, distributed::{DistributedConfig, ToDistributed, Expr, ColumnProjection}};

// Create a DataFrame with sales data
let mut df = DataFrame::new();
// ... add columns with sales data ...

// Convert to distributed DataFrame
let dist_df = df.to_distributed(DistributedConfig::new())?;

// Select columns with complex expressions
let selected = dist_df.select_expr(&[
    // Simple column selection
    ColumnProjection::column("region"),
    ColumnProjection::column("sales"),
    // Calculate a new column with expression
    ColumnProjection::with_alias(
        Expr::col("sales").mul(Expr::lit(1.1)),
        "sales_with_bonus"
    ),
    // Calculate profit margin percentage
    ColumnProjection::with_alias(
        Expr::col("profit").div(Expr::col("sales")).mul(Expr::lit(100.0)),
        "profit_margin"
    ),
])?;

// Filter using complex expressions
let high_margin = dist_df
    .filter_expr(
        Expr::col("profit")
            .div(Expr::col("sales"))
            .mul(Expr::lit(100.0))
            .gt(Expr::lit(15.0))
    )?;

// Create a user-defined function
let commission_udf = UdfDefinition::new(
    "calculate_commission",
    ExprDataType::Float,
    vec![ExprDataType::Float, ExprDataType::Float],
    "CASE
        WHEN param1 / param0 > 0.2 THEN param0 * 0.05
        WHEN param1 / param0 > 0.1 THEN param0 * 0.03
        ELSE param0 * 0.01
     END"
);

// Register and use the UDF
let with_commission = dist_df
    .create_udf(&[commission_udf])?
    .select_expr(&[
        ColumnProjection::column("region"),
        ColumnProjection::column("sales"),
        ColumnProjection::column("profit"),
        ColumnProjection::with_alias(
            Expr::call("calculate_commission", vec![
                Expr::col("sales"),
                Expr::col("profit"),
            ]),
            "commission"
        ),
    ])?;

// Chain operations together
let final_analysis = dist_df
    // Add calculated columns
    .with_column("profit_pct",
        Expr::col("profit").div(Expr::col("sales")).mul(Expr::lit(100.0))
    )?
    // Filter for high-profit regions
    .filter_expr(
        Expr::col("profit_pct").gt(Expr::lit(12.0))
    )?
    // Project final columns with calculations
    .select_expr(&[
        ColumnProjection::column("region"),
        ColumnProjection::column("product"),
        ColumnProjection::column("profit_pct"),
        // Calculate a bonus amount
        ColumnProjection::with_alias(
            Expr::col("profit")
                .mul(Expr::lit(0.1))
                .add(Expr::col("profit_pct").mul(Expr::lit(5.0))),
            "bonus"
        ),
    ])?;
```

### Statistical Analysis and Machine Learning Evaluation Functions

```rust
use pandrs::{DataFrame, Series, stats};
use pandrs::ml::metrics::regression::{mean_squared_error, r2_score};
use pandrs::ml::metrics::classification::{accuracy_score, f1_score};

// Descriptive statistics
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let stats_summary = stats::describe(&data)?;
println!("Mean: {}, Standard deviation: {}", stats_summary.mean, stats_summary.std);
println!("Median: {}, Quartiles: {} - {}", stats_summary.median, stats_summary.q1, stats_summary.q3);

// Calculate correlation coefficient
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let correlation = stats::correlation(&x, &y)?;
println!("Correlation coefficient: {}", correlation);

// Run t-test
let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
let alpha = 0.05; // significance level
let result = stats::ttest(&sample1, &sample2, alpha, true)?;
println!("t-statistic: {}, p-value: {}", result.statistic, result.pvalue);
println!("Significant difference: {}", result.significant);

// Regression analysis
let mut df = DataFrame::new();
df.add_column("x1".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x1".to_string()))?)?;
df.add_column("x2".to_string(), Series::new(vec![2.0, 3.0, 4.0, 5.0, 6.0], Some("x2".to_string()))?)?;
df.add_column("y".to_string(), Series::new(vec![3.0, 5.0, 7.0, 9.0, 11.0], Some("y".to_string()))?)?;

let model = stats::linear_regression(&df, "y", &["x1", "x2"])?;
println!("Coefficients: {:?}", model.coefficients());
println!("Coefficient of determination: {}", model.r_squared());

// Machine learning model evaluation - regression metrics
let y_true = vec![3.0, 5.0, 2.5, 7.0, 10.0];
let y_pred = vec![2.8, 4.8, 2.7, 7.2, 9.8];

let mse = mean_squared_error(&y_true, &y_pred)?;
let r2 = r2_score(&y_true, &y_pred)?;
println!("MSE: {:.4}, R²: {:.4}", mse, r2);

// Machine learning model evaluation - classification metrics
let true_labels = vec![true, false, true, true, false, false];
let pred_labels = vec![true, false, false, true, true, false];

let accuracy = accuracy_score(&true_labels, &pred_labels)?;
let f1 = f1_score(&true_labels, &pred_labels)?;
println!("Accuracy: {:.2}, F1 Score: {:.2}", accuracy, f1);
```

### Pivot Tables and Grouping

```rust
use pandrs::pivot::AggFunction;

// Grouping and aggregation
let grouped = df.groupby("category")?;
let category_sum = grouped.sum(&["sales"])?;

// Pivot table
let pivot_result = df.pivot_table(
    "category",   // index column
    "region",     // column column
    "sales",      // value column
    AggFunction::Sum
)?;
```

### Categorical Data Analysis

```rust
use pandrs::{DataFrame, stats};
use pandrs::series::{Series, Categorical};

// Create categorical data
let mut df = DataFrame::new();
df.add_column("gender".to_string(), Series::new(vec!["M", "F", "M", "F", "M"], Some("gender".to_string()))?)?;
df.add_column("response".to_string(), Series::new(vec!["Yes", "No", "Yes", "Yes", "No"], Some("response".to_string()))?)?;

// Convert columns to categorical
df.convert_to_categorical("gender")?;
df.convert_to_categorical("response")?;

// Create contingency table
let contingency = stats::contingency_table_from_df(&df, "gender", "response")?;
println!("Contingency table:\n{}", contingency);

// Chi-square test for independence
let chi2_result = stats::chi_square_independence(&df, "gender", "response", 0.05)?;
println!("Chi-square statistic: {}", chi2_result.chi2_statistic);
println!("p-value: {}", chi2_result.p_value);
println!("Significant association: {}", chi2_result.significant);

// Measure of association
let cramers_v = stats::cramers_v_from_df(&df, "gender", "response")?;
println!("Cramer's V: {}", cramers_v);

// Test association between categorical and numeric variables
df.add_column("score".to_string(), Series::new(vec![85, 92, 78, 95, 88], Some("score".to_string()))?)?;
let anova_result = stats::categorical_anova_from_df(&df, "gender", "score", 0.05)?;
println!("F-statistic: {}", anova_result.f_statistic);
println!("p-value: {}", anova_result.p_value);
```

## Development Plan and Implementation Status

- [x] Basic DataFrame structure
- [x] Series implementation
- [x] Index functionality
- [x] CSV input/output
- [x] JSON input/output
- [x] Parquet format support
- [x] Missing value handling
- [x] Grouping operations
- [x] Time series data support
  - [x] Date range generation
  - [x] Time filtering
  - [x] Moving average calculation
  - [x] Frequency conversion (resampling)
- [x] Pivot tables
- [x] Complete implementation of join operations
  - [x] Inner join (internal match)
  - [x] Left join (left side priority)
  - [x] Right join (right side priority)
  - [x] Outer join (all rows)
- [x] Visualization functionality integration
  - [x] Line graphs
  - [x] Scatter plots
  - [x] Text plot output
- [x] Parallel processing support
  - [x] Parallel conversion of Series/NASeries
  - [x] Parallel processing of DataFrames
  - [x] Parallel filtering (1.15x speedup)
  - [x] Parallel aggregation (3.91x speedup)
  - [x] Parallel computation processing (1.37x speedup)
  - [x] Adaptive parallel processing (automatic selection based on data size)
- [x] Enhanced visualization
  - [x] Text-based plots with textplots (line, scatter)
  - [x] High-quality graph output with plotters (PNG, SVG format)
  - [x] Various graph types (line, scatter, bar, histogram, area)
  - [x] Graph customization options (size, color, grid, legend)
  - [x] Intuitive plot API for Series, DataFrame
- [x] Multi-level indexes
  - [x] Hierarchical index structure
  - [x] Data grouping by multiple levels
  - [x] Level operations (swap, select)
- [x] Categorical data types
  - [x] Memory-efficient encoding
  - [x] Support for ordered and unordered categories
  - [x] Complete integration with NA values (missing values)
- [x] Advanced DataFrame operations
  - [x] Long-form and wide-form conversion (melt, stack, unstack)
  - [x] Conditional aggregation
  - [x] DataFrame concatenation
- [x] Memory usage optimization
  - [x] String pool optimization (up to 89.8% memory reduction)
  - [x] Categorical encoding (2.59x performance improvement)
  - [x] Global string pool implementation
  - [x] Improved memory locality with column-oriented storage
- [x] Python bindings
  - [x] Python module with PyO3
  - [x] Interoperability with numpy and pandas
  - [x] Jupyter Notebook support
  - [x] Speedup with string pool optimization (up to 3.33x)
- [x] Distributed processing enhancements
  - [x] SQL-like API with DistributedContext
  - [x] Window function support
  - [x] Expression system for complex transformations
  - [x] User-defined function registration and use
- [x] Lazy evaluation system
  - [x] Operation optimization with computation graph
  - [x] Operation fusion
  - [x] Avoiding unnecessary intermediate results
- [x] Statistical analysis features
  - [x] Descriptive statistics (mean, standard deviation, quantiles, etc.)
  - [x] Correlation coefficient and covariance
  - [x] Hypothesis testing (t-test)
  - [x] Regression analysis (simple and multiple regression)
  - [x] Sampling methods (bootstrap, etc.)
- [x] Machine learning evaluation metrics
  - [x] Regression evaluation (MSE, MAE, RMSE, R² score)
  - [x] Classification evaluation (accuracy, precision, recall, F1 score)
- [x] Codebase maintainability improvements
  - [x] File separation of OptimizedDataFrame by functionality
  - [x] API compatibility maintained through re-exports
  - [x] Independent implementation of ML metrics module
- [x] GPU acceleration
  - [x] CUDA integration for numerical operations
  - [x] GPU-accelerated matrix operations
  - [x] GPU-accelerated statistical functions
  - [x] GPU-accelerated machine learning algorithms
  - [x] Comprehensive benchmarking utility
  - [x] Python bindings for GPU acceleration

### Multi-level Index Operations

```rust
use pandrs::{DataFrame, MultiIndex};

// Create MultiIndex from tuples
let tuples = vec![
    vec!["A".to_string(), "a".to_string()],
    vec!["A".to_string(), "b".to_string()],
    vec!["B".to_string(), "a".to_string()],
    vec!["B".to_string(), "b".to_string()],
];

// Set level names
let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);
let multi_idx = MultiIndex::from_tuples(tuples, names)?;

// Create DataFrame with MultiIndex
let mut df = DataFrame::with_multi_index(multi_idx);

// Add data
let data = vec!["data1".to_string(), "data2".to_string(), "data3".to_string(), "data4".to_string()];
df.add_column("data".to_string(), pandrs::Series::new(data, Some("data".to_string()))?)?;

// Level operations
let level0_values = multi_idx.get_level_values(0)?;
let level1_values = multi_idx.get_level_values(1)?;

// Swap levels
let swapped_idx = multi_idx.swaplevel(0, 1)?;
```

### GPU Acceleration

```rust
use pandrs::gpu::{self, operations::{GpuMatrix, GpuVector}};
use pandrs::dataframe::gpu::DataFrameGpuExt;
use ndarray::{Array1, Array2};

// Initialize GPU
let device_status = gpu::init_gpu()?;
println!("GPU available: {}", device_status.available);

// Create matrices for GPU operations
let a_data = Array2::from_shape_vec((1000, 500),
    (0..500000).map(|i| i as f64).collect())?;
let b_data = Array2::from_shape_vec((500, 1000),
    (0..500000).map(|i| i as f64).collect())?;

// Create GPU matrices
let a = GpuMatrix::new(a_data);
let b = GpuMatrix::new(b_data);

// Perform GPU-accelerated matrix multiplication
let result = a.dot(&b)?;

// GPU-accelerated DataFrame operations
let mut df = DataFrame::new();
// Add data to DataFrame...

// Using GPU-accelerated correlation matrix
let corr_matrix = df.gpu_corr(&["col1", "col2", "col3"])?;

// GPU-accelerated linear regression
let model = df.gpu_linear_regression("y", &["x1", "x2", "x3"])?;

// GPU-accelerated PCA
let (components, explained_variance, transformed) = df.gpu_pca(&["x1", "x2", "x3"], 2)?;

// Benchmarking CPU vs GPU performance
let mut benchmark = gpu::benchmark::GpuBenchmark::new()?;
let matrix_multiply_result = benchmark.benchmark_matrix_multiply(1000, 1000, 1000)?;
println!("Matrix multiplication speedup: {}x", matrix_multiply_result.speedup.unwrap_or(0.0));
```

### Python Binding Usage Examples

```python
import pandrs as pr
import numpy as np
import pandas as pd

# Create optimized DataFrame
df = pr.OptimizedDataFrame()
df.add_int_column('A', [1, 2, 3, 4, 5])
df.add_string_column('B', ['a', 'b', 'c', 'd', 'e'])
df.add_float_column('C', [1.1, 2.2, 3.3, 4.4, 5.5])

# Traditional API compatible interface
df2 = pr.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': [1.1, 2.2, 3.3, 4.4, 5.5]
})

# Interoperability with pandas
pd_df = df.to_pandas()  # Convert from PandRS to pandas DataFrame
pr_df = pr.OptimizedDataFrame.from_pandas(pd_df)  # Convert from pandas DataFrame to PandRS

# Using lazy evaluation
lazy_df = pr.LazyFrame(df)
result = lazy_df.filter('A').select(['B', 'C']).execute()

# Direct use of string pool
string_pool = pr.StringPool()
idx1 = string_pool.add("repeated_value")
idx2 = string_pool.add("repeated_value")  # Returns the same index
print(string_pool.get(idx1))  # Returns "repeated_value"

# GPU acceleration in Python
pr.gpu.init_gpu()  # Initialize GPU
gpu_df = df.gpu_accelerate()
corr_matrix = gpu_df.gpu_corr(['A', 'C'])
pca_result = gpu_df.gpu_pca(['A', 'C'], n_components=1)

# CSV input/output
df.to_csv('data.csv')
df_loaded = pr.OptimizedDataFrame.read_csv('data.csv')

# NumPy integration
series = df['A']
np_array = series.to_numpy()

# Jupyter Notebook support
from pandrs.jupyter import display_dataframe
display_dataframe(df, max_rows=10, max_cols=5)
```

## Performance Optimization Results

The implementation of optimized column-oriented storage, lazy evaluation system, and GPU acceleration has achieved significant performance improvements:

### Performance Comparison of Key Operations

| Operation | Traditional Implementation | Optimized Implementation | Speedup |
|------|---------|-----------|----------|
| Series/Column Creation | 198.446ms | 149.528ms | 1.33x |
| DataFrame Creation (1 million rows) | NA | NA | NA |
| Filtering | 596.146ms | 161.816ms | 3.68x |
| Group Aggregation | 544.384ms | 107.837ms | 5.05x |

### GPU Acceleration Performance (vs CPU)

| Operation | Data Size | CPU Time | GPU Time | Speedup |
|-----------|-----------|----------|----------|---------|
| Matrix Multiplication | 1000x1000 | 232.8 ms | 11.5 ms | 20.2x |
| Element-wise Addition | 2000x2000 | 18.6 ms | 2.3 ms | 8.1x |
| Correlation Matrix | 10000x10 | 89.4 ms | 12.1 ms | 7.4x |
| Linear Regression | 10000x10 | 124.3 ms | 18.7 ms | 6.6x |
| Rolling Window | 100000, window=100 | 58.2 ms | 9.8 ms | 5.9x |

### String Processing Optimization

| Mode | Processing Time | vs Traditional | Notes |
|--------|---------|------------|------|
| Legacy Mode | 596.50ms | 1.00x | Traditional implementation |
| Categorical Mode | 230.11ms | 2.59x | Categorical optimization |
| Optimized Implementation | 232.38ms | 2.57x | Optimizer selection |

### Parallel Processing Performance Improvements

| Operation | Serial Processing | Parallel Processing | Speedup |
|------|---------|----------|---------|
| Group Aggregation | 696.85ms | 178.09ms | 3.91x |
| Filtering | 201.35ms | 175.48ms | 1.15x |
| Computation | 15.41ms | 11.23ms | 1.37x |

### Python Bindings String Optimization

| Data Size | Unique Rate | Without Pool | With Pool | Processing Speedup | Memory Reduction |
|------------|----------|-----------|-----------|------------|------------|
| 100,000 rows | 1% (high duplication) | 82ms | 35ms | 2.34x | 88.6% |
| 1,000,000 rows | 1% (high duplication) | 845ms | 254ms | 3.33x | 89.8% |

## 🆕 Major Features

### 🔧 Enhanced DataFrame API
- **Column Management**: New `rename_columns()` and `set_column_names()` methods for flexible DataFrame schema management
- **Series Operations**: Enhanced name management with `set_name()` and `with_name()` methods
- **Type Conversions**: Improved type conversion utilities like `to_string_series()` for Series
- **API Consistency**: Fluent interface design across all DataFrame operations

### 🌐 Production-Ready Distributed Processing
- **DataFusion Integration**: Complete integration with Apache DataFusion for scalable data processing
- **Schema Validation**: Compile-time schema validation preventing runtime errors
- **Fault Tolerance**: Checkpoint and recovery system for robust distributed operations
- **SQL Interface**: Full SQL query support with complex JOIN, window functions, and UDFs

### 💾 Enhanced Data I/O
- **Real Data Extraction**: Parquet and SQL I/O operations now use actual data instead of placeholders
- **Type Safety**: Improved Arrow integration with proper null value handling
- **Performance**: Optimized data conversion processes for better throughput

### 🐍 Complete Python Bindings
- **Feature Parity**: All features available through Python bindings
- **Pandas Integration**: Seamless interoperability with pandas DataFrames
- **Memory Optimization**: String pool integration reducing Python memory overhead

- **Comprehensive Test Coverage**
  - 26 new edge case tests covering boundary conditions, error handling, and invalid inputs
  - Stress testing for large datasets (100K+ rows) and concurrent operations
  - String pool concurrency testing with thread safety validation
  - Memory management and resource cleanup testing
  - Unicode and special character support validation
  - Null value handling with proper bitmask validation

- **Enhanced Parquet and SQL Support**
  - Real data extraction in Parquet write operations (replacing dummy implementations)
  - Improved column data access patterns for all data types
  - Better Arrow integration with proper null value handling
  - Enhanced type safety in data conversion processes

- **GPU Acceleration Integration**
  - CUDA-based acceleration for performance-critical operations
  - Up to 20x speedup for matrix operations
  - GPU-accelerated statistical functions and ML algorithms
  - Transparent CPU fallback when GPU is unavailable
  - Comprehensive benchmarking utility for CPU vs GPU performance comparison
  - Python bindings for GPU functionality

- **Specialized Categorical Data Statistics**
  - Comprehensive contingency table implementation
  - Chi-square test for independence
  - Cramer's V measure of association
  - Categorical ANOVA for comparing means across categories
  - Entropy and mutual information calculations
  - Statistical summaries specific to categorical variables

- **Large-Scale Data Processing**
  - Disk-based processing for datasets larger than available memory
  - Memory-mapped file support for efficient large data access
  - Chunked processing capabilities for scalable data operations
  - Spill-to-disk functionality when memory limits are reached

- **Streaming Data Support**
  - Real-time data processing interfaces
  - Stream connectors for various data sources
  - Windowed operations on streaming data
  - Real-time analytics capabilities

- **Column-Oriented Storage Engine**
  - Type-specialized column implementation (Int64Column, Float64Column, StringColumn, BooleanColumn)
  - Improved cache efficiency through memory locality
  - Operation acceleration and parallel processing efficiency

- **String Processing Optimization**
  - Elimination of duplicate strings with global string pool
  - String to index conversion with categorical encoding
  - Consistent API design and multiple optimization modes

- **Lazy Evaluation System Implementation**
  - Operation pipelining with computation graph
  - Avoiding unnecessary intermediate results
  - Improved efficiency through operation fusion

- **Significant Parallel Processing Improvements**
  - Efficient multi-threading with Rayon
  - Adaptive parallel processing (automatic selection based on data size)
  - Chunk processing optimization

- **Enhanced Python Integration**
  - Efficient data conversion between Python and Rust with string pool optimization
  - Utilization of NumPy buffer protocol
  - Near zero-copy data access
  - Type-specialized Python API
  - GPU acceleration support through Python bindings

- **Advanced DataFrame Operations**
  - Complete implementation of long-form and wide-form conversion (melt, stack, unstack)
  - Enhanced conditional aggregation processing
  - Optimization of complex join operations

- **Enhanced Time Series Data Processing**
  - Support for RFC3339 format date parsing
  - Complete implementation of advanced window operations
  - Support for complete format frequency specification (`DAILY`, `WEEKLY`, etc.)
  - GPU-accelerated time series operations

- **WebAssembly Support**
  - Browser-based visualization capabilities
  - Interactive dashboard functionality
  - Theme customization options
  - Multiple visualization types support

- **Stability and Quality Improvements**
  - Implementation of comprehensive test suite
  - Improved error handling and warning elimination
  - Enhanced documentation
  - Updated dependencies (Rust 2023 compatible)

### High-Quality Visualization (Plotters Integration)

```rust
use pandrs::{DataFrame, Series};
use pandrs::vis::plotters_ext::{PlotSettings, PlotKind, OutputType};

// Create plot from a single Series
let values = vec![15.0, 23.5, 18.2, 29.8, 32.1, 28.5, 19.2];
let series = Series::new(values, Some("temperature_change".to_string()))?;

// Create line graph
let line_settings = PlotSettings {
    title: "Temperature Change Over Time".to_string(),
    x_label: "Time".to_string(),
    y_label: "Temperature (°C)".to_string(),
    plot_kind: PlotKind::Line,
    ..PlotSettings::default()
};
series.plotters_plot("temp_line.png", line_settings)?;

// Create histogram
let hist_settings = PlotSettings {
    title: "Histogram of Temperature Distribution".to_string(),
    plot_kind: PlotKind::Histogram,
    ..PlotSettings::default()
};
series.plotters_histogram("histogram.png", 5, hist_settings)?;

// Visualization using DataFrame
let mut df = DataFrame::new();
df.add_column("temperature".to_string(), series)?;
df.add_column("humidity".to_string(), 
    Series::new(vec![67.0, 72.3, 69.5, 58.2, 62.1, 71.5, 55.8], Some("humidity".to_string()))?)?;

// Scatter plot (relationship between temperature and humidity)
let xy_settings = PlotSettings {
    title: "Relationship Between Temperature and Humidity".to_string(),
    plot_kind: PlotKind::Scatter,
    output_type: OutputType::SVG,  // Output in SVG format
    ..PlotSettings::default()
};
df.plotters_xy("temperature", "humidity", "temp_humidity.svg", xy_settings)?;

// Multiple series line graph
let multi_settings = PlotSettings {
    title: "Weather Data Trends".to_string(),
    plot_kind: PlotKind::Line,
    ..PlotSettings::default()
};
df.plotters_multi(&["temperature", "humidity"], "multi_series.png", multi_settings)?;
```

## Testing

Run the core library tests (no external dependencies):

```bash
cargo test --lib
```

Test core optimized features:

```bash
cargo test --features "test-core"
```

Test most features (excluding CUDA/WASM which require external tools):

```bash
cargo test --features "test-safe"
```

Test all features (requires CUDA toolkit and may have compilation issues):

```bash
cargo test --all-features
```

**Notes**: 
- The `cuda` feature requires NVIDIA CUDA toolkit installation
- The `visualization` feature currently has compilation issues and is excluded from safe testing
- Use `test-core` for reliable testing without external dependencies

### Edge Case and Error Condition Testing

Run comprehensive edge case tests:

```bash
cargo test --test edge_cases_test
```

Run concurrency and thread safety tests:

```bash
cargo test --test concurrency_test
```

Run I/O error condition tests:

```bash
cargo test --test io_error_conditions_test
```

These tests cover:
- **Boundary Conditions**: Empty data, single elements, extreme numeric values, NaN/infinity handling
- **Input Validation**: Invalid column names, duplicate columns, mismatched lengths, out-of-bounds access
- **Resource Management**: Large datasets (100K+ rows), string pool stress testing, memory cleanup
- **Concurrency**: Multi-threaded access, string pool thread safety, race condition prevention
- **Type Safety**: Type mismatches, column access patterns, error propagation
- **I/O Error Handling**: File permissions, malformed data, disk space issues, encoding problems

## Dependency Versions

Latest dependency versions:

```toml
[dependencies]
num-traits = "0.2.19"        # Numeric trait support
thiserror = "2.0.12"         # Error handling
serde = { version = "1.0.219", features = ["derive"] }  # Serialization
serde_json = "1.0.114"       # JSON processing
chrono = "0.4.38"            # Date and time processing (compatible with arrow ecosystem)
regex = "1.11.1"             # Regular expressions  
csv = "1.3.1"                # CSV processing
rayon = "1.10.0"             # Parallel processing
lazy_static = "1.5.0"        # Lazy initialization
rand = "0.9.0"               # Random number generation
tempfile = "3.8.1"           # Temporary files
textplots = "0.8.7"          # Text-based visualization
plotters = "0.3.7"           # High-quality visualization
chrono-tz = "0.9.0"          # Timezone processing (compatible with chrono 0.4.38)
parquet = "53.3.1"           # Parquet file support (compatible with arrow ecosystem)
arrow = "53.3.1"             # Arrow format support (stable version)
crossbeam-channel = "0.5.15" # For concurrent message passing
memmap2 = "0.9.5"            # For memory-mapped files
calamine = "0.23.1"          # For Excel reading
simple_excel_writer = "0.2.0" # For Excel writing

# Optional dependencies (feature-gated)
# CUDA support (requires CUDA toolkit installation)
cudarc = "0.10.0"            # CUDA bindings (requires NVIDIA CUDA toolkit)
half = "2.3.1"               # Half-precision floating point support
ndarray = "0.15.6"           # N-dimensional arrays

# Distributed processing
datafusion = "30.0.0"        # DataFrame processing engine (compatible with arrow 53.x)

# WebAssembly support
wasm-bindgen = "0.2.91"      # WebAssembly bindings
js-sys = "0.3.68"            # JavaScript interop
web-sys = "0.3.68"           # Web API bindings
plotters-canvas = "0.4.0"    # Canvas backend for plotters
```

## License

PandRS is dual-licensed under either

* MIT License (LICENSE-MIT or http://opensource.org/licenses/MIT)
* Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
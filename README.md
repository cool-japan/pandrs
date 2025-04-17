# PandRS

[![Rust CI](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml/badge.svg)](https://github.com/cool-japan/pandrs/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crate](https://img.shields.io/crates/v/pandrs.svg)](https://crates.io/crates/pandrs)

A DataFrame library for data analysis implemented in Rust. It has features and design inspired by Python's `pandas` library, combining fast data processing with type safety.

## Key Features

- Efficient data processing with high-performance column-oriented storage
- Low memory footprint with categorical data and string pool optimization
- Multi-core utilization through parallel processing
- Optimization with lazy evaluation system
- Thread-safe implementation
- Robustness leveraging Rust's type safety and ownership system
- Modularized design (implementation divided by functionality)
- Python integration (PyO3 bindings)

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
- Machine learning evaluation metrics (MSE, R², accuracy, F1, etc.)
- Optimized implementation (column-oriented storage, lazy evaluation, string pool)
- High-performance split implementation (sub-modularized files for each functionality)

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
```

### Numeric Operations

```rust
// Create numeric series
let numbers = Series::new(vec![10, 20, 30, 40, 50], Some("values".to_string()))?;

// Statistical calculations
let sum = numbers.sum();         // 150
let mean = numbers.mean()?;      // 30
let min = numbers.min()?;        // 10
let max = numbers.max()?;        // 50
```

## Installation

Add the following to your Cargo.toml:

```toml
[dependencies]
pandrs = "0.1.0-alpha.1"
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

The implementation of optimized column-oriented storage and lazy evaluation system has achieved significant performance improvements:

### Performance Comparison of Key Operations

| Operation | Traditional Implementation | Optimized Implementation | Speedup |
|------|---------|-----------|----------|
| Series/Column Creation | 198.446ms | 149.528ms | 1.33x |
| DataFrame Creation (1 million rows) | NA | NA | NA |
| Filtering | 596.146ms | 161.816ms | 3.68x |
| Group Aggregation | 544.384ms | 107.837ms | 5.05x |

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

## Recent Improvements

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

- **Advanced DataFrame Operations**
  - Complete implementation of long-form and wide-form conversion (melt, stack, unstack)
  - Enhanced conditional aggregation processing
  - Optimization of complex join operations

- **Enhanced Time Series Data Processing**
  - Support for RFC3339 format date parsing
  - Complete implementation of advanced window operations
  - Support for complete format frequency specification (`DAILY`, `WEEKLY`, etc.)

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

## Dependency Versions

Latest dependency versions (April 2024):

```toml
[dependencies]
num-traits = "0.2.19"        # Numeric trait support
thiserror = "2.0.12"          # Error handling
serde = { version = "1.0.219", features = ["derive"] }  # Serialization
serde_json = "1.0.114"       # JSON processing
chrono = "0.4.40"            # Date and time processing
regex = "1.10.2"             # Regular expressions
csv = "1.3.1"                # CSV processing
rayon = "1.9.0"              # Parallel processing
lazy_static = "1.5.0"        # Lazy initialization
rand = "0.9.0"               # Random number generation
tempfile = "3.8.1"           # Temporary files
textplots = "0.8.7"          # Text-based visualization
plotters = "0.3.7"          # High-quality visualization
chrono-tz = "0.10.3"         # Timezone processing
parquet = "54.3.1"           # Parquet file support
arrow = "54.3.1"             # Arrow format support
```

## License

Available under the Apache License 2.0.
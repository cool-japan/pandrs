# PandRS Roadmap - Implementation Plan for pandas-equivalent Functionality

This roadmap serves as a guideline for implementing Rust functionality inspired by Python's Pandas library.

## Currently Implemented Features

- Series (1-dimensional array) and DataFrame (2-dimensional table) data structures
- Support for missing values (NA)
- Grouping operations
- Row labeling with indexes
- CSV/JSON reading and writing
- Parquet data format support (dependencies added, implementation to be expanded)
- Basic operations (filtering, sorting, joining, etc.)
- Aggregation functions for numeric data
- Basic time series data processing
- Pivot tables
- Text-based visualization
- Parallel processing support
- Categorical data type
- Multi-level indexes
- String pool optimization
- High-performance column-oriented storage implementation
- Statistical functions (descriptive statistics, correlation/covariance, t-tests, ANOVA, non-parametric tests, chi-square tests, regression analysis, sampling)

## Short-term Implementation Goals (1-3 months)

### Enhancement and Expansion of Statistical Functions (May-June 2024)

1. **Expansion of Existing Statistical Module (stats/)**
   - ✅ Descriptive statistics functions (sample variance/standard deviation, quantiles)
   - ✅ Covariance and correlation analysis
   - ✅ Hypothesis testing (t-tests)
   - ✅ Basic regression analysis (simple/multiple regression, least squares method)
   - ✅ Sampling methods (bootstrap)
   - ✅ Implementation of analysis of variance (one-way ANOVA)
   - ✅ Non-parametric tests (Mann-Whitney U test)
   - ✅ Implementation of chi-square tests
   - ✅ Enhancement of confidence intervals and prediction intervals

2. **Strengthening Integration with Existing Features**
   - ✅ Providing independent API functions
   - ✅ Organizing public API interfaces (publishing linear_regression functions, etc.)
   - ✅ Adding statistical methods to DataFrame and Series
   - ✅ Interface design for integration with parallel processing
   - ✅ Integration with optimized implementation (optimized/)
   - Specialized statistical processing for categorical data

3. **Machine Learning Evaluation Metrics Module**
   - ✅ Regression model evaluation metrics (MSE, MAE, RMSE, R² score)
   - ✅ Classification model evaluation metrics (accuracy, precision, recall, F1 score)
   - ✅ Error handling and documentation

### Expansion of Data Structures and Operations

1. ✅ **MultiIndex (Multi-level Index)**
   - ✅ Hierarchical index structure
   - ✅ Level-based data access
   - ✅ Index swapping operations

2. ✅ **Categorical Data Type**
   - ✅ Efficient representation of categorical data
   - ✅ Support for ordered categorical data
   - ✅ Categorical data operations (transformation, aggregation, etc.)

3. ✅ **Expansion of DataFrame Operations**
   - ✅ Function application features equivalent to `apply`/`applymap`
   - ✅ Conditional replacement (`where`/`mask`/`replace`)
   - ✅ Improved detection and removal of duplicate rows

### Enhancement of Data Input/Output

1. ✅ **Excel Support**
   - ✅ Reading and writing xlsx files
   - ✅ Sheet specification and operations
   - ✅ Basic Excel output functionality

2. ✅ **SQL Interface**
   - ✅ Reading from SQLite (queries using SQL statements)
   - ✅ Writing to SQLite
   - ✅ Options for adding to/replacing existing tables

3. ✅ **Parquet and Column-Oriented Format Support**
   - ✅ Addition of dependencies (arrow 54.3.1, parquet 54.3.1)
   - ✅ Reading and writing Parquet files
   - ✅ Compression options (Snappy, GZIP, Brotli, LZO, LZ4, Zstd)
   - ✅ Integration with column-oriented data structures

### Enhancement of Time Series Data Processing

1. ✅ **Strengthening Periodic Indexes**
   - ✅ Custom frequencies (business days, etc.)
   - ✅ Support for quarterly and fiscal year calculations
   - ✅ Extension of calendar functionality (chrono-tz 0.10.3)

2. ✅ **Time Series-Specific Operations**
   - ✅ Seasonal decomposition
   - ✅ Expanded types of moving averages
   - ✅ Optimization of time series shift and difference operations

## Medium-term Implementation Goals (4-8 months)

### Advanced Analysis Features

1. ✅ **Window Operations**
   - ✅ Fixed, expanding, and variable window processing
   - ✅ Window aggregation functions
   - ✅ Diversification of rolling statistics

2. **Enhanced Statistical Functions**
   - ✅ Correlation coefficients and covariance
   - ✅ Hypothesis testing (t-tests)
   - ✅ Sampling and random number generation (rand 0.9.0)
   - ✅ Basic regression analysis (simple/multiple regression)
   - 🔄 Advanced statistical methods (expanded hypothesis testing, non-parametric tests)

3. ✅ **Enhanced String Operations**
   - ✅ Regular expression-based search and replacement (regex 1.10.2)
   - ✅ Optimization of string vector operations
   - ✅ Text processing utilities

### Data Visualization Enhancement

1. ✅ **Integration with Plotters**
   - ✅ Integration of high-quality visualization library (plotters v0.3.7)
   - ✅ Support for PNG and SVG output formats
   - ✅ Expanded graph types (line, bar, scatter, histogram, area charts)
   - ✅ Customization options (size, color, grid, legend)
   - 🔄 Direct plotting from DataFrame/Series (partially implemented)

2. **Interactive Visualization**
   - 🔄 Browser visualization with WebAssembly support (initial stage)
   - Dashboard functionality
   - Dynamic graph generation

### Memory and Performance Optimization

1. ✅ **Memory Usage Optimization**
   - ✅ Addition of zero-copy operations
   - ✅ Optimization of column-oriented storage
   - ✅ Disk-based processing for large datasets

2. ✅ **Enhanced Parallel Processing**
   - ✅ DataFrame-level parallel processing (rayon 1.9.0)
   - ✅ Parallel optimization of operation chains
   - ✅ GPU acceleration with CUDA (up to 20x speedup)

3. ✅ **Codebase Optimization**
   - ✅ Function-based file splitting for OptimizedDataFrame
   - ✅ Optimal division into core functionality, column operations, data operations, etc.
   - ✅ Re-export with API compatibility assurance
   - ✅ Module structure reorganization (Stage 1 & 2)
     - ✅ Creation of core/ directory with fundamental data structures
     - ✅ Creation of compute/ directory with computation functionality
     - ✅ Creation of storage/ directory with storage engines
     - ✅ Restructuring of dataframe/ and series/ directories
     - ✅ Implementation of backward compatibility layers
   - 🔄 Module structure reorganization (Stage 3)
     - 🔄 Feature module reorganization for stats/, ml/, temporal/, and vis/
     - Specialized module structures for advanced features
     - Documentation updates

## Long-term Implementation Goals (9+ months)

### Advanced Data Science Features

1. **Integration with Machine Learning**
   - ✅ Data transformation pipeline equivalent to scikit-learn
   - ✅ Feature engineering functionality
     - ✅ Polynomial feature generation
     - ✅ Binning (discretization)
     - ✅ Missing value imputation
     - ✅ Feature selection
   - ✅ Utilities for model training and evaluation
     - ✅ Linear regression and logistic regression models
     - ✅ Model selection (cross-validation, grid search)
     - ✅ Model evaluation metrics
     - ✅ Model saving and loading

2. **Dimensionality Reduction and Exploratory Data Analysis**
   - ✅ Implementation of PCA, t-SNE, etc.
     - ✅ Principal Component Analysis (PCA)
     - ✅ t-Distributed Stochastic Neighbor Embedding (t-SNE)
   - ✅ Clustering functionality
     - ✅ k-means clustering
     - ✅ Hierarchical clustering
     - ✅ DBSCAN (density-based clustering)
   - ✅ Anomaly detection
     - ✅ Isolation Forest
     - ✅ LOF (Local Outlier Factor)
     - ✅ One-Class SVM

3. 🔄 **Large-scale Data Processing**
   - ✅ Chunk processing functionality
   - 🔄 Streaming data support
   - Integration with distributed processing frameworks

### Ecosystem Integration

1. ✅ **Python Bindings**
   - ✅ Python module creation using PyO3
   - ✅ Interoperability with numpy and pandas
   - ✅ Jupyter Notebook support

2. **Integration with R Language**
   - Interoperability between R and Rust
   - tidyverse-style interface

3. **Database Integration**
   - Connectors for major databases
   - Query optimizer
   - ORM-like functionality

## Implementation Approach

1. **Incremental Implementation Strategy**
   - First design the API and create doc tests
   - Implement basic functionality simply
   - Optimize performance incrementally

2. **Usability Focus**
   - Intuitive API for those familiar with Python's pandas
   - API design leveraging Rust's strengths in type safety
   - Comprehensive documentation and examples

3. **Test Strategy**
   - Unit tests for each feature
   - Compatibility tests with pandas
   - Performance tests through benchmarks

## Next Steps

1. **Community Building**
   - Establishing contribution guidelines
   - Organizing milestones and issues
   - Creating issues for beginners

2. **Documentation Enhancement**
   - Expanding API documentation
   - Creating tutorials and cookbooks
   - Use case gallery

3. ✅ **Updating Dependencies**
   - ✅ Updating all dependencies to the latest versions (as of April 2024)
   - ✅ Ensuring compatibility with the Rust 2023 ecosystem
   - ✅ Updates for security and performance improvements
   - ✅ Adapting to rand 0.9.0 API changes (`gen_range` → `random_range`)
   - ✅ Adapting to new API for Parquet compression constants

4. **Packaging**
   - Publication and distribution on crates.io
   - Versioning strategy
   - Dependency management

## Key Dependencies (Latest as of April 2024)

```toml
[dependencies]
num-traits = "0.2.19"        # Numeric type trait support
thiserror = "2.0.12"         # Error handling
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
chrono-tz = "0.10.3"         # Timezone processing
parquet = "54.3.1"           # Parquet file support
arrow = "54.3.1"             # Arrow format support
```

---

This roadmap outlines PandRS's goal to provide functionality equivalent to Python's pandas library while leveraging Rust's characteristics to create a high-performance data analysis library. Implementation priorities should be adjusted according to community interests and needs.
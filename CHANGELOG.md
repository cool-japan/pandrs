# Changelog

All notable changes to PandRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.4]

### Added
- **Enhanced DataFrame Operations:**
  - New `rename_columns()` method for flexible column renaming with HashMap mapping
  - New `set_column_names()` method for setting all column names at once
  - Enhanced Series name management with `set_name()` and `with_name()` methods
  - Improved type conversion utilities like `to_string_series()` for Series

- **String Accessor (.str) Implementation:**
  - Complete string accessor module with 25+ methods
  - Methods: `contains`, `startswith`, `endswith`, `upper`, `lower`, `replace`, `split`, `len`, `strip`, `extract`
  - Additional methods: `isalpha`, `isdigit`, `isalnum`, `isspace`, `islower`, `isupper`, `swapcase`
  - Full regex support with pattern matching capabilities and caching
  - Unicode normalization and character count support
  - Vectorized string operations for performance

- **DateTime Accessor (.dt) Implementation:**
  - Comprehensive datetime accessor for temporal operations
  - Basic datetime component access: `year`, `month`, `day`, `hour`, `minute`, `second`
  - Enhanced temporal properties: `week`, `quarter`, `weekday`, `dayofyear`, `days_in_month`, `is_leap_year`
  - Advanced date arithmetic: `add_days`, `add_hours`, `add_months`, `add_years` with overflow handling
  - Timezone-aware operations with `DateTimeAccessorTz` and chrono-tz integration
  - Business day support: `is_business_day`, `business_day_count`, `is_weekend`
  - Enhanced rounding support for "15min", "30S", and custom intervals
  - Date formatting and parsing: `strftime`, `timestamp`, `normalize`

- **Advanced Window Operations:**
  - Enhanced DataFrame window operations with feature parity to Series level
  - Rolling operations: `rolling(n).mean()`, `rolling(n).sum()`, `rolling(n).std()`, `rolling(n).min()`, `rolling(n).max()`
  - Expanding operations: `expanding().mean()`, `expanding().count()`, `expanding().std()`, `expanding().var()`
  - Exponentially weighted functions: `ewm(span=n).mean()`, `ewm(alpha=0.1).var()`, `ewm(halflife=n).std()`
  - Advanced window parameters: `min_periods`, `center`, `closed` boundaries (Left, Right, Both, Neither)
  - Multi-column operations with automatic numeric column detection
  - Time-based rolling windows with datetime column support
  - Custom aggregation functions with Arc-based closures

- **Enhanced I/O Capabilities:**
  - Excel Support Enhancement with multi-sheet support and formula preservation
  - Advanced Parquet features with schema evolution and predicate pushdown
  - Database integration expansion with async PostgreSQL and MySQL drivers
  - Connection pooling with async support and transaction management
  - Type-safe SQL query builder with fluent API
  - Real data extraction in Parquet/SQL operations (replacing placeholder implementations)

- **Query and Eval Engine:**
  - String expression parser for DataFrame.query() operations with full SQL-like syntax
  - Mathematical expression evaluator for DataFrame.eval() with comprehensive function support
  - Boolean expression optimization with short-circuiting and constant folding
  - Vectorized operations for simple column comparisons and performance optimization
  - JIT compilation for repeated expressions with automatic compilation thresholds
  - Built-in mathematical functions: sqrt, log, sin, cos, abs, power operations
  - Complex logical operations (AND, OR, NOT) with proper precedence handling

- **Advanced Indexing System:**
  - DatetimeIndex with full timezone support and frequency-based operations
  - PeriodIndex for financial and business period analysis (quarterly, monthly, weekly, daily, annual)
  - IntervalIndex for range-based and binned data indexing with equal-width and quantile-based cutting
  - CategoricalIndex with memory optimization and dynamic category management
  - Index set operations: union, intersection, difference, symmetric_difference
  - Specialized indexing operations for datetime filtering, period grouping, and interval containment

- **GPU-Accelerated Window Operations:**
  - Comprehensive GPU window operations module with intelligent GPU/JIT/CPU hybrid acceleration
  - GPU support for rolling, expanding, and EWM operations
  - Intelligent threshold-based decision making (50K+ elements for GPU)
  - Real-time performance monitoring and GPU usage ratio analysis
  - Seamless fallback to JIT/CPU when GPU is unavailable

- **Group-wise Window Operations:**
  - Group-wise rolling, expanding, and EWM operations
  - Multi-column group-wise operations with flexible column selection
  - Time-based group-wise window operations with datetime support
  - Integration with existing enhanced GroupBy functionality

### Changed
- **Dependency Updates:**
  - chrono: 0.4.38 (for arrow ecosystem compatibility)
  - chrono-tz: 0.9.0 (compatible with chrono 0.4.38)
  - arrow: 53.3.1 (compatible versions)
  - parquet: 53.3.1 (compatible versions)
  - datafusion: 30.0.0 (compatible with arrow 53.x)
  - rayon: 1.10.0
  - regex: 1.11.1
  - serde_json: 1.0.140
  - memmap2: 0.9.5
  - crossbeam-channel: 0.5.15
- **Python Bindings Updates:**
  - pyo3: 0.25.0
  - numpy: 0.25.0
- Enhanced Arrow integration with proper null value handling
- Improved type safety in data conversion processes
- API consistency with fluent interface design across all DataFrame operations

### Fixed
- Fixed arrow-arith dependency conflict (E0034: multiple applicable items in scope)
- Fixed CUDA optional compilation (prevents build failures when CUDA toolkit unavailable)
- Fixed JIT parallel example compilation errors (closure signatures, trait imports, type mismatches)
- Fixed test_multi_index_simulation assertion failure (string pool race condition, now uses integer codes)
- Fixed get_column_string_values method to return actual data instead of dummy values
- Fixed IO error handling tests
- Improved error handling and comprehensive documentation

### Technical Improvements
- **Comprehensive Test Coverage:**
  - 143+ core tests passing successfully
  - 26 new edge case tests covering boundary conditions, error handling, and invalid inputs
  - Stress testing for large datasets (100K+ rows) and concurrent operations
  - String pool concurrency testing with thread safety validation
  - Memory management and resource cleanup testing
- **Performance Optimizations:**
  - Zero compilation warnings in core library
  - Production-ready string and datetime accessors
  - Enterprise-grade I/O capabilities
  - Memory-efficient implementations suitable for large-scale data processing
- **Enhanced Documentation:**
  - Comprehensive examples demonstrating all new features
  - Performance benchmarks and optimization guidelines
  - Real-world use cases with financial time series analysis

### Breaking Changes
- Some internal APIs have been reorganized for better performance and consistency
- String pool integration may affect memory usage patterns (generally improving them)

## [0.1.0-alpha.3]

### Added
- Complete Extended ML Pipeline implementation with advanced feature engineering
  - Polynomial features generation (configurable degree)
  - Interaction features between column pairs
  - Binning/discretization with multiple strategies (equal width, equal frequency, quantile)
  - Rolling window operations (mean, sum, min, max, std, count)
  - Custom transformation support with lambda functions
  - PipelineContext for stage metadata and execution history
  - Performance monitoring and execution summaries
- Comprehensive financial analysis pipeline examples
- Advanced pipeline with monitoring and execution tracking

### Changed
- **Dependency Updates:**
  - chrono: 0.4.40 → 0.4.41
  - rayon: 1.9.0 → 1.10.0  
  - regex: 1.10.2 → 1.11.1
  - serde_json: 1.0.114 → 1.0.140
  - memmap2: 0.7.1 → 0.9.5
  - crossbeam-channel: 0.5.8 → 0.5.15
- **Python Bindings Updates:**
  - pyo3: 0.24.0 → 0.25.0
  - numpy: 0.24.0 → 0.25.0
- Updated TODO.md to reflect completion of all major planned features
- Updated IMPLEMENTATION_COMPLETION_SUMMARY.md with Extended ML Pipeline completion
- Fixed compilation issues in test files and examples

### Fixed
- Resolved compilation issues in extended ML pipeline module
- Fixed column access APIs to use proper `column_view.column()` method
- Fixed integer overflow issues in rolling window calculations
- Fixed type mismatches in test files and examples
- Removed references to unimplemented group_by functionality in tests

### Technical Improvements
- All 52 core library tests passing (54 total, 2 intentionally ignored)
- Zero compilation warnings or errors
- Clean codebase with comprehensive test coverage
- Production-ready build system

## [0.1.0-alpha.2]

### Added
- Just-In-Time (JIT) compilation system for high-performance operations
- SIMD vectorization support (AVX2/SSE2)
- Comprehensive parallel processing with Rayon integration
- Module structure reorganization with core/, compute/, storage/ architecture
- Distributed processing framework with DataFusion integration
- GPU acceleration framework (CUDA support)
- Advanced statistical functions and ML metrics
- Streaming data support with real-time analytics
- Memory-mapped file support for large datasets
- WebAssembly support for browser-based visualization

### Changed
- Complete module hierarchy reorganization
- Improved public API interfaces with clear re-exports
- Enhanced error handling and type safety

### Technical Details
- 50+ passing unit tests
- Comprehensive benchmarking suite
- Production-ready performance optimizations
- Memory efficiency improvements with string pooling

## [0.1.0-alpha.1]

### Added
- Initial DataFrame and Series implementations
- Basic data operations (filtering, sorting, joining)
- CSV/JSON/Parquet file I/O support
- Statistical analysis functions
- Categorical data type support
- Python bindings with PyO3
- Basic visualization capabilities

### Technical Foundation
- Rust-native implementation with zero-cost abstractions
- Thread-safe design with ownership model
- Modular architecture for extensibility
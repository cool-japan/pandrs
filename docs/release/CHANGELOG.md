# Changelog

All notable changes to PandRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
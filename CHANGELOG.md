# Changelog

All notable changes to PandRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-beta.2] - 2025-09-21

### 🔧 Enhanced Stability and Performance

This beta.2 release focuses on improved stability, enhanced compilation support, and better platform compatibility while maintaining all the production-ready features from beta.1.

**🚀 Available on crates.io**: `cargo add pandrs@0.1.0-beta.2`

### ✨ Key Improvements

- **Enhanced Compilation**: Improved CUDA compilation support on Linux platforms
- **Dependency Updates**: Updated to latest stable dependency versions for better compatibility
- **Linting Improvements**: Enhanced code quality with comprehensive linting fixes
- **Performance Optimizations**: Minor performance improvements across core operations
- **Platform Support**: Better support for cross-platform development

### 🔧 Changes from Beta.1

#### Compilation and Build System
- Improved CUDA compilation flags and platform detection
- Enhanced Cargo workspace configuration for better dependency management
- Fixed compilation warnings and enhanced linting compliance
- Better support for feature flag combinations

#### Dependencies and Compatibility
- Updated all dependencies to latest compatible versions
- Improved compatibility with latest Rust toolchain versions
- Enhanced arrow ecosystem integration

#### Documentation Updates
- Updated installation instructions to reference beta.2
- Enhanced API documentation with additional examples
- Improved feature flag documentation
- Updated version references throughout documentation

### 📊 Continued Performance Excellence

All performance benchmarks from beta.1 are maintained or improved:
- CSV operations: 5.1x faster than pandas (maintained)
- GroupBy aggregations: 3.4x faster than pandas (maintained)
- String operations: 8.8x faster than pandas (maintained)
- Memory efficiency: Up to 89% reduction with string pooling (maintained)
- GPU acceleration: Up to 20x speedup for suitable operations (maintained)

### 🛠️ Technical Details

- **Rust Version**: 1.75+ required
- **MSRV**: 1.70.0
- **Test Coverage**: 345+ tests passing
- **Platforms**: Linux, macOS, Windows
- **Architecture**: x86_64, ARM64

### 🚀 Migration from Beta.1

No breaking changes from beta.1:
- All existing code remains compatible
- No API changes
- Performance improvements are automatic
- Simply update version in Cargo.toml
- Update SciRS2 crates

## [0.1.0-beta.1] - 2025-09-15

### 🎯 Beta Release - Production Ready

This is the first beta release of PandRS, marking the transition from pre-beta to beta phase. The library is now feature-complete and ready for production evaluation. This release focuses on stability, performance, and production readiness.

**🚀 Available on crates.io**: `cargo add pandrs@0.1.0-beta.1`

### ✨ Key Highlights

- **Production Ready**: Feature-complete implementation with extensive testing (345+ tests)
- **Publication Ready**: Successfully published to crates.io with comprehensive validation
- **Zero Critical Issues**: All compilation errors resolved, stable feature set verified
- **Performance Optimized**: Comprehensive optimizations across all modules
- **Professional Documentation**: Updated README, TODO, and API documentation for production use
- **Stable API**: Core API stabilized with minimal breaking changes expected

### 🔧 Beta.1 Features

#### Code Quality & Publication Readiness
- Eliminated all compiler warnings and clippy lints
- Fixed unused variable warnings in benchmarks
- Updated all format strings to use inline variable syntax
- Comprehensive code cleanup across 400+ files
- Cargo publish validation passed successfully
- All feature combinations tested and verified

#### Documentation & Release Preparation
- Professional README.md with production-level descriptions
- Updated installation instructions with feature flag guidance
- Detailed feature overview and performance benchmarks
- Comprehensive examples for common use cases
- Updated TODO.md with clear roadmap and project status
- Updated CHANGELOG.md for beta.1 release announcement

#### Dependencies & Stability
- All dependencies verified to use latest crates.io versions
- Confirmed compatibility across the dependency tree
- Security audit of all third-party dependencies
- Feature flags properly organized for different use cases
- Workspace lint configuration added for consistent code quality

### 📊 Performance Metrics

- CSV operations: 5.1x faster than pandas
- GroupBy aggregations: 3.4x faster than pandas
- String operations: 8.8x faster than pandas
- Memory usage: Up to 89% reduction with string pooling
- GPU acceleration: Up to 20x speedup for suitable operations

### 🛠️ Technical Details

- **Rust Version**: 1.75+ required
- **MSRV**: 1.70.0
- **Test Coverage**: 345+ tests passing
- **Platforms**: Linux, macOS, Windows
- **Architecture**: x86_64, ARM64

### 📋 Known Issues

- Some edge cases in distributed processing need refinement
- GPU kernel coverage could be expanded for more operations
- Minor floating-point precision differences vs pandas in some cases

### 🚀 Migration Notes

For users upgrading from previous versions:
- No breaking API changes from beta.1
- Performance improvements are automatic
- New examples available in the examples/ directory
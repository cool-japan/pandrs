# PandRS Module Reorganization Plan

This document outlines the plan for reorganizing the PandRS module structure to improve maintainability, extensibility, and developer experience.

## Current Structure Assessment

The current structure has evolved organically as features were added. While it has worked well for incremental development, there are areas that can be improved:

1. **Inconsistent Naming Patterns**: Some modules follow different naming conventions.
2. **Feature Fragmentation**: Related features are sometimes spread across different modules.
3. **Unclear API Boundaries**: Public vs. internal APIs are not always clearly marked.
4. **Re-export Complexity**: The re-export patterns have grown in complexity.
5. **Extension Traits Consistency**: Different approaches to extension traits.

## Goals of Reorganization

1. **Improve Developer Experience**: Make the codebase easier to navigate and understand.
2. **Standardize Module Patterns**: Create a consistent pattern for module organization.
3. **Clarify API Boundaries**: Clear distinction between public and internal APIs.
4. **Enhance Extensibility**: Make it easier to add new features without major refactoring.
5. **Preserve Backward Compatibility**: Avoid breaking existing code.

## Proposed Module Structure

```
pandrs/
│
├── core/                    - Core data structures and traits
│   ├── data_value.rs        - Data value traits and implementations
│   ├── error.rs             - Error types and handling
│   ├── column.rs            - Base column traits and implementations 
│   ├── index.rs             - Index functionality
│   ├── multi_index.rs       - Multi-level index functionality
│   └── mod.rs               - Core module re-exports
│
├── dataframe/               - DataFrame implementations
│   ├── base.rs              - Base DataFrame implementation
│   ├── optimized.rs         - OptimizedDataFrame implementation
│   ├── transform.rs         - Shape transformation operations
│   ├── join.rs              - Join operations
│   ├── apply.rs             - Function application
│   ├── view.rs              - DataFrame views
│   ├── serialize.rs         - Serialization functionality
│   ├── gpu.rs               - GPU acceleration for DataFrames
│   └── mod.rs               - DataFrame module re-exports
│
├── series/                  - Series implementations
│   ├── base.rs              - Base Series implementation
│   ├── na.rs                - Missing value (NA) support
│   ├── categorical.rs       - Categorical data type
│   ├── functions.rs         - Common Series functions
│   ├── gpu.rs               - GPU acceleration for Series
│   └── mod.rs               - Series module re-exports
│
├── io/                      - Input/Output operations
│   ├── csv.rs               - CSV file operations
│   ├── json.rs              - JSON operations
│   ├── parquet.rs           - Parquet file operations
│   ├── excel.rs             - Excel file operations
│   ├── sql.rs               - SQL operations
│   └── mod.rs               - I/O module re-exports
│
├── compute/                 - Computation functionality
│   ├── parallel.rs          - Parallel processing
│   ├── lazy.rs              - Lazy evaluation
│   ├── gpu/                 - GPU computation
│   │   ├── mod.rs           - GPU module exports
│   │   ├── operations.rs    - GPU operations
│   │   ├── cuda.rs          - CUDA implementations
│   │   └── benchmark.rs     - Benchmarking utilities
│   └── mod.rs               - Computation module re-exports
│
├── stats/                   - Statistical functionality 
│   ├── descriptive.rs       - Descriptive statistics
│   ├── inference.rs         - Inferential statistics
│   ├── regression.rs        - Regression analysis
│   ├── sampling.rs          - Sampling methods
│   ├── categorical.rs       - Categorical statistics
│   ├── gpu.rs               - GPU-accelerated statistics
│   └── mod.rs               - Statistics module re-exports
│
├── ml/                      - Machine Learning functionality
│   ├── preprocessing.rs     - Data preprocessing
│   ├── metrics/             - Evaluation metrics
│   │   ├── mod.rs           - Metrics module exports
│   │   ├── regression.rs    - Regression metrics
│   │   └── classification.rs - Classification metrics
│   ├── dimension.rs         - Dimensionality reduction
│   ├── clustering.rs        - Clustering algorithms
│   ├── gpu.rs               - GPU-accelerated ML
│   └── mod.rs               - ML module re-exports
│
├── temporal/                - Time series functionality
│   ├── date_range.rs        - Date range generation
│   ├── frequency.rs         - Frequency definitions
│   ├── window.rs            - Window operations
│   ├── resample.rs          - Resampling operations
│   ├── gpu.rs               - GPU-accelerated time series
│   └── mod.rs               - Temporal module re-exports
│
├── viz/                     - Visualization functionality
│   ├── config.rs            - Plot configuration
│   ├── text.rs              - Text-based visualization
│   ├── plotters.rs          - Plotters integration
│   ├── direct.rs            - Direct plotting methods
│   ├── wasm.rs              - WebAssembly visualization
│   └── mod.rs               - Visualization module re-exports
│
├── storage/                 - Storage engines
│   ├── column_store.rs      - Column-oriented storage
│   ├── string_pool.rs       - String pooling
│   ├── disk.rs              - Disk-based storage
│   ├── memory_mapped.rs     - Memory-mapped files
│   └── mod.rs               - Storage module re-exports
│
├── streaming/               - Streaming data support
│   ├── stream.rs            - Data stream definitions
│   ├── connector.rs         - Stream connectors
│   ├── window.rs            - Windowed operations
│   ├── analytics.rs         - Real-time analytics
│   └── mod.rs               - Streaming module re-exports
│
├── util/                    - Utility functions and helpers
│   ├── conversion.rs        - Type conversion utilities
│   ├── iterator.rs          - Iterator utilities
│   ├── math.rs              - Mathematical utilities
│   └── mod.rs               - Utilities module re-exports
│
├── python/                  - Python binding support
│   ├── conversion.rs        - Python ↔ Rust conversion
│   ├── dataframe.rs         - DataFrame Python bindings
│   ├── series.rs            - Series Python bindings
│   ├── gpu.rs               - GPU Python bindings
│   └── mod.rs               - Python module re-exports
│
├── web/                     - WebAssembly support
│   ├── canvas.rs            - Canvas rendering
│   ├── dashboard.rs         - Interactive dashboards
│   ├── dom.rs               - DOM interaction
│   └── mod.rs               - Web module re-exports
│
└── lib.rs                   - Library entry point with top-level re-exports
```

## Public API Organization

We will follow these principles for public API organization:

1. **Entry Point**: `lib.rs` will re-export the most commonly used types and functions.
2. **Module Level**: Each module's `mod.rs` will re-export all public items from that module.
3. **Feature Flags**: Feature-gated functionality will be clearly marked.
4. **Extension Traits**: Will use a consistent naming pattern `*Ext` (e.g., `DataFrameGpuExt`).
5. **Documentation**: Each public API will have comprehensive documentation.

## Re-export Strategy

```rust
// Example re-export structure in lib.rs
pub use core::{DataFrame, Series, NA, NASeries, DataValue, Index, MultiIndex};
pub use dataframe::{JoinType, StackOptions, MeltOptions, UnstackOptions};
pub use series::{Categorical, CategoricalOrder};
pub use stats::{DescriptiveStats, TTestResult, LinearRegressionResult};
pub use ml::metrics::regression::{mean_squared_error, r2_score};
pub use ml::metrics::classification::{accuracy_score, f1_score};
pub use viz::{OutputFormat, PlotConfig, PlotType};
pub use compute::parallel::ParallelUtils;
pub use compute::lazy::LazyFrame;

// Feature-gated re-exports
#[cfg(feature = "cuda")]
pub use compute::gpu::{GpuConfig, init_gpu, GpuDeviceStatus, GpuBenchmark};
#[cfg(feature = "cuda")]
pub use dataframe::gpu::DataFrameGpuExt;
#[cfg(feature = "cuda")]
pub use temporal::gpu::SeriesTimeGpuExt;

#[cfg(feature = "wasm")]
pub use web::{WebVisualization, WebVisualizationConfig, ColorTheme};

// Example module-level re-exports in dataframe/mod.rs
pub use self::base::DataFrame;
pub use self::optimized::OptimizedDataFrame;
pub use self::transform::{StackOptions, MeltOptions, UnstackOptions};
pub use self::join::JoinType;
pub use self::apply::Axis;
pub use self::view::ColumnView;

// Optional feature re-exports
#[cfg(feature = "cuda")]
pub use self::gpu::DataFrameGpuExt;
```

## Backward Compatibility Strategy

To maintain backward compatibility:

1. **Keep old paths**: Maintain the existing import paths for at least one major version cycle.
2. **Deprecation notices**: Add deprecation notices to old imports.
3. **Re-export transitionally**: Use re-exports to allow both old and new import paths.
4. **Migration guide**: Provide a clear migration guide for users.

Example transitional re-export:
```rust
// In old location (src/dataframe/mod.rs)
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use pandrs::dataframe::OptimizedDataFrame instead"
)]
pub use crate::dataframe::optimized::OptimizedDataFrame;
```

## Implementation Approach

The reorganization will be phased in stages:

1. **Stage 1**: Design finalization and documentation
   - Complete this design document
   - Create detailed implementation tasks
   - Establish testing strategy

2. **Stage 2**: Core reorganization
   - Reorganize core, dataframe, and series modules
   - Update re-exports
   - Add backward compatibility layers

3. **Stage 3**: Feature modules reorganization
   - Reorganize stats, ml, temporal, and other feature modules
   - Update imports and re-exports
   - Refine documentation

4. **Stage 4**: Advanced optimization modules
   - Reorganize compute, storage, and specialized modules
   - Update benchmarks and tests
   - Complete transitional compatibility work

5. **Stage 5**: Documentation and examples update
   - Update all examples to use new import paths
   - Enhance API documentation
   - Create migration guides

## Timeline

- Design and planning: May 2024 ✅
- Core reorganization: May-June 2024 ✅
  - Create core/ directory with core module structure ✅
  - Create compute/ directory with compute module structure ✅
  - Create storage/ directory with storage module structure ✅
  - Create dataframe/ directory with module structure ✅
  - Create series/ directory with module structure ✅
  - Implement backward compatibility layers ✅
  - Continue implementing full modules ✅
- Feature modules reorganization: June 2024 🔄
  - Create stats/ module structure with improved organization ✅
  - Create ml/ module with refined structure ✅
  - Create temporal/ module with improved organization ✅
  - Create vis/ module with improved organization ✅
  - Update re-exports and backward compatibility layers ✅
- Advanced modules reorganization: June-July 2024
  - Implement distributed/ module reorganization ✅
    - Created distributed/core/ directory structure ✅
    - Created distributed/execution/ directory structure ✅
    - Created distributed/engines/ directory structure ✅
    - Created distributed/expr/ directory structure ✅
    - Created distributed/api/ directory structure ✅
    - Created distributed/window/ directory structure ✅
    - Created distributed/fault_tolerance/ directory structure ✅
    - Created distributed/explain/ directory structure ✅
    - Created distributed/schema_validator/ directory structure ✅
    - Implemented backward compatibility layers ✅
  - Refine compute/ module implementations
  - Enhance GPU acceleration integrations
- Documentation and examples: July 2024
  - Update all examples to use new imports
  - Create migration guides
  - Update API documentation
- Target completion: July 2024

## Success Criteria

1. All modules follow the new consistent pattern
2. All public APIs are properly documented
3. All examples and tests pass with the new structure
4. No breaking changes for existing code
5. Clean import paths for new code
6. Code is more maintainable and extensible

## Future Extensibility

The new structure is designed to accommodate future extensions:

1. **New Data Types**: The core module can easily incorporate new data types.
2. **Algorithmic Extensions**: Feature modules (stats, ml, etc.) can be extended.
3. **Integration Points**: New integrations (databases, file formats) go to appropriate modules.
4. **Acceleration Methods**: New acceleration techniques fit into compute module.
5. **Visualization Extensions**: New visualization methods go into viz module.

## Conclusion

This reorganization will significantly improve the maintainability and extensibility of the PandRS codebase. By establishing clear module boundaries, consistent naming patterns, and comprehensive documentation, we will enhance the developer experience and make it easier to contribute to the project. The approach ensures backward compatibility while setting up a cleaner, more intuitive structure for the future.
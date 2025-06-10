# PandRS Test Plan and Execution Strategy

Efficient testing strategies are necessary for large-scale codebases. This file manages the test plan and execution status.

## Test Classification and Execution Commands

### 1. Module-specific Unit Tests

```bash
# Core module tests
cargo test --test dataframe_test
cargo test --test series_test
cargo test --test index_test
cargo test --test na_test

# Feature-specific tests
cargo test --test apply_test
cargo test --test transform_test
cargo test --test groupby_test
cargo test --test categorical_test
cargo test --test categorical_na_test
cargo test --test categorical_df_test
cargo test --test temporal_test
cargo test --test window_test
cargo test --test multi_index_test

# I/O related tests
cargo test --test io_test

# ML related module tests
cargo test --test ml_basic_test

# Optimized implementation tests
cargo test --test optimized_dataframe_test
cargo test --test optimized_series_test
cargo test --test optimized_transform_test
cargo test --test optimized_apply_test
cargo test --test optimized_groupby_test
cargo test --test optimized_io_test
cargo test --test optimized_lazy_test
cargo test --test optimized_multi_index_test
cargo test --test optimized_window_test
```

### 2. Example Code Verification

Verify example code by dividing into groups:

```bash
# Group 1: Basic functionality
cargo check --example basic_usage
cargo check --example benchmark_comparison
cargo check --example optimized_basic_usage
cargo check --example string_optimization_benchmark

# Group 2: Data operations
cargo check --example categorical_example
cargo check --example categorical_na_example
cargo check --example na_example
cargo check --example optimized_categorical_example
cargo check --example optimized_transform_example
cargo check --example transform_example

# Group 3: Grouping and aggregation
cargo check --example groupby_example
cargo check --example optimized_groupby_example
cargo check --example pivot_example

# Group 4: Time series and window operations
cargo check --example time_series_example
cargo check --example window_operations_example
cargo check --example optimized_window_example
cargo check --example dataframe_window_example

# Group 5: Index operations
cargo check --example multi_index_example
cargo check --example optimized_multi_index_example

# Group 6: ML basic
cargo check --example ml_basic_example
cargo check --example ml_model_example
cargo check --example ml_pipeline_example

# Group 7: ML applications
cargo check --example ml_clustering_example
cargo check --example ml_anomaly_detection_example
cargo check --example ml_dimension_reduction_example
cargo check --example ml_feature_engineering_example

# Group 8: Visualization
cargo check --example visualization_example
cargo check --example visualization_plotters_example
cargo check --example plotters_simple_example
cargo check --example plotters_visualization_example

# Group 9: Parallel processing
cargo check --example parallel_example
cargo check --example parallel_benchmark
cargo check --example lazy_parallel_example

# Group 10: Others
cargo check --example parquet_example
cargo check --example performance_bench
cargo check --example benchmark_million
```

## Testing Strategy

1. **Testing Based on Scope of Change**:
   - Changes to specific modules: Test that module and directly related example code
   - API changes: Test all modules affected by the change
   - Basic feature additions/fixes: Test related features and example code

2. **Phased Test Execution**:
   - First use `cargo check` to detect compilation errors
   - Then run related unit tests
   - Finally verify related example code

3. **Continuous Testing**:
   - Ensure all related tests pass before creating a PR
   - Wait for automated tests in CI/CD before merging

## Current Work and Test Progress (2025/04/06)

### ML-related Modification Test Plan and Results

#### Unit Tests
- [x] cargo test --test ml_basic_test

#### Example Code Compilation Check
- [x] cargo check --example ml_dimension_reduction_example
- [x] cargo check --example ml_clustering_example
- [x] cargo check --example ml_anomaly_detection_example
- [x] cargo check --example ml_basic_example
- [x] cargo check --example ml_feature_engineering_example
- [x] cargo check --example ml_pipeline_example
- [ ] cargo check --example ml_model_example (requires substantial modifications)

#### GitHub Actions Tests
- [ ] Confirm CI tests completion for latest commit (8e35eb7)

### Fixed Issues
1. Updated rand API (for 0.9.0 compatibility)
   - `gen_range` → `random_range`
   - `thread_rng` → `rng`
   - `random` → `gen` compatibility

2. Updated constructor patterns
   - `Float64Column::new(values, has_nulls, name)` → `Float64Column::with_name(values, name)`
   - `Int64Column::new(values, has_nulls, name)` → `Int64Column::with_name(values, name)`
   - `StringColumn::new(values, has_nulls, name)` → `StringColumn::with_name(values, name)`

3. Resolved ownership issues
   - Fixed reuse of moved values
   - Used `clone()` where necessary

4. Fixed string formatting
   - Output of values with hidden types: `{}` → `{:?}`
   - Fixed placeholders

5. API compatibility
   - Added `Transformer` trait import
   - Adapted to new API patterns
   - Implemented alternative processing for `head()` method
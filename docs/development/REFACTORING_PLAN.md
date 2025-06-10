# PandRS Refactoring Plan

## Implementation Status (2025/4/7)

The following work has been completed:

1. **Refactoring Plan Formulation and Implementation**:
   - Divided implementation by functionality while maintaining existing APIs
   - Completed migration using delegation pattern
   - Organized functionality into 13 modules

2. **Organization and Migration of All Features**:
   - Completed migration of all functionality to dedicated files in the split_dataframe directory
   - Separated core functionality into `core.rs`
   - Separated column operations into `column_ops.rs`
   - Separated data operations into `data_ops.rs`
   - Separated index operations into `index.rs`
   - Separated I/O operations into `io.rs`
   - Separated join operations into `join.rs`
   - Separated group operations into `group.rs`
   - Separated row operations into `row_ops.rs`
   - Separated selection operations into `select.rs`
   - Separated serialization operations into `serialize.rs`
   - Separated sort operations into `sort.rs`
   - Separated apply operations into `apply.rs`
   - Separated parallel processing into `parallel.rs`

3. **Compatibility Assurance**:
   - Improved internal implementation while maintaining old APIs
   - Significantly reduced code duplication
   - Unified all error handling
   - Optimized pattern matching

4. **Enhanced Testing**:
   - Verified all existing tests work correctly
   - Added dedicated tests for new functionality (join operations, etc.)
   - Expanded compatibility tests

## Background

The OptimizedDataFrame implementation has become concentrated in the `dataframe.rs` file, making it too large and difficult to manage. Meanwhile, a `split_dataframe/` directory has already been set up to store implementations divided by functionality.

This refactoring aims to divide the implementation by functionality while maintaining the existing API.

## Approach

1. **Maintain Existing API Unchanged**
   - Keep the public API in `dataframe.rs` unchanged so existing code and samples continue to work
   - Apply the same pattern when adding new functionality

2. **Move Internal Implementation**
   - Move the internal implementation of each method in `dataframe.rs` to the appropriate file in `split_dataframe/`
   - Classify by functionality and move implementation to the corresponding file

3. **Delegation Pattern**
   - Make methods in `dataframe.rs` thin wrappers that just call the implementation in `split_dataframe/`
   - This hides implementation details in `split_dataframe/` while maintaining the external API

## Example Implementation Pattern

```rust
// dataframe.rs
impl OptimizedDataFrame {
    pub fn some_method(&self, args) -> Result<T> {
        // Just calls the implementation in split_dataframe/module.rs
        use crate::optimized::split_dataframe::module;
        module::some_method_impl(self, args)
    }
}
```

```rust
// split_dataframe/module.rs
pub(crate) fn some_method_impl(df: &OptimizedDataFrame, args) -> Result<T> {
    // Actual implementation
    // ...
}
```

## Feature Classification and Destination Files

| Feature Category | Target Methods | Destination File |
|------------|------------|-------------|
| Basic Structures and Constructors | `new()`, `with_index()`, `with_multi_index()`, `with_range_index()` | `core.rs` |
| Index Operations | `set_multi_index()`, `set_column_as_index()`, `set_simple_index()`, `get_index()` | `index.rs` |
| Column Operations | `add_column()`, `rename_column()`, `remove_column()` | `column_ops.rs` |
| Data Transformation | `stack()`, `unstack()`, `conditional_aggregate()` | `data_ops.rs` |
| Join Functionality | `inner_join()`, `left_join()`, `right_join()`, `outer_join()` | `join.rs` |
| Input/Output | `to_csv()`, `from_csv()`, `to_json()`, `from_json()`, `to_parquet()`, `from_parquet()` | `io.rs` |
| Grouping and Aggregation | `groupby()`, `par_groupby()` | `group.rs` |
| Statistical Functions | `describe()`, `corr()` | `stats.rs` |

## Migration Strategy

1. **Phased Approach**
   - Start by implementing new functionality (e.g., join functionality) first with this pattern
   - Gradually migrate existing functionality

2. **Testing Strategy**
   - Run corresponding tests after each feature migration to verify behavior remains unchanged
   - Add dedicated tests for refactoring to verify API compatibility

3. **Documentation**
   - Update code comments to record that internal implementation has moved
   - Update coding style guide for adding new functionality

## Expected Benefits

1. Code becomes easier to manage
2. Separation of concerns by functionality is achieved
3. Code becomes easier to process within context by splitting across multiple files
4. No migration cost as existing external APIs are maintained

## Corresponding Migration Plan

This plan is implemented in coordination with the migration plan defined in `MIGRATION_STRATEGY.md`.
Work can proceed in parallel with Phase 1 (Facade Pattern Implementation).
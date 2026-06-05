# PandRS Project Status & Roadmap

## Current Release

**Version:** 0.4.0
**Release Date:** June 2026
**Status:** Production Ready — Full SciRS2-Core Integration Release
**Test Coverage:** 1893 tests passing (nextest, scirs2 features), zero clippy warnings

### Acknowledged upstream Pure Rust tech debt (feature-gated; default build unaffected)
- `parquet` / `distributed` / `flight` pull `flate2`/`zstd`/`lz4_flex`/`snap`/`bzip2`/`brotli`/`miniz_oxide` via upstream arrow/parquet/datafusion/async-compression.
- `cloud-storage` pulls `ring` (C+asm) via `object_store 0.13.2` — upstream uses `ring::{hmac,digest,signature,rand}` directly and exposes `ring::error` in public API (see `/tmp/pandrs_rustls_phase2_report.md`).

## Completed Features (v0.1.0)

### Core Data Structures ✓
- [x] Series with full pandas-compatible API
- [x] DataFrame with comprehensive operations
- [x] MultiIndex support for hierarchical data
- [x] Categorical data type with memory optimization
- [x] Missing value (NA) handling across all types

### Data Operations ✓
- [x] Advanced indexing and selection
- [x] Boolean indexing and filtering
- [x] Sorting and ranking operations
- [x] Duplicate detection and removal
- [x] Type conversion and casting

### Aggregation & Analytics ✓
- [x] GroupBy with multiple aggregation functions
- [x] Window functions (rolling, expanding, EWM)
- [x] Pivot tables and cross-tabulation
- [x] Statistical functions and hypothesis testing
- [x] Time series resampling and analysis

### String & DateTime Operations ✓
- [x] String accessor (.str) with 25+ methods
- [x] DateTime accessor (.dt) with timezone support
- [x] Regular expression support
- [x] Text processing and cleaning functions

### I/O & Interoperability ✓
- [x] CSV reader/writer with parallel processing
- [x] JSON support (records and columnar)
- [x] Parquet integration with compression
- [x] Excel read/write with multi-sheet support
- [x] ~~SQL database connectivity~~ (removed in v0.3.0 — Pure Rust policy)
- [x] Arrow format integration

### Performance Optimizations ✓
- [x] SIMD vectorization for numerical operations
- [x] SIMD-accelerated string operations (ASCII fast path)
  - Case conversion (upper/lower)
  - Character classification (digits, alpha, whitespace)
  - Pattern matching (byte search, count)
  - Batch operations with parallel processing
- [x] Parallel processing with Rayon
- [x] JIT compilation for hot paths
- [x] String pooling for memory efficiency
- [x] Zero-copy operations where possible

### Advanced Features ✓
- [x] Machine learning metrics and utilities
- [x] Distributed computing with DataFusion
- [x] GPU acceleration (CUDA support)
- [x] Python bindings with PyO3
- [x] WebAssembly compilation
- [x] Text-based visualization (ASCII/Unicode charts)
  - Histograms, bar charts, line plots, scatter plots
  - Sparklines for inline mini charts
  - No external dependencies required

### Pandas API Compatibility ✓
- [x] DataFrame functional methods
  - assign() for adding computed columns
  - pipe() for method chaining
  - isin() for membership testing (string and numeric)
  - apply() for custom row/column transformations
- [x] Selection and sorting
  - nlargest()/nsmallest() for top-N selection
  - idxmax()/idxmin() for finding extrema indices
  - head()/tail() for selecting first/last N rows
  - sample() for random sampling (with/without replacement)
- [x] Ranking and statistics
  - rank() with multiple methods (average, min, max, first, dense)
  - describe() for comprehensive statistical summaries
  - corr() for correlation matrices
  - cov() for covariance matrices
  - quantile() for percentile calculations
- [x] Data transformation
  - clip() for value clamping
  - between() for range checking
  - transpose() for row/column swapping
  - replace() for value substitution (string and numeric)
  - abs() for absolute values
  - round() for decimal rounding
  - drop_columns() for removing columns
  - rename_columns() for renaming with mapper
  - abs_column() for column-wise absolute value
  - round_column() for column-wise rounding with precision
- [x] Cumulative operations
  - cumsum(), cumprod(), cummax(), cummin()
  - shift() for time series operations
  - pct_change() for percentage changes
  - diff() for discrete differences
- [x] Frequency and counting
  - value_counts() for frequency analysis
  - nunique() for unique value counts
  - unique() for getting unique values (string and numeric)
- [x] Missing data handling
  - fillna() for filling NaN values with specified value
  - dropna() for removing rows with NaN values
  - isna() for detecting NaN values (boolean mask)
  - ffill() for forward-filling NaN values
  - bfill() for backward-filling NaN values
- [x] DataFrame-level aggregations
  - sum_all() for summing all numeric columns
  - mean_all() for averaging all numeric columns
  - std_all() for standard deviation of all columns
  - var_all() for variance of all columns
  - min_all() for minimum values across columns
  - max_all() for maximum values across columns
- [x] Multi-column sorting
  - sort_values() for sorting by single column
  - sort_by_columns() for multi-key sorting with mixed order
- [x] Metadata and utilities
  - memory_usage() for memory profiling
- [x] Conditional operations
  - where_cond() for conditional value replacement (keep where True)
  - mask() for conditional value replacement (replace where True)
- [x] Duplicate handling
  - drop_duplicates() with keep options (first, last, none)
- [x] Column type selection
  - select_dtypes() to filter columns by type (numeric, string/object)
- [x] Boolean aggregations
  - any_numeric() to check for non-zero values
  - all_numeric() to verify all values are non-zero
  - count_valid() to count non-NA values per column
  - any_column()/all_column() for column-level boolean tests
  - count_na() for counting NaN values
- [x] Element-wise comparisons
  - gt()/ge()/lt()/le() for comparison operators
  - eq_value()/ne_value() for equality comparisons
- [x] Advanced clipping
  - clip_lower()/clip_upper() for one-sided clipping
- [x] Product aggregation
  - prod() for computing product of values
- [x] Column arithmetic
  - add_columns()/sub_columns()/mul_columns()/div_columns()
  - mod_column()/floordiv() for modulo and floor division
  - neg() for negation, sign() for sign extraction
- [x] Value coalescing
  - coalesce() for combining columns with NaN fallback
  - first_valid()/last_valid() for finding valid values
- [x] Infinity handling
  - is_finite()/is_infinite() for detecting special values
  - replace_inf() for replacing infinite values
- [x] Numeric rounding functions
  - floor()/ceil()/trunc() for value rounding
  - fract() for fractional part extraction
  - reciprocal() for computing 1/x
- [x] Value utilities
  - count_value() for counting specific values
  - fillna_zero() for quick NaN replacement
  - nunique_all() for unique counts across columns
  - is_between() for range checking
- [x] Column analysis methods
  - describe_column() for single-column statistics
  - memory_usage_column() for column memory usage
  - range() for computing max-min
  - abs_sum() for sum of absolute values
  - is_unique() for checking uniqueness
  - mode_with_count() for mode and its frequency
  - has_nulls() for detecting NaN presence
- [x] DataFrame ordering
  - reverse_columns() to reverse column order
  - reverse_rows() to reverse row order
- [x] Data reshaping
  - melt() for unpivoting from wide to long format
  - explode() for expanding list-like columns into rows
- [x] Duplicate detection
  - duplicated() for marking duplicate rows
- [x] Statistical functions
  - skew() for computing skewness
  - kurtosis() for computing kurtosis
  - mode_numeric()/mode_string() for finding most frequent values
  - median_all() for median of all numeric columns
  - product_all() for product of all numeric columns
  - sem() for standard error of the mean
  - mad() for mean absolute deviation
  - pct_rank() for percentile ranking
  - argmax()/argmin() for index of extrema
  - geometric_mean() for geometric average
  - harmonic_mean() for harmonic average
  - iqr() for interquartile range
  - cv() for coefficient of variation
  - percentile_value() for specific percentile
  - trimmed_mean() for outlier-resistant mean
- [x] Exponential weighted functions
  - ewma() for exponentially weighted moving average
- [x] Index operations
  - iloc() for accessing rows by integer position
  - iloc_range() for slicing rows by integer range
  - first_valid_index()/last_valid_index() for finding valid data
- [x] Column naming utilities
  - add_prefix()/add_suffix() for bulk column renaming
- [x] Data filtering
  - filter_by_mask() for boolean filtering
  - notna() for detecting non-NA values
- [x] Conversion utilities
  - copy() for deep copying
  - to_dict() for converting to dictionary
  - percentile() for percentile calculations
- [x] DataFrame comparison
  - info() for DataFrame summary information
  - equals() for equality comparison (NaN-aware)
  - compare() for finding differences between DataFrames
  - keys() for getting column names
- [x] Column manipulation
  - pop_column() for removing and returning columns
  - insert_column() for inserting at specific position
  - reindex_columns() for reordering columns
  - align() for aligning two DataFrames
- [x] Rolling window functions
  - rolling_sum() for rolling sum with min_periods
  - rolling_mean() for rolling mean with NaN handling
  - rolling_std() for rolling standard deviation
  - rolling_var() for rolling variance
  - rolling_min()/rolling_max() for rolling extrema
  - rolling_median() for rolling median
  - rolling_count() for count of non-NaN values
  - rolling_apply() for custom rolling functions
- [x] Expanding window functions
  - expanding_sum() for cumulative sum
  - expanding_mean() for cumulative mean
  - expanding_std() for cumulative standard deviation
  - expanding_var() for cumulative variance
  - expanding_min()/expanding_max() for cumulative extrema
  - expanding_apply() for custom expanding functions
- [x] Data normalization
  - zscore() for z-score normalization
  - normalize() for min-max normalization
  - value_range() for getting min/max range
- [x] Binning operations
  - cut() for equal-width binning
  - qcut() for quantile-based binning
- [x] Additional utilities
  - crosstab() for cross-tabulation
  - transform() for element-wise transformation
  - cumcount() for cumulative non-NA count
  - nth() for accessing rows with negative indexing

## Known Issues & Limitations

### Performance
- GPU kernel coverage could be expanded
- Some edge cases in distributed processing need refinement
- Note: Large string operations have been optimized with SIMD acceleration

### Compatibility
- Minor differences in floating-point precision compared to pandas
- Some advanced pandas features not yet implemented
- Excel formula preservation not supported

### Documentation
- API documentation needs expansion
- More comprehensive examples needed
- Performance tuning guide in progress

## Roadmap

### v0.1.0 - First Stable Release (December 2025) ✓
- [x] Improve temporary file handling in tests ✓
  - RAII wrappers for automatic cleanup (TempTestFile, TempTestDir)
  - Environment variable support (TMPDIR, TEMP, TMP)
  - Unique file name generation to avoid collisions
  - Comprehensive test utilities documentation
- [x] Support TMPDIR and other environment variables for temporary files ✓
- [x] Implement proper test cleanup for all temporary files ✓
- [x] Enhanced SIMD coverage for all operations ✓
- [x] Improved memory management for large datasets ✓
  - Arena allocator for bulk memory allocation
  - Memory pool for efficient large dataset handling
  - Zero-copy operations and cache-aligned allocations
- [x] Advanced query optimization ✓
- [x] Comprehensive benchmark suite ✓
- [x] Production deployment guide
- [x] Native machine learning algorithms ✓
  - Decision Trees (CART algorithm)
  - Random Forest (bootstrap ensemble)
  - Gradient Boosting (sequential boosting)
  - Multi-layer Perceptron Neural Networks
- [x] Advanced time series forecasting ✓
- [x] Graph analytics support ✓
- [x] Streaming data processing ✓
- [x] Real-time analytics dashboard ✓
  - Metrics collection (counters, gauges, histograms, timers)
  - Operation tracking with category-based statistics
  - Alert management with configurable rules
  - Resource monitoring (CPU, memory, throughput)
- [x] Data versioning and lineage ✓
- [x] Advanced security features ✓
  - Enterprise authentication (JWT/OAuth 2.0)
  - API Key management with rate limiting
  - Session management
  - Role-based access control (RBAC)
- [x] Audit logging ✓
- [x] Multi-tenancy support ✓
- [x] Enterprise authentication ✓

### v0.2.0 - Enterprise-Grade Release (March 2026) ✓
- [x] ReBAC (Relationship-Based Access Control) ✓
- [x] Comprehensive documentation (300+ pages) ✓
- [x] Enterprise support (three-tier model) ✓
- [x] LTS commitment (24-month) ✓
- [x] Security hardening (zero vulnerabilities in default build) ✓
- [x] Eliminated 6,984 unwrap() calls ✓
- [x] Large file refactoring (<2000 lines policy) ✓

### v0.3.0 - Pure Rust & Ecosystem (March 2026) ✓
- [x] Advanced lazy evaluation engine with query optimization
- [x] Streaming DataFrame operations for out-of-core processing
- [ ] R and Julia language bindings
- [x] Cloud-native storage backends (S3, GCS, Azure Blob)
- [x] Advanced visualization library (SVG/HTML output)
- [x] DataFrame schema evolution and migration tools
- [x] Plugin system for custom data sources and transforms
- [ ] Automated performance regression testing in CI
- [x] Full SciRS2-Core integration for scientific computing ✓
- [x] Arrow Flight RPC support for distributed data transfer
- [x] Removed SQL dependencies for Pure Rust compliance (no C/Fortran deps)
- [x] Dependency upgrades (parquet/arrow 58.1, datafusion 53.0, cranelift 0.130)

### v0.3.1 - Quality & Pure Rust Polish (April 2026) ✓
- [x] Fix broken intra-doc link in `src/io/excel.rs` (private module reference)
- [x] Replace `calamine` + `simple_excel_writer` with OxiARC-backed xlsx (Pure Rust policy)
- [x] Remove -sys crate violations (Pure Rust policy follow-on)
- [x] All 1813 tests passing, 117 doc tests passing
- [x] Zero clippy warnings (including with `-D warnings`)
- [x] Rustdoc builds cleanly with `-D warnings`

### v0.3.2 - Workspace Consolidation (April 2026) ✓
- [x] Convert root Cargo.toml to virtual workspace with ~55 shared dependencies
- [x] py_bindings inherits version/authors/edition/license from workspace root
- [x] Fix clippy::assertions_on_constants violations in tests and stats modules
- [x] Replace hardcoded /tmp/ paths with std::env::temp_dir() for portability
- [x] Add shift() method to NASeries and PyNASeries with pandas semantics
- [x] Add prelude module for convenient access to core types
- [x] All 1818 tests passing (nextest) + 118 doc tests
- [x] Zero clippy warnings, clean rustdoc with -D warnings

### v0.4.0 - Full SciRS2-Core Integration (May 2026) ✓
- [x] SciRS2-Core policy compliance: all `rand`/`ndarray` direct deps routed through `scirs2_core::random` / `scirs2_core::ndarray`; `rand_compat` shim retired; direct `rand`/`ndarray` deps removed from pandrs crate
- [x] Expanded scirs2 stats: Spearman/Kendall correlation, covariance matrix, paired t-test, chi-square, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Shapiro-Wilk, KS two-sample
- [x] Expanded scirs2 linalg: QR, Cholesky, LU, lstsq, pinv, matrix_norm, matrix_rank, condition_number
- [x] `SciRS2Ext` DataFrame trait extended with 6 new ergonomic methods
- [x] `skewness` and `kurtosis_excess` free functions (no scirs2 feature required)
- [x] MultiIndex missing-value support (code == -1 pandas NA sentinel)
- [x] Model serving `load_model` fully implemented via `Arc<dyn ModelServing>`
- [x] JIT expression tree real recursive interpreter (all ExpressionNode variants)
- [x] AutoML `create_estimator` instantiates real `SupervisedAdapter<M>` wrappers
- [x] `RandomizedSearchCV::fit` implements real k-fold cross-validation
- [x] Dead `src/index_impl/` directory removed
- [x] 16 `.disabled`/`.bak` example files removed
- [x] Fixed oxiarc-core version conflict blocking scirs2 feature compilation
- [x] `examples/scirs2_integration_example.rs` added
- [x] 1771+ tests passing (nextest, scirs2 feature)

### v0.4.1 - ML/Statistical Correctness (May 2026) ✓
- [x] PCA: real Jacobi eigendecomposition + correct transform (was: zeros + String-column rename)
- [x] t-SNE: real gradient descent with perplexity tuning + momentum (was: all-zeros embedding)
- [x] DBSCAN: real density clustering with BFS region-growing + noise labelling (was: all zeros)
- [x] AgglomerativeClustering: real bottom-up hierarchical clustering, all 4 linkage variants (was: all zeros)
- [x] silhouette score: real mean a/b coefficient (was: hardcoded 0.75)
- [x] LogisticRegression: real IRLS training + sigmoid predict/proba + real metrics (was: all zeros)
- [x] LinearRegression::cross_validate: real k-fold (was: hardcoded r2=0.8)
- [x] IsolationForest: real isolation trees with path-length scoring (was: RNG-fabricated)
- [x] LocalOutlierFactor: real k-NN LOF algorithm (was: no-op)
- [x] OneClassSVM: real SVDD kernel-distance scoring (was: RNG-fabricated)
- [x] RocAuc scorer: real Mann-Whitney AUC (was: hardcoded 0.75)
- [x] chi2_scores / mutual_info_scores: real χ² and histogram-based MI (was: all 1.0 / all 0.5)
- [x] HyperparameterGrid::parameter_combinations: real Cartesian product (was: one combo)
- [x] GridSearchCV/RandomizedSearchCV (models/selection): real k-fold CV scoring (was: hardcoded 0.9)
- [x] RandomizedSearchCV (model_selection): real k-fold CV with populated cv_results_ (was: empty Vec)
- [x] learning_curve / validation_curve: real per-size/per-param k-fold CV (was: hardcoded 0.9/0.8)
- [x] select_features: all 4 strategies data-dependent (RecursiveElimination/L1Based/TreeBased/MutualInformation)
- [x] RobustScaler/QuantileTransformer/PowerTransformer: real implementations (was: silent StandardScaler)
- [x] Statistical p-values: real chi2_sf + normal_sf; Ljung-Box/Box-Pierce/Breusch-Godfrey/Friedman/KW use χ² distribution (was: binary thresholds)
- [x] Shapiro-Wilk p-value: Royston log-transform approximation (was: binary threshold)
- [x] Kruskal-Wallis: computes own H statistic instead of forwarding Friedman (was: wrong)
- [x] examples/ml_real_algorithms_example.rs added (PCA/DBSCAN/LogisticRegression/IsolationForest/LOF/AgglomerativeClustering)
- [x] 1800+ tests passing (all-safe feature)

### v1.0.0 - Production Release (Q2 2026)
- [ ] Full pandas API compatibility (Complete - 100%)
  - [ ] Core functional methods (assign, pipe, isin, apply)
  - [ ] Selection methods (nlargest, nsmallest, idxmax, idxmin, head, tail, sample)
  - [ ] Ranking and cumulative operations (rank, cumsum, cumprod, etc.)
  - [ ] Statistical analysis (describe, corr, cov, value_counts, quantile)
  - [ ] Data transformation (clip, between, transpose, replace, abs, round)
  - [ ] Time series operations (pct_change, diff, shift)
  - [ ] Column operations (drop_columns, rename_columns)
  - [ ] Utility methods (unique, unique_numeric, memory_usage)
  - [ ] Missing data handling (fillna, dropna, isna)
  - [ ] Advanced missing data methods (fillna_method with ffill/bfill, interpolate)
  - [ ] DataFrame-level aggregations (sum_all, mean_all, std_all, var_all, min_all, max_all)
  - [ ] Multi-column sorting (sort_values, sort_by_columns)
  - [ ] DataFrame merging and joining (merge with inner, left, right, outer join types)
  - [ ] GroupBy operations (groupby_multi with sum, mean, min, max, std, var, count, first, last, agg)
  - [ ] DataFrame concatenation (concat with row-wise and column-wise support)
  - [ ] Conditional operations (where_cond, mask)
  - [ ] Duplicate handling (drop_duplicates with first/last/none)
  - [ ] Column type selection (select_dtypes)
  - [ ] Boolean aggregations (any_numeric, all_numeric, count_valid)
  - [ ] DataFrame ordering (reverse_columns, reverse_rows)
  - [ ] Data reshaping (melt, explode)
  - [ ] Duplicate detection (duplicated)
  - [ ] Advanced statistics (skew, kurtosis, mode, median_all, product_all)
  - [ ] Exponential weighted functions (ewma)
  - [ ] Index operations (iloc, iloc_range, first_valid_index, last_valid_index)
  - [ ] Column naming utilities (add_prefix, add_suffix)
  - [ ] Data filtering and detection (filter_by_mask, notna)
  - [ ] Conversion utilities (copy, to_dict, percentile)
  - [ ] Advanced reshaping (stack, unstack, pivot)
  - [ ] Type conversion (astype for numeric/string conversion)
  - [ ] Element-wise operations (applymap for custom functions)
  - [ ] Multiple aggregations (agg for combining aggregations)
  - [ ] Data type inspection (dtypes for column type information)
  - [ ] Value manipulation (set_values for index-based assignment)
  - [ ] Query operations (query_eq, query_gt, query_lt, query_contains)
  - [ ] Column selection (select_columns for subsetting)
  - [ ] Scalar arithmetic (add_scalar, mul_scalar, sub_scalar, div_scalar)
  - [ ] Mathematical functions (pow, sqrt, log, exp)
  - [ ] Column-wise arithmetic (col_add, col_mul, col_sub, col_div)
  - [ ] Row iteration (iterrows, to_records, items)
  - [ ] Fast scalar access (at, iat, get_value)
  - [ ] Row manipulation (drop_rows, take, sample_frac)
  - [ ] Index management (set_index, reset_index)
  - [ ] DataFrame utilities (shape, size, empty, first_row, last_row)
  - [ ] Column operations (swap_columns, sort_columns, rename_column, get_column_by_index)
  - [ ] DataFrame combination (update, combine, lookup)
  - [ ] Categorical encoding (to_categorical with mapping)
  - [ ] Row hashing (row_hash, duplicated_rows)
  - [ ] Column statistics (var_column, std_column, corr_columns, cov_columns)
  - [ ] String operations (str_lower, str_upper, str_strip, str_contains, str_replace, str_split, str_len)
  - str_startswith()/str_endswith() for prefix/suffix matching
  - str_pad_left()/str_pad_right() for string padding
  - str_slice() for substring extraction
  - str_count() for counting pattern occurrences
  - str_repeat() for repeating strings
  - str_center() for centering strings
  - str_zfill() for zero-filling
  - [ ] Column type conversion (get_column_as_f64, get_column_as_string)
  - [ ] GroupBy with custom functions (groupby_apply)
  - [ ] Complete time series resampling and frequency conversion
  - [ ] Full categorical data support with efficient memory usage
- [ ] Stabilized public API ✓
- [ ] Comprehensive documentation ✓
- [ ] Enterprise support options ✓
- [ ] Long-term support (LTS) commitment ✓

## Development Priorities

### Immediate (Post-Release)
1. Address user feedback from initial release
2. Performance optimization for identified bottlenecks
3. Documentation improvements
4. Example notebook collection

### Short Term
1. Expand test coverage to 95%+
2. Implement missing pandas features based on usage
3. Optimize distributed processing
4. Enhance error messages and debugging

### Long Term
1. Full ecosystem integration (R, Julia)
2. Advanced visualization library
3. Cloud-native deployment options
4. Automated performance regression testing

## Contributing

We welcome contributions in the following areas:

### High Priority
- Performance optimizations
- Documentation and examples
- Test coverage improvements
- Bug fixes and stability

### Medium Priority
- New feature implementations
- Integration with other tools
- Benchmark comparisons
- Use case examples

### Getting Started
1. Fork the repository
2. Check open issues labeled "good first issue"
3. Read CONTRIBUTING.md for guidelines
4. Join our Discord for discussions

## Testing & Quality

### Current Status
- 1813 tests passing (nextest) + 117 doc tests
- Comprehensive property-based testing
- Continuous integration with GitHub Actions
- Regular performance regression testing

### Quality Metrics
- Zero compiler warnings policy ✓
- Clippy lints enforced ✓
- Rustfmt formatting required ✓
- Documentation coverage tracked

## Release Process

### Stable Release (Current)
1. Feature freeze - no new features
2. Focus on stability and performance
3. Address critical bugs only
4. Gather user feedback

### Release Criteria
- [x] All tests passing (1818 nextest + 118 doc tests)
- [x] No critical bugs
- [x] Documentation complete
- [x] Performance benchmarks met
- [x] Ready for publication to crates.io

## Community

### Support Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Discord: Real-time chat and support
- Stack Overflow: Tagged questions

### Resources
- [User Guide](https://github.com/cool-japan/pandrs/wiki)
- [API Documentation](https://docs.rs/pandrs)
- [Examples](./examples/)
- [Benchmarks](./benches/)

---

Last Updated: 2026-04-19
Maintainer: COOLJAPAN OU (Team Kitasan)
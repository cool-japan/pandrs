# PandRS Project Status & Roadmap

## Current Release

**Version:** 0.4.2
**Status:** In development — `0.4.2` is the current branch, freshly cut from the completed `0.4.1` release (2026-08-24) with no changes yet. All quality gates (tests, clippy, docs, dry-run publish) were last verified against 0.4.1's shipped state — see Testing & Quality below.
**Last updated:** 2026-08-24
**Test Coverage:** 2817 tests passing (`cargo nextest run --features all-safe`), 110 doc tests passing + 19 ignored (`cargo test --doc --features all-safe`). `cargo clippy --features all-safe -- -D warnings` clean (`clippy::correctness`/`suspicious`/`perf` enforced; `style`/`complexity` allowed — see CONTRIBUTING.md).

---

> **2026-08-19 update:** the audit below (2026-06-20) and its "✅ RESOLVED &
> verified" section were the *first* pass. A second, independent, more
> exhaustive audit — the "Productionization" effort, run in four waves —
> found and fixed a further round of correctness bugs and honesty gaps
> across the whole crate; the June audit's "all resolved" framing undersold
> what was still broken. The itemized Wave 1-4 fix list lives in
> `CHANGELOG.md`'s `[0.4.1]` entry (the 2026-08-19 addendum). The
> "Known Limitations / Future Work" section near the end of *this* file is
> the current, accurate list of what remains open — treat it, not the audit
> prose below, as the source of truth for open gaps.

## ⚠️ Code Honesty Audit (2026-06-20) — Reddit-critique verification & remediation backlog

A public critique alleged COOLJAPAN crates ship (1) "name-only" GPU backends, (2) stubs / fabricated-result functions, and (3) non-idiomatic, allocation-heavy code. A 7-track read-only audit of PandRS confirms the critique is **substantially TRUE in specific subsystems** and **false/overstated in others**. Verdict and backlog below; every item carries a verified `file:line`. NOTE: several items marked "✓ Completed" later in this file (GPU/CUDA, Parquet compression, Arrow integration) are in fact **stubs or fabrications** — see below.

### Verdict per pillar
- **GPU 看板倒れ — PARTLY TRUE.** Default build honestly ships *no* GPU API (cfg'd out), and there is **no** Metal/ROCm/Vulkan/OpenCL/TPU overclaim, and it is **f64** (not the alleged Float32). BUT the `cuda`-gated code launches **zero** real kernels/cuBLAS calls; every device path errors-after-alloc or silently runs CPU, and several `gpu_*` APIs return fabricated values (`gpu_sum`→`0.0`, `gpu_corr`→input clone, multivariate LR→`0.1*i`) plus a hardcoded "2.5× speedup" metric.
- **Stubs / fabricated results — TRUE (the serious one).** `todo!()`/`unimplemented!()` = 0 (clean), but ~193 "simplified" / ~127 NotImplemented / ~112 "placeholder" markers hide many functions that **silently return wrong data**: a broken `chi2_sf` (1-char sign bug → every χ² test "never significant"), `to_arrow()` fabricating data from row indices, `from_csv` returning hardcoded Alice/Bob/Charlie, joins that infinite-recurse, fake LZ4/ZSTD that *grows* data, ARIMA with hardcoded coefficients, ML serving returning `42.0`.
- **Non-idiomatic / allocation — PARTLY TRUE.** Real & severe: **O(N²·C)** quadratic allocation in `apply`/`drop_duplicates` (column re-materialized per row); core `Series`/`Column` miss `Index`/`IntoIterator`/`FromIterator`/`Extend`/`Display`. But the `[T;N]` charge is **misguided** (columnar data is correctly heap-backed), and the "panic-happy `.expect()`" charge is **false** (≈2755/2997 expects are test-only; 38 `unwrap()` total — the crate is `Result`-disciplined).

### ✅ RESOLVED & verified (2026-06-20) — integrated on branch 0.4.1; **1818 default / 1853 scirs2 tests pass; clippy clean; cuda compiles**

All six remediation clusters are implemented (real algorithms preferred; honest `Error::NotImplemented` only where a real impl needs external infra) and merged. Commits: `ca026df` stats, `da2072f` time_series, `83f63f7` dataframe, `abde6aa` ml, `84a5f32` gpu, `94761eb` storage.

- **Stats (Tier 0/1).** New single-source-of-truth `src/stats/special.rs` (Lanczos lnΓ; NR `gser`/`gcf` incomplete-gamma, `betacf` incomplete-beta; exact normal/χ²/t/F CDF·SF·quantile, unit-tested). Fixed the `chi2_sf` **sign bug** in BOTH `hypothesis.rs` and `time_series/stats.rs`; routed all T/χ²/F CDFs+quantiles, `inference` p-value fns, and OLS p-values (now Student-t) through it. Kills the >1 probability, the constant-1.0 F-quantile, and the always-significant χ².
- **time_series.** Real ARIMA (Levinson-Durbin via SarimaForecaster), real ADF OLS regression, real Durbin-Levinson PACF, Savitzky-Golay + natural cubic spline + seasonal fill + aggregation-aware resample, real isolation-forest outlier detection, Higuchi fractal dim + LZ76. Real p-values (incl. Royston Shapiro-Wilk). Honest `NotImplemented` for Kalman/Hodrick-Prescott/LOWESS/STL/X-13 (no more silent substitution).
- **core DataFrame/Arrow/IO.** Real `to_arrow` (from actual data + real type inference), real CSV/JSON/Parquet (de)serialization, real hash joins, real `.xs()`/`.select()`/`agg()`, real `is_numeric_column`/`is_categorical`/`filter`/`set_index`/`add_row`, `get_categorical` no longer `transmute`s, real parquet rename + predicate filter.
- **ML.** Real model-serving inference from stored weights (was `42.0`/`class_a`), real pipeline + backward_compat scalers/encoders/imputers (was `df.clone()`/`std=10%·mean`), real `SelectKBest::FRegression` F-stat, GridSearch/RandomizedSearch refit `best_estimator_`.
- **GPU.** Silent no-ops (`gpu_sum`→0.0, `gpu_corr`→clone, decomps→identity, LR→`0.1·i`) → honest `NotImplemented` or **real** (Jacobi eigen/SVD, normal-equation LR). Removed fabricated 2.5× speedup + benchmark "speedup"; real cudarc-0.19 device queries. (Verified on a real CUDA toolkit/GPU present in the env: 17 cuda tests pass.)
- **storage.** Real Pure-Rust LZ4 + ZSTD via **COOLJAPAN OxiARC** (`oxiarc_lz4`/`oxiarc_zstd`) — was copying data verbatim (it GREW), with fabricated 2.5/multiplier ratios. Ratios now measured from real bytes; round-trip+shrink test added.
- **Allocation (Tier 2 headline).** `apply.rs` `apply(Row)`/`duplicated`/`drop_duplicates` materialize each column once (O(N²·C)→O(N·C)).

**Still open (deferred, lower priority):** idiom polish (`column_names()`→`&[String]`; add `Index`/`IntoIterator`/`FromIterator`/`Extend`/`Display` to `Series`/`Column`); demote blanket `[workspace.lints.rust] allow`; `column_store.rs` `BitPacked` raw-byte no-op (true bit-packing N/A for opaque bytes).
*(Done: `time_series/stats.rs` split into `stats.rs` 1660 + `stats_normality.rs` 369 + `stats_tests.rs` 111, all under the 2000-line guideline — commit `ae9c531`.)*

---
*The tiered findings below are the original audit backlog, retained for reference / traceability.*

### TIER 0 — CRITICAL: silently returns WRONG data to users (corrupts analysis)
- [x] `stats/hypothesis.rs:841` `chi2_sf` continued-fraction **sign bug** `an = ia*(ia-a)` → returns ≈1.0 for almost all inputs; **every** χ² p-value (Ljung-Box, Friedman, KW, independence, GoF) is wrong. *(1-char fix; headline v0.4.1 claim is currently false.)*
- [x] `stats/inference/mod.rs:113` `chi2_to_pvalue` → returns ≈0 always (public `chi_square_test` always "significant"). Route through fixed `chi2_sf`.
- [x] `arrow_integration.rs:133` `series_to_arrow_array` + `:113` `infer_arrow_type` — `to_arrow()` **fabricates** column data from row index / infers type from column *name*. All Arrow/Flight/batch output is fake.
- [x] `dataframe/serialize.rs:41` `from_csv` ignores path, returns hardcoded `Alice/Bob/Charlie`.
- [x] `dataframe/join.rs:76,88,105` `inner/left/outer_join` — `#[allow(unconditional_recursion)]`, return empty / stack-overflow. `join`+`right_join` dispatch here.
- [x] `storage/unified_column_store.rs:73,115` fake `Lz4`/`Zstd` engines copy data verbatim (+len byte → **grows**); `Zstd` is the **default** codec (`:34`). Same in `adaptive_string_pool.rs:898,914`. Fabricated ratios (`:925`→`2.5`, `:879` multipliers).
- [x] `dataframe/multi_index_cross_section.rs:299` `select_rows`→`.xs()/.select()/.sort_index()` return clone of ALL rows; `:430` `extract_column_values`→`agg()` aggregates all-zeros.
- [x] `dataframe/base.rs:1258` `is_numeric_column`→`false` always; `:946` `set_index`/`set_multi_index` no-op; `:1136` `is_categorical`→true for every column; `:1051` `filter`→empty; `:1230` `get_categorical<T>` unsound `transmute`.
- [x] `time_series/forecasting.rs:850` `ArimaForecaster::fit` hardcodes `ar=0.5, ma=0.3` (never estimates). Delegate to real `SarimaForecaster` (Levinson-Durbin already exists).
- [x] `ml/serving/serialization.rs:286` `perform_prediction` ignores loaded weights, returns `42.0` / `class_a:0.7`.
- [x] `ml/backward_compat/preprocessing.rs:60` StandardScaler std = `mean*0.1`; OneHot/Polynomial/Binner/Imputer/FeatureSelector (`:313,352,388,437,486`) return input unchanged.
- [x] `io/parquet.rs:1455` schema-evolution rename writes literal `"renamed_from_X"` as cell values; `:1515` `apply_predicate_filters` returns unfiltered data.

### TIER 1 — HIGH: wrong numbers / fabricated stats (less central API)
- [x] `stats/distributions.rs:475` `FDistribution::inverse_cdf`→constant `1.0`; `:224` `TDistribution::cdf` can return >1 (impossible prob); `:251` t-quantile 36% low; `:201` 1-term Stirling `ln_gamma` corrupts all t/χ²/F PDFs. Replace with Lanczos log-gamma + real incomplete beta/gamma (Lentz CF).
- [x] `stats/regression/mod.rs:164` OLS p-values use `normal_cdf` not t-distribution (anti-conservative).
- [x] `stats/hypothesis.rs:719` Shapiro-Wilk W coefficients fabricated (`0.7071`, `0.5/(i+1)`).
  - **Goal:** Replace fabricated W coefficients (0.7071 / 0.5/(i+1)) with real Royston (1992) algorithm; share code with time_series/stats_normality.rs where the real impl already exists.
  - **Files:** src/stats/hypothesis.rs; possibly src/time_series/stats_normality.rs
  - **Tests:** known-W fixture, normal-data non-rejection, skewed-data rejection
- [x] `time_series/analysis.rs:614,727` ADF builds regression design matrix then **discards** it, returns one-sample t-test of differenced mean; `:904` PACF Durbin-Levinson `denominator=1.0` hardcoded (wrong for lag≥2); `:960` second Ljung-Box still hardcoded ladder.
- [x] `time_series` silent algorithm substitution (return error or implement): `preprocessing.rs` Spline→linear(`:704`), Kalman→exp-smooth(`:1150`), HP→MA(`:1155`), Savitzky-Golay→MA(`:1135`), Lowess→MA(`:1145`), SeasonalFill→ffill(`:766`), `resample` ignores aggregation(`:609`); `decomposition.rs:248,254` STL/X13→classical; `preprocessing.rs:884` IsolationForest→`vec![]` (always "no outliers").
- [x] `time_series/forecasting.rs:217` SMA residual-std hardcoded `1.0` → fabricated confidence intervals.
- [x] ~10 hardcoded p-value ladders in `time_series/stats.rs` (`:970,787,860,1395,1432,1509,1709,1780`) + `analysis.rs:749,971`. `stats.rs:94 normal_sf` is correct but **dead code** — wire it into Runs/Variance-Ratio.
- [x] `ml/pipeline/mod.rs:88,131` PipelineStage transformers all `df.clone()`; StandardScaler `fit` reads numeric cols as `Series<String>` (errors on real data).
- [x] GPU honesty: `dataframe/gpu/mod.rs:34`, `series/gpu/mod.rs:30`, `gpu/cuda.rs:937(to_cpu→zeros),693(sort→unsorted)`, `gpu/advanced_ops.rs:107(QR/SVD/eigen/inverse→identity)`, `stats/gpu.rs:461(LR→0.1*i)`, `dataframe/gpu_window.rs:263(fake 2.5×)`. Convert silent no-ops to `Error::NotImplemented`; delete fabricated metrics; fix stale "cudarc 0.18.x" strings; drop unused `half::f16`.
- [x] `gpu/cuda.rs:206,368,452,536,621,867` non-CUDA fallback matmul/elementwise/reduce paths return `Array2::zeros(...)` — silent wrong data. Implement real CPU computation or honest Error::NotImplemented.
  - **Files:** src/gpu/cuda.rs
  - **Tests:** CPU-fallback matmul unit test (cuda-gated)

### TIER 2 — MEDIUM: performance / idiom (matches "allocation-heavy" critique)
- [x] `dataframe/apply.rs:79,222,298,353` call `get_column_string_values` (`base.rs:576`, full-column `Vec<String>` copy) **inside the row loop** → **O(N²·C)**. Hoist column extraction out of the loop.
- [x] `dataframe/base.rs:366` `column_names()->Vec<String>` clones at 20+ loop sites → return `&[String]`.
  - **Goal:** Return &[String] instead of cloned Vec<String>; fix ~20 call sites
  - **Files:** src/dataframe/base.rs and call sites
- [x] Whole-object clones for 1-column ops: `dataframe/window.rs:71`, `categorical.rs:148`, `series/base.rs:167,177,187` accessors clone entire Series.
  - **Goal:** Cut whole-object clones in 1-column ops/accessors
  - **Files:** src/dataframe/window.rs, src/dataframe/categorical.rs, src/series/base.rs
- [x] Add idiomatic traits to `series/base.rs` `Series<T>` & `core/column.rs` `Column`: `Index<usize>`, `IntoIterator for &`, `FromIterator`, `Extend`, `Display`.
  - **Goal:** Add Index<usize>, IntoIterator for &, FromIterator, Extend, Display to Series<T> and Column
  - **Files:** src/series/base.rs, src/core/column.rs

### TIER 3 — policy / hygiene
- [x] `Cargo.toml:132` `[workspace.lints.rust]` blanket `allow(dead_code, unused_variables, unused_imports)` masks real stubs (e.g. `core/column.rs:80` `Column::from_any` ignores input). Demote to `warn`, fix churn properly.
  - **Goal:** Demote dead_code/unused_variables/unused_imports from allow→warn; fix root causes if ≤30 sites, else revert+flag. Also remove verified-orphaned src/ml/backward_compat/ sibling files.
  - **Files:** Cargo.toml; src/ml/backward_compat/ (orphaned files removal)
- [x] `Cargo.toml:24,~93` root lists `rand`/`ndarray` as direct deps (manifest drift; `src` uses `scirs2_core::*`). One stray `ml/backward_compat/dimension_reduction.rs:669 rand::rngs::StdRng`. Remove/scrub. *(orphaned file; scirs2_core re-exports used in compiled code; no live drift)*
- [x] `distributed/ballista/cluster.rs:11` references undeclared `ballista_client` crate (latent build break under `distributed`); `dataframe.rs:585` silently selects dead Ballista engine.
  - **Goal:** Remove ballista_client field (undeclared crate → build break under --features distributed); return Error::NotImplemented for non-datafusion engine selection
  - **Files:** src/distributed/ballista/cluster.rs, src/distributed/dataframe.rs
- [x] `connectors/local.rs:240` path-traversal guard computes `canonical_base` then never enforces it (dead security check).
  - **Goal:** Enforce canonical_base containment check in resolve(); reject ../escape paths
  - **Files:** src/connectors/local.rs
  - **Tests:** in-base resolves OK; ../escape rejected (use temp_dir() for base)
- [x] `core/column.rs:78` Column::from_any ignores its input and always returns an empty Int64Column — every clone_column() loses all data and mis-types the column.
  - **Goal:** downcast::<Column>() (common path) with defensive fallbacks for bare column structs; fix the silent data-loss bug
  - **Files:** src/core/column.rs
  - **Tests:** clone_column() round-trips data and type for each of the 4 column kinds

### CONFIRMED-HONEST (no action — critique does NOT apply)
- ML v0.4.1 "real algorithm" claims **all verified true**: PCA(Jacobi), t-SNE(real GD), DBSCAN(BFS), Agglomerative(4 linkages), IsolationForest(real trees), LOF, OLS, IRLS LogisticRegression, RocAuc(Mann-Whitney), chi2/MI scores, RobustScaler/QuantileTransformer/PowerTransformer, GridSearch/RandomizedSearch CV.
- `stats.rs` χ² tests (Ljung-Box/Box-Pierce/Breusch-Godfrey/Friedman/KW/JarqueBera) use the real `chi2_sf` (correct *once the sign bug above is fixed*); descriptive stats, OLS coefficients/R², rank statistics, Binomial/Poisson, `normal_sf`, Acklam normal-quantile — all real.
- `SarimaForecaster` (real Levinson-Durbin + AIC AutoArima), ExponentialSmoothing, Mann-Kendall, entropy/Hurst/DFA features — real.
- `schema_evolution/migrator.rs` — real (deliberately avoids the broken `is_numeric_column` stub).
- Cloud connectors (`connectors/cloud.rs`) — honest real `object_store` or clean `NotImplemented`; Ballista — honest `NotImplemented`. `.expect()` is test-only; crate is `Result`-disciplined.

---

### Acknowledged upstream Pure Rust tech debt (feature-gated; default build unaffected)
- `parquet` / `distributed` / `flight` pull `flate2`/`zstd`/`lz4_flex`/`snap`/`bzip2`/`brotli`/`miniz_oxide` via upstream arrow/parquet/datafusion/async-compression.
- `cloud-storage` pulls `ring` (C+asm) via `object_store 0.14.1` (bumped from 0.13.2 in Wave 3, closing RUSTSEC-2026-0194/0195) — upstream uses `ring::{hmac,digest,signature,rand}` directly and exposes `ring::error` in public API. Separately, `object_store` 0.14's `reqwest` now builds with `rustls-no-provider`; this crate installs `rustls::crypto::ring::default_provider()` once before the first cloud client is built (`src/connectors/cloud.rs`) — see the Wave 4 entry in CHANGELOG.md.

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

## Known Limitations / Future Work

Current, accurate list of open gaps (last verified 2026-08-19, Wave 4 of the
productionization effort). Every item below is sourced to a specific file or
a specific verification command — nothing here is a guess.

### API surface

- **Categorical public API gap:** `CategoricalExt` trait and
  `astype_categorical` are not public; the underlying operations
  (`is_categorical`/`get_categorical`/`from_categoricals`/`value_counts`/
  `add_categories`/`remove_categories`/`add_na_series_as_categorical`) exist
  only as inherent `DataFrame` methods. Wiring a public trait was attempted
  and reverted in Wave 3 (`E0592` method-resolution collisions plus a
  companion-column convention mismatch) — this is future feature work, not
  a regression.
- **`sample` shadow:** the inherent `DataFrame::sample(&[usize])`
  (positional take) shadows `PandasCompatExt::sample(n, replace)`. Renaming
  either is a breaking change and needs a project decision before 0.5.0.
- **`E0034` trait-import ambiguity:** importing `PandasCompatExt` and
  `AdvancedIndexingExt` together makes `iloc`/`at`/`head` ambiguous at the
  call site (both traits provide a same-named method on `DataFrame`). This
  is inherent to Rust's trait method resolution, not fixable by renaming the
  traits themselves — the `prelude` module itself avoids the collision by
  not importing both at once. Workaround: import only one of the two
  traits, or disambiguate with `Trait::method(&df, ..)`.

### Correctness — ✅ RESOLVED (2026-08-24)

- **`hierarchical_groupby.rs` carried the same NaN-propagation bug the base
  `groupby_single` path was fixed for in Wave 4 — now fixed.**
  `calculate_hierarchical_aggregation`, `cross_level_agg`, and
  `nested_transform` used to parse-then-accumulate with no `is_nan()`
  filter, so a `NaN` cell propagated through the whole aggregate instead of
  being excluded (pandas `skipna=True`), the same way the base path used to
  before its Wave 4 fix. A second, related divergence found alongside it —
  `.parse().ok()` silently dropping non-numeric cells instead of erroring
  loudly, unlike the base path it otherwise mirrors — is fixed too. All
  three functions now route through a new shared helper,
  `parse_hierarchical_numeric_pairs`, which excludes `NaN` per pandas
  `skipna=True` and errors loudly on genuinely non-numeric cells instead of
  silently dropping them. Proven by 9 passing regression tests in
  `tests/hierarchical_groupby_nan_w4_regression_test.rs`.
  (`src/dataframe/pandas_compat/groupby.rs` was checked separately and was
  already correct — it filters `!v.is_nan()` in every reduction; no action
  was needed there.)

### Performance / SIMD

- `simd_stats` (variance/std/covariance/correlation/skewness/kurtosis/
  dot-product/L2-norm/weighted-mean, `optimized::jit::simd_stats`) and
  `simd_string` (`optimized::jit::simd_string`) are real, tested, public
  API — but **not called from the main `DataFrame`/`Series` statistics or
  string-accessor paths** (see the "Wiring status" doc comment at the top
  of `src/optimized/jit/simd_stats.rs`). The element-wise
  `SIMDFloat64Ops`/`SIMDInt64Ops` column traits
  (`src/column/simd_operations.rs`) are in the same state: implemented,
  public, and exported from `pandrs::column`, but zero call sites anywhere
  in normal DataFrame/column arithmetic. Only the `sum` reduction
  (`OptimizedDataFrame::sum_simd`, in `src/optimized/direct_aggregations.rs`)
  is wired into a main-path method today; `mean_simd`/`min_simd`/`max_simd`
  on that same fast path are scalar by design (see below), not "unwired."
  Wiring `simd_variance_f64` into `optimized::split_dataframe::group::operations`'s
  `var`/`std` would close this gap for the highest-value case; it is a
  real, scoped perf task, not started.
- Within `src/optimized/jit/simd.rs` (the aggregation kernels
  `direct_aggregations.rs` calls), only `sum_f64`/`sum_i64` have a
  hand-written AVX2 kernel. `mean`/`min`/`max` there are scalar on every
  target *by design*: their pandas `skipna=True` contract (skip `NaN`, keep
  `±inf`, preserve the sign of zero) does not line up with the x86
  `MINPD`/`MAXPD` tie-break rules, so a vector kernel could not be made
  bit-identical without more work than it would save.
- **AVX2 kernels have never executed on any host available during
  development** — no AVX2 hardware; SSE2+scalar was exercised under
  Rosetta, scalar on aarch64. Bit-identity with the scalar reference rests
  on a construction argument in the source (see the "Bit-identity contract"
  doc comments), not an on-hardware measurement. Needs an AVX2 CI/dev host
  to actually verify.
- No NEON, no AVX-512 anywhere in the crate. Scalar (still
  auto-vectorizable by the compiler) is the fallback on every non-`x86_64`
  target, and on `x86_64` without AVX2.

### GPU / JIT

- `gpu_window.rs` (`gpu_rolling(...).mean()` etc., behind
  `cuda_available`): **fixed in Wave 4** — now returns real `f64`-typed
  columns (was `String`-typed) and honors a caller-supplied `ddof`. Its
  `#[cfg(cuda_available)]` real-GPU dispatch branch still has no kernel for
  `.min()`/`.max()` (honest `Err`, not a fabricated result).
- As of the two most-recent "CUDA" commits, `dataframe::gpu_window` (and
  its `crate::gpu` dependency) are gated behind `cuda_available` even under
  `--features all-safe` — `cargo nextest run --features all-safe
  gpu_window` currently matches **zero** tests. Earlier documentation
  described this as "the all-safe (CPU-fallback) path"; that framing is no
  longer accurate as written. Needs an explicit decision: correct the
  module's framing to "cuda-only", or loosen the `#[cfg]` gate so the CPU
  fallback is reachable without the CUDA toolkit.
- `jit_core.rs`: `jit()` performs no real Cranelift compilation. This is an
  intentional, documented fallback (not a fabrication — it says so in its
  own doc comment), but it is also not the JIT the name implies.
  Implementing real Cranelift codegen here is a large, separate decision,
  out of scope for Waves 1-4.
- **CUDA path is unverifiable on macOS** — `build.rs` compiles it out
  entirely on this platform, and there is no CUDA-capable host in the
  current dev environment. Needs a CUDA-capable Linux/Windows host to
  verify.

### Compatibility

- Minor floating-point precision differences vs. pandas remain expected
  (different reduction order / underlying library implementations).
- Excel formula preservation is not supported — read/write is values-only.

### Process / CI

- Repository policy currently permits only `pypi-publish.yml` and
  `npm-publish.yml` under `.github/workflows/` — there is **no** CI
  workflow that runs build/test/clippy/fmt on pull requests. Contributors
  must run those checks locally before requesting review (see
  CONTRIBUTING.md). This is an accepted policy constraint, not a bug, and
  not expected to change without an explicit policy decision.

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
- 2817 tests passing (`cargo nextest run --features all-safe`) + 110 doc tests passing (19 ignored)
- Property-based / regression testing for every Wave 1-4 fix (see `tests/*_w{1,2,3,4}_regression_test.rs`)
- **No CI on pull requests** — repo policy permits only the release-publish workflows (`pypi-publish.yml`/`npm-publish.yml`); see "Process / CI" under Known Limitations above
- Benchmark suite available (`cargo bench`, see BENCHMARKING.md); not run on a schedule (no CI)

### Quality Metrics
- Zero compiler warnings policy ✓
- Clippy `correctness` (deny) / `suspicious` / `perf` (warn) enforced on the library, `style`/`complexity` intentionally allowed — see CONTRIBUTING.md ✓
- Rustfmt formatting required ✓
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` clean on both `all-safe` and the docs.rs feature set ✓

## Release Process

### Stable Release (Current)
1. Feature freeze - no new features
2. Focus on stability and performance
3. Address critical bugs only
4. Gather user feedback

### Release Criteria
- [x] All tests passing (2817 nextest + 110 doc tests, 19 ignored)
- [x] No known critical bugs (see Known Limitations above for the open, non-critical gaps)
- [x] Documentation builds clean (`cargo doc -D warnings`, both feature sets)
- [x] `cargo publish --dry-run --allow-dirty --features all-safe` packaging verified (2026-08-19: 529 files, 9.0 MiB / 1.8 MiB compressed)
- [ ] Actually published to crates.io — publishing is a separate release-management decision, not yet made

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

Last Updated: 2026-08-24
Maintainer: COOLJAPAN OU (Team Kitasan)

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [x] `pandrs-py_bindings`: `py_bindings/src/py_gpu.rs:298` — implement GPU acceleration in `PyOptimizedDataFrame::gpu_accelerate` (currently no-op clone)
  - Priority: P2 | Scope: medium | Hint: none
  - **Goal:** Replace fabricated-zero outputs with real CPU computation via non-CUDA-gated pandrs APIs
  - **Files:** py_bindings/src/py_gpu.rs
  - **Tests:** corr diagonal=1 & symmetric; LR recovers known line; kmeans inertia≥0 & labels in range
- [x] `pandrs-py_bindings`: `py_bindings/src/py_gpu.rs:317` — implement GPU correlation matrix computation
  - Priority: P2 | Scope: medium | Hint: none
  - **Goal:** Replace fabricated-zero outputs with real CPU computation via non-CUDA-gated pandrs APIs
  - **Files:** py_bindings/src/py_gpu.rs
  - **Tests:** corr diagonal=1 & symmetric; LR recovers known line; kmeans inertia≥0 & labels in range
- [x] `pandrs-py_bindings`: `py_bindings/src/py_gpu.rs:347` — implement GPU PCA
  - Priority: P2 | Scope: medium | Hint: none
  - **Goal:** Replace fabricated-zero outputs with real CPU computation via non-CUDA-gated pandrs APIs
  - **Files:** py_bindings/src/py_gpu.rs
  - **Tests:** corr diagonal=1 & symmetric; LR recovers known line; kmeans inertia≥0 & labels in range
- [x] `pandrs-py_bindings`: `py_bindings/src/py_gpu.rs:388` — implement GPU k-means clustering
  - Priority: P2 | Scope: medium | Hint: none
  - **Goal:** Replace fabricated-zero outputs with real CPU computation via non-CUDA-gated pandrs APIs
  - **Files:** py_bindings/src/py_gpu.rs
  - **Tests:** corr diagonal=1 & symmetric; LR recovers known line; kmeans inertia≥0 & labels in range
- [x] `pandrs-py_bindings`: `py_bindings/src/py_gpu.rs:418` — implement GPU linear regression
  - Priority: P2 | Scope: medium | Hint: none
  - **Goal:** Replace fabricated-zero outputs with real CPU computation via non-CUDA-gated pandrs APIs
  - **Files:** py_bindings/src/py_gpu.rs
  - **Tests:** corr diagonal=1 & symmetric; LR recovers known line; kmeans inertia≥0 & labels in range
- [x] `pandrs`: `src/vis/direct/mod.rs:429` — implement `OptimizedDataFrame → DataFrame` conversion to enable direct plotting on OptimizedDataFrame
  - Priority: P2 | Scope: small | Hint: none
  - **Goal:** Use standard_dataframe() converter then delegate to DataFrame plotting impls
  - **Files:** src/vis/direct/mod.rs
- [x] `pandrs`: `src/distributed/schema_validator/validation.rs:345` — implement window function validation when window module is enabled
  - Priority: P2 | Scope: small | Hint: none
  - **Goal:** Validate referenced columns exist & have compatible types for window functions
  - **Files:** src/distributed/schema_validator/validation.rs
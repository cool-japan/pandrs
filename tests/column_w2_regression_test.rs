//! Regression tests for `src/column/**` (wave 2).
//!
//! Pins down behaviour that was previously wrong:
//!
//! * `Float64Column::{sum,mean}` now skip `NaN` the same way `min`/`max`
//!   already did (pandas `skipna=True`: `Series([1, NaN, 3]).mean() == 2.0`),
//!   while `min`/`max` keep skipping `NaN` only -- they no longer also
//!   exclude `+-infinity` the way the old `is_finite()` filter did
//!   (`max([1, inf, 2]) == inf`).
//! * `StringPool::from_strings`/`from_strings_legacy` build a LOCAL, dense
//!   id space per column instead of indexing a per-column `Vec` by the
//!   ever-growing GLOBAL pool id -- which used to force that `Vec` to be
//!   padded out to the global id's value (an unbounded blow-up) and
//!   corrupted `len()`/`all_strings()`/`merge()`.
//! * `GlobalStringPool::{get_or_insert,get,len}` propagate a
//!   `Result`/`Error::LockPoisoned` instead of silently returning `0`/`None`
//!   (which used to alias whichever string happened to be interned first).
//! * `StringColumn::new_legacy` no longer substitutes the pool's first
//!   string via `find(..).unwrap_or(0)`.
//! * `SimpleZeroCopyStringColumn`'s derived-column operations
//!   (`to_lowercase_optimized`/`to_uppercase_optimized`/`concat_with`/
//!   `substring_views`/`clone_column`) now carry the source NULL mask
//!   through instead of hardcoding `null_mask: None` (which turned NULLs
//!   into `""`).
//! * The process-wide default string-column optimization mode is read and
//!   written through an atomic accessor instead of an unsafe `static mut`.
//! * `simd_compare_scalar` compares directly against the broadcast scalar
//!   (no `vec![scalar; len]` allocation) with unchanged results.

use pandrs::column::string_pool::{StringPool, GLOBAL_STRING_POOL};
use pandrs::column::{
    default_optimization_mode, set_default_optimization_mode, ColumnTrait, SIMDFloat64Ops,
    SIMDInt64Ops, SimpleZeroCopyStringColumn, StringColumnOptimizationMode,
};
use pandrs::optimized::jit::simd_column_ops::ComparisonOp;
use pandrs::{Column, Float64Column, Int64Column, StringColumn};

/// Tests report failures through a boxed error so that the large
/// `pandrs::Error` never has to travel in a `Result` (which clippy
/// rightfully flags).
type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------
// NaN / infinity aggregation semantics (Float64Column)
// ---------------------------------------------------------------------

#[test]
fn float64_column_mean_and_sum_skip_nan_like_pandas() {
    let col = Float64Column::new(vec![1.0, f64::NAN, 3.0]);
    assert_eq!(
        col.mean(),
        Some(2.0),
        "pandas: Series([1, NaN, 3]).mean() == 2.0"
    );
    assert_eq!(col.sum(), 4.0, "sum must also skip NaN, not propagate it");
}

#[test]
fn float64_column_min_max_do_not_exclude_infinity() {
    let col = Float64Column::new(vec![1.0, f64::INFINITY, 2.0]);
    assert_eq!(
        col.max(),
        Some(f64::INFINITY),
        "pandas: max([1, inf, 2]) == inf"
    );
    assert_eq!(col.min(), Some(1.0));

    let col = Float64Column::new(vec![1.0, f64::NEG_INFINITY, 2.0]);
    assert_eq!(col.min(), Some(f64::NEG_INFINITY));
    assert_eq!(col.max(), Some(2.0));
}

#[test]
fn float64_column_all_nan_reports_empty_not_nan() {
    let col = Float64Column::new(vec![f64::NAN, f64::NAN]);
    assert_eq!(col.sum(), 0.0, "sum of zero real observations is 0.0");
    assert_eq!(col.mean(), None, "no honest average of zero observations");
    assert_eq!(col.min(), None);
    assert_eq!(col.max(), None);
}

#[test]
fn float64_column_large_column_aggregates_match_naive_reference() {
    // Exercise the 4-way unrolled fast path (F10) at a length that is not
    // a multiple of the unroll width (1003 % 4 == 3), with NaNs scattered
    // through both full unrolled chunks and the scalar remainder.
    let data: Vec<f64> = (0..1003)
        .map(|i| if i % 17 == 0 { f64::NAN } else { i as f64 })
        .collect();
    let col = Float64Column::new(data.clone());

    let naive_sum: f64 = data.iter().copied().filter(|v| !v.is_nan()).sum();
    let naive_count = data.iter().filter(|v| !v.is_nan()).count();
    let naive_min = data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::INFINITY, f64::min);
    let naive_max = data
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);

    assert_eq!(col.sum(), naive_sum);
    assert_eq!(col.mean(), Some(naive_sum / naive_count as f64));
    assert_eq!(col.min(), Some(naive_min));
    assert_eq!(col.max(), Some(naive_max));
}

#[test]
fn float64_column_null_mask_path_also_skips_nan_and_keeps_infinity() {
    let data = vec![1.0, f64::NAN, f64::INFINITY, 5.0, 2.0];
    let nulls = vec![false, false, false, true, false]; // index 3 (5.0) is NULL
    let col = Float64Column::with_nulls(data, nulls);

    // Surviving (non-NULL, non-NaN) values: 1.0, inf, 2.0.
    assert_eq!(col.sum(), f64::INFINITY);
    assert_eq!(col.max(), Some(f64::INFINITY));
    assert_eq!(col.min(), Some(1.0));
}

// ---------------------------------------------------------------------
// String pool: local id space, no blow-up from the global pool's ids
// ---------------------------------------------------------------------

#[test]
fn string_pool_from_strings_stays_dense_after_many_unrelated_columns() {
    // Inflate the *global* pool's id counter well past a small column's
    // own string count, the way many unrelated StringColumns built earlier
    // in the same process would. Before the fix, a fresh column's LOCAL
    // pool indexed itself by these global ids directly, so it padded its
    // own Vec out to the highest global id it was handed -- for 200
    // columns of 3 strings each that means the local pool for a *later*
    // 3-string column would balloon to hundreds of slots instead of 3.
    for i in 0..200 {
        let _ = StringColumn::new_with_global_pool(vec![
            format!("unrelated_{i}_a"),
            format!("unrelated_{i}_b"),
            format!("unrelated_{i}_c"),
        ]);
    }

    let strings = vec!["x".to_string(), "y".to_string(), "z".to_string()];
    let (pool, indices) = StringPool::from_strings(&strings).expect("lock not poisoned");

    assert_eq!(
        pool.len(),
        3,
        "a fresh 3-string pool must stay 3 entries regardless of how many \
         other columns/strings already exist in the global pool"
    );
    assert_eq!(pool.all_strings().len(), 3);
    for &idx in &indices {
        assert!(
            (idx as usize) < pool.len(),
            "returned indices must be valid LOCAL indices, not raw global ids \
             (index {idx} is out of bounds for a {}-entry pool)",
            pool.len()
        );
    }
}

#[test]
fn string_column_round_trips_values_after_many_columns() -> TestResult {
    for i in 0..200 {
        let _ = StringColumn::new_with_global_pool(vec![format!("noise_{i}")]);
    }

    let col = StringColumn::new_with_global_pool(vec![
        "alpha".to_string(),
        "beta".to_string(),
        "alpha".to_string(),
    ]);
    assert_eq!(col.get(0)?, Some("alpha"));
    assert_eq!(col.get(1)?, Some("beta"));
    assert_eq!(col.get(2)?, Some("alpha"));
    Ok(())
}

#[test]
fn string_pool_merge_is_correct_after_from_strings() {
    let a = vec!["a".to_string(), "b".to_string()];
    let b = vec!["b".to_string(), "c".to_string()];
    let (pool_a, _) = StringPool::from_strings(&a).expect("lock not poisoned");
    let (pool_b, _) = StringPool::from_strings(&b).expect("lock not poisoned");

    let merged = pool_a.merge(&pool_b);
    assert_eq!(merged.len(), 3, "a, b, c -- deduplicated across both pools");
    let mut all: Vec<&str> = merged.all_strings();
    all.sort_unstable();
    assert_eq!(all, vec!["a", "b", "c"]);
}

#[test]
fn string_column_new_legacy_never_substitutes_the_wrong_string() {
    let data: Vec<String> = (0..64).map(|i| format!("legacy_{i}")).collect();
    let col = StringColumn::new_legacy(data.clone());
    for (i, expected) in data.iter().enumerate() {
        assert_eq!(
            col.get(i).expect("in-bounds"),
            Some(expected.as_str()),
            "row {i} must round-trip its own string, not the pool's first one"
        );
    }
}

#[test]
fn global_string_pool_get_or_insert_is_fallible_and_consistent() -> TestResult {
    let idx1 = GLOBAL_STRING_POOL.get_or_insert("w2_regression_marker_string")?;
    let idx2 = GLOBAL_STRING_POOL.get_or_insert("w2_regression_marker_string")?;
    assert_eq!(
        idx1, idx2,
        "the same string must intern to the same global id"
    );

    let round_tripped = GLOBAL_STRING_POOL.get(idx1)?;
    assert_eq!(
        round_tripped.as_deref(),
        Some("w2_regression_marker_string")
    );
    Ok(())
}

// ---------------------------------------------------------------------
// Default optimization mode: atomic accessor, not `static mut`
// ---------------------------------------------------------------------

#[test]
fn default_optimization_mode_accessor_round_trips() {
    let previous = default_optimization_mode();

    set_default_optimization_mode(StringColumnOptimizationMode::Legacy);
    assert_eq!(
        default_optimization_mode(),
        StringColumnOptimizationMode::Legacy
    );
    let col = StringColumn::new(vec!["a".to_string()]);
    assert_eq!(
        col.optimization_mode(),
        StringColumnOptimizationMode::Legacy
    );

    set_default_optimization_mode(StringColumnOptimizationMode::GlobalPool);
    assert_eq!(
        default_optimization_mode(),
        StringColumnOptimizationMode::GlobalPool
    );

    // Restore, so this test doesn't leak its mode change to whatever else
    // (if anything) shares this test's process.
    set_default_optimization_mode(previous);
}

// ---------------------------------------------------------------------
// SimpleZeroCopyStringColumn: NULL mask preserved through derived columns
// ---------------------------------------------------------------------

#[test]
fn zero_copy_case_conversion_preserves_null_positions() -> TestResult {
    let col = SimpleZeroCopyStringColumn::with_nulls(
        vec!["Aa".to_string(), "Bb".to_string(), "Cc".to_string()],
        vec![false, true, false],
    )?;

    let lower = col.to_lowercase_optimized()?;
    assert_eq!(lower.get(0)?, Some("aa".to_string()));
    assert_eq!(
        lower.get(1)?,
        None,
        "a NULL source row must stay NULL, not become an empty string"
    );
    assert_eq!(lower.get(2)?, Some("cc".to_string()));

    let upper = col.to_uppercase_optimized()?;
    assert_eq!(upper.get(0)?, Some("AA".to_string()));
    assert_eq!(upper.get(1)?, None);
    assert_eq!(upper.get(2)?, Some("CC".to_string()));

    Ok(())
}

#[test]
fn zero_copy_clone_column_preserves_row_count_and_nulls() -> TestResult {
    let col = SimpleZeroCopyStringColumn::with_nulls(
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        vec![false, true, false],
    )?;

    let cloned = col.clone_column();
    assert_eq!(cloned.len(), 3, "clone_column must never change row count");

    match cloned {
        Column::String(string_col) => {
            assert_eq!(string_col.get(0)?, Some("a"));
            assert_eq!(string_col.get(1)?, None, "NULL must survive clone_column");
            assert_eq!(string_col.get(2)?, Some("c"));
        }
        other => return Err(format!("expected Column::String, got {other:?}").into()),
    }

    Ok(())
}

// ---------------------------------------------------------------------
// simd_compare_scalar: no vec![scalar; len] allocation, same results
// ---------------------------------------------------------------------

#[test]
fn simd_compare_scalar_f64_matches_elementwise_semantics() -> TestResult {
    let col = Float64Column::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = col.simd_compare_scalar(3.0, ComparisonOp::GreaterThanEqual)?;
    assert_eq!(result, vec![false, false, true, true, true]);

    let result = col.simd_compare_scalar(3.0, ComparisonOp::Equal)?;
    assert_eq!(result, vec![false, false, true, false, false]);
    Ok(())
}

#[test]
fn simd_compare_scalar_i64_matches_elementwise_semantics() -> TestResult {
    let col = Int64Column::new(vec![1, 2, 3, 4, 5]);
    let result = col.simd_compare_scalar(3, ComparisonOp::LessThan)?;
    assert_eq!(result, vec![true, true, false, false, false]);
    Ok(())
}

#[test]
fn simd_compare_scalar_respects_null_mask() -> TestResult {
    let col = Float64Column::with_nulls(vec![1.0, 2.0, 3.0], vec![false, true, false]);
    // Every value equals itself, but a NULL comparison must read false.
    let result = col.simd_compare_scalar(2.0, ComparisonOp::Equal)?;
    assert_eq!(result, vec![false, false, false]);
    Ok(())
}

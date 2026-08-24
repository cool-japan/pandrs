//! Sampling and random number generation module

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::RngExt;
use scirs2_core::random::SeedableRng;
use scirs2_core::random::SliceRandom;
use std::collections::HashMap;

/// Build a seeded RNG when `seed` is given, otherwise one seeded from the
/// system entropy source — matching the pattern already established at
/// `optimized::split_dataframe::row_ops::sample_rows`, the crate's other
/// seed-optional sampler.
fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed_val) => StdRng::seed_from_u64(seed_val),
        None => {
            let mut seed_bytes = [0u8; 32];
            scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        }
    }
}

/// Internal implementation for sampling from DataFrame.
///
/// `seed`, when given, makes the draw reproducible (`None` draws a fresh
/// system-entropy seed each call). Row reconstruction delegates to
/// [`DataFrame::sample`], which dispatches over each column's concrete
/// element type (`f64`/`i64`/`String`/`bool`/`i32`/`f32`, then a numeric or
/// string fallback for every other supported type) — the previous
/// hand-rolled loop here only ever tried `get_column::<String>`, so on a
/// DataFrame with no `String` columns (the common case for a purely
/// numeric/boolean dataset) *every* column silently failed that downcast
/// and was dropped, and `sample_impl` on such a DataFrame returned an
/// entirely empty (zero-column) result.
pub(crate) fn sample_impl(
    df: &DataFrame,
    fraction: f64,
    replace: bool,
    seed: Option<u64>,
) -> Result<DataFrame> {
    if fraction <= 0.0 {
        return Err(Error::InvalidValue(
            "Sample rate must be a positive value".into(),
        ));
    }

    // Get number of rows in DataFrame
    let n_rows = df.row_count();
    if n_rows == 0 {
        return Ok(DataFrame::new());
    }

    let sample_size = (n_rows as f64 * fraction).ceil() as usize;
    if !replace && sample_size > n_rows {
        return Err(Error::InvalidOperation(
            "For sampling without replacement, sample size must not exceed original data size"
                .into(),
        ));
    }

    let mut rng = seeded_rng(seed);

    // Generate indices
    let indices = if replace {
        // Sampling with replacement
        (0..sample_size)
            .map(|_| rng.random_range(0..n_rows))
            .collect::<Vec<_>>()
    } else {
        // Sampling without replacement
        let mut idx: Vec<usize> = (0..n_rows).collect();
        idx.shuffle(&mut rng);
        idx[0..sample_size].to_vec()
    };

    df.sample(&indices)
}

/// Internal implementation for generating bootstrap samples
pub(crate) fn bootstrap_impl(data: &[f64], n_samples: usize) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Bootstrap requires data".into()));
    }

    if n_samples == 0 {
        return Err(Error::InvalidValue(
            "Number of samples must be positive".into(),
        ));
    }

    let n = data.len();
    let mut rng = scirs2_core::random::rng();
    let mut result = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        // Sampling with replacement
        let sample: Vec<f64> = (0..n).map(|_| data[rng.random_range(0..n)]).collect();

        result.push(sample);
    }

    Ok(result)
}

/// Perform stratified sampling
///
/// Samples from each stratum (group) at the specified rate.
///
/// `seed`, when given, makes the draw reproducible: the *same* seed always
/// produces the *same* sampled indices for a given input and parameters.
/// This requires two things the previous implementation got wrong even
/// though its doc comment already claimed reproducibility: a single RNG
/// seeded once and threaded through every stratum (a fresh
/// `scirs2_core::random::rng()` — always system-entropy-seeded, `seed` or
/// not — was previously created *inside* the per-stratum loop, so even a
/// caller-supplied seed only would have governed the first stratum's draw
/// in a way that didn't even reproduce across runs), and a deterministic
/// stratum iteration order (strata were previously collected into a
/// `HashMap` and iterated in its unspecified, run-to-run-varying order).
///
/// # Arguments
/// * `df` - Input DataFrame
/// * `strata_column` - Column name specifying the strata
/// * `fraction` - Sampling rate from each stratum
/// * `replace` - Whether to sample with replacement
/// * `seed` - RNG seed for reproducible sampling; `None` seeds from system
///   entropy (a fresh draw each call, as before this fix)
pub fn stratified_sample_impl(
    df: &DataFrame,
    strata_column: &str,
    fraction: f64,
    replace: bool,
    seed: Option<u64>,
) -> Result<DataFrame> {
    if !df.contains_column(strata_column) {
        return Err(Error::ColumnNotFound(strata_column.to_string()));
    }

    if fraction <= 0.0 {
        return Err(Error::InvalidValue(
            "Sample rate must be a positive value".into(),
        ));
    }

    let strata_col = match df.get_column::<String>(strata_column) {
        Ok(col) => col,
        Err(_) => return Err(Error::ColumnNotFound(strata_column.to_string())),
    };

    // Collect indices for each stratum.
    let mut strata_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, value) in strata_col.values().iter().enumerate() {
        strata_indices.entry(value.clone()).or_default().push(i);
    }

    // Iterate strata in a deterministic (sorted) order rather than the
    // HashMap's unspecified order, so a given seed always consumes the RNG
    // stream in the same sequence and therefore always produces the same
    // sampled indices.
    let mut sorted_strata: Vec<&String> = strata_indices.keys().collect();
    sorted_strata.sort();

    let mut rng = seeded_rng(seed);

    // Sample from each stratum at the specified rate
    let mut all_sample_indices = Vec::new();
    for stratum in sorted_strata {
        let indices = &strata_indices[stratum];
        let sample_size = (indices.len() as f64 * fraction).ceil() as usize;
        if sample_size == 0 {
            continue;
        }

        if replace {
            // Sampling with replacement
            for _ in 0..sample_size {
                let idx = indices[rng.random_range(0..indices.len())];
                all_sample_indices.push(idx);
            }
        } else {
            // Sampling without replacement
            if sample_size > indices.len() {
                return Err(Error::InvalidOperation(
                    "For sampling without replacement, sample size must not exceed stratum size"
                        .into(),
                ));
            }

            let mut sampled_indices = indices.clone();
            sampled_indices.shuffle(&mut rng);
            all_sample_indices.extend_from_slice(&sampled_indices[0..sample_size]);
        }
    }

    // Sort sampled indices (to maintain original row order)
    all_sample_indices.sort();

    // Reconstruct via `DataFrame::sample`, preserving every column's
    // concrete element type (see `sample_impl`'s doc comment for why the
    // previous `get_column::<String>`-only reconstruction here silently
    // dropped every non-`String` column).
    df.sample(&all_sample_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;

    #[test]
    fn test_simple_sample() {
        let mut df = DataFrame::new();
        // Unsuffixed integer literals default to `i32` — deliberately a
        // *non*-`String` column type, since the previous implementation
        // reconstructed sampled rows via `get_column::<String>` only and
        // silently dropped every column that failed that downcast
        // (i.e. every column in this DataFrame).
        let data = Series::new(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Some("data".to_string()),
        )
        .expect("operation should succeed");
        df.add_column("data".to_string(), data)
            .expect("operation should succeed");

        // 50% sampling (without replacement): the sole `i32` column must
        // survive intact (not silently be dropped), and every sampled
        // value must actually come from the source column.
        let sample = sample_impl(&df, 0.5, false, None).expect("operation should succeed");
        assert_eq!(sample.column_count(), df.column_count());
        assert_eq!(sample.row_count(), 5);
        let sampled_values = sample
            .get_column::<i32>("data")
            .expect("the i32 column must round-trip through sampling")
            .values()
            .to_vec();
        assert_eq!(sampled_values.len(), 5);
        for v in &sampled_values {
            assert!((1..=10).contains(v));
        }

        // 30% sampling (with replacement)
        let sample = sample_impl(&df, 0.3, true, None).expect("operation should succeed");
        assert_eq!(sample.column_count(), df.column_count());

        // 200% sampling (with replacement)
        let sample = sample_impl(&df, 2.0, true, None).expect("operation should succeed");
        assert_eq!(sample.column_count(), df.column_count());

        // 200% sampling (without replacement) - should error
        let result = sample_impl(&df, 2.0, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_sample_is_reproducible_with_seed() {
        let mut df = DataFrame::new();
        let data = Series::new((0..50).collect::<Vec<i64>>(), Some("data".to_string()))
            .expect("operation should succeed");
        df.add_column("data".to_string(), data)
            .expect("operation should succeed");

        let a = sample_impl(&df, 0.5, false, Some(42))
            .expect("operation should succeed")
            .get_column::<i64>("data")
            .expect("column present")
            .values()
            .to_vec();
        let b = sample_impl(&df, 0.5, false, Some(42))
            .expect("operation should succeed")
            .get_column::<i64>("data")
            .expect("column present")
            .values()
            .to_vec();
        assert_eq!(a, b, "the same seed must draw byte-identical samples");

        let c = sample_impl(&df, 0.5, false, Some(43))
            .expect("operation should succeed")
            .get_column::<i64>("data")
            .expect("column present")
            .values()
            .to_vec();
        assert_ne!(
            a, c,
            "a different seed should (overwhelmingly likely) draw a different sample"
        );
    }

    #[test]
    fn test_bootstrap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // 10 bootstrap samples
        let bootstrap_samples = bootstrap_impl(&data, 10).expect("operation should succeed");
        assert_eq!(bootstrap_samples.len(), 10);

        // Each sample is the same length as the original data
        for sample in &bootstrap_samples {
            assert_eq!(sample.len(), data.len());
        }

        // Samples are drawn with replacement from the original data
        for sample in &bootstrap_samples {
            for value in sample {
                assert!(data.contains(value));
            }
        }
    }

    fn stratified_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        let strata = Series::new(
            vec![
                "a".to_string(),
                "a".to_string(),
                "a".to_string(),
                "a".to_string(),
                "b".to_string(),
                "b".to_string(),
                "b".to_string(),
                "b".to_string(),
            ],
            Some("stratum".to_string()),
        )
        .expect("operation should succeed");
        // A non-`String` value column, for the same dtype-preservation
        // reason as `test_simple_sample` above.
        let value = Series::new(vec![1i64, 2, 3, 4, 5, 6, 7, 8], Some("value".to_string()))
            .expect("operation should succeed");
        df.add_column("stratum".to_string(), strata)
            .expect("operation should succeed");
        df.add_column("value".to_string(), value)
            .expect("operation should succeed");
        df
    }

    #[test]
    fn test_stratified_sample_preserves_non_string_columns() {
        let df = stratified_test_df();
        let sample = stratified_sample_impl(&df, "stratum", 0.5, false, None)
            .expect("operation should succeed");
        assert_eq!(sample.column_count(), df.column_count());
        let values = sample
            .get_column::<i64>("value")
            .expect("the i64 column must round-trip through stratified sampling")
            .values()
            .to_vec();
        // 50% of 4 rows per stratum, ceil'd = 2 rows per stratum, 2 strata.
        assert_eq!(values.len(), 4);
    }

    #[test]
    fn test_stratified_sample_is_reproducible_with_seed() {
        let df = stratified_test_df();

        let a = stratified_sample_impl(&df, "stratum", 0.5, false, Some(7))
            .expect("operation should succeed")
            .get_column::<i64>("value")
            .expect("column present")
            .values()
            .to_vec();
        // Repeated many times: with a `HashMap`-ordered stratum loop and a
        // freshly-seeded RNG per stratum, this could still coincidentally
        // match once; running it several times makes a spurious pass
        // vanishingly unlikely while still finishing instantly.
        for _ in 0..20 {
            let b = stratified_sample_impl(&df, "stratum", 0.5, false, Some(7))
                .expect("operation should succeed")
                .get_column::<i64>("value")
                .expect("column present")
                .values()
                .to_vec();
            assert_eq!(
                a, b,
                "the same seed must draw a byte-identical stratified sample every time"
            );
        }
    }
}

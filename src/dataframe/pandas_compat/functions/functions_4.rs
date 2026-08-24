//! Companion pandas-compat extension traits.
//!
//! [`crate::dataframe::pandas_compat::trait_def::PandasCompatExt`] is owned
//! by a different module (its signatures can't be widened from here), so a
//! handful of pandas methods that need one more parameter than their
//! `PandasCompatExt` counterpart exposes are implemented as small
//! standalone traits instead: real, additional functionality rather than a
//! trait-signature change this module doesn't own.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use std::collections::HashMap;

/// `astype` with an explicit `errors` policy ("raise" or "coerce"),
/// matching pandas' `DataFrame.astype(dtype, errors=...)`.
///
/// [`PandasCompatExt::astype`](super::super::trait_def::PandasCompatExt::astype)
/// always behaves as `errors="raise"` (pandas' own default); call
/// `astype_errors` directly when unparsable values should become nulls
/// instead of failing the whole conversion.
pub trait AstypeErrorsExt {
    fn astype_errors(&self, column: &str, dtype: &str, errors: &str) -> Result<DataFrame>;
}

impl AstypeErrorsExt for DataFrame {
    fn astype_errors(&self, column: &str, dtype: &str, errors: &str) -> Result<DataFrame> {
        super::functions_2_impl_part2::astype_with_errors(self, column, dtype, errors)
    }
}

/// `value_counts` with pandas' full parameter set.
///
/// [`PandasCompatExt::value_counts`](super::super::trait_def::PandasCompatExt::value_counts)
/// always behaves as `normalize=false, sort=true, ascending=false,
/// dropna=true` (pandas' defaults); call `value_counts_with_options`
/// directly for anything else.
pub trait ValueCountsOptionsExt {
    /// * `normalize` - return each value's proportion of the total instead
    ///   of its raw count.
    /// * `sort` - sort the result by frequency (otherwise it is returned in
    ///   first-seen order).
    /// * `ascending` - sort direction when `sort` is true.
    /// * `dropna` - exclude missing values (empty strings for object
    ///   columns, `NaN` for numeric columns rendered through this column's
    ///   string form) from both the counts and the total used by
    ///   `normalize`.
    fn value_counts_with_options(
        &self,
        column: &str,
        normalize: bool,
        sort: bool,
        ascending: bool,
        dropna: bool,
    ) -> Result<Vec<(String, f64)>>;
}

impl ValueCountsOptionsExt for DataFrame {
    fn value_counts_with_options(
        &self,
        column: &str,
        normalize: bool,
        sort: bool,
        ascending: bool,
        dropna: bool,
    ) -> Result<Vec<(String, f64)>> {
        if !self.contains_column(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }
        let raw_values = self.get_column_string_values(column)?;
        let values: Vec<String> = if dropna {
            raw_values
                .into_iter()
                .filter(|v| !v.is_empty() && v != "NaN" && v != "nan")
                .collect()
        } else {
            raw_values
        };

        let mut counts: HashMap<String, usize> = HashMap::new();
        // Track first-seen order so the result is fully deterministic even
        // when `sort` is false (no HashMap-iteration order leaking through)
        // and so frequency ties under `sort` break by first appearance.
        let mut order: Vec<String> = Vec::new();
        for v in values {
            if !counts.contains_key(&v) {
                order.push(v.clone());
            }
            *counts.entry(v).or_insert(0) += 1;
        }

        let total: usize = counts.values().sum();
        let mut result: Vec<(String, f64)> = order
            .into_iter()
            .map(|v| {
                let count = counts.get(&v).copied().unwrap_or(0);
                let reported = if normalize {
                    if total == 0 {
                        0.0
                    } else {
                        count as f64 / total as f64
                    }
                } else {
                    count as f64
                };
                (v, reported)
            })
            .collect();

        if sort {
            result.sort_by(|a, b| {
                let cmp = a.1.total_cmp(&b.1);
                if ascending {
                    cmp
                } else {
                    cmp.reverse()
                }
            });
        }

        Ok(result)
    }
}

/// `between` with pandas' full `inclusive` vocabulary (`"both"`,
/// `"neither"`, `"left"`, `"right"`) instead of the plain boolean
/// [`PandasCompatExt::is_between`](super::super::trait_def::PandasCompatExt::is_between)
/// offers (`true`/`false` for "both"/"neither" only).
pub trait BetweenInclusiveExt {
    fn between_inclusive(
        &self,
        column: &str,
        lower: f64,
        upper: f64,
        inclusive: &str,
    ) -> Result<Vec<bool>>;
}

impl BetweenInclusiveExt for DataFrame {
    fn between_inclusive(
        &self,
        column: &str,
        lower: f64,
        upper: f64,
        inclusive: &str,
    ) -> Result<Vec<bool>> {
        let (lower_ok, upper_ok): (fn(f64, f64) -> bool, fn(f64, f64) -> bool) = match inclusive {
            "both" => (|v, b| v >= b, |v, b| v <= b),
            "neither" => (|v, b| v > b, |v, b| v < b),
            "left" => (|v, b| v >= b, |v, b| v < b),
            "right" => (|v, b| v > b, |v, b| v <= b),
            other => {
                return Err(Error::InvalidValue(format!(
                    "inclusive must be one of 'both', 'neither', 'left', 'right', got '{}'",
                    other
                )))
            }
        };
        let values = self.get_column_numeric_values(column)?;
        Ok(values
            .iter()
            .map(|&v| !v.is_nan() && lower_ok(v, lower) && upper_ok(v, upper))
            .collect())
    }
}

/// `describe` with an explicit, arbitrary percentile list, matching
/// pandas' `DataFrame.describe(percentiles=[...])`.
///
/// [`PandasCompatExt::describe`](super::super::trait_def::PandasCompatExt::describe)
/// always reports the fixed 25%/50%/75% quartiles (pandas' own default);
/// call `describe_percentiles` directly for any other set. As in pandas,
/// the 50th percentile (median) is always included even if not requested,
/// and results use the same skipna=True + ddof=1 + linear-interpolation
/// quantile convention as `describe`/`describe_column` so all three stay
/// numerically consistent with each other on the same column.
pub trait DescribePercentilesExt {
    /// `percentiles` are fractions in `[0, 1]` (e.g. `0.1` for the 10th
    /// percentile), matching pandas. Returns `(label, value)` pairs in
    /// pandas' own row order: count, mean, std, min, the requested
    /// percentiles (ascending, deduplicated, always including 50%), max.
    fn describe_percentiles(&self, column: &str, percentiles: &[f64])
        -> Result<Vec<(String, f64)>>;
}

impl DescribePercentilesExt for DataFrame {
    fn describe_percentiles(
        &self,
        column: &str,
        percentiles: &[f64],
    ) -> Result<Vec<(String, f64)>> {
        for &p in percentiles {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidValue(format!(
                    "percentiles must be between 0 and 1, got {}",
                    p
                )));
            }
        }

        let raw_values = self.get_column_numeric_values(column)?;
        if raw_values.is_empty() {
            return Err(Error::Empty("Cannot describe empty column".to_string()));
        }
        // skipna=True, matching `describe`'s own default.
        let values: Vec<f64> = raw_values.into_iter().filter(|v| !v.is_nan()).collect();
        let count = values.len();

        // pandas always includes the median even if the caller didn't ask
        // for it, and reports every percentile in ascending, deduplicated
        // order regardless of the order they were passed in.
        let mut requested: Vec<f64> = percentiles.to_vec();
        requested.push(0.5);
        requested.sort_by(|a, b| a.total_cmp(b));
        requested.dedup();
        let percentile_label = |p: f64| -> String {
            let pct = p * 100.0;
            if (pct - pct.round()).abs() < 1e-9 {
                format!("{}%", pct.round() as i64)
            } else {
                format!("{}%", pct)
            }
        };

        let mut result = Vec::with_capacity(4 + requested.len());
        result.push(("count".to_string(), count as f64));

        if count == 0 {
            // All-NaN column: pandas reports count=0 and NaN for the rest
            // rather than erroring (an *empty* column, handled above, is
            // the only case that raises).
            result.push(("mean".to_string(), f64::NAN));
            result.push(("std".to_string(), f64::NAN));
            result.push(("min".to_string(), f64::NAN));
            for p in &requested {
                result.push((percentile_label(*p), f64::NAN));
            }
            result.push(("max".to_string(), f64::NAN));
            return Ok(result);
        }

        let n = count as f64;
        let mean = values.iter().sum::<f64>() / n;
        // Sample standard deviation (ddof=1): undefined (NaN) for a single
        // observation, matching `describe`/`describe_column`.
        let std = if count < 2 {
            f64::NAN
        } else {
            let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        };
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        result.push(("mean".to_string(), mean));
        result.push(("std".to_string(), std));
        result.push(("min".to_string(), sorted[0]));
        for p in &requested {
            result.push((
                percentile_label(*p),
                linear_interpolated_percentile(&sorted, *p),
            ));
        }
        result.push(("max".to_string(), sorted[count - 1]));

        Ok(result)
    }
}

/// Linear-interpolation quantile of an already-sorted slice, `p` a
/// fraction in `[0, 1]`. Shared formula with `describe`/`describe_column`
/// (pandas' default `interpolation="linear"`), duplicated locally rather
/// than exposed from those `functions_2_impl_part1`/`helpers::aggregations`
/// modules since neither exports it as a standalone function.
fn linear_interpolated_percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = idx - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

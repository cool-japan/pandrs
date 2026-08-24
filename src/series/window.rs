//! Window operations for Series
//!
//! This module provides pandas-like window operations including rolling windows,
//! expanding windows, and exponentially weighted moving operations.

use crate::core::error::{Error, Result};
use crate::series::base::Series;
use std::fmt::Debug;

/// Convert a `Series<T>`'s values to `Option<f64>`, treating `NaN` as
/// missing -- `Series<T>` has no null bitmap, so `NaN` is this crate's
/// missing-value sentinel for a plain, non-`NA`-wrapped numeric series
/// (matching `Series::<f64>::sum`'s convention). Shared by `Rolling`,
/// `Expanding`, and `EWM` so all three treat "missing" the same way; this
/// is also what makes `EWM::ignore_na` meaningful at all (an input that
/// unconditionally wrapped every value in `Some` had no missing rows to
/// ever exercise that flag against).
fn series_values_as_f64_opt<T>(series: &Series<T>) -> Vec<Option<f64>>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    series
        .values()
        .iter()
        .map(|&v| {
            let f: f64 = v.into();
            if f.is_nan() {
                None
            } else {
                Some(f)
            }
        })
        .collect()
}

/// Rolling window configuration and operations
#[derive(Debug, Clone)]
pub struct Rolling<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    window_size: usize,
    min_periods: Option<usize>,
    center: bool,
    closed: WindowClosed,
}

/// Expanding window configuration and operations
#[derive(Debug, Clone)]
pub struct Expanding<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    min_periods: usize,
}

/// Exponentially weighted moving window configuration and operations
#[derive(Debug, Clone)]
pub struct EWM<T>
where
    T: Debug + Clone,
{
    series: Series<T>,
    alpha: Option<f64>,
    span: Option<usize>,
    halflife: Option<f64>,
    adjust: bool,
    ignore_na: bool,
    min_periods: usize,
}

/// How to handle window boundaries.
///
/// Applied to the trailing (non-centered) window `[i + 1 - window_size, i]`
/// at each position `i`, mirroring pandas' fixed-window indexer: `Right`
/// (the default) leaves it unshifted; `Left` shifts the whole window one
/// step earlier (drop the current row, include one more from the past);
/// `Both` keeps the current row and additionally includes one more from
/// the past (window grows by one); `Neither` drops the current row without
/// adding a past one (window shrinks by one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowClosed {
    /// Window includes both endpoints
    Both,
    /// Window includes left endpoint only
    Left,
    /// Window includes right endpoint only
    Right,
    /// Window includes neither endpoint
    Neither,
}

impl Default for WindowClosed {
    fn default() -> Self {
        WindowClosed::Right
    }
}

/// Trait for window aggregation operations
pub trait WindowOps<T>
where
    T: Debug + Clone,
{
    /// Calculate the mean of the window
    fn mean(&self) -> Result<Series<f64>>;

    /// Calculate the sum of the window
    fn sum(&self) -> Result<Series<f64>>;

    /// Calculate the standard deviation of the window
    fn std(&self, ddof: usize) -> Result<Series<f64>>;

    /// Calculate the variance of the window
    fn var(&self, ddof: usize) -> Result<Series<f64>>;

    /// Calculate the minimum value in the window
    fn min(&self) -> Result<Series<f64>>;

    /// Calculate the maximum value in the window
    fn max(&self) -> Result<Series<f64>>;

    /// Count non-null values in the window
    fn count(&self) -> Result<Series<usize>>;

    /// Calculate the median of the window
    fn median(&self) -> Result<Series<f64>>;

    /// Calculate a quantile of the window
    fn quantile(&self, q: f64) -> Result<Series<f64>>;

    /// Apply a custom aggregation function.
    ///
    /// The result is aligned with the input series: one entry per row,
    /// `None` wherever that row didn't have enough observations for
    /// `min_periods` (rather than being dropped, which would silently
    /// shorten and misalign the output relative to the input series).
    fn apply<F, R>(&self, func: F) -> Result<Series<Option<R>>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone;
}

// Implementation for Rolling windows
impl<T> Rolling<T>
where
    T: Debug + Clone,
{
    /// Create a new rolling window
    pub fn new(series: Series<T>, window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(Error::InvalidValue(
                "Window size must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            series,
            window_size,
            min_periods: None,
            center: false,
            closed: WindowClosed::default(),
        })
    }

    /// Set minimum number of observations required to have a value
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = Some(min_periods);
        self
    }

    /// Set whether to center the window around the current observation
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set how to handle window boundaries
    pub fn closed(mut self, closed: WindowClosed) -> Self {
        self.closed = closed;
        self
    }

    /// Get the effective minimum periods
    fn effective_min_periods(&self) -> usize {
        self.min_periods.unwrap_or(self.window_size)
    }

    /// Convert values to f64 for calculations, treating `NaN` as missing.
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        Ok(series_values_as_f64_opt(&self.series))
    }

    /// Apply window operation with generic aggregation function.
    ///
    /// Honors `window_size`, `min_periods`, `closed`, and `center`:
    /// - The base (non-centered) window at position `i` is the trailing
    ///   range `[i + 1 - window_size, i]`, clipped to `[0, len)`; `closed`
    ///   shifts that range's endpoints before clipping, mirroring pandas'
    ///   `FixedWindowIndexer` (`Left`/`Both` pull the start back by one,
    ///   `Left`/`Neither` pull the end back by one; `Right`, the default,
    ///   is the unshifted base range -- see [`WindowClosed`]'s docs).
    /// - `center` is applied as a *post-hoc shift* of the trailing results
    ///   by `window_size / 2` positions, exactly like pandas: the trailing
    ///   (non-centered) result array is computed first, honoring
    ///   `min_periods` as usual, and the final output at position `i` is
    ///   the trailing result at position `i + window_size / 2` (`None` if
    ///   that's out of range). This reports missing/insufficient data at
    ///   the series' edges as `None`, rather than clamping the window's
    ///   start to `0` and silently pulling in extra right-side context
    ///   there instead.
    /// - An empty window (reachable with `closed = Neither`/`Both` at the
    ///   series' edges, or when every value in range is `NaN`) is always
    ///   `None` regardless of `min_periods`: `min_periods == 0` means "at
    ///   least zero observations is fine", not "call the aggregator on
    ///   nothing" -- `median`/`quantile` would panic indexing an empty
    ///   slice, and `min`/`max`'s `INFINITY`/`NEG_INFINITY` fold seeds
    ///   would otherwise leak out as fabricated data.
    fn apply_window_op<F, R>(&self, mut func: F) -> Result<Series<Option<R>>>
    where
        T: Into<f64> + Copy,
        F: FnMut(&[f64]) -> R,
        R: Debug + Clone,
    {
        let values = self.values_as_f64()?;
        let n = values.len();
        let min_periods = self.effective_min_periods();

        let mut trailing: Vec<Option<R>> = Vec::with_capacity(n);
        for i in 0..n {
            let end1 = (i + 1) as i64;
            let start1 = end1 - self.window_size as i64;
            let (mut start, mut end) = (start1, end1);
            match self.closed {
                WindowClosed::Right => {}
                WindowClosed::Left => {
                    start -= 1;
                    end -= 1;
                }
                WindowClosed::Both => {
                    start -= 1;
                }
                WindowClosed::Neither => {
                    end -= 1;
                }
            }
            let start = start.clamp(0, n as i64) as usize;
            let end = end.clamp(0, n as i64) as usize;

            let window_values: Vec<f64> = if end > start {
                values[start..end].iter().filter_map(|&v| v).collect()
            } else {
                Vec::new()
            };

            if !window_values.is_empty() && window_values.len() >= min_periods {
                trailing.push(Some(func(&window_values)));
            } else {
                trailing.push(None);
            }
        }

        let result: Vec<Option<R>> = if self.center {
            let offset = self.window_size / 2;
            (0..n)
                .map(|i| {
                    let src = i + offset;
                    if src < n {
                        trailing[src].clone()
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            trailing
        };

        Series::new(result, self.series.name().cloned())
    }
}

impl<T> WindowOps<T> for Rolling<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    fn mean(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().sum::<f64>() / values.len() as f64)?;

        // Convert Option<f64> to f64 (with NaN for None)
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn sum(&self) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| values.iter().sum::<f64>())?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64;
                variance.sqrt()
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn min(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().fold(f64::INFINITY, |a, &b| a.min(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn max(&self) -> Result<Series<f64>> {
        let result =
            self.apply_window_op(|values| values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn count(&self) -> Result<Series<usize>> {
        let result = self.apply_window_op(|values| values.len())?;
        let values: Vec<usize> = result.values().iter().map(|&v| v.unwrap_or(0)).collect();
        Series::new(values, result.name().cloned())
    }

    fn median(&self) -> Result<Series<f64>> {
        let result = self.apply_window_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn quantile(&self, q: f64) -> Result<Series<f64>> {
        if q < 0.0 || q > 1.0 {
            return Err(Error::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        let result = self.apply_window_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (q * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn apply<F, R>(&self, func: F) -> Result<Series<Option<R>>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone,
    {
        self.apply_window_op(func)
    }
}

// Implementation for Expanding windows
impl<T> Expanding<T>
where
    T: Debug + Clone,
{
    /// Create a new expanding window
    pub fn new(series: Series<T>, min_periods: usize) -> Result<Self> {
        Ok(Self {
            series,
            min_periods,
        })
    }

    /// Convert values to f64 for calculations, treating `NaN` as missing.
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        Ok(series_values_as_f64_opt(&self.series))
    }

    /// Apply expanding operation with generic aggregation function.
    ///
    /// An empty window (reachable when `min_periods == 0` and every value
    /// from the start through position `i` is `NaN`) is always `None`
    /// regardless of `min_periods` -- see `Rolling::apply_window_op`'s docs
    /// for why calling the aggregator on an empty slice must be avoided.
    fn apply_expanding_op<F, R>(&self, mut func: F) -> Result<Series<Option<R>>>
    where
        T: Into<f64> + Copy,
        F: FnMut(&[f64]) -> R,
        R: Debug + Clone,
    {
        let values = self.values_as_f64()?;
        let mut result = Vec::with_capacity(values.len());

        for i in 0..values.len() {
            // Get all values from start to current position
            let window_values: Vec<f64> = values[0..=i].iter().filter_map(|&v| v).collect();

            if !window_values.is_empty() && window_values.len() >= self.min_periods {
                let agg_result = func(&window_values);
                result.push(Some(agg_result));
            } else {
                result.push(None);
            }
        }

        Series::new(result, self.series.name().cloned())
    }
}

impl<T> WindowOps<T> for Expanding<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    fn mean(&self) -> Result<Series<f64>> {
        let result =
            self.apply_expanding_op(|values| values.iter().sum::<f64>() / values.len() as f64)?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn sum(&self) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| values.iter().sum::<f64>())?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64;
                variance.sqrt()
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            if values.len() <= ddof {
                f64::NAN
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - ddof) as f64
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn min(&self) -> Result<Series<f64>> {
        let result =
            self.apply_expanding_op(|values| values.iter().fold(f64::INFINITY, |a, &b| a.min(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn max(&self) -> Result<Series<f64>> {
        let result = self
            .apply_expanding_op(|values| values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn count(&self) -> Result<Series<usize>> {
        let result = self.apply_expanding_op(|values| values.len())?;
        let values: Vec<usize> = result.values().iter().map(|&v| v.unwrap_or(0)).collect();
        Series::new(values, result.name().cloned())
    }

    fn median(&self) -> Result<Series<f64>> {
        let result = self.apply_expanding_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn quantile(&self, q: f64) -> Result<Series<f64>> {
        if q < 0.0 || q > 1.0 {
            return Err(Error::InvalidValue(
                "Quantile must be between 0 and 1".to_string(),
            ));
        }

        let result = self.apply_expanding_op(|values| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (q * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })?;
        let values: Vec<f64> = result
            .values()
            .iter()
            .map(|&v| v.unwrap_or(f64::NAN))
            .collect();
        Series::new(values, result.name().cloned())
    }

    fn apply<F, R>(&self, func: F) -> Result<Series<Option<R>>>
    where
        F: Fn(&[f64]) -> R + Copy,
        R: Debug + Clone,
    {
        self.apply_expanding_op(func)
    }
}

// Implementation for EWM windows
impl<T> EWM<T>
where
    T: Debug + Clone,
{
    /// Create a new exponentially weighted moving window
    pub fn new(series: Series<T>) -> Self {
        Self {
            series,
            alpha: None,
            span: None,
            halflife: None,
            adjust: true,
            ignore_na: false,
            min_periods: 0,
        }
    }

    /// Set the smoothing factor alpha directly
    pub fn alpha(mut self, alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(Error::InvalidValue(
                "Alpha must be between 0 and 1".to_string(),
            ));
        }
        self.alpha = Some(alpha);
        self.span = None;
        self.halflife = None;
        Ok(self)
    }

    /// Set the span (window size)
    pub fn span(mut self, span: usize) -> Self {
        self.span = Some(span);
        self.alpha = None;
        self.halflife = None;
        self
    }

    /// Set the halflife
    pub fn halflife(mut self, halflife: f64) -> Self {
        self.halflife = Some(halflife);
        self.alpha = None;
        self.span = None;
        self
    }

    /// Set whether to use adjustment
    pub fn adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    /// Set whether to ignore NA values
    pub fn ignore_na(mut self, ignore_na: bool) -> Self {
        self.ignore_na = ignore_na;
        self
    }

    /// Set the minimum number of observations required to have a value
    /// (default `0`, meaning a single observation is already enough --
    /// matching pandas' `EWM(min_periods=0)` default). Positions with
    /// fewer non-NA observations than this report `NaN`.
    pub fn min_periods(mut self, min_periods: usize) -> Self {
        self.min_periods = min_periods;
        self
    }

    /// Calculate the effective alpha value
    fn get_alpha(&self) -> Result<f64> {
        if let Some(alpha) = self.alpha {
            Ok(alpha)
        } else if let Some(span) = self.span {
            Ok(2.0 / (span as f64 + 1.0))
        } else if let Some(halflife) = self.halflife {
            Ok(1.0 - (-std::f64::consts::LN_2 / halflife).exp())
        } else {
            Err(Error::InvalidValue(
                "Must specify either alpha, span, or halflife".to_string(),
            ))
        }
    }

    /// Convert values to f64 for calculations, treating `NaN` as missing.
    fn values_as_f64(&self) -> Result<Vec<Option<f64>>>
    where
        T: Into<f64> + Copy,
    {
        Ok(series_values_as_f64_opt(&self.series))
    }
}

/// Running state produced by `ewm_recursion` for every position: the
/// exponentially weighted mean, the biased (`ddof = 0`) weighted variance,
/// the weight-based effective sample size (`sum_wt^2 / sum_wt2`, used to
/// bias-correct the variance for an arbitrary `ddof`), and the observation
/// count so far.
struct EwmState {
    mean: Vec<f64>,
    cov_biased: Vec<f64>,
    n_eff: Vec<f64>,
    nobs: Vec<usize>,
}

/// The recursive EWM algorithm pandas uses (`pandas.core.window.ewm`,
/// `_libs/window/aggregations.pyx`'s `ewma`/`ewmcov`), specialized here to
/// a single series with no `times=`/variable-spacing support (every step
/// decays by exactly one period). Hand-verified against the direct
/// weighted-average definition `y_t = sum_i w_i x_{t-i} / sum_i w_i` with
/// `w_i = (1 - alpha)^i` for `adjust = true`, the plain recursive form
/// `y_t = alpha * x_t + (1 - alpha) * y_{t-1}` for `adjust = false`, and
/// pandas' own `ignore_na` docstring example (`[x0, None, x2]`'s weights
/// are `(1-alpha)^2, 1` when `ignore_na = false` and `1-alpha, 1` when
/// `ignore_na = true`, for `adjust = true`) -- for both `adjust` settings
/// and both `ignore_na` settings; see the
/// `ewm_adjust_matches_pandas_formula` regression test.
///
/// `old_wt`/`new_wt` drive the mean recursion; `sum_wt`/`sum_wt2`
/// separately accumulate the weight moments used for the variance's
/// `ddof` bias correction in [`EWM::var`] (`n_eff = sum_wt^2 / sum_wt2`;
/// the corrected variance is `cov_biased * n_eff / (n_eff - ddof)`, which
/// reduces to pandas' own `sum_wt^2 / (sum_wt^2 - sum_wt2)` bias-correction
/// factor at `ddof = 1`, pandas' default).
fn ewm_recursion(values: &[Option<f64>], alpha: f64, adjust: bool, ignore_na: bool) -> EwmState {
    let n = values.len();
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };

    let mut mean_out = vec![f64::NAN; n];
    let mut cov_out = vec![f64::NAN; n];
    let mut n_eff_out = vec![f64::NAN; n];
    let mut nobs_out = vec![0usize; n];

    let mut mean: Option<f64> = None;
    let mut cov = 0.0_f64;
    let mut old_wt = 1.0_f64;
    let mut sum_wt = 1.0_f64;
    let mut sum_wt2 = 1.0_f64;
    let mut nobs = 0usize;

    for i in 0..n {
        let cur = values[i];
        let is_obs = cur.is_some();
        if is_obs {
            nobs += 1;
        }

        if i == 0 {
            if let Some(v) = cur {
                mean = Some(v);
                cov = 0.0;
            }
        } else if let Some(m) = mean {
            if is_obs || !ignore_na {
                old_wt *= old_wt_factor;
                sum_wt *= old_wt_factor;
                sum_wt2 *= old_wt_factor * old_wt_factor;
            }
            if let Some(c) = cur {
                let old_mean = m;
                let new_mean = if old_mean != c {
                    (old_wt * old_mean + new_wt * c) / (old_wt + new_wt)
                } else {
                    old_mean
                };
                cov = (old_wt * (cov + (old_mean - new_mean) * (old_mean - new_mean))
                    + new_wt * (c - new_mean) * (c - new_mean))
                    / (old_wt + new_wt);
                mean = Some(new_mean);

                sum_wt += new_wt;
                sum_wt2 += new_wt * new_wt;
                old_wt += new_wt;
                if !adjust {
                    sum_wt /= old_wt;
                    sum_wt2 /= old_wt * old_wt;
                    old_wt = 1.0;
                }
            }
        } else if is_obs {
            mean = cur;
            cov = 0.0;
        }

        nobs_out[i] = nobs;
        if let Some(m) = mean {
            mean_out[i] = m;
            cov_out[i] = cov;
            n_eff_out[i] = if sum_wt2 > 0.0 {
                sum_wt * sum_wt / sum_wt2
            } else {
                f64::NAN
            };
        }
    }

    EwmState {
        mean: mean_out,
        cov_biased: cov_out,
        n_eff: n_eff_out,
        nobs: nobs_out,
    }
}

impl<T> EWM<T>
where
    T: Debug + Clone + Into<f64> + Copy,
{
    /// Calculate exponentially weighted moving average.
    ///
    /// Honors `adjust` (default `true`, pandas' default: the "adjusted"
    /// weighted average over *all* prior observations, vs. the plain
    /// recursive form `y_t = alpha * x_t + (1 - alpha) * y_{t-1}` when
    /// `false`, which is what this crate previously computed
    /// unconditionally regardless of `adjust`), `ignore_na` (whether a gap
    /// left by a missing value still costs decay weight), and
    /// `min_periods`. See `ewm_recursion`'s docs for the algorithm and
    /// how it was checked.
    pub fn mean(&self) -> Result<Series<f64>> {
        let alpha = self.get_alpha()?;
        let values = self.values_as_f64()?;
        let state = ewm_recursion(&values, alpha, self.adjust, self.ignore_na);

        let result: Vec<f64> = (0..values.len())
            .map(|i| {
                if state.nobs[i] >= self.min_periods {
                    state.mean[i]
                } else {
                    f64::NAN
                }
            })
            .collect();
        Series::new(result, self.series.name().cloned())
    }

    /// Calculate exponentially weighted moving variance.
    ///
    /// `ddof` (previously accepted but silently ignored) bias-corrects the
    /// weighted variance via its effective sample size,
    /// `n_eff = sum_wt^2 / sum_wt2`: the reported variance is
    /// `cov_biased * n_eff / (n_eff - ddof)`, `NaN` when there isn't enough
    /// effective sample size for that correction (`n_eff <= ddof` -- e.g.
    /// `ddof = 1`, pandas' implied default, is always `NaN` at the first
    /// observation, matching pandas). Pass `ddof = 0` for the plain
    /// ("biased") weighted variance. Clamped to `0.0` to absorb float
    /// rounding noise near zero (the underlying weighted sum of squared
    /// deviations cannot be negative).
    pub fn var(&self, ddof: usize) -> Result<Series<f64>> {
        let alpha = self.get_alpha()?;
        let values = self.values_as_f64()?;
        let state = ewm_recursion(&values, alpha, self.adjust, self.ignore_na);
        let ddof = ddof as f64;

        let result: Vec<f64> = (0..values.len())
            .map(|i| {
                if state.nobs[i] < self.min_periods {
                    return f64::NAN;
                }
                let n_eff = state.n_eff[i];
                let denom = n_eff - ddof;
                if denom > 0.0 && state.cov_biased[i].is_finite() {
                    (state.cov_biased[i] * n_eff / denom).max(0.0)
                } else {
                    f64::NAN
                }
            })
            .collect();
        Series::new(result, self.series.name().cloned())
    }

    /// Calculate exponentially weighted moving standard deviation: `sqrt`
    /// of [`EWM::var`] (computed directly, not squared back out of a
    /// separately-computed `std`, so the `ddof` correction only ever needs
    /// to be applied once). See [`EWM::var`]'s docs for `ddof`.
    pub fn std(&self, ddof: usize) -> Result<Series<f64>> {
        let var_series = self.var(ddof)?;
        let std_values: Vec<f64> = var_series.values().iter().map(|&v| v.sqrt()).collect();
        Series::new(std_values, var_series.name().cloned())
    }
}

/// Extension trait to add window operations to Series
pub trait WindowExt<T>
where
    T: Debug + Clone,
{
    /// Create a rolling window
    fn rolling(&self, window_size: usize) -> Result<Rolling<T>>;

    /// Create an expanding window
    fn expanding(&self, min_periods: usize) -> Result<Expanding<T>>;

    /// Create an exponentially weighted moving window
    fn ewm(&self) -> EWM<T>;
}

impl<T> WindowExt<T> for Series<T>
where
    T: Debug + Clone,
{
    fn rolling(&self, window_size: usize) -> Result<Rolling<T>> {
        Rolling::new(self.clone(), window_size)
    }

    fn expanding(&self, min_periods: usize) -> Result<Expanding<T>> {
        Expanding::new(self.clone(), min_periods)
    }

    fn ewm(&self) -> EWM<T> {
        EWM::new(self.clone())
    }
}

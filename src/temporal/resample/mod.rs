use chrono::TimeZone;
use std::collections::HashMap;

use crate::error::{PandRSError, Result};
use crate::temporal::core::Temporal;
use crate::temporal::core::TimeSeries;
use crate::temporal::frequency::Frequency;

/// Structure representing resampling operations
#[derive(Debug)]
pub struct Resample<'a, T: Temporal> {
    /// Original time series
    series: &'a TimeSeries<T>,

    /// Resampling frequency
    frequency: Frequency,
}

impl<'a, T: Temporal> Resample<'a, T> {
    /// Create a new resampling operation
    pub fn new(series: &'a TimeSeries<T>, frequency: Frequency) -> Self {
        Resample { series, frequency }
    }

    /// Resample using mean. Empty buckets are `NA`, not `0.0` — see
    /// [`Resample::aggregate`], which never invokes the aggregator on an empty
    /// bucket.
    pub fn mean(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            let sum: f64 = values.iter().sum();
            sum / values.len() as f64
        })
    }

    /// Resample using sum
    pub fn sum(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| values.iter().sum())
    }

    /// Resample using maximum
    pub fn max(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    }

    /// Resample using minimum
    pub fn min(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| values.iter().copied().fold(f64::INFINITY, f64::min))
    }

    /// Resample using a custom aggregation function.
    ///
    /// The result is a **gapless** series at the requested frequency: every
    /// bucket between the first and last observation gets a row, and buckets
    /// that received no observation are `NA::NA`. `aggregator` is only ever
    /// called with a non-empty slice.
    ///
    /// Previously only the buckets that happened to contain data were emitted,
    /// while the result was still tagged with the resampling frequency — so a
    /// daily resample of a series with a two-day gap produced consecutive rows
    /// one day apart in the index but two days apart in reality, and any
    /// consumer trusting `frequency()` (differencing, seasonal alignment,
    /// plotting) silently mis-indexed everything after the gap.
    pub fn aggregate<F>(&self, aggregator: F) -> Result<TimeSeries<T>>
    where
        F: Fn(Vec<f64>) -> f64,
    {
        let timestamps = self.series.timestamps();
        if timestamps.is_empty() {
            return TimeSeries::new(Vec::new(), Vec::new(), self.series.name().cloned())
                .map(|ts| ts.with_frequency(self.frequency.clone()));
        }

        let freq_seconds = self.frequency.to_seconds();
        if freq_seconds <= 0 {
            return Err(PandRSError::Consistency(format!(
                "Resampling frequency must be positive, got {freq_seconds} seconds"
            )));
        }

        // Group by period
        let mut period_groups: HashMap<i64, Vec<f64>> = HashMap::new();

        // Assign each data point to the appropriate period
        let start_seconds = timestamps[0].to_utc().timestamp();
        let mut last_period = 0_i64;

        for (i, timestamp) in timestamps.iter().enumerate() {
            // Calculate which period it belongs to. `div_euclid` keeps buckets
            // uniform for timestamps that precede the first one.
            let ts_seconds = timestamp.to_utc().timestamp();
            let period = (ts_seconds - start_seconds).div_euclid(freq_seconds);
            last_period = last_period.max(period);

            // NA observations still define the bucket span but contribute no
            // value to the aggregate.
            if let crate::na::NA::Value(value) = self.series.values()[i] {
                period_groups.entry(period).or_default().push(value);
            }
        }

        // Create aggregation results over the *full* bucket range, so the
        // output really is uniform at `self.frequency`.
        let bucket_count = (last_period + 1).max(0) as usize;
        let mut result_values = Vec::with_capacity(bucket_count);
        let mut result_timestamps = Vec::with_capacity(bucket_count);

        for period in 0..=last_period {
            match period_groups.get(&period) {
                Some(values) if !values.is_empty() => {
                    result_values.push(crate::na::NA::Value(aggregator(values.clone())));
                }
                _ => result_values.push(crate::na::NA::NA),
            }

            // Calculate the representative time for this period.
            let period_start_seconds = start_seconds + period * freq_seconds;
            let period_time = match chrono::Utc.timestamp_opt(period_start_seconds, 0) {
                chrono::LocalResult::Single(t) => t,
                // UTC has no DST folds, so `Ambiguous` cannot occur here; both
                // remaining arms mean the instant is outside chrono's range.
                chrono::LocalResult::Ambiguous(earliest, _) => earliest,
                chrono::LocalResult::None => {
                    return Err(PandRSError::Consistency(format!(
                        "Resampled bucket start {period_start_seconds} is outside the \
                         representable timestamp range"
                    )))
                }
            };

            // Convert to the appropriate type
            result_timestamps.push(T::from_str(&period_time.to_rfc3339())?);
        }

        // Create a new time series
        TimeSeries::new(
            result_values,
            result_timestamps,
            self.series.name().cloned(),
        )
        .map(|ts| ts.with_frequency(self.frequency.clone()))
    }
}

use chrono::TimeZone;
use std::collections::HashMap;

use crate::error::{PandRSError, Result};
use crate::temporal::{Frequency, Temporal, TimeSeries};

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
    /// Mirrors `temporal::resample::Resample::aggregate`: the result is gapless
    /// at the requested frequency (buckets with no observation are `NA::NA`,
    /// not omitted, so the index really is uniform), the aggregator is never
    /// called with an empty slice, and an out-of-range bucket start is reported
    /// as an error rather than an `.expect(...)` panic.
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
        let start_seconds = timestamps[0].to_utc().timestamp();
        let mut last_period = 0_i64;

        for (i, timestamp) in timestamps.iter().enumerate() {
            let ts_seconds = timestamp.to_utc().timestamp();
            let period = (ts_seconds - start_seconds).div_euclid(freq_seconds);
            last_period = last_period.max(period);

            if let crate::na::NA::Value(value) = self.series.values()[i] {
                period_groups.entry(period).or_default().push(value);
            }
        }

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

            let period_start_seconds = start_seconds + period * freq_seconds;
            let period_time = match chrono::Utc.timestamp_opt(period_start_seconds, 0) {
                chrono::LocalResult::Single(t) => t,
                chrono::LocalResult::Ambiguous(earliest, _) => earliest,
                chrono::LocalResult::None => {
                    return Err(PandRSError::Consistency(format!(
                        "Resampled bucket start {period_start_seconds} is outside the \
                         representable timestamp range"
                    )))
                }
            };

            result_timestamps.push(T::from_str(&period_time.to_rfc3339())?);
        }

        TimeSeries::new(
            result_values,
            result_timestamps,
            self.series.name().cloned(),
        )
        .map(|ts| ts.with_frequency(self.frequency.clone()))
    }
}

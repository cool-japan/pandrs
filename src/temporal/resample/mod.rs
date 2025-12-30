use chrono::TimeZone;
use std::collections::HashMap;

use crate::error::Result;
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

    /// Resample using mean
    pub fn mean(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            if values.is_empty() {
                return 0.0;
            }
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
        self.aggregate(|values| {
            if values.is_empty() {
                return f64::NAN;
            }
            let mut max = values[0];
            for &value in &values[1..] {
                if value > max {
                    max = value;
                }
            }
            max
        })
    }

    /// Resample using minimum
    pub fn min(&self) -> Result<TimeSeries<T>> {
        self.aggregate(|values| {
            if values.is_empty() {
                return f64::NAN;
            }
            let mut min = values[0];
            for &value in &values[1..] {
                if value < min {
                    min = value;
                }
            }
            min
        })
    }

    /// Resample using a custom aggregation function
    pub fn aggregate<F>(&self, aggregator: F) -> Result<TimeSeries<T>>
    where
        F: Fn(Vec<f64>) -> f64,
    {
        // Group by period
        let mut period_groups: HashMap<i64, Vec<f64>> = HashMap::new();
        let freq_seconds = self.frequency.to_seconds();

        // Assign each data point to the appropriate period
        let start_time = self.series.timestamps()[0].to_utc();
        let start_seconds = start_time.timestamp();

        for (i, timestamp) in self.series.timestamps().iter().enumerate() {
            if let Some(value) = match self.series.values()[i] {
                crate::na::NA::Value(v) => Some(v),
                crate::na::NA::NA => None,
            } {
                // Calculate which period it belongs to
                let ts_seconds = timestamp.to_utc().timestamp();
                let offset = ts_seconds - start_seconds;
                let period = offset / freq_seconds;

                // Add to that period's group
                period_groups
                    .entry(period)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Sort periods in chronological order
        let mut periods: Vec<i64> = period_groups.keys().cloned().collect();
        periods.sort();

        // Create aggregation results
        let mut result_values = Vec::with_capacity(periods.len());
        let mut result_timestamps = Vec::with_capacity(periods.len());

        for period in periods {
            // Aggregate data for this period
            if let Some(values) = period_groups.get(&period) {
                let agg_value = aggregator(values.clone());
                result_values.push(crate::na::NA::Value(agg_value));

                // Calculate the representative time for this period
                let period_start_seconds = start_seconds + period * freq_seconds;
                let period_time = chrono::Utc.timestamp_opt(period_start_seconds, 0).unwrap();

                // Convert to the appropriate type
                let period_timestamp = T::from_str(&period_time.to_rfc3339())?;
                result_timestamps.push(period_timestamp);
            }
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

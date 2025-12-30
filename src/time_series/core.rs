//! Core Time Series Data Structures and Operations
//!
//! This module provides the fundamental time series data structures and basic operations
//! for temporal data analysis.

use crate::core::error::{Error, Result};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Simple data structure for time series values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    values: Vec<f64>,
}

impl TimeSeriesData {
    pub fn from_vec(values: Vec<f64>) -> Self {
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get_f64(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start >= self.values.len() || end > self.values.len() || start >= end {
            return Err(Error::InvalidInput("Invalid slice bounds".to_string()));
        }
        Ok(Self::from_vec(self.values[start..end].to_vec()))
    }

    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().copied()
    }
}

/// Time series frequency specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Frequency {
    /// Secondly data
    Second,
    /// Minutely data
    Minute,
    /// Hourly data
    Hour,
    /// Daily data
    Daily,
    /// Weekly data
    Weekly,
    /// Monthly data
    Monthly,
    /// Quarterly data
    Quarterly,
    /// Yearly data
    Yearly,
    /// Custom frequency with duration
    Custom(Duration),
}

impl Frequency {
    /// Get the duration for this frequency
    pub fn to_duration(&self) -> Duration {
        match self {
            Frequency::Second => Duration::seconds(1),
            Frequency::Minute => Duration::minutes(1),
            Frequency::Hour => Duration::hours(1),
            Frequency::Daily => Duration::days(1),
            Frequency::Weekly => Duration::weeks(1),
            Frequency::Monthly => Duration::days(30), // Approximate
            Frequency::Quarterly => Duration::days(90), // Approximate
            Frequency::Yearly => Duration::days(365), // Approximate
            Frequency::Custom(duration) => *duration,
        }
    }

    /// Get frequency name as string
    pub fn name(&self) -> &'static str {
        match self {
            Frequency::Second => "S",
            Frequency::Minute => "T",
            Frequency::Hour => "H",
            Frequency::Daily => "D",
            Frequency::Weekly => "W",
            Frequency::Monthly => "M",
            Frequency::Quarterly => "Q",
            Frequency::Yearly => "Y",
            Frequency::Custom(_) => "C",
        }
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// DateTime index for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateTimeIndex {
    /// Datetime values
    pub values: Vec<DateTime<Utc>>,
    /// Frequency (if regular)
    pub frequency: Option<Frequency>,
    /// Name of the index
    pub name: Option<String>,
}

impl DateTimeIndex {
    /// Create a new datetime index
    pub fn new(values: Vec<DateTime<Utc>>) -> Self {
        let frequency = Self::infer_frequency(&values);
        Self {
            values,
            frequency,
            name: None,
        }
    }

    /// Create with specified frequency
    pub fn with_frequency(values: Vec<DateTime<Utc>>, frequency: Frequency) -> Self {
        Self {
            values,
            frequency: Some(frequency),
            name: None,
        }
    }

    /// Create a date range
    pub fn date_range(
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        frequency: Frequency,
    ) -> Result<Self> {
        let mut dates = Vec::new();
        let mut current = start;
        let duration = frequency.to_duration();

        while current <= end {
            dates.push(current);
            current = current + duration;
        }

        if dates.is_empty() {
            return Err(Error::InvalidInput("Invalid date range".to_string()));
        }

        Ok(Self {
            values: dates,
            frequency: Some(frequency),
            name: None,
        })
    }

    /// Infer frequency from datetime values
    fn infer_frequency(values: &[DateTime<Utc>]) -> Option<Frequency> {
        if values.len() < 2 {
            return None;
        }

        let diff = values[1] - values[0];

        // Check if all differences are the same
        for i in 2..values.len() {
            if (values[i] - values[i - 1]) != diff {
                return None; // Irregular frequency
            }
        }

        // Determine frequency based on difference
        match diff.num_seconds() {
            1 => Some(Frequency::Second),
            60 => Some(Frequency::Minute),
            3600 => Some(Frequency::Hour),
            86400 => Some(Frequency::Daily),
            604800 => Some(Frequency::Weekly),
            _ => Some(Frequency::Custom(diff)),
        }
    }

    /// Get length of index
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value at index
    pub fn get(&self, index: usize) -> Option<&DateTime<Utc>> {
        self.values.get(index)
    }

    /// Check if index is regular (has consistent frequency)
    pub fn is_regular(&self) -> bool {
        self.frequency.is_some()
    }

    /// Get start date
    pub fn start(&self) -> Option<&DateTime<Utc>> {
        self.values.first()
    }

    /// Get end date
    pub fn end(&self) -> Option<&DateTime<Utc>> {
        self.values.last()
    }

    /// Slice the index
    pub fn slice(&self, start: usize, end: usize) -> Result<Self> {
        if start >= self.values.len() || end > self.values.len() || start >= end {
            return Err(Error::InvalidInput("Invalid slice bounds".to_string()));
        }

        Ok(Self {
            values: self.values[start..end].to_vec(),
            frequency: self.frequency.clone(),
            name: self.name.clone(),
        })
    }
}

/// A single time point with value and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimePoint<T> {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Value at this time point
    pub value: T,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl<T> TimePoint<T> {
    /// Create a new time point
    pub fn new(timestamp: DateTime<Utc>, value: T) -> Self {
        Self {
            timestamp,
            value,
            metadata: HashMap::new(),
        }
    }

    /// Create with metadata
    pub fn with_metadata(
        timestamp: DateTime<Utc>,
        value: T,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            timestamp,
            value,
            metadata,
        }
    }
}

/// Main time series data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// DateTime index
    pub index: DateTimeIndex,
    /// Values data
    pub values: TimeSeriesData,
    /// Name of the time series
    pub name: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(index: DateTimeIndex, values: TimeSeriesData) -> Result<Self> {
        if index.len() != values.len() {
            return Err(Error::DimensionMismatch(
                "Index and values must have the same length".to_string(),
            ));
        }

        Ok(Self {
            index,
            values,
            name: None,
            metadata: HashMap::new(),
        })
    }

    /// Create from vectors
    pub fn from_vecs(timestamps: Vec<DateTime<Utc>>, values: Vec<f64>) -> Result<Self> {
        let index = DateTimeIndex::new(timestamps);
        let data = TimeSeriesData::from_vec(values);
        Self::new(index, data)
    }

    /// Create from time points
    pub fn from_points(points: Vec<TimePoint<f64>>) -> Result<Self> {
        let timestamps: Vec<DateTime<Utc>> = points.iter().map(|p| p.timestamp).collect();
        let values: Vec<f64> = points.into_iter().map(|p| p.value).collect();
        Self::from_vecs(timestamps, values)
    }

    /// Get length of time series
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if time series is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get value at index
    pub fn get(&self, index: usize) -> Option<(&DateTime<Utc>, f64)> {
        match (self.index.get(index), self.values.get_f64(index)) {
            (Some(ts), Some(val)) => Some((ts, val)),
            _ => None,
        }
    }

    /// Get value by timestamp (exact match)
    pub fn at(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        self.index
            .values
            .iter()
            .position(|ts| ts == timestamp)
            .and_then(|idx| self.values.get_f64(idx))
    }

    /// Get values within time range
    pub fn between(&self, start: &DateTime<Utc>, end: &DateTime<Utc>) -> Result<TimeSeries> {
        let start_idx = self
            .index
            .values
            .iter()
            .position(|ts| ts >= start)
            .unwrap_or(0);

        let end_idx = self
            .index
            .values
            .iter()
            .rposition(|ts| ts <= end)
            .map(|idx| idx + 1)
            .unwrap_or(self.len());

        if start_idx >= end_idx {
            return Err(Error::InvalidInput("Invalid time range".to_string()));
        }

        let index = self.index.slice(start_idx, end_idx)?;
        let values = self.values.slice(start_idx, end_idx)?;

        Ok(TimeSeries {
            index,
            values,
            name: self.name.clone(),
            metadata: self.metadata.clone(),
        })
    }

    /// Slice time series by index positions
    pub fn slice(&self, start: usize, end: usize) -> Result<TimeSeries> {
        let index = self.index.slice(start, end)?;
        let values = self.values.slice(start, end)?;

        Ok(TimeSeries {
            index,
            values,
            name: self.name.clone(),
            metadata: self.metadata.clone(),
        })
    }

    /// Resample time series to new frequency
    pub fn resample(&self, frequency: Frequency, method: ResampleMethod) -> Result<TimeSeries> {
        if self.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot resample empty time series".to_string(),
            ));
        }

        let start = *self.index.start().unwrap();
        let end = *self.index.end().unwrap();
        let new_index = DateTimeIndex::date_range(start, end, frequency)?;

        let mut new_values = Vec::new();

        for new_timestamp in &new_index.values {
            let value = match method {
                ResampleMethod::Mean => self.interpolate_mean(new_timestamp),
                ResampleMethod::Linear => self.interpolate_linear(new_timestamp),
                ResampleMethod::Nearest => self.interpolate_nearest(new_timestamp),
                ResampleMethod::Forward => self.forward_fill(new_timestamp),
                ResampleMethod::Backward => self.backward_fill(new_timestamp),
            };

            new_values.push(value.unwrap_or(f64::NAN));
        }

        let new_series = TimeSeriesData::from_vec(new_values);
        TimeSeries::new(new_index, new_series)
    }

    /// Linear interpolation at timestamp
    fn interpolate_linear(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        // Find surrounding points
        let mut before_idx = None;
        let mut after_idx = None;

        for (i, ts) in self.index.values.iter().enumerate() {
            if ts <= timestamp {
                before_idx = Some(i);
            } else {
                after_idx = Some(i);
                break;
            }
        }

        match (before_idx, after_idx) {
            (Some(before), Some(after)) => {
                let ts_before = &self.index.values[before];
                let ts_after = &self.index.values[after];
                let val_before = self.values.get_f64(before)?;
                let val_after = self.values.get_f64(after)?;

                let total_duration = ts_after.signed_duration_since(*ts_before);
                let elapsed_duration = timestamp.signed_duration_since(*ts_before);

                if total_duration.num_seconds() == 0 {
                    return Some(val_before);
                }

                let ratio =
                    elapsed_duration.num_seconds() as f64 / total_duration.num_seconds() as f64;
                Some(val_before + (val_after - val_before) * ratio)
            }
            (Some(idx), None) => self.values.get_f64(idx), // Use last value
            (None, Some(idx)) => self.values.get_f64(idx), // Use first value
            _ => None,
        }
    }

    /// Mean interpolation (aggregate surrounding values)
    fn interpolate_mean(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        let window = Duration::minutes(30); // Default window
        let start = *timestamp - window;
        let end = *timestamp + window;

        let values: Vec<f64> = self
            .index
            .values
            .iter()
            .zip(self.values.iter())
            .filter(|(ts, _)| **ts >= start && **ts <= end)
            .map(|(_, val)| val)
            .collect();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    /// Nearest neighbor interpolation
    fn interpolate_nearest(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        let mut closest_idx = 0;
        let mut min_diff = Duration::MAX;

        for (i, ts) in self.index.values.iter().enumerate() {
            let diff = (*timestamp - *ts).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_idx = i;
            }
        }

        self.values.get_f64(closest_idx)
    }

    /// Forward fill
    fn forward_fill(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        for (i, ts) in self.index.values.iter().enumerate().rev() {
            if ts <= timestamp {
                return self.values.get_f64(i);
            }
        }
        None
    }

    /// Backward fill
    fn backward_fill(&self, timestamp: &DateTime<Utc>) -> Option<f64> {
        for (i, ts) in self.index.values.iter().enumerate() {
            if ts >= timestamp {
                return self.values.get_f64(i);
            }
        }
        None
    }

    /// Calculate rolling window statistics
    pub fn rolling_mean(&self, window: usize) -> Result<TimeSeries> {
        if window == 0 || window > self.len() {
            return Err(Error::InvalidInput("Invalid window size".to_string()));
        }

        let mut rolling_values = Vec::new();

        for i in 0..self.len() {
            if i + 1 < window {
                rolling_values.push(f64::NAN);
            } else {
                let start_idx = i + 1 - window;
                let window_sum: f64 = (start_idx..=i)
                    .filter_map(|idx| self.values.get_f64(idx))
                    .sum();
                rolling_values.push(window_sum / window as f64);
            }
        }

        let new_series = TimeSeriesData::from_vec(rolling_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Calculate rolling standard deviation
    pub fn rolling_std(&self, window: usize) -> Result<TimeSeries> {
        if window == 0 || window > self.len() {
            return Err(Error::InvalidInput("Invalid window size".to_string()));
        }

        let mut rolling_values = Vec::new();

        for i in 0..self.len() {
            if i + 1 < window {
                rolling_values.push(f64::NAN);
            } else {
                let start_idx = i + 1 - window;
                let window_values: Vec<f64> = (start_idx..=i)
                    .filter_map(|idx| self.values.get_f64(idx))
                    .collect();

                if window_values.len() == window {
                    let mean = window_values.iter().sum::<f64>() / window as f64;
                    let variance = window_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / window as f64;
                    rolling_values.push(variance.sqrt());
                } else {
                    rolling_values.push(f64::NAN);
                }
            }
        }

        let new_series = TimeSeriesData::from_vec(rolling_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Calculate differences (lag = 1 by default)
    pub fn diff(&self, periods: usize) -> Result<TimeSeries> {
        if periods >= self.len() {
            return Err(Error::InvalidInput(
                "Periods must be less than series length".to_string(),
            ));
        }

        let mut diff_values = vec![f64::NAN; periods];

        for i in periods..self.len() {
            if let (Some(current), Some(prev)) =
                (self.values.get_f64(i), self.values.get_f64(i - periods))
            {
                diff_values.push(current - prev);
            } else {
                diff_values.push(f64::NAN);
            }
        }

        let new_series = TimeSeriesData::from_vec(diff_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Calculate percentage change
    pub fn pct_change(&self, periods: usize) -> Result<TimeSeries> {
        if periods >= self.len() {
            return Err(Error::InvalidInput(
                "Periods must be less than series length".to_string(),
            ));
        }

        let mut pct_values = vec![f64::NAN; periods];

        for i in periods..self.len() {
            if let (Some(current), Some(prev)) =
                (self.values.get_f64(i), self.values.get_f64(i - periods))
            {
                if prev != 0.0 {
                    pct_values.push((current - prev) / prev);
                } else {
                    pct_values.push(f64::NAN);
                }
            } else {
                pct_values.push(f64::NAN);
            }
        }

        let new_series = TimeSeriesData::from_vec(pct_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Shift values by periods
    pub fn shift(&self, periods: i32) -> Result<TimeSeries> {
        let mut shifted_values = Vec::with_capacity(self.len());

        if periods > 0 {
            // Shift forward (add NaNs at beginning)
            for _ in 0..periods as usize {
                shifted_values.push(f64::NAN);
            }
            for i in 0..(self.len() - periods as usize) {
                shifted_values.push(self.values.get_f64(i).unwrap_or(f64::NAN));
            }
        } else if periods < 0 {
            // Shift backward (add NaNs at end)
            let abs_periods = (-periods) as usize;
            for i in abs_periods..self.len() {
                shifted_values.push(self.values.get_f64(i).unwrap_or(f64::NAN));
            }
            for _ in 0..abs_periods {
                shifted_values.push(f64::NAN);
            }
        } else {
            // No shift
            for i in 0..self.len() {
                shifted_values.push(self.values.get_f64(i).unwrap_or(f64::NAN));
            }
        }

        let new_series = TimeSeriesData::from_vec(shifted_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Fill missing values forward
    pub fn fillna_forward(&self) -> Result<TimeSeries> {
        let mut filled_values = Vec::new();
        let mut last_valid = None;

        for i in 0..self.len() {
            if let Some(val) = self.values.get_f64(i) {
                if val.is_finite() {
                    last_valid = Some(val);
                    filled_values.push(val);
                } else if let Some(last) = last_valid {
                    filled_values.push(last);
                } else {
                    filled_values.push(val);
                }
            } else if let Some(last) = last_valid {
                filled_values.push(last);
            } else {
                filled_values.push(f64::NAN);
            }
        }

        let new_series = TimeSeriesData::from_vec(filled_values);
        TimeSeries::new(self.index.clone(), new_series)
    }

    /// Fill missing values backward
    pub fn fillna_backward(&self) -> Result<TimeSeries> {
        let mut filled_values = vec![f64::NAN; self.len()];
        let mut next_valid = None;

        for i in (0..self.len()).rev() {
            if let Some(val) = self.values.get_f64(i) {
                if val.is_finite() {
                    next_valid = Some(val);
                    filled_values[i] = val;
                } else if let Some(next) = next_valid {
                    filled_values[i] = next;
                } else {
                    filled_values[i] = val;
                }
            } else if let Some(next) = next_valid {
                filled_values[i] = next;
            }
        }

        let new_series = TimeSeriesData::from_vec(filled_values);
        TimeSeries::new(self.index.clone(), new_series)
    }
}

/// Resampling methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResampleMethod {
    /// Mean of surrounding values
    Mean,
    /// Linear interpolation
    Linear,
    /// Nearest neighbor
    Nearest,
    /// Forward fill
    Forward,
    /// Backward fill
    Backward,
}

/// Time series builder for convenient construction
pub struct TimeSeriesBuilder {
    timestamps: Vec<DateTime<Utc>>,
    values: Vec<f64>,
    name: Option<String>,
    frequency: Option<Frequency>,
    metadata: HashMap<String, String>,
}

impl TimeSeriesBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            timestamps: Vec::new(),
            values: Vec::new(),
            name: None,
            frequency: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a data point
    pub fn add_point(mut self, timestamp: DateTime<Utc>, value: f64) -> Self {
        self.timestamps.push(timestamp);
        self.values.push(value);
        self
    }

    /// Set name
    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Set frequency
    pub fn frequency(mut self, frequency: Frequency) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Build the time series
    pub fn build(self) -> Result<TimeSeries> {
        if self.timestamps.len() != self.values.len() {
            return Err(Error::DimensionMismatch(
                "Timestamps and values must have the same length".to_string(),
            ));
        }

        let index = if let Some(freq) = self.frequency {
            DateTimeIndex::with_frequency(self.timestamps, freq)
        } else {
            DateTimeIndex::new(self.timestamps)
        };

        let series = TimeSeriesData::from_vec(self.values);
        let mut ts = TimeSeries::new(index, series)?;
        ts.name = self.name;
        ts.metadata = self.metadata;

        Ok(ts)
    }
}

impl Default for TimeSeriesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_series() -> TimeSeries {
        let timestamps = vec![
            Utc.timestamp_opt(1640995200, 0).unwrap(), // 2022-01-01
            Utc.timestamp_opt(1641081600, 0).unwrap(), // 2022-01-02
            Utc.timestamp_opt(1641168000, 0).unwrap(), // 2022-01-03
            Utc.timestamp_opt(1641254400, 0).unwrap(), // 2022-01-04
            Utc.timestamp_opt(1641340800, 0).unwrap(), // 2022-01-05
        ];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        TimeSeries::from_vecs(timestamps, values).unwrap()
    }

    #[test]
    fn test_time_series_creation() {
        let ts = create_test_series();
        assert_eq!(ts.len(), 5);
        assert!(!ts.is_empty());
        assert_eq!(ts.get(0).unwrap().1, 1.0);
    }

    #[test]
    fn test_frequency_inference() {
        let timestamps = vec![
            Utc.timestamp_opt(1640995200, 0).unwrap(),
            Utc.timestamp_opt(1641081600, 0).unwrap(),
            Utc.timestamp_opt(1641168000, 0).unwrap(),
        ];
        let index = DateTimeIndex::new(timestamps);
        assert_eq!(index.frequency, Some(Frequency::Daily));
    }

    #[test]
    fn test_rolling_mean() {
        let ts = create_test_series();
        let rolling = ts.rolling_mean(2).unwrap();

        assert!(rolling.values.get_f64(0).unwrap().is_nan());
        assert_eq!(rolling.values.get_f64(1).unwrap(), 1.5); // (1+2)/2
        assert_eq!(rolling.values.get_f64(2).unwrap(), 2.5); // (2+3)/2
    }

    #[test]
    fn test_diff() {
        let ts = create_test_series();
        let diff = ts.diff(1).unwrap();

        assert!(diff.values.get_f64(0).unwrap().is_nan());
        assert_eq!(diff.values.get_f64(1).unwrap(), 1.0); // 2-1
        assert_eq!(diff.values.get_f64(2).unwrap(), 1.0); // 3-2
    }

    #[test]
    fn test_time_series_builder() {
        let ts = TimeSeriesBuilder::new()
            .add_point(Utc.timestamp_opt(1640995200, 0).unwrap(), 1.0)
            .add_point(Utc.timestamp_opt(1641081600, 0).unwrap(), 2.0)
            .name("test_series".to_string())
            .frequency(Frequency::Daily)
            .metadata("source".to_string(), "test".to_string())
            .build()
            .unwrap();

        assert_eq!(ts.len(), 2);
        assert_eq!(ts.name, Some("test_series".to_string()));
        assert_eq!(ts.index.frequency, Some(Frequency::Daily));
    }

    #[test]
    fn test_slice() {
        let ts = create_test_series();
        let sliced = ts.slice(1, 4).unwrap();

        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.values.get_f64(0).unwrap(), 2.0);
        assert_eq!(sliced.values.get_f64(2).unwrap(), 4.0);
    }

    #[test]
    fn test_shift() {
        let ts = create_test_series();
        let shifted = ts.shift(1).unwrap();

        assert!(shifted.values.get_f64(0).unwrap().is_nan());
        assert_eq!(shifted.values.get_f64(1).unwrap(), 1.0);
        assert_eq!(shifted.values.get_f64(2).unwrap(), 2.0);
    }
}

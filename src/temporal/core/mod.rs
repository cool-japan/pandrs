//! Core functionality for temporal operations
//!
//! This module provides the fundamental types and traits for time series data processing.

use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use std::ops::{Add, Sub};

use crate::error::{PandRSError, Result};
use crate::na::NA;
use crate::temporal::frequency::Frequency;

/// Trait for representing date-time types
pub trait Temporal:
    Clone
    + std::fmt::Debug
    + PartialOrd
    + Add<Duration, Output = Self>
    + Sub<Duration, Output = Self>
    + 'static
{
    /// Get current time
    fn now() -> Self;

    /// Get duration between two times
    fn duration_between(&self, other: &Self) -> Duration;

    /// Convert to UTC timezone
    fn to_utc(&self) -> DateTime<Utc>;

    /// Convert from string
    fn from_str(s: &str) -> Result<Self>;

    /// Convert to string
    fn to_string(&self) -> String;
}

// Implementation of Temporal trait for various Chrono date-time types

impl Temporal for DateTime<Utc> {
    fn now() -> Self {
        Utc::now()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        *self
    }

    fn from_str(s: &str) -> Result<Self> {
        match s.parse::<DateTime<Utc>>() {
            Ok(dt) => Ok(dt),
            Err(e) => Err(PandRSError::Format(format!(
                "Date-time parsing error: {}",
                e
            ))),
        }
    }

    fn to_string(&self) -> String {
        self.to_rfc3339()
    }
}

impl Temporal for DateTime<Local> {
    fn now() -> Self {
        Local::now()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        self.with_timezone(&Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        match s.parse::<DateTime<Local>>() {
            Ok(dt) => Ok(dt),
            Err(e) => Err(PandRSError::Format(format!(
                "Date-time parsing error: {}",
                e
            ))),
        }
    }

    fn to_string(&self) -> String {
        self.to_rfc3339()
    }
}

impl Temporal for NaiveDateTime {
    fn now() -> Self {
        Local::now().naive_local()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        if self > other {
            *self - *other
        } else {
            *other - *self
        }
    }

    fn to_utc(&self) -> DateTime<Utc> {
        // NaiveDateTime doesn't have timezone info, so we assume UTC
        DateTime::<Utc>::from_naive_utc_and_offset(*self, Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        match NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            Ok(dt) => Ok(dt),
            Err(_) => match NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
                Ok(dt) => Ok(dt),
                Err(e) => Err(PandRSError::Format(format!(
                    "Date-time parsing error: {}",
                    e
                ))),
            },
        }
    }

    fn to_string(&self) -> String {
        self.format("%Y-%m-%d %H:%M:%S").to_string()
    }
}

impl Temporal for NaiveDate {
    fn now() -> Self {
        Local::now().date_naive()
    }

    fn duration_between(&self, other: &Self) -> Duration {
        let days = if self > other {
            (*self - *other).num_days()
        } else {
            (*other - *self).num_days()
        };
        Duration::days(days)
    }

    fn to_utc(&self) -> DateTime<Utc> {
        // Add default time (00:00:00) to the date and treat as UTC
        let naive_dt = self.and_time(NaiveTime::from_hms_opt(0, 0, 0).unwrap());
        DateTime::<Utc>::from_naive_utc_and_offset(naive_dt, Utc)
    }

    fn from_str(s: &str) -> Result<Self> {
        // Try parsing with standard format first
        match NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            Ok(dt) => Ok(dt),
            Err(_) => {
                // Try parsing from RFC3339 format (from DateTime)
                match chrono::DateTime::parse_from_rfc3339(s) {
                    Ok(dt) => Ok(dt.date_naive()),
                    Err(e) => Err(PandRSError::Format(format!("Date parsing error: {}", e))),
                }
            }
        }
    }

    fn to_string(&self) -> String {
        self.format("%Y-%m-%d").to_string()
    }
}

/// Structure representing a time series
#[derive(Debug, Clone)]
pub struct TimeSeries<T: Temporal> {
    /// Values of the time series data
    values: Vec<NA<f64>>,

    /// Time index of the series
    timestamps: Vec<T>,

    /// Name of the series
    name: Option<String>,

    /// Frequency (optional)
    frequency: Option<Frequency>,
}

impl<T: Temporal> TimeSeries<T> {
    /// Create a new time series
    pub fn new(values: Vec<NA<f64>>, timestamps: Vec<T>, name: Option<String>) -> Result<Self> {
        if values.len() != timestamps.len() {
            return Err(PandRSError::Consistency(format!(
                "Length of values ({}) does not match length of time index ({})",
                values.len(),
                timestamps.len()
            )));
        }

        Ok(TimeSeries {
            values,
            timestamps,
            name,
            frequency: None,
        })
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get name
    pub fn name(&self) -> Option<&String> {
        self.name.as_ref()
    }

    /// Get timestamps
    pub fn timestamps(&self) -> &[T] {
        &self.timestamps
    }

    /// Get values
    pub fn values(&self) -> &[NA<f64>] {
        &self.values
    }

    /// Get frequency
    pub fn frequency(&self) -> Option<&Frequency> {
        self.frequency.as_ref()
    }

    /// Set frequency
    pub fn with_frequency(mut self, freq: Frequency) -> Self {
        self.frequency = Some(freq);
        self
    }

    /// Filter by time range
    pub fn filter_by_time(&self, start: &T, end: &T) -> Result<Self> {
        let mut filtered_values = Vec::new();
        let mut filtered_timestamps = Vec::new();

        for (i, ts) in self.timestamps.iter().enumerate() {
            if ts >= start && ts <= end {
                filtered_values.push(self.values[i].clone());
                filtered_timestamps.push(ts.clone());
            }
        }

        Self::new(filtered_values, filtered_timestamps, self.name.clone())
    }

    /// Resample with specified frequency
    pub fn resample(&self, freq: Frequency) -> crate::temporal::resample::Resample<T> {
        crate::temporal::resample::Resample::new(self, freq)
    }

    /// Calculate moving average
    ///
    /// Note: This function is maintained for compatibility,
    /// but the more feature-rich `rolling()` method is recommended.
    pub fn rolling_mean(&self, window: usize) -> Result<Self> {
        if window > self.len() || window == 0 {
            return Err(PandRSError::Consistency(format!(
                "Invalid window size ({}). Must be greater than 0 and less than or equal to the data length ({}).",
                window, self.len()
            )));
        }

        // Implemented using the new Window API
        self.rolling(window)?.mean()
    }
}

// Define utilities for date calculations
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => panic!("Invalid month: {}", month),
    }
}

/// Check if a year is a leap year
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

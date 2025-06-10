//! Module for generating date ranges
//!
//! This module provides functionality for creating ranges of dates
//! with specified frequencies.

use crate::error::{PandRSError, Result};
use crate::temporal::core::{days_in_month, is_leap_year, Temporal};
use crate::temporal::frequency::Frequency;
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, Utc};

/// Structure to generate a date range
#[derive(Debug, Clone)]
pub struct DateRange<T: Temporal> {
    start: T,
    end: T,
    freq: Frequency,
    inclusive: bool,
}

impl<T: Temporal> DateRange<T> {
    /// Create a date range from start, end, and frequency
    pub fn new(start: T, end: T, freq: Frequency, inclusive: bool) -> Result<Self> {
        if start > end {
            return Err(PandRSError::Consistency(
                "Start date must be earlier than end date".to_string(),
            ));
        }

        Ok(DateRange {
            start,
            end,
            freq,
            inclusive,
        })
    }

    /// Get all points in the date range
    pub fn generate(&self) -> Vec<T> {
        let mut result = Vec::new();
        let mut current = self.start.clone();

        // Add the first date
        result.push(current.clone());

        loop {
            // Move to the next date
            current = match self.freq {
                Frequency::Secondly => current.add(Duration::seconds(1)),
                Frequency::Minutely => current.add(Duration::minutes(1)),
                Frequency::Hourly => current.add(Duration::hours(1)),
                Frequency::Daily => current.add(Duration::days(1)),
                Frequency::Weekly => current.add(Duration::weeks(1)),
                Frequency::Monthly => {
                    // Converting to UTC before operating since adding months is complex
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();

                    // Calculate year and month
                    let mut year = naive.year();
                    let mut month = naive.month() + 1;

                    if month > 12 {
                        month = 1;
                        year += 1;
                    }

                    // Calculate the new day (adjust to not exceed the last day of the month)
                    let day = naive.day().min(days_in_month(year, month));

                    // Create the new date
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time(),
                    );

                    // Convert back to the original type
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Quarterly => {
                    // Add quarters (every 3 months)
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();

                    // Calculate year and month
                    let mut year = naive.year();
                    let mut month = naive.month() + 3;

                    if month > 12 {
                        month = month - 12;
                        year += 1;
                    }

                    // Calculate the new day (adjust to not exceed the last day of the month)
                    let day = naive.day().min(days_in_month(year, month));

                    // Create the new date
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time(),
                    );

                    // Convert back to the original type
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Yearly => {
                    // Add years
                    let utc = current.to_utc();
                    let naive = utc.naive_utc();

                    // Increment the year
                    let year = naive.year() + 1i32;
                    let month = naive.month();

                    // Adjust February 29 to February 28 if not a leap year
                    let day = if month == 2 && naive.day() == 29 && !is_leap_year(year as i32) {
                        28
                    } else {
                        naive.day()
                    };

                    // Create the new date
                    let new_naive = NaiveDateTime::new(
                        NaiveDate::from_ymd_opt(year, month, day).unwrap(),
                        naive.time(),
                    );

                    // Convert back to the original type
                    let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
                    T::from_str(&new_utc.to_rfc3339()).unwrap()
                }
                Frequency::Custom(duration) => current.add(duration),
            };

            // Check end condition
            if self.inclusive {
                if current > self.end {
                    break;
                }
            } else if current >= self.end {
                break;
            }

            result.push(current.clone());
        }

        result
    }
}

/// Helper function to generate a date range
pub fn date_range<T: Temporal>(
    start: T,
    end: T,
    freq: Frequency,
    inclusive: bool,
) -> Result<Vec<T>> {
    DateRange::new(start, end, freq, inclusive).map(|range| range.generate())
}

//! Module for generating date ranges
//!
//! This module provides functionality for creating ranges of dates
//! with specified frequencies.

use crate::error::{PandRSError, Result};
use crate::temporal::core::{days_in_month, is_leap_year, normalize_year_month, Temporal};
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

    /// Get all points in the date range.
    ///
    /// Calendar steps (monthly, quarterly, yearly) **clamp to the end of the
    /// target month**, matching pandas: 2024-01-31 plus one month is
    /// 2024-02-29, not the non-existent 2024-02-31. Month arithmetic wraps
    /// through `normalize_year_month` rather than assuming the incremented
    /// month is in range.
    ///
    /// # Errors
    /// Returns [`PandRSError::Consistency`] if a step lands outside the
    /// representable calendar range or the resulting timestamp cannot be parsed
    /// back into `T`. Both were `.expect(...)` panics before — a library
    /// generating a range is not entitled to abort the process because a caller
    /// asked for a date near the year-`i32` boundary.
    pub fn generate(&self) -> Result<Vec<T>> {
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
                Frequency::Monthly => Self::add_months(&current, 1)?,
                Frequency::Quarterly => Self::add_months(&current, 3)?,
                Frequency::Yearly => Self::add_years(&current)?,
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

        Ok(result)
    }

    /// Advance by `months` calendar months, clamping the day to the last day of
    /// the destination month (pandas' `DateOffset(months=n)` semantics).
    fn add_months(current: &T, months: i64) -> Result<T> {
        let naive = current.to_utc().naive_utc();

        let (year, month) = normalize_year_month(naive.year(), i64::from(naive.month()) + months);
        let day = naive.day().min(days_in_month(year, month));

        Self::rebuild(naive, year, month, day)
    }

    /// Advance by one calendar year, mapping February 29 onto February 28 in
    /// non-leap destination years.
    fn add_years(current: &T) -> Result<T> {
        let naive = current.to_utc().naive_utc();

        let year = naive.year().saturating_add(1);
        let month = naive.month();
        let day = if month == 2 && naive.day() == 29 && !is_leap_year(year) {
            28
        } else {
            naive.day()
        };

        Self::rebuild(naive, year, month, day)
    }

    /// Rebuild a `T` from a calendar date plus the original time of day.
    fn rebuild(naive: NaiveDateTime, year: i32, month: u32, day: u32) -> Result<T> {
        let date = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
            PandRSError::Consistency(format!(
                "Date {year:04}-{month:02}-{day:02} is outside the representable calendar range"
            ))
        })?;

        let new_naive = NaiveDateTime::new(date, naive.time());
        let new_utc = DateTime::<Utc>::from_naive_utc_and_offset(new_naive, Utc);
        T::from_str(&new_utc.to_rfc3339())
    }
}

/// Helper function to generate a date range
pub fn date_range<T: Temporal>(
    start: T,
    end: T,
    freq: Frequency,
    inclusive: bool,
) -> Result<Vec<T>> {
    DateRange::new(start, end, freq, inclusive)?.generate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn day(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("valid test date")
    }

    #[test]
    fn monthly_range_clamps_to_month_end() {
        // Jan 31 + 1 month is Feb 29 (2024 is a leap year), not "Feb 31".
        let dates = date_range(day(2024, 1, 31), day(2024, 5, 31), Frequency::Monthly, true)
            .expect("monthly range must not panic on a month-end start");

        assert_eq!(dates[0], day(2024, 1, 31));
        assert_eq!(dates[1], day(2024, 2, 29));
        assert_eq!(dates[2], day(2024, 3, 29));
    }

    #[test]
    fn quarterly_range_clamps_to_month_end() {
        let dates = date_range(
            day(2023, 11, 30),
            day(2024, 9, 30),
            Frequency::Quarterly,
            true,
        )
        .expect("quarterly range must not panic when the month wraps the year");

        assert_eq!(dates[0], day(2023, 11, 30));
        assert_eq!(dates[1], day(2024, 2, 29));
        assert_eq!(dates[2], day(2024, 5, 29));
    }

    #[test]
    fn yearly_range_maps_leap_day_to_feb_28() {
        let dates = date_range(day(2024, 2, 29), day(2026, 3, 1), Frequency::Yearly, true)
            .expect("yearly range must not panic on a leap day");

        assert_eq!(dates[0], day(2024, 2, 29));
        assert_eq!(dates[1], day(2025, 2, 28));
    }
}

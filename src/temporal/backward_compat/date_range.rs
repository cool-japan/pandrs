use crate::error::{PandRSError, Result};
use crate::temporal::{Frequency, Temporal};
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
    /// Mirrors `temporal::date_range::DateRange::generate`: calendar steps clamp
    /// to the end of the destination month (pandas semantics), month arithmetic
    /// wraps through [`normalize_year_month`], and out-of-range dates are
    /// reported as errors instead of `.expect(...)` panics.
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
    /// the destination month.
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

/// Return the number of days in the specified year and month.
///
/// Months outside `1..=12` wrap into range, carrying the excess into the year,
/// instead of panicking.
fn days_in_month(year: i32, month: u32) -> u32 {
    let (year, month) = normalize_year_month(year, i64::from(month));
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        // Only February remains; normalization guarantees `1..=12`.
        _ => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
    }
}

/// Normalize `(year, month)` so that `month` lands in `1..=12`.
fn normalize_year_month(year: i32, month: i64) -> (i32, u32) {
    let zero_based = month - 1;
    let year_offset = zero_based.div_euclid(12);
    let normalized_month = zero_based.rem_euclid(12) + 1;
    let normalized_year = i64::from(year)
        .saturating_add(year_offset)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    (normalized_year, normalized_month as u32)
}

/// Check if a year is a leap year
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
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

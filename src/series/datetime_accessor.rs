use crate::core::error::Error as PandrsError;
use crate::series::base::Series;
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Timelike, Utc};
use chrono_tz::Tz;

/// DateTime accessor for Series containing datetime data
/// Provides pandas-like datetime operations through .dt accessor
#[derive(Clone)]
pub struct DateTimeAccessor {
    series: Series<NaiveDateTime>,
}

/// Build the "unknown frequency" error shared by [`DateTimeAccessor::floor`],
/// [`DateTimeAccessor::ceil`], and [`DateTimeAccessor::round`] -- these
/// previously fell back to silently returning the input unchanged for any
/// frequency string they didn't recognize (e.g. a typo like `"Hr"`), which
/// looks identical to a correctly-processed row with no way to tell the
/// two apart.
fn unknown_freq_err(freq: &str) -> PandrsError {
    PandrsError::InvalidValue(format!(
        "Unknown datetime frequency '{}'. Supported: 'D'/'day', 'H'/'hour', \
         'T'/'min'/'minute', 'S'/'second', '{{n}}min' (e.g. \"15min\"), '{{n}}S' (e.g. \"30S\")",
        freq
    ))
}

/// Truncate `dt` down to the start of the `freq` bucket it falls in.
///
/// Field-based (zeroes out the sub-frequency calendar fields) rather than
/// epoch/raw-timestamp arithmetic, so this is correct for any
/// `NaiveDateTime` -- including ones before 1970 -- without having to
/// separately reason about which direction integer division truncates a
/// negative epoch timestamp.
fn floor_to_freq(dt: &NaiveDateTime, freq: &str) -> Result<NaiveDateTime, PandrsError> {
    match freq {
        "D" | "day" => Ok(dt.date().and_hms_opt(0, 0, 0).unwrap_or(*dt)),
        "H" | "hour" => Ok(dt.date().and_hms_opt(dt.hour(), 0, 0).unwrap_or(*dt)),
        "T" | "min" | "minute" => Ok(dt
            .date()
            .and_hms_opt(dt.hour(), dt.minute(), 0)
            .unwrap_or(*dt)),
        "S" | "second" => Ok(dt
            .date()
            .and_hms_opt(dt.hour(), dt.minute(), dt.second())
            .unwrap_or(*dt)),
        // Specific minute intervals like "15min", "30min".
        freq_str if freq_str.ends_with("min") => {
            let n = freq_str
                .trim_end_matches("min")
                .parse::<u32>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| unknown_freq_err(freq))?;
            let rounded_minute = (dt.minute() / n) * n;
            Ok(dt
                .date()
                .and_hms_opt(dt.hour(), rounded_minute, 0)
                .unwrap_or(*dt))
        }
        // Specific second intervals like "30S", "15S".
        freq_str if freq_str.ends_with('S') => {
            let n = freq_str
                .trim_end_matches('S')
                .parse::<u32>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| unknown_freq_err(freq))?;
            let rounded_second = (dt.second() / n) * n;
            Ok(dt
                .date()
                .and_hms_opt(dt.hour(), dt.minute(), rounded_second)
                .unwrap_or(*dt))
        }
        _ => Err(unknown_freq_err(freq)),
    }
}

/// The fixed duration of one `freq` bucket, paired with [`floor_to_freq`]
/// to build `ceil`/`round` (the bucket after `floored` is
/// `floored + freq_period(freq)`).
fn freq_period(freq: &str) -> Result<chrono::Duration, PandrsError> {
    match freq {
        "D" | "day" => Ok(chrono::Duration::days(1)),
        "H" | "hour" => Ok(chrono::Duration::hours(1)),
        "T" | "min" | "minute" => Ok(chrono::Duration::minutes(1)),
        "S" | "second" => Ok(chrono::Duration::seconds(1)),
        freq_str if freq_str.ends_with("min") => freq_str
            .trim_end_matches("min")
            .parse::<i64>()
            .ok()
            .filter(|&n| n > 0)
            .map(chrono::Duration::minutes)
            .ok_or_else(|| unknown_freq_err(freq)),
        freq_str if freq_str.ends_with('S') => freq_str
            .trim_end_matches('S')
            .parse::<i64>()
            .ok()
            .filter(|&n| n > 0)
            .map(chrono::Duration::seconds)
            .ok_or_else(|| unknown_freq_err(freq)),
        _ => Err(unknown_freq_err(freq)),
    }
}

/// Round `dt` to the nearest multiple of `freq`; a value exactly halfway
/// between two boundaries rounds up, to the later one.
fn round_to_freq(dt: &NaiveDateTime, freq: &str) -> Result<NaiveDateTime, PandrsError> {
    let floored = floor_to_freq(dt, freq)?;
    if floored == *dt {
        return Ok(floored);
    }
    let ceiled = floored + freq_period(freq)?;
    let down_delta = *dt - floored;
    let up_delta = ceiled - *dt;
    if up_delta <= down_delta {
        Ok(ceiled)
    } else {
        Ok(floored)
    }
}

impl DateTimeAccessor {
    /// Create a new DateTimeAccessor
    pub fn new(series: Series<NaiveDateTime>) -> Result<Self, PandrsError> {
        Ok(DateTimeAccessor { series })
    }

    /// Extract year from datetime
    pub fn year(&self) -> Result<Series<i32>, PandrsError> {
        let years: Vec<i32> = self.series.values().iter().map(|dt| dt.year()).collect();

        Series::new(years, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract month from datetime
    pub fn month(&self) -> Result<Series<u32>, PandrsError> {
        let months: Vec<u32> = self.series.values().iter().map(|dt| dt.month()).collect();

        Series::new(months, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract day from datetime
    pub fn day(&self) -> Result<Series<u32>, PandrsError> {
        let days: Vec<u32> = self.series.values().iter().map(|dt| dt.day()).collect();

        Series::new(days, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract hour from datetime
    pub fn hour(&self) -> Result<Series<u32>, PandrsError> {
        let hours: Vec<u32> = self.series.values().iter().map(|dt| dt.hour()).collect();

        Series::new(hours, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract minute from datetime
    pub fn minute(&self) -> Result<Series<u32>, PandrsError> {
        let minutes: Vec<u32> = self.series.values().iter().map(|dt| dt.minute()).collect();

        Series::new(minutes, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract second from datetime
    pub fn second(&self) -> Result<Series<u32>, PandrsError> {
        let seconds: Vec<u32> = self.series.values().iter().map(|dt| dt.second()).collect();

        Series::new(seconds, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract weekday (0=Monday, 6=Sunday)
    pub fn weekday(&self) -> Result<Series<u32>, PandrsError> {
        let weekdays: Vec<u32> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.weekday().num_days_from_monday())
            .collect();

        Series::new(weekdays, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract day of year
    pub fn dayofyear(&self) -> Result<Series<u32>, PandrsError> {
        let dayofyears: Vec<u32> = self.series.values().iter().map(|dt| dt.ordinal()).collect();

        Series::new(dayofyears, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract quarter (1-4)
    pub fn quarter(&self) -> Result<Series<u32>, PandrsError> {
        let quarters: Vec<u32> = self
            .series
            .values()
            .iter()
            .map(|dt| ((dt.month() - 1) / 3) + 1)
            .collect();

        Series::new(quarters, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if date is weekend (Saturday or Sunday)
    pub fn is_weekend(&self) -> Result<Series<bool>, PandrsError> {
        let is_weekends: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                let weekday = dt.weekday().num_days_from_monday();
                weekday >= 5 // Saturday (5) or Sunday (6)
            })
            .collect();

        Series::new(is_weekends, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract date part (no time)
    pub fn date(&self) -> Result<Series<NaiveDate>, PandrsError> {
        let dates: Vec<NaiveDate> = self.series.values().iter().map(|dt| dt.date()).collect();

        Series::new(dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Format datetime as string
    pub fn strftime(&self, format: &str) -> Result<Series<String>, PandrsError> {
        let formatted: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.format(format).to_string())
            .collect();

        Series::new(formatted, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get timestamp (seconds since epoch)
    pub fn timestamp(&self) -> Result<Series<i64>, PandrsError> {
        let timestamps: Vec<i64> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.and_utc().timestamp())
            .collect();

        Series::new(timestamps, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add days to datetime
    pub fn add_days(&self, days: i64) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| *dt + chrono::Duration::days(days))
            .collect();

        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add hours to datetime
    pub fn add_hours(&self, hours: i64) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| *dt + chrono::Duration::hours(hours))
            .collect();

        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Normalize to start of day (set time to 00:00:00)
    pub fn normalize(&self) -> Result<Series<NaiveDateTime>, PandrsError> {
        let normalized: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.date().and_hms_opt(0, 0, 0).unwrap_or(*dt))
            .collect();

        Series::new(normalized, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Round datetime *down* to the specified frequency (truncate).
    ///
    /// See [`DateTimeAccessor::round`] for the supported `freq` strings.
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::{NaiveDate, Timelike};
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(14, 30, 45).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let floored = series.dt().expect("operation should succeed").floor("H").expect("operation should succeed");
    /// assert_eq!(floored.values()[0].hour(), 14);
    /// assert_eq!(floored.values()[0].minute(), 0);
    /// ```
    pub fn floor(&self, freq: &str) -> Result<Series<NaiveDateTime>, PandrsError> {
        let floored: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| floor_to_freq(dt, freq))
            .collect::<Result<_, PandrsError>>()?;

        Series::new(floored, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Round datetime *up* to the specified frequency.
    ///
    /// See [`DateTimeAccessor::round`] for the supported `freq` strings.
    /// Values already exactly on a frequency boundary are left unchanged
    /// (not bumped to the *next* boundary).
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::{NaiveDate, Timelike};
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(14, 30, 45).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let ceiled = series.dt().expect("operation should succeed").ceil("H").expect("operation should succeed");
    /// assert_eq!(ceiled.values()[0].hour(), 15);
    /// assert_eq!(ceiled.values()[0].minute(), 0);
    /// ```
    pub fn ceil(&self, freq: &str) -> Result<Series<NaiveDateTime>, PandrsError> {
        let ceiled: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                let floored = floor_to_freq(dt, freq)?;
                if floored == *dt {
                    Ok(floored)
                } else {
                    Ok(floored + freq_period(freq)?)
                }
            })
            .collect::<Result<_, PandrsError>>()?;

        Series::new(ceiled, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Round datetime to the *nearest* multiple of the specified frequency
    /// (round-half-up: a value exactly halfway between two boundaries
    /// rounds to the later one).
    ///
    /// # Arguments
    /// * `freq` - Frequency string: "D"/"day", "H"/"hour", "T"/"min"/"minute", "S"/"second", "{n}min" (e.g. "15min"), "{n}S" (e.g. "30S").
    ///
    /// Returns an error for an unrecognized `freq` rather than silently
    /// returning the input unchanged.
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::{NaiveDate, Timelike};
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(14, 20, 0).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let rounded = series.dt().expect("operation should succeed").round("H").expect("operation should succeed");
    /// // 14:20 is closer to 14:00 than to 15:00, so it rounds down.
    /// assert_eq!(rounded.values()[0].hour(), 14);
    /// assert_eq!(rounded.values()[0].minute(), 0);
    /// ```
    pub fn round(&self, freq: &str) -> Result<Series<NaiveDateTime>, PandrsError> {
        let rounded: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| round_to_freq(dt, freq))
            .collect::<Result<_, PandrsError>>()?;

        Series::new(rounded, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract week number of the year
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::NaiveDate;
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let weeks = series.dt().expect("operation should succeed").week().expect("operation should succeed");
    /// assert!(weeks.values()[0] >= 1 && weeks.values()[0] <= 53);
    /// ```
    pub fn week(&self) -> Result<Series<u32>, PandrsError> {
        let weeks: Vec<u32> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.iso_week().week())
            .collect();

        Series::new(weeks, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get number of days in the month
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::NaiveDate;
    /// let data = vec![
    ///     NaiveDate::from_ymd_opt(2023, 2, 15).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Feb 2023
    ///     NaiveDate::from_ymd_opt(2024, 2, 15).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Feb 2024 (leap year)
    /// ];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let days_in_month = series.dt().expect("operation should succeed").days_in_month().expect("operation should succeed");
    /// assert_eq!(days_in_month.values(), &[28, 29]); // 2023 vs 2024 leap year
    /// ```
    pub fn days_in_month(&self) -> Result<Series<u32>, PandrsError> {
        let days: Vec<u32> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                // Get the first day of next month and subtract 1 day to get last day of current month
                let year = dt.year();
                let month = dt.month();
                let next_month = if month == 12 { 1 } else { month + 1 };
                let next_year = if month == 12 { year + 1 } else { year };

                if let Some(next_month_first) = NaiveDate::from_ymd_opt(next_year, next_month, 1) {
                    if let Some(last_day_current_month) = next_month_first.pred_opt() {
                        return last_day_current_month.day();
                    }
                }
                // Fallback: use month-specific logic
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
                    _ => 30, // Fallback
                }
            })
            .collect();

        Series::new(days, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if year is a leap year
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::NaiveDate;
    /// let data = vec![
    ///     NaiveDate::from_ymd_opt(2023, 1, 1).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"),
    ///     NaiveDate::from_ymd_opt(2024, 1, 1).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"),
    /// ];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let is_leap = series.dt().expect("operation should succeed").is_leap_year().expect("operation should succeed");
    /// assert_eq!(is_leap.values(), &[false, true]);
    /// ```
    pub fn is_leap_year(&self) -> Result<Series<bool>, PandrsError> {
        let is_leap: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|dt| is_leap_year(dt.year()))
            .collect();

        Series::new(is_leap, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if date is a business day (Monday-Friday, excluding weekends)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::NaiveDate;
    /// let data = vec![
    ///     NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Monday
    ///     NaiveDate::from_ymd_opt(2023, 12, 23).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Saturday
    /// ];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let is_bday = series.dt().expect("operation should succeed").is_business_day().expect("operation should succeed");
    /// assert_eq!(is_bday.values(), &[true, false]);
    /// ```
    pub fn is_business_day(&self) -> Result<Series<bool>, PandrsError> {
        let is_bday: Vec<bool> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                let weekday = dt.weekday().num_days_from_monday();
                weekday < 5 // Monday (0) through Friday (4)
            })
            .collect();

        Series::new(is_bday, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add months to datetime
    ///
    /// # Arguments
    /// * `months` - Number of months to add (can be negative)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::{NaiveDate, Datelike};
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(14, 30, 45).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let plus_months = series.dt().expect("operation should succeed").add_months(3).expect("operation should succeed");
    /// assert_eq!(plus_months.values()[0].month(), 3);
    /// assert_eq!(plus_months.values()[0].year(), 2024);
    /// ```
    pub fn add_months(&self, months: i32) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                let mut year = dt.year();
                let mut month = dt.month() as i32;

                month += months;

                // Handle month overflow/underflow
                while month > 12 {
                    month -= 12;
                    year += 1;
                }
                while month < 1 {
                    month += 12;
                    year -= 1;
                }

                // Handle day overflow (e.g., Jan 31 + 1 month = Feb 28/29)
                let mut day = dt.day();
                if let Some(new_date) = NaiveDate::from_ymd_opt(year, month as u32, day) {
                    new_date
                        .and_hms_opt(dt.hour(), dt.minute(), dt.second())
                        .unwrap_or(*dt)
                } else {
                    // Day overflow, use last day of month
                    let days_in_new_month = match month {
                        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                        4 | 6 | 9 | 11 => 30,
                        2 => {
                            if is_leap_year(year) {
                                29
                            } else {
                                28
                            }
                        }
                        _ => 30,
                    };
                    day = day.min(days_in_new_month);
                    if let Some(new_date) = NaiveDate::from_ymd_opt(year, month as u32, day) {
                        new_date
                            .and_hms_opt(dt.hour(), dt.minute(), dt.second())
                            .unwrap_or(*dt)
                    } else {
                        *dt // Fallback to original date
                    }
                }
            })
            .collect();

        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add years to datetime
    ///
    /// # Arguments
    /// * `years` - Number of years to add (can be negative)
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::{NaiveDate, Datelike};
    /// let data = vec![NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(14, 30, 45).expect("operation should succeed")];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let plus_years = series.dt().expect("operation should succeed").add_years(2).expect("operation should succeed");
    /// assert_eq!(plus_years.values()[0].year(), 2025);
    /// ```
    pub fn add_years(&self, years: i32) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self
            .series
            .values()
            .iter()
            .map(|dt| {
                let new_year = dt.year() + years;

                // Handle leap year edge case (Feb 29 -> Feb 28 in non-leap year)
                let mut day = dt.day();
                let month = dt.month();

                if month == 2 && day == 29 && !is_leap_year(new_year) {
                    day = 28;
                }

                if let Some(new_date) = NaiveDate::from_ymd_opt(new_year, month, day) {
                    new_date
                        .and_hms_opt(dt.hour(), dt.minute(), dt.second())
                        .unwrap_or(*dt)
                } else {
                    *dt // Fallback to original date
                }
            })
            .collect();

        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Calculate business day count between dates (excluding weekends)
    /// Note: This is a simplified implementation that only excludes weekends.
    /// For a production system, you'd want to integrate with a holiday calendar.
    ///
    /// # Arguments
    /// * `end_date` - End date for business day calculation
    ///
    /// # Examples
    /// ```
    /// use pandrs::Series;
    /// use chrono::NaiveDate;
    /// let data = vec![
    ///     NaiveDate::from_ymd_opt(2023, 12, 25).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Monday
    ///     NaiveDate::from_ymd_opt(2023, 12, 18).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"), // Monday
    /// ];
    /// let series = Series::new(data, None).expect("operation should succeed");
    /// let end_date = NaiveDate::from_ymd_opt(2023, 12, 29).expect("operation should succeed").and_hms_opt(0, 0, 0).expect("operation should succeed"); // Friday
    /// let bday_count = series.dt().expect("operation should succeed").business_day_count(end_date).expect("operation should succeed");
    /// // Should count business days only
    /// ```
    pub fn business_day_count(&self, end_date: NaiveDateTime) -> Result<Series<i64>, PandrsError> {
        let counts: Vec<i64> = self
            .series
            .values()
            .iter()
            .map(|start_dt| {
                let start = if *start_dt <= end_date {
                    *start_dt
                } else {
                    end_date
                };
                let end = if *start_dt <= end_date {
                    end_date
                } else {
                    *start_dt
                };

                let mut count = 0i64;
                let mut current = start.date();
                let end_date_only = end.date();

                while current <= end_date_only {
                    let weekday = current.weekday().num_days_from_monday();
                    if weekday < 5 {
                        // Monday (0) through Friday (4)
                        count += 1;
                    }
                    if let Some(next_day) = current.succ_opt() {
                        current = next_day;
                    } else {
                        break;
                    }
                }

                if *start_dt > end_date {
                    -count
                } else {
                    count
                }
            })
            .collect();

        Series::new(counts, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }
}

/// DateTime accessor for timezone-aware Series
#[derive(Clone)]
pub struct DateTimeAccessorTz {
    series: Series<DateTime<Utc>>,
}

impl DateTimeAccessorTz {
    /// Create a new timezone-aware DateTimeAccessor
    pub fn new(series: Series<DateTime<Utc>>) -> Result<Self, PandrsError> {
        Ok(DateTimeAccessorTz { series })
    }

    /// Convert timezone
    pub fn tz_convert(&self, tz_str: &str) -> Result<Series<DateTime<Tz>>, PandrsError> {
        let tz = tz_str
            .parse::<Tz>()
            .map_err(|e| PandrsError::InvalidValue(format!("Invalid timezone: {}", e)))?;

        let converted: Vec<DateTime<Tz>> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.with_timezone(&tz))
            .collect();

        Series::new(converted, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get timezone name
    pub fn tz(&self) -> Result<Series<String>, PandrsError> {
        let tz_names: Vec<String> = self
            .series
            .values()
            .iter()
            .map(|dt| dt.timezone().to_string())
            .collect();

        Series::new(tz_names, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract UTC offset in hours
    pub fn utc_offset(&self) -> Result<Series<i32>, PandrsError> {
        let offsets: Vec<i32> = self.series.values()
            .iter()
            .map(|_dt| 0) // UTC always has 0 offset
            .collect();

        Series::new(offsets, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }
}

/// Helper function to check if a year is a leap year
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Helper functions for creating datetime series from various inputs
pub mod datetime_constructors {
    use super::*;
    use std::str::FromStr;

    /// Parse string datetime series
    pub fn parse_datetime_series(
        strings: Vec<String>,
        format: Option<&str>,
        name: Option<String>,
    ) -> Result<Series<NaiveDateTime>, PandrsError> {
        let datetimes: Result<Vec<NaiveDateTime>, PandrsError> = if let Some(fmt) = format {
            strings
                .iter()
                .map(|s| {
                    NaiveDateTime::parse_from_str(s, fmt).map_err(|e| {
                        PandrsError::InvalidValue(format!(
                            "Failed to parse datetime '{}': {}",
                            s, e
                        ))
                    })
                })
                .collect()
        } else {
            // Try common formats
            strings
                .iter()
                .map(|s| {
                    // Try RFC3339 first
                    if let Ok(dt) = DateTime::<Utc>::from_str(s) {
                        return Ok(dt.naive_utc());
                    }
                    // Try common ISO format
                    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
                        return Ok(dt);
                    }
                    // Try date only
                    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                        return Ok(date.and_hms_opt(0, 0, 0).unwrap_or_else(|| {
                            // Fallback to Unix epoch if time conversion fails (should never happen)
                            NaiveDate::from_ymd_opt(1970, 1, 1)
                                .expect("Unix epoch date is always valid")
                                .and_hms_opt(0, 0, 0)
                                .expect("Midnight time is always valid")
                        }));
                    }
                    Err(PandrsError::InvalidValue(format!(
                        "Unable to parse datetime: {}",
                        s
                    )))
                })
                .collect()
        };

        let dt_values = datetimes?;
        Series::new(dt_values, name)
            .map_err(|e| PandrsError::Type(format!("Failed to create datetime series: {:?}", e)))
    }

    /// Create date range
    pub fn date_range(
        start: NaiveDate,
        end: NaiveDate,
        freq: &str,
    ) -> Result<Series<NaiveDateTime>, PandrsError> {
        let mut dates = Vec::new();
        // SAFETY: Valid dates always convert to datetime with valid time
        let mut current = start
            .and_hms_opt(0, 0, 0)
            .expect("Valid date should convert to datetime with time 00:00:00");
        let end_dt = end
            .and_hms_opt(23, 59, 59)
            .expect("Valid date should convert to datetime with time 23:59:59");

        let duration = match freq {
            "D" | "day" => chrono::Duration::days(1),
            "H" | "hour" => chrono::Duration::hours(1),
            "W" | "week" => chrono::Duration::weeks(1),
            "M" | "month" => chrono::Duration::days(30), // Approximate
            _ => {
                return Err(PandrsError::InvalidValue(format!(
                    "Unsupported frequency: {}",
                    freq
                )))
            }
        };

        while current <= end_dt {
            dates.push(current);
            current = current + duration;
        }

        Series::new(dates, Some("date_range".to_string()))
            .map_err(|e| PandrsError::Type(format!("Failed to create date range: {:?}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn test_datetime_extraction() {
        let dt1 = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed");
        let dt2 = NaiveDate::from_ymd_opt(2024, 6, 15)
            .expect("operation should succeed")
            .and_hms_opt(9, 15, 30)
            .expect("operation should succeed");

        let data = vec![dt1, dt2];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Test year extraction
        let years = dt_accessor.year().expect("operation should succeed");
        assert_eq!(years.values(), &[2023, 2024]);

        // Test month extraction
        let months = dt_accessor.month().expect("operation should succeed");
        assert_eq!(months.values(), &[12, 6]);

        // Test day extraction
        let days = dt_accessor.day().expect("operation should succeed");
        assert_eq!(days.values(), &[25, 15]);

        // Test hour extraction
        let hours = dt_accessor.hour().expect("operation should succeed");
        assert_eq!(hours.values(), &[14, 9]);
    }

    #[test]
    fn test_datetime_formatting() {
        let dt = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed");
        let data = vec![dt];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        let formatted = dt_accessor
            .strftime("%Y-%m-%d %H:%M:%S")
            .expect("operation should succeed");
        assert_eq!(formatted.values(), &["2023-12-25 14:30:45".to_string()]);
    }

    #[test]
    fn test_weekend_detection() {
        // 2023-12-23 is Saturday, 2023-12-24 is Sunday, 2023-12-25 is Monday
        let dt1 = NaiveDate::from_ymd_opt(2023, 12, 23)
            .expect("operation should succeed")
            .and_hms_opt(10, 0, 0)
            .expect("operation should succeed");
        let dt2 = NaiveDate::from_ymd_opt(2023, 12, 24)
            .expect("operation should succeed")
            .and_hms_opt(10, 0, 0)
            .expect("operation should succeed");
        let dt3 = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(10, 0, 0)
            .expect("operation should succeed");

        let data = vec![dt1, dt2, dt3];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        let is_weekend = dt_accessor.is_weekend().expect("operation should succeed");
        assert_eq!(is_weekend.values(), &[true, true, false]);
    }

    #[test]
    fn test_date_arithmetic() {
        let dt = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed");
        let data = vec![dt];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Add 5 days
        let plus_days = dt_accessor.add_days(5).expect("operation should succeed");
        let expected = NaiveDate::from_ymd_opt(2023, 12, 30)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed");
        assert_eq!(plus_days.values(), &[expected]);

        // Add 3 hours
        let plus_hours = dt_accessor.add_hours(3).expect("operation should succeed");
        let expected = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(17, 30, 45)
            .expect("operation should succeed");
        assert_eq!(plus_hours.values(), &[expected]);
    }

    #[test]
    fn test_enhanced_datetime_features() {
        let dt1 = NaiveDate::from_ymd_opt(2023, 2, 15)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed"); // Feb 2023
        let dt2 = NaiveDate::from_ymd_opt(2024, 2, 15)
            .expect("operation should succeed")
            .and_hms_opt(9, 15, 30)
            .expect("operation should succeed"); // Feb 2024 (leap year)
        let dt3 = NaiveDate::from_ymd_opt(2023, 12, 23)
            .expect("operation should succeed")
            .and_hms_opt(10, 0, 0)
            .expect("operation should succeed"); // Saturday

        let data = vec![dt1, dt2, dt3];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Test week extraction
        let weeks = dt_accessor.week().expect("operation should succeed");
        assert!(weeks.values()[0] >= 1 && weeks.values()[0] <= 53);

        // Test days in month (leap year detection)
        let days_in_month = dt_accessor
            .days_in_month()
            .expect("operation should succeed");
        assert_eq!(days_in_month.values()[0], 28); // Feb 2023
        assert_eq!(days_in_month.values()[1], 29); // Feb 2024 (leap year)
        assert_eq!(days_in_month.values()[2], 31); // Dec 2023

        // Test leap year detection
        let is_leap = dt_accessor
            .is_leap_year()
            .expect("operation should succeed");
        assert_eq!(is_leap.values(), &[false, true, false]);

        // Test business day detection
        let is_bday = dt_accessor
            .is_business_day()
            .expect("operation should succeed");
        assert_eq!(is_bday.values()[0], true); // Wednesday
        assert_eq!(is_bday.values()[1], true); // Thursday
        assert_eq!(is_bday.values()[2], false); // Saturday
    }

    #[test]
    fn test_advanced_date_arithmetic() {
        let dt = NaiveDate::from_ymd_opt(2023, 1, 31)
            .expect("operation should succeed")
            .and_hms_opt(14, 30, 45)
            .expect("operation should succeed");
        let data = vec![dt];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Add months with day overflow handling
        let plus_months = dt_accessor.add_months(1).expect("operation should succeed");
        // Jan 31 + 1 month = Feb 28 (day overflow handled)
        assert_eq!(plus_months.values()[0].month(), 2);
        assert_eq!(plus_months.values()[0].day(), 28);

        // Add years
        let plus_years = dt_accessor.add_years(2).expect("operation should succeed");
        assert_eq!(plus_years.values()[0].year(), 2025);
        assert_eq!(plus_years.values()[0].month(), 1);
        assert_eq!(plus_years.values()[0].day(), 31);
    }

    #[test]
    fn test_enhanced_rounding() {
        let dt = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(14, 37, 23)
            .expect("operation should succeed");
        let data = vec![dt];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Test 15-minute rounding
        let rounded_15min = dt_accessor
            .round("15min")
            .expect("operation should succeed");
        assert_eq!(rounded_15min.values()[0].minute(), 30); // 37 minutes rounds down to 30

        // Test second rounding
        let rounded_sec = dt_accessor.round("S").expect("operation should succeed");
        assert_eq!(rounded_sec.values()[0].second(), 23);
        assert_eq!(rounded_sec.values()[0].nanosecond(), 0);
    }

    #[test]
    fn test_business_day_count() {
        // Test business day counting
        let start_dt = NaiveDate::from_ymd_opt(2023, 12, 25)
            .expect("operation should succeed")
            .and_hms_opt(0, 0, 0)
            .expect("operation should succeed"); // Monday
        let data = vec![start_dt];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        let end_dt = NaiveDate::from_ymd_opt(2023, 12, 29)
            .expect("operation should succeed")
            .and_hms_opt(0, 0, 0)
            .expect("operation should succeed"); // Friday
        let bday_count = dt_accessor
            .business_day_count(end_dt)
            .expect("operation should succeed");

        // Monday to Friday inclusive = 5 business days
        assert_eq!(bday_count.values()[0], 5);
    }

    #[test]
    fn test_leap_year_edge_cases() {
        // Test leap year Feb 29 handling when adding years
        let leap_day = NaiveDate::from_ymd_opt(2024, 2, 29)
            .expect("operation should succeed")
            .and_hms_opt(12, 0, 0)
            .expect("operation should succeed");
        let data = vec![leap_day];
        let series =
            Series::new(data, Some("test_dates".to_string())).expect("operation should succeed");
        let dt_accessor = DateTimeAccessor::new(series).expect("operation should succeed");

        // Adding 1 year to Feb 29, 2024 should give Feb 28, 2025
        let plus_year = dt_accessor.add_years(1).expect("operation should succeed");
        assert_eq!(plus_year.values()[0].year(), 2025);
        assert_eq!(plus_year.values()[0].month(), 2);
        assert_eq!(plus_year.values()[0].day(), 28);
    }
}

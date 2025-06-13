use crate::core::error::Error as PandrsError;
use crate::series::base::Series;
use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};
use chrono_tz::Tz;

/// DateTime accessor for Series containing datetime data
/// Provides pandas-like datetime operations through .dt accessor
#[derive(Clone)]
pub struct DateTimeAccessor {
    series: Series<NaiveDateTime>,
}

impl DateTimeAccessor {
    /// Create a new DateTimeAccessor
    pub fn new(series: Series<NaiveDateTime>) -> Result<Self, PandrsError> {
        Ok(DateTimeAccessor { series })
    }

    /// Extract year from datetime
    pub fn year(&self) -> Result<Series<i32>, PandrsError> {
        let years: Vec<i32> = self.series.values()
            .iter()
            .map(|dt| dt.year())
            .collect();
        
        Series::new(years, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract month from datetime
    pub fn month(&self) -> Result<Series<u32>, PandrsError> {
        let months: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.month())
            .collect();
        
        Series::new(months, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract day from datetime
    pub fn day(&self) -> Result<Series<u32>, PandrsError> {
        let days: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.day())
            .collect();
        
        Series::new(days, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract hour from datetime
    pub fn hour(&self) -> Result<Series<u32>, PandrsError> {
        let hours: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.hour())
            .collect();
        
        Series::new(hours, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract minute from datetime
    pub fn minute(&self) -> Result<Series<u32>, PandrsError> {
        let minutes: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.minute())
            .collect();
        
        Series::new(minutes, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract second from datetime
    pub fn second(&self) -> Result<Series<u32>, PandrsError> {
        let seconds: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.second())
            .collect();
        
        Series::new(seconds, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract weekday (0=Monday, 6=Sunday)
    pub fn weekday(&self) -> Result<Series<u32>, PandrsError> {
        let weekdays: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.weekday().num_days_from_monday())
            .collect();
        
        Series::new(weekdays, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract day of year
    pub fn dayofyear(&self) -> Result<Series<u32>, PandrsError> {
        let dayofyears: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| dt.ordinal())
            .collect();
        
        Series::new(dayofyears, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Extract quarter (1-4)
    pub fn quarter(&self) -> Result<Series<u32>, PandrsError> {
        let quarters: Vec<u32> = self.series.values()
            .iter()
            .map(|dt| ((dt.month() - 1) / 3) + 1)
            .collect();
        
        Series::new(quarters, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Check if date is weekend (Saturday or Sunday)
    pub fn is_weekend(&self) -> Result<Series<bool>, PandrsError> {
        let is_weekends: Vec<bool> = self.series.values()
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
        let dates: Vec<NaiveDate> = self.series.values()
            .iter()
            .map(|dt| dt.date())
            .collect();
        
        Series::new(dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Format datetime as string
    pub fn strftime(&self, format: &str) -> Result<Series<String>, PandrsError> {
        let formatted: Vec<String> = self.series.values()
            .iter()
            .map(|dt| dt.format(format).to_string())
            .collect();
        
        Series::new(formatted, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get timestamp (seconds since epoch)
    pub fn timestamp(&self) -> Result<Series<i64>, PandrsError> {
        let timestamps: Vec<i64> = self.series.values()
            .iter()
            .map(|dt| dt.and_utc().timestamp())
            .collect();
        
        Series::new(timestamps, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add days to datetime
    pub fn add_days(&self, days: i64) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self.series.values()
            .iter()
            .map(|dt| *dt + chrono::Duration::days(days))
            .collect();
        
        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Add hours to datetime
    pub fn add_hours(&self, hours: i64) -> Result<Series<NaiveDateTime>, PandrsError> {
        let new_dates: Vec<NaiveDateTime> = self.series.values()
            .iter()
            .map(|dt| *dt + chrono::Duration::hours(hours))
            .collect();
        
        Series::new(new_dates, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Normalize to start of day (set time to 00:00:00)
    pub fn normalize(&self) -> Result<Series<NaiveDateTime>, PandrsError> {
        let normalized: Vec<NaiveDateTime> = self.series.values()
            .iter()
            .map(|dt| dt.date().and_hms_opt(0, 0, 0).unwrap_or(*dt))
            .collect();
        
        Series::new(normalized, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Round datetime to specified frequency
    pub fn round(&self, freq: &str) -> Result<Series<NaiveDateTime>, PandrsError> {
        let rounded: Vec<NaiveDateTime> = self.series.values()
            .iter()
            .map(|dt| {
                match freq {
                    "D" | "day" => dt.date().and_hms_opt(0, 0, 0).unwrap_or(*dt),
                    "H" | "hour" => dt.date().and_hms_opt(dt.hour(), 0, 0).unwrap_or(*dt),
                    "T" | "min" | "minute" => dt.date().and_hms_opt(dt.hour(), dt.minute(), 0).unwrap_or(*dt),
                    _ => *dt, // Unknown frequency, return original
                }
            })
            .collect();
        
        Series::new(rounded, self.series.name().cloned())
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
        let tz = tz_str.parse::<Tz>()
            .map_err(|e| PandrsError::InvalidValue(format!("Invalid timezone: {}", e)))?;
        
        let converted: Vec<DateTime<Tz>> = self.series.values()
            .iter()
            .map(|dt| dt.with_timezone(&tz))
            .collect();
        
        Series::new(converted, self.series.name().cloned())
            .map_err(|e| PandrsError::Type(format!("Failed to create series: {:?}", e)))
    }

    /// Get timezone name
    pub fn tz(&self) -> Result<Series<String>, PandrsError> {
        let tz_names: Vec<String> = self.series.values()
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

/// Helper functions for creating datetime series from various inputs
pub mod datetime_constructors {
    use super::*;
    use std::str::FromStr;

    /// Parse string datetime series
    pub fn parse_datetime_series(
        strings: Vec<String>, 
        format: Option<&str>,
        name: Option<String>
    ) -> Result<Series<NaiveDateTime>, PandrsError> {
        let datetimes: Result<Vec<NaiveDateTime>, PandrsError> = if let Some(fmt) = format {
            strings.iter()
                .map(|s| {
                    NaiveDateTime::parse_from_str(s, fmt)
                        .map_err(|e| PandrsError::InvalidValue(format!("Failed to parse datetime '{}': {}", s, e)))
                })
                .collect()
        } else {
            // Try common formats
            strings.iter()
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
                            NaiveDate::from_ymd_opt(1970, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
                        }));
                    }
                    Err(PandrsError::InvalidValue(format!("Unable to parse datetime: {}", s)))
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
        freq: &str
    ) -> Result<Series<NaiveDateTime>, PandrsError> {
        let mut dates = Vec::new();
        let mut current = start.and_hms_opt(0, 0, 0).unwrap();
        let end_dt = end.and_hms_opt(23, 59, 59).unwrap();

        let duration = match freq {
            "D" | "day" => chrono::Duration::days(1),
            "H" | "hour" => chrono::Duration::hours(1),
            "W" | "week" => chrono::Duration::weeks(1),
            "M" | "month" => chrono::Duration::days(30), // Approximate
            _ => return Err(PandrsError::InvalidValue(format!("Unsupported frequency: {}", freq))),
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
        let dt1 = NaiveDate::from_ymd_opt(2023, 12, 25).unwrap().and_hms_opt(14, 30, 45).unwrap();
        let dt2 = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap().and_hms_opt(9, 15, 30).unwrap();
        
        let data = vec![dt1, dt2];
        let series = Series::new(data, Some("test_dates".to_string())).unwrap();
        let dt_accessor = DateTimeAccessor::new(series).unwrap();
        
        // Test year extraction
        let years = dt_accessor.year().unwrap();
        assert_eq!(years.values(), &[2023, 2024]);
        
        // Test month extraction
        let months = dt_accessor.month().unwrap();
        assert_eq!(months.values(), &[12, 6]);
        
        // Test day extraction
        let days = dt_accessor.day().unwrap();
        assert_eq!(days.values(), &[25, 15]);
        
        // Test hour extraction
        let hours = dt_accessor.hour().unwrap();
        assert_eq!(hours.values(), &[14, 9]);
    }

    #[test]
    fn test_datetime_formatting() {
        let dt = NaiveDate::from_ymd_opt(2023, 12, 25).unwrap().and_hms_opt(14, 30, 45).unwrap();
        let data = vec![dt];
        let series = Series::new(data, Some("test_dates".to_string())).unwrap();
        let dt_accessor = DateTimeAccessor::new(series).unwrap();
        
        let formatted = dt_accessor.strftime("%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(formatted.values(), &["2023-12-25 14:30:45".to_string()]);
    }

    #[test]
    fn test_weekend_detection() {
        // 2023-12-23 is Saturday, 2023-12-24 is Sunday, 2023-12-25 is Monday
        let dt1 = NaiveDate::from_ymd_opt(2023, 12, 23).unwrap().and_hms_opt(10, 0, 0).unwrap();
        let dt2 = NaiveDate::from_ymd_opt(2023, 12, 24).unwrap().and_hms_opt(10, 0, 0).unwrap();
        let dt3 = NaiveDate::from_ymd_opt(2023, 12, 25).unwrap().and_hms_opt(10, 0, 0).unwrap();
        
        let data = vec![dt1, dt2, dt3];
        let series = Series::new(data, Some("test_dates".to_string())).unwrap();
        let dt_accessor = DateTimeAccessor::new(series).unwrap();
        
        let is_weekend = dt_accessor.is_weekend().unwrap();
        assert_eq!(is_weekend.values(), &[true, true, false]);
    }

    #[test]
    fn test_date_arithmetic() {
        let dt = NaiveDate::from_ymd_opt(2023, 12, 25).unwrap().and_hms_opt(14, 30, 45).unwrap();
        let data = vec![dt];
        let series = Series::new(data, Some("test_dates".to_string())).unwrap();
        let dt_accessor = DateTimeAccessor::new(series).unwrap();
        
        // Add 5 days
        let plus_days = dt_accessor.add_days(5).unwrap();
        let expected = NaiveDate::from_ymd_opt(2023, 12, 30).unwrap().and_hms_opt(14, 30, 45).unwrap();
        assert_eq!(plus_days.values(), &[expected]);
        
        // Add 3 hours
        let plus_hours = dt_accessor.add_hours(3).unwrap();
        let expected = NaiveDate::from_ymd_opt(2023, 12, 25).unwrap().and_hms_opt(17, 30, 45).unwrap();
        assert_eq!(plus_hours.values(), &[expected]);
    }
}
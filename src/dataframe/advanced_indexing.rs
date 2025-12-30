//! Advanced specialized indexing types for DataFrames
//!
//! This module provides specialized index types for specific use cases:
//! - DatetimeIndex: Time series indexing with timezone support
//! - PeriodIndex: Business period indexing (quarters, months, etc.)
//! - IntervalIndex: Range-based and binned data indexing  
//! - CategoricalIndex: Memory-optimized categorical indexing
//! - Index set operations: union, intersection, difference, symmetric_difference

use std::collections::{HashMap, HashSet};
use std::fmt;

use chrono::{DateTime, Datelike, Duration, FixedOffset, NaiveDate, NaiveDateTime, Timelike, Utc};
use chrono_tz::Tz;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::indexing::AdvancedIndexingExt as DataFrameIndexingExt;
use crate::series::base::Series;

/// Trait for all index types
pub trait Index: fmt::Debug + Clone {
    /// Get the length of the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get string representation of index values
    fn to_string_vec(&self) -> Vec<String>;

    /// Get the name of the index
    fn name(&self) -> Option<&str>;

    /// Set the name of the index
    fn set_name(&mut self, name: Option<String>);

    /// Check if index contains duplicates
    fn has_duplicates(&self) -> bool;

    /// Get unique values (returns same type)
    fn unique(&self) -> Result<Self>
    where
        Self: Sized;

    /// Sort the index (returns same type)
    fn sort(&self, ascending: bool) -> Result<(Self, Vec<usize>)>
    where
        Self: Sized;
}

/// Enum wrapper for different index types (for polymorphic usage)
#[derive(Debug, Clone)]
pub enum IndexType {
    Datetime(DatetimeIndex),
    Period(PeriodIndex),
    Interval(IntervalIndex),
    Categorical(CategoricalIndex),
}

impl IndexType {
    /// Get the length of the index
    pub fn len(&self) -> usize {
        match self {
            IndexType::Datetime(idx) => idx.len(),
            IndexType::Period(idx) => idx.len(),
            IndexType::Interval(idx) => idx.len(),
            IndexType::Categorical(idx) => idx.len(),
        }
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get string representation of index values
    pub fn to_string_vec(&self) -> Vec<String> {
        match self {
            IndexType::Datetime(idx) => idx.to_string_vec(),
            IndexType::Period(idx) => idx.to_string_vec(),
            IndexType::Interval(idx) => idx.to_string_vec(),
            IndexType::Categorical(idx) => idx.to_string_vec(),
        }
    }

    /// Get the name of the index
    pub fn name(&self) -> Option<&str> {
        match self {
            IndexType::Datetime(idx) => idx.name(),
            IndexType::Period(idx) => idx.name(),
            IndexType::Interval(idx) => idx.name(),
            IndexType::Categorical(idx) => idx.name(),
        }
    }

    /// Check if index contains duplicates
    pub fn has_duplicates(&self) -> bool {
        match self {
            IndexType::Datetime(idx) => idx.has_duplicates(),
            IndexType::Period(idx) => idx.has_duplicates(),
            IndexType::Interval(idx) => idx.has_duplicates(),
            IndexType::Categorical(idx) => idx.has_duplicates(),
        }
    }
}

/// Index set operations trait
pub trait IndexSetOps<T: Index> {
    /// Union of two indexes
    fn union(&self, other: &T) -> Result<IndexType>;

    /// Intersection of two indexes
    fn intersection(&self, other: &T) -> Result<IndexType>;

    /// Difference of two indexes (self - other)
    fn difference(&self, other: &T) -> Result<IndexType>;

    /// Symmetric difference of two indexes
    fn symmetric_difference(&self, other: &T) -> Result<IndexType>;
}

/// DateTime index for time series data
#[derive(Debug, Clone)]
pub struct DatetimeIndex {
    /// DateTime values
    pub values: Vec<NaiveDateTime>,
    /// Index name
    pub name: Option<String>,
    /// Timezone information (optional)
    pub timezone: Option<Tz>,
    /// Frequency information (optional)
    pub frequency: Option<String>,
}

impl DatetimeIndex {
    /// Create a new datetime index
    pub fn new(values: Vec<NaiveDateTime>, name: Option<String>) -> Self {
        Self {
            values,
            name,
            timezone: None,
            frequency: None,
        }
    }

    /// Create a datetime index with timezone
    pub fn with_timezone(values: Vec<NaiveDateTime>, name: Option<String>, timezone: Tz) -> Self {
        Self {
            values,
            name,
            timezone: Some(timezone),
            frequency: None,
        }
    }

    /// Create a datetime range with frequency
    pub fn date_range(
        start: NaiveDateTime,
        end: Option<NaiveDateTime>,
        periods: Option<usize>,
        frequency: &str,
        name: Option<String>,
    ) -> Result<Self> {
        let freq_duration = Self::parse_frequency(frequency)?;
        let mut values = Vec::new();

        match (end, periods) {
            (Some(end_date), None) => {
                // Generate range from start to end
                let mut current = start;
                while current <= end_date {
                    values.push(current);
                    current = current + freq_duration;
                }
            }
            (None, Some(periods_count)) => {
                // Generate specific number of periods
                let mut current = start;
                for _ in 0..periods_count {
                    values.push(current);
                    current = current + freq_duration;
                }
            }
            (Some(end_date), Some(periods_count)) => {
                // Use periods count, ignore end date
                let mut current = start;
                for _ in 0..periods_count {
                    values.push(current);
                    current = current + freq_duration;
                    if current > end_date {
                        break;
                    }
                }
            }
            (None, None) => {
                return Err(Error::InvalidValue(
                    "Either end date or periods must be specified".to_string(),
                ));
            }
        }

        Ok(Self {
            values,
            name,
            timezone: None,
            frequency: Some(frequency.to_string()),
        })
    }

    /// Parse frequency string to Duration
    fn parse_frequency(freq: &str) -> Result<Duration> {
        let freq = freq.to_lowercase();
        match freq.as_str() {
            "d" | "day" | "days" => Ok(Duration::days(1)),
            "h" | "hour" | "hours" => Ok(Duration::hours(1)),
            "min" | "minute" | "minutes" => Ok(Duration::minutes(1)),
            "s" | "second" | "seconds" => Ok(Duration::seconds(1)),
            "w" | "week" | "weeks" => Ok(Duration::weeks(1)),
            "ms" | "millisecond" | "milliseconds" => Ok(Duration::milliseconds(1)),
            _ => {
                // Simple numeric frequency parsing for common patterns
                if freq.ends_with("min") || freq.ends_with("minutes") {
                    let num_str = freq.trim_end_matches("min").trim_end_matches("utes");
                    if let Ok(num) = num_str.parse::<i64>() {
                        Ok(Duration::minutes(num))
                    } else {
                        Err(Error::InvalidValue(format!(
                            "Invalid frequency format: {}",
                            freq
                        )))
                    }
                } else if freq.ends_with("h") || freq.ends_with("hour") || freq.ends_with("hours") {
                    let num_str = freq
                        .trim_end_matches("h")
                        .trim_end_matches("our")
                        .trim_end_matches("s");
                    if let Ok(num) = num_str.parse::<i64>() {
                        Ok(Duration::hours(num))
                    } else {
                        Err(Error::InvalidValue(format!(
                            "Invalid frequency format: {}",
                            freq
                        )))
                    }
                } else if freq.ends_with("d") || freq.ends_with("day") || freq.ends_with("days") {
                    let num_str = freq
                        .trim_end_matches("d")
                        .trim_end_matches("ay")
                        .trim_end_matches("s");
                    if let Ok(num) = num_str.parse::<i64>() {
                        Ok(Duration::days(num))
                    } else {
                        Err(Error::InvalidValue(format!(
                            "Invalid frequency format: {}",
                            freq
                        )))
                    }
                } else {
                    Err(Error::InvalidValue(format!(
                        "Invalid frequency format: {}",
                        freq
                    )))
                }
            }
        }
    }

    /// Get year component
    pub fn year(&self) -> Vec<i32> {
        self.values.iter().map(|dt| dt.year()).collect()
    }

    /// Get month component
    pub fn month(&self) -> Vec<u32> {
        self.values.iter().map(|dt| dt.month()).collect()
    }

    /// Get day component
    pub fn day(&self) -> Vec<u32> {
        self.values.iter().map(|dt| dt.day()).collect()
    }

    /// Get hour component
    pub fn hour(&self) -> Vec<u32> {
        self.values.iter().map(|dt| dt.hour()).collect()
    }

    /// Get minute component
    pub fn minute(&self) -> Vec<u32> {
        self.values.iter().map(|dt| dt.minute()).collect()
    }

    /// Get weekday component (0=Monday, 6=Sunday)
    pub fn weekday(&self) -> Vec<u32> {
        self.values
            .iter()
            .map(|dt| dt.weekday().num_days_from_monday())
            .collect()
    }

    /// Filter by date range
    pub fn filter_range(&self, start: NaiveDateTime, end: NaiveDateTime) -> Result<Vec<usize>> {
        Ok(self
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, dt)| {
                if *dt >= start && *dt <= end {
                    Some(i)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Resample to different frequency
    pub fn resample(&self, frequency: &str) -> Result<Vec<Vec<usize>>> {
        let freq_duration = Self::parse_frequency(frequency)?;
        let mut groups = Vec::new();

        if self.values.is_empty() {
            return Ok(groups);
        }

        let mut current_group = Vec::new();
        let mut group_start = self.values[0];

        for (i, dt) in self.values.iter().enumerate() {
            if *dt >= group_start + freq_duration {
                if !current_group.is_empty() {
                    groups.push(current_group);
                    current_group = Vec::new();
                }
                group_start = *dt;
            }
            current_group.push(i);
        }

        if !current_group.is_empty() {
            groups.push(current_group);
        }

        Ok(groups)
    }
}

impl Index for DatetimeIndex {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn to_string_vec(&self) -> Vec<String> {
        self.values
            .iter()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .collect()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    fn has_duplicates(&self) -> bool {
        let mut seen = HashSet::new();
        self.values.iter().any(|dt| !seen.insert(dt))
    }

    fn unique(&self) -> Result<Self> {
        let mut unique_values: Vec<NaiveDateTime> = self.values.clone();
        unique_values.sort();
        unique_values.dedup();

        Ok(DatetimeIndex::new(unique_values, self.name.clone()))
    }

    fn sort(&self, ascending: bool) -> Result<(Self, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..self.values.len()).collect();

        if ascending {
            indices.sort_by(|&a, &b| self.values[a].cmp(&self.values[b]));
        } else {
            indices.sort_by(|&a, &b| self.values[b].cmp(&self.values[a]));
        }

        let sorted_values: Vec<NaiveDateTime> = indices.iter().map(|&i| self.values[i]).collect();
        let sorted_index = DatetimeIndex::new(sorted_values, self.name.clone());

        Ok((sorted_index, indices))
    }
}

/// Period index for business periods (quarters, months, etc.)
#[derive(Debug, Clone)]
pub struct PeriodIndex {
    /// Period values
    pub periods: Vec<Period>,
    /// Index name
    pub name: Option<String>,
    /// Frequency of periods
    pub frequency: PeriodFrequency,
}

/// Period frequency enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum PeriodFrequency {
    /// Annual periods
    Annual,
    /// Quarterly periods  
    Quarterly,
    /// Monthly periods
    Monthly,
    /// Weekly periods
    Weekly,
    /// Daily periods
    Daily,
}

/// Individual period representation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Period {
    /// Period start date
    pub start: NaiveDate,
    /// Period end date
    pub end: NaiveDate,
    /// Period label (e.g., "2024-Q1", "2024-03")
    pub label: String,
}

impl Period {
    /// Create a new period
    pub fn new(start: NaiveDate, end: NaiveDate, label: String) -> Self {
        Self { start, end, label }
    }

    /// Check if a date falls within this period
    pub fn contains(&self, date: NaiveDate) -> bool {
        date >= self.start && date <= self.end
    }

    /// Get period duration in days
    pub fn duration_days(&self) -> i64 {
        (self.end - self.start).num_days() + 1
    }
}

impl PeriodIndex {
    /// Create a new period index
    pub fn new(periods: Vec<Period>, frequency: PeriodFrequency, name: Option<String>) -> Self {
        Self {
            periods,
            name,
            frequency,
        }
    }

    /// Create period range
    pub fn period_range(
        start: NaiveDate,
        end: NaiveDate,
        frequency: PeriodFrequency,
        name: Option<String>,
    ) -> Result<Self> {
        let mut periods = Vec::new();
        let mut current = start;

        while current <= end {
            let (period_end, label) = match frequency {
                PeriodFrequency::Annual => {
                    let year = current.year();
                    let period_start = NaiveDate::from_ymd_opt(year, 1, 1)
                        .ok_or_else(|| Error::InvalidValue("Invalid year".to_string()))?;
                    let period_end = NaiveDate::from_ymd_opt(year, 12, 31)
                        .ok_or_else(|| Error::InvalidValue("Invalid year".to_string()))?;
                    (period_end, format!("{}", year))
                }
                PeriodFrequency::Quarterly => {
                    let year = current.year();
                    let quarter = ((current.month() - 1) / 3) + 1;
                    let quarter_start_month = ((quarter - 1) * 3) + 1;
                    let quarter_end_month = quarter * 3;

                    let period_end = NaiveDate::from_ymd_opt(
                        year,
                        quarter_end_month,
                        match quarter_end_month {
                            3 => 31,
                            6 => 30,
                            9 => 30,
                            12 => 31,
                            _ => return Err(Error::InvalidValue("Invalid quarter".to_string())),
                        },
                    )
                    .ok_or_else(|| Error::InvalidValue("Invalid quarter end date".to_string()))?;

                    (period_end, format!("{}-Q{}", year, quarter))
                }
                PeriodFrequency::Monthly => {
                    let year = current.year();
                    let month = current.month();

                    // Get last day of month
                    let next_month = if month == 12 { 1 } else { month + 1 };
                    let next_year = if month == 12 { year + 1 } else { year };
                    let first_of_next = NaiveDate::from_ymd_opt(next_year, next_month, 1)
                        .ok_or_else(|| Error::InvalidValue("Invalid next month".to_string()))?;
                    let period_end = first_of_next
                        .pred_opt()
                        .ok_or_else(|| Error::InvalidValue("Invalid month end".to_string()))?;

                    (period_end, format!("{}-{:02}", year, month))
                }
                PeriodFrequency::Weekly => {
                    let period_end = current + Duration::days(6);
                    (
                        period_end.min(end),
                        format!("{}-W{:02}", current.year(), current.iso_week().week()),
                    )
                }
                PeriodFrequency::Daily => (current, current.format("%Y-%m-%d").to_string()),
            };

            periods.push(Period::new(current, period_end, label));

            // Move to next period
            current = match frequency {
                PeriodFrequency::Annual => NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                    .ok_or_else(|| Error::InvalidValue("Invalid next year".to_string()))?,
                PeriodFrequency::Quarterly => {
                    let quarter = ((current.month() - 1) / 3) + 1;
                    if quarter == 4 {
                        NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                    } else {
                        NaiveDate::from_ymd_opt(current.year(), ((quarter) * 3) + 1, 1)
                    }
                    .ok_or_else(|| Error::InvalidValue("Invalid next quarter".to_string()))?
                }
                PeriodFrequency::Monthly => if current.month() == 12 {
                    NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                } else {
                    NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1)
                }
                .ok_or_else(|| Error::InvalidValue("Invalid next month".to_string()))?,
                PeriodFrequency::Weekly => current + Duration::days(7),
                PeriodFrequency::Daily => current + Duration::days(1),
            };

            if current > end {
                break;
            }
        }

        Ok(Self::new(periods, frequency, name))
    }

    /// Find periods containing a specific date
    pub fn find_periods_containing(&self, date: NaiveDate) -> Vec<usize> {
        self.periods
            .iter()
            .enumerate()
            .filter_map(
                |(i, period)| {
                    if period.contains(date) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Get period labels
    pub fn labels(&self) -> Vec<&str> {
        self.periods.iter().map(|p| p.label.as_str()).collect()
    }
}

impl Index for PeriodIndex {
    fn len(&self) -> usize {
        self.periods.len()
    }

    fn to_string_vec(&self) -> Vec<String> {
        self.periods.iter().map(|p| p.label.clone()).collect()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    fn has_duplicates(&self) -> bool {
        let mut seen = HashSet::new();
        self.periods.iter().any(|p| !seen.insert(&p.label))
    }

    fn unique(&self) -> Result<Self> {
        let mut unique_periods: Vec<Period> = self.periods.clone();
        unique_periods.sort_by(|a, b| a.label.cmp(&b.label));
        unique_periods.dedup_by(|a, b| a.label == b.label);

        Ok(PeriodIndex::new(
            unique_periods,
            self.frequency.clone(),
            self.name.clone(),
        ))
    }

    fn sort(&self, ascending: bool) -> Result<(Self, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..self.periods.len()).collect();

        if ascending {
            indices.sort_by(|&a, &b| self.periods[a].start.cmp(&self.periods[b].start));
        } else {
            indices.sort_by(|&a, &b| self.periods[b].start.cmp(&self.periods[a].start));
        }

        let sorted_periods: Vec<Period> =
            indices.iter().map(|&i| self.periods[i].clone()).collect();
        let sorted_index =
            PeriodIndex::new(sorted_periods, self.frequency.clone(), self.name.clone());

        Ok((sorted_index, indices))
    }
}

/// Interval index for range-based and binned data indexing
#[derive(Debug, Clone)]
pub struct IntervalIndex {
    /// Interval values
    pub intervals: Vec<Interval>,
    /// Index name
    pub name: Option<String>,
    /// Whether intervals are closed on left, right, both, or neither
    pub closed: IntervalClosed,
}

/// Interval closed specification
#[derive(Debug, Clone, PartialEq)]
pub enum IntervalClosed {
    /// Intervals are closed on the left: [a, b)
    Left,
    /// Intervals are closed on the right: (a, b]
    Right,
    /// Intervals are closed on both sides: [a, b]
    Both,
    /// Intervals are open on both sides: (a, b)
    Neither,
}

/// Individual interval representation
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    /// Left boundary
    pub left: f64,
    /// Right boundary
    pub right: f64,
    /// Interval label
    pub label: String,
}

impl Interval {
    /// Create a new interval
    pub fn new(left: f64, right: f64) -> Self {
        let label = format!("({}, {})", left, right);
        Self { left, right, label }
    }

    /// Create interval with custom label
    pub fn with_label(left: f64, right: f64, label: String) -> Self {
        Self { left, right, label }
    }

    /// Check if a value falls within this interval
    pub fn contains(&self, value: f64, closed: &IntervalClosed) -> bool {
        match closed {
            IntervalClosed::Left => value >= self.left && value < self.right,
            IntervalClosed::Right => value > self.left && value <= self.right,
            IntervalClosed::Both => value >= self.left && value <= self.right,
            IntervalClosed::Neither => value > self.left && value < self.right,
        }
    }

    /// Get interval width
    pub fn width(&self) -> f64 {
        self.right - self.left
    }

    /// Get interval midpoint
    pub fn midpoint(&self) -> f64 {
        (self.left + self.right) / 2.0
    }
}

impl IntervalIndex {
    /// Create a new interval index
    pub fn new(intervals: Vec<Interval>, closed: IntervalClosed, name: Option<String>) -> Self {
        Self {
            intervals,
            name,
            closed,
        }
    }

    /// Create interval index from breaks
    pub fn from_breaks(
        breaks: Vec<f64>,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<Self> {
        if breaks.len() < 2 {
            return Err(Error::InvalidValue(
                "At least 2 breaks required".to_string(),
            ));
        }

        let mut sorted_breaks = breaks;
        sorted_breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let intervals: Vec<Interval> = sorted_breaks
            .windows(2)
            .map(|window| Interval::new(window[0], window[1]))
            .collect();

        Ok(Self::new(intervals, closed, name))
    }

    /// Create equal-width bins
    pub fn cut(
        values: &[f64],
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::InvalidValue("Cannot cut empty values".to_string()));
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            return Err(Error::InvalidValue("All values are identical".to_string()));
        }

        let width = (max_val - min_val) / bins as f64;
        let mut breaks = Vec::with_capacity(bins + 1);

        for i in 0..=bins {
            breaks.push(min_val + (i as f64 * width));
        }

        // Ensure the last break includes the maximum value
        breaks[bins] = max_val + f64::EPSILON;

        Self::from_breaks(breaks, closed, name)
    }

    /// Create quantile-based bins
    pub fn qcut(
        values: &[f64],
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::InvalidValue("Cannot qcut empty values".to_string()));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut breaks = Vec::with_capacity(bins + 1);
        breaks.push(sorted_values[0]);

        for i in 1..bins {
            let index = (i as f64 / bins as f64 * sorted_values.len() as f64) as usize;
            let clamped_index = index.min(sorted_values.len() - 1);
            breaks.push(sorted_values[clamped_index]);
        }

        breaks.push(sorted_values[sorted_values.len() - 1] + f64::EPSILON);

        // Remove duplicates while preserving order
        let mut unique_breaks = Vec::new();
        for &brk in &breaks {
            if unique_breaks.is_empty()
                || (brk - unique_breaks[unique_breaks.len() - 1] as f64).abs() > f64::EPSILON
            {
                unique_breaks.push(brk);
            }
        }

        if unique_breaks.len() < 2 {
            return Err(Error::InvalidValue(
                "Not enough unique values for quantile bins".to_string(),
            ));
        }

        Self::from_breaks(unique_breaks, closed, name)
    }

    /// Find intervals containing a specific value
    pub fn find_intervals_containing(&self, value: f64) -> Vec<usize> {
        self.intervals
            .iter()
            .enumerate()
            .filter_map(|(i, interval)| {
                if interval.contains(value, &self.closed) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get interval labels
    pub fn labels(&self) -> Vec<&str> {
        self.intervals.iter().map(|i| i.label.as_str()).collect()
    }

    /// Get interval midpoints
    pub fn midpoints(&self) -> Vec<f64> {
        self.intervals.iter().map(|i| i.midpoint()).collect()
    }

    /// Get interval widths
    pub fn widths(&self) -> Vec<f64> {
        self.intervals.iter().map(|i| i.width()).collect()
    }
}

impl Index for IntervalIndex {
    fn len(&self) -> usize {
        self.intervals.len()
    }

    fn to_string_vec(&self) -> Vec<String> {
        self.intervals.iter().map(|i| i.label.clone()).collect()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    fn has_duplicates(&self) -> bool {
        let mut seen = HashSet::new();
        self.intervals.iter().any(|i| !seen.insert(&i.label))
    }

    fn unique(&self) -> Result<Self> {
        let mut unique_intervals: Vec<Interval> = self.intervals.clone();
        unique_intervals.sort_by(|a, b| {
            a.left
                .partial_cmp(&b.left)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        unique_intervals.dedup_by(|a, b| a.label == b.label);

        Ok(IntervalIndex::new(
            unique_intervals,
            self.closed.clone(),
            self.name.clone(),
        ))
    }

    fn sort(&self, ascending: bool) -> Result<(Self, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..self.intervals.len()).collect();

        if ascending {
            indices.sort_by(|&a, &b| {
                self.intervals[a]
                    .left
                    .partial_cmp(&self.intervals[b].left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indices.sort_by(|&a, &b| {
                self.intervals[b]
                    .left
                    .partial_cmp(&self.intervals[a].left)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let sorted_intervals: Vec<Interval> =
            indices.iter().map(|&i| self.intervals[i].clone()).collect();
        let sorted_index =
            IntervalIndex::new(sorted_intervals, self.closed.clone(), self.name.clone());

        Ok((sorted_index, indices))
    }
}

/// Categorical index for memory-optimized categorical indexing
#[derive(Debug, Clone)]
pub struct CategoricalIndex {
    /// Category codes (indices into categories)
    pub codes: Vec<Option<usize>>,
    /// Unique categories
    pub categories: Vec<String>,
    /// Index name
    pub name: Option<String>,
    /// Whether categories are ordered
    pub ordered: bool,
}

impl CategoricalIndex {
    /// Create a new categorical index
    pub fn new(values: Vec<String>, name: Option<String>, ordered: bool) -> Self {
        let mut categories = Vec::new();
        let mut category_map = HashMap::new();
        let mut codes = Vec::with_capacity(values.len());

        for value in values {
            let code = if let Some(&existing_code) = category_map.get(&value) {
                existing_code
            } else {
                let new_code = categories.len();
                categories.push(value.clone());
                category_map.insert(value, new_code);
                new_code
            };
            codes.push(Some(code));
        }

        Self {
            codes,
            categories,
            name,
            ordered,
        }
    }

    /// Create categorical index with predefined categories
    pub fn with_categories(
        values: Vec<String>,
        categories: Vec<String>,
        name: Option<String>,
        ordered: bool,
    ) -> Result<Self> {
        let category_map: HashMap<String, usize> = categories
            .iter()
            .enumerate()
            .map(|(i, cat)| (cat.clone(), i))
            .collect();

        let mut codes = Vec::with_capacity(values.len());

        for value in values {
            let code = category_map.get(&value).copied();
            codes.push(code);
        }

        Ok(Self {
            codes,
            categories,
            name,
            ordered,
        })
    }

    /// Get category values
    pub fn values(&self) -> Vec<Option<String>> {
        self.codes
            .iter()
            .map(|&code| code.map(|c| self.categories[c].clone()))
            .collect()
    }

    /// Get category counts
    pub fn value_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for &code in &self.codes {
            if let Some(code) = code {
                let category = &self.categories[code];
                *counts.entry(category.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Add new categories
    pub fn add_categories(&mut self, new_categories: Vec<String>) -> Result<()> {
        let existing_set: HashSet<_> = self.categories.iter().collect();
        let categories_to_add: Vec<String> = new_categories
            .into_iter()
            .filter(|cat| !existing_set.contains(cat))
            .collect();

        self.categories.extend(categories_to_add);
        Ok(())
    }

    /// Remove categories (and set corresponding codes to None)
    pub fn remove_categories(&mut self, categories_to_remove: Vec<String>) -> Result<()> {
        let remove_set: HashSet<_> = categories_to_remove.iter().collect();
        let mut old_to_new_index = HashMap::new();
        let mut new_categories = Vec::new();

        for (old_idx, category) in self.categories.iter().enumerate() {
            if !remove_set.contains(category) {
                old_to_new_index.insert(old_idx, new_categories.len());
                new_categories.push(category.clone());
            }
        }

        // Update codes
        for code in &mut self.codes {
            if let Some(old_code) = *code {
                *code = old_to_new_index.get(&old_code).copied();
            }
        }

        self.categories = new_categories;
        Ok(())
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let codes_size = self.codes.len() * std::mem::size_of::<Option<usize>>();
        let categories_size: usize = self.categories.iter().map(|s| s.len()).sum();
        let name_size = self.name.as_ref().map_or(0, |s| s.len());

        codes_size + categories_size + name_size + std::mem::size_of::<Self>()
    }
}

impl Index for CategoricalIndex {
    fn len(&self) -> usize {
        self.codes.len()
    }

    fn to_string_vec(&self) -> Vec<String> {
        self.codes
            .iter()
            .map(|&code| match code {
                Some(c) => self.categories[c].clone(),
                None => "NaN".to_string(),
            })
            .collect()
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn set_name(&mut self, name: Option<String>) {
        self.name = name;
    }

    fn has_duplicates(&self) -> bool {
        let mut seen = HashSet::new();
        self.codes.iter().any(|&code| !seen.insert(code))
    }

    fn unique(&self) -> Result<Self> {
        let mut unique_codes = Vec::new();
        let mut seen = HashSet::new();

        for &code in &self.codes {
            if seen.insert(code) {
                unique_codes.push(code);
            }
        }

        Ok(CategoricalIndex {
            codes: unique_codes,
            categories: self.categories.clone(),
            name: self.name.clone(),
            ordered: self.ordered,
        })
    }

    fn sort(&self, ascending: bool) -> Result<(Self, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..self.codes.len()).collect();

        indices.sort_by(|&a, &b| match (self.codes[a], self.codes[b]) {
            (Some(code_a), Some(code_b)) => {
                let cat_a = &self.categories[code_a];
                let cat_b = &self.categories[code_b];
                if ascending {
                    cat_a.cmp(cat_b)
                } else {
                    cat_b.cmp(cat_a)
                }
            }
            (Some(_), None) => {
                if ascending {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            }
            (None, Some(_)) => {
                if ascending {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Less
                }
            }
            (None, None) => std::cmp::Ordering::Equal,
        });

        let sorted_codes: Vec<Option<usize>> = indices.iter().map(|&i| self.codes[i]).collect();
        let sorted_index = CategoricalIndex {
            codes: sorted_codes,
            categories: self.categories.clone(),
            name: self.name.clone(),
            ordered: self.ordered,
        };

        Ok((sorted_index, indices))
    }
}

/// Index set operations implementation
pub struct IndexOperations;

impl IndexOperations {
    /// Union of two string-based indexes (generic implementation)
    pub fn union_string_indexes(left: &[String], right: &[String]) -> Vec<String> {
        let mut result = left.to_vec();
        let left_set: HashSet<_> = left.iter().collect();

        for item in right {
            if !left_set.contains(item) {
                result.push(item.clone());
            }
        }

        result
    }

    /// Intersection of two string-based indexes
    pub fn intersection_string_indexes(left: &[String], right: &[String]) -> Vec<String> {
        let right_set: HashSet<_> = right.iter().collect();

        left.iter()
            .filter(|item| right_set.contains(item))
            .cloned()
            .collect()
    }

    /// Difference of two string-based indexes (left - right)
    pub fn difference_string_indexes(left: &[String], right: &[String]) -> Vec<String> {
        let right_set: HashSet<_> = right.iter().collect();

        left.iter()
            .filter(|item| !right_set.contains(item))
            .cloned()
            .collect()
    }

    /// Symmetric difference of two string-based indexes
    pub fn symmetric_difference_string_indexes(left: &[String], right: &[String]) -> Vec<String> {
        let left_set: HashSet<_> = left.iter().collect();
        let right_set: HashSet<_> = right.iter().collect();

        let mut result = Vec::new();

        // Items in left but not in right
        for item in left {
            if !right_set.contains(item) {
                result.push(item.clone());
            }
        }

        // Items in right but not in left
        for item in right {
            if !left_set.contains(item) {
                result.push(item.clone());
            }
        }

        result
    }
}

/// Extension trait to add advanced indexing to DataFrame
pub trait AdvancedIndexingExt {
    /// Set datetime index
    fn set_datetime_index(
        &self,
        column: &str,
        name: Option<String>,
    ) -> Result<(DataFrame, DatetimeIndex)>;

    /// Set period index
    fn set_period_index(
        &self,
        start_date: NaiveDate,
        frequency: PeriodFrequency,
        name: Option<String>,
    ) -> Result<(DataFrame, PeriodIndex)>;

    /// Set interval index by cutting a column
    fn set_interval_index_cut(
        &self,
        column: &str,
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<(DataFrame, IntervalIndex)>;

    /// Set interval index by quantile cutting a column
    fn set_interval_index_qcut(
        &self,
        column: &str,
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<(DataFrame, IntervalIndex)>;

    /// Set categorical index
    fn set_categorical_index(
        &self,
        column: &str,
        ordered: bool,
        name: Option<String>,
    ) -> Result<(DataFrame, CategoricalIndex)>;
}

impl AdvancedIndexingExt for DataFrame {
    fn set_datetime_index(
        &self,
        column: &str,
        name: Option<String>,
    ) -> Result<(DataFrame, DatetimeIndex)> {
        let column_values = self.get_column_string_values(column)?;
        let mut datetime_values = Vec::new();

        for value in &column_values {
            let dt = NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S")
                .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d"))
                .map_err(|_| Error::InvalidValue(format!("Cannot parse datetime: {}", value)))?;
            datetime_values.push(dt);
        }

        let index = DatetimeIndex::new(datetime_values, name);
        let df_without_index = self.drop_columns(&[column.to_string()])?;

        Ok((df_without_index, index))
    }

    fn set_period_index(
        &self,
        start_date: NaiveDate,
        frequency: PeriodFrequency,
        name: Option<String>,
    ) -> Result<(DataFrame, PeriodIndex)> {
        let row_count = self.row_count();

        // Calculate end date based on row count and frequency
        let end_date = match frequency {
            PeriodFrequency::Daily => start_date + Duration::days(row_count as i64 - 1),
            PeriodFrequency::Weekly => start_date + Duration::weeks(row_count as i64 - 1),
            PeriodFrequency::Monthly => {
                let mut current = start_date;
                for _ in 0..(row_count - 1) {
                    current = if current.month() == 12 {
                        NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                    } else {
                        NaiveDate::from_ymd_opt(current.year(), current.month() + 1, 1)
                    }
                    .ok_or_else(|| Error::InvalidValue("Invalid date calculation".to_string()))?;
                }
                current
            }
            PeriodFrequency::Quarterly => {
                let mut current = start_date;
                for _ in 0..(row_count - 1) {
                    let quarter = ((current.month() - 1) / 3) + 1;
                    current = if quarter == 4 {
                        NaiveDate::from_ymd_opt(current.year() + 1, 1, 1)
                    } else {
                        NaiveDate::from_ymd_opt(current.year(), ((quarter) * 3) + 1, 1)
                    }
                    .ok_or_else(|| {
                        Error::InvalidValue("Invalid quarter calculation".to_string())
                    })?;
                }
                current
            }
            PeriodFrequency::Annual => NaiveDate::from_ymd_opt(
                start_date.year() + row_count as i32 - 1,
                start_date.month(),
                start_date.day(),
            )
            .ok_or_else(|| Error::InvalidValue("Invalid year calculation".to_string()))?,
        };

        let index = PeriodIndex::period_range(start_date, end_date, frequency, name)?;
        Ok((self.clone(), index))
    }

    fn set_interval_index_cut(
        &self,
        column: &str,
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<(DataFrame, IntervalIndex)> {
        let column_values = self.get_column_string_values(column)?;
        let mut numeric_values = Vec::new();

        for value in &column_values {
            let num = value
                .parse::<f64>()
                .map_err(|_| Error::InvalidValue(format!("Cannot parse number: {}", value)))?;
            numeric_values.push(num);
        }

        let index = IntervalIndex::cut(&numeric_values, bins, closed, name)?;
        let df_without_index = self.drop_columns(&[column.to_string()])?;

        Ok((df_without_index, index))
    }

    fn set_interval_index_qcut(
        &self,
        column: &str,
        bins: usize,
        closed: IntervalClosed,
        name: Option<String>,
    ) -> Result<(DataFrame, IntervalIndex)> {
        let column_values = self.get_column_string_values(column)?;
        let mut numeric_values = Vec::new();

        for value in &column_values {
            let num = value
                .parse::<f64>()
                .map_err(|_| Error::InvalidValue(format!("Cannot parse number: {}", value)))?;
            numeric_values.push(num);
        }

        let index = IntervalIndex::qcut(&numeric_values, bins, closed, name)?;
        let df_without_index = self.drop_columns(&[column.to_string()])?;

        Ok((df_without_index, index))
    }

    fn set_categorical_index(
        &self,
        column: &str,
        ordered: bool,
        name: Option<String>,
    ) -> Result<(DataFrame, CategoricalIndex)> {
        let column_values = self.get_column_string_values(column)?;
        let index = CategoricalIndex::new(column_values, name, ordered);
        let df_without_index = self.drop_columns(&[column.to_string()])?;

        Ok((df_without_index, index))
    }
}

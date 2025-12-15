//! Windowed aggregations for streaming data processing
//!
//! This module provides sophisticated window types and windowed aggregation
//! capabilities for streaming data, including:
//!
//! - Tumbling windows (fixed-size, non-overlapping)
//! - Sliding windows (fixed-size, overlapping)
//! - Session windows (gap-based)
//! - Count-based windows
//! - Custom aggregation functions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::StreamRecord;
use crate::error::{Error, Result};

/// Types of windows for stream processing
#[derive(Debug, Clone)]
pub enum WindowType {
    /// Fixed-size non-overlapping windows
    Tumbling {
        /// Window size in duration
        size: Duration,
    },
    /// Fixed-size overlapping windows
    Sliding {
        /// Window size in duration
        size: Duration,
        /// Slide interval
        slide: Duration,
    },
    /// Variable-size windows based on activity gaps
    Session {
        /// Maximum gap between events
        gap: Duration,
        /// Maximum session duration (optional)
        max_duration: Option<Duration>,
    },
    /// Windows based on record count
    Count {
        /// Number of records per window
        size: usize,
        /// Slide by count (for sliding count windows)
        slide: Option<usize>,
    },
    /// Global window (all records in a single window)
    Global,
}

impl Default for WindowType {
    fn default() -> Self {
        WindowType::Tumbling {
            size: Duration::from_secs(60),
        }
    }
}

/// Configuration for windowed processing
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Type of window
    pub window_type: WindowType,
    /// Allowed lateness for late-arriving records
    pub allowed_lateness: Duration,
    /// Whether to emit on every record (vs only on window close)
    pub emit_on_every_record: bool,
    /// Whether to include partial windows
    pub include_partial_windows: bool,
    /// Watermark delay for event-time processing
    pub watermark_delay: Duration,
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig {
            window_type: WindowType::default(),
            allowed_lateness: Duration::from_secs(0),
            emit_on_every_record: false,
            include_partial_windows: false,
            watermark_delay: Duration::from_secs(1),
        }
    }
}

/// Builder for WindowConfig
pub struct WindowConfigBuilder {
    config: WindowConfig,
}

impl WindowConfigBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        WindowConfigBuilder {
            config: WindowConfig::default(),
        }
    }

    /// Sets a tumbling window
    pub fn tumbling(mut self, size: Duration) -> Self {
        self.config.window_type = WindowType::Tumbling { size };
        self
    }

    /// Sets a sliding window
    pub fn sliding(mut self, size: Duration, slide: Duration) -> Self {
        self.config.window_type = WindowType::Sliding { size, slide };
        self
    }

    /// Sets a session window
    pub fn session(mut self, gap: Duration, max_duration: Option<Duration>) -> Self {
        self.config.window_type = WindowType::Session { gap, max_duration };
        self
    }

    /// Sets a count-based window
    pub fn count(mut self, size: usize, slide: Option<usize>) -> Self {
        self.config.window_type = WindowType::Count { size, slide };
        self
    }

    /// Sets a global window
    pub fn global(mut self) -> Self {
        self.config.window_type = WindowType::Global;
        self
    }

    /// Sets allowed lateness
    pub fn allowed_lateness(mut self, lateness: Duration) -> Self {
        self.config.allowed_lateness = lateness;
        self
    }

    /// Sets emit on every record
    pub fn emit_on_every_record(mut self, emit: bool) -> Self {
        self.config.emit_on_every_record = emit;
        self
    }

    /// Sets include partial windows
    pub fn include_partial_windows(mut self, include: bool) -> Self {
        self.config.include_partial_windows = include;
        self
    }

    /// Sets watermark delay
    pub fn watermark_delay(mut self, delay: Duration) -> Self {
        self.config.watermark_delay = delay;
        self
    }

    /// Builds the config
    pub fn build(self) -> WindowConfig {
        self.config
    }
}

impl Default for WindowConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a time window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Window start time
    pub start: DateTime<Utc>,
    /// Window end time
    pub end: DateTime<Utc>,
    /// Window identifier
    pub id: String,
}

impl TimeWindow {
    /// Creates a new time window
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        let id = format!("{}-{}", start.timestamp(), end.timestamp());
        TimeWindow { start, end, id }
    }

    /// Checks if a timestamp falls within this window
    pub fn contains(&self, timestamp: DateTime<Utc>) -> bool {
        timestamp >= self.start && timestamp < self.end
    }

    /// Gets the duration of the window
    pub fn duration(&self) -> Duration {
        let diff = self.end.signed_duration_since(self.start);
        Duration::from_millis(diff.num_milliseconds().max(0) as u64)
    }
}

/// Aggregation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowAggregation {
    /// Sum of values
    Sum,
    /// Average (mean) of values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of values
    Count,
    /// First value in window
    First,
    /// Last value in window
    Last,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Median (50th percentile)
    Median,
    /// Percentile (specified)
    Percentile(u8),
}

/// Result of a windowed aggregation
#[derive(Debug, Clone)]
pub struct WindowResult {
    /// The window this result belongs to
    pub window: TimeWindow,
    /// Aggregated values by column
    pub values: HashMap<String, f64>,
    /// Number of records in the window
    pub count: usize,
    /// When the result was emitted
    pub emitted_at: DateTime<Utc>,
}

/// State for incremental aggregation
#[derive(Debug, Clone)]
struct AggregationState {
    /// Running sum
    sum: f64,
    /// Running count
    count: usize,
    /// Running min
    min: f64,
    /// Running max
    max: f64,
    /// First value seen
    first: Option<f64>,
    /// Last value seen
    last: Option<f64>,
    /// Sum of squares for variance calculation
    sum_squares: f64,
    /// All values for percentile calculations
    values: Vec<f64>,
}

impl AggregationState {
    fn new() -> Self {
        AggregationState {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            first: None,
            last: None,
            sum_squares: 0.0,
            values: Vec::new(),
        }
    }

    fn update(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if self.first.is_none() {
            self.first = Some(value);
        }
        self.last = Some(value);

        self.sum_squares += value * value;
        self.values.push(value);
    }

    fn compute(&self, agg: WindowAggregation) -> f64 {
        match agg {
            WindowAggregation::Sum => self.sum,
            WindowAggregation::Avg => {
                if self.count > 0 {
                    self.sum / self.count as f64
                } else {
                    0.0
                }
            }
            WindowAggregation::Min => {
                if self.count > 0 {
                    self.min
                } else {
                    0.0
                }
            }
            WindowAggregation::Max => {
                if self.count > 0 {
                    self.max
                } else {
                    0.0
                }
            }
            WindowAggregation::Count => self.count as f64,
            WindowAggregation::First => self.first.unwrap_or(0.0),
            WindowAggregation::Last => self.last.unwrap_or(0.0),
            WindowAggregation::StdDev => {
                if self.count > 1 {
                    let mean = self.sum / self.count as f64;
                    let variance = (self.sum_squares / self.count as f64) - (mean * mean);
                    variance.max(0.0).sqrt()
                } else {
                    0.0
                }
            }
            WindowAggregation::Variance => {
                if self.count > 1 {
                    let mean = self.sum / self.count as f64;
                    (self.sum_squares / self.count as f64) - (mean * mean)
                } else {
                    0.0
                }
            }
            WindowAggregation::Median => self.compute_percentile(50),
            WindowAggregation::Percentile(p) => self.compute_percentile(p),
        }
    }

    fn compute_percentile(&self, percentile: u8) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p = (percentile as f64 / 100.0).clamp(0.0, 1.0);
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx]
    }
}

/// Manages windows and aggregations for streaming data
#[derive(Debug)]
pub struct WindowedAggregator {
    /// Configuration
    config: WindowConfig,
    /// Column to aggregate
    column: String,
    /// Aggregation function
    aggregation: WindowAggregation,
    /// Active windows and their states
    windows: HashMap<String, (TimeWindow, AggregationState)>,
    /// Closed window results
    results: Vec<WindowResult>,
    /// Current watermark
    watermark: DateTime<Utc>,
    /// Record count for count-based windows
    record_count: usize,
    /// Session start times by key
    session_starts: HashMap<String, DateTime<Utc>>,
    /// Last record time by key
    last_record_times: HashMap<String, DateTime<Utc>>,
}

impl WindowedAggregator {
    /// Creates a new windowed aggregator
    pub fn new(config: WindowConfig, column: &str, aggregation: WindowAggregation) -> Self {
        WindowedAggregator {
            config,
            column: column.to_string(),
            aggregation,
            windows: HashMap::new(),
            results: Vec::new(),
            watermark: Utc::now() - chrono::Duration::days(1),
            record_count: 0,
            session_starts: HashMap::new(),
            last_record_times: HashMap::new(),
        }
    }

    /// Processes a record and returns any completed window results
    pub fn process(&mut self, record: &StreamRecord) -> Result<Vec<WindowResult>> {
        let mut completed_results = Vec::new();

        // Get the event time (or use current time if not available)
        let event_time = self.get_event_time(record);

        // Update watermark
        self.update_watermark(event_time);

        // Get the value to aggregate
        let value = record
            .fields
            .get(&self.column)
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);

        self.record_count += 1;

        // Handle based on window type
        match &self.config.window_type {
            WindowType::Tumbling { size } => {
                completed_results.extend(self.process_tumbling(event_time, value, *size)?);
            }
            WindowType::Sliding { size, slide } => {
                completed_results.extend(self.process_sliding(event_time, value, *size, *slide)?);
            }
            WindowType::Session { gap, max_duration } => {
                completed_results.extend(self.process_session(
                    event_time,
                    value,
                    *gap,
                    *max_duration,
                )?);
            }
            WindowType::Count { size, slide } => {
                completed_results.extend(self.process_count(event_time, value, *size, *slide)?);
            }
            WindowType::Global => {
                self.process_global(value);
            }
        }

        // Close windows that are past the watermark
        completed_results.extend(self.close_expired_windows()?);

        Ok(completed_results)
    }

    /// Gets the event time from a record
    fn get_event_time(&self, record: &StreamRecord) -> DateTime<Utc> {
        // Try to get event_time field from record
        if let Some(ts_str) = record.fields.get("event_time") {
            if let Ok(ts) = ts_str.parse::<i64>() {
                return DateTime::from_timestamp(ts, 0).unwrap_or_else(Utc::now);
            }
        }

        // Fall back to processing time
        Utc::now()
    }

    /// Updates the watermark
    fn update_watermark(&mut self, event_time: DateTime<Utc>) {
        let watermark_candidate = event_time
            - chrono::Duration::from_std(self.config.watermark_delay)
                .unwrap_or(chrono::Duration::zero());

        if watermark_candidate > self.watermark {
            self.watermark = watermark_candidate;
        }
    }

    /// Processes a tumbling window
    fn process_tumbling(
        &mut self,
        event_time: DateTime<Utc>,
        value: f64,
        size: Duration,
    ) -> Result<Vec<WindowResult>> {
        let window_start = self.align_to_window(event_time, size);
        let window_end =
            window_start + chrono::Duration::from_std(size).unwrap_or(chrono::Duration::zero());

        let window = TimeWindow::new(window_start, window_end);
        let window_id = window.id.clone();

        // Get or create window state
        let (_, state) = self
            .windows
            .entry(window_id.clone())
            .or_insert_with(|| (window.clone(), AggregationState::new()));

        state.update(value);

        // Emit if configured to emit on every record
        if self.config.emit_on_every_record {
            let result = WindowResult {
                window: window.clone(),
                values: [(self.column.clone(), state.compute(self.aggregation))]
                    .into_iter()
                    .collect(),
                count: state.count,
                emitted_at: Utc::now(),
            };
            return Ok(vec![result]);
        }

        Ok(vec![])
    }

    /// Processes a sliding window
    fn process_sliding(
        &mut self,
        event_time: DateTime<Utc>,
        value: f64,
        size: Duration,
        slide: Duration,
    ) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

        // Calculate all windows this record belongs to
        let slide_duration = chrono::Duration::from_std(slide).unwrap_or(chrono::Duration::zero());
        let size_duration = chrono::Duration::from_std(size).unwrap_or(chrono::Duration::zero());

        // Align to slide boundary
        let base_start = self.align_to_window(event_time, slide);

        // Go back to find all windows containing this event
        let mut window_start = base_start;
        while window_start + size_duration > event_time {
            let window_end = window_start + size_duration;
            let window = TimeWindow::new(window_start, window_end);
            let window_id = window.id.clone();

            let (_, state) = self
                .windows
                .entry(window_id)
                .or_insert_with(|| (window.clone(), AggregationState::new()));

            state.update(value);

            if self.config.emit_on_every_record {
                let result = WindowResult {
                    window: window.clone(),
                    values: [(self.column.clone(), state.compute(self.aggregation))]
                        .into_iter()
                        .collect(),
                    count: state.count,
                    emitted_at: Utc::now(),
                };
                results.push(result);
            }

            window_start = window_start - slide_duration;
        }

        Ok(results)
    }

    /// Processes a session window
    fn process_session(
        &mut self,
        event_time: DateTime<Utc>,
        value: f64,
        gap: Duration,
        max_duration: Option<Duration>,
    ) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();
        let key = "default".to_string(); // Could be keyed by record field

        let gap_duration = chrono::Duration::from_std(gap).unwrap_or(chrono::Duration::zero());

        // Check if we need to start a new session
        let last_time = self.last_record_times.get(&key).cloned();
        let session_start = self.session_starts.get(&key).cloned();

        let (should_start_new, close_old) = if let Some(last) = last_time {
            let time_since_last = event_time.signed_duration_since(last);
            if time_since_last > gap_duration {
                (true, session_start.is_some())
            } else if let (Some(start), Some(max)) = (session_start, max_duration) {
                let max_dur = chrono::Duration::from_std(max).unwrap_or(chrono::Duration::zero());
                if event_time.signed_duration_since(start) > max_dur {
                    (true, true)
                } else {
                    (false, false)
                }
            } else {
                (false, false)
            }
        } else {
            (true, false)
        };

        // Close old session if needed
        if close_old {
            if let Some(start) = session_start {
                let window = TimeWindow::new(start, last_time.unwrap_or(event_time));
                let window_id = format!("session_{}", start.timestamp());

                if let Some((_, state)) = self.windows.remove(&window_id) {
                    let result = WindowResult {
                        window,
                        values: [(self.column.clone(), state.compute(self.aggregation))]
                            .into_iter()
                            .collect(),
                        count: state.count,
                        emitted_at: Utc::now(),
                    };
                    results.push(result);
                }
            }
        }

        // Start new session if needed
        if should_start_new {
            self.session_starts.insert(key.clone(), event_time);
        }

        // Update current session
        let session_start = self.session_starts.get(&key).cloned().unwrap_or(event_time);
        let window_id = format!("session_{}", session_start.timestamp());

        let window = TimeWindow::new(session_start, event_time);
        let (_, state) = self
            .windows
            .entry(window_id)
            .or_insert_with(|| (window.clone(), AggregationState::new()));

        state.update(value);
        self.last_record_times.insert(key, event_time);

        Ok(results)
    }

    /// Processes a count-based window
    fn process_count(
        &mut self,
        event_time: DateTime<Utc>,
        value: f64,
        size: usize,
        slide: Option<usize>,
    ) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();
        let slide = slide.unwrap_or(size);

        // Calculate window indices this record belongs to
        let record_idx = self.record_count - 1;
        let first_window_idx = record_idx / slide;

        // For tumbling windows (slide == size), only one window
        // For sliding windows, multiple windows
        let windows_per_size = (size + slide - 1) / slide;

        for i in 0..windows_per_size {
            if first_window_idx < i {
                continue;
            }

            let window_idx = first_window_idx - i;
            let window_start_count = window_idx * slide;
            let window_end_count = window_start_count + size;

            // Check if this record belongs to this window
            if record_idx >= window_start_count && record_idx < window_end_count {
                let window_id = format!("count_{}", window_idx);

                let window = TimeWindow::new(
                    event_time - chrono::Duration::seconds(size as i64),
                    event_time,
                );

                let (_, state) = self
                    .windows
                    .entry(window_id.clone())
                    .or_insert_with(|| (window.clone(), AggregationState::new()));

                state.update(value);

                // Check if window is complete
                if state.count >= size {
                    let result = WindowResult {
                        window: window.clone(),
                        values: [(self.column.clone(), state.compute(self.aggregation))]
                            .into_iter()
                            .collect(),
                        count: state.count,
                        emitted_at: Utc::now(),
                    };
                    results.push(result);

                    // Remove completed window (for tumbling)
                    if slide == size {
                        self.windows.remove(&window_id);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Processes a global window
    fn process_global(&mut self, value: f64) {
        let window_id = "global".to_string();

        let (_, state) = self.windows.entry(window_id).or_insert_with(|| {
            let window = TimeWindow::new(
                DateTime::from_timestamp(0, 0).unwrap(),
                DateTime::from_timestamp(i64::MAX / 2, 0).unwrap(),
            );
            (window, AggregationState::new())
        });

        state.update(value);
    }

    /// Closes windows that are past the watermark
    fn close_expired_windows(&mut self) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

        let watermark_with_lateness = self.watermark
            - chrono::Duration::from_std(self.config.allowed_lateness)
                .unwrap_or(chrono::Duration::zero());

        // Find and close expired windows
        let expired_ids: Vec<String> = self
            .windows
            .iter()
            .filter(|(_, (window, _))| window.end <= watermark_with_lateness)
            .map(|(id, _)| id.clone())
            .collect();

        for window_id in expired_ids {
            if let Some((window, state)) = self.windows.remove(&window_id) {
                let result = WindowResult {
                    window,
                    values: [(self.column.clone(), state.compute(self.aggregation))]
                        .into_iter()
                        .collect(),
                    count: state.count,
                    emitted_at: Utc::now(),
                };
                self.results.push(result.clone());
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Aligns a timestamp to a window boundary
    fn align_to_window(&self, timestamp: DateTime<Utc>, window_size: Duration) -> DateTime<Utc> {
        let millis = timestamp.timestamp_millis();
        let window_millis = window_size.as_millis() as i64;
        let aligned_millis = (millis / window_millis) * window_millis;

        DateTime::from_timestamp_millis(aligned_millis).unwrap_or(timestamp)
    }

    /// Flushes all active windows and returns results
    pub fn flush(&mut self) -> Vec<WindowResult> {
        let mut results = Vec::new();

        for (_, (window, state)) in self.windows.drain() {
            if state.count > 0 || self.config.include_partial_windows {
                let result = WindowResult {
                    window,
                    values: [(self.column.clone(), state.compute(self.aggregation))]
                        .into_iter()
                        .collect(),
                    count: state.count,
                    emitted_at: Utc::now(),
                };
                results.push(result);
            }
        }

        results
    }

    /// Gets all completed window results
    pub fn results(&self) -> &[WindowResult] {
        &self.results
    }

    /// Gets the current watermark
    pub fn watermark(&self) -> DateTime<Utc> {
        self.watermark
    }
}

/// Multi-column windowed aggregator
#[derive(Debug)]
pub struct MultiColumnAggregator {
    /// Configuration
    config: WindowConfig,
    /// Aggregations by column
    aggregations: HashMap<String, WindowAggregation>,
    /// Active windows and their states
    windows: HashMap<String, (TimeWindow, HashMap<String, AggregationState>)>,
    /// Completed results
    results: Vec<WindowResult>,
    /// Current watermark
    watermark: DateTime<Utc>,
}

impl MultiColumnAggregator {
    /// Creates a new multi-column aggregator
    pub fn new(config: WindowConfig) -> Self {
        MultiColumnAggregator {
            config,
            aggregations: HashMap::new(),
            windows: HashMap::new(),
            results: Vec::new(),
            watermark: Utc::now() - chrono::Duration::days(1),
        }
    }

    /// Adds an aggregation for a column
    pub fn add_aggregation(&mut self, column: &str, agg: WindowAggregation) -> &mut Self {
        self.aggregations.insert(column.to_string(), agg);
        self
    }

    /// Processes a record
    pub fn process(&mut self, record: &StreamRecord) -> Result<Vec<WindowResult>> {
        let event_time = Utc::now(); // Simplified - could extract from record

        // Update watermark
        let watermark_candidate = event_time
            - chrono::Duration::from_std(self.config.watermark_delay)
                .unwrap_or(chrono::Duration::zero());
        if watermark_candidate > self.watermark {
            self.watermark = watermark_candidate;
        }

        // Handle tumbling windows (simplified)
        if let WindowType::Tumbling { size } = &self.config.window_type {
            let millis = event_time.timestamp_millis();
            let window_millis = size.as_millis() as i64;
            let window_start_millis = (millis / window_millis) * window_millis;
            let window_start =
                DateTime::from_timestamp_millis(window_start_millis).unwrap_or(event_time);
            let window_end = window_start
                + chrono::Duration::from_std(*size).unwrap_or(chrono::Duration::zero());

            let window = TimeWindow::new(window_start, window_end);
            let window_id = window.id.clone();

            // Get or create window state
            let (_, states) = self
                .windows
                .entry(window_id)
                .or_insert_with(|| (window.clone(), HashMap::new()));

            // Update states for each aggregated column
            for (column, _) in &self.aggregations {
                let state = states
                    .entry(column.clone())
                    .or_insert_with(AggregationState::new);

                if let Some(value) = record.fields.get(column) {
                    if let Ok(v) = value.parse::<f64>() {
                        state.update(v);
                    }
                }
            }
        }

        // Close expired windows
        self.close_expired_windows()
    }

    /// Closes windows past the watermark
    fn close_expired_windows(&mut self) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

        let watermark_with_lateness = self.watermark
            - chrono::Duration::from_std(self.config.allowed_lateness)
                .unwrap_or(chrono::Duration::zero());

        let expired_ids: Vec<String> = self
            .windows
            .iter()
            .filter(|(_, (window, _))| window.end <= watermark_with_lateness)
            .map(|(id, _)| id.clone())
            .collect();

        for window_id in expired_ids {
            if let Some((window, states)) = self.windows.remove(&window_id) {
                let mut values = HashMap::new();
                let mut total_count = 0;

                for (column, agg) in &self.aggregations {
                    if let Some(state) = states.get(column) {
                        values.insert(column.clone(), state.compute(*agg));
                        total_count = total_count.max(state.count);
                    }
                }

                if !values.is_empty() {
                    let result = WindowResult {
                        window,
                        values,
                        count: total_count,
                        emitted_at: Utc::now(),
                    };
                    results.push(result.clone());
                    self.results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Flushes all active windows
    pub fn flush(&mut self) -> Vec<WindowResult> {
        let mut results = Vec::new();

        for (_, (window, states)) in self.windows.drain() {
            let mut values = HashMap::new();
            let mut total_count = 0;

            for (column, agg) in &self.aggregations {
                if let Some(state) = states.get(column) {
                    values.insert(column.clone(), state.compute(*agg));
                    total_count = total_count.max(state.count);
                }
            }

            if !values.is_empty() || self.config.include_partial_windows {
                let result = WindowResult {
                    window,
                    values,
                    count: total_count,
                    emitted_at: Utc::now(),
                };
                results.push(result);
            }
        }

        results
    }

    /// Gets all completed results
    pub fn results(&self) -> &[WindowResult] {
        &self.results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_record(value: f64) -> StreamRecord {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), value.to_string());
        StreamRecord::new(fields)
    }

    fn create_record_with_time(value: f64, event_time: i64) -> StreamRecord {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), value.to_string());
        fields.insert("event_time".to_string(), event_time.to_string());
        StreamRecord::new(fields)
    }

    #[test]
    fn test_tumbling_window() {
        let config = WindowConfigBuilder::new()
            .tumbling(Duration::from_secs(10))
            .include_partial_windows(true)
            .build();

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        // Process some records
        for i in 0..5 {
            agg.process(&create_record(i as f64)).unwrap();
        }

        // Flush to get results
        let results = agg.flush();
        assert!(!results.is_empty());

        let total: f64 = results.iter().map(|r| r.values["value"]).sum();
        assert_eq!(total, 10.0); // 0 + 1 + 2 + 3 + 4
    }

    #[test]
    fn test_count_window() {
        let config = WindowConfigBuilder::new().count(3, None).build();

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Avg);

        // Process exactly 3 records
        for i in 1..=3 {
            let results = agg.process(&create_record(i as f64)).unwrap();
            if i == 3 {
                // Window should complete on 3rd record
                assert!(!results.is_empty());
                assert!((results[0].values["value"] - 2.0).abs() < 0.001); // avg(1, 2, 3) = 2
            }
        }
    }

    #[test]
    fn test_aggregation_types() {
        let config = WindowConfigBuilder::new()
            .tumbling(Duration::from_secs(60))
            .include_partial_windows(true)
            .build();

        // Test sum
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Sum);
        for i in 1..=5 {
            agg.process(&create_record(i as f64)).unwrap();
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 15.0);

        // Test avg
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Avg);
        for i in 1..=5 {
            agg.process(&create_record(i as f64)).unwrap();
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 3.0);

        // Test min
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Min);
        for i in 1..=5 {
            agg.process(&create_record(i as f64)).unwrap();
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 1.0);

        // Test max
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Max);
        for i in 1..=5 {
            agg.process(&create_record(i as f64)).unwrap();
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 5.0);
    }

    #[test]
    fn test_multi_column_aggregator() {
        let config = WindowConfigBuilder::new()
            .tumbling(Duration::from_secs(60))
            .include_partial_windows(true)
            .build();

        let mut agg = MultiColumnAggregator::new(config);
        agg.add_aggregation("price", WindowAggregation::Sum)
            .add_aggregation("quantity", WindowAggregation::Avg);

        for i in 1..=5 {
            let mut fields = HashMap::new();
            fields.insert("price".to_string(), (i * 10).to_string());
            fields.insert("quantity".to_string(), i.to_string());
            let record = StreamRecord::new(fields);
            agg.process(&record).unwrap();
        }

        let results = agg.flush();
        assert!(!results.is_empty());
        assert_eq!(results[0].values["price"], 150.0); // 10 + 20 + 30 + 40 + 50
        assert_eq!(results[0].values["quantity"], 3.0); // avg(1, 2, 3, 4, 5)
    }

    #[test]
    fn test_window_config_builder() {
        let config = WindowConfigBuilder::new()
            .sliding(Duration::from_secs(30), Duration::from_secs(10))
            .allowed_lateness(Duration::from_secs(5))
            .emit_on_every_record(true)
            .build();

        assert!(matches!(config.window_type, WindowType::Sliding { .. }));
        assert_eq!(config.allowed_lateness, Duration::from_secs(5));
        assert!(config.emit_on_every_record);
    }

    #[test]
    fn test_time_window() {
        let start = Utc::now();
        let end = start + chrono::Duration::seconds(60);
        let window = TimeWindow::new(start, end);

        assert!(window.contains(start));
        assert!(window.contains(start + chrono::Duration::seconds(30)));
        assert!(!window.contains(end));
        assert!(!window.contains(start - chrono::Duration::seconds(1)));
    }

    #[test]
    fn test_session_window() {
        let config = WindowConfigBuilder::new()
            .session(Duration::from_millis(100), None)
            .build();

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        // Process records - they should be in same session
        for i in 1..=3 {
            agg.process(&create_record(i as f64)).unwrap();
        }

        // Simulate gap - flush should close session
        let results = agg.flush();
        assert!(!results.is_empty());
    }
}

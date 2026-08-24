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
use std::collections::{HashMap, HashSet};
use std::time::Duration;

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
    /// For [`WindowType::Session`], the name of the record field whose
    /// value keys separate concurrent sessions (e.g. `"user_id"`, so each
    /// user accumulates independent sessions instead of all records sharing
    /// one global session). `None` (the default) means every record shares
    /// a single global session key -- documented explicitly here since that
    /// is rarely what a real session-window use case wants, but is the only
    /// sensible default in the absence of a configured key.
    pub session_key_field: Option<String>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        WindowConfig {
            window_type: WindowType::default(),
            allowed_lateness: Duration::from_secs(0),
            emit_on_every_record: false,
            include_partial_windows: false,
            watermark_delay: Duration::from_secs(1),
            session_key_field: None,
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

    /// Sets a sliding window.
    ///
    /// Note: a `slide` much smaller than `size` makes each record belong to
    /// `size / slide` overlapping windows simultaneously (e.g. a 1-hour
    /// window with a 1ms slide means roughly 3.6 million overlapping
    /// windows are touched per record) -- keep this ratio reasonable for
    /// your workload.
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

    /// Sets the record field used to key separate sessions for
    /// [`WindowType::Session`]. See [`WindowConfig::session_key_field`].
    pub fn session_key_field(mut self, field: impl Into<String>) -> Self {
        self.config.session_key_field = Some(field.into());
        self
    }

    /// Builds the config.
    ///
    /// Validates window-size invariants that would otherwise cause a panic
    /// deep inside stream processing (`Duration::ZERO` triggering a
    /// divide-by-zero in window alignment) or pathological/unbounded
    /// behavior (a `slide` larger than `size`) rather than being caught at
    /// construction time.
    pub fn build(self) -> Result<WindowConfig> {
        match &self.config.window_type {
            WindowType::Tumbling { size } => {
                if size.is_zero() {
                    return Err(Error::InvalidInput(
                        "tumbling window size must be greater than zero".into(),
                    ));
                }
            }
            WindowType::Sliding { size, slide } => {
                if size.is_zero() {
                    return Err(Error::InvalidInput(
                        "sliding window size must be greater than zero".into(),
                    ));
                }
                if slide.is_zero() {
                    return Err(Error::InvalidInput(
                        "sliding window slide must be greater than zero".into(),
                    ));
                }
                if slide > size {
                    return Err(Error::InvalidInput(
                        "sliding window slide must not exceed size".into(),
                    ));
                }
            }
            WindowType::Count { size, slide } => {
                if *size == 0 {
                    return Err(Error::InvalidInput(
                        "count window size must be greater than zero".into(),
                    ));
                }
                if let Some(slide) = slide {
                    if *slide == 0 {
                        return Err(Error::InvalidInput(
                            "count window slide must be greater than zero".into(),
                        ));
                    }
                    if *slide > *size {
                        return Err(Error::InvalidInput(
                            "count window slide must not exceed size".into(),
                        ));
                    }
                }
            }
            WindowType::Session { .. } | WindowType::Global => {}
        }

        Ok(self.config)
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
    /// Creates a new time window.
    ///
    /// `id` is derived at nanosecond precision (falling back, saturating
    /// rather than overflowing/panicking, to a second-precision value for
    /// timestamps outside chrono's nanosecond-representable range, such as
    /// the sentinel `DateTime::MIN_UTC`/`MAX_UTC` used for
    /// [`WindowType::Global`]). A second-precision id, as this used
    /// unconditionally before, can collide for two genuinely distinct
    /// windows that happen to start and end within the same wall-clock
    /// second -- which is routine for count-based windows (see
    /// `WindowedAggregator::process_count`, whose windows' `start`/`end` are
    /// the real observed event times of their records, often only
    /// microseconds apart under any real load) and for a burst of
    /// high-frequency tumbling/sliding windows.
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        let start_id = start
            .timestamp_nanos_opt()
            .unwrap_or_else(|| start.timestamp().saturating_mul(1_000_000_000));
        let end_id = end
            .timestamp_nanos_opt()
            .unwrap_or_else(|| end.timestamp().saturating_mul(1_000_000_000));
        let id = format!("{}-{}", start_id, end_id);
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
    /// Percentile (specified, 0-100)
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

/// A record that was rejected as too late for its target window (see
/// `WindowedAggregator::is_too_late`). Captured with just enough context
/// for diagnostics/side-channel handling rather than a full [`StreamRecord`]
/// clone (which would also retain every field of every dropped record for
/// the lifetime of the aggregator).
#[derive(Debug, Clone)]
pub struct LateRecord {
    /// The id of the window this record would have belonged to.
    pub window_id: String,
    /// The record's event time.
    pub event_time: DateTime<Utc>,
    /// The parsed value that would have been aggregated.
    pub value: f64,
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
    /// Welford's online mean, used both for `Avg` and as the basis for the
    /// numerically stable variance below.
    mean: f64,
    /// Welford's running sum of squared differences from the mean (`M2`).
    /// Population variance is `m2 / count`; this quantity is mathematically
    /// non-negative by construction (it is a literal sum of squares,
    /// computed incrementally without ever subtracting two similar-sized
    /// large numbers), unlike the previous `E[x^2] - mean^2` formula, which
    /// suffers catastrophic cancellation for data with a large mean and
    /// small spread (e.g. `1e9, 1e9+1, 1e9+2`) and could -- and did --
    /// produce a negative result there. `StdDev` happened to mask that with
    /// a `.max(0.0)` floor before taking the square root; `Variance` did
    /// not, and returned the raw negative value.
    m2: f64,
    /// All raw values seen, for percentile calculations. Populated *only*
    /// when the aggregation actually needs them (`Median`/`Percentile`);
    /// every other aggregation leaves this empty, since retaining every
    /// raw value indefinitely for a `Global`/long-lived `Sum` window would
    /// otherwise grow without bound.
    values: Vec<f64>,
    tracks_raw_values: bool,
}

impl AggregationState {
    fn new(aggregation: WindowAggregation) -> Self {
        AggregationState {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            first: None,
            last: None,
            mean: 0.0,
            m2: 0.0,
            values: Vec::new(),
            tracks_raw_values: matches!(
                aggregation,
                WindowAggregation::Median | WindowAggregation::Percentile(_)
            ),
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

        // Welford's online algorithm.
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if self.tracks_raw_values {
            self.values.push(value);
        }
    }

    fn compute(&self, agg: WindowAggregation) -> f64 {
        match agg {
            WindowAggregation::Sum => self.sum,
            WindowAggregation::Avg => {
                if self.count > 0 {
                    self.mean
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
                    (self.m2 / self.count as f64).sqrt()
                } else {
                    0.0
                }
            }
            WindowAggregation::Variance => {
                if self.count > 1 {
                    self.m2 / self.count as f64
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

/// What kind of window a [`WindowEntry`] represents, so
/// [`WindowedAggregator::close_expired_windows`] can tell time-based windows
/// (which it evicts on watermark passage) apart from count/session/global
/// windows (which complete via their own logic and must never be evicted by
/// the watermark -- previously, count windows were given a nonsensical
/// fabricated time range derived from interpreting the record *count* as
/// *seconds*, specifically so they could participate in this same
/// watermark-based eviction, which then either never fired at all or fired
/// for the wrong reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowKind {
    /// Tumbling or sliding: closed by watermark passage.
    Time,
    /// Count-based: closed (and removed) as soon as it reaches its target
    /// count, in `process_count` itself.
    Count,
    /// Session: closed by a gap timeout or `max_duration`, in
    /// `process_session` itself.
    Session,
    /// Global: closed only by `flush()` at end-of-stream.
    Global,
}

/// One active window's window metadata, kind, and aggregation state.
#[derive(Debug, Clone)]
struct WindowEntry {
    window: TimeWindow,
    kind: WindowKind,
    state: AggregationState,
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
    windows: HashMap<String, WindowEntry>,
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
    /// Ids of windows that have already closed, so a late-arriving record
    /// can never "resurrect" one under the same id as a lookalike -- but
    /// distinct -- window with an incomplete aggregate.
    closed_window_ids: HashSet<String>,
    /// Records rejected as too late for their target (time-based) window.
    late_records: Vec<LateRecord>,
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
            closed_window_ids: HashSet::new(),
            late_records: Vec::new(),
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
                let key = self.session_key(record);
                completed_results.extend(self.process_session(
                    event_time,
                    value,
                    *gap,
                    *max_duration,
                    key,
                )?);
            }
            WindowType::Count { size, slide } => {
                completed_results.extend(self.process_count(event_time, value, *size, *slide)?);
            }
            WindowType::Global => {
                self.process_global(value);
            }
        }

        // Close time-based windows that are past the watermark. Count,
        // session, and global windows are intentionally excluded (see
        // `WindowKind`): they close via their own logic above, not via the
        // watermark.
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

    /// Determines the session key for `record`, per
    /// `WindowConfig::session_key_field`.
    fn session_key(&self, record: &StreamRecord) -> String {
        match &self.config.session_key_field {
            Some(field) => record
                .fields
                .get(field)
                .cloned()
                .unwrap_or_else(|| format!("__missing_{field}__")),
            None => "__global__".to_string(),
        }
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

    /// Returns true if `window_end` is too late to admit a record into
    /// (whether the window in question currently exists, has already
    /// closed and been removed, or was never created at all): its window
    /// would already be eligible for eviction even accounting for
    /// `allowed_lateness`. Because the watermark only ever moves forward,
    /// this check is at least as strict for any *previously* closed window
    /// as the check that closed it was, so it also prevents resurrecting a
    /// closed window id (belt-and-suspenders alongside the explicit
    /// `closed_window_ids` check in the time-based window processors).
    fn is_too_late(&self, window_end: DateTime<Utc>) -> bool {
        let watermark_with_lateness = self.watermark
            - chrono::Duration::from_std(self.config.allowed_lateness)
                .unwrap_or(chrono::Duration::zero());
        window_end <= watermark_with_lateness
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

        if self.closed_window_ids.contains(&window_id) || self.is_too_late(window_end) {
            self.late_records.push(LateRecord {
                window_id,
                event_time,
                value,
            });
            return Ok(vec![]);
        }

        let entry = self
            .windows
            .entry(window_id)
            .or_insert_with(|| WindowEntry {
                window: window.clone(),
                kind: WindowKind::Time,
                state: AggregationState::new(self.aggregation),
            });
        entry.state.update(value);

        // Emit if configured to emit on every record
        if self.config.emit_on_every_record {
            let result = WindowResult {
                window: entry.window.clone(),
                values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                    .into_iter()
                    .collect(),
                count: entry.state.count,
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

            // Skip (rather than resurrect or wastefully re-create-then-evict)
            // any of this record's overlapping sub-windows that are
            // individually too late; the record can still be admitted into
            // whichever of its other overlapping sub-windows are not.
            if !self.closed_window_ids.contains(&window_id) && !self.is_too_late(window_end) {
                let entry = self
                    .windows
                    .entry(window_id)
                    .or_insert_with(|| WindowEntry {
                        window: window.clone(),
                        kind: WindowKind::Time,
                        state: AggregationState::new(self.aggregation),
                    });
                entry.state.update(value);

                if self.config.emit_on_every_record {
                    let result = WindowResult {
                        window: entry.window.clone(),
                        values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                            .into_iter()
                            .collect(),
                        count: entry.state.count,
                        emitted_at: Utc::now(),
                    };
                    results.push(result);
                }
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
        key: String,
    ) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

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
                let window_id = Self::session_window_id(&key, start);
                if let Some(entry) = self.windows.remove(&window_id) {
                    self.closed_window_ids.insert(window_id);
                    let result = WindowResult {
                        window: entry.window,
                        values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                            .into_iter()
                            .collect(),
                        count: entry.state.count,
                        emitted_at: Utc::now(),
                    };
                    self.results.push(result.clone());
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
        let window_id = Self::session_window_id(&key, session_start);

        let entry = self
            .windows
            .entry(window_id)
            .or_insert_with(|| WindowEntry {
                window: TimeWindow::new(session_start, event_time),
                kind: WindowKind::Session,
                state: AggregationState::new(self.aggregation),
            });
        // Extend the session's end to the latest record seen so far, rather
        // than leaving it fixed at whatever it was when the entry was first
        // created (which previously left `end == start` of the *first*
        // record forever, since `or_insert_with` only runs once per key).
        // Guarded with `max` rather than an unconditional assignment: a
        // record that arrives out of event-time order but still within gap
        // tolerance (so it does not start a new session) must not be able
        // to pull the window's `end` backward -- e.g. before its own
        // `start` -- just because its timestamp happens to be earlier than
        // one already folded into this session.
        entry.window.end = entry.window.end.max(event_time);
        entry.state.update(value);

        self.last_record_times.insert(key, event_time);

        Ok(results)
    }

    /// Builds a session window's id from its key and start time, at
    /// nanosecond precision. The previous second-granularity id
    /// (`session_{start.timestamp()}`) meant two *different* sessions --
    /// whether for different keys (there was only ever one key, `"default"`,
    /// before this configurable keying existed) or, now, two independent
    /// keys' sessions that happen to start within the same second --  could
    /// collide onto the same map entry and silently merge their states.
    fn session_window_id(key: &str, start: DateTime<Utc>) -> String {
        format!(
            "session_{}_{}",
            key,
            start.timestamp_nanos_opt().unwrap_or(0)
        )
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
        let slide = slide.unwrap_or(size).max(1);
        let size = size.max(1);

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

                // Real, observed event-time extent of the records actually
                // admitted into this window (extended below), rather than
                // the previous fabricated range that interpreted the record
                // *count* `size` as a number of *seconds*.
                let entry = self
                    .windows
                    .entry(window_id.clone())
                    .or_insert_with(|| WindowEntry {
                        window: TimeWindow::new(event_time, event_time),
                        kind: WindowKind::Count,
                        state: AggregationState::new(self.aggregation),
                    });
                entry.window.end = event_time;
                entry.state.update(value);

                // Check if window is complete
                if entry.state.count >= size {
                    let result = WindowResult {
                        window: entry.window.clone(),
                        values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                            .into_iter()
                            .collect(),
                        count: entry.state.count,
                        emitted_at: Utc::now(),
                    };
                    results.push(result);

                    // Emit-once: remove immediately on completion regardless
                    // of tumbling (slide == size) vs sliding (slide < size)
                    // -- a completed window must never be able to
                    // accumulate further updates and be re-emitted as a
                    // duplicate, monotonically-growing result. This also
                    // means count windows never need (and, per `WindowKind`,
                    // never receive) watermark-based eviction: they are
                    // always removed here, exactly once, right when they
                    // complete.
                    self.windows.remove(&window_id);
                    self.closed_window_ids.insert(window_id);
                }
            }
        }

        Ok(results)
    }

    /// Processes a global window
    fn process_global(&mut self, value: f64) {
        let window_id = "global".to_string();

        let entry = self
            .windows
            .entry(window_id)
            .or_insert_with(|| WindowEntry {
                // A sentinel spanning the full representable range, since a
                // global window has no natural start/end. The previous
                // implementation used `DateTime::from_timestamp(i64::MAX /
                // 2, 0)`, which is outside chrono's representable range and
                // therefore always returned `None` -- unconditionally
                // panicking (via `.expect(...)`) the first time any record
                // was ever processed with `WindowType::Global`.
                window: TimeWindow::new(DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC),
                kind: WindowKind::Global,
                state: AggregationState::new(self.aggregation),
            });

        entry.state.update(value);
    }

    /// Closes time-based windows that are past the watermark. Count,
    /// session, and global windows are excluded (see [`WindowKind`]): they
    /// close via their own dedicated logic, never via the watermark.
    fn close_expired_windows(&mut self) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();

        let watermark_with_lateness = self.watermark
            - chrono::Duration::from_std(self.config.allowed_lateness)
                .unwrap_or(chrono::Duration::zero());

        // Find and close expired windows
        let expired_ids: Vec<String> = self
            .windows
            .iter()
            .filter(|(_, entry)| {
                entry.kind == WindowKind::Time && entry.window.end <= watermark_with_lateness
            })
            .map(|(id, _)| id.clone())
            .collect();

        for window_id in expired_ids {
            if let Some(entry) = self.windows.remove(&window_id) {
                self.closed_window_ids.insert(window_id);
                let result = WindowResult {
                    window: entry.window,
                    values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                        .into_iter()
                        .collect(),
                    count: entry.state.count,
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
        // `WindowConfigBuilder::build` rejects `Duration::ZERO` sizes, but
        // `WindowConfig` is constructible directly (all fields are `pub`),
        // bypassing that validation -- guard here too rather than dividing
        // by zero. Also convert via `try_from` rather than `as i64`: the
        // latter silently truncates for values that don't fit (irrelevant
        // for realistic window sizes, but a real footgun for a
        // user-supplied `Duration` in principle).
        let window_millis: i64 = i64::try_from(window_size.as_millis())
            .unwrap_or(i64::MAX)
            .max(1);
        let aligned_millis = millis.div_euclid(window_millis) * window_millis;

        DateTime::from_timestamp_millis(aligned_millis).unwrap_or(timestamp)
    }

    /// Flushes all active windows and returns results
    pub fn flush(&mut self) -> Vec<WindowResult> {
        let mut results = Vec::new();

        for (_, entry) in self.windows.drain() {
            if entry.state.count > 0 || self.config.include_partial_windows {
                let result = WindowResult {
                    window: entry.window,
                    values: [(self.column.clone(), entry.state.compute(self.aggregation))]
                        .into_iter()
                        .collect(),
                    count: entry.state.count,
                    emitted_at: Utc::now(),
                };
                results.push(result);
            }
        }

        results
    }

    /// Gets all completed window results accumulated so far. Unbounded: for
    /// a long-running aggregator, use [`WindowedAggregator::take_results`]
    /// periodically instead if you don't want this history to grow for the
    /// aggregator's whole lifetime.
    pub fn results(&self) -> &[WindowResult] {
        &self.results
    }

    /// Drains and returns all completed window results accumulated so far,
    /// leaving `results()` empty afterward.
    pub fn take_results(&mut self) -> Vec<WindowResult> {
        std::mem::take(&mut self.results)
    }

    /// Drains and returns all records rejected as too late for their target
    /// window (see `WindowedAggregator::is_too_late`).
    pub fn take_late_records(&mut self) -> Vec<LateRecord> {
        std::mem::take(&mut self.late_records)
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
            let window_millis = i64::try_from(size.as_millis()).unwrap_or(i64::MAX).max(1);
            let window_start_millis = millis.div_euclid(window_millis) * window_millis;
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
            for (column, agg_type) in &self.aggregations {
                let state = states
                    .entry(column.clone())
                    .or_insert_with(|| AggregationState::new(*agg_type));

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

    /// Drains and returns all completed results accumulated so far, leaving
    /// `results()` empty afterward.
    pub fn take_results(&mut self) -> Vec<WindowResult> {
        std::mem::take(&mut self.results)
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

    fn create_record_with_field(value: f64, field: &str, field_value: &str) -> StreamRecord {
        let mut fields = HashMap::new();
        fields.insert("value".to_string(), value.to_string());
        fields.insert(field.to_string(), field_value.to_string());
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
            .build()
            .expect("valid config");

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        // Process some records
        for i in 0..5 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }

        // Flush to get results
        let results = agg.flush();
        assert!(!results.is_empty());

        let total: f64 = results.iter().map(|r| r.values["value"]).sum();
        assert_eq!(total, 10.0); // 0 + 1 + 2 + 3 + 4
    }

    #[test]
    fn test_count_window() {
        let config = WindowConfigBuilder::new()
            .count(3, None)
            .build()
            .expect("valid config");

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Avg);

        // Process exactly 3 records
        for i in 1..=3 {
            let results = agg
                .process(&create_record(i as f64))
                .expect("operation should succeed");
            if i == 3 {
                // Window should complete on 3rd record
                assert!(!results.is_empty());
                assert!((results[0].values["value"] - 2.0).abs() < 0.001); // avg(1, 2, 3) = 2
            } else {
                assert!(results.is_empty());
            }
        }
    }

    #[test]
    fn test_count_window_does_not_duplicate_after_completion() {
        // A sliding count window (slide < size): once a window completes it
        // must never be emitted again as later records that overlap other,
        // still-growing windows are processed. Uses explicit, widely spaced
        // `event_time` values (rather than relying on `Utc::now()`'s real
        // wall-clock resolution) so the test is deterministic regardless of
        // how fast it happens to run.
        let config = WindowConfigBuilder::new()
            .count(4, Some(2))
            .build()
            .expect("valid config");
        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        let mut emitted_window_ids = Vec::new();
        for i in 1..=10 {
            let results = agg
                .process(&create_record_with_time(i as f64, 1_700_000_000 + i * 1000))
                .expect("operation should succeed");
            for r in results {
                emitted_window_ids.push(r.window.id.clone());
            }
        }

        assert_eq!(
            emitted_window_ids.len(),
            4,
            "expected exactly 4 completed count windows (size=4, slide=2, 10 records), got {:?}",
            emitted_window_ids
        );
        let unique: std::collections::HashSet<_> = emitted_window_ids.iter().cloned().collect();
        assert_eq!(
            emitted_window_ids.len(),
            unique.len(),
            "each completed count window must be emitted exactly once, got {:?}",
            emitted_window_ids
        );
    }

    #[test]
    fn test_aggregation_types() {
        let config = WindowConfigBuilder::new()
            .tumbling(Duration::from_secs(60))
            .include_partial_windows(true)
            .build()
            .expect("valid config");

        // Test sum
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Sum);
        for i in 1..=5 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 15.0);

        // Test avg
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Avg);
        for i in 1..=5 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 3.0);

        // Test min
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Min);
        for i in 1..=5 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 1.0);

        // Test max
        let mut agg = WindowedAggregator::new(config.clone(), "value", WindowAggregation::Max);
        for i in 1..=5 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }
        let results = agg.flush();
        assert_eq!(results[0].values["value"], 5.0);
    }

    #[test]
    fn test_variance_is_never_negative_for_large_mean_small_spread() {
        // Values clustered tightly around a large mean are exactly the case
        // that made the old `E[x^2] - mean^2` formula cancel catastrophically
        // and go negative.
        let config = WindowConfigBuilder::new()
            .global()
            .include_partial_windows(true)
            .build()
            .expect("valid config");
        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Variance);

        for v in [1.0e9, 1.0e9 + 1.0, 1.0e9 + 2.0] {
            agg.process(&create_record(v))
                .expect("operation should succeed");
        }

        let results = agg.flush();
        let variance = results[0].values["value"];
        assert!(
            variance >= 0.0,
            "variance must never be negative, got {variance}"
        );
        // True population variance of {0, 1, 2} shifted by a constant is
        // exactly 2/3.
        assert!(
            (variance - (2.0 / 3.0)).abs() < 1e-6,
            "expected variance close to 2/3, got {variance}"
        );
    }

    #[test]
    fn test_multi_column_aggregator() {
        let config = WindowConfigBuilder::new()
            .tumbling(Duration::from_secs(60))
            .include_partial_windows(true)
            .build()
            .expect("valid config");

        let mut agg = MultiColumnAggregator::new(config);
        agg.add_aggregation("price", WindowAggregation::Sum)
            .add_aggregation("quantity", WindowAggregation::Avg);

        for i in 1..=5 {
            let mut fields = HashMap::new();
            fields.insert("price".to_string(), (i * 10).to_string());
            fields.insert("quantity".to_string(), i.to_string());
            let record = StreamRecord::new(fields);
            agg.process(&record).expect("operation should succeed");
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
            .build()
            .expect("valid config");

        assert!(matches!(config.window_type, WindowType::Sliding { .. }));
        assert_eq!(config.allowed_lateness, Duration::from_secs(5));
        assert!(config.emit_on_every_record);
    }

    #[test]
    fn test_window_config_builder_rejects_zero_size() {
        let result = WindowConfigBuilder::new().tumbling(Duration::ZERO).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_window_config_builder_rejects_slide_larger_than_size() {
        let result = WindowConfigBuilder::new()
            .sliding(Duration::from_secs(5), Duration::from_secs(10))
            .build();
        assert!(result.is_err());
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
            .build()
            .expect("valid config");

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        // Process records - they should be in same session
        for i in 1..=3 {
            agg.process(&create_record(i as f64))
                .expect("operation should succeed");
        }

        // Simulate gap - flush should close session
        let results = agg.flush();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_session_window_keyed_by_field_isolates_sessions() {
        let config = WindowConfigBuilder::new()
            .session(Duration::from_secs(60), None)
            .session_key_field("user")
            .include_partial_windows(true)
            .build()
            .expect("valid config");

        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        agg.process(&create_record_with_field(1.0, "user", "alice"))
            .expect("operation should succeed");
        agg.process(&create_record_with_field(10.0, "user", "bob"))
            .expect("operation should succeed");
        agg.process(&create_record_with_field(2.0, "user", "alice"))
            .expect("operation should succeed");
        agg.process(&create_record_with_field(20.0, "user", "bob"))
            .expect("operation should succeed");

        let results = agg.flush();
        // Two independent sessions (one per user), not one merged session.
        assert_eq!(results.len(), 2);
        let sums: Vec<f64> = results.iter().map(|r| r.values["value"]).collect();
        assert!(
            sums.contains(&3.0),
            "alice's session should sum to 3.0, got {sums:?}"
        );
        assert!(
            sums.contains(&30.0),
            "bob's session should sum to 30.0, got {sums:?}"
        );
    }

    #[test]
    fn test_session_window_end_never_regresses_for_out_of_order_record() {
        // gap tolerance wide enough that an out-of-order record (arriving
        // with an *earlier* event_time than one already folded into the
        // session) still belongs to the same session rather than starting a
        // new one.
        let config = WindowConfigBuilder::new()
            .session(Duration::from_secs(120), None)
            .include_partial_windows(true)
            .build()
            .expect("valid config");
        let mut agg = WindowedAggregator::new(config, "value", WindowAggregation::Sum);

        let base = 1_700_000_000i64;
        agg.process(&create_record_with_time(1.0, base))
            .expect("operation should succeed");
        agg.process(&create_record_with_time(2.0, base + 30))
            .expect("operation should succeed");
        // Out of order: earlier event_time than the previous record, but
        // still within the 120s gap tolerance of it, so it must extend the
        // *same* session rather than start a new one or shrink `end`.
        agg.process(&create_record_with_time(5.0, base + 10))
            .expect("operation should succeed");

        let results = agg.flush();
        assert_eq!(
            results.len(),
            1,
            "the out-of-order record must join the single existing session, not start a new one"
        );
        assert_eq!(results[0].values["value"], 1.0 + 2.0 + 5.0);
        assert_eq!(
            results[0].window.start.timestamp(),
            base,
            "session start must remain the first record's time"
        );
        assert_eq!(
            results[0].window.end.timestamp(),
            base + 30,
            "session end must stay at the latest-seen record's time, not regress to an \
             out-of-order record's earlier timestamp"
        );
    }
}

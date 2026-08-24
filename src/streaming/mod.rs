//! Module for streaming data processing
//!
//! This module provides functionality for processing data in a streaming fashion,
//! allowing for efficient handling of data streams, real-time analytics, and
//! continuous data processing.
//!
//! # Features
//!
//! - **Streaming Data Sources**: Read from CSV files, iterators, or custom connectors
//! - **Backpressure Handling**: Multiple strategies for handling slow consumers
//! - **Windowed Aggregations**: Tumbling, sliding, session, and count-based windows
//! - **Real-time Analytics**: Compute metrics like EMA, percentiles, and rate of change
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use pandrs::streaming::{DataStream, StreamConfig, StreamAggregator, AggregationType};
//! use pandrs::streaming::backpressure::{BackpressureBuffer, BackpressureConfig, BackpressureStrategy};
//! use pandrs::streaming::window::{WindowedAggregator, WindowConfigBuilder, WindowAggregation};
//! use std::time::Duration;
//!
//! // Basic streaming with backpressure
//! let config = BackpressureConfig {
//!     high_watermark: 1000,
//!     low_watermark: 500,
//!     strategy: BackpressureStrategy::DropOldest,
//!     ..Default::default()
//! };
//! let buffer = BackpressureBuffer::new(config);
//!
//! // Windowed aggregation
//! let window_config = WindowConfigBuilder::new()
//!     .tumbling(Duration::from_secs(60))
//!     .build()
//!     .expect("valid window config");
//! let mut agg = WindowedAggregator::new(window_config, "value", WindowAggregation::Sum);
//! ```

pub mod backpressure;
pub mod window;

// Re-export backpressure types
pub use backpressure::{
    BackpressureBuffer, BackpressureChannel, BackpressureConfig, BackpressureConfigBuilder,
    BackpressureStats, BackpressureStrategy, FlowController, PushOutcome,
};

// Re-export window types
pub use window::{
    MultiColumnAggregator, TimeWindow, WindowAggregation, WindowConfig, WindowConfigBuilder,
    WindowResult, WindowType, WindowedAggregator,
};

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::lock_safe;

/// Configuration for stream processing
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of records to buffer
    pub buffer_size: usize,
    /// Window size for operations (in number of records)
    pub window_size: Option<usize>,
    /// Window size for operations (in duration)
    pub window_duration: Option<Duration>,
    /// Processing interval (how often to process buffered data)
    pub processing_interval: Duration,
    /// Batch size for processing
    pub batch_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        StreamConfig {
            buffer_size: 10_000,
            window_size: None,
            window_duration: None,
            processing_interval: Duration::from_millis(100),
            batch_size: 1_000,
        }
    }
}

/// A record in a data stream
#[derive(Debug, Clone)]
pub struct StreamRecord {
    /// The data fields
    pub fields: HashMap<String, String>,
    /// Timestamp when the record was received
    pub timestamp: Instant,
}

impl StreamRecord {
    /// Create a new stream record
    pub fn new(fields: HashMap<String, String>) -> Self {
        StreamRecord {
            fields,
            timestamp: Instant::now(),
        }
    }

    /// Create a stream record from a CSV line
    pub fn from_csv(line: &str, headers: &[String]) -> Result<Self> {
        let mut fields = HashMap::new();
        let values: Vec<&str> = line.split(',').collect();

        if values.len() != headers.len() {
            return Err(Error::Cast(format!(
                "CSV line has {} fields but expected {} headers",
                values.len(),
                headers.len()
            )));
        }

        for (i, header) in headers.iter().enumerate() {
            fields.insert(header.clone(), values[i].trim().to_string());
        }

        Ok(StreamRecord::new(fields))
    }
}

/// Builds a [`DataFrame`] from one batch of records, given the stream's
/// column headers. Free function (rather than a method borrowing a whole
/// `DataStream`) so it can be reused by callers -- such as
/// [`StreamProcessor::process`] -- that only have the headers available
/// alongside a transformed batch, not a `DataStream` reference.
fn build_dataframe_from_batch(headers: &[String], batch: &[StreamRecord]) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    if batch.is_empty() {
        return Ok(df);
    }

    // Prepare columns
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();
    for header in headers {
        columns.insert(header.clone(), Vec::with_capacity(batch.len()));
    }

    // Fill columns
    for record in batch {
        for header in headers {
            let value = record.fields.get(header).cloned().unwrap_or_default();
            columns
                .get_mut(header)
                .ok_or_else(|| Error::InvalidOperation(format!("column not found: {}", header)))?
                .push(value);
        }
    }

    // Create DataFrame using add_column method
    for header in headers {
        let column_data = columns
            .get(header)
            .ok_or_else(|| Error::InvalidOperation(format!("column not found: {}", header)))?
            .clone();
        let series = crate::series::Series::new(column_data, Some(header.clone()))?;
        df.add_column(header.clone(), series)?;
    }

    Ok(df)
}

/// Represents a stream of data
#[derive(Debug)]
pub struct DataStream {
    /// Configuration for stream processing
    config: StreamConfig,
    /// Column headers/schema
    headers: Vec<String>,
    /// Sender for stream records. Kept only for callers that want to drive
    /// the stream manually via [`DataStream::get_sender`] after
    /// construction (see [`DataStream::new`]); stream sources that own their
    /// producer outright (e.g. [`DataStream::read_from_csv`]) do not
    /// populate this and instead move their sender into the producer
    /// thread, so that dropping it there is what lets the receiving side
    /// observe [`RecvTimeoutError::Disconnected`] -- see
    /// [`DataStream::process`].
    sender: Option<Sender<StreamRecord>>,
    /// Receiver for stream records
    receiver: Option<Receiver<StreamRecord>>,
}

impl DataStream {
    /// Create a new data stream with specified configuration.
    ///
    /// The returned stream retains its own sender (obtainable via
    /// [`DataStream::get_sender`]) so a caller can feed it manually. If you
    /// intend to call [`DataStream::process`] or
    /// [`DataStream::window_operation`] on this stream, obtain the sender
    /// (or hand it to a producer thread) *before* calling them: both of
    /// those methods drop this struct-held sender as soon as they start, so
    /// that a producer's own sender clone(s) being dropped is what lets the
    /// stream terminate (see their doc comments).
    pub fn new(headers: Vec<String>, config: Option<StreamConfig>) -> Self {
        let config = config.unwrap_or_default();
        let (sender, receiver) = bounded(config.buffer_size);

        DataStream {
            config,
            headers,
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }

    /// Get a sender for this stream
    pub fn get_sender(&self) -> Option<Sender<StreamRecord>> {
        self.sender.clone()
    }

    /// Read from a CSV file, simulating a stream.
    ///
    /// The producer thread owns the only sender for the returned stream's
    /// channel (the stream itself is constructed with no struct-held
    /// sender), so once the file is fully read the sender is dropped and
    /// the channel becomes genuinely disconnected -- which is what lets
    /// [`DataStream::process`]/[`DataStream::window_operation`] know the
    /// stream has actually ended, including after any idle gaps caused by
    /// `delay_ms`.
    pub fn read_from_csv<P: AsRef<Path>>(
        path: P,
        config: Option<StreamConfig>,
        delay_ms: Option<u64>,
    ) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read headers
        let header_line = lines
            .next()
            .ok_or_else(|| Error::Cast("CSV file is empty".into()))??
            .trim()
            .to_string();

        let headers: Vec<String> = header_line
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let config = config.unwrap_or_default();
        let (sender, receiver) = bounded(config.buffer_size);
        let thread_headers = headers.clone();

        // Start a thread to read lines and send to stream
        thread::spawn(move || {
            for line in lines {
                if let Ok(line) = line {
                    if let Ok(record) = StreamRecord::from_csv(&line, &thread_headers) {
                        if sender.send(record).is_err() {
                            // Channel closed, exit thread
                            break;
                        }
                    }

                    // Simulate delay between records if specified
                    if let Some(delay) = delay_ms {
                        thread::sleep(Duration::from_millis(delay));
                    }
                }
            }
            // `sender` drops here, disconnecting the channel once (and only
            // once) the whole file has been read.
        });

        Ok(DataStream {
            config,
            headers,
            sender: None,
            receiver: Some(receiver),
        })
    }

    /// Create a stream from an iterator.
    ///
    /// As with [`DataStream::read_from_csv`], the producer thread owns the
    /// only sender for the returned stream, so the channel disconnects
    /// (letting consumers terminate correctly) once the iterator is
    /// exhausted.
    pub fn from_iterator<I, T>(
        iter: I,
        headers: Vec<String>,
        field_extractor: impl Fn(&T) -> HashMap<String, String> + Send + 'static,
        config: Option<StreamConfig>,
    ) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
        T: Clone + Send + 'static,
    {
        let config = config.unwrap_or_default();
        let (sender, receiver) = bounded(config.buffer_size);

        // Start a thread to read from iterator and send to stream
        thread::spawn(move || {
            for item in iter {
                let fields = field_extractor(&item);
                let record = StreamRecord::new(fields);

                if sender.send(record).is_err() {
                    // Channel closed, exit thread
                    break;
                }
            }
        });

        DataStream {
            config,
            headers,
            sender: None,
            receiver: Some(receiver),
        }
    }

    /// Process the stream with a function, in batches of `batch_size`
    /// records (or `self.config.batch_size` if not specified).
    ///
    /// A batch is flushed to `processor` as soon as it reaches `batch_size`,
    /// and any partial trailing batch is flushed once the stream ends. The
    /// stream is considered ended **only** when the channel reports
    /// [`RecvTimeoutError::Disconnected`] -- an idle gap
    /// ([`RecvTimeoutError::Timeout`]) longer than
    /// `config.processing_interval` is not treated as end-of-stream, since
    /// the producer may simply be slow and could still send more data. This
    /// method drops any sender this struct itself still holds before
    /// entering the receive loop, since a live self-held sender would make
    /// `Disconnected` unreachable (the stream would then hang forever
    /// instead of ever terminating) once a real producer finishes sending.
    pub fn process<F, T>(&mut self, processor: F, batch_size: Option<usize>) -> Result<Vec<T>>
    where
        F: FnMut(&[StreamRecord]) -> Result<T>,
    {
        self.sender = None;

        let batch_size = batch_size.unwrap_or(self.config.batch_size);
        let mut results = Vec::new();
        let mut batch = Vec::with_capacity(batch_size);
        let mut processor = processor;

        // Get receiver
        let receiver = match self.receiver.as_ref() {
            Some(r) => r,
            None => {
                return Err(Error::InvalidValue(
                    "Stream receiver is not available".into(),
                ))
            }
        };

        loop {
            match receiver.recv_timeout(self.config.processing_interval) {
                Ok(record) => {
                    batch.push(record);

                    // Process batch if it's full
                    if batch.len() >= batch_size {
                        let result = processor(&batch)?;
                        results.push(result);
                        batch.clear();
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Just an idle gap: the producer may still be alive.
                    // Keep waiting instead of flushing/terminating.
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // The stream has genuinely ended: flush any partial
                    // trailing batch, then stop.
                    if !batch.is_empty() {
                        let result = processor(&batch)?;
                        results.push(result);
                        batch.clear();
                    }
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Apply a window operation to the stream, firing `operation` once per
    /// completed window rather than once per record.
    ///
    /// Precedence when both are configured: if
    /// `config.window_duration` is set, windows are duration-based (a
    /// window closes once the gap between its first and most recent record
    /// reaches that duration); otherwise, if `config.window_size` is set,
    /// windows are count-based (a window closes once it holds that many
    /// records); if neither is set, the entire stream is treated as a
    /// single window that fires once at end-of-stream. As with
    /// [`DataStream::process`], end-of-stream is detected only via
    /// [`RecvTimeoutError::Disconnected`], and this method drops any
    /// sender this struct itself still holds before consuming.
    pub fn window_operation<F, T>(&mut self, operation: F) -> Result<Vec<T>>
    where
        F: FnMut(&[StreamRecord]) -> Result<T>,
    {
        self.sender = None;

        let mut results = Vec::new();
        let mut operation = operation;

        // Get receiver
        let receiver = match self.receiver.as_ref() {
            Some(r) => r,
            None => {
                return Err(Error::InvalidValue(
                    "Stream receiver is not available".into(),
                ))
            }
        };

        let mut window: Vec<StreamRecord> = Vec::new();
        let mut window_start: Option<Instant> = None;

        loop {
            match receiver.recv_timeout(self.config.processing_interval) {
                Ok(record) => {
                    if window.is_empty() {
                        window_start = Some(record.timestamp);
                    }
                    window.push(record);

                    let boundary_reached = if let Some(duration) = self.config.window_duration {
                        match (window_start, window.last()) {
                            (Some(start), Some(last)) => {
                                last.timestamp.duration_since(start) >= duration
                            }
                            _ => false,
                        }
                    } else if let Some(win_size) = self.config.window_size {
                        window.len() >= win_size
                    } else {
                        // No window configured: accumulate until end-of-stream.
                        false
                    };

                    if boundary_reached {
                        let result = operation(&window)?;
                        results.push(result);
                        window.clear();
                        window_start = None;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    if !window.is_empty() {
                        let result = operation(&window)?;
                        results.push(result);
                    }
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Convert stream batch to DataFrame
    pub fn batch_to_dataframe(&self, batch: &[StreamRecord]) -> Result<DataFrame> {
        build_dataframe_from_batch(&self.headers, batch)
    }
}

/// Stream aggregator for computing aggregates over a stream
#[derive(Debug)]
pub struct StreamAggregator {
    /// Stream to aggregate
    pub stream: DataStream,
    /// Aggregation functions by column
    aggregators: HashMap<String, AggregationType>,
    /// Current aggregate values
    current_values: HashMap<String, f64>,
    /// Count of processed records
    count: usize,
}

/// Types of aggregation functions
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    /// Sum of values
    Sum,
    /// Average of values
    Average,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of values
    Count,
}

/// Applies one record's contribution to a running aggregate state.
///
/// This is a free function (rather than a `&mut self` method) so it can be
/// called from inside the closure passed to `DataStream::process`, which
/// already holds a mutable borrow of `self.stream` for the duration of the
/// call -- a method that needed `&mut self` (any field of it) could not be
/// called from within that closure without conflicting with that borrow.
fn apply_record_to_aggregates(
    aggregators: &HashMap<String, AggregationType>,
    current_values: &mut HashMap<String, f64>,
    count: &mut usize,
    record: &StreamRecord,
) -> Result<()> {
    for (column, agg_type) in aggregators {
        let value_str = record
            .fields
            .get(column)
            .ok_or_else(|| Error::Column(format!("Column '{}' not found in record", column)))?;

        let value = value_str
            .parse::<f64>()
            .map_err(|_| Error::Cast(format!("Could not parse '{}' as number", value_str)))?;

        let current = current_values.get_mut(column).ok_or_else(|| {
            Error::InvalidOperation(format!("aggregation column not found: {}", column))
        })?;

        match agg_type {
            AggregationType::Sum => {
                *current += value;
            }
            AggregationType::Average => {
                // Incremental average update
                let old_count = *count as f64;
                let new_count = (*count + 1) as f64;
                *current = (*current * old_count + value) / new_count;
            }
            AggregationType::Min => {
                *current = (*current).min(value);
            }
            AggregationType::Max => {
                *current = (*current).max(value);
            }
            AggregationType::Count => {
                *current += 1.0;
            }
        }
    }

    *count += 1;

    Ok(())
}

impl StreamAggregator {
    /// Create a new stream aggregator
    pub fn new(stream: DataStream) -> Self {
        StreamAggregator {
            stream,
            aggregators: HashMap::new(),
            current_values: HashMap::new(),
            count: 0,
        }
    }

    /// Add an aggregation function for a column
    pub fn add_aggregator(&mut self, column: &str, agg_type: AggregationType) -> Result<&mut Self> {
        if !self.stream.headers.contains(&column.to_string()) {
            return Err(Error::Column(format!("Column '{}' does not exist", column)));
        }

        self.aggregators.insert(column.to_string(), agg_type);

        // Initialize current value
        match agg_type {
            AggregationType::Min => {
                self.current_values
                    .insert(column.to_string(), f64::INFINITY);
            }
            AggregationType::Max => {
                self.current_values
                    .insert(column.to_string(), f64::NEG_INFINITY);
            }
            _ => {
                self.current_values.insert(column.to_string(), 0.0);
            }
        }

        Ok(self)
    }

    /// Process the stream and compute aggregates.
    ///
    /// Aggregates are updated incrementally as each batch arrives from
    /// `DataStream::process` -- this does **not** buffer the stream's
    /// records in memory first; only the (typically small) running
    /// aggregate state is held across batches.
    pub fn process(&mut self) -> Result<HashMap<String, f64>> {
        let aggregators = self.aggregators.clone();
        let mut current_values = self.current_values.clone();
        let mut count = self.count;

        self.stream.process(
            |batch| {
                for record in batch {
                    apply_record_to_aggregates(
                        &aggregators,
                        &mut current_values,
                        &mut count,
                        record,
                    )?;
                }
                Ok(())
            },
            None,
        )?;

        self.current_values = current_values;
        self.count = count;

        Ok(self.current_values.clone())
    }

    /// Get current aggregate values
    pub fn get_aggregates(&self) -> HashMap<String, f64> {
        self.current_values.clone()
    }
}

/// Stream processor for transforming data in a stream
pub struct StreamProcessor {
    /// Stream to process
    stream: DataStream,
    /// Transformation functions by column
    transformers: HashMap<String, Box<dyn Fn(&str) -> Result<String> + Send>>,
    /// Filter function
    filter: Option<Box<dyn Fn(&StreamRecord) -> bool + Send>>,
}

// Manual Debug implementation to handle closures
impl std::fmt::Debug for StreamProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamProcessor")
            .field("stream", &self.stream)
            .field("transformers_count", &self.transformers.len())
            .field("has_filter", &self.filter.is_some())
            .finish()
    }
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(stream: DataStream) -> Self {
        StreamProcessor {
            stream,
            transformers: HashMap::new(),
            filter: None,
        }
    }

    /// Add a transformation function for a column
    pub fn add_transformer<F>(&mut self, column: &str, transformer: F) -> Result<&mut Self>
    where
        F: Fn(&str) -> Result<String> + Send + 'static,
    {
        if !self.stream.headers.contains(&column.to_string()) {
            return Err(Error::Column(format!("Column '{}' does not exist", column)));
        }

        self.transformers
            .insert(column.to_string(), Box::new(transformer));

        Ok(self)
    }

    /// Set a filter function for records
    pub fn set_filter<F>(&mut self, filter: F) -> &mut Self
    where
        F: Fn(&StreamRecord) -> bool + Send + 'static,
    {
        self.filter = Some(Box::new(filter));
        self
    }

    /// Process the stream and transform data.
    ///
    /// Each batch is transformed and converted to a `DataFrame` as it
    /// arrives from `DataStream::process`, rather than first collecting
    /// every raw batch from the whole stream into memory and only then
    /// transforming them.
    pub fn process(&mut self) -> Result<Vec<DataFrame>> {
        let headers = self.stream.headers.clone();
        // Temporarily move the transformers/filter out of `self` so the
        // closure below can borrow them without conflicting with
        // `self.stream.process(...)`'s mutable borrow of `self.stream`.
        // They are moved back into `self` after the call, regardless of
        // outcome.
        let transformers = std::mem::take(&mut self.transformers);
        let filter = self.filter.take();
        let mut results: Vec<DataFrame> = Vec::new();

        let outcome = self.stream.process(
            |batch| {
                let mut transformed_batch = Vec::with_capacity(batch.len());

                for record in batch {
                    // Apply filter if any
                    if let Some(f) = &filter {
                        if !f(record) {
                            continue;
                        }
                    }

                    // Apply transformations
                    let mut new_fields = HashMap::with_capacity(record.fields.len());

                    for (column, value) in &record.fields {
                        if let Some(transformer) = transformers.get(column) {
                            new_fields.insert(column.clone(), transformer(value)?);
                        } else {
                            new_fields.insert(column.clone(), value.clone());
                        }
                    }

                    transformed_batch.push(StreamRecord {
                        fields: new_fields,
                        timestamp: record.timestamp,
                    });
                }

                let df = build_dataframe_from_batch(&headers, &transformed_batch)?;
                results.push(df);
                Ok(())
            },
            None,
        );

        self.transformers = transformers;
        self.filter = filter;
        outcome?;

        Ok(results)
    }
}

/// Stream connector for connecting to external data sources
#[derive(Debug)]
pub struct StreamConnector {
    /// The stream's declared schema, used by [`StreamConnector::send_fields`]
    /// to validate incoming field names. Genuinely read (not just retained
    /// for a hypothetical future use): without this check, a field sent
    /// under a name that doesn't match any declared header would silently
    /// vanish later, since [`DataStream::batch_to_dataframe`] builds each
    /// `DataFrame` column by iterating over `headers` (not over whatever
    /// keys a given record happens to carry) -- so a typo'd field name
    /// would be dropped with no error anywhere, rather than surfacing at
    /// the point the mistake was actually made.
    headers: Vec<String>,
    /// Data sender
    sender: Sender<StreamRecord>,
}

impl StreamConnector {
    /// Create a new stream connector
    pub fn new(headers: Vec<String>, config: Option<StreamConfig>) -> (Self, DataStream) {
        let config = config.unwrap_or_default();
        let (sender, receiver) = bounded(config.buffer_size);

        let stream = DataStream {
            config,
            headers: headers.clone(),
            sender: None,
            receiver: Some(receiver),
        };

        let connector = StreamConnector { headers, sender };

        (connector, stream)
    }

    /// Send a record to the stream
    pub fn send(&self, record: StreamRecord) -> Result<()> {
        self.sender
            .send(record)
            .map_err(|_| Error::IoError("Failed to send record to stream".into()))
    }

    /// Send a record from field values.
    ///
    /// Every key in `fields` must be one of this connector's declared
    /// `headers`; see the field's doc comment for why this validation
    /// matters. This does *not* require every declared header to be
    /// present in `fields` -- a genuinely absent field is a separate,
    /// pre-existing concern of the schema (`StreamRecord`'s field map has
    /// no representation for "missing" narrower than the key being absent
    /// at all) rather than something this connector-level check should
    /// paper over.
    pub fn send_fields(&self, fields: HashMap<String, String>) -> Result<()> {
        for key in fields.keys() {
            if !self.headers.contains(key) {
                return Err(Error::Column(format!(
                    "field '{}' is not one of this stream's declared headers: {:?}",
                    key, self.headers
                )));
            }
        }

        let record = StreamRecord::new(fields);
        self.send(record)
    }

    /// Close the stream
    pub fn close(self) {
        // Sender is dropped, which closes the channel
    }
}

/// Identifies one configured real-time metric by its user-given name and
/// target column.
///
/// The public, user-facing key remains the combined `"{name}_{column}"`
/// string (as returned by [`RealTimeAnalytics::get_metrics`]); this type
/// exists so the background thread can recover the metric's *column*
/// without parsing that combined string back apart via `split('_')` --
/// which silently breaks for any name or column containing an underscore
/// (e.g. name `"moving_avg"` + column `"value"` previously produced the
/// combined key `"moving_avg_value"`, which `split('_')` then recovered as
/// column `"avg_value"`, a column that does not exist, so the metric never
/// updated again).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MetricKey {
    name: String,
    column: String,
}

impl MetricKey {
    fn display_key(&self) -> String {
        format!("{}_{}", self.name, self.column)
    }
}

/// Real-time stream analytics for computing metrics over streaming data
#[derive(Debug)]
pub struct RealTimeAnalytics {
    /// Stream to analyze
    pub stream: DataStream,
    /// Window size in number of records
    window_size: usize,
    /// Computing interval
    interval: Duration,
    /// Metrics to compute
    metrics: HashMap<MetricKey, MetricType>,
    /// Current metric values
    current_values: Arc<Mutex<HashMap<String, f64>>>,
    /// Stop signal for the background thread
    stop: Arc<AtomicBool>,
    /// Handle of the background processing thread, retained so it can
    /// actually be joined (rather than leaked) on `stop()`/`Drop`.
    thread_handle: Option<thread::JoinHandle<()>>,
}

/// Types of real-time metrics
#[derive(Debug, Clone, Copy)]
pub enum MetricType {
    /// Average over window
    WindowAverage,
    /// Rate of change
    RateOfChange,
    /// Exponential moving average
    ExponentialMovingAverage(f64), // Alpha parameter
    /// Standard deviation
    StandardDeviation,
    /// Percentile, in the 0.0-1.0 convention (e.g. 0.9 for the 90th
    /// percentile). Values outside that range are clamped in
    /// `add_metric`.
    Percentile(f64),
}

impl RealTimeAnalytics {
    /// Create a new real-time analytics processor
    pub fn new(stream: DataStream, window_size: usize, interval: Duration) -> Self {
        RealTimeAnalytics {
            stream,
            window_size,
            interval,
            metrics: HashMap::new(),
            current_values: Arc::new(Mutex::new(HashMap::new())),
            stop: Arc::new(AtomicBool::new(false)),
            thread_handle: None,
        }
    }

    /// Add a metric to compute
    pub fn add_metric(
        &mut self,
        name: &str,
        column: &str,
        metric_type: MetricType,
    ) -> Result<&mut Self> {
        if !self.stream.headers.contains(&column.to_string()) {
            return Err(Error::Column(format!("Column '{}' does not exist", column)));
        }

        // Unify on the 0.0-1.0 convention documented on `MetricType::Percentile`.
        let metric_type = match metric_type {
            MetricType::Percentile(p) => MetricType::Percentile(p.clamp(0.0, 1.0)),
            other => other,
        };

        let key = MetricKey {
            name: name.to_string(),
            column: column.to_string(),
        };
        let display_key = key.display_key();
        self.metrics.insert(key, metric_type);

        // Create a clone to avoid borrowing self in the closure
        let values_clone = self.current_values.clone();
        // Insert the initial value
        {
            let mut values = lock_safe!(values_clone, "stream metric values lock")?;
            values.insert(display_key, 0.0);
        }

        Ok(self)
    }

    /// Start computing metrics in a background thread
    pub fn start_background_processing(&mut self) -> Result<Arc<Mutex<HashMap<String, f64>>>> {
        let receiver = match self.stream.receiver.take() {
            Some(r) => r,
            None => {
                return Err(Error::InvalidValue(
                    "Stream receiver is not available".into(),
                ))
            }
        };

        let window_size = self.window_size;
        let metrics = self.metrics.clone();
        let current_values = self.current_values.clone();
        let stop = self.stop.clone();
        let interval = self.interval;

        // Start background thread
        let handle = thread::spawn(move || {
            let mut window: std::collections::VecDeque<StreamRecord> =
                std::collections::VecDeque::with_capacity(window_size);
            let mut last_values: HashMap<String, f64> = HashMap::new();
            // Per-metric EMA state, seeded from `None` (genuinely absent)
            // rather than read back from `current_values` (which is
            // pre-seeded with a `0.0` placeholder in `add_metric` for
            // external readers, and so could never be told apart from "the
            // previous EMA really was zero" -- biasing every metric's first
            // emission toward zero).
            let mut ema_state: HashMap<MetricKey, f64> = HashMap::new();

            while !stop.load(Ordering::Acquire) {
                // Process records
                while let Ok(record) = receiver.try_recv() {
                    // Add to window
                    window.push_back(record);
                    if window.len() > window_size {
                        window.pop_front();
                    }
                }

                // Compute metrics
                if !window.is_empty() {
                    let mut new_values = HashMap::new();

                    for (metric_key, metric_type) in &metrics {
                        let column = &metric_key.column;
                        let display_key = metric_key.display_key();

                        // Collect values for this column
                        let values: Vec<f64> = window
                            .iter()
                            .filter_map(|record| {
                                record
                                    .fields
                                    .get(column)
                                    .and_then(|v| v.parse::<f64>().ok())
                            })
                            .collect();

                        if values.is_empty() {
                            continue;
                        }

                        // Compute metric
                        let metric_value = match metric_type {
                            MetricType::WindowAverage => {
                                values.iter().sum::<f64>() / values.len() as f64
                            }
                            MetricType::RateOfChange => {
                                if values.len() >= 2 {
                                    let last = values[values.len() - 1];
                                    let prev = values[values.len() - 2];
                                    last - prev
                                } else if let Some(&last_value) = last_values.get(column) {
                                    values[0] - last_value
                                } else {
                                    0.0
                                }
                            }
                            MetricType::ExponentialMovingAverage(alpha) => {
                                let last = values[values.len() - 1];
                                let computed = match ema_state.get(metric_key) {
                                    Some(&prev_ema) => alpha * last + (1.0 - alpha) * prev_ema,
                                    None => last, // genuine cold start
                                };
                                ema_state.insert(metric_key.clone(), computed);
                                computed
                            }
                            MetricType::StandardDeviation => {
                                let mean = values.iter().sum::<f64>() / values.len() as f64;
                                let variance =
                                    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                                        / values.len() as f64;
                                variance.max(0.0).sqrt()
                            }
                            MetricType::Percentile(p) => {
                                let mut sorted = values.clone();
                                sorted.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });

                                let p = p.clamp(0.0, 1.0);
                                let idx = (p * (sorted.len() - 1) as f64).round() as usize;
                                sorted[idx]
                            }
                        };

                        new_values.insert(display_key, metric_value);

                        // Save last value for rate of change
                        if let Some(&last) = values.last() {
                            last_values.insert(column.clone(), last);
                        }
                    }

                    // Update current values
                    if let Ok(mut current) =
                        lock_safe!(current_values, "stream current values lock")
                    {
                        for (key, value) in new_values {
                            current.insert(key, value);
                        }
                    }
                }

                // Wait for next interval (or until stopped)
                thread::sleep(interval);
            }
        });

        self.thread_handle = Some(handle);

        Ok(self.current_values.clone())
    }

    /// Stop background processing and wait for the background thread to
    /// actually exit (bounded by roughly one `interval`), rather than only
    /// flipping a flag and leaking the thread for the rest of the process's
    /// lifetime.
    pub fn stop(&mut self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    /// Get current metric values
    pub fn get_metrics(&self) -> Result<HashMap<String, f64>> {
        Ok(lock_safe!(self.current_values, "stream current values lock")?.clone())
    }
}

impl Drop for RealTimeAnalytics {
    fn drop(&mut self) {
        // Mirror `stop()` so a caller that simply drops `RealTimeAnalytics`
        // without calling `stop()` still doesn't leak the background
        // thread.
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

//! Module for streaming data processing
//!
//! This module provides functionality for processing data in a streaming fashion,
//! allowing for efficient handling of data streams, real-time analytics, and
//! continuous data processing.

use crossbeam_channel::{bounded, Receiver, Sender};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::dataframe::DataFrame;
use crate::error::{Error, PandRSError, Result};
use crate::optimized::dataframe::OptimizedDataFrame;
use crate::series::Series;
use crate::series::Series as LegacySeries;

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

/// Represents a stream of data
#[derive(Debug)]
pub struct DataStream {
    /// Configuration for stream processing
    config: StreamConfig,
    /// Buffer for received records
    buffer: VecDeque<StreamRecord>,
    /// Column headers/schema
    headers: Vec<String>,
    /// Sender for stream records
    sender: Option<Sender<StreamRecord>>,
    /// Receiver for stream records
    receiver: Option<Receiver<StreamRecord>>,
}

impl DataStream {
    /// Create a new data stream with specified configuration
    pub fn new(headers: Vec<String>, config: Option<StreamConfig>) -> Self {
        let config = config.unwrap_or_default();
        let buffer = VecDeque::with_capacity(config.buffer_size);
        let (sender, receiver) = bounded(config.buffer_size);

        DataStream {
            config,
            buffer,
            headers,
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }

    /// Get a sender for this stream
    pub fn get_sender(&self) -> Option<Sender<StreamRecord>> {
        self.sender.clone()
    }

    /// Read from a CSV file, simulating a stream
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

        let stream = DataStream::new(headers.clone(), config);
        let sender = stream.get_sender().unwrap();

        // Start a thread to read lines and send to stream
        thread::spawn(move || {
            for line in lines {
                if let Ok(line) = line {
                    if let Ok(record) = StreamRecord::from_csv(&line, &headers) {
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
        });

        Ok(stream)
    }

    /// Create a stream from an iterator
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
        let stream = DataStream::new(headers, config);
        let sender = stream.get_sender().unwrap();

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

        stream
    }

    /// Process the stream with a function
    pub fn process<F, T>(&mut self, processor: F, batch_size: Option<usize>) -> Result<Vec<T>>
    where
        F: FnMut(&[StreamRecord]) -> Result<T>,
    {
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
            // Try to receive a record with timeout
            match receiver.recv_timeout(self.config.processing_interval) {
                Ok(record) => {
                    // Add to buffer and batch
                    self.buffer.push_back(record.clone());
                    if self.buffer.len() > self.config.buffer_size {
                        self.buffer.pop_front();
                    }

                    batch.push(record);

                    // Process batch if it's full
                    if batch.len() >= batch_size {
                        let result = processor(&batch)?;
                        results.push(result);
                        batch.clear();
                    }
                }
                Err(_) => {
                    // Timeout or channel closed
                    // Process remaining records in batch
                    if !batch.is_empty() {
                        let result = processor(&batch)?;
                        results.push(result);
                        batch.clear();
                    }

                    // If channel is disconnected, exit
                    // Check if the receiver is disconnected by seeing if all senders have been dropped
                    if receiver.is_empty() {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Apply a window operation to the stream
    pub fn window_operation<F, T>(&mut self, operation: F) -> Result<Vec<T>>
    where
        F: FnMut(&[StreamRecord]) -> Result<T>,
    {
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

        // Track window
        let mut window = VecDeque::new();
        let window_size = self.config.window_size.unwrap_or(self.config.buffer_size);
        let start_time = Instant::now();

        loop {
            // Try to receive a record with timeout
            match receiver.recv_timeout(self.config.processing_interval) {
                Ok(record) => {
                    // Add to window
                    window.push_back(record.clone());

                    // Add to buffer
                    self.buffer.push_back(record);
                    if self.buffer.len() > self.config.buffer_size {
                        self.buffer.pop_front();
                    }

                    // Maintain window size
                    if let Some(win_size) = self.config.window_size {
                        while window.len() > win_size {
                            window.pop_front();
                        }
                    }

                    // Check time-based window
                    if let Some(duration) = self.config.window_duration {
                        let now = Instant::now();
                        while !window.is_empty() {
                            let front = &window[0];
                            if now.duration_since(front.timestamp) > duration {
                                window.pop_front();
                            } else {
                                break;
                            }
                        }
                    }

                    // Process window
                    let window_vec: Vec<StreamRecord> = window.iter().cloned().collect();
                    let result = operation(&window_vec)?;
                    results.push(result);
                }
                Err(_) => {
                    // Timeout or channel closed
                    if !window.is_empty() {
                        // Process final window
                        let window_vec: Vec<StreamRecord> = window.iter().cloned().collect();
                        let result = operation(&window_vec)?;
                        results.push(result);
                    }

                    // If channel is disconnected, exit
                    // Check if the receiver is disconnected by seeing if all senders have been dropped
                    if receiver.is_empty() {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    /// Convert stream batch to DataFrame
    pub fn batch_to_dataframe(&self, batch: &[StreamRecord]) -> Result<DataFrame> {
        let mut df = DataFrame::new();

        if batch.is_empty() {
            return Ok(df);
        }

        // Prepare columns
        let mut columns: HashMap<String, Vec<String>> = HashMap::new();
        for header in &self.headers {
            columns.insert(header.clone(), Vec::with_capacity(batch.len()));
        }

        // Fill columns
        for record in batch {
            for header in &self.headers {
                let value = record.fields.get(header).cloned().unwrap_or_default();
                columns.get_mut(header).unwrap().push(value);
            }
        }

        // Create DataFrame using add_column method
        for header in &self.headers {
            let column_data = columns.get(header).unwrap().clone();
            let series = crate::series::Series::new(column_data, Some(header.clone()))?;
            df.add_column(header.clone(), series)?;
        }

        Ok(df)
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

    /// Process the stream and compute aggregates
    pub fn process(&mut self) -> Result<HashMap<String, f64>> {
        // Collect all records from the stream first
        let mut all_records = Vec::new();
        self.stream.process(
            |batch| {
                for record in batch {
                    all_records.push(record.clone());
                }
                Ok(())
            },
            None,
        )?;

        // Now process all records to update aggregates
        for record in &all_records {
            self.update_aggregates(record)?;
        }

        Ok(self.current_values.clone())
    }

    /// Update aggregates with a new record
    fn update_aggregates(&mut self, record: &StreamRecord) -> Result<()> {
        for (column, agg_type) in &self.aggregators {
            let value_str = record
                .fields
                .get(column)
                .ok_or_else(|| Error::Column(format!("Column '{}' not found in record", column)))?;

            let value = value_str
                .parse::<f64>()
                .map_err(|_| Error::Cast(format!("Could not parse '{}' as number", value_str)))?;

            let current = self.current_values.get_mut(column).unwrap();

            match agg_type {
                AggregationType::Sum => {
                    *current += value;
                }
                AggregationType::Average => {
                    // Incremental average update
                    let old_count = self.count as f64;
                    let new_count = (self.count + 1) as f64;
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

        self.count += 1;

        Ok(())
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

    /// Process the stream and transform data
    pub fn process(&mut self) -> Result<Vec<DataFrame>> {
        // Collect all records from the stream first
        let mut all_batches = Vec::new();
        self.stream.process(
            |batch| {
                all_batches.push(batch.to_vec());
                Ok(())
            },
            None,
        )?;

        let mut results = Vec::new();

        // Process each batch
        for batch in all_batches {
            let mut transformed_batch = Vec::new();

            for record in &batch {
                // Apply filter if any
                if let Some(filter) = &self.filter {
                    if !filter(record) {
                        continue;
                    }
                }

                // Apply transformations
                let mut new_fields = HashMap::new();

                for (column, value) in &record.fields {
                    if let Some(transformer) = self.transformers.get(column) {
                        let new_value = transformer(value)?;
                        new_fields.insert(column.clone(), new_value);
                    } else {
                        new_fields.insert(column.clone(), value.clone());
                    }
                }

                transformed_batch.push(StreamRecord {
                    fields: new_fields,
                    timestamp: record.timestamp,
                });
            }

            // Convert transformed batch to DataFrame
            let df = self.stream.batch_to_dataframe(&transformed_batch)?;
            results.push(df);
        }

        Ok(results)
    }
}

/// Stream connector for connecting to external data sources
#[derive(Debug)]
pub struct StreamConnector {
    /// Stream configuration
    config: StreamConfig,
    /// Stream headers
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
            config: config.clone(),
            buffer: VecDeque::with_capacity(config.buffer_size),
            headers: headers.clone(),
            sender: None,
            receiver: Some(receiver),
        };

        let connector = StreamConnector {
            config,
            headers,
            sender,
        };

        (connector, stream)
    }

    /// Send a record to the stream
    pub fn send(&self, record: StreamRecord) -> Result<()> {
        self.sender
            .send(record)
            .map_err(|_| Error::IoError("Failed to send record to stream".into()))
    }

    /// Send a record from field values
    pub fn send_fields(&self, fields: HashMap<String, String>) -> Result<()> {
        let record = StreamRecord::new(fields);
        self.send(record)
    }

    /// Close the stream
    pub fn close(self) {
        // Sender is dropped, which closes the channel
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
    metrics: HashMap<String, MetricType>,
    /// Current metric values
    current_values: Arc<Mutex<HashMap<String, f64>>>,
    /// Stop signal
    stop: Arc<Mutex<bool>>,
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
    /// Percentile
    Percentile(f64), // Percentile to compute (0.0-1.0)
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
            stop: Arc::new(Mutex::new(false)),
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

        let metric_key = format!("{}_{}", name, column);
        self.metrics.insert(metric_key.clone(), metric_type);

        // Create a clone to avoid borrowing self in the closure
        let values_clone = self.current_values.clone();
        // Insert the initial value
        {
            let mut values = values_clone.lock().unwrap();
            values.insert(metric_key, 0.0);
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
        let headers = self.stream.headers.clone();
        let interval = self.interval;

        // Start background thread
        thread::spawn(move || {
            let mut window: VecDeque<StreamRecord> = VecDeque::with_capacity(window_size);
            let mut last_values: HashMap<String, f64> = HashMap::new();

            loop {
                // Check if stopped
                if *stop.lock().unwrap() {
                    break;
                }

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
                        let parts: Vec<&str> = metric_key.split('_').collect();
                        if parts.len() < 2 {
                            continue;
                        }

                        let column = parts[1..].join("_");

                        // Collect values for this column
                        let values: Vec<f64> = window
                            .iter()
                            .filter_map(|record| {
                                record
                                    .fields
                                    .get(&column)
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
                                } else if let Some(&last_value) = last_values.get(&column) {
                                    values[0] - last_value
                                } else {
                                    0.0
                                }
                            }
                            MetricType::ExponentialMovingAverage(alpha) => {
                                let last = values[values.len() - 1];
                                if let Some(&prev_ema) =
                                    current_values.lock().unwrap().get(metric_key)
                                {
                                    alpha * last + (1.0 - alpha) * prev_ema
                                } else {
                                    last
                                }
                            }
                            MetricType::StandardDeviation => {
                                let mean = values.iter().sum::<f64>() / values.len() as f64;
                                let variance =
                                    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                                        / values.len() as f64;
                                variance.sqrt()
                            }
                            MetricType::Percentile(p) => {
                                let mut sorted = values.clone();
                                sorted.sort_by(|a, b| {
                                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                                });

                                let idx = (p * (sorted.len() - 1) as f64).round() as usize;
                                sorted[idx]
                            }
                        };

                        new_values.insert(metric_key.clone(), metric_value);

                        // Save last value for rate of change
                        if let Some(&last) = values.last() {
                            last_values.insert(column, last);
                        }
                    }

                    // Update current values
                    let mut current = current_values.lock().unwrap();
                    for (key, value) in new_values {
                        current.insert(key, value);
                    }
                }

                // Wait for next interval
                thread::sleep(interval);
            }
        });

        Ok(self.current_values.clone())
    }

    /// Stop background processing
    pub fn stop(&self) {
        let mut stop = self.stop.lock().unwrap();
        *stop = true;
    }

    /// Get current metric values
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        self.current_values.lock().unwrap().clone()
    }
}
